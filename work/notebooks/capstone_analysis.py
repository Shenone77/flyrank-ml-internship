"""Build and measure the FlyRank refresh-triage capstone on held-out dates."""
from datetime import date, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path

import duckdb
from huggingface_hub import get_token
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from refresh_triage import MIN_LOSS, MIN_PRIOR_IMPRESSIONS, SEED, baseline_score, ranked

RELEASE = "50cbf7c3909d07be4d1b5906b4d09e882e5acbf2"
BASE = f"hf://datasets/FlyRank/internship-warehouse@{RELEASE}"
CUTOFFS = {"train": date(2026, 4, 1), "test": date(2026, 6, 1)}
FEATURES = ("log_prior_impressions", "prior_ctr_pct", "age_at_cutoff_days",
            "prior_active_days", "prior_late_share")
PER_CLIENT_K = 10
GLOBAL_K = 50
SQL = """
WITH daily AS (
    SELECT client_hash_id, content_hash_id, report_date, gsc_impressions, gsc_clicks
    FROM read_parquet($partitions)
    WHERE report_date >= $start AND report_date < $finish
      AND gsc_data_available IS TRUE
      AND gsc_impressions IS NOT NULL AND gsc_clicks IS NOT NULL
      AND gsc_impressions >= 0 AND gsc_clicks >= 0
), totals AS (
    SELECT client_hash_id, content_hash_id,
           count(*) FILTER (WHERE report_date < $cutoff) AS prior_rows,
           count(DISTINCT report_date) FILTER (WHERE report_date < $cutoff) AS prior_days,
           count(*) FILTER (WHERE report_date >= $cutoff) AS next_rows,
           count(DISTINCT report_date) FILTER (WHERE report_date >= $cutoff) AS next_days,
           sum(gsc_impressions) FILTER (WHERE report_date < $cutoff) AS impressions_prev_30d,
           sum(gsc_clicks) FILTER (WHERE report_date < $cutoff) AS prior_clicks,
           count(*) FILTER (WHERE report_date < $cutoff AND gsc_impressions > 0) AS prior_active_days,
           sum(gsc_impressions) FILTER (
               WHERE report_date >= $midpoint AND report_date < $cutoff
           ) AS late_impressions,
           sum(gsc_impressions) FILTER (WHERE report_date >= $cutoff) AS next_impressions
    FROM daily
    GROUP BY client_hash_id, content_hash_id
    HAVING prior_days = 30 AND impressions_prev_30d >= $minimum
)
SELECT totals.*,
       date_diff('day', d.content_created_date, $cutoff) AS age_at_cutoff_days
FROM totals
JOIN read_parquet($clients) c USING (client_hash_id)
JOIN read_parquet($content) d USING (client_hash_id, content_hash_id)
WHERE c.gsc_data_start <= $start AND d.content_created_date <= $cutoff
ORDER BY client_hash_id, content_hash_id
"""


def warehouse_connection(token):
    con = duckdb.connect()
    try:
        if (extensions := os.environ.get("CAPSTONE_DUCKDB_EXTENSIONS")):
            con.execute("SET extension_directory = ?", [extensions])
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        token = token or get_token()
        if not token:
            raise RuntimeError("Request access to the FlyRank warehouse and configure HF_TOKEN.")
        con.execute("CREATE SECRET flyrank (TYPE huggingface, TOKEN ?)", [token])
    except Exception:
        con.close()
        raise
    return con


def load_frame(root, cutoff, token=None):
    start, finish = cutoff - timedelta(days=30), cutoff + timedelta(days=30)
    months = sorted({(start + timedelta(days=day)).strftime("%Y-%m") for day in range(60)})
    fingerprint = sha256((RELEASE + SQL + str(MIN_PRIOR_IMPRESSIONS)).encode()).hexdigest()[:12]
    cache = root / "data/processed" / f"refresh_capstone_{cutoff}_{fingerprint}.parquet"
    with duckdb.connect() as local:
        if cache.is_file():
            frame = local.execute("SELECT * FROM read_parquet(?) ORDER BY client_hash_id, content_hash_id",
                                  [str(cache)]).df()
        else:
            query_params = {
                "partitions": [f"{BASE}/fact_content_daily_performance/month={month}/*.parquet"
                               for month in months],
                "clients": f"{BASE}/dim_clients.parquet",
                "content": f"{BASE}/dim_content.parquet",
                "start": start,
                "cutoff": cutoff,
                "midpoint": cutoff - timedelta(days=15),
                "finish": finish,
                "minimum": MIN_PRIOR_IMPRESSIONS,
            }
            with warehouse_connection(token) as remote:
                frame = remote.execute(SQL, query_params).df()
            validate_frame(frame)
            cache.parent.mkdir(parents=True, exist_ok=True)
            local.register("page_window", frame)
            pending = cache.with_suffix(".tmp.parquet")
            local.execute("COPY page_window TO ? (FORMAT PARQUET)", [str(pending)])
            pending.replace(cache)
    validate_frame(frame)
    frame["prior_ctr_pct"] = 100 * frame["prior_clicks"] / frame["impressions_prev_30d"]
    frame["log_prior_impressions"] = np.log1p(frame["impressions_prev_30d"])
    frame["prior_late_share"] = frame["late_impressions"] / frame["impressions_prev_30d"]
    frame["tie_order"] = np.random.default_rng(SEED).permutation(len(frame))
    frame["observed"] = frame["next_days"].eq(30)
    frame["loss"] = frame["impressions_prev_30d"] - frame["next_impressions"]
    frame["material_drop"] = material_drop(frame["impressions_prev_30d"], frame["loss"])
    assert np.isfinite(frame.loc[:, list(FEATURES)]).all().all()
    return frame


def validate_frame(frame):
    if frame.empty or frame.duplicated(["client_hash_id", "content_hash_id"]).any():
        raise ValueError("Expected unique page-window rows after joining the warehouse dimensions.")
    if not (frame["prior_rows"].eq(frame["prior_days"]) & frame["next_rows"].eq(frame["next_days"])).all():
        raise ValueError("Repeated page-days would inflate impression totals.")


def material_drop(prior, loss):
    return (loss >= MIN_LOSS) & (10 * loss >= 3 * prior)


def queue_metrics(frame, scores):
    queue = ranked(frame, scores)
    result = queue.head(GLOBAL_K)
    by_client = queue.groupby("client_hash_id", sort=False)
    eligible = by_client.filter(lambda group: len(group) >= PER_CLIENT_K)
    per_client = (eligible.groupby("client_hash_id", sort=False).head(PER_CLIENT_K)
                  .groupby("client_hash_id")["material_drop"].mean())
    return {
        "precision_at_50": float(result["material_drop"].mean()),
        "drops_at_50": int(result["material_drop"].sum()),
        "clients_at_50": int(result["client_hash_id"].nunique()),
        "lost_impressions_among_hits_at_50": int(result.loc[result["material_drop"], "loss"].sum()),
        "average_precision": float(average_precision_score(frame["material_drop"], scores)),
        "macro_precision_at_10": float(per_client.mean()),
    }, per_client


def population(frame):
    observed = frame.loc[frame["observed"]]
    return {
        "eligible_pages": len(frame),
        "observed_outcomes": len(observed),
        "unresolved_outcomes": len(frame) - len(observed),
        "clients_with_observed_outcomes": int(observed["client_hash_id"].nunique()),
        "material_drops": int(observed["material_drop"].sum()),
        "base_rate": float(observed["material_drop"].mean()),
    }


def evaluate(train, test):
    assert material_drop(500, 150) and not material_drop(501, 150) and not material_drop(300, 149)
    fitted = train.loc[train["observed"]]
    holdout = test.loc[test["observed"]]
    x = list(FEATURES)
    if len(holdout) < GLOBAL_K or fitted["material_drop"].nunique() != 2 or holdout["material_drop"].nunique() != 2:
        raise ValueError("The fixed windows must supply usable positives and negatives for evaluation.")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=SEED))
    model.fit(fitted[x], fitted["material_drop"])
    predictions = model.predict_proba(test[x])[:, 1]
    scores = pd.Series(predictions, index=test.index).loc[holdout.index]
    baseline, rule_clients = queue_metrics(holdout, baseline_score(holdout))
    learned, learned_clients = queue_metrics(holdout, scores)
    if rule_clients.empty or not rule_clients.index.equals(learned_clients.index):
        raise ValueError("Both methods must score the same clients at the same review budget.")
    delta = (learned_clients - rule_clients).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
    eligible_clients = rule_clients.index
    macro_base_rate = holdout.groupby("client_hash_id")["material_drop"].mean().loc[eligible_clients].mean()
    metrics = {
        "warehouse_revision": RELEASE,
        "seed": SEED,
        "cutoffs": {key: value.isoformat() for key, value in CUTOFFS.items()},
        "train": population(train),
        "holdout": population(test),
        "evaluation": {"clients_at_k10": len(delta), "macro_random_precision_at_10": float(macro_base_rate),
                       "better_clients": int((delta > 0).sum()), "tied_clients": int((delta == 0).sum()),
                       "worse_clients": int((delta < 0).sum()), "macro_delta": float(delta.mean()),
                       "macro_delta_client_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist()},
        "methods": {"Exposure rule": baseline, "Logistic ranking": learned},
        "standardized_coefficients": dict(zip(x, model.named_steps["logisticregression"].coef_[0].tolist())),
        "versions": {"duckdb": duckdb.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    operational = test.copy()
    operational["exposure_score"] = baseline_score(test)
    operational["model_score"] = predictions
    return metrics, operational


def export_review_queue(root, operational, score_name):
    queue = ranked(operational, operational[score_name])
    queue["client_rank"] = queue.groupby("client_hash_id", sort=False).cumcount() + 1
    review = queue["client_rank"].le(PER_CLIENT_K)
    weak_ctr = queue["prior_ctr_pct"].le(0.5)
    fading = queue["prior_late_share"].lt(0.35)
    queue["reason_code"] = np.select(
        [~review, fading, weak_ctr],
        ["OUTSIDE_BUDGET", "PRIOR_EXPOSURE_FADING", "LOW_PRIOR_CTR"],
        default="EXPOSURE_AT_STAKE",
    )
    queue["action"] = np.select(
        [~review, fading, weak_ctr],
        ["Monitor", "Check indexing and demand", "Inspect query mix and snippet"],
        default="Review page and query trends",
    )
    columns = ["client_hash_id", "content_hash_id", "client_rank", "score", "reason_code", "action"]
    destination = root / "work/outputs/capstone_review_queue.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    queue[columns].to_csv(destination, index=False)
    return queue.loc[review, "reason_code"].value_counts().to_dict()


def run(root, token=None):
    frames = {}
    for stage, cutoff in CUTOFFS.items():
        print(f"Loading {stage} page windows at {cutoff}...", flush=True)
        frames[stage] = load_frame(root, cutoff, token)
        print(f"{stage}: {len(frames[stage]):,} eligible pages.", flush=True)
    metrics, operational = evaluate(frames["train"], frames["test"])
    lower, _ = metrics["evaluation"]["macro_delta_client_bootstrap_95"]
    policy = "model_score" if lower > 0 else "exposure_score"
    metrics["recommended_queue_score"] = policy
    metrics["review_queue_reasons"] = export_review_queue(root, operational, policy)
    output_path = root / "work/outputs/capstone_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    output = run(Path(__file__).resolve().parents[2])
    print(json.dumps(output, indent=2))
