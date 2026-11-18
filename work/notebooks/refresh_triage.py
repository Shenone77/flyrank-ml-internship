"""Shared data, baseline, and holdout for the content refresh assignments."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

STARTER_CSV = Path("data/raw/content_refresh_anonymized.csv")
SEED = 42
TEST_SIZE = 0.25
REVIEW_BUDGET = 50
MIN_PRIOR_IMPRESSIONS = 300
MIN_LOSS = 150
LOW_CTR_PCT = 0.5
FEATURES = ("log_prior_impressions", "prior_ctr_pct", "age_at_cutoff_days")


def load_pages(root):
    keys = ["client_id", "content_id"]
    counts = ["impressions_prev_30d", "clicks_prev_30d", "impressions_last_30d", "content_age_days"]
    raw = pd.read_csv(root / STARTER_CSV, usecols=keys + counts)
    if raw[keys].isna().any().any() or raw.duplicated(keys).any():
        raise ValueError("Expected one row per nonmissing client and content key.")
    if not np.isfinite(raw[counts]).all().all() or raw[counts].lt(0).any().any():
        raise ValueError("Expected finite nonnegative counts.")
    if raw["clicks_prev_30d"].gt(raw["impressions_prev_30d"]).any():
        raise ValueError("Prior clicks cannot exceed prior impressions.")
    if raw["content_age_days"].lt(30).any():
        raise ValueError("A page in this slice was not live at the decision cutoff.")

    eligible = raw["impressions_prev_30d"].ge(MIN_PRIOR_IMPRESSIONS)
    pages = raw.loc[eligible].reset_index(drop=True)
    prior = pages["impressions_prev_30d"]
    loss = prior - pages["impressions_last_30d"]
    pages["material_drop"] = loss.ge(MIN_LOSS) & loss.mul(10).ge(prior.mul(3))
    pages["log_prior_impressions"] = np.log1p(prior)
    pages["prior_ctr_pct"] = 100 * pages["clicks_prev_30d"] / prior
    pages["age_at_cutoff_days"] = pages["content_age_days"] - 30
    pages["tie_order"] = np.random.default_rng(SEED).permutation(len(pages))
    return pages


def client_holdout(pages):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, holdout_idx = next(splitter.split(pages, groups=pages["client_id"]))
    train, holdout = pages.iloc[train_idx], pages.iloc[holdout_idx]
    assert set(train["client_id"]).isdisjoint(holdout["client_id"])
    assert len(holdout) >= REVIEW_BUDGET
    assert train["material_drop"].nunique() == holdout["material_drop"].nunique() == 2
    return train, holdout


def baseline_score(pages):
    boost = pages["prior_ctr_pct"].le(LOW_CTR_PCT)
    return pages["impressions_prev_30d"] * (1 + 0.25 * boost)


def ranked(pages, scores):
    return pages.assign(score=np.asarray(scores)).sort_values(
        ["score", "tie_order"], ascending=[False, True]
    )


def precision_at_k(queue, k=REVIEW_BUDGET):
    assert 0 < k <= len(queue)
    return float(queue["material_drop"].head(k).mean())
