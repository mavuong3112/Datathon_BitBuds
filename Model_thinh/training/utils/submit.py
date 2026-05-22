"""Submission validator + writer."""
from __future__ import annotations

import pandas as pd


def validate_submission(sub: pd.DataFrame, valid_item_ids: set,
                         expected_users: set, k: int = 10) -> None:
    assert {"user_id", "rank", "item_id"}.issubset(sub.columns), \
        f"Missing required columns; got {list(sub.columns)}"
    assert sub["item_id"].notna().all(), "item_id has nulls"
    assert sub["user_id"].notna().all(), "user_id has nulls"
    bad = ~sub["item_id"].isin(valid_item_ids)
    assert not bad.any(), f"{int(bad.sum())} item_id not in dim_listing"
    g = sub.groupby("user_id")
    sizes = g.size()
    assert (sizes <= k).all(), f"Some users have > {k} predictions"
    assert sub["rank"].between(1, k).all(), "rank outside [1, k]"
    dup = sub.duplicated(subset=["user_id", "item_id"])
    assert not dup.any(), f"{int(dup.sum())} duplicate (user, item) pairs"
    rank_uniq = g["rank"].apply(lambda x: x.nunique() == len(x))
    assert rank_uniq.all(), "rank not unique within user"
    missing_users = expected_users - set(sub["user_id"])
    if missing_users:
        print(f"WARNING: {len(missing_users)} expected users missing from submission")
    print(f"Submission OK: {len(sub):,} rows, "
          f"{sub['user_id'].nunique():,} users, "
          f"min/max rank {sub['rank'].min()}/{sub['rank'].max()}")


def write_submission(sub: pd.DataFrame, path: str) -> None:
    out = sub[["user_id", "rank", "item_id"]].copy()
    out.insert(0, "ID", range(len(out)))
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"Wrote {path} ({len(out):,} rows)")
