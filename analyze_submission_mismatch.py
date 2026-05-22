#!/usr/bin/env python3
"""Analyze (user,rank,item) mismatches vs Recall@10 set overlap."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent


def analyze(best_path: Path, cand_path: Path, sample_users: int | None = None) -> None:
    best = pd.read_csv(best_path, usecols=["user_id", "rank", "item_id"])
    cand = pd.read_csv(cand_path, usecols=["user_id", "rank", "item_id"])
    best["item_id"] = best["item_id"].astype(str)
    cand["item_id"] = cand["item_id"].astype(str)

    if sample_users:
        rng = __import__("numpy").random.default_rng(42)
        users = rng.choice(best["user_id"].unique(), min(sample_users, best["user_id"].nunique()), replace=False)
        best = best[best["user_id"].isin(users)]
        cand = cand[cand["user_id"].isin(users)]

    m = best.merge(cand, on=["user_id", "rank"], suffixes=("_best", "_cand"))
    pos_match = m["item_id_best"] == m["item_id_cand"]
    n = len(m)
    print(f"\n=== {cand_path.name} vs {best_path.name} ===")
    print(f"Rows compared: {n:,}")
    print(f"Same (user, rank, item): {pos_match.mean():.4f}  ({(~pos_match).sum():,} mismatches)")

    bb = best.groupby("user_id")["item_id"].apply(set)
    cc = cand.groupby("user_id")["item_id"].apply(set)
    set_ov = bb.combine(cc, lambda x, y: len(x & y))
    print(f"Avg SET overlap (Recall@10 metric): {set_ov.mean():.4f}/10")
    print(f"Users with identical SET: {(bb == cc).mean():.4f}")

    # Decompose position mismatches
    mm = m[~pos_match].copy()
    reorder = 0
    cand_new = 0
    best_lost = 0
    for _, row in mm.iterrows():
        u = row["user_id"]
        ib, ic = row["item_id_best"], row["item_id_cand"]
        bset, cset = bb[u], cc[u]
        if ic in bset and ib in cset:
            reorder += 1
        if ic not in bset:
            cand_new += 1
        if ib not in cset:
            best_lost += 1

    t = len(mm)
    print(f"\n--- Decompose {t:,} position mismatches ---")
    print(f"  Reorder only (both items still in top-10 SET): {reorder:,} ({100*reorder/t:.1f}%)")
    print(f"  Cand item NOT in best SET (injection):       {cand_new:,} ({100*cand_new/t:.1f}%)")
    print(f"  Best item NOT in cand SET (removed):         {best_lost:,} ({100*best_lost/t:.1f}%)")

    print("\n--- Mismatch rate BY RANK ---")
    for r in range(1, 11):
        sub = m[m["rank"] == r]
        pm = (sub["item_id_best"] == sub["item_id_cand"]).mean()
        inj = (~sub.apply(lambda row: row["item_id_cand"] in bb[row["user_id"]], axis=1)).mean()
        print(f"  rank {r:2d}: pos_match={pm:.3f}  cand_item_new_to_set={inj:.3f}")

    removed_ranks: list[int] = []
    added_ranks: list[int] = []
    for u in bb.index:
        rem = bb[u] - cc[u]
        add = cc[u] - bb[u]
        brow = best[best["user_id"] == u].set_index("item_id")["rank"]
        crow = cand[cand["user_id"] == u].set_index("item_id")["rank"]
        removed_ranks.extend(int(brow[it]) for it in rem)
        added_ranks.extend(int(crow[it]) for it in add)

    print("\n--- Rank in BEST of REMOVED items (GT killers if model was right) ---")
    print(pd.Series(removed_ranks).value_counts().sort_index().to_string())
    print("\n--- Rank in CAND of ADDED items (replacements) ---")
    print(pd.Series(added_ranks).value_counts().sort_index().to_string())

    slots_lost = 10 - set_ov
    slots_added = cc.combine(bb, lambda c, b: len(c - b))
    print(f"\nPer user avg REMOVED from best set: {slots_lost.mean():.3f}")
    print(f"Per user avg ADDED to cand set:      {slots_added.mean():.3f}")


def rebuild_aggressive(out: Path) -> None:
    """Fast rebuild: interactions-only RRF (same failure mode as full aggressive)."""
    import sys

    sys.path.insert(0, str(ROOT))
    from build_submission_postprocess import (  # noqa: WPS433
        TOP_N,
        build_hist_scores,
        load_base_submission,
        rrf_merge_aggressive,
    )

    base = load_base_submission(ROOT / "Sub_Score" / "submission_ (0.2184).csv")
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='5GB'; PRAGMA threads=4")
    inter = str(ROOT / "fact_post_contact_interactions" / "*.parquet")
    events = str(ROOT / "fact_user_events" / "*.parquet")
    build_hist_scores(con, inter, events)
    hist = con.execute(
        "SELECT user_id, item_id, hist_raw, hist_rank FROM hist_rank WHERE hist_rank <= 50"
    ).df()
    dim = set(
        con.execute(f"SELECT item_id FROM read_parquet('{ROOT / 'dim_listing'}/*.parquet')")
        .df()["item_id"]
        .astype(str)
    )
    sub = rrf_merge_aggressive(base, hist, dim)
    sub["ID"] = range(1, len(sub) + 1)
    sub[["ID", "user_id", "rank", "item_id"]].to_csv(out, index=False)
    print("Rebuilt aggressive ->", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--best", type=Path, default=ROOT / "Sub_Score" / "submission_ (0.2184).csv")
    p.add_argument("--cand", type=Path, default=ROOT / "Sub_Score" / "submission_(0.0114).csv")
    p.add_argument("--rebuild-aggressive", action="store_true")
    p.add_argument("--sample", type=int, default=0, help="0 = full data")
    args = p.parse_args()

    if args.rebuild_aggressive:
        rebuild_aggressive(args.cand)

    analyze(args.best, args.cand, sample_users=args.sample or None)


if __name__ == "__main__":
    main()
