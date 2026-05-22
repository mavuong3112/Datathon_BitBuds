#!/usr/bin/env python3
"""Run Góc 4 other_interaction EDA for all categories."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

from eda_engagement_g4_module import cross_lift_summary, run_corner4_category

DATA_ROOT = Path(__file__).resolve().parent
OUT_DIR = DATA_ROOT / "outputs" / "eda_engagement"
OUT_CSV = OUT_DIR / "g4_tables"
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
INTER_GLOB = str(DATA_ROOT / "fact_post_contact_interactions" / "*.parquet")
DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
SNAP_GLOB = str(DATA_ROOT / "fact_listing_snapshot" / "*.parquet")
PALETTE = {1010: "#238b45", 1020: "#2171b5", 1030: "#6a51a3", 1040: "#cb181d", 1050: "#d94801"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--categories", default="1010,1020,1030,1040,1050")
    p.add_argument("--sample-frac", type=float, default=0.08)
    p.add_argument("--memory", default="4GB")
    args = p.parse_args()
    cats = [int(x.strip()) for x in args.categories.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{args.memory}'")
    con.execute("PRAGMA threads=4")

    parts = []
    for cat in cats:
        print(f"=== category {cat} (sample={args.sample_frac}) ===", flush=True)
        ab = run_corner4_category(
            con, cat, EVENTS_GLOB, INTER_GLOB, DIM_GLOB, SNAP_GLOB,
            OUT_DIR, OUT_CSV, args.sample_frac, PALETTE[cat],
        )
        print(ab.to_string(index=False), flush=True)
        parts.append(ab)

    if parts:
        ab_all = pd.concat(parts, ignore_index=True)
        lift = cross_lift_summary(ab_all, OUT_CSV, OUT_DIR)
        print("\n=== cross lift ===\n", lift.to_string(index=False), flush=True)
    print("Done →", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
