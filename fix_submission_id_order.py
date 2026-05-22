#!/usr/bin/env python3
"""Re-map predictions onto Kaggle sample_submission row order (ID column)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "Sub_Score" / "submission_ (0.2184).csv"


def fix(src: Path, dst: Path | None = None, template: Path = TEMPLATE) -> Path:
    dst = dst or src
    pred = pd.read_csv(src, usecols=["user_id", "rank", "item_id"])
    base = pd.read_csv(template, usecols=["ID", "user_id", "rank"])
    out = base.merge(pred, on=["user_id", "rank"], how="left", validate="one_to_one")
    if out["item_id"].isna().any():
        raise ValueError(f"{out['item_id'].isna().sum()} rows missing item_id")
    out[["ID", "user_id", "rank", "item_id"]].to_csv(dst, index=False)
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description="Fix submission ID row order")
    p.add_argument("src", type=Path, help="Submission to fix")
    p.add_argument("--out", type=Path, default=None, help="Output path (default: overwrite src)")
    p.add_argument("--template", type=Path, default=TEMPLATE)
    args = p.parse_args()
    out = fix(args.src, args.out, args.template)
    print("Wrote", out)


if __name__ == "__main__":
    main()
