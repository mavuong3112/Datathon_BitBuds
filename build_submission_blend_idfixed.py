#!/usr/bin/env python3
"""
Blend submissions on 0.2184 ID template (Kaggle row order).

Strategy: maximize set overlap with 0.2184 while injecting
high-confidence slots from alt (0.0114 history / 0.2182).

  tier1: all items in intersection(primary, alt) — score = 1/rank_p + 1/rank_a
  tier2: remaining primary by rank_p
  tier3: remaining alt (only if in union top-10 of both files)

Default: conservative — cap alt-only slots at 2 per user.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "Sub_Score" / "submission_ (0.2184).csv"
PRIMARY = TEMPLATE
ALT = ROOT / "Sub_Score" / "submission_(0.0114).csv"
OUT = ROOT / "Sub_Score" / "submission_blend_2184_0114_idfixed.csv"
SCORE_K = 60.0
TOP_N = 10
MAX_ALT_ONLY = 2


def build(primary: Path, alt: Path, max_alt_only: int) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute(
        f"""
        CREATE TABLE p AS SELECT * FROM read_csv('{primary}', header=true);
        CREATE TABLE a AS SELECT * FROM read_csv('{alt}', header=true);
        CREATE TABLE pool AS
        SELECT user_id, item_id,
               MIN(CASE WHEN src='p' THEN rank END) rank_p,
               MIN(CASE WHEN src='a' THEN rank END) rank_a,
               MAX(CASE WHEN src='p' THEN 1 ELSE 0 END) in_p,
               MAX(CASE WHEN src='a' THEN 1 ELSE 0 END) in_a
        FROM (
            SELECT user_id, item_id, rank, 'p' src FROM p
            UNION ALL SELECT user_id, item_id, rank, 'a' src FROM a
        ) GROUP BY 1,2;
        CREATE TABLE scored AS
        SELECT *,
            CASE WHEN in_p=1 AND in_a=1 THEN 1 ELSE 0 END in_both,
            COALESCE(1.0/({SCORE_K}+rank_p),0) + COALESCE(1.0/({SCORE_K}+rank_a),0) pa_score,
            COALESCE(1.0/({SCORE_K}+rank_p),0) p_score,
            COALESCE(1.0/({SCORE_K}+rank_a),0) a_score
        FROM pool;
        """
    )
    rows = con.execute(
        f"""
        WITH ranked AS (
            SELECT user_id, item_id, in_both, rank_p, rank_a, pa_score, p_score, a_score,
                CASE
                    WHEN in_both=1 THEN 1
                    WHEN in_p=1 AND in_a=0 THEN 2
                    WHEN in_p=0 AND in_a=1 THEN 3
                    ELSE 4
                END tier,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id,
                    CASE
                        WHEN in_both=1 THEN 1
                        WHEN in_p=1 AND in_a=0 THEN 2
                        WHEN in_p=0 AND in_a=1 THEN 3
                        ELSE 4
                    END
                    ORDER BY
                        CASE WHEN in_both=1 THEN pa_score ELSE 0 END DESC,
                        p_score DESC, a_score DESC, rank_p, rank_a, item_id
                ) rn_tier
            FROM scored
        ),
        capped AS (
            SELECT *,
                SUM(CASE WHEN tier=3 THEN 1 ELSE 0 END) OVER (PARTITION BY user_id) alt_only_n
            FROM ranked
            WHERE tier IN (1,2) OR (tier=3 AND rn_tier <= {max_alt_only})
        ),
        final AS (
            SELECT user_id, item_id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY tier, rn_tier
                ) rank
            FROM capped
        )
        SELECT user_id, rank, item_id FROM final WHERE rank <= {TOP_N}
        """
    ).df()
    tpl = pd.read_csv(TEMPLATE, usecols=["ID", "user_id", "rank"])
    out = tpl.merge(rows, on=["user_id", "rank"], how="left", validate="one_to_one")
    if out["item_id"].isna().any():
        out = tpl.merge(
            pd.read_csv(primary, usecols=["user_id", "rank", "item_id"]),
            on=["user_id", "rank"],
            how="left",
        )
    return out[["ID", "user_id", "rank", "item_id"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=Path, default=ALT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-alt-only", type=int, default=MAX_ALT_ONLY)
    args = ap.parse_args()
    out = build(PRIMARY, args.alt, args.max_alt_only)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows", len(out))


if __name__ == "__main__":
    main()
