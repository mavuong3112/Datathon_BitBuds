#!/usr/bin/env python3
"""
Optimal post-processing — 3 strategies (conservative, data-driven):

1. PRUNE hard-rules (no inject outside pool union 0.2184∪0.2182):
   - user×item purchased=true
   - expected_expired_date < train cutoff
   - ad_status = 'refused' (not all 'deleted' — most dim rows are deleted but still in LB)

2. Cold users: intersection(0.2184,0.2182) core + fill by pseudo-score 1/(k+rank_p)
   + city popular backfill (not global blind rank1)

3. Warm users: default strict = exact 0.2184; optional --warm-mode prune

  env/bin/python build_submission_optimal_postprocess.py
  env/bin/python build_submission_optimal_postprocess.py --warm-mode prune
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "Sub_Score" / "submission_ (0.2184).csv"
SECONDARY = ROOT / "Sub_Score" / "submission_(0.2182).csv"
OUT = ROOT / "submission.csv"
CUTOFF = "2026-04-09"
SCORE_K = 60.0
TOP_N = 10


def prepare(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("PRAGMA memory_limit='5GB'; PRAGMA threads=4")
    dim = str(ROOT / "dim_listing" / "*.parquet")
    inter = str(ROOT / "fact_post_contact_interactions" / "*.parquet")
    events = str(ROOT / "fact_user_events" / "*.parquet")
    test = str(ROOT / "test" / "test_users.parquet")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim AS
        SELECT item_id, city_name, ad_status, expected_expired_date
        FROM read_parquet('{dim}');

        CREATE OR REPLACE TABLE test_users AS
        SELECT user_id FROM read_parquet('{test}');

        CREATE OR REPLACE TABLE warm_users AS
        SELECT DISTINCT user_id FROM (
            SELECT user_id FROM read_parquet('{inter}')
            WHERE user_id IN (SELECT user_id FROM test_users)
            UNION
            SELECT user_id FROM read_parquet('{events}')
            WHERE is_login = 'login'
              AND user_id IN (SELECT user_id FROM test_users)
        );

        CREATE OR REPLACE TABLE user_purchased AS
        SELECT DISTINCT user_id, item_id
        FROM read_parquet('{inter}')
        WHERE purchased = true
          AND user_id IN (SELECT user_id FROM test_users);

        CREATE OR REPLACE TABLE user_city AS
        SELECT user_id, city_name FROM (
            SELECT e.user_id, e.city_name,
                   ROW_NUMBER() OVER (PARTITION BY e.user_id ORDER BY COUNT(*) DESC) rn
            FROM read_parquet('{events}') e
            WHERE e.is_login = 'login' AND e.city_name IS NOT NULL
              AND e.user_id IN (SELECT user_id FROM test_users)
              AND e.date <= DATE '{CUTOFF}'
            GROUP BY e.user_id, e.city_name
        ) WHERE rn = 1;

        -- Conservative prune list (NOT all deleted listings)
        CREATE OR REPLACE TABLE prune_global AS
        SELECT item_id FROM dim
        WHERE ad_status = 'refused'
           OR (expected_expired_date IS NOT NULL
               AND expected_expired_date < DATE '{CUTOFF}');

        -- City popular: positive events only (faster than full interactions scan)
        CREATE OR REPLACE TABLE city_popular AS
        SELECT city_name, item_id,
               ROW_NUMBER() OVER (PARTITION BY city_name ORDER BY cnt DESC) AS city_rank
        FROM (
            SELECT e.city_name, e.item_id, COUNT(*) cnt
            FROM read_parquet('{events}') e
            WHERE e.city_name IS NOT NULL
              AND e.event_type IN (
                  'view_phone','contact_chat','contact_zalo','contact_sms','other_interaction'
              )
              AND e.date <= DATE '{CUTOFF}'
              AND e.item_id NOT IN (SELECT item_id FROM prune_global)
            GROUP BY 1, 2
        );
        """
    )


def load_subs(con: duckdb.DuckDBPyConnection, primary: Path, secondary: Path) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sub_p AS
        SELECT user_id, rank::INT rank, item_id FROM read_csv('{primary}', header=true);
        CREATE OR REPLACE TABLE sub_s AS
        SELECT user_id, rank::INT rank, item_id FROM read_csv('{secondary}', header=true);
        CREATE OR REPLACE TABLE pool AS
        SELECT user_id, item_id,
               MIN(CASE WHEN src='p' THEN rank END) rank_p,
               MIN(CASE WHEN src='s' THEN rank END) rank_s,
               MAX(CASE WHEN src='p' THEN 1 ELSE 0 END) in_p,
               MAX(CASE WHEN src='s' THEN 1 ELSE 0 END) in_s
        FROM (
            SELECT user_id, item_id, rank, 'p' src FROM sub_p
            UNION ALL SELECT user_id, item_id, rank, 's' FROM sub_s
        ) GROUP BY 1, 2;
        CREATE OR REPLACE TABLE pool_scored AS
        SELECT user_id, item_id, rank_p, rank_s, in_p, in_s,
               CASE WHEN in_p=1 AND in_s=1 THEN 1 ELSE 0 END in_both,
               COALESCE(1.0/({SCORE_K}+rank_p),0) fill_score
        FROM pool;
        """
    )


def build_cold(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH cold AS (
            SELECT user_id FROM test_users
            WHERE user_id NOT IN (SELECT user_id FROM warm_users)
        ),
        dropped AS (SELECT user_id, item_id FROM user_purchased),
        tier1 AS (
            SELECT ps.user_id, ps.item_id, ps.fill_score, ps.rank_p, 1 tier
            FROM pool_scored ps
            JOIN cold c ON ps.user_id = c.user_id
            WHERE ps.in_both = 1
              AND ps.item_id NOT IN (SELECT item_id FROM prune_global)
              AND NOT EXISTS (
                  SELECT 1 FROM dropped d
                  WHERE d.user_id = ps.user_id AND d.item_id = ps.item_id
              )
        ),
        tier2 AS (
            SELECT ps.user_id, ps.item_id, ps.fill_score, ps.rank_p, 2 tier
            FROM pool_scored ps
            JOIN cold c ON ps.user_id = c.user_id
            WHERE ps.in_p = 1 AND ps.in_both = 0
              AND ps.item_id NOT IN (SELECT item_id FROM prune_global)
              AND NOT EXISTS (
                  SELECT 1 FROM dropped d
                  WHERE d.user_id = ps.user_id AND d.item_id = ps.item_id
              )
        ),
        tier3 AS (
            SELECT c.user_id, cp.item_id,
                   (1.0/(3+cp.city_rank)) fill_score, cp.city_rank rank_p, 3 tier
            FROM cold c
            LEFT JOIN user_city uc ON c.user_id = uc.user_id
            JOIN city_popular cp ON cp.city_name = uc.city_name AND cp.city_rank <= 15
            WHERE NOT EXISTS (
                SELECT 1 FROM dropped d
                WHERE d.user_id = c.user_id AND d.item_id = cp.item_id
            )
        ),
        all_t AS (
            SELECT * FROM tier1 UNION ALL SELECT * FROM tier2
            UNION ALL SELECT * FROM tier3
        ),
        dedup AS (
            SELECT user_id, item_id, MIN(tier) tier,
                   MAX(fill_score) fill_score, MIN(rank_p) rank_p
            FROM all_t GROUP BY 1, 2
        ),
        ranked AS (
            SELECT user_id, item_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id
                       ORDER BY tier, fill_score DESC, rank_p, item_id
                   ) rank
            FROM dedup
        )
        SELECT user_id, rank, item_id FROM ranked WHERE rank <= 10
        """
    ).df()


def build_warm_prune(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        WITH dropped AS (
            SELECT user_id, item_id FROM user_purchased
        ),
        valid_p AS (
            SELECT p.user_id, p.rank, p.item_id
            FROM sub_p p
            WHERE p.user_id IN (SELECT user_id FROM warm_users)
              AND p.item_id NOT IN (SELECT item_id FROM prune_global)
              AND NOT EXISTS (
                  SELECT 1 FROM dropped d
                  WHERE d.user_id = p.user_id AND d.item_id = p.item_id
              )
        ),
        fill AS (
            SELECT ps.user_id, ps.item_id, ps.fill_score, ps.rank_p, ps.in_both
            FROM pool_scored ps
            WHERE ps.user_id IN (SELECT user_id FROM warm_users) AND ps.in_p = 1
              AND ps.item_id NOT IN (SELECT item_id FROM prune_global)
              AND NOT EXISTS (
                  SELECT 1 FROM dropped d
                  WHERE d.user_id = ps.user_id AND d.item_id = ps.item_id
              )
        ),
        comb AS (
            SELECT user_id, item_id, rank::DOUBLE sk FROM valid_p
            UNION ALL
            SELECT user_id, item_id,
                   50.0 + ROW_NUMBER() OVER (
                       PARTITION BY user_id
                       ORDER BY in_both DESC, fill_score DESC, rank_p, item_id
                   ) sk
            FROM fill
        ),
        dedup AS (
            SELECT user_id, item_id, MIN(sk) sk FROM comb GROUP BY 1, 2
        ),
        ranked AS (
            SELECT user_id, item_id,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY sk) rank
            FROM dedup
        )
        SELECT user_id, rank, item_id FROM ranked WHERE rank <= 10
        """
    ).df()


def pad_users(sub: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    pri = primary.sort_values(["user_id", "rank"])
    sub = sub.sort_values(["user_id", "rank"])
    merged = sub.merge(
        pri[pri["user_id"].isin(sub["user_id"].unique())],
        on="user_id",
        how="outer",
        suffixes=("_s", "_p"),
    )
    # users only in primary handled separately
    missing_users = set(pri["user_id"]) - set(sub["user_id"])
    if missing_users:
        extra = pri[pri["user_id"].isin(missing_users)]
        sub = pd.concat([sub, extra.rename(columns={"rank": "rank", "item_id": "item_id"})], ignore_index=True)

    rows: list[dict] = []
    all_users = pri["user_id"].unique()
    sub_by = {u: g.sort_values("rank")["item_id"].tolist() for u, g in sub.groupby("user_id")}
    pri_by = {u: g.sort_values("rank")["item_id"].tolist() for u, g in pri.groupby("user_id")}
    for uid in all_users:
        chosen = list(sub_by.get(uid, []))
        if len(chosen) < TOP_N:
            for it in pri_by[uid]:
                if it not in chosen:
                    chosen.append(it)
                if len(chosen) >= TOP_N:
                    break
        for r, it in enumerate(chosen[:TOP_N], 1):
            rows.append({"user_id": uid, "rank": r, "item_id": it})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, default=PRIMARY)
    ap.add_argument("--secondary", type=Path, default=SECONDARY)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--warm-mode", choices=("strict", "prune"), default="strict")
    args = ap.parse_args()

    con = duckdb.connect(":memory:")
    print("Prepare…")
    prepare(con)
    load_subs(con, args.primary, args.secondary)
    nw = con.execute("SELECT COUNT(*) FROM warm_users").fetchone()[0]
    nc = con.execute("SELECT COUNT(*) FROM test_users").fetchone()[0] - nw
    print(f"Warm {nw:,} | Cold {nc:,}")

    if args.warm_mode == "strict":
        warm = con.execute(
            "SELECT user_id, rank, item_id FROM sub_p "
            "WHERE user_id IN (SELECT user_id FROM warm_users) ORDER BY 1,2"
        ).df()
        print("Warm: exact 0.2184")
    else:
        warm = build_warm_prune(con)
        print("Warm: prune + backfill (pool)")

    print("Cold: intersection + score fill + city popular…")
    cold = build_cold(con)
    primary_df = pd.read_csv(args.primary)
    sub = pad_users(pd.concat([warm, cold], ignore_index=True), primary_df)

    # CRITICAL: Kaggle grades by ID row order (= sample_submission), NOT sorted user_id.
    sub = primary_df[["ID", "user_id", "rank"]].merge(
        sub[["user_id", "rank", "item_id"]],
        on=["user_id", "rank"],
        how="left",
        validate="one_to_one",
    )
    assert sub["item_id"].notna().all(), "missing item_id after template merge"
    sub = sub[["ID", "user_id", "rank", "item_id"]]
    p = pd.read_csv(args.primary, usecols=["user_id", "rank", "item_id"])
    m = p.merge(sub, on=["user_id", "rank"], suffixes=("_p", "_o"))
    bp, bo = p.groupby("user_id").item_id.apply(set), sub.groupby("user_id").item_id.apply(set)
    print(f"vs 0.2184: pos_match={(m.item_id_p==m.item_id_o).mean():.4f}, set_ov={bp.combine(bo,lambda a,b:len(a&b)).mean():.3f}")

    sub.to_csv(args.out, index=False)
    sub.to_csv(ROOT / "Sub_Score" / "submission_optimal_postprocess.csv", index=False)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
