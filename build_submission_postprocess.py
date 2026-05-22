#!/usr/bin/env python3
"""
Build submission.csv from the best base file, with optional SAFE post-processing.

IMPORTANT (public LB 0.0114 lesson):
  Do NOT inject items from user history that are outside the base model top-10.
  That displaced ~19% of slots with past-interaction items and collapsed Recall@10.

Modes:
  restore   — copy base submission unchanged
  ensemble  — RRF within union(top-10 base + top-10 alt); recommended post-process
  safe      — re-rank ONLY within each user's existing top-10 (tiny history tie-break)
  aggressive — inject history items (public LB ~0.01; do not use)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "Sub_Score" / "submission_ (0.2184).csv"
DEFAULT_ALT = ROOT / "Sub_Score" / "submission_(0.2182).csv"
OUT_PATH = ROOT / "submission.csv"

POSITIVE_EVENTS = (
    "view_phone",
    "contact_chat",
    "contact_zalo",
    "contact_sms",
    "other_interaction",
)
EXPLICIT_EVENTS = POSITIVE_EVENTS[:-1]
POS_SQL = ", ".join(repr(x) for x in POSITIVE_EVENTS)
EXP_SQL = ", ".join(repr(x) for x in EXPLICIT_EVENTS)

# Weights for interaction table (lead >> chat >> view)
W_LEAD = 8.0
W_CHAT = 3.0
W_VIEW = 0.15
# Event-type weights (explicit contact stronger than other_interaction)
EVENT_W = {
    "view_phone": 10.0,
    "contact_chat": 9.0,
    "contact_zalo": 9.0,
    "contact_sms": 8.0,
    "other_interaction": 2.5,
}
HALF_LIFE_DAYS = 21.0
RRF_K = 60.0
W_BASE = 1.0
W_HIST = 1.35
W_EVT = 1.15
SAFE_HIST_BOOST = 0.08  # small tie-break inside base top-10 only
ENSEMBLE_W_PRIMARY = 1.0
ENSEMBLE_W_ALT = 0.95
ENSEMBLE_CONSENSUS_BONUS = 0.12  # both rankers agree on item
TOP_N = 10


def _event_weight_sql(col: str = "event_type") -> str:
    parts = [f"WHEN {col} = {repr(k)} THEN {v}" for k, v in EVENT_W.items()]
    return f"CASE {' '.join(parts)} ELSE 0 END"


def build_hist_scores(con: duckdb.DuckDBPyConnection, inter_glob: str, events_glob: str) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE hist_inter AS
        SELECT
            i.user_id,
            i.item_id,
            SUM(
                ({W_LEAD} * COALESCE(i.lead_count, 0)
                 + {W_CHAT} * COALESCE(i.chat_message_count, 0)
                 + {W_VIEW} * COALESCE(i.adview_count, 0))
                * EXP(-LN(2) * DATE_DIFF('day', i.date, DATE '2026-04-09') / {HALF_LIFE_DAYS})
            ) AS inter_score,
            MAX(i.date) AS last_inter_dt
        FROM read_parquet('{inter_glob}') i
        INNER JOIN read_parquet('{ROOT / "test/test_users.parquet"}') t
            ON i.user_id = t.user_id
        GROUP BY 1, 2
        """
    )

    ew = _event_weight_sql()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE hist_evt AS
        SELECT
            e.user_id,
            e.item_id,
            SUM(
                ({ew}) * EXP(-LN(2) * DATE_DIFF('day', e.date, DATE '2026-04-09') / {HALF_LIFE_DAYS})
            ) AS evt_score,
            MAX(e.date) AS last_evt_dt
        FROM read_parquet('{events_glob}') e
        INNER JOIN read_parquet('{ROOT / "test/test_users.parquet"}') t
            ON e.user_id = t.user_id
        WHERE e.is_login = 'login'
          AND e.event_type IN ({POS_SQL})
          AND e.item_id IS NOT NULL
        GROUP BY 1, 2
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE hist_score AS
        SELECT
            COALESCE(i.user_id, e.user_id) AS user_id,
            COALESCE(i.item_id, e.item_id) AS item_id,
            COALESCE(i.inter_score, 0) + COALESCE(e.evt_score, 0) AS hist_raw,
            GREATEST(COALESCE(i.last_inter_dt, DATE '1900-01-01'),
                     COALESCE(e.last_evt_dt, DATE '1900-01-01')) AS last_touch
        FROM hist_inter i
        FULL OUTER JOIN hist_evt e
            ON i.user_id = e.user_id AND i.item_id = e.item_id
        WHERE COALESCE(i.inter_score, 0) + COALESCE(e.evt_score, 0) > 0
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE hist_rank AS
        SELECT
            user_id,
            item_id,
            hist_raw,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY hist_raw DESC, last_touch DESC, item_id
            ) AS hist_rank
        FROM hist_score
        """
    )


def load_base_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ID", "user_id", "rank", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Base submission missing columns: {missing}")
    return df


def ensemble_two_submissions(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    k: float = RRF_K,
    w_primary: float = ENSEMBLE_W_PRIMARY,
    w_alt: float = ENSEMBLE_W_ALT,
    consensus_bonus: float = ENSEMBLE_CONSENSUS_BONUS,
) -> pd.DataFrame:
    """
    RRF over union of two strong submissions only (~12 candidates/user on average).
    Never adds items outside either model's top-10.
    """
    p = primary.copy()
    s = secondary.copy()
    p["item_id"] = p["item_id"].astype(str)
    s["item_id"] = s["item_id"].astype(str)

    pr = p.groupby(["user_id", "item_id"], as_index=False)["rank"].min()
    pr["rrf_p"] = w_primary / (k + pr["rank"].astype(float))
    pr["rank_p"] = pr["rank"]

    sr = s.groupby(["user_id", "item_id"], as_index=False)["rank"].min()
    sr["rrf_s"] = w_alt / (k + sr["rank"].astype(float))
    sr["rank_s"] = sr["rank"]

    merged = pr.merge(
        sr[["user_id", "item_id", "rrf_s", "rank_s"]],
        on=["user_id", "item_id"],
        how="outer",
    )
    merged["rrf_p"] = merged["rrf_p"].fillna(0.0)
    merged["rrf_s"] = merged["rrf_s"].fillna(0.0)
    merged["rank_p"] = merged["rank_p"].fillna(99)
    merged["rank_s"] = merged["rank_s"].fillna(99)
    both = (merged["rrf_p"] > 0) & (merged["rrf_s"] > 0)
    merged["score"] = merged["rrf_p"] + merged["rrf_s"] + consensus_bonus * both.astype(float)

    merged = merged.sort_values(
        ["user_id", "score", "rank_p", "rank_s", "item_id"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    merged["rank"] = merged.groupby("user_id", sort=False).cumcount() + 1
    out = merged.loc[merged["rank"] <= TOP_N, ["user_id", "rank", "item_id"]].copy()
    out.insert(0, "ID", range(1, len(out) + 1))
    return out


def rerank_within_base_only(
    base: pd.DataFrame,
    hist_rank: pd.DataFrame,
) -> pd.DataFrame:
    """Re-order each user's top-10 without adding or removing items."""
    hist = hist_rank.copy()
    hist["item_id"] = hist["item_id"].astype(str)
    hist_score = hist.groupby(["user_id", "item_id"], as_index=False)["hist_raw"].max()
    user_max = hist_score.groupby("user_id")["hist_raw"].transform("max")

    rows: list[dict] = []
    for user_id, grp in base.groupby("user_id", sort=False):
        g = grp.copy()
        g["item_id"] = g["item_id"].astype(str)
        merged = g.merge(
            hist_score[hist_score.user_id == user_id][["item_id", "hist_raw"]],
            on="item_id",
            how="left",
        )
        merged["hist_raw"] = merged["hist_raw"].fillna(0.0)
        hm = float(user_max.get(user_id, 0.0) or 0.0)
        merged["score"] = 1.0 / merged["rank"].astype(float)
        if hm > 0:
            merged["score"] += SAFE_HIST_BOOST * (merged["hist_raw"] / hm)
        merged = merged.sort_values(
            ["score", "rank", "item_id"], ascending=[False, True, True], kind="mergesort"
        )
        for rank, (_, row) in enumerate(merged.iterrows(), start=1):
            rows.append({"user_id": user_id, "rank": rank, "item_id": row["item_id"]})

    out = pd.DataFrame(rows)
    out.insert(0, "ID", range(1, len(out) + 1))
    return out


def rrf_merge_aggressive(
    base: pd.DataFrame,
    hist_rank: pd.DataFrame,
    valid_items: set[str],
) -> pd.DataFrame:
    """Reciprocal rank fusion — can inject history items (hurt public LB)."""
    base = base.copy()
    base["item_id"] = base["item_id"].astype(str)
    base = base[base["item_id"].isin(valid_items)]

    hist = hist_rank.copy()
    hist["item_id"] = hist["item_id"].astype(str)
    hist = hist[hist["item_id"].isin(valid_items)]

    base_rrf = base.groupby(["user_id", "item_id"], as_index=False)["rank"].min()
    base_rrf["rrf_base"] = W_BASE / (RRF_K + base_rrf["rank"].astype(float))

    if not hist.empty:
        hist_rrf = hist.groupby(["user_id", "item_id"], as_index=False).agg(
            hist_rank=("hist_rank", "min"),
            hist_raw=("hist_raw", "max"),
        )
        hist_rrf["rrf_hist"] = W_HIST / (RRF_K + hist_rrf["hist_rank"].astype(float))
        user_max = hist_rrf.groupby("user_id")["hist_raw"].transform("max")
        hist_rrf["rrf_raw"] = W_EVT * (hist_rrf["hist_raw"] / (user_max + 1e-9))
    else:
        hist_rrf = pd.DataFrame(
            columns=["user_id", "item_id", "hist_rank", "hist_raw", "rrf_hist", "rrf_raw"]
        )

    merged = base_rrf.merge(
        hist_rrf[["user_id", "item_id", "hist_raw", "rrf_hist", "rrf_raw"]],
        on=["user_id", "item_id"],
        how="outer",
    )
    for c in ("rrf_base", "rrf_hist", "rrf_raw", "hist_raw"):
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)
    merged["score"] = merged["rrf_base"] + merged["rrf_hist"] + merged["rrf_raw"]

    rows: list[dict] = []
    for user_id, grp in merged.groupby("user_id", sort=False):
        g = grp.sort_values(
            ["score", "hist_raw", "item_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        seen: set[str] = set()
        rank = 1
        for _, row in g.iterrows():
            it = row["item_id"]
            if it in seen:
                continue
            seen.add(it)
            rows.append({"user_id": user_id, "rank": rank, "item_id": it})
            rank += 1
            if rank > TOP_N:
                break

        if rank <= TOP_N:
            pad = base[(base.user_id == user_id) & (~base.item_id.isin(seen))].sort_values("rank")
            for _, prow in pad.iterrows():
                it = prow["item_id"]
                if it in seen:
                    continue
                seen.add(it)
                rows.append({"user_id": user_id, "rank": rank, "item_id": it})
                rank += 1
                if rank > TOP_N:
                    break

    out = pd.DataFrame(rows)
    out.insert(0, "ID", range(1, len(out) + 1))
    return out


def postprocess_dedupe_seller(
    sub: pd.DataFrame,
    seller_map: dict[str, str],
    max_per_seller: int = 2,
) -> pd.DataFrame:
    """Optional: cap items per seller to improve diversity (marketplace health)."""
    rows: list[dict] = []
    for user_id, grp in sub.groupby("user_id", sort=False):
        g = grp.sort_values("rank")
        kept: list[dict] = []
        deferred: list[dict] = []
        seller_counts: dict[str, int] = {}
        for _, row in g.iterrows():
            sid = seller_map.get(row["item_id"])
            if sid is None:
                kept.append(row.to_dict())
                continue
            c = seller_counts.get(sid, 0)
            if c < max_per_seller:
                kept.append(row.to_dict())
                seller_counts[sid] = c + 1
            else:
                deferred.append(row.to_dict())
        final = kept + deferred
        for i, d in enumerate(final[:TOP_N], start=1):
            d["rank"] = i
            d["user_id"] = user_id
            rows.append(d)
    out = pd.DataFrame(rows)
    out.insert(0, "ID", range(1, len(out) + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--mode",
        choices=("restore", "ensemble", "safe", "aggressive"),
        default="ensemble",
        help="ensemble=RRF two good subs (recommended); restore=copy base; safe/aggressive=see docstring",
    )
    parser.add_argument("--alt", type=Path, default=DEFAULT_ALT, help="Second submission for ensemble mode")
    parser.add_argument("--seller-cap", type=int, default=0, help="0=disable seller diversity cap")
    args = parser.parse_args()

    print("Loading base submission:", args.base)
    base = load_base_submission(args.base)
    test_users = pd.read_parquet(ROOT / "test" / "test_users.parquet")
    assert set(base.user_id.unique()) == set(test_users.user_id)

    if args.mode == "restore":
        sub = base.copy()
        sub["ID"] = range(1, len(sub) + 1)
        sub = sub[["ID", "user_id", "rank", "item_id"]]
        print("Mode restore: writing base submission unchanged.")
    elif args.mode == "ensemble":
        alt = load_base_submission(args.alt)
        assert set(alt.user_id.unique()) == set(test_users.user_id)
        print("Mode ensemble:", args.base.name, "+", args.alt.name)
        sub = ensemble_two_submissions(base, alt)
    else:
        con = duckdb.connect(":memory:")
        con.execute("PRAGMA memory_limit='5GB'")
        con.execute("PRAGMA threads=4")

        inter_glob = str(ROOT / "fact_post_contact_interactions" / "*.parquet")
        events_glob = str(ROOT / "fact_user_events" / "*.parquet")
        dim_glob = str(ROOT / "dim_listing" / "*.parquet")

        print("Building history scores (interactions + events)…")
        build_hist_scores(con, inter_glob, events_glob)
        hist_rank = con.execute(
            "SELECT user_id, item_id, hist_raw, hist_rank FROM hist_rank WHERE hist_rank <= 50"
        ).df()
        n_users_hist = con.execute("SELECT COUNT(DISTINCT user_id) FROM hist_rank").fetchone()[0]
        print(f"Users with history signal: {n_users_hist:,}")

        valid_items = set(
            con.execute(f"SELECT item_id FROM read_parquet('{dim_glob}')").df()["item_id"].astype(str)
        )
        print(f"Valid listings: {len(valid_items):,}")

        if args.mode == "safe":
            print("Mode safe: re-rank within base top-10 only…")
            sub = rerank_within_base_only(base, hist_rank)
        else:
            print("Mode aggressive: RRF with history injection (public LB ~0.01)…")
            sub = rrf_merge_aggressive(base, hist_rank, valid_items)

    # Ensure all test users present
    have = set(sub.user_id.unique())
    need = set(test_users.user_id) - have
    if need:
        extra = base[base.user_id.isin(need)].copy()
        sub = pd.concat([sub, extra], ignore_index=True)
        sub["ID"] = range(1, len(sub) + 1)

    if args.seller_cap > 0:
        print(f"Applying seller cap (max {args.seller_cap} per user)…")
        con = duckdb.connect(":memory:")
        dim_glob = str(ROOT / "dim_listing" / "*.parquet")
        seller_map = (
            con.execute(f"SELECT item_id, seller_id FROM read_parquet('{dim_glob}')")
            .df()
            .set_index("item_id")["seller_id"]
            .astype(str)
            .to_dict()
        )
        sub = postprocess_dedupe_seller(sub, seller_map, max_per_seller=args.seller_cap)

    # Validation
    assert len(sub) == len(test_users) * TOP_N
    assert sub.groupby("user_id").size().eq(TOP_N).all()
    assert not sub.duplicated(["user_id", "item_id"]).any()
    assert not sub.duplicated(["user_id", "rank"]).any()
    dim_glob = str(ROOT / "dim_listing" / "*.parquet")
    valid_items = set(
        duckdb.connect(":memory:")
        .execute(f"SELECT item_id FROM read_parquet('{dim_glob}')")
        .df()["item_id"]
        .astype(str)
    )
    invalid = ~sub["item_id"].astype(str).isin(valid_items)
    assert not invalid.any(), f"{invalid.sum()} invalid item_ids"

    sub.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows=", len(sub))


if __name__ == "__main__":
    main()
