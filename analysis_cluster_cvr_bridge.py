#!/usr/bin/env python3
"""
Bridge clustering outputs → listing-level CVR (dim + events).

RAM-safe design:
  - DuckDB (default 2GB) for joins; CSV/parquet via read_* without full pandas loads.
  - Events scan restricted to scoped item_id / user_id from clustering CSVs.
  - Session funnel: chunked pandas on 04_session_journey_features.csv only.

Run from Datathon_Data:
  ./env/bin/python analysis_cluster_cvr_bridge.py
  ./env/bin/python analysis_cluster_cvr_bridge.py --session-chunk 40000 --memory 1500MB
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Config (tune here if OOM)
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent
CLUSTER_DIR = DATA_ROOT / "outputs" / "eda_category_1010_1020" / "clustering"
PERF_DIR = DATA_ROOT / "outputs" / "eda_category_1010_1020"
OUT_DIR = DATA_ROOT / "outputs" / "eda_category_1010_1020" / "bridge"

DUCKDB_MEMORY_LIMIT = "2GB"
DUCKDB_THREADS = 2
SESSION_CHUNK_ROWS = 50_000
MIN_GROUP_N = 30

POSITIVE_TYPES = (
    "view_phone",
    "contact_chat",
    "other_interaction",
    "contact_zalo",
    "contact_sms",
)
CAT_SQL = "1010, 1020"
POS_SQL = ", ".join(repr(x) for x in POSITIVE_TYPES)
UI_LABELS = {1010: "Căn hộ / Chung cư", 1020: "Nhà ở"}


def log(msg: str) -> None:
    print(msg, flush=True)


def cvr_agg_sql(group_cols: str) -> str:
    return f"""
    SELECT
        {group_cols},
        COUNT(*)::BIGINT AS n,
        SUM(CASE WHEN has_positive THEN 1 ELSE 0 END)::BIGINT AS n_positive,
        ROUND(
            100.0 * SUM(CASE WHEN has_positive THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
            2
        ) AS cvr_pct
    FROM joined
    GROUP BY {group_cols}
    HAVING COUNT(*) >= {MIN_GROUP_N}
    ORDER BY {group_cols}
    """


def layer_register_cluster_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Layer 0 — register clustering CSVs in DuckDB (no pandas full load)."""
    log("Layer 0: register clustering CSVs …")
    paths = {
        "health": CLUSTER_DIR / "20_marketplace_health_segments.csv",
        "listing_cl": CLUSTER_DIR / "11_listing_clusters.csv",
        "user_cl": CLUSTER_DIR / "10_user_clusters.csv",
        "session_cl": CLUSTER_DIR / "13_session_clusters.csv",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — chạy eda_category_1010_1020_clustering.ipynb trước.")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT * FROM read_csv_auto(?, header=true)
            """,
            [str(p)],
        )

    con.execute("""
        CREATE OR REPLACE TABLE listing_bridge AS
        SELECT
            h.item_id,
            h.category,
            h.ad_type,
            h.health_segment,
            h.district_name,
            h.seller_type,
            h.contact_rate_pct AS event_contact_rate_pct,
            h.exposure,
            h.exposure_pct_rank,
            h.contact_pct_rank,
            h.n_pageviews,
            h.n_unique_users,
            h.repeat_viewer_pct,
            h.median_age_days,
            COALESCE(c.cluster_id, -1) AS cluster_id
        FROM health h
        LEFT JOIN listing_cl c
          ON h.item_id = c.item_id AND h.category = c.category
    """)

    con.execute("""
        CREATE OR REPLACE TABLE scope_items AS
        SELECT DISTINCT item_id FROM listing_bridge
    """)
    con.execute("""
        CREATE OR REPLACE TABLE scope_users AS
        SELECT DISTINCT user_id, category, cluster_id
        FROM user_cl
    """)
    n_items = con.execute("SELECT COUNT(*) FROM scope_items").fetchone()[0]
    n_users = con.execute("SELECT COUNT(*) FROM scope_users").fetchone()[0]
    log(f"  scope_items={n_items:,}  scope_users={n_users:,}")


def layer_pos_items(
    con: duckdb.DuckDBPyConnection,
    events_glob: str,
    event_sample_frac: float | None,
) -> None:
    """Layer 1 — positive item_id (semi-join scope only)."""
    sample = ""
    if event_sample_frac is not None:
        if not (0 < event_sample_frac < 1):
            raise ValueError("event_sample_frac must be in (0,1)")
        sample = f"AND random() < {float(event_sample_frac)}"
        log(f"Layer 1: pos_items (sample {event_sample_frac}) …")
    else:
        log("Layer 1: pos_items (full events, scoped item_id) …")

    con.execute(f"""
        CREATE OR REPLACE TABLE pos_items AS
        SELECT DISTINCT e.item_id
        FROM read_parquet('{events_glob}') e
        WHERE e.category IN ({CAT_SQL})
          AND e.event_type IN ({POS_SQL})
          AND e.item_id IN (SELECT item_id FROM scope_items)
          {sample}
    """)
    n = con.execute("SELECT COUNT(*) FROM pos_items").fetchone()[0]
    log(f"  pos_items={n:,}")
    gc.collect()


def layer_pos_users(
    con: duckdb.DuckDBPyConnection,
    events_glob: str,
    event_sample_frac: float | None,
) -> None:
    """Layer 1b — users with ≥1 positive event (scoped users)."""
    sample = ""
    if event_sample_frac is not None:
        sample = f"AND random() < {float(event_sample_frac)}"

    log("Layer 1b: pos_users (scoped user_id) …")
    con.execute(f"""
        CREATE OR REPLACE TABLE pos_users AS
        SELECT DISTINCT e.user_id, CAST(e.category AS INTEGER) AS category
        FROM read_parquet('{events_glob}') e
        INNER JOIN scope_users s
          ON e.user_id = s.user_id AND CAST(e.category AS INTEGER) = s.category
        WHERE e.category IN ({CAT_SQL})
          AND e.event_type IN ({POS_SQL})
          {sample}
    """)
    n = con.execute("SELECT COUNT(*) FROM pos_users").fetchone()[0]
    log(f"  pos_users={n:,}")
    gc.collect()


def layer_listing_cvr(con: duckdb.DuckDBPyConnection, dim_glob: str) -> None:
    """Layer 2 — listing-level CVR × health × cluster (dim truth)."""
    log("Layer 2: listing CVR bridge …")
    con.execute(f"""
        CREATE OR REPLACE TABLE dim_scoped AS
        SELECT
            CAST(d.item_id AS VARCHAR) AS item_id,
            CAST(d.category AS INTEGER) AS category,
            d.ad_type
        FROM read_parquet('{dim_glob}') d
        WHERE d.category IN ({CAT_SQL})
          AND d.item_id IN (SELECT item_id FROM scope_items)
    """)

    con.execute("""
        CREATE OR REPLACE TABLE joined AS
        SELECT
            b.*,
            p.item_id IS NOT NULL AS has_positive
        FROM listing_bridge b
        LEFT JOIN pos_items p ON b.item_id = p.item_id
    """)

    for tag, group_cols in (
        ("01_cvr_by_health_segment", "category, health_segment"),
        ("02_cvr_by_listing_cluster", "category, cluster_id"),
        ("02b_cvr_health_x_cluster", "category, health_segment, cluster_id"),
        ("02c_cvr_by_adtype_health", "category, ad_type, health_segment"),
    ):
        df = con.execute(cvr_agg_sql(group_cols)).df()
        path = OUT_DIR / f"{tag}.csv"
        df.to_csv(path, index=False)
        log(f"  wrote {path.name} ({len(df)} rows)")

    gc.collect()


def layer_user_cvr(con: duckdb.DuckDBPyConnection) -> None:
    """Layer 3 — user positive rate by cluster (scoped clustered users)."""
    log("Layer 3: user positive rate by cluster …")
    con.execute("""
        CREATE OR REPLACE TABLE user_joined AS
        SELECT
            s.user_id,
            s.category,
            s.cluster_id,
            p.user_id IS NOT NULL AS has_positive
        FROM scope_users s
        LEFT JOIN pos_users p
          ON s.user_id = p.user_id AND s.category = p.category
    """)

    df = con.execute(cvr_agg_sql("category, cluster_id")).df()
    df["metric"] = "user_positive_rate_pct"
    df.to_csv(OUT_DIR / "03_user_positive_by_cluster.csv", index=False)

    # Named clusters only (exclude noise -1 for interpretability)
    df_named = df[df["cluster_id"] != -1].copy()
    df_named.to_csv(OUT_DIR / "03b_user_positive_named_clusters.csv", index=False)
    log(f"  wrote user cluster tables ({len(df)} rows)")
    gc.collect()


def layer_event_efficiency(con: duckdb.DuckDBPyConnection) -> None:
    """Layer 3b — event efficiency (cohort đã có footprint; CVR listing ≈100%)."""
    log("Layer 3b: event efficiency by health / cluster …")
    eff = con.execute("""
        SELECT
            category,
            health_segment,
            COUNT(*)::BIGINT AS n,
            ROUND(AVG(event_contact_rate_pct), 2) AS avg_event_contact_rate_pct,
            ROUND(MEDIAN(event_contact_rate_pct), 2) AS med_event_contact_rate_pct,
            ROUND(AVG(exposure), 2) AS avg_exposure,
            ROUND(MEDIAN(exposure), 2) AS med_exposure,
            ROUND(AVG(n_pageviews), 2) AS avg_pageviews,
            ROUND(AVG(n_unique_users), 2) AS avg_unique_users,
            ROUND(AVG(repeat_viewer_pct), 2) AS avg_repeat_viewer_pct,
            ROUND(AVG(median_age_days), 1) AS avg_median_age_days
        FROM listing_bridge
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    eff.to_csv(OUT_DIR / "04_segment_event_efficiency.csv", index=False)

    cl = con.execute("""
        SELECT
            category,
            cluster_id,
            COUNT(*)::BIGINT AS n,
            ROUND(AVG(event_contact_rate_pct), 2) AS avg_event_contact_rate_pct,
            ROUND(AVG(exposure), 2) AS avg_exposure,
            ROUND(AVG(n_pageviews), 2) AS avg_pageviews
        FROM listing_bridge
        GROUP BY 1, 2
        HAVING COUNT(*) >= 30
        ORDER BY 1, 2
    """).df()
    cl.to_csv(OUT_DIR / "04b_listing_cluster_event_efficiency.csv", index=False)
    gc.collect()


def layer_health_ranked(con: duckdb.DuckDBPyConnection, top_n: int = 500) -> None:
    """Layer 4 — top underexposed / oversaturated with dim CVR flag."""
    log("Layer 4: marketplace health ranked …")
    for cat in (1010, 1020):
        q = f"""
        SELECT
            b.item_id,
            b.category,
            b.ad_type,
            b.health_segment,
            b.district_name,
            b.seller_type,
            b.event_contact_rate_pct,
            b.exposure,
            b.exposure_pct_rank,
            b.contact_pct_rank,
            b.cluster_id,
            b.has_positive,
            (b.contact_pct_rank - b.exposure_pct_rank) AS lift_score
        FROM joined b
        WHERE b.category = {cat}
          AND b.health_segment = 'high_quality_underexposed'
        ORDER BY lift_score DESC
        LIMIT {top_n}
        """
        df = con.execute(q).df()
        df["ui_label"] = UI_LABELS[cat]
        path = OUT_DIR / f"10_health_ranked_underexposed_{cat}.csv"
        df.to_csv(path, index=False)
        log(f"  wrote {path.name} ({len(df)} rows)")

    seg = con.execute("""
        SELECT
            category,
            health_segment,
            COUNT(*)::BIGINT AS n,
            ROUND(100.0 * SUM(CASE WHEN has_positive THEN 1 ELSE 0 END) / COUNT(*), 2) AS cvr_pct
        FROM joined
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    seg.to_csv(OUT_DIR / "04_cvr_summary_health_segment.csv", index=False)
    gc.collect()


def layer_session_funnel_chunked(
    session_path: Path,
    session_cl_path: Path,
    chunk_rows: int,
    bounce_dwell_max: float,
) -> None:
    """Layer 5 — session funnel by cluster; chunked read (~110MB file)."""
    log(f"Layer 5: session funnel (chunksize={chunk_rows:,}) …")
    usecols = [
        "session_id",
        "category",
        "has_contact",
        "has_search",
        "deep_compare",
        "n_pageviews",
        "avg_dwell_sec",
    ]
    clusters = pd.read_csv(
        session_cl_path,
        usecols=["session_id", "category", "cluster_id"],
    )
    clusters["session_id"] = clusters["session_id"].astype(str)

    # cluster_id -> aggregated sums
    acc: dict[tuple, dict] = defaultdict(
        lambda: {
            "n": 0,
            "has_contact": 0.0,
            "has_search": 0.0,
            "deep_compare": 0.0,
            "bounce_v2": 0.0,
        }
    )

    n_rows = 0
    for i, chunk in enumerate(
        pd.read_csv(session_path, usecols=usecols, chunksize=chunk_rows, low_memory=False)
    ):
        n_rows += len(chunk)
        chunk = chunk.merge(clusters, on=["session_id", "category"], how="left")
        chunk["cluster_id"] = chunk["cluster_id"].fillna(-1).astype(int)
        dwell = chunk["avg_dwell_sec"].fillna(9999.0)
        chunk["bounce_v2"] = (
            (chunk["n_pageviews"] >= 2)
            & (chunk["has_contact"].fillna(0) == 0)
            & (dwell < bounce_dwell_max)
        ).astype("int8")

        for (cat, cid), g in chunk.groupby(["category", "cluster_id"], observed=True):
            key = (int(cat), int(cid))
            acc[key]["n"] += len(g)
            acc[key]["has_contact"] += g["has_contact"].fillna(0).sum()
            acc[key]["has_search"] += g["has_search"].fillna(0).sum()
            acc[key]["deep_compare"] += g["deep_compare"].sum()
            acc[key]["bounce_v2"] += g["bounce_v2"].sum()

        if (i + 1) % 5 == 0:
            log(f"  … processed {n_rows:,} session rows")
        del chunk
        gc.collect()

    rows = []
    for (cat, cid), v in sorted(acc.items()):
        n = v["n"]
        if n < MIN_GROUP_N:
            continue
        rows.append(
            {
                "category": cat,
                "cluster_id": cid,
                "n_sessions": n,
                "rate_has_contact": round(v["has_contact"] / n, 4),
                "rate_has_search": round(v["has_search"] / n, 4),
                "rate_deep_compare": round(v["deep_compare"] / n, 4),
                "rate_bounce_v2": round(v["bounce_v2"] / n, 4),
            }
        )
    funnel = pd.DataFrame(rows)
    funnel.to_csv(OUT_DIR / "05_session_funnel_by_cluster.csv", index=False)
    log(f"  wrote 05_session_funnel_by_cluster.csv ({len(funnel)} groups, {n_rows:,} sessions)")

    overall = pd.DataFrame(
        [
            {
                "metric": "all_sessions_in_file",
                "n": n_rows,
                "bounce_dwell_max_sec": bounce_dwell_max,
            }
        ]
    )
    overall.to_csv(OUT_DIR / "05b_session_funnel_meta.csv", index=False)
    del clusters, acc
    gc.collect()


def layer_insights(con: duckdb.DuckDBPyConnection) -> None:
    """Layer 6 — compact JSON + markdown summary."""
    log("Layer 6: insights summary …")
    health = pd.read_csv(OUT_DIR / "04_cvr_summary_health_segment.csv")
    eff_path = OUT_DIR / "04_segment_event_efficiency.csv"
    eff = pd.read_csv(eff_path) if eff_path.exists() else None
    baseline = pd.read_csv(PERF_DIR / "02_cvr_baseline_adtype.csv")

    bullets: list[str] = []
    bullets.append("# Bridge insights — clustering × performance\n")
    bullets.append(
        "**Lưu ý cohort:** `20_marketplace_health` chỉ gồm tin có event trong sample clustering "
        "→ CVR listing (≥1 positive) trên cohort này ~100%. So sánh segment bằng **event efficiency** "
        "(contact/pageview, exposure), và session funnel.\n"
    )
    bullets.append("**CVR catalog (dim toàn bộ):** xem `outputs/eda_category_1010_1020/02_cvr_baseline_adtype.csv`.\n\n")

    if eff is not None:
        bullets.append("## Event efficiency theo health_segment\n")
        for _, r in eff.iterrows():
            bullets.append(
                f"- **{int(r['category'])}** `{r['health_segment']}`: n={int(r['n']):,}, "
                f"avg contact/pageview={r['avg_event_contact_rate_pct']}%, "
                f"med exposure={r['med_exposure']}\n"
            )

    bullets.append("\n## CVR listing trên cohort clustering (tham khảo)\n")
    for _, r in health.iterrows():
        bullets.append(
            f"- **{int(r['category'])}** `{r['health_segment']}`: n={int(r['n']):,}, CVR={r['cvr_pct']}%\n"
        )

    bullets.append("\n## Baseline dim (toàn catalog)\n")
    for _, r in baseline.iterrows():
        bullets.append(
            f"- **{int(r['category'])}** {r['ad_type']}: CVR={r['cvr_pct']}% (n={int(r['listings']):,})\n"
        )

    funnel_path = OUT_DIR / "05_session_funnel_by_cluster.csv"
    if funnel_path.exists():
        f = pd.read_csv(funnel_path)
        bullets.append("\n## Session funnel (cluster_id=-1 vs search-heavy)\n")
        for cat in (1010, 1020):
            noise = f[(f.category == cat) & (f.cluster_id == -1)].iloc[0]
            bullets.append(
                f"- **{cat}** noise: contact={noise['rate_has_contact']:.1%}, "
                f"search={noise['rate_has_search']:.1%}, deep_compare={noise['rate_deep_compare']:.1%}\n"
            )
        searchy = f[(f.rate_has_search >= 0.9) & (f.n_sessions >= 100)].head(3)
        for _, r in searchy.iterrows():
            bullets.append(
                f"- cat={int(r['category'])} cl={int(r['cluster_id'])}: search={r['rate_has_search']:.0%}, "
                f"contact={r['rate_has_contact']:.0%}, deep_compare={r['rate_deep_compare']:.0%}\n"
            )

    meta = {
        "cluster_dir": str(CLUSTER_DIR),
        "out_dir": str(OUT_DIR),
        "duckdb_memory": DUCKDB_MEMORY_LIMIT,
    }
    (OUT_DIR / "00_bridge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT_DIR / "00_bridge_insights.md").write_text("".join(bullets), encoding="utf-8")
    log("  wrote 00_bridge_insights.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering ↔ CVR bridge (RAM-safe)")
    parser.add_argument("--memory", default=DUCKDB_MEMORY_LIMIT, help="DuckDB memory_limit")
    parser.add_argument("--threads", type=int, default=DUCKDB_THREADS)
    parser.add_argument(
        "--event-sample",
        type=float,
        default=None,
        help="Optional random sample on events (0<frac<1) if OOM on Layer 1",
    )
    parser.add_argument("--session-chunk", type=int, default=SESSION_CHUNK_ROWS)
    parser.add_argument(
        "--bounce-dwell-max",
        type=float,
        default=60.0,
        help="bounce_v2: avg_dwell_sec < this (pageview sessions)",
    )
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    dim_glob = str(DATA_ROOT / "dim_listing" / "*.parquet")
    events_glob = str(DATA_ROOT / "fact_user_events" / "*.parquet")
    for name in ("dim_listing", "fact_user_events"):
        if not (DATA_ROOT / name).exists():
            raise FileNotFoundError(f"Thiếu `{name}` trong {DATA_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{args.memory}'")
    con.execute(f"PRAGMA threads={args.threads}")

    layer_register_cluster_tables(con)
    layer_pos_items(con, events_glob, args.event_sample)
    layer_pos_users(con, events_glob, args.event_sample)
    layer_listing_cvr(con, dim_glob)
    layer_user_cvr(con)
    layer_event_efficiency(con)
    layer_health_ranked(con, top_n=args.top_n)

    session_path = CLUSTER_DIR / "04_session_journey_features.csv"
    session_cl_path = CLUSTER_DIR / "13_session_clusters.csv"
    if session_path.exists() and session_cl_path.exists():
        layer_session_funnel_chunked(
            session_path,
            session_cl_path,
            chunk_rows=args.session_chunk,
            bounce_dwell_max=args.bounce_dwell_max,
        )
    else:
        log("Layer 5 skipped (missing session CSVs)")

    layer_insights(con)
    con.close()
    log(f"Done. Outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
