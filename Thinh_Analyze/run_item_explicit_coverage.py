"""
EDA: item_id × đủ 4 explicit event_type × contact rate.

Run: env/bin/python Thinh_Analyze/run_item_explicit_coverage.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import matplotlib

_bk = matplotlib.get_backend().lower()
if "inline" not in _bk and "agg" not in _bk:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.ioff()

DATA_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "item_explicit_coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
SNAP_GLOB = str(DATA_ROOT / "fact_listing_snapshot" / "*.parquet")
INTER_GLOB = str(DATA_ROOT / "fact_post_contact_interactions" / "*.parquet")

DUCKDB_MEMORY_LIMIT = "3GB"
DUCKDB_THREADS = 2
# None = full scan (cần RAM lớn); 0.25 ≈ 25% session — giữ nguyên contact rate trong session
SESSION_SAMPLE_FRAC: float | None = 0.25

ACTIVE_AD_STATUS = ("accepted", "hidden", "shop_accepted")
ACTIVE_AD_SQL = ", ".join(repr(x) for x in ACTIVE_AD_STATUS)

CAT_IN = "1010, 1020, 1030, 1040, 1050"
CAT_META = {
    1010: "1010 — Căn hộ / Chung cư",
    1020: "1020 — Nhà ở",
    1030: "1030 — VP / Mặt bằng",
    1040: "1040 — Đất",
    1050: "1050 — Phòng trọ",
}
CATEGORIES = tuple(CAT_META)

EXPLICIT_TYPES = ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
EXPLICIT_SQL = ", ".join(repr(x) for x in EXPLICIT_TYPES)

con: duckdb.DuckDBPyConnection | None = None
eda_min: object = None
eda_max: object = None


def session_sample_clause(frac: float | None) -> str:
    if frac is None:
        return ""
    if not (0 < float(frac) < 1):
        raise ValueError("SESSION_SAMPLE_FRAC must be in (0, 1)")
    bucket = max(1, int(float(frac) * 1000))
    return f"AND (abs(hash(CAST(session_id AS VARCHAR))) % 1000) < {bucket}"


def _finish_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def coverage_group_expr(prefix: str = "") -> str:
    p = f"{prefix}." if prefix else ""
    return f"""
        CASE
            WHEN COALESCE({p}n_explicit_types, 0) = 4 THEN 'full_four'
            WHEN COALESCE({p}n_explicit_types, 0) BETWEEN 1 AND 3 THEN 'partial_1_3'
            WHEN COALESCE({p}has_any_event, 0) = 1 AND COALESCE({p}n_explicit_types, 0) = 0 THEN 'zero_explicit'
            ELSE 'no_events'
        END
    """


def init_db() -> tuple[object, object]:
    global con, eda_min, eda_max
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")

    row = con.execute(f"""
    SELECT
        (SELECT MIN(date) FROM read_parquet('{EVENTS_GLOB}')) AS events_min,
        (SELECT MAX(date) FROM read_parquet('{EVENTS_GLOB}')) AS events_max,
        (SELECT MIN(date) FROM read_parquet('{SNAP_GLOB}')) AS snap_min,
        (SELECT MAX(date) FROM read_parquet('{SNAP_GLOB}')) AS snap_max
    """).fetchone()
    eda_min, eda_max = row[0], row[1]
    print(f"EDA window (events): {eda_min} → {eda_max}")
    print(f"Snapshot window: {row[2]} → {row[3]}")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE dim_all AS
    SELECT
        CAST(item_id AS VARCHAR) AS item_id,
        category, seller_type, ad_type, ad_status,
        title, area_sqm, bedrooms, bathrooms, images_count,
        city_name, district_name, price_bucket, posted_date,
        CASE
            WHEN posted_date IS NULL THEN 'unknown_posted'
            WHEN posted_date < DATE '{eda_min}' THEN 'pre_eda_window'
            WHEN posted_date > DATE '{eda_max}' THEN 'post_eda_window'
            ELSE 'in_eda_window'
        END AS posted_cohort
    FROM read_parquet('{DIM_GLOB}')
    WHERE category IN ({CAT_IN})
    """)

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE dim_active AS
    SELECT * FROM dim_all
    WHERE ad_status IN ({ACTIVE_AD_SQL})
      AND posted_cohort = 'in_eda_window'
    """)
    n_active = con.execute("SELECT COUNT(*) FROM dim_active").fetchone()[0]
    print(f"dim_active (ad_status active + in_eda_window): {n_active:,}")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE item_event_flags AS
    SELECT
        CAST(item_id AS VARCHAR) AS item_id,
        1 AS has_any_event,
        MAX(CASE WHEN event_type = 'view_phone' THEN 1 ELSE 0 END) AS has_view_phone,
        MAX(CASE WHEN event_type = 'contact_chat' THEN 1 ELSE 0 END) AS has_contact_chat,
        MAX(CASE WHEN event_type = 'contact_zalo' THEN 1 ELSE 0 END) AS has_contact_zalo,
        MAX(CASE WHEN event_type = 'contact_sms' THEN 1 ELSE 0 END) AS has_contact_sms,
        SUM(CASE WHEN event_type = 'view_phone' THEN 1 ELSE 0 END)::BIGINT AS n_view_phone,
        SUM(CASE WHEN event_type = 'contact_chat' THEN 1 ELSE 0 END)::BIGINT AS n_contact_chat,
        SUM(CASE WHEN event_type = 'contact_zalo' THEN 1 ELSE 0 END)::BIGINT AS n_contact_zalo,
        SUM(CASE WHEN event_type = 'contact_sms' THEN 1 ELSE 0 END)::BIGINT AS n_contact_sms,
        SUM(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END)::BIGINT AS n_pageview,
        (
            MAX(CASE WHEN event_type = 'view_phone' THEN 1 ELSE 0 END)
            + MAX(CASE WHEN event_type = 'contact_chat' THEN 1 ELSE 0 END)
            + MAX(CASE WHEN event_type = 'contact_zalo' THEN 1 ELSE 0 END)
            + MAX(CASE WHEN event_type = 'contact_sms' THEN 1 ELSE 0 END)
        )::INT AS n_explicit_types
    FROM read_parquet('{EVENTS_GLOB}')
    WHERE item_id IS NOT NULL
      AND date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
      AND category IN ({CAT_IN})
    GROUP BY 1
    """)

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE item_coverage AS
    SELECT
        d.item_id,
        d.category,
        d.ad_status,
        d.seller_type,
        d.ad_type,
        d.title,
        d.area_sqm,
        d.images_count,
        d.city_name,
        d.price_bucket,
        d.posted_date,
        COALESCE(e.has_any_event, 0) AS has_any_event,
        COALESCE(e.has_view_phone, 0) AS has_view_phone,
        COALESCE(e.has_contact_chat, 0) AS has_contact_chat,
        COALESCE(e.has_contact_zalo, 0) AS has_contact_zalo,
        COALESCE(e.has_contact_sms, 0) AS has_contact_sms,
        COALESCE(e.n_explicit_types, 0) AS n_explicit_types,
        COALESCE(e.n_view_phone, 0) AS n_view_phone,
        COALESCE(e.n_contact_chat, 0) AS n_contact_chat,
        COALESCE(e.n_contact_zalo, 0) AS n_contact_zalo,
        COALESCE(e.n_contact_sms, 0) AS n_contact_sms,
        COALESCE(e.n_pageview, 0) AS n_pageview,
        {coverage_group_expr()} AS coverage_group,
        CASE WHEN COALESCE(e.n_explicit_types, 0) = 4 THEN 1 ELSE 0 END AS is_full_four
    FROM dim_active d
    LEFT JOIN item_event_flags e ON d.item_id = e.item_id
    """)
    return eda_min, eda_max


def section_0() -> pd.DataFrame:
    time_cov = con.execute(f"""
    SELECT * FROM (
        SELECT 'dim_listing.posted_date (all 5 cat)' AS src,
               MIN(posted_date), MAX(posted_date), COUNT(*)::BIGINT
        FROM dim_all
        UNION ALL
        SELECT 'dim_active (ad_status active + in_eda_window)',
               MIN(posted_date), MAX(posted_date), COUNT(*)::BIGINT
        FROM dim_active
        UNION ALL
        SELECT 'fact_user_events.date', MIN(date), MAX(date), COUNT(*)::BIGINT
        FROM read_parquet('{EVENTS_GLOB}')
        UNION ALL
        SELECT 'fact_listing_snapshot.date', MIN(date), MAX(date), COUNT(*)::BIGINT
        FROM read_parquet('{SNAP_GLOB}')
    ) ORDER BY 1
    """).df()
    time_cov.columns = ["src", "t_min", "t_max", "n_rows"]
    time_cov.to_csv(OUT_DIR / "00_time_coverage.csv", index=False)

    overlap = con.execute("""
    SELECT ad_status, posted_cohort, COUNT(*)::BIGINT AS n
    FROM dim_all
    WHERE ad_status IN (""" + ACTIVE_AD_SQL + """)
    GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    overlap.to_csv(OUT_DIR / "00_dim_active_overlap.csv", index=False)
    print("§0 time coverage OK")
    return time_cov


def section_1() -> None:
    overall = con.execute(f"""
    SELECT coverage_group,
           COUNT(*)::BIGINT AS n_listings,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct
    FROM item_coverage
    GROUP BY 1 ORDER BY 1
    """).df()
    overall.to_csv(OUT_DIR / "01_coverage_overall.csv", index=False)

    full_vs_not = con.execute("""
    SELECT
        CASE WHEN n_explicit_types = 4 THEN 'full_four' ELSE 'not_full_four' END AS bucket,
        COUNT(*)::BIGINT AS n_listings,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct
    FROM item_coverage
    GROUP BY 1
    """).df()
    full_vs_not.to_csv(OUT_DIR / "01_coverage_full_vs_not.csv", index=False)

    by_cat = con.execute("""
    SELECT category, coverage_group,
           COUNT(*)::BIGINT AS n_listings,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY category), 3) AS pct_within_cat
    FROM item_coverage
    GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    by_cat.to_csv(OUT_DIR / "01_coverage_by_category.csv", index=False)

    by_status = con.execute("""
    SELECT ad_status, coverage_group,
           COUNT(*)::BIGINT AS n_listings,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY ad_status), 3) AS pct_within_status
    FROM item_coverage
    GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    by_status.to_csv(OUT_DIR / "01_coverage_by_ad_status.csv", index=False)

    missing = con.execute("""
    WITH partial AS (
        SELECT * FROM item_coverage WHERE coverage_group = 'partial_1_3'
    )
    SELECT 'missing_view_phone' AS missing_channel, COUNT(*)::BIGINT AS n
    FROM partial WHERE has_view_phone = 0
    UNION ALL SELECT 'missing_contact_chat', COUNT(*) FROM partial WHERE has_contact_chat = 0
    UNION ALL SELECT 'missing_contact_zalo', COUNT(*) FROM partial WHERE has_contact_zalo = 0
    UNION ALL SELECT 'missing_contact_sms', COUNT(*) FROM partial WHERE has_contact_sms = 0
    ORDER BY 2 DESC
    """).df()
    missing.to_csv(OUT_DIR / "02_missing_channel_counts.csv", index=False)

    # Stacked bar by category
    pivot = by_cat.pivot(index="category", columns="coverage_group", values="pct_within_cat").fillna(0)
    col_order = [c for c in ("full_four", "partial_1_3", "zero_explicit", "no_events") if c in pivot.columns]
    pivot = pivot.reindex(columns=col_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("% within category")
    ax.set_title("Coverage group by category (dim active)")
    ax.legend(title="group", bbox_to_anchor=(1.02, 1))
    ax.set_xticklabels([CAT_META.get(int(x), x) for x in pivot.index], rotation=0, ha="center")
    _finish_fig(fig, OUT_DIR / "fig_01_coverage_by_category.png")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    miss_plot = missing.set_index("missing_channel")["n"]
    miss_plot.plot(kind="barh", ax=ax2, color="#756bb1")
    ax2.set_xlabel("Listings (partial_1_3)")
    ax2.set_title("Missing explicit channel among partial listings")
    _finish_fig(fig2, OUT_DIR / "fig_02_missing_channels.png")

    print("§1 coverage OK — full_four:", overall.loc[overall.coverage_group == "full_four", "pct"].iloc[0] if len(overall) else "n/a")


def _two_prop_ztest(n1: int, c1: int, n2: int, c2: int) -> dict:
    if n1 == 0 or n2 == 0:
        return {"z_stat": np.nan, "p_value": np.nan}
    p1, p2 = c1 / n1, c2 / n2
    p_pool = (c1 + c2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z_stat": np.nan, "p_value": np.nan}
    z = (p1 - p2) / se
    return {"z_stat": float(z), "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))}


def _session_item_joined_sql(category_filter: str) -> str:
    samp = session_sample_clause(SESSION_SAMPLE_FRAC)
    cat_clause = f"AND category = {category_filter}" if category_filter else f"AND category IN ({CAT_IN})"
    return f"""
    WITH session_item AS (
        SELECT
            CAST(session_id AS VARCHAR) AS session_id,
            CAST(item_id AS VARCHAR) AS item_id,
            MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS has_pv,
            MAX(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END) AS has_explicit
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_login = 'login'
          AND item_id IS NOT NULL
          AND date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
          {cat_clause}
          {samp}
        GROUP BY 1, 2
        HAVING MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) = 1
    )
    SELECT si.has_explicit, ic.coverage_group, ic.n_explicit_types
    FROM session_item si
    INNER JOIN item_coverage ic ON si.item_id = ic.item_id
    """


def section_2() -> None:
    parts: list[pd.DataFrame] = []
    for cat in CATEGORIES:
        parts.append(con.execute(_session_item_joined_sql(str(cat))).df())
    joined = pd.concat(parts, ignore_index=True)
    joined["bucket_binary"] = np.where(joined["n_explicit_types"] == 4, "full_four", "not_full_four")

    contact = (
        joined.groupby(["coverage_group", "bucket_binary"], as_index=False)
        .agg(
            n_session_items=("has_explicit", "count"),
            n_with_explicit=("has_explicit", "sum"),
        )
        .rename(columns={"coverage_group": "coverage_group_detail"})
    )
    contact["session_item_contact_rate_pct"] = (
        100.0 * contact["n_with_explicit"] / contact["n_session_items"]
    ).round(3)
    contact = contact.sort_values(["coverage_group_detail", "bucket_binary"])
    contact.to_csv(OUT_DIR / "03_contact_rate_by_coverage.csv", index=False)

    contact_binary = (
        joined.groupby("bucket_binary", as_index=False)
        .agg(
            n_session_items=("has_explicit", "count"),
            n_with_explicit=("has_explicit", "sum"),
        )
    )
    contact_binary["session_item_contact_rate_pct"] = (
        100.0 * contact_binary["n_with_explicit"] / contact_binary["n_session_items"]
    ).round(3)
    contact_binary.to_csv(OUT_DIR / "03_contact_rate_binary.csv", index=False)

    meta = pd.DataFrame([{
        "session_sample_frac": SESSION_SAMPLE_FRAC,
        "note": "session_sample_frac=None for full scan if RAM allows",
    }])
    meta.to_csv(OUT_DIR / "03_contact_rate_meta.csv", index=False)

    if len(contact_binary) == 2:
        ff = contact_binary.loc[contact_binary.bucket_binary == "full_four"].iloc[0]
        nf = contact_binary.loc[contact_binary.bucket_binary == "not_full_four"].iloc[0]
        test = _two_prop_ztest(
            int(nf.n_session_items), int(nf.n_with_explicit),
            int(ff.n_session_items), int(ff.n_with_explicit),
        )
        pd.DataFrame([{
            "comparison": "full_four vs not_full_four",
            "rate_full_four_pct": ff.session_item_contact_rate_pct,
            "rate_not_full_four_pct": nf.session_item_contact_rate_pct,
            "lift_pp": round(ff.session_item_contact_rate_pct - nf.session_item_contact_rate_pct, 3),
            **test,
        }]).to_csv(OUT_DIR / "03_contact_rate_ztest.csv", index=False)

    snap = con.execute(f"""
    WITH snap_item AS (
        SELECT CAST(item_id AS VARCHAR) AS item_id,
               SUM(contacts_24h) AS contacts,
               SUM(views_24h) AS views
        FROM read_parquet('{SNAP_GLOB}')
        WHERE date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
        GROUP BY 1
    ),
    rates AS (
        SELECT ic.coverage_group,
               CASE WHEN ic.n_explicit_types = 4 THEN 'full_four' ELSE 'not_full_four' END AS bucket_binary,
               s.contacts / NULLIF(s.views, 0) AS snap_cvr
        FROM item_coverage ic
        INNER JOIN snap_item s ON ic.item_id = s.item_id
        WHERE s.views > 0
    )
    SELECT coverage_group AS coverage_group_detail,
           bucket_binary,
           COUNT(*)::BIGINT AS n_listings_with_snap,
           ROUND(quantile_cont(snap_cvr, 0.5) * 100, 4) AS median_snap_cvr_pct,
           ROUND(AVG(snap_cvr) * 100, 4) AS mean_snap_cvr_pct
    FROM rates
    GROUP BY 1, 2
    ORDER BY 1, 2
    """).df()
    snap.to_csv(OUT_DIR / "04_snapshot_rate_by_coverage.csv", index=False)

    # Sensitivity: contact_chat verified
    try:
        sens = con.execute(f"""
        WITH chat_real AS (
            SELECT DISTINCT CAST(e.item_id AS VARCHAR) AS item_id
            FROM read_parquet('{EVENTS_GLOB}') e
            INNER JOIN read_parquet('{INTER_GLOB}') i
              ON CAST(e.user_id AS VARCHAR) = CAST(i.user_id AS VARCHAR)
             AND CAST(e.item_id AS VARCHAR) = CAST(i.item_id AS VARCHAR)
             AND e.date = i.date
            WHERE e.event_type = 'contact_chat' AND e.is_login = 'login'
              AND COALESCE(i.chat_message_count, 0) > 0
              AND e.date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
        ),
        flags AS (
            SELECT
                CAST(e.item_id AS VARCHAR) AS item_id,
                MAX(CASE WHEN e.event_type = 'view_phone' THEN 1 ELSE 0 END) AS has_view_phone,
                MAX(CASE WHEN e.event_type = 'contact_zalo' THEN 1 ELSE 0 END) AS has_contact_zalo,
                MAX(CASE WHEN e.event_type = 'contact_sms' THEN 1 ELSE 0 END) AS has_contact_sms,
                MAX(CASE WHEN e.event_type = 'contact_chat' THEN 1 ELSE 0 END) AS has_chat_raw,
                MAX(CASE WHEN c.item_id IS NOT NULL THEN 1 ELSE 0 END) AS has_chat_verified
            FROM read_parquet('{EVENTS_GLOB}') e
            LEFT JOIN chat_real c ON CAST(e.item_id AS VARCHAR) = c.item_id
            WHERE e.item_id IS NOT NULL
              AND e.date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
              AND e.category IN ({CAT_IN})
            GROUP BY 1
        )
        SELECT
            SUM(CASE WHEN has_view_phone=1 AND has_contact_zalo=1
                      AND has_contact_sms=1 AND has_chat_raw=1 THEN 1 ELSE 0 END)::BIGINT AS full_four_raw,
            SUM(CASE WHEN has_view_phone=1 AND has_contact_zalo=1
                      AND has_contact_sms=1 AND has_chat_verified=1 THEN 1 ELSE 0 END)::BIGINT AS full_four_verified
        FROM flags f
        INNER JOIN item_coverage ic ON f.item_id = ic.item_id
        """).df()
        sens.to_csv(OUT_DIR / "03_sensitivity_chat_verified.csv", index=False)
    except Exception as exc:
        print("§2 sensitivity skip:", exc)

    if len(contact_binary) >= 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = contact_binary["bucket_binary"]
        y = contact_binary["session_item_contact_rate_pct"]
        colors = ["#31a354" if b == "full_four" else "#cb181d" for b in x]
        ax.bar(x, y, color=colors)
        ax.set_ylabel("session_item_contact_rate_pct")
        ax.set_title("Contact rate after pageview (login)")
        for i, (_, row) in enumerate(contact_binary.iterrows()):
            ax.text(i, row.session_item_contact_rate_pct + 0.15, f"n={int(row.n_session_items):,}", ha="center", fontsize=9)
        _finish_fig(fig, OUT_DIR / "fig_03_contact_rate_binary.png")

    print("§2 contact rate OK")


def _item_session_rates() -> pd.DataFrame:
    samp = session_sample_clause(SESSION_SAMPLE_FRAC)
    parts: list[pd.DataFrame] = []
    for cat in CATEGORIES:
        parts.append(con.execute(f"""
        WITH session_item AS (
            SELECT CAST(item_id AS VARCHAR) AS item_id,
                   MAX(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END) AS has_explicit
            FROM read_parquet('{EVENTS_GLOB}')
            WHERE is_login = 'login' AND item_id IS NOT NULL
              AND date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
              AND category = {cat}
              {samp}
            GROUP BY 1
            HAVING MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) = 1
        )
        SELECT item_id,
               COUNT(*)::BIGINT AS n_session_items_pv,
               ROUND(100.0 * AVG(has_explicit), 3) AS item_session_contact_rate_pct
        FROM session_item
        GROUP BY 1
        """).df())
    return pd.concat(parts, ignore_index=True)


def section_3() -> None:
    profile = con.execute("""
    SELECT coverage_group,
           COUNT(*)::BIGINT AS n_listings,
           ROUND(AVG(images_count), 2) AS mean_images,
           ROUND(quantile_cont(images_count, 0.5), 2) AS median_images,
           ROUND(AVG(area_sqm), 2) AS mean_area_sqm,
           ROUND(quantile_cont(area_sqm, 0.5), 2) AS median_area_sqm,
           ROUND(100.0 * AVG(CASE WHEN seller_type = 'agent' THEN 1.0 ELSE 0.0 END), 2) AS pct_agent,
           ROUND(100.0 * AVG(CASE WHEN ad_type = 'let' THEN 1.0 ELSE 0.0 END), 2) AS pct_let
    FROM item_coverage
    GROUP BY 1 ORDER BY 1
    """).df()
    profile.to_csv(OUT_DIR / "05_listing_profile_by_group.csv", index=False)

    item_rate = _item_session_rates()
    ic = con.execute("""
    SELECT coverage_group, item_id, category, ad_status,
           LEFT(title, 80) AS title, city_name, price_bucket, posted_date,
           n_pageview, n_view_phone, n_contact_chat, n_contact_zalo, n_contact_sms,
           n_explicit_types
    FROM item_coverage
    """).df()
    merged = ic.merge(item_rate, on="item_id", how="left")
    samples = []
    for grp in ("full_four", "partial_1_3", "zero_explicit", "no_events"):
        sub = merged.loc[merged.coverage_group == grp].sort_values(
            ["n_pageview", "n_view_phone"], ascending=False
        ).head(40)
        samples.append(sub)
    sample_all = pd.concat(samples, ignore_index=True)
    sample_all.to_csv(OUT_DIR / "06_sample_listings.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["full_four", "partial_1_3", "zero_explicit", "no_events"]
    plot_df = con.execute("""
    SELECT coverage_group, images_count FROM item_coverage
    WHERE images_count IS NOT NULL AND images_count > 0
    """).df()
    plot_df["coverage_group"] = pd.Categorical(plot_df["coverage_group"], categories=order, ordered=True)
    sns.boxplot(data=plot_df, x="coverage_group", y="images_count", order=order, ax=ax)
    ax.set_title("images_count by coverage group")
    ax.set_xlabel("")
    _finish_fig(fig, OUT_DIR / "fig_04_images_by_group.png")

    print("§3 listing profile OK")


def main() -> None:
    for name in ("dim_listing", "fact_user_events", "fact_listing_snapshot"):
        if not (DATA_ROOT / name).exists():
            raise FileNotFoundError(f"Missing `{name}` in {DATA_ROOT}")
    init_db()
    section_0()
    section_1()
    section_2()
    section_3()
    print(f"\nDone. Outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
