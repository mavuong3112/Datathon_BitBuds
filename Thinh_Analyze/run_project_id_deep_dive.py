"""
Project_id deep dive — runnable script (same logic as eda_project_id_deep_dive.ipynb).
Run: env/bin/python Thinh_Analyze/run_project_id_deep_dive.py
"""
from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import duckdb
import matplotlib

# Không popup GUI: Agg khi chạy script; giữ inline nếu notebook đã %matplotlib inline
_bk = matplotlib.get_backend().lower()
if "inline" not in _bk and "agg" not in _bk:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt

plt.ioff()
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

DATA_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "project_id_deep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
SNAP_GLOB = str(DATA_ROOT / "fact_listing_snapshot" / "*.parquet")
INTER_GLOB = str(DATA_ROOT / "fact_post_contact_interactions" / "*.parquet")

DUCKDB_MEMORY_LIMIT = "3GB"
DUCKDB_THREADS = 4
SAMPLE_PCT = 10
CAT_IN = "1010, 1020, 1030, 1040, 1050"
CAT_META = {
    1010: "1010 — Căn hộ / Chung cư",
    1020: "1020 — Nhà ở",
    1030: "1030 — VP / Mặt bằng",
    1040: "1040 — Đất",
    1050: "1050 — Phòng trọ",
}
CATEGORIES = tuple(CAT_META)

POSITIVE_TYPES = (
    "view_phone", "contact_chat", "other_interaction", "contact_zalo", "contact_sms",
)
EXPLICIT_TYPES = ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
POS_SQL = ", ".join(repr(x) for x in POSITIVE_TYPES)
EXPLICIT_SQL = ", ".join(repr(x) for x in EXPLICIT_TYPES)

RULES_CSV = DATA_ROOT / "outputs" / "category_mapping" / "13_dim_listing_eda_rules_long.csv"
EDA_RULES = pd.read_csv(RULES_CSV)

CATEGORICAL_DIM = {
    "ad_type", "seller_type", "price_bucket", "furnishing", "legal_status",
    "house_type", "direction", "city_name", "district_name", "ward_name", "ad_status",
}
NUMERIC_DIM = {"area_sqm", "bedrooms", "bathrooms", "floors", "width_m", "images_count"}
AREA_BUCKET_ORDER = ("<30", "30-50", "50-80", "80-120", "120+", "unknown_or_0")

# Notebook: True → display(fig) dưới cell (không plt.show / không popup)
SHOW_INLINE = False


def _finish_fig(fig: plt.Figure, path: Path | None = None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    if SHOW_INLINE:
        try:
            from IPython.display import display

            display(fig)
        except ImportError:
            pass
    plt.close(fig)
    return fig


def _clean_label(s: str, max_len: int = 40) -> str:
    t = "".join(c for c in str(s) if c.isprintable()).strip()
    return t[:max_len] + ("…" if len(t) > max_len else "")


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def top_pct_share(values: np.ndarray, top_pct: float) -> float:
    x = np.sort(values)[::-1]
    total = float(x.sum())
    if total == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(x) * top_pct / 100.0)))
    return 100.0 * float(x[:k].sum()) / total


def fields_for_category(cat: int) -> list[str]:
    roles = EDA_RULES.loc[EDA_RULES["category"] == cat]
    out = []
    for _, row in roles.iterrows():
        attr = row["attribute"]
        if attr in ("item_id", "project_id"):
            continue
        if str(row["eda_role"]).lower() == "ignore":
            continue
        out.append(attr)
    return out


def compare_categorical(cat: int, field: str) -> pd.DataFrame:
    col = field
    return con.execute(f"""
    SELECT has_project, COALESCE(NULLIF(TRIM(CAST({col} AS VARCHAR)), ''), '(blank)') AS bucket,
           COUNT(*)::BIGINT AS n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY has_project), 2) AS pct
    FROM dim_base WHERE category = {cat}
    GROUP BY 1, 2 ORDER BY 1, n DESC
    """).df()


def compare_numeric(cat: int, field: str) -> pd.DataFrame:
    return con.execute(f"""
    SELECT has_project,
           COUNT(*)::BIGINT AS n,
           ROUND(quantile_cont({field}, 0.25), 2) AS p25,
           ROUND(quantile_cont({field}, 0.5), 2) AS median,
           ROUND(quantile_cont({field}, 0.75), 2) AS p75,
           ROUND(AVG({field}), 2) AS mean
    FROM dim_base
    WHERE category = {cat} AND {field} IS NOT NULL AND {field} > 0
    GROUP BY 1 ORDER BY 1
    """).df()


con: duckdb.DuckDBPyConnection | None = None
eda_min: object = None
eda_max: object = None


def init_db() -> tuple[object, object]:
    global con, eda_min, eda_max
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    eda_dates = con.execute(f"""
    SELECT (SELECT MIN(date) FROM read_parquet('{EVENTS_GLOB}')) AS t0,
           (SELECT MAX(date) FROM read_parquet('{EVENTS_GLOB}')) AS t1
    """).fetchone()
    eda_min, eda_max = eda_dates[0], eda_dates[1]
    print(f"EDA window: {eda_min} → {eda_max}")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE dim_base AS
    SELECT
        CAST(item_id AS VARCHAR) AS item_id,
        category, seller_id, seller_type, ad_type, ad_status,
        area_sqm, bedrooms, bathrooms, floors, width_m,
        direction, legal_status, house_type, furnishing,
        city_name, district_name, ward_name, project_id, price_bucket,
        images_count, posted_date,
        CASE WHEN project_id IS NOT NULL AND TRIM(CAST(project_id AS VARCHAR)) <> ''
             THEN 1 ELSE 0 END AS has_project,
        CASE
            WHEN posted_date IS NULL THEN 'unknown_posted'
            WHEN posted_date < DATE '{eda_min}' THEN 'pre_eda_window'
            WHEN posted_date > DATE '{eda_max}' THEN 'post_eda_window'
            ELSE 'in_eda_window'
        END AS posted_cohort
    FROM read_parquet('{DIM_GLOB}')
    WHERE category IN ({CAT_IN})
    """)
    return eda_min, eda_max


def section_0() -> pd.DataFrame:
    time_cov = con.execute(f"""
    SELECT * FROM (
        SELECT 'dim_listing.posted_date' AS src, MIN(posted_date), MAX(posted_date), COUNT(*)::BIGINT
        FROM dim_base
        UNION ALL
        SELECT 'fact_user_events', MIN(date), MAX(date), COUNT(*)::BIGINT FROM read_parquet('{EVENTS_GLOB}')
        UNION ALL
        SELECT 'fact_listing_snapshot', MIN(date), MAX(date), COUNT(*)::BIGINT FROM read_parquet('{SNAP_GLOB}')
    ) ORDER BY 1
    """).df()
    time_cov.to_csv(OUT_DIR / "00_time_coverage.csv", index=False)

    pct_pid = con.execute("""
    SELECT category, ROUND(100.0 * AVG(has_project), 2) AS pct FROM dim_base GROUP BY 1
    """).df()
    assert pct_pid.loc[pct_pid.category == 1050, "pct"].iloc[0] < 1.0
    assert pct_pid.loc[pct_pid.category == 1010, "pct"].iloc[0] > 30.0
    print("§0 QA OK")
    return time_cov


def section_1() -> None:
    overview = con.execute("""
    SELECT category, SUM(has_project)::BIGINT AS with_pid, COUNT(*)::BIGINT AS total,
           ROUND(100.0 * AVG(has_project), 2) AS pct_with_pid
    FROM dim_base GROUP BY 1 ORDER BY 1
    """).df()
    overview.to_csv(OUT_DIR / "01_overview_by_category.csv", index=False)

    cross_cat = con.execute("""
    WITH pc AS (
        SELECT project_id, COUNT(DISTINCT category)::INT AS n_cat
        FROM dim_base WHERE has_project = 1 GROUP BY 1
    )
    SELECT CASE WHEN n_cat=1 THEN '1' WHEN n_cat=2 THEN '2' ELSE '3+' END AS bucket,
           COUNT(*)::BIGINT AS n_pids FROM pc GROUP BY 1
    """).df()
    cross_cat.to_csv(OUT_DIR / "01_pid_cross_category.csv", index=False)

    con.execute("""
    SELECT project_id, COUNT(*)::BIGINT AS n_items,
           STRING_AGG(DISTINCT CAST(category AS VARCHAR), ', ') AS categories
    FROM dim_base WHERE has_project = 1
    GROUP BY 1 ORDER BY n_items DESC LIMIT 15
    """).df().to_csv(OUT_DIR / "01_top15_project_ids.csv", index=False)

    con.execute("""
    SELECT category, has_project, seller_type,
           COUNT(*)::BIGINT AS n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY category, has_project), 1) AS pct
    FROM dim_base GROUP BY 1, 2, 3 ORDER BY 1, 2, 4 DESC
    """).df().to_csv(OUT_DIR / "01_seller_type_by_project.csv", index=False)

    items_per_pid = con.execute("""
    SELECT COUNT(*)::BIGINT AS n FROM dim_base WHERE has_project=1 GROUP BY project_id
    """).df()["n"].values
    conc = {
        "gini": round(gini(items_per_pid), 4),
        "top_10pct_share": round(top_pct_share(items_per_pid, 10), 2),
        "median_items_per_pid": float(np.median(items_per_pid)),
        "p90": float(np.percentile(items_per_pid, 90)),
    }
    pd.DataFrame([conc]).to_csv(OUT_DIR / "01_pid_concentration.csv", index=False)

    pid_stats = con.execute("""
    SELECT category, COUNT(DISTINCT project_id)::BIGINT AS unique_pids
    FROM dim_base WHERE has_project = 1 GROUP BY 1 ORDER BY 1
    """).df()
    seller_mix = pd.read_csv(OUT_DIR / "01_seller_type_by_project.csv")
    top15 = pd.read_csv(OUT_DIR / "01_top15_project_ids.csv")
    plot_fig01_overview(overview, cross_cat, items_per_pid, pid_stats, seller_mix, top15)
    print("§1 done")


def section_2() -> None:
    for cat in CATEGORIES:
        if cat == 1050:
            note = pd.DataFrame([{"note": "0% project_id — skip pid cohort compare"}])
            note.to_csv(OUT_DIR / f"02_{cat}_dim_compare_note.csv", index=False)
            continue
        rows = []
        for field in fields_for_category(cat):
            if field in CATEGORICAL_DIM:
                df = compare_categorical(cat, field)
                df["field"] = field
                df["kind"] = "categorical"
                rows.append(df)
            elif field in NUMERIC_DIM:
                df = compare_numeric(cat, field)
                df["field"] = field
                df["kind"] = "numeric"
                rows.append(df)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(
                OUT_DIR / f"02_{cat}_dim_compare.csv", index=False
            )
        _save_dim_figures(cat)
    print("§2 done")


def section_3() -> None:
    """Project entity tables (also run inside section_1 path — idempotent)."""
    con.execute("""
    SELECT project_id, COUNT(*)::BIGINT AS n_items,
           COUNT(DISTINCT category)::INT AS n_categories,
           COUNT(DISTINCT seller_id)::BIGINT AS n_sellers,
           COUNT(DISTINCT city_name)::BIGINT AS n_cities,
           MODE(ad_type) AS mode_ad_type
    FROM dim_base WHERE has_project = 1
    GROUP BY 1
    """).df().to_csv(OUT_DIR / "03_project_entity_summary.csv", index=False)
    con.execute("""
    SELECT project_id, STRING_AGG(DISTINCT CAST(category AS VARCHAR), ',') AS categories,
           COUNT(*)::BIGINT AS n_items
    FROM dim_base WHERE has_project = 1
    GROUP BY 1 HAVING COUNT(DISTINCT category) >= 2
    ORDER BY n_items DESC
    """).df().to_csv(OUT_DIR / "03_cross_category_pids.csv", index=False)
    print("§3 done")


def section_4() -> pd.DataFrame:
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE snap_item AS
    SELECT
        CAST(s.item_id AS VARCHAR) AS item_id,
        SUM(s.views_24h)::DOUBLE AS sum_views,
        SUM(s.contacts_24h)::DOUBLE AS sum_contacts,
        MEDIAN(s.listing_age_days)::DOUBLE AS median_age,
        COUNT(*)::BIGINT AS n_days
    FROM read_parquet('{SNAP_GLOB}') s
    INNER JOIN dim_base d ON CAST(s.item_id AS VARCHAR) = d.item_id
    WHERE d.posted_cohort = 'in_eda_window'
      AND s.date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
    GROUP BY 1
    """)
    snap_cmp = con.execute("""
    SELECT d.category, d.has_project,
           COUNT(*)::BIGINT AS listings,
           ROUND(AVG(s.sum_views), 2) AS avg_views,
           ROUND(AVG(s.sum_contacts), 2) AS avg_contacts,
           ROUND(100.0 * SUM(s.sum_contacts) / NULLIF(SUM(s.sum_views), 0), 4) AS contact_per_view_pct
    FROM dim_base d
    INNER JOIN snap_item s ON d.item_id = s.item_id
    GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    snap_cmp.to_csv(OUT_DIR / "04_snapshot_performance_all.csv", index=False)
    for cat in CATEGORIES:
        sub = snap_cmp[snap_cmp.category == cat]
        if len(sub):
            sub.to_csv(OUT_DIR / f"04_{cat}_snapshot_performance.csv", index=False)
    plot_snapshot_performance(snap_cmp)
    for cat in (1010, 1030, 1040):
        plot_snapshot_decay(cat)
    plot_snapshot_decay(1020, min_n=2000)
    print("§4 done")
    return snap_cmp


def section_5a() -> None:
    print("Building pos_items (full scan)…")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE pos_items AS
    SELECT DISTINCT CAST(item_id AS VARCHAR) AS item_id
    FROM read_parquet('{EVENTS_GLOB}')
    WHERE event_type IN ({POS_SQL})
    """)
    con.execute("""
    CREATE OR REPLACE TEMP TABLE listing_feats AS
    SELECT
        item_id, category, ad_type, has_project, posted_cohort,
        CASE WHEN bedrooms IS NULL OR bedrooms <= 0 THEN 'unknown_or_0'
             WHEN bedrooms = 1 THEN '1' WHEN bedrooms = 2 THEN '2'
             WHEN bedrooms = 3 THEN '3' ELSE '4+' END AS bed_bucket,
        CASE WHEN furnishing IS NOT NULL AND TRIM(CAST(furnishing AS VARCHAR)) <> ''
             THEN 1 ELSE 0 END AS has_furnishing,
        COALESCE(NULLIF(TRIM(CAST(house_type AS VARCHAR)), ''), '(blank)') AS house_type,
        CASE WHEN floors IS NULL OR floors <= 0 THEN 'unknown_or_0'
             WHEN floors <= 2 THEN '1-2' WHEN floors <= 4 THEN '3-4' ELSE '5+' END AS floors_bucket,
        CASE WHEN legal_status IS NOT NULL AND TRIM(CAST(legal_status AS VARCHAR)) <> ''
             THEN 1 ELSE 0 END AS has_legal,
        CASE
            WHEN area_sqm IS NULL OR area_sqm <= 0 THEN 'unknown_or_0'
            WHEN area_sqm < 30 THEN '<30' WHEN area_sqm < 50 THEN '30-50'
            WHEN area_sqm < 80 THEN '50-80' WHEN area_sqm < 120 THEN '80-120'
            ELSE '120+' END AS area_bucket
    FROM dim_base
    """)

    def cvr_sql(dims: str, where: str = "TRUE", min_n: int = 500) -> pd.DataFrame:
        return con.execute(f"""
        SELECT {dims},
            COUNT(*)::BIGINT AS listings,
            SUM(CASE WHEN p.item_id IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS with_pos,
            ROUND(100.0 * SUM(CASE WHEN p.item_id IS NOT NULL THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2) AS cvr_pct
        FROM listing_feats l
        LEFT JOIN pos_items p ON l.item_id = p.item_id
        WHERE {where}
        GROUP BY ALL HAVING COUNT(*) >= {min_n}
        """).df()

    cvr_sql("category, has_project, ad_type").to_csv(
        OUT_DIR / "05_cvr_full_category_has_project_adtype.csv", index=False
    )
    cvr_sql("category, has_project", "posted_cohort = 'in_eda_window'").to_csv(
        OUT_DIR / "05_cvr_in_eda_window.csv", index=False
    )
    cvr_sql(
        "category, has_project, ad_type, bed_bucket, has_furnishing",
        "category = 1010",
        500,
    ).to_csv(OUT_DIR / "05_cvr_1010_slices.csv", index=False)
    cvr_sql(
        "has_project, house_type, floors_bucket",
        "category = 1020",
        2000,
    ).to_csv(OUT_DIR / "05_cvr_1020_slices.csv", index=False)
    cvr_sql(
        "has_project, area_bucket, has_legal",
        "category IN (1030, 1040)",
        500,
    ).to_csv(OUT_DIR / "05_cvr_1030_1040_slices.csv", index=False)
    plot_cvr_in_window()
    plot_cvr_heatmap(1010, "05_cvr_1010_slices.csv", "bed_bucket")
    plot_cvr_heatmap(1020, "05_cvr_1020_slices.csv", "house_type")
    print("§5A done")


def section_5b() -> None:
    print(f"§5B SYSTEM {SAMPLE_PCT}% …")
    ch = con.execute(f"""
    SELECT d.has_project, e.event_type, COUNT(*)::BIGINT AS n
    FROM read_parquet('{EVENTS_GLOB}') e TABLESAMPLE {SAMPLE_PCT} PERCENT (SYSTEM)
    INNER JOIN dim_base d ON CAST(e.item_id AS VARCHAR) = d.item_id
    WHERE e.is_login = 'login' AND e.event_type IN ({EXPLICIT_SQL})
      AND d.category IN ({CAT_IN})
    GROUP BY 1, 2 ORDER BY 1, 3 DESC
    """).df()
    ch.to_csv(OUT_DIR / "05_events_sample10_explicit_channel.csv", index=False)

    dwell = con.execute(f"""
    SELECT d.has_project,
           quantile_cont(e.dwell_time_sec, 0.5) AS raw_median
    FROM read_parquet('{EVENTS_GLOB}') e TABLESAMPLE {SAMPLE_PCT} PERCENT (SYSTEM)
    INNER JOIN dim_base d ON CAST(e.item_id AS VARCHAR) = d.item_id
    WHERE e.event_type = 'pageview' AND d.category IN ({CAT_IN})
    GROUP BY 1
    """).df()
    scale = 1000.0 if dwell["raw_median"].max() > 1000 else 1.0
    dwell["median_dwell_sec"] = dwell["raw_median"] / scale
    dwell.to_csv(OUT_DIR / "05_events_sample10_dwell.csv", index=False)
    plot_explicit_channel(ch)
    print("§5B done")
    return ch, dwell


def section_6() -> pd.DataFrame:
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE login_users AS
    SELECT DISTINCT user_id FROM read_parquet('{EVENTS_GLOB}')
    WHERE is_login = 'login'
    """)
    inter = con.execute(f"""
    SELECT d.category, d.has_project,
           COUNT(DISTINCT i.item_id)::BIGINT AS listings,
           ROUND(AVG(i.lead_count), 3) AS avg_lead,
           ROUND(AVG(i.chat_message_count), 3) AS avg_chat_msg,
           ROUND(AVG(i.chat_turn_count), 3) AS avg_chat_turn
    FROM read_parquet('{INTER_GLOB}') i
    INNER JOIN dim_base d ON CAST(i.item_id AS VARCHAR) = d.item_id
    INNER JOIN login_users u ON i.user_id = u.user_id
    WHERE i.date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
    GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    inter.to_csv(OUT_DIR / "06_interactions_by_project.csv", index=False)
    plot_interactions(inter)
    print("§6 done")
    return inter


def section_7(snap_cmp: pd.DataFrame) -> pd.DataFrame:
    pct_pid = con.execute("""
    SELECT category, ROUND(100.0 * AVG(has_project), 2) AS pct FROM dim_base GROUP BY 1
    """).df()
    score_rows = []
    for cat in CATEGORIES:
        row = {"category": cat, "label": CAT_META[cat]}
        pr = pct_pid.loc[pct_pid.category == cat, "pct"]
        row["pct_has_project_id"] = float(pr.iloc[0]) if len(pr) else 0.0
        sm = con.execute(f"""
        SELECT has_project,
               ROUND(100.0 * SUM(CASE WHEN seller_type='private' THEN 1 ELSE 0 END)
                     / COUNT(*), 1) AS pct_private
        FROM dim_base WHERE category={cat} GROUP BY 1
        """).df()
        if len(sm) == 2:
            row["private_pct_delta"] = (
                sm.loc[sm.has_project == 1, "pct_private"].iloc[0]
                - sm.loc[sm.has_project == 0, "pct_private"].iloc[0]
            )
        sc = snap_cmp[snap_cmp.category == cat] if len(snap_cmp) else pd.DataFrame()
        if len(sc) == 2:
            row["contact_per_view_delta"] = (
                sc.loc[sc.has_project == 1, "contact_per_view_pct"].iloc[0]
                - sc.loc[sc.has_project == 0, "contact_per_view_pct"].iloc[0]
            )
        cv = con.execute(f"""
        SELECT has_project,
               ROUND(100.0 * SUM(CASE WHEN p.item_id IS NOT NULL THEN 1 ELSE 0 END)
                     / COUNT(*), 2) AS cvr
        FROM listing_feats l
        LEFT JOIN pos_items p ON l.item_id = p.item_id
        WHERE category={cat} AND posted_cohort='in_eda_window'
        GROUP BY 1
        """).df()
        if len(cv) == 2:
            row["cvr_delta_in_window"] = (
                cv.loc[cv.has_project == 1, "cvr"].iloc[0]
                - cv.loc[cv.has_project == 0, "cvr"].iloc[0]
            )
        if cat == 1010:
            row["modeling_note"] = "has_project_id + optional pid embedding; stratify ad_type"
        elif cat == 1050:
            row["modeling_note"] = "ignore project_id (0%); furnishing + area primary"
        else:
            row["modeling_note"] = "has_project_id flag (sparse); do not pool cross-category pids"
        score_rows.append(row)
    score = pd.DataFrame(score_rows)
    score.to_csv(OUT_DIR / "07_scorecard_all_categories.csv", index=False)

    summary = textwrap.dedent(f"""
    # Project_id deep dive — SUMMARY

    - **1010** dominates project_id prevalence (~41%); cohort with pid skews more **private** sellers.
    - **1050** has 0% project_id — use category-specific features only.
    - CVR comparisons use **in_eda_window** postings to avoid pre-fact bias.
    - Full CVR: `05_cvr_*.csv`; sample 10% events: `05_events_sample10_*.csv`.
    - Cross-category pids exist (~23% of pids in 3+ categories) — avoid naive global pid embedding.

    Generated by run_project_id_deep_dive.py
    """)
    (OUT_DIR / "SUMMARY.md").write_text(summary.strip() + "\n", encoding="utf-8")
    print("§7 done — all exports in", OUT_DIR)
    return score


def build_interest_table(snap_cmp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quan tâm KH: CVR positive, explicit (login), snapshot views/contacts — in_eda_window."""
    cvr = con.execute("""
    SELECT l.category, l.has_project,
           COUNT(*)::BIGINT AS listings,
           SUM(CASE WHEN p.item_id IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS listings_positive,
           ROUND(100.0 * SUM(CASE WHEN p.item_id IS NOT NULL THEN 1 ELSE 0 END)
                 / NULLIF(COUNT(*), 0), 2) AS cvr_positive_pct
    FROM listing_feats l
    LEFT JOIN pos_items p ON l.item_id = p.item_id
    WHERE l.posted_cohort = 'in_eda_window'
    GROUP BY 1, 2 ORDER BY 1, 2
    """).df()

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE explicit_items AS
    SELECT DISTINCT CAST(item_id AS VARCHAR) AS item_id
    FROM read_parquet('{EVENTS_GLOB}')
    WHERE is_login = 'login' AND event_type IN ({EXPLICIT_SQL})
    """)
    explicit = con.execute("""
    SELECT l.category, l.has_project,
           COUNT(*)::BIGINT AS listings,
           SUM(CASE WHEN e.item_id IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS listings_explicit,
           ROUND(100.0 * SUM(CASE WHEN e.item_id IS NOT NULL THEN 1 ELSE 0 END)
                 / NULLIF(COUNT(*), 0), 2) AS cvr_explicit_pct
    FROM listing_feats l
    LEFT JOIN explicit_items e ON l.item_id = e.item_id
    WHERE l.posted_cohort = 'in_eda_window'
    GROUP BY 1, 2 ORDER BY 1, 2
    """).df()

    if snap_cmp is None:
        try:
            snap_cmp = con.execute("""
            SELECT d.category, d.has_project,
                   COUNT(*)::BIGINT AS listings,
                   ROUND(AVG(s.sum_views), 2) AS avg_views,
                   ROUND(AVG(s.sum_contacts), 2) AS avg_contacts,
                   ROUND(100.0 * SUM(s.sum_contacts) / NULLIF(SUM(s.sum_views), 0), 4) AS contact_per_view_pct
            FROM dim_base d
            INNER JOIN snap_item s ON d.item_id = s.item_id
            WHERE d.posted_cohort = 'in_eda_window'
            GROUP BY 1, 2 ORDER BY 1, 2
            """).df()
        except duckdb.CatalogException:
            snap_cmp = None

    out = cvr.merge(
        explicit[["category", "has_project", "cvr_explicit_pct", "listings_explicit"]],
        on=["category", "has_project"],
        how="left",
    )
    if snap_cmp is not None and len(snap_cmp):
        snap = snap_cmp.rename(columns={"listings": "listings_in_snap"})
        out = out.merge(
            snap[["category", "has_project", "avg_views", "avg_contacts", "contact_per_view_pct"]],
            on=["category", "has_project"],
            how="left",
        )

    out["label"] = out["category"].map(lambda c: CAT_META[int(c)])
    out["has_project_label"] = out["has_project"].map({0: "Không project_id", 1: "Có project_id"})

    # Lift % (có vs không) theo category — chỉ category có cả 2 cohort
    lifts = []
    for cat in CATEGORIES:
        sub = out[out.category == cat]
        if cat == 1050 or sub["has_project"].nunique() < 2:
            lifts.append({"category": cat, "lift_cvr_positive_pct": np.nan, "winner_cvr": "n/a"})
            continue
        no_p = sub.loc[sub.has_project == 0, "cvr_positive_pct"].iloc[0]
        yes_p = sub.loc[sub.has_project == 1, "cvr_positive_pct"].iloc[0]
        lift = (yes_p - no_p) / no_p * 100 if no_p > 0 else np.nan
        winner = "Có pid" if yes_p > no_p else ("Không pid" if yes_p < no_p else "Hòa")
        lifts.append({"category": cat, "lift_cvr_positive_pct": round(lift, 2), "winner_cvr": winner})
    out = out.merge(pd.DataFrame(lifts), on="category", how="left")
    return out


def plot_interest_comparison(interest: pd.DataFrame) -> plt.Figure:
    """So sánh quan tâm KH: có vs không project_id trong từng category."""
    cats = [c for c in CATEGORIES if c != 1050]
    labels = [CAT_META[c].split("—")[-1].strip() for c in cats]
    x = np.arange(len(cats))
    w = 0.36
    colors = {0: "#9E9E9E", 1: "#2E86AB"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Quan tâm khách hàng — Có vs Không project_id (tin posted in_eda_window)\n"
        "CVR positive = datathon intent (README); Explicit = 4 kênh lead (login); Snapshot = views/contacts",
        fontsize=11,
        y=1.02,
    )

    def _grouped_bars(ax, metric: str, ylabel: str, title: str):
        for i, hp in enumerate([0, 1]):
            sub = interest[interest.has_project == hp].set_index("category")
            vals = [float(sub.loc[c, metric]) if c in sub.index and pd.notna(sub.loc[c, metric]) else 0 for c in cats]
            lab = "Có project_id" if hp == 1 else "Không project_id"
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lab, color=colors[hp])
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    _grouped_bars(
        axes[0, 0], "cvr_positive_pct", "CVR %",
        "Tỉ lệ tin có tương tác tích cực (positive)",
    )
    if "cvr_explicit_pct" in interest.columns:
        _grouped_bars(
            axes[0, 1], "cvr_explicit_pct", "CVR explicit %",
            "Tỉ lệ tin có lead explicit (login)",
        )
    if "contact_per_view_pct" in interest.columns:
        _grouped_bars(
            axes[1, 0], "contact_per_view_pct", "Contact / view %",
            "Snapshot: contact / views",
        )
    if "avg_contacts" in interest.columns:
        _grouped_bars(
            axes[1, 1], "avg_contacts", "Avg contacts / tin",
            "Snapshot: contacts TB / tin",
        )

    # Annotate winner per category on CVR panel
    ax0 = axes[0, 0]
    for j, cat in enumerate(cats):
        row = interest[interest.category == cat]
        if row["has_project"].nunique() < 2:
            continue
        wtxt = row["winner_cvr"].iloc[0]
        lift = row["lift_cvr_positive_pct"].iloc[0]
        if pd.notna(lift):
            ax0.text(j, ax0.get_ylim()[1] * 0.95, f"{wtxt}\n(+{lift:.0f}%)" if lift > 0 else f"{wtxt}\n({lift:.0f}%)",
                     ha="center", fontsize=7, color="#333")

    return _finish_fig(fig, OUT_DIR / "fig_08_interest_has_vs_no_project.png")


def section_interest(snap_cmp: pd.DataFrame | None = None) -> pd.DataFrame:
    interest = build_interest_table(snap_cmp)
    interest.to_csv(OUT_DIR / "08_interest_by_category_has_project.csv", index=False)
    plot_interest_comparison(interest)
    print("§8 interest comparison done")
    return interest


def main() -> None:
    init_db()
    section_0()
    section_1()
    section_2()
    section_3()
    snap = section_4()
    section_5a()
    section_interest(snap)
    section_5b()
    section_6()
    section_7(snap)


def plot_fig01_overview(
    overview: pd.DataFrame,
    cross_cat: pd.DataFrame,
    items_per_pid: np.ndarray,
    pid_stats: pd.DataFrame | None = None,
    seller_mix: pd.DataFrame | None = None,
    top15: pd.DataFrame | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Thành phần project_id — dim_listing (5 category, nhãn UI đúng)", fontsize=13, y=1.02)

    labels = [CAT_META[int(c)] for c in overview["category"]]
    p = overview["pct_with_pid"].values
    ax = axes[0, 0]
    ax.barh(labels, p, color="#4C72B0", label="Có project_id")
    ax.barh(labels, 100 - p, left=p, color="#DDDDDD", label="NULL")
    for i, (pi, row) in enumerate(zip(p, overview.itertuples())):
        ax.text(pi / 2, i, f"{pi:.1f}%", va="center", ha="center", fontsize=8, color="white")
    ax.set_xlabel("% listings")
    ax.set_title("Tỷ lệ CÓ / KHÔNG project_id")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    if pid_stats is not None and len(pid_stats):
        ps = pid_stats.copy()
        ps["label"] = ps["category"].map(lambda c: CAT_META[int(c)])
        ax.barh(ps["label"], ps["unique_pids"], color="#55A868")
        ax.set_title("Unique project_id / category")
        ax.set_xlabel("Count")
    else:
        ax.axis("off")

    ax = axes[0, 2]
    ax.pie(
        cross_cat["n_pids"],
        labels=cross_cat["bucket"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("project_id × số category")

    ax = axes[1, 0]
    ax.hist(items_per_pid, bins=50, log=True, color="#C44E52", edgecolor="white")
    ax.axvline(np.median(items_per_pid), color="k", ls="--", label=f"p50={np.median(items_per_pid):.0f}")
    ax.axvline(
        np.percentile(items_per_pid, 90), color="gray", ls=":",
        label=f"p90={np.percentile(items_per_pid, 90):.0f}",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("Items / project_id (log)")
    ax.set_title("Phân phối items / pid")

    ax = axes[1, 1]
    if top15 is not None and len(top15):
        y = np.arange(len(top15))
        ax.barh(y, top15["n_items"], color="#8172B2")
        ax.set_yticks(y)
        ax.set_yticklabels([_clean_label(x, 14) for x in top15["project_id"]], fontsize=7)
        ax.invert_yaxis()
        ax.set_title("Top 15 project_id")
    else:
        ax.axis("off")

    ax = axes[1, 2]
    if seller_mix is not None and len(seller_mix):
        cats_plot = [c for c in CATEGORIES if c != 1050]
        x = np.arange(len(cats_plot))
        w = 0.35
        for i, hp in enumerate([0, 1]):
            sub = seller_mix[
                (seller_mix.has_project == hp) & (seller_mix.seller_type == "agent")
            ].set_index("category")
            vals = [float(sub.loc[c, "pct"]) if c in sub.index else 0 for c in cats_plot]
            ax.bar(
                x + (i - 0.5) * w, vals, w,
                label=f"{'Có' if hp else 'Không'} pid — % agent",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CAT_META[c].split("—")[-1].strip() for c in cats_plot],
            rotation=15, ha="right", fontsize=8,
        )
        ax.set_ylabel("% agent trong cohort")
        ax.set_title("Seller type")
        ax.legend(fontsize=7)
    else:
        ax.axis("off")

    return _finish_fig(fig, OUT_DIR / "fig_01_overview_dashboard.png")


def plot_dim_categorical(cat: int, field: str, df: pd.DataFrame) -> plt.Figure | None:
    sub = df[(df.field == field) & (df.kind == "categorical")]
    if sub.empty:
        return None
    top_buckets = (
        sub.groupby("bucket")["n"].sum().sort_values(ascending=False).head(12).index.tolist()
    )
    sub = sub[sub.bucket.isin(top_buckets)]
    piv = sub.pivot(index="bucket", columns="has_project", values="pct").fillna(0)
    if piv.shape[1] < 2:
        return None
    piv.index = [_clean_label(b, 35) for b in piv.index]
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(piv) * 0.35)))
    piv.plot(kind="barh", ax=ax, width=0.75)
    ax.set_title(f"{CAT_META[cat]} — {field}: có vs không project_id (%)")
    ax.set_xlabel("% trong cohort has_project")
    ax.legend(["Không pid", "Có pid"], title="has_project")
    return _finish_fig(fig, OUT_DIR / f"fig_02_{cat}_{field}.png")


def plot_snapshot_performance(snap_cmp: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cats = sorted(snap_cmp["category"].unique())
    x = np.arange(len(cats))
    w = 0.35
    for i, hp in enumerate([0, 1]):
        sub = snap_cmp[snap_cmp.has_project == hp].set_index("category")
        axes[0].bar(
            x + (i - 0.5) * w,
            [sub.loc[c, "avg_views"] if c in sub.index else 0 for c in cats],
            w, label=f"has_project={hp}",
        )
        axes[1].bar(
            x + (i - 0.5) * w,
            [sub.loc[c, "contact_per_view_pct"] if c in sub.index else 0 for c in cats],
            w, label=f"has_project={hp}",
        )
    for ax, ylab in zip(axes, ["Avg views (in_eda_window)", "Contact / view %"]):
        ax.set_xticks(x)
        ax.set_xticklabels([CAT_META[int(c)].split("—")[0].strip() for c in cats], fontsize=9)
        ax.legend()
        ax.set_ylabel(ylab)
    axes[0].set_title("Snapshot — exposure")
    axes[1].set_title("Snapshot — conversion proxy")
    fig.suptitle("fact_listing_snapshot (posted in_eda_window)", fontsize=11)
    return _finish_fig(fig, OUT_DIR / "fig_04_snapshot_performance.png")


def plot_snapshot_decay(cat: int, min_n: int = 200) -> plt.Figure | None:
    decay = con.execute(f"""
    SELECT
        d.has_project,
        CASE
            WHEN s.listing_age_days < 7 THEN '0-6'
            WHEN s.listing_age_days < 14 THEN '7-13'
            WHEN s.listing_age_days < 30 THEN '14-29'
            WHEN s.listing_age_days < 60 THEN '30-59'
            ELSE '60+'
        END AS age_bucket,
        AVG(s.contacts_24h / NULLIF(s.views_24h, 0)) AS contact_rate,
        COUNT(*)::BIGINT AS n
    FROM read_parquet('{SNAP_GLOB}') s
    INNER JOIN dim_base d ON CAST(s.item_id AS VARCHAR) = d.item_id
    WHERE d.category = {cat} AND d.posted_cohort = 'in_eda_window'
      AND s.date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
      AND s.views_24h > 0
    GROUP BY 1, 2
    HAVING COUNT(*) >= {min_n}
    ORDER BY 1, 2
    """).df()
    if decay.empty:
        return None
    order = ["0-6", "7-13", "14-29", "30-59", "60+"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for hp, lab in [(0, "Không pid"), (1, "Có pid")]:
        sub = decay[decay.has_project == hp].set_index("age_bucket").reindex(order)
        ax.plot(order, sub["contact_rate"] * 100, marker="o", label=lab)
    ax.set_ylabel("Contact/view % (daily snap)")
    ax.set_xlabel("listing_age_days bucket")
    ax.set_title(f"{CAT_META[cat]} — decay theo tuổi tin")
    ax.legend()
    return _finish_fig(fig, OUT_DIR / f"fig_04_{cat}_decay.png")


def plot_cvr_heatmap(cat: int, csv_name: str, index_col: str, col_col: str = "has_project") -> plt.Figure | None:
    path = OUT_DIR / csv_name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "cvr_pct" not in df.columns:
        return None
    if index_col not in df.columns:
        return None
    piv = df.pivot_table(index=index_col, columns=col_col, values="cvr_pct", aggfunc="mean")
    if piv.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(5, piv.shape[1] * 2), max(4, piv.shape[0] * 0.4)))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"{CAT_META[cat]} — CVR % ({index_col} × has_project)")
    return _finish_fig(fig, OUT_DIR / f"fig_05_{cat}_cvr_heatmap.png")


def plot_explicit_channel(ch: pd.DataFrame) -> plt.Figure:
    ch = ch.copy()
    ch["has_project"] = ch["has_project"].map({0: "Không pid", 1: "Có pid"})
    piv = ch.pivot(index="event_type", columns="has_project", values="n").fillna(0)
    piv_pct = piv.div(piv.sum(axis=0), axis=1) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    piv_pct.plot(kind="bar", ax=ax)
    ax.set_ylabel("% trong cohort (SYSTEM 10% sample)")
    ax.set_title("Explicit contact — mix kênh × has_project (login)")
    ax.legend(title="")
    plt.xticks(rotation=20, ha="right")
    return _finish_fig(fig, OUT_DIR / "fig_05_explicit_channel.png")


def plot_cvr_in_window() -> plt.Figure:
    df = pd.read_csv(OUT_DIR / "05_cvr_in_eda_window.csv")
    df["label"] = df["category"].map(lambda c: CAT_META[int(c)])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df["label"].unique()))
    cats = sorted(df["category"].unique())
    w = 0.35
    for i, hp in enumerate([0, 1]):
        sub = df[df.has_project == hp].set_index("category")
        ax.bar(
            np.arange(len(cats)) + (i - 0.5) * w,
            [sub.loc[c, "cvr_pct"] if c in sub.index else 0 for c in cats],
            w, label=f"has_project={hp}",
        )
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([CAT_META[int(c)] for c in cats], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("CVR % (≥1 positive event)")
    ax.set_title("CVR catalog — chỉ tin posted in_eda_window")
    ax.legend()
    return _finish_fig(fig, OUT_DIR / "fig_05_cvr_in_window.png")


def plot_interactions(inter: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["avg_lead", "avg_chat_msg", "avg_chat_turn"]
    titles = ["Lead (lộ SĐT)", "Chat messages", "Chat turns"]
    cats = sorted(inter["category"].unique())
    x = np.arange(len(cats))
    w = 0.35
    for ax, m, tit in zip(axes, metrics, titles):
        for i, hp in enumerate([0, 1]):
            sub = inter[inter.has_project == hp].set_index("category")
            ax.bar(
                x + (i - 0.5) * w,
                [sub.loc[c, m] if c in sub.index else 0 for c in cats],
                w, label=f"pid={hp}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cats], fontsize=8)
        ax.set_title(tit)
        ax.legend(fontsize=7)
    fig.suptitle("Post-contact interactions (login users, in EDA window)", fontsize=11)
    return _finish_fig(fig, OUT_DIR / "fig_06_interactions.png")


def _save_dim_figures(cat: int) -> None:
    path = OUT_DIR / f"02_{cat}_dim_compare.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    for field in df["field"].unique():
        if field in CATEGORICAL_DIM:
            plot_dim_categorical(cat, field, df)


if __name__ == "__main__":
    main()
