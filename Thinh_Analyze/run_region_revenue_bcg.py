"""
Doanh thu proxy theo miền + ma trận BCG (loại HCM khỏi tỷ trọng).

Run: env/bin/python Thinh_Analyze/run_region_revenue_bcg.py
     env/bin/python Thinh_Analyze/run_region_revenue_bcg.py --event-sample-frac 0.25
"""
from __future__ import annotations

import argparse
import textwrap
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

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.ioff()

DATA_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "config"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "region_revenue_bcg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
MAPPING_CSV = CONFIG_DIR / "city_region_mapping.csv"

DUCKDB_MEMORY_LIMIT = "3GB"
DUCKDB_THREADS = 4
MIN_N_LISTINGS = 40
MIN_N_EXPLICIT = 20

ACTIVE_AD_STATUS = ("accepted", "hidden", "shop_accepted")
ACTIVE_AD_SQL = ", ".join(repr(x) for x in ACTIVE_AD_STATUS)

CAT_IN = "1010, 1020, 1030, 1040, 1050"
CAT_META = {
    1010: "Căn hộ",
    1020: "Nhà ở",
    1030: "VP/MB",
    1040: "Đất",
    1050: "Phòng trọ",
}
CATEGORIES = tuple(CAT_META)
REGIONS = ("Bac", "Trung", "Nam")
AD_TYPES = ("let", "sell")

EXPLICIT_TYPES = ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
POSITIVE_TYPES = EXPLICIT_TYPES + ("other_interaction",)
EXPLICIT_SQL = ", ".join(repr(x) for x in EXPLICIT_TYPES)
POSITIVE_SQL = ", ".join(repr(x) for x in POSITIVE_TYPES)

QUADRANT_LABELS = {
    "stars": "Stars (Sao)",
    "cash_cows": "Cash cows (Bò sữa)",
    "question_marks": "Question marks (Dấu hỏi)",
    "dogs": "Dogs (Chó)",
    "new_market": "New market (Mới)",
    "low_volume": "Low volume",
    "hcmc_excluded": "HCM (loại khỏi BCG)",
}

con: duckdb.DuckDBPyConnection | None = None
eda_min: object = None
eda_mid: object = None
eda_max: object = None


def _finish_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def _clean_city(s: str) -> str:
    return str(s).strip() if s is not None else ""


def is_hcmc_name(city: str) -> int:
    c = _clean_city(city).lower()
    c = c.replace("tp.", "tp").replace("  ", " ")
    if "hồ chí minh" in c or "ho chi minh" in c:
        return 1
    if c in ("tp hồ chí minh", "tp ho chi minh", "hồ chí minh"):
        return 1
    return 0


def load_city_mapping() -> pd.DataFrame:
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(f"Missing {MAPPING_CSV}")
    m = pd.read_csv(MAPPING_CSV)
    m["city_name_raw"] = m["city_name_raw"].map(_clean_city)
    m["is_hcmc"] = m["is_hcmc"].astype(int)
    return m


def build_dim_listing_prep(eda_min_d: object, eda_max_d: object) -> None:
    """Phase 0: dim_listing QA — cohort, active status, dedup item_id → dim_scoped."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dim_all AS
        SELECT
            CAST(item_id AS VARCHAR) AS item_id,
            category,
            ad_type,
            ad_status,
            TRIM(CAST(city_name AS VARCHAR)) AS city_name,
            posted_date,
            CASE
                WHEN posted_date IS NULL THEN 'unknown_posted'
                WHEN posted_date < DATE '{eda_min_d}' THEN 'pre_eda_window'
                WHEN posted_date > DATE '{eda_max_d}' THEN 'post_eda_window'
                ELSE 'in_eda_window'
            END AS posted_cohort
        FROM read_parquet('{DIM_GLOB}')
        WHERE category IN ({CAT_IN})
          AND ad_type IN ('let', 'sell')
    """)

    n_raw_5cat = con.execute(
        "SELECT COUNT(*)::BIGINT FROM dim_all"
    ).fetchone()[0]

    cohort_df = con.execute("""
        SELECT ad_status, posted_cohort, COUNT(*)::BIGINT AS n
        FROM dim_all
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    cohort_df.to_csv(OUT_DIR / "00_dim_listing_posted_cohort.csv", index=False)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dim_active AS
        SELECT *
        FROM dim_all
        WHERE ad_status IN ({ACTIVE_AD_SQL})
          AND posted_cohort = 'in_eda_window'
          AND city_name IS NOT NULL
          AND LENGTH(city_name) > 0
    """)
    n_active = con.execute("SELECT COUNT(*)::BIGINT FROM dim_active").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TEMP TABLE dim_ranked AS
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY item_id
                ORDER BY posted_date DESC NULLS LAST, item_id
            ) AS rn,
            COUNT(*) OVER (PARTITION BY item_id) AS n_dup
        FROM dim_active
    """)

    dup_drop = con.execute("""
        SELECT item_id, n_dup::BIGINT AS n_duplicate_rows
        FROM dim_ranked
        WHERE n_dup > 1 AND rn > 1
        ORDER BY n_dup DESC, item_id
    """).df()
    dup_drop.to_csv(OUT_DIR / "00_dim_listing_duplicate_drop.csv", index=False)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE dim_deduped AS
        SELECT
            item_id,
            category,
            ad_type,
            city_name,
            posted_date,
            posted_cohort,
            ad_status
        FROM dim_ranked
        WHERE rn = 1
    """)
    n_deduped = con.execute("SELECT COUNT(*)::BIGINT FROM dim_deduped").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TEMP TABLE dim_scoped AS
        SELECT item_id, category, ad_type, city_name, posted_date
        FROM dim_deduped
    """)

    by_region = con.execute("""
        SELECT
            COALESCE(cr.region, 'Unknown') AS region,
            d.category,
            d.ad_type,
            COUNT(*)::BIGINT AS n_listings
        FROM dim_scoped d
        LEFT JOIN city_region cr ON d.city_name = cr.city_name
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).df()
    by_region.to_csv(OUT_DIR / "00_dim_listing_by_region_category.csv", index=False)

    pd.DataFrame(
        [
            {"stage": "dim_all_5cat", "n_rows": n_raw_5cat},
            {"stage": "dim_active_in_eda_window", "n_rows": n_active},
            {"stage": "dim_scoped_deduped", "n_rows": n_deduped},
            {
                "stage": "duplicate_rows_dropped",
                "n_rows": int(len(dup_drop)),
            },
        ]
    ).to_csv(OUT_DIR / "00_dim_listing_prep_summary.csv", index=False)
    print(
        f"§ dim_listing prep OK: active={n_active:,} → deduped={n_deduped:,} "
        f"(dropped {len(dup_drop):,} dup rows)"
    )


def init_db(event_sample_frac: float | None) -> None:
    global con, eda_min, eda_mid, eda_max
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")

    row = con.execute(f"""
        SELECT MIN(date) AS dmin, MAX(date) AS dmax
        FROM read_parquet('{EVENTS_GLOB}')
    """).fetchone()
    eda_min, eda_max = row[0], row[1]
    eda_mid = con.execute(f"""
        SELECT MIN(date) + ((MAX(date) - MIN(date)) / 2)::INTEGER
        FROM read_parquet('{EVENTS_GLOB}')
    """).fetchone()[0]
    print(f"EDA window: {eda_min} → {eda_max} (mid={eda_mid})")

    mapping = load_city_mapping()
    con.register("city_map_df", mapping)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE city_region AS
        SELECT city_name_raw AS city_name, region, is_hcmc
        FROM city_map_df
    """)

    build_dim_listing_prep(eda_min, eda_max)

    sample_clause = ""
    if event_sample_frac is not None and 0 < event_sample_frac < 1:
        bucket = max(1, int(event_sample_frac * 1000))
        sample_clause = (
            f"AND (abs(hash(CAST(event_id AS VARCHAR))) % 1000) < {bucket}"
        )
        print(f"Event sample: {event_sample_frac:.2%} ({sample_clause.strip()})")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE events_scoped AS
        SELECT
            CAST(e.item_id AS VARCHAR) AS item_id,
            e.date,
            e.event_type,
            e.category AS event_category,
            d.category AS dim_category,
            d.ad_type,
            d.city_name
        FROM read_parquet('{EVENTS_GLOB}') e
        INNER JOIN dim_scoped d ON CAST(e.item_id AS VARCHAR) = d.item_id
        WHERE e.is_login = 'login'
          AND e.date BETWEEN DATE '{eda_min}' AND DATE '{eda_max}'
          {sample_clause}
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE events_geo AS
        SELECT
            e.*,
            COALESCE(cr.region, 'Unknown') AS region,
            COALESCE(cr.is_hcmc, 0) AS is_hcmc
        FROM events_scoped e
        LEFT JOIN city_region cr ON e.city_name = cr.city_name
    """)


def export_distinct_cities() -> pd.DataFrame:
    df = con.execute(f"""
        SELECT TRIM(CAST(city_name AS VARCHAR)) AS city_name,
               COUNT(*)::BIGINT AS n_listings
        FROM read_parquet('{DIM_GLOB}')
        WHERE city_name IS NOT NULL
          AND LENGTH(TRIM(CAST(city_name AS VARCHAR))) > 0
        GROUP BY 1 ORDER BY n_listings DESC
    """).df()
    df.to_csv(OUT_DIR / "00_distinct_cities.csv", index=False)
    mapped = set(load_city_mapping()["city_name_raw"])
    unknown = df[~df["city_name"].isin(mapped)]
    if len(unknown):
        print("WARNING unmapped cities:", unknown["city_name"].tolist())
    return df


def build_listing_counts() -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE listing_counts AS
        SELECT
            COALESCE(cr.region, 'Unknown') AS region,
            d.city_name,
            COALESCE(cr.is_hcmc, 0) AS is_hcmc,
            d.category,
            d.ad_type,
            COUNT(*)::BIGINT AS n_listings
        FROM dim_scoped d
        LEFT JOIN city_region cr ON d.city_name = cr.city_name
        GROUP BY 1, 2, 3, 4, 5
    """)


def build_event_totals() -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE event_totals AS
        SELECT
            region,
            city_name,
            is_hcmc,
            dim_category AS category,
            ad_type,
            SUM(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END)::BIGINT
                AS n_explicit_events,
            SUM(CASE WHEN event_type IN ({POSITIVE_SQL}) THEN 1 ELSE 0 END)::BIGINT
                AS n_positive_events
        FROM events_geo
        GROUP BY 1, 2, 3, 4, 5
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE event_half AS
        SELECT
            region,
            city_name,
            is_hcmc,
            dim_category AS category,
            ad_type,
            SUM(CASE WHEN date < DATE '{eda_mid}'
                      AND event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END)::BIGINT
                AS explicit_h1,
            SUM(CASE WHEN date >= DATE '{eda_mid}'
                      AND event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END)::BIGINT
                AS explicit_h2,
            SUM(CASE WHEN date < DATE '{eda_mid}'
                      AND event_type IN ({POSITIVE_SQL}) THEN 1 ELSE 0 END)::BIGINT
                AS positive_h1,
            SUM(CASE WHEN date >= DATE '{eda_mid}'
                      AND event_type IN ({POSITIVE_SQL}) THEN 1 ELSE 0 END)::BIGINT
                AS positive_h2
        FROM events_geo
        GROUP BY 1, 2, 3, 4, 5
    """)


def export_revenue_tables() -> pd.DataFrame:
    build_listing_counts()
    build_event_totals()

    rev = con.execute("""
        SELECT
            COALESCE(l.region, e.region) AS region,
            COALESCE(l.city_name, e.city_name) AS city_name,
            COALESCE(l.is_hcmc, e.is_hcmc, 0) AS is_hcmc,
            COALESCE(l.category, e.category) AS category,
            COALESCE(l.ad_type, e.ad_type) AS ad_type,
            COALESCE(l.n_listings, 0) AS n_listings,
            COALESCE(e.n_explicit_events, 0) AS n_explicit_events,
            COALESCE(e.n_positive_events, 0) AS n_positive_events
        FROM listing_counts l
        FULL OUTER JOIN event_totals e
          ON l.region = e.region
         AND l.city_name = e.city_name
         AND l.category = e.category
         AND l.ad_type = e.ad_type
        WHERE COALESCE(l.region, e.region) != 'Unknown'
        ORDER BY region, category, ad_type, n_explicit_events DESC
    """).df()

    rev["cvr_per_listing"] = np.where(
        rev["n_listings"] > 0,
        rev["n_explicit_events"] / rev["n_listings"],
        np.nan,
    )
    rev["explicit_per_1k_listings"] = rev["cvr_per_listing"] * 1000.0
    rev["positive_per_1k_listings"] = np.where(
        rev["n_listings"] > 0,
        1000.0 * rev["n_positive_events"] / rev["n_listings"],
        np.nan,
    )
    rev.to_csv(OUT_DIR / "01_revenue_by_province.csv", index=False)

    summary = (
        rev.groupby(["region", "category", "ad_type"], as_index=False)
        .agg(
            n_listings=("n_listings", "sum"),
            n_explicit_events=("n_explicit_events", "sum"),
            n_positive_events=("n_positive_events", "sum"),
        )
    )
    summary["explicit_per_1k_listings"] = np.where(
        summary["n_listings"] > 0,
        1000.0 * summary["n_explicit_events"] / summary["n_listings"],
        np.nan,
    )
    summary.to_csv(OUT_DIR / "02_revenue_region_category_adtype.csv", index=False)
    print("§ revenue tables OK")
    return rev


def export_share_ex_hcmc(rev: pd.DataFrame) -> pd.DataFrame:
    sub = rev[rev["is_hcmc"] == 0].copy()
    grp = ["region", "category", "ad_type"]
    for col in ("n_explicit_events", "n_positive_events"):
        denom = sub.groupby(grp)[col].transform("sum")
        sub[f"share_{col.replace('n_', '')}_ex_hcmc_pct"] = np.where(
            denom > 0, 100.0 * sub[col] / denom, np.nan
        )

    hcmc = rev[rev["is_hcmc"] == 1].copy()
    if len(hcmc):
        vn_explicit = rev["n_explicit_events"].sum()
        vn_positive = rev["n_positive_events"].sum()
        hcmc["share_explicit_vn_pct"] = np.where(
            vn_explicit > 0, 100.0 * hcmc["n_explicit_events"] / vn_explicit, np.nan
        )
        hcmc["share_positive_vn_pct"] = np.where(
            vn_positive > 0, 100.0 * hcmc["n_positive_events"] / vn_positive, np.nan
        )
        hcmc.to_csv(OUT_DIR / "hcmc_reference_stats.csv", index=False)

    out_cols = [
        "region", "city_name", "is_hcmc", "category", "ad_type",
        "n_listings", "n_explicit_events", "n_positive_events",
        "share_explicit_events_ex_hcmc_pct", "share_positive_events_ex_hcmc_pct",
    ]
    share_df = pd.concat([
        sub[out_cols],
        rev[rev["is_hcmc"] == 1][
            ["region", "city_name", "is_hcmc", "category", "ad_type",
             "n_listings", "n_explicit_events", "n_positive_events"]
        ].assign(
            share_explicit_events_ex_hcmc_pct=np.nan,
            share_positive_events_ex_hcmc_pct=np.nan,
        ),
    ], ignore_index=True)
    share_df.to_csv(OUT_DIR / "03_share_ex_hcmc.csv", index=False)
    print("§ share ex-HCM OK")
    return sub


def _growth_pct(h1: float, h2: float) -> float:
    if h1 <= 0 and h2 > 0:
        return np.nan
    if h1 <= 0:
        return 0.0
    return 100.0 * (h2 - h1) / h1


def _bcg_eligible_mask(
    df: pd.DataFrame,
    min_n_listings: int,
    min_n_explicit: int,
) -> pd.Series:
    return (
        (df["is_hcmc"] == 0)
        & (df["n_listings"] >= min_n_listings)
        & (df["n_explicit_events"] >= min_n_explicit)
    )


def _apply_conversion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """CVR explicit/tin (4 loại) — dùng cùng n_listings cho H1/H2 (snapshot)."""
    out = df.copy()
    nl = out["n_listings"].replace(0, np.nan)
    out["cvr_per_listing"] = out["n_explicit_events"] / nl
    out["explicit_per_1k_listings"] = out["cvr_per_listing"] * 1000.0
    out["cvr_h1"] = out["explicit_h1"] / nl
    out["cvr_h2"] = out["explicit_h2"] / nl
    out["growth_cvr_pct"] = [
        _growth_pct(a, b) for a, b in zip(out["cvr_h1"], out["cvr_h2"])
    ]
    return out


def _assign_relative_cvr(
    g: pd.DataFrame, eligible: pd.Series
) -> pd.Series:
    max_cvr = float(g.loc[eligible, "explicit_per_1k_listings"].max()) if eligible.any() else 0.0
    return pd.Series(
        np.where(
            eligible & (max_cvr > 0),
            g["explicit_per_1k_listings"] / max_cvr,
            np.nan,
        ),
        index=g.index,
    )


BCG_MIN_ELIGIBLE_FOR_MEDIAN = 2


def _find_bcg_reference_pool(
    ex_hcmc: pd.DataFrame,
    *,
    region: str | None,
    category: int,
    ad_type: str,
    min_n_listings: int,
    min_n_explicit: int,
) -> tuple[pd.DataFrame, float, float, str]:
    """
    Median BCG từ pool đủ ≥2 tỉnh eligible; thử segment → national cat×ad → cat → ad → all.
  """
    candidates: list[tuple[str, pd.Series]] = []
    if region is not None:
        candidates.append(
            (
                "regional",
                (ex_hcmc["region"] == region)
                & (ex_hcmc["category"] == category)
                & (ex_hcmc["ad_type"] == ad_type),
            )
        )
    candidates.extend(
        [
            (
                "national_cat_ad",
                (ex_hcmc["category"] == category) & (ex_hcmc["ad_type"] == ad_type),
            ),
            ("national_cat", ex_hcmc["category"] == category),
            ("national_ad", ex_hcmc["ad_type"] == ad_type),
            ("national_all", pd.Series(True, index=ex_hcmc.index)),
        ]
    )
    for scope, mask in candidates:
        pool = ex_hcmc[mask]
        eligible = _bcg_eligible_mask(pool, min_n_listings, min_n_explicit)
        if int(eligible.sum()) < BCG_MIN_ELIGIBLE_FOR_MEDIAN:
            continue
        pool = pool.copy()
        pool["relative_cvr"] = _assign_relative_cvr(pool, eligible)
        eg = pool.loc[eligible]
        return (
            pool,
            float(eg["growth_cvr_pct"].median()),
            float(eg["relative_cvr"].median()),
            scope,
        )
    return ex_hcmc.iloc[0:0], np.nan, np.nan, "none"


def _apply_bcg_quadrants(
    g: pd.DataFrame,
    ref_pool: pd.DataFrame,
    med_g: float,
    med_s: float,
    median_scope: str,
) -> pd.DataFrame:
    """Gán ô BCG; relative_cvr lấy từ ref_pool (cùng chuẩn với median)."""
    g = g.copy()
    if not ref_pool.empty:
        rel_map = ref_pool.drop_duplicates("city_name").set_index("city_name")[
            "relative_cvr"
        ]
        g["relative_cvr"] = g["city_name"].map(rel_map)
    g["bcg_median_scope"] = median_scope
    g["median_growth"] = med_g
    g["median_relative_share"] = med_s
    g["median_growth_cvr"] = med_g
    g["median_relative_cvr"] = med_s
    g["bcg_quadrant"] = g.apply(
        lambda row: _assign_bcg_quadrant(row, med_g, med_s), axis=1
    )
    return g


def build_bcg_tables(
    rev: pd.DataFrame,
    min_n_listings: int = MIN_N_LISTINGS,
    min_n_explicit: int = MIN_N_EXPLICIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    half = con.execute("""
        SELECT * FROM event_half
        WHERE region != 'Unknown'
    """).df()

    bcg = rev[rev["is_hcmc"] == 0].merge(
        half,
        on=["region", "city_name", "is_hcmc", "category", "ad_type"],
        how="left",
    )
    bcg["explicit_h1"] = bcg["explicit_h1"].fillna(0).astype(int)
    bcg["explicit_h2"] = bcg["explicit_h2"].fillna(0).astype(int)
    bcg["positive_h1"] = bcg["positive_h1"].fillna(0).astype(int)
    bcg["positive_h2"] = bcg["positive_h2"].fillna(0).astype(int)

    bcg["growth_explicit_pct"] = [
        _growth_pct(a, b) for a, b in zip(bcg["explicit_h1"], bcg["explicit_h2"])
    ]
    bcg["growth_positive_pct"] = [
        _growth_pct(a, b) for a, b in zip(bcg["positive_h1"], bcg["positive_h2"])
    ]
    bcg["is_new_market"] = (bcg["explicit_h1"] == 0) & (bcg["explicit_h2"] > 0)
    bcg = _apply_conversion_metrics(bcg)

    share = bcg.copy()
    grp = ["region", "category", "ad_type"]
    denom = share.groupby(grp)["n_explicit_events"].transform("sum")
    share["share_explicit_ex_hcmc_pct"] = np.where(
        denom > 0, 100.0 * share["n_explicit_events"] / denom, 0.0
    )
    max_share = share.groupby(grp)["share_explicit_ex_hcmc_pct"].transform("max")
    share["relative_share"] = np.where(
        max_share > 0, share["share_explicit_ex_hcmc_pct"] / max_share, 0.0
    )

    ex_hcmc = share[share["is_hcmc"] == 0]
    quadrants = []
    for keys, g in share.groupby(["region", "ad_type", "category"]):
        region, ad_type, category = keys
        g = g.copy()
        eligible = _bcg_eligible_mask(g, min_n_listings, min_n_explicit)
        g["bcg_eligible"] = eligible
        ref_pool, med_g, med_s, scope = _find_bcg_reference_pool(
            ex_hcmc,
            region=region,
            category=category,
            ad_type=ad_type,
            min_n_listings=min_n_listings,
            min_n_explicit=min_n_explicit,
        )
        g = _apply_bcg_quadrants(g, ref_pool, med_g, med_s, scope)
        quadrants.append(g)

    quad_df = pd.concat(quadrants, ignore_index=True)
    quad_df["bcg_label"] = quad_df["bcg_quadrant"].map(QUADRANT_LABELS)
    input_cols = [
        "region", "city_name", "category", "ad_type",
        "n_listings", "n_explicit_events", "n_positive_events",
        "cvr_per_listing", "explicit_per_1k_listings",
        "share_explicit_ex_hcmc_pct", "relative_share", "relative_cvr",
        "explicit_h1", "explicit_h2", "cvr_h1", "cvr_h2",
        "growth_explicit_pct", "growth_cvr_pct",
        "growth_positive_pct", "is_new_market",
        "bcg_eligible", "bcg_median_scope",
    ]
    inputs = quad_df[[c for c in input_cols if c in quad_df.columns]].copy()
    inputs.to_csv(OUT_DIR / "04_bcg_inputs.csv", index=False)
    quad_df.to_csv(OUT_DIR / "05_bcg_quadrants.csv", index=False)
    print("§ BCG quadrants OK")
    return inputs, quad_df


def _assign_bcg_quadrant(
    row: pd.Series,
    med_g: float,
    med_s: float,
) -> str:
    if row.get("is_hcmc", 0) == 1:
        return "hcmc_excluded"
    if not row["bcg_eligible"]:
        return "low_volume"
    if row.get("is_new_market"):
        return "question_marks"
    if pd.isna(med_g) or pd.isna(med_s):
        return "low_volume"
    gh = row.get("growth_cvr_pct", row.get("growth_explicit_pct"))
    sh = row.get("relative_cvr", row.get("relative_share"))
    if pd.isna(gh):
        return "question_marks"
    high_g = gh >= med_g
    high_s = sh >= med_s
    if high_g and high_s:
        return "stars"
    if not high_g and high_s:
        return "cash_cows"
    if high_g and not high_s:
        return "question_marks"
    return "dogs"


def _ensure_event_half() -> None:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "event_half" not in tables:
        build_listing_counts()
        build_event_totals()


def build_bcg_national_choropleth(
    rev: pd.DataFrame,
    min_n_listings: int = MIN_N_LISTINGS,
    min_n_explicit: int = MIN_N_EXPLICIT,
) -> pd.DataFrame:
    """BCG toàn quốc — trục CVR explicit/tin (4 loại); HCM excluded from median."""
    _ensure_event_half()
    half = con.execute("""
        SELECT * FROM event_half
        WHERE region != 'Unknown'
    """).df()

    base = rev[rev["region"] != "Unknown"].merge(
        half,
        on=["region", "city_name", "is_hcmc", "category", "ad_type"],
        how="left",
    )
    for col in ("explicit_h1", "explicit_h2"):
        base[col] = base[col].fillna(0).astype(int)
    base["growth_explicit_pct"] = [
        _growth_pct(a, b) for a, b in zip(base["explicit_h1"], base["explicit_h2"])
    ]
    base["is_new_market"] = (base["explicit_h1"] == 0) & (base["explicit_h2"] > 0)
    base = _apply_conversion_metrics(base)
    base["demand_per_1k_supply"] = base["explicit_per_1k_listings"]

    parts: list[pd.DataFrame] = []
    warnings_national: list[str] = []
    ex_hcmc_all = base[base["is_hcmc"] == 0]

    for (category, ad_type), g in base.groupby(["category", "ad_type"]):
        g = g.copy()
        denom = float(g.loc[g["is_hcmc"] == 0, "n_explicit_events"].sum())
        g["share_demand_pct"] = np.where(
            g["is_hcmc"] == 1,
            np.nan,
            np.where(denom > 0, 100.0 * g["n_explicit_events"] / denom, 0.0),
        )
        non_hcm = g[g["is_hcmc"] == 0]
        max_share = float(non_hcm["share_demand_pct"].max()) if len(non_hcm) else 0.0
        g["relative_share"] = np.where(
            (g["is_hcmc"] == 0) & (max_share > 0),
            g["share_demand_pct"] / max_share,
            np.nan,
        )

        eligible = _bcg_eligible_mask(g, min_n_listings, min_n_explicit)
        g["bcg_eligible"] = eligible
        ref_pool, med_g, med_s, scope = _find_bcg_reference_pool(
            ex_hcmc_all,
            region=None,
            category=category,
            ad_type=ad_type,
            min_n_listings=min_n_listings,
            min_n_explicit=min_n_explicit,
        )
        if scope == "none":
            warnings_national.append(
                f"category={category} ad_type={ad_type}: no pool with ≥2 eligible provinces"
            )
        elif scope != "national_cat_ad":
            warnings_national.append(
                f"category={category} ad_type={ad_type}: median fallback → {scope}"
            )
        g = _apply_bcg_quadrants(g, ref_pool, med_g, med_s, scope)
        g["bcg_label"] = g["bcg_quadrant"].map(QUADRANT_LABELS)
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT_DIR / "06_bcg_national_choropleth.csv", index=False)
    if warnings_national:
        warn_path = OUT_DIR / "06_bcg_national_warnings.txt"
        warn_path.write_text("\n".join(warnings_national) + "\n", encoding="utf-8")
    print("§ BCG national choropleth OK →", OUT_DIR / "06_bcg_national_choropleth.csv")
    return out


def plot_bcg_matrix(quad_df: pd.DataFrame) -> None:
    for region in REGIONS:
        for ad_type in AD_TYPES:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig.suptitle(
                f"BCG CVR — {region} × {ad_type} (explicit/tin, HCM excluded)",
                fontsize=14,
                y=1.02,
            )
            cats = list(CATEGORIES)
            for idx, cat in enumerate(cats):
                ax = axes.flat[idx]
                sub = quad_df[
                    (quad_df["region"] == region)
                    & (quad_df["ad_type"] == ad_type)
                    & (quad_df["category"] == cat)
                ]
                eligible = sub[sub["bcg_eligible"]]
                low = sub[~sub["bcg_eligible"]]

                if len(eligible) == 0:
                    ax.set_title(f"{cat} — {CAT_META[cat]}")
                    ax.text(0.5, 0.5, "No eligible provinces", ha="center", va="center")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-50, 50)
                    continue

                med_g = eligible["median_growth_cvr"].iloc[0]
                if pd.isna(med_g):
                    med_g = eligible["median_growth"].iloc[0]
                med_s = eligible["median_relative_cvr"].iloc[0]
                if pd.isna(med_s):
                    med_s = eligible["median_relative_share"].iloc[0]
                xlim = (-0.05, 1.1)
                ycol = "growth_cvr_pct" if "growth_cvr_pct" in eligible.columns else "growth_explicit_pct"
                xcol = "relative_cvr" if "relative_cvr" in eligible.columns else "relative_share"
                ylo = min(-20, eligible[ycol].min() - 10)
                yhi = max(20, eligible[ycol].max() + 10)
                if pd.notna(med_g):
                    ax.axhline(med_g, color="gray", ls="--", lw=1, alpha=0.7)
                    ax.axvline(med_s, color="gray", ls="--", lw=1, alpha=0.7)
                    ax.fill_between([med_s, xlim[1]], med_g, yhi, alpha=0.08, color="green")
                    ax.fill_between([med_s, xlim[1]], ylo, med_g, alpha=0.08, color="gold")
                    ax.fill_between([xlim[0], med_s], med_g, yhi, alpha=0.08, color="skyblue")
                    ax.fill_between([xlim[0], med_s], ylo, med_g, alpha=0.08, color="salmon")

                colors = {
                    "stars": "#2ca02c",
                    "cash_cows": "#ff7f0e",
                    "question_marks": "#1f77b4",
                    "dogs": "#d62728",
                    "new_market": "#9467bd",
                    "low_volume": "#aaaaaa",
                }
                for q in eligible["bcg_quadrant"].unique():
                    pts = eligible[eligible["bcg_quadrant"] == q]
                    ax.scatter(
                        pts[xcol],
                        pts[ycol],
                        c=colors.get(q, "#333"),
                        label=QUADRANT_LABELS.get(q, q),
                        s=40,
                        alpha=0.85,
                        edgecolors="white",
                        linewidths=0.5,
                    )
                if len(low):
                    ax.scatter(
                        low[xcol] if xcol in low.columns else low.get("relative_share", 0),
                        low[ycol].fillna(0) if ycol in low.columns else low.get("growth_explicit_pct", 0),
                        c="#cccccc",
                        s=25,
                        alpha=0.5,
                        marker="x",
                        label="Low volume",
                    )

                top = eligible.nlargest(8, "explicit_per_1k_listings")
                for _, r in top.iterrows():
                    ax.annotate(
                        r["city_name"][:12],
                        (r[xcol], r[ycol]),
                        fontsize=7,
                        alpha=0.9,
                    )

                ax.set_xlim(xlim)
                ax.set_ylim(ylo, yhi)
                ax.set_xlabel("Relative CVR (explicit/1k tin, ex-HCM)")
                ax.set_ylabel("Growth CVR % (H2 vs H1)")
                ax.set_title(f"{cat} — {CAT_META[cat]}")
                if idx == 0:
                    ax.legend(loc="upper left", fontsize=6, framealpha=0.9)

            axes.flat[-1].axis("off")
            path = OUT_DIR / f"fig_bcg_{region}_{ad_type}.png"
            _finish_fig(fig, path)


def qa_summary(
    rev: pd.DataFrame,
    quad_df: pd.DataFrame,
    min_n_listings: int = MIN_N_LISTINGS,
    min_n_explicit: int = MIN_N_EXPLICIT,
) -> None:
    mismatch = con.execute("""
        SELECT COUNT(*)::BIGINT AS n
        FROM events_geo
        WHERE event_category != dim_category
    """).fetchone()[0]

    unknown = rev[rev["region"] == "Unknown"]
    prep_path = OUT_DIR / "00_dim_listing_prep_summary.csv"
    prep_note = ""
    if prep_path.exists():
        prep = pd.read_csv(prep_path)
        prep_note = "\n".join(
            f"- dim prep **{r['stage']}**: {int(r['n_rows']):,}"
            for _, r in prep.iterrows()
        )

    lines = [
        "# Region revenue & BCG — summary",
        "",
        f"- EDA window: `{eda_min}` → `{eda_max}` (mid `{eda_mid}`)",
        f"- Explicit types (lead/CVR): {', '.join(EXPLICIT_TYPES)}",
        f"- BCG metric: **CVR** = explicit / tin (`explicit_per_1k_listings`)",
        f"- BCG eligibility: `n_listings >= {min_n_listings}` AND `n_explicit >= {min_n_explicit}`",
        f"- CVR growth H1→H2 uses same `n_listings` snapshot (see Cursor.md time note)",
        f"- HCM excluded from BCG median / relative CVR",
        "",
        "## dim_listing prep",
        prep_note or "- (run pipeline to generate 00_dim_listing_*.csv)",
        "",
        "## QA",
        f"- Events category ≠ dim category rows: **{mismatch:,}**",
        f"- Unknown region rows in revenue table: **{len(unknown)}**",
        "",
        "## Totals by region (explicit events)",
    ]
    reg = (
        rev.groupby("region")[["n_explicit_events", "n_positive_events"]]
        .sum()
        .reset_index()
    )
    for _, r in reg.iterrows():
        lines.append(
            f"- **{r['region']}**: explicit={int(r['n_explicit_events']):,}, "
            f"positive={int(r['n_positive_events']):,}"
        )

    hcmc = rev[rev["is_hcmc"] == 1]["n_explicit_events"].sum()
    lines.extend([
        "",
        f"- **HCM explicit** (reference only): {int(hcmc):,}",
        "",
        "## BCG quadrant counts (eligible provinces)",
    ])
    qc = (
        quad_df[quad_df["bcg_eligible"]]
        .groupby(["region", "ad_type", "bcg_quadrant"])
        .size()
        .reset_index(name="n")
    )
    for _, r in qc.iterrows():
        lines.append(f"- {r['region']} / {r['ad_type']} / {r['bcg_quadrant']}: {r['n']}")

    lines.append("\nGenerated by `run_region_revenue_bcg.py`.\n")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("§ SUMMARY.md OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-sample-frac",
        type=float,
        default=None,
        help="Optional hash sample on events (0-1) if RAM limited",
    )
    parser.add_argument(
        "--min-n-listings",
        type=int,
        default=MIN_N_LISTINGS,
        help="Min listings per province-segment for BCG eligibility",
    )
    parser.add_argument(
        "--min-n-explicit",
        type=int,
        default=MIN_N_EXPLICIT,
        help="Min explicit events per province-segment for BCG eligibility",
    )
    args = parser.parse_args()

    init_db(args.event_sample_frac)
    export_distinct_cities()
    rev = export_revenue_tables()
    export_share_ex_hcmc(rev)
    _, quad_df = build_bcg_tables(
        rev,
        min_n_listings=args.min_n_listings,
        min_n_explicit=args.min_n_explicit,
    )
    build_bcg_national_choropleth(
        rev,
        min_n_listings=args.min_n_listings,
        min_n_explicit=args.min_n_explicit,
    )
    plot_bcg_matrix(quad_df)
    qa_summary(
        rev,
        quad_df,
        min_n_listings=args.min_n_listings,
        min_n_explicit=args.min_n_explicit,
    )
    try:
        import importlib.util
        import sys as _sys

        _chor_path = Path(__file__).resolve().parent / "run_region_choropleth.py"
        _spec = importlib.util.spec_from_file_location("run_region_choropleth", _chor_path)
        _chor = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_chor)
        print("\n--- Choropleth maps ---")
        _chor.main()
    except Exception as exc:
        print("Choropleth skipped:", exc)

    try:
        _pb_path = Path(__file__).resolve().parent / "run_bcg_quadrant_playbook.py"
        _spec2 = importlib.util.spec_from_file_location(
            "run_bcg_quadrant_playbook", _pb_path
        )
        _pb = importlib.util.module_from_spec(_spec2)
        _spec2.loader.exec_module(_pb)
        print("\n--- BCG quadrant playbook ---")
        _pb.main()
    except Exception as exc:
        print("Playbook skipped:", exc)

    print("Done →", OUT_DIR)


if __name__ == "__main__":
    main()
