"""
Step 1 — Data Quality & Overview
- Missing value rates per category (with business-valid null annotation)
- Time coverage of dim_listing vs. fact period
- Scale: unique items / users / events per category (via DuckDB)
- Sub-setting: category × seller_type × ad_type breakdowns
"""
import os, glob, logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DIM_LISTING_DIR, FACT_INTER_DIR, FACT_EVENTS_DIR, OUTPUT_DIR,
    CATEGORIES, CAT_COLORS, VALID_NULLS, FACT_START, FACT_END, DIM_START,
    CATEGORY_FILTER,
)

log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────

def _parquet_files(directory: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    return [f.replace("\\", "/") for f in files]


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path)


# ── 1A: Load dim_listing ───────────────────────────────────────────────

def load_dim_listing() -> pd.DataFrame:
    """Load all 40 dim_listing parquet files (~3.1 M rows, ~2–3 GB in RAM)."""
    files = _parquet_files(DIM_LISTING_DIR)
    log.info("Loading dim_listing: %d files …", len(files))
    chunks = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        chunks.append(df)
        if (i + 1) % 10 == 0:
            log.info("  … %d / %d files loaded", i + 1, len(files))
    dim = pd.concat(chunks, ignore_index=True)
    log.info("dim_listing loaded: %s rows × %s cols", f"{len(dim):,}", dim.shape[1])
    return dim


# ── 1B: Time coverage ─────────────────────────────────────────────────

def analyze_time_coverage(dim: pd.DataFrame) -> pd.DataFrame:
    """
    dim_listing covers 2024-09-15 → 2026-04-09, but fact tables only start
    2025-11-09. Items posted before FACT_START may still be 'alive' during
    the fact window (expected_expired_date >= FACT_START).
    """
    dim["posted_date"] = pd.to_datetime(dim["posted_date"])
    dim["expected_expired_date"] = pd.to_datetime(dim["expected_expired_date"])

    fact_start = pd.Timestamp(FACT_START)
    fact_end   = pd.Timestamp(FACT_END)

    dim["in_fact_window"] = (
        (dim["expected_expired_date"] >= fact_start) &
        (dim["posted_date"]           <= fact_end)
    )
    dim["pre_fact"] = dim["posted_date"] < fact_start

    rows = []
    for cat, name in CATEGORIES.items():
        sub = dim[dim["category"] == cat]
        rows.append({
            "category":          cat,
            "name":              name,
            "total_listings":    len(sub),
            "pre_fact_period":   int(sub["pre_fact"].sum()),
            "in_fact_window":    int(sub["in_fact_window"].sum()),
            "pct_pre_fact":      round(sub["pre_fact"].mean() * 100, 1),
            "pct_in_window":     round(sub["in_fact_window"].mean() * 100, 1),
        })

    tbl = pd.DataFrame(rows)
    log.info("\nTime Coverage:\n%s", tbl.to_string(index=False))

    # ── plot ──
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(CATEGORIES))
    w = 0.35
    ax.bar(x - w/2, tbl["total_listings"],    w, label="Total listings",       color="#9ecae1")
    ax.bar(x + w/2, tbl["in_fact_window"],    w, label="Active in fact window", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['name']}\n({r['category']})" for _, r in tbl.iterrows()], fontsize=9)
    ax.set_ylabel("Listing count")
    ax.set_title("dim_listing: Total vs. Active in Fact Window (Nov 2025 – Apr 2026)")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _save(fig, "s1_time_coverage.png")

    tbl.to_csv(os.path.join(OUTPUT_DIR, "s1_time_coverage.csv"), index=False)
    return dim   # with new columns


# ── 1C: Missing values per category ───────────────────────────────────

def analyze_missing_values(dim: pd.DataFrame):
    """Compute null rates per column per category, annotate business-valid nulls."""
    dim_cols = [
        "area_sqm", "bedrooms", "bathrooms", "floors", "width_m",
        "direction", "legal_status", "house_type", "furnishing",
        "project_id", "price_bucket", "images_count",
    ]

    records = []
    for cat, name in CATEGORIES.items():
        sub = dim[dim["category"] == cat]
        n   = len(sub)
        for col in dim_cols:
            null_pct = round(sub[col].isnull().mean() * 100, 1)
            valid    = col in VALID_NULLS.get(cat, [])
            records.append({
                "category": cat, "name": name, "column": col,
                "null_pct": null_pct, "valid_null": valid, "n": n,
            })

    mv = pd.DataFrame(records)

    # ── heatmap ──
    pivot = mv.pivot(index="column", columns="name", values="null_pct")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=0, vmax=100,
        cbar_kws={"label": "Null %"},
    )
    ax.set_title("Missing Value Rate (%) by Column × Category\n"
                 "Note: some 100% nulls are business-valid (see annotation)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    _save(fig, "s1_missing_heatmap.png")

    # ── valid-null annotation table ──
    valid_df = mv[mv["valid_null"]]
    log.info("\nBusiness-Valid 100%% Nulls:\n%s", valid_df[["category","name","column","null_pct"]].to_string(index=False))

    mv.to_csv(os.path.join(OUTPUT_DIR, "s1_missing_values.csv"), index=False)
    log.info("Missing-values heatmap saved.")
    return mv


# ── 1D: Scale measurement via DuckDB ──────────────────────────────────

def analyze_scale(conn) -> pd.DataFrame:
    """
    Count unique items / users / events per category.
    Uses DuckDB to avoid loading 41 GB of fact_user_events into RAM.
    """
    log.info("Measuring scale (DuckDB queries on Parquet files) …")

    # -- unique items in dim per category (already in RAM)
    # -- unique items + users in fact_post_contact_interactions
    inter_files = _parquet_files(FACT_INTER_DIR)
    events_files = _parquet_files(FACT_EVENTS_DIR)

    log.info("  Querying fact_post_contact_interactions …")
    inter_scale = conn.execute(f"""
        SELECT
            category,
            APPROX_COUNT_DISTINCT(item_id)  AS unique_items_inter,
            APPROX_COUNT_DISTINCT(user_id)  AS unique_users_inter,
            COUNT(*)                        AS total_interaction_rows
        FROM read_parquet({inter_files})
        WHERE {CATEGORY_FILTER}
        GROUP BY category
        ORDER BY category
    """).df()

    log.info("  Querying fact_user_events (500 files — using APPROX_COUNT_DISTINCT) …")
    events_scale = conn.execute(f"""
        SELECT
            category,
            is_login,
            APPROX_COUNT_DISTINCT(item_id)                AS unique_items_events,
            APPROX_COUNT_DISTINCT(
                CASE WHEN is_login = 'login' THEN user_id END
            )                                             AS unique_login_users,
            COUNT(*)                                      AS total_events,
            SUM(CASE WHEN event_type IN
                ('view_phone','contact_chat','other_interaction',
                 'contact_zalo','contact_sms') THEN 1 ELSE 0 END
            )                                             AS positive_events
        FROM read_parquet({events_files})
        WHERE {CATEGORY_FILTER}
        GROUP BY category, is_login
        ORDER BY category, is_login
    """).df()

    log.info("\nInteraction scale:\n%s", inter_scale.to_string(index=False))
    log.info("\nEvents scale:\n%s", events_scale.to_string(index=False))

    inter_scale.to_csv(os.path.join(OUTPUT_DIR, "s1_scale_interactions.csv"), index=False)
    events_scale.to_csv(os.path.join(OUTPUT_DIR, "s1_scale_events.csv"), index=False)

    # Log and drop any unexpected category codes not in our schema
    known_cats = set(CATEGORIES.keys())
    unexpected = set(events_scale["category"].unique()) - known_cats
    if unexpected:
        log.warning("  Unexpected category codes in events (will be excluded from plots): %s", unexpected)

    # ── bar chart: unique login users per category ──
    ev_login = events_scale[
        (events_scale["is_login"] == "login") &
        (events_scale["category"].isin(known_cats))
    ].copy()
    ev_login["cat_name"] = ev_login["category"].map(CATEGORIES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        ev_login["cat_name"],
        ev_login["unique_login_users"],
        color=[CAT_COLORS[c] for c in ev_login["category"]],
    )
    axes[0].set_title("Unique Login Users per Category\n(fact_user_events)")
    axes[0].set_ylabel("Users")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    axes[1].bar(
        ev_login["cat_name"],
        ev_login["total_events"],
        color=[CAT_COLORS[c] for c in ev_login["category"]],
    )
    axes[1].set_title("Total Events per Category\n(fact_user_events, login)")
    axes[1].set_ylabel("Events")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    _save(fig, "s1_scale.png")

    return events_scale


# ── 1E: Category × ad_type × seller_type breakdown ────────────────────

def analyze_subset_breakdown(dim: pd.DataFrame):
    """Per-category distribution of ad_type and seller_type."""
    sub = (
        dim.groupby(["category", "ad_type", "seller_type"])
        .size()
        .reset_index(name="count")
    )
    sub["cat_name"] = sub["category"].map(CATEGORIES)

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
    for ax, (cat, name) in zip(axes, CATEGORIES.items()):
        grp = sub[sub["category"] == cat].copy()
        grp["label"] = grp["ad_type"] + " / " + grp["seller_type"]
        ax.barh(grp["label"], grp["count"], color=CAT_COLORS[cat], alpha=0.85)
        ax.set_title(f"{name}\n({cat})", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v/1000)}K"))
        ax.invert_yaxis()
    fig.suptitle("Listing Count by ad_type × seller_type per Category", fontsize=11)
    fig.tight_layout()
    _save(fig, "s1_subset_breakdown.png")

    sub.to_csv(os.path.join(OUTPUT_DIR, "s1_subset_breakdown.csv"), index=False)


# ── main entry ─────────────────────────────────────────────────────────

def run_step1(conn) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 1 — Data Quality & Overview")
    log.info("=" * 60)

    dim = load_dim_listing()
    dim = analyze_time_coverage(dim)
    analyze_missing_values(dim)
    analyze_subset_breakdown(dim)
    analyze_scale(conn)

    log.info("Step 1 complete. Outputs in %s", OUTPUT_DIR)
    return dim
