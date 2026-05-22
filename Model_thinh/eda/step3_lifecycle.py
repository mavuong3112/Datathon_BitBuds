"""
Step 3 — Lifecycle & Feature Differentiation
- Decay curve: listing_age_days → contacts_24h / views_24h
- Price sensitivity: price_bucket → positive event rate
- Area sensitivity: area_sqm buckets → contact rate
- Geographic hotspots: district/ward with highest lead density
All aggregation-heavy queries run via DuckDB.
"""
import os, glob, logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import (
    FACT_SNAPSHOT_DIR, FACT_EVENTS_DIR, OUTPUT_DIR,
    CATEGORIES, CAT_COLORS, POSITIVE_EVENTS, CATEGORY_FILTER,
)

log = logging.getLogger(__name__)


def _parquet_files(directory: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    return [f.replace("\\", "/") for f in files]


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path)


# ── 3A: Decay effect ──────────────────────────────────────────────────

def analyze_decay(conn) -> pd.DataFrame:
    """
    Listing age (days) vs avg daily contacts and views.
    Uses fact_listing_snapshot. Age bucketed to 5-day bins up to 180 days.
    """
    log.info("3A: Decay effect (age vs contacts/views) …")
    snap_files = _parquet_files(FACT_SNAPSHOT_DIR)

    decay = conn.execute(f"""
        SELECT
            item_id,
            FLOOR(listing_age_days / 5) * 5  AS age_bucket,
            AVG(views_24h)                    AS avg_views,
            AVG(contacts_24h)                 AS avg_contacts
        FROM read_parquet({snap_files})
        WHERE listing_age_days >= 0
          AND listing_age_days <= 180
        GROUP BY item_id, age_bucket
    """).df()

    # Aggregate per age bucket (across all items)
    agg = (
        decay.groupby("age_bucket")
        .agg(avg_views=("avg_views", "mean"), avg_contacts=("avg_contacts", "mean"), n=("item_id", "nunique"))
        .reset_index()
        .sort_values("age_bucket")
    )

    log.info("  Decay profile computed for %d age buckets.", len(agg))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(agg["age_bucket"], agg["avg_views"], marker="o", ms=4, color="#2171b5")
    axes[0].set_xlabel("Listing age (days)")
    axes[0].set_ylabel("Avg daily views")
    axes[0].set_title("Views Decay Curve (all categories)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(agg["age_bucket"], agg["avg_contacts"], marker="o", ms=4, color="#e6550d")
    axes[1].set_xlabel("Listing age (days)")
    axes[1].set_ylabel("Avg daily contacts")
    axes[1].set_title("Contacts Decay Curve (all categories)")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Listing Decay Effect: Engagement vs. Age (0–180 days, 5-day bins)", fontsize=11)
    fig.tight_layout()
    _save(fig, "s3_decay_overall.png")

    # Per-category decay — join with dim to get category
    # Use fact_events instead (has category column directly on snapshot)
    snap_files2 = _parquet_files(FACT_SNAPSHOT_DIR)
    evt_files   = _parquet_files(FACT_EVENTS_DIR)

    # fact_listing_snapshot has no category → get from fact_user_events item_id→category map
    log.info("  Building item→category map from fact_user_events sample …")
    item_cat = conn.execute(f"""
        SELECT DISTINCT item_id, category
        FROM read_parquet({evt_files})
        WHERE {CATEGORY_FILTER}
    """).df()
    # Register as DuckDB table for join
    conn.register("item_cat", item_cat)

    decay_cat = conn.execute(f"""
        SELECT
            ic.category,
            FLOOR(s.listing_age_days / 5) * 5 AS age_bucket,
            AVG(s.contacts_24h)                AS avg_contacts,
            AVG(s.views_24h)                   AS avg_views,
            APPROX_COUNT_DISTINCT(s.item_id)    AS n_items
        FROM read_parquet({snap_files2}) s
        JOIN item_cat ic ON s.item_id = ic.item_id
        WHERE s.listing_age_days >= 0
          AND s.listing_age_days <= 180
          AND ic.category IN (1010, 1020, 1030, 1040, 1050)
        GROUP BY ic.category, age_bucket
        ORDER BY ic.category, age_bucket
    """).df()

    decay_cat["cat_name"] = decay_cat["category"].map(CATEGORIES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cat, name in CATEGORIES.items():
        sub = decay_cat[decay_cat["category"] == cat].sort_values("age_bucket")
        if sub.empty:
            continue
        axes[0].plot(sub["age_bucket"], sub["avg_views"],    label=name, color=CAT_COLORS[cat])
        axes[1].plot(sub["age_bucket"], sub["avg_contacts"], label=name, color=CAT_COLORS[cat])

    for ax, title in zip(axes, ["Views Decay", "Contacts Decay"]):
        ax.set_xlabel("Listing age (days)")
        ax.set_ylabel("Avg daily value")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Listing Decay per Category (0–180 days, 5-day bins)", fontsize=11)
    fig.tight_layout()
    _save(fig, "s3_decay_per_category.png")

    decay_cat.to_csv(os.path.join(OUTPUT_DIR, "s3_decay.csv"), index=False)
    return decay_cat


# ── 3B: Price sensitivity ─────────────────────────────────────────────

def analyze_price_sensitivity(conn, dim: pd.DataFrame):
    """
    For each category: positive event count per price_bucket.
    Join dim_listing (price_bucket) with fact_user_events (positive events).
    """
    log.info("3B: Price sensitivity analysis …")
    pos_str   = ", ".join(f"'{e}'" for e in POSITIVE_EVENTS)
    evt_files = _parquet_files(FACT_EVENTS_DIR)

    # Aggregate positive events per item_id from events
    log.info("  Aggregating positive events per item …")
    item_leads = conn.execute(f"""
        SELECT
            item_id,
            category,
            COUNT(*) AS positive_events
        FROM read_parquet({evt_files})
        WHERE event_type IN ({pos_str})
          AND {CATEGORY_FILTER}
        GROUP BY item_id, category
    """).df()

    # Join with dim for price_bucket
    dim_price = dim[["item_id", "category", "price_bucket"]].dropna(subset=["price_bucket"])
    merged = item_leads.merge(dim_price, on=["item_id", "category"], how="inner")

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (cat, name) in zip(axes, CATEGORIES.items()):
        sub = merged[merged["category"] == cat]
        if sub.empty:
            ax.set_visible(False)
            continue
        pricegrp = (
            sub.groupby("price_bucket")["positive_events"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "total_leads", "count": "n_items"})
        )
        pricegrp["lead_per_item"] = pricegrp["total_leads"] / pricegrp["n_items"]
        pricegrp = pricegrp.sort_values("lead_per_item", ascending=False).head(15)

        ax.barh(pricegrp["price_bucket"], pricegrp["lead_per_item"],
                color=CAT_COLORS[cat], alpha=0.85)
        ax.set_title(f"{name}\n({cat})", fontsize=8)
        ax.set_xlabel("Avg leads/item")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Top Price Buckets by Avg Leads per Item per Category", fontsize=11)
    fig.tight_layout()
    _save(fig, "s3_price_sensitivity.png")

    merged.groupby(["category", "price_bucket"])["positive_events"].agg(["sum","count"]).reset_index()\
        .to_csv(os.path.join(OUTPUT_DIR, "s3_price_sensitivity.csv"), index=False)


# ── 3C: Area sensitivity ──────────────────────────────────────────────

def analyze_area_sensitivity(conn, dim: pd.DataFrame):
    """
    Bucket area_sqm (log scale) and compare avg lead rates.
    Clip area outliers at 500 m² for visualization sanity.
    """
    log.info("3C: Area sensitivity analysis …")
    pos_str   = ", ".join(f"'{e}'" for e in POSITIVE_EVENTS)
    evt_files = _parquet_files(FACT_EVENTS_DIR)

    item_leads = conn.execute(f"""
        SELECT item_id, category, COUNT(*) AS leads
        FROM read_parquet({evt_files})
        WHERE event_type IN ({pos_str})
          AND {CATEGORY_FILTER}
        GROUP BY item_id, category
    """).df()

    # Clip & bucket area
    dim_area = dim[["item_id", "category", "area_sqm"]].dropna(subset=["area_sqm"])
    dim_area = dim_area[dim_area["area_sqm"].between(1, 1000)].copy()
    bins  = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
    labels = ["<25", "25-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300-500", "500-1000"]
    dim_area["area_bucket"] = pd.cut(dim_area["area_sqm"], bins=bins, labels=labels)

    merged = item_leads.merge(dim_area, on=["item_id", "category"], how="inner")

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (cat, name) in zip(axes, CATEGORIES.items()):
        sub = merged[merged["category"] == cat]
        if sub.empty:
            ax.set_visible(False)
            continue
        grp = (
            sub.groupby("area_bucket", observed=True)["leads"]
            .agg(["sum", "count"])
            .reset_index()
        )
        grp["lead_per_item"] = grp["sum"] / grp["count"]
        ax.bar(grp["area_bucket"].astype(str), grp["lead_per_item"],
               color=CAT_COLORS[cat], alpha=0.85)
        ax.set_title(f"{name}\n({cat})", fontsize=8)
        ax.set_xlabel("Area (m²)")
        ax.set_ylabel("Avg leads/item")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("Area Sweet Spot: Avg Leads per Item per Area Bucket", fontsize=11)
    fig.tight_layout()
    _save(fig, "s3_area_sensitivity.png")

    merged.groupby(["category","area_bucket"], observed=True)["leads"].agg(["sum","count"]).reset_index()\
        .to_csv(os.path.join(OUTPUT_DIR, "s3_area_sensitivity.csv"), index=False)


# ── 3D: Geographic hotspots ────────────────────────────────────────────

def analyze_geo_hotspots(conn, dim: pd.DataFrame):
    """
    Top 20 districts by total positive leads per category.
    Uses dim_listing city_name + district_name joined with fact_user_events.
    """
    log.info("3D: Geographic hotspot analysis …")
    pos_str   = ", ".join(f"'{e}'" for e in POSITIVE_EVENTS)
    evt_files = _parquet_files(FACT_EVENTS_DIR)

    item_leads = conn.execute(f"""
        SELECT item_id, category, COUNT(*) AS leads
        FROM read_parquet({evt_files})
        WHERE event_type IN ({pos_str})
          AND {CATEGORY_FILTER}
        GROUP BY item_id, category
    """).df()

    dim_geo = dim[["item_id", "category", "city_name", "district_name"]].dropna(subset=["district_name"])
    merged  = item_leads.merge(dim_geo, on=["item_id", "category"], how="inner")
    merged["location"] = merged["city_name"] + " / " + merged["district_name"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 8))
    for ax, (cat, name) in zip(axes, CATEGORIES.items()):
        sub = merged[merged["category"] == cat]
        if sub.empty:
            ax.set_visible(False)
            continue
        geo = (
            sub.groupby("location")["leads"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )
        ax.barh(geo["location"], geo["leads"],
                color=CAT_COLORS[cat], alpha=0.85)
        ax.set_title(f"{name}\n({cat})", fontsize=8)
        ax.set_xlabel("Total leads")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=6)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1e3)}K" if v >= 1000 else str(int(v))))

    fig.suptitle("Top 20 Geographic Hotspots by Lead Count per Category", fontsize=11)
    fig.tight_layout()
    _save(fig, "s3_geo_hotspots.png")

    geo_agg = merged.groupby(["category","city_name","district_name"])["leads"].sum().reset_index()
    geo_agg.sort_values(["category","leads"], ascending=[True, False])\
        .to_csv(os.path.join(OUTPUT_DIR, "s3_geo_hotspots.csv"), index=False)


# ── main entry ─────────────────────────────────────────────────────────

def run_step3(conn, dim: pd.DataFrame):
    log.info("=" * 60)
    log.info("STEP 3 — Lifecycle & Feature Differentiation")
    log.info("=" * 60)

    analyze_decay(conn)
    analyze_price_sensitivity(conn, dim)
    analyze_area_sensitivity(conn, dim)
    analyze_geo_hotspots(conn, dim)

    log.info("Step 3 complete. Outputs in %s", OUTPUT_DIR)
