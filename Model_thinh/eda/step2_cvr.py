"""
Step 2 — CVR & Behavioral Analysis
- CVR = positive_events / total_events per category
- Funnel: pageview → adview → lead (view_phone + contact_*)
- Dwell-time distribution (detect ms vs sec unit automatically)
- Temporal heatmap: hour × day_of_week for each category
All heavy queries run through DuckDB on Parquet — no full 41 GB load.
"""
import os, glob, logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from config import (
    FACT_EVENTS_DIR, FACT_INTER_DIR, OUTPUT_DIR,
    CATEGORIES, CAT_COLORS, POSITIVE_EVENTS, CATEGORY_FILTER,
)

log = logging.getLogger(__name__)

DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _parquet_files(directory: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    return [f.replace("\\", "/") for f in files]


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path)


# ── 2A: CVR per category ───────────────────────────────────────────────

def analyze_cvr(conn) -> pd.DataFrame:
    log.info("2A: Computing CVR per category …")
    pos_str = ", ".join(f"'{e}'" for e in POSITIVE_EVENTS)
    files   = _parquet_files(FACT_EVENTS_DIR)

    cvr = conn.execute(f"""
        SELECT
            category,
            COUNT(*)  AS total_events,
            SUM(CASE WHEN event_type IN ({pos_str}) THEN 1 ELSE 0 END) AS positive_events,
            SUM(CASE WHEN event_type = 'pageview'          THEN 1 ELSE 0 END) AS pageviews,
            SUM(CASE WHEN event_type = 'view_phone'        THEN 1 ELSE 0 END) AS view_phone,
            SUM(CASE WHEN event_type = 'contact_chat'      THEN 1 ELSE 0 END) AS contact_chat,
            SUM(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END) AS other_interaction,
            SUM(CASE WHEN event_type = 'contact_zalo'      THEN 1 ELSE 0 END) AS contact_zalo,
            SUM(CASE WHEN event_type = 'contact_sms'       THEN 1 ELSE 0 END) AS contact_sms
        FROM read_parquet({files})
        WHERE {CATEGORY_FILTER}
        GROUP BY category
        ORDER BY category
    """).df()

    cvr["CVR"]      = (cvr["positive_events"] / cvr["total_events"] * 100).round(2)
    cvr["cat_name"] = cvr["category"].map(CATEGORIES)

    log.info("\nCVR summary:\n%s",
             cvr[["cat_name","total_events","positive_events","CVR"]].to_string(index=False))

    # ── bar chart ──
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        cvr["cat_name"], cvr["CVR"],
        color=[CAT_COLORS[c] for c in cvr["category"]], alpha=0.88,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("CVR (%)")
    ax.set_title("Conversion Rate (Positive Events / Total Events) per Category")
    ax.set_ylim(0, cvr["CVR"].max() * 1.25)
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "s2_cvr_per_category.png")

    cvr.to_csv(os.path.join(OUTPUT_DIR, "s2_cvr.csv"), index=False)
    return cvr


# ── 2B: Funnel analysis ────────────────────────────────────────────────

def analyze_funnel(conn) -> pd.DataFrame:
    """
    Funnel stages per category:
      Stage 1: pageview (browsing noise — baseline impression)
      Stage 2: other_interaction (engaged scrolling / gallery)
      Stage 3: view_phone (high-intent lead)
      Stage 4: contact_chat | contact_zalo | contact_sms (direct contact)
    """
    log.info("2B: Funnel analysis …")
    files = _parquet_files(FACT_EVENTS_DIR)

    funnel = conn.execute(f"""
        SELECT
            category,
            SUM(CASE WHEN event_type = 'pageview'                                   THEN 1 ELSE 0 END) AS s1_pageview,
            SUM(CASE WHEN event_type = 'other_interaction'                           THEN 1 ELSE 0 END) AS s2_other_interaction,
            SUM(CASE WHEN event_type = 'view_phone'                                 THEN 1 ELSE 0 END) AS s3_view_phone,
            SUM(CASE WHEN event_type IN ('contact_chat','contact_zalo','contact_sms') THEN 1 ELSE 0 END) AS s4_direct_contact
        FROM read_parquet({files})
        WHERE {CATEGORY_FILTER}
        GROUP BY category
        ORDER BY category
    """).df()

    funnel["cat_name"] = funnel["category"].map(CATEGORIES)

    # Compute drop-off rates
    funnel["pv_to_other"]   = (funnel["s2_other_interaction"] / funnel["s1_pageview"] * 100).round(1)
    funnel["pv_to_phone"]   = (funnel["s3_view_phone"]        / funnel["s1_pageview"] * 100).round(1)
    funnel["pv_to_contact"] = (funnel["s4_direct_contact"]    / funnel["s1_pageview"] * 100).round(1)

    log.info("\nFunnel drop-off:\n%s",
             funnel[["cat_name","pv_to_other","pv_to_phone","pv_to_contact"]].to_string(index=False))

    # ── grouped bar plot ──
    stages = ["s1_pageview", "s2_other_interaction", "s3_view_phone", "s4_direct_contact"]
    stage_labels = ["Pageview", "Other Interaction", "View Phone", "Direct Contact"]
    cats = list(CATEGORIES.keys())
    x = np.arange(len(cats))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (stage, label) in enumerate(zip(stages, stage_labels)):
        vals = [funnel.loc[funnel["category"] == c, stage].values[0] for c in cats]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORIES[c] for c in cats], rotation=20, ha="right")
    ax.set_ylabel("Event count")
    ax.set_title("Event Funnel per Category")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}K"))
    _save(fig, "s2_funnel.png")

    funnel.to_csv(os.path.join(OUTPUT_DIR, "s2_funnel.csv"), index=False)
    return funnel


# ── 2C: Dwell time distribution ───────────────────────────────────────

def analyze_dwell_time(conn) -> pd.DataFrame:
    """
    dwell_time_sec is 62 % null (only populated on ad_view events).
    Suspected unit issue: raw median ~18,052 → likely milliseconds (18 s).
    We auto-detect by checking if median > 1 000 → assume ms, convert to s.
    """
    log.info("2C: Dwell time distribution …")
    files = _parquet_files(FACT_EVENTS_DIR)

    stats = conn.execute(f"""
        SELECT
            category,
            COUNT(*)                                          AS n_with_dwell,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY dwell_time_sec) AS p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY dwell_time_sec) AS p50,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY dwell_time_sec) AS p75,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY dwell_time_sec) AS p90,
            AVG(dwell_time_sec)                               AS mean_dwell
        FROM read_parquet({files})
        WHERE dwell_time_sec IS NOT NULL AND dwell_time_sec > 0
          AND {CATEGORY_FILTER}
        GROUP BY category
        ORDER BY category
    """).df()

    # Unit detection: if median > 1000, likely stored as milliseconds
    if stats["p50"].mean() > 1000:
        log.warning("  dwell_time_sec median = %.0f → likely MILLISECONDS. Converting to seconds.", stats["p50"].mean())
        for col in ["p25", "p50", "p75", "p90", "mean_dwell"]:
            stats[col] = stats[col] / 1000
        stats["unit"] = "ms→sec"
    else:
        stats["unit"] = "sec"

    stats["cat_name"] = stats["category"].map(CATEGORIES)
    log.info("\nDwell time (seconds) stats:\n%s",
             stats[["cat_name","p25","p50","p75","p90","mean_dwell","unit"]].round(1).to_string(index=False))

    # ── box-like bar chart (p25-p75 IQR + median line) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stats))
    iqr_low  = stats["p25"].values
    iqr_high = stats["p75"].values
    median   = stats["p50"].values

    ax.bar(x, iqr_high - iqr_low, bottom=iqr_low, width=0.5,
           color=[CAT_COLORS[c] for c in stats["category"]], alpha=0.7, label="IQR (p25–p75)")
    ax.scatter(x, median, color="black", zorder=5, s=60, label="Median")
    ax.scatter(x, stats["p90"].values, color="red", zorder=5, s=40, marker="^", label="p90")

    ax.set_xticks(x)
    ax.set_xticklabels(stats["cat_name"], rotation=20, ha="right")
    ax.set_ylabel("Dwell time (seconds)")
    ax.set_title("Dwell Time Distribution per Category\n(non-null ad_view events only)")
    ax.legend()
    _save(fig, "s2_dwell_time.png")

    stats.to_csv(os.path.join(OUTPUT_DIR, "s2_dwell_time.csv"), index=False)
    return stats


# ── 2D: Temporal heatmap ──────────────────────────────────────────────

def analyze_temporal(conn):
    """
    Aggregate event counts by hour_of_day × day_of_week × category.
    Produces: one overall heatmap + one per category (positive events only).
    """
    log.info("2D: Temporal heatmap (hour × day) …")
    pos_str = ", ".join(f"'{e}'" for e in POSITIVE_EVENTS)
    files   = _parquet_files(FACT_EVENTS_DIR)

    temporal = conn.execute(f"""
        SELECT
            category,
            EXTRACT(hour FROM event_ts)::INT AS hour_of_day,
            EXTRACT(dow  FROM event_ts)::INT AS day_of_week,
            COUNT(*) AS total_events,
            SUM(CASE WHEN event_type IN ({pos_str}) THEN 1 ELSE 0 END) AS positive_events
        FROM read_parquet({files})
        WHERE {CATEGORY_FILTER}
        GROUP BY category, hour_of_day, day_of_week
        ORDER BY category, day_of_week, hour_of_day
    """).df()

    temporal["cat_name"] = temporal["category"].map(CATEGORIES)
    temporal.to_csv(os.path.join(OUTPUT_DIR, "s2_temporal.csv"), index=False)

    # ── Overall heatmap (all categories combined) ──
    overall = (
        temporal.groupby(["hour_of_day", "day_of_week"])
        ["positive_events"].sum()
        .reset_index()
    )
    pivot_all = overall.pivot(index="day_of_week", columns="hour_of_day", values="positive_events").fillna(0)
    pivot_all.index = [DAYS[i] for i in pivot_all.index]

    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(pivot_all, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Positive Events"},
                linewidths=0.3, linecolor="white")
    ax.set_title("Positive Events — Hour × Day of Week (All Categories)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    _save(fig, "s2_temporal_overall.png")

    # ── Per-category heatmaps ──
    fig, axes = plt.subplots(1, 5, figsize=(24, 4), sharey=True)
    for ax, (cat, name) in zip(axes, CATEGORIES.items()):
        sub = temporal[temporal["category"] == cat]
        piv = sub.pivot(index="day_of_week", columns="hour_of_day", values="positive_events").fillna(0)
        # Ensure full grid
        piv = piv.reindex(index=range(7), columns=range(24), fill_value=0)
        piv.index = DAYS
        sns.heatmap(piv, cmap="YlOrRd", ax=ax, cbar=False,
                    linewidths=0.2, linecolor="white")
        ax.set_title(f"{name}\n({cat})", fontsize=9)
        ax.set_xlabel("Hour")
        ax.set_ylabel("")
    fig.suptitle("Positive Events Heatmap per Category (Hour × Day)", fontsize=11)
    fig.tight_layout()
    _save(fig, "s2_temporal_per_category.png")

    log.info("Temporal analysis complete.")
    return temporal


# ── main entry ─────────────────────────────────────────────────────────

def run_step2(conn):
    log.info("=" * 60)
    log.info("STEP 2 — CVR & Behavioral Analysis")
    log.info("=" * 60)

    analyze_cvr(conn)
    analyze_funnel(conn)
    analyze_dwell_time(conn)
    analyze_temporal(conn)

    log.info("Step 2 complete. Outputs in %s", OUTPUT_DIR)
