"""
EDA sâu fact_listing_snapshot — pipeline theo step, full data.

CLI: python eda_listing_snapshot_temporal_deep.py --step all
Notebook: import module + %matplotlib inline (xem eda_listing_snapshot_temporal_deep.ipynb)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path(__file__).resolve().parent
OUT_ROOT = DATA_ROOT / "outputs" / "eda_listing_snapshot"
CACHE_DIR = OUT_ROOT / "_cache"
QA_DIR = OUT_ROOT / "_qa"
CROSS_DIR = OUT_ROOT / "_cross"
STATUS_FILE = CACHE_DIR / "pipeline_status.json"

SNAP_CACHE = CACHE_DIR / "snap_enriched.parquet"
SNAP_CACHE_PART = CACHE_DIR / "snap_enriched"

DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
SNAP_GLOB = str(DATA_ROOT / "fact_listing_snapshot" / "*.parquet")
INTER_GLOB = str(DATA_ROOT / "fact_post_contact_interactions" / "*.parquet")

DATA_RANGE = "fact_listing_snapshot · 2025-11-09 → 2026-04-09 · full data"
DOW_ORDER = [1, 2, 3, 4, 5, 6, 0]  # T2 … CN

DUCKDB_MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "4GB")
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "2"))
EVENT_SAMPLE_FRAC: float | None = None
MIN_CELL_N = 1000

LOGIN_WHERE = "is_login = 'login'"
EXPLICIT_SQL = ", ".join(
    repr(x) for x in ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
)

CAT_META = {
    1010: "1010 — Căn hộ / Chung cư",
    1020: "1020 — Nhà ở",
    1030: "1030 — Văn phòng / Mặt bằng",
    1040: "1040 — Đất",
    1050: "1050 — Phòng trọ",
}
CATEGORIES = (1010, 1020, 1030, 1040, 1050)
PALETTE = {
    1010: "#238b45",
    1020: "#2171b5",
    1030: "#6a51a3",
    1040: "#cb181d",
    1050: "#d94801",
}
DOW_LABELS = {0: "CN", 1: "T2", 2: "T3", 3: "T4", 4: "T5", 5: "T6", 6: "T7"}
AD_TYPE_LABELS = {"sell": "Bán", "let": "Cho thuê"}

CHART_LABELS = {
    "views_24h": "Lượt xem TB / ngày-tin (views_24h)",
    "contacts_24h": "Lượt liên hệ TB / ngày-tin (contacts_24h)",
    "contact_rate_pct": "Tỷ lệ liên hệ (%) = Σcontacts ÷ Σviews",
    "listing_age_days": "Tuổi tin (ngày kể từ ngày đăng)",
    "dow": "Thứ trong tuần",
    "month": "Tháng (YYYY-MM)",
    "hour": "Giờ trong ngày (0–23h)",
    "explicit_contacts": "Liên hệ thật (SĐT, chat, Zalo, SMS) — user đăng nhập",
    "pageviews": "Lượt xem trang tin (pageview)",
    "leads": "Lượt lộ SĐT/email (lead_count)",
    "heatmap_contact_rate": "Tỷ lệ liên hệ (%)",
    "heatmap_explicit": "Số sự kiện liên hệ thật",
    "heatmap_hour_share": "Tỷ trọng liên hệ (%) trong category",
}


def configure_matplotlib(notebook: bool = False) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".mplconfig"))
    if not notebook:
        import matplotlib

        matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="notebook")


def mpl_show_default() -> bool:
    return os.environ.get("MPL_SHOW", "0").strip() in ("1", "true", "yes")


def age_bucket_sql(col: str = "listing_age_days") -> str:
    return f"""
        CASE
            WHEN {col} IS NULL THEN 'unk'
            WHEN {col} <= 2 THEN '0-2d'
            WHEN {col} <= 7 THEN '3-7d'
            WHEN {col} <= 14 THEN '8-14d'
            WHEN {col} <= 30 THEN '15-30d'
            WHEN {col} <= 60 THEN '31-60d'
            ELSE '60+d'
        END
    """


def cat_out(cat: int) -> Path:
    p = OUT_ROOT / str(cat)
    p.mkdir(parents=True, exist_ok=True)
    return p


def events_cache_path(cat: int) -> Path:
    return CACHE_DIR / f"events_hour_agg_{cat}.parquet"


def login_cache_path(cat: int) -> Path:
    return CACHE_DIR / f"login_users_{cat}.parquet"


def show_fig(path: Path | None = None, dpi: int = 130, show: bool | None = None) -> None:
    if show is None:
        show = mpl_show_default()
    plt.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        if not show:
            print("Saved", path)
    if show:
        plt.show()
    else:
        plt.close()


def chart_subtitle(n_listing_days: int | None = None, n_events: int | None = None) -> str:
    parts = [DATA_RANGE]
    if n_listing_days is not None:
        parts.append(f"n = {n_listing_days:,} listing-days")
    if n_events is not None:
        parts.append(f"n = {n_events:,} events (login)")
    return " · ".join(parts)


def dow_label_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: DOW_LABELS.get(int(x), str(x)))


def sort_dow_df(df: pd.DataFrame, dow_col: str = "dow") -> pd.DataFrame:
    if dow_col not in df.columns or df.empty:
        return df
    order_map = {d: i for i, d in enumerate(DOW_ORDER)}
    out = df.copy()
    out["_ord"] = out[dow_col].map(lambda x: order_map.get(int(x), 99))
    return out.sort_values("_ord").drop(columns="_ord")


def mark_status(step: str, detail: str = "ok") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    status = {}
    if STATUS_FILE.exists():
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    status[step] = {"detail": detail}
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{CACHE_DIR / 'duckdb_tmp'}'")
    return con


def snap_glob_path() -> str:
    if SNAP_CACHE.exists():
        return str(SNAP_CACHE)
    if SNAP_CACHE_PART.exists():
        return str(SNAP_CACHE_PART / "**" / "*.parquet")
    return ""


def register_snap_view(con: duckdb.DuckDBPyConnection) -> None:
    glob = snap_glob_path()
    if not glob:
        raise FileNotFoundError("Chưa có cache snap_enriched. Chạy: --step cache_snap")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW snap_enriched AS
        SELECT * FROM read_parquet('{glob}')
        """
    )


def check_data_dirs() -> None:
    for name in (
        "dim_listing",
        "fact_user_events",
        "fact_listing_snapshot",
        "fact_post_contact_interactions",
    ):
        if not (DATA_ROOT / name).exists():
            raise FileNotFoundError(f"Thiếu `{name}` trong {DATA_ROOT}")


def ensure_caches(con: duckdb.DuckDBPyConnection, cats: tuple[int, ...] = CATEGORIES) -> None:
    if not SNAP_CACHE.exists():
        step_cache_snap(con)
    step_cache_events(con, cats)
    step_cache_login(con, cats)


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_daily_trend(
    trend: pd.DataFrame, label: str, color: str, out: Path | None, show: bool
) -> None:
    n = int(trend["listing_days"].sum()) if "listing_days" in trend.columns else len(trend)
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ln1 = ax1.plot(
        trend["date"],
        trend["avg_views_24h"],
        color="#2171b5",
        label=CHART_LABELS["views_24h"],
    )
    ax1.set_ylabel(CHART_LABELS["views_24h"], color="#2171b5")
    ax1.tick_params(axis="y", labelcolor="#2171b5")
    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        trend["date"],
        trend["avg_contacts_24h"],
        color="#cb181d",
        label=CHART_LABELS["contacts_24h"],
    )
    ax2.set_ylabel(CHART_LABELS["contacts_24h"], color="#cb181d")
    ax2.tick_params(axis="y", labelcolor="#cb181d")
    ax1.set_xlabel("Ngày chụp snapshot")
    ax1.set_title(
        f"{label}\nXu hướng exposure & liên hệ theo ngày",
        fontsize=12,
        fontweight="bold",
    )
    ax1.text(0.01, -0.22, chart_subtitle(n), transform=ax1.transAxes, fontsize=9, color="#444")
    ax1.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], loc="upper left", fontsize=8)
    show_fig(out, show=show)


def plot_dow_bars(dow: pd.DataFrame, label: str, color: str, out: Path | None, show: bool) -> None:
    dow = sort_dow_df(dow)
    dow = dow.copy()
    dow["dow_label"] = dow_label_series(dow["dow"])
    n = int(dow["listing_days"].sum())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(dow))
    w = 0.38
    ax.bar(x - w / 2, dow["avg_views_24h"], width=w, label=CHART_LABELS["views_24h"], color="#9ecae1")
    ax.bar(x + w / 2, dow["avg_contacts_24h"], width=w, label=CHART_LABELS["contacts_24h"], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(dow["dow_label"])
    ax.set_ylabel("Giá trị trung bình / ngày-tin")
    ax.set_xlabel(CHART_LABELS["dow"])
    ax.set_title(f"{label}\nLượt xem & liên hệ theo thứ trong tuần", fontweight="bold")
    ax.text(0.01, -0.18, chart_subtitle(n), transform=ax.transAxes, fontsize=9, color="#444")
    ax.legend()
    show_fig(out, show=show)


def plot_dow_contact_rate(
    dow: pd.DataFrame, label: str, color: str, out: Path | None, show: bool
) -> None:
    dow = sort_dow_df(dow.copy())
    dow["dow_label"] = dow_label_series(dow["dow"])
    n = int(dow["listing_days"].sum())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(dow["dow_label"], dow["contact_rate_pct"], marker="o", color=color, linewidth=2)
    ax.set_ylabel(CHART_LABELS["contact_rate_pct"])
    ax.set_xlabel(CHART_LABELS["dow"])
    ax.set_title(f"{label}\nNgày nào trong tuần có tỷ lệ liên hệ cao nhất?", fontweight="bold")
    ax.text(0.01, -0.2, chart_subtitle(n), transform=ax.transAxes, fontsize=9, color="#444")
    show_fig(out, show=show)


def plot_month_contact_rate(
    month: pd.DataFrame, label: str, color: str, out: Path | None, show: bool
) -> None:
    month = month.copy()
    month["month_str"] = month["month"].astype(str).str[:7]
    n = int(month["listing_days"].sum())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(month["month_str"], month["contact_rate_pct"], color=color)
    ax.set_ylabel(CHART_LABELS["contact_rate_pct"])
    ax.set_xlabel(CHART_LABELS["month"])
    ax.set_title(f"{label}\nTỷ lệ liên hệ theo tháng", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.text(0.01, -0.22, chart_subtitle(n), transform=ax.transAxes, fontsize=9, color="#444")
    show_fig(out, show=show)


def plot_decay_listing_age(
    decay: pd.DataFrame, label: str, color: str, out: Path | None, show: bool
) -> None:
    n = int(decay["listing_days"].sum()) if "listing_days" in decay.columns else len(decay)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ln1 = ax.plot(
        decay["listing_age_days"],
        decay["avg_views_24h"],
        color="#2171b5",
        marker="o",
        ms=4,
        label=CHART_LABELS["views_24h"],
    )
    ax.set_ylabel(CHART_LABELS["views_24h"], color="#2171b5")
    ax.set_xlabel(CHART_LABELS["listing_age_days"])
    ax2 = ax.twinx()
    ln2 = ax2.plot(
        decay["listing_age_days"],
        decay["avg_contacts_24h"],
        color="#cb181d",
        marker="s",
        ms=4,
        label=CHART_LABELS["contacts_24h"],
    )
    ax2.set_ylabel(CHART_LABELS["contacts_24h"], color="#cb181d")
    ax.set_title(
        f"{label}\nĐộ “sống” của tin theo tuổi (0–60 ngày)",
        fontweight="bold",
    )
    ax.text(0.01, -0.2, chart_subtitle(n), transform=ax.transAxes, fontsize=9, color="#444")
    ax.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], loc="upper right", fontsize=8)
    show_fig(out, show=show)


def plot_heatmap_age_x_dow(
    age_dow: pd.DataFrame, label: str, out: Path | None, show: bool
) -> None:
    if age_dow.empty:
        return
    pivot = age_dow.pivot(index="age_bucket", columns="dow", values="contact_rate_pct")
    pivot = pivot.reindex(columns=[c for c in DOW_ORDER if c in pivot.columns])
    pivot.columns = [DOW_LABELS[int(c)] for c in pivot.columns]
    order = ["0-2d", "3-7d", "8-14d", "15-30d", "31-60d", "60+d"]
    pivot = pivot.reindex([b for b in order if b in pivot.index])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": CHART_LABELS["heatmap_contact_rate"]},
    )
    ax.set_xlabel(CHART_LABELS["dow"])
    ax.set_ylabel("Nhóm tuổi tin")
    ax.set_title(
        f"{label}\nTuổi tin × thứ trong tuần → tỷ lệ liên hệ (%)",
        fontweight="bold",
    )
    fig.text(0.5, -0.02, f"Ô có n < {MIN_CELL_N:,} listing-days: đọc thận trọng", ha="center", fontsize=8)
    show_fig(out, show=show)


def plot_heatmap_hour_x_dow(
    hour_dow: pd.DataFrame, label: str, out: Path | None, show: bool
) -> None:
    if hour_dow.empty:
        return
    n_ev = int(hour_dow["explicit_contacts"].sum())
    if "pageviews" in hour_dow.columns:
        n_ev += int(hour_dow["pageviews"].sum())
    heat = hour_dow.pivot(index="hod", columns="dow", values="explicit_contacts").fillna(0)
    heat = heat.reindex(columns=[c for c in DOW_ORDER if c in heat.columns])
    heat.columns = [DOW_LABELS[int(c)] for c in heat.columns]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(
        heat,
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": CHART_LABELS["heatmap_explicit"]},
    )
    ax.set_xlabel(CHART_LABELS["dow"])
    ax.set_ylabel(CHART_LABELS["hour"])
    ax.set_title(
        f"{label}\nKhung giờ × thứ — liên hệ thật (events, user đăng nhập)",
        fontweight="bold",
    )
    ax.text(0.01, -0.12, chart_subtitle(n_events=n_ev), transform=ax.transAxes, fontsize=9, color="#444")
    show_fig(out, show=show)


def plot_hour_line(hour_line: pd.DataFrame, label: str, color: str, out: Path | None, show: bool) -> None:
    if hour_line.empty:
        return
    n_ev = int(hour_line["explicit_contacts"].sum() + hour_line["pageviews"].sum())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        hour_line["hod"],
        hour_line["pageviews"],
        label=CHART_LABELS["pageviews"],
        color="#9ecae1",
        linewidth=2,
    )
    ax.plot(
        hour_line["hod"],
        hour_line["explicit_contacts"],
        label=CHART_LABELS["explicit_contacts"],
        color=color,
        linewidth=2,
    )
    ax.set_xlabel(CHART_LABELS["hour"])
    ax.set_ylabel("Số sự kiện (đếm thô)")
    ax.set_title(f"{label}\nPhân bố hoạt động theo giờ trong ngày", fontweight="bold")
    ax.text(0.01, -0.18, chart_subtitle(n_events=n_ev), transform=ax.transAxes, fontsize=9, color="#444")
    ax.legend()
    show_fig(out, show=show)


def plot_adtype_dow_facet(
    ad_dow: pd.DataFrame, label: str, color: str, out: Path | None, show: bool
) -> None:
    if ad_dow.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, ad in zip(axes, ("sell", "let")):
        sub = sort_dow_df(ad_dow[ad_dow["ad_type"] == ad])
        if len(sub):
            sub = sub.copy()
            sub["dow_label"] = dow_label_series(sub["dow"])
            ax.bar(sub["dow_label"], sub["contact_rate_pct"], color=color, alpha=0.85)
        ax.set_title(AD_TYPE_LABELS.get(ad, ad))
        ax.set_xlabel(CHART_LABELS["dow"])
    axes[0].set_ylabel(CHART_LABELS["contact_rate_pct"])
    fig.suptitle(f"{label}\nTỷ lệ liên hệ theo thứ — so sánh Bán vs Cho thuê", fontweight="bold", y=1.02)
    show_fig(out, show=show)


def plot_adtype_decay_facet(
    ad_decay: pd.DataFrame, label: str, color: str, out: Path | None, show: bool
) -> None:
    if ad_decay.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, ad in zip(axes, ("sell", "let")):
        sub = ad_decay[ad_decay["ad_type"] == ad]
        ax.plot(
            sub["listing_age_days"],
            sub["avg_contacts_24h"],
            color=color,
            marker="o",
            ms=3,
        )
        ax.set_title(AD_TYPE_LABELS.get(ad, ad))
        ax.set_xlabel(CHART_LABELS["listing_age_days"])
    axes[0].set_ylabel(CHART_LABELS["contacts_24h"])
    fig.suptitle(f"{label}\nLiên hệ theo tuổi tin — Bán vs Cho thuê", fontweight="bold", y=1.02)
    show_fig(out, show=show)


def plot_interactions_dow(
    inter_dow: pd.DataFrame, label: str, color: str, out: Path | None, show: bool
) -> None:
    if inter_dow.empty:
        return
    inter_dow = sort_dow_df(inter_dow.copy())
    inter_dow["dow_label"] = dow_label_series(inter_dow["dow"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(inter_dow["dow_label"], inter_dow["leads"], color=color, alpha=0.75)
    ax.set_ylabel(CHART_LABELS["leads"])
    ax.set_xlabel(CHART_LABELS["dow"])
    ax.set_title(
        f"{label}\nLead (lộ SĐT) theo thứ — user đăng nhập, fact_post_contact_interactions",
        fontweight="bold",
    )
    show_fig(out, show=show)


def fetch_executive_frames(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    register_snap_view(con)
    ranking = con.execute(
        """
        SELECT category, COUNT(*)::BIGINT AS listing_days,
               COUNT(DISTINCT item_id)::BIGINT AS distinct_listings,
               ROUND(AVG(views_24h), 3) AS avg_views_24h,
               ROUND(AVG(contacts_24h), 3) AS avg_contacts_24h,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct,
               ROUND(100.0 * SUM(CASE WHEN contacts_24h > 0 THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0), 2) AS pct_days_with_contact,
               ROUND(MEDIAN(listing_age_days), 1) AS median_listing_age_days
        FROM snap_enriched GROUP BY 1 ORDER BY contact_rate_pct DESC
        """
    ).df()
    ranking["category_label"] = ranking["category"].map(CAT_META)

    cat_dow = con.execute(
        """
        SELECT category, dow, COUNT(*)::BIGINT AS listing_days,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct
        FROM snap_enriched GROUP BY 1, 2
        """
    ).df()

    decay_all = con.execute(
        """
        SELECT category, CAST(listing_age_days AS INT) AS listing_age_days,
               ROUND(AVG(contacts_24h), 3) AS avg_contacts_24h
        FROM snap_enriched
        WHERE listing_age_days BETWEEN 0 AND 60
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()

    return {"ranking": ranking, "cat_dow": cat_dow, "decay_all": decay_all}


def plot_executive_ranking_bars(ranking: pd.DataFrame, out: Path | None, show: bool) -> None:
    r = ranking.sort_values("contact_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    y = r["category_label"].astype(str)
    ax.barh(y, r["contact_rate_pct"], color=[PALETTE[int(c)] for c in r["category"]])
    ax.set_xlabel(CHART_LABELS["contact_rate_pct"])
    ax.set_title("So sánh 5 category — Tỷ lệ liên hệ tổng thể", fontweight="bold")
    ax.text(0.01, -0.12, chart_subtitle(int(r["listing_days"].sum())), transform=ax.transAxes, fontsize=9)
    show_fig(out, show=show)


def plot_executive_decay_multiples(decay_all: pd.DataFrame, out: Path | None, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cat in CATEGORIES:
        sub = decay_all[decay_all["category"] == cat]
        ax.plot(
            sub["listing_age_days"],
            sub["avg_contacts_24h"],
            label=CAT_META[cat],
            color=PALETTE[cat],
            marker="o",
            ms=3,
        )
    ax.set_xlabel(CHART_LABELS["listing_age_days"])
    ax.set_ylabel(CHART_LABELS["contacts_24h"])
    ax.set_title("So sánh 5 category — Liên hệ TB theo tuổi tin", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    show_fig(out, show=show)


def plot_executive_heatmap_dow(cat_dow: pd.DataFrame, out: Path | None, show: bool) -> None:
    pivot = cat_dow.pivot(index="category", columns="dow", values="contact_rate_pct")
    pivot = pivot.reindex(columns=[c for c in DOW_ORDER if c in pivot.columns])
    pivot.index = [CAT_META[int(c)] for c in pivot.index]
    pivot.columns = [DOW_LABELS[int(c)] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": CHART_LABELS["heatmap_contact_rate"]},
    )
    ax.set_xlabel(CHART_LABELS["dow"])
    ax.set_ylabel("Category")
    ax.set_title("So sánh 5 category × thứ trong tuần", fontweight="bold")
    show_fig(out, show=show)


def plot_executive_summary(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path | None = None,
    show: bool = False,
) -> pd.DataFrame:
    frames = fetch_executive_frames(con)
    ranking = frames["ranking"]
    prefix = out_dir or CROSS_DIR
    if out_dir:
        prefix.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(prefix / "executive_ranking.csv", index=False)
    plot_executive_ranking_bars(
        ranking,
        prefix / "fig_executive_contact_rate.png" if out_dir else CROSS_DIR / "fig_executive_contact_rate.png",
        show,
    )
    plot_executive_decay_multiples(
        frames["decay_all"],
        prefix / "fig_executive_decay.png" if out_dir else CROSS_DIR / "fig_executive_decay.png",
        show,
    )
    plot_executive_heatmap_dow(
        frames["cat_dow"],
        prefix / "fig_executive_dow_heatmap.png" if out_dir else CROSS_DIR / "fig_executive_dow_heatmap.png",
        show,
    )
    return ranking


def fetch_category_frames(con: duckdb.DuckDBPyConnection, cat: int) -> dict[str, Any]:
    register_snap_view(con)
    w = f"category = {cat}"
    trend = con.execute(
        f"""
        SELECT date, COUNT(*)::BIGINT AS listing_days,
               ROUND(AVG(views_24h), 3) AS avg_views_24h,
               ROUND(AVG(contacts_24h), 3) AS avg_contacts_24h,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct
        FROM snap_enriched WHERE {w} GROUP BY 1 ORDER BY 1
        """
    ).df()
    dow = con.execute(
        f"""
        SELECT dow, COUNT(*)::BIGINT AS listing_days,
               ROUND(AVG(views_24h), 3) AS avg_views_24h,
               ROUND(AVG(contacts_24h), 3) AS avg_contacts_24h,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct
        FROM snap_enriched WHERE {w} GROUP BY 1 ORDER BY 1
        """
    ).df()
    month = con.execute(
        f"""
        SELECT month, COUNT(*)::BIGINT AS listing_days,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct,
               ROUND(AVG(views_24h), 3) AS avg_views_24h,
               ROUND(AVG(contacts_24h), 3) AS avg_contacts_24h
        FROM snap_enriched WHERE {w} GROUP BY 1 ORDER BY 1
        """
    ).df()
    decay = con.execute(
        f"""
        SELECT CAST(listing_age_days AS INT) AS listing_age_days,
               COUNT(*)::BIGINT AS listing_days,
               ROUND(AVG(views_24h), 3) AS avg_views_24h,
               ROUND(AVG(contacts_24h), 3) AS avg_contacts_24h,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct
        FROM snap_enriched WHERE {w} AND listing_age_days BETWEEN 0 AND 60
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    age_dow = con.execute(
        f"""
        SELECT age_bucket, dow, COUNT(*)::BIGINT AS listing_days,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct
        FROM snap_enriched WHERE {w} AND age_bucket != 'unk'
        GROUP BY 1, 2
        """
    ).df()
    age_bucket = con.execute(
        f"""
        SELECT age_bucket, COUNT(*)::BIGINT AS listing_days,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct
        FROM snap_enriched WHERE {w} AND age_bucket != 'unk'
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    ad_dow = con.execute(
        f"""
        SELECT ad_type, dow,
               ROUND(100.0 * SUM(contacts_24h) / NULLIF(SUM(views_24h), 0), 4) AS contact_rate_pct
        FROM snap_enriched WHERE {w} GROUP BY 1, 2
        """
    ).df()
    ad_decay = con.execute(
        f"""
        SELECT ad_type, CAST(listing_age_days AS INT) AS listing_age_days,
               ROUND(AVG(contacts_24h), 3) AS avg_contacts_24h
        FROM snap_enriched WHERE {w} AND listing_age_days BETWEEN 0 AND 60
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()
    qa_row = con.execute(
        f"""
        SELECT ROUND(100.0 * AVG(flag_contact_gt_view), 4) AS pct_contacts_gt_views,
               ROUND(100.0 * AVG(CASE WHEN views_24h = 0 OR views_24h IS NULL THEN 1.0 ELSE 0.0 END), 4)
                   AS pct_views_zero
        FROM snap_enriched WHERE {w}
        """
    ).df()
    hour_dow, hour_line = load_events_agg(cat)
    inter_dow = pd.DataFrame()
    lp = login_cache_path(cat)
    if lp.exists():
        con.execute(f"CREATE OR REPLACE TEMP TABLE login_users AS SELECT user_id FROM read_parquet('{lp}')")
        inter_dow = con.execute(
            f"""
            SELECT EXTRACT('dow' FROM i.date)::INT AS dow,
                   SUM(i.lead_count)::BIGINT AS leads,
                   SUM(i.chat_message_count)::BIGINT AS chat_messages
            FROM read_parquet('{INTER_GLOB}') i
            INNER JOIN login_users u ON i.user_id = u.user_id
            WHERE i.category = {cat}
            GROUP BY 1 ORDER BY 1
            """
        ).df()

    return {
        "trend": trend,
        "dow": dow,
        "month": month,
        "decay": decay,
        "age_dow": age_dow,
        "age_bucket": age_bucket,
        "hour_dow": hour_dow,
        "hour_line": hour_line,
        "ad_dow": ad_dow,
        "ad_decay": ad_decay,
        "inter_dow": inter_dow,
        "qa_pct": {
            "pct_contacts_gt_views": float(qa_row.iloc[0]["pct_contacts_gt_views"] or 0),
            "pct_views_zero": float(qa_row.iloc[0]["pct_views_zero"] or 0),
        },
    }


def render_category_plots(
    cat: int,
    frames: dict[str, Any],
    out: Path | None = None,
    show: bool = False,
    save_csv: bool = True,
) -> tuple[pd.DataFrame, str]:
    label = CAT_META[cat]
    color = PALETTE[cat]
    if out is None:
        out = cat_out(cat)
    else:
        out.mkdir(parents=True, exist_ok=True)

    if save_csv:
        frames["trend"].to_csv(out / "01_daily_trend.csv", index=False)
        sort_dow_df(frames["dow"]).assign(
            dow_label=dow_label_series(frames["dow"]["dow"])
        ).to_csv(out / "02_dow_summary.csv", index=False)
        frames["month"].to_csv(out / "03_month_summary.csv", index=False)
        frames["decay"].to_csv(out / "04_decay_by_listing_age.csv", index=False)
        frames["age_dow"].to_csv(out / "05_age_bucket_x_dow.csv", index=False)
        frames["hour_dow"].to_csv(out / "06_hour_x_dow_events.csv", index=False)
        frames["hour_line"].to_csv(out / "07_hour_of_day_events.csv", index=False)
        frames["ad_dow"].to_csv(out / "08_adtype_x_dow.csv", index=False)
        frames["ad_decay"].to_csv(out / "08b_adtype_decay.csv", index=False)
        if len(frames["inter_dow"]):
            frames["inter_dow"].to_csv(out / "09_interactions_dow.csv", index=False)

    plot_daily_trend(frames["trend"], label, color, out / "fig_01_daily_trend.png", show)
    plot_dow_bars(frames["dow"], label, color, out / "fig_02_dow_bars.png", show)
    plot_dow_contact_rate(frames["dow"], label, color, out / "fig_02b_dow_contact_rate.png", show)
    plot_month_contact_rate(frames["month"], label, color, out / "fig_03_month_contact_rate.png", show)
    plot_decay_listing_age(frames["decay"], label, color, out / "fig_04_decay_listing_age.png", show)
    plot_heatmap_age_x_dow(frames["age_dow"], label, out / "fig_05_heatmap_age_x_dow.png", show)
    plot_heatmap_hour_x_dow(frames["hour_dow"], label, out / "fig_06_heatmap_hour_x_dow.png", show)
    plot_hour_line(frames["hour_line"], label, color, out / "fig_07_hour_line.png", show)
    plot_adtype_dow_facet(frames["ad_dow"], label, color, out / "fig_08_adtype_dow_facet.png", show)
    plot_adtype_decay_facet(frames["ad_decay"], label, color, out / "fig_08b_adtype_decay_facet.png", show)
    if len(frames["inter_dow"]):
        plot_interactions_dow(
            frames["inter_dow"], label, color, out / "fig_09_interactions_dow.png", show
        )

    dow_sc = sort_dow_df(frames["dow"].copy())
    dow_sc["dow_label"] = dow_label_series(dow_sc["dow"])
    scorecard = build_scorecard_rows(
        cat, dow_sc, frames["month"], frames["age_bucket"], frames["hour_line"], frames["qa_pct"]
    )
    scorecard.to_csv(out / "99_temporal_scorecard.csv", index=False)
    narrative = marketing_narrative(cat, scorecard)
    (out / "99_marketing_notes.md").write_text(narrative, encoding="utf-8")
    return scorecard, narrative


def plot_cross_heatmap_hour(cat_hour: pd.DataFrame, out: Path | None, show: bool) -> None:
    if cat_hour.empty:
        return
    cat_hour = cat_hour.copy()
    cat_hour["share_pct"] = 100.0 * cat_hour["explicit_contacts"] / cat_hour.groupby("category")[
        "explicit_contacts"
    ].transform("sum")
    heat = cat_hour.pivot(index="category", columns="hod", values="share_pct")
    heat.index = [CAT_META[int(c)] for c in heat.index]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(
        heat.astype(float),
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": CHART_LABELS["heatmap_hour_share"]},
    )
    ax.set_xlabel(CHART_LABELS["hour"])
    ax.set_ylabel("Category")
    ax.set_title(
        "Cross-category — Tỷ trọng liên hệ thật theo giờ (trong từng category)",
        fontweight="bold",
    )
    show_fig(out, show=show)


# ── Scorecard ─────────────────────────────────────────────────────────────────


def build_scorecard_rows(
    cat: int,
    dow_df: pd.DataFrame,
    month_df: pd.DataFrame,
    age_df: pd.DataFrame,
    hour_line: pd.DataFrame,
    qa_pct: dict,
) -> pd.DataFrame:
    rows = []
    label = CAT_META[cat]
    if len(dow_df):
        for _, r in dow_df.sort_values("contact_rate_pct", ascending=False).head(3).iterrows():
            rows.append(
                {
                    "metric": "top_dow_contact_rate",
                    "value": r["dow_label"],
                    "detail": f"rate={r['contact_rate_pct']:.3f}% n={int(r['listing_days'])}",
                    "category": cat,
                    "category_label": label,
                }
            )
    if len(month_df):
        hi = month_df.loc[month_df["contact_rate_pct"].idxmax()]
        lo = month_df.loc[month_df["contact_rate_pct"].idxmin()]
        rows.append(
            {
                "metric": "peak_month",
                "value": str(hi["month"])[:7],
                "detail": f"rate={hi['contact_rate_pct']:.3f}%",
                "category": cat,
                "category_label": label,
            }
        )
        rows.append(
            {
                "metric": "low_month",
                "value": str(lo["month"])[:7],
                "detail": f"rate={lo['contact_rate_pct']:.3f}%",
                "category": cat,
                "category_label": label,
            }
        )
    if len(age_df):
        valid = age_df[age_df["listing_days"] >= MIN_CELL_N]
        if len(valid):
            best = valid.loc[valid["contact_rate_pct"].idxmax()]
            rows.append(
                {
                    "metric": "best_listing_age_bucket",
                    "value": best["age_bucket"],
                    "detail": f"rate={best['contact_rate_pct']:.3f}% n={int(best['listing_days'])}",
                    "category": cat,
                    "category_label": label,
                }
            )
    if len(hour_line):
        h = hour_line.copy()
        h["block"] = (h["hod"] // 3) * 3
        blocks = h.groupby("block", as_index=False)["explicit_contacts"].sum()
        blocks = blocks.sort_values("explicit_contacts", ascending=False)
        if len(blocks):
            b = int(blocks.iloc[0]["block"])
            rows.append(
                {
                    "metric": "peak_hour_block_3h",
                    "value": f"{b:02d}-{b + 3:02d}h",
                    "detail": f"n_events={int(blocks.iloc[0]['explicit_contacts'])}",
                    "category": cat,
                    "category_label": label,
                }
            )
    for k, v in qa_pct.items():
        rows.append(
            {
                "metric": k,
                "value": f"{v:.2f}%",
                "detail": "",
                "category": cat,
                "category_label": label,
            }
        )
    return pd.DataFrame(rows)


def marketing_narrative(cat: int, scorecard: pd.DataFrame) -> str:
    label = CAT_META[cat]
    lines = [f"### {label}\n"]
    top_dow = scorecard[scorecard["metric"] == "top_dow_contact_rate"]
    if len(top_dow):
        lines.append(
            f"- **Thứ có tỷ lệ liên hệ cao:** {', '.join(top_dow['value'].astype(str))}"
        )
    peak_m = scorecard[scorecard["metric"] == "peak_month"]
    if len(peak_m):
        lines.append(f"- **Tháng mạnh nhất:** {peak_m.iloc[0]['value']} ({peak_m.iloc[0]['detail']})")
    age = scorecard[scorecard["metric"] == "best_listing_age_bucket"]
    if len(age):
        lines.append(f"- **Tuổi tin tối ưu:** {age.iloc[0]['value']} ({age.iloc[0]['detail']})")
    hb = scorecard[scorecard["metric"] == "peak_hour_block_3h"]
    if len(hb):
        lines.append(f"- **Khung giờ liên hệ thật:** {hb.iloc[0]['value']} ({hb.iloc[0]['detail']})")
    return "\n".join(lines) + "\n"


# ── Pipeline steps ─────────────────────────────────────────────────────────────


def step_qa(con: duckdb.DuckDBPyConnection) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT COUNT(*)::BIGINT AS orphan_snapshot_rows
            FROM read_parquet('{SNAP_GLOB}') s
            LEFT JOIN read_parquet('{DIM_GLOB}') d
              ON CAST(s.item_id AS VARCHAR) = CAST(d.item_id AS VARCHAR)
            WHERE d.item_id IS NULL
        ) TO '{QA_DIR / "01_orphan_snapshot_rows.csv"}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT d.category, COUNT(*)::BIGINT AS listing_days,
                SUM(CASE WHEN s.contacts_24h > s.views_24h THEN 1 ELSE 0 END)::BIGINT AS n_contact_gt_view,
                ROUND(100.0 * SUM(CASE WHEN s.contacts_24h > s.views_24h THEN 1 ELSE 0 END)
                      / NULLIF(COUNT(*), 0), 4) AS pct_contact_gt_view,
                SUM(CASE WHEN s.views_24h = 0 OR s.views_24h IS NULL THEN 1 ELSE 0 END)::BIGINT AS n_views_zero,
                ROUND(100.0 * SUM(CASE WHEN s.views_24h = 0 OR s.views_24h IS NULL THEN 1 ELSE 0 END)
                      / NULLIF(COUNT(*), 0), 4) AS pct_views_zero
            FROM read_parquet('{SNAP_GLOB}') s
            INNER JOIN read_parquet('{DIM_GLOB}') d
              ON CAST(s.item_id AS VARCHAR) = CAST(d.item_id AS VARCHAR)
            WHERE d.category IN (1010, 1020, 1030, 1040, 1050)
            GROUP BY 1 ORDER BY 1
        ) TO '{QA_DIR / "02_contact_gt_views_by_category.csv"}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
            WITH snap AS (
                SELECT d.category, s.date, SUM(s.contacts_24h)::DOUBLE AS snap_contacts
                FROM read_parquet('{SNAP_GLOB}') s
                INNER JOIN read_parquet('{DIM_GLOB}') d
                  ON CAST(s.item_id AS VARCHAR) = CAST(d.item_id AS VARCHAR)
                WHERE d.category IN (1010, 1020, 1030, 1040, 1050)
                GROUP BY 1, 2
            ),
            ev AS (
                SELECT category, date, COUNT(*)::DOUBLE AS explicit_events
                FROM read_parquet('{EVENTS_GLOB}')
                WHERE category IN (1010, 1020, 1030, 1040, 1050)
                  AND {LOGIN_WHERE} AND event_type IN ({EXPLICIT_SQL})
                GROUP BY 1, 2
            )
            SELECT snap.category,
                ROUND(AVG(snap.snap_contacts), 2) AS avg_daily_snap_contacts,
                ROUND(AVG(ev.explicit_events), 2) AS avg_daily_explicit_events,
                ROUND(AVG(ev.explicit_events / NULLIF(snap.snap_contacts, 0)), 4) AS avg_ratio_events_to_snap
            FROM snap LEFT JOIN ev ON snap.category = ev.category AND snap.date = ev.date
            GROUP BY 1 ORDER BY 1
        ) TO '{QA_DIR / "03_snap_vs_events_sanity.csv"}' (HEADER, DELIMITER ',')
        """
    )
    mark_status("qa")


def step_cache_snap(con: duckdb.DuckDBPyConnection) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if SNAP_CACHE.exists():
        mark_status("cache_snap", "skipped_exists")
        return
    con.execute(
        f"""
        COPY (
            SELECT CAST(s.item_id AS VARCHAR) AS item_id, s.date, s.views_24h, s.contacts_24h,
                s.listing_age_days, d.category, d.ad_type,
                EXTRACT('dow' FROM s.date)::INT AS dow,
                DATE_TRUNC('month', s.date) AS month,
                {age_bucket_sql("s.listing_age_days")} AS age_bucket,
                CASE WHEN s.contacts_24h > s.views_24h THEN 1 ELSE 0 END AS flag_contact_gt_view,
                CASE WHEN s.views_24h > 0 THEN 100.0 * s.contacts_24h / s.views_24h ELSE NULL END
                    AS contact_rate_pct
            FROM read_parquet('{SNAP_GLOB}') s
            INNER JOIN read_parquet('{DIM_GLOB}') d
              ON CAST(s.item_id AS VARCHAR) = CAST(d.item_id AS VARCHAR)
            WHERE d.category IN (1010, 1020, 1030, 1040, 1050)
        ) TO '{SNAP_CACHE}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    mark_status("cache_snap")


def step_cache_events(con: duckdb.DuckDBPyConnection, cats: tuple[int, ...]) -> None:
    for cat in cats:
        out = events_cache_path(cat)
        if out.exists():
            continue
        con.execute(
            f"""
            COPY (
                SELECT {cat}::INT AS category,
                    EXTRACT('hour' FROM event_ts)::INT AS hod,
                    EXTRACT('dow' FROM event_ts)::INT AS dow,
                    SUM(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END)::BIGINT AS explicit_contacts,
                    SUM(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END)::BIGINT AS pageviews
                FROM read_parquet('{EVENTS_GLOB}')
                WHERE category = {cat} AND {LOGIN_WHERE}
                GROUP BY 2, 3
            ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    mark_status("cache_events")


def step_cache_login(con: duckdb.DuckDBPyConnection, cats: tuple[int, ...]) -> None:
    for cat in cats:
        out = login_cache_path(cat)
        if out.exists():
            continue
        con.execute(
            f"""
            COPY (
                SELECT DISTINCT user_id FROM read_parquet('{EVENTS_GLOB}')
                WHERE category = {cat} AND {LOGIN_WHERE}
            ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    mark_status("cache_login")


def load_events_agg(cat: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = events_cache_path(cat)
    if not path.exists():
        raise FileNotFoundError(f"Chưa cache events cho {cat}. Chạy: --step cache_events")
    raw = pd.read_parquet(path)
    hour_line = raw.groupby("hod", as_index=False)[["explicit_contacts", "pageviews"]].sum()
    return raw, hour_line.sort_values("hod")


def step_category(
    con: duckdb.DuckDBPyConnection,
    cats: tuple[int, ...],
    show: bool = False,
) -> list[pd.DataFrame]:
    all_sc = []
    narratives = []
    for cat in cats:
        frames = fetch_category_frames(con, cat)
        sc, nar = render_category_plots(cat, frames, show=show)
        all_sc.append(sc)
        narratives.append(nar)
    mark_status("category")
    (OUT_ROOT / "99_all_marketing_notes.md").write_text("\n".join(narratives), encoding="utf-8")
    return all_sc


def step_cross(
    con: duckdb.DuckDBPyConnection,
    all_scorecards: list[pd.DataFrame],
    show: bool = False,
) -> None:
    CROSS_DIR.mkdir(parents=True, exist_ok=True)
    frames = fetch_executive_frames(con)
    ranking = frames["ranking"]
    ranking.to_csv(CROSS_DIR / "01_ranking_by_category.csv", index=False)
    frames["cat_dow"].to_csv(CROSS_DIR / "02_category_x_dow.csv", index=False)
    frames["decay_all"].to_csv(CROSS_DIR / "04_decay_all_categories.csv", index=False)

    plot_executive_ranking_bars(ranking, CROSS_DIR / "fig_02_executive_contact_rate.png", show)
    pivot = frames["cat_dow"].pivot(index="category", columns="dow", values="contact_rate_pct")
    pivot = pivot.reindex(columns=[c for c in DOW_ORDER if c in pivot.columns])
    pivot.index = [CAT_META[int(c)] for c in pivot.index]
    pivot.columns = [DOW_LABELS[int(c)] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": CHART_LABELS["heatmap_contact_rate"]},
    )
    ax.set_xlabel(CHART_LABELS["dow"])
    ax.set_ylabel("Category")
    ax.set_title("Cross — Tỷ lệ liên hệ theo category × thứ", fontweight="bold")
    show_fig(CROSS_DIR / "fig_02_heatmap_category_x_dow.png", show=show)

    parts = [str(events_cache_path(c)) for c in CATEGORIES if events_cache_path(c).exists()]
    if parts:
        paths_sql = ", ".join(repr(p) for p in parts)
        cat_hour = con.execute(
            f"""
            SELECT category, hod, SUM(explicit_contacts)::BIGINT AS explicit_contacts
            FROM read_parquet([{paths_sql}]) GROUP BY 1, 2
            """
        ).df()
        cat_hour.to_csv(CROSS_DIR / "03_category_x_hour.csv", index=False)
        plot_cross_heatmap_hour(cat_hour, CROSS_DIR / "fig_03_heatmap_category_x_hour.png", show)

    plot_executive_decay_multiples(
        frames["decay_all"], CROSS_DIR / "fig_04_decay_small_multiples.png", show
    )
    if all_scorecards:
        pd.concat(all_scorecards, ignore_index=True).to_csv(
            CROSS_DIR / "05_all_scorecards.csv", index=False
        )
    lines = ["# Đánh giá marketing — tổng hợp cross-category\n"]
    for cat in CATEGORIES:
        sub = ranking[ranking["category"] == cat]
        if len(sub):
            r = sub.iloc[0]
            lines.append(
                f"- **{CAT_META[cat]}:** {CHART_LABELS['contact_rate_pct']} = {r['contact_rate_pct']:.2f}%, "
                f"views TB = {r['avg_views_24h']:.2f}"
            )
    hi, lo = ranking.iloc[0], ranking.iloc[-1]
    lines.append(
        f"\n**Cao nhất:** {CAT_META[int(hi['category'])]} ({hi['contact_rate_pct']:.2f}%). "
        f"**Thấp nhất:** {CAT_META[int(lo['category'])]} ({lo['contact_rate_pct']:.2f}%)."
    )
    (CROSS_DIR / "00_cross_marketing_summary.md").write_text("\n".join(lines), encoding="utf-8")
    mark_status("cross")


def parse_categories(arg: str | None) -> tuple[int, ...]:
    if arg is None or arg.lower() == "all":
        return CATEGORIES
    return tuple(int(x.strip()) for x in arg.split(","))


def main() -> None:
    configure_matplotlib(notebook=False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["qa", "cache_snap", "cache_events", "cache_login", "category", "cross", "all"],
        default="all",
    )
    parser.add_argument("--category", default="all")
    args = parser.parse_args()
    cats = parse_categories(args.category)
    check_data_dirs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    con = connect_duckdb()
    run_cats = CATEGORIES if args.step == "all" else cats
    steps = (
        ["qa", "cache_snap", "cache_events", "cache_login", "category", "cross"]
        if args.step == "all"
        else [args.step]
    )
    all_sc: list[pd.DataFrame] = []
    for step in steps:
        if step == "qa":
            step_qa(con)
        elif step == "cache_snap":
            step_cache_snap(con)
        elif step == "cache_events":
            step_cache_events(con, run_cats)
        elif step == "cache_login":
            step_cache_login(con, run_cats)
        elif step == "category":
            all_sc = step_category(con, run_cats, show=False)
        elif step == "cross":
            if (CROSS_DIR / "05_all_scorecards.csv").exists():
                all_sc = [pd.read_csv(CROSS_DIR / "05_all_scorecards.csv")]
            else:
                all_sc = [
                    pd.read_csv(OUT_ROOT / str(c) / "99_temporal_scorecard.csv")
                    for c in CATEGORIES
                    if (OUT_ROOT / str(c) / "99_temporal_scorecard.csv").exists()
                ]
            step_cross(con, all_sc, show=False)


if __name__ == "__main__":
    main()
