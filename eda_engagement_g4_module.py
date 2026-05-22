"""Góc 4 — other_interaction micro-conversion EDA (used by eda_engagement_content.ipynb)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXPLICIT_TYPES = ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
EXPLICIT_SQL = ", ".join(repr(x) for x in EXPLICIT_TYPES)
LOGIN_WHERE = "is_login = 'login'"

CAT_LABELS = {
    1010: "1010 — Căn hộ / Chung cư",
    1020: "1020 — Nhà ở",
    1030: "1030 — Văn phòng / Mặt bằng",
    1040: "1040 — Đất",
    1050: "1050 — Phòng trọ",
}


def _sample_suffix(frac: float | None) -> str:
    if frac is None:
        return ""
    pct = max(1, min(100, int(round(float(frac) * 100))))
    return f" TABLESAMPLE {pct} PERCENT (SYSTEM)"


def cat_table_dir(out_csv: Path, cat: int) -> Path:
    p = out_csv / str(cat)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ab_session_explicit(con: duckdb.DuckDBPyConnection, cat: int, events_glob: str, inter_glob: str, sample_frac: float | None) -> pd.DataFrame:
    samp = _sample_suffix(sample_frac)
    return con.execute(
        f"""
        WITH ev AS (
            SELECT session_id, event_type, is_login,
                   CAST(user_id AS VARCHAR) AS user_id,
                   CAST(item_id AS VARCHAR) AS item_id,
                   CAST(date AS DATE) AS dt
            FROM read_parquet('{events_glob}'){samp}
            WHERE category = {cat} AND session_id IS NOT NULL
        ),
        chat_real_sess AS (
            SELECT DISTINCT e.session_id
            FROM ev e
            INNER JOIN read_parquet('{inter_glob}') i
                ON e.user_id = i.user_id AND e.item_id = i.item_id AND e.dt = i.date
            WHERE e.event_type = 'contact_chat' AND e.is_login = 'login'
              AND COALESCE(i.chat_message_count, 0) > 0
        ),
        sess AS (
            SELECT e.session_id,
                MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS has_pv,
                MAX(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END) AS has_other,
                MAX(CASE WHEN event_type = 'view_phone' THEN 1 ELSE 0 END) AS has_view_phone,
                MAX(CASE WHEN event_type = 'contact_zalo' THEN 1 ELSE 0 END) AS has_contact_zalo,
                MAX(CASE WHEN event_type = 'contact_sms' THEN 1 ELSE 0 END) AS has_contact_sms,
                MAX(CASE WHEN event_type = 'contact_chat' THEN 1 ELSE 0 END) AS has_contact_chat_raw,
                MAX(CASE WHEN c.session_id IS NOT NULL THEN 1 ELSE 0 END) AS has_chat_real
            FROM ev e
            LEFT JOIN chat_real_sess c ON e.session_id = c.session_id
            GROUP BY e.session_id
        ),
        pv_sess AS (
            SELECT *,
                GREATEST(has_view_phone, has_contact_zalo, has_contact_sms, has_chat_real) AS has_explicit_any
            FROM sess WHERE has_pv = 1
        )
        SELECT
            CASE WHEN has_other = 1 THEN 'co_other_interaction' ELSE 'khong_other' END AS grp,
            COUNT(*)::BIGINT AS sessions,
            ROUND(100.0 * SUM(has_view_phone) / COUNT(*), 3) AS view_phone_pct,
            ROUND(100.0 * SUM(has_contact_chat_raw) / COUNT(*), 3) AS contact_chat_raw_pct,
            ROUND(100.0 * SUM(has_chat_real) / COUNT(*), 3) AS contact_chat_verified_pct,
            ROUND(100.0 * SUM(has_contact_zalo) / COUNT(*), 3) AS contact_zalo_pct,
            ROUND(100.0 * SUM(has_contact_sms) / COUNT(*), 3) AS contact_sms_pct,
            ROUND(100.0 * SUM(has_explicit_any) / COUNT(*), 3) AS explicit_any_pct
        FROM pv_sess
        GROUP BY 1
        ORDER BY 1
        """
    ).df()


def dwell_pv_vs_oi(con: duckdb.DuckDBPyConnection, cat: int, events_glob: str, sample_frac: float | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samp = _sample_suffix(sample_frac)
    summary = con.execute(
        f"""
        SELECT event_type,
            ROUND(AVG(dwell_time_sec / 1000.0), 2) AS avg_dwell_sec,
            ROUND(quantile_cont(dwell_time_sec / 1000.0, 0.5), 2) AS p50_dwell_sec,
            ROUND(quantile_cont(dwell_time_sec / 1000.0, 0.9), 2) AS p90_dwell_sec,
            COUNT(*)::BIGINT AS n_with_dwell
        FROM read_parquet('{events_glob}'){samp}
        WHERE category = {cat}
          AND event_type IN ('pageview', 'other_interaction')
          AND dwell_time_sec IS NOT NULL AND dwell_time_sec > 0
          AND dwell_time_sec / 1000.0 <= 600
        GROUP BY 1
        """
    ).df()
    pv_proxy = con.execute(
        f"""
        WITH si AS (
            SELECT session_id, item_id,
                MAX(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END) AS has_oi,
                MAX(CASE WHEN event_type = 'pageview' THEN dwell_time_sec / 1000.0 END) AS pv_dwell_sec
            FROM read_parquet('{events_glob}'){samp}
            WHERE category = {cat} AND session_id IS NOT NULL AND item_id IS NOT NULL
            GROUP BY 1, 2
            HAVING MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) = 1
              AND MAX(CASE WHEN event_type = 'pageview' THEN dwell_time_sec END) IS NOT NULL
        )
        SELECT
            CASE WHEN has_oi = 1 THEN 'co_other_interaction' ELSE 'khong_other' END AS grp,
            ROUND(AVG(pv_dwell_sec), 2) AS avg_pv_dwell_sec,
            ROUND(quantile_cont(pv_dwell_sec, 0.5), 2) AS p50_pv_dwell_sec,
            COUNT(*)::BIGINT AS n_session_items
        FROM si
        WHERE pv_dwell_sec > 0 AND pv_dwell_sec <= 600
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    oi_dwell = con.execute(
        f"""
        WITH si AS (
            SELECT session_id, item_id,
                SUM(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END)::BIGINT AS n_oi,
                MAX(CASE WHEN event_type = 'other_interaction' THEN dwell_time_sec / 1000.0 END) AS max_oi_dwell_sec
            FROM read_parquet('{events_glob}'){samp}
            WHERE category = {cat} AND session_id IS NOT NULL AND item_id IS NOT NULL
            GROUP BY 1, 2
            HAVING SUM(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END) >= 2
        )
        SELECT
            CASE WHEN max_oi_dwell_sec < 5 THEN 'fast_swipe_lt5s'
                 WHEN max_oi_dwell_sec < 15 THEN 'short_5-15s'
                 ELSE 'long_15s+' END AS oi_dwell_band,
            COUNT(*)::BIGINT AS n_session_items,
            ROUND(AVG(n_oi), 2) AS avg_oi_count
        FROM si
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    return summary, oi_dwell, pv_proxy


def dwell_violin_sample(con: duckdb.DuckDBPyConnection, cat: int, events_glob: str, sample_frac: float | None, limit: int = 80000) -> pd.DataFrame:
    samp = _sample_suffix(sample_frac)
    return con.execute(
        f"""
        SELECT event_type, dwell_time_sec / 1000.0 AS dwell_sec
        FROM read_parquet('{events_glob}'){samp}
        WHERE category = {cat}
          AND event_type IN ('pageview', 'other_interaction')
          AND dwell_time_sec IS NOT NULL AND dwell_time_sec > 0
          AND dwell_time_sec / 1000.0 <= 600
        LIMIT {limit}
        """
    ).df()


def content_blackholes(
    con: duckdb.DuckDBPyConnection,
    cat: int,
    events_glob: str,
    dim_glob: str,
    snap_glob: str,
    sample_frac: float | None,
    top_n: int = 100,
) -> pd.DataFrame:
    samp = _sample_suffix(sample_frac)
    return con.execute(
        f"""
        WITH stats AS (
            SELECT item_id,
                SUM(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END)::BIGINT AS pageviews,
                SUM(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END)::BIGINT AS other_ix,
                SUM(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END)::BIGINT AS explicit_events
            FROM read_parquet('{events_glob}'){samp}
            WHERE category = {cat} AND item_id IS NOT NULL
            GROUP BY item_id
        ),
        holes AS (
            SELECT * FROM stats
            WHERE pageviews >= 20 AND other_ix >= 5 AND explicit_events <= 1
            ORDER BY pageviews DESC, other_ix DESC
            LIMIT {top_n}
        ),
        snap AS (
            SELECT s.item_id,
                MEDIAN(s.listing_age_days) AS median_listing_age_days,
                AVG(s.contacts_24h) AS avg_contacts_24h
            FROM read_parquet('{snap_glob}') s
            INNER JOIN holes h ON s.item_id = h.item_id
            GROUP BY s.item_id
        )
        SELECT h.item_id, h.pageviews, h.other_ix, h.explicit_events,
               d.title, d.ad_type, d.city_name,
               sn.median_listing_age_days, sn.avg_contacts_24h
        FROM holes h
        LEFT JOIN read_parquet('{dim_glob}') d ON h.item_id = d.item_id AND d.category = {cat}
        LEFT JOIN snap sn ON h.item_id = sn.item_id
        ORDER BY h.pageviews DESC, h.other_ix DESC
        """
    ).df()


def funnel_session(con: duckdb.DuckDBPyConnection, cat: int, events_glob: str, inter_glob: str, sample_frac: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    samp = _sample_suffix(sample_frac)
    steps = con.execute(
        f"""
        WITH ev AS (
            SELECT session_id, event_type, is_login,
                   CAST(user_id AS VARCHAR) AS user_id,
                   CAST(item_id AS VARCHAR) AS item_id,
                   CAST(date AS DATE) AS dt,
                   CASE WHEN query IS NOT NULL AND TRIM(CAST(query AS VARCHAR)) <> '' THEN 1 ELSE 0 END AS is_search
            FROM read_parquet('{events_glob}'){samp}
            WHERE category = {cat} AND session_id IS NOT NULL
        ),
        chat_real AS (
            SELECT DISTINCT session_id
            FROM ev e
            INNER JOIN read_parquet('{inter_glob}') i
                ON e.user_id = i.user_id AND e.item_id = i.item_id AND e.dt = i.date
            WHERE e.event_type = 'contact_chat' AND e.is_login = 'login'
              AND COALESCE(i.chat_message_count, 0) > 0
        ),
        sess AS (
            SELECT e.session_id,
                MAX(is_search) AS s_search,
                MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS s_pv,
                MAX(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END) AS s_other,
                MAX(CASE WHEN event_type = 'view_phone' THEN 1 ELSE 0 END) AS s_view_phone,
                MAX(CASE WHEN event_type = 'contact_zalo' THEN 1 ELSE 0 END) AS s_zalo,
                MAX(CASE WHEN event_type = 'contact_sms' THEN 1 ELSE 0 END) AS s_sms,
                MAX(CASE WHEN c.session_id IS NOT NULL THEN 1 ELSE 0 END) AS s_chat_real,
                MAX(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END) AS s_explicit_raw
            FROM ev e
            LEFT JOIN chat_real c ON e.session_id = c.session_id
            GROUP BY e.session_id
        ),
        n AS (SELECT COUNT(*)::DOUBLE AS total FROM sess),
        enriched AS (
            SELECT *,
                GREATEST(s_view_phone, s_zalo, s_sms, s_chat_real) AS s_explicit_any
            FROM sess
        )
        SELECT step, sessions, pct FROM (
            SELECT '1_search_or_feed' AS step, SUM(s_search)::BIGINT AS sessions,
                   ROUND(SUM(s_search) * 100.0 / (SELECT total FROM n), 2) AS pct FROM enriched
            UNION ALL SELECT '2_pageview', SUM(s_pv), ROUND(SUM(s_pv) * 100.0 / (SELECT total FROM n), 2) FROM enriched
            UNION ALL SELECT '3_other_interaction', SUM(s_other), ROUND(SUM(s_other) * 100.0 / (SELECT total FROM n), 2) FROM enriched
            UNION ALL SELECT '4_view_phone', SUM(s_view_phone), ROUND(SUM(s_view_phone) * 100.0 / (SELECT total FROM n), 2) FROM enriched
            UNION ALL SELECT '4_contact_zalo', SUM(s_zalo), ROUND(SUM(s_zalo) * 100.0 / (SELECT total FROM n), 2) FROM enriched
            UNION ALL SELECT '4_contact_sms', SUM(s_sms), ROUND(SUM(s_sms) * 100.0 / (SELECT total FROM n), 2) FROM enriched
            UNION ALL SELECT '4_contact_chat_verified', SUM(s_chat_real), ROUND(SUM(s_chat_real) * 100.0 / (SELECT total FROM n), 2) FROM enriched
            UNION ALL SELECT '4_explicit_any', SUM(s_explicit_any), ROUND(SUM(s_explicit_any) * 100.0 / (SELECT total FROM n), 2) FROM enriched
        ) ORDER BY step
        """
    ).df()

    transitions = con.execute(
        f"""
        WITH ev AS (
            SELECT session_id, event_type, is_login,
                   CAST(user_id AS VARCHAR) AS user_id,
                   CAST(item_id AS VARCHAR) AS item_id,
                   CAST(date AS DATE) AS dt
            FROM read_parquet('{events_glob}'){samp}
            WHERE category = {cat} AND session_id IS NOT NULL
        ),
        chat_real AS (
            SELECT DISTINCT session_id FROM ev e
            INNER JOIN read_parquet('{inter_glob}') i
                ON e.user_id = i.user_id AND e.item_id = i.item_id AND e.dt = i.date
            WHERE e.event_type = 'contact_chat' AND e.is_login = 'login'
              AND COALESCE(i.chat_message_count, 0) > 0
        ),
        sess AS (
            SELECT e.session_id,
                MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS s_pv,
                MAX(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END) AS s_other,
                GREATEST(
                    MAX(CASE WHEN event_type = 'view_phone' THEN 1 ELSE 0 END),
                    MAX(CASE WHEN event_type = 'contact_zalo' THEN 1 ELSE 0 END),
                    MAX(CASE WHEN event_type = 'contact_sms' THEN 1 ELSE 0 END),
                    MAX(CASE WHEN c.session_id IS NOT NULL THEN 1 ELSE 0 END)
                ) AS s_explicit_any
            FROM ev e
            LEFT JOIN chat_real c ON e.session_id = c.session_id
            GROUP BY e.session_id
        ),
        n AS (SELECT COUNT(*)::DOUBLE AS total FROM sess WHERE s_pv = 1)
        SELECT path_label, n_sessions,
               ROUND(n_sessions * 100.0 / (SELECT total FROM n), 2) AS pct_of_pv_sessions
        FROM (
            SELECT 'pv_only_no_explicit' AS path_label,
                SUM(CASE WHEN s_pv = 1 AND s_other = 0 AND s_explicit_any = 0 THEN 1 ELSE 0 END)::BIGINT AS n_sessions
            FROM sess
            UNION ALL
            SELECT 'pv_oi_no_explicit', SUM(CASE WHEN s_pv = 1 AND s_other = 1 AND s_explicit_any = 0 THEN 1 ELSE 0 END) FROM sess
            UNION ALL
            SELECT 'pv_oi_explicit', SUM(CASE WHEN s_pv = 1 AND s_other = 1 AND s_explicit_any = 1 THEN 1 ELSE 0 END) FROM sess
            UNION ALL
            SELECT 'pv_explicit_skip_oi', SUM(CASE WHEN s_pv = 1 AND s_other = 0 AND s_explicit_any = 1 THEN 1 ELSE 0 END) FROM sess
        ) WHERE n_sessions > 0
        """
    ).df()
    return steps, transitions


def path_session_item(con: duckdb.DuckDBPyConnection, cat: int, events_glob: str, sample_frac: float | None) -> pd.DataFrame:
    samp = _sample_suffix(sample_frac)
    return con.execute(
        f"""
        WITH session_item AS (
            SELECT session_id, item_id,
                MAX(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS has_pageview,
                MAX(CASE WHEN event_type = 'other_interaction' AND lower(coalesce(surface,'')) = 'ad_view' THEN 1 ELSE 0 END) AS has_oi_ad_view,
                MAX(CASE WHEN event_type = 'other_interaction' AND lower(coalesce(surface,'')) = 'adview' THEN 1 ELSE 0 END) AS has_oi_adview,
                MAX(CASE WHEN event_type = 'other_interaction'
                    AND lower(coalesce(surface,'')) NOT IN ('ad_view', 'adview') THEN 1 ELSE 0 END) AS has_oi_other,
                MAX(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END) AS has_explicit
            FROM read_parquet('{events_glob}'){samp}
            WHERE category = {cat} AND {LOGIN_WHERE} AND item_id IS NOT NULL
            GROUP BY 1, 2
        ),
        base AS (SELECT * FROM session_item WHERE has_pageview = 1)
        SELECT path_label, n_items, n_explicit,
            ROUND(100.0 * n_explicit / NULLIF(n_items, 0), 3) AS explicit_rate_pct
        FROM (
            SELECT 'A_pageview_only_no_other_ix' AS path_label,
                SUM(CASE WHEN has_oi_ad_view = 0 AND has_oi_adview = 0 AND has_oi_other = 0 THEN 1 ELSE 0 END)::BIGINT AS n_items,
                SUM(CASE WHEN has_oi_ad_view = 0 AND has_oi_adview = 0 AND has_oi_other = 0 AND has_explicit = 1 THEN 1 ELSE 0 END)::BIGINT AS n_explicit
            FROM base
            UNION ALL
            SELECT 'B_pageview_plus_oi_ad_view',
                SUM(CASE WHEN has_oi_ad_view = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN has_oi_ad_view = 1 AND has_explicit = 1 THEN 1 ELSE 0 END) FROM base
            UNION ALL
            SELECT 'C_pageview_plus_oi_adview',
                SUM(CASE WHEN has_oi_adview = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN has_oi_adview = 1 AND has_explicit = 1 THEN 1 ELSE 0 END) FROM base
        ) ORDER BY path_label
        """
    ).df()


def plot_ab_session(ab: pd.DataFrame, cat: int, label: str, out_dir: Path, palette: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ab["grp"], ab["explicit_any_pct"], color=["#6baed6", "#2171b5"], edgecolor="black", linewidth=0.3)
    ax.set_ylabel("% session có explicit (có pageview)")
    ax.set_title(f"other_interaction → explicit — {label}")
    for i, v in enumerate(ab["explicit_any_pct"]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"g4_ab_other_contact_{cat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = ["view_phone_pct", "contact_chat_verified_pct", "contact_zalo_pct", "contact_sms_pct"]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, grp in enumerate(["khong_other", "co_other_interaction"]):
        row = ab.loc[ab["grp"] == grp]
        if row.empty:
            continue
        vals = [float(row[m].iloc[0]) for m in metrics]
        offset = -w / 2 if i == 0 else w / 2
        ax.bar(x + offset, vals, width=w, label=grp.replace("_", " "), edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["view_phone", "chat (verified)", "zalo", "sms"], rotation=15, ha="right")
    ax.set_ylabel("% sessions")
    ax.set_title(f"Explicit by channel × other_interaction — {label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"g4_ab_explicit_by_channel_{cat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dwell(pv_proxy: pd.DataFrame, dwell_summary: pd.DataFrame, cat: int, label: str, out_dir: Path) -> None:
    if pv_proxy.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(pv_proxy["grp"], pv_proxy["p50_pv_dwell_sec"], color=["#6baed6", "#2171b5"], edgecolor="black", linewidth=0.3)
    ax.set_ylabel("p50 pageview dwell (giây) trên session×item")
    note = ""
    if dwell_summary["event_type"].eq("other_interaction").any():
        oi_n = dwell_summary.loc[dwell_summary["event_type"] == "other_interaction", "n_with_dwell"]
        if len(oi_n) and int(oi_n.iloc[0]) == 0:
            note = " — OI không có dwell_time_sec; proxy từ PV cùng tin"
    ax.set_title(f"Dwell proxy khi có/không other_ix{note} — {label}")
    fig.tight_layout()
    fig.savefig(out_dir / f"g4_dwell_violin_{cat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_blackholes_scatter(bh: pd.DataFrame, cat: int, label: str, out_dir: Path, palette: str) -> None:
    if bh.empty:
        return
    top = bh.head(30)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(top["pageviews"], top["other_ix"], c=top["explicit_events"], cmap="YlOrRd", s=60, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="explicit_events")
    ax.set_xlabel("pageviews")
    ax.set_ylabel("other_interaction count")
    ax.set_title(f"Content blackholes (top 30) — {label}")
    fig.tight_layout()
    fig.savefig(out_dir / f"g4_blackholes_scatter_{cat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_funnel(steps: pd.DataFrame, cat: int, label: str, out_dir: Path, palette: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(steps["step"], steps["pct"], color=palette, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("% session (mẫu số = tất cả session trong sample)")
    ax.set_title(f"Funnel session — {label}")
    for y, v in enumerate(steps["pct"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"g4_funnel4_{cat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_path(path: pd.DataFrame, cat: int, label: str, out_dir: Path) -> None:
    plot = path[path["path_label"].str.startswith(("A_", "B_", "C_"))]
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot["path_label"], plot["explicit_rate_pct"], color="#2171b5", edgecolor="black", linewidth=0.3)
    ax.set_ylabel("% explicit (login, session×item có pageview)")
    ax.set_title(f"Path A/B/C — {label}")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / f"g4_path_session_item_{cat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_corner4_category(
    con: duckdb.DuckDBPyConnection,
    cat: int,
    events_glob: str,
    inter_glob: str,
    dim_glob: str,
    snap_glob: str,
    out_dir: Path,
    out_csv: Path,
    sample_frac: float | None,
    palette: str,
) -> pd.DataFrame:
    label = CAT_LABELS[cat]
    tdir = cat_table_dir(out_csv, cat)

    ab = ab_session_explicit(con, cat, events_glob, inter_glob, sample_frac)
    ab.to_csv(tdir / "01_ab_session_rates.csv", index=False)
    plot_ab_session(ab, cat, label, out_dir, palette)

    dwell_summary, oi_dwell, pv_proxy = dwell_pv_vs_oi(con, cat, events_glob, sample_frac)
    dwell_summary.to_csv(tdir / "02_dwell_summary_by_event.csv", index=False)
    pv_proxy.to_csv(tdir / "02b_pv_dwell_proxy_when_oi.csv", index=False)
    oi_dwell.to_csv(tdir / "03_oi_count_vs_dwell.csv", index=False)
    plot_dwell(pv_proxy, dwell_summary, cat, label, out_dir)

    bh = content_blackholes(con, cat, events_glob, dim_glob, snap_glob, sample_frac)
    bh.to_csv(tdir / "04_blackholes_top100.csv", index=False)
    plot_blackholes_scatter(bh, cat, label, out_dir, palette)

    steps, trans = funnel_session(con, cat, events_glob, inter_glob, sample_frac)
    steps.to_csv(tdir / "05_funnel_steps.csv", index=False)
    trans.to_csv(tdir / "06_funnel_transitions.csv", index=False)
    plot_funnel(steps, cat, label, out_dir, palette)

    path = path_session_item(con, cat, events_glob, sample_frac)
    path.to_csv(tdir / "07_path_session_item_login.csv", index=False)
    plot_path(path, cat, label, out_dir)

    ab = ab.copy()
    ab["category"] = cat
    ab["category_label"] = label
    return ab


def cross_lift_summary(ab_all: pd.DataFrame, out_csv: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for cat, g in ab_all.groupby("category"):
        no = g.loc[g["grp"] == "khong_other", "explicit_any_pct"]
        yes = g.loc[g["grp"] == "co_other_interaction", "explicit_any_pct"]
        if len(no) and len(yes):
            rows.append({
                "category": cat,
                "category_label": g["category_label"].iloc[0],
                "explicit_any_khong_other_pct": float(no.iloc[0]),
                "explicit_any_co_other_pct": float(yes.iloc[0]),
                "lift_pp": float(yes.iloc[0]) - float(no.iloc[0]),
            })
    lift = pd.DataFrame(rows)
    cross_dir = out_csv / "_cross"
    cross_dir.mkdir(parents=True, exist_ok=True)
    lift.to_csv(cross_dir / "ab_lift_summary.csv", index=False)

    if not lift.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        cats = lift["category"].astype(str)
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w / 2, lift["explicit_any_khong_other_pct"], width=w, label="không other_ix")
        ax.bar(x + w / 2, lift["explicit_any_co_other_pct"], width=w, label="có other_ix")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel("% session explicit_any")
        ax.set_title("Lift explicit_any: có vs không other_interaction")
        ax.legend()
        for i, lv in enumerate(lift["lift_pp"]):
            ax.text(i, max(lift["explicit_any_co_other_pct"].iloc[i], lift["explicit_any_khong_other_pct"].iloc[i]) + 0.3,
                    f"+{lv:.1f}pp", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "g4_cross_lift_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return lift
