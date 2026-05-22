"""
BCG 4 nhóm — playbook, Action Matrix (Expected Impact), borderline, survivorship QA.

Run after run_region_revenue_bcg.py:
  env/bin/python Thinh_Analyze/run_bcg_quadrant_playbook.py
  env/bin/python Thinh_Analyze/run_bcg_quadrant_playbook.py --init-db   # item-level survival + deep dive
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import textwrap
from pathlib import Path

import matplotlib

try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_rev_path = Path(__file__).resolve().parent / "run_region_revenue_bcg.py"
_spec = importlib.util.spec_from_file_location("run_region_revenue_bcg", _rev_path)
rev = importlib.util.module_from_spec(_spec)
sys.modules["run_region_revenue_bcg"] = rev
_spec.loader.exec_module(rev)

OUT_DIR = rev.OUT_DIR
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BCG4 = ("stars", "cash_cows", "question_marks", "dogs")
BCG_MEDIAN_BUFFER_PCT = 5.0
SURVIVAL_UNSTABLE_THRESHOLD = 0.30

STRATEGIC_MULTIPLIER = {
    "stars": 1.0,
    "cash_cows": 0.95,
    "question_marks": 0.85,
    "dogs": 0.50,
}

LEVER_BY_QUADRANT = {
    "stars": "scale_quality,peer_replicate,defend_growth",
    "cash_cows": "harvest_refresh,explicit_mix,anti_decline",
    "question_marks": "ab_test_cvr,intent_fit,agent_quality",
    "dogs": "cut_low_supply,category_shift,fix_funnel",
    "low_volume": "reach_threshold_n_e",
}


def _buffer_bounds(median: float, buffer_pct: float) -> tuple[float, float]:
    if pd.isna(median):
        return np.nan, np.nan
    b = buffer_pct / 100.0
    return median * (1.0 - b), median * (1.0 + b)


def _borderline_flags(
    growth: float,
    rel_cvr: float,
    med_g: float,
    med_s: float,
    buffer_pct: float,
) -> tuple[int, str]:
    if pd.isna(growth) or pd.isna(rel_cvr) or pd.isna(med_g) or pd.isna(med_s):
        return 0, ""
    low_g, high_g = _buffer_bounds(med_g, buffer_pct)
    low_s, high_s = _buffer_bounds(med_s, buffer_pct)
    g_bl = low_g < growth < high_g
    s_bl = low_s < rel_cvr < high_s
    if g_bl and s_bl:
        return 1, "both"
    if g_bl:
        return 1, "growth"
    if s_bl:
        return 1, "cvr"
    return 0, ""


def _cvr_target_from_pool(
    ex_hcmc: pd.DataFrame,
    scope: str,
    category: int,
    ad_type: str,
    min_n_listings: int,
    min_n_explicit: int,
) -> float:
    mask = rev._bcg_eligible_mask(ex_hcmc, min_n_listings, min_n_explicit)
    if scope == "national_cat_ad":
        pool = ex_hcmc[
            (ex_hcmc["category"] == category) & (ex_hcmc["ad_type"] == ad_type)
        ]
    elif scope == "national_cat":
        pool = ex_hcmc[ex_hcmc["category"] == category]
    elif scope == "national_ad":
        pool = ex_hcmc[ex_hcmc["ad_type"] == ad_type]
    elif scope == "national_all":
        pool = ex_hcmc
    else:
        pool = ex_hcmc[
            (ex_hcmc["category"] == category) & (ex_hcmc["ad_type"] == ad_type)
        ]
    el = pool[rev._bcg_eligible_mask(pool, min_n_listings, min_n_explicit)]
    if el.empty:
        return np.nan
    return float(el["cvr_per_listing"].median())


def _stability_factor(survival_rate: float) -> float:
    if pd.isna(survival_rate):
        return 1.0
    if survival_rate < SURVIVAL_UNSTABLE_THRESHOLD:
        return 0.7
    return 1.0


def _assign_quadrant_simple(
    growth_pct: float,
    rel_cvr: float,
    med_g: float,
    med_s: float,
) -> str:
    if pd.isna(med_g) or pd.isna(med_s) or pd.isna(rel_cvr):
        return "low_volume"
    if pd.isna(growth_pct):
        growth_pct = 0.0
    high_g = growth_pct >= med_g
    high_s = rel_cvr >= med_s
    if high_g and high_s:
        return "stars"
    if not high_g and high_s:
        return "cash_cows"
    if high_g and not high_s:
        return "question_marks"
    return "dogs"


def load_national_bcg() -> pd.DataFrame:
    path = OUT_DIR / "06_bcg_national_choropleth.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run run_region_revenue_bcg.py first — missing {path}")
    return pd.read_csv(path)


def _ensure_db() -> None:
    if rev.con is None:
        rev.init_db(None)


def build_listing_survival_metrics(init_db: bool) -> pd.DataFrame:
    """Item-level survival H1→H2 explicit per city×segment (optional DuckDB)."""
    if not init_db:
        return pd.DataFrame()

    _ensure_db()
    mid = rev.eda_mid
    sql = f"""
        WITH item_half AS (
            SELECT
                region,
                city_name,
                dim_category AS category,
                ad_type,
                item_id,
                SUM(CASE WHEN date < DATE '{mid}'
                          AND event_type IN ({rev.EXPLICIT_SQL}) THEN 1 ELSE 0 END)::BIGINT AS e1,
                SUM(CASE WHEN date >= DATE '{mid}'
                          AND event_type IN ({rev.EXPLICIT_SQL}) THEN 1 ELSE 0 END)::BIGINT AS e2
            FROM events_geo
            WHERE is_hcmc = 0
            GROUP BY 1, 2, 3, 4, 5
        )
        SELECT
            region,
            city_name,
            category,
            ad_type,
            COUNT(*) FILTER (WHERE e1 > 0)::BIGINT AS items_explicit_h1,
            COUNT(*) FILTER (WHERE e1 > 0 AND e2 > 0)::BIGINT AS items_explicit_both_halves,
            COUNT(*) FILTER (WHERE e1 > 0 AND e2 = 0)::BIGINT AS items_h1_only,
            COUNT(*) FILTER (WHERE e1 = 0 AND e2 > 0)::BIGINT AS items_h2_only
        FROM item_half
        GROUP BY 1, 2, 3, 4
    """
    df = rev.con.execute(sql).df()
    df["listing_survival_rate"] = np.where(
        df["items_explicit_h1"] > 0,
        df["items_explicit_both_halves"] / df["items_explicit_h1"],
        np.nan,
    )
    df["listing_churn_pct"] = np.where(
        df["items_explicit_h1"] > 0,
        100.0 * df["items_h1_only"] / df["items_explicit_h1"],
        np.nan,
    )
    df["growth_quality_flag"] = np.where(
        df["listing_survival_rate"] < SURVIVAL_UNSTABLE_THRESHOLD,
        "unstable_growth",
        "stable",
    )
    return df


def enrich_action_rows(
    nat: pd.DataFrame,
    survival: pd.DataFrame,
    buffer_pct: float,
    min_n_listings: int,
    min_n_explicit: int,
) -> pd.DataFrame:
    ex = nat[nat["is_hcmc"] == 0].copy()
    eligible_mask = rev._bcg_eligible_mask(ex, min_n_listings, min_n_explicit)
    rows = ex[eligible_mask & ex["bcg_quadrant"].isin(BCG4)].copy()

    if not survival.empty:
        rows = rows.merge(
            survival,
            on=["region", "city_name", "category", "ad_type"],
            how="left",
        )
    else:
        h1 = rows["explicit_h1"].fillna(0).astype(float)
        h2 = rows["explicit_h2"].fillna(0).astype(float)
        rows["items_explicit_h1"] = np.nan
        rows["listing_survival_rate"] = np.where(
            h1 > 0, np.minimum(h1, h2) / h1, np.nan
        )
        rows["listing_churn_pct"] = np.where(
            h1 > 0, 100.0 * (h1 - np.minimum(h1, h2)) / h1, np.nan
        )
        rows["growth_quality_flag"] = np.where(
            rows["listing_survival_rate"] < SURVIVAL_UNSTABLE_THRESHOLD,
            "unstable_growth",
            "stable",
        )

    med_g_col = rows["median_growth_cvr"]
    med_s_col = rows["median_relative_cvr"]

    bl_flags, bl_axes, growth_gaps, cvr_gaps = [], [], [], []
    cvr_targets, uplifts, impacts, defends = [], [], [], []
    primary_weak, scenarios = [], []

    for _, r in rows.iterrows():
        mg = r["median_growth_cvr"]
        ms = r["median_relative_cvr"]
        gh = r["growth_cvr_pct"]
        sh = r["relative_cvr"]
        cvr = r["cvr_per_listing"]
        n = r["n_listings"]
        quad = r["bcg_quadrant"]

        bf, ba = _borderline_flags(gh, sh, mg, ms, buffer_pct)
        bl_flags.append(bf)
        bl_axes.append(ba)

        growth_gaps.append(gh - mg if pd.notna(gh) and pd.notna(mg) else np.nan)
        cvr_gaps.append(sh - ms if pd.notna(sh) and pd.notna(ms) else np.nan)

        ct = _cvr_target_from_pool(
            ex,
            str(r.get("bcg_median_scope", "national_cat_ad")),
            int(r["category"]),
            str(r["ad_type"]),
            min_n_listings,
            min_n_explicit,
        )
        cvr_targets.append(ct)
        uplift = max(0.0, (ct - cvr) * n) if pd.notna(ct) and pd.notna(cvr) else 0.0
        uplifts.append(uplift)

        stab = _stability_factor(r.get("listing_survival_rate", np.nan))
        mult = STRATEGIC_MULTIPLIER.get(quad, 0.5)
        impacts.append(uplift * mult * stab)

        e = r["n_explicit_events"]
        defend = 0.0
        if quad in ("stars", "cash_cows") and pd.notna(gh) and pd.notna(mg) and gh < mg:
            defend = float(e) * (mg - gh) / 100.0
        defends.append(defend)

        if pd.notna(gh) and pd.notna(mg) and pd.notna(sh) and pd.notna(ms):
            if gh < mg and sh < ms:
                pw = "both"
            elif gh < mg:
                pw = "growth"
            elif sh < ms:
                pw = "cvr"
            else:
                pw = "none"
        else:
            pw = "unknown"
        primary_weak.append(pw)

        need_g = max(0.0, mg - gh) if pd.notna(mg) and pd.notna(gh) else np.nan
        need_cvr = max(0.0, (ms - sh) * ct) if pd.notna(ms) and pd.notna(sh) and pd.notna(ct) else np.nan
        parts = []
        if need_g > 0:
            parts.append(f"growth+{need_g:.1f}pp")
        if need_cvr > 0:
            parts.append(f"cvr+{need_cvr:.2f}/tin")
        scenarios.append(";".join(parts))

    rows["borderline_flag"] = bl_flags
    rows["borderline_axis"] = bl_axes
    rows["growth_gap_pp"] = growth_gaps
    rows["cvr_gap_vs_median"] = cvr_gaps
    rows["cvr_target"] = cvr_targets
    rows["uplift_explicit"] = uplifts
    rows["priority_impact"] = impacts
    rows["priority_defend"] = defends
    rows["primary_weakness"] = primary_weak
    rows["scenario_to_median"] = scenarios
    rows["stability_factor"] = [
        _stability_factor(r.get("listing_survival_rate", np.nan)) for _, r in rows.iterrows()
    ]
    rows["strategic_multiplier"] = rows["bcg_quadrant"].map(STRATEGIC_MULTIPLIER)
    rows["suggested_levers"] = rows["bcg_quadrant"].map(LEVER_BY_QUADRANT)

    rows["explicit_pct_rank"] = (
        rows["n_explicit_events"].rank(pct=True, method="average") * 100.0
    )

    return rows


def add_peer_columns(action: pd.DataFrame) -> pd.DataFrame:
    peers = []
    for _, r in action.iterrows():
        sub = action[
            (action["category"] == r["category"])
            & (action["ad_type"] == r["ad_type"])
            & (action["bcg_quadrant"] == r["bcg_quadrant"])
            & (action["city_name"] != r["city_name"])
        ].nlargest(3, "n_explicit_events")
        peers.append(
            "; ".join(
                f"{x['city_name']}({x['cvr_per_listing']:.1f}/tin)"
                for _, x in sub.iterrows()
            )
        )
    action = action.copy()
    action["peer_top3"] = peers
    return action


def mark_deep_dive_eligible(action: pd.DataFrame) -> pd.DataFrame:
    action = action.copy()
    p75_e = action["n_explicit_events"].quantile(0.75)
    top20_cut = action["priority_impact"].quantile(0.80)
    defend_top = action["priority_defend"].nlargest(10).index

    action["deep_dive_eligible"] = (
        (action["priority_impact"] >= top20_cut)
        | (
            action["bcg_quadrant"].isin(["question_marks", "dogs"])
            & (action["n_explicit_events"] >= p75_e)
        )
        | action.index.isin(defend_top)
    ).astype(int)
    return action


def build_quadrant_profiles(action: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    profiles = []
    for q in list(BCG4) + ["low_volume", "hcmc_excluded"]:
        sub_a = action[action["bcg_quadrant"] == q] if q in BCG4 else pd.DataFrame()
        sub_n = nat[nat["bcg_quadrant"] == q]
        profiles.append(
            {
                "bcg_quadrant": q,
                "n_cells_colored": len(sub_a),
                "n_rows_all": len(sub_n),
                "total_explicit": int(sub_a["n_explicit_events"].sum()) if len(sub_a) else 0,
                "median_cvr_per_listing": sub_a["cvr_per_listing"].median() if len(sub_a) else np.nan,
                "median_growth_cvr_pct": sub_a["growth_cvr_pct"].median() if len(sub_a) else np.nan,
                "median_n_listings": sub_a["n_listings"].median() if len(sub_a) else np.nan,
                "pct_borderline": (
                    100.0 * sub_a["borderline_flag"].mean() if len(sub_a) else np.nan
                ),
                "pct_unstable_growth": (
                    100.0
                    * (sub_a["growth_quality_flag"] == "unstable_growth").mean()
                    if len(sub_a) and "growth_quality_flag" in sub_a.columns
                    else np.nan
                ),
                "median_priority_impact": sub_a["priority_impact"].median() if len(sub_a) else np.nan,
            }
        )
    return pd.DataFrame(profiles)


def build_migration_table(
    nat: pd.DataFrame, min_n_listings: int, min_n_explicit: int
) -> pd.DataFrame:
    ex = nat[nat["is_hcmc"] == 0]
    ex = ex[rev._bcg_eligible_mask(ex, min_n_listings, min_n_explicit)]
    rows = []
    for _, r in ex.iterrows():
        ref_pool, med_g, med_s, scope = rev._find_bcg_reference_pool(
            ex,
            region=None,
            category=int(r["category"]),
            ad_type=str(r["ad_type"]),
            min_n_listings=min_n_listings,
            min_n_explicit=min_n_explicit,
        )
        if ref_pool.empty:
            continue
        elig = ref_pool[rev._bcg_eligible_mask(ref_pool, min_n_listings, min_n_explicit)]
        max_cvr = float(elig["explicit_per_1k_listings"].max()) if len(elig) else np.nan
        cvr_h1 = float(r.get("cvr_h1", np.nan))
        cvr_h2 = float(r.get("cvr_h2", np.nan))
        rel_h1 = cvr_h1 * 1000 / max_cvr if max_cvr > 0 and pd.notna(cvr_h1) else np.nan
        rel_h2 = cvr_h2 * 1000 / max_cvr if max_cvr > 0 and pd.notna(cvr_h2) else np.nan
        gh = float(r.get("growth_cvr_pct", np.nan))
        q_h1 = _assign_quadrant_simple(0.0, rel_h1, med_g, med_s)
        q_h2 = _assign_quadrant_simple(gh, rel_h2, med_g, med_s)
        e1, e2 = int(r.get("explicit_h1", 0)), int(r.get("explicit_h2", 0))
        mig = "flat" if e2 == e1 else ("up" if e2 > e1 else "down")
        rows.append(
            {
                "region": r["region"],
                "city_name": r["city_name"],
                "category": r["category"],
                "ad_type": r["ad_type"],
                "bcg_quadrant_current": r["bcg_quadrant"],
                "quadrant_h1_cvr_proxy": q_h1,
                "quadrant_h2_cvr_proxy": q_h2,
                "explicit_migration": mig,
                "bcg_median_scope": scope,
                "note": "H1 proxy: rel_cvr_h1 vs median, growth=0; H2: full growth_cvr_pct",
            }
        )
    return pd.DataFrame(rows)


def run_deep_dive_levers(action: pd.DataFrame, init_db: bool) -> pd.DataFrame:
    if not init_db:
        return pd.DataFrame()

    targets = action[action["deep_dive_eligible"] == 1]
    if targets.empty:
        return pd.DataFrame()

    _ensure_db()
    dim_glob = rev.DIM_GLOB
    out_rows = []
    for _, r in targets.iterrows():
        city = str(r["city_name"]).replace("'", "''")
        cat = int(r["category"])
        adt = str(r["ad_type"]).replace("'", "''")
        qdf = rev.con.execute(f"""
            WITH dim_city AS (
                SELECT
                    CAST(item_id AS VARCHAR) AS item_id,
                    seller_type
                FROM read_parquet('{dim_glob}')
                WHERE category = {cat}
                  AND ad_type = '{adt}'
                  AND TRIM(CAST(city_name AS VARCHAR)) = '{city}'
            )
            SELECT
                COALESCE(d.seller_type, 'unknown') AS seller_type,
                e.event_type,
                COUNT(*)::BIGINT AS n_events
            FROM events_geo e
            INNER JOIN dim_city d ON e.item_id = d.item_id
            WHERE e.city_name = '{city}'
              AND e.dim_category = {cat}
              AND e.ad_type = '{adt}'
              AND e.event_type IN ({rev.EXPLICIT_SQL})
            GROUP BY 1, 2
        """).df()
        agent_pct = 0.0
        top_event = ""
        if not qdf.empty:
            tot = float(qdf["n_events"].sum())
            agent = float(
                qdf.loc[qdf["seller_type"] == "agent", "n_events"].sum()
            )
            agent_pct = 100.0 * agent / tot if tot else 0.0
            top_event = str(
                qdf.sort_values("n_events", ascending=False).iloc[0]["event_type"]
            )
        out_rows.append(
            {
                "region": r["region"],
                "city_name": r["city_name"],
                "category": cat,
                "ad_type": adt,
                "bcg_quadrant": r["bcg_quadrant"],
                "pct_agent_explicit": round(agent_pct, 2),
                "top_explicit_event": top_event,
                "deep_dive_eligible": 1,
            }
        )
    return pd.DataFrame(out_rows)


def merge_levers_into_action(
    action: pd.DataFrame, levers: pd.DataFrame
) -> pd.DataFrame:
    if levers.empty:
        return action
    extra = levers[
        [
            "region",
            "city_name",
            "category",
            "ad_type",
            "pct_agent_explicit",
            "top_explicit_event",
        ]
    ]
    out = action.merge(
        extra,
        on=["region", "city_name", "category", "ad_type"],
        how="left",
    )
    out["suggested_levers"] = out.apply(
        lambda r: (
            f"{r['suggested_levers']};agent={r.get('pct_agent_explicit', '')}%"
            f";top={r.get('top_explicit_event', '')}"
            if pd.notna(r.get("top_explicit_event"))
            and str(r.get("top_explicit_event", "")) != ""
            else r["suggested_levers"]
        ),
        axis=1,
    )
    return out


def write_playbook_md(profiles: pd.DataFrame) -> Path:
    path = OUT_DIR / "BCG_QUADRANT_PLAYBOOK.md"
    disclaimer = textwrap.dedent("""
    > **Disclaimer:** Explicit events (login) là proxy lead, không phải GMV VND.
    > Growth H1→H2 dùng snapshot `n_listings` — kiểm tra `listing_survival_rate` / `growth_quality_flag`.
    > Median có thể fallback (`bcg_median_scope`); ô borderline (±5%) — theo dõi, không đổi chiến lược mạnh.
    """).strip()

    sections = {
        "stars": (
            "**Mục tiêu:** Bảo vệ + nhân bản peer.\n"
            "**KPI:** `cvr_per_listing`, `growth_cvr_pct`, `priority_defend`.\n"
            "**Lever:** chất lượng tin, scale supply có kiểm soát.\n"
            "**Cải thiện:** Bảo toàn — ưu tiên `priority_defend` khi growth < median."
        ),
        "cash_cows": (
            "**Mục tiêu:** Thu hoạch, chống suy thoái.\n"
            "**KPI:** `n_explicit_events`, refresh listing, mix kênh explicit.\n"
            "**Lever:** harvest_refresh, explicit_mix.\n"
            "**Cải thiện:** Cao nếu kéo growth → Star; `uplift_explicit` ∝ `n×ΔCVR`."
        ),
        "question_marks": (
            "**Mục tiêu:** A/B CVR có chọn lọc.\n"
            "**KPI:** `cvr_per_listing`, `uplift_explicit`.\n"
            "**Lever:** ab_test_cvr, intent_fit, agent_quality.\n"
            "**Cải thiện:** Upside CVR cao nhất khi `n` đủ."
        ),
        "dogs": (
            "**Mục tiêu:** Sửa hoặc thu hẹp — tránh scale mù.\n"
            "**KPI:** CVR/tin, chỉ ưu tiên khi `e` lớn (`priority_impact` thấp hơn QM).\n"
            "**Lever:** cut_low_supply, category_shift.\n"
            "**Cải thiện:** Thấp/trung bình; multiplier 0.5 trong `priority_impact`."
        ),
        "low_volume": (
            "**Mục tiêu:** Đạt `n≥40`, `e≥20` trước khi gán màu BCG.\n"
            "**Không** diễn giải CVR cao trên `n` nhỏ là Sao."
        ),
    }

    lines = [
        "# BCG Quadrant Playbook (v2)\n",
        disclaimer,
        "\n## Tổng quan profile\n\n```csv\n",
        profiles.to_csv(index=False),
        "```\n",
        "\n## Chiến lược theo nhóm\n",
    ]
    for q, body in sections.items():
        lines.append(f"\n### {q}\n\n{body}\n")

    lines.append(
        "\n## Borderline (±5% median)\n\n"
        "Ô `borderline_flag=1`: theo dõi 2 kỳ; không flip chiến lược vì biên 5.1% vs 4.9%.\n"
    )
    lines.append(
        "\n## Priority\n\n"
        "- `priority_impact` = `uplift_explicit` × strategic_multiplier × stability_factor\n"
        "- `priority_defend` = Stars/Cows khi growth < median (cảnh báo suy thoái)\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved", path)
    return path


def plot_quadrant_summary(profiles: pd.DataFrame, action: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("BCG 4 nhóm — profile (eligible cells)", fontsize=13, y=1.02)

    prof4 = profiles[profiles["bcg_quadrant"].isin(BCG4)]
    colors = {
        "stars": "#F4C430",
        "cash_cows": "#2E7D32",
        "question_marks": "#1565C0",
        "dogs": "#C62828",
    }

    axes[0, 0].bar(
        prof4["bcg_quadrant"],
        prof4["n_cells_colored"],
        color=[colors.get(q, "#888") for q in prof4["bcg_quadrant"]],
    )
    axes[0, 0].set_title("Số ô (tỉnh×segment)")
    axes[0, 0].tick_params(axis="x", rotation=15)

    axes[0, 1].bar(
        prof4["bcg_quadrant"],
        prof4["total_explicit"],
        color=[colors.get(q, "#888") for q in prof4["bcg_quadrant"]],
    )
    axes[0, 1].set_title("Tổng explicit")

    axes[1, 0].bar(
        prof4["bcg_quadrant"],
        prof4["median_cvr_per_listing"],
        color=[colors.get(q, "#888") for q in prof4["bcg_quadrant"]],
    )
    axes[1, 0].set_title("Median CVR/tin")

    bl_pct = action.groupby("bcg_quadrant")["borderline_flag"].mean() * 100.0
    bl_pct = bl_pct.reindex([q for q in BCG4 if q in bl_pct.index])
    axes[1, 1].bar(
        bl_pct.index,
        bl_pct.values,
        color=[colors.get(q, "#888") for q in bl_pct.index],
    )
    axes[1, 1].set_title("% ô borderline (±5% median)")

    out = FIG_DIR / "fig_bcg_quadrant_summary.png"
    rev._finish_fig(fig, out)
    return out


def plot_priority_top20(action: pd.DataFrame) -> Path:
    top_imp = action.nlargest(20, "priority_impact")
    top_def = action[action["priority_defend"] > 0].nlargest(20, "priority_defend")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Top ưu tiên — Impact vs Defend", fontsize=13)

    labels = [
        f"{r['city_name'][:12]} {r['category']}{'L' if r['ad_type']=='let' else 'S'}"
        for _, r in top_imp.iterrows()
    ]
    ax1.barh(labels[::-1], top_imp["priority_impact"].values[::-1], color="#1565C0")
    ax1.set_title("priority_impact (uplift × multiplier)")

    if len(top_def):
        labels2 = [
            f"{r['city_name'][:12]} {r['category']}{'L' if r['ad_type']=='let' else 'S'}"
            for _, r in top_def.iterrows()
        ]
        ax2.barh(labels2[::-1], top_def["priority_defend"].values[::-1], color="#F4C430")
    ax2.set_title("priority_defend (Stars/Cows suy growth)")
    if not len(top_def):
        ax2.text(0.5, 0.5, "Không có ô defend", ha="center", va="center")

    out = FIG_DIR / "fig_bcg_priority_top20.png"
    rev._finish_fig(fig, out)
    return out


def append_summary_md(action: pd.DataFrame, profiles: pd.DataFrame) -> None:
    path = OUT_DIR / "SUMMARY.md"
    if not path.exists():
        return
    lines = ["\n## BCG playbook priorities (v2)\n"]
    for q in BCG4:
        sub = action[action["bcg_quadrant"] == q]
        if sub.empty:
            continue
        lines.append(f"\n### Top impact — {q}\n")
        for _, r in sub.nlargest(5, "priority_impact").iterrows():
            lines.append(
                f"- {r['city_name']} {r['category']} {r['ad_type']}: "
                f"impact={r['priority_impact']:.0f}, CVR={r['cvr_per_listing']:.2f}/tin, "
                f"scope={r.get('bcg_median_scope','')}\n"
            )
    path.write_text(path.read_text(encoding="utf-8") + "".join(lines), encoding="utf-8")
    print("Updated", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="DuckDB: item-level survival + deep-dive levers (chậm hơn)",
    )
    parser.add_argument(
        "--buffer-pct",
        type=float,
        default=BCG_MEDIAN_BUFFER_PCT,
        help="Borderline band around median (percent)",
    )
    parser.add_argument("--min-n-listings", type=int, default=rev.MIN_N_LISTINGS)
    parser.add_argument("--min-n-explicit", type=int, default=rev.MIN_N_EXPLICIT)
    args = parser.parse_args()

    nat = load_national_bcg()
    survival = build_listing_survival_metrics(args.init_db)

    action = enrich_action_rows(
        nat,
        survival,
        args.buffer_pct,
        args.min_n_listings,
        args.min_n_explicit,
    )
    action = add_peer_columns(action)
    action = mark_deep_dive_eligible(action)

    levers = run_deep_dive_levers(action, args.init_db)
    if not levers.empty:
        levers.to_csv(OUT_DIR / "13_bcg_lever_by_quadrant.csv", index=False)
        print("Saved", OUT_DIR / "13_bcg_lever_by_quadrant.csv")
        action = merge_levers_into_action(action, levers)

    profiles = build_quadrant_profiles(action, nat)
    detail = nat.merge(
        action[
            [
                "city_name",
                "category",
                "ad_type",
                "borderline_flag",
                "borderline_axis",
                "uplift_explicit",
                "priority_impact",
                "priority_defend",
                "listing_survival_rate",
                "listing_churn_pct",
                "growth_quality_flag",
                "peer_top3",
                "deep_dive_eligible",
            ]
        ],
        on=["city_name", "category", "ad_type"],
        how="left",
    )
    migration = build_migration_table(
        nat, args.min_n_listings, args.min_n_explicit
    )

    profiles.to_csv(OUT_DIR / "09_bcg_quadrant_profiles.csv", index=False)
    action.to_csv(OUT_DIR / "10_bcg_action_matrix.csv", index=False)
    detail.to_csv(OUT_DIR / "11_bcg_quadrant_cells_detail.csv", index=False)
    migration.to_csv(OUT_DIR / "12_bcg_h1_h2_migration.csv", index=False)
    for p in (
        "09_bcg_quadrant_profiles.csv",
        "10_bcg_action_matrix.csv",
        "11_bcg_quadrant_cells_detail.csv",
        "12_bcg_h1_h2_migration.csv",
    ):
        print("Saved", OUT_DIR / p)

    write_playbook_md(profiles)
    plot_quadrant_summary(profiles, action)
    plot_priority_top20(action)
    append_summary_md(action, profiles)
    print("Done →", OUT_DIR)


if __name__ == "__main__":
    main()
