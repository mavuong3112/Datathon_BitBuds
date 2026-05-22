"""Build SCIS business decks for categories 1010 and 1020 from EDA CSV exports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
PERF_OUT = DATA_ROOT / "outputs" / "eda_category_1010_1020"
BEHAV_ROOT = DATA_ROOT / "outputs" / "eda_category_behavior"
BRIDGE_OUT = PERF_OUT / "bridge"
CLUSTER_OUT = PERF_OUT / "clustering"

CAT_META = {
    1010: {
        "label": "1010 — Căn hộ / Chung cư",
        "short": "Căn hộ / Chung cư",
        "market_note": "thị trường cho thuê căn hộ",
        "struct_note": "project × bedrooms (41% có project_id)",
        "cta_note": "Chat-first CTA — 30% explicit là contact_chat",
        "device_note": "Desktop + iOS (40% + 33%) — UX cross-device cho thuê căn hộ",
        "weak_slice_label": "1-bed let ~18%",
        "weak_slice_listings_key": "weak_1bed_let",
    },
    1020: {
        "label": "1020 — Nhà ở",
        "short": "Nhà ở",
        "market_note": "thị trường bán nhà",
        "struct_note": "house_type 100%, floors 61%, width 76%",
        "cta_note": "Phone-first CTA — 76% explicit là view_phone",
        "device_note": "Android 26% — mobile-first cho Nhà ở",
        "weak_slice_label": "sell 30–50m² CVR 17.5%",
        "weak_slice_listings_key": "weak_sell_3050",
    },
}

SECTION_CHARTS = {
    "SITUATION": [
        "01_supply_adtype.png",
        "02_cvr_baseline_adtype.png",
        "03_hcm_concentration.png",
    ],
    "CHALLENGES": [
        "04_session_funnel_gap.png",
        "05_event_layer_mix.png",
        "06_health_segment_efficiency.png",
    ],
    "STRATEGIES": [
        "07_contact_channel_mix.png",
        "08_device_mix.png",
        "09_session_archetypes.png",
    ],
    "IMPACTS": ["10_impact_whatif.png"],
}

HEALTH_LABELS = {
    "high_quality_underexposed": "HQ underexposed",
    "normal": "Normal",
    "oversaturated_low_conversion": "Oversaturated",
}

HEALTH_COLORS = {
    "high_quality_underexposed": "#2ca02c",
    "normal": "#1f77b4",
    "oversaturated_low_conversion": "#d62728",
}


@dataclass
class CategoryData:
    cat: int
    profile: pd.Series
    cvr_baseline: pd.DataFrame
    supply_adtype: pd.DataFrame
    supply_cities: pd.DataFrame
    snapshot: pd.DataFrame
    event_layers: pd.DataFrame
    explicit_mix: pd.DataFrame
    device: pd.DataFrame
    funnel: pd.DataFrame
    segments: pd.DataFrame
    scorecard: pd.DataFrame
    health_efficiency: pd.DataFrame
    session_funnel: pd.DataFrame
    area_cvr: pd.DataFrame
    slices_1010: pd.DataFrame | None = None
    slices_1020: pd.DataFrame | None = None
    seller_profile: pd.DataFrame | None = None
    health_adtype: pd.DataFrame | None = None


@dataclass
class ImpactScenarios:
    boost_contacts: float
    session_leads: float
    demote_impressions_saved: float
    weak_slice_contacts: float
    rows: list[dict[str, Any]] = field(default_factory=list)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _scorecard_value(df: pd.DataFrame, metric: str) -> float:
    row = df.loc[df["metric"] == metric]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["value"])


def load_category_data(cat: int) -> CategoryData:
    beh = BEHAV_ROOT / str(cat)
    profile = _read_csv(PERF_OUT / "01_profile_1010_1020.csv")
    profile = profile.loc[profile["category"] == cat].iloc[0]
    cvr = _read_csv(PERF_OUT / "02_cvr_baseline_adtype.csv")
    cvr = cvr.loc[cvr["category"] == cat]
    health = _read_csv(BRIDGE_OUT / "04_segment_event_efficiency.csv")
    health = health.loc[health["category"] == cat]
    session = _read_csv(BRIDGE_OUT / "05_session_funnel_by_cluster.csv")
    session = session.loc[session["category"] == cat]
    area = _read_csv(PERF_OUT / "05_cvr_area_bucket_shared.csv")
    area = area.loc[area["category"] == cat]
    health_adtype = _read_csv(BRIDGE_OUT / "02c_cvr_by_adtype_health.csv")
    health_adtype = health_adtype.loc[health_adtype["category"] == cat]

    data = CategoryData(
        cat=cat,
        profile=profile,
        cvr_baseline=cvr,
        supply_adtype=_read_csv(beh / "01_supply_ad_type.csv"),
        supply_cities=_read_csv(beh / "01_supply_top_cities.csv"),
        snapshot=_read_csv(beh / "02_snapshot_by_adtype.csv"),
        event_layers=_read_csv(beh / "03_events_login_event_layers.csv"),
        explicit_mix=_read_csv(beh / "03_events_login_explicit_contact_mix.csv"),
        device=_read_csv(beh / "03_events_login_device.csv"),
        funnel=_read_csv(beh / "03_events_login_funnel.csv"),
        segments=_read_csv(beh / "04_interactions_login_segments.csv"),
        scorecard=_read_csv(beh / "05_marketing_scorecard.csv"),
        health_efficiency=health,
        session_funnel=session,
        area_cvr=area,
        health_adtype=health_adtype,
        seller_profile=_read_csv(CLUSTER_OUT / f"profile_seller_{cat}.csv"),
    )
    if cat == 1010:
        data.slices_1010 = _read_csv(PERF_OUT / "03a_cvr_1010_slices.csv")
    else:
        data.slices_1020 = _read_csv(PERF_OUT / "04_cvr_1020_slices.csv")
    return data


def _fmt(n: float | int, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{int(round(n)):,}".replace(",", ".")
    return f"{n:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}%"


def _weak_slice_stats(data: CategoryData) -> tuple[int, float, float, float]:
    if data.cat == 1010:
        sl = data.slices_1010
        assert sl is not None
        sub = sl.loc[(sl["ad_type"] == "let") & (sl["bed_bucket"] == "1")]
        listings = int(sub["listings"].sum())
        cvr = float(sub["listings_with_positive"].sum() / listings * 100) if listings else 0.0
        return listings, cvr, 18.0, 20.0
    sl = data.slices_1020
    assert sl is not None
    row = data.area_cvr.loc[
        (data.area_cvr["ad_type"] == "sell") & (data.area_cvr["area_bucket"] == "30-50")
    ]
    listings = int(row.iloc[0]["listings"])
    cvr = float(row.iloc[0]["cvr_pct"])
    return listings, cvr, cvr, cvr + 2.0


def compute_impact_scenarios(data: CategoryData) -> ImpactScenarios:
    hq = data.health_efficiency.loc[
        data.health_efficiency["health_segment"] == "high_quality_underexposed"
    ].iloc[0]
    over = data.health_efficiency.loc[
        data.health_efficiency["health_segment"] == "oversaturated_low_conversion"
    ].iloc[0]

    n_hq = int(hq["n"])
    med_exp = float(hq["med_exposure"])
    avg_pv = float(hq["avg_pageviews"])
    eff = float(hq["avg_event_contact_rate_pct"]) / 100.0
    exp_lift = max(4.0 - med_exp, 0.0)
    boost_contacts = n_hq * avg_pv * exp_lift * eff * 0.35

    total_sessions = float(data.event_layers["sessions"].max())
    session_pct = _scorecard_value(data.scorecard, "session_explicit_contact_pct")
    session_leads = total_sessions * 0.01

    n_over = int(over["n"])
    over_exp = float(over["med_exposure"])
    over_pv = float(over["avg_pageviews"])
    demote_impressions_saved = n_over * over_exp * over_pv * 0.25

    w_listings, w_cvr, cvr_from, cvr_to = _weak_slice_stats(data)
    weak_slice_contacts = w_listings * (cvr_to - cvr_from) / 100.0

    rows = [
        {
            "scenario": "Boost HQ underexposed",
            "value": boost_contacts,
            "unit": "contact events",
        },
        {
            "scenario": "Session explicit +1pp",
            "value": session_leads,
            "unit": "sessions w/ contact",
        },
        {
            "scenario": "Demote oversaturated",
            "value": demote_impressions_saved,
            "unit": "impression-equiv saved",
        },
        {
            "scenario": "Weak slice CVR lift",
            "value": weak_slice_contacts,
            "unit": "listings w/ contact",
        },
    ]
    return ImpactScenarios(
        boost_contacts=boost_contacts,
        session_leads=session_leads,
        demote_impressions_saved=demote_impressions_saved,
        weak_slice_contacts=weak_slice_contacts,
        rows=rows,
    )


def infer_scis_bullets(data: CategoryData, impacts: ImpactScenarios) -> dict[str, list[str]]:
    meta = CAT_META[data.cat]
    cat = data.cat
    n_listings = int(data.profile["n"])
    pct_let = float(data.profile["pct_let"])
    pct_sell = 100.0 - pct_let
    hcm_pct = float(data.supply_cities.iloc[0]["pct"])

    cvr_let = float(data.cvr_baseline.loc[data.cvr_baseline["ad_type"] == "let", "cvr_pct"].iloc[0])
    cvr_sell = float(data.cvr_baseline.loc[data.cvr_baseline["ad_type"] == "sell", "cvr_pct"].iloc[0])

    snap_let = data.snapshot.loc[data.snapshot["ad_type"] == "let"]
    snap_sell = data.snapshot.loc[data.snapshot["ad_type"] == "sell"]
    let_contact_day = float(snap_let.iloc[0]["pct_days_with_contact"]) if len(snap_let) else float("nan")
    sell_contact_day = float(snap_sell.iloc[0]["pct_days_with_contact"]) if len(snap_sell) else float("nan")
    let_age = float(snap_let.iloc[0]["avg_age_days"]) if len(snap_let) else float("nan")

    explicit_pct = _scorecard_value(data.scorecard, "explicit_contact_event_pct")
    other_pct = _scorecard_value(data.scorecard, "other_interaction_event_pct")
    ad_view_pct = _scorecard_value(data.scorecard, "other_interaction_top_surface_pct")
    session_explicit = _scorecard_value(data.scorecard, "session_explicit_contact_pct")
    high_intent = _scorecard_value(data.scorecard, "high_intent_user_pct")
    spam = float(data.segments.loc[data.segments["segment"] == "C_spam_broker", "pct"].iloc[0])

    hq = data.health_efficiency.loc[
        data.health_efficiency["health_segment"] == "high_quality_underexposed"
    ].iloc[0]
    over = data.health_efficiency.loc[
        data.health_efficiency["health_segment"] == "oversaturated_low_conversion"
    ].iloc[0]

    login_users = int(data.event_layers["users"].max())
    w_listings, w_cvr, cvr_from, cvr_to = _weak_slice_stats(data)

    noise = data.session_funnel.loc[data.session_funnel["cluster_id"] == -1].iloc[0]
    deep_compare = float(noise["rate_deep_compare"]) * 100

    situation: list[str] = []
    if cat == 1010:
        situation = [
            f"**{_fmt(n_listings)} listings**, **{_pct(pct_let, 1)} let** — {meta['market_note']}; sell {_pct(pct_sell, 1)}",
            f"**{_fmt(login_users)} login users** (sample); cấu trúc **{meta['struct_note']}**",
            f"**TP.HCM ~{_pct(hcm_pct, 1)}** supply — chiến lược theo thành phố, không nationwide generic",
            f"Catalog CVR: let **{_pct(cvr_let)}**, sell **{_pct(cvr_sell)}** — sell outperform let (+{cvr_sell - cvr_let:.1f}pp)",
            f"Snapshot let: contact-day **{_pct(let_contact_day)}**; listing age ~{let_age:.0f} ngày",
        ]
    else:
        situation = [
            f"**{_fmt(n_listings)} listings**, **{_pct(pct_sell, 1)} sell** — {meta['market_note']}; let {_pct(pct_let, 1)}",
            f"**{_fmt(login_users)} login users** (sample); {meta['struct_note']}",
            f"**TP.HCM ~{_pct(hcm_pct, 1)}** supply — tập trung micro-market HCMC",
            f"Catalog CVR: let **{_pct(cvr_let)}**, sell **{_pct(cvr_sell)}** — let hơn sell (+{cvr_let - cvr_sell:.1f}pp), **ngược 1010**",
            f"Snapshot: let contact-day **{_pct(let_contact_day)}** (mạnh); sell **{_pct(sell_contact_day)}** trên volume lớn",
        ]

    challenges: list[str] = [
        f"Funnel login: chỉ **{_pct(session_explicit)} session** có explicit contact; **{_pct(other_pct, 1)} events** là `other_interaction` ({_pct(ad_view_pct, 0)} ad_view)",
        f"**High-intent users {_pct(high_intent)}**; broker/spam **{_pct(spam)}**",
        f"**~{_fmt(hq['n'])} listings HQ underexposed** ({_pct(float(hq['avg_event_contact_rate_pct']), 0)} contact/pageview, exposure={hq['med_exposure']:.0f}) vs **~{_fmt(over['n'])} oversaturated** ({_pct(float(over['avg_event_contact_rate_pct']), 0)}, exposure={over['med_exposure']:.0f})",
    ]
    if cat == 1010:
        challenges.append(
            f"Weak pocket: **{meta['weak_slice_label']}** CVR trên **~{_fmt(w_listings)} listings** — volume lớn, conversion thấp"
        )
        challenges.append(
            "Catalog CVR bị méo bởi **~85% tin pre-EDA window** — filter in-window cho quyết định product"
        )
    else:
        challenges.append(
            f"Weak pocket: **{meta['weak_slice_label']}** trên **~{_fmt(w_listings)} listings**; let mặt phố ~15%"
        )
        bulk = data.seller_profile
        assert bulk is not None
        cl0 = bulk.loc[bulk["cluster_id"] == 0].iloc[0]
        challenges.append(
            f"**Bulk seller cluster** (~{cl0['n_listings']:.0f} listings/seller, {cl0['pageviews_per_listing']:.2f} PV/listing) — spam/cold supply risk"
        )

    hq_n = int(hq["n"])
    over_n = int(over["n"])
    strategies: list[str] = [
        f"**Boost { _fmt(hq_n) } HQ underexposed** — recsys rank / feed diversity (`10_health_ranked_underexposed_{cat}.csv`)",
        f"**Demote { _fmt(over_n) } oversaturated** — cap impression share (exposure median {over['med_exposure']:.0f})",
        meta["cta_note"],
        meta["device_note"],
    ]
    if cat == 1010:
        strategies.extend(
            [
                "**Night-active micro-segment** (~500 users, 99% night) — push/notification theo giờ",
                "**Listing quality:** ưu tiên 3–4+ PN, project-linked 2-bed sell; completeness furnishing",
            ]
        )
    else:
        hq_sell = 0
        if data.health_adtype is not None:
            hq_rows = data.health_adtype.loc[
                data.health_adtype["health_segment"] == "high_quality_underexposed"
            ]
            hq_sell = int(hq_rows.loc[hq_rows["ad_type"] == "sell", "n"].sum())
        strategies.extend(
            [
                f"**Deep-compare UX** — {deep_compare:.1f}% session noise có deep_compare (house buyer journey dài)",
                f"**Listing completeness:** house_type + floors + legal + width; ưu tiên sell HCMC ({_fmt(hq_sell)} HQ underexposed sell)",
            ]
        )

    impact_bullets = [
        f"Boost underexposed: **{_fmt(hq_n)} listings**, exposure {hq['med_exposure']:.0f}→4 → ước tính **+{_fmt(impacts.boost_contacts)} contact events**",
        f"Session explicit **+1pp** ({session_explicit:.2f}→{session_explicit + 1:.2f}%) → **+{_fmt(impacts.session_leads)} sessions** có contact (trong sample login)",
        f"Demote oversaturated: tiết kiệm **~{_fmt(impacts.demote_impressions_saved)} impression-equiv** trên {_fmt(over_n)} tin exposure cao",
        f"Cải thiện {meta['weak_slice_label']} **{cvr_from:.1f}→{cvr_to:.1f}%** → **~{_fmt(impacts.weak_slice_contacts)} listings** thêm contact",
    ]

    return {
        "SITUATION": situation,
        "CHALLENGES": challenges,
        "STRATEGIES": strategies,
        "IMPACTS": impact_bullets,
    }


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _style_axes(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def chart_supply_adtype(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    df = data.supply_adtype
    colors = ["#1f77b4", "#ff7f0e"]
    ax.bar(df["ad_type"], df["pct"], color=colors)
    for i, row in df.iterrows():
        ax.text(i, row["pct"] + 1, f"{row['pct']:.1f}%", ha="center", fontsize=9)
    _style_axes(ax, f"{data.cat} — Supply mix (ad_type)", "Tỷ lệ %")
    _savefig(out)


def chart_sell_house_type_cvr(data: CategoryData, out: Path) -> None:
    assert data.slices_1020 is not None
    sl = data.slices_1020.loc[data.slices_1020["ad_type"] == "sell"].copy()
    agg = (
        sl.groupby("house_type", as_index=False)
        .agg(listings=("listings", "sum"), pos=("listings_with_positive", "sum"))
        .assign(cvr_pct=lambda d: d["pos"] / d["listings"] * 100)
        .sort_values("listings", ascending=False)
        .head(6)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [t[:22] + "…" if len(t) > 22 else t for t in agg["house_type"]]
    ax.barh(labels[::-1], agg["cvr_pct"][::-1], color="#9467bd")
    for i, (_, row) in enumerate(agg.iloc[::-1].iterrows()):
        ax.text(row["cvr_pct"] + 0.2, i, f"{row['cvr_pct']:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("CVR %")
    ax.set_title(f"{data.cat} — Sell CVR theo house_type (top volume)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    _savefig(out)


def chart_cvr_baseline(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    df = data.cvr_baseline
    ax.bar(df["ad_type"], df["cvr_pct"], color=["#2ca02c", "#9467bd"])
    for i, row in df.iterrows():
        ax.text(i, row["cvr_pct"] + 0.3, f"{row['cvr_pct']:.1f}%", ha="center", fontsize=9)
    _style_axes(ax, f"{data.cat} — Catalog CVR theo ad_type", "CVR %")
    _savefig(out)


def chart_hcm_concentration(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    df = data.supply_cities.head(6)
    ax.barh(df["city_name"][::-1], df["pct"][::-1], color="#17becf")
    for i, (_, row) in enumerate(df.iloc[::-1].iterrows()):
        ax.text(row["pct"] + 0.3, i, f"{row['pct']:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Tỷ lệ supply %")
    ax.set_title(f"{data.cat} — Top cities (supply)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    _savefig(out)


def chart_session_funnel(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    df = data.funnel
    stages = ["search", "view", "consider", "explicit_contact", "other_interaction"]
    df = df.set_index("stage").reindex(stages).reset_index()
    colors = ["#aec7e8", "#1f77b4", "#ffbb78", "#2ca02c", "#c5b0d5"]
    ax.bar(df["stage"], df["pct_of_sessions"], color=colors)
    for i, row in df.iterrows():
        ax.text(i, row["pct_of_sessions"] + 1, f"{row['pct_of_sessions']:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["stage"], rotation=25, ha="right")
    _style_axes(ax, f"{data.cat} — Session funnel (% sessions login)", "Tỷ lệ %")
    _savefig(out)


def chart_event_layers(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    df = data.event_layers
    colors = {"pageview": "#1f77b4", "other_interaction": "#ff7f0e", "explicit_contact": "#2ca02c"}
    ax.bar(
        df["event_layer"],
        df["pct_events"],
        color=[colors.get(x, "#888") for x in df["event_layer"]],
    )
    for i, row in df.iterrows():
        ax.text(i, row["pct_events"] + 1, f"{row['pct_events']:.1f}%", ha="center", fontsize=9)
    _style_axes(ax, f"{data.cat} — Event layer mix (login)", "Tỷ lệ events %")
    _savefig(out)


def chart_health_segment(data: CategoryData, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    df = data.health_efficiency.copy()
    order = [
        "high_quality_underexposed",
        "normal",
        "oversaturated_low_conversion",
    ]
    df["health_segment"] = pd.Categorical(df["health_segment"], categories=order, ordered=True)
    df = df.sort_values("health_segment")
    x = np.arange(len(df))
    labels = [HEALTH_LABELS[s] for s in df["health_segment"]]
    colors = [HEALTH_COLORS[s] for s in df["health_segment"]]
    bars = ax1.bar(x, df["avg_event_contact_rate_pct"], color=colors, alpha=0.85)
    ax1.set_ylabel("Contact / pageview %")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, df["med_exposure"], color="black", marker="o", linewidth=2, label="Med exposure")
    ax2.set_ylabel("Median exposure")
    for bar, val in zip(bars, df["avg_event_contact_rate_pct"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.0f}%",
            ha="center",
            fontsize=8,
        )
    ax1.set_title(f"{data.cat} — Marketplace health segments", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    _savefig(out)


def chart_contact_channel(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    df = data.explicit_mix
    ax.bar(df["event_type"], df["pct_of_explicit"], color="#9467bd")
    for i, row in df.iterrows():
        ax.text(i, row["pct_of_explicit"] + 1, f"{row['pct_of_explicit']:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["event_type"], rotation=20, ha="right")
    _style_axes(ax, f"{data.cat} — Explicit contact mix", "Tỷ lệ %")
    _savefig(out)


def chart_device_mix(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    df = data.device
    ax.bar(df["device"], df["pct_events"], color="#8c564b")
    for i, row in df.iterrows():
        ax.text(i, row["pct_events"] + 0.8, f"{row['pct_events']:.1f}%", ha="center", fontsize=9)
    _style_axes(ax, f"{data.cat} — Device mix (login events)", "Tỷ lệ %")
    _savefig(out)


def chart_session_archetypes(data: CategoryData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sf = data.session_funnel
    rows = []
    noise = sf.loc[sf["cluster_id"] == -1]
    if not noise.empty:
        r = noise.iloc[0]
        rows.append(
            {
                "archetype": "Noise (mainstream)",
                "contact": r["rate_has_contact"] * 100,
                "search": r["rate_has_search"] * 100,
                "deep_compare": r["rate_deep_compare"] * 100,
            }
        )
    for cl, name in [(0, "Search-only"), (1, "Search+contact")]:
        sub = sf.loc[sf["cluster_id"] == cl]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append(
                {
                    "archetype": name,
                    "contact": r["rate_has_contact"] * 100,
                    "search": r["rate_has_search"] * 100,
                    "deep_compare": r["rate_deep_compare"] * 100,
                }
            )
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["contact"], w, label="Contact", color="#2ca02c")
    ax.bar(x, df["search"], w, label="Search", color="#1f77b4")
    ax.bar(x + w, df["deep_compare"], w, label="Deep compare", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(df["archetype"], rotation=15, ha="right")
    ax.set_ylabel("Tỷ lệ %")
    ax.set_title(f"{data.cat} — Session archetypes", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _savefig(out)


def chart_impact_whatif(data: CategoryData, impacts: ImpactScenarios, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df = pd.DataFrame(impacts.rows)
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    ax.barh(df["scenario"], df["value"], color=colors)
    for i, row in df.iterrows():
        ax.text(row["value"] * 1.01, i, _fmt(row["value"]), va="center", fontsize=8)
    ax.set_xlabel("Giá trị ước tính")
    ax.set_title(f"{data.cat} — Impact what-if scenarios", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    _savefig(out)


def generate_charts(data: CategoryData, impacts: ImpactScenarios, chart_dir: Path) -> list[str]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    cvr_chart = (
        ("02_cvr_baseline_adtype.png", lambda p: chart_sell_house_type_cvr(data, p))
        if data.cat == 1020
        else ("02_cvr_baseline_adtype.png", lambda p: chart_cvr_baseline(data, p))
    )
    generators = [
        ("01_supply_adtype.png", lambda p: chart_supply_adtype(data, p)),
        cvr_chart,
        ("03_hcm_concentration.png", lambda p: chart_hcm_concentration(data, p)),
        ("04_session_funnel_gap.png", lambda p: chart_session_funnel(data, p)),
        ("05_event_layer_mix.png", lambda p: chart_event_layers(data, p)),
        ("06_health_segment_efficiency.png", lambda p: chart_health_segment(data, p)),
        ("07_contact_channel_mix.png", lambda p: chart_contact_channel(data, p)),
        ("08_device_mix.png", lambda p: chart_device_mix(data, p)),
        ("09_session_archetypes.png", lambda p: chart_session_archetypes(data, p)),
        ("10_impact_whatif.png", lambda p: chart_impact_whatif(data, impacts, p)),
    ]
    created: list[str] = []
    for name, fn in generators:
        path = chart_dir / name
        fn(path)
        created.append(name)
    return created


def write_markdown(
    cat: int,
    bullets: dict[str, list[str]],
    chart_dir: Path,
    out_path: Path,
) -> None:
    meta = CAT_META[cat]
    lines = [
        f"# Business insights — {meta['label']}",
        "",
        "> Nguồn: `eda_category_1010_1020_performance`, `behavior_deepdive`, `clustering`, `cluster_bridge` + CSV exports.",
        "",
    ]
    for section, items in bullets.items():
        lines.append(f"## {section}")
        lines.append("")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
        for chart_name in SECTION_CHARTS[section]:
            rel = f"charts/{chart_name}"
            lines.append(f"![{section} — {chart_name}]({rel})")
            lines.append("")

    lines.extend(
        [
            "---",
            "### Caveats",
            "- Explicit contact ≠ `other_interaction`/ad_view",
            "- CVR catalog ≠ CVR clustering cohort (~100% trên event sample)",
            "- Event sample 6–8% — số tuyệt đối mang tính ước lượng",
            "",
            "### Charts index",
        ]
    )
    for section, charts in SECTION_CHARTS.items():
        for c in charts:
            lines.append(f"- `charts/{c}` — {section}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    cat: int,
    bullets: dict[str, list[str]],
    charts: list[str],
    impacts: ImpactScenarios,
    out_path: Path,
) -> None:
    payload = {
        "category": cat,
        "label": CAT_META[cat]["label"],
        "charts": charts,
        "sections": bullets,
        "impact_scenarios": impacts.rows,
        "markdown": str(out_path.parent / f"category_{cat}_business_scis.md"),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_impact_csv(impacts: ImpactScenarios, csv_dir: Path) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(impacts.rows).to_csv(csv_dir / "impact_scenarios.csv", index=False)


def build_deck(cat: int) -> Path:
    out_root = DATA_ROOT / "outputs" / f"category_{cat}_business"
    chart_dir = out_root / "charts"
    csv_dir = out_root / "csv"
    md_path = out_root / f"category_{cat}_business_scis.md"
    manifest_path = out_root / "manifest.json"

    data = load_category_data(cat)
    impacts = compute_impact_scenarios(data)
    bullets = infer_scis_bullets(data, impacts)
    charts = generate_charts(data, impacts, chart_dir)
    write_impact_csv(impacts, csv_dir)
    write_markdown(cat, bullets, chart_dir, md_path)
    write_manifest(cat, bullets, charts, impacts, manifest_path)
    print(f"[{cat}] wrote {md_path} ({len(charts)} charts)")
    return md_path


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})
    for cat in (1010, 1020):
        build_deck(cat)


if __name__ == "__main__":
    main()
