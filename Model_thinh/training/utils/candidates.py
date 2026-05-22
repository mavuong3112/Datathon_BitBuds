"""Candidate generation: 4 sources (history, co-vis, popularity, content)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .covis import INTENT_WEIGHT, score_user_covis


def gen_history_candidates(user_events: pd.DataFrame, allowed_items: set,
                           top_n: int = 30, cutoff_ts=None) -> dict[str, dict[str, float]]:
    """Per user, items they have interacted with, scored by recency * intent weight."""
    if cutoff_ts is not None:
        user_events = user_events[user_events["event_ts"] < cutoff_ts]
    user_events = user_events[user_events["item_id"].isin(allowed_items)]
    out: dict[str, dict[str, float]] = {}
    if len(user_events) == 0:
        return out
    max_ts = user_events["event_ts"].max()
    user_events = user_events.assign(
        _w=user_events["event_type"].map(INTENT_WEIGHT).fillna(1.0),
        _age_days=(max_ts - user_events["event_ts"]).dt.total_seconds() / 86400.0,
    )
    user_events["_score"] = user_events["_w"] * np.exp(-user_events["_age_days"] / 14.0)
    agg = user_events.groupby(["user_id", "item_id"])["_score"].sum().reset_index()
    for uid, grp in agg.groupby("user_id", sort=False):
        top = grp.nlargest(top_n, "_score")
        out[uid] = dict(zip(top["item_id"], top["_score"]))
    return out


def gen_covis_candidates(user_events: pd.DataFrame, covis: dict, allowed_items: set,
                         top_n: int = 200, cutoff_ts=None) -> dict[str, dict[str, float]]:
    if cutoff_ts is not None:
        user_events = user_events[user_events["event_ts"] < cutoff_ts]
    out: dict[str, dict[str, float]] = {}
    for uid, grp in user_events.groupby("user_id", sort=False):
        history = list(zip(grp["item_id"].tolist(), grp["event_type"].tolist()))
        scores = score_user_covis(history, covis)
        scores = {k: v for k, v in scores.items() if k in allowed_items}
        if not scores:
            continue
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        out[uid] = dict(top)
    return out


def gen_popularity_candidates(user_profile: pd.DataFrame, dim_enriched: pd.DataFrame,
                              snap_recent: pd.DataFrame, top_n_per_seg: int = 50,
                              top_n_per_user: int = 100) -> dict[str, dict[str, float]]:
    """Top items by contacts in user's (category, city) segment in last 14d.

    user_profile: cols = [user_id, u_top_category, u_top_city]
    snap_recent: last 14d snapshot, cols = [item_id, contacts_24h, date]
    dim_enriched: cols include [item_id, category, city_name, tier]
    """
    pop = snap_recent.groupby("item_id", as_index=False)["contacts_24h"].sum()
    dim_a = dim_enriched[dim_enriched["tier"].isin(["A", "B"])][
        ["item_id", "category", "city_name"]
    ]
    pop = pop.merge(dim_a, on="item_id", how="inner")
    seg_top: dict[tuple, list[tuple]] = {}
    for (cat, city), grp in pop.groupby(["category", "city_name"], sort=False):
        top = grp.nlargest(top_n_per_seg, "contacts_24h")
        seg_top[(cat, city)] = list(zip(top["item_id"], top["contacts_24h"].astype(float)))

    global_top = pop.nlargest(top_n_per_seg, "contacts_24h")
    global_list = list(zip(global_top["item_id"], global_top["contacts_24h"].astype(float)))

    out: dict[str, dict[str, float]] = {}
    for _, row in user_profile.iterrows():
        uid = row["user_id"]
        seg = (row.get("u_top_category"), row.get("u_top_city"))
        items = seg_top.get(seg, global_list)[:top_n_per_user]
        if not items:
            items = global_list[:top_n_per_user]
        out[uid] = dict(items)
    return out


def gen_content_candidates(user_profile: pd.DataFrame, dim_enriched: pd.DataFrame,
                            top_n: int = 50) -> dict[str, dict[str, float]]:
    """Cold-item match by (category, district, ad_type) with user's top preferences.

    user_profile cols: [user_id, u_top_category, u_top_district, u_top_ad_type]
    """
    cold = dim_enriched[dim_enriched["tier"] == "C"][
        ["item_id", "category", "district_name", "ad_type", "posted_date"]
    ]
    if len(cold) == 0:
        return {}
    cold = cold.sort_values("posted_date", ascending=False)
    seg_top: dict[tuple, list[str]] = {}
    for key, grp in cold.groupby(["category", "district_name", "ad_type"], sort=False):
        seg_top[key] = grp["item_id"].head(top_n).tolist()

    out: dict[str, dict[str, float]] = {}
    for _, row in user_profile.iterrows():
        uid = row["user_id"]
        key = (row.get("u_top_category"), row.get("u_top_district"), row.get("u_top_ad_type"))
        items = seg_top.get(key, [])[:top_n]
        if not items:
            continue
        out[uid] = {it: 1.0 - i / max(len(items), 1) for i, it in enumerate(items)}
    return out


def merge_candidates(sources: dict[str, dict[str, dict[str, float]]],
                     cap_total: int = 500) -> pd.DataFrame:
    """Merge candidate sources into long DataFrame.

    sources = {"history": {uid: {iid: score}}, "covis": ..., "pop": ..., "content": ...}
    """
    rows = []
    all_users: set = set()
    for src_cands in sources.values():
        all_users.update(src_cands.keys())
    for uid in all_users:
        merged: dict[str, dict[str, float]] = defaultdict(dict)
        for src_name, src_cands in sources.items():
            for iid, sc in src_cands.get(uid, {}).items():
                merged[iid][f"src_{src_name}"] = float(sc)
        if not merged:
            continue
        items = list(merged.items())
        if len(items) > cap_total:
            items.sort(key=lambda x: -sum(x[1].values()))
            items = items[:cap_total]
        for iid, scores in items:
            row = {"user_id": uid, "item_id": iid,
                   "src_history": np.nan, "src_covis": np.nan,
                   "src_pop": np.nan, "src_content": np.nan}
            row.update(scores)
            rows.append(row)
    return pd.DataFrame(rows)
