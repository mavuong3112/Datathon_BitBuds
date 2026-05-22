"""User-side / item-side feature aggregators + price parser."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


HIGH_INTENT = {"view_phone", "contact_chat", "contact_zalo", "contact_sms"}


def build_user_features(events_pos: pd.DataFrame, events_pv: pd.DataFrame,
                        cutoff_ts: pd.Timestamp,
                        dim_enriched: pd.DataFrame | None = None) -> pd.DataFrame:
    """events_pos: positive events (5 types). events_pv: pageview last 30d.
    dim_enriched: optional, used to compute u_price_pref_log (median price of interacted items).
    """
    e = events_pos[events_pos["event_ts"] < cutoff_ts].copy()
    e["_age"] = (cutoff_ts - e["event_ts"]).dt.total_seconds() / 86400.0
    e["_hi"] = e["event_type"].isin(HIGH_INTENT)

    agg_specs = {
        "u_n_pos": ("item_id", "count"),
        "u_n_pos_7d": ("_age", lambda x: int((x <= 7).sum())),
        "u_n_pos_30d": ("_age", lambda x: int((x <= 30).sum())),
        "u_n_high_intent_30d": ("_hi", lambda x: int(x[e.loc[x.index, "_age"] <= 30].sum())),
        "u_recency_days": ("_age", "min"),
        "u_n_unique_items": ("item_id", "nunique"),
    }
    if "session_id" in e.columns:
        agg_specs["u_n_sessions"] = ("session_id", "nunique")
    if "is_login" in e.columns:
        agg_specs["u_is_login_majority"] = ("is_login", lambda x: (x == "login").mean() > 0.5)
    agg = e.groupby("user_id").agg(**agg_specs).reset_index()
    agg["u_high_intent_ratio"] = (
        agg["u_n_high_intent_30d"] / agg["u_n_pos_30d"].clip(lower=1)
    )

    def _mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else None

    mode_specs = {}
    for src_col, out_col in [("category", "u_top_category"),
                              ("city_name", "u_top_city"),
                              ("surface", "u_main_surface"),
                              ("device", "u_main_device")]:
        if src_col in e.columns:
            mode_specs[out_col] = (src_col, _mode)
    if mode_specs:
        mode_cols = e.groupby("user_id").agg(**mode_specs).reset_index()
        agg = agg.merge(mode_cols, on="user_id", how="left")

    if "dwell_time_sec" in events_pv.columns:
        pv = events_pv[events_pv["event_ts"] < cutoff_ts]
        pv_agg = pv.groupby("user_id").agg(
            u_n_pageview_30d=("item_id", "count"),
            u_avg_dwell_pageview=("dwell_time_sec", "mean"),
        ).reset_index()
        agg = agg.merge(pv_agg, on="user_id", how="left")

    # Price preference: median log-price of items user has interacted with
    if dim_enriched is not None and "price_bucket" in dim_enriched.columns:
        price_map = dim_enriched.set_index("item_id")["price_bucket"].map(parse_price_to_log)
        e2 = e.copy()
        e2["_price_log"] = e2["item_id"].map(price_map)
        price_pref = (
            e2.dropna(subset=["_price_log"])
            .groupby("user_id")["_price_log"]
            .median()
            .rename("u_price_pref_log")
            .reset_index()
        )
        agg = agg.merge(price_pref, on="user_id", how="left")

    return agg


def build_item_features(events_pos: pd.DataFrame, snap_60d: pd.DataFrame,
                         dim_enriched: pd.DataFrame,
                         cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    e = events_pos[events_pos["event_ts"] < cutoff_ts].copy()
    e["_age"] = (cutoff_ts - e["event_ts"]).dt.total_seconds() / 86400.0
    agg = e.groupby("item_id").agg(
        i_n_pos_7d=("_age", lambda x: int((x <= 7).sum())),
        i_n_pos_30d=("_age", lambda x: int((x <= 30).sum())),
        i_n_unique_users_train=("user_id", "nunique"),
    ).reset_index()

    snap = snap_60d.copy()
    snap["_age"] = (cutoff_ts - pd.to_datetime(snap["date"])).dt.total_seconds() / 86400.0
    s7 = snap[snap["_age"] <= 7].groupby("item_id").agg(
        i_views_24h_mean_7d=("views_24h", "mean"),
        i_contacts_24h_mean_7d=("contacts_24h", "mean"),
    )
    s30 = snap[snap["_age"] <= 30].groupby("item_id").agg(
        i_views_24h_mean_30d=("views_24h", "mean"),
        i_contacts_24h_mean_30d=("contacts_24h", "mean"),
    )
    snap_agg = s7.join(s30, how="outer").reset_index()
    snap_agg["i_CR_30d"] = (
        snap_agg["i_contacts_24h_mean_30d"] /
        snap_agg["i_views_24h_mean_30d"].clip(lower=0.01)
    )
    # Velocity: relative change 7d vs 30d avg (positive = trending up)
    snap_agg["i_velocity_contacts"] = (
        (snap_agg["i_contacts_24h_mean_7d"] - snap_agg["i_contacts_24h_mean_30d"]) /
        snap_agg["i_contacts_24h_mean_30d"].clip(lower=0.01)
    )
    snap_agg["i_velocity_views"] = (
        (snap_agg["i_views_24h_mean_7d"] - snap_agg["i_views_24h_mean_30d"]) /
        snap_agg["i_views_24h_mean_30d"].clip(lower=0.01)
    )

    out = dim_enriched.merge(agg, on="item_id", how="left")
    out = out.merge(snap_agg, on="item_id", how="left")
    out["i_price_log"] = out["price_bucket"].map(parse_price_to_log) if "price_bucket" in out.columns else np.nan
    out["i_evt_density"] = out["n_pos_train"] / out["n_snap_days"].clip(lower=1)

    # Price deviation vs segment median (z-score within category × city)
    if "i_price_log" in out.columns and "category" in out.columns and "city_name" in out.columns:
        seg_med = out.groupby(["category", "city_name"])["i_price_log"].transform("median")
        seg_std = out.groupby(["category", "city_name"])["i_price_log"].transform("std").clip(lower=0.1)
        out["i_price_deviation_seg"] = (out["i_price_log"] - seg_med) / seg_std

    out = out.rename(columns={
        "category": "i_category", "ad_type": "i_ad_type",
        "seller_type": "i_seller_type", "tier": "i_tier",
    })
    return out


_price_pat_billion = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:tỷ|ty|billion)", re.I)
_price_pat_million = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:triệu|tr|million)", re.I)


def parse_price_to_log(bucket) -> float:
    if pd.isna(bucket):
        return np.nan
    s = str(bucket).lower()
    val = 0.0
    m = _price_pat_billion.search(s)
    if m:
        try:
            val += float(m.group(1).replace(",", ".")) * 1_000_000_000
        except ValueError:
            pass
    m = _price_pat_million.search(s)
    if m:
        try:
            val += float(m.group(1).replace(",", ".")) * 1_000_000
        except ValueError:
            pass
    if val <= 0:
        nums = re.findall(r"\d+", s.replace(",", ""))
        if nums:
            try:
                val = float(nums[0])
            except ValueError:
                return np.nan
    return float(np.log1p(val)) if val > 0 else np.nan


def add_cross_features(cands: pd.DataFrame, user_feats: pd.DataFrame,
                        item_feats: pd.DataFrame) -> pd.DataFrame:
    df = cands.merge(user_feats, on="user_id", how="left")
    df = df.merge(item_feats, on="item_id", how="left")
    if "u_top_category" in df.columns and "i_category" in df.columns:
        df["same_category"] = (df["u_top_category"] == df["i_category"]).astype("int8")
    if "u_top_city" in df.columns and "city_name" in df.columns:
        df["same_city"] = (df["u_top_city"] == df["city_name"]).astype("int8")
    if "u_top_ad_type" in df.columns and "i_ad_type" in df.columns:
        df["same_ad_type"] = (df["u_top_ad_type"] == df["i_ad_type"]).astype("int8")
    # Price affinity: how close item price is to user's historical price preference
    if "u_price_pref_log" in df.columns and "i_price_log" in df.columns:
        df["x_price_affinity"] = -(df["i_price_log"] - df["u_price_pref_log"]).abs()
    return df
