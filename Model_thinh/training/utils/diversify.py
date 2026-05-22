"""Post-rank diversification: cap per seller / district, freshness boost."""
from __future__ import annotations

import pandas as pd


def diversify_top_k(scored: pd.DataFrame, dim_lookup: pd.DataFrame,
                    k: int = 10, max_per_seller: int = 7,
                    max_per_district: int = 8,
                    freshness_boost: float = 0.05,
                    fresh_age_days: int = 7) -> pd.DataFrame:
    """scored: cols [user_id, item_id, score], sorted by user, score desc.
    dim_lookup: cols [item_id, seller_id, district_name, i_listing_age_days_latest]
    """
    df = scored.merge(dim_lookup, on="item_id", how="left")
    if "i_listing_age_days_latest" in df.columns:
        df["score"] = df["score"] + (
            (df["i_listing_age_days_latest"].fillna(999) <= fresh_age_days) * freshness_boost
        )
    df = df.sort_values(["user_id", "score"], ascending=[True, False])

    out_rows = []
    for uid, grp in df.groupby("user_id", sort=False):
        seller_count: dict = {}
        district_count: dict = {}
        picked = 0
        for _, row in grp.iterrows():
            seller = row.get("seller_id")
            district = row.get("district_name")
            if seller_count.get(seller, 0) >= max_per_seller and seller is not None:
                continue
            if district_count.get(district, 0) >= max_per_district and district is not None:
                continue
            seller_count[seller] = seller_count.get(seller, 0) + 1
            district_count[district] = district_count.get(district, 0) + 1
            out_rows.append({
                "user_id": uid, "item_id": row["item_id"],
                "rank": picked + 1, "score": row["score"],
            })
            picked += 1
            if picked >= k:
                break

        if picked < k:
            picked_items = {r["item_id"] for r in out_rows if r["user_id"] == uid}
            for _, row in grp.iterrows():
                if row["item_id"] in picked_items:
                    continue
                out_rows.append({
                    "user_id": uid, "item_id": row["item_id"],
                    "rank": picked + 1, "score": row["score"],
                })
                picked_items.add(row["item_id"])
                picked += 1
                if picked >= k:
                    break

    return pd.DataFrame(out_rows)
