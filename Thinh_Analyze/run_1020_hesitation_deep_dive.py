"""
1020 — delayed confidence / reassurance market (follow-up probes).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1]
WORKDIR = Path(__file__).resolve().parent
CAT = 1020

FILTERED = WORKDIR / "filtered_events.parquet"
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
SNAP_GLOB = str(DATA_ROOT / "fact_listing_snapshot" / "*.parquet")
EXPLICIT_SQL = ", ".join(
    repr(x) for x in ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
)

OUT = WORKDIR / "outputs" / "demand_side_1020" / "1020_deep_dive"
OUT.mkdir(parents=True, exist_ok=True)

use_filtered = FILTERED.exists()
if use_filtered:
    ev_from = f"read_parquet('{FILTERED.as_posix()}')"
    ev_where = f"category = {CAT}"
else:
    ev_from = f"read_parquet('{EVENTS_GLOB}')"
    ev_where = f"category = {CAT} AND is_login = 'login'"

con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=4; PRAGMA memory_limit='5GB'")
print("Events:", "filtered" if use_filtered else "raw")

# 1. Time-to-first-contact
ttc = con.execute(
    f"""
    WITH ev AS (
        SELECT user_id, event_ts,
            CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END AS is_explicit,
            CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END AS is_pv
        FROM {ev_from} WHERE {ev_where}
    ),
    bounds AS (
        SELECT user_id,
            MIN(CASE WHEN is_pv = 1 THEN event_ts END) AS first_pv_ts,
            MIN(CASE WHEN is_explicit = 1 THEN event_ts END) AS first_contact_ts
        FROM ev GROUP BY 1
        HAVING first_pv_ts IS NOT NULL AND first_contact_ts IS NOT NULL
           AND first_contact_ts >= first_pv_ts
    ),
    lag AS (
        SELECT date_diff('day', CAST(first_pv_ts AS DATE), CAST(first_contact_ts AS DATE)) AS days_to_contact
        FROM bounds
    )
    SELECT
        COUNT(*)::BIGINT AS users,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY days_to_contact), 1) AS p50_days,
        ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY days_to_contact), 1) AS p75_days,
        ROUND(percentile_cont(0.90) WITHIN GROUP (ORDER BY days_to_contact), 1) AS p90_days,
        ROUND(avg(days_to_contact), 2) AS mean_days,
        ROUND(100.0 * SUM(CASE WHEN days_to_contact = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_same_day,
        ROUND(100.0 * SUM(CASE WHEN days_to_contact >= 2 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_2plus_days,
        ROUND(100.0 * SUM(CASE WHEN days_to_contact >= 3 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_3plus_days
    FROM lag
    """
).df()
ttc.to_csv(OUT / "01_time_to_first_contact_days.csv", index=False)
print("\n=== 1. Time-to-first-contact ===\n", ttc.to_string(index=False))

# 2. Listing age at first contact
fresh = con.execute(
    f"""
    WITH first_contact AS (
        SELECT user_id, CAST(item_id AS VARCHAR) AS item_id,
            MIN(CAST(date AS DATE)) AS contact_date
        FROM {ev_from}
        WHERE {ev_where} AND event_type IN ({EXPLICIT_SQL}) AND item_id IS NOT NULL
        GROUP BY 1, 2
    ),
    joined AS (
        SELECT fc.*, s.listing_age_days,
            CASE
                WHEN s.listing_age_days IS NULL THEN 'unknown'
                WHEN s.listing_age_days <= 3 THEN '0-3d fresh'
                WHEN s.listing_age_days <= 14 THEN '4-14d'
                WHEN s.listing_age_days <= 30 THEN '15-30d'
                ELSE '31d+'
            END AS age_bucket
        FROM first_contact fc
        LEFT JOIN read_parquet('{SNAP_GLOB}') s
            ON fc.item_id = CAST(s.item_id AS VARCHAR) AND fc.contact_date = CAST(s.date AS DATE)
    )
    SELECT age_bucket, COUNT(*)::BIGINT AS contacts,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM joined GROUP BY 1 ORDER BY contacts DESC
    """
).df()
fresh.to_csv(OUT / "02_contact_by_listing_age.csv", index=False)
print("\n=== 2. Listing age at contact ===\n", fresh.to_string(index=False))

# 3. Repeat PV same listing before contact
repeat_pv = con.execute(
    f"""
    WITH ev AS (
        SELECT user_id, CAST(item_id AS VARCHAR) AS item_id, event_ts, event_type
        FROM {ev_from} WHERE {ev_where}
    ),
    fc AS (
        SELECT user_id, item_id, MIN(event_ts) AS t0
        FROM ev WHERE event_type IN ({EXPLICIT_SQL}) GROUP BY 1, 2
    ),
    pv_before AS (
        SELECT e.user_id, e.item_id, COUNT(*)::BIGINT AS n_pv
        FROM ev e JOIN fc f USING (user_id, item_id)
        WHERE e.event_type = 'pageview' AND e.event_ts < f.t0
        GROUP BY 1, 2
    )
    SELECT COUNT(*)::BIGINT AS pairs,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY n_pv), 1) AS median_pv,
        ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY n_pv), 1) AS p75_pv,
        ROUND(100.0 * SUM(CASE WHEN n_pv >= 2 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_2plus_pv,
        ROUND(100.0 * SUM(CASE WHEN n_pv >= 3 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_3plus_pv
    FROM pv_before
    """
).df()
repeat_pv.to_csv(OUT / "03_repeat_pv_same_listing.csv", index=False)
print("\n=== 3. Repeat PV same listing ===\n", repeat_pv.to_string(index=False))

# 4–5. let vs sell
split = con.execute(
    f"""
    WITH ev AS (
        SELECT e.user_id, e.session_id, CAST(e.item_id AS VARCHAR) AS item_id,
            e.date, e.event_ts, e.event_type,
            CASE WHEN e.event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END AS is_explicit,
            CASE WHEN e.event_type = 'pageview' THEN 1 ELSE 0 END AS is_pv
        FROM {ev_from} e WHERE {ev_where.replace('category', 'e.category')}
    ),
    ev_ad AS (
        SELECT ev.*, COALESCE(d.ad_type, 'unknown') AS ad_type
        FROM ev
        LEFT JOIN read_parquet('{DIM_GLOB}') d
            ON ev.item_id = CAST(d.item_id AS VARCHAR) AND d.category = {CAT}
    ),
    fc AS (
        SELECT ad_type, user_id, MIN(event_ts) AS t0
        FROM ev_ad WHERE is_explicit = 1 GROUP BY 1, 2
    ),
    per_user AS (
        SELECT e.ad_type, e.user_id,
            COUNT(DISTINCT CASE WHEN e.is_explicit = 1 THEN e.item_id END)::BIGINT AS contacted,
            COUNT(DISTINCT CASE WHEN e.is_pv = 1 AND e.event_ts < f.t0 THEN e.date END)::BIGINT AS pv_days_before
        FROM ev_ad e
        JOIN fc f ON e.ad_type = f.ad_type AND e.user_id = f.user_id
        GROUP BY 1, 2
    ),
    sess AS (
        WITH ss AS (
            SELECT ad_type, user_id, session_id, MIN(event_ts) AS st
            FROM ev_ad WHERE session_id IS NOT NULL GROUP BY 1, 2, 3
        )
        SELECT s.ad_type, s.user_id,
            SUM(CASE WHEN s.st < f.t0 THEN 1 ELSE 0 END)::BIGINT AS sessions_before
        FROM ss s JOIN fc f ON s.ad_type = f.ad_type AND s.user_id = f.user_id
        GROUP BY 1, 2
    )
    SELECT p.ad_type, COUNT(*)::BIGINT AS converting_users,
        ROUND(median(p.contacted), 1) AS median_contacted,
        ROUND(avg(p.contacted), 2) AS mean_contacted,
        ROUND(median(s.sessions_before), 1) AS median_sessions_before,
        ROUND(100.0 * SUM(CASE WHEN p.pv_days_before >= 2 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_2plus_pv_days
    FROM per_user p
    LEFT JOIN sess s ON p.ad_type = s.ad_type AND p.user_id = s.user_id
    GROUP BY p.ad_type ORDER BY converting_users DESC
    """
).df()
split.to_csv(OUT / "04_metrics_by_ad_type.csv", index=False)
print("\n=== 4–5. let vs sell ===\n", split.to_string(index=False))

con.close()
print(f"\n→ {OUT}")
