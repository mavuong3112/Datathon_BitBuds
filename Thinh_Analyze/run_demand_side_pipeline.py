"""
Demand-side concentration — SYSTEM sample 5% trên fact_user_events.

Median / mean / % ổn định với ~5% mẫu; converting_users_sample chỉ là quy mô mẫu (×20 ≈ ước lượng).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1]
WORKDIR = Path(__file__).resolve().parent
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
EXPLICIT_SQL = ", ".join(
    repr(x) for x in ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
)
OUT = WORKDIR / "outputs" / "demand_side_1020"
OUT.mkdir(parents=True, exist_ok=True)

# 5% SYSTEM — đổi thành 10 hoặc 20 nếu muốn mượt hơn
SAMPLE_PCT = 5

con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=4")

print(f"Đang quét ~{SAMPLE_PCT}% dữ liệu (SYSTEM sample)…", flush=True)

query = f"""
CREATE TEMP TABLE base_events AS
SELECT
    category,
    user_id,
    session_id,
    item_id,
    date,
    event_ts,
    event_type,
    CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END AS is_explicit,
    CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END AS is_pv,
    CASE
        WHEN event_type = 'pageview'
         AND query IS NOT NULL
         AND trim(CAST(query AS VARCHAR)) <> ''
        THEN regexp_replace(lower(trim(CAST(query AS VARCHAR))), '\\s+', ' ', 'g')
        ELSE NULL
    END AS qnorm
FROM read_parquet('{EVENTS_GLOB}')
TABLESAMPLE {SAMPLE_PCT} PERCENT (SYSTEM)
WHERE is_login = 'login'
  AND category IN (1010, 1020, 1030, 1040, 1050);

CREATE TEMP TABLE user_first_contact AS
SELECT
    category,
    user_id,
    MIN(event_ts) AS t0,
    COUNT(DISTINCT item_id)::BIGINT AS contacted_listings
FROM base_events
WHERE is_explicit = 1 AND item_id IS NOT NULL
GROUP BY category, user_id;

CREATE TEMP TABLE user_sessions AS
WITH session_starts AS (
    SELECT category, user_id, session_id, MIN(event_ts) AS st
    FROM base_events
    WHERE session_id IS NOT NULL
    GROUP BY category, user_id, session_id
)
SELECT
    s.category,
    s.user_id,
    SUM(CASE WHEN s.st < f.t0 THEN 1 ELSE 0 END)::BIGINT AS sessions_before_contact,
    COUNT(s.session_id)::BIGINT AS total_sessions
FROM session_starts s
JOIN user_first_contact f USING (category, user_id)
GROUP BY s.category, s.user_id;

CREATE TEMP TABLE user_pv_days AS
SELECT
    e.category,
    e.user_id,
    COUNT(DISTINCT e.date)::BIGINT AS pv_days_before
FROM base_events e
JOIN user_first_contact f USING (category, user_id)
WHERE e.is_pv = 1 AND e.event_ts < f.t0
GROUP BY e.category, e.user_id;

CREATE TEMP TABLE all_searches AS
SELECT
    category,
    user_id,
    session_id,
    event_ts,
    qnorm,
    LAG(qnorm) OVER (
        PARTITION BY category, user_id, session_id ORDER BY event_ts
    ) AS pq
FROM base_events
WHERE qnorm IS NOT NULL;

CREATE TEMP TABLE session_refinements AS
SELECT
    s.category,
    s.user_id,
    s.session_id,
    SUM(CASE WHEN s.pq IS NOT NULL AND s.qnorm <> s.pq THEN 1 ELSE 0 END)::BIGINT AS refinements
FROM all_searches s
JOIN user_first_contact f USING (category, user_id)
GROUP BY s.category, s.user_id, s.session_id;

CREATE TEMP TABLE user_refinements_before AS
SELECT
    s.category,
    s.user_id,
    SUM(CASE WHEN s.pq IS NOT NULL AND s.qnorm <> s.pq THEN 1 ELSE 0 END)::BIGINT AS total_refinements
FROM all_searches s
JOIN user_first_contact f USING (category, user_id)
WHERE s.event_ts < f.t0
GROUP BY s.category, s.user_id;

WITH cat_user_metrics AS (
    SELECT
        c.category,
        COUNT(c.user_id)::BIGINT AS converting_users_sample,
        median(c.contacted_listings) AS median_contacted_listings,
        quantile_cont(c.contacted_listings, 0.75) AS p75_contacted_listings,
        quantile_cont(c.contacted_listings, 0.90) AS p90_contacted_listings,
        avg(c.contacted_listings) AS mean_contacted_listings,
        median(us.sessions_before_contact) AS median_sessions_before_contact,
        avg(us.sessions_before_contact) AS mean_sessions_before_contact,
        median(us.total_sessions) AS median_total_sessions_to_convert,
        avg(CASE WHEN pd.pv_days_before >= 2 THEN 100.0 ELSE 0.0 END) AS pct_users_repeat_daily_2plus_pv_days,
        median(pd.pv_days_before) AS median_pv_days_before_contact,
        median(urb.total_refinements) AS median_total_refinements_before_contact,
        avg(CASE WHEN urb.total_refinements >= 5 THEN 100.0 ELSE 0.0 END) AS pct_users_5plus_refinements_before_contact
    FROM user_first_contact c
    LEFT JOIN user_sessions us USING (category, user_id)
    LEFT JOIN user_pv_days pd USING (category, user_id)
    LEFT JOIN user_refinements_before urb USING (category, user_id)
    GROUP BY c.category
),
cat_session_metrics AS (
    SELECT
        category,
        median(refinements) AS median_search_refinements_per_session,
        avg(refinements) AS mean_search_refinements_per_session,
        avg(CASE WHEN refinements >= 3 THEN 100.0 ELSE 0.0 END) AS pct_search_sessions_with_3plus_refine
    FROM session_refinements
    GROUP BY category
)
SELECT
    u.*,
    s.median_search_refinements_per_session,
    s.mean_search_refinements_per_session,
    s.pct_search_sessions_with_3plus_refine
FROM cat_user_metrics u
LEFT JOIN cat_session_metrics s USING (category)
ORDER BY u.category;
"""

df = con.execute(query).df()

cols = [
    "category",
    "converting_users_sample",
    "median_contacted_listings",
    "p75_contacted_listings",
    "p90_contacted_listings",
    "mean_contacted_listings",
    "median_sessions_before_contact",
    "mean_sessions_before_contact",
    "median_total_sessions_to_convert",
    "pct_users_repeat_daily_2plus_pv_days",
    "median_pv_days_before_contact",
    "median_search_refinements_per_session",
    "mean_search_refinements_per_session",
    "pct_search_sessions_with_3plus_refine",
    "median_total_refinements_before_contact",
    "pct_users_5plus_refinements_before_contact",
]
df = df[cols].fillna(np.nan)
df["converting_users_est_full"] = (df["converting_users_sample"] * (100 / SAMPLE_PCT)).round(0)
df.to_csv(OUT / "01_demand_side_by_category_sampled.csv", index=False)

print(df.to_string(index=False))

metrics = [
    "median_contacted_listings",
    "median_sessions_before_contact",
    "pct_users_repeat_daily_2plus_pv_days",
    "median_total_refinements_before_contact",
    "pct_users_5plus_refinements_before_contact",
]
focus = df.set_index("category")
print(f"\n1020 rank (1 = cao nhất / ‘đào mỏ’ hơn) — sample {SAMPLE_PCT}% SYSTEM:")
for m in metrics:
    print(f"  {m}: {int(focus[m].rank(ascending=False).loc[1020])}/5")

con.close()
print(f"\nSaved → {OUT / '01_demand_side_by_category_sampled.csv'}")
