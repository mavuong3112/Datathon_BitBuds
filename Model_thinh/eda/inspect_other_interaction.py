import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, pandas as pd

conn = duckdb.connect()
conn.execute("SET memory_limit='20GB'")
conn.execute("SET threads=4")
conn.execute("SET enable_progress_bar=false")

files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/fact_user_events/*.parquet')]

def q(sql): return conn.execute(sql).df()

sep = "\n" + "="*60 + "\n"

# ── 1. surface ─────────────────────────────────────────────────
print(sep + "1. SURFACE DISTRIBUTION")
print(q(f"""
    SELECT surface,
           COUNT(*) AS cnt,
           ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),2) AS pct
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
    GROUP BY surface ORDER BY cnt DESC
""").to_string(index=False))

# ── 2. device ─────────────────────────────────────────────────
print(sep + "2. DEVICE DISTRIBUTION")
print(q(f"""
    SELECT device,
           COUNT(*) AS cnt,
           ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),2) AS pct
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
    GROUP BY device ORDER BY cnt DESC
""").to_string(index=False))

# ── 3. is_login split ─────────────────────────────────────────
print(sep + "3. LOGIN vs NON-LOGIN")
print(q(f"""
    SELECT is_login,
           COUNT(*) AS cnt,
           ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),2) AS pct
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
    GROUP BY is_login ORDER BY cnt DESC
""").to_string(index=False))

# ── 4. category breakdown ─────────────────────────────────────
print(sep + "4. CATEGORY BREAKDOWN")
print(q(f"""
    SELECT category,
           COUNT(*) AS cnt,
           ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),2) AS pct
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
      AND category IN (1010,1020,1030,1040,1050)
    GROUP BY category ORDER BY cnt DESC
""").to_string(index=False))

# ── 5. position stats ─────────────────────────────────────────
print(sep + "5. POSITION STATS (non-null)")
print(q(f"""
    SELECT
        COUNT(*) AS n_with_position,
        ROUND(AVG(position),1) AS mean_pos,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY position) AS p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY position) AS p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY position) AS p75,
        MAX(position) AS max_pos
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
      AND position IS NOT NULL
""").to_string(index=False))

# ── 6. dwell_time stats ───────────────────────────────────────
print(sep + "6. DWELL_TIME_SEC STATS (non-null, raw value)")
print(q(f"""
    SELECT
        COUNT(*) AS n_with_dwell,
        ROUND(AVG(dwell_time_sec)/1000,1) AS mean_sec,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY dwell_time_sec)/1000,1) AS p25_sec,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY dwell_time_sec)/1000,1) AS p50_sec,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY dwell_time_sec)/1000,1) AS p75_sec,
        ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY dwell_time_sec)/1000,1) AS p90_sec
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
      AND dwell_time_sec IS NOT NULL AND dwell_time_sec > 0
""").to_string(index=False))

# ── 7. null rates per column ──────────────────────────────────
print(sep + "7. NULL RATES PER COLUMN (other_interaction rows only)")
print(q(f"""
    SELECT
        COUNT(*) AS total_rows,
        ROUND(SUM(CASE WHEN query          IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS null_query_pct,
        ROUND(SUM(CASE WHEN position       IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS null_position_pct,
        ROUND(SUM(CASE WHEN dwell_time_sec IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS null_dwell_pct,
        ROUND(SUM(CASE WHEN item_id        IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS null_item_pct
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
""").to_string(index=False))

# ── 8. co-occurrence: other_interaction before/after positive ─
print(sep + "8. SESSIONS WITH other_interaction: do they also have high-intent events?")
print(q(f"""
    SELECT
        has_phone,
        has_chat,
        has_zalo,
        COUNT(DISTINCT session_id) AS n_sessions
    FROM (
        SELECT session_id,
               MAX(CASE WHEN event_type='view_phone'   THEN 1 ELSE 0 END) AS has_phone,
               MAX(CASE WHEN event_type='contact_chat' THEN 1 ELSE 0 END) AS has_chat,
               MAX(CASE WHEN event_type='contact_zalo' THEN 1 ELSE 0 END) AS has_zalo
        FROM read_parquet({files})
        WHERE is_login = 'login'
          AND session_id IN (
              SELECT DISTINCT session_id FROM read_parquet({files})
              WHERE event_type = 'other_interaction' AND is_login='login'
          )
        GROUP BY session_id
    )
    GROUP BY has_phone, has_chat, has_zalo
    ORDER BY n_sessions DESC
    LIMIT 8
""").to_string(index=False))

# ── 9. query field population ─────────────────────────────────
print(sep + "9. QUERY FIELD — populated rows sample (search-sourced events)")
sample = q(f"""
    SELECT query, category, surface, device
    FROM read_parquet({files})
    WHERE event_type = 'other_interaction'
      AND query IS NOT NULL
    LIMIT 10
""")
print(sample.to_string(index=False))

conn.close()
print("\nDone.")
