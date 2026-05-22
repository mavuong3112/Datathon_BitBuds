"""
Step 1: DuckDB extraction → parquet cache
Resume-safe: skips any output that already exists on disk.
Memory strategy: delete DataFrames + gc.collect() between steps,
                 restart DuckDB connection before each heavy query.
"""
import sys, glob, time, os, gc
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, pandas as pd
import numpy as np
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

def mem_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6

def fresh_conn():
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")
    c.execute("SET enable_progress_bar=false")
    c.execute("SET preserve_insertion_order=false")
    return c

dim_files  = [f.replace('\\','/') for f in glob.glob(f'{DIM_DIR}/*.parquet')]
evt_files  = [f.replace('\\','/') for f in glob.glob(f'{EVT_DIR}/*.parquet')]
inter_files= [f.replace('\\','/') for f in glob.glob(f'{INTER_DIR}/*.parquet')]
seq_tmp_dir = f"{CACHE_DIR}/seq_tmp"
os.makedirs(seq_tmp_dir, exist_ok=True)

print(f"{elapsed()} dim={len(dim_files)} evt={len(evt_files)} inter={len(inter_files)}")

# ── 1a: Positive interactions ─────────────────────────────────────────────────
POS_FILE = f"{CACHE_DIR}/user_item_pos.parquet"
if os.path.exists(POS_FILE):
    print(f"{elapsed()} [SKIP] user_item_pos.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Extracting positive interactions …")
    pos = conn.execute(f"""
        SELECT user_id, item_id, category, city_name,
               COUNT(*) AS pos_count,
               MAX(event_ts) AS last_ts,
               MIN(event_ts) AS first_ts,
               SUM(CASE WHEN event_type='view_phone'        THEN 1 ELSE 0 END) AS n_view_phone,
               SUM(CASE WHEN event_type='contact_chat'      THEN 1 ELSE 0 END) AS n_chat,
               SUM(CASE WHEN event_type='contact_zalo'      THEN 1 ELSE 0 END) AS n_zalo,
               SUM(CASE WHEN event_type='contact_sms'       THEN 1 ELSE 0 END) AS n_sms,
               SUM(CASE WHEN event_type='other_interaction' THEN 1 ELSE 0 END) AS n_other
        FROM read_parquet({evt_files})
        WHERE event_type IN ({POS_STR})
          AND is_login = 'login'
          AND {CATEGORY_FILTER}
        GROUP BY user_id, item_id, category, city_name
    """).df()
    conn.close(); del conn
    print(f"{elapsed()} pos: {len(pos):,} rows, {pos['user_id'].nunique():,} users  [RAM:{mem_mb():.0f}MB]")
    pos.to_parquet(POS_FILE, index=False)

# ── 1b: Sequence batches ──────────────────────────────────────────────────────
SEQ_FILE  = f"{CACHE_DIR}/user_item_seq.parquet"
SEQ_BATCH = 50
n_batches = (len(evt_files) + SEQ_BATCH - 1) // SEQ_BATCH

if os.path.exists(SEQ_FILE):
    print(f"{elapsed()} [SKIP] user_item_seq.parquet already exists")
else:
    # Extract each batch independently with its own DuckDB connection
    for bi, batch_start in enumerate(range(0, len(evt_files), SEQ_BATCH)):
        tmp_path = f"{seq_tmp_dir}/seq_batch_{bi:03d}.parquet"
        if os.path.exists(tmp_path):
            print(f"{elapsed()} [SKIP] seq batch {bi+1}/{n_batches}")
            continue
        batch = evt_files[batch_start:batch_start+SEQ_BATCH]
        conn  = fresh_conn()
        bdf   = conn.execute(f"""
            SELECT user_id, item_id, category,
                   MAX(event_ts) AS event_ts
            FROM read_parquet({batch})
            WHERE is_login = 'login'
              AND {CATEGORY_FILTER}
            GROUP BY user_id, item_id, category
        """).df()
        conn.close(); del conn
        bdf.to_parquet(tmp_path, index=False)
        print(f"{elapsed()} seq batch {bi+1}/{n_batches}: {len(bdf):,} rows  [RAM:{mem_mb():.0f}MB]")
        del bdf; gc.collect()

    # Streaming merge: process one batch at a time
    print(f"{elapsed()} Streaming merge of seq batches …")
    tmp_files = sorted(glob.glob(f"{seq_tmp_dir}/seq_batch_*.parquet"))
    running = pd.read_parquet(tmp_files[0])
    for f in tmp_files[1:]:
        batch = pd.read_parquet(f)
        combined = pd.concat([running, batch], ignore_index=True)
        del running, batch
        running = (combined
                   .sort_values('event_ts', ascending=False)
                   .drop_duplicates(['user_id','item_id'])
                   .reset_index(drop=True))
        del combined; gc.collect()

    seq = (running.sort_values(['user_id','event_ts'])
                  .groupby('user_id').tail(50)
                  .reset_index(drop=True))
    del running; gc.collect()
    print(f"{elapsed()} seq final: {len(seq):,} rows, {seq['user_id'].nunique():,} users  [RAM:{mem_mb():.0f}MB]")
    seq.to_parquet(SEQ_FILE, index=False)
    del seq; gc.collect()

# ── 1c: fact_post_contact aggregates ─────────────────────────────────────────
INTER_FILE = f"{CACHE_DIR}/user_item_inter.parquet"
if os.path.exists(INTER_FILE):
    print(f"{elapsed()} [SKIP] user_item_inter.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Extracting fact_post_contact aggregates …  [RAM:{mem_mb():.0f}MB]")
    inter = conn.execute(f"""
        SELECT user_id, item_id, category,
               SUM(adview_count)       AS total_adviews,
               SUM(lead_count)         AS total_leads,
               SUM(chat_message_count) AS total_chat_msgs,
               SUM(chat_turn_count)    AS total_chat_turns,
               MAX(CASE WHEN purchased THEN 1 ELSE 0 END) AS ever_purchased,
               COUNT(DISTINCT date)    AS active_days_inter
        FROM read_parquet({inter_files})
        WHERE {CATEGORY_FILTER}
        GROUP BY user_id, item_id, category
    """).df()
    conn.close(); del conn
    print(f"{elapsed()} inter: {len(inter):,} rows  [RAM:{mem_mb():.0f}MB]")
    inter.to_parquet(INTER_FILE, index=False)
    del inter; gc.collect()

# ── 1d: Item catalog ──────────────────────────────────────────────────────────
ITEMS_FILE = f"{CACHE_DIR}/items.parquet"
if os.path.exists(ITEMS_FILE):
    print(f"{elapsed()} [SKIP] items.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Extracting item catalog …")
    items = conn.execute(f"""
        SELECT item_id, category, ad_type, seller_type,
               area_sqm, bedrooms, bathrooms, images_count,
               city_name, district_name, ward_name,
               price_bucket, direction, legal_status, furnishing,
               project_id, posted_date, expected_expired_date
        FROM read_parquet({dim_files})
        WHERE {CATEGORY_FILTER}
    """).df()
    conn.close(); del conn
    print(f"{elapsed()} items: {len(items):,} rows")
    items.to_parquet(ITEMS_FILE, index=False)
    del items; gc.collect()

# ── 1e: Item quality stats ────────────────────────────────────────────────────
IQ_FILE = f"{CACHE_DIR}/item_quality.parquet"
if os.path.exists(IQ_FILE):
    print(f"{elapsed()} [SKIP] item_quality.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Extracting item quality stats …  [RAM:{mem_mb():.0f}MB]")
    item_qual = conn.execute(f"""
        SELECT item_id, category,
               COUNT(*) AS total_events,
               SUM(CASE WHEN event_type IN ({POS_STR}) THEN 1 ELSE 0 END) AS pos_events,
               SUM(CASE WHEN event_type='pageview' THEN 1 ELSE 0 END) AS pageviews,
               APPROX_COUNT_DISTINCT(user_id) AS unique_users,
               APPROX_COUNT_DISTINCT(session_id) AS unique_sessions,
               AVG(CASE WHEN dwell_time_sec BETWEEN 1 AND 3600 THEN dwell_time_sec END) AS avg_dwell
        FROM read_parquet({evt_files})
        WHERE {CATEGORY_FILTER}
        GROUP BY item_id, category
    """).df()
    conn.close(); del conn
    if item_qual['avg_dwell'].median() > 1000:
        item_qual['avg_dwell'] /= 1000
    item_qual['item_cvr'] = item_qual['pos_events'] / item_qual['total_events'].clip(lower=1)
    print(f"{elapsed()} item_quality: {len(item_qual):,} items")
    item_qual.to_parquet(IQ_FILE, index=False)
    del item_qual; gc.collect()

# ── 1f: Trending items (last 28 days) ────────────────────────────────────────
POP_FILE = f"{CACHE_DIR}/popular_items.parquet"
if os.path.exists(POP_FILE):
    print(f"{elapsed()} [SKIP] popular_items.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Extracting trending items …  [RAM:{mem_mb():.0f}MB]")
    popular = conn.execute(f"""
        SELECT item_id, category, city_name,
               COUNT(*) AS trend_events,
               SUM(CASE WHEN event_type IN ({POS_STR}) THEN 1 ELSE 0 END) AS trend_pos
        FROM read_parquet({evt_files})
        WHERE {CATEGORY_FILTER}
          AND date >= DATE '{TRAIN_END}'::DATE - INTERVAL 28 DAY
        GROUP BY item_id, category, city_name
        ORDER BY trend_pos DESC
    """).df()
    conn.close(); del conn
    print(f"{elapsed()} popular: {len(popular):,} rows")
    popular.to_parquet(POP_FILE, index=False)
    del popular; gc.collect()

# ── 1g: User profiles (from pos — reload from disk to avoid stale ref) ────────
PROF_FILE = f"{CACHE_DIR}/user_profiles.parquet"
if os.path.exists(PROF_FILE):
    print(f"{elapsed()} [SKIP] user_profiles.parquet already exists")
else:
    print(f"{elapsed()} Building user profiles …  [RAM:{mem_mb():.0f}MB]")
    pos = pd.read_parquet(POS_FILE)

    pref_cat  = (pos.groupby(['user_id','category'])['pos_count'].sum()
                   .reset_index()
                   .sort_values('pos_count', ascending=False)
                   .drop_duplicates('user_id')
                   .rename(columns={'category':'pref_category','pos_count':'pref_cat_score'}))
    pref_city = (pos.groupby(['user_id','city_name'])['pos_count'].sum()
                   .reset_index()
                   .sort_values('pos_count', ascending=False)
                   .drop_duplicates('user_id')
                   .rename(columns={'city_name':'pref_city','pos_count':'pref_city_score'}))

    train_end_dt = pd.Timestamp(TRAIN_END)
    pos['last_ts']  = pd.to_datetime(pos['last_ts'])
    pos['first_ts'] = pd.to_datetime(pos['first_ts'])

    user_agg = (pos.groupby('user_id').agg(
        total_pos_events = ('pos_count','sum'),
        unique_items     = ('item_id','nunique'),
        last_activity    = ('last_ts','max'),
        first_activity   = ('first_ts','min'),
    ).reset_index())
    user_agg['days_since_last']  = (train_end_dt - user_agg['last_activity']).dt.days
    user_agg['active_span_days'] = (user_agg['last_activity'] - user_agg['first_activity']).dt.days.clip(lower=1)

    profiles = (user_agg
        .merge(pref_cat,  on='user_id', how='left')
        .merge(pref_city, on='user_id', how='left'))
    print(f"{elapsed()} profiles: {len(profiles):,} users")
    profiles.to_parquet(PROF_FILE, index=False)
    del pos, pref_cat, pref_city, user_agg, profiles; gc.collect()

# ── Stage 8 outputs: _train versions with Tết filter (Feb 14-22, 2026) + < VAL_SPLIT ─

TET_EXCLUDE = "AND NOT (event_ts >= '2026-02-14' AND event_ts < '2026-02-23')"
TET_EXCLUDE_DATE = "AND NOT (date >= '2026-02-14' AND date < '2026-02-23')"

# Stage 8.1: pos_train — events < VAL_SPLIT, Tết-filtered (for retrievers_train + features_train)
POS_TRAIN_FILE = f"{CACHE_DIR}/user_item_pos_train.parquet"
if os.path.exists(POS_TRAIN_FILE):
    print(f"{elapsed()} [SKIP] user_item_pos_train.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Stage 8: Extracting pos_train (< VAL_SPLIT, Tết filtered) …")
    pos = conn.execute(f"""
        SELECT user_id, item_id, category, city_name,
               COUNT(*) AS pos_count,
               MAX(event_ts) AS last_ts,
               MIN(event_ts) AS first_ts,
               SUM(CASE WHEN event_type='view_phone'        THEN 1 ELSE 0 END) AS n_view_phone,
               SUM(CASE WHEN event_type='contact_chat'      THEN 1 ELSE 0 END) AS n_chat,
               SUM(CASE WHEN event_type='contact_zalo'      THEN 1 ELSE 0 END) AS n_zalo,
               SUM(CASE WHEN event_type='contact_sms'       THEN 1 ELSE 0 END) AS n_sms,
               SUM(CASE WHEN event_type='other_interaction' THEN 1 ELSE 0 END) AS n_other
        FROM read_parquet({evt_files})
        WHERE event_type IN ({POS_STR})
          AND is_login = 'login'
          AND {CATEGORY_FILTER}
          AND event_ts < '{VAL_SPLIT}'
          {TET_EXCLUDE}
        GROUP BY user_id, item_id, category, city_name
    """).df()
    conn.close(); del conn
    print(f"{elapsed()} pos_train: {len(pos):,} rows, {pos['user_id'].nunique():,} users  [RAM:{mem_mb():.0f}MB]")
    pos.to_parquet(POS_TRAIN_FILE, index=False)
    del pos; gc.collect()

# Stage 8.2: inter_train — date < VAL_SPLIT, Tết-filtered
INTER_TRAIN_FILE = f"{CACHE_DIR}/user_item_inter_train.parquet"
if os.path.exists(INTER_TRAIN_FILE):
    print(f"{elapsed()} [SKIP] user_item_inter_train.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Stage 8: Extracting inter_train (< VAL_SPLIT, Tết filtered) …")
    inter = conn.execute(f"""
        SELECT user_id, item_id, category,
               SUM(adview_count)       AS total_adviews,
               SUM(lead_count)         AS total_leads,
               SUM(chat_message_count) AS total_chat_msgs,
               SUM(chat_turn_count)    AS total_chat_turns,
               MAX(CASE WHEN purchased THEN 1 ELSE 0 END) AS ever_purchased,
               COUNT(DISTINCT date)    AS active_days_inter
        FROM read_parquet({inter_files})
        WHERE {CATEGORY_FILTER}
          AND date < '{VAL_SPLIT}'
          {TET_EXCLUDE_DATE}
        GROUP BY user_id, item_id, category
    """).df()
    conn.close(); del conn
    print(f"{elapsed()} inter_train: {len(inter):,} rows  [RAM:{mem_mb():.0f}MB]")
    inter.to_parquet(INTER_TRAIN_FILE, index=False)
    del inter; gc.collect()

# ── 1h: Positive events RAW (per-event, no aggregate) — Stage 2 ───────────────
# Unblocks Stage 3 features: time-decay history, same-project boost.
POS_RAW_FILE = f"{CACHE_DIR}/user_item_pos_events_raw.parquet"
if os.path.exists(POS_RAW_FILE):
    print(f"{elapsed()} [SKIP] user_item_pos_events_raw.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Extracting positive events RAW (per-event) …  [RAM:{mem_mb():.0f}MB]")
    # Stream to parquet directly via DuckDB COPY (avoids materializing huge DataFrame in RAM)
    conn.execute(f"""
        COPY (
            SELECT user_id, item_id, event_ts, event_type, session_id
            FROM read_parquet({evt_files})
            WHERE event_type IN ({POS_STR})
              AND is_login = 'login'
              AND {CATEGORY_FILTER}
              AND event_ts BETWEEN '{TRAIN_START}' AND '{TRAIN_END}'
        ) TO '{POS_RAW_FILE}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{POS_RAW_FILE}')").fetchone()[0]
    conn.close(); del conn
    print(f"{elapsed()} pos_events_raw: {n:,} rows  [RAM:{mem_mb():.0f}MB]")

# ── 1i: Dwell pair-level aggregates from pageviews — Stage 2 ──────────────────
# TEST version: full train window (correct for inference).
DWELL_FILE = f"{CACHE_DIR}/user_item_dwell_pair.parquet"
if os.path.exists(DWELL_FILE):
    print(f"{elapsed()} [SKIP] user_item_dwell_pair.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Aggregating dwell stats from pageviews (full train) …  [RAM:{mem_mb():.0f}MB]")
    conn.execute(f"""
        COPY (
            WITH pv AS (
                SELECT user_id, item_id,
                       CASE WHEN dwell_time_sec BETWEEN 1 AND 3600000
                            THEN dwell_time_sec ELSE NULL END AS dwell_clean
                FROM read_parquet({evt_files})
                WHERE event_type = 'pageview'
                  AND is_login = 'login'
                  AND {CATEGORY_FILTER}
                  AND event_ts BETWEEN '{TRAIN_START}' AND '{TRAIN_END}'
            )
            SELECT user_id, item_id,
                   COUNT(*) AS n_pageview,
                   MAX(dwell_clean) AS max_dwell_raw,
                   AVG(dwell_clean) AS avg_dwell_raw,
                   MAX(CASE WHEN dwell_clean > 30 OR dwell_clean > 30000 THEN 1 ELSE 0 END) AS has_consider_pv
            FROM pv
            GROUP BY user_id, item_id
        ) TO '{DWELL_FILE}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{DWELL_FILE}')").fetchone()[0]
    conn.close(); del conn
    print(f"{elapsed()} dwell_pair (test): {n:,} pairs  [RAM:{mem_mb():.0f}MB]")

# TRAIN version: events < VAL_SPLIT (leak-free for training labels in Mar-Apr).
# Fixes -0.0038 features-induced drop seen on submission_stage3_global.csv.
DWELL_TRAIN_FILE = f"{CACHE_DIR}/user_item_dwell_pair_train.parquet"
if os.path.exists(DWELL_TRAIN_FILE):
    print(f"{elapsed()} [SKIP] user_item_dwell_pair_train.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Aggregating dwell stats from pageviews (train — event_ts < {VAL_SPLIT}) …  [RAM:{mem_mb():.0f}MB]")
    conn.execute(f"""
        COPY (
            WITH pv AS (
                SELECT user_id, item_id,
                       CASE WHEN dwell_time_sec BETWEEN 1 AND 3600000
                            THEN dwell_time_sec ELSE NULL END AS dwell_clean
                FROM read_parquet({evt_files})
                WHERE event_type = 'pageview'
                  AND is_login = 'login'
                  AND {CATEGORY_FILTER}
                  AND event_ts >= '{TRAIN_START}' AND event_ts < '{VAL_SPLIT}'
            )
            SELECT user_id, item_id,
                   COUNT(*) AS n_pageview,
                   MAX(dwell_clean) AS max_dwell_raw,
                   AVG(dwell_clean) AS avg_dwell_raw,
                   MAX(CASE WHEN dwell_clean > 30 OR dwell_clean > 30000 THEN 1 ELSE 0 END) AS has_consider_pv
            FROM pv
            GROUP BY user_id, item_id
        ) TO '{DWELL_TRAIN_FILE}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{DWELL_TRAIN_FILE}')").fetchone()[0]
    conn.close(); del conn
    print(f"{elapsed()} dwell_pair (train): {n:,} pairs  [RAM:{mem_mb():.0f}MB]")

# ── 1j: User preference extended (price/area/adtype/district/channel) ─────────
# Unblocks Stage 3 features: price_match, area_match, adtype_match, district_match,
# seller_match, user_channel_ratios.
PREF_EXT_FILE = f"{CACHE_DIR}/user_pref_extended.parquet"
if os.path.exists(PREF_EXT_FILE):
    print(f"{elapsed()} [SKIP] user_pref_extended.parquet already exists")
else:
    print(f"{elapsed()} Building user_pref_extended …  [RAM:{mem_mb():.0f}MB]")
    pos    = pd.read_parquet(POS_FILE)
    items  = pd.read_parquet(ITEMS_FILE)[
        ['item_id','price_bucket','ad_type','district_name','seller_type','bedrooms','area_sqm']]
    joined = pos.merge(items, on='item_id', how='left')
    del items; gc.collect()

    # Most-frequent value per user for categoricals (price/ad_type/district/seller)
    pref_parts = []
    for col in ['price_bucket','ad_type','district_name','seller_type']:
        cnt = (joined.groupby(['user_id', col]).size().reset_index(name='c')
                     .sort_values('c', ascending=False).drop_duplicates('user_id'))
        cnt = cnt[['user_id', col]].rename(columns={col: f'pref_{col}'})
        pref_parts.append(cnt)

    # Numeric medians
    num_med = joined.groupby('user_id').agg(
        pref_bedrooms=('bedrooms','median'),
        pref_area_sqm=('area_sqm','median'),
    ).reset_index()
    pref_parts.append(num_med)

    # Channel mix (argmax over view_phone/chat/zalo/sms)
    chan_cols = ['n_view_phone','n_chat','n_zalo','n_sms']
    chan = pos.groupby('user_id')[chan_cols].sum().reset_index()
    chan['pref_channel'] = chan[chan_cols].idxmax(axis=1).str.replace('n_','', regex=False)
    chan['_explicit_total'] = chan[chan_cols].sum(axis=1).clip(lower=1)
    for c in chan_cols:
        chan[f'user_{c.replace("n_","")}_ratio'] = chan[c] / chan['_explicit_total']
    chan = chan[['user_id','pref_channel',
                 'user_view_phone_ratio','user_chat_ratio','user_zalo_ratio','user_sms_ratio']]
    pref_parts.append(chan)

    prefs = pref_parts[0]
    for p in pref_parts[1:]:
        prefs = prefs.merge(p, on='user_id', how='outer')
    print(f"{elapsed()} pref_extended: {len(prefs):,} users, {prefs.shape[1]} cols")
    prefs.to_parquet(PREF_EXT_FILE, index=False)
    del pos, joined, pref_parts, prefs, chan, num_med; gc.collect()

# ── 1k: First click per user (for Stage 6 cold session logic) ─────────────────
FIRST_CLICK_FILE = f"{CACHE_DIR}/user_first_click.parquet"
if os.path.exists(FIRST_CLICK_FILE):
    print(f"{elapsed()} [SKIP] user_first_click.parquet already exists")
else:
    conn = fresh_conn()
    print(f"{elapsed()} Extracting first click per user …  [RAM:{mem_mb():.0f}MB]")
    conn.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT user_id, item_id, event_ts, event_type, session_id,
                       category, city_name,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_ts) AS rn
                FROM read_parquet({evt_files})
                WHERE is_login = 'login'
                  AND {CATEGORY_FILTER}
                  AND user_id IS NOT NULL
                  AND event_ts BETWEEN '{TRAIN_START}' AND '{TRAIN_END}'
            )
            SELECT user_id, item_id, event_ts, event_type, session_id, category, city_name
            FROM ranked
            WHERE rn = 1
        ) TO '{FIRST_CLICK_FILE}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{FIRST_CLICK_FILE}')").fetchone()[0]
    conn.close(); del conn
    print(f"{elapsed()} first_click: {n:,} users  [RAM:{mem_mb():.0f}MB]")

print(f"{elapsed()} DONE — all cache files saved to {CACHE_DIR}  [RAM:{mem_mb():.0f}MB]")
