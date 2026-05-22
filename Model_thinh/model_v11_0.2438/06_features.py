"""
Step 6: Feature engineering for LightGBM reranker — Stage 8 version.

Architecture:
  features_train: candidates_train (leak-free retrievers) + pos_train + inter_train
                  → reranker learns CLEAN signal
  features_test:  candidates (full retrievers) + pos (full) + inter (full)
                  → inference with maximum knowledge

Feature subset (~58 cols, drops Stage 3 features proven non-helpful):
  KEEP: 49 v6 features + 4 match features (price/adtype/district/seller)
        + age_boost_cat + is_renewal_week + hist_decay (2 cols)
  DROP: dwell × 3, snapshot × 5, channel ratios × 4, pref_channel_enc,
        n_chat_verified, same_project_score, bedrooms_match
"""
import sys, time, gc, os
sys.stdout.reconfigure(encoding='utf-8')

# Stage 9: CLI mode — split test/train build into separate processes to avoid RAM fragmentation.
# Usage: python 06_features.py [test|train|both]  (default: both, runs sequentially)
BUILD_MODE = sys.argv[1].lower() if len(sys.argv) > 1 else 'both'
assert BUILD_MODE in ('test','train','both'), f"BUILD_MODE={BUILD_MODE} invalid"
print(f"[STAGE 9] BUILD_MODE={BUILD_MODE}")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

EVENT_WEIGHTS = {
    'view_phone':        10.0,
    'contact_chat':       9.0,
    'contact_zalo':       9.0,
    'contact_sms':        8.0,
    'other_interaction':  2.5,
}
HALF_LIFE_DAYS = 21.0

print(f"{elapsed()} Loading base data …")
items    = pd.read_parquet(f"{CACHE_DIR}/items.parquet")
iq       = pd.read_parquet(f"{CACHE_DIR}/item_quality.parquet")
profiles = pd.read_parquet(f"{CACHE_DIR}/user_profiles.parquet")
pop      = pd.read_parquet(f"{CACHE_DIR}/popular_items.parquet")

# Stage 8: TWO sets of pos/inter/candidates
# Stage 11: env TEST_CANDS_MODE=lukewarm uses candidates_lukewarm.parquet (expanded coverage)
TEST_CANDS_MODE = os.environ.get('TEST_CANDS_MODE', 'full').lower()
assert TEST_CANDS_MODE in ('full','lukewarm'), f"TEST_CANDS_MODE={TEST_CANDS_MODE} invalid"
test_cands_file = (f"{CACHE_DIR}/candidates_lukewarm.parquet"
                   if TEST_CANDS_MODE == 'lukewarm'
                   else f"{CACHE_DIR}/candidates.parquet")
print(f"[STAGE 11] TEST_CANDS_MODE={TEST_CANDS_MODE} → test from {test_cands_file}")

print(f"{elapsed()} Loading Stage 8 _train data …")
cands_train = pd.read_parquet(f"{CACHE_DIR}/candidates_train.parquet")
cands_test  = pd.read_parquet(test_cands_file)
pos_train   = pd.read_parquet(f"{CACHE_DIR}/user_item_pos_train.parquet")
pos_test    = pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet")
inter_train = pd.read_parquet(f"{CACHE_DIR}/user_item_inter_train.parquet")
inter_test  = pd.read_parquet(f"{CACHE_DIR}/user_item_inter.parquet")
print(f"{elapsed()}   cands: train={len(cands_train):,} test={len(cands_test):,}")
print(f"{elapsed()}   pos:   train={len(pos_train):,} test={len(pos_test):,}")
print(f"{elapsed()}   inter: train={len(inter_train):,} test={len(inter_test):,}")

train_end_dt = pd.Timestamp(TRAIN_END)
val_split_dt = pd.Timestamp(VAL_SPLIT)

# ── Item features (SHARED — same for train/test rows) ─────────────────────────
print(f"{elapsed()} Building item features …")
items_feat = items.drop_duplicates('item_id').copy()
items_feat['posted_date']       = pd.to_datetime(items_feat['posted_date'])
items_feat['days_since_posted'] = (train_end_dt - items_feat['posted_date']).dt.days.clip(lower=0)
items_feat['has_project_id']    = items_feat['project_id'].notna().astype(int)
items_feat['is_renewal_week']   = items_feat['days_since_posted'].isin([15,16,30,31]).astype(np.int8)

# age_boost_cat
def age_boost(row):
    age = row['days_since_posted']; cat = row['category']
    if cat == 1050:
        if   age < 15:  return 1.00
        elif age < 30:  return 1.05
        elif age < 60:  return 1.08
        else:           return 1.10
    else:
        if   age <= 2: return 1.20
        elif age <= 7: return 1.10
        elif age <=14: return 1.03
        elif age <=29: return 1.00
        else:          return 0.92
items_feat['age_boost_cat'] = items_feat.apply(age_boost, axis=1).astype(np.float32)

le_cat = LabelEncoder(); le_adtype = LabelEncoder(); le_seller = LabelEncoder()
le_price = LabelEncoder(); le_city = LabelEncoder(); le_district = LabelEncoder()
items_feat['cat_enc']      = le_cat.fit_transform(items_feat['category'].astype(str))
items_feat['adtype_enc']   = le_adtype.fit_transform(items_feat['ad_type'].fillna('unknown'))
items_feat['seller_enc']   = le_seller.fit_transform(items_feat['seller_type'].fillna('unknown'))
items_feat['price_enc']    = le_price.fit_transform(items_feat['price_bucket'].fillna('unknown'))
items_feat['city_enc']     = le_city.fit_transform(items_feat['city_name'].fillna('unknown'))
items_feat['district_enc'] = le_district.fit_transform(items_feat['district_name'].fillna('unknown'))

AREA_BINS = [0,20,30,45,60,80,100,150,200,300,500,10_000]
AREA_LBLS = list(range(len(AREA_BINS)-1))
items_feat['area_bucket'] = pd.cut(items_feat['area_sqm'].clip(lower=0),
                                     bins=AREA_BINS, labels=AREA_LBLS, right=False).cat.codes
items_feat['area_bucket']      = items_feat['area_bucket'].clip(lower=0)
items_feat['area_sqm_log']     = np.log1p(items_feat['area_sqm'].fillna(0))
items_feat['images_count_log'] = np.log1p(items_feat['images_count'].fillna(0))
items_feat['bedrooms_filled']  = items_feat['bedrooms'].fillna(-1)

items_feat = items_feat.merge(
    iq[['item_id','item_cvr','total_events','pos_events','pageviews','unique_users','avg_dwell']],
    on='item_id', how='left')
items_feat['item_cvr']         = items_feat['item_cvr'].fillna(0)
items_feat['total_events_log'] = np.log1p(items_feat['total_events'].fillna(0))
items_feat['unique_users_log'] = np.log1p(items_feat['unique_users'].fillna(0))
items_feat['avg_dwell']        = items_feat['avg_dwell'].fillna(0)

# Stage 10: item_contact_rate_pct = pos_events / pageviews (cleaner CVR than item_cvr)
# item_cvr = pos_events / total_events; new metric = pos_events / pageviews focuses on view→contact funnel
items_feat['item_contact_rate_pct'] = (
    items_feat['pos_events'].fillna(0) / items_feat['pageviews'].fillna(0).clip(lower=1)
).astype(np.float32)
# Stage 10: item_repeat_viewer_pct = 1 - (unique_users / total_events) when events > unique_users
items_feat['item_repeat_viewer_pct'] = (
    1.0 - (items_feat['unique_users'].fillna(0) / items_feat['total_events'].fillna(0).clip(lower=1))
).clip(lower=0).astype(np.float32)

pop_agg = pop.groupby('item_id').agg(
    trend_pos=('trend_pos','sum'), trend_events=('trend_events','sum')).reset_index()
pop_agg['trend_cvr'] = pop_agg['trend_pos'] / pop_agg['trend_events'].clip(lower=1)
items_feat = items_feat.merge(pop_agg, on='item_id', how='left')
items_feat['trend_pos_log']     = np.log1p(items_feat['trend_pos'].fillna(0))
items_feat['trend_cvr']         = items_feat['trend_cvr'].fillna(0)
items_feat['item_velocity']     = items_feat['trend_pos'].fillna(0) / (items_feat['total_events'].fillna(0) + 1)
items_feat['item_velocity_log'] = np.log1p(items_feat['item_velocity'])

# Stage 9: Multi-window velocity (7d, 14d) + acceleration
pop_7d  = pd.read_parquet(f"{CACHE_DIR}/popular_items_7d.parquet")
pop_14d = pd.read_parquet(f"{CACHE_DIR}/popular_items_14d.parquet")
items_feat = items_feat.merge(pop_7d[['item_id','trend_pos_7d']],   on='item_id', how='left')
items_feat = items_feat.merge(pop_14d[['item_id','trend_pos_14d']], on='item_id', how='left')
items_feat['trend_pos_7d']  = items_feat['trend_pos_7d'].fillna(0)
items_feat['trend_pos_14d'] = items_feat['trend_pos_14d'].fillna(0)
# Velocity: fraction of 28-day events that happened in recent windows
items_feat['velocity_7d']    = items_feat['trend_pos_7d']  / (items_feat['trend_pos'].fillna(0) + 1)
items_feat['velocity_14d']   = items_feat['trend_pos_14d'] / (items_feat['trend_pos'].fillna(0) + 1)
items_feat['velocity_accel'] = items_feat['velocity_7d']   - items_feat['velocity_14d']  # ↑ = accelerating
del pop_7d, pop_14d; gc.collect()

ITEM_COLS = [
    'cat_enc','adtype_enc','seller_enc','price_enc','city_enc','district_enc',
    'area_bucket','area_sqm_log','images_count_log','bedrooms_filled',
    'days_since_posted','has_project_id',
    'item_cvr','total_events_log','unique_users_log','avg_dwell',
    'trend_pos_log','trend_cvr','item_velocity_log',
    'is_renewal_week','age_boost_cat',
    'velocity_7d','velocity_14d','velocity_accel',  # Stage 9
    'item_contact_rate_pct','item_repeat_viewer_pct',  # Stage 10
]

# Keep raw bucket cols for match-feature joins later
items_keep = items_feat[['item_id'] + ITEM_COLS + ['price_bucket','ad_type','district_name',
                                                    'seller_type']].copy()
del items, iq, pop, pop_agg, items_feat; gc.collect()

# ── User features (SHARED) ────────────────────────────────────────────────────
print(f"{elapsed()} Building user features …")
le_ucat  = LabelEncoder()
le_ucity = LabelEncoder()
profiles['pref_cat_enc']  = le_ucat.fit_transform(profiles['pref_category'].fillna(-1).astype(str))
profiles['pref_city_enc'] = le_ucity.fit_transform(profiles['pref_city'].fillna('unknown'))
profiles['total_pos_log']    = np.log1p(profiles['total_pos_events'].fillna(0))
profiles['unique_items_log'] = np.log1p(profiles['unique_items'].fillna(0))
profiles['days_since_last']  = profiles['days_since_last'].fillna(999)
profiles['active_span_days'] = profiles['active_span_days'].fillna(1)
profiles['avg_pos_per_item'] = (profiles['total_pos_events'] / profiles['unique_items'].clip(lower=1)).fillna(0)

# Stage 9: user_explicit_cvr — % của user's events là explicit contact (vs ad_view).
# "Serious" users (high CVR) khác "browsers" (low CVR).
print(f"{elapsed()}   Computing user_explicit_cvr …")
user_cvr_df = pos_test.groupby('user_id').agg(
    _explicit=('n_view_phone','sum'),
    _chat=('n_chat','sum'),
    _zalo=('n_zalo','sum'),
    _sms=('n_sms','sum'),
    _total=('pos_count','sum'),
).reset_index()
user_cvr_df['_explicit_sum'] = (user_cvr_df['_explicit'].fillna(0) + user_cvr_df['_chat'].fillna(0)
                                + user_cvr_df['_zalo'].fillna(0) + user_cvr_df['_sms'].fillna(0))
user_cvr_df['user_explicit_cvr'] = user_cvr_df['_explicit_sum'] / user_cvr_df['_total'].clip(lower=1)
user_cvr_df = user_cvr_df[['user_id','user_explicit_cvr']]
profiles = profiles.merge(user_cvr_df, on='user_id', how='left')
profiles['user_explicit_cvr'] = profiles['user_explicit_cvr'].fillna(0).astype(np.float32)
del user_cvr_df; gc.collect()

# Stage 10: merge user_behavioral.parquet (n_districts, district_entropy, pct_night, avg_dwell_sec)
print(f"{elapsed()}   Merging user_behavioral …")
ub = pd.read_parquet(f"{CACHE_DIR}/user_behavioral.parquet")
profiles = profiles.merge(ub, on='user_id', how='left')
profiles['user_n_districts']       = profiles['user_n_districts'].fillna(0).astype(np.float32)
profiles['user_district_entropy']  = profiles['user_district_entropy'].fillna(0).astype(np.float32)
profiles['user_pct_night']         = profiles['user_pct_night'].fillna(0).astype(np.float32)
profiles['user_avg_dwell_sec']     = profiles['user_avg_dwell_sec'].fillna(0).astype(np.float32)
profiles['n_sessions_log']         = np.log1p(profiles['n_sessions'].fillna(0)).astype(np.float32)
del ub; gc.collect()

USER_COLS = ['pref_cat_enc','pref_city_enc','total_pos_log','unique_items_log',
             'days_since_last','active_span_days','avg_pos_per_item',
             'user_explicit_cvr',  # Stage 9
             'user_n_districts','user_district_entropy','user_pct_night',  # Stage 10
             'user_avg_dwell_sec','n_sessions_log']  # Stage 10

# pref_extended for match features
pref_ext = pd.read_parquet(f"{CACHE_DIR}/user_pref_extended.parquet",
                            columns=['user_id','pref_price_bucket','pref_ad_type',
                                      'pref_district_name','pref_seller_type'])
profiles = profiles.merge(pref_ext, on='user_id', how='left')

profiles_keep = profiles[['user_id'] + USER_COLS + ['pref_price_bucket','pref_ad_type',
                                                     'pref_district_name','pref_seller_type']].copy()
del profiles, pref_ext; gc.collect()

# ── Inter features (TWO versions: train vs test) ──────────────────────────────
print(f"{elapsed()} Building inter features (2 versions) …")
def build_inter_feat(inter_df):
    f = inter_df[['user_id','item_id','total_leads','total_chat_turns',
                  'ever_purchased','active_days_inter']].copy()
    f['total_leads_log']      = np.log1p(f['total_leads'].fillna(0)).astype(np.float32)
    f['total_chat_turns_log'] = np.log1p(f['total_chat_turns'].fillna(0)).astype(np.float32)
    f['ever_purchased']       = f['ever_purchased'].fillna(0).astype(np.int8)
    f['active_days_inter']    = f['active_days_inter'].fillna(0).astype(np.float32)
    return f[['user_id','item_id','total_leads_log','total_chat_turns_log',
              'ever_purchased','active_days_inter']]
inter_feat_train = build_inter_feat(inter_train); del inter_train; gc.collect()
inter_feat_test  = build_inter_feat(inter_test);  del inter_test; gc.collect()
print(f"{elapsed()}   inter_feat: train={len(inter_feat_train):,} test={len(inter_feat_test):,}")

# ── Pos features (TWO versions) ───────────────────────────────────────────────
print(f"{elapsed()} Building pos_feat (2 versions) …")
pos_train['last_ts']  = pd.to_datetime(pos_train['last_ts'])
pos_train['first_ts'] = pd.to_datetime(pos_train['first_ts'])
pos_test['last_ts']   = pd.to_datetime(pos_test['last_ts'])
pos_test['first_ts']  = pd.to_datetime(pos_test['first_ts'])

def build_pos_feat(pos_df, ref_dt):
    p = pos_df.copy()
    p['days_since_ui'] = (ref_dt - p['last_ts']).dt.days.clip(lower=0)
    p['pos_count_log'] = np.log1p(p['pos_count'])
    p['intent_score']  = (3*(p['n_chat'].fillna(0)+p['n_zalo'].fillna(0)+p['n_sms'].fillna(0))
                          + 2*p['n_view_phone'].fillna(0) + 1*p['n_other'].fillna(0))
    p['intent_score_log']     = np.log1p(p['intent_score'])
    p['explicit_contact']     = (p['n_view_phone'].fillna(0)+p['n_chat'].fillna(0)
                                +p['n_zalo'].fillna(0)+p['n_sms'].fillna(0))
    p['explicit_contact_log'] = np.log1p(p['explicit_contact'])
    p['user_intent_ratio']    = p['intent_score'] / p['pos_count'].clip(lower=1)
    p['is_weekend_interaction'] = p['last_ts'].dt.dayofweek.isin([5,6]).astype(np.int8)
    return p[['user_id','item_id','pos_count_log','days_since_ui',
              'n_view_phone','n_chat','n_zalo','n_sms','n_other',
              'intent_score_log','explicit_contact_log','user_intent_ratio',
              'is_weekend_interaction']]

pos_feat_train = build_pos_feat(pos_train, val_split_dt)
pos_feat_test  = build_pos_feat(pos_test,  train_end_dt)
print(f"{elapsed()}   pos_feat: train={len(pos_feat_train):,} test={len(pos_feat_test):,}")

POS_COLS = ['pos_count_log','days_since_ui',
            'n_view_phone','n_chat','n_zalo','n_sms','n_other',
            'intent_score_log','explicit_contact_log','user_intent_ratio',
            'is_weekend_interaction']

# ── hist_decay (2 versions via DuckDB) ────────────────────────────────────────
print(f"{elapsed()} Building hist_decay (2 versions) via DuckDB …")
import duckdb
RAW = f"{CACHE_DIR}/user_item_pos_events_raw.parquet"

def build_hist_decay(cutoff_ts, only_before):
    where = f"WHERE event_ts < TIMESTAMP '{cutoff_ts}'" if only_before else ""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    return con.execute(f"""
        SELECT user_id, item_id,
               SUM(weight * exp(-days_ago / {HALF_LIFE_DAYS})) AS hist_decay_score,
               SUM(weight) AS hist_decay_total
        FROM (
            SELECT user_id, item_id,
                   CASE event_type
                     WHEN 'view_phone'        THEN {EVENT_WEIGHTS['view_phone']}
                     WHEN 'contact_chat'      THEN {EVENT_WEIGHTS['contact_chat']}
                     WHEN 'contact_zalo'      THEN {EVENT_WEIGHTS['contact_zalo']}
                     WHEN 'contact_sms'       THEN {EVENT_WEIGHTS['contact_sms']}
                     WHEN 'other_interaction' THEN {EVENT_WEIGHTS['other_interaction']}
                     ELSE 1.0 END AS weight,
                   GREATEST(0, DATE_DIFF('day', event_ts, TIMESTAMP '{cutoff_ts}')) AS days_ago
            FROM read_parquet('{RAW}')
            {where}
        )
        GROUP BY user_id, item_id
    """).df()

hist_train = build_hist_decay(val_split_dt.isoformat(), only_before=True)
hist_train['hist_decay_score']     = np.log1p(hist_train['hist_decay_score']).astype(np.float32)
hist_train['hist_decay_total_log'] = np.log1p(hist_train['hist_decay_total']).astype(np.float32)
hist_train = hist_train[['user_id','item_id','hist_decay_score','hist_decay_total_log']]
print(f"{elapsed()}   hist_train: {len(hist_train):,} pairs")

hist_test = build_hist_decay(train_end_dt.isoformat(), only_before=False)
hist_test['hist_decay_score']     = np.log1p(hist_test['hist_decay_score']).astype(np.float32)
hist_test['hist_decay_total_log'] = np.log1p(hist_test['hist_decay_total']).astype(np.float32)
hist_test = hist_test[['user_id','item_id','hist_decay_score','hist_decay_total_log']]
print(f"{elapsed()}   hist_test: {len(hist_test):,} pairs")

# ── FEATURE_COLS (~58 features) ───────────────────────────────────────────────
FEATURE_COLS = (
    # Retrieval (8)
    ['als_score_norm','ease_score_norm','itemcf_score_norm','sasrec_score_norm',
     'blend_score','source_count','is_repeat','repeat_count'] +
    # Pos (11)
    POS_COLS +
    # Inter (4)
    ['total_leads_log','total_chat_turns_log','ever_purchased','active_days_inter'] +
    # Hist_decay (2)
    ['hist_decay_score','hist_decay_total_log'] +
    # Item (21 — includes age_boost, is_renewal)
    ITEM_COLS +
    # User (7)
    USER_COLS +
    # Match (6: category, city + 4 new clean)
    ['category_match','city_match','price_match','adtype_match','district_match','seller_match']
)
print(f"{elapsed()} FEATURE_COLS: {len(FEATURE_COLS)} cols")

# ── Build features helper (used twice: train and test) ────────────────────────
def build_features(cands_df, pos_feat_df, hist_df, inter_feat_df,
                    out_path, with_label=False, label_set=None):
    print(f"{elapsed()} >>> Building {out_path} <<<")
    feat = cands_df  # no copy — caller already passed by reference and we own it
    # Downcast cands_df float64 → float32 BEFORE merges (reduces RAM during join)
    for _col in feat.select_dtypes('float64').columns:
        feat[_col] = feat[_col].astype(np.float32)
    gc.collect()
    feat = feat.merge(items_keep,    on='item_id', how='left'); gc.collect()
    # Downcast after first merge (items_keep adds float64 from items table)
    for _col in feat.select_dtypes('float64').columns:
        feat[_col] = feat[_col].astype(np.float32)
    gc.collect()
    feat = feat.merge(profiles_keep, on='user_id', how='left'); gc.collect()

    # Match features
    feat['category_match']  = (feat['pref_cat_enc']  == feat['cat_enc']).astype(np.int8)
    feat['city_match']      = (feat['pref_city_enc'] == feat['city_enc']).astype(np.int8)
    feat['price_match']     = (feat['pref_price_bucket'].fillna('') == feat['price_bucket'].fillna('')).astype(np.int8)
    feat['adtype_match']    = (feat['pref_ad_type'].fillna('') == feat['ad_type'].fillna('')).astype(np.int8)
    feat['district_match']  = (feat['pref_district_name'].fillna('') == feat['district_name'].fillna('')).astype(np.int8)
    feat['seller_match']    = (feat['pref_seller_type'].fillna('') == feat['seller_type'].fillna('')).astype(np.int8)
    feat = feat.drop(columns=['price_bucket','ad_type','district_name','seller_type',
                              'pref_price_bucket','pref_ad_type','pref_district_name','pref_seller_type'])

    # Downcast
    for _col in feat.select_dtypes('float64').columns:
        feat[_col] = feat[_col].astype(np.float32)
    gc.collect()

    # Downcast all float64 → float32 BEFORE next merges
    for _col in feat.select_dtypes('float64').columns:
        feat[_col] = feat[_col].astype(np.float32)
    gc.collect()

    # Merge pos_feat
    feat = feat.merge(pos_feat_df, on=['user_id','item_id'], how='left'); gc.collect()
    feat['pos_count_log']          = feat['pos_count_log'].fillna(0).astype(np.float32)
    feat['days_since_ui']          = feat['days_since_ui'].fillna(999).astype(np.float32)
    feat['n_view_phone']           = feat['n_view_phone'].fillna(0).astype(np.float32)
    feat['n_chat']                 = feat['n_chat'].fillna(0).astype(np.float32)
    feat['n_zalo']                 = feat['n_zalo'].fillna(0).astype(np.float32)
    feat['n_sms']                  = feat['n_sms'].fillna(0).astype(np.float32)
    feat['n_other']                = feat['n_other'].fillna(0).astype(np.float32)
    feat['intent_score_log']       = feat['intent_score_log'].fillna(0).astype(np.float32)
    feat['explicit_contact_log']   = feat['explicit_contact_log'].fillna(0).astype(np.float32)
    feat['user_intent_ratio']      = feat['user_intent_ratio'].fillna(0).astype(np.float32)
    feat['is_weekend_interaction'] = feat['is_weekend_interaction'].fillna(0).astype(np.int8)

    # Merge hist_decay
    feat = feat.merge(hist_df, on=['user_id','item_id'], how='left'); gc.collect()
    feat['hist_decay_score']     = feat['hist_decay_score'].fillna(0).astype(np.float32)
    feat['hist_decay_total_log'] = feat['hist_decay_total_log'].fillna(0).astype(np.float32)

    # Merge inter_feat
    feat = feat.merge(inter_feat_df, on=['user_id','item_id'], how='left'); gc.collect()
    feat['total_leads_log']      = feat['total_leads_log'].fillna(0).astype(np.float32)
    feat['total_chat_turns_log'] = feat['total_chat_turns_log'].fillna(0).astype(np.float32)
    feat['ever_purchased']       = feat['ever_purchased'].fillna(0).astype(np.int8)
    feat['active_days_inter']    = feat['active_days_inter'].fillna(0).astype(np.float32)

    # Cands repair
    feat['is_repeat']    = feat['is_repeat'].fillna(0).astype(int)
    feat['repeat_count'] = feat['repeat_count'].fillna(0)
    feat['blend_score']  = feat['blend_score'].fillna(0)
    feat['source_count'] = feat['source_count'].fillna(1)
    for col in ['als_score','ease_score','itemcf_score','sasrec_score',
                'als_score_norm','ease_score_norm','itemcf_score_norm','sasrec_score_norm']:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0)

    if with_label:
        u = feat['user_id'].tolist(); i = feat['item_id'].tolist()
        feat['label'] = np.array([1 if (uu, ii) in label_set else 0 for uu, ii in zip(u, i)],
                                  dtype=np.int8)
        del u, i
        print(f"{elapsed()}   Label positive rate: {feat['label'].mean():.4f} "
              f"({feat['label'].sum():,}/{len(feat):,})")
        cols_out = ['user_id','item_id','label'] + FEATURE_COLS
    else:
        cols_out = ['user_id','item_id'] + FEATURE_COLS

    missing = [c for c in cols_out if c not in feat.columns]
    if missing:
        print(f"⚠ MISSING cols: {missing}")
        for c in missing:
            feat[c] = 0
    feat[cols_out].to_parquet(out_path, index=False)
    print(f"{elapsed()}   Saved: {out_path}  rows={len(feat):,}  cols={len(cols_out)}")
    return feat

# ── Build labels (events ≥ VAL_SPLIT) ─────────────────────────────────────────
print(f"{elapsed()} Building val labels …")
_pos_val_mask = pos_test['last_ts'] >= val_split_dt
pos_val_set = set(zip(pos_test.loc[_pos_val_mask, 'user_id'].tolist(),
                       pos_test.loc[_pos_val_mask, 'item_id'].tolist()))
print(f"{elapsed()}   Positive labels: {len(pos_val_set):,}")
del _pos_val_mask, pos_train, pos_test; gc.collect()

# Stage 9: free unused branches BEFORE big merge
if BUILD_MODE == 'test':
    del cands_train, pos_feat_train, hist_train, inter_feat_train; gc.collect()
elif BUILD_MODE == 'train':
    del cands_test, pos_feat_test, hist_test, inter_feat_test; gc.collect()

# ── Build features_test (deploy with full retrievers) ─────────────────────────
if BUILD_MODE in ('test','both'):
    build_features(cands_test, pos_feat_test, hist_test, inter_feat_test,
                   f"{CACHE_DIR}/features_test.parquet", with_label=False)
    del cands_test, pos_feat_test, hist_test, inter_feat_test; gc.collect()

# ── Build features_train (leak-free retrievers + pre-val features) ────────────
feat_train = None
if BUILD_MODE in ('train','both'):
    feat_train = build_features(cands_train, pos_feat_train, hist_train, inter_feat_train,
                                 f"{CACHE_DIR}/features_train.parquet",
                                 with_label=True, label_set=pos_val_set)
    del cands_train, pos_feat_train, hist_train, inter_feat_train, pos_val_set; gc.collect()

# ── Correlation diagnostic ────────────────────────────────────────────────────
if feat_train is not None:
    print(f"\n{elapsed()} Correlation analysis (300K sample) …")
    sample = feat_train.sample(n=min(300_000, len(feat_train)), random_state=42)
    correlations = sample[FEATURE_COLS + ['label']].corr()['label'].sort_values(ascending=False)
    print("\nTop 15 features positively correlated with label:")
    print(correlations.head(15).to_string())
    print("\nTop 5 negatively correlated:")
    print(correlations.tail(5).to_string())
    del sample, feat_train; gc.collect()
print(f"\n{elapsed()} DONE")
