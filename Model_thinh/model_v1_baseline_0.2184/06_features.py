"""
Step 6: Feature engineering for LightGBM reranker.
~60 features per (user_id, item_id) candidate pair.
Outputs:
  cache/features_train.parquet  — labeled pairs (for LightGBM training)
  cache/features_test.parquet   — unlabeled pairs (for inference)
"""
import sys, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

# ── Fast-path: if features_test exists but features_train doesn't, only build labels ──
import os as _os
if (_os.path.exists(f"{CACHE_DIR}/features_test.parquet") and
        _os.path.exists(f"{CACHE_DIR}/features_train.parquet")):
    print(f"{elapsed()} Both feature files exist — skipping rebuild. DONE")
    sys.exit(0)
elif (_os.path.exists(f"{CACHE_DIR}/features_test.parquet") and
        not _os.path.exists(f"{CACHE_DIR}/features_train.parquet")):
    print(f"{elapsed()} Fast-path: features_test exists — loading it to build training labels …")
    _feat = pd.read_parquet(f"{CACHE_DIR}/features_test.parquet")
    _FCOLS = [c for c in _feat.columns if c not in {'user_id','item_id'}]
    _pos = pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet")
    _pos['last_ts'] = pd.to_datetime(_pos['last_ts'])
    _val_split_dt = pd.Timestamp(VAL_SPLIT)
    _last_ts = _pos['last_ts'].tolist()
    _mask    = [ts >= _val_split_dt for ts in _last_ts]; del _last_ts
    _pu = [u for u, m in zip(_pos['user_id'].tolist(), _mask) if m]
    _pi = [i for i, m in zip(_pos['item_id'].tolist(), _mask) if m]
    del _mask, _pos; gc.collect()
    _pos_set = set(zip(_pu, _pi)); del _pu, _pi; gc.collect()
    print(f"{elapsed()} Positive val pairs: {len(_pos_set):,}")
    _u = _feat['user_id'].tolist(); _i = _feat['item_id'].tolist()
    _feat['label'] = np.array([1 if (u,i) in _pos_set else 0 for u,i in zip(_u,_i)], dtype=np.int8)
    del _u, _i, _pos_set; gc.collect()
    _pr = _feat['label'].mean()
    print(f"{elapsed()} Positive rate: {_pr:.4f} ({_feat['label'].sum():,}/{len(_feat):,})")
    _feat[['user_id','item_id','label'] + _FCOLS].to_parquet(
        f"{CACHE_DIR}/features_train.parquet", index=False)
    print(f"{elapsed()} Saved features_train.parquet — DONE (fast-path)")
    sys.exit(0)

print(f"{elapsed()} Loading data …")
cands    = pd.read_parquet(f"{CACHE_DIR}/candidates.parquet")
items    = pd.read_parquet(f"{CACHE_DIR}/items.parquet")
iq       = pd.read_parquet(f"{CACHE_DIR}/item_quality.parquet")
profiles = pd.read_parquet(f"{CACHE_DIR}/user_profiles.parquet")
inter    = pd.read_parquet(f"{CACHE_DIR}/user_item_inter.parquet")
pos      = pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet")
pop      = pd.read_parquet(f"{CACHE_DIR}/popular_items.parquet")

train_end_dt = pd.Timestamp(TRAIN_END)
val_split_dt = pd.Timestamp(VAL_SPLIT)

# ── Item features ─────────────────────────────────────────────────────────────
print(f"{elapsed()} Building item features …")
items_feat = items.drop_duplicates('item_id').copy()
items_feat['posted_date'] = pd.to_datetime(items_feat['posted_date'])
items_feat['days_since_posted'] = (train_end_dt - items_feat['posted_date']).dt.days.clip(lower=0)
items_feat['has_project_id']    = items_feat['project_id'].notna().astype(int)

# Label encode categoricals
le_cat     = LabelEncoder()
le_adtype  = LabelEncoder()
le_seller  = LabelEncoder()
le_price   = LabelEncoder()
le_city    = LabelEncoder()
le_district= LabelEncoder()

items_feat['cat_enc']      = le_cat.fit_transform(items_feat['category'].astype(str))
items_feat['adtype_enc']   = le_adtype.fit_transform(items_feat['ad_type'].fillna('unknown'))
items_feat['seller_enc']   = le_seller.fit_transform(items_feat['seller_type'].fillna('unknown'))
items_feat['price_enc']    = le_price.fit_transform(items_feat['price_bucket'].fillna('unknown'))
items_feat['city_enc']     = le_city.fit_transform(items_feat['city_name'].fillna('unknown'))
items_feat['district_enc'] = le_district.fit_transform(items_feat['district_name'].fillna('unknown'))

# Area bucket
AREA_BINS  = [0,20,30,45,60,80,100,150,200,300,500,10_000]
AREA_LBLS  = list(range(len(AREA_BINS)-1))
items_feat['area_bucket'] = pd.cut(
    items_feat['area_sqm'].clip(lower=0),
    bins=AREA_BINS, labels=AREA_LBLS, right=False
).cat.codes  # -1 for NaN → fill 0

items_feat['area_bucket'] = items_feat['area_bucket'].clip(lower=0)
items_feat['area_sqm_log'] = np.log1p(items_feat['area_sqm'].fillna(0))
items_feat['images_count_log'] = np.log1p(items_feat['images_count'].fillna(0))
items_feat['bedrooms_filled'] = items_feat['bedrooms'].fillna(-1)

# Merge item quality
items_feat = items_feat.merge(
    iq[['item_id','item_cvr','total_events','pos_events','unique_users','avg_dwell']],
    on='item_id', how='left'
)
items_feat['item_cvr']         = items_feat['item_cvr'].fillna(0)
items_feat['total_events_log'] = np.log1p(items_feat['total_events'].fillna(0))
items_feat['unique_users_log'] = np.log1p(items_feat['unique_users'].fillna(0))
items_feat['avg_dwell']        = items_feat['avg_dwell'].fillna(0)

# Trending score (last 28 days)
pop_agg = pop.groupby('item_id').agg(
    trend_pos=('trend_pos','sum'),
    trend_events=('trend_events','sum')
).reset_index()
pop_agg['trend_cvr'] = pop_agg['trend_pos'] / pop_agg['trend_events'].clip(lower=1)
items_feat = items_feat.merge(pop_agg, on='item_id', how='left')
items_feat['trend_pos_log']    = np.log1p(items_feat['trend_pos'].fillna(0))
items_feat['trend_cvr']        = items_feat['trend_cvr'].fillna(0)
# Velocity: ratio of recent 7-day contacts vs all-time (proxy for "hot listing")
ITEM_COLS = [
    'cat_enc','adtype_enc','seller_enc','price_enc','city_enc','district_enc',
    'area_bucket','area_sqm_log','images_count_log','bedrooms_filled',
    'days_since_posted','has_project_id',
    'item_cvr','total_events_log','unique_users_log','avg_dwell',
    'trend_pos_log','trend_cvr',
]
items_feat = items_feat[['item_id'] + ITEM_COLS]

# ── User features ─────────────────────────────────────────────────────────────
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

USER_COLS = [
    'pref_cat_enc','pref_city_enc','total_pos_log','unique_items_log',
    'days_since_last','active_span_days','avg_pos_per_item',
]
profiles_feat = profiles[['user_id'] + USER_COLS]

# ── User-Item interaction features ───────────────────────────────────────────
print(f"{elapsed()} Building user-item features …")
pos['last_ts'] = pd.to_datetime(pos['last_ts'])
pos['days_since_ui'] = (train_end_dt - pos['last_ts']).dt.days.clip(lower=0)
pos_feat = pos[['user_id','item_id','pos_count','days_since_ui',
                'n_view_phone','n_chat','n_other']].copy()
pos_feat['pos_count_log'] = np.log1p(pos_feat['pos_count'])

inter_feat = inter[['user_id','item_id','total_leads','total_chat_msgs',
                     'total_chat_turns','ever_purchased','active_days_inter']].copy()
inter_feat['total_leads_log']     = np.log1p(inter_feat['total_leads'].fillna(0))
inter_feat['total_chat_turns_log']= np.log1p(inter_feat['total_chat_turns'].fillna(0))
inter_feat['ever_purchased']      = inter_feat['ever_purchased'].fillna(0)

# ── Assemble feature matrix (sequential merges to avoid OOM) ─────────────────
print(f"{elapsed()} Assembling feature matrix …")
feat = cands; del cands; gc.collect()
feat = feat.merge(items_feat,    on='item_id',             how='left'); del items_feat; gc.collect()
feat = feat.merge(profiles_feat, on='user_id',             how='left'); del profiles_feat; gc.collect()
feat = feat.merge(pos_feat,      on=['user_id','item_id'], how='left'); del pos_feat; gc.collect()
# Downcast to float32 to free ~3.5 GB before the final (largest) merge
for _col in feat.select_dtypes('float64').columns:
    feat[_col] = feat[_col].astype(np.float32)
gc.collect()
print(f"{elapsed()} feat memory after float32 downcast: {feat.memory_usage(deep=True).sum()/1e9:.2f} GB")
# Trim inter_feat to only FEATURE_COLS columns (drop raw total_leads, total_chat_msgs, total_chat_turns)
inter_feat = inter_feat[['user_id','item_id','total_leads_log','total_chat_turns_log',
                          'ever_purchased','active_days_inter']]
feat = feat.merge(inter_feat,    on=['user_id','item_id'], how='left'); del inter_feat; gc.collect()

# User-item match features
feat['category_match'] = (feat['pref_cat_enc'] == feat['cat_enc']).astype(int)
feat['city_match']     = (feat['pref_city_enc'] == feat['city_enc']).astype(int)

print(f"{elapsed()} Cold users in candidates: {feat['user_id'].isin(set(pos['user_id'].tolist())).eq(False).mean():.2%}")

# Recency of user-item interaction
feat['days_since_ui']        = feat['days_since_ui'].fillna(999)
feat['pos_count_log']        = feat['pos_count_log'].fillna(0)
feat['n_view_phone']         = feat['n_view_phone'].fillna(0)
feat['n_chat']               = feat['n_chat'].fillna(0)
feat['n_other']              = feat['n_other'].fillna(0)
feat['total_leads_log']   = feat['total_leads_log'].fillna(0)
feat['total_chat_turns_log'] = feat['total_chat_turns_log'].fillna(0)
feat['ever_purchased']    = feat['ever_purchased'].fillna(0)
feat['active_days_inter'] = feat['active_days_inter'].fillna(0)

# Fill remaining NaNs from retrieval scores
for col in ['als_score','ease_score','itemcf_score','sasrec_score',
            'als_score_norm','ease_score_norm','itemcf_score_norm','sasrec_score_norm']:
    if col in feat.columns:
        feat[col] = feat[col].fillna(0)

feat['is_repeat']    = feat['is_repeat'].fillna(0).astype(int)
feat['repeat_count'] = feat['repeat_count'].fillna(0)
feat['source_count'] = feat['source_count'].fillna(1)
feat['blend_score']  = feat['blend_score'].fillna(0)

FEATURE_COLS = (
    # Retrieval scores
    ['als_score_norm','ease_score_norm','itemcf_score_norm','sasrec_score_norm',
     'blend_score','source_count','is_repeat','repeat_count'] +
    # User-item interaction
    ['pos_count_log','days_since_ui','n_view_phone','n_chat','n_other',
     'total_leads_log','total_chat_turns_log','ever_purchased','active_days_inter'] +
    # Item features
    ITEM_COLS +
    # User features
    USER_COLS +
    # Match features
    ['category_match','city_match']
)

# Ensure all feature columns exist
for col in FEATURE_COLS:
    if col not in feat.columns:
        feat[col] = 0.0

print(f"{elapsed()} Feature matrix: {len(feat):,} rows × {len(FEATURE_COLS)} features")

# Save test features
feat_test = feat[['user_id','item_id'] + FEATURE_COLS].copy()
feat_test.to_parquet(f"{CACHE_DIR}/features_test.parquet", index=False)
del feat_test; gc.collect()  # free 7.6 GB before label building
print(f"{elapsed()} Saved features_test.parquet")

# ── Build training data from temporal split ───────────────────────────────────
# Train ALS/retrieval on Nov→Feb, validate on Mar→Apr
# For LightGBM: need (user,item) pairs with label=1 if positive in Mar→Apr
print(f"{elapsed()} Building training labels (temporal split) …")
pos['last_ts'] = pd.to_datetime(pos['last_ts'])
pos['first_ts']= pd.to_datetime(pos['first_ts'])

# Items interacted after VAL_SPLIT = ground truth labels for LightGBM training
# Use tolist() to escape PyArrow allocator for boolean mask + set operations
_last_ts_list = pos['last_ts'].tolist()
_pos_val_mask = [ts >= val_split_dt for ts in _last_ts_list]
pos_val_users = [u for u, m in zip(pos['user_id'].tolist(), _pos_val_mask) if m]
pos_val_items = [i for i, m in zip(pos['item_id'].tolist(), _pos_val_mask) if m]
del _last_ts_list, _pos_val_mask; gc.collect()
pos_val_set = set(zip(pos_val_users, pos_val_items))
print(f"{elapsed()} Positive labels (val): {len(pos_val_set):,}")
del pos_val_users, pos_val_items; gc.collect()

# Assign labels via set lookup directly onto feat (no copy — saves 7.6 GB)
_u = feat['user_id'].tolist()
_i = feat['item_id'].tolist()
feat['label'] = np.array([1 if (u, i) in pos_val_set else 0 for u, i in zip(_u, _i)], dtype=np.int8)
del _u, _i, pos_val_set; gc.collect()

pos_rate = feat['label'].mean()
print(f"{elapsed()} Label positive rate: {pos_rate:.4f} ({feat['label'].sum():,}/{len(feat):,})")

feat[['user_id','item_id','label'] + FEATURE_COLS].to_parquet(f"{CACHE_DIR}/features_train.parquet", index=False)
print(f"{elapsed()} Saved features_train.parquet")
print(f"{elapsed()} DONE")
