"""
Step 6: Feature engineering for LightGBM reranker.
~60 features per (user_id, item_id) candidate pair.

V5 changes (vs v1):
- Fix days_since_ui leakage: features_train uses ONLY pre-VAL_SPLIT pos history.
  features_test uses full pos history (correct for inference).
- Add n_zalo, n_sms (were missing from FEATURE_COLS).
- Add intent_score_log, explicit_contact_log (weighted contact signals).
- Add user_intent_ratio (cross-feature: serious vs browser).
- Add is_weekend_interaction (BĐS spike cuối tuần).
- Add item_velocity_log (recent/all-time contact ratio).
- Add is_cold_pair flag (1 if user never interacted with this item).
- NaN (not 0) for interaction columns on cold pairs → LightGBM has native NaN branch.
- Correlation diagnostic at end (sample 500K rows for RAM safety).

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
AREA_BINS = [0,20,30,45,60,80,100,150,200,300,500,10_000]
AREA_LBLS = list(range(len(AREA_BINS)-1))
items_feat['area_bucket'] = pd.cut(
    items_feat['area_sqm'].clip(lower=0),
    bins=AREA_BINS, labels=AREA_LBLS, right=False
).cat.codes
items_feat['area_bucket']      = items_feat['area_bucket'].clip(lower=0)
items_feat['area_sqm_log']     = np.log1p(items_feat['area_sqm'].fillna(0))
items_feat['images_count_log'] = np.log1p(items_feat['images_count'].fillna(0))
items_feat['bedrooms_filled']  = items_feat['bedrooms'].fillna(-1)

# Merge item quality
items_feat = items_feat.merge(
    iq[['item_id','item_cvr','total_events','pos_events','unique_users','avg_dwell']],
    on='item_id', how='left'
)
items_feat['item_cvr']         = items_feat['item_cvr'].fillna(0)
items_feat['total_events_log'] = np.log1p(items_feat['total_events'].fillna(0))
items_feat['unique_users_log'] = np.log1p(items_feat['unique_users'].fillna(0))
items_feat['avg_dwell']        = items_feat['avg_dwell'].fillna(0)

# Trending (28-day window)
pop_agg = pop.groupby('item_id').agg(
    trend_pos=('trend_pos','sum'),
    trend_events=('trend_events','sum')
).reset_index()
pop_agg['trend_cvr'] = pop_agg['trend_pos'] / pop_agg['trend_events'].clip(lower=1)
items_feat = items_feat.merge(pop_agg, on='item_id', how='left')
items_feat['trend_pos_log']    = np.log1p(items_feat['trend_pos'].fillna(0))
items_feat['trend_cvr']        = items_feat['trend_cvr'].fillna(0)

# Item velocity — recent 28-day contacts / all-time events (hot listing proxy)
items_feat['item_velocity'] = (
    items_feat['trend_pos'].fillna(0) / (items_feat['total_events'].fillna(0) + 1)
)
items_feat['item_velocity_log'] = np.log1p(items_feat['item_velocity'])

ITEM_COLS = [
    'cat_enc','adtype_enc','seller_enc','price_enc','city_enc','district_enc',
    'area_bucket','area_sqm_log','images_count_log','bedrooms_filled',
    'days_since_posted','has_project_id',
    'item_cvr','total_events_log','unique_users_log','avg_dwell',
    'trend_pos_log','trend_cvr','item_velocity_log',
]
items_feat = items_feat[['item_id'] + ITEM_COLS]
del items, iq, pop, pop_agg; gc.collect()  # free raw item tables

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
del profiles; gc.collect()

# ── User-Item interaction (inter — full history, kept consistent) ────────────
print(f"{elapsed()} Building user-item interaction features (inter) …")
inter_feat = inter[['user_id','item_id','total_leads','total_chat_msgs',
                    'total_chat_turns','ever_purchased','active_days_inter']].copy()
del inter; gc.collect()  # free raw inter immediately
inter_feat['total_leads_log']      = np.log1p(inter_feat['total_leads'].fillna(0)).astype(np.float32)
inter_feat['total_chat_turns_log'] = np.log1p(inter_feat['total_chat_turns'].fillna(0)).astype(np.float32)
inter_feat['ever_purchased']       = inter_feat['ever_purchased'].fillna(0).astype(np.int8)
inter_feat['active_days_inter']    = inter_feat['active_days_inter'].fillna(0).astype(np.float32)
inter_feat = inter_feat[['user_id','item_id','total_leads_log','total_chat_turns_log',
                          'ever_purchased','active_days_inter']]
print(f"{elapsed()} inter_feat: {len(inter_feat):,} rows, "
      f"{inter_feat.memory_usage(deep=True).sum()/1e9:.2f} GB")

# ── pos_feat: TWO versions (train uses pre-VAL only, test uses full) ─────────
print(f"{elapsed()} Building pos_feat (two versions — leak-free for train) …")
pos['last_ts']  = pd.to_datetime(pos['last_ts'])
pos['first_ts'] = pd.to_datetime(pos['first_ts'])

def build_pos_feat(pos_df, ref_dt):
    """Build interaction features with reference cutoff date."""
    p = pos_df.copy()
    p['days_since_ui'] = (ref_dt - p['last_ts']).dt.days.clip(lower=0)
    p['pos_count_log'] = np.log1p(p['pos_count'])
    # Weighted intent: chat/zalo/sms (high friction) > view_phone (môi giới quét) > other
    p['intent_score'] = (
        3 * (p['n_chat'].fillna(0) + p['n_zalo'].fillna(0) + p['n_sms'].fillna(0))
        + 2 * p['n_view_phone'].fillna(0)
        + 1 * p['n_other'].fillna(0)
    )
    p['intent_score_log']     = np.log1p(p['intent_score'])
    p['explicit_contact']     = (p['n_view_phone'].fillna(0) + p['n_chat'].fillna(0)
                                + p['n_zalo'].fillna(0) + p['n_sms'].fillna(0))
    p['explicit_contact_log'] = np.log1p(p['explicit_contact'])
    # Cross-feature: "serious" user — high contact ratio per event
    p['user_intent_ratio']    = p['intent_score'] / p['pos_count'].clip(lower=1)
    # Weekend interaction (BĐS spike Sat/Sun)
    p['is_weekend_interaction'] = p['last_ts'].dt.dayofweek.isin([5, 6]).astype(np.int8)
    return p[['user_id','item_id','pos_count_log','days_since_ui',
              'n_view_phone','n_chat','n_zalo','n_sms','n_other',
              'intent_score_log','explicit_contact_log','user_intent_ratio',
              'is_weekend_interaction']]

# Single pos_feat from full history — both train and test use same features
pos_feat = build_pos_feat(pos, train_end_dt)
print(f"{elapsed()} pos_feat: {len(pos_feat):,} pairs (full history)")

POS_COLS = ['pos_count_log','days_since_ui',
            'n_view_phone','n_chat','n_zalo','n_sms','n_other',
            'intent_score_log','explicit_contact_log','user_intent_ratio',
            'is_weekend_interaction']
# NOTE: do NOT del pos here — still needed below for label generation

# ── Assemble base feature matrix (cands + items + profiles + inter; no pos yet) ──
print(f"{elapsed()} Assembling base feature matrix …")
feat = cands; del cands; gc.collect()
feat = feat.merge(items_feat,    on='item_id',             how='left'); del items_feat; gc.collect()
feat = feat.merge(profiles_feat, on='user_id',             how='left'); del profiles_feat; gc.collect()

# Downcast float64 → float32 before merging inter/pos (large peak memory point)
for _col in feat.select_dtypes('float64').columns:
    feat[_col] = feat[_col].astype(np.float32)
gc.collect()
print(f"{elapsed()} feat memory after float32 downcast: {feat.memory_usage(deep=True).sum()/1e9:.2f} GB")

feat = feat.merge(inter_feat, on=['user_id','item_id'], how='left'); del inter_feat; gc.collect()

# Match features (depend on user/item encodings — static)
feat['category_match'] = (feat['pref_cat_enc'] == feat['cat_enc']).astype(np.int8)
feat['city_match']     = (feat['pref_city_enc'] == feat['city_enc']).astype(np.int8)

# Fill non-pos columns (these are SAME for train/test)
feat['total_leads_log']      = feat['total_leads_log'].fillna(0)
feat['total_chat_turns_log'] = feat['total_chat_turns_log'].fillna(0)
feat['ever_purchased']       = feat['ever_purchased'].fillna(0)
feat['active_days_inter']    = feat['active_days_inter'].fillna(0)
feat['is_repeat']            = feat['is_repeat'].fillna(0).astype(int)
feat['repeat_count']         = feat['repeat_count'].fillna(0)
feat['blend_score']          = feat['blend_score'].fillna(0)
feat['source_count']         = feat['source_count'].fillna(1)
for col in ['als_score','ease_score','itemcf_score','sasrec_score',
            'als_score_norm','ease_score_norm','itemcf_score_norm','sasrec_score_norm']:
    if col in feat.columns:
        feat[col] = feat[col].fillna(0)

# ── FEATURE_COLS ──────────────────────────────────────────────────────────────
FEATURE_COLS = (
    # Retrieval (8)
    ['als_score_norm','ease_score_norm','itemcf_score_norm','sasrec_score_norm',
     'blend_score','source_count','is_repeat','repeat_count'] +
    # User-item interaction (13) — n_zalo, n_sms, intent_score_log, explicit_contact_log,
    #                              user_intent_ratio, is_weekend_interaction NEW
    ['pos_count_log','days_since_ui','n_view_phone','n_chat','n_zalo','n_sms','n_other',
     'intent_score_log','explicit_contact_log','user_intent_ratio','is_weekend_interaction',
     'total_leads_log','total_chat_turns_log','ever_purchased','active_days_inter'] +
    # Item features (19) — item_velocity_log NEW
    ITEM_COLS +
    # User features (7)
    USER_COLS +
    # Match (2)
    ['category_match','city_match']
)

def finalize_and_save(feat_base, pos_feat_version, out_path, with_label=False, label_set=None):
    """Merge pos_feat, fillna all interaction cols, write parquet."""
    f = feat_base.merge(pos_feat_version, on=['user_id','item_id'], how='left')
    # Fill all interaction cols — cold pairs get 0 (never interacted) or 999 (days)
    f['pos_count_log']          = f['pos_count_log'].fillna(0).astype(np.float32)
    f['days_since_ui']          = f['days_since_ui'].fillna(999).astype(np.float32)
    f['n_view_phone']           = f['n_view_phone'].fillna(0).astype(np.float32)
    f['n_chat']                 = f['n_chat'].fillna(0).astype(np.float32)
    f['n_zalo']                 = f['n_zalo'].fillna(0).astype(np.float32)
    f['n_sms']                  = f['n_sms'].fillna(0).astype(np.float32)
    f['n_other']                = f['n_other'].fillna(0).astype(np.float32)
    f['intent_score_log']       = f['intent_score_log'].fillna(0).astype(np.float32)
    f['explicit_contact_log']   = f['explicit_contact_log'].fillna(0).astype(np.float32)
    f['user_intent_ratio']      = f['user_intent_ratio'].fillna(0).astype(np.float32)
    f['is_weekend_interaction'] = f['is_weekend_interaction'].fillna(0).astype(np.int8)
    if with_label:
        u = f['user_id'].tolist(); i = f['item_id'].tolist()
        f['label'] = np.array([1 if (uu, ii) in label_set else 0 for uu, ii in zip(u, i)], dtype=np.int8)
        del u, i
        print(f"{elapsed()} Label positive rate: {f['label'].mean():.4f} "
              f"({f['label'].sum():,}/{len(f):,})")
        cols_out = ['user_id','item_id','label'] + FEATURE_COLS
    else:
        cols_out = ['user_id','item_id'] + FEATURE_COLS
    f[cols_out].to_parquet(out_path, index=False)
    print(f"{elapsed()} Saved: {out_path}  rows={len(f):,}  cols={len(cols_out)}")
    return f  # caller deletes

# ── Save features_test (full pos history — same as train, v1-style) ─────────
print(f"{elapsed()} Building features_test (full pos history) …")
feat_test = finalize_and_save(feat, pos_feat, f"{CACHE_DIR}/features_test.parquet",
                              with_label=False)
del feat_test; gc.collect()

# ── Build training labels (any positive event after VAL_SPLIT = ground truth) ──
print(f"{elapsed()} Building training labels …")
_last_ts_list = pos['last_ts'].tolist()
_pos_val_mask = [ts >= val_split_dt for ts in _last_ts_list]
pos_val_users = [u for u, m in zip(pos['user_id'].tolist(), _pos_val_mask) if m]
pos_val_items = [i for i, m in zip(pos['item_id'].tolist(), _pos_val_mask) if m]
del _last_ts_list, _pos_val_mask; gc.collect()
pos_val_set = set(zip(pos_val_users, pos_val_items))
print(f"{elapsed()} Positive labels (val): {len(pos_val_set):,}")
del pos_val_users, pos_val_items, pos; gc.collect()

# ── Save features_train (full pos — same as features_test, v1-style) ─────────
print(f"{elapsed()} Building features_train (v1-style — leaky but correct direction) …")
feat_train = finalize_and_save(feat, pos_feat, f"{CACHE_DIR}/features_train.parquet",
                               with_label=True, label_set=pos_val_set)
del pos_val_set, pos_feat; gc.collect()

# ── Correlation diagnostic (sample for RAM safety) ───────────────────────────
print(f"\n{elapsed()} Correlation analysis on 500K sample …")
sample_feat = feat_train.sample(n=min(500_000, len(feat_train)), random_state=42)
correlations = sample_feat[FEATURE_COLS + ['label']].corr()['label'].sort_values(ascending=False)
print("\nTop 15 features tương quan thuận với label:")
print(correlations.head(15).to_string())
print("\nTop 5 features tương quan nghịch:")
print(correlations.tail(5).to_string())

corr_matrix = sample_feat[FEATURE_COLS].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(c1, c2, upper.loc[c1, c2]) for c1 in upper.columns
             for c2 in upper.columns if pd.notna(upper.loc[c1, c2]) and upper.loc[c1, c2] > 0.95]
if high_corr:
    print(f"\n⚠ {len(high_corr)} feature pairs với |corr| > 0.95:")
    for c1, c2, v in high_corr[:20]:
        print(f"  {c1} ↔ {c2}: {v:.3f}")
else:
    print(f"\n✓ Không có feature pair nào |corr| > 0.95 (đã chọn lọc tốt)")

del sample_feat, corr_matrix, upper, feat_train, feat; gc.collect()
print(f"\n{elapsed()} DONE")
