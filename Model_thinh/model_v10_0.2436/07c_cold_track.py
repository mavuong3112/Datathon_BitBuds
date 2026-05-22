"""
Step 7c: Cold-Track Specialist — Stage 9.

User History Dropout: train LightGBM LambdaRank với ALL user-derived features MASKED.
Force model to learn item-intrinsic quality (item_velocity, trend_pos, age_boost, etc.).

Inference: rank top-1000 popular pool → cold_top10.
All cold users get SAME top-10 (smart popular by ML-learned weights).

Output: cache/cold_top10.pkl — list of 10 item_id strings.
"""
import sys, os, time, gc, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

# ── Features to MASK (user-derived → set to 0) ────────────────────────────────
USER_DERIVED = {
    # POS-derived (pair-level history)
    'pos_count_log','days_since_ui','n_view_phone','n_chat','n_zalo','n_sms','n_other',
    'intent_score_log','explicit_contact_log','user_intent_ratio','is_weekend_interaction',
    'hist_decay_score','hist_decay_total_log',
    # Inter (pair-level interactions)
    'total_leads_log','total_chat_turns_log','ever_purchased','active_days_inter',
    'is_repeat','repeat_count',
    # Match (user×item)
    'category_match','city_match','price_match','adtype_match','district_match','seller_match',
    # User-level
    'pref_cat_enc','pref_city_enc','total_pos_log','unique_items_log',
    'days_since_last','active_span_days','avg_pos_per_item','user_explicit_cvr',
    # Retriever (user-conditional)
    'itemcf_score_norm','sasrec_score_norm',
}

# ── Load features_train, mask user features ───────────────────────────────────
print(f"{elapsed()} Loading features_train …")
train_df = pd.read_parquet(f"{CACHE_DIR}/features_train.parquet")
skip_cols = {'user_id','item_id','label'}
FEATURE_COLS_ALL = [c for c in train_df.columns if c not in skip_cols]
COLD_FEATURE_COLS = [c for c in FEATURE_COLS_ALL if c not in USER_DERIVED]
print(f"{elapsed()} Total features: {len(FEATURE_COLS_ALL)}, Cold-track: {len(COLD_FEATURE_COLS)}")
print(f"{elapsed()} Masked features: {len(FEATURE_COLS_ALL) - len(COLD_FEATURE_COLS)}")

# Cast
for c in FEATURE_COLS_ALL:
    train_df[c] = train_df[c].astype(np.float32).replace([np.inf, -np.inf], np.nan)

# Mask user-derived to 0 in TRAINING
for c in USER_DERIVED:
    if c in train_df.columns:
        train_df[c] = 0.0

# Train/val split (last 20% users by appearance order)
n_val_users = max(1, int(train_df['user_id'].nunique() * 0.2))
val_users = set(train_df['user_id'].unique()[-n_val_users:])
val_mask = train_df['user_id'].isin(val_users).values
trn_mask = ~val_mask

y_all = train_df['label'].values.astype(np.int32)
y_trn, y_val = y_all[trn_mask], y_all[val_mask]
g_trn = train_df[trn_mask].groupby('user_id', sort=False).size().values
g_val = train_df[val_mask].groupby('user_id', sort=False).size().values

# Materialize X_all → COLD subset
print(f"{elapsed()} Materializing X (float32, {len(train_df):,}×{len(COLD_FEATURE_COLS)}) …")
X_all = np.empty((len(train_df), len(COLD_FEATURE_COLS)), dtype=np.float32)
for j, c in enumerate(COLD_FEATURE_COLS):
    X_all[:, j] = train_df[c].to_numpy(dtype=np.float32, copy=False)
del train_df; gc.collect()

X_trn, X_val = X_all[trn_mask], X_all[val_mask]
del X_all, y_all, trn_mask, val_mask; gc.collect()

trn_data = lgb.Dataset(X_trn, label=y_trn, group=g_trn,
                        feature_name=COLD_FEATURE_COLS, free_raw_data=True)
trn_data.construct()
del X_trn, y_trn, g_trn; gc.collect()
val_data = lgb.Dataset(X_val, label=y_val, group=g_val,
                        feature_name=COLD_FEATURE_COLS, free_raw_data=True)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"{elapsed()} Training Cold-Track LightGBM …")
params = dict(LGBM_PARAMS); params['seed'] = 42
cbs = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]
model = lgb.train(params, trn_data,
                  num_boost_round=LGBM_PARAMS['n_estimators'],
                  valid_sets=[val_data], callbacks=cbs)
print(f"{elapsed()} Best iter: {model.best_iteration}")
model.save_model(f"{CACHE_DIR}/cold_track_model.txt")

fi = pd.Series(model.feature_importance(importance_type='gain'),
                index=COLD_FEATURE_COLS).sort_values(ascending=False)
print(f"\n{elapsed()} Cold-Track Top 15 FI:")
print(fi.head(15).to_string())
del X_val, y_val, g_val, trn_data, val_data; gc.collect()

# ── Build cold candidate pool (top-1000 items by composite score) ─────────────
print(f"\n{elapsed()} Building cold candidate pool (top-1000 items) …")
items_feat = pd.read_parquet(f"{CACHE_DIR}/features_test.parquet",
                              columns=['item_id'] + COLD_FEATURE_COLS)
items_feat = items_feat.drop_duplicates('item_id').reset_index(drop=True)
print(f"{elapsed()} Unique items in test: {len(items_feat):,}")

# Composite score for INITIAL pool selection (broad popular by interaction volume)
# Then cold-track model RE-RANKS within this pool
# Pool size = 1000 (~10K candidates if we don't dedup)
pop = pd.read_parquet(f"{CACHE_DIR}/popular_items.parquet")
pop_agg = pop.groupby('item_id').agg(trend_pos=('trend_pos','sum')).reset_index()
items_feat = items_feat.merge(pop_agg, on='item_id', how='left')
items_feat['trend_pos'] = items_feat['trend_pos'].fillna(0)
items_feat['_pool_score'] = items_feat['trend_pos']  # initial popularity rank
pool = items_feat.sort_values('_pool_score', ascending=False).head(1000).copy()
print(f"{elapsed()} Pool size: {len(pool):,}")

# Score pool with cold-track model
X_pool = pool[COLD_FEATURE_COLS].fillna(0).astype(np.float32).values
pool_scores = model.predict(X_pool, num_iteration=model.best_iteration)
pool['cold_score'] = pool_scores
pool = pool.sort_values('cold_score', ascending=False)
cold_top10 = pool.head(10)['item_id'].tolist()
print(f"\n{elapsed()} cold_top10:")
for i, iid in enumerate(cold_top10, 1):
    sc = pool[pool['item_id']==iid]['cold_score'].iloc[0]
    pos = pool[pool['item_id']==iid]['trend_pos'].iloc[0]
    print(f"  {i:2d}. {iid[:16]}…  cold_score={sc:.4f}  trend_pos={pos:.0f}")

# Save
with open(f"{CACHE_DIR}/cold_top10.pkl", 'wb') as f:
    pickle.dump(cold_top10, f)
print(f"\n{elapsed()} Saved: {CACHE_DIR}/cold_top10.pkl")
print(f"{elapsed()} DONE")
