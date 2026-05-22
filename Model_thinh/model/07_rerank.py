"""
Step 7: LightGBM LambdaRank re-ranker (GPU).
Trains on temporal split, re-ranks candidates → top-30 per user.
Multi-seed ensembling (5 seeds) for variance reduction.
Outputs:
  cache/ranked_predictions.parquet — top-30 item_ids per user with scores
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

import gc as _gc

# ── Stage 1: Load train_df only, build datasets, then free it ─────────────────
print(f"{elapsed()} Loading train features …")
train_df = pd.read_parquet(f"{CACHE_DIR}/features_train.parquet")
skip_cols = {'user_id','item_id','label'}
FEATURE_COLS = [c for c in train_df.columns if c not in skip_cols]
print(f"{elapsed()} Features: {len(FEATURE_COLS)}")
print(f"{elapsed()} Train: {len(train_df):,} rows, {train_df['label'].mean():.4f} pos rate")

# Per-column float32 + inf→NaN
print(f"{elapsed()} Converting train to float32 + inf→NaN …")
for _col in FEATURE_COLS:
    train_df[_col] = train_df[_col].astype(np.float32)
    train_df[_col] = train_df[_col].replace([np.inf, -np.inf], np.nan)

# User-based val split (last 20% users)
n_val_users = max(1, int(train_df['user_id'].nunique() * 0.2))
val_users   = set(train_df['user_id'].unique()[-n_val_users:])
val_mask    = train_df['user_id'].isin(val_users).values
trn_mask    = ~val_mask

y_all   = train_df['label'].values.astype(np.int32)
y_trn   = y_all[trn_mask]; y_val = y_all[val_mask]
g_trn   = train_df[trn_mask].groupby('user_id', sort=False).size().values
g_val   = train_df[val_mask].groupby('user_id', sort=False).size().values

# Materialize X as float32 directly — avoid .values intermediate copy that triggers OOM
print(f"{elapsed()} Materializing X (np.float32, {len(train_df):,}×{len(FEATURE_COLS)}) …")
X_all = np.empty((len(train_df), len(FEATURE_COLS)), dtype=np.float32)
for j, c in enumerate(FEATURE_COLS):
    X_all[:, j] = train_df[c].to_numpy(dtype=np.float32, copy=False)
del train_df; _gc.collect()

X_trn = X_all[trn_mask]; X_val = X_all[val_mask]
del X_all, y_all, trn_mask, val_mask; _gc.collect()

trn_data = lgb.Dataset(X_trn, label=y_trn, group=g_trn,
                        feature_name=FEATURE_COLS, free_raw_data=True)
trn_data.construct()
del X_trn, y_trn, g_trn; _gc.collect()

val_data = lgb.Dataset(X_val, label=y_val, group=g_val,
                        feature_name=FEATURE_COLS, free_raw_data=True)
print(f"{elapsed()} Datasets ready — training …")

# ── Stage 2: Load test_df AFTER train is freed ────────────────────────────────
print(f"{elapsed()} Loading test features …")
test_df  = pd.read_parquet(f"{CACHE_DIR}/features_test.parquet")
print(f"{elapsed()} Test:  {len(test_df):,} rows")
for _col in FEATURE_COLS:
    test_df[_col] = test_df[_col].astype(np.float32)
    test_df[_col] = test_df[_col].replace([np.inf, -np.inf], np.nan)
test_uids  = test_df['user_id'].copy()
test_iids  = test_df['item_id'].copy()
test_blend = test_df['blend_score'].astype(np.float32).values
X_test = np.empty((len(test_df), len(FEATURE_COLS)), dtype=np.float32)
for j, c in enumerate(FEATURE_COLS):
    X_test[:, j] = test_df[c].to_numpy(dtype=np.float32, copy=False)
del test_df; _gc.collect()

# ── Train LightGBM ────────────────────────────────────────────────────────────
params = dict(LGBM_PARAMS)
# Try GPU, fallback to CPU if GPU not available
try:
    import subprocess, json
    result = subprocess.run(['python','-c','import lightgbm as lgb; print(lgb.__version__)'],
                           capture_output=True, text=True)
    print(f"{elapsed()} LightGBM {result.stdout.strip()}")
except:
    pass

# Stage 12: BIGGER LightGBM — 10 seeds (variance reduction), depth 8, num_leaves 95
print(f"{elapsed()} Training LightGBM LambdaRank — 10-seed ensemble (Stage 12 BIGGER) …")
SEEDS = [42, 123, 456, 789, 2024, 31, 314, 1729, 31415, 6710]
all_scores = []
last_model  = None
last_fi     = None

# Stage 12: bigger model config
params['max_depth']   = 8   # was 7
params['num_leaves']  = 95  # was 63
print(f"{elapsed()} Stage 12 LGBM config: depth={params['max_depth']}, leaves={params['num_leaves']}, seeds={len(SEEDS)}")
print(f"{elapsed()} Training with early stopping (patience=50), max {LGBM_PARAMS['n_estimators']} trees …")
for seed_i, seed in enumerate(SEEDS):
    print(f"{elapsed()} Seed {seed_i+1}/{len(SEEDS)} (seed={seed}) …")
    params_s = dict(params)
    params_s['seed'] = seed
    cbs = [
        lgb.early_stopping(50, verbose=(seed_i == 0)),
        lgb.log_evaluation(50 if seed_i == 0 else 0),
    ]
    try:
        m = lgb.train(params_s, trn_data,
                      num_boost_round=LGBM_PARAMS['n_estimators'],
                      valid_sets=[val_data], callbacks=cbs)
    except Exception as e:
        print(f"  GPU failed ({e}), falling back to CPU …")
        ps = dict(params_s); ps['device'] = 'cpu'
        m = lgb.train(ps, trn_data,
                      num_boost_round=LGBM_PARAMS['n_estimators'],
                      valid_sets=[val_data], callbacks=cbs)
    print(f"{elapsed()}   best_iter={m.best_iteration}")
    all_scores.append(m.predict(X_test, num_iteration=m.best_iteration))
    last_model = m
    last_fi = pd.Series(m.feature_importance(importance_type='gain'), index=FEATURE_COLS)
    _gc.collect()

last_model.save_model(f"{CACHE_DIR}/lgbm_ranker.txt")
print(f"{elapsed()} Saved last-seed model")

# Feature importance (from last seed — representative)
fi = last_fi.sort_values(ascending=False)
print(f"\n{elapsed()} Top 15 features (last seed):")
print(fi.head(15).to_string())

# ── Ensemble scores (average across seeds) ────────────────────────────────────
print(f"\n{elapsed()} Averaging {len(SEEDS)} seed predictions …")
scores = np.mean(all_scores, axis=0)
del all_scores, X_test; _gc.collect()

test_df = pd.DataFrame({
    'user_id': test_uids,
    'item_id': test_iids,
    'lgbm_score': scores,
    'blend_score': test_blend,
})
del test_uids, test_iids, scores, test_blend; _gc.collect()

# ── Take top-30 per user (buffer for diversification in submit step) ──────────
# Use blend_score as tiebreaker — critical for cold users whose lgbm_score ties at
# the single-tree leaf value (all days_since_ui=999 → same leaf).
print(f"{elapsed()} Selecting top-30 per user (lgbm_score, blend_score tiebreaker) …")
ranked = (test_df
    .sort_values(['lgbm_score','blend_score'], ascending=[False, False])
    .groupby('user_id', sort=False)
    .head(30)
    .copy())
ranked['rank'] = (ranked
    .sort_values(['lgbm_score','blend_score'], ascending=[False, False])
    .groupby('user_id', sort=False)
    .cumcount() + 1)

user_counts = ranked.groupby('user_id').size()
print(f"{elapsed()} Users with <10 items: {(user_counts < 10).sum():,} — will fill in submit step")

ranked.to_parquet(f"{CACHE_DIR}/ranked_predictions.parquet", index=False)
print(f"{elapsed()} Saved: {len(ranked):,} predictions for {ranked['user_id'].nunique():,} users")
print(f"{elapsed()} DONE")
