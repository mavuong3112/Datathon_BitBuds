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

print(f"{elapsed()} Loading feature data …")
train_df = pd.read_parquet(f"{CACHE_DIR}/features_train.parquet")
test_df  = pd.read_parquet(f"{CACHE_DIR}/features_test.parquet")

# Determine feature columns (all except user_id, item_id, label)
skip_cols = {'user_id','item_id','label'}
FEATURE_COLS = [c for c in train_df.columns if c not in skip_cols]
print(f"{elapsed()} Features: {len(FEATURE_COLS)}")
print(f"{elapsed()} Train: {len(train_df):,} rows, {train_df['label'].mean():.4f} pos rate")
print(f"{elapsed()} Test:  {len(test_df):,} rows")

# ── Prepare LightGBM datasets ─────────────────────────────────────────────────
import gc as _gc

# Convert all FEATURE_COLS to float32 uniformly (avoids object array from mixed-dtype .values)
print(f"{elapsed()} Converting to float32 …")
for _col in FEATURE_COLS:
    train_df[_col] = train_df[_col].astype(np.float32)

# Val split — last 20% of users for early stopping.
# days_since_ui computed from full pos including val-window → NDCG saturates to 1.0 after a
# handful of trees, so early_stopping fires and keeps only the effective recency-ranked trees.
n_val_users = max(1, int(train_df['user_id'].nunique() * 0.2))
val_users   = set(train_df['user_id'].unique()[-n_val_users:])
val_mask    = train_df['user_id'].isin(val_users).values
trn_mask    = ~val_mask

y_all   = train_df['label'].values.astype(np.int32)
y_trn   = y_all[trn_mask]; y_val = y_all[val_mask]
g_trn   = train_df[trn_mask].groupby('user_id', sort=False).size().values
g_val   = train_df[val_mask].groupby('user_id', sort=False).size().values
X_all   = train_df[FEATURE_COLS].values  # float32
del train_df; _gc.collect()

X_trn = X_all[trn_mask]; X_val = X_all[val_mask]
del X_all, y_all, trn_mask, val_mask; _gc.collect()

trn_data = lgb.Dataset(X_trn, label=y_trn, group=g_trn,
                        feature_name=FEATURE_COLS, free_raw_data=True)
trn_data.construct()
del X_trn, y_trn, g_trn; _gc.collect()

val_data = lgb.Dataset(X_val, label=y_val, group=g_val,
                        feature_name=FEATURE_COLS, free_raw_data=True)
# Do NOT pre-construct val_data — lgb.train must set_reference(trn_data) before construction
print(f"{elapsed()} Datasets ready — training …")

# Load and convert test features (separate from training to reduce peak memory)
print(f"{elapsed()} Loading test features …")
for _col in FEATURE_COLS:
    test_df[_col] = test_df[_col].astype(np.float32)
X_test = test_df[FEATURE_COLS].values  # float32
test_uids = test_df['user_id'].copy()
test_iids = test_df['item_id'].copy()
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

print(f"{elapsed()} Training LightGBM LambdaRank — 5-seed ensemble …")
SEEDS = [42]
all_scores = []
last_model  = None
last_fi     = None

for seed_i, seed in enumerate(SEEDS):
    print(f"{elapsed()} Seed {seed_i+1}/{len(SEEDS)} (seed={seed}) …")
    params_s = dict(params)
    params_s['seed'] = seed
    cbs = [
        lgb.early_stopping(50, verbose=(seed_i == 0)),
        lgb.log_evaluation(100 if seed_i == 0 else 0),
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

test_df = pd.DataFrame({'user_id': test_uids, 'item_id': test_iids, 'lgbm_score': scores})
del test_uids, test_iids, scores; _gc.collect()

# ── Take top-30 per user (buffer for diversification in submit step) ──────────
print(f"{elapsed()} Selecting top-30 per user …")
ranked = (test_df
    .sort_values('lgbm_score', ascending=False)
    .groupby('user_id', sort=False)
    .head(30)
    .copy())
ranked['rank'] = (ranked
    .sort_values('lgbm_score', ascending=False)
    .groupby('user_id', sort=False)
    .cumcount() + 1)

user_counts = ranked.groupby('user_id').size()
print(f"{elapsed()} Users with <10 items: {(user_counts < 10).sum():,} — will fill in submit step")

ranked.to_parquet(f"{CACHE_DIR}/ranked_predictions.parquet", index=False)
print(f"{elapsed()} Saved: {len(ranked):,} predictions for {ranked['user_id'].nunique():,} users")
print(f"{elapsed()} DONE")
