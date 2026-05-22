"""
Step 7d: XGBoost ranker for Stage 12 ensemble.

Trains XGBoost rank:ndcg on same features_train.parquet.
Output: xgboost_predictions.parquet aligned with features_test.

Blended in 08_submit:
  final_score = 0.4*LGBM + 0.3*CatBoost + 0.3*XGBoost (per-user normalized)
"""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    raise SystemExit(1)

print(f"{elapsed()} XGBoost {xgb.__version__}")

# ── Load features ─────────────────────────────────────────────────────────────
print(f"{elapsed()} Loading train features …")
train_df = pd.read_parquet(f"{CACHE_DIR}/features_train.parquet")
skip_cols = {'user_id','item_id','label'}
FEATURE_COLS = [c for c in train_df.columns if c not in skip_cols]
print(f"{elapsed()} Features: {len(FEATURE_COLS)}, Train: {len(train_df):,} rows, "
      f"pos rate {train_df['label'].mean():.4f}")

for c in FEATURE_COLS:
    train_df[c] = train_df[c].astype(np.float32).replace([np.inf, -np.inf], np.nan)

# XGBoost requires CONSECUTIVE group_id per query → sort by user_id
print(f"{elapsed()} Sorting train_df by user_id for XGBoost qid requirement …")
train_df = train_df.sort_values('user_id', kind='stable').reset_index(drop=True)

# Train/val split (last 20% unique users in sorted order)
unique_users_sorted = train_df['user_id'].drop_duplicates().tolist()
n_val_users = max(1, int(len(unique_users_sorted) * 0.2))
val_users = set(unique_users_sorted[-n_val_users:])
val_mask = train_df['user_id'].isin(val_users).values
trn_mask = ~val_mask

y_all = train_df['label'].values.astype(np.int32)
# Use user_id codes as group_id (consecutive after sort)
g_all = train_df['user_id'].astype('category').cat.codes.values.astype(np.int32)

# Materialize X
print(f"{elapsed()} Materializing X (float32, {len(train_df):,}×{len(FEATURE_COLS)}) …")
X_all = np.empty((len(train_df), len(FEATURE_COLS)), dtype=np.float32)
for j, c in enumerate(FEATURE_COLS):
    X_all[:, j] = train_df[c].to_numpy(dtype=np.float32, copy=False)
del train_df; gc.collect()

X_trn, X_val = X_all[trn_mask], X_all[val_mask]
y_trn, y_val = y_all[trn_mask], y_all[val_mask]
g_trn, g_val = g_all[trn_mask], g_all[val_mask]
del X_all, y_all, g_all, trn_mask, val_mask; gc.collect()

# Compute group sizes (contiguous user blocks)
def group_sizes(arr):
    sizes = []
    cur = arr[0]; cnt = 1
    for v in arr[1:]:
        if v == cur:
            cnt += 1
        else:
            sizes.append(cnt); cur = v; cnt = 1
    sizes.append(cnt)
    return sizes
grp_trn_sizes = group_sizes(g_trn)
grp_val_sizes = group_sizes(g_val)
print(f"{elapsed()} X_trn={X_trn.shape} ({len(grp_trn_sizes):,} queries), X_val={X_val.shape}")

# ── Train XGBoost ranker ──────────────────────────────────────────────────────
print(f"{elapsed()} Training XGBoost rank:ndcg …")
dtrn = xgb.DMatrix(X_trn, label=y_trn, group=grp_trn_sizes,
                    feature_names=FEATURE_COLS)
dval = xgb.DMatrix(X_val, label=y_val, group=grp_val_sizes,
                    feature_names=FEATURE_COLS)

base_params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'eta': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'max_bin': 128,
    'device': os.environ.get('XGB_DEVICE', 'cpu'),
    'nthread': -1,
}

# Stage 16: 3-seed bagging for variance reduction
SEEDS = [int(s) for s in os.environ.get('XGB_SEEDS', '42,123,789').split(',')]
print(f"{elapsed()} Bagging seeds: {SEEDS}")

# Free X arrays AFTER DMatrix construction — release ~9GB before training starts
del X_trn, X_val; gc.collect()

models = []
for si, seed in enumerate(SEEDS):
    print(f"{elapsed()}   Seed {si+1}/{len(SEEDS)} (seed={seed}) …")
    params = dict(base_params); params['seed'] = seed
    evals_result = {}
    m = xgb.train(params, dtrn,
                  num_boost_round=500,
                  evals=[(dval, 'val')],
                  evals_result=evals_result,
                  early_stopping_rounds=50,
                  verbose_eval=(50 if si == 0 else 0))
    print(f"{elapsed()}     best_iter={m.best_iteration}")
    models.append(m)

# Save last-seed model + FI (representative)
models[-1].save_model(f"{CACHE_DIR}/xgboost_ranker.json")
fi = pd.Series(models[-1].get_score(importance_type='gain'), name='gain').sort_values(ascending=False)
print(f"\n{elapsed()} XGBoost Top 15 FI (last seed):")
print(fi.head(15).to_string())
del y_trn, y_val, g_trn, g_val, dtrn, dval; gc.collect()

# ── Predict on test ───────────────────────────────────────────────────────────
print(f"\n{elapsed()} Loading test features …")
test_df = pd.read_parquet(f"{CACHE_DIR}/features_test.parquet")
for c in FEATURE_COLS:
    test_df[c] = test_df[c].astype(np.float32).replace([np.inf, -np.inf], np.nan)

print(f"{elapsed()} Materializing X_test …")
X_test = np.empty((len(test_df), len(FEATURE_COLS)), dtype=np.float32)
for j, c in enumerate(FEATURE_COLS):
    X_test[:, j] = test_df[c].to_numpy(dtype=np.float32, copy=False)

test_uids = test_df['user_id'].copy()
test_iids = test_df['item_id'].copy()
del test_df; gc.collect()

print(f"{elapsed()} XGBoost predicting (chunked, averaging {len(models)} seeds) …")
for m in models:
    m.set_param({'device': 'cpu'})
CHUNK = 2_000_000
xgb_scores_sum = np.zeros(len(X_test), dtype=np.float64)
for start in range(0, len(X_test), CHUNK):
    end = min(start + CHUNK, len(X_test))
    dchunk = xgb.DMatrix(X_test[start:end], feature_names=FEATURE_COLS)
    chunk_pred = np.zeros(end - start, dtype=np.float64)
    for m in models:
        chunk_pred += m.predict(dchunk, iteration_range=(0, m.best_iteration + 1))
    xgb_scores_sum[start:end] = chunk_pred
    if (start // CHUNK) % 3 == 0:
        print(f"{elapsed()}   predicted {end:,}/{len(X_test):,}")
xgb_scores = (xgb_scores_sum / len(models)).astype(np.float32)
print(f"{elapsed()} Bagged predictions: {xgb_scores.shape}  mean={xgb_scores.mean():.4f}  std={xgb_scores.std():.4f}")

out = pd.DataFrame({
    'user_id': test_uids,
    'item_id': test_iids,
    'xgboost_score': xgb_scores.astype(np.float32),
})
out.to_parquet(f"{CACHE_DIR}/xgboost_predictions.parquet", index=False)
print(f"{elapsed()} Saved: {CACHE_DIR}/xgboost_predictions.parquet ({len(out):,} rows)")
print(f"{elapsed()} DONE")
