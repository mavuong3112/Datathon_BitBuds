"""
Step 7b: CatBoost warm-track ensemble — Stage 9.

Trains CatBoost YetiRank on same features_train.parquet as 07_rerank LightGBM.
Outputs catboost_predictions.npy aligned with features_test row order.

Will be blended in 08_submit.py with LightGBM predictions:
  final_score = BLEND_WEIGHT * lgbm_norm + (1-BLEND_WEIGHT) * catboost_norm
"""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

try:
    import catboost as cb
except ImportError:
    print("ERROR: catboost not installed. Run: pip install catboost")
    raise SystemExit(1)

print(f"{elapsed()} CatBoost {cb.__version__}")

# ── Load features (same as 07_rerank) ─────────────────────────────────────────
print(f"{elapsed()} Loading train features …")
train_df = pd.read_parquet(f"{CACHE_DIR}/features_train.parquet")
skip_cols = {'user_id','item_id','label'}
FEATURE_COLS = [c for c in train_df.columns if c not in skip_cols]
print(f"{elapsed()} Features: {len(FEATURE_COLS)}, Train: {len(train_df):,} rows, "
      f"pos rate {train_df['label'].mean():.4f}")

# Cast to float32 + replace inf
for c in FEATURE_COLS:
    train_df[c] = train_df[c].astype(np.float32).replace([np.inf, -np.inf], np.nan)

# CatBoost requires group_id CONSECUTIVE → sort train_df by user_id first
print(f"{elapsed()} Sorting train_df by user_id for CatBoost group_id requirement …")
train_df = train_df.sort_values('user_id', kind='stable').reset_index(drop=True)

# Train/val split (last 20% unique users in sorted order)
unique_users_sorted = train_df['user_id'].drop_duplicates().tolist()
n_val_users = max(1, int(len(unique_users_sorted) * 0.2))
val_users = set(unique_users_sorted[-n_val_users:])
val_mask = train_df['user_id'].isin(val_users).values
trn_mask = ~val_mask

# Group_id (contiguous after sort)
y_all = train_df['label'].values.astype(np.int32)
g_all = train_df['user_id'].astype('category').cat.codes.values.astype(np.int32)

# Build X
print(f"{elapsed()} Materializing X (float32, {len(train_df):,}×{len(FEATURE_COLS)}) …")
X_all = np.empty((len(train_df), len(FEATURE_COLS)), dtype=np.float32)
for j, c in enumerate(FEATURE_COLS):
    X_all[:, j] = train_df[c].to_numpy(dtype=np.float32, copy=False)
del train_df; gc.collect()

X_trn, X_val = X_all[trn_mask], X_all[val_mask]
y_trn, y_val = y_all[trn_mask], y_all[val_mask]
g_trn, g_val = g_all[trn_mask], g_all[val_mask]
del X_all, y_all, g_all, trn_mask, val_mask; gc.collect()
print(f"{elapsed()} X_trn={X_trn.shape}, X_val={X_val.shape}")

# ── Train CatBoost YetiRank ───────────────────────────────────────────────────
print(f"{elapsed()} Training CatBoost YetiRank …")

# Try GPU first, fall back to CPU
device = 'GPU'
try:
    import subprocess
    # quick GPU test
    pool = cb.Pool(data=X_trn[:1000], label=y_trn[:1000], group_id=g_trn[:1000])
    test_model = cb.CatBoostRanker(iterations=2, task_type='GPU', devices='0', verbose=0)
    test_model.fit(pool)
    del pool, test_model
    print(f"{elapsed()} GPU available")
except Exception as e:
    print(f"{elapsed()} GPU failed ({e}), using CPU")
    device = 'CPU'

trn_pool = cb.Pool(data=X_trn, label=y_trn, group_id=g_trn)
val_pool = cb.Pool(data=X_val, label=y_val, group_id=g_val)

model = cb.CatBoostRanker(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='YetiRank',
    eval_metric='NDCG:top=10',
    task_type=device,
    devices='0' if device == 'GPU' else None,
    od_type='Iter',
    od_wait=50,
    random_seed=42,
    verbose=50,
)
model.fit(trn_pool, eval_set=val_pool)
print(f"{elapsed()} Best iteration: {model.best_iteration_}")

# Save model + FI
model.save_model(f"{CACHE_DIR}/catboost_ranker.cbm")
fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\n{elapsed()} CatBoost Top 15 FI:")
print(fi.head(15).to_string())
del X_trn, X_val, y_trn, y_val, g_trn, g_val, trn_pool, val_pool; gc.collect()

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

print(f"{elapsed()} CatBoost predicting …")
cat_scores = model.predict(X_test)
print(f"{elapsed()} Predictions: {cat_scores.shape}  mean={cat_scores.mean():.4f}  std={cat_scores.std():.4f}")

# Save predictions aligned with features_test row order
out = pd.DataFrame({
    'user_id': test_uids,
    'item_id': test_iids,
    'catboost_score': cat_scores.astype(np.float32),
})
out.to_parquet(f"{CACHE_DIR}/catboost_predictions.parquet", index=False)
print(f"{elapsed()} Saved: {CACHE_DIR}/catboost_predictions.parquet ({len(out):,} rows)")
print(f"{elapsed()} DONE")
