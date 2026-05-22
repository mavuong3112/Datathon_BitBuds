"""Quick XGBoost prediction from saved model (avoid retraining)."""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

# Load model
print(f"{elapsed()} Loading saved XGBoost model …")
model = xgb.Booster()
model.load_model(f"{CACHE_DIR}/xgboost_ranker.json")
model.set_param({'device': 'cpu'})
print(f"{elapsed()} Model loaded, best_iteration: {model.best_iteration}")

# Load test features
print(f"{elapsed()} Loading test features …")
test_df = pd.read_parquet(f"{CACHE_DIR}/features_test.parquet")
FEATURE_COLS = [c for c in test_df.columns if c not in {'user_id','item_id','label'}]
for c in FEATURE_COLS:
    test_df[c] = test_df[c].astype(np.float32).replace([np.inf, -np.inf], np.nan)

print(f"{elapsed()} Materializing X_test ({len(test_df):,}×{len(FEATURE_COLS)}) …")
X_test = np.empty((len(test_df), len(FEATURE_COLS)), dtype=np.float32)
for j, c in enumerate(FEATURE_COLS):
    X_test[:, j] = test_df[c].to_numpy(dtype=np.float32, copy=False)

test_uids = test_df['user_id'].copy()
test_iids = test_df['item_id'].copy()
del test_df; gc.collect()

print(f"{elapsed()} XGBoost predicting (chunked CPU, 2M rows/chunk) …")
CHUNK = 2_000_000
xgb_scores = np.empty(len(X_test), dtype=np.float32)
for start in range(0, len(X_test), CHUNK):
    end = min(start + CHUNK, len(X_test))
    dchunk = xgb.DMatrix(X_test[start:end], feature_names=FEATURE_COLS)
    xgb_scores[start:end] = model.predict(dchunk, iteration_range=(0, model.best_iteration + 1))
    print(f"{elapsed()}   predicted {end:,}/{len(X_test):,}")

print(f"{elapsed()} Predictions: mean={xgb_scores.mean():.4f}  std={xgb_scores.std():.4f}")
out = pd.DataFrame({
    'user_id': test_uids,
    'item_id': test_iids,
    'xgboost_score': xgb_scores.astype(np.float32),
})
out.to_parquet(f"{CACHE_DIR}/xgboost_predictions.parquet", index=False)
print(f"{elapsed()} Saved {CACHE_DIR}/xgboost_predictions.parquet")
