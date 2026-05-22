"""
Step 2b: ALS-only — loads minimal data, trains implicit ALS, saves ease_candidates.
Run this after BPR (02_als_ease.py) has already saved als_candidates.parquet.
"""
import sys, os, time, pickle, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import scipy.sparse as sp
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"
def mem_mb():
    import psutil; return psutil.Process().memory_info().rss / 1e6

OUT_PATH = f"{CACHE_DIR}/ease_candidates.parquet"
if os.path.exists(OUT_PATH):
    df_check = pd.read_parquet(OUT_PATH)
    if len(df_check) > 0:
        print(f"{elapsed()} [SKIP] ease_candidates.parquet already exists ({len(df_check):,} rows)")
        raise SystemExit(0)
    del df_check
    os.remove(OUT_PATH)
    print(f"{elapsed()} Removed empty ease_candidates.parquet — rerunning")

print(f"{elapsed()} Loading mappings …")
with open(f"{CACHE_DIR}/mappings.pkl", 'rb') as f:
    maps = pickle.load(f)
user2idx = maps['user2idx']
item2idx = maps['item2idx']
idx2item = maps['idx2item']
n_users, n_items = len(user2idx), len(item2idx)
print(f"{elapsed()} users={n_users:,}  items={n_items:,}  [RAM:{mem_mb():.0f}MB]")

# Build sparse matrix via row-group streaming — minimal peak memory
print(f"{elapsed()} Building sparse matrix (row-group streaming) …")
import pyarrow.parquet as pq

POS_PATH = f"{CACHE_DIR}/user_item_pos.parquet"
train_end_dt = pd.Timestamp(TRAIN_END)

pf = pq.ParquetFile(POS_PATH)
all_ui_u, all_ii_i, all_ww = [], [], []

for batch in pf.iter_batches(batch_size=500_000,
                              columns=['user_id','item_id','pos_count','last_ts']):
    df = batch.to_pandas()
    df['last_ts']  = pd.to_datetime(df['last_ts'])
    df['days_ago'] = (train_end_dt - df['last_ts']).dt.days.clip(lower=0)
    df['weight']   = (df['pos_count'] * np.exp(-RECENCY_DECAY * df['days_ago'])).clip(lower=1.0)

    ui = df['user_id'].map(user2idx)
    ii = df['item_id'].map(item2idx)
    valid = ui.notna() & ii.notna()
    all_ui_u.append(ui[valid].values.astype(np.int32))
    all_ii_i.append(ii[valid].values.astype(np.int32))
    all_ww.append(df.loc[valid, 'weight'].values.astype(np.float32))
    del df, ui, ii, valid

ui_u = np.concatenate(all_ui_u).astype(np.int32)
ii_i = np.concatenate(all_ii_i).astype(np.int32)
ww   = np.concatenate(all_ww).astype(np.float32)
del all_ui_u, all_ii_i, all_ww; gc.collect()

user_item_csr = sp.csr_matrix((ww, (ui_u, ii_i)), shape=(n_users, n_items))
item_user_csr = user_item_csr.T.tocsr()
del ui_u, ii_i, ww, user_item_csr; gc.collect()
print(f"{elapsed()} Sparse matrix built ({item_user_csr.nnz:,} nnz)  [RAM:{mem_mb():.0f}MB]")

# Load test users
test = pd.read_parquet(TEST_FILE)
test_users    = set(test['user_id'].tolist())
test_in_train = [u for u in test_users if u in user2idx]
print(f"{elapsed()} warm_test={len(test_in_train):,}")
del test; gc.collect()

# ── ALS model ─────────────────────────────────────────────────────────────────
ALS_FACTORS  = 128   # 256 needs 675MB contiguous — fragmentation causes OOM; 128=337MB works
ALS_ITER     = 20
ALS_ALPHA    = 40
CHUNK        = 1000

ALS_MODEL_U = f"{CACHE_DIR}/als_model/user_factors_als.npy"
ALS_MODEL_I = f"{CACHE_DIR}/als_model/item_factors_als.npy"
os.makedirs(f"{CACHE_DIR}/als_model", exist_ok=True)

try:
    import implicit
    if os.path.exists(ALS_MODEL_U) and os.path.exists(ALS_MODEL_I):
        print(f"{elapsed()} [RESUME] Loading saved ALS weights …")
        u_factors = np.load(ALS_MODEL_U)
        i_factors = np.load(ALS_MODEL_I)
    else:
        print(f"{elapsed()} Training ALS (factors={ALS_FACTORS}, iter={ALS_ITER}) …")
        als = implicit.als.AlternatingLeastSquares(
            factors=ALS_FACTORS, regularization=0.01,
            iterations=ALS_ITER, alpha=ALS_ALPHA,
            use_gpu=False, num_threads=4, random_state=42)
        als.fit(item_user_csr)
        u_factors = als.user_factors
        i_factors = als.item_factors
        np.save(ALS_MODEL_U, u_factors)
        np.save(ALS_MODEL_I, i_factors)
        del als; gc.collect()
    print(f"{elapsed()} ALS ready  [RAM:{mem_mb():.0f}MB]")

    del item_user_csr; gc.collect()

    print(f"{elapsed()} Generating ALS candidates …")
    rows_als = []
    for i in range(0, len(test_in_train), CHUNK):
        batch_users = test_in_train[i:i + CHUNK]
        batch_idx   = [user2idx[u] for u in batch_users]
        u_emb       = u_factors[batch_idx]
        scores      = u_emb @ i_factors.T
        top_ids     = np.argpartition(scores, -N_EASE, axis=1)[:, -N_EASE:]
        for j, uid in enumerate(batch_users):
            top_s = top_ids[j][np.argsort(scores[j][top_ids[j]])[::-1]]
            for rank, iid in enumerate(top_s):
                rows_als.append({'user_id': uid, 'item_id': idx2item[int(iid)],
                                 'ease_score': float(scores[j][iid]),
                                 'ease_rank': rank + 1})
        if (i // CHUNK) % 20 == 0:
            print(f"{elapsed()} ALS inference {i//CHUNK+1}/{(len(test_in_train)-1)//CHUNK+1}")

    df_als = pd.DataFrame(rows_als)
    print(f"{elapsed()} ALS candidates: {len(df_als):,}")

except Exception as e:
    import traceback
    print(f"{elapsed()} ALS failed: {e}")
    traceback.print_exc()
    df_als = pd.DataFrame(columns=['user_id','item_id','ease_score','ease_rank'])

df_als.to_parquet(OUT_PATH, index=False)
print(f"{elapsed()} Saved ease_candidates.parquet  [RAM:{mem_mb():.0f}MB]")
print(f"{elapsed()} DONE")
