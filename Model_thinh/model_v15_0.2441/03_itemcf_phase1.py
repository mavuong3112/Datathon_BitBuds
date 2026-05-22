"""
Phase 1 of Item-CF: Stream parquet → build 54K test-user embeddings → save → EXIT.
Exiting cleanly releases all 3 GB RAM so Phase 2 (GPU) gets a fresh heap.

Stage 8: env RETRIEVER_MODE=train uses pos_train, mappings_train, als_model_train.
"""
import sys, os, time, pickle, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"
def mem_mb():
    import psutil; return psutil.Process().memory_info().rss / 1e6

MODE = os.environ.get('RETRIEVER_MODE', 'full').lower()
SUFFIX = {'train': '_train', 'lukewarm': '_lukewarm', 'full': ''}[MODE]
OUT_EMB  = f"{CACHE_DIR}/test_user_emb{SUFFIX}.npy"
OUT_META = f"{CACHE_DIR}/test_user_meta{SUFFIX}.pkl"
MAPPINGS_PATH = f"{CACHE_DIR}/mappings{SUFFIX}.pkl"
POS_PARQUET   = f"{CACHE_DIR}/user_item_pos{SUFFIX}.parquet"
BPR_ITEM_EMB  = f"{CACHE_DIR}/als_model{SUFFIX}/item_factors_bpr.npy"

if os.path.exists(OUT_EMB) and os.path.exists(OUT_META):
    print(f"{elapsed()} [SKIP] Phase 1 outputs already exist", flush=True)
    raise SystemExit(0)

# ── Load mappings ─────────────────────────────────────────────────────────────
with open(MAPPINGS_PATH, 'rb') as f:
    maps = pickle.load(f)
user2idx = maps['user2idx']
item2idx = maps['item2idx']

test = pd.read_parquet(TEST_FILE)
test_in_train = [u for u in test['user_id'].tolist() if u in user2idx]
n_test = len(test_in_train)
test_uidx = [user2idx[u] for u in test_in_train]
test_uidx_set = set(test_uidx)
test_uidx_rank = {uidx: r for r, uidx in enumerate(test_uidx)}
print(f"{elapsed()} warm_test={n_test:,}  [RAM:{mem_mb():.0f}MB]", flush=True)

# ── Load item embeddings via mmap (no malloc) ─────────────────────────────────
item_emb_mmap = np.load(BPR_ITEM_EMB, mmap_mode='r')
F = item_emb_mmap.shape[1]
print(f"{elapsed()} item_emb mmap: {item_emb_mmap.shape}  F={F}  [RAM:{mem_mb():.0f}MB]",
      flush=True)

# ── Pre-allocate output buffers ───────────────────────────────────────────────
test_user_emb        = np.zeros((n_test, F), dtype=np.float32)   # 56 MB
test_user_wsum       = np.zeros(n_test, dtype=np.float32)
test_user_interacted = [set() for _ in range(n_test)]
print(f"{elapsed()} Buffers pre-allocated  [RAM:{mem_mb():.0f}MB]", flush=True)

# ── Stream parquet → build weighted-average user embeddings ──────────────────
print(f"{elapsed()} Streaming {POS_PARQUET} …", flush=True)
pf = pq.ParquetFile(POS_PARQUET)
batch_count = 0
for batch in pf.iter_batches(batch_size=500_000,
                              columns=['user_id', 'item_id', 'pos_count']):
    df = batch.to_pandas()
    ui = df['user_id'].map(user2idx)
    ii = df['item_id'].map(item2idx)
    valid = ui.notna() & ii.notna() & ui.isin(test_uidx_set)

    if valid.any():
        uidx_arr  = ui[valid].astype(int).values
        iidx_arr  = ii[valid].astype(int).values
        w_arr     = df.loc[valid, 'pos_count'].values.astype(np.float32)
        ranks_arr = np.fromiter(
            (test_uidx_rank[u] for u in uidx_arr),
            dtype=np.int32, count=len(uidx_arr))

        # Sort by item index for sequential mmap reads (better page locality)
        order        = np.argsort(iidx_arr)
        iidx_sorted  = iidx_arr[order]
        w_sorted     = w_arr[order]
        ranks_sorted = ranks_arr[order]

        # Load unique item embeddings from mmap in one shot
        unique_iidx, inv = np.unique(iidx_sorted, return_inverse=True)
        embs = np.array(item_emb_mmap[unique_iidx], dtype=np.float32)  # (n_unique, F)
        embs_full = embs[inv]   # (n_valid, F)

        # Vectorised scatter-add (handles duplicate rank indices correctly)
        np.add.at(test_user_emb,  ranks_sorted, embs_full * w_sorted[:, np.newaxis])
        np.add.at(test_user_wsum, ranks_sorted, w_sorted)

        # Track interacted item indices per user (needed for filtering in Phase 2)
        for r, iidx_val in zip(ranks_sorted.tolist(), iidx_sorted.tolist()):
            test_user_interacted[r].add(iidx_val)

    batch_count += 1
    del df, ui, ii, valid; gc.collect()

print(f"{elapsed()} Streaming done ({batch_count} batches)  [RAM:{mem_mb():.0f}MB]",
      flush=True)

# ── Normalise: weighted mean → L2-unit vectors ────────────────────────────────
wsum_safe = np.maximum(test_user_wsum, 1e-10)[:, np.newaxis]
test_user_emb /= wsum_safe
norms = np.linalg.norm(test_user_emb, axis=1, keepdims=True).clip(min=1e-10)
test_user_emb /= norms
print(f"{elapsed()} test-user embs normalised  [RAM:{mem_mb():.0f}MB]", flush=True)

# ── Save outputs ──────────────────────────────────────────────────────────────
np.save(OUT_EMB, test_user_emb)
with open(OUT_META, 'wb') as f:
    pickle.dump({
        'test_in_train':        test_in_train,
        'test_user_interacted': test_user_interacted,
    }, f, protocol=4)
print(f"{elapsed()} Saved test_user_emb.npy + test_user_meta.pkl", flush=True)
print(f"{elapsed()} Phase 1 DONE — releasing all RAM on exit", flush=True)
