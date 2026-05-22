"""
Step 2: BPR (PyTorch GPU) + implicit ALS (CPU) retrieval.

BPR training: vectorized negative sampling (no Python per-sample loop)
              → fast on GPU RTX 5070.
BPR inference: numpy CPU (avoids 5GB score matrix OOM on GPU).
ALS: implicit CPU fallback for complementary signal.

Stage 8: env RETRIEVER_MODE=train uses pos_train (events < VAL_SPLIT, Tết-filtered)
         → outputs *_candidates_train.parquet for leak-free reranker training.

Outputs (RETRIEVER_MODE=full, default):
  cache/als_candidates.parquet   — BPR top-N per user
  cache/ease_candidates.parquet  — ALS top-N per user
  cache/als_model/               — BPR + ALS weights
  cache/mappings.pkl             — user/item index mappings

Outputs (RETRIEVER_MODE=train):
  cache/als_candidates_train.parquet, cache/ease_candidates_train.parquet
  cache/als_model_train/, cache/mappings_train.pkl
"""
import sys, os, time, pickle, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from config import *

# ── Stage 8/11: mode = full | train | lukewarm ────────────────────────────────
# - full:     pos.parquet (warm users, positive contacts only) — Stage 8 default
# - train:    pos_train.parquet (events < VAL_SPLIT) — Stage 8 leak-free training
# - lukewarm: pos_lukewarm.parquet (login users + pageviews, weighted) — Stage 11
MODE = os.environ.get('RETRIEVER_MODE', 'full').lower()
assert MODE in ('full','train','lukewarm'), f"RETRIEVER_MODE={MODE} invalid"
SUFFIX = {'train': '_train', 'lukewarm': '_lukewarm', 'full': ''}[MODE]
POS_FILE_USED   = f"{CACHE_DIR}/user_item_pos{SUFFIX}.parquet"
MAPPINGS_PATH   = f"{CACHE_DIR}/mappings{SUFFIX}.pkl"
MODEL_DIR_USED  = f"{CACHE_DIR}/als_model{SUFFIX}"
BPR_CANDS_PATH  = f"{CACHE_DIR}/als_candidates{SUFFIX}.parquet"
ALS_CANDS_PATH  = f"{CACHE_DIR}/ease_candidates{SUFFIX}.parquet"
print(f"[STAGE 8] RETRIEVER_MODE={MODE} → reading {POS_FILE_USED}, outputting {BPR_CANDS_PATH}")

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"
def mem_mb():
    import psutil; return psutil.Process().memory_info().rss / 1e6

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{elapsed()} Device: {DEVICE}")
if DEVICE == 'cuda':
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{elapsed()} GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f}GB")

# ── BPR hyperparams ───────────────────────────────────────────────────────────
BPR_FACTORS  = 256
BPR_LR       = 1e-2
BPR_REG      = 1e-4
BPR_EPOCHS   = 20
BPR_BATCH    = 32768
CHUNK        = 1000  # users per chunk for numpy matmul inference (BPR + ALS)

# ── ALS hyperparams ───────────────────────────────────────────────────────────
ALS_FACTORS  = 256
ALS_ITER     = 30
ALS_ALPHA    = 40

# ── Load interactions ─────────────────────────────────────────────────────────
print(f"{elapsed()} Loading interactions from {POS_FILE_USED} …")
pos  = pd.read_parquet(POS_FILE_USED)
test = pd.read_parquet(TEST_FILE)
test_users = set(test['user_id'].tolist())

train_end_dt = pd.Timestamp(TRAIN_END)
pos['last_ts']   = pd.to_datetime(pos['last_ts'])
pos['days_ago']  = (train_end_dt - pos['last_ts']).dt.days.clip(lower=0)
pos['recency_w'] = np.exp(-RECENCY_DECAY * pos['days_ago'])
pos['weight']    = (pos['pos_count'] * pos['recency_w']).clip(lower=1.0)

# ── Index mappings ────────────────────────────────────────────────────────────
all_users = pos['user_id'].unique().tolist()
all_items = pos['item_id'].unique().tolist()
user2idx  = {u: i for i, u in enumerate(all_users)}
item2idx  = {it: i for i, it in enumerate(all_items)}
idx2user  = {i: u for u, i in user2idx.items()}
idx2item  = {i: it for it, i in item2idx.items()}
n_users, n_items = len(all_users), len(all_items)
print(f"{elapsed()} users={n_users:,}  items={n_items:,}  [RAM:{mem_mb():.0f}MB]")

with open(MAPPINGS_PATH, 'wb') as f:
    pickle.dump({'user2idx': user2idx, 'idx2user': idx2user,
                 'item2idx': item2idx, 'idx2item': idx2item}, f)

# Sparse matrix for ALS
ui_u = pos['user_id'].map(user2idx).values.astype(np.int32)
ii_i = pos['item_id'].map(item2idx).values.astype(np.int32)
ww   = pos['weight'].values.astype(np.float32)
user_item_csr = sp.csr_matrix((ww, (ui_u, ii_i)), shape=(n_users, n_items))
item_user_csr = user_item_csr.T.tocsr()

# Pre-build positive arrays for BPR (as tensors — avoids Python loops)
pos_u_np = ui_u                                          # (N,)
pos_i_np = ii_i                                          # (N,)
pos_w_np = pos['weight'].values.astype(np.float32)       # (N,)
N_pairs  = len(pos_u_np)
print(f"{elapsed()} Training pairs: {N_pairs:,}")

# ── BPR Model ─────────────────────────────────────────────────────────────────
class BPR(nn.Module):
    def __init__(self, n_users, n_items, factors):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, factors, sparse=True)
        self.item_emb = nn.Embedding(n_items, factors, sparse=True)
        nn.init.normal_(self.user_emb.weight, std=1.0/factors**0.5)
        nn.init.normal_(self.item_emb.weight, std=1.0/factors**0.5)

    def forward(self, u, i, j):
        pu = self.user_emb(u)           # (B, F)
        qi = self.item_emb(i)           # (B, F)
        qj = self.item_emb(j)           # (B, F)
        return (pu * (qi - qj)).sum(1)  # (B,)

BPR_U_PATH = f"{MODEL_DIR_USED}/user_factors_bpr.npy"
BPR_I_PATH = f"{MODEL_DIR_USED}/item_factors_bpr.npy"
os.makedirs(MODEL_DIR_USED, exist_ok=True)

if os.path.exists(BPR_U_PATH) and os.path.exists(BPR_I_PATH):
    print(f"{elapsed()} [SKIP] BPR weights found — skipping training")
else:
    model = BPR(n_users, n_items, BPR_FACTORS).to(DEVICE)
    optim = torch.optim.SparseAdam(list(model.parameters()), lr=BPR_LR)

    # ── Vectorized BPR training ────────────────────────────────────────────────
    print(f"{elapsed()} Training BPR ({BPR_EPOCHS} epochs, batch={BPR_BATCH}) …")
    all_u = torch.from_numpy(pos_u_np)
    all_i = torch.from_numpy(pos_i_np)
    all_w = torch.from_numpy(pos_w_np)

    for epoch in range(1, BPR_EPOCHS + 1):
        model.train()
        neg_j = torch.randint(0, n_items, (N_pairs,), dtype=torch.long)
        perm  = torch.randperm(N_pairs)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, N_pairs, BPR_BATCH):
            idx  = perm[start:start + BPR_BATCH]
            u_b  = all_u[idx].to(DEVICE)
            i_b  = all_i[idx].to(DEVICE)
            j_b  = neg_j[idx].to(DEVICE)
            w_b  = all_w[idx].to(DEVICE)
            diff = model(u_b, i_b, j_b)
            loss = -(w_b * torch.log(torch.sigmoid(diff) + 1e-10)).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item(); n_batches += 1

        if epoch % 5 == 0 or epoch == BPR_EPOCHS:
            print(f"{elapsed()} Epoch {epoch}/{BPR_EPOCHS}  loss={ep_loss/n_batches:.4f}")

    model.eval()
    with torch.no_grad():
        np.save(BPR_U_PATH, model.user_emb.weight.detach().cpu().numpy())
        np.save(BPR_I_PATH, model.item_emb.weight.detach().cpu().numpy())
    del model; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

print(f"{elapsed()} BPR done  [RAM:{mem_mb():.0f}MB]")

# ── BPR Candidate Generation (CPU numpy — avoids GPU OOM) ────────────────────
if os.path.exists(BPR_CANDS_PATH):
    print(f"{elapsed()} [SKIP] BPR candidates already exist — loading factors for ALS step")
    user_factors_bpr = np.load(BPR_U_PATH)
    item_factors_bpr = np.load(BPR_I_PATH)
    test_in_train = [u for u in test_users if u in user2idx]
    test_cold     = [u for u in test_users if u not in user2idx]
else:
    print(f"{elapsed()} Generating BPR candidates (CPU numpy) …")
    test_in_train = [u for u in test_users if u in user2idx]
    test_cold     = [u for u in test_users if u not in user2idx]
    print(f"{elapsed()} warm={len(test_in_train):,}  cold={len(test_cold):,}")
    user_factors_bpr = np.load(BPR_U_PATH)
    item_factors_bpr = np.load(BPR_I_PATH)

    rows_bpr = []

    for i in range(0, len(test_in_train), CHUNK):
        batch_users = test_in_train[i:i + CHUNK]
        batch_idx   = [user2idx[u] for u in batch_users]
        u_emb       = user_factors_bpr[batch_idx]           # (chunk, F)
        scores      = u_emb @ item_factors_bpr.T            # (chunk, n_items)
        top_ids     = np.argpartition(scores, -N_ALS, axis=1)[:, -N_ALS:]
        for j, uid in enumerate(batch_users):
            top_sorted = top_ids[j][np.argsort(scores[j][top_ids[j]])[::-1]]
            for rank, iid in enumerate(top_sorted):
                rows_bpr.append({'user_id': uid, 'item_id': idx2item[int(iid)],
                                 'als_score': float(scores[j][iid]), 'als_rank': rank+1})
        if (i // CHUNK) % 20 == 0:
            print(f"{elapsed()} BPR inference chunk {i//CHUNK+1}/{(len(test_in_train)-1)//CHUNK+1}")

    df_bpr = pd.DataFrame(rows_bpr)
    print(f"{elapsed()} BPR candidates: {len(df_bpr):,} rows")
    df_bpr.to_parquet(BPR_CANDS_PATH, index=False)
    del df_bpr, rows_bpr; gc.collect()

del user_factors_bpr, item_factors_bpr; gc.collect()

# ── ALS (implicit CPU) — complementary signal ─────────────────────────────────
if os.path.exists(ALS_CANDS_PATH):
    print(f"{elapsed()} [SKIP] ALS candidates already exist — skipping ALS training")
    print(f"{elapsed()} DONE")
else:
    print(f"{elapsed()} Training ALS (implicit CPU, factors={ALS_FACTORS}, iter={ALS_ITER}) …")
    try:
        import implicit
        ALS_MODEL_U = f"{MODEL_DIR_USED}/user_factors_als.npy"
        ALS_MODEL_I = f"{MODEL_DIR_USED}/item_factors_als.npy"
        if os.path.exists(ALS_MODEL_U) and os.path.exists(ALS_MODEL_I):
            print(f"{elapsed()} [RESUME] ALS weights found — skipping training")
            u_factors_als = np.load(ALS_MODEL_U)
            i_factors_als = np.load(ALS_MODEL_I)
        else:
            als = implicit.als.AlternatingLeastSquares(
                factors=ALS_FACTORS, regularization=0.01,
                iterations=ALS_ITER, alpha=ALS_ALPHA,
                use_gpu=False, num_threads=DUCKDB_THREADS, random_state=42)
            als.fit(user_item_csr)  # fit expects (n_users, n_items), not transpose
            u_factors_als = als.user_factors  # shape (n_users, factors)
            i_factors_als = als.item_factors  # shape (n_items, factors)
            np.save(ALS_MODEL_U, u_factors_als)
            np.save(ALS_MODEL_I, i_factors_als)
            del als; gc.collect()
        print(f"{elapsed()} ALS ready  [RAM:{mem_mb():.0f}MB]")

        rows_als = []
        for i in range(0, len(test_in_train), CHUNK):
            batch_users = test_in_train[i:i + CHUNK]
            batch_idx   = [user2idx[u] for u in batch_users]
            u_emb       = u_factors_als[batch_idx]
            scores      = u_emb @ i_factors_als.T
            top_ids     = np.argpartition(scores, -N_EASE, axis=1)[:, -N_EASE:]
            for j, uid in enumerate(batch_users):
                top_s = top_ids[j][np.argsort(scores[j][top_ids[j]])[::-1]]
                for rank, iid in enumerate(top_s):
                    rows_als.append({'user_id': uid, 'item_id': idx2item[int(iid)],
                                     'ease_score': float(scores[j][iid]), 'ease_rank': rank+1})
            if (i // CHUNK) % 20 == 0:
                print(f"{elapsed()} ALS inference {i//CHUNK+1}")

        df_als = pd.DataFrame(rows_als)
        print(f"{elapsed()} ALS candidates: {len(df_als):,}")
    except Exception as e:
        import traceback
        print(f"{elapsed()} ALS failed: {e}")
        traceback.print_exc()
        df_als = pd.DataFrame(columns=['user_id','item_id','ease_score','ease_rank'])

    df_als.to_parquet(ALS_CANDS_PATH, index=False)
    print(f"{elapsed()} DONE")
