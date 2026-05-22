"""
Phase 2 of Item-CF: GPU scoring with a clean heap.
Loads 56 MB of user embeddings, streams item embeddings via mmap, scores on GPU.
Runs as a fresh subprocess so heap fragmentation from Phase 1 is gone.
"""
import sys, os, time, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"
def mem_mb():
    import psutil; return psutil.Process().memory_info().rss / 1e6

MODE = os.environ.get('RETRIEVER_MODE', 'full').lower()
SUFFIX = '_train' if MODE == 'train' else ''
OUT_PATH      = f"{CACHE_DIR}/itemcf_candidates{SUFFIX}.parquet"
EMB_USER_PATH = f"{CACHE_DIR}/test_user_emb{SUFFIX}.npy"
META_PATH     = f"{CACHE_DIR}/test_user_meta{SUFFIX}.pkl"
EMB_ITEM_PATH = f"{CACHE_DIR}/als_model{SUFFIX}/item_factors_bpr.npy"
MAPPINGS_PATH = f"{CACHE_DIR}/mappings{SUFFIX}.pkl"

if os.path.exists(OUT_PATH):
    try:
        df_check = pd.read_parquet(OUT_PATH)
        if len(df_check) > 0:
            print(f"{elapsed()} [SKIP] itemcf_candidates.parquet exists "
                  f"({len(df_check):,} rows)", flush=True)
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{elapsed()} Device: {DEVICE}  [RAM:{mem_mb():.0f}MB]", flush=True)

# ── Load user embeddings (56 MB, clean allocation) ────────────────────────────
test_user_emb = np.load(EMB_USER_PATH)          # (n_test, F) float32
with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
test_in_train        = meta['test_in_train']
test_user_interacted = meta['test_user_interacted']
n_test = len(test_in_train)
F      = test_user_emb.shape[1]
print(f"{elapsed()} User embs: {test_user_emb.shape}  [RAM:{mem_mb():.0f}MB]", flush=True)

# ── Load mappings ─────────────────────────────────────────────────────────────
with open(MAPPINGS_PATH, 'rb') as f:
    maps = pickle.load(f)
idx2item = maps['idx2item']
n_items  = len(maps['item2idx'])
print(f"{elapsed()} n_items={n_items:,}  [RAM:{mem_mb():.0f}MB]", flush=True)

# ── Load item embeddings via mmap (OS pages on demand, no malloc) ─────────────
item_emb_mmap = np.load(EMB_ITEM_PATH, mmap_mode='r')
print(f"{elapsed()} item_emb mmap: {item_emb_mmap.shape}  [RAM:{mem_mb():.0f}MB]",
      flush=True)

# ── Pre-allocate the chunk buffer ONCE (no realloc during scoring) ────────────
ITEM_CHUNK  = 50_000   # 50K × 256 × 4 = 51 MB, reused via np.copyto
INFER_BATCH = 512
chunk_buf = np.empty((ITEM_CHUNK, F), dtype=np.float32)
print(f"{elapsed()} chunk_buf pre-allocated ({ITEM_CHUNK}×{F})  "
      f"[RAM:{mem_mb():.0f}MB]", flush=True)

# ── GPU chunked scoring ────────────────────────────────────────────────────────
print(f"{elapsed()} GPU scoring "
      f"(user_batch={INFER_BATCH}, item_chunk={ITEM_CHUNK}) …", flush=True)

rows_cf  = []
n_batches = (n_test - 1) // INFER_BATCH + 1

with torch.no_grad():
    for ui_start in range(0, n_test, INFER_BATCH):
        ui_end    = min(ui_start + INFER_BATCH, n_test)
        B         = ui_end - ui_start
        batch_num = ui_start // INFER_BATCH + 1

        # User slice → GPU (float16 for matmul speed)
        u_emb_gpu = torch.tensor(
            test_user_emb[ui_start:ui_end], dtype=torch.float16, device=DEVICE)  # (B, F)

        # Running top-K accumulators
        best_scores = torch.full((B, N_ITEMCF), float('-inf'), device=DEVICE)
        best_ids    = torch.zeros((B, N_ITEMCF), dtype=torch.long, device=DEVICE)

        for c_start in range(0, n_items, ITEM_CHUNK):
            c_end = min(c_start + ITEM_CHUNK, n_items)
            sz    = c_end - c_start

            # Copy mmap slice into pre-allocated buffer → no new malloc
            np.copyto(chunk_buf[:sz], item_emb_mmap[c_start:c_end])

            # Move to GPU as float16; from_numpy shares buffer, .to() copies to GPU
            chunk_gpu = torch.from_numpy(chunk_buf[:sz]).to(
                device=DEVICE, dtype=torch.float16)                       # (sz, F)
            chunk_gpu = chunk_gpu / chunk_gpu.norm(dim=1, keepdim=True).clamp(min=1e-10)

            chunk_scores = (u_emb_gpu @ chunk_gpu.T).float()              # (B, sz)

            # Merge into running top-K without extra allocation where possible
            full_sc = torch.cat([best_scores, chunk_scores], dim=1)       # (B, K+sz)
            ids_chunk = torch.arange(c_start, c_end,
                                     device=DEVICE, dtype=torch.long)
            full_id = torch.cat(
                [best_ids, ids_chunk.unsqueeze(0).expand(B, -1)], dim=1)  # (B, K+sz)

            topk        = torch.topk(full_sc, N_ITEMCF, dim=1)
            best_scores = topk.values
            best_ids    = full_id.gather(1, topk.indices)

            del chunk_gpu, chunk_scores, full_sc, full_id, topk, ids_chunk

        # ── Decode results: ONE bulk GPU→CPU transfer, no per-element .item() ──
        best_ids_np    = best_ids.cpu().numpy()    # (B, N_ITEMCF) int64
        best_scores_np = best_scores.cpu().numpy() # (B, N_ITEMCF) float32
        del u_emb_gpu, best_scores, best_ids

        for j in range(B):
            uid  = test_in_train[ui_start + j]
            seen = test_user_interacted[ui_start + j]
            rank_out = 0
            for k in range(N_ITEMCF):
                iidx = int(best_ids_np[j, k])
                sc   = float(best_scores_np[j, k])
                if iidx in seen or sc == float('-inf') or iidx not in idx2item:
                    continue
                rows_cf.append({
                    'user_id':      uid,
                    'item_id':      idx2item[iidx],
                    'itemcf_score': sc,
                    'itemcf_rank':  rank_out + 1,
                })
                rank_out += 1
                if rank_out >= N_ITEMCF:
                    break

        if batch_num % 20 == 0 or batch_num == 1:
            vram = torch.cuda.memory_allocated() / 1e6 if DEVICE == 'cuda' else 0
            print(f"{elapsed()} ItemCF {batch_num}/{n_batches}  "
                  f"VRAM:{vram:.0f}MB  [RAM:{mem_mb():.0f}MB]", flush=True)

        # Flush every 10 user-batches to avoid 512MB+ in-memory list at the end
        FLUSH_EVERY = 10
        if len(rows_cf) > 0 and (batch_num % FLUSH_EVERY == 0 or
                                   ui_start + INFER_BATCH >= n_test):
            chunk_path = f"{CACHE_DIR}/_itemcf_chunk_{batch_num:04d}.parquet"
            pd.DataFrame(rows_cf).to_parquet(chunk_path, index=False)
            rows_cf.clear()

# ── Concatenate chunk files → final output ────────────────────────────────────
import glob as _glob
chunk_files = sorted(_glob.glob(f"{CACHE_DIR}/_itemcf_chunk_*.parquet"))
if chunk_files:
    df_cf = pd.concat([pd.read_parquet(c) for c in chunk_files], ignore_index=True)
    for c in chunk_files:
        os.remove(c)
else:
    df_cf = pd.DataFrame(rows_cf)  # fallback (shouldn't happen)
print(f"{elapsed()} ItemCF candidates: {len(df_cf):,} rows", flush=True)
df_cf.to_parquet(OUT_PATH, index=False)
print(f"{elapsed()} Phase 2 DONE", flush=True)
