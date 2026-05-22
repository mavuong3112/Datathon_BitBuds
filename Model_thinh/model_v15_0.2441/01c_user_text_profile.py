"""
Stage 13 Tier 2 (Part 2): Build user_text_profile + compute title_semantic_sim.

Steps:
1. Load item_title_emb.parquet (3.1M items × 768 float16)
2. Build user_text_profile by averaging title embeddings of pos events per user
3. Compute title_semantic_sim for all (user, item) pairs in candidates_lukewarm
4. Save user_item_title_sim.parquet

Memory strategy: process candidates in 1M-row chunks to avoid OOM.
"""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

OUT_SIM_FILE     = f"{CACHE_DIR}/user_item_title_sim.parquet"
OUT_PROFILE_FILE = f"{CACHE_DIR}/user_text_profile.parquet"

if os.path.exists(OUT_SIM_FILE):
    print(f"{elapsed()} [SKIP] {OUT_SIM_FILE} already exists")
    raise SystemExit(0)

# ── Load item embeddings ──────────────────────────────────────────────────────
print(f"{elapsed()} Loading item_title_emb (~4.7GB) …")
emb_df = pd.read_parquet(f"{CACHE_DIR}/item_title_emb.parquet")
te_cols = [c for c in emb_df.columns if c.startswith('te_')]
print(f"{elapsed()}   {len(emb_df):,} items × {len(te_cols)} dims")

# Convert to numpy array indexed by item_id
items_list = emb_df['item_id'].tolist()
item2idx = {iid: i for i, iid in enumerate(items_list)}
item_emb_arr = emb_df[te_cols].to_numpy(dtype=np.float32)  # (N_items, 768)
title_lengths = emb_df['title_length'].to_numpy()
del emb_df; gc.collect()
print(f"{elapsed()}   item_emb_arr shape: {item_emb_arr.shape}, RAM: ~{item_emb_arr.nbytes/1e9:.1f} GB")

# Normalize item embeddings (precomputed for cosine sim)
item_norms = np.linalg.norm(item_emb_arr, axis=1, keepdims=True).clip(min=1e-10)
item_emb_normed = item_emb_arr / item_norms
del item_emb_arr, item_norms; gc.collect()

# ── Build user_text_profile from pos events ───────────────────────────────────
print(f"{elapsed()} Loading user_item_pos.parquet …")
pos = pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet",
                       columns=['user_id','item_id'])
print(f"{elapsed()}   pos: {len(pos):,} rows, {pos['user_id'].nunique():,} users")

# Filter to items we have embeddings for
pos['item_idx'] = pos['item_id'].map(item2idx)
pos = pos.dropna(subset=['item_idx'])
pos['item_idx'] = pos['item_idx'].astype(np.int32)
print(f"{elapsed()}   after filter to embedded items: {len(pos):,} rows")

# Group by user, average item embeddings
print(f"{elapsed()} Building user text profiles (group-and-average) …")
unique_users = pos['user_id'].unique().tolist()
user2idx_text = {u: i for i, u in enumerate(unique_users)}
n_users = len(unique_users)
print(f"{elapsed()}   n_users: {n_users:,}")

# Pre-allocate sum + count
user_emb_sum = np.zeros((n_users, 768), dtype=np.float32)
user_count   = np.zeros(n_users, dtype=np.int32)

# Stream through pos in chunks to avoid creating huge intermediate
CHUNK = 1_000_000
user_idx_arr = pos['user_id'].map(user2idx_text).astype(np.int32).values
item_idx_arr = pos['item_idx'].values

print(f"{elapsed()} Accumulating user profiles ({len(pos):,} pairs in {len(pos)//CHUNK+1} chunks) …")
for start in range(0, len(pos), CHUNK):
    end = min(start + CHUNK, len(pos))
    u_idx = user_idx_arr[start:end]
    i_idx = item_idx_arr[start:end]
    # scatter add embeddings
    np.add.at(user_emb_sum, u_idx, item_emb_normed[i_idx])
    np.add.at(user_count, u_idx, 1)
    if start % (CHUNK*3) == 0:
        print(f"{elapsed()}   accum {end:,}/{len(pos):,}")

# Average + normalize
user_count_safe = user_count.clip(min=1)[:, np.newaxis]
user_emb = user_emb_sum / user_count_safe
user_norms = np.linalg.norm(user_emb, axis=1, keepdims=True).clip(min=1e-10)
user_emb_normed = user_emb / user_norms
del user_emb_sum, user_emb, user_count_safe, user_norms; gc.collect()
print(f"{elapsed()} user_emb_normed: {user_emb_normed.shape}")
del pos; gc.collect()

# ── Compute title_semantic_sim for candidates_lukewarm ────────────────────────
print(f"{elapsed()} Loading candidates_lukewarm.parquet …")
cands = pd.read_parquet(f"{CACHE_DIR}/candidates_lukewarm.parquet",
                        columns=['user_id','item_id'])
print(f"{elapsed()}   candidates: {len(cands):,}")

# Map to indices
cands['user_idx_text'] = cands['user_id'].map(user2idx_text)
cands['item_idx_text'] = cands['item_id'].map(item2idx)

# For unmapped (cold users / unknown items): set sim = 0, title_length = 0
mask_valid = cands['user_idx_text'].notna() & cands['item_idx_text'].notna()
print(f"{elapsed()}   valid pairs (mapped both): {mask_valid.sum():,}/{len(cands):,}")

sims = np.zeros(len(cands), dtype=np.float32)
title_len_arr = np.zeros(len(cands), dtype=np.int16)

valid_idx = cands.index[mask_valid].values
u_arr = cands.loc[mask_valid, 'user_idx_text'].astype(np.int32).values
i_arr = cands.loc[mask_valid, 'item_idx_text'].astype(np.int32).values

# Compute cosine sim in chunks (avoid temp arrays of 27M × 768)
SIM_CHUNK = 500_000
print(f"{elapsed()} Computing cosine sim ({len(u_arr):,} valid pairs in {len(u_arr)//SIM_CHUNK+1} chunks) …")
for start in range(0, len(u_arr), SIM_CHUNK):
    end = min(start + SIM_CHUNK, len(u_arr))
    u_chunk = user_emb_normed[u_arr[start:end]]  # (chunk, 768)
    i_chunk = item_emb_normed[i_arr[start:end]]  # (chunk, 768)
    chunk_sim = (u_chunk * i_chunk).sum(axis=1)  # row-wise dot
    sims[valid_idx[start:end]] = chunk_sim.astype(np.float32)
    if start % (SIM_CHUNK * 5) == 0:
        print(f"{elapsed()}   sim {end:,}/{len(u_arr):,}")
    del u_chunk, i_chunk, chunk_sim

# Title length for all candidates
title_len_arr[mask_valid] = title_lengths[i_arr]
print(f"{elapsed()} sim stats: mean={sims.mean():.4f} std={sims.std():.4f} min={sims.min():.4f} max={sims.max():.4f}")

# Save
out_df = pd.DataFrame({
    'user_id': cands['user_id'].values,
    'item_id': cands['item_id'].values,
    'title_semantic_sim': sims,
    'title_length':       title_len_arr,
})
out_df.to_parquet(OUT_SIM_FILE, index=False)
print(f"{elapsed()} Saved {OUT_SIM_FILE} ({len(out_df):,} rows)")

# Also save user_text_profile for potential reuse
profile_df = pd.DataFrame({'user_id': unique_users})
for j in range(768):
    profile_df[f'up_{j:03d}'] = user_emb_normed[:, j].astype(np.float16)
profile_df.to_parquet(OUT_PROFILE_FILE, index=False)
print(f"{elapsed()} Saved {OUT_PROFILE_FILE} ({len(profile_df):,} users)")
print(f"{elapsed()} DONE")
