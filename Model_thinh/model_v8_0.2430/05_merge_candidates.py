"""
Step 5: Merge all retrieval strategies into unified candidate pool.
Sources: ALS, EASE, ItemCF, SASRec, Repeat history, Trending

Stage 8: env RETRIEVER_MODE=train uses *_train retrievers + pos_train,
         outputs candidates_train.parquet (leak-free for reranker training).
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

MODE = os.environ.get('RETRIEVER_MODE', 'full').lower()
SUFFIX = '_train' if MODE == 'train' else ''
OUT_CANDS  = f"{CACHE_DIR}/candidates{SUFFIX}.parquet"
POS_FILE   = f"{CACHE_DIR}/user_item_pos{SUFFIX}.parquet"
print(f"[STAGE 8] RETRIEVER_MODE={MODE} → output {OUT_CANDS}")

print(f"{elapsed()} Loading retrieval results …")
pos   = pd.read_parquet(POS_FILE)
test  = pd.read_parquet(TEST_FILE)
test_users = set(test['user_id'].tolist())

# ── Load each strategy ────────────────────────────────────────────────────────
def load_if_exists(path, cols):
    if os.path.exists(path):
        return pd.read_parquet(path)
    print(f"  WARNING: {path} not found — skipping")
    return pd.DataFrame(columns=['user_id','item_id'] + cols)

als   = load_if_exists(f"{CACHE_DIR}/als_candidates{SUFFIX}.parquet",    ['als_score','als_rank'])
ease  = load_if_exists(f"{CACHE_DIR}/ease_candidates{SUFFIX}.parquet",   ['ease_score','ease_rank'])
cf    = load_if_exists(f"{CACHE_DIR}/itemcf_candidates{SUFFIX}.parquet", ['itemcf_score','itemcf_rank'])
sr    = load_if_exists(f"{CACHE_DIR}/sasrec_candidates{SUFFIX}.parquet", ['sasrec_score','sasrec_rank'])
pop   = pd.read_parquet(f"{CACHE_DIR}/popular_items.parquet")

print(f"{elapsed()} als={len(als):,}  ease={len(ease):,}  cf={len(cf):,}  sr={len(sr):,}")

# ── Repeat history (positive interactions from training) ──────────────────────
repeat = pos[pos['user_id'].isin(test_users)][['user_id','item_id','pos_count']].copy()
repeat = repeat.rename(columns={'pos_count':'repeat_count'})
repeat['is_repeat'] = 1
print(f"{elapsed()} repeat: {len(repeat):,}")

# ── Per-user trending fallback (for cold users) ────────────────────────────────
# Build trending candidates per category+city from popular_items
items_df = pd.read_parquet(f"{CACHE_DIR}/items.parquet")
profiles  = pd.read_parquet(f"{CACHE_DIR}/user_profiles.parquet")

# Top-N per category globally — category-diverse pool for cold users
COLD_POOL_PER_CAT = 10   # 5 categories × 10 = 50 diverse items for cold users
                          # (200 was too memory-heavy at step 06 — 38M rows OOM)
pop_ranked = (pop.sort_values('trend_pos', ascending=False)
                 .drop_duplicates('item_id')
                 .groupby('category').head(COLD_POOL_PER_CAT)
                 [['item_id','category','trend_pos']])

# ── Merge all sources on (user_id, item_id) ───────────────────────────────────
print(f"{elapsed()} Merging sources …")

# Start with union of all (user_id, item_id) pairs
def get_pairs(df, user_col='user_id', item_col='item_id'):
    return df[[user_col, item_col]].drop_duplicates()

all_pairs = pd.concat([
    get_pairs(als),
    get_pairs(ease),
    get_pairs(cf),
    get_pairs(sr),
    get_pairs(repeat),
], ignore_index=True).drop_duplicates(['user_id','item_id'])

# Add cold-user fallback: assign category-diverse popular items
users_with_cands = set(all_pairs['user_id'].unique())
cold_users = [u for u in test_users if u not in users_with_cands]
print(f"{elapsed()} warm={len(users_with_cands):,}  cold={len(cold_users):,}")

if cold_users:
    # Build INTERLEAVED diverse pool: pick rank-1 from each cat, then rank-2, ...
    # Guarantees first 5 items span all 5 categories (căn hộ, phòng trọ, nhà ở, đất nền, dự án)
    diverse_pool = []
    seen = set()
    for rank_idx in range(COLD_POOL_PER_CAT):
        for cat in [1010, 1020, 1030, 1040, 1050]:
            cat_items = pop_ranked[pop_ranked['category']==cat]['item_id'].tolist()
            if rank_idx < len(cat_items):
                iid = cat_items[rank_idx]
                if iid not in seen:
                    diverse_pool.append(iid)
                    seen.add(iid)
    print(f"{elapsed()} Cold pool: {len(diverse_pool)} category-diverse items")

    # Vectorized cross-join: cold_users × diverse_pool (replaces O(N_cold * MAX_CANDS) loop)
    cold_df = pd.MultiIndex.from_product(
        [cold_users, diverse_pool], names=['user_id','item_id']
    ).to_frame(index=False)
    all_pairs = pd.concat([all_pairs, cold_df], ignore_index=True).drop_duplicates(['user_id','item_id'])
    del cold_df

print(f"{elapsed()} Total pairs: {len(all_pairs):,}")

# ── Join retrieval scores ─────────────────────────────────────────────────────
cands = all_pairs.copy()
if len(als):
    cands = cands.merge(als[['user_id','item_id','als_score']],
                        on=['user_id','item_id'], how='left')
else:
    cands['als_score'] = np.nan

if len(ease):
    cands = cands.merge(ease[['user_id','item_id','ease_score']],
                        on=['user_id','item_id'], how='left')
else:
    cands['ease_score'] = np.nan

if len(cf):
    cands = cands.merge(cf[['user_id','item_id','itemcf_score']],
                        on=['user_id','item_id'], how='left')
else:
    cands['itemcf_score'] = np.nan

if len(sr):
    cands = cands.merge(sr[['user_id','item_id','sasrec_score']],
                        on=['user_id','item_id'], how='left')
else:
    cands['sasrec_score'] = np.nan

cands = cands.merge(repeat[['user_id','item_id','repeat_count','is_repeat']],
                    on=['user_id','item_id'], how='left')
cands['repeat_count'] = cands['repeat_count'].fillna(0)
cands['is_repeat']    = cands['is_repeat'].fillna(0).astype(int)

# Source count (how many strategies recommended this item to this user)
score_cols = ['als_score','ease_score','itemcf_score','sasrec_score']
cands['source_count'] = cands[score_cols].notna().sum(axis=1) + cands['is_repeat']

# ── Normalize scores per strategy within each user ───────────────────────────
def minmax_norm_per_user(df, col):
    mn = df.groupby('user_id')[col].transform('min')
    mx = df.groupby('user_id')[col].transform('max')
    return (df[col] - mn) / (mx - mn + 1e-10)

for col in score_cols:
    valid = cands[col].notna()
    cands.loc[valid, f'{col}_norm'] = minmax_norm_per_user(cands[valid], col)
    cands[f'{col}_norm'] = cands[f'{col}_norm'].fillna(0.0)

# Blended retrieval score (before LightGBM reranking)
cands['blend_score'] = (
    0.35 * cands['als_score_norm'] +
    0.25 * cands['ease_score_norm'] +
    0.15 * cands['itemcf_score_norm'] +
    0.15 * cands['sasrec_score_norm'] +
    0.10 * cands['is_repeat'].clip(0, 1)
)

# ── Cap at MAX_CANDS per user ─────────────────────────────────────────────────
print(f"{elapsed()} Capping at {MAX_CANDS} candidates per user …")
cands = (cands.sort_values('blend_score', ascending=False)
              .groupby('user_id').head(MAX_CANDS)
              .reset_index(drop=True))

print(f"{elapsed()} Final candidates: {len(cands):,} rows, {cands['user_id'].nunique():,} users")
cands.to_parquet(OUT_CANDS, index=False)
print(f"{elapsed()} DONE")
