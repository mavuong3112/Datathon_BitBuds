"""
Step 5: Merge all retrieval strategies into unified candidate pool.
Sources: ALS, EASE, ItemCF, SASRec, Repeat history, Trending
Outputs:
  cache/candidates.parquet — deduped candidate pool with source scores
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

print(f"{elapsed()} Loading retrieval results …")
pos   = pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet")
test  = pd.read_parquet(TEST_FILE)
test_users = set(test['user_id'].tolist())

# ── Load each strategy ────────────────────────────────────────────────────────
def load_if_exists(path, cols):
    if os.path.exists(path):
        return pd.read_parquet(path)
    print(f"  WARNING: {path} not found — skipping")
    return pd.DataFrame(columns=['user_id','item_id'] + cols)

als   = load_if_exists(f"{CACHE_DIR}/als_candidates.parquet",    ['als_score','als_rank'])
ease  = load_if_exists(f"{CACHE_DIR}/ease_candidates.parquet",   ['ease_score','ease_rank'])
cf    = load_if_exists(f"{CACHE_DIR}/itemcf_candidates.parquet", ['itemcf_score','itemcf_rank'])
sr    = load_if_exists(f"{CACHE_DIR}/sasrec_candidates.parquet", ['sasrec_score','sasrec_rank'])
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

# Top-N_TRENDING per category globally (for users without preference)
pop_ranked = (pop.sort_values('trend_pos', ascending=False)
                 .groupby('category').head(N_TRENDING)
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

# Add cold-user fallback: assign popular items to users with no candidates
users_with_cands = set(all_pairs['user_id'].unique())
cold_users = [u for u in test_users if u not in users_with_cands]
print(f"{elapsed()} warm={len(users_with_cands):,}  cold={len(cold_users):,}")

if cold_users:
    # Assign global top-trending per category for cold users
    top_global = pop.sort_values('trend_pos', ascending=False).head(MAX_CANDS)
    cold_rows = []
    for uid in cold_users:
        for _, row in top_global.iterrows():
            cold_rows.append({'user_id': uid, 'item_id': row['item_id']})
    cold_df = pd.DataFrame(cold_rows[:len(cold_users)*N_TRENDING])
    all_pairs = pd.concat([all_pairs, cold_df], ignore_index=True).drop_duplicates(['user_id','item_id'])

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
cands.to_parquet(f"{CACHE_DIR}/candidates.parquet", index=False)
print(f"{elapsed()} DONE")
