"""
Step 8: Validate format + generate final submission.csv
- Scale-aware freshness boost for listings ≤7 days old
- Category-weighted popular fallback (200 items, weighted by contact volume)
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

print(f"{elapsed()} Loading predictions …")
ranked   = pd.read_parquet(f"{CACHE_DIR}/ranked_predictions.parquet")
test     = pd.read_parquet(TEST_FILE)
items    = pd.read_parquet(f"{CACHE_DIR}/items.parquet")
pop      = pd.read_parquet(f"{CACHE_DIR}/popular_items.parquet")
pos_users = set(pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet",
                                 columns=['user_id'])['user_id'].unique())
train_end_dt = pd.Timestamp(TRAIN_END)

test_users    = test['user_id'].tolist()
valid_items   = set(items['item_id'].unique())
cold_users    = set(test_users) - pos_users   # v1-style cold detection
print(f"{elapsed()} test_users={len(test_users):,}  valid_items={len(valid_items):,}")
print(f"{elapsed()} cold_users={len(cold_users):,} ({100*len(cold_users)/len(test_users):.1f}%) — will get global popular top-10 (v1 behavior)")

# ── Remove invalid item_ids ───────────────────────────────────────────────────
before = len(ranked)
ranked = ranked[ranked['item_id'].isin(valid_items)]
print(f"{elapsed()} Removed {before-len(ranked):,} invalid item_ids")

# ── Scale-aware freshness boost (listings ≤7 days old) ───────────────────────
print(f"{elapsed()} Applying freshness boost …")
items_meta = (items[['item_id','posted_date','category']].drop_duplicates('item_id').copy())
items_meta['posted_date'] = pd.to_datetime(items_meta['posted_date'])
items_meta['days_since_posted'] = (train_end_dt - items_meta['posted_date']).dt.days.clip(lower=0)
ranked = ranked.merge(items_meta[['item_id','days_since_posted','category']],
                      on='item_id', how='left')
score_range = ranked['lgbm_score'].max() - ranked['lgbm_score'].min()
freshness_boost = score_range * 0.015
ranked['lgbm_score'] += (ranked['days_since_posted'].fillna(999) <= 7).astype(float) * freshness_boost

# ── v6d: Category match boost for warm users (pref_category from POS) ────────
# pref_category for warm users derived from positive interactions → reliable signal.
# Cold users won't have pref_category (NaN merge) → no boost → no effect on them.
# 1-tree LGBM ignores category_match feature → apply as post-hoc score boost.
print(f"{elapsed()} Applying category-match boost for warm users …")
profiles = pd.read_parquet(f"{CACHE_DIR}/user_profiles.parquet",
                            columns=['user_id','pref_category'])
ranked = ranked.merge(profiles, on='user_id', how='left')
cat_boost = freshness_boost * 0.4  # ~0.4% of score range, enough to break ties
ranked['cat_match'] = (
    ranked['pref_category'].notna() &
    (ranked['category'] == ranked['pref_category'])
).astype(float)
ranked['lgbm_score'] += ranked['cat_match'] * cat_boost
n_matches = int(ranked['cat_match'].sum())
print(f"{elapsed()}   {n_matches:,} item-pair matches boosted by {cat_boost:.4f}")

# blend_score as tiebreaker — critical for cold users whose lgbm_score ties at single-leaf value
ranked = ranked.sort_values(['user_id','lgbm_score','blend_score'],
                            ascending=[True, False, False])
print(f"{elapsed()} score_range={score_range:.4f}  freshness_boost={freshness_boost:.4f}  cat_boost={cat_boost:.4f}")

# ── Category-weighted INTERLEAVED diverse fallback ──────────────────────────
# Slot allocation theo contact volume per category, top-N each per trend_pos
# Interleave (rank-1 mỗi cat, rồi rank-2, …) → top-10 fallback spans nhiều categories
POOL_SIZE = 50
COLD_POOL_PER_CAT = 40  # max per category before interleave

# Allocate slots theo contact volume (trend_pos) per category
cat_volume = pop.groupby('category')['trend_pos'].sum().sort_values(ascending=False)
weights = cat_volume / cat_volume.sum()
slots_per_cat = (weights * POOL_SIZE).round().clip(lower=3).astype(int)
# Adjust slots tổng = POOL_SIZE
while slots_per_cat.sum() > POOL_SIZE:
    slots_per_cat[slots_per_cat.idxmax()] -= 1
while slots_per_cat.sum() < POOL_SIZE:
    slots_per_cat[slots_per_cat.idxmax()] += 1
print(f"{elapsed()} Category slot allocation:")
for cat, n in slots_per_cat.items():
    print(f"  cat={cat}: {n} slots ({weights[cat]*100:.1f}% volume)")

# Top items per category (max COLD_POOL_PER_CAT each)
pop_per_cat = (pop.sort_values('trend_pos', ascending=False)
                  .drop_duplicates('item_id')
                  .groupby('category').head(COLD_POOL_PER_CAT)
                  [['item_id','category','trend_pos']])

# Precompute per-cat item lists once (avoid O(N²) lookup in loop)
cat_items_map = {cat: pop_per_cat[pop_per_cat['category']==cat]['item_id'].tolist()
                 for cat in slots_per_cat.index}

# Interleave: rank-1 cho mỗi cat, rồi rank-2, … cho đến khi đủ slots/cat
diverse_pool, seen = [], set()
cat_counts = {cat: 0 for cat in slots_per_cat.index}
for rank_idx in range(COLD_POOL_PER_CAT):
    for cat, max_slots in slots_per_cat.items():
        if cat_counts[cat] >= max_slots:
            continue
        cat_items = cat_items_map[cat]
        if rank_idx < len(cat_items):
            iid = cat_items[rank_idx]
            if iid in valid_items and iid not in seen:
                diverse_pool.append(iid)
                seen.add(iid)
                cat_counts[cat] += 1
diverse_pool = diverse_pool[:POOL_SIZE]
print(f"{elapsed()} Diverse fallback pool: {len(diverse_pool)} items")
print(f"{elapsed()} Pool category breakdown: {cat_counts}")

# ── Ensure every test user has exactly 10 items ──────────────────────────────
print(f"{elapsed()} Filling to 10 items per user …")
user_items  = {uid: [] for uid in test_users}
user_seen   = {uid: set() for uid in test_users}
for _, row in ranked.iterrows():
    uid = row['user_id']
    iid = row['item_id']
    if uid in user_items and iid not in user_seen[uid]:
        user_items[uid].append(iid)
        user_seen[uid].add(iid)

n_filled = 0
for uid in test_users:
    items_list = user_items[uid]
    if len(items_list) < 10:
        seen = user_seen[uid]
        for it in diverse_pool:
            if len(items_list) >= 10:
                break
            if it not in seen:
                items_list.append(it)
                seen.add(it)
                n_filled += 1
    user_items[uid] = items_list[:10]

print(f"{elapsed()} Filled {n_filled:,} slots with popularity fallback")

# ── v6+B: All cold users get global popular top-10 (v1 behavior, restored) ──
# v6c lukewarm category personalization scored 0.2259 (-0.0162 vs v6+B 0.2421)
# → Cold/lukewarm users CONTACT items distinct from browse category. Globally
#   popular always wins over category-segmented for users without pos history.
print(f"{elapsed()} Building global popular top-10 for cold user override …")
global_pop_top10 = (pop[pop['item_id'].isin(valid_items)]
                    .sort_values('trend_pos', ascending=False)
                    .drop_duplicates('item_id')
                    .head(10)['item_id'].tolist())
print(f"{elapsed()} Global popular top-10: {global_pop_top10[0][:16]}…")

n_overridden = 0
for uid in cold_users:
    if uid in user_items:
        user_items[uid] = global_pop_top10[:10]
        n_overridden += 1
print(f"{elapsed()} Overrode {n_overridden:,} cold users with global popular top-10")

# ── Format submission ─────────────────────────────────────────────────────────
print(f"{elapsed()} Formatting submission …")
rows = []
for uid in test_users:
    for rank, item_id in enumerate(user_items[uid], start=1):
        rows.append({'user_id': uid, 'rank': rank, 'item_id': item_id})

sub = pd.DataFrame(rows)
sub.insert(0, 'ID', range(1, len(sub)+1))

# ── Validation checks ─────────────────────────────────────────────────────────
print(f"{elapsed()} Validating …")
assert len(sub) == len(test_users) * 10,  f"Expected {len(test_users)*10} rows, got {len(sub)}"
assert sub['rank'].between(1,10).all(),   "rank out of [1,10]"
assert sub.groupby('user_id')['rank'].nunique().eq(10).all(), "Not all users have 10 ranks"
assert sub['item_id'].isin(valid_items).all(), "Invalid item_ids found"
print(f"{elapsed()} All checks passed ✓")

# ── Save ──────────────────────────────────────────────────────────────────────
sub[['ID','user_id','rank','item_id']].to_csv(SUBMIT_OUT, index=False, encoding='utf-8')
size_mb = sub.memory_usage(deep=True).sum() / 1e6
print(f"{elapsed()} Saved: {SUBMIT_OUT}")
print(f"{elapsed()} Rows: {len(sub):,}  Users: {sub['user_id'].nunique():,}  Size: ~{size_mb:.0f}MB")
print(f"{elapsed()} DONE — ready to submit to Kaggle!")
