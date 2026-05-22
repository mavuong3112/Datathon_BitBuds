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
train_end_dt = pd.Timestamp(TRAIN_END)

test_users    = test['user_id'].tolist()
valid_items   = set(items['item_id'].unique())
print(f"{elapsed()} test_users={len(test_users):,}  valid_items={len(valid_items):,}")

# ── Remove invalid item_ids ───────────────────────────────────────────────────
before = len(ranked)
ranked = ranked[ranked['item_id'].isin(valid_items)]
print(f"{elapsed()} Removed {before-len(ranked):,} invalid item_ids")

# ── Scale-aware freshness boost (listings ≤7 days old) ───────────────────────
print(f"{elapsed()} Applying freshness boost …")
items_meta = (items[['item_id','posted_date']].drop_duplicates('item_id').copy())
items_meta['posted_date'] = pd.to_datetime(items_meta['posted_date'])
items_meta['days_since_posted'] = (train_end_dt - items_meta['posted_date']).dt.days.clip(lower=0)
ranked = ranked.merge(items_meta[['item_id','days_since_posted']], on='item_id', how='left')
score_range = ranked['lgbm_score'].max() - ranked['lgbm_score'].min()
freshness_boost = score_range * 0.015
ranked['lgbm_score'] += (ranked['days_since_posted'].fillna(999) <= 7).astype(float) * freshness_boost
ranked = ranked.sort_values(['user_id','lgbm_score'], ascending=[True, False])
print(f"{elapsed()} score_range={score_range:.4f}  boost={freshness_boost:.4f}")

# ── Global popular fallback (top-200 by trend_pos, same logic as baseline v1) ─
POOL_SIZE = 50
pop_ranked = (pop.sort_values('trend_pos', ascending=False)
                 .drop_duplicates('item_id'))
diverse_pool = [it for it in pop_ranked['item_id'].tolist()
                if it in valid_items][:POOL_SIZE]
print(f"{elapsed()} Global fallback pool: {len(diverse_pool)} items")

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
