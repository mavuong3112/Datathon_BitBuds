"""
Step 4b (Stage 14, Hook #3 Part C): Content-Based Filtering candidate stream.

Why this is MANDATORY (not optional):
  Fresh listings (<24h since posted) have no click data yet → ALS/EASE/ItemCF/SASRec
  CANNOT retrieve them. Without a CBF stream injecting fresh+quality items into the
  candidate pool, the Freshness_Multiplier in 08_submit (Part B) multiplies a score of
  zero → useless. CBF is the only retriever that can find fresh items by attribute.

Logic:
  1. Load items, compute item_quality_score (same formula as 06_features) + velocity_24h.
  2. Filter items: days_since_posted ≤ 7 AND item_quality_score > 0.3.
  3. For each test user with pref_category + pref_district_name (warm-with-profile):
     - Match items in same (category, district_name).
     - Optionally filter price_bucket within ±1 ordinal of pref_price_bucket.
  4. Rank by cbf_score = quality_score × log1p(velocity_24h * 10 + trend_pos_1d * 0.1).
     (Use additive log term so trending items still surface when velocity ≈ 1.0.)
  5. Top-50 per user → cbf_candidates.parquet.

Modes (env CBF_MODE):
  - test (default): builds for test_users → cbf_candidates.parquet
  - train: builds for users in user_item_pos_train → cbf_candidates_train.parquet
           (needed for reranker training to see CBF feature distribution)

Output cols: user_id, item_id, cbf_score, cbf_rank
"""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

MODE = os.environ.get('CBF_MODE', os.environ.get('RETRIEVER_MODE', 'full')).lower()
TOP_N = int(os.environ.get('CBF_TOP_N', '50'))
FRESH_MAX_DAYS = int(os.environ.get('CBF_FRESH_DAYS', '7'))
QUALITY_MIN = float(os.environ.get('CBF_QUALITY_MIN', '0.3'))

SUFFIX = {'train': '_train', 'lukewarm': '_lukewarm', 'full': '', 'test': ''}.get(MODE)
if SUFFIX is None:
    raise ValueError(f"Invalid CBF_MODE={MODE}; expected train/lukewarm/full/test")
OUT_FILE = f"{CACHE_DIR}/cbf_candidates{SUFFIX}.parquet"
USERS_SRC = (f"{CACHE_DIR}/user_item_pos_train.parquet" if MODE == 'train' else TEST_FILE)

print(f"[STAGE 14 CBF] mode={MODE}  top_n={TOP_N}  fresh_days≤{FRESH_MAX_DAYS}  q≥{QUALITY_MIN}")
print(f"{elapsed()} output → {OUT_FILE}")

# ── 1. Load items + compute quality_score + days_since_posted ─────────────────
print(f"{elapsed()} Loading items …")
items = pd.read_parquet(f"{CACHE_DIR}/items.parquet")
print(f"{elapsed()}   items: {len(items):,}")

train_end_dt = pd.Timestamp(TRAIN_END)
items['posted_date'] = pd.to_datetime(items['posted_date'], errors='coerce')
items['days_since_posted'] = (train_end_dt - items['posted_date']).dt.days.clip(lower=0).fillna(999).astype(np.int32)

# Mirror 06_features.py quality_score formula
strong_legal = {'Đã có sổ', 'Sổ hồng riêng', 'Sổ hồng / sổ đỏ', 'Sổ đỏ'}
trusted_sellers = {'Cá nhân', 'Cá nhân tự đăng', 'Môi giới chuyên nghiệp'}
has_strong_legal = items['legal_status'].isin(strong_legal).astype(np.float32)
has_direction    = items['direction'].notna().astype(np.float32)
has_house_type   = items['house_type'].notna().astype(np.float32) if 'house_type' in items.columns else 0.0
has_many_images  = (items['images_count'].fillna(0) > 3).astype(np.float32)
is_trusted_seller = items['seller_type'].isin(trusted_sellers).astype(np.float32)
items['item_quality_score'] = (
    0.20 * has_many_images
  + 0.20 * is_trusted_seller
  + 0.30 * has_strong_legal
  + 0.15 * has_house_type
  + 0.15 * has_direction
).astype(np.float32)

PRICE_ORDER = {
    '<500M': 0, '500M–800M': 1, '800M–1B': 2, '1B–1.5B': 3, '1.5B–2B': 4,
    '2B–3B': 5, '3B–5B': 6, '5B–7B': 7, '7B–10B': 8, '10B–15B': 9,
    '15B–20B': 10, '>20B': 11,
    '<2M/tháng': 0, '2M–3M/tháng': 1, '3M–5M/tháng': 2, '5M–7M/tháng': 3,
    '7M–10M/tháng': 4, '10M–15M/tháng': 5, '15M–20M/tháng': 6,
    '20M–30M/tháng': 7, '>30M/tháng': 8,
}
items['price_ordinal'] = items['price_bucket'].map(PRICE_ORDER).fillna(-1).astype(np.int8)

# ── 2. Merge popular_items_1d / 7d for velocity_24h ───────────────────────────
print(f"{elapsed()} Loading popular_items_1d + 7d …")
pop_1d = pd.read_parquet(f"{CACHE_DIR}/popular_items_1d.parquet")
pop_7d = pd.read_parquet(f"{CACHE_DIR}/popular_items_7d.parquet")
items = items.merge(pop_1d[['item_id','trend_pos_1d']], on='item_id', how='left')
items = items.merge(pop_7d[['item_id','trend_pos_7d']], on='item_id', how='left')
items['trend_pos_1d'] = items['trend_pos_1d'].fillna(0).astype(np.float32)
items['trend_pos_7d'] = items['trend_pos_7d'].fillna(0).astype(np.float32)
items['velocity_24h'] = (items['trend_pos_1d'] / (items['trend_pos_7d'] + 1)).astype(np.float32)

# ── 3. Filter fresh + quality items ───────────────────────────────────────────
fresh = items[
    (items['days_since_posted'] <= FRESH_MAX_DAYS) &
    (items['item_quality_score'] > QUALITY_MIN) &
    items['district_name'].notna() &
    items['category'].notna()
].copy()
print(f"{elapsed()}   fresh+quality items: {len(fresh):,} / {len(items):,} "
      f"(days_since≤{FRESH_MAX_DAYS}, q>{QUALITY_MIN})")

# CBF score: quality × (1 + log popularity boost). Adding +trend_pos_1d term ensures
# items with 0 velocity but some 24h activity rank above completely-stale items.
fresh['cbf_score'] = (
    fresh['item_quality_score'].astype(np.float32) *
    np.log1p(fresh['velocity_24h'] * 10 + fresh['trend_pos_1d'] * 0.1)
).astype(np.float32)

fresh = fresh[['item_id','category','district_name','price_ordinal','cbf_score']].copy()
del items, pop_1d, pop_7d, has_strong_legal, has_direction, has_house_type, has_many_images, is_trusted_seller
gc.collect()

# ── 4. Load user profiles + pref_extended ─────────────────────────────────────
print(f"{elapsed()} Loading user profiles …")
profs = pd.read_parquet(f"{CACHE_DIR}/user_profiles.parquet")[['user_id','pref_category']]
upe = pd.read_parquet(f"{CACHE_DIR}/user_pref_extended.parquet")[
    ['user_id','pref_district_name','pref_price_bucket']]

if MODE == 'train':
    src_users = pd.read_parquet(USERS_SRC)[['user_id']].drop_duplicates()
else:
    src_users = pd.read_parquet(USERS_SRC)[['user_id']].drop_duplicates()

users = (src_users
    .merge(profs, on='user_id', how='inner')
    .merge(upe,   on='user_id', how='inner'))
users = users[users['pref_district_name'].notna() & users['pref_category'].notna()].copy()
users['pref_price_ordinal'] = users['pref_price_bucket'].map(PRICE_ORDER).fillna(-1).astype(np.int8)
print(f"{elapsed()}   eligible users (have pref_cat + pref_district): {len(users):,} / {len(src_users):,}")

# ── 5. Match: inner-join users × fresh on (pref_category=category, pref_district_name=district_name)
print(f"{elapsed()} Joining users × fresh items …")
matches = users.merge(
    fresh.rename(columns={'category':'pref_category','district_name':'pref_district_name'}),
    on=['pref_category','pref_district_name'], how='inner')
print(f"{elapsed()}   raw matches: {len(matches):,}")
del users, fresh; gc.collect()

# Optional price filter: ±1 ordinal of user pref, but only if both have valid ordinals.
# If user has no pref_price (ordinal=-1) → skip price filter for that user.
price_diff = (matches['price_ordinal'] - matches['pref_price_ordinal']).abs()
keep_mask = (matches['pref_price_ordinal'] < 0) | (matches['price_ordinal'] < 0) | (price_diff <= 1)
matches = matches[keep_mask].copy()
print(f"{elapsed()}   after price filter (±1): {len(matches):,}")

# ── 6. Rank by cbf_score, take top-N per user ─────────────────────────────────
print(f"{elapsed()} Ranking top-{TOP_N} per user …")
matches = matches.sort_values(['user_id','cbf_score'], ascending=[True, False])
matches['cbf_rank'] = matches.groupby('user_id').cumcount() + 1
matches = matches[matches['cbf_rank'] <= TOP_N].copy()

out = matches[['user_id','item_id','cbf_score','cbf_rank']].reset_index(drop=True)
print(f"{elapsed()} Final CBF candidates: {len(out):,} rows  "
      f"({out['user_id'].nunique():,} users, avg {len(out)/max(out['user_id'].nunique(),1):.1f}/user)")
out.to_parquet(OUT_FILE, index=False)
print(f"{elapsed()} Saved → {OUT_FILE}")
