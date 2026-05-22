"""
Step 8: Validate format + generate final submission.csv
- Scale-aware freshness boost for listings ≤7 days old
- Category-weighted popular fallback (200 items, weighted by contact volume)
- Cold-user strategy (env COLD_STRATEGY): 'global' (v6 default) | 'session' | 'cvr'
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

COLD_STRATEGY = os.environ.get('COLD_STRATEGY', 'global').lower()
assert COLD_STRATEGY in ('global','session','cvr'), f"COLD_STRATEGY={COLD_STRATEGY} invalid"

# Stage 14 Hook #3 Part B: post-rank freshness multiplier (1.5x for fresh+quality items)
# OFF by default — only enabled for submission B (A/B-safety-gated)
FRESH_MULTIPLIER = os.environ.get('FRESH_MULTIPLIER', 'off').lower()
assert FRESH_MULTIPLIER in ('on','off'), f"FRESH_MULTIPLIER={FRESH_MULTIPLIER} invalid"
FRESH_SAFETY_MIN_OVERLAP = float(os.environ.get('FRESH_SAFETY_MIN_OVERLAP', '7.0'))  # require ≥7/10 avg overlap

# Stage 9: COLD_MODE controls cold-user override source
#  - global: raw trend_pos (Stage 8 default, Public LB 0.2430)
#  - track:  Cold-Track Specialist top-10 from cold_top10.pkl
#  - track_safe: blend cold_track + trend_pos (anchored to popular)
#  - age_boost: trend_pos × age_boost_cat (fallback heuristic)
COLD_MODE = os.environ.get('COLD_MODE', 'global').lower()
assert COLD_MODE in ('global','track','track_safe','age_boost'), f"COLD_MODE={COLD_MODE} invalid"

# Stage 9: BLEND_WEIGHT for LightGBM vs CatBoost — Stage 12: extended to 3-way with XGBoost
BLEND_WEIGHT     = float(os.environ.get('BLEND_WEIGHT', '1.0'))         # LGBM weight
BLEND_WEIGHT_CB  = float(os.environ.get('BLEND_WEIGHT_CB', '0.0'))      # CatBoost weight
BLEND_WEIGHT_XGB = float(os.environ.get('BLEND_WEIGHT_XGB', '0.0'))     # XGBoost weight (Stage 12)
# If weights don't sum to 1.0, normalize
_total = BLEND_WEIGHT + BLEND_WEIGHT_CB + BLEND_WEIGHT_XGB
if _total > 0 and abs(_total - 1.0) > 0.001:
    BLEND_WEIGHT, BLEND_WEIGHT_CB, BLEND_WEIGHT_XGB = (BLEND_WEIGHT/_total, BLEND_WEIGHT_CB/_total, BLEND_WEIGHT_XGB/_total)

# Backwards-compat: if only BLEND_WEIGHT set and < 1.0, derive CB from rest (Stage 9 behavior)
if BLEND_WEIGHT_CB == 0.0 and BLEND_WEIGHT_XGB == 0.0 and BLEND_WEIGHT < 1.0:
    BLEND_WEIGHT_CB = 1.0 - BLEND_WEIGHT

print(f"COLD_STRATEGY = {COLD_STRATEGY}  COLD_MODE = {COLD_MODE}")
print(f"BLEND: LGBM={BLEND_WEIGHT:.2f}, CatBoost={BLEND_WEIGHT_CB:.2f}, XGBoost={BLEND_WEIGHT_XGB:.2f}")

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

print(f"{elapsed()} Loading predictions …")
ranked   = pd.read_parquet(f"{CACHE_DIR}/ranked_predictions.parquet")

# Stage 9/12: 3-way blend with CatBoost + XGBoost
def per_user_norm(df, col):
    mn = df.groupby('user_id')[col].transform('min')
    mx = df.groupby('user_id')[col].transform('max')
    return (df[col] - mn) / (mx - mn + 1e-10)

ranked['lgbm_norm'] = per_user_norm(ranked, 'lgbm_score')
final_score = BLEND_WEIGHT * ranked['lgbm_norm']
cols_to_drop = ['lgbm_norm']

if BLEND_WEIGHT_CB > 0:
    cb_path = f"{CACHE_DIR}/catboost_predictions.parquet"
    if os.path.exists(cb_path):
        cb = pd.read_parquet(cb_path)
        print(f"{elapsed()} Loaded CatBoost predictions: {len(cb):,} rows")
        ranked = ranked.merge(cb[['user_id','item_id','catboost_score']],
                                on=['user_id','item_id'], how='left')
        ranked['catboost_norm'] = per_user_norm(ranked, 'catboost_score').fillna(0)
        final_score = final_score + BLEND_WEIGHT_CB * ranked['catboost_norm']
        cols_to_drop += ['catboost_norm', 'catboost_score']
        del cb
    else:
        print(f"{elapsed()} ⚠ catboost_predictions.parquet not found")

if BLEND_WEIGHT_XGB > 0:
    xgb_path = f"{CACHE_DIR}/xgboost_predictions.parquet"
    if os.path.exists(xgb_path):
        xg = pd.read_parquet(xgb_path)
        print(f"{elapsed()} Loaded XGBoost predictions: {len(xg):,} rows")
        ranked = ranked.merge(xg[['user_id','item_id','xgboost_score']],
                                on=['user_id','item_id'], how='left')
        ranked['xgboost_norm'] = per_user_norm(ranked, 'xgboost_score').fillna(0)
        final_score = final_score + BLEND_WEIGHT_XGB * ranked['xgboost_norm']
        cols_to_drop += ['xgboost_norm', 'xgboost_score']
        del xg
    else:
        print(f"{elapsed()} ⚠ xgboost_predictions.parquet not found")

# Replace lgbm_score with blended score (downstream code uses lgbm_score)
ranked['lgbm_score'] = final_score
ranked = ranked.drop(columns=cols_to_drop)
print(f"{elapsed()} Blended scores: {BLEND_WEIGHT:.2f}×LGBM + {BLEND_WEIGHT_CB:.2f}×CB + {BLEND_WEIGHT_XGB:.2f}×XGB")
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
items_meta = (items[['item_id','posted_date','category','legal_status','direction','seller_type',
                      'images_count','house_type' if 'house_type' in items.columns else 'category']]
                .drop_duplicates('item_id').copy())
items_meta['posted_date'] = pd.to_datetime(items_meta['posted_date'])
items_meta['days_since_posted'] = (train_end_dt - items_meta['posted_date']).dt.days.clip(lower=0)
ranked = ranked.merge(items_meta[['item_id','days_since_posted','category']],
                      on='item_id', how='left')
score_range = ranked['lgbm_score'].max() - ranked['lgbm_score'].min()
freshness_boost = score_range * 0.015
ranked['lgbm_score'] += (ranked['days_since_posted'].fillna(999) <= 7).astype(float) * freshness_boost

# ── Stage 14 Hook #3 Part B: Post-rank Freshness Multiplier (A/B-safety-gated) ──
if FRESH_MULTIPLIER == 'on':
    print(f"{elapsed()} [FRESH_MULTIPLIER=on] Computing item_quality_score for multiplier …")
    # Mirror 06_features.py composite formula (item-level only)
    strong_legal = {'Đã có sổ', 'Sổ hồng riêng', 'Sổ hồng / sổ đỏ', 'Sổ đỏ'}
    trusted_sellers = {'Cá nhân', 'Cá nhân tự đăng', 'Môi giới chuyên nghiệp'}
    iq_df = items_meta.copy()
    _has_legal  = iq_df['legal_status'].isin(strong_legal).astype(np.float32)
    _has_dir    = iq_df['direction'].notna().astype(np.float32)
    _has_house  = iq_df['house_type'].notna().astype(np.float32) if 'house_type' in iq_df.columns else 0.0
    _has_imgs   = (iq_df['images_count'].fillna(0) > 3).astype(np.float32)
    _trusted    = iq_df['seller_type'].isin(trusted_sellers).astype(np.float32)
    iq_df['item_quality_score'] = (
        0.20 * _has_imgs + 0.20 * _trusted + 0.30 * _has_legal +
        0.15 * _has_house + 0.15 * _has_dir
    ).astype(np.float32)
    ranked = ranked.merge(iq_df[['item_id','item_quality_score']], on='item_id', how='left')
    ranked['item_quality_score'] = ranked['item_quality_score'].fillna(0).astype(np.float32)

    # Multiplier formula per plan:
    #   <1d & q>0.7 → 1.5x ; <3d & q>0.6 → 1.25x ; <7d → linear decay 1.0+0.1*(7-d)/7 ; else 1.0
    d = ranked['days_since_posted'].fillna(999).astype(np.float32)
    q = ranked['item_quality_score']
    mult = np.where((d < 1) & (q > 0.7), 1.50,
           np.where((d < 3) & (q > 0.6), 1.25,
           np.where(d < 7,                1.0 + 0.1 * (7 - d) / 7,
                                          1.0))).astype(np.float32)
    ranked['fresh_mult'] = mult

    # A/B safety: compute top-10 per user WITHOUT vs WITH multiplier; require ≥7/10 avg overlap.
    print(f"{elapsed()}   A/B safety check (top-10 overlap) …")
    safety_sample = ranked.sort_values(['user_id','lgbm_score','blend_score'],
                                        ascending=[True, False, False])
    safety_sample['rk_before'] = safety_sample.groupby('user_id').cumcount() + 1
    safety_before = safety_sample[safety_sample['rk_before'] <= 10][['user_id','item_id','rk_before']].copy()

    safety_sample['lgbm_score_mult'] = safety_sample['lgbm_score'] * safety_sample['fresh_mult']
    safety_sample = safety_sample.sort_values(['user_id','lgbm_score_mult','blend_score'],
                                                ascending=[True, False, False])
    safety_sample['rk_after'] = safety_sample.groupby('user_id').cumcount() + 1
    safety_after = safety_sample[safety_sample['rk_after'] <= 10][['user_id','item_id','rk_after']].copy()

    merged_ab = safety_before.merge(safety_after, on=['user_id','item_id'], how='inner')
    overlap_per_user = merged_ab.groupby('user_id').size()
    avg_overlap = overlap_per_user.mean()
    print(f"{elapsed()}   Avg top-10 overlap WITH vs WITHOUT multiplier: {avg_overlap:.2f}/10")
    del safety_sample, safety_before, safety_after, merged_ab, overlap_per_user

    if avg_overlap >= FRESH_SAFETY_MIN_OVERLAP:
        print(f"{elapsed()}   ✓ A/B SAFE (≥{FRESH_SAFETY_MIN_OVERLAP:.1f}/10) — applying multiplier")
        ranked['lgbm_score'] = ranked['lgbm_score'] * ranked['fresh_mult']
    else:
        print(f"{elapsed()}   ✗ A/B UNSAFE (<{FRESH_SAFETY_MIN_OVERLAP:.1f}/10) — REVERTED (no multiplier)")
    ranked = ranked.drop(columns=['fresh_mult','item_quality_score'])
    del iq_df, _has_legal, _has_dir, _has_house, _has_imgs, _trusted

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

# ── Cold-user strategy ───────────────────────────────────────────────────────
print(f"{elapsed()} Building cold-user recommendations (strategy={COLD_STRATEGY}, mode={COLD_MODE}) …")

# Build raw global popular top-10 (always available as baseline/fallback)
if COLD_STRATEGY == 'cvr':
    pop_w = pop.groupby('item_id').agg(
        trend_pos=('trend_pos','sum'),
        trend_events=('trend_events','sum')).reset_index()
    pop_w['cvr'] = pop_w['trend_pos'] / pop_w['trend_events'].clip(lower=1)
    pop_w['cvr_score'] = pop_w['trend_pos'] * pop_w['cvr']
    global_pop_top10 = (pop_w[pop_w['item_id'].isin(valid_items)]
                        .sort_values('cvr_score', ascending=False)
                        .head(10)['item_id'].tolist())
else:
    global_pop_top10 = (pop[pop['item_id'].isin(valid_items)]
                        .sort_values('trend_pos', ascending=False)
                        .drop_duplicates('item_id')
                        .head(10)['item_id'].tolist())
print(f"{elapsed()} Global popular top-10 ready: {global_pop_top10[0][:16]}…")

# Stage 9: COLD_MODE selection
cold_top10 = global_pop_top10  # default
if COLD_MODE == 'track':
    import pickle
    cold_pkl = f"{CACHE_DIR}/cold_top10.pkl"
    if os.path.exists(cold_pkl):
        with open(cold_pkl, 'rb') as f:
            cold_track_top10 = pickle.load(f)
        # Safety check: overlap with raw global
        overlap = len(set(cold_track_top10) & set(global_pop_top10))
        print(f"{elapsed()} Cold-Track top-10 overlap with global: {overlap}/10")
        if overlap >= 7:
            cold_top10 = cold_track_top10
            print(f"{elapsed()} Cold-Track SAFE (>=7/10 overlap) — using cold_track top-10")
        elif overlap >= 4:
            print(f"{elapsed()} Cold-Track MEDIUM RISK — blending with global (track_safe mode)")
            # Take 5 from track + 5 from global, dedup
            cold_top10 = []
            for items in zip(cold_track_top10, global_pop_top10):
                for it in items:
                    if it not in cold_top10:
                        cold_top10.append(it)
                    if len(cold_top10) >= 10:
                        break
                if len(cold_top10) >= 10:
                    break
            cold_top10 = cold_top10[:10]
        else:
            print(f"{elapsed()} ⚠ Cold-Track HIGH RISK (<4/10 overlap) — fall back to global")
            # cold_top10 stays as global_pop_top10
    else:
        print(f"{elapsed()} cold_top10.pkl not found, using global")
elif COLD_MODE == 'age_boost':
    # trend_pos × age_boost_cat heuristic
    items_with_pop = pop.groupby('item_id').agg(trend_pos=('trend_pos','sum')).reset_index()
    items_meta_local = items[['item_id','category','posted_date']].drop_duplicates('item_id').copy()
    items_meta_local['posted_date'] = pd.to_datetime(items_meta_local['posted_date'])
    items_meta_local['days_since_posted'] = (train_end_dt - items_meta_local['posted_date']).dt.days.clip(lower=0)
    def age_boost(row):
        age, cat = row['days_since_posted'], row['category']
        if cat == 1050:
            return 1.00 if age < 15 else (1.05 if age < 30 else (1.08 if age < 60 else 1.10))
        else:
            return 1.20 if age <= 2 else (1.10 if age <= 7 else (1.03 if age <= 14 else (1.00 if age <= 29 else 0.92)))
    items_meta_local['age_boost_cat'] = items_meta_local.apply(age_boost, axis=1)
    merged = items_with_pop.merge(items_meta_local[['item_id','age_boost_cat']], on='item_id', how='left')
    merged['adj_score'] = merged['trend_pos'] * merged['age_boost_cat'].fillna(1.0)
    merged = merged[merged['item_id'].isin(valid_items)]
    cold_top10 = merged.sort_values('adj_score', ascending=False).head(10)['item_id'].tolist()
    print(f"{elapsed()} COLD_MODE=age_boost: top-10 by trend_pos × age_boost_cat")

print(f"{elapsed()} Cold top-10 selected ({COLD_MODE}): {cold_top10[0][:16]}…")

n_overridden_global  = 0
n_overridden_session = 0

if COLD_STRATEGY == 'session':
    # Tier 1: cold users with a first_click → item-to-item by (cat, city, district, price)
    # Tier 3: cold users without first_click → global popular fallback
    print(f"{elapsed()} Loading session first_click + snapshot for item-to-item …")
    first_click = pd.read_parquet(f"{CACHE_DIR}/user_first_click.parquet",
                                   columns=['user_id','item_id','category','city_name'])
    # Snapshot for ranking item-to-item candidates by recent demand
    try:
        snap_feat = pd.read_parquet(f"{CACHE_DIR}/snapshot_features.parquet",
                                     columns=['item_id','contacts_24h','views_24h'])
    except Exception:
        snap_feat = pd.DataFrame(columns=['item_id','contacts_24h','views_24h'])

    # Item lookup for first-click metadata (district, price_bucket)
    item_meta = (items[['item_id','category','city_name','district_name','price_bucket']]
                  .drop_duplicates('item_id'))
    first_click = (first_click.merge(item_meta.rename(columns={
        'category':'fc_category','city_name':'fc_city','district_name':'fc_district','price_bucket':'fc_price'}),
        on='item_id', how='left'))
    # Drop the click item itself when generating recommendations
    first_click = first_click.rename(columns={'item_id':'fc_item_id'})
    first_click = first_click[first_click['user_id'].isin(cold_users)]
    fc_dict = first_click.set_index('user_id').to_dict('index')
    print(f"{elapsed()} cold users with first_click: {len(fc_dict):,} / {len(cold_users):,}")

    # Precompute per-(category, district, price_bucket) top items by snapshot contacts_24h
    items_ranked = item_meta.merge(snap_feat, on='item_id', how='left')
    items_ranked['contacts_24h'] = items_ranked['contacts_24h'].fillna(0)
    items_ranked = items_ranked.sort_values('contacts_24h', ascending=False)
    items_ranked = items_ranked[items_ranked['item_id'].isin(valid_items)]
    # Index by (category, district) for fast lookup — primary affinity
    cat_dist_groups = items_ranked.groupby(['category','district_name'])
    cat_city_groups = items_ranked.groupby(['category','city_name'])
    cat_only_groups = items_ranked.groupby('category')

    def session_rec(uid, click_item_id, fc_category, fc_city, fc_district, fc_price):
        recs, seen = [], {click_item_id}
        # Layer 1: same (category, district)
        key = (fc_category, fc_district)
        if key in cat_dist_groups.groups:
            for iid in cat_dist_groups.get_group(key)['item_id'].head(15):
                if iid not in seen:
                    recs.append(iid); seen.add(iid)
                if len(recs) >= 10:
                    return recs
        # Layer 2: same (category, city)
        key = (fc_category, fc_city)
        if key in cat_city_groups.groups:
            for iid in cat_city_groups.get_group(key)['item_id'].head(15):
                if iid not in seen:
                    recs.append(iid); seen.add(iid)
                if len(recs) >= 10:
                    return recs
        # Layer 3: same category only
        if fc_category in cat_only_groups.groups:
            for iid in cat_only_groups.get_group(fc_category)['item_id'].head(15):
                if iid not in seen:
                    recs.append(iid); seen.add(iid)
                if len(recs) >= 10:
                    return recs
        # Layer 4: fall back to global popular
        for iid in global_pop_top10:
            if iid not in seen:
                recs.append(iid); seen.add(iid)
            if len(recs) >= 10:
                return recs
        return recs[:10]

    for uid in cold_users:
        if uid not in user_items:
            continue
        if uid in fc_dict:
            d = fc_dict[uid]
            recs = session_rec(uid, d['fc_item_id'], d['fc_category'],
                               d['fc_city'], d['fc_district'], d['fc_price'])
            if len(recs) == 10:
                user_items[uid] = recs
                n_overridden_session += 1
                continue
        # Tier 3 — true cold or fallback (uses cold_top10 which respects COLD_MODE)
        user_items[uid] = cold_top10[:10]
        n_overridden_global += 1
    print(f"{elapsed()} Session override: {n_overridden_session:,} | Cold fallback: {n_overridden_global:,}")
else:
    # 'global' or 'cvr' — all cold users get cold_top10 (respects COLD_MODE)
    for uid in cold_users:
        if uid in user_items:
            user_items[uid] = cold_top10[:10]
            n_overridden_global += 1
    print(f"{elapsed()} Overrode {n_overridden_global:,} cold users with {COLD_MODE} top-10")

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
