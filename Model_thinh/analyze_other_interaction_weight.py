"""
Phân tích tương quan other_interaction vs explicit contact
để đánh giá trọng số phù hợp cho other_interaction trong pipeline.

Câu hỏi chính:
  1. other_interaction có predict explicit contact không? (item-level)
  2. User có other_interaction cao → contact rate có khác không? (user-level)
  3. Trọng số tối ưu theo precision@k với các weight thử nghiệm
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats

CACHE = "model/cache"

print("=" * 60)
print("LOAD DATA")
print("=" * 60)

pos = pd.read_parquet(f"{CACHE}/user_item_pos.parquet")
iq  = pd.read_parquet(f"{CACHE}/item_quality.parquet")

print(f"pos: {len(pos):,} rows, {pos['user_id'].nunique():,} users, {pos['item_id'].nunique():,} items")
print(f"iq:  {len(iq):,} items")
print(f"\npos columns: {list(pos.columns)}")
print(f"iq  columns: {list(iq.columns)}")

# ── 1. Phân phối event types ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. PHÂN PHỐI EVENT TYPES TRONG pos TABLE")
print("=" * 60)

pos['explicit'] = pos['n_view_phone'].fillna(0) + pos['n_chat'].fillna(0) + \
                  pos['n_zalo'].fillna(0) + pos['n_sms'].fillna(0)
pos['n_other']  = pos['n_other'].fillna(0)
pos['pos_count'] = pos['pos_count'].fillna(0)

total_explicit = pos['explicit'].sum()
total_other    = pos['n_other'].sum()
total_pos      = pos['pos_count'].sum()

print(f"  explicit contacts : {total_explicit:,.0f}  ({100*total_explicit/total_pos:.1f}%)")
print(f"  other_interaction : {total_other:,.0f}  ({100*total_other/total_pos:.1f}%)")
print(f"  total pos_events  : {total_pos:,.0f}")

# ── 2. User-level: other_interaction → có contact không? ──────────────────────
print("\n" + "=" * 60)
print("2. USER-LEVEL: other_interaction → explicit contact rate")
print("=" * 60)

user_agg = pos.groupby('user_id').agg(
    total_explicit=('explicit', 'sum'),
    total_other=('n_other', 'sum'),
    n_items=('item_id', 'nunique'),
).reset_index()

user_agg['has_explicit']    = (user_agg['total_explicit'] > 0).astype(int)
user_agg['other_per_item']  = user_agg['total_other'] / user_agg['n_items'].clip(lower=1)
user_agg['explicit_ratio']  = user_agg['total_explicit'] / (
    user_agg['total_explicit'] + user_agg['total_other']).clip(lower=1)

# Group users by other_interaction level
user_agg['other_bucket'] = pd.qcut(
    user_agg['total_other'].clip(upper=200), q=5,
    labels=['Q1(thấp)', 'Q2', 'Q3', 'Q4', 'Q5(cao)'], duplicates='drop'
)
grp = user_agg.groupby('other_bucket', observed=True).agg(
    n_users=('user_id', 'count'),
    pct_has_explicit=('has_explicit', 'mean'),
    avg_explicit=('total_explicit', 'mean'),
    avg_other=('total_other', 'mean'),
    explicit_ratio=('explicit_ratio', 'mean'),
).reset_index()
print(grp.to_string(index=False))

corr_spearman, p_val = stats.spearmanr(
    user_agg['total_other'], user_agg['total_explicit'])
print(f"\n  Spearman corr(other, explicit) = {corr_spearman:.4f}  (p={p_val:.2e})")

# ── 3. Item-level: other_interaction → item CVR ────────────────────────────────
print("\n" + "=" * 60)
print("3. ITEM-LEVEL: other_interaction → explicit CVR")
print("=" * 60)

item_agg = pos.groupby('item_id').agg(
    total_explicit=('explicit', 'sum'),
    total_other=('n_other', 'sum'),
    n_users=('user_id', 'nunique'),
).reset_index()
item_agg = item_agg.merge(
    iq[['item_id','total_events','pos_events','pageviews']], on='item_id', how='left')

item_agg['item_explicit_cvr'] = item_agg['total_explicit'] / \
    item_agg['pageviews'].fillna(0).clip(lower=1)
item_agg['item_other_rate']   = item_agg['total_other'] / \
    item_agg['pageviews'].fillna(0).clip(lower=1)

item_agg['other_bucket'] = pd.qcut(
    item_agg['total_other'].clip(upper=500), q=5,
    labels=['Q1(thấp)', 'Q2', 'Q3', 'Q4', 'Q5(cao)'], duplicates='drop'
)
igrp = item_agg.groupby('other_bucket', observed=True).agg(
    n_items=('item_id', 'count'),
    avg_other=('total_other', 'mean'),
    avg_explicit=('total_explicit', 'mean'),
    avg_explicit_cvr=('item_explicit_cvr', 'mean'),
    avg_other_rate=('item_other_rate', 'mean'),
).reset_index()
print(igrp.to_string(index=False))

corr_i, p_i = stats.spearmanr(item_agg['total_other'], item_agg['total_explicit'])
print(f"\n  Spearman corr(item other, item explicit) = {corr_i:.4f}  (p={p_i:.2e})")

# ── 4. Pair-level: với cùng 1 user-item pair, other trước → explicit sau ──────
print("\n" + "=" * 60)
print("4. PAIR-LEVEL: other_interaction có dự báo explicit không?")
print("=" * 60)

pos_both = pos[pos['n_other'] > 0].copy()
print(f"  Pairs có other_interaction: {len(pos_both):,}  ({100*len(pos_both)/len(pos):.1f}%)")

# Trong những pair có other → tỷ lệ cũng có explicit
pct_also_explicit = (pos_both['explicit'] > 0).mean()
pct_explicit_all  = (pos['explicit'] > 0).mean()
print(f"  Tỷ lệ pairs có BOTH other + explicit : {100*pct_also_explicit:.1f}%")
print(f"  Tỷ lệ pairs có explicit (toàn bộ)    : {100*pct_explicit_all:.1f}%")
print(f"  Lift: {pct_also_explicit/pct_explicit_all:.2f}x")

# ── 5. Đề xuất trọng số dựa trên lift ─────────────────────────────────────────
print("\n" + "=" * 60)
print("5. ĐỀ XUẤT TRỌNG SỐ")
print("=" * 60)

# Baseline: explicit = 1.0 (normalized)
# other → explicit lift ở pair level
lift = pct_also_explicit / pct_explicit_all

# Spearman corr item-level (strength of signal)
# Weight proposal: corr * lift (bounded 0..1) * baseline_explicit_weight
explicit_weight = 10.0
proposed_weight = max(0.5, round(explicit_weight * corr_i * min(lift, 2.0) * 0.5, 1))

print(f"  Spearman corr (item) : {corr_i:.4f}")
print(f"  Pair-level lift      : {lift:.2f}x")
print(f"\n  Trọng số hiện tại:")
print(f"    view_phone/chat/zalo/sms = 10.0  (hist_decay)")
print(f"    other_interaction        = 2.5   (hist_decay)  /  1 (lukewarm)")
print(f"    pageview                 = -      (hist_decay)  /  2 (lukewarm)")
print(f"\n  Đề xuất (dựa trên corr × lift):")
print(f"    other_interaction hist_decay weight ≈ {proposed_weight}")
print(f"    Nếu corr yếu (<0.2): nên dùng ~1.0 (noise buffer)")
print(f"    Nếu corr mạnh (>0.4): có thể dùng 2.0-3.0")

# ── 6. Phân tích theo category ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. THEO CATEGORY")
print("=" * 60)

if 'category' in pos.columns:
    cat_agg = pos.groupby('category').agg(
        total_explicit=('explicit','sum'),
        total_other=('n_other','sum'),
    ).reset_index()
    cat_agg['other_pct'] = 100 * cat_agg['total_other'] / \
        (cat_agg['total_explicit'] + cat_agg['total_other']).clip(lower=1)
    cat_agg['explicit_pct'] = 100 - cat_agg['other_pct']
    print(cat_agg.sort_values('other_pct', ascending=False).to_string(index=False))

print("\nDONE")
