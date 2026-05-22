"""Compare submission_stage8.csv vs submission_v6b_0.2421.csv."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

STAGE8 = 'd:/Datathon_Data/datathon-chung-ket/submission_stage8.csv'
V6_BEST = 'd:/Datathon_Data/model_v6_0.2421/submission_v6b_0.2421.csv'

print(f"Loading {STAGE8} …")
s8 = pd.read_csv(STAGE8)
print(f"Loading {V6_BEST} …")
v6 = pd.read_csv(V6_BEST)

print(f"\nstage8: {len(s8):,} rows, cols={s8.columns.tolist()}")
print(f"v6:     {len(v6):,} rows, cols={v6.columns.tolist()}")

# Build {user_id: [item_ids in rank order]} for each
def to_dict(df):
    df = df.sort_values(['user_id','rank'])
    return df.groupby('user_id')['item_id'].apply(list).to_dict()

print("\nBuilding per-user dicts …")
s8_d = to_dict(s8)
v6_d = to_dict(v6)

common_users = set(s8_d.keys()) & set(v6_d.keys())
print(f"Common users: {len(common_users):,} (s8={len(s8_d):,}, v6={len(v6_d):,})")

# Per-user metrics
jaccards = []         # Jaccard top-10 set overlap
top1_same = 0         # # users where top-1 item identical
top5_overlap = []     # # items overlap in top-5
top10_overlap = []    # # items overlap in top-10
identical_users = 0   # # users with EXACT same top-10 (same items, same order)
spearman_rhos = []    # Spearman rank correlation for shared items

for uid in common_users:
    a = s8_d[uid][:10]
    b = v6_d[uid][:10]
    set_a, set_b = set(a), set(b)
    inter = set_a & set_b
    union = set_a | set_b
    jaccards.append(len(inter) / len(union) if union else 0)
    top10_overlap.append(len(inter))
    top5_overlap.append(len(set(a[:5]) & set(b[:5])))
    if a[0] == b[0]:
        top1_same += 1
    if a == b:
        identical_users += 1
    # Rank correlation on shared items only
    if len(inter) >= 2:
        # Rank items in each list
        rank_a = {it: i for i, it in enumerate(a)}
        rank_b = {it: i for i, it in enumerate(b)}
        ranks_a = [rank_a[it] for it in inter]
        ranks_b = [rank_b[it] for it in inter]
        from scipy.stats import spearmanr
        rho, _ = spearmanr(ranks_a, ranks_b)
        if not np.isnan(rho):
            spearman_rhos.append(rho)

jaccards = np.array(jaccards)
top10_overlap = np.array(top10_overlap)
top5_overlap = np.array(top5_overlap)
spearman_rhos = np.array(spearman_rhos)

print(f"\n=== Similarity metrics (n={len(common_users):,} common users) ===")
print(f"Top-10 Jaccard:     mean={jaccards.mean():.4f}  median={np.median(jaccards):.4f}  std={jaccards.std():.4f}")
print(f"Top-10 overlap (#): mean={top10_overlap.mean():.2f}/10  median={np.median(top10_overlap):.0f}/10")
print(f"Top-5  overlap (#): mean={top5_overlap.mean():.2f}/5   median={np.median(top5_overlap):.0f}/5")
print(f"Top-1 same item:    {top1_same:,} users ({100*top1_same/len(common_users):.2f}%)")
print(f"Identical top-10:   {identical_users:,} users ({100*identical_users/len(common_users):.2f}%)")
if len(spearman_rhos):
    print(f"Spearman on shared (≥2 items): mean={spearman_rhos.mean():.4f}  median={np.median(spearman_rhos):.4f}")

# Distribution of overlap
print(f"\n=== Top-10 overlap distribution ===")
for k in range(11):
    pct = 100 * (top10_overlap == k).mean()
    print(f"  {k}/10 overlap: {pct:5.2f}% of users")

# Compare COLD vs WARM splits
print(f"\n=== Cold vs Warm analysis (cold = no pos history) ===")
pos = pd.read_parquet('d:/Datathon_Data/model/cache/user_item_pos.parquet', columns=['user_id'])
warm_users = set(pos['user_id'].unique())
cold_users = common_users - warm_users
warm_in_common = common_users & warm_users
print(f"Warm: {len(warm_in_common):,}  Cold: {len(cold_users):,}")

warm_overlap = np.array([top10_overlap[i] for i, uid in enumerate(common_users) if uid in warm_users])
cold_overlap = np.array([top10_overlap[i] for i, uid in enumerate(common_users) if uid not in warm_users])
if len(warm_overlap):
    print(f"Warm users top-10 overlap: mean={warm_overlap.mean():.2f}/10")
if len(cold_overlap):
    print(f"Cold users top-10 overlap: mean={cold_overlap.mean():.2f}/10")
