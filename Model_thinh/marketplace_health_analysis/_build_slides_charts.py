"""
Pre-generates slide-optimized chart PNGs (16:9 dense layout style).
Output to: marketplace_health_analysis/slides/charts/
"""
import os, sys, glob, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path('/Volumes/mavuong3112/Datathon_Data')
CACHE_V16  = ROOT / 'model_v16_0.xxxx' / 'cache'
CACHE_OLD  = ROOT / 'model' / 'cache'
DIM_DIR    = ROOT / 'dim_listing'
OUT_BASE   = ROOT / 'marketplace_health_analysis'
OUT_DIR    = OUT_BASE / 'slides' / 'charts'
BEST_SUB   = ROOT / 'model_v16_0.xxxx' / 'submission_stage15_0.2441.csv'

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Output: {OUT_DIR}')

# ── Slide-friendly palette (more business style) ──────────────────────────────
P = {
    'PRIMARY':   '#0F2C5F',   # deep navy (slide headers)
    'TEAL':      '#1F8E9D',   # accent teal
    'ORANGE':    '#E07B00',   # warm orange (warnings/highlights)
    'GREEN':     '#2E7D32',   # success
    'MUTED':     '#9AA5B1',
    'LIGHT_BG':  '#F4F6FA',
    'DARK_TEXT': '#1A1A2E',
    'ACCENT':    '#C62828',
    'TILE_BLUE': '#2196F3',
    'TILE_AMBER':'#FB8C00',
    'TILE_RED':  '#E53935',
}

# Set base style
plt.rcParams['font.family']     = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans','Arial Unicode MS','sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titleweight']= 'bold'
plt.rcParams['axes.titlesize']  = 11
plt.rcParams['axes.titlecolor'] = P['PRIMARY']

def save_fig(name, fig, dpi=180):
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  ✓ {name}  ({path.stat().st_size//1024} KB)')

# ═════════════════════════════════════════════════════════════════════════════
# CHART 1 — LB Progression (wide compact for slide top)
# ═════════════════════════════════════════════════════════════════════════════
progression = pd.DataFrame([
    ('v1\nbaseline',  0.2184, 'ALS only'),
    ('v6',            0.2421, '+ Multi-retriever'),
    ('v8',            0.2430, '+ Behavioral'),
    ('v10',           0.2436, '+ Freshness'),
    ('v11',           0.2438, '+ District'),
    ('v12',           0.2440, 'BIGGER LGBM'),
    ('v15\nfinal',    0.2441, '+ XGB blend'),
], columns=['Version', 'Recall', 'Change'])

fig, ax = plt.subplots(figsize=(7.5, 3.0), facecolor='white')
bars = ax.bar(progression['Version'], progression['Recall'],
              color=P['TILE_BLUE'], edgecolor='white', linewidth=1.5, zorder=3)
bars[-1].set_color(P['GREEN'])
for bar, val in zip(bars, progression['Recall']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{val:.4f}', ha='center', fontsize=8, fontweight='bold', color=P['DARK_TEXT'])
ax.axhline(0.0058, ls='--', color=P['MUTED'], linewidth=1, label='Popularity baseline (0.0058)')
ax.set_ylim(0, 0.265)
ax.set_ylabel('Recall@10', fontsize=9, color=P['DARK_TEXT'])
ax.set_title('Public LB Progression — Recall@10 (v1 → v15)', pad=8)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0); ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=7.5, frameon=False)
ax.tick_params(axis='x', labelsize=8)
plt.tight_layout()
save_fig('chart_lb_progression.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 2 — Feature importance horizontal bar (compact)
# ═════════════════════════════════════════════════════════════════════════════
booster = lgb.Booster(model_file=str(CACHE_V16 / 'lgbm_ranker.txt'))
fi = pd.Series(
    booster.feature_importance(importance_type='gain'),
    index=booster.feature_name()
).sort_values(ascending=True)
top15 = fi.tail(15)

fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor='white')
ax.barh(top15.index, top15.values, color=P['TEAL'], edgecolor='white', linewidth=0.8)
ax.set_title('Top-15 Feature Importance (LGBM gain)', pad=8)
ax.set_xlabel('Gain', fontsize=8, color=P['DARK_TEXT'])
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.35); ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=7)
plt.tight_layout()
save_fig('chart_feature_importance.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 3 — Personalization vs Popularity (compact)
# ═════════════════════════════════════════════════════════════════════════════
baseline_sub = pd.read_csv(BEST_SUB, usecols=['user_id','rank','item_id'])
pop = pq.read_table(CACHE_OLD / 'popular_items.parquet',
                    columns=['item_id','trend_pos']).to_pandas()
pop_top10 = (pop.sort_values('trend_pos', ascending=False)
                .drop_duplicates('item_id')
                .head(10)['item_id'].tolist())
base_sets = baseline_sub.groupby('user_id')['item_id'].apply(set)
pop_set = set(pop_top10)
overlap = base_sets.apply(lambda s: len(s & pop_set))

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), facecolor='white')
axes[0].bar(['Popularity\nbaseline','Our best\n(stage15)'],
            [0.0058, 0.2441],
            color=[P['MUTED'], P['GREEN']], edgecolor='white', linewidth=1.5)
for i, v in enumerate([0.0058, 0.2441]):
    axes[0].text(i, v+0.005, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold',
                 color=P['DARK_TEXT'])
axes[0].annotate('42× lift', xy=(1, 0.244), xytext=(0.45, 0.16),
                 fontsize=10, color=P['ACCENT'], ha='center', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=P['ACCENT'], lw=1.5))
axes[0].set_ylim(0, 0.30)
axes[0].set_title('Recall@10 vs Popularity Baseline', pad=8)
axes[0].set_ylabel('Recall@10', fontsize=9)
axes[0].spines[['top','right']].set_visible(False)
axes[0].grid(axis='y', linestyle='--', alpha=0.35); axes[0].set_axisbelow(True)

axes[1].hist(overlap.values, bins=range(0, 12), color=P['TILE_BLUE'],
             edgecolor='white', linewidth=1, alpha=0.85)
axes[1].axvline(overlap.mean(), ls='--', color=P['ACCENT'], lw=1.5,
                label=f'mean = {overlap.mean():.2f}/10')
axes[1].set_title('Top-10 Overlap với Popularity', pad=8)
axes[1].set_xlabel('# items trùng popular', fontsize=9)
axes[1].set_ylabel('# users', fontsize=9)
axes[1].legend(fontsize=8, loc='upper right')
axes[1].spines[['top','right']].set_visible(False)
axes[1].grid(axis='y', linestyle='--', alpha=0.35); axes[1].set_axisbelow(True)
plt.tight_layout()
save_fig('chart_personalization.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 4 — Score distribution + rank bucket
# ═════════════════════════════════════════════════════════════════════════════
rp = pq.read_table(CACHE_V16 / 'ranked_predictions.parquet',
                   columns=['user_id','item_id','lgbm_score','blend_score','rank']).to_pandas()

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), facecolor='white')

# Score distribution
sample = rp.sample(n=150_000, random_state=42)
axes[0].hist(sample['lgbm_score'], bins=50, alpha=0.6, color=P['TILE_BLUE'],
             label='LGBM score', density=True)
ax_r = axes[0].twinx()
ax_r.hist(sample['blend_score'], bins=50, alpha=0.45, color=P['ORANGE'],
          label='blend_score', density=True)
axes[0].set_title('Phân bố score (LGBM vs blend)', pad=8)
axes[0].set_xlabel('Score', fontsize=9)
axes[0].set_ylabel('LGBM density', fontsize=8, color=P['TILE_BLUE'])
ax_r.set_ylabel('Blend density', fontsize=8, color=P['ORANGE'])
axes[0].spines[['top']].set_visible(False); ax_r.spines[['top']].set_visible(False)
axes[0].legend(loc='upper left', fontsize=7.5)
ax_r.legend(loc='upper right', fontsize=7.5)

# Rank bucket boxplot
rp['rank_bucket'] = pd.cut(rp['rank'], bins=[0,3,10,30],
                            labels=['Top-3','Mid 4-10','Tail 11-30'])
data_box = [
    rp[rp['rank_bucket']=='Top-3']['lgbm_score'].sample(40_000, random_state=42),
    rp[rp['rank_bucket']=='Mid 4-10']['lgbm_score'].sample(40_000, random_state=42),
    rp[rp['rank_bucket']=='Tail 11-30']['lgbm_score'].sample(40_000, random_state=42),
]
bp = axes[1].boxplot(data_box, labels=['Top-3','Mid 4-10','Tail 11-30'],
                     patch_artist=True, showfliers=False, widths=0.55)
for patch, c in zip(bp['boxes'], [P['GREEN'], P['ORANGE'], P['MUTED']]):
    patch.set_facecolor(c); patch.set_alpha(0.75)
for med in bp['medians']: med.set_color('#000')
axes[1].set_title('LGBM score theo rank bucket', pad=8)
axes[1].set_ylabel('lgbm_score', fontsize=9)
axes[1].spines[['top','right']].set_visible(False)
axes[1].grid(axis='y', linestyle='--', alpha=0.35); axes[1].set_axisbelow(True)
axes[1].tick_params(axis='x', labelsize=8.5)

plt.tight_layout()
save_fig('chart_score_distribution.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 5 — Lorenz curve + HHI (combined)
# ═════════════════════════════════════════════════════════════════════════════
print('Loading dim_listing…')
dim = pq.ParquetDataset(sorted(glob.glob(str(DIM_DIR / '*.parquet')))).read(
    columns=['item_id','seller_id','category','seller_type','posted_date']
).to_pandas().drop_duplicates('item_id')
TRAIN_END = pd.Timestamp('2026-04-09')
dim['posted_date'] = pd.to_datetime(dim['posted_date'], errors='coerce')
dim['days_since_post'] = (TRAIN_END - dim['posted_date']).dt.days.clip(lower=0)
dim['is_fresh']    = (dim['days_since_post'] <= 7).astype(np.int8)
dim['is_private']  = (dim['seller_type'] == 'private').astype(np.int8)

sub_full = baseline_sub.merge(
    dim[['item_id','seller_id','category','seller_type','is_fresh','is_private','days_since_post']],
    on='item_id', how='left')

seller_expo = sub_full.groupby('seller_id').size().sort_values()
item_expo   = sub_full['item_id'].value_counts()

def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return (2*np.sum(np.arange(1,n+1)*x) - (n+1)*x.sum()) / (n*x.sum())
def hhi(x):
    x = np.asarray(x, dtype=float)
    share = x / x.sum()
    return float((share**2).sum() * 10_000)

g = gini(seller_expo.values)
hhi_base = hhi(item_expo.values)
hhi_pop  = hhi(pd.Series([baseline_sub['user_id'].nunique()] * 10).values)
hhi_unif = 10_000 / item_expo.shape[0]

cum = seller_expo.cumsum() / seller_expo.sum()
x = np.linspace(0, 1, len(cum))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), facecolor='white')

axes[0].fill_between(x, cum.values, alpha=0.2, color=P['TILE_BLUE'])
axes[0].plot(x, cum.values, color=P['TILE_BLUE'], linewidth=2.5)
axes[0].plot([0,1], [0,1], '--', color=P['MUTED'], linewidth=1.2)
axes[0].set_title(f'Lorenz — Seller exposure  (Gini={g:.3f})', pad=8)
axes[0].set_xlabel('% sellers (cumulative)', fontsize=8.5)
axes[0].set_ylabel('% exposure (cumulative)', fontsize=8.5)
axes[0].spines[['top','right']].set_visible(False)
axes[0].set_xlim(0,1); axes[0].set_ylim(0,1.02)
axes[0].grid(linestyle='--', alpha=0.35); axes[0].set_axisbelow(True)

vals = [hhi_base, hhi_pop, hhi_unif]
labels = ['Best\n(stage15)','Popularity\nonly','Uniform\n(ideal)']
colors = [P['GREEN'], P['MUTED'], P['TILE_BLUE']]
axes[1].bar(labels, vals, color=colors, edgecolor='white', linewidth=1.5)
for i, v in enumerate(vals):
    axes[1].text(i, v+max(vals)*0.03, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
axes[1].set_title('HHI — Item exposure concentration', pad=8)
axes[1].set_ylabel('HHI (0-10,000; lower=diverse)', fontsize=8.5)
axes[1].spines[['top','right']].set_visible(False)
axes[1].grid(axis='y', linestyle='--', alpha=0.35); axes[1].set_axisbelow(True)

plt.tight_layout()
save_fig('chart_concentration.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 6 — Age bucket + Category mix
# ═════════════════════════════════════════════════════════════════════════════
def age_bucket(d):
    if d <= 7:    return '0-7d\n(fresh)'
    elif d <= 30: return '8-30d'
    elif d <= 90: return '31-90d'
    elif d <= 365:return '91-365d'
    else:         return '>365d'
dim['age_bucket'] = dim['days_since_post'].apply(age_bucket)
order = ['0-7d\n(fresh)','8-30d','31-90d','91-365d','>365d']
sub_age = baseline_sub.merge(dim[['item_id','age_bucket']], on='item_id', how='left')
exposed_age = sub_age['age_bucket'].value_counts(normalize=True).reindex(order).fillna(0) * 100
pool_age    = dim['age_bucket'].value_counts(normalize=True).reindex(order).fillna(0) * 100

cat_map = {1010:'Phòng\ntrọ',1020:'Căn hộ',1030:'Nhà ở',1040:'Đất nền',1050:'Dự án\nmới'}
sub_cat = baseline_sub.merge(dim[['item_id','category']], on='item_id', how='left')
exposed_cat = sub_cat['category'].value_counts(normalize=True).sort_index() * 100
pool_cat    = dim['category'].value_counts(normalize=True).sort_index() * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor='white')

# Age
x_pos = np.arange(len(order))
w = 0.38
axes[0].bar(x_pos - w/2, pool_age.values,    w, color=P['MUTED'], alpha=0.8, label='Pool', edgecolor='white')
axes[0].bar(x_pos + w/2, exposed_age.values, w, color=P['TILE_BLUE'], label='Exposed', edgecolor='white')
ymax = max(pool_age.max(), exposed_age.max())
for i, (p, e) in enumerate(zip(pool_age.values, exposed_age.values)):
    axes[0].text(i-w/2, p+ymax*0.02, f'{p:.0f}', ha='center', fontsize=7.5, color='#555')
    axes[0].text(i+w/2, e+ymax*0.02, f'{e:.0f}', ha='center', fontsize=7.5, fontweight='bold')
axes[0].set_title('Exposure theo tuổi listing (%)', pad=8)
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(order, fontsize=8)
axes[0].set_ylim(0, ymax*1.15)
axes[0].legend(fontsize=8, loc='upper left')
axes[0].spines[['top','right']].set_visible(False)
axes[0].grid(axis='y', linestyle='--', alpha=0.35); axes[0].set_axisbelow(True)

# Category
cat_labels = [cat_map[c] for c in exposed_cat.index]
x_pos = np.arange(len(exposed_cat))
axes[1].bar(x_pos - w/2, pool_cat.values,    w, color=P['MUTED'], alpha=0.8, label='Pool', edgecolor='white')
axes[1].bar(x_pos + w/2, exposed_cat.values, w, color=P['TILE_BLUE'], label='Exposed', edgecolor='white')
ymax = max(pool_cat.max(), exposed_cat.max())
for i, (p, e) in enumerate(zip(pool_cat.values, exposed_cat.values)):
    axes[1].text(i-w/2, p+ymax*0.02, f'{p:.0f}', ha='center', fontsize=7.5, color='#555')
    axes[1].text(i+w/2, e+ymax*0.02, f'{e:.0f}', ha='center', fontsize=7.5, fontweight='bold')
axes[1].set_title('Category mix (%)', pad=8)
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(cat_labels, fontsize=8)
axes[1].set_ylim(0, ymax*1.15)
axes[1].legend(fontsize=8, loc='upper right')
axes[1].spines[['top','right']].set_visible(False)
axes[1].grid(axis='y', linestyle='--', alpha=0.35); axes[1].set_axisbelow(True)

plt.tight_layout()
save_fig('chart_age_category.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 7 — Seller type donut (2 panels)
# ═════════════════════════════════════════════════════════════════════════════
expo_st = sub_full['seller_type'].value_counts(normalize=True) * 100
pool_st = dim['seller_type'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), facecolor='white')
colors_pie = [P['TILE_BLUE'], P['ORANGE']]
labels_pie = ['agent','private']

for ax, data, title in [(axes[0], pool_st, 'Pool (toàn bộ 3.1M items)'),
                         (axes[1], expo_st, 'Exposed (top-10 submission)')]:
    vals = [data.get(lbl, 0) for lbl in labels_pie]
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2.5, width=0.4),
        textprops={'fontsize': 9, 'fontweight':'bold'},
        pctdistance=0.78, labeldistance=1.15)
    for at in autotexts:
        at.set_color('white'); at.set_fontsize(9.5); at.set_fontweight('bold')
    ax.set_title(title, pad=12, fontsize=10)
plt.tight_layout()
save_fig('chart_seller_donut.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 8 — Memory footprint
# ═════════════════════════════════════════════════════════════════════════════
artifacts = [
    ('LGBM',          CACHE_V16 / 'lgbm_ranker.txt'),
    ('XGBoost',       CACHE_V16 / 'xgboost_ranker.json'),
    ('CatBoost',      CACHE_V16 / 'catboost_ranker.cbm'),
    ('Popular items', CACHE_OLD / 'popular_items.parquet'),
    ('User profiles', CACHE_OLD / 'user_profiles.parquet'),
    ('District trans.', CACHE_V16 / 'district_transition.pkl'),
    ('Cold top-10',   CACHE_OLD / 'cold_top10.pkl'),
]
sizes = [(n, p.stat().st_size / 1e6) for n,p in artifacts if p.exists()]
sizes_df = pd.DataFrame(sizes, columns=['Artifact','MB']).sort_values('MB')

fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor='white')
ax.barh(sizes_df['Artifact'], sizes_df['MB'], color=P['TEAL'], edgecolor='white', linewidth=1.2)
for i, v in enumerate(sizes_df['MB']):
    ax.text(v + max(sizes_df['MB'])*0.02, i, f'{v:,.1f} MB', va='center', fontsize=8)
ax.set_title(f'Memory footprint per artifact (tổng = {sizes_df["MB"].sum():.1f} MB)', pad=8)
ax.set_xlabel('MB', fontsize=8.5)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.35); ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=8.5)
plt.tight_layout()
save_fig('chart_memory.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 9 — Health scorecard 6-tile (for dashboard slide)
# ═════════════════════════════════════════════════════════════════════════════
freshness_pct = sub_full['is_fresh'].mean() * 100
seller_cov = sub_full['seller_id'].nunique() / dim['seller_id'].nunique() * 100
item_cov   = sub_full['item_id'].nunique() / len(dim) * 100
def normalized_entropy(x):
    x = np.asarray(x, dtype=float)
    share = x / x.sum(); share = share[share>0]
    if len(share)<=1: return 0.
    return float(-np.sum(share*np.log2(share)) / np.log2(len(share)))
cat_entropy = normalized_entropy(sub_full['category'].value_counts().values)

scorecard = [
    ('Freshness@10',  f'{freshness_pct:.1f}%', freshness_pct, [1.0, 3.0], 'higher'),
    ('Seller Coverage', f'{seller_cov:.2f}%', seller_cov,     [5.0, 15.0], 'higher'),
    ('Item Coverage', f'{item_cov:.2f}%',     item_cov,       [2.0, 5.0],  'higher'),
    ('Category Entropy', f'{cat_entropy:.3f}', cat_entropy,   [0.7, 0.9],  'higher'),
    ('Seller Gini',   f'{g:.3f}',              g,             [0.85, 0.95],'lower'),
    ('Item HHI',      f'{hhi_base:.0f}',       hhi_base,      [200, 1000], 'lower'),
]

def color_for(val, thr, direction):
    low, high = thr
    if direction == 'higher':
        if val >= high: return P['GREEN']
        if val >= low:  return P['TILE_AMBER']
        return P['TILE_RED']
    else:
        if val <= low:  return P['GREEN']
        if val <= high: return P['TILE_AMBER']
        return P['TILE_RED']

fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.8), facecolor='white')
for ax, (name, fmt, val, thr, direction) in zip(axes.flat, scorecard):
    c = color_for(val, thr, direction)
    ax.set_facecolor(c)
    ax.text(0.5, 0.6, fmt, ha='center', va='center', fontsize=24,
            fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.22, name, ha='center', va='center', fontsize=11.5,
            color='white', transform=ax.transAxes)
    arrow = '↑' if direction=='higher' else '↓'
    ax.text(0.5, 0.06, f'{arrow} ngưỡng: {thr[0]} / {thr[1]}',
            ha='center', va='center', fontsize=8, color='white', alpha=0.9,
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
plt.suptitle('Health Scorecard — Baseline v15 (LB 0.2441)', fontsize=12.5,
             fontweight='bold', color=P['PRIMARY'], y=1.00)
plt.tight_layout()
save_fig('chart_health_scorecard.png', fig)

# ═════════════════════════════════════════════════════════════════════════════
# CHART 10 — Trade-off compact bar (Recall vs Freshness)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 3.5), facecolor='white')
scenarios = ['Baseline\n(v15)','Variant A\nFreshness +5%','Variant B\nSeller Cap ≤2']
recall    = [0.2441, 0.1064, 0.1077]
freshness = [1.1,    3.0,    0.9]
colors_bars=[P['TILE_BLUE'], P['ORANGE'], P['GREEN']]

x_pos = np.arange(3)
w = 0.36
ax2 = ax.twinx()
bars1 = ax.bar(x_pos - w/2, recall, w, color=colors_bars, edgecolor='white', linewidth=1.5, label='Recall@10')
bars2 = ax2.bar(x_pos + w/2, freshness, w, color=colors_bars, edgecolor='white', linewidth=1.5,
                hatch='//', alpha=0.55, label='Freshness@10 (%)')
for i, v in enumerate(recall):
    ax.text(i - w/2, v + 0.005, f'{v:.4f}', ha='center', fontsize=8.5, fontweight='bold')
for i, v in enumerate(freshness):
    ax2.text(i + w/2, v + 0.1, f'{v:.1f}%', ha='center', fontsize=8.5, fontweight='bold')

ax.set_xticks(x_pos); ax.set_xticklabels(scenarios, fontsize=9)
ax.set_ylabel('Recall@10', fontsize=9, color=P['TILE_BLUE'])
ax2.set_ylabel('Freshness@10 (%)', fontsize=9, color=P['ORANGE'])
ax.set_ylim(0, 0.30); ax2.set_ylim(0, 4.0)
ax.set_title('Trade-off: Accuracy vs Freshness — 3 kịch bản', pad=8)
ax.spines[['top']].set_visible(False); ax2.spines[['top']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.35); ax.set_axisbelow(True)
ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
plt.tight_layout()
save_fig('chart_tradeoff_compact.png', fig)

print('\n✓ All slide charts generated.')
