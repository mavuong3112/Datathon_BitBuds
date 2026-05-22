"""
Generates 2 combo dashboard charts cho 5-slide deck:
- dashboard_performance.png  (4-panel: LB · Personalization · Score · Features)
- dashboard_health.png       (4-panel: Scorecard · Trade-off · Concentration · Age/Category)
"""
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path('/Volumes/mavuong3112/Datathon_Data')
CACHE_V16  = ROOT / 'model_v16_0.xxxx' / 'cache'
CACHE_OLD  = ROOT / 'model' / 'cache'
DIM_DIR    = ROOT / 'dim_listing'
OUT_DIR    = ROOT / 'marketplace_health_analysis' / 'slides' / 'charts'
BEST_SUB   = ROOT / 'model_v16_0.xxxx' / 'submission_stage15_0.2441.csv'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette (must match _build_slides.py) ─────────────────────────────────────
SLATE      = '#1B2538'
INDIGO     = '#3F51B5'
SKY        = '#00B0FF'
MINT       = '#00BFA5'
CORAL      = '#FF5252'
AMBER      = '#FFB74D'
MUTED      = '#607D8B'
DARK_TEXT  = '#102030'
GREEN      = '#2E7D32'
TILE_BLUE  = '#2196F3'

plt.rcParams['font.family']     = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titleweight']= 'bold'
plt.rcParams['axes.titlecolor'] = SLATE

def save(name, fig, dpi=180):
    p = OUT_DIR / name
    fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  ✓ {name}  ({p.stat().st_size // 1024} KB)')

# Load data once
print('Loading data…')
baseline_sub = pd.read_csv(BEST_SUB, usecols=['user_id','rank','item_id'])
booster = lgb.Booster(model_file=str(CACHE_V16 / 'lgbm_ranker.txt'))
rp = pq.read_table(CACHE_V16 / 'ranked_predictions.parquet',
                   columns=['user_id','item_id','lgbm_score','blend_score','rank']).to_pandas()
pop = pq.read_table(CACHE_OLD / 'popular_items.parquet',
                    columns=['item_id','trend_pos']).to_pandas()

# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD 1 — PERFORMANCE (2×2 grid)
# ═══════════════════════════════════════════════════════════════════════════
print('\n[1/2] Building dashboard_performance.png…')

fig = plt.figure(figsize=(13, 7.0), facecolor='white')
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.28,
               left=0.06, right=0.97, top=0.93, bottom=0.08)

# (1,1) LB Progression
ax1 = fig.add_subplot(gs[0, 0])
progression = [
    ('v1\nbaseline', 0.2184),
    ('v6',           0.2421),
    ('v8',           0.2430),
    ('v10',          0.2436),
    ('v11',          0.2438),
    ('v12',          0.2440),
    ('v15\nfinal',   0.2441),
]
labels  = [p[0] for p in progression]
values  = [p[1] for p in progression]
bars = ax1.bar(labels, values, color=TILE_BLUE, edgecolor='white', linewidth=1.2, zorder=3)
bars[-1].set_color(GREEN)
for b, v in zip(bars, values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
             f'{v:.4f}', ha='center', fontsize=7.5, fontweight='bold', color=DARK_TEXT)
ax1.axhline(0.0058, ls='--', color=MUTED, linewidth=0.9,
            label='Popularity (0.0058)')
ax1.set_ylim(0, 0.275)
ax1.set_ylabel('Recall@10', fontsize=9, color=DARK_TEXT)
ax1.set_title('① LB PROGRESSION  ·  v1 → v15 (+11.8%)', pad=10, fontsize=11, loc='left')
ax1.spines[['top','right']].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.30, zorder=0); ax1.set_axisbelow(True)
ax1.legend(loc='lower right', fontsize=7.5, frameon=False)
ax1.tick_params(axis='x', labelsize=7.5)

# (1,2) Personalization comparison + 42× annotation
ax2 = fig.add_subplot(gs[0, 1])
pop_top10 = (pop.sort_values('trend_pos', ascending=False)
                .drop_duplicates('item_id').head(10)['item_id'].tolist())
base_sets = baseline_sub.groupby('user_id')['item_id'].apply(set)
pop_set = set(pop_top10)
overlap = base_sets.apply(lambda s: len(s & pop_set))

# Bar comparison
ax2.bar(['Popularity\nbaseline','Our best\n(stage15)'],
        [0.0058, 0.2441],
        color=[MUTED, GREEN], edgecolor='white', linewidth=1.5, width=0.55)
for i, v in enumerate([0.0058, 0.2441]):
    ax2.text(i, v + 0.006, f'{v:.4f}', ha='center', fontsize=10.5, fontweight='bold',
             color=DARK_TEXT)
ax2.annotate('42× lift\n(+0.2383)', xy=(1, 0.244), xytext=(0.4, 0.16),
             fontsize=10, color=CORAL, ha='center', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.6))
ax2.set_ylim(0, 0.30)
ax2.set_ylabel('Recall@10', fontsize=9, color=DARK_TEXT)
ax2.set_title('② PERSONALIZATION  ·  vs popularity baseline', pad=10, fontsize=11, loc='left')
ax2.spines[['top','right']].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.30); ax2.set_axisbelow(True)
ax2.tick_params(axis='x', labelsize=9)

# (2,1) Score by rank bucket
ax3 = fig.add_subplot(gs[1, 0])
rp['rank_bucket'] = pd.cut(rp['rank'], bins=[0,3,10,30],
                            labels=['Top-3\n(rank 1-3)','Mid\n(rank 4-10)','Tail\n(rank 11-30)'])
data_box = [
    rp[rp['rank_bucket']=='Top-3\n(rank 1-3)']['lgbm_score'].sample(40_000, random_state=42),
    rp[rp['rank_bucket']=='Mid\n(rank 4-10)']['lgbm_score'].sample(40_000, random_state=42),
    rp[rp['rank_bucket']=='Tail\n(rank 11-30)']['lgbm_score'].sample(40_000, random_state=42),
]
bp = ax3.boxplot(data_box, labels=['Top-3\n(rank 1-3)','Mid\n(rank 4-10)','Tail\n(rank 11-30)'],
                 patch_artist=True, showfliers=False, widths=0.50)
for patch, c in zip(bp['boxes'], [GREEN, AMBER, MUTED]):
    patch.set_facecolor(c); patch.set_alpha(0.72)
for med in bp['medians']: med.set_color('#000')
ax3.set_title('③ SCORE BY RANK BUCKET  ·  Top-3 confident · Mid fragile',
              pad=10, fontsize=11, loc='left')
ax3.set_ylabel('LGBM score', fontsize=9, color=DARK_TEXT)
ax3.spines[['top','right']].set_visible(False)
ax3.grid(axis='y', linestyle='--', alpha=0.30); ax3.set_axisbelow(True)
ax3.tick_params(axis='x', labelsize=8.5)

# (2,2) Top-10 features
ax4 = fig.add_subplot(gs[1, 1])
fi = pd.Series(
    booster.feature_importance(importance_type='gain'),
    index=booster.feature_name()
).sort_values(ascending=True)
top10 = fi.tail(10)
ax4.barh(top10.index, top10.values, color=INDIGO, edgecolor='white', linewidth=0.8)
ax4.set_title('④ TOP-10 FEATURE IMPORTANCE  ·  LGBM gain', pad=10, fontsize=11, loc='left')
ax4.set_xlabel('Gain', fontsize=8.5, color=DARK_TEXT)
ax4.spines[['top','right']].set_visible(False)
ax4.grid(axis='x', linestyle='--', alpha=0.30); ax4.set_axisbelow(True)
ax4.tick_params(axis='y', labelsize=8)
ax4.tick_params(axis='x', labelsize=7.5)

fig.suptitle('PERFORMANCE DASHBOARD  ·  Recall@10 = 0.2441  ·  42× lift vs popularity',
             fontsize=14, fontweight='bold', color=SLATE, y=0.99, x=0.06, ha='left')

save('dashboard_performance.png', fig)

# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD 2 — MARKETPLACE HEALTH (2×2 grid)
# ═══════════════════════════════════════════════════════════════════════════
print('\n[2/2] Building dashboard_health.png…')

print('  Loading dim_listing…')
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

def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    return (2*np.sum(np.arange(1,n+1)*x) - (n+1)*x.sum()) / (n*x.sum())
def hhi(x):
    x = np.asarray(x, dtype=float); share = x / x.sum()
    return float((share**2).sum() * 10_000)
def normalized_entropy(x):
    x = np.asarray(x, dtype=float); share = x / x.sum(); share = share[share>0]
    if len(share)<=1: return 0.
    return float(-np.sum(share*np.log2(share)) / np.log2(len(share)))

seller_expo = sub_full.groupby('seller_id').size().sort_values()
item_expo   = sub_full['item_id'].value_counts()

freshness_pct = sub_full['is_fresh'].mean() * 100
seller_cov = sub_full['seller_id'].nunique() / dim['seller_id'].nunique() * 100
item_cov   = sub_full['item_id'].nunique() / len(dim) * 100
cat_entropy = normalized_entropy(sub_full['category'].value_counts().values)
g     = gini(seller_expo.values)
hhi_b = hhi(item_expo.values)

# Build dashboard
fig = plt.figure(figsize=(13, 7.0), facecolor='white')
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28,
               left=0.06, right=0.97, top=0.92, bottom=0.08)

# (1,1) 6-tile scorecard (mini)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_axis_off()
ax1.set_title('① HEALTH SCORECARD  ·  6 metrics', pad=10, fontsize=11, loc='left',
              fontweight='bold', color=SLATE)

scorecard = [
    ('Freshness@10',   f'{freshness_pct:.1f}%', freshness_pct, [1.0, 3.0],  'higher'),
    ('Seller Cov',     f'{seller_cov:.2f}%',    seller_cov,    [5.0, 15.0], 'higher'),
    ('Item Cov',       f'{item_cov:.2f}%',      item_cov,      [2.0, 5.0],  'higher'),
    ('Cat Entropy',    f'{cat_entropy:.3f}',    cat_entropy,   [0.7, 0.9],  'higher'),
    ('Seller Gini',    f'{g:.3f}',              g,             [0.85, 0.95],'lower'),
    ('Item HHI',       f'{hhi_b:.0f}',          hhi_b,         [200, 1000], 'lower'),
]
def color_for(val, thr, direction):
    low, high = thr
    if direction == 'higher':
        if val >= high: return GREEN
        if val >= low:  return AMBER
        return CORAL
    else:
        if val <= low:  return GREEN
        if val <= high: return AMBER
        return CORAL

# 3x2 mini tile grid inside ax1
for i, (lbl, fmt, val, thr, direction) in enumerate(scorecard):
    row = i // 3; col = i % 3
    c = color_for(val, thr, direction)
    # Tile rectangle (use ax coords 0-1)
    tw, th = 0.30, 0.42
    tx = 0.02 + col * (tw + 0.02)
    ty = 0.50 - row * (th + 0.05)
    rect = mpatches.FancyBboxPatch((tx, ty), tw, th, boxstyle='round,pad=0.005',
                                    facecolor=c, edgecolor='none',
                                    transform=ax1.transAxes)
    ax1.add_patch(rect)
    ax1.text(tx + tw/2, ty + th*0.65, fmt,
             ha='center', va='center', fontsize=15, fontweight='bold', color='white',
             transform=ax1.transAxes)
    ax1.text(tx + tw/2, ty + th*0.22, lbl,
             ha='center', va='center', fontsize=8, fontweight='bold', color='white',
             transform=ax1.transAxes)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

# (1,2) Trade-off bar
ax2 = fig.add_subplot(gs[0, 1])
scenarios = ['Baseline\n(v15)','Variant A\nFresh +5%','Variant B\nCap ≤2']
recall    = [0.2441, 0.1064, 0.1077]
freshness = [1.1,    3.0,    0.9]
colors_b  = [TILE_BLUE, AMBER, CORAL]

x_pos = np.arange(3)
w = 0.36
ax2r = ax2.twinx()
bars1 = ax2.bar(x_pos - w/2, recall, w, color=colors_b, edgecolor='white', linewidth=1.5)
bars2 = ax2r.bar(x_pos + w/2, freshness, w, color=colors_b, edgecolor='white',
                 linewidth=1.5, hatch='//', alpha=0.50)
for i, v in enumerate(recall):
    ax2.text(i - w/2, v + 0.006, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold',
             color=DARK_TEXT)
for i, v in enumerate(freshness):
    ax2r.text(i + w/2, v + 0.10, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold',
              color=DARK_TEXT)
ax2.set_xticks(x_pos); ax2.set_xticklabels(scenarios, fontsize=8.5)
ax2.set_ylabel('Recall@10', fontsize=8.5, color=TILE_BLUE)
ax2r.set_ylabel('Freshness (%)', fontsize=8.5, color=AMBER)
ax2.set_ylim(0, 0.30); ax2r.set_ylim(0, 4.0)
ax2.set_title('② TRADE-OFF  ·  Recall vs Freshness  ·  3 scenarios',
              pad=10, fontsize=11, loc='left')
ax2.spines[['top']].set_visible(False); ax2r.spines[['top']].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.30); ax2.set_axisbelow(True)
ax2.tick_params(axis='y', labelsize=7); ax2r.tick_params(axis='y', labelsize=7)

# (2,1) Lorenz curve
ax3 = fig.add_subplot(gs[1, 0])
cum = seller_expo.cumsum() / seller_expo.sum()
x_lr = np.linspace(0, 1, len(cum))
ax3.fill_between(x_lr, cum.values, alpha=0.20, color=TILE_BLUE)
ax3.plot(x_lr, cum.values, color=TILE_BLUE, linewidth=2.0)
ax3.plot([0,1], [0,1], '--', color=MUTED, linewidth=1.1)
ax3.set_title(f'③ LORENZ · SELLER EXPOSURE  ·  Gini = {g:.3f}',
              pad=10, fontsize=11, loc='left')
ax3.set_xlabel('% sellers (cumulative)', fontsize=8.5, color=DARK_TEXT)
ax3.set_ylabel('% exposure (cumulative)', fontsize=8.5, color=DARK_TEXT)
ax3.spines[['top','right']].set_visible(False)
ax3.set_xlim(0,1); ax3.set_ylim(0,1.02)
ax3.grid(linestyle='--', alpha=0.30); ax3.set_axisbelow(True)
ax3.tick_params(axis='both', labelsize=7.5)

# (2,2) Age bucket pool vs exposed
ax4 = fig.add_subplot(gs[1, 1])
def age_bucket(d):
    if d <= 7:    return '0-7d'
    elif d <= 30: return '8-30d'
    elif d <= 90: return '31-90d'
    elif d <= 365:return '91-365d'
    else:         return '>365d'
dim['age_bucket'] = dim['days_since_post'].apply(age_bucket)
order = ['0-7d','8-30d','31-90d','91-365d','>365d']
sub_age = baseline_sub.merge(dim[['item_id','age_bucket']], on='item_id', how='left')
exposed_age = sub_age['age_bucket'].value_counts(normalize=True).reindex(order).fillna(0) * 100
pool_age    = dim['age_bucket'].value_counts(normalize=True).reindex(order).fillna(0) * 100

x_pos = np.arange(len(order))
w = 0.36
ax4.bar(x_pos - w/2, pool_age.values,    w, color=MUTED, alpha=0.75, label='Pool',
        edgecolor='white')
ax4.bar(x_pos + w/2, exposed_age.values, w, color=TILE_BLUE, label='Exposed',
        edgecolor='white')
ymax = max(pool_age.max(), exposed_age.max())
for i, (p, e) in enumerate(zip(pool_age.values, exposed_age.values)):
    ax4.text(i - w/2, p + ymax*0.02, f'{p:.0f}', ha='center', fontsize=7, color='#555')
    ax4.text(i + w/2, e + ymax*0.02, f'{e:.0f}', ha='center', fontsize=7, fontweight='bold')
ax4.set_title('④ EXPOSURE BY AGE  ·  Pool vs Submission (%)',
              pad=10, fontsize=11, loc='left')
ax4.set_xticks(x_pos); ax4.set_xticklabels(order, fontsize=8.5)
ax4.set_ylim(0, ymax * 1.18)
ax4.legend(fontsize=8, loc='upper left', frameon=False)
ax4.spines[['top','right']].set_visible(False)
ax4.grid(axis='y', linestyle='--', alpha=0.30); ax4.set_axisbelow(True)
ax4.tick_params(axis='y', labelsize=7.5)

fig.suptitle('MARKETPLACE HEALTH DASHBOARD  ·  Coverage · Freshness · Fairness · Concentration',
             fontsize=14, fontweight='bold', color=SLATE, y=0.99, x=0.06, ha='left')

save('dashboard_health.png', fig)

print('\n✓ Both dashboards generated.')
