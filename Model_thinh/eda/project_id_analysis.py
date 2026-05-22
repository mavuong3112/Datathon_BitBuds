import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

conn = duckdb.connect()
conn.execute("SET memory_limit='20GB'")
conn.execute("SET threads=4")
conn.execute("SET enable_progress_bar=false")

dim_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/dim_listing/*.parquet')]

CATEGORIES = {1010:"Phòng trọ\n/Cho thuê", 1020:"Căn hộ\n/Chung cư",
               1030:"Nhà ở", 1040:"Đất nền", 1050:"Dự án mới"}
CAT_NAMES  = {1010:"Phòng trọ/Cho thuê", 1020:"Căn hộ/Chung cư",
               1030:"Nhà ở", 1040:"Đất nền", 1050:"Dự án mới"}
CAT_COLORS = {1010:"#4C72B0", 1020:"#DD8452", 1030:"#55A868",
              1040:"#C44E52", 1050:"#8172B2"}

SEP = "\n" + "="*65

print("Loading dim_listing …")
df = conn.execute(f"""
    SELECT item_id, category, project_id, seller_type, ad_type,
           city_name, posted_date
    FROM read_parquet({dim_files})
    WHERE category IN (1010,1020,1030,1040,1050)
""").df()
conn.close()
print(f"  Loaded: {len(df):,} rows")

# ── 1. Null / Non-null per category ──────────────────────────────────
print(SEP + "\n1. NULL vs NON-NULL project_id per category")
rows = []
for cat in [1010,1020,1030,1040,1050]:
    sub = df[df['category'] == cat]
    has   = sub['project_id'].notna().sum()
    total = len(sub)
    rows.append({
        'category'  : cat,
        'cat_name'  : CAT_NAMES[cat],
        'total'     : total,
        'has_pid'   : int(has),
        'null_pid'  : int(total - has),
        'has_pct'   : round(has/total*100, 2),
        'null_pct'  : round((total-has)/total*100, 2),
    })
null_df = pd.DataFrame(rows)
print(null_df[['cat_name','total','has_pid','null_pid','has_pct','null_pct']].to_string(index=False))

# ── 2. Unique project_id per category ────────────────────────────────
print(SEP + "\n2. UNIQUE project_id count per category")
uniq = (
    df[df['project_id'].notna()]
    .groupby('category')['project_id']
    .nunique()
    .reset_index(name='unique_pids')
)
uniq['items_per_pid'] = [
    round(null_df.loc[null_df['category']==c,'has_pid'].values[0] /
          uniq.loc[uniq['category']==c,'unique_pids'].values[0], 1)
    if c in uniq['category'].values else 0
    for c in uniq['category']
]
uniq['cat_name'] = uniq['category'].map(CAT_NAMES)
print(uniq[['cat_name','unique_pids','items_per_pid']].to_string(index=False))

# ── 3. Cross-category sharing — do project_ids appear in multiple cats?
print(SEP + "\n3. CROSS-CATEGORY SHARING — project_id xuất hiện ở bao nhiêu category?")
pid_cats = (
    df[df['project_id'].notna()]
    .groupby('project_id')['category']
    .nunique()
    .reset_index(name='n_categories')
)
sharing = pid_cats['n_categories'].value_counts().sort_index()
print("  n_categories  count_of_project_ids")
for n, cnt in sharing.items():
    print(f"       {n}           {cnt:,}")

# Sample cross-category project_ids
cross = pid_cats[pid_cats['n_categories'] > 1]['project_id'].tolist()[:5]
if cross:
    print(f"\n  Sample cross-category project_ids:")
    for pid in cross:
        sub = df[df['project_id'] == pid]
        cats_in = sub['category'].unique()
        print(f"    pid={pid[:16]}…  categories={cats_in}  n_items={len(sub)}")

# ── 4. Distribution — items per project_id (top 20 largest projects) ─
print(SEP + "\n4. TOP 20 PROJECT_ID (most items attached)")
pid_size = (
    df[df['project_id'].notna()]
    .groupby(['project_id','category'])
    .agg(n_items=('item_id','nunique'),
         n_cities=('city_name','nunique'),
         n_sellers=('item_id','count'))
    .reset_index()
    .sort_values('n_items', ascending=False)
)
pid_size['cat_name'] = pid_size['category'].map(CAT_NAMES)
pid_size['pid_short'] = pid_size['project_id'].str[:12] + '…'
top20 = pid_size.head(20)
print(top20[['pid_short','cat_name','n_items','n_cities']].to_string(index=False))

# ── 5. Seller type of items WITH project_id ───────────────────────────
print(SEP + "\n5. SELLER TYPE — items WITH vs WITHOUT project_id")
for cat in [1010,1020,1030,1040,1050]:
    sub = df[df['category'] == cat]
    has_pid  = sub[sub['project_id'].notna()]['seller_type'].value_counts(normalize=True)*100
    no_pid   = sub[sub['project_id'].isna()]['seller_type'].value_counts(normalize=True)*100
    print(f"\n  [{CAT_NAMES[cat]}]")
    print(f"    HAS project_id  → agent={has_pid.get('agent',0):.1f}%  private={has_pid.get('private',0):.1f}%  (n={sub['project_id'].notna().sum():,})")
    print(f"    NO  project_id  → agent={no_pid.get('agent',0):.1f}%   private={no_pid.get('private',0):.1f}%  (n={sub['project_id'].isna().sum():,})")

# ── 6. Items-per-pid distribution (histogram buckets) ─────────────────
print(SEP + "\n6. PHÂN PHỐI — số items gắn với mỗi project_id")
size_dist = pid_size['n_items'].describe([.25,.5,.75,.9,.99])
print(size_dist.round(1).to_string())

# ══════════════════ CHARTS ═══════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Thành phần project_id trên toàn bộ dim_listing — 5 Category',
             fontsize=14, fontweight='bold')

gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.38)

cats = [1010,1020,1030,1040,1050]
cat_labels = [CATEGORIES[c] for c in cats]
null_pcts  = [null_df.loc[null_df['category']==c,'null_pct'].values[0] for c in cats]
has_pcts   = [null_df.loc[null_df['category']==c,'has_pct'].values[0] for c in cats]
has_abs    = [null_df.loc[null_df['category']==c,'has_pid'].values[0] for c in cats]
total_abs  = [null_df.loc[null_df['category']==c,'total'].values[0] for c in cats]

# ── Chart 1: Stacked 100% bar — Has/No project_id ────────────────────
ax1 = fig.add_subplot(gs[0, :2])
x  = np.arange(len(cats))
w  = 0.55
b1 = ax1.bar(x, null_pcts, w, label='NULL (không có project_id)',
             color='#d9d9d9', edgecolor='#999', linewidth=0.6)
b2 = ax1.bar(x, has_pcts,  w, bottom=null_pcts,
             label='CÓ project_id',
             color=[CAT_COLORS[c] for c in cats], alpha=0.9)

for i, (hp, np_, ha, tot) in enumerate(zip(has_pcts, null_pcts, has_abs, total_abs)):
    # label for has-pid bar
    if hp > 3:
        ax1.text(x[i], np_ + hp/2, f'{hp:.1f}%\n({ha/1000:.0f}K)',
                 ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    # label for null bar
    ax1.text(x[i], np_/2, f'{np_:.1f}%',
             ha='center', va='center', fontsize=8, color='#444')
    # total label on top
    ax1.text(x[i], 103, f'n={tot/1000:.0f}K',
             ha='center', va='bottom', fontsize=7, color='#333')

ax1.set_xticks(x)
ax1.set_xticklabels(cat_labels, fontsize=9)
ax1.set_ylim(0, 115)
ax1.set_ylabel('%')
ax1.set_title('Tỷ lệ CÓ / KHÔNG CÓ project_id theo Category (%)', fontsize=10)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.legend(fontsize=8, loc='lower right')
ax1.spines[['top','right']].set_visible(False)

# ── Chart 2: Unique project_ids per category (bar) ───────────────────
ax2 = fig.add_subplot(gs[0, 2])
has_cats = [c for c in cats if c in uniq['category'].values]
u_vals   = [uniq.loc[uniq['category']==c,'unique_pids'].values[0] for c in has_cats]
u_labels = [CATEGORIES[c] for c in has_cats]
bars = ax2.bar(range(len(has_cats)), u_vals,
               color=[CAT_COLORS[c] for c in has_cats], alpha=0.88)
for i, (bar, v) in enumerate(zip(bars, u_vals)):
    ipp = uniq.loc[uniq['category']==has_cats[i],'items_per_pid'].values[0]
    ax2.text(bar.get_x()+bar.get_width()/2, v+50,
             f'{v:,}\n({ipp:.0f} items/pid)', ha='center', va='bottom', fontsize=7)
ax2.set_xticks(range(len(has_cats)))
ax2.set_xticklabels(u_labels, fontsize=7, rotation=15, ha='right')
ax2.set_title('Unique project_id\nper Category', fontsize=9)
ax2.set_ylabel('Số unique project_id')
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f'{int(v/1000)}K' if v>=1000 else str(int(v))))
ax2.spines[['top','right']].set_visible(False)

# ── Chart 3: Cross-category sharing pie ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
share_labels = [f'Chỉ 1 category\n({sharing.get(1,0):,} pids)',
                f'2 categories\n({sharing.get(2,0):,} pids)',
                f'3+ categories\n({sum(v for k,v in sharing.items() if k>=3):,} pids)']
share_vals = [sharing.get(1,0),
              sharing.get(2,0),
              sum(v for k,v in sharing.items() if k>=3)]
share_vals = [v for v in share_vals if v > 0]
share_labels_f = [l for l,v in zip(share_labels, [sharing.get(1,0), sharing.get(2,0),
                  sum(v for k,v in sharing.items() if k>=3)]) if v > 0]
ax3.pie(share_vals, labels=share_labels_f, autopct='%1.1f%%',
        colors=['#4C72B0','#DD8452','#55A868'],
        textprops={'fontsize':8}, startangle=90)
ax3.set_title('project_id dùng chung\nnhiều category?', fontsize=9)

# ── Chart 4: Distribution items-per-pid (log-scale histogram) ────────
ax4 = fig.add_subplot(gs[1, 1])
sizes = pid_size['n_items'].values
bins = [1,2,3,5,10,20,50,100,500,10000]
ax4.hist(sizes, bins=bins, color='#4C72B0', alpha=0.8, edgecolor='white')
ax4.set_xscale('log')
ax4.set_xlabel('Số items / project_id (log scale)')
ax4.set_ylabel('Số lượng project_id')
ax4.set_title('Phân phối: bao nhiêu items\ngắn với 1 project_id?', fontsize=9)
ax4.spines[['top','right']].set_visible(False)
# Annotate percentiles
p50 = np.percentile(sizes, 50)
p90 = np.percentile(sizes, 90)
ax4.axvline(p50, color='orange', linestyle='--', linewidth=1.5, label=f'p50={p50:.0f}')
ax4.axvline(p90, color='red',    linestyle='--', linewidth=1.5, label=f'p90={p90:.0f}')
ax4.legend(fontsize=8)

# ── Chart 5: Top 15 largest project_ids (horizontal bar) ─────────────
ax5 = fig.add_subplot(gs[1, 2])
top15 = pid_size.head(15)
y = np.arange(len(top15))
bars5 = ax5.barh(y, top15['n_items'],
                 color=[CAT_COLORS[c] for c in top15['category']], alpha=0.85)
ax5.set_yticks(y)
ax5.set_yticklabels(
    [f"{r['pid_short']} [{r['cat_name'][:8]}]" for _,r in top15.iterrows()],
    fontsize=6.5
)
ax5.invert_yaxis()
ax5.set_xlabel('Số items')
ax5.set_title('Top 15 project_id\n(nhiều items nhất)', fontsize=9)
ax5.spines[['top','right']].set_visible(False)
legend_patches = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_NAMES[c]) for c in cats]
ax5.legend(handles=legend_patches, fontsize=6, loc='lower right')

# ── Chart 6: Seller type breakdown (has vs no pid, per category) ──────
ax6 = fig.add_subplot(gs[2, :])
cat_labels_full = [CAT_NAMES[c] for c in cats]
n_groups = len(cats)
x6 = np.arange(n_groups)
w6 = 0.2

has_agent   = []
has_private = []
no_agent    = []
no_private  = []

for c in cats:
    sub = df[df['category'] == c]
    has = sub[sub['project_id'].notna()]['seller_type'].value_counts(normalize=True)*100
    no  = sub[sub['project_id'].isna()]['seller_type'].value_counts(normalize=True)*100
    has_agent.append(has.get('agent',0))
    has_private.append(has.get('private',0))
    no_agent.append(no.get('agent',0))
    no_private.append(no.get('private',0))

ax6.bar(x6 - 1.5*w6, has_agent,   w6, label='CÓ pid — Agent',   color='#2166ac', alpha=0.9)
ax6.bar(x6 - 0.5*w6, has_private, w6, label='CÓ pid — Private', color='#92c5de', alpha=0.9)
ax6.bar(x6 + 0.5*w6, no_agent,    w6, label='NO pid — Agent',   color='#d6604d', alpha=0.9)
ax6.bar(x6 + 1.5*w6, no_private,  w6, label='NO pid — Private', color='#fddbc7', alpha=0.9, edgecolor='#d6604d', linewidth=0.5)

ax6.set_xticks(x6)
ax6.set_xticklabels(cat_labels_full, fontsize=9)
ax6.set_ylabel('% Seller Type')
ax6.set_title('Seller Type: CÓ project_id vs KHÔNG CÓ project_id — per Category', fontsize=10)
ax6.yaxis.set_major_formatter(mtick.PercentFormatter())
ax6.legend(fontsize=8, ncol=4, loc='upper right')
ax6.spines[['top','right']].set_visible(False)
ax6.set_ylim(0, 115)

for i in range(n_groups):
    for offset, val, color in [(-1.5*w6, has_agent[i], 'white'),
                                (-0.5*w6, has_private[i], '#333'),
                                ( 0.5*w6, no_agent[i], 'white'),
                                ( 1.5*w6, no_private[i], '#333')]:
        if val > 8:
            ax6.text(x6[i]+offset, val/2, f'{val:.0f}%',
                     ha='center', va='center', fontsize=7, color=color, fontweight='bold')

fig.tight_layout()
out = 'd:/Datathon_Data/eda/outputs/project_id_composition.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
print("Done.")
