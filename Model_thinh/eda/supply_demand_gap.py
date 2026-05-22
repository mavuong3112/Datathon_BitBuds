"""
Supply vs Demand gap analysis per area bucket × ad_type × category
- Supply  = số tin đăng (item count từ dim_listing)
- Demand  = số positive events (leads) từ fact_user_events
- Gap     = Demand% - Supply% → dương = cầu > cung, âm = cung > cầu
"""
import sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUTPUT_DIR = 'd:/Datathon_Data/eda/outputs'

conn = duckdb.connect()
conn.execute("SET memory_limit='20GB'")
conn.execute("SET threads=4")
conn.execute("SET enable_progress_bar=false")

evt_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/fact_user_events/*.parquet')]
dim_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/dim_listing/*.parquet')]

CATEGORIES = {
    1010: "Phòng trọ\n/Cho thuê",
    1020: "Căn hộ\n/Chung cư",
    1030: "Nhà ở",
    1040: "Đất nền",
    1050: "Dự án mới",
}
CAT_COLORS = {
    1010:"#4C72B0", 1020:"#DD8452",
    1030:"#55A868", 1040:"#C44E52", 1050:"#8172B2",
}
POS_EVENTS = ('view_phone','contact_chat','other_interaction','contact_zalo','contact_sms')
pos_str = ", ".join(f"'{e}'" for e in POS_EVENTS)

AREA_BINS   = [0, 20, 30, 45, 60, 80, 100, 150, 200, 300, 500, 10_000]
AREA_LABELS = ['<20','20-30','30-45','45-60','60-80','80-100','100-150','150-200','200-300','300-500','>500']

# ── SUPPLY: listing count từ dim_listing ──────────────────────────────────────
print("Loading supply (dim_listing) …")
dim = conn.execute(f"""
    SELECT item_id, category, ad_type, area_sqm
    FROM read_parquet({dim_files})
    WHERE category IN (1010,1020,1030,1040,1050)
      AND ad_type   IS NOT NULL
      AND area_sqm BETWEEN 1 AND 5000
""").df()
dim['area_bucket'] = pd.cut(dim['area_sqm'], bins=AREA_BINS, labels=AREA_LABELS, right=False)

supply = (
    dim.groupby(['category','ad_type','area_bucket'], observed=True)['item_id']
    .count().reset_index(name='supply_count')
)

# ── DEMAND: positive events từ fact_user_events, join area từ dim ─────────────
print("Loading demand (positive events per item) …")
item_leads = conn.execute(f"""
    SELECT item_id, category,
           SUM(CASE WHEN event_type IN ({pos_str}) THEN 1 ELSE 0 END) AS leads,
           COUNT(*) AS total_events
    FROM read_parquet({evt_files})
    WHERE category IN (1010,1020,1030,1040,1050)
    GROUP BY item_id, category
""").df()

conn.close()

# Join demand với dim để lấy ad_type, area_sqm
dim_key = dim[['item_id','category','ad_type','area_sqm','area_bucket']].drop_duplicates('item_id')
demand_dim = item_leads.merge(dim_key, on=['item_id','category'], how='inner')

demand = (
    demand_dim.groupby(['category','ad_type','area_bucket'], observed=True)
    .agg(demand_leads=('leads','sum'), demand_events=('total_events','sum'))
    .reset_index()
)

# ── Merge supply + demand ─────────────────────────────────────────────────────
merged = supply.merge(demand, on=['category','ad_type','area_bucket'], how='outer').fillna(0)

# Normalize to % within (category × ad_type)
def add_pct(df, col_raw, col_pct):
    df[col_pct] = df.groupby(['category','ad_type'])[col_raw]\
                    .transform(lambda x: x / x.sum() * 100)
    return df

merged = add_pct(merged, 'supply_count',   'supply_pct')
merged = add_pct(merged, 'demand_leads',   'demand_pct')
merged['gap'] = merged['demand_pct'] - merged['supply_pct']   # + = undersupply

# ── Figure: 5 columns (categories) × 2 rows (sell / let) ─────────────────────
ad_types = ['sell','let']
fig, axes = plt.subplots(
    len(ad_types), len(CATEGORIES),
    figsize=(28, 10), sharey=False
)
fig.suptitle(
    'Supply (Tin đăng) vs Demand (Leads) phân bổ theo Bucket Diện tích\n'
    'Gap = Demand% − Supply%  →  (+) Cầu > Cung  |  (−) Cung > Cầu',
    fontsize=12, fontweight='bold', y=1.02
)

x = np.arange(len(AREA_LABELS))
w = 0.35

for row_i, ad in enumerate(ad_types):
    for col_i, (cat, cat_name) in enumerate(CATEGORIES.items()):
        ax = axes[row_i][col_i]
        sub = merged[
            (merged['category'] == cat) & (merged['ad_type'] == ad)
        ].set_index('area_bucket').reindex(AREA_LABELS).fillna(0)

        s_pct = sub['supply_pct'].values.astype(float)
        d_pct = sub['demand_pct'].values.astype(float)
        gap   = sub['gap'].values.astype(float)

        bar_s = ax.bar(x - w/2, s_pct, w, label='Supply (tin đăng)',
                       color='#9ecae1', alpha=0.9, edgecolor='#3182bd', linewidth=0.5)
        bar_d = ax.bar(x + w/2, d_pct, w, label='Demand (leads)',
                       color='#fdae6b', alpha=0.9, edgecolor='#e6550d', linewidth=0.5)

        # Gap line on twin axis
        ax2 = ax.twinx()
        ax2.plot(x, gap, color='#d62728', linewidth=1.5,
                 marker='o', ms=3.5, linestyle='--', alpha=0.8, label='Gap')
        ax2.axhline(0, color='#d62728', linewidth=0.6, linestyle=':')
        ax2.set_ylabel('Gap (pp)', fontsize=6, color='#d62728')
        ax2.tick_params(axis='y', labelsize=5.5, colors='#d62728')
        yabs = max(abs(gap).max(), 1)
        ax2.set_ylim(-yabs*1.6, yabs*1.6)

        # Shade undersupply zones (gap > 2pp)
        for i, g in enumerate(gap):
            if g > 2:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
            elif g < -2:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='blue')

        ax.set_xticks(x)
        ax.set_xticklabels(AREA_LABELS, rotation=45, ha='right', fontsize=6.5)
        ax.set_ylabel('Tỷ lệ (%)' if col_i == 0 else '', fontsize=7)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.spines[['top']].set_visible(False)

        title_color = CAT_COLORS[cat]
        if row_i == 0:
            ax.set_title(f'{cat_name}\n[{ad.upper()}]', fontsize=8,
                         fontweight='bold', color=title_color)
        else:
            ax.set_title(f'[{ad.upper()}]', fontsize=8, color=title_color)

        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=6.5, loc='upper right')
            ax2.legend(fontsize=6.5, loc='upper left')

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'supply_demand_gap.png')
fig.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Summary: top undersupply/oversupply buckets ───────────────────────────────
print("\n=== Top Undersupply (Demand > Supply, gap > 3pp) ===")
under = merged[merged['gap'] > 3].sort_values('gap', ascending=False)
under['cat_name'] = under['category'].map({k: v.replace('\n','') for k,v in CATEGORIES.items()})
print(under[['cat_name','ad_type','area_bucket','supply_pct','demand_pct','gap']]
      .head(20).round(1).to_string(index=False))

print("\n=== Top Oversupply (Supply > Demand, gap < -3pp) ===")
over = merged[merged['gap'] < -3].sort_values('gap')
over['cat_name'] = over['category'].map({k: v.replace('\n','') for k,v in CATEGORIES.items()})
print(over[['cat_name','ad_type','area_bucket','supply_pct','demand_pct','gap']]
      .head(20).round(1).to_string(index=False))

merged.to_csv(os.path.join(OUTPUT_DIR, 'supply_demand_gap.csv'), index=False)
print("\nDone.")
