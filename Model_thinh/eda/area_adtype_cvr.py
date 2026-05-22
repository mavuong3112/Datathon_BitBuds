"""
Area bucket × ad_type CVR analysis
- CVR = positive_events / total_events per (category, area_bucket, ad_type)
- Top geographic zones by CVR per category
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
    1010: "Phòng trọ/Cho thuê",
    1020: "Căn hộ/Chung cư",
    1030: "Nhà ở",
    1040: "Đất nền",
    1050: "Dự án mới",
}
CAT_COLORS = {
    1010: "#4C72B0", 1020: "#DD8452",
    1030: "#55A868", 1040: "#C44E52", 1050: "#8172B2",
}
POS_EVENTS = ('view_phone','contact_chat','other_interaction','contact_zalo','contact_sms')
pos_str = ", ".join(f"'{e}'" for e in POS_EVENTS)

# Area bins — category-aware
#  1010 phòng trọ: nhỏ, 1040 đất nền: lớn
AREA_BINS   = [0, 20, 30, 45, 60, 80, 100, 150, 200, 300, 500, 10_000]
AREA_LABELS = ['<20','20-30','30-45','45-60','60-80','80-100','100-150','150-200','200-300','300-500','>500']

print("Step 1: Aggregate events per item …")
item_events = conn.execute(f"""
    SELECT
        item_id,
        category,
        COUNT(*)                                                          AS total_events,
        SUM(CASE WHEN event_type IN ({pos_str}) THEN 1 ELSE 0 END)        AS pos_events
    FROM read_parquet({evt_files})
    WHERE category IN (1010,1020,1030,1040,1050)
    GROUP BY item_id, category
""").df()

print(f"  item_events: {len(item_events):,} rows")

print("Step 2: Load dim_listing (area_sqm, ad_type, city, district) …")
dim = conn.execute(f"""
    SELECT item_id, category, ad_type, area_sqm,
           city_name, district_name
    FROM read_parquet({dim_files})
    WHERE category IN (1010,1020,1030,1040,1050)
      AND ad_type IS NOT NULL
      AND area_sqm IS NOT NULL
      AND area_sqm BETWEEN 1 AND 5000
""").df()

print(f"  dim: {len(dim):,} rows")

conn.close()

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = item_events.merge(dim, on=['item_id','category'], how='inner')
print(f"  merged: {len(merged):,} rows")

# Area bucket
merged['area_bucket'] = pd.cut(
    merged['area_sqm'], bins=AREA_BINS, labels=AREA_LABELS, right=False
)
merged['CVR'] = merged['pos_events'] / merged['total_events'] * 100

# ── Plot 1: CVR by area_bucket × ad_type per category ────────────────────────
print("\nPlotting CVR by area bucket × ad_type …")

fig, axes = plt.subplots(2, 5, figsize=(26, 11))
fig.suptitle('CVR (%) theo Phân cụm Diện tích × Loại giao dịch (Sell vs Let)\nper Category',
             fontsize=13, fontweight='bold', y=1.01)

SELL_COLOR = '#2171b5'
LET_COLOR  = '#e6550d'

for col_idx, (cat, cat_name) in enumerate(CATEGORIES.items()):
    sub = merged[merged['category'] == cat]

    # ── TOP chart: item count distribution (area histogram) ──
    ax_top = axes[0][col_idx]
    count_grp = sub.groupby(['area_bucket','ad_type'], observed=True)['item_id'].count().unstack(fill_value=0)
    total_per_bucket = count_grp.sum(axis=1)

    if 'sell' in count_grp.columns and 'let' in count_grp.columns:
        sell_pct = count_grp['sell'] / total_per_bucket * 100
        let_pct  = count_grp['let']  / total_per_bucket * 100
        x = np.arange(len(AREA_LABELS))
        ax_top.bar(x, sell_pct, 0.65, label='sell', color=SELL_COLOR, alpha=0.85)
        ax_top.bar(x, let_pct,  0.65, bottom=sell_pct, label='let', color=LET_COLOR, alpha=0.85)
        ax_top.set_ylim(0, 115)
        ax_top.yaxis.set_major_formatter(mtick.PercentFormatter())
    elif 'sell' in count_grp.columns:
        ax_top.bar(np.arange(len(AREA_LABELS)), [100]*len(AREA_LABELS), 0.65,
                   label='sell', color=SELL_COLOR, alpha=0.85)
    else:
        ax_top.bar(np.arange(len(AREA_LABELS)), [100]*len(AREA_LABELS), 0.65,
                   label='let', color=LET_COLOR, alpha=0.85)

    ax_top.set_title(f'{cat_name}\n({cat})', fontsize=9, fontweight='bold',
                     color=CAT_COLORS[cat])
    ax_top.set_ylabel('Tỷ lệ Sell/Let (%)', fontsize=7)
    ax_top.set_xticks(np.arange(len(AREA_LABELS)))
    ax_top.set_xticklabels(AREA_LABELS, rotation=45, ha='right', fontsize=7)
    ax_top.legend(fontsize=7, loc='upper right')
    ax_top.spines[['top','right']].set_visible(False)

    # Annotate n per bucket (total)
    for i, (bucket, n) in enumerate(zip(AREA_LABELS, total_per_bucket)):
        if n > 0:
            label = f'{n/1000:.0f}K' if n >= 1000 else str(n)
            ax_top.text(i, min(sell_pct.iloc[i] if 'sell' in count_grp.columns else 100, 105) + 3,
                        label, ha='center', va='bottom', fontsize=5.5, color='#333')

    # ── BOTTOM chart: CVR line by ad_type ──
    ax_bot = axes[1][col_idx]
    cvr_grp = (
        sub.groupby(['area_bucket','ad_type'], observed=True)
        .agg(pos=('pos_events','sum'), total=('total_events','sum'))
        .reset_index()
    )
    cvr_grp['cvr'] = cvr_grp['pos'] / cvr_grp['total'] * 100

    x = np.arange(len(AREA_LABELS))
    for ad_type, color, marker in [('sell', SELL_COLOR, 'o'), ('let', LET_COLOR, 's')]:
        sub_ad = cvr_grp[cvr_grp['ad_type'] == ad_type].set_index('area_bucket')
        # Reindex to all buckets
        sub_ad = sub_ad.reindex(AREA_LABELS)
        vals = sub_ad['cvr'].values.astype(float)
        ax_bot.plot(x, vals, marker=marker, ms=5, color=color, label=ad_type,
                    linewidth=1.8, alpha=0.9)
        # Annotate peak
        if not np.all(np.isnan(vals)):
            peak_i = int(np.nanargmax(vals))
            ax_bot.annotate(f'{vals[peak_i]:.0f}%',
                            xy=(peak_i, vals[peak_i]),
                            xytext=(0, 6), textcoords='offset points',
                            ha='center', fontsize=6, color=color, fontweight='bold')

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(AREA_LABELS, rotation=45, ha='right', fontsize=7)
    ax_bot.set_ylabel('CVR (%)', fontsize=7)
    ax_bot.set_xlabel('Diện tích (m²)', fontsize=7)
    ax_bot.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_bot.legend(fontsize=7)
    ax_bot.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bot.spines[['top','right']].set_visible(False)

    if col_idx == 0:
        axes[0][0].set_ylabel('Tỷ lệ Sell/Let (%)', fontsize=8)
        axes[1][0].set_ylabel('CVR (%)', fontsize=8)

fig.text(0.01, 0.75, 'ROW 1: Phân bố Sell/Let theo bucket m²', va='center', rotation=90, fontsize=9, color='#555')
fig.text(0.01, 0.28, 'ROW 2: CVR (%) theo bucket m²', va='center', rotation=90, fontsize=9, color='#555')
fig.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'area_adtype_cvr.png')
fig.savefig(out1, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Saved: {out1}")

# ── Plot 2: Geographic CVR — top 15 cities per category ──────────────────────
print("Plotting geographic CVR …")

geo = (
    merged.groupby(['category','city_name'])
    .agg(pos=('pos_events','sum'), total=('total_events','sum'), n_items=('item_id','nunique'))
    .reset_index()
)
geo['CVR'] = geo['pos'] / geo['total'] * 100
geo['lead_per_item'] = geo['pos'] / geo['n_items']

fig2, axes2 = plt.subplots(2, 5, figsize=(26, 11))
fig2.suptitle('CVR (%) và Lead/Item theo Thành phố — Top 15 mỗi Category',
              fontsize=13, fontweight='bold', y=1.01)

for col_idx, (cat, cat_name) in enumerate(CATEGORIES.items()):
    sub = geo[geo['category'] == cat].copy()
    if sub.empty:
        continue

    # Top 15 by CVR
    top_cvr = sub.nlargest(15, 'CVR')
    # Top 15 by lead_per_item
    top_lead = sub.nlargest(15, 'lead_per_item')

    color = CAT_COLORS[cat]

    # CVR chart
    ax = axes2[0][col_idx]
    ax.barh(top_cvr['city_name'], top_cvr['CVR'], color=color, alpha=0.85)
    for i, (_, row) in enumerate(top_cvr.iterrows()):
        ax.text(row['CVR'] + 0.3, i, f"{row['CVR']:.1f}%  (n={int(row['n_items']):,})",
                va='center', fontsize=6.5, color='#333')
    ax.set_title(f'{cat_name} ({cat})\nTop 15 by CVR', fontsize=8, fontweight='bold', color=color)
    ax.set_xlabel('CVR (%)', fontsize=7)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=7)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(0, top_cvr['CVR'].max() * 1.25)

    # Lead/item chart
    ax2 = axes2[1][col_idx]
    ax2.barh(top_lead['city_name'], top_lead['lead_per_item'], color=color, alpha=0.6)
    for i, (_, row) in enumerate(top_lead.iterrows()):
        ax2.text(row['lead_per_item'] + 0.3, i, f"{row['lead_per_item']:.0f}",
                 va='center', fontsize=6.5, color='#333')
    ax2.set_title(f'{cat_name} ({cat})\nTop 15 by Avg Lead/Item', fontsize=8, fontweight='bold', color=color)
    ax2.set_xlabel('Avg Lead per Item', fontsize=7)
    ax2.invert_yaxis()
    ax2.tick_params(axis='y', labelsize=7)
    ax2.spines[['top','right']].set_visible(False)
    ax2.set_xlim(0, top_lead['lead_per_item'].max() * 1.25)

fig2.text(0.01, 0.75, 'ROW 1: Top 15 City by CVR (%)', va='center', rotation=90, fontsize=9, color='#555')
fig2.text(0.01, 0.28, 'ROW 2: Top 15 City by Avg Lead/Item', va='center', rotation=90, fontsize=9, color='#555')
fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'geo_cvr.png')
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Saved: {out2}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n=== CVR by Category × ad_type ===")
summary = (
    merged.groupby(['category','ad_type'])
    .agg(pos=('pos_events','sum'), total=('total_events','sum'), n_items=('item_id','nunique'))
    .reset_index()
)
summary['CVR'] = (summary['pos'] / summary['total'] * 100).round(2)
summary['cat_name'] = summary['category'].map(CATEGORIES)
print(summary[['cat_name','ad_type','CVR','n_items','total','pos']].to_string(index=False))

print("\n=== Top 3 Cities by CVR per Category × ad_type ===")
geo2 = (
    merged.groupby(['category','ad_type','city_name'])
    .agg(pos=('pos_events','sum'), total=('total_events','sum'), n_items=('item_id','nunique'))
    .reset_index()
)
geo2['CVR'] = geo2['pos'] / geo2['total'] * 100
geo2 = geo2[geo2['n_items'] >= 20]  # min 20 items for stability

for cat, cat_name in CATEGORIES.items():
    for ad in ['sell','let']:
        sub = geo2[(geo2['category']==cat) & (geo2['ad_type']==ad)]
        if sub.empty: continue
        top3 = sub.nlargest(3,'CVR')[['city_name','CVR','n_items']]
        top3['CVR'] = top3['CVR'].round(1)
        cities = ' | '.join(f"{r['city_name']} {r['CVR']}% (n={int(r['n_items'])})" for _,r in top3.iterrows())
        print(f"  {cat_name} [{ad}]: {cities}")

# Save CSVs
summary.to_csv(os.path.join(OUTPUT_DIR, 'area_adtype_cvr_summary.csv'), index=False)
geo2.sort_values(['category','ad_type','CVR'], ascending=[True,True,False])\
    .to_csv(os.path.join(OUTPUT_DIR, 'geo_cvr.csv'), index=False)

print("\nDone. All outputs saved.")
