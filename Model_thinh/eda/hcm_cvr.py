import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

conn = duckdb.connect()
conn.execute("SET memory_limit='20GB'")
conn.execute("SET threads=4")
conn.execute("SET enable_progress_bar=false")

evt_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/fact_user_events/*.parquet')]
dim_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/dim_listing/*.parquet')]

CATEGORIES = {1010:"Phòng trọ/Cho thuê", 1020:"Căn hộ/Chung cư",
               1030:"Nhà ở", 1040:"Đất nền", 1050:"Dự án mới"}
CAT_COLORS = {1010:"#4C72B0", 1020:"#DD8452", 1030:"#55A868",
              1040:"#C44E52", 1050:"#8172B2"}
POS = ('view_phone','contact_chat','other_interaction','contact_zalo','contact_sms')
pos_str = ", ".join(f"'{e}'" for e in POS)

# ── Tìm tên thành phố chính xác trong data ────────────────────────────────────
print("Kiểm tra tên city_name cho HCM trong fact_user_events …")
sample = conn.execute(f"""
    SELECT DISTINCT city_name
    FROM read_parquet({evt_files})
    WHERE LOWER(city_name) LIKE '%h%chi%minh%'
       OR LOWER(city_name) LIKE '%hcm%'
       OR LOWER(city_name) LIKE '%sài gòn%'
       OR city_name LIKE '%Minh%'
    LIMIT 10
""").df()
print(sample)

hcm_name = sample['city_name'].iloc[0] if len(sample) > 0 else 'Hồ Chí Minh'
print(f"\nDùng city_name = '{hcm_name}'\n")

# ── Query 1: CVR tổng theo category tại HCM ───────────────────────────────────
print("Query CVR HCM per category …")
cvr_cat = conn.execute(f"""
    SELECT
        category,
        COUNT(*)                                                          AS total_events,
        SUM(CASE WHEN event_type IN ({pos_str}) THEN 1 ELSE 0 END)        AS pos_events,
        SUM(CASE WHEN event_type = 'pageview'          THEN 1 ELSE 0 END) AS pageviews,
        SUM(CASE WHEN event_type = 'view_phone'        THEN 1 ELSE 0 END) AS view_phone,
        SUM(CASE WHEN event_type = 'contact_chat'      THEN 1 ELSE 0 END) AS contact_chat,
        SUM(CASE WHEN event_type = 'other_interaction' THEN 1 ELSE 0 END) AS other_interaction,
        SUM(CASE WHEN event_type = 'contact_zalo'      THEN 1 ELSE 0 END) AS contact_zalo,
        SUM(CASE WHEN event_type = 'contact_sms'       THEN 1 ELSE 0 END) AS contact_sms,
        APPROX_COUNT_DISTINCT(item_id)                                    AS unique_items,
        APPROX_COUNT_DISTINCT(
            CASE WHEN is_login='login' THEN user_id END
        )                                                                 AS unique_users
    FROM read_parquet({evt_files})
    WHERE city_name = '{hcm_name}'
      AND category IN (1010,1020,1030,1040,1050)
    GROUP BY category
    ORDER BY category
""").df()

cvr_cat['CVR'] = (cvr_cat['pos_events'] / cvr_cat['total_events'] * 100).round(2)
cvr_cat['cat_name'] = cvr_cat['category'].map(CATEGORIES)

# ── Query 2: CVR theo category × ad_type tại HCM ─────────────────────────────
print("Query CVR HCM per category × ad_type …")
item_evt = conn.execute(f"""
    SELECT item_id, category,
           COUNT(*)                                                    AS total_events,
           SUM(CASE WHEN event_type IN ({pos_str}) THEN 1 ELSE 0 END)  AS pos_events
    FROM read_parquet({evt_files})
    WHERE city_name = '{hcm_name}'
      AND category IN (1010,1020,1030,1040,1050)
    GROUP BY item_id, category
""").df()

dim = conn.execute(f"""
    SELECT item_id, category, ad_type, area_sqm, district_name
    FROM read_parquet({dim_files})
    WHERE category IN (1010,1020,1030,1040,1050)
      AND ad_type IS NOT NULL
""").df()

conn.close()

merged = item_evt.merge(dim, on=['item_id','category'], how='inner')

cvr_adtype = (
    merged.groupby(['category','ad_type'])
    .agg(pos=('pos_events','sum'), total=('total_events','sum'),
         n_items=('item_id','nunique'))
    .reset_index()
)
cvr_adtype['CVR'] = (cvr_adtype['pos'] / cvr_adtype['total'] * 100).round(2)
cvr_adtype['cat_name'] = cvr_adtype['category'].map(CATEGORIES)

# ── Query 3: CVR theo district tại HCM (top 20 per category) ─────────────────
cvr_dist = (
    merged.groupby(['category','district_name'])
    .agg(pos=('pos_events','sum'), total=('total_events','sum'),
         n_items=('item_id','nunique'))
    .reset_index()
)
cvr_dist['CVR'] = cvr_dist['pos'] / cvr_dist['total'] * 100
cvr_dist = cvr_dist[cvr_dist['n_items'] >= 10]

# ── Print kết quả ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"CVR tại {hcm_name} — theo Category")
print(f"{'='*65}")
display = cvr_cat[['cat_name','total_events','pos_events','CVR',
                   'unique_items','unique_users',
                   'view_phone','contact_chat','other_interaction',
                   'contact_zalo','contact_sms']].copy()
display['view_phone%']   = (display['view_phone']/display['total_events']*100).round(2)
display['chat%']         = (display['contact_chat']/display['total_events']*100).round(2)
display['other%']        = (display['other_interaction']/display['total_events']*100).round(2)
for _, r in display.iterrows():
    print(f"\n  [{r['cat_name']}]")
    print(f"    Total events : {int(r['total_events']):>12,}")
    print(f"    Positive evts: {int(r['pos_events']):>12,}  →  CVR = {r['CVR']:.2f}%")
    print(f"    Unique items : {int(r['unique_items']):>12,}")
    print(f"    Unique users : {int(r['unique_users']):>12,}")
    print(f"    view_phone   : {int(r['view_phone']):>12,}  ({r['view_phone%']:.2f}% of all events)")
    print(f"    contact_chat : {int(r['contact_chat']):>12,}  ({r['chat%']:.2f}%)")
    print(f"    other_inter  : {int(r['other_interaction']):>12,}  ({r['other%']:.2f}%)")

print(f"\n{'='*65}")
print("CVR theo Category × ad_type")
print(f"{'='*65}")
print(cvr_adtype[['cat_name','ad_type','CVR','n_items','total']].to_string(index=False))

print(f"\n{'='*65}")
print("Top 5 Quận có CVR cao nhất per Category")
print(f"{'='*65}")
for cat, name in CATEGORIES.items():
    sub = cvr_dist[cvr_dist['category']==cat].nlargest(5,'CVR')
    if sub.empty: continue
    print(f"\n  {name}:")
    for _, r in sub.iterrows():
        print(f"    {r['district_name']:<30} CVR={r['CVR']:.1f}%  n_items={int(r['n_items'])}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle(f'CVR tại {hcm_name} — 5 Category', fontsize=14, fontweight='bold')

gs = fig.add_gridspec(3, 5, hspace=0.55, wspace=0.35)

cats = list(CATEGORIES.keys())

# Row 0: CVR tổng per category (bar)
ax_main = fig.add_subplot(gs[0, :])
bars = ax_main.bar(
    [CATEGORIES[c] for c in cats],
    [cvr_cat.loc[cvr_cat['category']==c,'CVR'].values[0] if c in cvr_cat['category'].values else 0 for c in cats],
    color=[CAT_COLORS[c] for c in cats], alpha=0.88, width=0.5
)
# National average line (from previous EDA)
nat_avg = {'Phòng trọ/Cho thuê':64.62, 'Căn hộ/Chung cư':61.66,
           'Nhà ở':52.98, 'Đất nền':55.54, 'Dự án mới':54.69}
for i, (bar, cat) in enumerate(zip(bars, cats)):
    h = bar.get_height()
    ax_main.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    nat = nat_avg.get(CATEGORIES[cat].replace('\n',''), None)
    if nat:
        ax_main.plot([bar.get_x(), bar.get_x()+bar.get_width()], [nat, nat],
                     color='#333', linewidth=1.8, linestyle='--')
        delta = h - nat
        color_d = '#d62728' if delta > 0 else '#1a6bba'
        ax_main.text(bar.get_x()+bar.get_width()/2, nat - 1.8,
                     f'Quốc gia: {nat}%\n(Δ {delta:+.1f}pp)',
                     ha='center', va='top', fontsize=7, color=color_d)
ax_main.set_ylabel('CVR (%)', fontsize=10)
ax_main.set_title(f'CVR tổng per Category tại {hcm_name}  (-- = mức toàn quốc)', fontsize=10)
ax_main.yaxis.set_major_formatter(mtick.PercentFormatter())
ax_main.set_ylim(0, 80)
ax_main.spines[['top','right']].set_visible(False)

# Row 1: CVR breakdown by event_type (stacked 100% bar)
ax_event = fig.add_subplot(gs[1, :])
event_cols = ['view_phone','contact_chat','contact_zalo','contact_sms','other_interaction','pageviews']
event_colors = ['#d62728','#2ca02c','#17becf','#bcbd22','#ff7f0e','#aec7e8']
event_labels = ['view_phone','contact_chat','contact_zalo','contact_sms','other_interaction','pageview']

bottom = np.zeros(len(cats))
for col, color, label in zip(event_cols, event_colors, event_labels):
    vals = []
    for c in cats:
        row = cvr_cat[cvr_cat['category']==c]
        if row.empty:
            vals.append(0)
        else:
            vals.append(row[col].values[0] / row['total_events'].values[0] * 100)
    vals = np.array(vals, dtype=float)
    bars2 = ax_event.bar([CATEGORIES[c] for c in cats], vals, 0.5,
                          bottom=bottom, color=color, alpha=0.85, label=label)
    for i, v in enumerate(vals):
        if v > 3:
            ax_event.text(i, bottom[i] + v/2, f'{v:.1f}%',
                          ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
    bottom += vals
ax_event.set_ylabel('%')
ax_event.set_title('Phân bổ Event Type (% of total events)', fontsize=10)
ax_event.yaxis.set_major_formatter(mtick.PercentFormatter())
ax_event.legend(fontsize=8, bbox_to_anchor=(1.01,1), loc='upper left')
ax_event.spines[['top','right']].set_visible(False)

# Row 2: per-category top 5 district CVR
for col_i, (cat, cat_name) in enumerate(CATEGORIES.items()):
    ax = fig.add_subplot(gs[2, col_i])
    sub = cvr_dist[cvr_dist['category']==cat].nlargest(8,'CVR')
    if sub.empty:
        ax.set_visible(False)
        continue
    ax.barh(sub['district_name'], sub['CVR'], color=CAT_COLORS[cat], alpha=0.85)
    ax.set_title(f'{cat_name.replace(chr(10),"")}\nTop Quận/Huyện', fontsize=7.5,
                 fontweight='bold', color=CAT_COLORS[cat])
    ax.set_xlabel('CVR (%)', fontsize=7)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(axis='y', labelsize=6.5)
    ax.invert_yaxis()
    ax.spines[['top','right']].set_visible(False)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(r['CVR']+0.3, i, f"{r['CVR']:.1f}%", va='center', fontsize=6)

out = 'd:/Datathon_Data/eda/outputs/hcm_cvr.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
