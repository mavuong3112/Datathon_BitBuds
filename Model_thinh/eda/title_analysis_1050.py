import sys, glob, re
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import Counter

conn = duckdb.connect()
conn.execute("SET memory_limit='20GB'")
conn.execute("SET threads=4")
conn.execute("SET enable_progress_bar=false")

dim_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/dim_listing/*.parquet')]

print("Loading 1050 titles …")
df = conn.execute(f"""
    SELECT item_id, title, ad_type, area_sqm, price_bucket,
           seller_type, city_name, district_name, posted_date
    FROM read_parquet({dim_files})
    WHERE category = 1050
      AND title IS NOT NULL
""").df()
conn.close()

df['title_lower'] = df['title'].str.lower().str.strip()
print(f"  Total 1050 rows with title: {len(df):,}")

SEP = "\n" + "="*65

# ─────────────────────────────────────────────────────────────────────
# Keyword taxonomy — theo thứ tự ưu tiên (first-match wins)
# ─────────────────────────────────────────────────────────────────────
TAXONOMY = [
    # ── Dự án BĐS thực sự ────────────────────────────────────────────
    ("🏗 Dự án mở bán",       r"dự án|mở bán|chủ đầu tư|shophouse|officetel|condotel|sky villa|penthouse|liền kề|biệt thự dự án|căn hộ dự án"),
    # ── Studio / mini apartment ───────────────────────────────────────
    ("🛋 Studio/Mini",         r"studio|mini apartment|căn mini|studio full|studio bancol|studio thang máy"),
    # ── Căn hộ cho thuê (chung cư có thương hiệu) ────────────────────
    ("🏢 Căn hộ chung cư",    r"căn hộ|chung cư|cc |apartment|vinhomes|masteri|the one|city garden|estella|sunrise|gateway|riviera|botanica|opal|an gia|hà đô|eco|gold view|river gate|icon56|saigon|pearl|sky park|scenic valley|gem riverside"),
    # ── Phòng trọ rõ ràng ─────────────────────────────────────────────
    ("🚪 Phòng trọ/Phòng ở",  r"phòng trọ|phòng ở|phòng cho thuê|phòng tháng|phòng thoáng|phòng sạch|phòng gác|phòng máy lạnh|phòng có wc|nhà trọ|gác trọ|trọ cao cấp"),
    # ── Duplex / Loft ─────────────────────────────────────────────────
    ("🏠 Duplex/Loft/Thông tầng", r"duplex|loft|thông tầng|2 tầng|nhà 2 tầng"),
    # ── Nhà nguyên căn cho thuê ───────────────────────────────────────
    ("🏡 Nhà nguyên căn",     r"nhà nguyên căn|nguyên căn|nhà phố|nhà trệt|nhà cấp 4|nhà 1 trệt|1 trệt|nhà cho thuê|cho thuê nhà"),
    # ── Số phòng ngủ rõ ràng ──────────────────────────────────────────
    ("🛏 Căn [1-4] phòng ngủ",r"\b[1-4]\s*(?:pn|phòng ngủ|bedroom|phòng)\b"),
    # ── Mặt bằng kinh doanh ───────────────────────────────────────────
    ("🏪 Mặt bằng/Văn phòng", r"mặt bằng|văn phòng|kinh doanh|thương mại|officetel|showroom|kiot|ki.t"),
]

OTHER_LABEL = "❓ Khác / Không xác định"

def classify(title: str) -> str:
    t = title.lower()
    for label, pattern in TAXONOMY:
        if re.search(pattern, t):
            return label
    return OTHER_LABEL

df['segment'] = df['title_lower'].apply(classify)

# ── 1. Phân bổ segment ────────────────────────────────────────────────
print(SEP)
print("1. PHÂN BỔ SEGMENT TỪ TITLE (1050)")
seg_count = df['segment'].value_counts()
seg_pct   = (seg_count / len(df) * 100).round(2)
seg_df    = pd.DataFrame({'count': seg_count, 'pct': seg_pct})
print(seg_df.to_string())

# ── 2. Từ khóa phổ biến nhất (unigram + bigram) ───────────────────────
print(SEP)
print("2. TOP 40 TỪ KHÓA TRONG TITLE (1050)")
# tokenize đơn giản — tách bằng space, loại ký tự đặc biệt
all_words = []
for t in df['title_lower']:
    words = re.findall(r'[a-zàáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵýỷỹ]{2,}', t)
    all_words.extend(words)

stop = {'cho','thuê','và','của','có','với','trong','tại','ngay','căn','phòng',
        'được','nhà','full','nội','thất','sạch','thoáng','mát','đẹp','mới',
        'đường','quận','huyện','thành','phố','hồ','chí','minh','không',
        'gần','từ','giá','rẻ','tiện','lợi','view','ban','công','thang','máy'}
top_words = Counter(w for w in all_words if w not in stop and len(w) > 2)
top_40 = top_words.most_common(40)
for i, (w, c) in enumerate(top_40):
    bar = '█' * (c * 40 // top_40[0][1])
    print(f"  {w:<20} {c:>8,}  {bar}")

# ── 3. area_sqm × segment ─────────────────────────────────────────────
print(SEP)
print("3. AREA_SQM THEO SEGMENT")
area_stats = (
    df[df['area_sqm'].between(1,500)]
    .groupby('segment')['area_sqm']
    .agg(['median','mean','min','max','count'])
    .round(1)
    .sort_values('count', ascending=False)
)
print(area_stats.to_string())

# ── 4. price_bucket × segment (top 3 per segment) ────────────────────
print(SEP)
print("4. PRICE BUCKET PHỔ BIẾN NHẤT THEO SEGMENT")
for seg in seg_count.index:
    sub = df[df['segment'] == seg]['price_bucket'].value_counts().head(3)
    prices = ' | '.join(f"{k} ({v:,})" for k, v in sub.items())
    print(f"  {seg[:35]:<35}: {prices}")

# ── 5. Seller type × segment ──────────────────────────────────────────
print(SEP)
print("5. SELLER TYPE THEO SEGMENT (%)")
pivot = pd.crosstab(df['segment'], df['seller_type'], normalize='index') * 100
print(pivot.round(1).to_string())

# ── 6. Sample titles per segment ─────────────────────────────────────
print(SEP)
print("6. SAMPLE TITLES THEO SEGMENT (3 mẫu mỗi loại)")
for seg in seg_count.index[:7]:
    samples = df[df['segment'] == seg]['title'].dropna().head(3).tolist()
    print(f"\n  [{seg}]")
    for s in samples:
        print(f"    · {s[:90]}")

# ── PLOT ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Phân tích Title — Category 1050 "Dự án mới"', fontsize=13, fontweight='bold')

colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999']

# Chart 1: Pie phân bổ segment
labels_short = [s.split(' ',1)[1][:22] if ' ' in s else s[:22] for s in seg_count.index]
wedges, texts, autotexts = axes[0].pie(
    seg_count.values, labels=labels_short, autopct='%1.1f%%',
    colors=colors[:len(seg_count)], startangle=140,
    textprops={'fontsize': 7.5}, pctdistance=0.78
)
for at in autotexts: at.set_fontsize(7)
axes[0].set_title('Phân bổ theo Segment\n(từ phân tích Title)', fontsize=10)

# Chart 2: Top 20 từ khóa
words20, counts20 = zip(*top_40[:20])
y_pos = np.arange(len(words20))
axes[1].barh(y_pos, counts20, color='#4C72B0', alpha=0.85)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(words20, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_title('Top 20 Từ khóa trong Title (1050)', fontsize=10)
axes[1].set_xlabel('Tần suất xuất hiện')
axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f'{int(v/1000)}K'))
axes[1].spines[['top','right']].set_visible(False)

# Chart 3: Box plot area_sqm theo segment
df_box = df[df['area_sqm'].between(1,200)].copy()
segs_ordered = area_stats.index.tolist()
box_data = [df_box[df_box['segment']==s]['area_sqm'].dropna().values for s in segs_ordered]
bp = axes[2].boxplot(box_data, vert=True, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
short_labels = [s.split(' ',1)[0] + '\n' + s.split(' ',1)[1][:15] if ' ' in s else s[:15]
                for s in segs_ordered]
axes[2].set_xticks(range(1, len(segs_ordered)+1))
axes[2].set_xticklabels(short_labels, fontsize=6.5, rotation=20, ha='right')
axes[2].set_title('Phân bổ Diện tích (m²) theo Segment\n(clip 1–200m²)', fontsize=10)
axes[2].set_ylabel('area_sqm (m²)')
axes[2].spines[['top','right']].set_visible(False)

fig.tight_layout()
out = 'd:/Datathon_Data/eda/outputs/title_analysis_1050.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
print("Done.")
