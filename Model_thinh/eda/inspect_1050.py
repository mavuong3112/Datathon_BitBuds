import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, pandas as pd
import numpy as np

conn = duckdb.connect()
conn.execute("SET memory_limit='20GB'")
conn.execute("SET threads=4")
conn.execute("SET enable_progress_bar=false")

dim_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/dim_listing/*.parquet')]
evt_files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/fact_user_events/*.parquet')]

SEP = "\n" + "="*65

# ── Load toàn bộ dim_listing cho cat 1050 và 1020 để so sánh ─────────
print("Loading dim_listing cat 1050 & 1020 …")
dim = conn.execute(f"""
    SELECT * FROM read_parquet({dim_files})
    WHERE category IN (1020, 1050)
""").df()

cat1050 = dim[dim['category'] == 1050].copy()
cat1020 = dim[dim['category'] == 1020].copy()
print(f"  1050 rows: {len(cat1050):,}   |   1020 rows: {len(cat1020):,}")

# ── 1. Null rate từng cột cho 1050 vs 1020 ────────────────────────────
print(SEP)
print("1. NULL RATE MỖI CỘT — 1050 (Dự án mới) vs 1020 (Căn hộ)")
cols = ['area_sqm','bedrooms','bathrooms','floors','width_m','direction',
        'legal_status','house_type','furnishing','project_id',
        'price_bucket','images_count']
rows = []
for col in cols:
    n50 = cat1050[col].isnull().mean()*100
    n20 = cat1020[col].isnull().mean()*100
    flag = '⚠' if n50 == 100 else ('★' if n50 < n20 else '')
    rows.append({'column': col,
                 'null_1050%': round(n50,1),
                 'null_1020%': round(n20,1),
                 'note': flag})
null_df = pd.DataFrame(rows)
print(null_df.to_string(index=False))

# ── 2. project_id — tại sao null 100% ở 1050? ────────────────────────
print(SEP)
print("2. project_id — phân tích chi tiết")
print(f"  1050 — project_id non-null: {cat1050['project_id'].notna().sum():,} / {len(cat1050):,}")
print(f"  1020 — project_id non-null: {cat1020['project_id'].notna().sum():,} / {len(cat1020):,}")
print(f"  1010 — check:")
cat1010 = conn.execute(f"""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN project_id IS NOT NULL THEN 1 ELSE 0 END) as non_null
    FROM read_parquet({dim_files}) WHERE category = 1010
""").df()
print(f"  1010 total={cat1010['total'].values[0]:,}  non_null={cat1010['non_null'].values[0]:,}")
# Sample project_id từ 1020
sample_pid = cat1020[cat1020['project_id'].notna()]['project_id'].head(5).tolist()
print(f"  Sample project_id (từ 1020): {sample_pid[:3]}")
# Xem 1050 có project_id nào trong title không
sample_title = cat1050['title'].dropna().head(10).tolist()
print(f"\n  Sample titles (1050):")
for t in sample_title[:8]:
    print(f"    · {t[:80]}")

# ── 3. bedrooms — tại sao null 100%? ─────────────────────────────────
print(SEP)
print("3. bedrooms — tại sao null 100% ở 1050?")
print(f"  Một dự án = nhiều loại căn hộ → không thể gán 1 con số cố định")
# Verify qua title
kw = cat1050['title'].dropna()
has_bedroom_kw = kw.str.contains(r'phòng ngủ|PN|bedroom|studio|1PN|2PN|3PN', case=False, na=False)
print(f"  Titles đề cập đến phòng ngủ: {has_bedroom_kw.sum():,} / {len(kw):,} ({has_bedroom_kw.mean()*100:.1f}%)")
samples_bed = kw[has_bedroom_kw].head(5).tolist()
for s in samples_bed:
    print(f"    · {s[:80]}")

# ── 4. legal_status — tại sao null 100%? ─────────────────────────────
print(SEP)
print("4. legal_status — tại sao null 100% ở 1050?")
print(f"  1050 non-null: {cat1050['legal_status'].notna().sum():,}")
print(f"  1020 legal_status distribution:")
print(cat1020['legal_status'].value_counts().head(8).to_string())
# Xem liệu có đề cập pháp lý trong title 1050 không
kw2 = cat1050['title'].dropna()
has_legal = kw2.str.contains(r'pháp lý|sổ hồng|sổ đỏ|quyền sử dụng', case=False, na=False)
print(f"\n  Titles 1050 đề cập pháp lý: {has_legal.sum():,} ({has_legal.mean()*100:.1f}%)")

# ── 5. area_sqm — phân phối 1050 vs 1020 ─────────────────────────────
print(SEP)
print("5. area_sqm — phân phối 1050 vs 1020")
for label, df_ in [("1050", cat1050), ("1020", cat1020)]:
    a = df_['area_sqm'].dropna()
    print(f"  [{label}] n={len(a):,}  null={df_['area_sqm'].isnull().mean()*100:.1f}%"
          f"  min={a.min():.0f}  p25={a.quantile(.25):.0f}"
          f"  median={a.median():.0f}  p75={a.quantile(.75):.0f}"
          f"  max={a.max():.0f}")

# ── 6. price_bucket — dạng giá khác nhau thế nào? ────────────────────
print(SEP)
print("6. price_bucket — top values cho 1050 vs 1020")
print("\n  [1050 Dự án mới] top 12:")
print(cat1050['price_bucket'].value_counts().head(12).to_string())
print("\n  [1020 Căn hộ] top 8:")
print(cat1020['price_bucket'].value_counts().head(8).to_string())

# ── 7. seller_type cho 1050 ───────────────────────────────────────────
print(SEP)
print("7. seller_type & ad_type cho 1050")
print(cat1050['seller_type'].value_counts().to_string())
print()
print(cat1050['ad_type'].value_counts().to_string())

# ── 8. images_count — dự án đầu tư nhiều vào ảnh không? ─────────────
print(SEP)
print("8. images_count — 1050 vs 1020")
for label, df_ in [("1050", cat1050), ("1020", cat1020)]:
    a = df_['images_count'].dropna()
    print(f"  [{label}]  mean={a.mean():.1f}  median={a.median():.0f}"
          f"  p90={a.quantile(.9):.0f}  max={a.max():.0f}")

# ── 9. Fact coverage — 1050 có bao nhiêu item xuất hiện trong events? ─
print(SEP)
print("9. Coverage trong fact_user_events (1050 items có event không?)")
evt_items_1050 = conn.execute(f"""
    SELECT APPROX_COUNT_DISTINCT(item_id) AS unique_items_in_events
    FROM read_parquet({evt_files})
    WHERE category = 1050
""").df()
print(f"  Unique item_id 1050 trong events : {evt_items_1050.values[0][0]:,}")
print(f"  Total item_id 1050 trong dim     : {len(cat1050):,}")
print(f"  Coverage: {evt_items_1050.values[0][0]/len(cat1050)*100:.1f}%")

# ── 10. Summary bảng tổng hợp ─────────────────────────────────────────
print(SEP)
print("10. TỔNG HỢP: Giải thích null theo nghiệp vụ")
explanations = [
    ("bedrooms",     "100%", "1 dự án = nhiều loại căn (studio, 1PN, 2PN...) → không thể gán 1 con số"),
    ("legal_status", "100%", "Dự án đang xây/mở bán → chưa có sổ riêng lẻ, pháp lý là của cả dự án"),
    ("project_id",   "100%", "⚠ ANOMALY: bắt buộc phải có nhưng null hoàn toàn → lỗi ETL hoặc field chưa map"),
    ("floors",       "~70%", "Dự án nhiều tòa, nhiều tầng → thông tin tổng hợp, khó điền 1 con số"),
    ("house_type",   "100%", "Không áp dụng cho căn hộ/dự án — field này dành cho nhà riêng lẻ"),
    ("furnishing",   "~55%", "Bàn giao thô vs bàn giao nội thất — không phải dự án nào cũng ghi rõ"),
    ("direction",    "~80%", "Dự án nhiều căn hướng khác nhau → không thể điền 1 hướng chung"),
    ("width_m",      "~95%", "Mặt tiền không có nghĩa với chung cư/dự án"),
]
print(f"  {'Cột':<15} {'Null%':<8} {'Giải thích'}")
print(f"  {'-'*60}")
for col, pct, reason in explanations:
    print(f"  {col:<15} {pct:<8} {reason}")

conn.close()
print("\nDone.")
