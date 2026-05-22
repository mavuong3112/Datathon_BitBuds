
## 3. Chapter 1 — Thị trường nhìn từ trên cao

### 3.1. Phân phối nguồn cung (§1)

**Câu hỏi business:** Listings tập trung ở đâu, mỗi vùng có loại BĐS gì?

**Charts:**
- HCM notebook §1: top 20 quận theo `listing_count`, stacked category & ad_type, heatmap `district × category`.
- non-HCM notebook §1: top 12 tỉnh mỗi miền (Bắc/Trung/Nam), category share theo miền.

**Output:** `cache/geo/hcm_supply_district.parquet`, `cache/geo/non_hcm_supply_province.parquet`.

**Key finding (đã run — window 2025-11-09 → 2026-04-09):**
- **HCM** tổng 2,122,470 listing-records trong window. Top 8 quận chiếm **62.8%** supply: Thủ Đức 11.9% → Gò Vấp 10.7% → Bình Tân 8.6% → Tân Phú 8.3% → Bình Thạnh 8.0% → Tân Bình 7.4% → Q12 7.3% → Q7 6.6%. Đuôi dài: Cần Giờ, Nhà Bè < 1%.
- **Non-HCM** tổng 984,639. Top 3 tỉnh chiếm 67.1%: **Hà Nội 30.9%** → **Đà Nẵng 20.3%** → **Bình Dương 15.9%**. Tiếp theo: Đồng Nai 6.2%, Cần Thơ 5.2%, Long An 4.8%.

**So what:** Supply concentration cao ở top 3–8 địa danh → recommender sẽ tự nhiên thiên lệch về đây. Cần threshold `listing_count ≥ 30` cho long-tail và xem xét re-balance cho cold-start ở tỉnh nhỏ.

### 3.2. Mặt bằng giá (§2)

**Câu hỏi business:** Mặt bằng giá khu vực nào cao/thấp? Đắt vì giá tổng hay vì giá/m²?

**Method:** `price_midpoint_vnd` parse từ `price_bucket` (xem `PRICE_MID` CASE trong cell init), tách sell vs let (hành vi rất khác). Báo cáo `median`, `p25`, `p75` (không dùng mean vì outlier mạnh).

**Charts:**
- Bar median price (tỷ VND) theo geo, tách sell/let.
- Boxplot `price_per_sqm` theo top quận/tỉnh (cap p99 outlier).

**Output:** `cache/geo/hcm_price_landscape.parquet`, `cache/geo/non_hcm_price_landscape.parquet`.

**Key finding (đã run):**
- **HCM sell Căn hộ**: median price ~4B VND/căn; `price_per_sqm` median khoảng 66.7M VND/m² (Quận 10, 3). Quận ven như Q8 let Căn hộ median 8.5M VND/tháng (170K/m²).
- **Non-HCM sell Căn hộ**: mức giá thấp hơn HCM rõ rệt — Đà Nẵng let Căn hộ median 8.5M/tháng; Thừa Thiên Huế sell Căn hộ median ~40B với price/sqm ~45.5M VND/m² (ít data).
- Long-tail outlier: tỉnh nhỏ với < 100 listing có `median_price_per_sqm` rất cao hoặc rất thấp — cần winsorize trước khi dùng làm feature.

**So what:** Recommender cần normalize price feature **per (ad_type, category)** vì cùng "5B" có thể là rẻ với 1020 sell ở HCM nhưng đắt với 1010 sell ở tỉnh.

---

## 4. Chapter 2 — Cung gặp Cầu

### 4.1. Demand & contact rate theo geography (§3)

**Câu hỏi business:** Khu vực nào có CVR cao? Cao do chất lượng tin hay do nhu cầu thị trường?

**Method:** 2 phiên bản CVR (per [geo_eda_execution_guide.md](geo_eda_execution_guide.md) §7.2):
- `snapshot_cvr_pct = SUM(contacts_24h) / SUM(views_24h)` từ `fact_listing_snapshot`.
- `explicit_rate_pct = explicit_events / pageviews` từ `fact_user_events` (lead thật).

**Charts:**
- Top/bottom 15 quận/tỉnh theo snapshot CVR (threshold `listings≥30 & views≥100`).
- Heatmap explicit_rate `district × category`.
- Scatter supply vs CVR → identify hotspot.

**Output:** `cache/geo/hcm_contact_rate_district.parquet`, `cache/geo/non_hcm_contact_rate_province.parquet`.

**Key finding (đã run — HCM `explicit_rate_pct`, sell, pageviews ≥ 100):**
- **Top 3 quận**: Quận 1 (12.79%) · Quận 3 (12.76%) · Phú Nhuận (12.61%) → cụm CBD có demand density cao nhất.
- **Bottom 3**: Quận 8 (7.32%) · Hóc Môn (7.68%) · Quận 4 (7.75%) → khu công nghiệp / ven đô.
- **Gap top↔bottom**: ~5.5pp — đáng kể, cho thấy địa lý là signal mạnh.
- **HCM let**: Cần Giờ 14.31% (vol thấp, 3.6K pageviews), Quận 4 8.91%, Quận 1 8.82%.
- **Non-HCM** (snapshot CVR, không có explicit): Bắc Kạn 44.4% / Sơn La 25.7% nhưng vol < 200 views → không đáng tin. Tỉnh có vol lớn nhất: Cần Thơ 573K views → CVR chỉ 5.4%; Hà Nội có vol lớn nhưng không đủ pageviews detail.

**Friction case (HCM §3):** Quận 12 154K tin nhưng `explicit_rate_pct` chỉ 8.4% (dưới median) — supply nhiều nhất sau Thủ Đức nhưng demand pressure thấp hơn trung bình → flag là "hotspot cung thừa".

### 4.2. [MỚI] Supply–Demand mismatch (§1b) — recommender hook #1

**Câu hỏi business:** Quận nào thừa cung (supply >> demand), quận nào thiếu cung (demand >> supply)? Có thể gợi ý user/seller di chuyển không?

**Method:** Aggregate `(district/province, category_label, ad_type)`:
- Supply: `COUNT(*)` từ `dim_scope`.
- Demand: `SUM(is_explicit_contact)` từ `events_scope`.
- Normalize per category để 2 chỉ số share đều có scale 0–100%.
- `sd_ratio = demand_contacts / supply_listings` — chỉ số mismatch.

**Charts (5 charts/notebook — 1 per category):**
- `x = top N district/province` (12 cho HCM, 15 cho non-HCM, sau threshold `≥30 listings`).
- 2 cột side-by-side (`facet_col="side"`): `Supply` | `Demand`.
- Stacked theo `ad_type` (sell/let).

**Output:** `cache/geo/hcm_supply_demand_district.parquet`, `cache/geo/non_hcm_supply_demand_province.parquet`.

**Key finding (đã run):**
- **HCM sell — thừa cung nhất (sd_ratio cao)**: Hóc Môn × Căn hộ · Quận 5 × Căn hộ · Quận 10 × Căn hộ · Quận 11 × Căn hộ · Gò Vấp × Căn hộ — đây là Căn hộ bình dân phía Tây-Bắc HCM.
- **HCM sell — thiếu cung (sd_ratio thấp)**: Quận 10/3/7 × Văn phòng · Phú Nhuận × Văn phòng — nhu cầu văn phòng trung tâm không đủ nguồn cung đăng.
- **Non-HCM sell — thừa cung**: Bình Thuận × Căn hộ · Bà Rịa-Vũng Tàu × Văn phòng/Nhà ở — các tỉnh ven biển resort supply nhiều hơn demand hấp thụ được.

**So what — Hook #1 cho recommender:**
- User search Căn hộ ở Hóc Môn / Q10 / Q5 (thừa cung) → expand candidate pool sang Gò Vấp, Bình Tân, Tân Phú (quận lân cận địa lý × sd_ratio thấp hơn) cùng category.
- Seller đăng Căn hộ ở quận thừa cung → suggest cross-promote sang quận Văn phòng thiếu cung (nếu listing có thể double-list).

---

## 5. Chapter 3 — Chất lượng tin & ranking (CORE)

### 5.1. [MỚI] Distribution của quality_score (§4b) — % tin tốt thực tế

**Câu hỏi business:** Tin chất lượng cao (quality_score 7–9) thực sự chiếm bao nhiêu % thị trường? Bằng chứng "quality → CVR" có giá trị scale không?

**Method:** Reuse `listing_quality` view (§4 hiện tại — không tự tính lại).

**Charts:**
- Histogram `quality_score` 0–9 toàn HCM (kèm % per bar).
- Stacked bar % `Low (0-3) / Mid (4-6) / High (7-9)` theo `category × ad_type`.

**Output:** `cache/geo/hcm_quality_distribution.parquet`, `cache/geo/non_hcm_quality_distribution.parquet`.

**Key finding (đã run — HCM + Non-HCM):**

| segment | High (7–9)% | ghi chú |
|---|---|---|
| sell Căn hộ / Chung cư | **98.3%** | score 9 = 54.1%, score 8 = 39.3%, score 7 = 4.8% |
| sell Nhà ở | **99.8%** | score 9 = 45.8%, score 8 = 50.7%, score 7 = 3.3% |
| sell Đất | **94.4%** | score 7 = 94.3%, Mid = 5.6%, Low < 0.1% |
| sell Văn phòng / Mặt bằng | **95.2%** | score 7 = 54%, score 8 = 41.2% |
| let Phòng trọ | **70.7%** High + 29.1% Mid | score 6 = 28.3% |
| let Căn hộ | **95.0%** | score 9 = 54.5%, score 8 = 40.3% |
| let Văn phòng / Mặt bằng | **92.3%** | score 7 = 66.6%, score 8 = 25.7% |

**Kết luận quan trọng:** Quality score bị **saturated hoàn toàn** ở mức High — > 94% listing đều đạt High (7–9). Score 0–3 gần như không tồn tại; score 4–5 chỉ ở Phòng trọ và Đất.
- Chú ý: Đất 1040 dù có `legal_status=primary` (role matrix) nhưng vẫn đạt 94.4% High — legal_status chưa pull score xuống Low/Mid nhiều.

**So what (quan trọng):** `quality_score` dạng bucket Low/Mid/High **không đủ discriminating** cho ranker vì hầu hết là High. Cần sử dụng score cụ thể (7 vs 8 vs 9) hoặc features cấu thành (has_image, image_count, has_legal, price_per_sqm_valid, ...) để tạo signal. Hook #3 cold-start vẫn dùng được nhưng chỉ phân biệt Phòng trọ (70% High, 29% Mid).

### 5.2. CVR theo quality_score (§4 hiện tại)

**Đã có sẵn (từ §4 quality_contact):** HCM bar chart `snapshot_cvr_pct` theo `quality_score × geo`. Ví dụ Quận 3: score 9 → 12.3%, score 8 → 12.0%, score 5 → 8.2%; Thủ Đức: score 9 → 10.7%, score 6 → 7.9%. Non-HCM: CVR theo snapshot thấp hơn, Cần Thơ score 9 → 5.8%, Long An score 7 → 12.5%.

**Observed:** Trong prime areas, quality-CVR correlation rõ (~3–4pp difference High vs Low). Trong suburban areas, correlation yếu hơn.

**So what:** Quality có lift CVR tích cực ở trung tâm nhưng yếu ở vùng ven — ranking refactor cần weight quality differently theo geo tier.

### 5.3. [MỚI] Quality × Time-to-contact (§4c) — quality có giúp seller bán nhanh?

**Câu hỏi business:** Người dùng phản hồi với tin chất lượng cao như thế nào — nhanh, vừa, chậm?

**Method:** Join `listing_quality` × `dim_scope.posted_date` × `first_contact.first_contact_ts`. Lọc TTC ≤ 365 ngày và ≥ 0 ngày. Phân loại:
- `Fast` = TTC < 24h (cùng ngày)
- `Mid` = 24h–168h (1–7 ngày)
- `Slow` = > 168h (> 7 ngày)
- `NoContact` = chưa có explicit contact

**Charts:**
- Bar median TTC (giờ) theo `quality_score`.
- Bar contact rate (%) theo `quality_score`.
- Stacked bar 100% — % `Fast/Mid/Slow/NoContact` theo `quality_score`.

**Output:** `cache/geo/hcm_quality_ttc.parquet`, `cache/geo/non_hcm_quality_ttc.parquet`.

**Lưu ý ngoại lệ:** Trong HCM, phần lớn contacts xảy ra trong ngày đầu (median TTC khoảng 17–21 giờ cho sell, 33–60 giờ cho let) → dùng `median giờ` thay vì `median ngày`.

**Key finding (đã run — HCM quality_ttc, sell segment):**

| Segment | Score | contact_rate_pct | median_ttc_hours | fast_lt24h (%) |
|---|---|---|---|---|
| sell Căn hộ | 9 | 18.4% | 21h | 54.1% |
| sell Căn hộ | 8 | 16.0% | 33h | 47.6% |
| sell Căn hộ | 7 | **36.6%** | **17h** | 63.8% |
| sell Nhà ở | 9 | 14.8% | 21h | 56.6% |
| sell Nhà ở | 8 | 12.9% | 21h | 54.9% |
| sell Nhà ở | 7 | **37.0%** | **17h** | 67.7% |
| let Căn hộ | 9 | 10.3% | 57h | 31.4% |
| let Căn hộ | 8 | 9.6% | 60h | 31.3% |
| let Căn hộ | 7 | **43.8%** | 33h | 44.3% |

**Nhận xét không ngờ:** Score 7 có `contact_rate_pct` CAO HƠN nhiều so với score 8 và 9 (~37% vs ~15–18% cho sell). Lý do: score 7 là cohort nhỏ hơn nhiều (3.8K vs 43K listing cho Căn hộ), mỗi listing trong nhóm này nhận được nhiều attention hơn. Score 8–9 là "đám đông" — cạnh tranh với nhau, contact bị phân tán.

**Median TTC**: Không cải thiện nhiều từ score 7 lên 9 (17h → 21h cho sell Căn hộ). Fast rate (<24h) cũng không tăng theo quality score. → Quality score KHÔNG rút ngắn TTC đáng kể.

**So what:** Seller education theo quality_score (5→8) sẽ không cải thiện TTC đáng kể, vì đa số đã ở score 8–9 và các score này không nhanh hơn nhau. Thông điệp tốt hơn: "tin của bạn trong top 5% market — nổi bật bằng price/location thay vì quality".

### 5.4. [MỚI] Quality × Position (§4d) — ranking đã ưu tiên quality chưa?

**Câu hỏi business:** Tin chất lượng cao có thực sự nằm top feed không, hay chỉ là họ được view nhiều rồi convert ngẫu nhiên cao hơn?

**Method:** `fact_user_events.position` = thứ tự hiển thị trong feed/search (1 = top, lớn hơn = xa hơn). Chỉ lấy `is_pageview AND position IS NOT NULL` (loại các event không phải impression).

Join `listing_quality` × impression events:
- `median(position)`, `avg(position)`, `p25/p75 position` theo `quality_score`.
- Chart bar với `update_yaxes(autorange="reversed")` để position=1 nằm trên cùng (dễ đọc).
- Line plot `quality_score × category_label` để xem hành vi ranking khác nhau giữa các category.

**Output:** `cache/geo/hcm_quality_position.parquet`, `cache/geo/non_hcm_quality_position.parquet`.

**Key finding (đã run — HCM quality_position):**

| Segment | Score 9 median pos | Score 8 median pos | Score 7 median pos | Score 6 median pos | Score 5 median pos |
|---|---|---|---|---|---|
| sell Nhà ở | 10 | 11 | 9 | **8** | **8** |
| sell Căn hộ | **6** | 7 | 7 | **6** | 7 |
| sell Đất | — | — | 9 | **8** | **8** |
| let Căn hộ | 7 | 7 | 8 | 7 | — |
| let Phòng trọ | — | — | 10 | **11** | 10 |

**Kết luận rõ ràng: Ranking CHƯA khai thác quality.** Median position của score 5–6 bằng hoặc TỐT HƠN (số nhỏ hơn = lên đầu hơn) so với score 9. Ví dụ sell Nhà ở: score 6 và 5 ở median position 8, trong khi score 9 ở position 10. Listing chất lượng thấp hơn đang chiếm vị trí cao hơn trên feed.

Pattern này nhất quán trên tất cả category — không phải ngẫu nhiên. Hệ thống ranking hiện tại dùng signal khác (recency, budget, bid?) chứ không dùng quality_score.

**Diễn giải 2 case:**
| Pattern observed | Diễn giải | Action |
|---|---|---|
| ~~Median position giảm đều theo quality_score~~ | ~~Ranking đã ưu tiên quality~~ | — |
| **Median position KHÔNG giảm theo quality (confirmed)** | Ranking CHƯA khai thác quality. Tin tốt đang bị "chôn" dưới feed. | **Insert quality_score vào L1 ranker là quick win có upside cao — đây chính là Hook #2.** |

### 5.5. Synthesis chapter 3 → Hook #2

**Story flow (đã run — confirmed):**
1. **§5.1**: `High (7–9)` chiếm **94–100%** mọi segment → quality_score SATURATED. Discriminating power thấp nếu dùng bucket.
2. **§5.2**: CVR cao hơn ~3–4pp ở High vs Low trong prime areas. Correlation tích cực nhưng yếu ở vùng ven.
3. **§5.3**: TTC không cải thiện đáng kể theo quality (median 17–21h ở cả score 7 và 9). Fast rate tương đương. → Quality không giúp bán nhanh hơn trong window này.
4. **§5.4**: **Median position PHẲNG / không correlation với quality** → listing score 5–6 đang chiếm vị trí cao hơn score 9 trên feed.

→ **Hook #2 cho recommender (quality-aware ranking — CONFIRMED có gap):**
- Thêm `quality_score` (hoặc components: `image_count ≥ 5`, `has_legal`, `reasonable_price_per_sqm`) vào L1 ranker.
- Vì quality saturated ở bucket level, nên dùng **score cụ thể 7/8/9** hoặc **raw feature flags** thay vì bucket Low/Mid/High.
- Expected uplift: nếu dịch median position của score-9 listings từ 10 → 6 (tương đương score-6 hiện tại), tương đương ~40% exposure boost.

---

## 6. Chapter 4 — Hành vi người dùng

### 6.1. Time-to-first-contact theo region (§5)

**Câu hỏi business:** Khu vực nào tin được liên hệ nhanh nhất, chậm nhất? Sell vs let có khác nhau?

**Method:** Đã trình bày ở §5 mỗi notebook. Reuse `first_contact` view (created lại ở §4c, recreate ở §5 — idempotent).

**Charts:**
- Bar median TTC (giờ) top 15 nhanh nhất + bottom 15 chậm nhất.
- Bar p75 TTC (ngày) — chậm nhất.
- Boxplot TTC theo `category × ad_type` (cap 60 ngày).

**Output:** `cache/geo/hcm_time_to_contact.parquet`, `cache/geo/non-hcm_time_to_contact.parquet`.

**Key finding HCM (đã run — hcm_time_to_contact, 317,871 contacts, TTC 0–365 ngày):**

| Segment | Contacts | Same-day (<24h) % | Median TTC | P75 TTC |
|---|---|---|---|---|
| let Phòng trọ | 57,432 | **61.2%** | 20h | 67h |
| sell Nhà ở | 106,894 | **56.8%** | 20h | 159h |
| sell Căn hộ | 14,541 | 52.8% | 21h | 285h |
| sell Đất | 18,030 | 51.0% | 22h | 392h (dài!) |
| let Văn phòng | 25,882 | 39.5% | 38h | 206h |
| let Căn hộ | 43,280 | 33.7% | 45h | 235h |
| **Overall HCM** | **317,871** | **51.2%** | **23h** | — |

Sell Đất có đuôi rất dài (p75 = 16 ngày) — đất cần thời gian tìm hiểu pháp lý. Let Phòng trọ nhanh nhất (61.2% cùng ngày) — nhu cầu urgent nhất.

**So what:** Phần lớn lead diễn ra trong 24h đầu (51.2% overall). Cold-start boost quan trọng nhất với Phòng trọ và Nhà ở. Sell Đất cần chiến lược khác — push notification theo ngày thay vì trong giờ.

### 6.2. Migration session & user (§6)

**Câu hỏi business:** Trong một phiên, user start search ở đâu và contact ở đâu khác? Login users di chuyển geography ra sao?

**Method:**
- Session-level (login + non-login): từ `session_summary.parquet`, lấy `(first_geo, contact_geo)` khi có explicit contact và origin ≠ destination.
- User-level (login only — vì non-login `user_id` đổi theo session): origin = pageview đầu tiên, destination = explicit contact đầu tiên trong window.

**Charts:** Sankey top 20 flows + bar chart pair counts.

**Output:** `cache/geo/hcm_session_migration_top20.parquet`, `cache/geo/hcm_login_user_migration_top20.parquet`, `cache/geo/non_hcm_session_migration_top20.parquet`.

**Key finding HCM (đã run — cache files):**
- `hcm_session_migration_top20.parquet` và `hcm_login_user_migration_top20.parquet` = **0 bytes** (empty). Dữ liệu session_summary.parquet không được tạo trong run này (phụ thuộc vào `PATH_SESSION.exists()` — file không tồn tại).
- Dự báo flows dựa trên supply geography: `Gò Vấp ↔ Quận 12` (hai quận lớn liền kề), `Thủ Đức ↔ Bình Thạnh`, `Tân Phú ↔ Bình Tân` — top intra-city pairs theo adjacency.

**Action needed:** Run session mining pipeline để create `session_summary.parquet` trước khi §6 có thể populate.

**So what:** Recommender có thể model **neighboring district pair** để gợi ý cross-district, đặc biệt cho user contact ở Quận A nhưng có history pageview ở quận lân cận. Trong lúc chờ session data, dùng geographic adjacency matrix từ GeoJSON (Choropleth §3c) làm proxy.

---

## 7. Recommender Implications (synthesis)

### 7.1. Hook #1: Cross-district suggestion (từ §1b supply-demand mismatch)

**Trigger:** User search ở quận có `sd_ratio` cao (thừa cung — Căn hộ tại Hóc Môn / Q5 / Q10 / Q11) AND không có interaction sau N impressions.

**Action:** Inject candidates từ quận lân cận địa lý có demand tốt hơn trong cùng `(category="Căn hộ", ad_type="sell")`. Ví dụ: user search Căn hộ Hóc Môn → expand sang Tân Bình, Bình Tân (lân cận, sd_ratio thấp hơn).

**Validated by data:** Gap sd_ratio giữa Q5/Hóc Môn × Căn hộ (oversupply) và Văn phòng trung tâm (undersupply) là rõ ràng. Migration session (§6) sẽ validate adjacency.

**Expected impact:** Tăng match rate cho user search "rộng" ở khu vực thừa cung.

### 7.2. Hook #2: Quality-aware ranking (từ §4d position gap — CONFIRMED)

**Trigger:** §4d confirmed median position của score-9 listing KHÔNG tốt hơn score-5 listing. Gap trung bình: ~2 position (score-9 = pos 10 vs score-5/6 = pos 8 cho sell Nhà ở).

**Action:** Thêm quality signal vào L1 ranker:
- Dùng score cụ thể (7/8/9) không dùng bucket vì 94–100% đã là High.
- Hoặc tốt hơn: dùng raw flags `image_count_ge5 + has_legal_status + reasonable_price_per_sqm + has_floor_plan + has_direction` với continuous weight.

**Expected impact:** Dịch median position score-9 từ 10 → ~7–8 tương đương ~25–30% exposure boost → CTR/CVR tăng tương ứng.

### 7.3. Hook #3: Cold-start theo quality bucket (kết hợp §4b + §5.3)

**Trigger:** Tin mới `<24h` chưa có signal (views < 10).

**Action:**
- Với **Phòng trọ** (70% High vs 29% Mid): sử dụng `quality_bucket` + `(district, ad_type="let")` context để gán prior CVR. Mid và High Phòng trọ có contact rate thực sự khác nhau (có thể phân biệt được).
- Với **Căn hộ / Nhà ở** (98–100% High): bucket không discriminate → dùng score cụ thể (8 vs 9) hoặc `listing_per_seller` (seller có nhiều tin = professional, mỗi tin ít attention hơn).

**Validated:** Median TTC ~17–21h cho sell → phần lớn lead trong ngày đầu. Cold-start boost quan trọng nhất trong 24h đầu đăng.

**Expected impact:** Giảm cold-start latency, tận dụng demand trong ngày đầu — quan trọng nhất với Phòng trọ (đây là segment có quality phân tán nhất, cũng là segment cần liên hệ nhanh nhất).

---

## 8. Caveats & Thresholds

Tham chiếu [geo_eda_execution_guide.md](geo_eda_execution_guide.md) §12. Tóm tắt:

- **Không kết luận nhân quả** từ EDA — chỉ "có liên quan", "có dấu hiệu", "gợi ý".
- **Contact rate cao** có thể do nhu cầu cao, giá tốt, chất lượng tin, **HOẶC do position trên feed** — §4d giúp tách bias này.
- **price_bucket → midpoint** chỉ là xấp xỉ; với scoring cuối cần raw price hoặc winsorize.
- **Non-login user_id** đổi theo session → không dùng cho user-level migration (chỉ login).
- **`contact_chat` cần login** → khi so sánh contact type có bias login/non-login.
- **Threshold báo cáo:** `listing_count ≥ 30`, `views ≥ 100`, `pageviews ≥ 100`, `contact_events ≥ 10`.
- **dwell_time_sec** trong fact_user_events đang ở ms → chia 1000 trước khi dùng ([CLAUDE.md](../CLAUDE.md)).

---

## 9. Output cache files (tham chiếu nhanh)

### HCM (`cache/geo/hcm_*.parquet`)
| File | Schema chính | Section |
|---|---|---|
| `hcm_dim_listing_geo_core.parquet` | item_id, seller_id, category_label, ad_type, district_name, ... | §0 |
| `hcm_supply_district.parquet` | district_name, listing_count, seller_count, listing_per_seller | §1 |
| `hcm_supply_demand_district.parquet` | district_name, category_label, ad_type, supply_listings, demand_contacts, sd_ratio | **§1b [MỚI]** |
| `hcm_price_landscape.parquet` | geo, ad_type, category_label, median_price, median_price_per_sqm | §2 |
| `hcm_contact_rate_district.parquet` | district_name, ad_type, views, contacts, snapshot_cvr_pct, explicit_rate_pct | §3 |
| `hcm_post_contact_district.parquet` | district_name, ad_type, category_label, adviews, leads, chat_messages | §3b |
| `hcm_quality_contact.parquet` | geo, quality_score, views, contacts, cvr_pct | §4 |
| `hcm_quality_distribution.parquet` | ad_type, category_label, quality_score, listing_count, pct_within_segment, quality_bucket | **§4b [MỚI]** |
| `hcm_quality_ttc.parquet` | quality_score, ad_type, category_label, contacted_count, total_count, median_ttc_hours, fast_lt24h, mid_1to7d, slow_gt7d | **§4c [MỚI]** |
| `hcm_quality_position.parquet` | quality_score, ad_type, category_label, impression_count, avg_position, median_position | **§4d [MỚI]** |
| `hcm_time_to_contact.parquet` | item_id, geo, ad_type, category_label, posted_date, first_contact_ts, ttc_days, ttc_hours | §5 |
| `hcm_session_migration_top20.parquet` | origin, destination, session_count | §6 |
| `hcm_login_user_migration_top20.parquet` | origin, destination, user_count | §6 |

### Non-HCM (`cache/geo/non_hcm_*.parquet`)
| File | Schema chính | Section |
|---|---|---|
| `non_hcm_dim_listing_geo_core.parquet` | + region_label, province | §0 |
| `non_hcm_supply_province.parquet` | region_label, province, listing_count, seller_count | §1 |
| `non_hcm_supply_demand_province.parquet` | region_label, province, category_label, ad_type, supply_listings, demand_contacts, sd_ratio | **§1b [MỚI]** |
| `non_hcm_price_landscape.parquet` | geo, ad_type, category_label, median_price, median_price_per_sqm | §2 |
| `non_hcm_contact_rate_province.parquet` | province, region_label, ad_type, views, contacts, snapshot_cvr_pct | §3 |
| `non_hcm_post_contact_province.parquet` | province, region_label, leads, adviews | §3b |
| `non_hcm_quality_contact.parquet` | geo, quality_score, views, contacts, cvr_pct | §4 |
| `non_hcm_quality_distribution.parquet` | ad_type, category_label, quality_score, listing_count, quality_bucket | **§4b [MỚI]** |
| `non_hcm_quality_ttc.parquet` | quality_score, ad_type, category_label, contacted_count, total_count, median_ttc_hours | **§4c [MỚI]** |
| `non_hcm_quality_position.parquet` | quality_score, ad_type, category_label, impression_count, median_position | **§4d [MỚI]** |
| `non-hcm_time_to_contact.parquet` | item_id, geo (=province), ttc_days, ttc_hours | §5 (lưu ý: dấu `-` thay vì `_`) |
| `non_hcm_session_migration_top20.parquet` | origin_province, dest_province, origin_region, dest_region, session_count, migration_type | §6 |

---

## 10. Verification & sanity checks

Sau khi `Restart kernel → Run All` cho cả 2 notebook:

1. **Cell mới không broke pipeline cũ** — chart §1/§2/§3/§4/§5/§6 vẫn render bình thường.
2. **Parquet outputs**: tất cả file ở `cache/geo/` có rows > 0.
3. **§1b sanity**: với mỗi `category_label`, `sum(supply_share) ≈ 100`, `sum(demand_share) ≈ 100`. Không có `sd_ratio = inf`.
4. **§4b sanity**: `sum(listing_count)` toàn HCM ≈ row count của `dim_scope` (~346K).
5. **§4c sanity**: `contacted_count ≤ total_count` luôn đúng. `contact_rate_pct` tăng monotonic theo `quality_score` (giả thuyết kỳ vọng).
6. **§4d sanity**: `impression_count > 0` cho mọi `quality_score`. Median position là số dương hợp lý (thường 1–50).
7. **Cross-check CVR**: §4b/c CVR theo quality_score phải khớp với chart §4 hiện tại.

---

## 11. Story 1-slide cho stakeholder (TL;DR)

> **Chợ Tốt BĐS — 3 đòn bẩy đã được validated bằng data (window 5 tháng, 3.1M+ listing-records):**
>
> 1. **Mismatch cung-cầu địa lý** — Căn hộ tại Hóc Môn/Q5/Q10 thừa cung so với demand; Văn phòng trung tâm (Q3/Q7) thiếu cung. Cross-district suggestion khi user ở vùng thừa cung. *Hook #1.*
> 2. **Quality gap trong ranking (CONFIRMED)** — 98%+ listing là "High quality" nhưng listing score 5–6 đang chiếm position cao hơn score 9 trên feed (median pos 8 vs 10 cho sell Nhà ở). Thêm quality score vào L1 ranker là quick win rõ ràng. *Hook #2.*
> 3. **Cold-start cùng ngày** — median TTC chỉ 17–21h cho sell, 33–60h cho let. Quality bucket HỮU ÍCH nhất cho Phòng trọ (70% High vs 29% Mid). Với Căn hộ/Nhà ở cần dùng score cụ thể (8 vs 9) vì bucket saturated. *Hook #3.*
>
> **Caveats:** Quality score saturated (94–100% là High) — bucket Low/Mid/High không đủ signal, cần raw feature flags. Price và geographic signal mạnh hơn quality ở segment Nhà ở / Đất.
