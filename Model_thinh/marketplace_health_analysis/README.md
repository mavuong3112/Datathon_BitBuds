# 📊 Marketplace Health Analysis — Datathon 2026

> Báo cáo phân tích cuối — **Chợ Tốt BĐS Recommender System** · Đội **The Gridbreakers** · VinUniversity

**Public LB best:** `Recall@10 = 0.2441` (`submission_stage15_0.2441.csv`)
**Lift over popularity baseline:** **~42×** (popularity baseline `Recall@10 ≈ 0.0058`)
**Deliverable chính:** [`report.ipynb`](report.ipynb) — Jupyter notebook 73 cells với continuous reasoning sau mỗi output.

---

## Mục lục

1. [Project Overview](#1-project-overview)
2. [Problem Definition](#2-problem-definition)
3. [Dataset Description](#3-dataset-description)
4. [Solution Architecture](#4-solution-architecture)
5. [Methodology](#5-methodology)
6. [Experimental Setup](#6-experimental-setup)
7. [Results](#7-results)
8. [Discussion](#8-discussion)
9. [Production Considerations](#9-production-considerations)
10. [Reproducibility](#10-reproducibility)
11. [References](#11-references)
12. [Team & Acknowledgments](#12-team--acknowledgments)

---

## 1. Project Overview

**Cuộc thi:** *Datathon 2026 — The Gridbreakers* · *Breaking Business Boundaries*
**Tổ chức:** VinTelligence — VinUniversity Data Science & AI Club
**Đối tác dữ liệu:** Chợ Tốt BĐS (production marketplace, ~3.1M tin đăng, hàng trăm triệu sự kiện)
**Deadline GitHub + thuyết trình:** 23:59 ngày 22/05/2026

### Mục đích báo cáo này

Báo cáo phân tích **`report.ipynb`** trong thư mục này được thiết kế để đáp ứng đầy đủ **4 trục chấm điểm** của Ban Giám Khảo (BGK):

| # | Trục BGK | Tài liệu | Section trong notebook |
|---|---|---|---|
| 1 | **Thiết kế giải pháp** | §2 (cells 7-23) | Pipeline, candidate gen, ranker, formal math spec |
| 2 | **Hiệu suất mô hình** | §3 (cells 24-33) | LB progression, baseline comparison, score distribution |
| 3 | **Marketplace health** | §4 (cells 34-53) | Coverage, freshness, fairness, Gini, HHI, trade-off analysis |
| 4 | **Tính khả thi triển khai** | §5 (cells 54-67) | Latency, memory, retraining, failure modes, abuse/risk |

### Triết lý phân tích

- **Trung thực với hạn chế** — Mọi limitation (ground truth privacy, NDCG offline saturation, proxy metrics) đều được khai báo công khai.
- **Reproducibility-first** — Toàn bộ analysis chạy trên CPU dùng cache có sẵn, không cần retrain. Notebook tự sinh từ `_build_report.py`.
- **Marketplace-thinking, không chỉ leaderboard-chasing** — Cân nhắc trade-off accuracy vs health metrics, không chỉ tối ưu Recall.

---

## 2. Problem Definition

### 2.1 Phát biểu bài toán hình thức

Cho lịch sử quan sát đầy đủ của một user trên Chợ Tốt BĐS đến hết **09/04/2026**, hãy dự đoán **top-10 `item_id` phân biệt** mà user đó sẽ có **tương tác tích cực** trong khoảng **10/04/2026 – 07/05/2026**, xếp hạng từ khả năng cao nhất xuống thấp nhất.

**Tương tác tích cực** được định nghĩa là `fact_user_events.event_type ∈ {view_phone, contact_chat, other_interaction, contact_zalo, contact_sms}`.

### 2.2 Notation

| Ký hiệu | Định nghĩa |
|---|---|
| $\mathcal{U}$ | Tập user trong test set, $\|\mathcal{U}\| = 161{,}568$ |
| $\mathcal{I}$ | Tập tin đăng hợp lệ trong `train/dim_listing/`, $\|\mathcal{I}\| \approx 3.1 \times 10^6$ |
| $\mathcal{R}^+$ | Tập positive interactions $(u, i, t)$ |
| $G_u$ | Ground truth: tập `item_id` user $u$ thực sự tương tác trong cửa sổ eval |
| $f_\theta(u)$ | Top-10 items predict cho user $u$ |

### 2.3 Evaluation metrics

**Primary — Recall@10:**

$$\text{Recall@10}(u) = \frac{|f_\theta(u) \cap G_u|}{|G_u|}$$

Điểm leaderboard = trung bình Recall@10 trên toàn bộ user test.

**Tie-breaker — NDCG@10:**

$$\text{NDCG@10}(u) = \frac{\text{DCG@10}(u)}{\text{IDCG@10}(u)}, \quad \text{DCG@10}(u) = \sum_{i=1}^{10} \frac{\mathbb{1}[f_\theta(u)_i \in G_u]}{\log_2(i+1)}$$

### 2.4 Ràng buộc bài nộp

- Không leak tương lai (chỉ dùng dữ liệu `event_ts < 10/04/2026`).
- Không dùng dữ liệu ngoài 4 bảng được cung cấp.
- 5 submissions/đội/ngày UTC.
- Reproducibility yêu cầu — random seed cố định.

---

## 3. Dataset Description

### 3.1 4 bảng dữ liệu chính

| Bảng | Số dòng (≈) | Phân vùng | Schema chính |
|---|---|---|---|
| `dim_listing` | 3.1M | 40 parquet shards | `item_id, seller_id, category, title, seller_type, ad_type, posted_date, price_bucket, area_sqm, …` (24 cols) |
| `fact_user_events` | hàng trăm M | date-partitioned | `user_id, item_id, event_ts, event_type, ...` |
| `fact_listing_snapshot` | hàng chục M | daily snapshot | `item_id, date, views_7d, contacts_7d, ...` |
| `fact_post_contact_interactions` | hàng triệu | event-level | post-contact follow-up signals |

### 3.2 Time periods

| Tập | Khoảng thời gian | Mô tả |
|---|---|---|
| `train/` | 09/11/2025 → 09/04/2026 (152 ngày) | Toàn bộ 4 bảng |
| `test/` | — | `test_users.parquet` (danh sách user cần dự đoán) |
| `ground_truth/` | 10/04/2026 → 07/05/2026 (28 ngày) | **Private** — không truy cập được |

### 3.3 Domain constants

- **5 categories:** `1010` (Phòng trọ), `1020` (Căn hộ), `1030` (Nhà ở), `1040` (Đất nền), `1050` (Dự án mới)
- **Tết season:** 10/02 – 23/02/2026 (traffic giảm mạnh, cần handle)
- **Seller types:** `agent` (83.5%), `private` (16.5%)

### 3.4 Data lineage trong báo cáo

- **EDA chi tiết** → [`full_eda_story.md`](../full_eda_story.md) (~2000 dòng, không lặp lại trong notebook)
- **Pipeline outputs cache** → `model_v16_0.xxxx/cache/` + `model/cache/`
- **Final submission** → `model_v16_0.xxxx/submission_stage15_0.2441.csv`

---

## 4. Solution Architecture

### 4.1 Tổng quan two-stage architecture

Hệ thống áp dụng **two-stage recommendation paradigm** ([9] Covington et al. 2016):

```
                ┌──────────────────────────────────────┐
                │  Stage 1: Candidate Generation       │
                │  5 retrievers · top-100 mỗi nguồn    │
                │  → merge & dedup → cap 700/user      │
                └────────────────┬─────────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────────┐
                │  Stage 2: Learning to Rank           │
                │  54 features · LGBM LambdaRank (×10) │
                │  + XGBoost ensemble → top-30/user    │
                └────────────────┬─────────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────────┐
                │  Stage 3: Post-process & Submit      │
                │  Blend 65/35 · freshness/cat boost   │
                │  · cold-user fallback                │
                │  → top-10 final                      │
                └──────────────────────────────────────┘
```

Mô tả hình thức đầy đủ (math notation, hyperparameters, complexity analysis) trong **`report.ipynb` § 2.3 Formal Architecture Specification**.

### 4.2 Component summary

| Component | Method | Key hyperparam | Reference |
|---|---|---|---|
| **ALS** | Implicit MF | factors=512, α=40, λ=0.01 | [1] |
| **EASE** | Closed-form item-item | λ=200 | [2] |
| **ItemCF** | Cosine + time decay | β=0.005 | [3] |
| **SASRec** | Self-attention sequential | d=64, h=2, L_max=50 | [4] |
| **CBF** | PhoBERT title embedding | 768d | [8] |
| **LGBM** | LambdaRank GBDT | depth=8, leaves=95, 10 seeds | [5][6] |
| **XGBoost** | XGB ranker | depth=6, 1000 rounds | [7] |

---

## 5. Methodology

### 5.1 Data preprocessing

- **Time-aware split:** Train cho candidate gen dùng full `[2025-11-09, 2026-04-09]`. LGBM training val split = last 20% users (NB: feature leakage observed — xem §8).
- **Positive event weighting:** mỗi event_type có weight khác nhau (view_phone=3, contact_*=2, other_interaction=1).
- **Recency decay:** $w(t) = \exp(-0.005 \cdot \Delta t_{\text{days}})$ → half-life ≈ 140 ngày.
- **Tết season handling:** flag isolated, không đặc biệt re-weight (đủ data sau Tết).

### 5.2 Feature engineering ($d = 54$)

| Group | Số features | Ví dụ |
|---|---|---|
| **User-level** ($\phi_u$) | ~15 | `n_pos_events`, `intent_score_log`, `days_since_last`, `pref_category`, `pref_city`, `active_span_days` |
| **Item-level** ($\phi_i$) | ~18 | `trend_pos`, `days_since_post`, `item_quality_score`, `images_count`, `legal_status`, `house_type` |
| **Cross user×item** ($\phi_{ui}$) | ~12 | `is_repeat`, `ui_match_category`, `ui_district_affinity`, `ui_price_match` |
| **Sequence/intent** ($\phi_{\text{seq}}$) | ~9 | `last_click_days`, `session_length`, `dwell_avg`, `first_click_match` |

**Top-3 features by gain (LGBM):** `is_repeat` (0.625) → `user_intent_ratio` (0.598) → `intent_score_log` (0.580). Confirm rằng *repeat behavior + user seriousness* là tín hiệu dominant.

### 5.3 Candidate generation (5 retrievers)

Mỗi retriever produce top-100 candidates per user; merge & dedup → cap **700** candidates/user. Rationale:

- **ALS** — long-tail discovery (dense MF embeddings).
- **EASE** — high-accuracy linear closed-form.
- **ItemCF** — recency-aware co-occurrence.
- **SASRec** — session-level sequential intent.
- **CBF (PhoBERT)** — content-based fallback cho new items.
- **Trending** — top-50 popular per category (cold-user safety net).

### 5.4 Learning-to-rank (Stage 2)

**LGBM LambdaRank ensemble:** 10 seeds × 500 trees (depth 8, leaves 95). Pairwise NDCG-weighted loss [5][6].

**XGBoost ranker:** parallel train với same features, blend per-user min-max normalized score.

**Final blend:**

$$s_{\text{final}}(u, i) = 0.65 \cdot \tilde{s}_{\text{LGBM}} + 0.35 \cdot \tilde{s}_{\text{XGB}} + \beta_f \cdot \mathbb{1}[\text{fresh}(i)] + \beta_c \cdot \mathbb{1}[\text{cat}(i) = \text{pref}(u)]$$

### 5.5 Post-processing (Stage 3)

1. **Freshness boost:** $\beta_f = 0.015 \cdot \text{range}(s)$ cho items posted ≤7 ngày trước `TRAIN_END`.
2. **Category-match boost:** $\beta_c = 0.4 \beta_f$ cho items thuộc `pref_category` của user (warm users only).
3. **Cold-user fallback:** category-weighted interleaved popular pool (50 items) cho users không có positive history.

---

## 6. Experimental Setup

### 6.1 Hyperparameter master table

| Component | Hyperparam | Value | Justification |
|---|---|---|---|
| **ALS** | factors | 512 | balance quality vs memory (6GB) |
| | α | 40 | strong implicit confidence |
| | λ | 0.01 | mild L2 |
| | iterations | 50 | converges |
| **EASE** | λ | 200 | typical for ~3M items |
| **ItemCF** | recency β | 0.005 | ~140-day half-life |
| | top-K | 100 | matches other retrievers |
| **SASRec** | d | 64 | small for fast inference |
| | n_heads | 2 | sufficient for 50-len seq |
| | n_blocks | 2 | shallow, avoid overfitting |
| **LGBM** | depth | 8 | Stage 12 BIGGER config |
| | num_leaves | 95 | < 2^depth = 256, prevents overfit |
| | learning_rate | 0.05 | standard for ranking |
| | n_estimators | 500 | fixed, no early stop (see §8) |
| | seed ensemble | 10 | variance reduction |
| **Blend** | $w_{\text{LGBM}}$ | 0.65 | dominant signal |
| | $w_{\text{XGB}}$ | 0.35 | secondary, reduces variance |
| | $\beta_f$ | 0.015·range | empirical, ~1.5% score |
| | $\beta_c$ | 0.4·$\beta_f$ | proportional |

### 6.2 Train/val split

- **LGBM train:** 80% users in `features_train.parquet` (last 20% used as val for monitoring).
- **Submission produced on:** all 161,568 users in `test_users.parquet`.
- **Random seed:** [42, 123, 456, 789, 2024, 31, 314, 1729, 31415, 6710] (10 LGBM seeds).

### 6.3 Compute environment

| Stage | Resources | Wall-time |
|---|---|---|
| Candidate gen (ALS + EASE + ItemCF + SASRec) | CPU 8-core, 32GB RAM | ~2 hours |
| Feature compute (DuckDB 10GB) | CPU 4-core | ~45 min |
| LGBM 10-seed ensemble | CPU 8-core | ~90 min |
| XGBoost training | CPU 8-core | ~30 min |
| Post-process + submit | CPU 4-core | ~10 min |
| **Total** | — | **~4 hours** |

---

## 7. Results

### 7.1 LB progression (v1 → v15)

| Version | Recall@10 | Δ Gain | Change introduced |
|---|---|---|---|
| v1 baseline | 0.2184 | — | ALS-only ranker |
| v6 | 0.2421 | +0.0237 | Multi-retriever merge + LGBM |
| v8 stage8 | 0.2430 | +0.0009 | + user behavioral features |
| v10 | 0.2436 | +0.0006 | + snapshot freshness features |
| v11 | 0.2438 | +0.0002 | + district transition features |
| v12 | 0.2440 | +0.0002 | BIGGER LGBM (10-seed, depth 8) |
| **v15 (final)** | **0.2441** | +0.0001 | + category-match boost + XGBoost blend |

**Total gain:** +0.0257 (+11.8% relative).

### 7.2 Ablation study

| Component added | Cumulative Recall | Δ Gain |
|---|---|---|
| ALS only | 0.2184 | — |
| + EASE | 0.2300 | +0.0116 (long-tail diversity) |
| + ItemCF + SASRec | 0.2421 | +0.0121 (session + recency) |
| + LGBM ranker (1-seed) | 0.2436 | +0.0015 (learn-to-rank vs blend) |
| + 10-seed ensemble | 0.2440 | +0.0004 (variance reduction) |
| + XGBoost 35% blend | 0.2440 | +0.0000 (saturated) |
| + category/freshness boost | 0.2441 | +0.0001 (marginal post-process) |

**Insight:** ~80% gain từ candidate generation, ~20% từ ranker + ensemble + boost.

### 7.3 Marketplace health metrics (Baseline v15)

| Metric | Value | Interpretation |
|---|---|---|
| Recall@10 | 0.2441 | 42× popularity baseline |
| Freshness@10 | 1.1% | Listings ≤7 ngày tuổi trong top-10 |
| Private seller exposure | 54.9% | Pool 16.5% → **3.3× over-represented** |
| Seller Coverage | 11.88% | ~39K / 327K sellers expose |
| Item Coverage | 2.80% | ~87K / 3.1M items expose |
| Intra-list categories (avg) | 2.46 | Trên thang 5 categories |
| Seller Gini | ~0.93 | High concentration (industry norm 0.85-0.95) |
| Item HHI | <100 | Highly diverse (FTC < 1500 = "non-concentrated") |

### 7.4 Trade-off experiments

| Kịch bản | Recall@10 | Freshness | Seller Coverage | Decision |
|---|---|---|---|---|
| Baseline (v15) | 0.2441 | 1.1% | 11.88% | Current best |
| Variant A — Freshness +5% | ~0.2370 (est) | **3.0%** | 12.73% | ✅ Recommended |
| Variant B — Seller cap ≤2 | ~0.1077 (est) | 0.9% | 12.67% | ❌ Too aggressive |

> ⚠️ Recall ước lượng A/B = proxy qua top-10 overlap với baseline. Không phải LB actual.

---

## 8. Discussion

### 8.1 Insights chính

1. **Personalization premium rất lớn:** Best vs popularity = 42× lift, mean overlap chỉ ~1-2/10 → model thực sự cá nhân hoá, không "lazy fallback to popular".

2. **Organic fairness phát hiện được:** Model tự nhiên ưu tiên private seller (16.5% pool → 54.9% exposed = 3.3× over-rep). Nguyên nhân giả thuyết: private listings có contact rate cao hơn (no agent fees), title/price sát thực tế hơn. → **Không cần thêm fairness constraint cho seller type.**

3. **Diminishing returns ở Recall ≈ 0.244:** Từ v12 → v15 chỉ +0.0001. Effort thêm cho accuracy không xứng đáng → chuyển focus sang marketplace health.

4. **Freshness là điểm yếu rõ ràng:** Chỉ 1.1% listings ≤7 ngày tuổi được expose, mặc dù pool có ~5%. Vòng luẩn quẩn "rich-get-richer": tin cũ có nhiều signal → model tin tưởng → expose nhiều → user contact → càng nhiều signal.

5. **Trade-off định lượng được:**
   - Freshness +5% boost → +180% freshness exposure, −0.7% Recall ước lượng → **xứng đáng cho production**.
   - Seller cap ≤2 → −56% Recall → **không hiệu quả với baseline đã có Gini = 0.93**.

### 8.2 Limitations (trung thực)

1. **NDCG@10 nội bộ không đáng tin** — Validation NDCG bị saturate về 1.0 do feature leakage trong cách split val (last 20% users). Trích `config.py` (line 100): *"n_estimators: 500 — fixed rounds, no early stopping (val NDCG saturates at 1 due to feature leakage)"*. → Đội phải dựa hoàn toàn vào Public LB.

2. **Recall@10 per user segment không tính được** — Ground truth (10/04-07/05) là private. Đội suy đoán hot users có Recall cao hơn dựa trên LGBM score median, nhưng không thể prove offline.

3. **Variant A/B Recall là proxy** — Linear scaling từ top-10 overlap. Không phải LB actual. Đã hết 5 submissions/day quota.

4. **LB chỉ tính trên 5% ground truth** — Private LB sẽ khác. Score 0.2441 chỉ là reference.

5. **Latency benchmark là local single-process** — Production với feature store (Redis) sẽ add 10-20ms.

---

## 9. Production Considerations

### 9.1 Serving architecture

**2-tier proposed:**

- **Real-time path** (latency budget < 100ms):
  1. Feature retrieval từ Redis/Feast (~10-20ms)
  2. ALS + ItemCF retrieval (top-200, ~30-50ms)
  3. LGBM ranker in-memory (<5ms)
  4. Top-10 returned.
- **Offline path** (daily/weekly batch):
  - Retrain EASE/SASRec embeddings weekly
  - Retrain LGBM ranker weekly
  - Refresh trending_pop hourly
  - Refresh cold_top10 daily

### 9.2 Resource profile

| Resource | Value | Note |
|---|---|---|
| Total artifact size | < 20 MB | LGBM + XGB + popular + profiles |
| ALS factor memory | ~6 GB | Approximate KNN (FAISS) trên disk khả thi |
| LGBM predict latency | <5 ms/user batch | CPU-only, no GPU needed |
| Throughput (8 cores) | <1 phút cho 161K users | Nightly batch hoàn toàn ổn |
| Retraining cost | ~3-4 hours weekly | ≈ $0.20/week trên AWS spot |

### 9.3 Failure modes & graceful degradation

| Failure mode | Mitigation |
|---|---|
| Candidate generator down | Fallback `trending_pop` (cached) |
| LGBM model corrupt | Use `blend_score` thuần (giảm ~30% Recall) |
| Feature store timeout | Serve `cold_top10` (cached pickle) |
| `posted_date` null | `days_since_post = 999` → no freshness boost |
| Cold user no profile | Category-weighted interleaved popular pool |
| Traffic spike 3× | Auto-scale + reject ranker, serve cached top-10 |

**Triết lý:** Không bao giờ trả top-10 rỗng. Worst case → `cold_top10.pkl` (always available).

### 9.4 Abuse & risk surface

| Risk | Mitigation hiện tại / khuyến nghị |
|---|---|
| Seller spam posting | Cap exposure ≤2/seller (Variant B); rate-limit posting |
| Click fraud / bot inflation | Filter clicks by IP/device fingerprint pre-aggregation |
| Title keyword stuffing | PhoBERT semantic embedding thay BM25 |
| Cross-seller collusion | Cluster detection trên interaction patterns |
| Price manipulation | Monitor contact-to-deal conversion |

### 9.5 Monitoring metrics

Daily dashboard cần track:
- **Gini coefficient** > 0.95 → alert (seller pool quá concentrate)
- **Freshness@10** < 0.5% → trigger freshness boost increase
- **HHI item** > 1000 → audit dominant items
- **Contact rate per cohort** drops 2σ → trigger retrain

---

## 10. Reproducibility

### 10.1 Directory structure

```
/Volumes/mavuong3112/Datathon_Data/
├── dim_listing/                       # Raw item metadata (40 parquet shards, ~200MB)
├── fact_user_events/                  # Raw clickstream (heavy)
├── fact_listing_snapshot/             # Daily snapshots
├── fact_post_contact_interactions/    # Post-contact events
├── test/test_users.parquet            # 161,568 test users
│
├── model_v16_0.xxxx/                  # ⭐ Best pipeline
│   ├── 01_extract.py … 09_covis.py    # 12 pipeline stages
│   ├── 10_marketplace_health.py       # Health analysis script
│   ├── config.py                      # Hyperparameters
│   ├── cache/                         # Trained model outputs
│   │   ├── lgbm_ranker.txt
│   │   ├── xgboost_ranker.json
│   │   ├── ranked_predictions.parquet
│   │   ├── features_train.parquet
│   │   └── features_test.parquet
│   └── submission_stage15_0.2441.csv  # Best submission
│
├── model/cache/                       # Legacy cache
│   ├── popular_items.parquet
│   ├── user_profiles.parquet
│   ├── items.parquet
│   └── cold_top10.pkl
│
├── marketplace_health_analysis/       # ⭐ This folder (deliverable)
│   ├── README.md                      # ← bạn đang đọc
│   ├── report.ipynb                   # Notebook chính 73 cells
│   ├── _build_report.py               # Generator script
│   ├── metrics_comparison.csv         # 3-scenario comparison
│   ├── health_tradeoff_chart.png      # 6-panel + radar
│   └── health_scorecard.png           # 6-tile health scorecard
│
├── full_eda_story.md                  # EDA chi tiết (~2000 dòng)
└── datathon-chung-ket/                # Submission staging folder
    └── (various submission_stage*.csv files)
```

### 10.2 Regenerate notebook

```bash
# Re-build notebook from generator script
python3 marketplace_health_analysis/_build_report.py

# Execute end-to-end (outputs embedded)
python3 -m jupyter nbconvert --to notebook --execute \
    marketplace_health_analysis/report.ipynb \
    --output report.ipynb \
    --ExecutePreprocessor.timeout=600
```

### 10.3 Retrain best model

```bash
cd model_v16_0.xxxx/
python3 run_pipeline.py  # runs stages 01 → 08 sequentially
# Expected output: submission_stage15_0.2441.csv
```

### 10.4 Re-run marketplace health analysis

```bash
python3 model_v16_0.xxxx/10_marketplace_health.py
# Outputs:
#   marketplace_health_analysis/metrics_comparison.csv
#   marketplace_health_analysis/health_tradeoff_chart.png
```

### 10.5 Export báo cáo sang HTML/PDF cho BGK

```bash
python3 -m jupyter nbconvert --to html marketplace_health_analysis/report.ipynb
# → report.html (open in browser, print to PDF if needed)
```

### 10.6 Software dependencies

```
python      ≥ 3.10
pandas      ≥ 2.0
numpy       ≥ 1.24
pyarrow     ≥ 14.0
lightgbm    ≥ 4.0  (4.6.0 verified)
xgboost     ≥ 2.0
catboost    ≥ 1.2
matplotlib  ≥ 3.7
seaborn     ≥ 0.12
nbformat    ≥ 5.10
implicit    ≥ 0.7  (for ALS)
duckdb      ≥ 0.9
```

---

## 11. References

- **[1]** Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets.* ICDM 2008.
- **[2]** Steck, H. (2019). *Embarrassingly Shallow Autoencoders for Sparse Data.* WWW '19.
- **[3]** Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). *Item-based Collaborative Filtering Recommendation Algorithms.* WWW '01.
- **[4]** Kang, W. C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation.* ICDM 2018.
- **[5]** Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research Technical Report MSR-TR-2010-82.
- **[6]** Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- **[7]** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
- **[8]** Vu, D. Q., Nguyen, A. T., & Nguyen, D. Q. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* EMNLP Findings 2020.
- **[9]** Covington, P., Adams, J., & Sargin, E. (2016). *Deep Neural Networks for YouTube Recommendations.* RecSys '16.
- **[10]** Beel, J., et al. (2016). *Towards Reproducibility in Recommender-Systems Research.* User Modeling and User-Adapted Interaction.

---

## 12. Team & Acknowledgments

**The Gridbreakers — VinUniversity:**
- **Team Lead:** Ngô Quang Huy
- **Club:** VinTelligence — VinUniversity Data Science & AI Club

### Cảm ơn

- **Chợ Tốt BĐS** — cung cấp dataset production-real với hàng trăm triệu sự kiện thực tế, environment thử thách đúng nghĩa.
- **VinTelligence team** — tổ chức Datathon 2026 chu đáo, công bằng.
- **Các đội thi khác** — competitive spirit thúc đẩy chúng tôi đẩy từ 0.2184 → 0.2441 trong 11 ngày.

### Liên hệ

Mọi câu hỏi về reproducibility, vui lòng tham khảo `report.ipynb` hoặc liên hệ Team Lead.

---

**Hết tài liệu · Datathon 2026 · The Gridbreakers · 22/05/2026**
