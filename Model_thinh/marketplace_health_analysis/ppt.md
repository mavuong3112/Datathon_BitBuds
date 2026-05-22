# 📑 `datathon_final_deck.pptx` — Tài liệu chi tiết (5-slide condensed)

> Tài liệu mô tả đầy đủ nội dung, layout, và visual design của slide deck cuối cùng cho **Datathon 2026 — The Gridbreakers**. Đây là phiên bản **condensed 5 slides** rút gọn từ deck 17 slides, mỗi slide là một dashboard tổng hợp nhiều charts + content.

**File:** [`slides/datathon_final_deck.pptx`](slides/datathon_final_deck.pptx)
**Kích thước file:** ~479 KB
**Số slide:** 5 (không có cover/thank-you slide)
**Format:** 16:9 widescreen (13.33" × 7.50")
**Font:** Arial (universal — Mac/Win/Linux + full Vietnamese diacritic support)

---

## 🎨 Design System

### Color Palette (Modern Data-Analyst)

| Color | Hex | Vai trò |
|---|---|---|
| **SLATE** | `#1B2538` | Header strip, panel headers chính, hero KPI |
| **SLATE_DARK** | `#0F1624` | Body title text |
| **INDIGO** | `#3F51B5` | Accent secondary, ranker, item-level |
| **SKY** | `#00B0FF` | Data viz, candidate gen, user-level |
| **MINT** | `#00BFA5` | Success metrics, positive findings, organic fairness |
| **CORAL** | `#FF5252` | Section accent stripe, alerts, limitations |
| **AMBER** | `#FFB74D` | Warnings, callouts, retrain |
| **MUTED** | `#607D8B` | Subtitle text, pool comparison |
| **SOFT_GREY** | `#ECEFF3` | Dividers, panel borders, progress bg |

### Slide Template (Common Chrome)

```
┌──────────────────────────────────────────────────────────────────────┐
│ ⬛ SLATE HEADER STRIP                                                │
│  DATATHON 2026 › SECTION       Chợ Tốt BĐS · The Gridbreakers   [#] │
│ ▬▬▬▬▬▬▬▬▬▬▬▬▬▬ coral accent stripe ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬│
│                                                                     │
│  SLIDE TITLE (22pt bold)                                            │
│  ▪ italic subtitle (10.5pt muted)                                   │
│                                                                     │
│  [DASHBOARD BODY — multiple components per slide]                   │
│                                                                     │
│  #/5                                                                │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ coral progress bar                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Typography Hierarchy

| Element | Font | Size | Weight |
|---|---|---|---|
| Slide title | Arial | 22pt | Bold |
| Subtitle | Arial | 10.5pt | Italic |
| Hero KPI number | Arial | 48pt | Bold |
| Sub-KPI number | Arial | 20-32pt | Bold |
| KPI label | Arial | 9-10pt | Bold |
| Panel header | Arial | 10pt | Bold (white on color) |
| Body bullet | Arial | 8.5-9pt | Regular |
| Code/table (math, ASCII) | Courier New | 7.8-8.5pt | Regular |
| Caption | Arial | 7.5-8.5pt | Italic |

### Components

- **KPI Tile** — Rounded rect với big number + label + sublabel
- **Panel** — Card with colored header bar + white body
- **Callout** — Highlighted box with side stripe + label + body
- **Stage Card** — Pipeline stage representation with color band + number + bullets
- **Progress Bar** — For self-assessment scores

---

## 🗺️ Slide Map · BGK Criterion Coverage

| Slide | Tên slide | Section | BGK Criterion |
|---|---|---|---|
| **1** | Tổng quan · Bài toán · Dataset | OVERVIEW & DATA | Đặt vấn đề + Context |
| **2** | Solution Architecture · Math · Features | ARCHITECTURE | **① Thiết kế giải pháp** |
| **3** | Performance Dashboard · Recall@10 = 0.2441 | PERFORMANCE | **② Hiệu suất mô hình** |
| **4** | Marketplace Health Dashboard · 5 trục | MARKETPLACE HEALTH | **③ Marketplace Health** ⭐ |
| **5** | Production · Roadmap · Limitations | PRODUCTION & FUTURE | **④ Tính khả thi triển khai** |

---

## 📋 Slide-by-Slide Breakdown

---

### 📍 SLIDE 1 — Tổng quan · Bài toán · Dataset

**Section:** OVERVIEW & DATA · **Page:** 1/5

**Title:** Tổng quan · Bài toán · Dataset
**Subtitle:** 161,568 users · 3.1M items · Recall@10 = 0.2441 (Public LB) · 42× lift vs popularity

**Layout:** 3-row dashboard

#### Row 1 — Hero + 3 KPIs + Pipeline Summary (y=1.55-3.40)
- **Hero KPI Tile (slate, left):** `0.2441` Recall@10 (48pt) · "v15 stage15 · best score"
- **3 Sub-KPI Tiles (middle, horizontal):**
  - `42×` LIFT (mint) — ↑ vs popularity
  - `161K` TEST USERS (sky) — top-10/user
  - `3.1M` ITEMS (indigo) — 5 categories
- **Pipeline Overview Panel (indigo, right):** 7 bullets về 5 retrievers, features, ensemble, blend, fallback

#### Row 2 — Problem Equation + 4 Data KPIs (y=3.55-5.05)
- **Formal Objective Card (sky bg, left):**
  - Equation: `argmax_θ  E_u [ |f_θ(u) ∩ G_u| / |G_u| ]` (16pt mono center)
  - Sub-text: `f_θ : U → I^10`, `G_u` definition, metric notes
- **4 Mini KPI Cards (right):**
  - `U` `161K` users (sky)
  - `I` `3.1M` items (indigo)
  - `T` `152d` train (mint)
  - `Δt` `28d` eval (coral)

#### Row 3 — 4 Detail Panels (y=5.15-7.00)
- **POSITIVE EVENTS** (mint): 5 event types
- **5 CATEGORIES** (indigo): 1010-1050
- **CONSTRAINTS** (coral): 5 ràng buộc đề thi
- **KEY INSIGHTS** (amber): 5 findings highlights

**Charts:** Không có chart — pure KPI tiles + panels (compact information density)

**Purpose:** Snapshot 1-slide tóm gọn toàn bộ dự án, dataset, problem formulation cho BGK trong 60 giây.

---

### 📍 SLIDE 2 — Solution Architecture · Math · Features

**Section:** ARCHITECTURE · **Page:** 2/5

**Title:** Solution Architecture · Math · Features
**Subtitle:** Two-stage paradigm · 5 retrievers · 54 features · LGBM + XGBoost ensemble

**Layout:** 2-row dashboard

#### Row 1 — 3 Stage Cards (y=1.55-3.70)
3 stage cards với arrows giữa các stage:

- **STAGE 1 · CANDIDATE GENERATION** (sky):
  - 5 retrievers · top-100 each (ALS, EASE, ItemCF, SASRec, CBF)
  - Hyperparams cụ thể (factors=512, λ=200, β=0.005, d=64, …)
  - merge & cap 700/user
- **STAGE 2 · LEARNING TO RANK** (indigo):
  - 54 features → LGBM (depth=8, leaves=95, lr=0.05, n_est=500, 10 seeds)
  - XGBoost (depth=6, 1000 rounds)
- **STAGE 3 · BLEND + POST-PROCESS** (mint):
  - Per-user min-max norm
  - `s_final = 0.65·LGBM + 0.35·XGB + β_f·1[fresh] + β_c·1[cat=pref]`
  - Cold-user fallback

#### Row 2 — Math + Chart + Taxonomy (y=3.90-7.00)

- **MATH NOTATION Panel (slate, left, ~5"):** Mono-spaced formulas cho ALS [1], EASE [2], ItemCF [3], SASRec [4], CBF [8], LambdaRank [5,6], final blend
- **Feature Importance Chart (middle, ~4"):** `chart_feature_importance.png` — Top-15 LGBM gain bars
- **Feature Taxonomy Panel (amber, right, ~3.2"):** 4 groups breakdown:
  - USER-LEVEL · 15 (n_pos_events, intent_score_log, pref_category)
  - ITEM-LEVEL · 18 (trend_pos, days_since_post, item_quality)
  - CROSS user×item · 12 (is_repeat, ui_match_category)
  - SEQUENCE/INTENT · 9 (last_click_days, dwell_avg)

**Charts:** `chart_feature_importance.png` (1 chart)

**Purpose:** Toàn bộ kiến trúc model + math notation + feature engineering trong 1 slide — covers BGK criterion #1 (Thiết kế giải pháp).

---

### 📍 SLIDE 3 — Performance Dashboard ⭐

**Section:** PERFORMANCE · **Page:** 3/5

**Title:** Performance Dashboard · Recall@10 = 0.2441
**Subtitle:** LB progression · Personalization 42× · Score diagnostics · Top features

**Layout:** 3-row dashboard

#### Row 1 — 4 KPI Tiles (y=1.55-2.85)
- `0.2441` Recall@10 LB (slate) · v15 best
- `42×` Personalization lift (mint) · vs popularity 0.0058
- `+11.8%` Gain vs v1 (indigo) · 0.2184 → 0.2441 in 11 days
- `1.2/10` Avg overlap (amber) · với popularity top-10

#### Row 2 — Combo Dashboard Chart + Insights Stack (y=2.95-6.75)
- **`dashboard_performance.png` (left, 9.5"×3.8"):** 2×2 grid với 4 panels:
  1. **① LB Progression** — bar chart v1→v15 với popularity baseline
  2. **② Personalization** — bar comparison + 42× lift annotation
  3. **③ Score by Rank Bucket** — boxplot Top-3 / Mid / Tail
  4. **④ Top-10 Feature Importance** — horizontal bar
- **3 Insight Callouts (right column, stacked):**
  - **STRATEGIC DECISION** (amber): Diminishing returns ở 0.244 → dừng tối ưu accuracy
  - **PERSONALIZATION PROOF** (mint): 80-90% slot personalized
  - **SCORE STRUCTURE** (sky): Top-3 confident · Mid fragile · Tail uncertainty

#### Row 3 — Honest Disclaimer Banner (y=6.80-7.22)
Amber-tinted banner với 4 limitations:
- NDCG@10 saturate về 1.0 (feature leakage)
- Recall per segment không đo được (GT private)
- Variant A/B Recall = proxy
- Public LB chỉ 5% GT

**Charts:** `dashboard_performance.png` (combo 4-panel)

**Purpose:** Toàn bộ performance story + chứng minh personalization + honest limitations — covers BGK criterion #2.

---

### 📍 SLIDE 4 — Marketplace Health Dashboard ⭐⭐

**Section:** MARKETPLACE HEALTH · **Page:** 4/5

**Title:** Marketplace Health Dashboard · 5 trục đánh giá
**Subtitle:** Scorecard · Trade-off · Concentration · Age skew · Fairness — Organic 3.3× private boost

**Layout:** Multi-zone dashboard

#### Zone A — Health Combo Dashboard (left, ~8.4"×4.5")
**`dashboard_health.png`:** 2×2 grid:
1. **① HEALTH SCORECARD** — 6 mini tiles (Freshness, Seller Cov, Item Cov, Cat Entropy, Gini, HHI) với color-coding green/amber/red
2. **② TRADE-OFF** — 3-scenario dual-axis bar (Recall vs Freshness)
3. **③ LORENZ · SELLER EXPOSURE** — Lorenz curve với Gini value
4. **④ EXPOSURE BY AGE** — Pool vs Submission grouped bar by age bucket

#### Zone B — Fairness Highlight (right top, ~4"×3.0")
- **`chart_seller_donut.png`:** 2-donut comparison (Pool vs Exposed seller_type)
- **Organic Fairness Tile** (mint): `private 16.5% pool → 54.9% exposed · 3.3× boost`

#### Zone C — Why Insight Callout (right bottom, ~4"×1.3")
Mint-tinted callout giải thích tại sao model ưu tiên private:
- Contact rate cao hơn (no commission)
- Title/desc match user intent tốt hơn
- Pricing thực tế hơn
- → Không cần fairness constraint thêm

#### Zone D — Trade-off Matrix Table (bottom, full-width, ~12.5"×1.1")
Slate-headed table với 4 rows × 5 cols:
- Baseline (v15) — Current LB best
- Freshness +5% — ✓ YES production (−0.7% Recall · +180% freshness)
- Hard seller cap ≤2 — ✗ TOO AGGRESSIVE (−56% Recall)
- Soft seller cap = 3 — ⟳ NEEDS A/B TEST

**Charts:** `dashboard_health.png` + `chart_seller_donut.png` (2 chart files, 5 visualizations total)

**Purpose:** **THE big slide** — toàn bộ marketplace health analysis bao gồm organic fairness finding + trade-off decision matrix — covers BGK criterion #3 ⭐.

---

### 📍 SLIDE 5 — Production · Roadmap · Limitations

**Section:** PRODUCTION & FUTURE · **Page:** 5/5

**Title:** Production Readiness · Roadmap · Honest Limitations
**Subtitle:** Serving · Latency · Failure modes · Roadmap 30/60/90 · Self-assessment 4 trục BGK

**Layout:** 3-row dashboard

#### Row 1 — 4 Production KPI Tiles (y=1.55-2.65)
- `< 5 ms` Inference Latency (mint) · LGBM per user batch
- `< 20 MB` Model Footprint (indigo) · fit 1 pod · no GPU
- `< 1 min` Throughput 8 cores (sky) · full 161K users
- `~$0.20` Retrain Cost/Week (amber) · 3-4h AWS spot

#### Row 2 — Architecture + Failure Modes (y=2.80-5.25)
- **SERVING ARCHITECTURE 2-TIER Panel** (slate, left, ~6.3"):
  - Real-time path (Feature store ↔ retrieval ↔ ranker) <100ms
  - Offline path (hourly/daily/weekly/monthly cadence)
- **FAILURE MODES Panel** (coral, right, ~6.1"):
  - 7 failure scenarios với mitigation
  - "TRIẾT LÝ: Không bao giờ trả top-10 rỗng. Worst case → cold_top10.pkl"

#### Row 3 — Roadmap + Self-Assessment + Limitations (y=5.35-7.28)
- **ROADMAP 30/60/90 Panel** (slate header, ~5.4"): 3 phase mini-cards với bullets
  - `30d` CANARY DEPLOY (sky): Variant A canary, monitoring, A/B test infra
  - `60d` TWO-TOWER (indigo): user×item tower PoC, online learning, seller cold-start
  - `90d` GNN + MULTI-OBJ (mint): GNN graph, multi-objective optimization, cross-category
- **SELF-ASSESSMENT Panel** (slate header, ~3.65"): 4 progress bars
  - Thiết kế: 4.0/5 (sky)
  - Hiệu suất: 4.0/5 (indigo)
  - Health: 4.5/5 (mint)
  - Production: 4.0/5 (amber)
- **HẠN CHẾ TRUNG THỰC Panel** (coral, ~3.2"): 5 limitations bullets

**Charts:** Không có chart — pure structured panels

**Purpose:** Production-readiness + forward-looking roadmap + honest self-assessment + limitations — covers BGK criterion #4 + closing.

---

## 📊 Charts Reference

| Chart file | Used in | Size | Description |
|---|---|---|---|
| `chart_feature_importance.png` | Slide 2 | 57 KB | Top-15 LGBM feature gain (horizontal bar) |
| `chart_seller_donut.png` | Slide 4 | 57 KB | Seller type 2-donut (pool vs exposed) |
| **`dashboard_performance.png`** | Slide 3 | **174 KB** | **2×2 combo:** LB · Personalization · Score · Features |
| **`dashboard_health.png`** | Slide 4 | **184 KB** | **2×2 combo:** Scorecard · Trade-off · Lorenz · Age |

**Tổng:** 4 chart files (2 combo dashboards + 2 single charts)

**Lưu ý:** 6 chart cũ (chart_lb_progression, chart_personalization, chart_score_distribution, chart_concentration, chart_age_category, chart_memory, chart_health_scorecard, chart_tradeoff_compact) vẫn còn trong `slides/charts/` để reuse — đã được hợp nhất vào 2 dashboard combo trên.

---

## 🛠️ Regenerate

```bash
# 1. Regen 2 combo dashboards (slow ~30s — loads dim_listing)
python3 marketplace_health_analysis/_build_dashboard_charts.py

# 2. (Optional) Regen các chart đơn lẻ
python3 marketplace_health_analysis/_build_slides_charts.py

# 3. Build pptx (fast ~3s)
python3 marketplace_health_analysis/_build_slides.py
```

**Output:** `/Volumes/mavuong3112/Datathon_Data/marketplace_health_analysis/slides/datathon_final_deck.pptx`

---

## 📋 Quality Checklist

- ✅ **5 slides** (rút gọn từ 17), không cover/thank-you
- ✅ 16:9 widescreen format (13.33" × 7.50")
- ✅ Arial font universal — Vietnamese diacritics render đúng trên Mac/Win/Linux
- ✅ **2 combo dashboard charts** — performance + health đều 2×2 grid
- ✅ Consistent design system: SLATE header + CORAL accent + footer progress
- ✅ **Tỉ lệ chart hợp lý**: dashboard charts ~9.5"×3.8" cho slide 3, ~8.4"×4.5" cho slide 4
- ✅ Font sizes hierarchical: hero 48pt → sub 32pt → KPI 22pt → body 9pt
- ✅ Màu chữ trên nền: white-on-colored-tile cho hero metrics, dark-on-white cho body
- ✅ Mỗi slide map 1-1 với BGK criterion (trừ slide 1 = context, slide 5 = production)
- ✅ Honest limitations clearly stated (slide 3 banner + slide 5 panel)
- ✅ Hero KPI 0.2441 luôn nổi bật ở slide 1

---

## 🎯 So sánh v1 (17 slides) vs v2 (5 slides)

| Tiêu chí | v1 (17 slides) | v2 (5 slides) |
|---|---|---|
| Số slide | 17 | 5 |
| Tổng size | 630 KB | 479 KB |
| Charts | 10 single charts | 2 combo + 2 single = 4 files |
| Time to present | ~25-30 phút | ~10-15 phút |
| Information density | Trung bình | Cao (dashboard-style) |
| Phù hợp khi | Buổi present dài, đi sâu | Pitch ngắn, Q&A nhiều |

**Khi nào dùng v2 (5 slides) — recommended cho Datathon:**
- BGK có ~15 phút mỗi đội
- Cần thể hiện key insights nhanh
- Dashboard format giống business intelligence reports → professional impression

---

**Author:** The Gridbreakers · VinUniversity
**Date:** 22/05/2026
**Deck:** `slides/datathon_final_deck.pptx` (~479 KB · 5 slides)
**Builder:** `_build_slides.py` + `_build_dashboard_charts.py`
