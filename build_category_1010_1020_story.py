"""Build data-story folder for categories 1010 vs 1020."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
OUT = DATA_ROOT / "outputs" / "category_1010_1020_story"
CHART = OUT / "charts"
COMPARE = CHART / "compare"
SRC_1010 = DATA_ROOT / "outputs" / "category_1010_business"
SRC_1020 = DATA_ROOT / "outputs" / "category_1020_business"
PERF = DATA_ROOT / "outputs" / "eda_category_1010_1020"
BEHAV = DATA_ROOT / "outputs" / "eda_category_behavior"
BRIDGE = PERF / "bridge"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_deck_charts() -> None:
    for cat, src in ((1010, SRC_1010), (1020, SRC_1020)):
        dst = CHART / str(cat)
        dst.mkdir(parents=True, exist_ok=True)
        for png in (src / "charts").glob("*.png"):
            shutil.copy2(png, dst / png.name)


def chart_compare_supply_mix() -> None:
    rows = []
    for cat in (1010, 1020):
        df = pd.read_csv(BEHAV / str(cat) / "01_supply_ad_type.csv")
        for _, r in df.iterrows():
            rows.append({"category": cat, "ad_type": r["ad_type"], "pct": r["pct"]})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    w = 0.35
    let_p = [d.loc[(d.category == c) & (d.ad_type == "let"), "pct"].iloc[0] for c in (1010, 1020)]
    sell_p = [d.loc[(d.category == c) & (d.ad_type == "sell"), "pct"].iloc[0] for c in (1010, 1020)]
    ax.bar(x - w / 2, let_p, w, label="Cho thuê (let)", color="#1f77b4")
    ax.bar(x + w / 2, sell_p, w, label="Bán (sell)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(["1010\nCăn hộ", "1020\nNhà ở"])
    ax.set_ylabel("Tỷ lệ supply (%)")
    ax.set_title("Hai category — hai playbook supply", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, COMPARE / "01_supply_let_sell_side_by_side.png")


def chart_compare_cvr() -> None:
    cvr = pd.read_csv(PERF / "02_cvr_baseline_adtype.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    w = 0.35
    for i, ad in enumerate(["let", "sell"]):
        vals = [
            cvr.loc[(cvr.category == c) & (cvr.ad_type == ad), "cvr_pct"].iloc[0]
            for c in (1010, 1020)
        ]
        ax.bar(x + (i - 0.5) * w, vals, w, label=ad, color=["#2ca02c", "#9467bd"][i])
    ax.set_xticks(x)
    ax.set_xticklabels(["1010", "1020"])
    ax.set_ylabel("Catalog CVR (%)")
    ax.set_title("CVR đảo chiều: 1010 sell > let · 1020 let > sell", fontweight="bold")
    ax.legend(title="ad_type")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, COMPARE / "02_cvr_baseline_side_by_side.png")


def chart_compare_funnel() -> None:
    rows = []
    for cat in (1010, 1020):
        f = pd.read_csv(BEHAV / str(cat) / "03_events_login_funnel.csv")
        ec = f.loc[f.stage == "explicit_contact", "pct_of_sessions"].iloc[0]
        oi = f.loc[f.stage == "other_interaction", "pct_of_sessions"].iloc[0]
        rows.append({"category": cat, "explicit_contact": ec, "other_interaction": oi})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, d["explicit_contact"], w, label="Explicit contact", color="#2ca02c")
    ax.bar(x + w / 2, d["other_interaction"], w, label="Other interaction", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(["1010", "1020"])
    ax.set_ylabel("% sessions (login sample)")
    ax.set_title("Funnel: lead thật vs exposure ảo", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, COMPARE / "03_session_funnel_side_by_side.png")


def chart_compare_health() -> None:
    h = pd.read_csv(BRIDGE / "04_segment_event_efficiency.csv")
    order = [
        "high_quality_underexposed",
        "normal",
        "oversaturated_low_conversion",
    ]
    labels = ["HQ underexposed", "Normal", "Oversaturated"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cat in zip(axes, (1010, 1020)):
        sub = h.loc[h.category == cat].set_index("health_segment").reindex(order)
        ax.bar(labels, sub["avg_event_contact_rate_pct"], color=["#2ca02c", "#1f77b4", "#d62728"])
        ax.set_title(f"Category {cat}")
        ax.set_ylabel("Contact / pageview %")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Marketplace health — cùng pattern, khác quy mô", fontweight="bold", y=1.02)
    _save(fig, COMPARE / "04_health_efficiency_side_by_side.png")


def chart_compare_impact() -> None:
    rows = []
    for cat in (1010, 1020):
        imp = pd.read_csv(DATA_ROOT / "outputs" / f"category_{cat}_business" / "csv" / "impact_scenarios.csv")
        boost = imp.loc[imp.scenario.str.contains("Boost"), "value"].iloc[0]
        rows.append({"category": cat, "boost_contacts": boost})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["1010\nCăn hộ", "1020\nNhà ở"], d["boost_contacts"], color=["#1f77b4", "#ff7f0e"])
    for i, v in enumerate(d["boost_contacts"]):
        ax.text(i, v * 1.02, f"{v:,.0f}".replace(",", "."), ha="center", fontsize=9)
    ax.set_ylabel("Contact events (ước tính)")
    ax.set_title("What-if: boost HQ underexposed", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, COMPARE / "05_impact_boost_side_by_side.png")


def build_comparison_csv() -> None:
    rows = []
    prof = pd.read_csv(PERF / "01_profile_1010_1020.csv")
    cvr = pd.read_csv(PERF / "02_cvr_baseline_adtype.csv")
    health = pd.read_csv(BRIDGE / "04_segment_event_efficiency.csv")
    for cat in (1010, 1020):
        p = prof.loc[prof.category == cat].iloc[0]
        sc = pd.read_csv(BEHAV / str(cat) / "05_marketing_scorecard.csv")
        funnel = pd.read_csv(BEHAV / str(cat) / "03_events_login_funnel.csv")
        imp = pd.read_csv(DATA_ROOT / "outputs" / f"category_{cat}_business" / "csv" / "impact_scenarios.csv")
        hq = health.loc[
            (health.category == cat)
            & (health.health_segment == "high_quality_underexposed")
        ].iloc[0]
        rows.append(
            {
                "category": cat,
                "label": p["ui_label"],
                "n_listings": int(p["n"]),
                "pct_let": float(p["pct_let"]),
                "cvr_let": float(cvr.loc[(cvr.category == cat) & (cvr.ad_type == "let"), "cvr_pct"].iloc[0]),
                "cvr_sell": float(cvr.loc[(cvr.category == cat) & (cvr.ad_type == "sell"), "cvr_pct"].iloc[0]),
                "session_explicit_pct": float(
                    sc.loc[sc.metric == "session_explicit_contact_pct", "value"].iloc[0]
                ),
                "hq_underexposed_n": int(hq["n"]),
                "hq_efficiency_pct": float(hq["avg_event_contact_rate_pct"]),
                "boost_contact_events_est": float(
                    imp.loc[imp.scenario.str.contains("Boost"), "value"].iloc[0]
                ),
            }
        )
    csv_dir = OUT / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_dir / "comparison_1010_vs_1020.csv", index=False)


def _fmt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def _story_body(r10: pd.Series, r20: pd.Series) -> str:
    return f"""# Hai thị trường trên một nền tảng — 1010 vs 1020

> *Data story · tổng hợp từ performance, behavior, clustering, bridge*  
> Deck gốc: [`category_1010_business`](../category_1010_business/) · [`category_1020_business`](../category_1020_business/)

---

## Mở đầu — Câu hỏi của PM

Khi nhìn dashboard tổng, **1010 (Căn hộ)** và **1020 (Nhà ở)** thường bị gộp chung nhóm BĐS. Nhưng sau khi lần theo supply → snapshot → clickstream → cluster health, hai category này **không cùng một cuộc chơi**:

| | **1010 Căn hộ** | **1020 Nhà ở** |
|---|-----------------|-----------------|
| Quy mô catalog | {_fmt_int(r10['n_listings'])} tin | {_fmt_int(r20['n_listings'])} tin |
| DNA supply | **{r10['pct_let']:.1f}% cho thuê** | **{100 - r20['pct_let']:.1f}% bán** |
| CVR let / sell | {r10['cvr_let']:.2f}% / {r10['cvr_sell']:.2f}% | {r20['cvr_let']:.2f}% / {r20['cvr_sell']:.2f}% |
| Session có lead thật | {r10['session_explicit_pct']:.2f}% | {r20['session_explicit_pct']:.2f}% |
| Tin “ngon” bị underexposed | {_fmt_int(r10['hq_underexposed_n'])} | {_fmt_int(r20['hq_underexposed_n'])} |

**Insight mở:** 1010 thắng khi **bán** căn hộ; 1020 thắng khi **cho thuê** nhà — đảo ngược kỳ vọng nếu chỉ đọc CVR trung bình ~20%.

![So sánh supply let/sell](charts/compare/01_supply_let_sell_side_by_side.png)

---

## Chương 1 — Bối cảnh (SITUATION)

### 1010: Thành phố thuê căn hộ

- **611k tin**, **74% let** — thị trường **thuê**, không phải mua.
- **TP.HCM ~74%** supply → chiến lược **HCM-first**.
- Catalog CVR: let **18.8%**, sell **23.0%** (+4.2pp).
- Snapshot: **11%** listing-days có contact trên cho thuê.

→ [Deck 1010 SCIS](../category_1010_business/category_1010_business_scis.md)

![1010 supply](charts/1010/01_supply_adtype.png)

### 1020: Đại dương bán nhà

- **1.51M tin** — gấp **2.5×** 1010; **72% sell**.
- CVR: let **20.5%** > sell **19.3%** — **ngược 1010**.
- Cho thuê: **19%** ngày có contact; **bán: 8.6%** — nút thắt chính.

→ [Deck 1020 SCIS](../category_1020_business/category_1020_business_scis.md)

![1020 supply](charts/1020/01_supply_adtype.png)

![CVR baseline so sánh](charts/compare/02_cvr_baseline_side_by_side.png)

---

## Chương 2 — Nơi lead bị “rò” (CHALLENGES)

**~50% events login** là `other_interaction`; **94–96%** là `ad_view` — không phải lead.

| Category | Session explicit | Events other_interaction |
|----------|------------------|--------------------------|
| 1010 | **7.78%** | **50.6%** |
| 1020 | **8.38%** | **50.8%** |

![Funnel so sánh](charts/compare/03_session_funnel_side_by_side.png)

**Marketplace health:** 1010 ~2.3k vs 1020 ~6.2k tin HQ underexposed; oversaturated exposure **13** (1020) vs **8** (1010).

![Health segments](charts/compare/04_health_efficiency_side_by_side.png)

**Weak pockets:** 1010 — 1PN thuê ~289k tin; 1020 — sell 30–50m² ~310k tin + bulk seller spam.

---

## Chương 3 — Hai playbook (STRATEGIES)

| 1010 | 1020 |
|------|------|
| Boost 2.3k underexposed | Boost 6.2k underexposed |
| Chat-first (~30%) | Phone-first (~77%) |
| Desktop + iOS | Android + deep-compare |
| Night push | Listing completeness (legal, width) |

![1010 channels](charts/1010/07_contact_channel_mix.png)
![1020 channels](charts/1020/07_contact_channel_mix.png)

---

## Chương 4 — IMPACTS (what-if)

| Scenario | 1010 | 1020 |
|----------|------|------|
| Boost underexposed | +{_fmt_int(r10['boost_contact_events_est'])} contacts | +{_fmt_int(r20['boost_contact_events_est'])} contacts |
| Session +1pp | +3.7k sessions | +9.7k sessions |
| Demote oversaturated | ~68k imp-equiv | ~322k imp-equiv |

![Impact boost](charts/compare/05_impact_boost_side_by_side.png)

![1010 what-if](charts/1010/10_impact_whatif.png)

![1020 what-if](charts/1020/10_impact_whatif.png)

---

## Kết — Một câu cho từng category

- **1010:** *Thuê căn hộ ở HCM — lead thật hiếm, gem bị chôn; boost underexposed và đừng nhầm ad_view với contact.*
- **1020:** *Bán nhà là đại dương — let convert tốt, sell cần fix 30–50m²; boost scale gấp 3× 1010.*

## Phụ lục

| File | Mô tả |
|------|--------|
| [`csv/comparison_1010_vs_1020.csv`](csv/comparison_1010_vs_1020.csv) | Bảng số so sánh |
| [`chapters/1010_situation_to_impact.md`](chapters/1010_situation_to_impact.md) | Deck 1010 đầy đủ |
| [`chapters/1020_situation_to_impact.md`](chapters/1020_situation_to_impact.md) | Deck 1020 đầy đủ |

### Caveats

- Explicit contact ≠ `other_interaction` / ad_view · CVR catalog ≠ clustering cohort · Sample 6–8%.
"""


def write_chapter_copies() -> None:
    chap = OUT / "chapters"
    chap.mkdir(parents=True, exist_ok=True)
    for cat in (1010, 1020):
        src = DATA_ROOT / "outputs" / f"category_{cat}_business" / f"category_{cat}_business_scis.md"
        text = src.read_text(encoding="utf-8")
        # Fix chart paths for chapter subfolder
        text = text.replace("](charts/", f"](../charts/{cat}/")
        (chap / f"{cat}_situation_to_impact.md").write_text(text, encoding="utf-8")


def write_cases_story(comp: pd.DataFrame) -> None:
    r10 = comp.loc[comp.category == 1010].iloc[0]
    r20 = comp.loc[comp.category == 1020].iloc[0]
    imp10 = pd.read_csv(SRC_1010 / "csv" / "impact_scenarios.csv")
    imp20 = pd.read_csv(SRC_1020 / "csv" / "impact_scenarios.csv")

    cases_index = {
        "1010": [
            {"id": "A1", "title": "DNA thị trường thuê HCM", "priority": 1},
            {"id": "A2", "title": "Bán căn hộ convert tốt hơn thuê", "priority": 2},
            {"id": "A3", "title": "Bẫy KPI: ad_view không phải lead", "priority": 1},
            {"id": "A4", "title": "Gem bị chôn — boost underexposed", "priority": 1},
            {"id": "A5", "title": "Pocket 1PN thuê — volume lớn CVR thấp", "priority": 2},
        ],
        "1020": [
            {"id": "B1", "title": "DNA thị trường bán nhà quy mô 1.5M", "priority": 1},
            {"id": "B2", "title": "Cho thuê nóng / bán lạnh trên snapshot", "priority": 1},
            {"id": "B3", "title": "Bẫy KPI: ad_view không phải lead", "priority": 1},
            {"id": "B4", "title": "Gem scale 6.2k — boost underexposed", "priority": 1},
            {"id": "B5", "title": "Pocket sell 30–50m² + bulk seller", "priority": 2},
        ],
        "cross": [
            {"id": "X1", "title": "CVR đảo chiều giữa 1010 và 1020"},
            {"id": "X2", "title": "Cùng pattern health segment, khác scale"},
        ],
    }

    body = f"""# Case study — 1010 & 1020 · Góc nhìn Business Data Analyst

> **Cách đọc:** Mỗi *case* = một quyết định kinh doanh có thể bàn với PM/Leadership.  
> Chỉ giữ **5 case/category** + **2 case chéo** — phần còn lại của EDA là bối cảnh.

**Deck số liệu:** [1010 SCIS](../category_1010_business/category_1010_business_scis.md) · [1020 SCIS](../category_1020_business/category_1020_business_scis.md)

---

## Tóm tắt cho leadership (30 giây)

| | 1010 Căn hộ | 1020 Nhà ở |
|---|-------------|------------|
| **Một câu** | Thuê HCM — gem bị chôn, đừng nhầm view với lead | Bán nhà khổng lồ — fix sell 30–50m², boost gem gấp 3× |
| **Case #1 ưu tiên** | A4 Boost 2.3k underexposed | B4 Boost 6.2k underexposed |
| **Case rủi ro #1** | A3 KPI ad_view | B5 Bulk seller + pocket sell |

---

# PHẦN A — Category 1010 (Căn hộ / Chung cư)

## Case A1 · DNA thị trường thuê tại TP.HCM

**Vì sao đây là case quan trọng**  
Nếu copy playbook “bán nhà” sang căn hộ, team sẽ optimize sai sản phẩm và sai địa lý.

**Bằng chứng**
- **{_fmt_int(r10['n_listings'])}** tin catalog; **{r10['pct_let']:.1f}%** là cho thuê (`let`).
- **~74%** supply tập trung TP.HCM — không phải thị trường phân tán toàn quốc.

**Vậy thì sao (insight)**  
1010 là **rental apartment marketplace tại HCM**, không phải “BĐS generic”. Budget UA, SEO, telesales nên **HCM × thuê**, không scatter.

**Hành động đề xuất**
- Landing & CRM theo quận HCM; gói listing “verified căn hộ cho thuê”.
- Không dùng CVR trung bình toàn category làm KPI duy nhất — tách `let` vs `sell`.

![A1 supply](charts/1010/01_supply_adtype.png)
![A1 geo](charts/1010/03_hcm_concentration.png)

---

## Case A2 · Bán căn hộ convert tốt hơn thuê (+4.2 điểm % CVR)

**Vì sao quan trọng**  
Đây là case **đảo ngược 1020** — chứng minh hai category không so sánh chung một ranking `ad_type`.

**Bằng chứng**
- CVR catalog: **let {r10['cvr_let']:.2f}%** vs **sell {r10['cvr_sell']:.2f}%** (+{r10['cvr_sell'] - r10['cvr_let']:.1f}pp).
- Snapshot cho thuê: chỉ **~11%** listing-days có contact — nhiều view, ít chốt liên hệ.

**Vậy thì sao**  
Người **mua** căn hộ chủ động liên hệ nhiều hơn người **thuê** ở cùng catalog. Doanh thu contact có thể đến từ **sell** dù **let** chiếm 3/4 supply.

**Hành động**
- Funnel riêng cho **sell căn hộ**: project × 2PN, completeness furnishing.
- Thuê: giảm friction chat (xem Case A3/A4), không ép phone như nhà bán.

![A2 CVR](charts/1010/02_cvr_baseline_adtype.png)

---

## Case A3 · Bẫy KPI — 50% events là exposure, không phải lead

**Vì sao quan trọng (case chéo cả 1020)**  
Đây là lỗi phân tích phổ biến nhất: nhầm **ad_view** với **explicit contact**.

**Bằng chứng**
- **{r10['session_explicit_pct']:.2f}%** session login có explicit contact (`view_phone`, chat, zalo, sms).
- **50.6%** events là `other_interaction`; **~94%** trong đó là `ad_view`.
- Chỉ **~4%** events là contact thật.

**Vậy thì sao**  
Dashboard “tương tác cao” có thể đang đo **impression feed**, không đo **lead**. PM cần North Star = **explicit/session** hoặc **contact/listing-day**.

**Hành động**
- Tách layer KPI: `pageview` | `other_interaction` | `explicit_contact`.
- Loại `ad_view` khỏi mọi báo cáo “conversion”.

![A3 funnel](charts/1010/04_session_funnel_gap.png)
![A3 layers](charts/1010/05_event_layer_mix.png)

---

## Case A4 · Gem bị chôn — {_fmt_int(r10['hq_underexposed_n'])} tin HQ underexposed ⭐ ƯU TIÊN

**Vì sao đây là case #1 về hành động**  
Đây là **cơ hội recsys rõ nhất**: tin đã chứng minh được contact nhưng feed không cho đủ exposure.

**Bằng chứng**
- **{_fmt_int(r10['hq_underexposed_n'])}** listings: contact/pageview **~{r10['hq_efficiency_pct']:.0f}%**, median exposure **= 2**.
- So với **~2.092** tin oversaturated: efficiency **~61%**, exposure median **8** — “nhiều view, ít contact”.
- What-if boost exposure 2→4: **+{_fmt_int(r10['boost_contact_events_est'])}** contact events (ước tính).

**Vậy thì sao**  
Marketplace đang **bỏ lỡ lead đã sẵn sàng** trên một segment nhỏ nhưng cực kỳ hiệu quả. ROI boost > demote toàn feed.

**Hành động**
1. Feed rank / diversity boost cho `high_quality_underexposed` (file `bridge/10_health_ranked_underexposed_1010.csv`).
2. Cap impression cho oversaturated (exposure cao, contact rank thấp).
3. Không dùng CVR cohort clustering (~100%) để rank — dùng **event efficiency**.

![A4 health](charts/1010/06_health_segment_efficiency.png)
![A4 impact](charts/1010/10_impact_whatif.png)

---

## Case A5 · Pocket 1PN cho thuê — ~289k tin, CVR ~18%

**Vì sao quan trọng**  
Đây là **volume trap**: segment lớn nhất nhưng conversion dưới trung bình category.

**Bằng chứng**
- Tổng hợp slice `bed_bucket=1`, `ad_type=let`: **~289.540** listings, CVR **~18%**.
- What-if CVR 18→20%: **~{_fmt_int(imp10.loc[imp10.scenario.str.contains('Weak'), 'value'].iloc[0])}** listings thêm contact.

**Vậy thì sao**  
Cải thiện 2pp trên pocket này có impact listing-level lớn hơn nhiều micro-campaign nhỏ.

**Hành động**
- Quality gate 1PN: ảnh, giá/m², furnishing; A/B CTA **chat-first** (30% explicit là `contact_chat`).
- Desktop + iOS ~73% events — UX đa thiết bị cho thuê.

![A5 channel](charts/1010/07_contact_channel_mix.png)

---

# PHẦN B — Category 1020 (Nhà ở)

## Case B1 · DNA thị trường bán nhà — {_fmt_int(r20['n_listings'])} tin

**Vì sao quan trọng**  
1020 chiếm phần lớn catalog toàn sàn; sai một điểm % trên sell 30–50m² = hàng trăm nghìn tin.

**Bằng chứng**
- **{_fmt_int(r20['n_listings'])}** listings — **~2.5×** 1010.
- **{100 - r20['pct_let']:.1f}%** sell; cấu trúc `house_type`, `floors`, `width` đầy đủ hơn căn hộ.

**Vậy thì sao**  
Đây là **transaction-heavy house market**, không phải rental-first. Product matching & form field khác hẳn 1010.

**Hành động**
- Completeness bắt buộc trước khi rank: house_type + floors + legal + width.
- Chiến lược **HCM micro-market** (~70% supply).

![B1 supply](charts/1020/01_supply_adtype.png)

---

## Case B2 · Cho thuê nóng, bán lạnh trên snapshot ⭐

**Vì sao quan trọng**  
Đây là case **phân biệt 1020 với mọi category khác**: CVR let > sell nhưng **contact-day sell rất thấp**.

**Bằng chứng**
- CVR catalog: let **{r20['cvr_let']:.2f}%** > sell **{r20['cvr_sell']:.2f}%** (ngược 1010).
- Snapshot: let **~19%** ngày có contact; sell chỉ **~8.6%** trên **~7.5M** listing-days.

**Vậy thì sao**  
**Bán nhà** là nút thắt thật sự của 1020 — không phải cho thuê nhà. Marketing “Nhà ở” phải tách message let vs sell.

**Hành động**
- Playbook **sell**: phone-first CTA, deep-compare UX (**60%** session có deep_compare).
- Playbook **let**: giữ tốc độ contact hiện tại; fix pocket mặt phố thuê ~15% CVR.

![B2 house type CVR](charts/1020/02_cvr_baseline_adtype.png)

---

## Case B3 · Bẫy KPI — cùng pattern 1010

**Bằng chứng**
- **{r20['session_explicit_pct']:.2f}%** session explicit; **50.8%** events `other_interaction` (**96%** ad_view).

**Hành động** — giống Case A3; thống nhất định nghĩa lead toàn platform.

![B3 funnel](charts/compare/03_session_funnel_side_by_side.png)

---

## Case B4 · Gem scale {_fmt_int(r20['hq_underexposed_n'])} tin — boost underexposed ⭐ ƯU TIÊN

**Vì sao case #1 của 1020**  
Cơ hội boost **gấp ~2.7×** 1010; nhiều trong đó là **sell HCMC** (4.853 tin trong segment HQ).

**Bằng chứng**
- **{_fmt_int(r20['hq_underexposed_n'])}** HQ underexposed: **~{r20['hq_efficiency_pct']:.0f}%** contact/pageview, exposure **2**.
- **4.295** oversaturated: **~58%** efficiency, exposure median **13**.
- What-if boost: **+{_fmt_int(r20['boost_contact_events_est'])}** contact events.

**Vậy thì sao**  
1020 là nơi **recsys re-rank** trả dividend lớn nhất trên toàn cặp 1010/1020.

**Hành động**
- Ưu tiên boost **sell** trong `10_health_ranked_underexposed_1020.csv`.
- Demote oversaturated — tiết kiệm **~{_fmt_int(imp20.loc[imp20.scenario.str.contains('Demote'), 'value'].iloc[0])}** impression-equiv.

![B4 health](charts/1020/06_health_segment_efficiency.png)
![B4 compare boost](charts/compare/05_impact_boost_side_by_side.png)

---

## Case B5 · Pocket sell 30–50m² + bulk seller spam

**Vì sao quan trọng**  
Hai rủi ro supply khác nhau: **chất lượng slice** và **hành vi seller**.

**Bằng chứng**
- Sell **30–50m²**: **~310.484** listings, CVR **17.5%** — bucket lớn nhất, conversion thấp.
- Bulk seller cluster: **~102** listings/seller, **0.29** pageviews/listing — cold inventory.

**Vậy thì sao**  
1020 cần **quality + fairness** song song: sửa tin yếu và cap seller phình inventory.

**Hành động**
- Completeness & pricing review cho ngõ/mặt phố 30–50m².
- Cap listing/seller; onboarding seller chất lượng.
- Phone-first (**76%** explicit = `view_phone`); **Android 26%** — mobile-first.

![B5 channel](charts/1020/07_contact_channel_mix.png)
![B5 sessions](charts/1020/09_session_archetypes.png)

---

# PHẦN X — Case chéo (so sánh 1010 vs 1020)

## Case X1 · CVR đảo chiều theo `ad_type`

| Category | Let CVR | Sell CVR | Ai thắng? |
|----------|---------|----------|------------|
| 1010 | {r10['cvr_let']:.2f}% | {r10['cvr_sell']:.2f}% | **Sell** (+{r10['cvr_sell'] - r10['cvr_let']:.1f}pp) |
| 1020 | {r20['cvr_let']:.2f}% | {r20['cvr_sell']:.2f}% | **Let** (+{r20['cvr_let'] - r20['cvr_sell']:.1f}pp) |

**Insight analyst:** Không có “category BĐS convert tốt hơn” — chỉ có **đúng ad_type trong đúng category**.

![X1](charts/compare/02_cvr_baseline_side_by_side.png)

---

## Case X2 · Cùng bệnh health segment, khác quy mô

| Segment | 1010 (n) | 1020 (n) |
|---------|----------|----------|
| HQ underexposed | {_fmt_int(r10['hq_underexposed_n'])} | {_fmt_int(r20['hq_underexposed_n'])} |
| Oversaturated | 2.092 | 4.295 |

**Insight:** Cùng rule percentile → cùng “bệnh” marketplace; 1020 cần **cùng liều thuốc** nhưng **liều gấp đôi**.

![X2](charts/compare/04_health_efficiency_side_by_side.png)

---

## Ma trận ưu tiên (Q2 roadmap gợi ý)

| Ưu tiên | 1010 | 1020 |
|---------|------|------|
| **P0** | A4 Boost underexposed | B4 Boost underexposed |
| **P0** | A3 KPI tách layer | B3 KPI tách layer |
| **P1** | A5 Fix 1PN let | B5 Fix sell 30–50m² + seller cap |
| **P1** | A2 Sell funnel | B2 Sell vs let messaging |

---

### Caveats

- Số từ sample login/clustering 6–8%; dùng cho **prioritization**, không audit tài chính.
- CVR catalog ≠ cohort event (~100% CVR trên tin có event).
"""
    (OUT / "CASES_STORY.md").write_text(body, encoding="utf-8")
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    (OUT / "cases" / "cases_index.json").write_text(
        json.dumps(cases_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_readme() -> None:
    readme = """# Category 1010 × 1020 — Data Story

Thư mục kể chuyện phân tích dữ liệu: **hai category, hai playbook**.

## Đọc theo thứ tự

1. **[CASES_STORY.md](CASES_STORY.md)** — ⭐ **case study** (5 case/category + 2 case chéo) — đọc trước khi pitch
2. **[STORY.md](STORY.md)** — narrative tổng (so sánh 1010 vs 1020)
3. **[chapters/1010_situation_to_impact.md](chapters/1010_situation_to_impact.md)** — deck đầy đủ 1010
4. **[chapters/1020_situation_to_impact.md](chapters/1020_situation_to_impact.md)** — deck đầy đủ 1020
5. **[csv/comparison_1010_vs_1020.csv](csv/comparison_1010_vs_1020.csv)** — bảng số

## Charts

- `charts/compare/` — side-by-side 1010 vs 1020
- `charts/1010/` · `charts/1020/` — copy từ business SCIS decks

## Build lại

```bash
python build_category_1010_1020_story.py
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "title": "Hai thị trường trên một nền tảng — 1010 vs 1020",
        "main": "STORY.md",
        "cases_story": "CASES_STORY.md",
        "chapters": [
            "chapters/1010_situation_to_impact.md",
            "chapters/1020_situation_to_impact.md",
        ],
        "compare_charts": sorted(p.name for p in COMPARE.glob("*.png")),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})
    OUT.mkdir(parents=True, exist_ok=True)
    copy_deck_charts()
    build_comparison_csv()
    chart_compare_supply_mix()
    chart_compare_cvr()
    chart_compare_funnel()
    chart_compare_health()
    chart_compare_impact()
    comp = pd.read_csv(OUT / "csv" / "comparison_1010_vs_1020.csv")
    (OUT / "STORY.md").write_text(
        _story_body(comp.loc[comp.category == 1010].iloc[0], comp.loc[comp.category == 1020].iloc[0]),
        encoding="utf-8",
    )
    write_cases_story(comp)
    write_chapter_copies()
    write_readme()
    write_manifest()
    print(f"Story folder: {OUT}")
    print(f"Cases story: {OUT / 'CASES_STORY.md'}")


if __name__ == "__main__":
    main()
