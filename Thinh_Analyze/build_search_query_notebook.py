"""Generate eda_search_query_province_nlp.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).parent

INTRO = """# NLP search query theo tỉnh + cách người dùng gõ query

Ước tính **volume search** (`fact_user_events.query` trên pageview) theo tỉnh/thành:

| Phương án | Định nghĩa |
|-----------|------------|
| **Method A** | Chỉ tỉnh trích từ **text query** (rule-based lexicon) |
| **Method B** | Method A; nếu không có tỉnh trong query → fallback `city_name` của tin |

**Chạy nhanh (export CSV/PNG):**
`env/bin/python Thinh_Analyze/run_search_query_province_nlp.py --query-sample-frac 0.15 --choropleth`

**Output:** `Thinh_Analyze/outputs/search_query_province_nlp/`

### Đọc nhanh (sau khi chạy pipeline)

| Câu hỏi | Gợi ý đọc |
|---------|-----------|
| Có phải chỉ ~10 tỉnh được gõ tên? | **Không** — xem `SUMMARY.md` § Method A vs B |
| Bao nhiêu % query có tên tỉnh trong text? | ~10% (Method A); ~90% là `khong_ro_tinh` |
| Phân bổ thực tế theo tỉnh? | **Method B** + choropleth `maps/` |
"""

SETUP = """
%matplotlib inline
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

WORKDIR = Path.cwd().resolve()
if not (WORKDIR / "dim_listing").exists() and (WORKDIR.parent / "dim_listing").exists():
    DATA_ROOT = WORKDIR.parent
else:
    DATA_ROOT = WORKDIR
sys.path.insert(0, str(DATA_ROOT))

from Thinh_Analyze.search_query_province_nlp_lib import OUT_DIR, CAT_META, CATEGORIES
import Thinh_Analyze.run_search_query_province_nlp as pipeline

OUT = OUT_DIR
print("DATA_ROOT =", DATA_ROOT)
print("OUT =", OUT)
"""


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in s.strip().split("\n")]}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in s.strip().split("\n")],
    }


cells = [
    md(INTRO),
    code(SETUP),
    md("## Chạy pipeline (hoặc đọc CSV đã export)"),
    code(
        """
# Đặt RUN_PIPELINE=True để tái chạy (~1–2 ph với sample 15%)
RUN_PIPELINE = False
if RUN_PIPELINE:
    import sys
    _argv = sys.argv
    sys.argv = [
        "run_search_query_province_nlp.py",
        "--query-sample-frac", "0.15",
        "--choropleth",
    ]
    pipeline.main()
    sys.argv = _argv
else:
    display(Markdown("Đọc output có sẵn. Bật `RUN_PIPELINE=True` để tái tạo."))
"""
    ),
    md("## 1. Coverage"),
    code(
        """
cov = pd.read_csv(OUT / "00_coverage_overall.csv")
dims = pd.read_csv(OUT / "00_coverage_by_dims.csv")
display(cov)
display(dims.head(12))
"""
    ),
    md(
        """## 2. Thống kê tỉnh — Method A vs B (markdown)

Bảng dưới trả lời: *có bao nhiêu tỉnh được match*, *% query không ghi tỉnh*, *taxonomy theo category*.
"""
    ),
    code(
        """
stats = pd.read_csv(OUT / "10_province_coverage_stats.csv")
display(stats)
summary_md = OUT / "SUMMARY.md"
if summary_md.exists():
    display(Markdown(summary_md.read_text(encoding="utf-8")))
"""
    ),
    md("## 2b. Volume theo tỉnh — bảng & chart"),
    code(
        """
vol_a = pd.read_csv(OUT / "01_volume_by_province_method_a.csv")
vol_b = pd.read_csv(OUT / "02_volume_by_province_method_b.csv")
delta = pd.read_csv(OUT / "03_delta_a_vs_b.csv")
for title, df in [
    ("Method A (NLP text)", vol_a.groupby("province")["n_searches_est"].sum().nlargest(12)),
    ("Method B (NLP + fallback)", vol_b.groupby("province")["n_searches_est"].sum().nlargest(12)),
]:
    display(Markdown(f"**{title}**"))
    display(df.reset_index())
display(Markdown("**Delta B − A (national)**"))
display(delta.head(12))
"""
    ),
    code(
        """
for name in [
    "fig_top_provinces_method_a.png",
    "fig_top_provinces_method_b.png",
    "fig_region_stack_method_b.png",
]:
    p = OUT / name
    if p.exists():
        display(Markdown(f"### {name}"))
        display(Image(filename=str(p)))
"""
    ),
    md("## 3. Validation NLP vs `city_name` tin"),
    code(
        """
val = pd.read_csv(OUT / "08_validation_summary.csv")
mis = pd.read_csv(OUT / "08_validation_mislabels.csv")
display(val)
display(mis.head(15))
p = OUT / "fig_validation_heatmap.png"
if p.exists():
    display(Image(filename=str(p)))
"""
    ),
    md("## 4. Cách người dùng query (taxonomy & n-grams)"),
    code(
        """
tax = pd.read_csv(OUT / "04_query_taxonomy_counts.csv")
ngrams = pd.read_csv(OUT / "06_bigrams_trigrams.csv")
refine = pd.read_csv(OUT / "07_refinement_proxy.csv")
display(tax.sort_values(["category", "n"], ascending=[True, False]).groupby("category").head(8))
display(ngrams.head(20))
display(refine)
p = OUT / "fig_taxonomy_donut.png"
if p.exists():
    display(Image(filename=str(p)))
"""
    ),
    md("## 5. Top query theo tỉnh (method B) — ví dụ category 1050"),
    code(
        """
cat = 1050
path = OUT / f"05_top_queries_by_province_{cat}.csv"
if path.exists():
  top = pd.read_csv(path)
  display(Markdown(f"**{CAT_META[cat]} ({cat})**"))
  for prov, g in top.groupby("province"):
      display(Markdown(f"#### {prov}"))
      display(g.nlargest(8, "n")[["query", "n"]])
      break  # first province demo
  display(top.head(25))
"""
    ),
    md("## 6. Choropleth search volume (method B)"),
    code(
        """
from IPython.display import Image
maps = sorted((OUT / "maps").glob("fig_search_volume_choropleth_*.png"))
for p in maps[:3]:
    display(Markdown(f"### {p.name}"))
    display(Image(filename=str(p)))
if len(maps) > 3:
    display(Markdown(f"_… và {len(maps)-3} map khác trong `{OUT / 'maps'}`_"))
"""
    ),
    md("## 7. Search vs explicit events (exploratory)"),
    code(
        """
gap_path = OUT / "09_search_vs_explicit_gap.csv"
if gap_path.exists():
    gap = pd.read_csv(gap_path)
    display(Markdown("**Search volume vs explicit events (exploratory)**"))
    display(gap.nlargest(10, "n_searches_est")[["province", "n_searches_est", "n_explicit_events"]])
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out_path = ROOT / "eda_search_query_province_nlp.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out_path)
