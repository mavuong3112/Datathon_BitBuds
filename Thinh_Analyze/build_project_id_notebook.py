"""Generate eda_project_id_deep_dive.ipynb with inline charts."""
import json
from pathlib import Path

ROOT = Path(__file__).parent

INTRO = """# Phân tích `project_id` — 5 category BĐS

Tin có **`project_id`** = dự án BĐS (chủ đầu tư). So sánh **có vs không** `project_id` theo category (nhãn UI đúng — Cursor.md).

| Mã | Label | ~% pid |
|----|-------|--------|
| 1010 | Căn hộ / Chung cư | 41% |
| 1020 | Nhà ở | 3% |
| 1030 | VP / Mặt bằng | 6.5% |
| 1040 | Đất | 8.6% |
| 1050 | Phòng trọ | 0% |

**Chart hiển thị ngay dưới mỗi cell** (`%matplotlib inline`). Export CSV/PNG → `outputs/project_id_deep/`.

**Run All** ~10–15 ph (full events scan). Kernel: `../env/bin/python`.
"""

SETUP = """
%matplotlib inline
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
plt.ioff()

import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

WORKDIR = Path.cwd().resolve()
if not (WORKDIR / "dim_listing").exists():
    WORKDIR = WORKDIR.parent
sys.path.insert(0, str(WORKDIR / "Thinh_Analyze"))

import run_project_id_deep_dive as pid
pid.SHOW_INLINE = True  # display(fig) dưới cell — không popup

from run_project_id_deep_dive import (
    OUT_DIR, CAT_META, CATEGORIES, EDA_RULES,
    init_db, section_0, section_1, section_2, section_3,
    section_4, section_5a, section_5b, section_6, section_7, section_interest,
    plot_dim_categorical,
)

def show_df(df, title):
    display(Markdown(f"**{title}**"))
    display(df)

print("SHOW_INLINE =", pid.SHOW_INLINE)
print("OUT_DIR =", OUT_DIR)
"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in s.strip().split("\n")]}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [l + "\n" for l in s.strip().split("\n")],
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(INTRO),
    code(SETUP),
    md("## §0 — Time coverage & QA"),
    code("""
init_db()
tc = section_0()
show_df(tc, "Time coverage dim vs facts")
"""),
    md("## §1 — Tổng quan supply (6-panel dashboard)\n\nChart xuất hiện **ngay bên dưới**."),
    code("""
section_1()
show_df(pd.read_csv(OUT_DIR / "01_overview_by_category.csv"), "Overview")
"""),
    md("## §2 — dim_listing: có vs không project_id\n\nMỗi field categorical (role matrix) → **chart ngay bên dưới** khi chạy cell."),
    code("""
section_2()  # vẽ fig_02_{cat}_{field}.png + hiển thị inline (SHOW_INLINE=True)
for cat in [1010, 1020, 1030, 1040]:
    p = OUT_DIR / f"02_{cat}_dim_compare.csv"
    if p.exists():
        df = pd.read_csv(p)
        show_df(
            df[df.field.isin(["ad_type", "seller_type", "price_bucket"])].head(20),
            f"{CAT_META[cat]} — bảng tóm tắt",
        )
"""),
    md("## §2b — 1050 (không có project_id)"),
    code("""
display(Markdown("**1050 — Phòng trọ:** 0% `project_id` → không so cohort pid."))
show_df(pd.read_csv(OUT_DIR / "02_1050_dim_compare_note.csv"), "Note")
"""),
    md("## §3 — Thực thể project_id & cross-category"),
    code("""
section_3()
show_df(pd.read_csv(OUT_DIR / "01_pid_cross_category.csv"), "Pid × số category")
show_df(pd.read_csv(OUT_DIR / "03_cross_category_pids.csv").head(12), "Top cross-category pids")
"""),
    md("## §4 — Snapshot performance (`in_eda_window`)"),
    code("""
snap = section_4()
show_df(snap, "Views / contacts — has_project vs not")
"""),
    md("## §5A — CVR full scan (bảng + heatmap)"),
    code("""
section_5a()
show_df(pd.read_csv(OUT_DIR / "05_cvr_in_eda_window.csv"), "CVR in_eda_window")
show_df(pd.read_csv(OUT_DIR / "05_cvr_full_category_has_project_adtype.csv"), "CVR × ad_type")
"""),
    md(
        "## §8 — Quan tâm khách hàng: có vs không `project_id` theo category\n\n"
        "So sánh tin **có vs không** `project_id` (1010–1040): CVR positive, explicit lead (login), "
        "snapshot views/contacts. Chỉ tin `posted in_eda_window` (Cursor.md). **Chạy §4 trước** (biến `snap`)."
    ),
    code("""
interest = section_interest(snap)
show_df(
    interest[[
        "label", "has_project_label", "listings", "cvr_positive_pct",
        "cvr_explicit_pct", "avg_views", "avg_contacts", "contact_per_view_pct",
        "winner_cvr", "lift_cvr_positive_pct",
    ]],
    "Quan tâm KH — có vs không project_id",
)
"""),
    md("## §5B — Events SYSTEM 10% (explicit channel)"),
    code("""
ch, dwell = section_5b()
show_df(ch, "Explicit channel (sample 10%)")
show_df(dwell, "Dwell median pageview (sec)")
"""),
    md("## §6 — Post-contact interactions (login)"),
    code("""
inter = section_6()
show_df(inter, "Interactions by has_project")
"""),
    md("## §7 — Scorecard & SUMMARY"),
    code("""
score = section_7(snap)
show_df(score, "Scorecard — modeling notes")
display(Markdown((OUT_DIR / "SUMMARY.md").read_text()))
"""),
    md("## Phụ lục — xem lại PNG đã export (nếu cần)"),
    code("""
from IPython.display import Image
for p in sorted(OUT_DIR.glob("fig_*.png")):
    display(Markdown(f"**`{p.name}`**"))
    display(Image(filename=str(p)))
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (env)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}
(ROOT / "eda_project_id_deep_dive.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("Wrote", ROOT / "eda_project_id_deep_dive.ipynb")
