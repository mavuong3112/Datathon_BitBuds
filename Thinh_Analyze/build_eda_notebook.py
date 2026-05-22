"""Generate Thinh_Analyze/eda_market_analysis.ipynb from notebook_cells/."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
CELLS_DIR = ROOT / "notebook_cells"


def md(text: str) -> dict:
    lines = text.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(text: str) -> dict:
    lines = text.strip().split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def load_stem(stem: str) -> str:
    path = CELLS_DIR / f"{stem}.py"
    return path.read_text(encoding="utf-8")


def load_md(stem: str) -> str:
    path = CELLS_DIR / f"{stem}.md"
    return path.read_text(encoding="utf-8")


cells: list[dict] = []

cells.append(md(load_md("00_intro")))
cells.append(code(load_stem("00_setup")))

for part_md, part_py in [
    ("01_part1", "01_part1_contact"),
    ("02_part2", "02_part2_duplicate"),
    ("03_part3", "03_part3_demand"),
    ("04_part4", "04_part4_1020"),
    ("05_part5", "05_part5_nonlogin"),
]:
    cells.append(md(load_md(part_md)))
    cells.append(code(load_stem(part_py)))

cells.append(md(load_md("06_synthesis")))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out = ROOT / "eda_market_analysis.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out, "cells:", len(cells))
