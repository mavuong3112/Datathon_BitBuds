"""Execute eda_market_analysis.ipynb cells in-process (smoke test)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).parent
NB = ROOT / "eda_market_analysis.ipynb"


def main() -> int:
    if not NB.exists():
        print("Missing notebook — run build_eda_notebook.py first", file=sys.stderr)
        return 1

    nb = json.loads(NB.read_text(encoding="utf-8"))
    g: dict = {"__name__": "__main__"}
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"Smoke: {len(code_cells)} code cells")

    magic = re.compile(r"^\s*%.*$", re.MULTILINE)

    for i, cell in enumerate(code_cells, 1):
        src = magic.sub("", "".join(cell["source"])).strip()
        print(f"\n--- Cell {i}/{len(code_cells)} ({len(src.splitlines())} lines) ---")
        t0 = time.perf_counter()
        try:
            exec(compile(src, f"<cell_{i}>", "exec"), g, g)
        except Exception:
            traceback.print_exc()
            return 1
        print(f"OK in {time.perf_counter() - t0:.1f}s")

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
