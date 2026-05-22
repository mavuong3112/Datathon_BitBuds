%matplotlib inline
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

WORKDIR = Path.cwd().resolve()
if not (WORKDIR / "dim_listing").exists() and (WORKDIR.parent / "dim_listing").exists():
    DATA_ROOT = WORKDIR.parent
else:
    DATA_ROOT = WORKDIR
if not (DATA_ROOT / "dim_listing").exists():
    raise FileNotFoundError(f"Thiếu dim_listing trong {DATA_ROOT}")

DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")
SNAP_GLOB = str(DATA_ROOT / "fact_listing_snapshot" / "*.parquet")
INTER_GLOB = str(DATA_ROOT / "fact_post_contact_interactions" / "*.parquet")
FILTERED_FILE = DATA_ROOT / "Thinh_Analyze" / "filtered_events.parquet"

SAMPLE_PCT = 5
SAMPLE_PCT_QA = 10
USE_FILTERED = FILTERED_FILE.exists()
EXPORT_CSV = True
CAT_1020 = 1020

EXPLICIT_TYPES = ("view_phone", "contact_chat", "contact_zalo", "contact_sms")
POSITIVE_TYPES = EXPLICIT_TYPES + ("other_interaction",)
EXPLICIT_SQL = ", ".join(repr(x) for x in EXPLICIT_TYPES)
POSITIVE_SQL = ", ".join(repr(x) for x in POSITIVE_TYPES)
LOGIN_WHERE = "is_login = 'login'"
CAT_IN = "1010, 1020, 1030, 1040, 1050"

CAT_META = {
    1010: "1010 — Căn hộ / Chung cư",
    1020: "1020 — Nhà ở",
    1030: "1030 — Văn phòng / Mặt bằng",
    1040: "1040 — Đất",
    1050: "1050 — Phòng trọ",
}
CATEGORIES = tuple(CAT_META)

OUT = {
    "concentration": DATA_ROOT / "Thinh_Analyze" / "outputs" / "contact_concentration",
    "duplicate": DATA_ROOT / "Thinh_Analyze" / "outputs" / "dim_duplicate_after_timecut",
    "demand": DATA_ROOT / "Thinh_Analyze" / "outputs" / "demand_side_1020",
    "deep1020": DATA_ROOT / "Thinh_Analyze" / "outputs" / "demand_side_1020" / "1020_deep_dive",
    "nonlogin": DATA_ROOT / "Thinh_Analyze" / "outputs" / "nonlogin_chat_qa",
}
for p in OUT.values():
    p.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=4")


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def top_pct_share(values: np.ndarray, top_pct: float) -> float:
    x = np.sort(values)[::-1]
    total = float(x.sum())
    if total == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(x) * top_pct / 100.0)))
    return 100.0 * float(x[:k].sum()) / total


def concentration_row(entity: str, counts: np.ndarray) -> dict:
    cl = gini(counts)
    return {
        "entity": entity,
        "n_with_contact": int(len(counts)),
        "total_contacts": int(counts.sum()),
        "gini": round(cl, 4),
        "top_10pct_share": round(top_pct_share(counts, 10), 2),
    }


def show_df(df: pd.DataFrame, title: str):
    display(Markdown(f"**{title}**"))
    display(df)


def save_fig(path: Path, dpi: int = 120):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        get_ipython()  # type: ignore[name-defined]
        plt.show()
    except NameError:
        pass
    plt.close()
    print("Saved", path)


EDA_MIN, EDA_MAX = con.execute(
    f"SELECT MIN(date), MAX(date) FROM read_parquet('{EVENTS_GLOB}')"
).fetchone()
print("DATA_ROOT =", DATA_ROOT)
print("EDA window:", EDA_MIN, "→", EDA_MAX)
print(f"SAMPLE_PCT={SAMPLE_PCT}% | SAMPLE_PCT_QA={SAMPLE_PCT_QA}% | USE_FILTERED={USE_FILTERED}")
