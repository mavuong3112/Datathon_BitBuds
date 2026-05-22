"""Build 7 training notebooks for Datathon 2026 (Cho Tot BDS Recommender).

Run: python build_training_notebooks.py
Generates 01_prepare_data.ipynb ... 07_predict_and_submit.ipynb.

LOCAL MODE: notebooks run on a local Python kernel and read parquet files from
`Datathon_Data/{train,test}/` on disk (all 5 categories). Cache is written to
`d:/Datathon_2/cache_drive/`.
"""
from __future__ import annotations

import nbformat as nbf


# ============================================================
# Shared cell builders
# ============================================================

def _new_cells() -> list:
    return []


def md(cells: list, src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(cells: list, src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


def setup_cell_drive_mount() -> str:
    """Local setup: define paths and add training/utils to sys.path."""
    return r"""
# === Local setup: paths + add training/utils to sys.path ===
import os, sys

PROJECT_DIR  = r"d:/Datathon_2"
TRAINING_DIR = os.path.join(PROJECT_DIR, "training")
CACHE_DIR    = os.path.join(PROJECT_DIR, "cache_drive")
DATA_DIR     = os.path.join(PROJECT_DIR, "Datathon_Data")  # contains train/ and test/
os.makedirs(CACHE_DIR, exist_ok=True)

if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

print("PROJECT_DIR:", PROJECT_DIR)
print("DATA_DIR   :", DATA_DIR)
print("CACHE_DIR  :", CACHE_DIR)
print("utils available:", os.path.isdir(os.path.join(TRAINING_DIR, "utils")))
"""


def pip_install_cell() -> str:
    """No-op on local kernel; assumes deps already installed.
    Uncomment the block below to install on a fresh environment."""
    return r"""
# Local kernel: assume deps already installed.
# To install run once:
#   pip install pyarrow pandas numpy scipy lightgbm scikit-learn tqdm
print("Skipping pip install (local kernel).")
"""


def constants_cell() -> str:
    return r"""
TRAIN_DATE_START = "2025-11-09"
TRAIN_DATE_END = "2026-04-09"
VALID_START = "2026-03-13"

POSITIVE_EVENT_TYPES = frozenset({
    "view_phone", "contact_chat", "other_interaction", "contact_zalo", "contact_sms",
})
HIGH_INTENT_EVENTS = frozenset({"view_phone", "contact_chat", "contact_zalo", "contact_sms"})

INTENT_WEIGHT = {
    "view_phone": 3.0, "contact_chat": 2.0,
    "contact_zalo": 2.0, "contact_sms": 2.0,
    "other_interaction": 1.0,
}

# Local data paths (relative to DATA_DIR defined in the setup cell)
TRAIN_PATH = os.path.join(DATA_DIR, "train") + os.sep
TEST_PATH  = os.path.join(DATA_DIR, "test", "test") + os.sep

print("Constants loaded. TRAIN_END:", TRAIN_DATE_END, "VALID_START:", VALID_START)
print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH :", TEST_PATH)
"""


def assert_no_gcs_cell() -> str:
    """Local mode: nothing to guard. Kept as no-op for backwards-compat."""
    return r"""
# Local mode: no GCS egress guardrail needed.
print("Local kernel: reading from CACHE_DIR.")
"""


# ============================================================
# Notebook 01 — prepare_data
# ============================================================

def build_nb01() -> list:
    cells = _new_cells()
    md(cells, r"""
# 01 — Prepare Data (local files → cache)

**Mục đích:** đọc `Datathon_Data/train/` & `Datathon_Data/test/test/` từ ổ đĩa local,
cache 6 file parquet xuống `cache_drive/`. Mọi notebook sau train từ cache.

**Setup trước khi chạy:**
1. Đảm bảo dữ liệu nằm ở `d:/Datathon_2/Datathon_Data/{train,test}/`.
2. Kernel Python local đã cài: `pyarrow pandas numpy scipy lightgbm scikit-learn tqdm`.

Output files trong `d:/Datathon_2/cache_drive/`:
- `dim_listing.parquet` (FULL, ~50MB)
- `snapshot_60d.parquet` (last 60d)
- `pci_full.parquet`
- `events_positive.parquet` (5 event_type, date ≤ 2026-04-09, ~96M row)
- `events_pageview_30d.parquet`
- `test_users.parquet`
""")

    code(cells, pip_install_cell())
    code(cells, setup_cell_drive_mount())
    code(cells, constants_cell())

    code(cells, r"""
# Local kernel: no auth needed.
import os
assert os.path.isdir(TRAIN_PATH), f"Missing train dir: {TRAIN_PATH}"
assert os.path.isfile(os.path.join(TEST_PATH, "test_users.parquet")), \
    f"Missing test_users.parquet under {TEST_PATH}"
print("Local data OK.")
""")

    code(cells, r"""
import time
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

t_start = time.time()

dim_dset  = ds.dataset(os.path.join(TRAIN_PATH, "dim_listing"),                    format="parquet")
snap_dset = ds.dataset(os.path.join(TRAIN_PATH, "fact_listing_snapshot"),          format="parquet")
pci_dset  = ds.dataset(os.path.join(TRAIN_PATH, "fact_post_contact_interactions"), format="parquet")
fue_dset  = ds.dataset(os.path.join(TRAIN_PATH, "fact_user_events"),               format="parquet")

train_end_d = pa.scalar(pd.Timestamp(TRAIN_DATE_END).date(), type=pa.date32())
snap_start_d = pa.scalar(pd.Timestamp("2026-02-09").date(), type=pa.date32())
pv_start_d   = pa.scalar(pd.Timestamp("2026-03-10").date(), type=pa.date32())

print(f"dim files     : {len(dim_dset.files):,}")
print(f"snap files    : {len(snap_dset.files):,}")
print(f"pci files     : {len(pci_dset.files):,}")
print(f"events files  : {len(fue_dset.files):,}")
""")

    code(cells, r"""
# ---- 0) Print schema for each dataset (verify before reading) ----
def _safe_cols(dset, wanted):
    have = set(dset.schema.names)
    keep = [c for c in wanted if c in have]
    miss = [c for c in wanted if c not in have]
    if miss:
        print(f"  [warn] missing cols dropped: {miss}")
    return keep

for name, dset in [("dim_listing", dim_dset), ("fact_listing_snapshot", snap_dset),
                   ("fact_post_contact_interactions", pci_dset),
                   ("fact_user_events", fue_dset)]:
    print(f"\n=== {name} schema ===")
    print(dset.schema)
""")

    code(cells, r"""
# ---- 1) dim_listing FULL ----
DIM_COLS_WANT = [
    "item_id", "seller_id", "category", "title",
    "seller_type", "ad_type", "ad_status",
    "area_sqm", "bedrooms", "bathrooms", "floors", "width_m",
    "direction", "legal_status", "house_type", "furnishing",
    "city_name", "district_name", "ward_name", "project_id",
    "price_bucket", "images_count", "posted_date", "expected_expired_date",
]
DIM_COLS = _safe_cols(dim_dset, DIM_COLS_WANT)
t0 = time.time()
dim_tbl = dim_dset.to_table(columns=DIM_COLS)
df_dim = dim_tbl.to_pandas()
print(f"dim_listing: {df_dim.shape} | {time.time()-t0:.1f}s")
print(f"posted_date: {df_dim['posted_date'].min()} -> {df_dim['posted_date'].max()}")
print(df_dim["ad_status"].value_counts(dropna=False).head(10))

df_dim.to_parquet(f"{CACHE_DIR}/dim_listing.parquet", index=False)
del dim_tbl, df_dim
""")

    code(cells, r"""
# ---- 2) fact_listing_snapshot last 60d (streaming write, 8GB-safe) ----
import pyarrow.parquet as pq
flt_snap = (pc.field("date") >= snap_start_d) & (pc.field("date") <= train_end_d)
SNAP_COLS_WANT = ["item_id", "date", "views_24h", "contacts_24h",
                  "listing_age_days", "category"]
SNAP_COLS = _safe_cols(snap_dset, SNAP_COLS_WANT)
t0 = time.time()
out_snap = f"{CACHE_DIR}/snapshot_60d.parquet"
writer = None
n_rows = 0
for batch in snap_dset.scanner(columns=SNAP_COLS, filter=flt_snap,
                                batch_size=500_000).to_batches():
    if writer is None:
        writer = pq.ParquetWriter(out_snap, batch.schema, compression="snappy")
    writer.write_batch(batch)
    n_rows += batch.num_rows
if writer is not None:
    writer.close()
print(f"snapshot_60d: {n_rows:,} rows | {time.time()-t0:.1f}s -> {out_snap}")
""")

    code(cells, r"""
# ---- 3) fact_post_contact_interactions FULL (date <= TRAIN_END, streaming) ----
flt_pci = pc.field("date") <= train_end_d
PCI_COLS_WANT = ["user_id", "item_id", "date", "category", "purchased"]
PCI_COLS = _safe_cols(pci_dset, PCI_COLS_WANT)
t0 = time.time()
out_pci = f"{CACHE_DIR}/pci_full.parquet"
writer = None
n_rows = 0
for batch in pci_dset.scanner(columns=PCI_COLS, filter=flt_pci,
                               batch_size=500_000).to_batches():
    if writer is None:
        writer = pq.ParquetWriter(out_pci, batch.schema, compression="snappy")
    writer.write_batch(batch)
    n_rows += batch.num_rows
if writer is not None:
    writer.close()
print(f"pci_full: {n_rows:,} rows | {time.time()-t0:.1f}s -> {out_pci}")
""")

    code(cells, r"""
# ---- 4) fact_user_events POSITIVE only (5 event_type, date <= TRAIN_END) ----
pos_arr = pa.array(sorted(POSITIVE_EVENT_TYPES))
flt_pos = (
    (pc.field("date") <= train_end_d)
    & pc.is_in(pc.field("event_type"), value_set=pos_arr)
)
POS_COLS_WANT = ["user_id", "session_id", "item_id", "event_type", "event_ts",
                 "date", "category", "surface", "device", "dwell_time_sec",
                 "is_login", "city_name"]
POS_COLS = _safe_cols(fue_dset, POS_COLS_WANT)
t0 = time.time()
import pyarrow.parquet as pq
out_path_pos = f"{CACHE_DIR}/events_positive.parquet"
writer = None
n_rows = 0
for batch in fue_dset.scanner(
    columns=POS_COLS, filter=flt_pos, batch_size=400_000
).to_batches():
    if writer is None:
        writer = pq.ParquetWriter(out_path_pos, batch.schema, compression="snappy")
    writer.write_batch(batch)
    n_rows += batch.num_rows
if writer is not None:
    writer.close()
print(f"events_positive: {n_rows:,} rows | {time.time()-t0:.1f}s -> {out_path_pos}")
""")

    code(cells, r"""
# ---- 5) fact_user_events PAGEVIEW last 30d ----
flt_pv = (
    (pc.field("date") >= pv_start_d)
    & (pc.field("date") <= train_end_d)
    & (pc.field("event_type") == pa.scalar("pageview"))
)
PV_COLS_WANT = ["user_id", "item_id", "event_ts", "date",
                "dwell_time_sec", "is_login"]
PV_COLS = _safe_cols(fue_dset, PV_COLS_WANT)
t0 = time.time()
out_path_pv = f"{CACHE_DIR}/events_pageview_30d.parquet"
writer = None
n_rows = 0
for batch in fue_dset.scanner(
    columns=PV_COLS, filter=flt_pv, batch_size=400_000
).to_batches():
    if writer is None:
        writer = pq.ParquetWriter(out_path_pv, batch.schema, compression="snappy")
    writer.write_batch(batch)
    n_rows += batch.num_rows
if writer is not None:
    writer.close()
print(f"events_pageview_30d: {n_rows:,} rows | {time.time()-t0:.1f}s")
""")

    code(cells, r"""
# ---- 6) test_users ----
df_test = pd.read_parquet(os.path.join(TEST_PATH, "test_users.parquet"))
print(f"test_users: {len(df_test):,}")
df_test.to_parquet(f"{CACHE_DIR}/test_users.parquet", index=False)
print(f"\nTOTAL ELAPSED: {time.time()-t_start:.0f}s")
""")

    code(cells, r"""
# ---- Verification: confirm all 6 cache files exist + sane ----
import os
for name in ["dim_listing.parquet", "snapshot_60d.parquet", "pci_full.parquet",
             "events_positive.parquet", "events_pageview_30d.parquet",
             "test_users.parquet"]:
    p = os.path.join(CACHE_DIR, name)
    sz = os.path.getsize(p) / (1024**2)
    print(f"  {name:35s} {sz:8.1f} MB")

df_t = pd.read_parquet(f"{CACHE_DIR}/test_users.parquet")
assert len(df_t) == 161_568, f"test_users mismatch: {len(df_t)}"
df_p = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet", columns=["event_type", "date"])
assert df_p["event_type"].isin(POSITIVE_EVENT_TYPES).all()
assert pd.to_datetime(df_p["date"]).max().date() <= pd.Timestamp(TRAIN_DATE_END).date()
print("\nAll assertions passed. Cache ready for notebooks 02-07.")
""")
    return cells


# ============================================================
# Notebook 02 — enrich_dim
# ============================================================

def build_nb02() -> list:
    cells = _new_cells()
    md(cells, r"""
# 02 — Enrich dim_listing (build tier classification)

**Input:** `cache_drive/dim_listing.parquet`, `events_positive.parquet`, `snapshot_60d.parquet`.
**Output:** `cache_drive/dim_listing_enriched.parquet` với:
- aggregates: `first_evt_date`, `last_evt_date`, `n_pos_train`, `n_unique_users`
- snapshot agg: `last_snap_date`, `n_snap_days`, `views_total`, `contacts_total`
- flags: `is_pre_fact_window`, `has_any_event_train`, `is_dead`
- `recency_evt_days`, `age_at_train_end`, `tier ∈ {A, B, C, D}`

Đọc cache, ghi cache. Không đụng tới `Datathon_Data/`.
""")
    code(cells, pip_install_cell())
    code(cells, setup_cell_drive_mount())
    code(cells, constants_cell())
    code(cells, assert_no_gcs_cell())

    code(cells, r"""
import time
import numpy as np
import pandas as pd

t0 = time.time()
dim = pd.read_parquet(f"{CACHE_DIR}/dim_listing.parquet")
print(f"dim: {dim.shape} | {time.time()-t0:.1f}s")
dim["posted_date"] = pd.to_datetime(dim["posted_date"], errors="coerce")
""")

    code(cells, r"""
# --- aggregate events_positive per item (pyarrow native group_by) ---
import pyarrow as pa, pyarrow.dataset as ds
t0 = time.time()
_evt_dset = ds.dataset(f"{CACHE_DIR}/events_positive.parquet", format="parquet")
_evt_tbl = _evt_dset.to_table(columns=["item_id", "user_id", "date"])
_g = _evt_tbl.group_by("item_id").aggregate([
    ("user_id", "count"),
    ("user_id", "count_distinct"),
    ("date",    "min"),
    ("date",    "max"),
])
evt_agg = _g.rename_columns([
    "item_id", "n_pos_train", "n_unique_users", "first_evt_date", "last_evt_date",
]).to_pandas()
evt_agg["first_evt_date"] = pd.to_datetime(evt_agg["first_evt_date"], errors="coerce")
evt_agg["last_evt_date"]  = pd.to_datetime(evt_agg["last_evt_date"],  errors="coerce")
print(f"evt_agg: {evt_agg.shape} | {time.time()-t0:.1f}s")
del _evt_tbl, _g
""")

    code(cells, r"""
# --- aggregate snapshot (pyarrow native group_by) ---
t0 = time.time()
_snap_dset = ds.dataset(f"{CACHE_DIR}/snapshot_60d.parquet", format="parquet")
_snap_tbl = _snap_dset.to_table(columns=["item_id", "date", "views_24h", "contacts_24h"])
_g = _snap_tbl.group_by("item_id").aggregate([
    ("date",         "max"),
    ("date",         "count_distinct"),
    ("views_24h",    "sum"),
    ("contacts_24h", "sum"),
])
snap_agg = _g.rename_columns([
    "item_id", "last_snap_date", "n_snap_days", "views_total", "contacts_total",
]).to_pandas()
snap_agg["last_snap_date"] = pd.to_datetime(snap_agg["last_snap_date"], errors="coerce")
print(f"snap_agg: {snap_agg.shape} | {time.time()-t0:.1f}s")
del _snap_tbl, _g
""")

    code(cells, r"""
# --- merge + flags + tier ---
TRAIN_END_TS = pd.Timestamp(TRAIN_DATE_END)
FACT_START_TS = pd.Timestamp("2025-11-09")

enr = dim.merge(evt_agg, on="item_id", how="left")
enr = enr.merge(snap_agg, on="item_id", how="left")

for c in ("n_pos_train", "n_unique_users", "n_snap_days",
          "views_total", "contacts_total"):
    enr[c] = enr[c].fillna(0).astype("int64")

enr["age_at_train_end"] = (TRAIN_END_TS - enr["posted_date"]).dt.days
enr["is_pre_fact_window"] = (enr["posted_date"] < FACT_START_TS).astype("int8")
enr["has_any_event_train"] = (enr["n_pos_train"] > 0).astype("int8")
enr["last_evt_date"] = pd.to_datetime(enr["last_evt_date"], errors="coerce")
enr["recency_evt_days"] = (TRAIN_END_TS - enr["last_evt_date"]).dt.days

if "ad_status" in enr.columns:
    ad_status_norm = enr["ad_status"].astype(str).str.lower()
    is_deleted = (ad_status_norm == "deleted")
else:
    is_deleted = pd.Series(False, index=enr.index)
enr["is_dead"] = (
    (~enr["has_any_event_train"].astype(bool) & (enr["contacts_total"] == 0))
    | is_deleted
).astype("int8")

def _tier(row):
    if row["is_dead"]:
        return "D"
    if row["has_any_event_train"]:
        if pd.notna(row["recency_evt_days"]) and row["recency_evt_days"] <= 30:
            return "A"
        return "B"
    return "C"

enr["tier"] = enr.apply(_tier, axis=1)
print(enr["tier"].value_counts(dropna=False))
print(f"\nA share: {(enr['tier']=='A').mean()*100:.1f}%")
print(f"D share: {(enr['tier']=='D').mean()*100:.1f}%")
""")

    code(cells, r"""
out = f"{CACHE_DIR}/dim_listing_enriched.parquet"
enr.to_parquet(out, index=False)
print(f"Wrote {out} | {enr.shape}")
""")
    return cells


# ============================================================
# Notebook 03 — validation_split
# ============================================================

def build_nb03() -> list:
    cells = _new_cells()
    md(cells, r"""
# 03 — Validation split (28-day holdout) + popularity baseline

**Split:**
- `train_pos`: events_positive với `date < VALID_START` (2026-03-13)
- `valid_pos`: events_positive với `date ∈ [VALID_START, TRAIN_END]`
- `valid_users`: user có trong cả 2 (mô phỏng test_users là user có history)

**Metric:** `recall_at_k`, `ndcg_at_k` từ `utils.metrics`.

**Baseline:** top-10 item theo `contacts_total` (tier A only).
Mục tiêu sau ranker: Recall@10 ≥ 2x baseline.
""")
    code(cells, pip_install_cell())
    code(cells, setup_cell_drive_mount())
    code(cells, constants_cell())
    code(cells, assert_no_gcs_cell())

    code(cells, r"""
import time
import pandas as pd
import numpy as np

from utils.metrics import mean_recall_at_k, mean_ndcg_at_k

t0 = time.time()
evt = pd.read_parquet(
    f"{CACHE_DIR}/events_positive.parquet",
    columns=["user_id", "item_id", "event_type", "date"],
)
evt["date"] = pd.to_datetime(evt["date"])
print(f"events_positive: {len(evt):,} | {time.time()-t0:.1f}s")

VALID_START_TS = pd.Timestamp(VALID_START)
TRAIN_END_TS = pd.Timestamp(TRAIN_DATE_END)
train_mask = evt["date"] < VALID_START_TS
valid_mask = (evt["date"] >= VALID_START_TS) & (evt["date"] <= TRAIN_END_TS)
train_pos = evt[train_mask]
valid_pos = evt[valid_mask]
print(f"train_pos: {len(train_pos):,} ({len(train_pos)/len(evt)*100:.1f}%)")
print(f"valid_pos: {len(valid_pos):,} ({len(valid_pos)/len(evt)*100:.1f}%)")
""")

    code(cells, r"""
# valid_users = unique user trong cả train_pos và valid_pos
train_users = set(train_pos["user_id"].unique())
valid_users_full = set(valid_pos["user_id"].unique())
valid_users = train_users & valid_users_full
print(f"|train users|: {len(train_users):,}")
print(f"|valid users (any)|: {len(valid_users_full):,}")
print(f"|valid_users (∩ train)|: {len(valid_users):,}")

# Ground-truth: user_id -> set(item_id) in valid window
gt = (valid_pos[valid_pos["user_id"].isin(valid_users)]
      .groupby("user_id")["item_id"]
      .agg(set)
      .to_dict())
print(f"GT users: {len(gt):,} | mean items/user: "
      f"{np.mean([len(v) for v in gt.values()]):.2f}")
""")

    code(cells, r"""
# Save split for downstream notebooks
train_pos.to_parquet(f"{CACHE_DIR}/train_pos.parquet", index=False)
valid_pos.to_parquet(f"{CACHE_DIR}/valid_pos.parquet", index=False)

# Save valid_users + gt as parquet (gt as long form)
gt_long = (valid_pos[valid_pos["user_id"].isin(valid_users)]
           [["user_id", "item_id"]].drop_duplicates())
gt_long.to_parquet(f"{CACHE_DIR}/valid_gt.parquet", index=False)
print(f"valid_gt: {len(gt_long):,} pairs")
""")

    code(cells, r"""
# ---- Popularity baseline: tier-A items by contacts_total, top-10 ----
enr = pd.read_parquet(
    f"{CACHE_DIR}/dim_listing_enriched.parquet",
    columns=["item_id", "tier", "contacts_total", "n_pos_train"],
)
pop_pool = enr[enr["tier"] == "A"].sort_values(
    ["contacts_total", "n_pos_train"], ascending=False
)
top10_global = pop_pool["item_id"].head(10).tolist()
print("Top-10 popularity:", top10_global[:5], "...")

# Predict same top-10 for every valid user
preds_baseline = {u: top10_global for u in gt.keys()}
r10 = mean_recall_at_k(preds_baseline, gt, k=10)
n10 = mean_ndcg_at_k(preds_baseline, gt, k=10)
print(f"\nBaseline Recall@10: {r10:.4f}")
print(f"Baseline NDCG@10  : {n10:.4f}")
print("(Mục tiêu sau ranker: Recall@10 ≥ 2x baseline)")
""")

    code(cells, r"""
# Save baseline preds for sanity comparison later
import json
with open(f"{CACHE_DIR}/baseline_preds.json", "w") as f:
    json.dump({"top10_global": top10_global,
               "recall_at_10": r10, "ndcg_at_10": n10}, f, indent=2)
print("Baseline saved.")
""")
    return cells


# ============================================================
# Notebook 04 — candidates
# ============================================================

def build_nb04() -> list:
    cells = _new_cells()
    md(cells, r"""
# 04 — Candidate generation (4 sources, cap 500/user)

Sources:
1. **History repeat** (~30): items user đã interact, weighted by recency × intent.
2. **Co-visitation** (~200, trục chính): cặp item cùng session trên tier A items.
3. **Popularity per segment** (~100): top-50 contacts_24h last 14d theo (category × city).
4. **Content-based cho cold** (~50): match (category, district, ad_type) cho tier C items.

Output: `candidates_valid.parquet` với cột `user_id, item_id, src_history, src_covis, src_pop, src_content`.

Target: `recall@500 ≥ 0.6` trên valid_users.
""")
    code(cells, pip_install_cell())
    code(cells, setup_cell_drive_mount())
    code(cells, constants_cell())
    code(cells, assert_no_gcs_cell())

    code(cells, r"""
import time
import pandas as pd
import numpy as np

from utils.covis import build_covis, score_user_covis
from utils.candidates import (
    gen_history_candidates, gen_covis_candidates,
    gen_popularity_candidates, gen_content_candidates,
    merge_candidates,
)
from utils.metrics import mean_recall_at_k

t0 = time.time()
train_pos = pd.read_parquet(f"{CACHE_DIR}/train_pos.parquet")
train_pos["event_ts"] = pd.to_datetime(train_pos["event_ts"])
enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet")
valid_gt = pd.read_parquet(f"{CACHE_DIR}/valid_gt.parquet")
print(f"train_pos: {len(train_pos):,} | enr: {len(enr):,} | "
      f"valid_gt: {len(valid_gt):,} | {time.time()-t0:.1f}s")

VALID_USERS = set(valid_gt["user_id"].unique())
print(f"|valid users|: {len(VALID_USERS):,}")
""")

    code(cells, r"""
# Limit train_pos to valid users to save memory in candidate gen
train_pos_valid = train_pos[train_pos["user_id"].isin(VALID_USERS)].copy()
print(f"train_pos for valid users: {len(train_pos_valid):,}")

allowed_a = set(enr[enr["tier"] == "A"]["item_id"])
allowed_ab = set(enr[enr["tier"].isin(["A", "B"])]["item_id"])
print(f"|tier A|: {len(allowed_a):,} | |tier A+B|: {len(allowed_ab):,}")
""")

    code(cells, r"""
# ---- Source 1: history (tier A+B allowed) ----
t0 = time.time()
hist_cands = gen_history_candidates(train_pos_valid, allowed_ab, top_n=30)
print(f"history: {sum(len(v) for v in hist_cands.values()):,} pairs | {time.time()-t0:.1f}s")
""")

    code(cells, r"""
# ---- Source 2: co-visitation (tier A only) ----
# Build covis from train_pos (all users, not just valid - more signal)
t0 = time.time()
covis_input = train_pos[train_pos["item_id"].isin(allowed_a)][
    ["user_id", "session_id", "item_id", "event_type", "event_ts"]
]
print(f"covis input rows: {len(covis_input):,}")
covis = build_covis(covis_input, allowed_items=allowed_a,
                    top_k_per_item=20, time_decay=True)
print(f"covis items: {len(covis):,} | {time.time()-t0:.1f}s")

t0 = time.time()
covis_cands = gen_covis_candidates(train_pos_valid, covis,
                                    allowed_items=allowed_a, top_n=200)
print(f"covis cands: {sum(len(v) for v in covis_cands.values()):,} pairs | {time.time()-t0:.1f}s")
""")

    code(cells, r"""
# ---- Source 3: popularity per segment ----
# Build user_profile from train_pos: top category/city/district/ad_type per user
t0 = time.time()
enr_seg_cols = [c for c in ["item_id", "category", "city_name",
                             "district_name", "ad_type"] if c in enr.columns]
tp = train_pos_valid.merge(enr[enr_seg_cols], on="item_id", how="left")

def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None

prof_specs = {}
for src, out in [("category", "u_top_category"), ("city_name", "u_top_city"),
                  ("district_name", "u_top_district"), ("ad_type", "u_top_ad_type")]:
    if src in tp.columns:
        prof_specs[out] = (src, _mode)
user_profile = tp.groupby("user_id").agg(**prof_specs).reset_index()
print(f"user_profile: {len(user_profile):,} | {time.time()-t0:.1f}s")

snap = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
snap["date"] = pd.to_datetime(snap["date"])
last14_start = pd.Timestamp(TRAIN_DATE_END) - pd.Timedelta(days=14)
snap_14 = snap[snap["date"] >= last14_start]
print(f"snap_14d: {len(snap_14):,}")

pop_cands = gen_popularity_candidates(user_profile, enr, snap_14,
                                       top_n_per_seg=50, top_n_per_user=100)
print(f"pop cands: {sum(len(v) for v in pop_cands.values()):,} pairs")
""")

    code(cells, r"""
# ---- Source 4: content-based for cold items ----
t0 = time.time()
content_cands = gen_content_candidates(user_profile, enr, top_n=50)
print(f"content cands: {sum(len(v) for v in content_cands.values()):,} pairs | {time.time()-t0:.1f}s")
""")

    code(cells, r"""
# ---- Merge + cap 500 / user ----
t0 = time.time()
cands = merge_candidates({
    "history": hist_cands,
    "covis": covis_cands,
    "pop": pop_cands,
    "content": content_cands,
}, cap_total=500)
print(f"candidates: {len(cands):,} | users: {cands['user_id'].nunique():,} | "
      f"avg cand/user: {len(cands)/max(cands['user_id'].nunique(),1):.1f} | "
      f"{time.time()-t0:.1f}s")

cands.to_parquet(f"{CACHE_DIR}/candidates_valid.parquet", index=False)

# Verify no tier-D leakage
chk = cands.merge(enr[["item_id", "tier"]], on="item_id", how="left")
assert chk["tier"].isin(["A", "B", "C"]).all(), \
    f"Tier D leakage: {chk[chk['tier']=='D'].shape[0]}"
print("No tier-D leakage. OK.")
""")

    code(cells, r"""
# ---- Compute recall@500 on valid ----
gt = (valid_gt.groupby("user_id")["item_id"].agg(set).to_dict())
cand_by_user = (cands.groupby("user_id")["item_id"].apply(list).to_dict())
r500 = mean_recall_at_k(cand_by_user, gt, k=500)
print(f"Recall@500 (ceiling for ranker): {r500:.4f}")
if r500 < 0.5:
    print("WARNING: recall@500 below 0.5 — ranker can't recover. "
          "Consider widening candidate sources.")
""")
    return cells


# ============================================================
# Notebook 05 — features
# ============================================================

def build_nb05() -> list:
    cells = _new_cells()
    md(cells, r"""
# 05 — Feature matrix (user × item)

Build user_features.parquet + item_features.parquet, join với candidates → full train matrix.
NA giữ nguyên (LightGBM xử lý native).
""")
    code(cells, pip_install_cell())
    code(cells, setup_cell_drive_mount())
    code(cells, constants_cell())
    code(cells, assert_no_gcs_cell())

    code(cells, r"""
import time
import pandas as pd
import numpy as np

from utils.features import build_user_features, build_item_features, add_cross_features

t0 = time.time()
train_pos = pd.read_parquet(f"{CACHE_DIR}/train_pos.parquet")
train_pos["event_ts"] = pd.to_datetime(train_pos["event_ts"])
enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet")
print(f"train_pos: {len(train_pos):,} | enr: {len(enr):,} | {time.time()-t0:.1f}s")
""")

    code(cells, r"""
import os
pv_path = f"{CACHE_DIR}/events_pageview_30d.parquet"
if os.path.exists(pv_path):
    pv = pd.read_parquet(pv_path)
    pv["event_ts"] = pd.to_datetime(pv["event_ts"])
else:
    pv = pd.DataFrame(columns=["user_id", "item_id", "event_ts", "dwell_time_sec"])
print(f"pageview: {len(pv):,}")
""")

    code(cells, r"""
CUTOFF = pd.Timestamp(VALID_START)
t0 = time.time()
user_feats = build_user_features(train_pos, pv, cutoff_ts=CUTOFF)
print(f"user_feats: {user_feats.shape} | {time.time()-t0:.1f}s")
user_feats.to_parquet(f"{CACHE_DIR}/user_features.parquet", index=False)
""")

    code(cells, r"""
snap = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
t0 = time.time()
item_feats = build_item_features(train_pos, snap, enr, cutoff_ts=CUTOFF)
print(f"item_feats: {item_feats.shape} | {time.time()-t0:.1f}s")
item_feats.to_parquet(f"{CACHE_DIR}/item_features.parquet", index=False)
""")

    code(cells, r"""
# Join candidates with features
t0 = time.time()
cands = pd.read_parquet(f"{CACHE_DIR}/candidates_valid.parquet")
full = add_cross_features(cands, user_feats, item_feats)
print(f"full matrix: {full.shape} | {time.time()-t0:.1f}s")

# Attach label from valid_gt
valid_gt = pd.read_parquet(f"{CACHE_DIR}/valid_gt.parquet")
valid_gt["label"] = 1
full = full.merge(valid_gt, on=["user_id", "item_id"], how="left")
full["label"] = full["label"].fillna(0).astype("int8")
print(f"positive labels: {full['label'].sum():,} / {len(full):,} "
      f"({full['label'].mean()*100:.3f}%)")
full.to_parquet(f"{CACHE_DIR}/train_matrix_valid.parquet", index=False)
""")
    return cells


# ============================================================
# Notebook 06 — train_ranker
# ============================================================

def build_nb06() -> list:
    cells = _new_cells()
    md(cells, r"""
# 06 — Train LightGBM LambdaRank

Group = số candidate / user. Label = 1 nếu (u, i) ∈ valid_gt.
Sample weight: view_phone=3, contact_*=2, other_interaction=1 (cho positive).
""")
    code(cells, pip_install_cell())
    code(cells, setup_cell_drive_mount())
    code(cells, constants_cell())
    code(cells, assert_no_gcs_cell())

    code(cells, r"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from utils.metrics import mean_recall_at_k, mean_ndcg_at_k

t0 = time.time()
full = pd.read_parquet(f"{CACHE_DIR}/train_matrix_valid.parquet")
print(f"full: {full.shape} | {time.time()-t0:.1f}s")

# Drop non-feature columns
DROP = {"user_id", "item_id", "label", "title", "posted_date",
        "expected_expired_date", "first_evt_date", "last_evt_date",
        "last_snap_date", "project_id"}
cat_cols = [c for c in full.columns
            if c not in DROP and full[c].dtype == "object"]
num_cols = [c for c in full.columns
            if c not in DROP and full[c].dtype != "object"
            and full[c].dtype != "datetime64[ns]"]
for c in cat_cols:
    full[c] = full[c].astype("category")
feat_cols = cat_cols + num_cols
print(f"#features: {len(feat_cols)} (cat={len(cat_cols)}, num={len(num_cols)})")
""")

    code(cells, r"""
# Split inside the valid users: 80/20 by user_id hash to avoid leakage
from hashlib import md5
def _hash01(s):
    return int(md5(str(s).encode()).hexdigest(), 16) % 100

full["_h"] = full["user_id"].map(_hash01)
tr_mask = full["_h"] < 80
va_mask = full["_h"] >= 80

X_tr = full.loc[tr_mask, feat_cols]
y_tr = full.loc[tr_mask, "label"].values
g_tr = full.loc[tr_mask].groupby("user_id", sort=False).size().values

X_va = full.loc[va_mask, feat_cols]
y_va = full.loc[va_mask, "label"].values
g_va = full.loc[va_mask].groupby("user_id", sort=False).size().values

print(f"train: {len(X_tr):,} ({tr_mask.sum()} rows, "
      f"{len(g_tr)} groups, pos={y_tr.sum()})")
print(f"valid: {len(X_va):,} (pos={y_va.sum()})")
""")

    code(cells, r"""
params = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
    learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=1.0, max_bin=255, seed=42, verbosity=-1,
)
d_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr,
                   categorical_feature=cat_cols, free_raw_data=False)
d_va = lgb.Dataset(X_va, label=y_va, group=g_va,
                   categorical_feature=cat_cols, reference=d_tr,
                   free_raw_data=False)

t0 = time.time()
model = lgb.train(
    params, d_tr, num_boost_round=2000,
    valid_sets=[d_va], valid_names=["valid"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
)
print(f"trained in {time.time()-t0:.0f}s | best iter: {model.best_iteration}")
model.save_model(f"{CACHE_DIR}/model.txt")

fi = pd.DataFrame({"feature": feat_cols,
                    "gain": model.feature_importance(importance_type="gain"),
                    "split": model.feature_importance(importance_type="split")})
fi = fi.sort_values("gain", ascending=False)
fi.to_csv(f"{CACHE_DIR}/feature_importance.csv", index=False)
print(fi.head(20))
""")

    code(cells, r"""
# Internal Recall@10 / NDCG@10 on valid split
full.loc[va_mask, "_pred"] = model.predict(X_va, num_iteration=model.best_iteration)
preds_dict = {}
for uid, grp in full[va_mask].groupby("user_id", sort=False):
    top10 = grp.nlargest(10, "_pred")["item_id"].tolist()
    preds_dict[uid] = top10

valid_gt = pd.read_parquet(f"{CACHE_DIR}/valid_gt.parquet")
gt = valid_gt.groupby("user_id")["item_id"].agg(set).to_dict()
gt_va = {u: gt[u] for u in preds_dict if u in gt}
r10 = mean_recall_at_k(preds_dict, gt_va, k=10)
n10 = mean_ndcg_at_k(preds_dict, gt_va, k=10)
print(f"Internal Recall@10: {r10:.4f}")
print(f"Internal NDCG@10  : {n10:.4f}")

import json
with open(f"{CACHE_DIR}/ranker_metrics.json", "w") as f:
    json.dump({"recall_at_10": r10, "ndcg_at_10": n10,
               "best_iter": int(model.best_iteration)}, f, indent=2)
""")
    return cells


# ============================================================
# Notebook 07 — predict and submit
# ============================================================

def build_nb07() -> list:
    cells = _new_cells()
    md(cells, r"""
# 07 — Predict full test_users + diversify + submission

1. Sinh candidate cho **toàn bộ test_users** (cùng pipeline notebook 04, dùng FULL train data).
2. Build feature matrix với cutoff = TRAIN_END.
3. Predict score → top-30 / user.
4. Diversify (max_per_seller=7, max_per_district=8, freshness_boost).
5. Submission validator + write csv.
""")
    code(cells, pip_install_cell())
    code(cells, setup_cell_drive_mount())
    code(cells, constants_cell())
    code(cells, assert_no_gcs_cell())

    code(cells, r"""
import time, os
import numpy as np
import pandas as pd
import lightgbm as lgb

from utils.covis import build_covis
from utils.candidates import (
    gen_history_candidates, gen_covis_candidates,
    gen_popularity_candidates, gen_content_candidates,
    merge_candidates,
)
from utils.features import build_user_features, build_item_features, add_cross_features
from utils.diversify import diversify_top_k
from utils.submit import validate_submission, write_submission

t0 = time.time()
events_pos = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet")
events_pos["event_ts"] = pd.to_datetime(events_pos["event_ts"])
enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet")
test_users = pd.read_parquet(f"{CACHE_DIR}/test_users.parquet")
print(f"events_pos: {len(events_pos):,} | enr: {len(enr):,} | "
      f"test_users: {len(test_users):,} | {time.time()-t0:.1f}s")

TEST_UIDS = set(test_users["user_id"].unique())
events_pos_test = events_pos[events_pos["user_id"].isin(TEST_UIDS)].copy()
print(f"events_pos for test users: {len(events_pos_test):,}")
""")

    code(cells, r"""
# ---- Candidates (4 sources) for ALL test_users ----
allowed_a = set(enr[enr["tier"] == "A"]["item_id"])
allowed_ab = set(enr[enr["tier"].isin(["A", "B"])]["item_id"])

t0 = time.time()
hist_cands = gen_history_candidates(events_pos_test, allowed_ab, top_n=30)
print(f"history: {sum(len(v) for v in hist_cands.values()):,} | {time.time()-t0:.1f}s")

t0 = time.time()
covis_input = events_pos[events_pos["item_id"].isin(allowed_a)][
    ["user_id", "session_id", "item_id", "event_type", "event_ts"]
]
covis = build_covis(covis_input, allowed_items=allowed_a,
                    top_k_per_item=20, time_decay=True)
covis_cands = gen_covis_candidates(events_pos_test, covis,
                                    allowed_items=allowed_a, top_n=200)
print(f"covis: {sum(len(v) for v in covis_cands.values()):,} | {time.time()-t0:.1f}s")
""")

    code(cells, r"""
# user_profile for popularity + content
def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None

enr_seg_cols = [c for c in ["item_id", "category", "city_name",
                             "district_name", "ad_type"] if c in enr.columns]
tp_t = events_pos_test.merge(enr[enr_seg_cols], on="item_id", how="left")

prof_specs = {}
SEG_MAP = [("category", "u_top_category"), ("city_name", "u_top_city"),
            ("district_name", "u_top_district"), ("ad_type", "u_top_ad_type")]
for src, out in SEG_MAP:
    if src in tp_t.columns:
        prof_specs[out] = (src, _mode)
user_profile = tp_t.groupby("user_id").agg(**prof_specs).reset_index()

# For cold test_users (no history), fill with global mode
missing = TEST_UIDS - set(user_profile["user_id"])
print(f"Cold test users (no history): {len(missing):,}")
if missing:
    cold_row = {"user_id": list(missing)}
    for src, out in SEG_MAP:
        if src in tp_t.columns:
            m = tp_t[src].mode()
            cold_row[out] = m.iloc[0] if len(m) else None
    user_profile = pd.concat([user_profile, pd.DataFrame(cold_row)], ignore_index=True)

snap = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
snap["date"] = pd.to_datetime(snap["date"])
last14_start = pd.Timestamp(TRAIN_DATE_END) - pd.Timedelta(days=14)
snap_14 = snap[snap["date"] >= last14_start]

pop_cands = gen_popularity_candidates(user_profile, enr, snap_14,
                                       top_n_per_seg=50, top_n_per_user=100)
content_cands = gen_content_candidates(user_profile, enr, top_n=50)
print(f"pop: {sum(len(v) for v in pop_cands.values()):,} | "
      f"content: {sum(len(v) for v in content_cands.values()):,}")
""")

    code(cells, r"""
# Merge candidates
cands = merge_candidates({
    "history": hist_cands,
    "covis": covis_cands,
    "pop": pop_cands,
    "content": content_cands,
}, cap_total=500)
print(f"candidates total: {len(cands):,} | users: {cands['user_id'].nunique():,}")

# Ensure every test user has at least global pop fallback
missing_uids = TEST_UIDS - set(cands["user_id"])
print(f"Users still missing: {len(missing_uids):,}")
if missing_uids:
    pop_global = (snap_14.groupby("item_id")["contacts_24h"].sum()
                  .sort_values(ascending=False).head(100).index.tolist())
    pop_global = [it for it in pop_global if it in set(enr[enr["tier"].isin(["A","B"])]["item_id"])]
    fb_rows = []
    for u in missing_uids:
        for it in pop_global[:100]:
            fb_rows.append({"user_id": u, "item_id": it,
                            "src_history": np.nan, "src_covis": np.nan,
                            "src_pop": 1.0, "src_content": np.nan})
    cands = pd.concat([cands, pd.DataFrame(fb_rows)], ignore_index=True)
print(f"After fallback: users {cands['user_id'].nunique():,}")
""")

    code(cells, r"""
# Build features at cutoff = TRAIN_END (use full events_pos, not just train_pos)
CUTOFF = pd.Timestamp(TRAIN_DATE_END) + pd.Timedelta(seconds=1)
pv_path = f"{CACHE_DIR}/events_pageview_30d.parquet"
if os.path.exists(pv_path):
    pv = pd.read_parquet(pv_path)
    pv["event_ts"] = pd.to_datetime(pv["event_ts"])
else:
    pv = pd.DataFrame(columns=["user_id", "item_id", "event_ts", "dwell_time_sec"])

user_feats = build_user_features(events_pos, pv, cutoff_ts=CUTOFF)
item_feats = build_item_features(events_pos, snap, enr, cutoff_ts=CUTOFF)
print(f"user_feats: {user_feats.shape} | item_feats: {item_feats.shape}")

full = add_cross_features(cands, user_feats, item_feats)
print(f"test matrix: {full.shape}")
""")

    code(cells, r"""
# Load trained model + predict
model = lgb.Booster(model_file=f"{CACHE_DIR}/model.txt")

# Align features with trained model
trained_feats = model.feature_name()
for c in trained_feats:
    if c not in full.columns:
        full[c] = np.nan
X = full[trained_feats].copy()
# Re-cast categoricals
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = X[c].astype("category")

t0 = time.time()
full["_pred"] = model.predict(X, num_iteration=model.best_iteration)
print(f"predicted in {time.time()-t0:.0f}s")
""")

    code(cells, r"""
# Top-30 per user → diversify → top-10
top30 = (full.sort_values(["user_id", "_pred"], ascending=[True, False])
         .groupby("user_id").head(30)[["user_id", "item_id", "_pred"]]
         .rename(columns={"_pred": "score"}))
print(f"top30: {len(top30):,}")

dim_lookup = enr[["item_id", "seller_id", "district_name"]].copy()
if "i_listing_age_days_latest" in item_feats.columns:
    dim_lookup = dim_lookup.merge(
        item_feats[["item_id", "i_listing_age_days_latest"]],
        on="item_id", how="left",
    )

top10 = diversify_top_k(top30, dim_lookup, k=10,
                         max_per_seller=7, max_per_district=8,
                         freshness_boost=0.05, fresh_age_days=7)
print(f"top10 after diversify: {len(top10):,} | users: {top10['user_id'].nunique():,}")
""")

    code(cells, r"""
# Validate + write submission
valid_items = set(enr["item_id"])
validate_submission(top10, valid_items, TEST_UIDS, k=10)
out = f"{CACHE_DIR}/submission.csv"
write_submission(top10, out)
print(f"\nSubmission ready: {out}")
print(f"File size: {os.path.getsize(out)/(1024**2):.2f} MB")
""")
    return cells


# ============================================================
# Build all + write
# ============================================================

NOTEBOOKS = [
    ("01_prepare_data.ipynb",       build_nb01),
    ("02_enrich_dim.ipynb",         build_nb02),
    ("03_validation_split.ipynb",   build_nb03),
    ("04_candidates.ipynb",         build_nb04),
    ("05_features.ipynb",           build_nb05),
    ("06_train_ranker.ipynb",       build_nb06),
    ("07_predict_and_submit.ipynb", build_nb07),
]

META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}


def main() -> None:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for name, builder in NOTEBOOKS:
        cells = builder()
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = META
        out = os.path.join(here, name)
        with open(out, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        n_code = sum(1 for c in cells if c["cell_type"] == "code")
        n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
        print(f"  wrote {name}  ({n_code} code, {n_md} md)")
    print("\nAll 7 notebooks generated.")


if __name__ == "__main__":
    main()
