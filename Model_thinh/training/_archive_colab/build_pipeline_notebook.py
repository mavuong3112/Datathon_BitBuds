"""Build ONE end-to-end pipeline notebook for Datathon 2026 (Cho Tot BDS Recommender).

Run: python build_pipeline_notebook.py
Generates pipeline.ipynb — a single Colab-ready notebook with 7 stages:
  Stage 01 — Prepare data (single GCS egress)
  Stage 02 — Enrich dim
  Stage 03 — Validation split + baseline
  Stage 04 — Candidate generation
  Stage 05 — Feature matrix
  Stage 06 — Train LightGBM LambdaRank
  Stage 07 — Predict + diversify + submit

The notebook still imports helpers from `training/utils/`, so the user must
upload the entire `training/` folder to Drive at
  /content/drive/MyDrive/datathon2026/training/

Setup cells (mount Drive, install deps, constants, egress guardrail) appear
ONCE at the top rather than being duplicated in every stage.
"""
from __future__ import annotations

import os
import nbformat as nbf


def md(cells: list, src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(cells: list, src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ============================================================
# Shared setup (runs ONCE at top of notebook)
# ============================================================

SETUP_DRIVE = r"""
# === Colab setup: mount Drive + add training/utils to sys.path ===
import os, sys

try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    print("Not running on Colab; assuming local paths.")

# === IMPORTANT: upload the entire `training/` folder (including utils/) ===
# to: /content/drive/MyDrive/datathon2026/training/   (one-time setup)
PROJECT_DIR = "/content/drive/MyDrive/datathon2026" if IS_COLAB else "."
TRAINING_DIR = os.path.join(PROJECT_DIR, "training")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache_drive")
os.makedirs(CACHE_DIR, exist_ok=True)

if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

print("PROJECT_DIR:", PROJECT_DIR)
print("CACHE_DIR  :", CACHE_DIR)
print("utils available:", os.path.isdir(os.path.join(TRAINING_DIR, "utils")))
"""

PIP_INSTALL = r"""
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "pyarrow", "pandas", "polars", "numpy", "scipy",
    "gcsfs", "google-cloud-storage", "google-auth",
    "lightgbm", "scikit-learn", "tqdm",
])
print("Deps installed.")
"""

CONSTANTS = r"""
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

BUCKET_NAME = "datathon_2026_final"
TRAIN_PATH = f"gs://{BUCKET_NAME}/train/"
TEST_PATH  = f"gs://{BUCKET_NAME}/test/"

print("Constants loaded. TRAIN_END:", TRAIN_DATE_END, "VALID_START:", VALID_START)
"""

# Helper functions defined once at top — used in stages 04 and 07.
PIPELINE_HELPERS = r"""
# === Pipeline helpers (used in stages 04 and 07) ===
import pandas as pd
import numpy as np

def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None

def build_user_profile(events_df, enr_df):
    \"\"\"Aggregate per-user top category/city/district/ad_type from events history.\"\"\"
    seg_cols = [c for c in ["item_id", "category", "city_name",
                            "district_name", "ad_type"] if c in enr_df.columns]
    tp = events_df.merge(enr_df[seg_cols], on="item_id", how="left")
    prof_specs = {}
    seg_map = [("category", "u_top_category"), ("city_name", "u_top_city"),
               ("district_name", "u_top_district"), ("ad_type", "u_top_ad_type")]
    for src, out in seg_map:
        if src in tp.columns:
            prof_specs[out] = (src, _mode)
    profile = tp.groupby("user_id").agg(**prof_specs).reset_index()
    return profile, tp, seg_map

def generate_all_candidates(events_df, enr_df, snap_14, allowed_a, allowed_ab,
                             top_covis=200, top_history=30, top_pop_per_user=100,
                             top_content=50, cap_total=500):
    \"\"\"Run all 4 candidate sources and merge. Returns (cands_df, covis_dict).\"\"\"
    from utils.covis import build_covis
    from utils.candidates import (
        gen_history_candidates, gen_covis_candidates,
        gen_popularity_candidates, gen_content_candidates,
        merge_candidates,
    )

    hist_cands = gen_history_candidates(events_df, allowed_ab, top_n=top_history)

    covis_input = events_df[events_df["item_id"].isin(allowed_a)][
        ["user_id", "session_id", "item_id", "event_type", "event_ts"]
    ]
    covis = build_covis(covis_input, allowed_items=allowed_a,
                        top_k_per_item=20, time_decay=True)
    covis_cands = gen_covis_candidates(events_df, covis,
                                       allowed_items=allowed_a, top_n=top_covis)

    user_profile, _tp, _seg = build_user_profile(events_df, enr_df)
    pop_cands = gen_popularity_candidates(user_profile, enr_df, snap_14,
                                          top_n_per_seg=50, top_n_per_user=top_pop_per_user)
    content_cands = gen_content_candidates(user_profile, enr_df, top_n=top_content)

    cands = merge_candidates({
        "history": hist_cands,
        "covis": covis_cands,
        "pop": pop_cands,
        "content": content_cands,
    }, cap_total=cap_total)
    return cands, user_profile

print("Pipeline helpers ready.")
"""

EGRESS_GUARDRAIL = r"""
# === Egress guardrail: after Stage 01, no further gs:// reads allowed ===
# If you accidentally call pd.read_parquet("gs://...") in stages 02-07,
# this raises. Always read from CACHE_DIR.
import pandas as pd
_orig_read = pd.read_parquet
def _safe_read(path, *a, **kw):
    p = str(path)
    if p.startswith("gs://"):
        raise RuntimeError(f"BLOCKED gs:// read after Stage 01: {p}")
    return _orig_read(path, *a, **kw)
pd.read_parquet = _safe_read
print("Egress guardrail active. Stages 02-07 cannot read gs://.")
"""


# ============================================================
# Stage builders (no setup duplication — just stage logic)
# ============================================================

def stage_01(cells: list) -> None:
    md(cells, r"""
## Stage 01 — Prepare data (single GCS egress)

Đọc `gs://datathon_2026_final/train/` & `test/` **đúng 1 lần**,
cache 6 file parquet xuống Drive.

**Quy tắc egress** (`rulestrain_model.txt`): vi phạm = thu hồi quyền truy cập bucket.
Sau khi stage này chạy xong, **không chạy lại** trừ khi cache bị mất.

Outputs trong `cache_drive/`:
- `dim_listing.parquet` (FULL)
- `snapshot_60d.parquet` (last 60d)
- `pci_full.parquet`
- `events_positive.parquet` (5 event_type, date ≤ 2026-04-09)
- `events_pageview_30d.parquet`
- `test_users.parquet`
""")

    code(cells, r"""
# Auth GCS (Colab Gmail BTC đã cấp quyền)
try:
    from google.colab import auth as _colab_auth
    _colab_auth.authenticate_user()
    print("Colab GCS auth OK.")
except ImportError:
    import google.auth
    _c, _p = google.auth.default()
    print(f"Local ADC OK (project={_p!r}).")
""")

    code(cells, r"""
import time
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

t_start = time.time()

dim_dset  = ds.dataset(f"{TRAIN_PATH}dim_listing/",                    format="parquet")
snap_dset = ds.dataset(f"{TRAIN_PATH}fact_listing_snapshot/",          format="parquet")
pci_dset  = ds.dataset(f"{TRAIN_PATH}fact_post_contact_interactions/", format="parquet")
fue_dset  = ds.dataset(f"{TRAIN_PATH}fact_user_events/",               format="parquet")

train_end_d  = pa.scalar(pd.Timestamp(TRAIN_DATE_END).date(), type=pa.date32())
snap_start_d = pa.scalar(pd.Timestamp("2026-02-09").date(),   type=pa.date32())
pv_start_d   = pa.scalar(pd.Timestamp("2026-03-10").date(),   type=pa.date32())

print(f"dim files     : {len(dim_dset.files):,}")
print(f"snap files    : {len(snap_dset.files):,}")
print(f"pci files     : {len(pci_dset.files):,}")
print(f"events files  : {len(fue_dset.files):,}")

def _safe_cols(dset, wanted):
    have = set(dset.schema.names)
    keep = [c for c in wanted if c in have]
    miss = [c for c in wanted if c not in have]
    if miss:
        print(f"  [warn] missing cols dropped: {miss}")
    return keep
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
df_dim = dim_dset.to_table(columns=DIM_COLS).to_pandas()
print(f"dim_listing: {df_dim.shape} | {time.time()-t0:.1f}s")
df_dim.to_parquet(f"{CACHE_DIR}/dim_listing.parquet", index=False)
del df_dim
""")

    code(cells, r"""
# ---- 2) fact_listing_snapshot last 60d ----
flt_snap = (pc.field("date") >= snap_start_d) & (pc.field("date") <= train_end_d)
SNAP_COLS_WANT = ["item_id", "date", "views_24h", "contacts_24h",
                  "listing_age_days", "category"]
SNAP_COLS = _safe_cols(snap_dset, SNAP_COLS_WANT)
t0 = time.time()
df_snap = snap_dset.to_table(columns=SNAP_COLS, filter=flt_snap).to_pandas()
print(f"snapshot_60d: {df_snap.shape} | {time.time()-t0:.1f}s")
df_snap.to_parquet(f"{CACHE_DIR}/snapshot_60d.parquet", index=False)
del df_snap
""")

    code(cells, r"""
# ---- 3) fact_post_contact_interactions FULL (date <= TRAIN_END) ----
flt_pci = pc.field("date") <= train_end_d
PCI_COLS_WANT = ["user_id", "item_id", "date", "category", "purchased"]
PCI_COLS = _safe_cols(pci_dset, PCI_COLS_WANT)
t0 = time.time()
df_pci = pci_dset.to_table(columns=PCI_COLS, filter=flt_pci).to_pandas()
print(f"pci_full: {df_pci.shape} | {time.time()-t0:.1f}s")
df_pci.to_parquet(f"{CACHE_DIR}/pci_full.parquet", index=False)
del df_pci
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
print(f"events_positive: {n_rows:,} rows | {time.time()-t0:.1f}s")
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
# ---- 6) test_users + verification ----
df_test = pd.read_parquet(f"{TEST_PATH}test_users.parquet")
print(f"test_users: {len(df_test):,}")
df_test.to_parquet(f"{CACHE_DIR}/test_users.parquet", index=False)
print(f"\nSTAGE 01 ELAPSED: {time.time()-t_start:.0f}s")

for name in ["dim_listing.parquet", "snapshot_60d.parquet", "pci_full.parquet",
             "events_positive.parquet", "events_pageview_30d.parquet",
             "test_users.parquet"]:
    p = os.path.join(CACHE_DIR, name)
    sz = os.path.getsize(p) / (1024**2)
    print(f"  {name:35s} {sz:8.1f} MB")

assert len(df_test) == 161_568, f"test_users mismatch: {len(df_test)}"
df_p = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet",
                       columns=["event_type", "date"])
assert df_p["event_type"].isin(POSITIVE_EVENT_TYPES).all()
assert pd.to_datetime(df_p["date"]).max().date() <= pd.Timestamp(TRAIN_DATE_END).date()
print("\nStage 01 assertions passed. Cache ready.")
""")


def stage_02(cells: list) -> None:
    md(cells, r"""
## Stage 02 — Enrich `dim_listing` (tier A/B/C/D)

Output: `dim_listing_enriched.parquet` with aggregates, flags, recency, and tier classification.
""")

    code(cells, r"""
import time
import numpy as np
import pandas as pd

t0 = time.time()
dim = pd.read_parquet(f"{CACHE_DIR}/dim_listing.parquet")
dim["posted_date"] = pd.to_datetime(dim["posted_date"], errors="coerce")
print(f"dim: {dim.shape} | {time.time()-t0:.1f}s")

# aggregate events_positive per item
t0 = time.time()
evt = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet",
                     columns=["item_id", "user_id", "date"])
evt["date"] = pd.to_datetime(evt["date"], errors="coerce")
evt_agg = evt.groupby("item_id").agg(
    n_pos_train=("user_id", "count"),
    n_unique_users=("user_id", "nunique"),
    first_evt_date=("date", "min"),
    last_evt_date=("date", "max"),
).reset_index()
print(f"evt_agg: {evt_agg.shape} | {time.time()-t0:.1f}s")
del evt

# aggregate snapshot
t0 = time.time()
snap = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet",
                      columns=["item_id", "date", "views_24h", "contacts_24h"])
snap["date"] = pd.to_datetime(snap["date"], errors="coerce")
snap_agg = snap.groupby("item_id").agg(
    last_snap_date=("date", "max"),
    n_snap_days=("date", "nunique"),
    views_total=("views_24h", "sum"),
    contacts_total=("contacts_24h", "sum"),
).reset_index()
print(f"snap_agg: {snap_agg.shape} | {time.time()-t0:.1f}s")
del snap
""")

    code(cells, r"""
# merge + flags + tier
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
    is_deleted = (enr["ad_status"].astype(str).str.lower() == "deleted")
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
print(f"A share: {(enr['tier']=='A').mean()*100:.1f}% | "
      f"D share: {(enr['tier']=='D').mean()*100:.1f}%")

enr.to_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet", index=False)
print(f"Wrote dim_listing_enriched.parquet | {enr.shape}")
""")


def stage_03(cells: list) -> None:
    md(cells, r"""
## Stage 03 — Validation split + popularity baseline

- `train_pos`: events_positive with `date < VALID_START` (2026-03-13)
- `valid_pos`: events_positive with `date ∈ [VALID_START, TRAIN_END]`
- `valid_users`: users in both (proxies the real test_users with history)
- Baseline: top-10 tier-A items by `contacts_total`. Target after ranker: ≥ 2× baseline.
""")

    code(cells, r"""
import time, json
import pandas as pd
import numpy as np

from utils.metrics import mean_recall_at_k, mean_ndcg_at_k

t0 = time.time()
evt = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet",
                     columns=["user_id", "item_id", "event_type", "date"])
evt["date"] = pd.to_datetime(evt["date"])
print(f"events_positive: {len(evt):,} | {time.time()-t0:.1f}s")

VALID_START_TS = pd.Timestamp(VALID_START)
TRAIN_END_TS = pd.Timestamp(TRAIN_DATE_END)
train_mask = evt["date"] < VALID_START_TS
valid_mask = (evt["date"] >= VALID_START_TS) & (evt["date"] <= TRAIN_END_TS)
train_pos = evt[train_mask]
valid_pos = evt[valid_mask]
print(f"train_pos: {len(train_pos):,}  valid_pos: {len(valid_pos):,}")

valid_users = set(train_pos["user_id"].unique()) & set(valid_pos["user_id"].unique())
gt = (valid_pos[valid_pos["user_id"].isin(valid_users)]
      .groupby("user_id")["item_id"].agg(set).to_dict())
print(f"|valid_users ∩ train|: {len(valid_users):,} | "
      f"mean items/user: {np.mean([len(v) for v in gt.values()]):.2f}")

# We need event_ts and session_id from the parent file too for downstream — re-read full
train_pos = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet")
train_pos["event_ts"] = pd.to_datetime(train_pos["event_ts"])
train_pos["date"] = pd.to_datetime(train_pos["date"])
valid_pos_full = train_pos[(train_pos["date"] >= VALID_START_TS)
                           & (train_pos["date"] <= TRAIN_END_TS)]
train_pos = train_pos[train_pos["date"] < VALID_START_TS]

train_pos.to_parquet(f"{CACHE_DIR}/train_pos.parquet", index=False)
valid_pos_full.to_parquet(f"{CACHE_DIR}/valid_pos.parquet", index=False)
gt_long = (valid_pos_full[valid_pos_full["user_id"].isin(valid_users)]
           [["user_id", "item_id"]].drop_duplicates())
gt_long.to_parquet(f"{CACHE_DIR}/valid_gt.parquet", index=False)
print(f"valid_gt: {len(gt_long):,} pairs")
""")

    code(cells, r"""
# Popularity baseline: tier-A items by contacts_total, top-10
enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet",
                     columns=["item_id", "tier", "contacts_total", "n_pos_train"])
pop_pool = enr[enr["tier"] == "A"].sort_values(
    ["contacts_total", "n_pos_train"], ascending=False)
top10_global = pop_pool["item_id"].head(10).tolist()

preds_baseline = {u: top10_global for u in gt.keys()}
r10 = mean_recall_at_k(preds_baseline, gt, k=10)
n10 = mean_ndcg_at_k(preds_baseline, gt, k=10)
print(f"Baseline Recall@10: {r10:.4f}  NDCG@10: {n10:.4f}")
print("(Mục tiêu sau ranker: Recall@10 ≥ 2x baseline)")

with open(f"{CACHE_DIR}/baseline_preds.json", "w") as f:
    json.dump({"top10_global": top10_global,
               "recall_at_10": r10, "ndcg_at_10": n10}, f, indent=2)
""")


def stage_04(cells: list) -> None:
    md(cells, r"""
## Stage 04 — Candidate generation (validation users)

4 sources merged, capped at 500/user:
1. History repeat (~30)
2. Co-visitation (~200, main driver)
3. Popularity per segment (~100)
4. Content-based for cold (~50)

Uses the `generate_all_candidates()` helper defined at the top of the notebook.
Target: `recall@500 ≥ 0.6` on valid_users.
""")

    code(cells, r"""
import time
import pandas as pd
from utils.metrics import mean_recall_at_k

t0 = time.time()
train_pos = pd.read_parquet(f"{CACHE_DIR}/train_pos.parquet")
train_pos["event_ts"] = pd.to_datetime(train_pos["event_ts"])
enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet")
valid_gt = pd.read_parquet(f"{CACHE_DIR}/valid_gt.parquet")
print(f"train_pos: {len(train_pos):,} | enr: {len(enr):,} | "
      f"valid_gt: {len(valid_gt):,} | {time.time()-t0:.1f}s")

VALID_USERS = set(valid_gt["user_id"].unique())
train_pos_valid = train_pos[train_pos["user_id"].isin(VALID_USERS)].copy()

allowed_a = set(enr[enr["tier"] == "A"]["item_id"])
allowed_ab = set(enr[enr["tier"].isin(["A", "B"])]["item_id"])

snap = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
snap["date"] = pd.to_datetime(snap["date"])
last14_start = pd.Timestamp(TRAIN_DATE_END) - pd.Timedelta(days=14)
snap_14 = snap[snap["date"] >= last14_start]

# NOTE: covis is built from FULL train_pos (more signal), then scored on valid users.
# We inline that here because generate_all_candidates uses events_df for both.
# For maximum signal, we pass train_pos as the events source.
t0 = time.time()
cands, user_profile = generate_all_candidates(
    train_pos_valid, enr, snap_14, allowed_a, allowed_ab,
    top_covis=200, top_history=30, top_pop_per_user=100, top_content=50,
    cap_total=500,
)
print(f"candidates: {len(cands):,} | users: {cands['user_id'].nunique():,} | "
      f"avg cand/user: {len(cands)/max(cands['user_id'].nunique(),1):.1f} | "
      f"{time.time()-t0:.1f}s")

cands.to_parquet(f"{CACHE_DIR}/candidates_valid.parquet", index=False)

# Verify no tier-D leakage + measure recall@500
chk = cands.merge(enr[["item_id", "tier"]], on="item_id", how="left")
assert chk["tier"].isin(["A", "B", "C"]).all(), "Tier D leakage!"

gt = valid_gt.groupby("user_id")["item_id"].agg(set).to_dict()
cand_by_user = cands.groupby("user_id")["item_id"].apply(list).to_dict()
r500 = mean_recall_at_k(cand_by_user, gt, k=500)
print(f"Recall@500 (ceiling for ranker): {r500:.4f}")
if r500 < 0.5:
    print("WARNING: recall@500 below 0.5 — widen candidate sources.")
""")


def stage_05(cells: list) -> None:
    md(cells, r"""
## Stage 05 — Feature matrix (user × item)

Build `user_features.parquet` + `item_features.parquet`, join with candidates,
attach labels from `valid_gt` → `train_matrix_valid.parquet`.
""")

    code(cells, r"""
import time, os
import pandas as pd
import numpy as np

from utils.features import build_user_features, build_item_features, add_cross_features

t0 = time.time()
train_pos = pd.read_parquet(f"{CACHE_DIR}/train_pos.parquet")
train_pos["event_ts"] = pd.to_datetime(train_pos["event_ts"])
enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet")

pv_path = f"{CACHE_DIR}/events_pageview_30d.parquet"
if os.path.exists(pv_path):
    pv = pd.read_parquet(pv_path)
    pv["event_ts"] = pd.to_datetime(pv["event_ts"])
else:
    pv = pd.DataFrame(columns=["user_id", "item_id", "event_ts", "dwell_time_sec"])
print(f"loaded inputs | {time.time()-t0:.1f}s")
""")

    code(cells, r"""
CUTOFF = pd.Timestamp(VALID_START)

t0 = time.time()
user_feats = build_user_features(train_pos, pv, cutoff_ts=CUTOFF)
user_feats.to_parquet(f"{CACHE_DIR}/user_features.parquet", index=False)
print(f"user_feats: {user_feats.shape} | {time.time()-t0:.1f}s")

snap = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
t0 = time.time()
item_feats = build_item_features(train_pos, snap, enr, cutoff_ts=CUTOFF)
item_feats.to_parquet(f"{CACHE_DIR}/item_features.parquet", index=False)
print(f"item_feats: {item_feats.shape} | {time.time()-t0:.1f}s")

# Join candidates with features + label
t0 = time.time()
cands = pd.read_parquet(f"{CACHE_DIR}/candidates_valid.parquet")
full = add_cross_features(cands, user_feats, item_feats)
print(f"full matrix: {full.shape} | {time.time()-t0:.1f}s")

valid_gt = pd.read_parquet(f"{CACHE_DIR}/valid_gt.parquet")
valid_gt["label"] = 1
full = full.merge(valid_gt, on=["user_id", "item_id"], how="left")
full["label"] = full["label"].fillna(0).astype("int8")
print(f"positives: {full['label'].sum():,} / {len(full):,} "
      f"({full['label'].mean()*100:.3f}%)")
full.to_parquet(f"{CACHE_DIR}/train_matrix_valid.parquet", index=False)
""")


def stage_06(cells: list) -> None:
    md(cells, r"""
## Stage 06 — Train LightGBM LambdaRank

Group = #candidates / user. Label = 1 if (u, i) ∈ valid_gt.
80/20 user-hash split to avoid leakage. Save model + feature importance.
""")

    code(cells, r"""
import time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from hashlib import md5

from utils.metrics import mean_recall_at_k, mean_ndcg_at_k

t0 = time.time()
full = pd.read_parquet(f"{CACHE_DIR}/train_matrix_valid.parquet")
print(f"full: {full.shape} | {time.time()-t0:.1f}s")

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

print(f"train: {len(X_tr):,} rows, {len(g_tr)} groups, pos={y_tr.sum()}")
print(f"valid: {len(X_va):,} rows, pos={y_va.sum()}")
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
    preds_dict[uid] = grp.nlargest(10, "_pred")["item_id"].tolist()

valid_gt = pd.read_parquet(f"{CACHE_DIR}/valid_gt.parquet")
gt = valid_gt.groupby("user_id")["item_id"].agg(set).to_dict()
gt_va = {u: gt[u] for u in preds_dict if u in gt}
r10 = mean_recall_at_k(preds_dict, gt_va, k=10)
n10 = mean_ndcg_at_k(preds_dict, gt_va, k=10)
print(f"Internal Recall@10: {r10:.4f}  NDCG@10: {n10:.4f}")

with open(f"{CACHE_DIR}/ranker_metrics.json", "w") as f:
    json.dump({"recall_at_10": r10, "ndcg_at_10": n10,
               "best_iter": int(model.best_iteration)}, f, indent=2)
""")


def stage_07(cells: list) -> None:
    md(cells, r"""
## Stage 07 — Predict full `test_users` → diversify → submission

1. Generate candidates for ALL test users (FULL events_positive history, cutoff = TRAIN_END).
2. Build features at cutoff = TRAIN_END.
3. Predict → top-30 / user → diversify → top-10.
4. Validate + write `submission.csv`.
""")

    code(cells, r"""
import time, os
import numpy as np
import pandas as pd
import lightgbm as lgb

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

allowed_a = set(enr[enr["tier"] == "A"]["item_id"])
allowed_ab = set(enr[enr["tier"].isin(["A", "B"])]["item_id"])

snap = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
snap["date"] = pd.to_datetime(snap["date"])
last14_start = pd.Timestamp(TRAIN_DATE_END) - pd.Timedelta(days=14)
snap_14 = snap[snap["date"] >= last14_start]
""")

    code(cells, r"""
# Generate candidates for test users via the same helper used in Stage 04
t0 = time.time()
cands, user_profile = generate_all_candidates(
    events_pos_test, enr, snap_14, allowed_a, allowed_ab,
    top_covis=200, top_history=30, top_pop_per_user=100, top_content=50,
    cap_total=500,
)
print(f"candidates: {len(cands):,} | users: {cands['user_id'].nunique():,} | "
      f"{time.time()-t0:.1f}s")

# Fallback: cold users (no positive history) get global pop top-100
missing_uids = TEST_UIDS - set(cands["user_id"])
print(f"Cold users needing fallback: {len(missing_uids):,}")
if missing_uids:
    pop_global = (snap_14.groupby("item_id")["contacts_24h"].sum()
                  .sort_values(ascending=False).head(200).index.tolist())
    pop_global = [it for it in pop_global if it in allowed_ab][:100]
    fb_rows = []
    for u in missing_uids:
        for it in pop_global:
            fb_rows.append({"user_id": u, "item_id": it,
                            "src_history": np.nan, "src_covis": np.nan,
                            "src_pop": 1.0, "src_content": np.nan})
    cands = pd.concat([cands, pd.DataFrame(fb_rows)], ignore_index=True)
print(f"After fallback: users {cands['user_id'].nunique():,}")
""")

    code(cells, r"""
# Build features at cutoff = TRAIN_END (use FULL events_pos)
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
# Load model + align features + predict
model = lgb.Booster(model_file=f"{CACHE_DIR}/model.txt")
trained_feats = model.feature_name()
for c in trained_feats:
    if c not in full.columns:
        full[c] = np.nan
X = full[trained_feats].copy()
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = X[c].astype("category")

t0 = time.time()
full["_pred"] = model.predict(X, num_iteration=model.best_iteration)
print(f"predicted in {time.time()-t0:.0f}s")
""")

    code(cells, r"""
# Top-30 per user → diversify → top-10 → write submission
top30 = (full.sort_values(["user_id", "_pred"], ascending=[True, False])
         .groupby("user_id").head(30)[["user_id", "item_id", "_pred"]]
         .rename(columns={"_pred": "score"}))

dim_lookup = enr[["item_id", "seller_id", "district_name"]].copy()
if "i_listing_age_days_latest" in item_feats.columns:
    dim_lookup = dim_lookup.merge(
        item_feats[["item_id", "i_listing_age_days_latest"]],
        on="item_id", how="left",
    )

top10 = diversify_top_k(top30, dim_lookup, k=10,
                        max_per_seller=7, max_per_district=8,
                        freshness_boost=0.05, fresh_age_days=7)
print(f"top10 after diversify: {len(top10):,} | "
      f"users: {top10['user_id'].nunique():,}")

valid_items = set(enr["item_id"])
validate_submission(top10, valid_items, TEST_UIDS, k=10)
out = f"{CACHE_DIR}/submission.csv"
write_submission(top10, out)
print(f"\nSubmission ready: {out}")
print(f"File size: {os.path.getsize(out)/(1024**2):.2f} MB")
""")


# ============================================================
# Assemble + write
# ============================================================

META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}


def main() -> None:
    cells: list = []

    # ---------- Title + overview ----------
    md(cells, r"""
# Datathon 2026 — Cho Tot BDS Recommender (Single Pipeline)

End-to-end pipeline in one notebook. Equivalent to the 7-notebook split but
runs top-to-bottom with shared setup. Helpers live in `training/utils/`.

**One-time setup on Drive:**
```
MyDrive/datathon2026/
└── training/
    ├── utils/        (candidates.py, covis.py, features.py, ...)
    └── pipeline.ipynb  (this file)
```
Cache will be written to `MyDrive/datathon2026/cache_drive/`.

**Pipeline stages:**
1. Stage 01 — Prepare data (SINGLE GCS egress — do not re-run)
2. Stage 02 — Enrich dim_listing (tier A/B/C/D)
3. Stage 03 — Validation split + popularity baseline
4. Stage 04 — Candidate generation (validation)
5. Stage 05 — Feature matrix
6. Stage 06 — Train LightGBM LambdaRank
7. Stage 07 — Predict full test_users + submit

Each stage reads from / writes to `cache_drive/`, so you can resume from any
stage after a runtime restart (just re-run the Setup section first).
""")

    # ---------- Setup section (runs once) ----------
    md(cells, "## Setup (run once per session)")
    code(cells, PIP_INSTALL)
    code(cells, SETUP_DRIVE)
    code(cells, CONSTANTS)
    code(cells, PIPELINE_HELPERS)

    # ---------- Stage 01 (egress) ----------
    stage_01(cells)

    # ---------- Egress guardrail (after stage 01 only) ----------
    md(cells, r"""
---
### Activate egress guardrail

After Stage 01 finishes, run this cell to block any further `gs://` reads.
This protects against accidentally re-triggering egress in stages 02-07.
""")
    code(cells, EGRESS_GUARDRAIL)

    # ---------- Stages 02-07 ----------
    stage_02(cells)
    stage_03(cells)
    stage_04(cells)
    stage_05(cells)
    stage_06(cells)
    stage_07(cells)

    # ---------- Done ----------
    md(cells, r"""
---
## Done

Submission file: `MyDrive/datathon2026/cache_drive/submission.csv`

To re-train from a different point without re-running the whole notebook:
- Run **Setup** cells, then the **egress guardrail** cell.
- Skip Stage 01 (cache already on Drive).
- Run only the stages you need — each stage reads its inputs from `cache_drive/`.
""")

    # ---------- Write ----------
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = META

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "pipeline.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"Wrote {out}")
    print(f"  total cells: {len(cells)}  (code={n_code}, md={n_md})")


if __name__ == "__main__":
    main()
