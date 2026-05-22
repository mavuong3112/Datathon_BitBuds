"""Generate 06b_moe_ranker.ipynb and 07b_moe_predict.ipynb."""
import json, os

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

def md(src):
    return {"cell_type": "markdown", "id": f"md{abs(hash(src))%99999:05d}",
            "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "id": f"cd{abs(hash(src))%99999:05d}",
            "metadata": {}, "outputs": [], "execution_count": None, "source": src}

def make_nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": cells,
    }

SETUP = """\
import os, sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from hashlib import md5

PROJECT_DIR  = r"d:/Datathon_2"
TRAINING_DIR = os.path.join(PROJECT_DIR, "training")
CACHE_DIR    = os.path.join(PROJECT_DIR, "cache_drive")
os.makedirs(CACHE_DIR, exist_ok=True)
for p in (TRAINING_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.features import build_user_features, build_item_features, add_cross_features
from utils.metrics import mean_recall_at_k, mean_ndcg_at_k
print("Imports OK | CACHE_DIR:", CACHE_DIR)
"""

CONSTANTS = """\
TRAIN_DATE_END = "2026-04-09"
VALID_START    = "2026-03-13"

# HCM city names as they appear in city_name column
HCM_CITY_NAMES = {
    "Hồ Chí Minh", "Ho Chi Minh", "TP. Hồ Chí Minh", "TP.HCM",
    "TP Hồ Chí Minh", "Thành phố Hồ Chí Minh",
}

MONO_MAP = {
    "age_at_train_end": -1, "recency_evt_days": -1, "u_recency_days": -1,
    "i_CR_30d": 1, "i_contacts_24h_mean_30d": 1,
    "i_n_pos_30d": 1, "u_n_pos_30d": 1,
}
DROP_COLS = {"user_id","item_id","label","title","posted_date","expected_expired_date",
             "first_evt_date","last_evt_date","last_snap_date","project_id","_h","_geo"}

INTENT_WEIGHT = {
    "view_phone": 3.0, "contact_chat": 2.0,
    "contact_zalo": 2.0, "contact_sms": 2.0,
    "other_interaction": 1.0,
}
print("Constants OK | HCM aliases:", len(HCM_CITY_NAMES))
"""

# ============================================================
# 06b — MoE Ranker training
# ============================================================
NB06B_CELLS = [

md("""\
# 06b — MoE Geo Router: Train Sub-Models (HCM Expert + Generalist)

**Ý tưởng:** Thị trường HCM (chung cư/phòng trọ, thanh khoản nhanh) và các tỉnh khác
(đất nền, thanh khoản chậm) có pattern khác nhau. Ép 1 model học cả 2 sẽ bị "trung bình hóa".

**Thiết kế:**
- **Router**: dựa vào `u_top_city` từ lịch sử clickstream → phân luồng user
- **Sub-Model A (HCM Expert)**: train trên user × item trong HCM
- **Sub-Model B (Generalist)**: train trên user × item ngoài HCM
  (+ 20% undersample từ HCM để giữ cross-market generalisation)

Output: `cache_drive/model_hcm.txt`, `cache_drive/model_general.txt`,
`cache_drive/model_hcm_feats.json`, `cache_drive/model_general_feats.json`
"""),

code("print('Skipping pip install (local kernel).')"),
code(SETUP),
code(CONSTANTS),

code("""\
# ---- Load train matrix (output of nb05) ----
t0 = time.time()
full = pd.read_parquet(f"{CACHE_DIR}/train_matrix_valid.parquet")
print(f"full matrix: {full.shape} | {time.time()-t0:.1f}s")

# Load dim to get city_name per item
dim = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet",
                       columns=["item_id","city_name"])
full = full.merge(dim, on="item_id", how="left")
print(f"city_name null: {full['city_name'].isna().sum():,}")
"""),

code("""\
# ---- Router: classify each row's city segment ----
# Use city_name from item (listing location), not user's top_city,
# so the split is about ITEM geography (where the listing is).
full["_geo"] = full["city_name"].apply(
    lambda c: "HCM" if (isinstance(c, str) and c.strip() in HCM_CITY_NAMES) else "OTHER"
)
geo_counts = full["_geo"].value_counts()
print("Geo distribution in train matrix:")
print(geo_counts)
print(f"HCM share: {geo_counts.get('HCM',0)/len(full)*100:.1f}%")
"""),

code("""\
def build_and_train(df, seg_label, save_model_path, save_feats_path,
                    n_boost=2000, early_stop=100):
    \"\"\"Train a lambdarank LightGBM on dataframe df, return model.\"\"\"
    cat_cols  = [c for c in df.columns if c not in DROP_COLS and df[c].dtype == "object"]
    num_cols  = [c for c in df.columns if c not in DROP_COLS and df[c].dtype != "object"
                 and "datetime" not in str(df[c].dtype)]
    feat_cols = cat_cols + num_cols
    for c in cat_cols:
        df[c] = df[c].astype("category")

    df["_h"] = df["user_id"].map(lambda s: int(md5(str(s).encode()).hexdigest(), 16) % 100)
    tr_m = df["_h"] < 80;  va_m = df["_h"] >= 80
    X_tr = df.loc[tr_m, feat_cols];  y_tr = df.loc[tr_m, "label"].values
    g_tr = df.loc[tr_m].groupby("user_id", sort=False).size().values
    X_va = df.loc[va_m, feat_cols];  y_va = df.loc[va_m, "label"].values
    g_va = df.loc[va_m].groupby("user_id", sort=False).size().values

    mono_list = [MONO_MAP.get(c, 0) for c in feat_cols]
    active_mono = sum(v != 0 for v in mono_list)
    print(f"  [{seg_label}] rows={len(df):,} | feats={len(feat_cols)} "
          f"(cat={len(cat_cols)}) | mono_active={active_mono}")
    print(f"  train groups={len(g_tr)} pos={y_tr.sum()} | "
          f"val groups={len(g_va)} pos={y_va.sum()}")

    params = dict(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=1.0, max_bin=255, seed=42, verbosity=-1,
        monotone_constraints=mono_list,
        monotone_constraints_method="advanced",
    )
    d_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr,
                       categorical_feature=cat_cols, free_raw_data=False)
    d_va = lgb.Dataset(X_va, label=y_va, group=g_va,
                       categorical_feature=cat_cols, reference=d_tr, free_raw_data=False)

    t0 = time.time()
    model = lgb.train(
        params, d_tr, num_boost_round=n_boost,
        valid_sets=[d_va], valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(100)],
    )
    print(f"  [{seg_label}] trained {time.time()-t0:.0f}s | best_iter={model.best_iteration}")
    model.save_model(save_model_path)
    with open(save_feats_path, "w") as fj:
        json.dump(feat_cols, fj)

    # Internal eval on va split
    preds = {}
    df.loc[va_m, "_pred"] = model.predict(X_va, num_iteration=model.best_iteration)
    for uid, grp in df[va_m].groupby("user_id", sort=False):
        preds[uid] = grp.nlargest(10, "_pred")["item_id"].tolist()
    valid_gt = pd.read_parquet(f"{CACHE_DIR}/valid_gt.parquet")
    gt_dict = valid_gt.groupby("user_id")["item_id"].agg(set).to_dict()
    gt_va = {u: gt_dict[u] for u in preds if u in gt_dict}
    r10 = mean_recall_at_k(preds, gt_va, k=10)
    n10 = mean_ndcg_at_k(preds,  gt_va, k=10)
    print(f"  [{seg_label}] INTERNAL (val split) => Recall@10={r10:.4f}  NDCG@10={n10:.4f}")
    return model, feat_cols
"""),

code("""\
# ---- Sub-Model A: HCM Expert ----
df_hcm = full[full["_geo"] == "HCM"].copy()
print(f"HCM subset: {len(df_hcm):,} rows")

model_hcm, feats_hcm = build_and_train(
    df_hcm, "HCM",
    save_model_path=f"{CACHE_DIR}/model_hcm.txt",
    save_feats_path=f"{CACHE_DIR}/model_hcm_feats.json",
)
"""),

code("""\
# ---- Sub-Model B: Generalist (non-HCM + 20% HCM undersample) ----
df_other = full[full["_geo"] == "OTHER"].copy()
n_hcm_sample = max(1, int(len(df_other) * 0.20))
df_hcm_sample = df_hcm.sample(n=min(n_hcm_sample, len(df_hcm)), random_state=42)
df_general = pd.concat([df_other, df_hcm_sample], ignore_index=True)
print(f"Generalist subset: {len(df_general):,} rows "
      f"(other={len(df_other):,} + hcm_sample={len(df_hcm_sample):,})")

model_general, feats_general = build_and_train(
    df_general, "GENERAL",
    save_model_path=f"{CACHE_DIR}/model_general.txt",
    save_feats_path=f"{CACHE_DIR}/model_general_feats.json",
)
"""),

code("""\
print("\\nMoE models saved:")
for fname in ["model_hcm.txt","model_general.txt",
              "model_hcm_feats.json","model_general_feats.json"]:
    p = f"{CACHE_DIR}/{fname}"
    size_mb = os.path.getsize(p) / 1024**2
    print(f"  {fname}: {size_mb:.1f} MB")
"""),

]  # end NB06B_CELLS

# ============================================================
# 07b — MoE predict + submit
# ============================================================
NB07B_CELLS = [

md("""\
# 07b — MoE Predict + Submit (Router: HCM Expert vs Generalist)

Dùng `u_top_city` từ lịch sử test_user để route qua đúng sub-model.
- user top_city ∈ HCM_CITY_NAMES → Model HCM Expert
- user top_city khác hoặc cold (null) → Model Generalist

Output: `cache_drive/submission_moe.csv`
"""),

code("print('Skipping pip install (local kernel).')"),
code(SETUP),
code(CONSTANTS),

code("""\
# ---- Load candidates + features for full test set ----
# (run nb07 first to produce the test matrix OR reproduce inline)
import os

# Check if test feature matrix already exists from nb07 run
test_mat_path = f"{CACHE_DIR}/test_matrix.parquet"

t0 = time.time()
if os.path.exists(test_mat_path):
    full = pd.read_parquet(test_mat_path)
    print(f"Loaded test_matrix: {full.shape} | {time.time()-t0:.1f}s")
else:
    # Rebuild from cache (same as nb07 steps 1-4)
    from utils.covis import build_covis
    from utils.candidates import (gen_history_candidates, gen_covis_candidates,
                                   gen_popularity_candidates, gen_content_candidates,
                                   merge_candidates)

    events_pos = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet")
    events_pos["event_ts"] = pd.to_datetime(events_pos["event_ts"])
    enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet")
    test_users_df = pd.read_parquet(f"{CACHE_DIR}/test_users.parquet")
    snap_60d = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
    snap_60d["date"] = pd.to_datetime(snap_60d["date"])

    TEST_UIDS = set(test_users_df["user_id"])
    events_pos_test = events_pos[events_pos["user_id"].isin(TEST_UIDS)].copy()

    allowed_a  = set(enr[enr["tier"] == "A"]["item_id"])
    allowed_ab = set(enr[enr["tier"].isin(["A","B"])]["item_id"])

    hist  = gen_history_candidates(events_pos_test, allowed_ab, top_n=30)
    covis = build_covis(events_pos[events_pos["item_id"].isin(allowed_a)][
        ["user_id","session_id","item_id","event_type","event_ts"]],
        allowed_items=allowed_a, top_k_per_item=20, time_decay=True)
    cvis  = gen_covis_candidates(events_pos_test, covis, allowed_a, top_n=200)

    def _mode(s):
        s = s.dropna(); return s.mode().iloc[0] if len(s) else None
    SEG_MAP = [("category","u_top_category"),("city_name","u_top_city"),
               ("district_name","u_top_district"),("ad_type","u_top_ad_type")]
    enr_seg = enr[[c for c in ["item_id","category","city_name","district_name","ad_type"] if c in enr.columns]]
    tp_t = events_pos_test.merge(enr_seg, on="item_id", how="left")
    prof_specs = {out: (src, _mode) for src, out in SEG_MAP if src in tp_t.columns}
    user_prof  = tp_t.groupby("user_id").agg(**prof_specs).reset_index()
    missing = TEST_UIDS - set(user_prof["user_id"])
    if missing:
        cold_row = {"user_id": list(missing)}
        for src, out in SEG_MAP:
            if src in tp_t.columns:
                m = tp_t[src].mode(); cold_row[out] = m.iloc[0] if len(m) else None
        user_prof = pd.concat([user_prof, pd.DataFrame(cold_row)], ignore_index=True)

    last14_start = pd.Timestamp(TRAIN_DATE_END) - pd.Timedelta(days=14)
    snap_14 = snap_60d[snap_60d["date"] >= last14_start]
    pop  = gen_popularity_candidates(user_prof, enr, snap_14, 50, 100)
    cont = gen_content_candidates(user_prof, enr, top_n=50)
    cands = merge_candidates({"history":hist,"covis":cvis,"pop":pop,"content":cont}, cap_total=500)

    # Purchased filter
    pci_raw = pd.read_parquet(f"{CACHE_DIR}/pci_full.parquet",
                               columns=["user_id","item_id","purchased"])
    _purchased = pci_raw[pci_raw["purchased"]==True][["user_id","item_id"]].drop_duplicates().copy()
    _purchased["_drop"] = True
    cands = cands.merge(_purchased, on=["user_id","item_id"], how="left")
    cands = cands[cands["_drop"].isna()].drop(columns=["_drop"])
    del pci_raw, _purchased

    # Fallback for cold users
    missing_uids = TEST_UIDS - set(cands["user_id"])
    if missing_uids:
        pop_global = (snap_14.groupby("item_id")["contacts_24h"].sum()
                      .sort_values(ascending=False).head(100).index.tolist())
        pop_global = [it for it in pop_global if it in allowed_ab]
        fb = [{"user_id":u,"item_id":it,"src_history":float("nan"),"src_covis":float("nan"),
               "src_pop":1.0,"src_content":float("nan")} for u in missing_uids for it in pop_global[:100]]
        cands = pd.concat([cands, pd.DataFrame(fb)], ignore_index=True)

    CUTOFF = pd.Timestamp(TRAIN_DATE_END) + pd.Timedelta(seconds=1)
    pv_path = f"{CACHE_DIR}/events_pageview_30d.parquet"
    pv = pd.read_parquet(pv_path) if os.path.exists(pv_path) else \
         pd.DataFrame(columns=["user_id","item_id","event_ts","dwell_time_sec"])
    if "event_ts" in pv.columns:
        pv["event_ts"] = pd.to_datetime(pv["event_ts"])

    uf  = build_user_features(events_pos, pv, cutoff_ts=CUTOFF)
    itf = build_item_features(events_pos, snap_60d, enr, cutoff_ts=CUTOFF)
    full = add_cross_features(cands, uf, itf)
    full.to_parquet(test_mat_path, index=False)
    print(f"Built + saved test_matrix: {full.shape} | {time.time()-t0:.1f}s")
"""),

code("""\
# ---- Load both sub-models ----
with open(f"{CACHE_DIR}/model_hcm_feats.json")     as f: feats_hcm     = json.load(f)
with open(f"{CACHE_DIR}/model_general_feats.json")  as f: feats_general = json.load(f)
model_hcm     = lgb.Booster(model_file=f"{CACHE_DIR}/model_hcm.txt")
model_general = lgb.Booster(model_file=f"{CACHE_DIR}/model_general.txt")
print(f"model_hcm     feats: {len(feats_hcm)}")
print(f"model_general feats: {len(feats_general)}")
"""),

code("""\
# ---- Router: classify each user as HCM or OTHER based on u_top_city ----
if "u_top_city" in full.columns:
    full["_route"] = full["u_top_city"].apply(
        lambda c: "HCM" if (isinstance(c, str) and c.strip() in HCM_CITY_NAMES) else "OTHER"
    )
else:
    full["_route"] = "OTHER"

route_counts = full.groupby("user_id")["_route"].first().value_counts()
print("Router distribution (users):")
print(route_counts)
"""),

code("""\
# ---- Predict with the correct sub-model per user ----
def _align_and_predict(df, model, feat_list):
    \"\"\"Align df columns to model's feature list, predict.\"\"\"
    for c in feat_list:
        if c not in df.columns:
            df[c] = float("nan")
    X = df[feat_list].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")
    return model.predict(X, num_iteration=model.best_iteration)

t0 = time.time()
full["_pred"] = float("nan")

mask_hcm   = full["_route"] == "HCM"
mask_other = full["_route"] == "OTHER"

if mask_hcm.any():
    full.loc[mask_hcm, "_pred"] = _align_and_predict(
        full[mask_hcm].copy(), model_hcm, feats_hcm)

if mask_other.any():
    full.loc[mask_other, "_pred"] = _align_and_predict(
        full[mask_other].copy(), model_general, feats_general)

print(f"Predicted in {time.time()-t0:.0f}s | null preds: {full['_pred'].isna().sum():,}")
"""),

code("""\
from utils.diversify import diversify_top_k
from utils.submit import validate_submission, write_submission

# Top-30 per user -> diversify -> top-10
top30 = (full.sort_values(["user_id","_pred"], ascending=[True,False])
         .groupby("user_id").head(30)[["user_id","item_id","_pred"]]
         .rename(columns={"_pred":"score"}))

enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet",
                       columns=["item_id","seller_id","district_name"])
top10 = diversify_top_k(top30, enr, k=10,
                         max_per_seller=7, max_per_district=8,
                         freshness_boost=0.05, fresh_age_days=7)
print(f"top10 after diversify: {len(top10):,} | users: {top10['user_id'].nunique():,}")

# Validate + write
test_users_df = pd.read_parquet(f"{CACHE_DIR}/test_users.parquet")
TEST_UIDS = set(test_users_df["user_id"])
enr_full = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet", columns=["item_id"])
valid_items = set(enr_full["item_id"])

validate_submission(top10, valid_items, TEST_UIDS, k=10)
out = f"{CACHE_DIR}/submission_moe.csv"
write_submission(top10, out)
print(f"\\nMoE Submission ready: {out}")
print(f"File size: {os.path.getsize(out)/1024**2:.2f} MB")
"""),

]  # end NB07B_CELLS

# Write notebooks
nb06b_path = os.path.join(TRAINING_DIR, "06b_moe_ranker.ipynb")
nb07b_path = os.path.join(TRAINING_DIR, "07b_moe_predict.ipynb")

with open(nb06b_path, "w", encoding="utf-8") as f:
    json.dump(make_nb(NB06B_CELLS), f, ensure_ascii=False, indent=1)
print(f"Written: {nb06b_path}")

with open(nb07b_path, "w", encoding="utf-8") as f:
    json.dump(make_nb(NB07B_CELLS), f, ensure_ascii=False, indent=1)
print(f"Written: {nb07b_path}")
