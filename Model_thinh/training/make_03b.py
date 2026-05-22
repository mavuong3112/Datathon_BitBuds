"""Generate 03b_walkforward_cv.ipynb."""
import json, os

OUT = os.path.join(os.path.dirname(__file__), "03b_walkforward_cv.ipynb")

def md(src):
    return {"cell_type": "markdown", "id": f"md{abs(hash(src))%9999:04d}",
            "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "id": f"cd{abs(hash(src))%9999:04d}",
            "metadata": {}, "outputs": [], "execution_count": None, "source": src}

CELLS = [

md("""\
# 03b — Walk-Forward Cross-Validation (3 temporal folds)

| Fold | Train end | Valid window |
|------|-----------|-------------|
| 1 | 2026-01-13 | 14/01 → 09/02/2026 |
| 2 | 2026-02-09 | 10/02 → 13/03/2026 |
| 3 | 2026-03-13 | 14/03 → 09/04/2026 |

Output: `cache_drive/cv_results.json` với mean / std Recall@10 và NDCG@10.
"""),

code("print('Skipping pip install (local kernel).')"),

code("""\
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

from utils.covis import build_covis
from utils.candidates import (gen_history_candidates, gen_covis_candidates,
                               gen_popularity_candidates, gen_content_candidates,
                               merge_candidates)
from utils.features import build_user_features, build_item_features, add_cross_features
from utils.metrics import mean_recall_at_k, mean_ndcg_at_k
print("Imports OK.")
"""),

code("""\
CV_FOLDS = [
    ("2026-01-13", "2026-01-14", "2026-02-09"),
    ("2026-02-09", "2026-02-10", "2026-03-13"),
    ("2026-03-13", "2026-03-14", "2026-04-09"),
]

INTENT_WEIGHT = {
    "view_phone": 3.0, "contact_chat": 2.0,
    "contact_zalo": 2.0, "contact_sms": 2.0,
    "other_interaction": 1.0,
}
MONO_MAP = {
    "age_at_train_end": -1, "recency_evt_days": -1, "u_recency_days": -1,
    "i_CR_30d": 1, "i_contacts_24h_mean_30d": 1,
    "i_n_pos_30d": 1, "u_n_pos_30d": 1,
}
DROP_COLS = {"user_id","item_id","label","title","posted_date","expected_expired_date",
             "first_evt_date","last_evt_date","last_snap_date","project_id","_h"}
print("Constants OK.")
"""),

code("""\
# ---- Load shared data once ----
t0 = time.time()
print("Loading events_positive (19GB) — may take 2-3 min on first load ...")
events_pos = pd.read_parquet(f"{CACHE_DIR}/events_positive.parquet")
events_pos["event_ts"] = pd.to_datetime(events_pos["event_ts"])
events_pos["date"]     = pd.to_datetime(events_pos["date"])

snap_60d = pd.read_parquet(f"{CACHE_DIR}/snapshot_60d.parquet")
snap_60d["date"] = pd.to_datetime(snap_60d["date"])

enr = pd.read_parquet(f"{CACHE_DIR}/dim_listing_enriched.parquet")
pci_raw = pd.read_parquet(f"{CACHE_DIR}/pci_full.parquet",
                           columns=["user_id","item_id","purchased"])

print(f"events_pos: {len(events_pos):,} | enr: {len(enr):,} | {time.time()-t0:.1f}s")
allowed_a  = set(enr[enr["tier"] == "A"]["item_id"])
allowed_ab = set(enr[enr["tier"].isin(["A","B"])]["item_id"])

purchased_pairs = (pci_raw[pci_raw["purchased"] == True]
                   [["user_id","item_id"]].drop_duplicates().copy())
purchased_pairs["_drop"] = True
del pci_raw
print(f"purchased pairs to exclude: {len(purchased_pairs):,}")

enr_seg_cols = [c for c in ["item_id","category","city_name","district_name","ad_type"]
                if c in enr.columns]
enr_seg = enr[enr_seg_cols]
"""),

code("""\
def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None

SEG_MAP = [("category","u_top_category"), ("city_name","u_top_city"),
           ("district_name","u_top_district"), ("ad_type","u_top_ad_type")]

cv_results = []

for fold_i, (fold_train_end, fold_valid_start, fold_valid_end) in enumerate(CV_FOLDS, 1):
    fts = pd.Timestamp(fold_train_end)
    fvs = pd.Timestamp(fold_valid_start)
    fve = pd.Timestamp(fold_valid_end)
    print(f"\\n{'='*60}")
    print(f"FOLD {fold_i}: train<=  {fold_train_end}  |  valid {fold_valid_start} -> {fold_valid_end}")

    train_pos = events_pos[events_pos["date"] < fvs].copy()
    valid_pos = events_pos[(events_pos["date"] >= fvs) & (events_pos["date"] <= fve)]
    valid_users = set(train_pos["user_id"]) & set(valid_pos["user_id"])
    print(f"  train: {len(train_pos):,}  valid_users: {len(valid_users):,}")
    if not valid_users:
        print("  SKIP: no overlap users")
        continue

    gt = (valid_pos[valid_pos["user_id"].isin(valid_users)]
          .groupby("user_id")["item_id"].agg(set).to_dict())

    # --- Candidates ---
    t0 = time.time()
    pos_vu = train_pos[train_pos["user_id"].isin(valid_users)]
    hist   = gen_history_candidates(pos_vu, allowed_ab, top_n=30, cutoff_ts=fvs)

    covis_inp = train_pos[train_pos["item_id"].isin(allowed_a)][
        ["user_id","session_id","item_id","event_type","event_ts"]]
    covis = build_covis(covis_inp, allowed_items=allowed_a, top_k_per_item=20, time_decay=True)
    cvis  = gen_covis_candidates(pos_vu, covis, allowed_items=allowed_a, top_n=200, cutoff_ts=fvs)

    tp = pos_vu.merge(enr_seg, on="item_id", how="left")
    prof_specs = {out: (src, _mode) for src, out in SEG_MAP if src in tp.columns}
    user_prof  = tp.groupby("user_id").agg(**prof_specs).reset_index()

    snap_cut = snap_60d[snap_60d["date"] <= fts]
    snap_14  = snap_cut[snap_cut["date"] >= (fts - pd.Timedelta(days=14))]
    pop  = gen_popularity_candidates(user_prof, enr, snap_14, 50, 100)
    cont = gen_content_candidates(user_prof, enr, top_n=50)

    cands = merge_candidates({"history":hist,"covis":cvis,"pop":pop,"content":cont}, cap_total=500)

    _before = len(cands)
    cands = cands.merge(purchased_pairs, on=["user_id","item_id"], how="left")
    cands = cands[cands["_drop"].isna()].drop(columns=["_drop"])
    print(f"  cands: {len(cands):,} (excl. {_before-len(cands):,} purchased) | {time.time()-t0:.1f}s")

    cand_by_user = cands.groupby("user_id")["item_id"].apply(list).to_dict()
    r500 = mean_recall_at_k(cand_by_user, gt, k=500)
    print(f"  Recall@500 ceiling: {r500:.4f}")

    # --- Feature matrix ---
    t0 = time.time()
    pv_dummy = pd.DataFrame(columns=["user_id","item_id","event_ts","dwell_time_sec"])
    uf  = build_user_features(train_pos, pv_dummy, cutoff_ts=fvs)
    itf = build_item_features(train_pos, snap_cut, enr, cutoff_ts=fvs)
    full = add_cross_features(cands, uf, itf)

    gt_long = (valid_pos[valid_pos["user_id"].isin(valid_users)]
               [["user_id","item_id"]].drop_duplicates())
    gt_long = gt_long.copy(); gt_long["label"] = 1
    full = full.merge(gt_long, on=["user_id","item_id"], how="left")
    full["label"] = full["label"].fillna(0).astype("int8")
    print(f"  full: {full.shape}  pos={full['label'].mean()*100:.3f}%  | {time.time()-t0:.1f}s")

    # --- Train LightGBM lambdarank ---
    cat_cols  = [c for c in full.columns if c not in DROP_COLS and full[c].dtype == "object"]
    num_cols  = [c for c in full.columns if c not in DROP_COLS and full[c].dtype != "object"
                 and "datetime" not in str(full[c].dtype)]
    feat_cols = cat_cols + num_cols
    for c in cat_cols:
        full[c] = full[c].astype("category")

    full["_h"] = full["user_id"].map(lambda s: int(md5(str(s).encode()).hexdigest(), 16) % 100)
    tr_m = full["_h"] < 80;  va_m = full["_h"] >= 80
    X_tr = full.loc[tr_m, feat_cols];  y_tr = full.loc[tr_m, "label"].values
    g_tr = full.loc[tr_m].groupby("user_id", sort=False).size().values
    X_va = full.loc[va_m, feat_cols];  y_va = full.loc[va_m, "label"].values
    g_va = full.loc[va_m].groupby("user_id", sort=False).size().values

    params = dict(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=1.0, max_bin=255, seed=42, verbosity=-1,
        monotone_constraints=[MONO_MAP.get(c, 0) for c in feat_cols],
        monotone_constraints_method="advanced",
    )
    d_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr,
                       categorical_feature=cat_cols, free_raw_data=False)
    d_va = lgb.Dataset(X_va, label=y_va, group=g_va,
                       categorical_feature=cat_cols, reference=d_tr, free_raw_data=False)

    t0 = time.time()
    model = lgb.train(
        params, d_tr, num_boost_round=1000,
        valid_sets=[d_va], valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"  trained {time.time()-t0:.0f}s | best_iter={model.best_iteration}")

    full.loc[va_m, "_pred"] = model.predict(X_va, num_iteration=model.best_iteration)
    preds = {}
    for uid, grp in full[va_m].groupby("user_id", sort=False):
        preds[uid] = grp.nlargest(10, "_pred")["item_id"].tolist()
    gt_va = {u: gt[u] for u in preds if u in gt}

    r10 = mean_recall_at_k(preds, gt_va, k=10)
    n10 = mean_ndcg_at_k(preds,  gt_va, k=10)
    print(f"  FOLD {fold_i}  =>  Recall@10={r10:.4f}   NDCG@10={n10:.4f}")
    cv_results.append({
        "fold": fold_i, "train_end": fold_train_end,
        "valid_start": fold_valid_start, "valid_end": fold_valid_end,
        "recall_at_10": r10, "ndcg_at_10": n10,
        "ceiling_recall_500": float(r500), "n_valid_users": len(gt_va),
    })
    del train_pos, valid_pos, cands, full, model, covis

print("\\n" + "="*60)
print(f"CV  Mean Recall@10 : {np.mean([r['recall_at_10'] for r in cv_results]):.4f}")
print(f"    Std  Recall@10 : {np.std( [r['recall_at_10'] for r in cv_results]):.4f}")
print(f"    Mean NDCG@10   : {np.mean([r['ndcg_at_10']   for r in cv_results]):.4f}")
print(f"    Std  NDCG@10   : {np.std( [r['ndcg_at_10']   for r in cv_results]):.4f}")
"""),

code("""\
with open(f"{CACHE_DIR}/cv_results.json", "w") as f:
    json.dump({
        "folds": cv_results,
        "mean_recall_at_10": float(np.mean([r["recall_at_10"] for r in cv_results])),
        "std_recall_at_10":  float(np.std( [r["recall_at_10"] for r in cv_results])),
        "mean_ndcg_at_10":   float(np.mean([r["ndcg_at_10"]   for r in cv_results])),
        "std_ndcg_at_10":    float(np.std( [r["ndcg_at_10"]   for r in cv_results])),
    }, f, indent=2)
print(f"Saved: {CACHE_DIR}/cv_results.json")
"""),

]

NB = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "cells": CELLS,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, ensure_ascii=False, indent=1)
print(f"Written: {OUT}")
