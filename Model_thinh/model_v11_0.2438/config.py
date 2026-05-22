"""Model pipeline config — Chợ Tốt BĐS Datathon."""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = "d:/Datathon_Data"
MODEL_DIR  = os.path.join(ROOT, "model")
CACHE_DIR  = os.path.join(MODEL_DIR, "cache")
SUBMIT_DIR = os.path.join(ROOT, "datathon-chung-ket")
os.makedirs(CACHE_DIR, exist_ok=True)

DIM_DIR    = os.path.join(ROOT, "dim_listing")
EVT_DIR    = os.path.join(ROOT, "fact_user_events")
SNAP_DIR   = os.path.join(ROOT, "fact_listing_snapshot")
INTER_DIR  = os.path.join(ROOT, "fact_post_contact_interactions")
TEST_FILE  = os.path.join(ROOT, "test", "test_users.parquet")
SAMPLE_SUB = os.path.join(SUBMIT_DIR, "sample_submission.csv")
SUBMIT_OUT = os.path.join(SUBMIT_DIR, "submission.csv")

# ── Domain constants ───────────────────────────────────────────────────────────
CATEGORIES = {1010: "Phòng trọ", 1020: "Căn hộ", 1030: "Nhà ở",
              1040: "Đất nền",   1050: "Dự án mới"}
CATEGORY_FILTER = "category IN (1010,1020,1030,1040,1050)"
POSITIVE_EVENTS = ["view_phone","contact_chat","other_interaction",
                   "contact_zalo","contact_sms"]
POS_STR = ", ".join(f"'{e}'" for e in POSITIVE_EVENTS)

# ── Time boundaries ────────────────────────────────────────────────────────────
TRAIN_START = "2025-11-09"
TRAIN_END   = "2026-04-09"   # inclusive — last day before test window
VAL_SPLIT   = "2026-03-01"   # offline validation: train=Nov-Feb, val=Mar-Apr
RECENT_DAYS = 28             # popularity window = last 28 days of train

# ── ALS hyperparameters ────────────────────────────────────────────────────────
ALS_FACTORS      = 512
ALS_ITERATIONS   = 50
ALS_REGULARIZE   = 0.01
ALS_ALPHA        = 40        # confidence = 1 + alpha * count
RECENCY_DECAY    = 0.005     # exp(-decay * days) → ~140-day half-life

# ── EASE ───────────────────────────────────────────────────────────────────────
EASE_LAMBDA = 200.0          # regularization

# ── Candidate generation ───────────────────────────────────────────────────────
N_ALS        = 100
N_EASE       = 100
N_ITEMCF     = 100
N_SASREC     = 100
N_TRENDING   = 50
MAX_CANDS    = 500           # cap per user before reranking

# ── LightGBM ───────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_eval_at":      [10],
    "boosting_type":     "gbdt",
    "device":            "cpu",
    "num_threads":       -1,    # use all CPU cores
    "num_leaves":        63,
    "max_depth":         7,
    "learning_rate":     0.05,
    "n_estimators":      500,   # fixed rounds — no early stopping (val NDCG saturates at 1 due to feature leakage)
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "min_child_samples": 20,
    "verbose":           -1,
}

# ── DuckDB ─────────────────────────────────────────────────────────────────────
DUCKDB_MEMORY  = "20GB"
DUCKDB_THREADS = 4
