"""
EDA Pipeline Config — Chợ Tốt BĐS Datathon
RAM budget: 32 GB system → cap DuckDB at 20 GB, pandas operations stay < 8 GB.
"""
import os

DATA_ROOT        = "d:/Datathon_Data"
EDA_DIR          = os.path.join(DATA_ROOT, "eda")
OUTPUT_DIR       = os.path.join(EDA_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Parquet directories -----------------------------------------------
DIM_LISTING_DIR      = os.path.join(DATA_ROOT, "dim_listing")
FACT_SNAPSHOT_DIR    = os.path.join(DATA_ROOT, "fact_listing_snapshot")
FACT_INTER_DIR       = os.path.join(DATA_ROOT, "fact_post_contact_interactions")
FACT_EVENTS_DIR      = os.path.join(DATA_ROOT, "fact_user_events")

# --- Domain constants --------------------------------------------------
CATEGORIES = {
    1010: "Phòng trọ/Cho thuê",
    1020: "Căn hộ/Chung cư",
    1030: "Nhà ở",
    1040: "Đất nền",
    1050: "Dự án mới",
}
CAT_COLORS = {
    1010: "#4C72B0", 1020: "#DD8452",
    1030: "#55A868", 1040: "#C44E52", 1050: "#8172B2",
}

# SQL-ready category filter (drops anomalous codes like 3030, 6020, 8030)
CATEGORY_FILTER = "category IN (1010, 1020, 1030, 1040, 1050)"

POSITIVE_EVENTS = [
    "view_phone", "contact_chat",
    "other_interaction", "contact_zalo", "contact_sms",
]

# Business-valid 100% nulls (do NOT flag as data quality issues)
VALID_NULLS = {
    1030: ["bedrooms"],
    1040: ["bedrooms", "furnishing"],
    1050: ["bedrooms", "legal_status"],
}

# --- Time boundaries ---------------------------------------------------
DIM_START  = "2024-09-15"   # earliest posting in dim_listing
FACT_START = "2025-11-09"   # start of all fact tables (train period)
FACT_END   = "2026-04-09"   # end of train period
