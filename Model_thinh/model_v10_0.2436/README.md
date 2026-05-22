# Stage 10 Pipeline — Public LB 0.2436 (best as of 2026-05-20)

**Created:** 2026-05-20
**Public LB:** 0.2436 (vs v6 baseline 0.2421, +0.0015)
**Status:** Cumulative best. Behavioral features from clustering notebook added.

## What's in this backup

| File | Purpose |
|------|---------|
| `01_extract.py` – `08_submit.py` | Pipeline scripts (Stage 10 versions) |
| `07b_catboost.py` | CatBoost YetiRank warm-track |
| `07c_cold_track.py` | Cold-Track Specialist (aborted by A/B safety in Stage 9) |
| `06b_snapshot.py` | Snapshot features (deprecated but kept) |
| `config.py` | Paths, LGBM_PARAMS, dates |
| `submission_stage10_0.2436.csv` | Submission at 0.2436 |
| Various `*_stage10_run.log` | Run logs |

## Stage 10 vs Stage 9 changes

**+7 behavioral features inspired by clustering notebook** (eda_category_1010_1020_clustering.ipynb):

### Item-level (2 from LISTING_CLUSTER_COLS)
- `item_contact_rate_pct` = pos_events / pageviews (per item)
- `item_repeat_viewer_pct` = 1 - (unique_users / total_events)

### User-level (5 from USER_CLUSTER_COLS)
- `user_n_districts` = COUNT DISTINCT districts per user
- `user_district_entropy` = -Σ p log p of district distribution
- `user_pct_night` = % of events between 22h-6h
- `user_avg_dwell_sec` = AVG dwell of user's pageviews
- `n_sessions_log` = log1p(unique sessions per user)

## Score progression

| Submission | Public LB | Δ |
|------|-------|---|
| v6 baseline | 0.2421 | — |
| Stage 8 (leak-free retrievers) | 0.2430 | +0.0009 |
| Stage 9 (CatBoost blend, 0.6/0.4) | 0.2434 | +0.0004 |
| **Stage 10 (behavioral features)** | **0.2436** | **+0.0002** |

## Architecture

### Retrieval (4 retrievers, leak-free training)
- BPR/ALS (factors=256)
- EASE/ALS (factors=256, implicit)
- ItemCF (BPR embeddings)
- SASRec (hidden=128, 2 layers, 4 heads, 80 epochs)

Train mode (events < VAL_SPLIT) → candidates_train.parquet (19.4M rows)
Test mode (full data) → candidates.parquet (21.8M rows)

### Reranker
- LightGBM LambdaRank, 290+ trees with early stopping
- CatBoost YetiRank, 500 iter (blend 0.6 LGBM / 0.4 CatBoost)
- 70 features

### Cold strategy
- 107K cold users (66.3%) → global popular top-10 by trend_pos
- (Cold-Track Specialist tried, aborted by A/B safety: 0/10 overlap with global)

## Reproduce from scratch

```bash
# Existing cache should have: user_item_pos.parquet, user_item_seq.parquet, items.parquet, etc.

# Step 1: Extract Stage 10 user_behavioral.parquet
python model/01_extract.py    # ~4 min for user_behavioral

# Step 2: Build features
python model/06_features.py test   # ~3 min (features_test.parquet)
python model/06_features.py train  # ~3 min (features_train.parquet)

# Step 3: Train rerankers
python model/07_rerank.py          # ~18 min (LightGBM 5 seeds)
python model/07b_catboost.py       # ~2.5 min (CatBoost GPU)

# Step 4: Submit
BLEND_WEIGHT=0.6 COLD_MODE=global python model/08_submit.py
cp datathon-chung-ket/submission.csv datathon-chung-ket/submission_stage10.csv
```

## Top 15 Feature Importance (Stage 10 LightGBM)

| Rank | Feature | Gain |
|---|---|---|
| 1 | item_velocity_log | 642K |
| 2 | trend_pos_log | 213K |
| 3 | unique_items_log | 165K |
| 4 | days_since_posted | 89K |
| 5 | days_since_last | 84K |
| 6 | total_pos_log | 82K |
| 7 | district_match | 55K |
| 8 | active_days_inter | 34K |
| 9 | blend_score | 34K |
| 10 | trend_cvr | 33K |
| 11 | category_match | 33K |
| 12 | price_match | 23K |
| 13 | **user_district_entropy** ⭐NEW | **19.5K** |
| 14 | active_span_days | 18K |
| 15 | **n_sessions_log** ⭐NEW | **17K** |

→ 2 of 7 new Stage 10 features made top-15. Confirmed clustering notebook insight.

## Public LB observation: 5% sample noise

Public LB = ~8K users sample. Score noise ~0.0001-0.0002.
Stage 10 lift +0.0002 is within noise margin, but cumulative +0.0015 over v6 is real signal.

Private LB (95% = 153K users) will be more reliable judgment.

## What NOT to do (proven failures)

- ❌ Cold strategy `trend_pos × trend_cvr` (stage3_cvr -0.0650)
- ❌ Cold session-based personalization (stage3_session -0.0105)
- ❌ 500 trees with leaky features (stage5 -0.0033)
- ❌ Cold-Track Specialist (A/B 0/10 overlap → aborted in Stage 9)

## Next direction (Stage 11)

**User feedback**: 0.2436 still far from top 1 (0.3293). Need bigger architecture change.

Ideas being explored:
- Lukewarm expansion: extend retrieval mappings to include users with browse-only (not just contact) events
- Hard negative mining
- Multi-task learning (separate models for chat vs phone)
- Bigger LightGBM (depth 10, num_leaves 127)
- Ensemble more bosters (LightGBM + CatBoost + XGBoost)
