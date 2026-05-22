# Stage 8 Pipeline — Public LB 0.2430 (best as of 2026-05-20)

**Created:** 2026-05-20
**Public LB:** 0.2430 (vs v6 baseline 0.2421, +0.0009)
**Status:** First WIN over v6 baseline. Root cause (retrieval leak) fixed.

## What's in this backup

| File | Purpose |
|------|---------|
| `01_extract.py` – `08_submit.py` | Pipeline scripts (Stage 8 versions) |
| `02b_als_only.py`, `06b_snapshot.py`, `09_covis.py` | Auxiliary scripts |
| `_compare_submissions.py`, `_estimate_raw.py` | Diagnostic scripts |
| `config.py` | Paths, LGBM_PARAMS, dates |
| `run_pipeline.py`, `validate.py` | Helpers |
| `submission_stage8_0.2430.csv` | Submission at 0.2430 |
| `06_stage8_run.log`, `07_stage8_run.log`, `08_stage8_run.log` | Run logs |

## Reproduce from scratch

```bash
# From d:/Datathon_Data/ — requires cache from 02-05 to exist
rm -f model/cache/features_train.parquet
rm -f model/cache/features_test.parquet
rm -f model/cache/ranked_predictions.parquet

# Train mode (build leak-free _train retrievers)
RETRIEVER_MODE=train python model/02_als_ease.py    # ~30 min (ALS)
RETRIEVER_MODE=train python model/03_itemcf.py      # ~5 min (GPU)
RETRIEVER_MODE=train python model/04_sasrec.py      # ~45 min (80 epochs)
RETRIEVER_MODE=train python model/05_merge_candidates.py  # ~1 min

# Stage 8 features + reranker (uses both train + test retrievers)
python model/06_features.py    # ~5 min (FEATURE_COLS=59)
python model/07_rerank.py      # ~14 min (5 seeds × 200-300 trees)
COLD_STRATEGY=global python model/08_submit.py   # ~2 min

# Output: datathon-chung-ket/submission.csv → submit to Kaggle
```

## Key innovation vs Stage 5/v6: Leak-free retrievers

### Root cause identified
v6 best_iter=1 because ALS/EASE/ItemCF/SASRec embeddings trained on FULL train (incl. val period).
Embeddings leak val-period interactions → val NDCG@10 saturates at 1.0 → early stopping fires at iter 1.

### Architecture (TWO sets of retrievers)
- **`_train` retrievers** (NEW): trained on `user_item_pos_train.parquet` (events < VAL_SPLIT, Tết-filtered)
  → outputs `*_candidates_train.parquet` → builds `candidates_train.parquet` (19.4M rows)
  → feeds `features_train.parquet` for reranker training
- **`_test` retrievers** (existing): trained on full TRAIN window
  → outputs `candidates.parquet` (21.8M rows)
  → feeds `features_test.parquet` for inference

→ Reranker learns CLEAN signal, inference uses MAXIMUM retriever knowledge.

## Result: best_iter from 1 → 290+

| Metric | Stage 5 (best_iter=1) | **Stage 8 (best_iter=290)** |
|---|---|---|
| Val NDCG@10 iter 1 | 1.000 (leak) | 0.997 (real signal) |
| Val NDCG growth | Flat | 0.9971 → 0.9976 over 290 trees |
| Features used | ~5 (1 tree) | 15+ (290 trees) |
| Top FI #1 | days_since_ui (54K) | item_velocity_log (601K) |

## Feature Importance Top 15 (Stage 8)

| Rank | Feature | Gain | Group |
|---|---|---|---|
| 1 | item_velocity_log | 601K | Item velocity |
| 2 | unique_items_log | 188K | User activity |
| 3 | trend_pos_log | 181K | Item popularity |
| 4 | trend_cvr | 104K | Item conversion |
| 5 | days_since_last | 83K | User recency |
| 6 | days_since_posted | 79K | Item freshness |
| 7 | total_pos_log | 67K | User activity |
| 8 | **district_match** | 63K | **NEW Stage 8 match feature** |
| 9 | active_days_inter | 38K | User activity |
| 10 | blend_score | 35K | Retrieval blend |
| 11 | category_match | 27K | Match feature |
| 12 | **price_match** | 21K | **NEW Stage 8 match feature** |
| 13 | active_span_days | 20K | User activity |
| 14 | ease_score_norm | 17K | Retrieval ALS |
| 15 | days_since_ui | 14K | Pair temporal (was leak king, now minor) |

## Feature subset (59 features — DOWN from Stage 3's 75)

**KEPT** (clean signals):
- 8 retrieval: als/ease/itemcf/sasrec norm, blend_score, source_count, is_repeat, repeat_count
- 11 pos: pos_count_log, days_since_ui, n_view_phone/chat/zalo/sms/other, intent_score_log, explicit_contact_log, user_intent_ratio, is_weekend_interaction
- 4 inter: total_leads_log, total_chat_turns_log, ever_purchased, active_days_inter
- 2 hist_decay: hist_decay_score, hist_decay_total_log
- 21 item: cat/adtype/seller/price/city/district_enc + area_bucket + area_sqm_log + images_count_log + bedrooms_filled + days_since_posted + has_project_id + item_cvr + total_events_log + unique_users_log + avg_dwell + trend_pos_log + trend_cvr + item_velocity_log + is_renewal_week + age_boost_cat
- 7 user: pref_cat_enc, pref_city_enc, total_pos_log, unique_items_log, days_since_last, active_span_days, avg_pos_per_item
- 6 match: category_match, city_match, price_match, adtype_match, district_match, seller_match

**DROPPED from Stage 3** (75 → 59):
- ❌ Dwell × 3 (max_dwell_pair_log, has_consider_pv, n_pageview_log) — corr <0.10 after leak fix
- ❌ Snapshot × 5 (views_24h_log, contacts_24h_log, contact_rate_24h, pct_days_contact, listing_age_days_snap)
- ❌ Channel ratios × 4 (user_view_phone_ratio, user_chat_ratio, ...)
- ❌ pref_channel_enc
- ❌ n_chat_verified (inter table full-history leak)
- ❌ same_project_score (gain thấp)
- ❌ bedrooms_match (corr nhỏ)

## Cold strategy (locked-in)

- 66.3% test users are cold (no pos history) → global popular top-10
- A/B tested ALL alternatives, all FAILED:
  - cvr (trend_pos × trend_cvr): -0.0650 — niche items
  - session-based (3K cold-with-click): -0.0067 — click intent noisy
- **Stage 8 keeps cold = raw global popular by trend_pos**

## Public LB volatility (5% sample = 8K users)

Score noise ~0.0001-0.0002. Stage 8's +0.0009 is at edge of noise (~7 users).
Private LB (95% = 153K) will give more reliable judgment.

## Submission History

| File | Score | Notes |
|------|-------|-------|
| stage3_session.csv | 0.2316 | -0.0105 (session cold strategy bad) |
| stage3_cvr.csv | 0.1771 | -0.0650 (cvr cold catastrophic) |
| stage3_global.csv | 0.2383 | -0.0038 (leaky dwell) |
| stage4_dwellfix.csv | 0.2403 | -0.0018 (clean dwell, still leaky retrievers) |
| stage5_500trees.csv | 0.2388 | -0.0033 (500 trees overfit) |
| **stage8.csv** | **0.2430** | **+0.0009** ← this backup |
| v6_baseline | 0.2421 | Previous baseline |

## What NOT to do (lessons from failed attempts)

- Do NOT change cold strategy (cvr, session, conversion-weighted) — all FAIL
- Do NOT add dwell features without temporal split (leak)
- Do NOT add inter table features without filter (leak)
- Do NOT use 500 trees (overfits when features have leak)
- Do NOT skip retriever retrain on pre-val data (the root cause)
