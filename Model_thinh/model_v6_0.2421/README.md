# Backup: v6 Pipeline — Public Score 0.2421

**Created:** 2026-05-18
**Status:** Current best public score

## What's in this backup

| File | Purpose |
|------|---------|
| `01_extract.py` – `09_covis.py` | Pipeline scripts (steps 1–8 active, 09 unused) |
| `config.py` | Paths, LGBM_PARAMS, dates |
| `run_pipeline.py`, `validate.py` | Helpers |
| `submission_v6b_0.2421.csv` | First submission at 0.2421 — Option B (no cat boost) |
| `submission_v6d_0.2421.csv` | Second submission at 0.2421 — Option B + cat match boost |
| `06_v6_run.log`, `07_v6_run.log`, `08_v6d_run.log` | Run output for verification |

Both submissions scored **0.2421**: cat_boost reordered ties but didn't change top-10 set membership → Recall@10 unchanged.

## Reproduce from scratch

```bash
# From d:/Datathon_Data/ — assumes steps 02-04 cache already built
rm -f model/cache/features_train.parquet
rm -f model/cache/features_test.parquet
rm -f model/cache/ranked_predictions.parquet

python model/06_features.py    # ~3 min
python model/07_rerank.py      # ~3 min (5 seeds, best_iter=1)
python model/08_submit.py      # ~2 min

# Output: datathon-chung-ket/submission.csv → submit to Kaggle
```

## Key implementation details vs v1 baseline (0.2184)

### Step 6 — `06_features.py`
- Single `pos_feat` from full history (LEAKY but correct direction)
- `fillna(0)` for all interaction cols, `fillna(999)` for days_since_ui
- New features added: n_zalo, n_sms, intent_score_log, explicit_contact_log, user_intent_ratio, is_weekend_interaction, item_velocity_log (1-tree LGBM ignores them, kept for future use)

### Step 7 — `07_rerank.py`
- 5-seed ensemble (variance reduction)
- `blend_score` passed through to ranked_predictions (tiebreaker for cold users)
- `best_iter=1` for all seeds (expected — `days_since_ui` is perfect val separator)
- Top feature: `days_since_ui` (gain ~113K)

### Step 8 — `08_submit.py`
- `cold_users = test_users - pos_users` — detect 107,066 users (66.3%)
- Freshness boost: `score_range × 0.015` for listings ≤7 days
- **Category match boost: `score_range × 0.006` for warm users where item.category == user.pref_category** (v6d only; same score as without)
- Sort by (lgbm_score, blend_score) — blend tiebreaker
- Category-weighted interleaved fallback pool (only 325 slots needed)
- **CRITICAL: Override all 107,066 cold users with global popular top-10** (v1 behavior — this is what worked)

## What failed (do NOT re-introduce)

| Attempt | Score | Lesson |
|---------|-------|--------|
| NaN for cold interaction cols + override_repeat (v5) | 0.0662 | Reversed ranking direction: NaN became "positive" signal |
| Cold pool diverse 50 items + LGBM rank for cold (v6 pre-Option B) | 0.2018 | Category-segmented popular < global popular for cold users |
| Lukewarm cat-mode personalization (v6c) | 0.2259 | Same category-segmented failure pattern |
| Cat_match boost only (v6d, no v6+B baseline) | 0.2421 | Boost reorders ties but doesn't change top-10 set |

## Category mapping (verified, see memory)
- 1010 = Căn hộ/Chung cư (45,651)
- 1020 = Nhà ở (106,746 — dominant by count)
- 1030 = Văn phòng/Mặt bằng (22,678)
- 1040 = Đất (40,056)
- 1050 = Phòng trọ (33,119 — dominant by engagement)

## Open improvement directions (ranked by safety)

1. **Conversion-weighted popular for cold** — `trend_pos × (trend_pos / trend_events)` instead of raw `trend_pos`
2. **Bigger cat_boost** — push boost from 0.6% to 5-10% of range to actually change top-10 set membership
3. **Diversify warm user candidates** — add cold pool items to warm user candidate list to introduce exploration
4. **Remove leakage features from training** — drop `days_since_ui` and `is_repeat` from FEATURE_COLS → forces LGBM to train 100+ trees on retrieval scores (HIGH RISK / HIGH UPSIDE)

## Backup workflow for future scores

When a new high score is achieved:
```bash
SCORE=0.2X  # e.g. 0.2500
mkdir -p model_v7_${SCORE}
cp model/*.py model_v7_${SCORE}/
cp datathon-chung-ket/submission.csv model_v7_${SCORE}/submission_v7_${SCORE}.csv
# Copy current run logs
cp model/cache/06_v*.log model/cache/07_v*.log model/cache/08_v*.log model_v7_${SCORE}/ 2>/dev/null
# Document what changed in README.md
```
