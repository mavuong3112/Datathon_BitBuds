# Stage 11 Pipeline — Public LB 0.2438 (best as of 2026-05-20)

**Created:** 2026-05-20
**Public LB:** 0.2438 (+0.0017 vs v6 baseline 0.2421, +0.0002 vs Stage 10)
**Status:** Cumulative best. Lukewarm coverage expansion via retraining all retrievers on broader matrix.

## Stage 11 = Lukewarm Coverage Expansion

### Key change vs Stage 10
- **user_item_pos_lukewarm.parquet**: 31.3M rows / 989K users (vs 13M rows / 705K users in pos)
  - Includes pageviews (weight=2) + other_interaction (weight=1) + positive contacts (weight=10)
  - Login users only (need user_id)
- **All 4 retrievers retrained** on lukewarm matrix:
  - BPR + ALS (factors=256): 50 min training
  - ItemCF: 5 min
  - SASRec (80 epochs, 994K user sequences): 65 min
- **candidates_lukewarm.parquet**: 27.6M rows (vs 21.8M Stage 10)
- **TEST_CANDS_MODE=lukewarm** env var added to 06_features.py

### Coverage stats
- Stage 10 warm users: 54K
- Stage 11 warm users: **57K (+3K)** lukewarm signal captured
- Cold users: 104K (down from 107K, marginal)

## Score progression

| Stage | Public LB | Δ vs prev | Cum vs v6 |
|---|---|---|---|
| v6 baseline | 0.2421 | — | — |
| Stage 8 (leak-free retrievers) | 0.2430 | +0.0009 | +0.0009 |
| Stage 9 (CatBoost blend) | 0.2434 | +0.0004 | +0.0013 |
| Stage 10 (behavioral features) | 0.2436 | +0.0002 | +0.0015 |
| **Stage 11 (lukewarm expansion)** | **0.2438** | **+0.0002** | **+0.0017** |

## Reproduce from scratch

```bash
# All retriever caches needed (built sequentially)
RETRIEVER_MODE=lukewarm python model/02_als_ease.py   # ~60 min (ALS 50 min)
RETRIEVER_MODE=lukewarm python model/03_itemcf.py     # ~3 min
RETRIEVER_MODE=lukewarm python model/04_sasrec.py     # ~65 min (80 epochs)
RETRIEVER_MODE=lukewarm python model/05_merge_candidates.py   # ~2 min

# Reranker pipeline (TEST_CANDS_MODE=lukewarm)
TEST_CANDS_MODE=lukewarm python model/06_features.py test  # ~4 min
python model/06_features.py train                          # ~3 min
python model/07_rerank.py                                  # ~19 min (5 seeds)
python model/07b_catboost.py                               # ~3 min (GPU)
BLEND_WEIGHT=0.6 COLD_MODE=global python model/08_submit.py
cp datathon-chung-ket/submission.csv datathon-chung-ket/submission_stage11.csv
```

## Why improvement is slow

1. **Public LB = 5% sample (8K users)** → noise ±0.0002 → many our +0.0002 lifts are within noise
2. **66.3% cold users** all get IDENTICAL top-10 (raw global popular) — locked-in ceiling
3. **Warm side feature engineering** hitting diminishing returns: signal is mostly captured by retrieval scores + item velocity
4. **Architecture is fundamentally LightGBM-based** — can't beat by 10%+ vs top-1 without different model class

## What's NOT explored yet (potential big lifts)

- **GNN retrieval (LightGCN)**: 3-5h, replaces ALS with stronger neural CF
- **Two-tower neural model**: user encoder + item encoder, 5-7h
- **Multi-objective training**: separate ranker for chat vs phone contacts
- **Cold pool cohort diversification**: cluster cold users by behavioral hints
- **Add XGBoost to ensemble**: 3-way blend, ~2h
- **Hard negative mining**: improve training data quality
- **Bigger LightGBM**: depth 10, leaves 127, 10+ seeds
