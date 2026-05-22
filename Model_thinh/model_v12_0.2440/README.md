# Stage 12 Pipeline — Public LB 0.2440

**Created:** 2026-05-20
**Public LB:** 0.2440 (+0.0019 vs v6 baseline 0.2421, +0.0002 vs Stage 11)
**Status:** Cumulative best. Multi-model ensemble (LGBM + CatBoost + XGBoost) + bigger LGBM.

## Stage 12 Changes vs Stage 11

| Component | Stage 11 | Stage 12 |
|---|---|---|
| LightGBM seeds | 5 | **10** (variance reduction) |
| LGBM depth | 7 | **8** |
| LGBM num_leaves | 63 | **95** |
| CatBoost | 500 iter (kept from Stage 9) | same |
| XGBoost | ❌ | ✅ **NEW** rank:ndcg, 385 best iter |
| Blend | 0.6 LGBM + 0.4 CB | **0.4 LGBM + 0.3 CB + 0.3 XGB** |

## Score progression

| Stage | Public LB | Δ |
|---|---|---|
| v6 baseline | 0.2421 | — |
| Stage 8 | 0.2430 | +0.0009 |
| Stage 9 | 0.2434 | +0.0004 |
| Stage 10 | 0.2436 | +0.0002 |
| Stage 11 | 0.2438 | +0.0002 |
| **Stage 12** | **0.2440** | **+0.0002** |

Cumulative +0.0019 (~15 users on 8K public LB sample). Consistent trend (every stage adds ~0.0002), within noise margin but cumulative real.

## Reproduce

```bash
# Build candidates_lukewarm (Stage 11 already done)
# Reranker pipeline:
python model/07_rerank.py          # ~30 min (10 seeds, depth 8)
python model/07b_catboost.py       # ~3 min (GPU, reuse Stage 11)
python model/07d_xgboost.py        # ~3 min train + 1 min predict
BLEND_WEIGHT=0.4 BLEND_WEIGHT_CB=0.3 BLEND_WEIGHT_XGB=0.3 COLD_MODE=global python model/08_submit.py
cp datathon-chung-ket/submission.csv datathon-chung-ket/submission_stage12.csv
```

## XGBoost note

Train on GPU but prediction OOM (needs 13.6GB > 10GB free).
Workaround: switch model to CPU device + chunk prediction (2M rows/chunk).
See `_xgb_predict_only.py` for predict-only script.

## What's still NOT working

After 5 stages of optimization, lift only +0.0019. Why:
1. **66.3% cold users** all get same top-10 → fundamental limit
2. **5% public LB sample (8K users)** → noise hides any lift < 0.0003
3. **GBT class saturation** → LGBM/CatBoost/XGBoost share similar feature interactions
4. **Top-1 at 0.3293** likely uses fundamentally different approach (GNN, neural reranker, or cohort-based cold)
