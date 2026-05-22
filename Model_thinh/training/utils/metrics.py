"""Recall@K and NDCG@K for top-K recommendation evaluation."""
from __future__ import annotations

import numpy as np


def recall_at_k(pred: list[str], gt: set[str], k: int = 10) -> float:
    if not gt:
        return 0.0
    topk = pred[:k]
    hits = sum(1 for p in topk if p in gt)
    return hits / len(gt)


def ndcg_at_k(pred: list[str], gt: set[str], k: int = 10) -> float:
    if not gt:
        return 0.0
    dcg = 0.0
    for i, p in enumerate(pred[:k]):
        if p in gt:
            dcg += 1.0 / np.log2(i + 2)
    ideal_n = min(len(gt), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def mean_recall_at_k(preds: dict[str, list[str]], gts: dict[str, set[str]], k: int = 10) -> float:
    users = [u for u in gts if gts[u]]
    if not users:
        return 0.0
    return float(np.mean([recall_at_k(preds.get(u, []), gts[u], k) for u in users]))


def mean_ndcg_at_k(preds: dict[str, list[str]], gts: dict[str, set[str]], k: int = 10) -> float:
    users = [u for u in gts if gts[u]]
    if not users:
        return 0.0
    return float(np.mean([ndcg_at_k(preds.get(u, []), gts[u], k) for u in users]))


def recall_at_k_per_user(preds: dict[str, list[str]], gts: dict[str, set[str]], k: int = 10) -> dict[str, float]:
    return {u: recall_at_k(preds.get(u, []), gt, k) for u, gt in gts.items() if gt}
