"""Co-visitation matrix builder (sparse, session-based)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


INTENT_WEIGHT = {
    "view_phone": 3.0,
    "contact_chat": 2.0,
    "contact_zalo": 2.0,
    "contact_sms": 2.0,
    "other_interaction": 1.0,
}


def build_covis(events: pd.DataFrame, allowed_items: set | None = None,
                top_k_per_item: int = 20, time_decay: bool = True) -> dict:
    """Build co-visitation pairs from positive events grouped by session.

    events: cols = [user_id, session_id, item_id, event_ts, event_type]
    Returns dict: item_id -> list[(neighbor_item_id, score)] (top_k_per_item).
    """
    if allowed_items is not None:
        events = events[events["item_id"].isin(allowed_items)]
    events = events.sort_values(["session_id", "event_ts"])
    pair_scores: dict[tuple, float] = defaultdict(float)
    for _, grp in events.groupby("session_id", sort=False):
        items = grp["item_id"].tolist()
        ts = grp["event_ts"].tolist()
        n = len(items)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i], items[j]
                if a == b:
                    continue
                if time_decay:
                    dt_min = max((ts[j] - ts[i]).total_seconds() / 60.0, 0.0)
                    w = 1.0 / np.log2(dt_min + 2.0)
                else:
                    w = 1.0
                if a < b:
                    pair_scores[(a, b)] += w
                else:
                    pair_scores[(b, a)] += w

    item_neighbors: dict = defaultdict(list)
    for (a, b), s in pair_scores.items():
        item_neighbors[a].append((b, s))
        item_neighbors[b].append((a, s))

    for k, lst in item_neighbors.items():
        lst.sort(key=lambda x: -x[1])
        item_neighbors[k] = lst[:top_k_per_item]
    return dict(item_neighbors)


def covis_to_sparse(covis: dict, item_index: dict[str, int]) -> sp.csr_matrix:
    n = len(item_index)
    rows, cols, vals = [], [], []
    for src, neigh in covis.items():
        if src not in item_index:
            continue
        i = item_index[src]
        for nb, sc in neigh:
            if nb in item_index:
                rows.append(i)
                cols.append(item_index[nb])
                vals.append(sc)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def score_user_covis(user_history: list[tuple], covis: dict,
                     intent_weight_map: dict[str, float] | None = None) -> dict[str, float]:
    """Score candidate items for a user given history list of (item_id, event_type).

    Returns dict candidate_item -> score.
    """
    if intent_weight_map is None:
        intent_weight_map = INTENT_WEIGHT
    scores: dict[str, float] = defaultdict(float)
    history_set = {h[0] for h in user_history}
    for it, et in user_history:
        iw = intent_weight_map.get(et, 1.0)
        for nb, s in covis.get(it, []):
            if nb in history_set:
                continue
            scores[nb] += iw * s
    return dict(scores)
