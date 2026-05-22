"""
Offline validation: Recall@10 and NDCG@10 on temporal split.
Train = Nov→Feb, Validate = Mar→Apr (before TRAIN_END).
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

def recall_at_k(pred_items, true_items, k=10):
    if not true_items:
        return 0.0
    hits = len(set(pred_items[:k]) & set(true_items))
    return hits / len(true_items)

def ndcg_at_k(pred_items, true_items, k=10):
    if not true_items:
        return 0.0
    true_set = set(true_items)
    dcg = sum(1.0/np.log2(i+2) for i, it in enumerate(pred_items[:k]) if it in true_set)
    idcg= sum(1.0/np.log2(i+2) for i in range(min(k, len(true_items))))
    return dcg / idcg if idcg > 0 else 0.0

print(f"{elapsed()} Loading ranked predictions and ground truth …")
ranked = pd.read_parquet(f"{CACHE_DIR}/ranked_predictions.parquet")
pos    = pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet")

val_split_dt = pd.Timestamp(VAL_SPLIT)
train_end_dt = pd.Timestamp(TRAIN_END)

pos['last_ts'] = pd.to_datetime(pos['last_ts'])
# Ground truth: positive interactions in the validation window (Mar-Apr)
ground_truth = (pos[(pos['last_ts'] >= val_split_dt) & (pos['last_ts'] <= train_end_dt)]
                .groupby('user_id')['item_id']
                .apply(list)
                .to_dict())

# Only evaluate users who have ground truth
eval_users = [u for u in ranked['user_id'].unique() if u in ground_truth]
print(f"{elapsed()} Eval users: {len(eval_users):,}  (have GT in Mar-Apr)")

pred_dict = (ranked.sort_values('lgbm_score', ascending=False)
             .groupby('user_id')['item_id']
             .apply(list)
             .to_dict())

recalls, ndcgs = [], []
for uid in eval_users:
    preds = pred_dict.get(uid, [])
    truth = ground_truth[uid]
    recalls.append(recall_at_k(preds, truth))
    ndcgs.append(ndcg_at_k(preds, truth))

print(f"\n{'='*50}")
print(f"OFFLINE VALIDATION RESULTS (temporal split)")
print(f"  Val window : {VAL_SPLIT} → {TRAIN_END}")
print(f"  Eval users : {len(eval_users):,}")
print(f"  Recall@10  : {np.mean(recalls):.6f}")
print(f"  NDCG@10    : {np.mean(ndcgs):.6f}")
print(f"{'='*50}")

# Distribution
recalls_arr = np.array(recalls)
print(f"\n  Recall@10 distribution:")
print(f"    p0  = {recalls_arr.min():.4f}")
print(f"    p25 = {np.percentile(recalls_arr,25):.4f}")
print(f"    p50 = {np.percentile(recalls_arr,50):.4f}")
print(f"    p75 = {np.percentile(recalls_arr,75):.4f}")
print(f"    p100= {recalls_arr.max():.4f}")
print(f"    >0  = {(recalls_arr>0).mean()*100:.1f}% of users")
