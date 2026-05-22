"""
Stage 14 Hook #1: District transition probability matrix.

Insight: Users cross-search adjacent districts (Gò Vấp ↔ Q12, Q7 ↔ Nhà Bè).
We want a SOFT probability — not a hard adjacency table — so the reranker can
learn priority among multiple candidate districts.

Method:
  1. From user_item_seq (browsing+contact events), order by (user, ts).
  2. Compute prev_district for each event (within 24h session window).
  3. Count global transitions A→B (excluding self A→A — same-district is handled
     by the existing district_match feature).
  4. Normalize per-source: P(B|A) = count(A→B) / sum_x count(A→x).
  5. Save district_transition.pkl = {dist_A: {dist_B: prob}}.

Feature `district_transition_score` (added in 06_features.py):
  - 1.0 if item.district == user.pref_district (perfect match)
  - else: P(item.district | user.pref_district), 0.0 if no observed transition

Why session window 24h:
  Cross-district browsing within the same day captures intentional comparison
  shopping. Multi-day gaps blend in unrelated browsing → noise.
"""
import sys, os, time, gc, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from collections import defaultdict
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

OUT_PKL = f"{CACHE_DIR}/district_transition.pkl"
SESSION_GAP_HOURS = 24

print(f"[STAGE 14 Hook #1] district transition matrix → {OUT_PKL}")

print(f"{elapsed()} Loading user_item_seq + items.district …")
seq = pd.read_parquet(f"{CACHE_DIR}/user_item_seq.parquet")[['user_id','item_id','event_ts']]
items_dist = pd.read_parquet(f"{CACHE_DIR}/items.parquet")[['item_id','district_name']]
seq = seq.merge(items_dist, on='item_id', how='left')
seq = seq[seq['district_name'].notna()].copy()
seq['event_ts'] = pd.to_datetime(seq['event_ts'])
print(f"{elapsed()}   events with district: {len(seq):,}")

print(f"{elapsed()} Sorting + computing prev_district …")
seq = seq.sort_values(['user_id','event_ts']).reset_index(drop=True)
seq['prev_user']     = seq['user_id'].shift(1)
seq['prev_district'] = seq['district_name'].shift(1)
seq['prev_ts']       = seq['event_ts'].shift(1)
seq['same_user']     = (seq['prev_user'] == seq['user_id'])
seq['gap_hours']     = (seq['event_ts'] - seq['prev_ts']).dt.total_seconds() / 3600.0
trans = seq[
    seq['same_user'] &
    (seq['gap_hours'] <= SESSION_GAP_HOURS) &
    (seq['district_name'] != seq['prev_district'])  # only cross-district
][['prev_district','district_name']].copy()
del seq; gc.collect()
print(f"{elapsed()}   cross-district transitions (≤{SESSION_GAP_HOURS}h gap): {len(trans):,}")

print(f"{elapsed()} Counting transitions …")
counts = (trans.groupby(['prev_district','district_name'])
                .size().reset_index(name='n'))
print(f"{elapsed()}   unique (A→B) pairs: {len(counts):,}")

print(f"{elapsed()} Normalizing per source …")
src_totals = counts.groupby('prev_district')['n'].sum().to_dict()
counts['prob'] = counts.apply(lambda r: r['n'] / src_totals[r['prev_district']], axis=1)

# Build dict: {src_district: {tgt_district: prob, ...}}
transition = defaultdict(dict)
for _, row in counts.iterrows():
    transition[row['prev_district']][row['district_name']] = float(row['prob'])
transition = dict(transition)

print(f"{elapsed()}   source districts: {len(transition):,}")
top_examples = sorted(transition.items(), key=lambda kv: -src_totals.get(kv[0], 0))[:3]
for src, tgts in top_examples:
    top3 = sorted(tgts.items(), key=lambda x: -x[1])[:3]
    print(f"     {src}: " + ", ".join([f"{d}({p:.3f})" for d, p in top3]))

with open(OUT_PKL, 'wb') as f:
    pickle.dump(transition, f)
print(f"{elapsed()} Saved → {OUT_PKL}")
