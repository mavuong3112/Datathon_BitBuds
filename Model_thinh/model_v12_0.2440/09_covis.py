"""
Step 9: Co-visitation retrieval (session-based item co-occurrence).
Uses user_item_seq.parquet (event_ts) to build pseudo-sessions,
then scores test user candidates via intent-weighted co-visitation.

Intent weights (corrected — chat/sms > view_phone):
  contact_chat / contact_zalo / contact_sms = 3.0  (high friction, true intent)
  view_phone                                 = 2.0  (medium intent)
  other_interaction                          = 1.0  (exposure)

Output:
  cache/covis_candidates.parquet — top-100 per user with covis_score
"""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from collections import defaultdict
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"
def mem_mb():
    import psutil; return psutil.Process().memory_info().rss / 1e6

OUT_PATH = f"{CACHE_DIR}/covis_candidates.parquet"
if os.path.exists(OUT_PATH):
    try:
        _df = pd.read_parquet(OUT_PATH)
        if len(_df) > 0:
            print(f"{elapsed()} [SKIP] covis_candidates exists ({len(_df):,} rows)")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass

# Corrected intent weights: chat/sms friction higher → higher true intent
INTENT_WEIGHT = {
    'view_phone':        2.0,
    'contact_chat':      3.0,
    'contact_zalo':      3.0,
    'contact_sms':       3.0,
    'other_interaction': 1.0,
}

COVIS_TOP_K     = 20    # top neighbors stored per item
COVIS_CANDS     = 100   # top candidates returned per test user
SESSION_GAP_MIN = 30    # gap in minutes → new session
PAIR_WINDOW     = 5     # max forward window within session for pair generation

print(f"{elapsed()} Loading data …  [RAM:{mem_mb():.0f}MB]")
seq  = pd.read_parquet(f"{CACHE_DIR}/user_item_seq.parquet")
pos  = pd.read_parquet(f"{CACHE_DIR}/user_item_pos.parquet",
                       columns=['user_id','item_id',
                                'n_view_phone','n_chat','n_zalo','n_sms','n_other'])
iq   = pd.read_parquet(f"{CACHE_DIR}/item_quality.parquet",
                       columns=['item_id','total_events'])
test = pd.read_parquet(TEST_FILE)
test_users = set(test['user_id'].tolist())

print(f"{elapsed()} seq={len(seq):,} rows  {seq['user_id'].nunique():,} users  "
      f"[RAM:{mem_mb():.0f}MB]")

# Filter to items with training events (skip dead listings in co-vis)
allowed_items = set(iq[iq['total_events'] > 0]['item_id'].tolist())
print(f"{elapsed()} Allowed items (total_events>0): {len(allowed_items):,}")

seq = seq[seq['item_id'].isin(allowed_items)].copy()
seq['event_ts'] = pd.to_datetime(seq['event_ts'])
seq = seq.sort_values(['user_id', 'event_ts']).reset_index(drop=True)
print(f"{elapsed()} Filtered seq: {len(seq):,} rows  [RAM:{mem_mb():.0f}MB]")

# ── Build pseudo-sessions ─────────────────────────────────────────────────────
print(f"{elapsed()} Creating pseudo-sessions (gap>{SESSION_GAP_MIN}min = new session) …")
gap_sec = seq.groupby('user_id')['event_ts'].diff().dt.total_seconds().fillna(0)
new_sess = (gap_sec / 60.0 > SESSION_GAP_MIN) | (seq['user_id'] != seq['user_id'].shift(1))
seq['session_id'] = new_sess.cumsum().astype(np.int32)
del gap_sec, new_sess

n_sessions = seq['session_id'].nunique()
print(f"{elapsed()} Sessions: {n_sessions:,}")

# Convert timestamps to float seconds for fast subtraction
seq['_ts_sec'] = seq['event_ts'].astype(np.int64) / 1e9
seq_items    = seq['item_id'].values
seq_sessions = seq['session_id'].values
seq_ts       = seq['_ts_sec'].values
del seq; gc.collect()
print(f"{elapsed()} Arrays ready  [RAM:{mem_mb():.0f}MB]")

# ── Build co-visitation pairs ─────────────────────────────────────────────────
print(f"{elapsed()} Building co-visitation pairs (window={PAIR_WINDOW}) …")
pair_scores: dict = defaultdict(float)
n = len(seq_items)
log2 = np.log2

for i in range(n):
    si = seq_sessions[i]
    ai = seq_items[i]
    for j in range(i + 1, min(i + PAIR_WINDOW + 1, n)):
        if seq_sessions[j] != si:
            break
        bj = seq_items[j]
        if ai == bj:
            continue
        dt_min = max((seq_ts[j] - seq_ts[i]) / 60.0, 0.0)
        w = 1.0 / log2(dt_min + 2.0)
        key = (ai, bj) if ai < bj else (bj, ai)
        pair_scores[key] += w

print(f"{elapsed()} Pairs: {len(pair_scores):,}  [RAM:{mem_mb():.0f}MB]")
del seq_items, seq_sessions, seq_ts; gc.collect()

# Build item → top-K neighbors
print(f"{elapsed()} Building item→neighbors …")
item_neighbors: dict = defaultdict(list)
for (a, b), s in pair_scores.items():
    item_neighbors[a].append((b, s))
    item_neighbors[b].append((a, s))
del pair_scores; gc.collect()

covis: dict = {}
for item, neighbors in item_neighbors.items():
    neighbors.sort(key=lambda x: -x[1])
    covis[item] = neighbors[:COVIS_TOP_K]
del item_neighbors; gc.collect()
print(f"{elapsed()} Co-vis matrix: {len(covis):,} items  [RAM:{mem_mb():.0f}MB]")

# ── Score test users ──────────────────────────────────────────────────────────
print(f"{elapsed()} Scoring test users …")
pos_test = pos[pos['user_id'].isin(test_users)].copy()
pos_test['intent_score'] = (
    INTENT_WEIGHT['view_phone']        * pos_test['n_view_phone'].fillna(0) +
    INTENT_WEIGHT['contact_chat']      * pos_test['n_chat'].fillna(0) +
    INTENT_WEIGHT['contact_zalo']      * pos_test['n_zalo'].fillna(0) +
    INTENT_WEIGHT['contact_sms']       * pos_test['n_sms'].fillna(0) +
    INTENT_WEIGHT['other_interaction'] * pos_test['n_other'].fillna(0)
)
pos_test = pos_test[(pos_test['intent_score'] > 0) &
                    (pos_test['item_id'].isin(allowed_items))]
del pos, iq; gc.collect()
print(f"{elapsed()} Test users with history: {pos_test['user_id'].nunique():,}  "
      f"[RAM:{mem_mb():.0f}MB]")

rows = []
for uid, grp in pos_test.groupby('user_id', sort=False):
    history_set = set(grp['item_id'].tolist())
    cand_scores: dict = defaultdict(float)
    for item_id, iw in zip(grp['item_id'].tolist(), grp['intent_score'].tolist()):
        for nb, cs in covis.get(item_id, []):
            if nb in history_set:
                continue
            cand_scores[nb] += iw * cs
    if not cand_scores:
        continue
    top = sorted(cand_scores.items(), key=lambda x: -x[1])[:COVIS_CANDS]
    for iid, sc in top:
        rows.append({'user_id': uid, 'item_id': iid, 'covis_score': float(sc)})

del covis, pos_test; gc.collect()

covis_df = pd.DataFrame(rows)
del rows; gc.collect()
print(f"{elapsed()} Covis candidates: {len(covis_df):,} rows  "
      f"{covis_df['user_id'].nunique():,} users  [RAM:{mem_mb():.0f}MB]")

covis_df.to_parquet(OUT_PATH, index=False)
print(f"{elapsed()} Saved → {OUT_PATH}")
print(f"{elapsed()} DONE")
