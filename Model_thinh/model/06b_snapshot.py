"""
Step 6b: Load fact_listing_snapshot → item-level snapshot features.
Outputs cache/snapshot_features.parquet with cols:
  item_id, views_24h, contacts_24h, contact_rate_24h,
  views_24h_log, contacts_24h_log, pct_days_contact, listing_age_days_snap
"""
import sys, time, gc, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from config import CACHE_DIR, SNAP_DIR, TRAIN_END

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

OUT = f"{CACHE_DIR}/snapshot_features.parquet"

print(f"{elapsed()} Loading snapshot parquet files …")
files = sorted(glob.glob(f"{SNAP_DIR}/*.parquet"))
print(f"{elapsed()} {len(files)} files")

snap = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
snap['date'] = pd.to_datetime(snap['date'])
train_end_dt = pd.Timestamp(TRAIN_END)
print(f"{elapsed()} snap raw: {len(snap):,} rows, "
      f"date range {snap['date'].min().date()} → {snap['date'].max().date()}")

# Cap to TRAIN window so we don't peek post-train
snap = snap[snap['date'] <= train_end_dt].copy()
print(f"{elapsed()} after train-end cap: {len(snap):,} rows")

# Sort once for groupby.last() efficiency
snap = snap.sort_values(['item_id','date'])

# Latest snapshot row per item (closest to TRAIN_END)
print(f"{elapsed()} Building latest-snapshot per item …")
latest = (snap.groupby('item_id', sort=False)
              .agg(views_24h=('views_24h','last'),
                   contacts_24h=('contacts_24h','last'),
                   listing_age_days_snap=('listing_age_days','last'))
              .reset_index())
latest['views_24h']    = latest['views_24h'].fillna(0)
latest['contacts_24h'] = latest['contacts_24h'].fillna(0)
latest['views_24h_log']    = np.log1p(latest['views_24h'])
latest['contacts_24h_log'] = np.log1p(latest['contacts_24h'])
latest['contact_rate_24h'] = latest['contacts_24h'] / (latest['views_24h'] + 1.0)
print(f"{elapsed()} latest: {len(latest):,} items")

# pct_days_with_contact aggregate over full history per item
print(f"{elapsed()} Building pct_days_contact …")
snap['has_contact'] = (snap['contacts_24h'].fillna(0) > 0).astype(np.int8)
pct = (snap.groupby('item_id', sort=False)['has_contact']
            .mean().rename('pct_days_contact').reset_index())
print(f"{elapsed()} pct_days_contact: {len(pct):,} items, "
      f"mean={pct['pct_days_contact'].mean():.3f}")

snap_feat = latest.merge(pct, on='item_id', how='left')
snap_feat['pct_days_contact'] = snap_feat['pct_days_contact'].fillna(0)

# Downcast
for c in ['views_24h','contacts_24h','listing_age_days_snap',
          'views_24h_log','contacts_24h_log','contact_rate_24h','pct_days_contact']:
    snap_feat[c] = snap_feat[c].astype(np.float32)

snap_feat.to_parquet(OUT, index=False)
print(f"{elapsed()} Saved {OUT}: {len(snap_feat):,} items, {snap_feat.shape[1]} cols")
del snap, latest, pct, snap_feat; gc.collect()
print(f"{elapsed()} DONE")
