"""Quick estimate of raw event sizes."""
import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import duckdb
from config import *

evt_files = [f.replace('\\','/') for f in glob.glob(f'{EVT_DIR}/*.parquet')]
print(f'evt files: {len(evt_files)}')

conn = duckdb.connect()
conn.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
conn.execute(f"SET threads={DUCKDB_THREADS}")

print('Counting positive events in train window...')
n_pos = conn.execute(f"""
    SELECT COUNT(*) FROM read_parquet({evt_files})
    WHERE event_type IN ({POS_STR})
      AND is_login = 'login'
      AND {CATEGORY_FILTER}
      AND event_ts BETWEEN '{TRAIN_START}' AND '{TRAIN_END}'
""").fetchone()[0]
print(f'Positive events: {n_pos:,}')

print('Counting pageview events in train window...')
n_pv = conn.execute(f"""
    SELECT COUNT(*) FROM read_parquet({evt_files})
    WHERE event_type = 'pageview'
      AND is_login = 'login'
      AND {CATEGORY_FILTER}
      AND event_ts BETWEEN '{TRAIN_START}' AND '{TRAIN_END}'
""").fetchone()[0]
print(f'Pageview events: {n_pv:,}')
