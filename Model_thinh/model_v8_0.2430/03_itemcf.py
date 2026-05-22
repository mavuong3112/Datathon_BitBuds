"""
Step 3: Item-CF via GPU using BPR item embeddings.

Two-phase approach eliminates heap fragmentation / CUDA DMA failures:
  Phase 1 (03_itemcf_phase1.py): stream parquet → build 54K user embs → save → EXIT
  Phase 2 (03_itemcf_phase2.py): fresh process → GPU scoring with clean heap

Stage 8: env RETRIEVER_MODE=train uses pos_train + als_model_train factors,
         outputs itemcf_candidates_train.parquet.
"""
import sys, os, time, subprocess
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

MODE = os.environ.get('RETRIEVER_MODE', 'full').lower()
SUFFIX = '_train' if MODE == 'train' else ''
OUT_PATH  = f"{CACHE_DIR}/itemcf_candidates{SUFFIX}.parquet"
EMB_USER  = f"{CACHE_DIR}/test_user_emb{SUFFIX}.npy"
META_PATH = f"{CACHE_DIR}/test_user_meta{SUFFIX}.pkl"
print(f"[STAGE 8] RETRIEVER_MODE={MODE} → output {OUT_PATH}")

# ── Skip if already done ──────────────────────────────────────────────────────
if os.path.exists(OUT_PATH):
    try:
        df = pd.read_parquet(OUT_PATH)
        if len(df) > 0:
            print(f"{elapsed()} [SKIP] itemcf_candidates.parquet exists "
                  f"({len(df):,} rows)", flush=True)
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass  # empty / corrupt → re-run

script_dir = os.path.dirname(os.path.abspath(__file__))

def run_phase(script_name, label):
    path = os.path.join(script_dir, script_name)
    print(f"\n{'='*60}", flush=True)
    print(f"{elapsed()} === {label} ===", flush=True)
    print(f"{'='*60}", flush=True)
    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '1'   # prevent threadpool OOM on fragmented heap
    result = subprocess.run(
        [sys.executable, path],
        cwd=script_dir,
        env=env,
    )
    if result.returncode not in (0, None):
        print(f"\nERROR: {script_name} exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    print(f"{elapsed()} {label} completed.", flush=True)

# ── Phase 1: build user embeddings (subprocess → own heap → exits cleanly) ───
if os.path.exists(EMB_USER) and os.path.exists(META_PATH):
    print(f"{elapsed()} [SKIP] Phase 1 outputs exist", flush=True)
else:
    run_phase('03_itemcf_phase1.py', 'Phase 1: build user embeddings')

# ── Phase 2: GPU scoring (fresh subprocess with clean heap) ───────────────────
run_phase('03_itemcf_phase2.py', 'Phase 2: GPU scoring')

print(f"\n{elapsed()} ItemCF pipeline complete.", flush=True)
