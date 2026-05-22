"""
End-to-end pipeline runner — runs each step in an isolated subprocess.
Subprocess isolation guarantees a clean heap for every step, preventing
the heap fragmentation / OOM errors that accumulate in a single long-lived process.

Usage:
  python run_pipeline.py            # run all pending steps (4-8)
  python run_pipeline.py --from 6   # re-run from step 6 onwards
"""
import sys, os, time, subprocess, argparse

sys.stdout.reconfigure(encoding='utf-8')
script_dir = os.path.dirname(os.path.abspath(__file__))

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

STEPS = [
    (4, '04_sasrec.py',           'SASRec 80-epoch GPU training + inference'),
    (5, '05_merge_candidates.py', 'Merge retrieval candidates'),
    (6, '06_features.py',         'Feature engineering (~60 features)'),
    (7, '07_rerank.py',           'LightGBM LambdaRank GPU reranker'),
    (8, '08_submit.py',           'Validate + write submission.csv'),
]

parser = argparse.ArgumentParser()
parser.add_argument('--from', dest='from_step', type=int, default=4,
                    help='Start from this step number (default: 4)')
args = parser.parse_args()

# Env for all subprocesses: single OpenBLAS thread prevents startup OOM
# when system heap is fragmented from previous steps
env = os.environ.copy()
env['OPENBLAS_NUM_THREADS'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUNBUFFERED'] = '1'   # force line-flush so logs appear in real time

print(f"{elapsed()} Pipeline runner starting from step {args.from_step}")
print("=" * 65)

for step_num, script, desc in STEPS:
    if step_num < args.from_step:
        print(f"{elapsed()} [SKIP step {step_num}] {desc}")
        continue

    print(f"\n{elapsed()} ── Step {step_num}: {desc} ──", flush=True)
    step_t0 = time.time()

    result = subprocess.run(
        [sys.executable, os.path.join(script_dir, script)],
        cwd=script_dir,
        env=env,
    )

    elapsed_step = time.time() - step_t0
    if result.returncode not in (0, None):
        print(f"\n{elapsed()} FAILED: step {step_num} ({script}) "
              f"exit code {result.returncode} after {elapsed_step:.0f}s")
        print(f"  Fix the error above and re-run with:  python run_pipeline.py --from {step_num}")
        sys.exit(result.returncode)

    print(f"{elapsed()} Step {step_num} DONE  ({elapsed_step:.0f}s)", flush=True)

print(f"\n{elapsed()} {'=' * 65}")
print(f"{elapsed()} All steps complete — submission.csv ready!")
