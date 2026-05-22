"""
EDA Pipeline Runner — Chợ Tốt BĐS Datathon
Usage:
    cd d:/Datathon_Data/eda
    python run_eda.py              # all 3 steps
    python run_eda.py --steps 1   # only Step 1
    python run_eda.py --steps 2 3 # Steps 2 and 3
"""
import sys, os, time, logging, argparse
import psutil
import duckdb

# Ensure this script's folder is on path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import OUTPUT_DIR

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "eda_run.log"), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("eda")


def _ram_gb() -> float:
    return psutil.virtual_memory().used / 1024 ** 3


def _build_conn() -> duckdb.DuckDBPyConnection:
    """
    In-memory DuckDB connection.
    Memory cap = 20 GB (leaves ~12 GB for OS + pandas).
    Threads = 4 to avoid thrashing on large Parquet scans.
    """
    conn = duckdb.connect()
    conn.execute("SET memory_limit = '20GB'")
    conn.execute("SET threads = 4")
    conn.execute("SET enable_progress_bar = true")
    log.info("DuckDB connected (memory_limit=20 GB, threads=4)")
    return conn


def main():
    parser = argparse.ArgumentParser(description="Run EDA pipeline steps")
    parser.add_argument(
        "--steps", nargs="+", type=int, choices=[1, 2, 3],
        default=[1, 2, 3], help="Which steps to run (default: 1 2 3)"
    )
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  Chợ Tốt BĐS EDA Pipeline                              ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("Steps requested: %s", args.steps)
    log.info("Output directory: %s", OUTPUT_DIR)
    log.info("System RAM used at start: %.1f GB / %.1f GB",
             _ram_gb(), psutil.virtual_memory().total / 1024 ** 3)

    conn = _build_conn()
    dim  = None  # will be loaded in Step 1 if needed

    t_start = time.time()

    # ── Step 1 ──
    if 1 in args.steps:
        from step1_quality import run_step1
        t = time.time()
        dim = run_step1(conn)
        log.info("Step 1 finished in %.0f s | RAM: %.1f GB", time.time() - t, _ram_gb())

    # ── Step 2 ──
    if 2 in args.steps:
        from step2_cvr import run_step2
        t = time.time()
        run_step2(conn)
        log.info("Step 2 finished in %.0f s | RAM: %.1f GB", time.time() - t, _ram_gb())

    # ── Step 3 ──
    if 3 in args.steps:
        from step3_lifecycle import run_step3
        # dim is required for Step 3 price/area/geo analysis
        if dim is None:
            log.info("Step 3 needs dim_listing — loading now …")
            from step1_quality import load_dim_listing, analyze_time_coverage
            import pandas as pd
            dim = load_dim_listing()
            dim = analyze_time_coverage(dim)
        t = time.time()
        run_step3(conn, dim)
        log.info("Step 3 finished in %.0f s | RAM: %.1f GB", time.time() - t, _ram_gb())

    conn.close()
    total = time.time() - t_start
    log.info("══════════════════════════════════════════════════════════")
    log.info("All steps done in %.0f s (%.1f min)", total, total / 60)
    log.info("Outputs saved to: %s", OUTPUT_DIR)

    # Print summary of saved files
    saved = sorted(os.listdir(OUTPUT_DIR))
    log.info("Generated %d output files:", len(saved))
    for f in saved:
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        log.info("  %-45s  %6.0f KB", f, size / 1024)


if __name__ == "__main__":
    main()
