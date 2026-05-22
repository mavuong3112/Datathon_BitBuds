#!/usr/bin/env python3
"""Fast vectorized mismatch analysis via DuckDB."""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent
BEST = str(ROOT / "Sub_Score" / "submission_ (0.2184).csv")
CAND = str(ROOT / "submission_(0.0114).csv")

con = duckdb.connect()
con.execute(f"""
CREATE TABLE best AS SELECT * FROM read_csv('{BEST}', header=true);
CREATE TABLE cand AS SELECT * FROM read_csv('{CAND}', header=true);
""")

print("=== Files ===")
print(con.execute("SELECT COUNT(*) FROM best").fetchone())
print(con.execute("SELECT COUNT(*) FROM cand").fetchone())

r = con.execute("""
SELECT
    AVG(CASE WHEN b.item_id = c.item_id THEN 1.0 ELSE 0.0 END) AS pos_match,
    COUNT(*) FILTER (WHERE b.item_id != c.item_id) AS n_mismatch
FROM best b
JOIN cand c USING (user_id, rank)
""").fetchone()
print(f"\nSame (user,rank,item): {r[0]:.4f}  mismatches={r[1]:,}")

r2 = con.execute("""
WITH per_user AS (
    SELECT
        b.user_id,
        COUNT(DISTINCT b.item_id) FILTER (
            WHERE b.item_id IN (SELECT item_id FROM cand c WHERE c.user_id = b.user_id)
        ) AS set_overlap
    FROM best b
    GROUP BY b.user_id
)
SELECT AVG(set_overlap), AVG(CASE WHEN set_overlap = 10 THEN 1.0 ELSE 0.0 END)
FROM per_user
""").fetchone()
# fix query - use list overlap properly
con.execute("""
CREATE OR REPLACE TABLE best_set AS
SELECT user_id, LIST(item_id ORDER BY rank) AS items FROM best GROUP BY user_id;
CREATE OR REPLACE TABLE cand_set AS
SELECT user_id, LIST(item_id ORDER BY rank) AS items FROM cand GROUP BY user_id;
""")
r2 = con.execute("""
SELECT
    AVG(list_intersect(b.items, c.items).len()) AS avg_set_overlap,
    AVG(CASE WHEN list_sort(b.items) = list_sort(c.items) THEN 1.0 ELSE 0.0 END) AS identical_set
FROM best_set b
JOIN cand_set c USING (user_id)
""").fetchone()
print(f"Avg SET overlap (Recall@10): {r2[0]:.4f}/10")
print(f"Users identical SET: {r2[1]:.4f}")

print("\n=== Mismatch BY RANK ===")
print(con.execute("""
SELECT
    b.rank,
    ROUND(AVG(CASE WHEN b.item_id = c.item_id THEN 1.0 ELSE 0.0 END), 4) AS pos_match,
    ROUND(AVG(CASE WHEN c.item_id NOT IN (
        SELECT item_id FROM best b2 WHERE b2.user_id = b.user_id
    ) THEN 1.0 ELSE 0.0 END), 4) AS cand_new_at_rank
FROM best b
JOIN cand c USING (user_id, rank)
GROUP BY b.rank ORDER BY b.rank
""").df().to_string(index=False))

print("\n=== REMOVED items: rank in BEST ===")
print(con.execute("""
WITH removed AS (
    SELECT b.user_id, b.item_id, b.rank AS rank_in_best
    FROM best b
    WHERE NOT EXISTS (
        SELECT 1 FROM cand c
        WHERE c.user_id = b.user_id AND c.item_id = b.item_id
    )
)
SELECT rank_in_best, COUNT(*) AS n_removed
FROM removed GROUP BY 1 ORDER BY 1
""").df().to_string(index=False))

print("\n=== ADDED items: rank in CAND ===")
print(con.execute("""
WITH added AS (
    SELECT c.user_id, c.item_id, c.rank AS rank_in_cand
    FROM cand c
    WHERE NOT EXISTS (
        SELECT 1 FROM best b
        WHERE b.user_id = c.user_id AND b.item_id = b.item_id
    )
)
SELECT rank_in_cand, COUNT(*) AS n_added
FROM added GROUP BY 1 ORDER BY 1
""").df().to_string(index=False))

print("\n=== Decompose mismatches (position level) ===")
print(con.execute("""
SELECT
    SUM(CASE WHEN b.item_id != c.item_id
        AND c.item_id IN (SELECT item_id FROM best b2 WHERE b2.user_id = b.user_id)
        AND b.item_id IN (SELECT item_id FROM cand c2 WHERE c2.user_id = b.user_id)
        THEN 1 ELSE 0 END) AS reorder_only,
    SUM(CASE WHEN b.item_id != c.item_id
        AND c.item_id NOT IN (SELECT item_id FROM best b2 WHERE b2.user_id = b.user_id)
        THEN 1 ELSE 0 END) AS cand_injection,
    SUM(CASE WHEN b.item_id != c.item_id
        AND b.item_id NOT IN (SELECT item_id FROM cand c2 WHERE c2.user_id = b.user_id)
        THEN 1 ELSE 0 END) AS best_removed
FROM best b JOIN cand c USING (user_id, rank)
WHERE b.item_id != c.item_id
""").df().to_string(index=False))

print("\n=== Per-user slot changes ===")
print(con.execute("""
SELECT
    AVG(10 - ov.cnt) AS avg_removed,
    AVG(adds.cnt) AS avg_added
FROM (
    SELECT b.user_id, COUNT(DISTINCT b.item_id) FILTER (
        WHERE EXISTS (SELECT 1 FROM cand c WHERE c.user_id=b.user_id AND c.item_id=b.item_id)
    ) AS cnt
    FROM best b GROUP BY 1
) ov
JOIN (
    SELECT c.user_id, COUNT(*) AS cnt
    FROM cand c
    WHERE NOT EXISTS (SELECT 1 FROM best b WHERE b.user_id=c.user_id AND b.item_id=c.item_id)
    GROUP BY 1
) adds USING (user_id)
""").fetchone())

PY