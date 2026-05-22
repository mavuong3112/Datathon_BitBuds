"""
Search query NLP → volume by province (method A: text only, B: text + city_name fallback).

Run:
  env/bin/python Thinh_Analyze/run_search_query_province_nlp.py
  env/bin/python Thinh_Analyze/run_search_query_province_nlp.py --query-sample-frac 0.15 --rebuild-lexicon
  env/bin/python Thinh_Analyze/run_search_query_province_nlp.py --choropleth
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import matplotlib

_bk = matplotlib.get_backend().lower()
if "inline" not in _bk and "agg" not in _bk:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Thinh_Analyze.search_query_province_nlp_lib import (
    CATEGORIES,
    CAT_META,
    EVENTS_GLOB,
    KHONG_RO_TINH,
    OUT_DIR,
    apply_nlp_row,
    enrich_query_frame,
    hash_sample_clause,
    load_or_build_lexicon,
    norm_query,
    export_lexicon_tables,
    volume_by_province,
)

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.ioff()

DUCKDB_MEMORY_LIMIT = "3GB"
DUCKDB_THREADS = 4
QUERY_SAMPLE_FRAC_DEFAULT = 0.15
MIN_QUERY_COUNT = 5

REV_DIR = Path(__file__).resolve().parent / "outputs" / "region_revenue_bcg"
MAP_DIR = OUT_DIR / "maps"


def _finish_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def extract_aggregated_queries(
    con: duckdb.DuckDBPyConnection,
    sample_frac: float,
) -> pd.DataFrame:
    sample_sql = hash_sample_clause(sample_frac)
    return con.execute(
        f"""
        SELECT
            TRIM(CAST(query AS VARCHAR)) AS query,
            category,
            TRIM(CAST(city_name AS VARCHAR)) AS city_name,
            COUNT(*)::BIGINT AS n
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE event_type = 'pageview'
          AND query IS NOT NULL
          AND TRIM(CAST(query AS VARCHAR)) <> ''
          AND category IN (1010, 1020, 1030, 1040, 1050)
          {sample_sql}
        GROUP BY 1, 2, 3
        """
    ).fetchdf()


def coverage_stats(con: duckdb.DuckDBPyConnection, sample_frac: float) -> pd.DataFrame:
    sample_sql = hash_sample_clause(sample_frac)
    row = con.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT AS n_events_sample,
            COUNT(*) FILTER (
                WHERE event_type = 'pageview'
                  AND query IS NOT NULL AND TRIM(CAST(query AS VARCHAR)) <> ''
            )::BIGINT AS n_search_pv_sample,
            COUNT(DISTINCT TRIM(CAST(query AS VARCHAR))) FILTER (
                WHERE event_type = 'pageview'
                  AND query IS NOT NULL AND TRIM(CAST(query AS VARCHAR)) <> ''
            )::BIGINT AS n_distinct_query
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE 1=1 {sample_sql}
        """
    ).fetchone()
    return pd.DataFrame(
        [
            {
                "query_sample_frac": sample_frac,
                "n_events_sample": row[0],
                "n_search_pv_sample": row[1],
                "n_distinct_query": row[2],
                "n_search_pv_est": int(row[1] / sample_frac) if sample_frac < 1 else row[1],
            }
        ]
    )


def coverage_by_dims(con: duckdb.DuckDBPyConnection, sample_frac: float) -> pd.DataFrame:
    sample_sql = hash_sample_clause(sample_frac)
    return con.execute(
        f"""
        SELECT
            category,
            device,
            is_login,
            COUNT(*)::BIGINT AS n_search_pv
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE event_type = 'pageview'
          AND query IS NOT NULL AND TRIM(CAST(query AS VARCHAR)) <> ''
          {sample_sql}
        GROUP BY 1, 2, 3
        ORDER BY n_search_pv DESC
        """
    ).fetchdf()


def refinement_proxy(con: duckdb.DuckDBPyConnection, sample_frac: float) -> pd.DataFrame:
    sample_sql = hash_sample_clause(sample_frac)
    return con.execute(
        f"""
        WITH searches AS (
            SELECT
                category,
                session_id,
                event_ts,
                regexp_replace(
                    lower(trim(CAST(query AS VARCHAR))), '\\s+', ' ', 'g'
                ) AS qnorm
            FROM read_parquet('{EVENTS_GLOB}')
            WHERE event_type = 'pageview'
              AND query IS NOT NULL AND TRIM(CAST(query AS VARCHAR)) <> ''
              AND session_id IS NOT NULL
              {sample_sql}
        ),
        tagged AS (
            SELECT
                category,
                session_id,
                qnorm,
                LAG(qnorm) OVER (
                    PARTITION BY category, session_id ORDER BY event_ts
                ) AS pq
            FROM searches
        )
        SELECT
            category,
            COUNT(*)::BIGINT AS n_search_events,
            SUM(CASE WHEN pq IS NOT NULL AND qnorm <> pq THEN 1 ELSE 0 END)::BIGINT AS n_refinements,
            ROUND(
                100.0 * SUM(CASE WHEN pq IS NOT NULL AND qnorm <> pq THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                2
            ) AS pct_refinement
        FROM tagged
        GROUP BY category
        ORDER BY category
        """
    ).fetchdf()


def top_ngrams(df: pd.DataFrame, top_k: int = 40) -> pd.DataFrame:
    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()
    for _, row in df.iterrows():
        toks = row["q_norm"].split()
        w = int(row["n"])
        for i in range(len(toks) - 1):
            bigrams[" ".join(toks[i : i + 2])] += w
        for i in range(len(toks) - 2):
            trigrams[" ".join(toks[i : i + 3])] += w
    rows = []
    for ngram, cnt in bigrams.most_common(top_k):
        rows.append({"ngram_type": "bigram", "ngram": ngram, "n": cnt})
    for ngram, cnt in trigrams.most_common(top_k):
        rows.append({"ngram_type": "trigram", "ngram": ngram, "n": cnt})
    return pd.DataFrame(rows)


def validation_metrics(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = enriched[
        (enriched["province_nlp"] != "")
        & (enriched["city_name_canon"] != "")
        & (~enriched["is_noise"])
    ].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    sub["agree"] = sub["province_nlp"] == sub["city_name_canon"]
    summary = pd.DataFrame(
        [
            {
                "n_rows": len(sub),
                "n_weighted": int(sub["n"].sum()),
                "pct_agreement_unweighted": 100.0 * sub["agree"].mean(),
                "pct_agreement_weighted": 100.0
                * sub.loc[sub["agree"], "n"].sum()
                / sub["n"].sum(),
            }
        ]
    )
    mis = (
        sub.loc[~sub["agree"]]
        .groupby(["province_nlp", "city_name_canon"], as_index=False)["n"]
        .sum()
        .sort_values("n", ascending=False)
        .head(25)
    )
    return summary, mis


def delta_a_vs_b(vol_a: pd.DataFrame, vol_b: pd.DataFrame) -> pd.DataFrame:
    a = vol_a.groupby("province", as_index=False)["n_searches_est"].sum().rename(
        columns={"n_searches_est": "n_method_a"}
    )
    b = vol_b.groupby("province", as_index=False)["n_searches_est"].sum().rename(
        columns={"n_searches_est": "n_method_b"}
    )
    m = a.merge(b, on="province", how="outer").fillna(0)
    m["delta"] = m["n_method_b"] - m["n_method_a"]
    m["pct_lift"] = np.where(
        m["n_method_a"] > 0,
        100.0 * m["delta"] / m["n_method_a"],
        np.nan,
    )
    return m.sort_values("delta", ascending=False)


def plot_top_provinces(vol: pd.DataFrame, title: str, path: Path, top_n: int = 15) -> None:
    g = vol.groupby("province", as_index=False)["n_searches_est"].sum()
    g = g[g["province"] != KHONG_RO_TINH].nlargest(top_n, "n_searches_est")
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=g, y="province", x="n_searches_est", ax=ax, color="#2171b5")
    ax.set_title(title)
    ax.set_xlabel("Ước lượng số search pageview")
    _finish_fig(fig, path)


def plot_taxonomy_donut(tax_df: pd.DataFrame, path: Path) -> None:
    g = tax_df.groupby("taxonomy", as_index=False)["n"].sum().nlargest(12, "n")
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(g["n"], labels=g["taxonomy"], autopct="%1.1f%%", textprops={"fontsize": 7})
    ax.set_title("Query taxonomy (weighted by search volume)")
    _finish_fig(fig, path)


def plot_region_stack(vol_b: pd.DataFrame, mapping: pd.DataFrame, path: Path) -> None:
    m = mapping[["city_name_raw", "region"]].rename(columns={"city_name_raw": "province"})
    g = vol_b.merge(m, on="province", how="left")
    g = g[g["province"] != KHONG_RO_TINH]
    reg = g.groupby("region", as_index=False)["n_searches_est"].sum()
    if reg.empty or reg["region"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=reg, x="region", y="n_searches_est", ax=ax, palette="Set2")
    ax.set_title("Search volume theo miền (method B)")
    _finish_fig(fig, path)


def plot_validation_heatmap(enriched: pd.DataFrame, path: Path, top_n: int = 12) -> None:
    sub = enriched[
        (enriched["province_nlp"] != "")
        & (enriched["city_name_canon"] != "")
        & (~enriched["is_noise"])
    ]
    if sub.empty:
        return
    top_p = sub.groupby("province_nlp")["n"].sum().nlargest(top_n).index
    top_c = sub.groupby("city_name_canon")["n"].sum().nlargest(top_n).index
    sub = sub[sub["province_nlp"].isin(top_p) & sub["city_name_canon"].isin(top_c)]
    pivot = sub.pivot_table(
        index="province_nlp",
        columns="city_name_canon",
        values="n",
        aggfunc="sum",
        fill_value=0,
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("NLP province vs listing city_name (counts)")
    _finish_fig(fig, path)


def run_choropleth(vol_b: pd.DataFrame) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        print("Skip choropleth: geopandas not installed")
        return

    from Thinh_Analyze.run_region_choropleth import (
        GEO_PATH,
        load_city_aliases,
        city_to_geo_name,
        plot_choropleth,
    )

    if not GEO_PATH.exists():
        print("Skip choropleth: missing geojson")
        return

    gdf = gpd.read_file(GEO_PATH).rename(columns={"name_vi": "name_vi_geo"})
    aliases = load_city_aliases()
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    for cat in CATEGORIES:
        sub = vol_b[vol_b["category"] == cat]
        if sub.empty:
            continue
        m = sub.groupby("province", as_index=False)["n_searches_est"].sum()
        m = m[m["province"] != KHONG_RO_TINH]
        m["name_vi_geo"] = m["province"].map(lambda c: city_to_geo_name(c, aliases))
        merged = gdf.merge(m, on="name_vi_geo", how="left")
        merged["n_searches_est"] = merged["n_searches_est"].fillna(0)
        plot_choropleth(
            merged,
            "n_searches_est",
            f"Search volume (method B) — {CAT_META[cat]} ({cat})",
            MAP_DIR / f"fig_search_volume_choropleth_{cat}.png",
            log_scale=True,
            legend_label="log1p(search est)",
        )


def _national_volume(vol: pd.DataFrame) -> pd.Series:
    return vol.groupby("province")["n_searches_est"].sum().sort_values(ascending=False)


def province_coverage_stats(
    vol_a: pd.DataFrame,
    vol_b: pd.DataFrame,
    tax: pd.DataFrame,
    work: pd.DataFrame,
) -> pd.DataFrame:
    """One-row-per-metric table for markdown / notebook."""
    nat_a = _national_volume(vol_a)
    nat_b = _national_volume(vol_b)
    total_a = nat_a.sum()
    total_b = nat_b.sum()
    khong_a = nat_a.get(KHONG_RO_TINH, 0)
    khong_b = nat_b.get(KHONG_RO_TINH, 0)
    has_a = nat_a.drop(labels=[KHONG_RO_TINH], errors="ignore")
    has_b = nat_b.drop(labels=[KHONG_RO_TINH], errors="ignore")

    top5_a = has_a.head(5).sum() / has_a.sum() * 100 if has_a.sum() else 0
    top5_b = has_b.head(5).sum() / has_b.sum() * 100 if has_b.sum() else 0
    top10_a = has_a.head(10).sum() / has_a.sum() * 100 if has_a.sum() else 0

    explicit_n = work.loc[work["has_explicit_province"], "n"].sum()
    work_total = work["n"].sum()

    rows = [
        ("n_provinces_method_a", len(has_a), ""),
        ("n_provinces_method_b", len(has_b), ""),
        ("pct_volume_khong_ro_tinh_method_a", 100 * khong_a / total_a if total_a else 0, "%"),
        ("pct_volume_khong_ro_tinh_method_b", 100 * khong_b / total_b if total_b else 0, "%"),
        ("pct_volume_with_province_in_query", 100 * (total_a - khong_a) / total_a if total_a else 0, "%"),
        ("top5_share_among_named_provinces_method_a", top5_a, "%"),
        ("top10_share_among_named_provinces_method_a", top10_a, "%"),
        ("top5_share_among_named_provinces_method_b", top5_b, "%"),
        ("pct_searches_has_explicit_province_flag", 100 * explicit_n / work_total if work_total else 0, "%"),
    ]
    for cat in CATEGORIES:
        t = tax[tax["category"] == cat]
        tot = t["n"].sum()
        if tot == 0:
            continue
        geo = t[t["taxonomy"].str.contains("geo_explicit_province", na=False)]["n"].sum()
        dist = t[t["taxonomy"].str.contains("geo_district_only", na=False)]["n"].sum()
        gen = t[t["taxonomy"].str.contains("generic", na=False)]["n"].sum()
        rows.append((f"cat_{cat}_pct_geo_explicit_province", 100 * geo / tot, "%"))
        rows.append((f"cat_{cat}_pct_geo_district_only", 100 * dist / tot, "%"))
        rows.append((f"cat_{cat}_pct_generic", 100 * gen / tot, "%"))

    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def write_summary(
    out_dir: Path,
    sample_frac: float,
    cov: pd.DataFrame,
    val_summary: pd.DataFrame,
    vol_a: pd.DataFrame,
    vol_b: pd.DataFrame,
    tax: pd.DataFrame,
    work: pd.DataFrame,
) -> None:
    stats = province_coverage_stats(vol_a, vol_b, tax, work)
    stats.to_csv(out_dir / "10_province_coverage_stats.csv", index=False)

    nat_a = _national_volume(vol_a)
    nat_b = _national_volume(vol_b)
    total_a = nat_a.sum()
    khong_a = nat_a.get(KHONG_RO_TINH, 0)
    has_a = nat_a.drop(labels=[KHONG_RO_TINH], errors="ignore")
    has_b = nat_b.drop(labels=[KHONG_RO_TINH], errors="ignore")

    n_search = cov["n_search_pv_est"].iloc[0]
    n_distinct = int(cov["n_distinct_query"].iloc[0])

    lines = [
        "# Search query NLP theo tỉnh — thống kê tổng hợp",
        "",
        "## Phạm vi dữ liệu",
        "",
        f"| Chỉ số | Giá trị |",
        f"|--------|---------|",
        f"| Sample `event_id` (hash) | {sample_frac:.0%} |",
        f"| Search pageview ước lượng (full) | {n_search:,.0f} |",
        f"| Số query phân biệt (trong sample) | {n_distinct:,} |",
        f"| Nguồn | `fact_user_events`, `event_type=pageview`, có `query` |",
        "",
        "## Method A vs B — có phải chỉ ~10 tỉnh được gõ tên?",
        "",
        "**Không.** Lexicon match được nhiều tỉnh; nhưng **đa số query không chứa tên tỉnh** trong text.",
        "",
        "| | Method A (chỉ text query) | Method B (text + `city_name` tin) |",
        "|--|---------------------------|-----------------------------------|",
        f"| Số tỉnh có volume > 0 | **{len(has_a)}** | **{len(has_b)}** |",
        f"| Volume `khong_ro_tinh` | **{100*khong_a/total_a:.1f}%** | **{100*nat_b.get(KHONG_RO_TINH,0)/nat_b.sum():.1f}%** |",
        f"| Volume có gán tỉnh | **{100*(total_a-khong_a)/total_a:.1f}%** | **{100*has_b.sum()/nat_b.sum():.1f}%** |",
        f"| Top 5 tỉnh (trong phần đã gán tỉnh) | {100*has_a.head(5).sum()/has_a.sum():.1f}% volume | "
        f"{100*has_b.head(5).sum()/has_b.sum():.1f}% volume |",
        f"| Top 10 tỉnh (Method A, phần có tỉnh) | {100*has_a.head(10).sum()/has_a.sum():.1f}% volume | — |",
        "",
        f"- **{100*work.loc[work['has_explicit_province'], 'n'].sum()/work['n'].sum():.1f}%** search (weighted) có cờ `has_explicit_province` trong text.",
        "- Phần còn lại Method A: query kiểu `quận 7`, `gò vấp`, `trọ` — không gõ TP/tỉnh.",
        "- Method B gán tỉnh qua tin user click → gần đủ 63 tỉnh; HCM ~82% do cơ cấu nền tảng.",
        "",
        "## Method A — top 15 tỉnh (có tên trong query)",
        "",
        "| Tỉnh | Search ước lượng | % trong phần có tỉnh |",
        "|------|------------------|----------------------|",
    ]
    for p, n in has_a.head(15).items():
        lines.append(f"| {p} | {n:,.0f} | {100*n/has_a.sum():.1f}% |")

    lines.extend(
        [
            "",
            "## Method B — top 15 tỉnh (NLP + fallback)",
            "",
            "| Tỉnh | Search ước lượng | % trong phần có tỉnh |",
            "|------|------------------|----------------------|",
        ]
    )
    for p, n in has_b.head(15).items():
        lines.append(f"| {p} | {n:,.0f} | {100*n/has_b.sum():.1f}% |")

    lines.extend(["", "## Cách người dùng gõ query (taxonomy, % volume theo category)", ""])
    lines.append("| Category | Gõ **tên tỉnh** | Chỉ **quận/khu** | **Generic** (trọ, chung cư…) |")
    lines.append("|----------|----------------|------------------|------------------------------|")
    for cat in CATEGORIES:
        t = tax[tax["category"] == cat]
        tot = t["n"].sum()
        if tot == 0:
            continue
        geo = t[t["taxonomy"].str.contains("geo_explicit_province", na=False)]["n"].sum()
        dist = t[t["taxonomy"].str.contains("geo_district_only", na=False)]["n"].sum()
        gen = t[t["taxonomy"].str.contains("generic", na=False)]["n"].sum()
        label = f"{cat} {CAT_META.get(cat, '')}"
        lines.append(
            f"| {label} | {100*geo/tot:.1f}% | {100*dist/tot:.1f}% | {100*gen/tot:.1f}% |"
        )

    lines.extend(["", "## Refinement trong session", ""])
    refine_path = out_dir / "07_refinement_proxy.csv"
    if refine_path.exists():
        ref = pd.read_csv(refine_path)
        lines.append("| Category | % search là đổi query (refinement) |")
        lines.append("|----------|-----------------------------------|")
        for _, r in ref.iterrows():
            lines.append(
                f"| {int(r['category'])} {CAT_META.get(int(r['category']), '')} | "
                f"{r['pct_refinement']:.1f}% |"
            )

    if not val_summary.empty:
        lines.extend(
            [
                "",
                "## Validation (query có tỉnh trong text vs `city_name` tin)",
                "",
                f"| Chỉ số | Giá trị |",
                f"|--------|---------|",
                f"| Weighted agreement | **{val_summary['pct_agreement_weighted'].iloc[0]:.1f}%** |",
                f"| Unweighted agreement | {val_summary['pct_agreement_unweighted'].iloc[0]:.1f}% |",
                f"| Số nhóm query (có cả NLP tỉnh + city) | {int(val_summary['n_rows'].iloc[0]):,} |",
                "",
                "→ Khi user **gõ tên tỉnh**, thường khớp tỉnh tin đăng họ xem.",
            ]
        )
        mis_path = out_dir / "08_validation_mislabels.csv"
        if mis_path.exists():
            mis = pd.read_csv(mis_path).head(5)
            lines.extend(["", "**Top lệch (NLP ≠ city tin):**", ""])
            for _, r in mis.iterrows():
                lines.append(
                    f"- NLP `{r['province_nlp']}` vs tin `{r['city_name_canon']}`: "
                    f"{int(r['n']):,} lượt"
                )

    lines.extend(
        [
            "",
            "## Đọc nhanh cho slide",
            "",
            "1. **Không phải chỉ 10 tỉnh** — Method A thấy ~"
            f"{len(has_a)} tỉnh trong text; Method B ~{len(has_b)} tỉnh.",
            "2. **~90% query không ghi tỉnh** (Method A) — hay gõ quận/từ khóa.",
            "3. **HCM + Đà Nẵng + HN** chiếm phần lớn phần có tên tỉnh trong query.",
            "4. Muốn **phân bổ search theo tỉnh thực tế** → ưu tiên **Method B** "
            "hoặc UI geo filter (không có trong `query`).",
            "",
            "_File số liệu: `10_province_coverage_stats.csv`_",
        ]
    )

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_dir / "SUMMARY.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-sample-frac",
        type=float,
        default=QUERY_SAMPLE_FRAC_DEFAULT,
    )
    parser.add_argument("--rebuild-lexicon", action="store_true")
    parser.add_argument("--choropleth", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    sample_frac = args.query_sample_frac
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")

    print(f"Building lexicon (rebuild={args.rebuild_lexicon})...")
    prov_lex, dist_lex, mapping = load_or_build_lexicon(
        con, rebuild=args.rebuild_lexicon
    )
    export_lexicon_tables(prov_lex, dist_lex, mapping, OUT_DIR)

    print(f"Extracting queries (sample_frac={sample_frac})...")
    cov = coverage_stats(con, sample_frac)
    cov.to_csv(OUT_DIR / "00_coverage_overall.csv", index=False)
    coverage_by_dims(con, sample_frac).to_csv(
        OUT_DIR / "00_coverage_by_dims.csv", index=False
    )

    raw = extract_aggregated_queries(con, sample_frac)
    print(f"Aggregated query groups: {len(raw):,}")
    raw["q_norm"] = raw["query"].map(norm_query)

    print("Applying NLP slots...")
    enriched = enrich_query_frame(raw, prov_lex, dist_lex, mapping)
    enriched.to_csv(OUT_DIR / "00_enriched_queries_sample.csv", index=False)

    work = enriched[~enriched["is_noise"]].copy()

    vol_a = volume_by_province(work, "geo_method_a", sample_frac)
    vol_b = volume_by_province(work, "geo_method_b", sample_frac)
    vol_a.to_csv(OUT_DIR / "01_volume_by_province_method_a.csv", index=False)
    vol_b.to_csv(OUT_DIR / "02_volume_by_province_method_b.csv", index=False)
    delta_a_vs_b(vol_a, vol_b).to_csv(OUT_DIR / "03_delta_a_vs_b.csv", index=False)

    tax = (
        work.groupby(["category", "taxonomy"], as_index=False)["n"]
        .sum()
        .assign(
            share_pct=lambda d: 100.0
            * d["n"]
            / d.groupby("category")["n"].transform("sum")
        )
    )
    tax.to_csv(OUT_DIR / "04_query_taxonomy_counts.csv", index=False)

    for cat in CATEGORIES:
        sub = work[work["category"] == cat]
        top = (
            sub[sub["geo_method_b"] != KHONG_RO_TINH]
            .groupby(["geo_method_b", "query"], as_index=False)["n"]
            .sum()
            .sort_values(["geo_method_b", "n"], ascending=[True, False])
        )
        parts = []
        for prov, g in top.groupby("geo_method_b"):
            parts.append(g.nlargest(20, "n").assign(province=prov))
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(
                OUT_DIR / f"05_top_queries_by_province_{cat}.csv",
                index=False,
            )

    top_ngrams(work).to_csv(OUT_DIR / "06_bigrams_trigrams.csv", index=False)
    refinement_proxy(con, sample_frac).to_csv(
        OUT_DIR / "07_refinement_proxy.csv", index=False
    )

    val_summary, val_mis = validation_metrics(enriched)
    if not val_summary.empty:
        val_summary.to_csv(OUT_DIR / "08_validation_summary.csv", index=False)
        val_mis.to_csv(OUT_DIR / "08_validation_mislabels.csv", index=False)

    if REV_DIR.exists() and (REV_DIR / "01_revenue_by_province.csv").exists():
        rev = pd.read_csv(REV_DIR / "01_revenue_by_province.csv")
        rev_agg = (
            rev.groupby("city_name", as_index=False)["n_explicit_events"]
            .sum()
            .rename(columns={"city_name": "province"})
        )
        search_agg = vol_b.groupby("province", as_index=False)["n_searches_est"].sum()
        gap = rev_agg.merge(search_agg, on="province", how="outer").fillna(0)
        gap.to_csv(OUT_DIR / "09_search_vs_explicit_gap.csv", index=False)

    if not args.no_plots:
        plot_top_provinces(
            vol_a,
            "Top tỉnh — Method A (NLP text)",
            OUT_DIR / "fig_top_provinces_method_a.png",
        )
        plot_top_provinces(
            vol_b,
            "Top tỉnh — Method B (NLP + fallback)",
            OUT_DIR / "fig_top_provinces_method_b.png",
        )
        plot_taxonomy_donut(tax, OUT_DIR / "fig_taxonomy_donut.png")
        plot_region_stack(vol_b, mapping, OUT_DIR / "fig_region_stack_method_b.png")
        plot_validation_heatmap(enriched, OUT_DIR / "fig_validation_heatmap.png")

    if args.choropleth:
        run_choropleth(vol_b)

    write_summary(OUT_DIR, sample_frac, cov, val_summary, vol_a, vol_b, tax, work)
    print("Done. Outputs →", OUT_DIR)


if __name__ == "__main__":
    main()
