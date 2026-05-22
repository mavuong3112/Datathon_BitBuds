"""
Lexicon + rule-based NLP for search queries → province / query behavior.

Used by run_search_query_province_nlp.py and eda_search_query_province_nlp.ipynb.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    from unidecode import unidecode
except ImportError:

    def unidecode(s: str) -> str:
        return s


DATA_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "config"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "search_query_province_nlp"

MAPPING_CSV = CONFIG_DIR / "city_region_mapping.csv"
ALIAS_CSV = CONFIG_DIR / "geo_city_alias.csv"
LEXICON_JSON = CONFIG_DIR / "query_geo_lexicon.json"

DIM_GLOB = str(DATA_ROOT / "dim_listing" / "*.parquet")
EVENTS_GLOB = str(DATA_ROOT / "fact_user_events" / "*.parquet")

KHONG_RO_TINH = "khong_ro_tinh"
NOISE_QUERIES = frozenset({"none", "null", "nan", "n/a", ""})

CAT_META = {
    1010: "Căn hộ",
    1020: "Nhà ở",
    1030: "VP/MB",
    1040: "Đất",
    1050: "Phòng trọ",
}
CATEGORIES = tuple(CAT_META)

PRICE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(tr|trieu|ty|t)\b", re.I)
DIST_RE = re.compile(r"\b(q|quan|huyen)\s*([0-9]+|[a-z0-9]+)", re.I)

KEYWORDS = {
    "phong tro": ["phong tro", "ptro", "nha tro", "tro"],
    "nha nguyen can": ["nha nguyen can", "nguyen can"],
    "chung cu": ["chung cu", "cc", "can ho"],
    "mat bang": ["mat bang", "mb", "mat bang kinh doanh"],
    "dat": ["dat nen", "ban dat", " dat "],
    "homestay": ["homestay"],
}

# Extra province aliases (normalized, no diacritics)
EXTRA_PROVINCE_ALIASES: dict[str, list[str]] = {
    "Tp Hồ Chí Minh": [
        "tp hcm",
        "tp ho chi minh",
        "tp hồ chí minh",
        "ho chi minh",
        "hcm",
        "sai gon",
        "saigon",
        "tp hcm",
    ],
    "Hà Nội": ["ha noi", "hn", "tp ha noi"],
    "Đà Nẵng": ["da nang", "dn"],
    "Cần Thơ": ["can tho"],
    "Hải Phòng": ["hai phong", "hp"],
    "Bình Dương": ["binh duong", "bd"],
    "Đồng Nai": ["dong nai"],
    "Khánh Hòa": ["khanh hoa", "nha trang"],
    "Lâm Đồng": ["lam dong", "da lat", "dalat"],
}


def norm_text(s: str | None) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = unidecode(str(s).lower().strip())
    return re.sub(r"\s+", " ", t)


def norm_query(q: str | None) -> str:
    return norm_text(q)


def _pad_alias(a: str) -> str:
    """Word-boundary safe: match alias as substring with spaces."""
    a = norm_text(a)
    if not a:
        return ""
    return f" {a} "


def build_province_lexicon(mapping: pd.DataFrame) -> list[tuple[str, str]]:
    """Return [(alias_padded, city_name_raw), ...] sorted by alias length desc."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_alias(alias: str, canonical: str) -> None:
        a = norm_text(alias)
        if not a or len(a) < 2:
            return
        key = (a, canonical)
        if key in seen:
            return
        seen.add(key)
        pairs.append((_pad_alias(a), canonical))

    for _, row in mapping.iterrows():
        canonical = str(row["city_name_raw"]).strip()
        add_alias(canonical, canonical)
        add_alias(unidecode(canonical), canonical)
        for extra in EXTRA_PROVINCE_ALIASES.get(canonical, []):
            add_alias(extra, canonical)

    if ALIAS_CSV.exists():
        adf = pd.read_csv(ALIAS_CSV)
        for _, r in adf.iterrows():
            raw = str(r["city_name_raw"]).strip()
            geo = str(r["name_vi_geo"]).strip()
            add_alias(geo, raw)
            add_alias(unidecode(geo), raw)

    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def build_district_lexicon(
    con: duckdb.DuckDBPyConnection,
    *,
    top_n_per_city: int = 80,
    min_listings: int = 5,
) -> list[tuple[str, str, str]]:
    """[(alias_padded, district_label, city_name_raw), ...] longest first."""
    df = con.execute(
        f"""
        SELECT
            TRIM(CAST(city_name AS VARCHAR)) AS city_name,
            TRIM(CAST(district_name AS VARCHAR)) AS district_name,
            COUNT(*)::BIGINT AS n
        FROM read_parquet('{DIM_GLOB}')
        WHERE city_name IS NOT NULL AND TRIM(CAST(city_name AS VARCHAR)) <> ''
          AND district_name IS NOT NULL AND TRIM(CAST(district_name AS VARCHAR)) <> ''
        GROUP BY 1, 2
        HAVING COUNT(*) >= {min_listings}
        """
    ).fetchdf()

    rows: list[tuple[str, str, str, int]] = []
    for city, g in df.groupby("city_name"):
        g = g.nlargest(top_n_per_city, "n")
        for _, r in g.iterrows():
            dnorm = norm_text(r["district_name"])
            if len(dnorm) < 3:
                continue
            label = f"{r['district_name']}|{city}"
            rows.append((_pad_alias(dnorm), label, str(city), int(r["n"])))

    # quận N patterns handled separately via DIST_RE
    rows.sort(key=lambda x: len(x[0]), reverse=True)
    return [(a, d, c) for a, d, c, _ in rows]


def load_or_build_lexicon(
    con: duckdb.DuckDBPyConnection,
    *,
    rebuild: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], pd.DataFrame]:
    if LEXICON_JSON.exists() and not rebuild:
        data = json.loads(LEXICON_JSON.read_text(encoding="utf-8"))
        prov = [(p[0], p[1]) for p in data["province_aliases"]]
        dist = [(d[0], d[1], d[2]) for d in data["district_aliases"]]
        mapping = pd.read_csv(MAPPING_CSV)
        return prov, dist, mapping

    mapping = pd.read_csv(MAPPING_CSV)
    prov = build_province_lexicon(mapping)
    dist = build_district_lexicon(con)
    payload = {
        "province_aliases": prov,
        "district_aliases": [[a, d, c] for a, d, c in dist],
    }
    LEXICON_JSON.parent.mkdir(parents=True, exist_ok=True)
    LEXICON_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return prov, dist, mapping


def city_norm_to_canonical(city: str | None, mapping: pd.DataFrame) -> str | None:
    if city is None or (isinstance(city, float) and np.isnan(city)):
        return None
    c = str(city).strip()
    if not c:
        return None
    cn = norm_text(c)
    for _, row in mapping.iterrows():
        raw = str(row["city_name_raw"]).strip()
        if norm_text(raw) == cn or c == raw:
            return raw
    if ALIAS_CSV.exists():
        adf = pd.read_csv(ALIAS_CSV)
        for _, r in adf.iterrows():
            if norm_text(r["city_name_raw"]) == cn or norm_text(r["name_vi_geo"]) == cn:
                return str(r["city_name_raw"]).strip()
    return c


def extract_province(
    q_norm: str,
    province_lexicon: list[tuple[str, str]],
) -> tuple[str | None, str | None]:
    """Returns (canonical_province, matched_alias)."""
    if not q_norm:
        return None, None
    padded = f" {q_norm} "
    for alias, canonical in province_lexicon:
        if alias in padded:
            return canonical, alias.strip()
    return None, None


def extract_district_from_lexicon(
    q_norm: str,
    district_lexicon: list[tuple[str, str, str]],
) -> tuple[str | None, str | None]:
    padded = f" {q_norm} "
    for alias, label, city in district_lexicon:
        if alias in padded:
            return label, city
    m = DIST_RE.search(q_norm)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}", None
    return None, None


def price_bucket_vnd(text: str) -> str:
    m = PRICE_RE.search(text)
    if not m:
        return "khong_ro_gia"
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    if unit in ("ty", "t"):
        val *= 1_000_000_000
    else:
        val *= 1_000_000
    if val < 3e6:
        return "<3tr"
    if val < 5e6:
        return "3-5tr"
    if val < 7e6:
        return "5-7tr"
    if val < 10e6:
        return "7-10tr"
    if val < 15e6:
        return "10-15tr"
    return ">15tr"


def extract_keyword(text: str) -> str:
    padded = f" {text} "
    for label, kws in KEYWORDS.items():
        for k in sorted(kws, key=len, reverse=True):
            if f" {k} " in padded or text == k.strip():
                return label
    return "khac"


def extract_intent(text: str) -> str:
    if "cho thue" in text or "thue " in text:
        return "cho_thue"
    if "ban " in text or "mua " in text:
        return "ban_mua"
    if "thue" in text:
        return "thue"
    return "khac"


def is_noise_query(q_norm: str) -> bool:
    return q_norm in NOISE_QUERIES or len(q_norm) <= 1


def classify_taxonomy(
    q_norm: str,
    *,
    province_nlp: str | None,
    district_nlp: str | None,
    price_bucket: str,
    n_tokens: int,
) -> str:
    if is_noise_query(q_norm):
        return "noise"
    tags: list[str] = []
    if province_nlp:
        tags.append("geo_explicit_province")
    elif district_nlp and district_nlp != "khac":
        tags.append("geo_district_only")
    if price_bucket != "khong_ro_gia":
        tags.append("price_led")
    if n_tokens >= 4 and ("cho thue" in q_norm or " o " in q_norm or " tai " in q_norm):
        tags.append("full_phrase")
    kw = extract_keyword(q_norm)
    if kw != "khac" and n_tokens <= 3:
        tags.append("generic")
    if not tags:
        return "other"
    return "|".join(tags)


def apply_nlp_row(
    q_norm: str,
    city_name: str | None,
    province_lexicon: list[tuple[str, str]],
    district_lexicon: list[tuple[str, str, str]],
    mapping: pd.DataFrame,
) -> dict:
    province_nlp, match_span = extract_province(q_norm, province_lexicon)
    district_label, _district_city = extract_district_from_lexicon(q_norm, district_lexicon)

    city_canon = city_norm_to_canonical(city_name, mapping)
    tokens = q_norm.split() if q_norm else []
    n_tokens = len(tokens)

    if province_nlp:
        geo_a = province_nlp
    else:
        geo_a = KHONG_RO_TINH

    if province_nlp:
        geo_b = province_nlp
    elif city_canon:
        geo_b = city_canon
    else:
        geo_b = KHONG_RO_TINH

    price_b = price_bucket_vnd(q_norm)
    tax = classify_taxonomy(
        q_norm,
        province_nlp=province_nlp if province_nlp else None,
        district_nlp=district_label,
        price_bucket=price_b,
        n_tokens=n_tokens,
    )

    return {
        "province_nlp": province_nlp or "",
        "district_nlp": district_label or "",
        "match_span": (match_span or "")[:80],
        "price_bucket": price_b,
        "kw_type": extract_keyword(q_norm),
        "intent_ad_type": extract_intent(q_norm),
        "query_len": len(q_norm),
        "n_tokens": n_tokens,
        "has_explicit_province": bool(province_nlp and match_span),
        "geo_method_a": geo_a,
        "geo_method_b": geo_b,
        "city_name_canon": city_canon or "",
        "taxonomy": tax,
        "is_noise": is_noise_query(q_norm),
    }


def enrich_query_frame(
    df: pd.DataFrame,
    province_lexicon: list[tuple[str, str]],
    district_lexicon: list[tuple[str, str, str]],
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """df must have columns: query, q_norm (optional), city_name, n."""
    out = df.copy()
    if "q_norm" not in out.columns:
        out["q_norm"] = out["query"].map(norm_query)

    def _row_slots(row: pd.Series) -> dict:
        city = row["city_name"] if "city_name" in row.index else None
        return apply_nlp_row(
            row["q_norm"],
            city,
            province_lexicon,
            district_lexicon,
            mapping,
        )

    slot_df = pd.DataFrame(out.apply(_row_slots, axis=1).tolist())
    return pd.concat([out.reset_index(drop=True), slot_df], axis=1)


def hash_sample_clause(frac: float, id_col: str = "event_id") -> str:
    if frac is None or frac >= 1.0:
        return ""
    bucket = max(1, int(frac * 1000))
    return f"AND (abs(hash(CAST({id_col} AS VARCHAR))) % 1000) < {bucket}"


def volume_by_province(
    df: pd.DataFrame,
    geo_col: str,
    sample_frac: float,
) -> pd.DataFrame:
    """Aggregate n_searches by province × category."""
    scale = 1.0 / sample_frac if sample_frac and sample_frac < 1 else 1.0
    g = (
        df.groupby([geo_col, "category"], as_index=False)["n"]
        .sum()
        .rename(columns={geo_col: "province", "n": "n_searches_sample"})
    )
    g["n_searches_est"] = (g["n_searches_sample"] * scale).round().astype(np.int64)
    total = g["n_searches_est"].sum()
    g["share_pct"] = np.where(total > 0, 100.0 * g["n_searches_est"] / total, 0.0)
    return g.sort_values("n_searches_est", ascending=False)


def export_lexicon_tables(
    province_lexicon: list[tuple[str, str]],
    district_lexicon: list[tuple[str, str, str]],
    mapping: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_rows = []
    for alias, canon in province_lexicon:
        prov_rows.append({"alias_norm": alias.strip(), "city_name_raw": canon})
    pd.DataFrame(prov_rows).drop_duplicates().to_csv(
        out_dir / "00_lexicon_provinces.csv", index=False
    )
    dist_rows = [
        {"alias_norm": a.strip(), "district_label": d, "city_name_raw": c}
        for a, d, c in district_lexicon
    ]
    pd.DataFrame(dist_rows).to_csv(out_dir / "00_lexicon_districts.csv", index=False)
    mapping.to_csv(out_dir / "00_city_region_mapping_copy.csv", index=False)
