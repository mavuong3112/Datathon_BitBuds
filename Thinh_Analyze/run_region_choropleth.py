"""
Choropleth Việt Nam — doanh thu proxy theo tỉnh (từ outputs region_revenue_bcg).

Run: env/bin/python Thinh_Analyze/run_region_choropleth.py

Cần: geopandas, pyogrio (xem requirements-eda.txt)
Geo: Thinh_Analyze/config/vietnam_provinces.geojson (TopoJSON adm2, 64 tỉnh)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

_bk = matplotlib.get_backend().lower()
if "inline" not in _bk and "agg" not in _bk:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm

warnings.filterwarnings("ignore", category=UserWarning)
plt.ioff()

DATA_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "config"
REV_DIR = Path(__file__).resolve().parent / "outputs" / "region_revenue_bcg"
MAP_DIR = REV_DIR / "maps"
GEO_PATH = CONFIG_DIR / "vietnam_provinces.geojson"
ALIAS_CSV = CONFIG_DIR / "geo_city_alias.csv"

CAT_META = {1010: "Căn hộ", 1020: "Nhà ở", 1030: "VP/MB", 1040: "Đất", 1050: "Phòng trọ"}
REGIONS = ("Bac", "Trung", "Nam")
REGION_LABELS = {"Bac": "Bắc", "Trung": "Trung", "Nam": "Nam"}

BCG_COLORS = {
    "stars": "#2ca02c",
    "cash_cows": "#ff7f0e",
    "question_marks": "#1f77b4",
    "dogs": "#d62728",
    "new_market": "#9467bd",
    "low_volume": "#bbbbbb",
    "hcmc_excluded": "#d9d9d9",
    "none": "#f0f0f0",
}
BCG_MAP_LEGEND = ("stars", "cash_cows", "question_marks", "dogs")
BCG_LEGEND_LABELS = {
    "stars": "Stars — tăng trưởng cao, thị phần cầu cao",
    "cash_cows": "Cash cows — tăng trưởng thấp, thị phần cầu cao",
    "question_marks": "Question marks — tăng trưởng cao, thị phần cầu thấp",
    "dogs": "Dogs — tăng trưởng thấp, thị phần cầu thấp",
}
AD_TYPE_LABELS = {"let": "Thuê", "sell": "Bán"}
CATEGORIES = tuple(CAT_META)

BCG_PILL_COLORS = {
    "stars": "#F4C430",
    "cash_cows": "#2E7D32",
    "question_marks": "#1565C0",
    "dogs": "#C62828",
    "low_volume": "#BBBBBB",
    "hcmc_excluded": "#D9D9D9",
    "none": "#F0F0F0",
}
BCG_PILL_LEGEND = ("stars", "cash_cows", "question_marks", "dogs")
BCG_PILL_LABELS = {
    "stars": "Stars (Sao)",
    "cash_cows": "Cash cows (Bò sữa)",
    "question_marks": "Question marks (Dấu hỏi)",
    "dogs": "Dogs (Chó)",
}
BCG4_QUADRANTS = frozenset(BCG_PILL_LEGEND)
REGION_ORDER = {"Bac": 0, "Trung": 1, "Nam": 2}
PILL_LIGHT_BG = frozenset({"stars", "low_volume", "none", "hcmc_excluded"})
# Bản đồ phủ sóng tỉnh trong ma trận BCG
BCG_COVERAGE_OTHER = "#ECECEC"
HCM_CITY = "Tp Hồ Chí Minh"
HCM_HIGHLIGHT_EDGE = "#7B1FA2"
HCM_HIGHLIGHT_FILL = "#EDE7F6"
BCG_CITY_LABEL_SHORT = {
    "Bà Rịa - Vũng Tàu": "BR-VT",
    "Thừa Thiên Huế": "TT Huế",
    HCM_CITY: "TP.HCM",
}
# (dx, dy) độ — đẩy nhãn ra khỏi centroid; Nam đẩy sang phía đông
BCG_LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "Hà Nội": (0.0, 0.35),
    "Hải Phòng": (0.45, 0.2),
    "Thừa Thiên Huế": (-0.45, 0.25),
    "Đà Nẵng": (0.5, 0.05),
    "Quảng Nam": (-0.4, -0.3),
    "Khánh Hòa": (0.55, 0.2),
    "Đắk Lắk": (0.5, 0.15),
    "Lâm Đồng": (0.55, -0.2),
    "Bình Thuận": (0.6, 0.05),
    "Bình Phước": (0.65, 0.55),
    "Tây Ninh": (-0.75, 0.45),
    "Bình Dương": (0.75, 0.55),
    "Đồng Nai": (0.95, 0.25),
    "Bà Rịa - Vũng Tàu": (1.05, -0.05),
    "Long An": (-0.7, -0.15),
    "Tiền Giang": (-0.95, -0.35),
    "Bến Tre": (-0.35, -0.85),
    "Vĩnh Long": (-1.05, -0.55),
    "Cần Thơ": (-1.15, -0.75),
    "Kiên Giang": (-1.25, -0.95),
}
BCG_LABEL_MIN_DIST = 0.28
# Đồng bộ với run_region_revenue_bcg.py
PILL_MIN_N_LISTINGS = 40
PILL_MIN_N_EXPLICIT = 20
EXPLICIT_TYPES_LABEL = "view_phone, contact_chat, contact_zalo, contact_sms (login)"


def _finish_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", path)


def load_geo() -> gpd.GeoDataFrame:
    if not GEO_PATH.exists():
        raise FileNotFoundError(
            f"Missing {GEO_PATH}. Download from kcjpop/vietnam-topojson legacy/adm2/adm2.json"
        )
    gdf = gpd.read_file(GEO_PATH)
    gdf = gdf.rename(columns={"name_vi": "name_vi_geo"})
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def load_city_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    if ALIAS_CSV.exists():
        df = pd.read_csv(ALIAS_CSV)
        for _, r in df.iterrows():
            aliases[str(r["city_name_raw"]).strip()] = str(r["name_vi_geo"]).strip()
    return aliases


def city_to_geo_name(city: str, aliases: dict[str, str]) -> str:
    c = str(city).strip()
    if c in aliases:
        return aliases[c]
    return c


def load_revenue() -> pd.DataFrame:
    path = REV_DIR / "01_revenue_by_province.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run run_region_revenue_bcg.py first — missing {path}")
    return pd.read_csv(path)


def load_bcg_national() -> pd.DataFrame:
    path = REV_DIR / "06_bcg_national_choropleth.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run run_region_revenue_bcg.py first"
        )
    return pd.read_csv(path)


def load_bcg_pill_cities() -> pd.DataFrame:
    """Tỉnh/thành có trong ma trận BCG pill (07_bcg_pill_matrix_wide.csv)."""
    path = REV_DIR / "07_bcg_pill_matrix_wide.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run plot_bcg_pill_matrix / run_region_choropleth first"
        )
    df = pd.read_csv(path)
    return df[["city_name", "region"]].drop_duplicates(subset=["city_name"])


def _dominant_bcg_quadrant_for_city(bcg_city: pd.DataFrame) -> str:
    """Ô BCG đại diện tỉnh = quadrant có tổng explicit lớn nhất (trong 4 ô BCG)."""
    sub4 = bcg_city[bcg_city["bcg_quadrant"].isin(BCG4_QUADRANTS)]
    if len(sub4):
        return (
            sub4.groupby("bcg_quadrant", as_index=False)["n_explicit_events"]
            .sum()
            .sort_values("n_explicit_events", ascending=False)
            .iloc[0]["bcg_quadrant"]
        )
    if len(bcg_city):
        modes = bcg_city["bcg_quadrant"].mode()
        if len(modes):
            return str(modes.iloc[0])
    return "low_volume"


def _bcg_label_anchor(city: str, x: float, y: float) -> tuple[float, float]:
    dx, dy = BCG_LABEL_OFFSETS.get(city, (0.45, 0.12))
    return x + dx, y + dy


def _repel_bcg_label_anchors(
    anchors: list[tuple[float, float, str]],
    min_dist: float = BCG_LABEL_MIN_DIST,
) -> list[tuple[float, float, str]]:
    """Tách nhãn chồng nhau bằng repulsion đơn giản."""
    pts = [list(a) for a in anchors]
    for _ in range(100):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                xi, yi = pts[i][0], pts[i][1]
                xj, yj = pts[j][0], pts[j][1]
                dx, dy = xi - xj, yi - yj
                dist = (dx * dx + dy * dy) ** 0.5
                if dist >= min_dist or dist < 1e-9:
                    continue
                push = (min_dist - dist) / 2
                ux, uy = dx / dist, dy / dist
                pts[i][0] = xi + ux * push
                pts[i][1] = yi + uy * push
                pts[j][0] = xj - ux * push
                pts[j][1] = yj - uy * push
                moved = True
        if not moved:
            break
    return [(p[0], p[1], p[2]) for p in pts]


def build_bcg_coverage_city_summary(bcg_nat: pd.DataFrame) -> pd.DataFrame:
    """Một dòng/tỉnh: màu BCG dominant + HCM (xử lý riêng)."""
    pill = load_bcg_pill_cities()
    rows: list[dict] = []
    for _, prow in pill.iterrows():
        city = prow["city_name"]
        sub = bcg_nat[(bcg_nat["city_name"] == city) & (bcg_nat["is_hcmc"] == 0)]
        rows.append(
            {
                "city_name": city,
                "region": prow["region"],
                "bcg_quadrant": _dominant_bcg_quadrant_for_city(sub),
                "is_hcmc": 0,
            }
        )
    hcm = bcg_nat[bcg_nat["is_hcmc"] == 1]
    if len(hcm):
        rows.append(
            {
                "city_name": HCM_CITY,
                "region": str(hcm["region"].iloc[0]),
                "bcg_quadrant": "hcmc_excluded",
                "is_hcmc": 1,
            }
        )
    return pd.DataFrame(rows)


def aggregate_slice(
    rev: pd.DataFrame,
    *,
    category: int | None = None,
    ad_type: str | None = None,
    region: str | None = None,
) -> pd.DataFrame:
    df = rev.copy()
    if category is not None:
        df = df[df["category"] == category]
    if ad_type is not None:
        df = df[df["ad_type"] == ad_type]
    if region is not None:
        df = df[df["region"] == region]

    agg = (
        df.groupby(["city_name", "region", "is_hcmc"], as_index=False)
        .agg(
            n_explicit_events=("n_explicit_events", "sum"),
            n_positive_events=("n_positive_events", "sum"),
            n_listings=("n_listings", "sum"),
        )
    )
    agg["explicit_per_1k"] = np.where(
        agg["n_listings"] > 0,
        1000.0 * agg["n_explicit_events"] / agg["n_listings"],
        np.nan,
    )
    return agg


def merge_geo_metrics(
    gdf: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
    aliases: dict[str, str],
    value_col: str,
) -> gpd.GeoDataFrame:
    m = metrics.copy()
    m["name_vi_geo"] = m["city_name"].map(lambda c: city_to_geo_name(c, aliases))
    merged = gdf.merge(m, on="name_vi_geo", how="left")
    merged[value_col] = merged[value_col].fillna(0)
    return merged


def plot_choropleth(
    merged: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    path: Path,
    *,
    cmap: str = "YlOrRd",
    log_scale: bool = False,
    exclude_hcmc: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    legend_label: str | None = None,
) -> None:
    plot_gdf = merged.copy()
    if exclude_hcmc:
        plot_gdf = plot_gdf[plot_gdf["is_hcmc"].fillna(0) != 1]

    vals = plot_gdf[value_col].astype(float)
    if log_scale:
        vals = np.log1p(vals)
        legend_label = legend_label or f"log1p({value_col})"
    else:
        legend_label = legend_label or value_col

    fig, ax = plt.subplots(1, 1, figsize=(7, 10))
    plot_gdf.plot(
        column=vals,
        ax=ax,
        cmap=cmap,
        linewidth=0.35,
        edgecolor="#666666",
        legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.55, "pad": 0.02},
        missing_kwds={"color": "#eeeeee", "label": "Không có dữ liệu"},
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    _finish_fig(fig, path)


def plot_national_bcg_choropleth(
    gdf: gpd.GeoDataFrame,
    bcg_row: pd.DataFrame,
    aliases: dict[str, str],
    category: int,
    ad_type: str,
    path: Path,
) -> None:
    if len(bcg_row) == 0:
        mgeo = gdf.copy()
        mgeo["bcg_quadrant"] = "none"
    else:
        sub = bcg_row.copy()
        sub["name_vi_geo"] = sub["city_name"].map(lambda c: city_to_geo_name(c, aliases))
        mgeo = gdf.merge(
            sub[
                [
                    "name_vi_geo",
                    "city_name",
                    "bcg_quadrant",
                    "n_listings",
                    "n_explicit_events",
                    "share_demand_pct",
                    "explicit_per_1k_listings",
                    "growth_cvr_pct",
                    "growth_explicit_pct",
                    "median_growth_cvr",
                    "median_relative_cvr",
                ]
            ],
            on="name_vi_geo",
            how="left",
        )
        mgeo["bcg_quadrant"] = mgeo["bcg_quadrant"].fillna("none")

    def color_for(q: str) -> str:
        if q in BCG_MAP_LEGEND:
            return BCG_COLORS[q]
        return BCG_COLORS.get(q, BCG_COLORS["none"])

    colors = mgeo["bcg_quadrant"].map(color_for)

    fig, ax = plt.subplots(1, 1, figsize=(8, 11))
    mgeo.plot(ax=ax, color=colors, linewidth=0.4, edgecolor="#555555")

    handles = [
        mpatches.Patch(color=BCG_COLORS[k], label=BCG_LEGEND_LABELS[k])
        for k in BCG_MAP_LEGEND
        if (mgeo["bcg_quadrant"] == k).any()
    ]
    if (mgeo["bcg_quadrant"] == "hcmc_excluded").any():
        handles.append(
            mpatches.Patch(
                color=BCG_COLORS["hcmc_excluded"],
                label="HCM — loại khỏi median / tỷ trọng",
            )
        )
    if (mgeo["bcg_quadrant"] == "low_volume").any():
        handles.append(
            mpatches.Patch(color=BCG_COLORS["low_volume"], label="Tỉnh volume thấp")
        )
    if (mgeo["bcg_quadrant"] == "none").any():
        handles.append(mpatches.Patch(color=BCG_COLORS["none"], label="Không có dữ liệu"))

    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.95)
    med_g = np.nan
    med_s = np.nan
    if len(bcg_row):
        if "median_growth_cvr" in bcg_row.columns:
            med_g = bcg_row["median_growth_cvr"].iloc[0]
        elif "median_growth" in bcg_row.columns:
            med_g = bcg_row["median_growth"].iloc[0]
        if "median_relative_cvr" in bcg_row.columns:
            med_s = bcg_row["median_relative_cvr"].iloc[0]
        elif "median_relative_share" in bcg_row.columns:
            med_s = bcg_row["median_relative_share"].iloc[0]
    no_data_note = " | Không có tin trong cửa sổ EDA" if len(bcg_row) == 0 else ""
    subtitle = (
        f"BCG=CVR explicit/tin | Median growth CVR={med_g:.1f}% | "
        f"Median rel. CVR={med_s:.2f}"
        if pd.notna(med_g) and pd.notna(med_s)
        else f"BCG=CVR explicit/tin (4 loại, login) | ex-HCM{no_data_note}"
    )
    ax.set_title(
        f"{CAT_META[category]} — {AD_TYPE_LABELS[ad_type]}\n{subtitle}",
        fontsize=11,
    )
    ax.set_axis_off()
    _finish_fig(fig, path)


def plot_national_bcg_choropleth_maps(
    gdf: gpd.GeoDataFrame,
    bcg: pd.DataFrame,
    aliases: dict[str, str],
) -> None:
    for category in CATEGORIES:
        for ad_type in ("let", "sell"):
            sub = bcg[(bcg["category"] == category) & (bcg["ad_type"] == ad_type)].copy()
            if sub.empty:
                sub = pd.DataFrame(
                    {
                        "city_name": pd.Series(dtype=str),
                        "bcg_quadrant": pd.Series(dtype=str),
                        "n_listings": pd.Series(dtype=float),
                        "n_explicit_events": pd.Series(dtype=float),
                        "share_demand_pct": pd.Series(dtype=float),
                        "growth_explicit_pct": pd.Series(dtype=float),
                        "median_growth": pd.Series(dtype=float),
                        "median_relative_share": pd.Series(dtype=float),
                    }
                )
            path = MAP_DIR / f"fig_bcg_choropleth_{category}_{ad_type}.png"
            plot_national_bcg_choropleth(gdf, sub, aliases, category, ad_type, path)


def _pill_quadrant_color(quadrant: str) -> str:
    return BCG_PILL_COLORS.get(str(quadrant), BCG_PILL_COLORS["none"])


def _pill_text_color(quadrant: str) -> str:
    return "#1a1a1a" if quadrant in PILL_LIGHT_BG else "#ffffff"


def _pill_cell_row(
    bcg: pd.DataFrame, city: str, category: int, ad_type: str
) -> dict[str, object]:
    """Một ô L/S: quadrant + CVR + cỡ mẫu (tin, explicit)."""
    empty: dict[str, object] = {
        "quadrant": "none",
        "cvr_per_listing": np.nan,
        "n_listings": 0,
        "n_explicit": 0,
    }
    sub = bcg[
        (bcg["city_name"] == city)
        & (bcg["category"] == category)
        & (bcg["ad_type"] == ad_type)
    ]
    if sub.empty:
        return empty
    row = sub.iloc[0]
    n_listings = int(row.get("n_listings", 0) or 0)
    n_explicit = int(row.get("n_explicit_events", 0) or 0)
    cvr = row.get("cvr_per_listing", np.nan)
    if pd.isna(cvr) and n_listings > 0:
        cvr = float(n_explicit) / float(n_listings)
    return {
        "quadrant": str(row["bcg_quadrant"]),
        "cvr_per_listing": float(cvr) if pd.notna(cvr) else np.nan,
        "n_listings": n_listings,
        "n_explicit": n_explicit,
    }


def _pill_sample_line(cell: dict[str, object]) -> str:
    """Cỡ mẫu: tin (n) và explicit (e)."""
    n = int(cell["n_listings"])
    e = int(cell["n_explicit"])
    return f"(n={n}, e={e})"


def _pill_half_label(side: str, cell: dict[str, object]) -> str:
    """VD: L \\n 24,50 / tin \\n (n=2, e=49) — rate và cỡ mẫu tách bạch."""
    n = int(cell["n_listings"])
    if cell["quadrant"] == "none" and n == 0:
        return f"{side}\n—\n{_pill_sample_line(cell)}"
    cvr = cell["cvr_per_listing"]
    if pd.isna(cvr):
        rate_line = "—"
    else:
        rate_line = f"{float(cvr):.2f}".replace(".", ",") + " / tin"
    return f"{side}\n{rate_line}\n{_pill_sample_line(cell)}"


def _pill_matrix_methodology_text() -> str:
    return "\n".join(
        [
            "ĐỊNH NGHĨA & CÔNG THỨC (mỗi ô = 1 tỉnh × category × thuê/bán)",
            f"• Segment: category (1010–1050) × ad_type (L=let thuê, S=sell bán).",
            f"• (n=…, e=…) trên ô: n_listings = tin active (posted, prep dim_listing); "
            f"e = n_explicit_events ({EXPLICIT_TYPES_LABEL}).",
            "• cvr_per_listing (số trên ô) = e ÷ n  →  explicit trên 1 tin.",
            "• Màu 4 ô BCG khi: n≥"
            f"{PILL_MIN_N_LISTINGS} & e≥{PILL_MIN_N_EXPLICIT} (ex-HCM); median: segment → "
            "toàn quốc cùng cat → cùng cat (L+S) → cùng L/S → tất cả (nếu segment <2 tỉnh).",
            "• Xám: chưa đủ n/e — hoặc không có pool ≥2 tỉnh để so median.",
        ]
    )


def _write_pill_matrix_methodology(path: Path) -> None:
    path.write_text(_pill_matrix_methodology_text() + "\n", encoding="utf-8")
    print("Saved", path)


def build_bcg_pill_matrix_wide(bcg: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per BCG-eligible city, columns per category × L/S."""
    bcg = bcg.copy()
    if "cvr_per_listing" not in bcg.columns:
        if "explicit_per_1k_listings" in bcg.columns:
            bcg["cvr_per_listing"] = bcg["explicit_per_1k_listings"] / 1000.0
        elif {"n_listings", "n_explicit_events"}.issubset(bcg.columns):
            bcg["cvr_per_listing"] = np.where(
                bcg["n_listings"] > 0,
                bcg["n_explicit_events"] / bcg["n_listings"],
                np.nan,
            )

    bcg4_rows = bcg[bcg["bcg_quadrant"].isin(BCG4_QUADRANTS)]
    city_meta = (
        bcg4_rows.groupby("city_name", as_index=False)
        .agg(
            region=("region", "first"),
            max_cvr=("cvr_per_listing", "max"),
        )
    )
    city_meta["_region_ord"] = city_meta["region"].map(REGION_ORDER).fillna(9)
    city_meta = city_meta.sort_values(
        ["_region_ord", "max_cvr"], ascending=[True, False]
    )

    rows: list[dict] = []
    for _, cm in city_meta.iterrows():
        city = cm["city_name"]
        rec: dict = {"city_name": city, "region": cm["region"]}
        for cat in CATEGORIES:
            let = _pill_cell_row(bcg, city, cat, "let")
            sell = _pill_cell_row(bcg, city, cat, "sell")
            rec[f"{cat}_let_quadrant"] = let["quadrant"]
            rec[f"{cat}_let_cvr_per_listing"] = let["cvr_per_listing"]
            rec[f"{cat}_let_n_listings"] = let["n_listings"]
            rec[f"{cat}_let_n_explicit"] = let["n_explicit"]
            rec[f"{cat}_sell_quadrant"] = sell["quadrant"]
            rec[f"{cat}_sell_cvr_per_listing"] = sell["cvr_per_listing"]
            rec[f"{cat}_sell_n_listings"] = sell["n_listings"]
            rec[f"{cat}_sell_n_explicit"] = sell["n_explicit"]
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_bcg_pill_matrix(bcg: pd.DataFrame) -> Path:
    """
    Ma trận BCG: hàng = tỉnh; cột = category; ô chữ nhật L/S + cvr_per_listing (/ tin).
    """
    wide = build_bcg_pill_matrix_wide(bcg)
    wide_path = REV_DIR / "07_bcg_pill_matrix_wide.csv"
    wide.to_csv(wide_path, index=False)
    print("Saved", wide_path)
    _write_pill_matrix_methodology(REV_DIR / "08_bcg_pill_matrix_methodology.txt")

    cities = wide["city_name"].tolist()
    n_cities = len(cities)
    n_cats = len(CATEGORIES)
    if n_cities == 0:
        raise ValueError("No BCG-eligible cities for pill matrix")

    # cell_w < 1 → gutter ngang; cell_h cao → đủ padding dọc cho 3 dòng chữ
    cell_w, cell_h = 0.72, 0.90
    half_gap = cell_w * 0.05
    half_w = (cell_w - half_gap) / 2
    fig_w = max(20.0, 3.6 * n_cats)
    fig_h = max(18.0, 0.88 * n_cities)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.28, top=0.88)
    ax.set_xlim(0, n_cats)
    ax.set_ylim(0, n_cities)
    ax.invert_yaxis()

    for i, city in enumerate(cities):
        row_y = i + 0.5
        for j, cat in enumerate(CATEGORIES):
            x0 = j + (1.0 - cell_w) / 2
            y0 = row_y - cell_h / 2

            cell_outline = Rectangle(
                (x0, y0),
                cell_w,
                cell_h,
                fill=False,
                linewidth=0.9,
                edgecolor="#333333",
                zorder=2,
            )
            ax.add_patch(cell_outline)

            halves = (
                ("L", "let", x0),
                ("S", "sell", x0 + half_w + half_gap),
            )
            for label, ad_type, hx0 in halves:
                cell = _pill_cell_row(bcg, city, cat, ad_type)
                quadrant = str(cell["quadrant"])
                color = _pill_quadrant_color(quadrant)
                half = Rectangle(
                    (hx0, y0),
                    half_w,
                    cell_h,
                    facecolor=color,
                    edgecolor="#333333",
                    linewidth=0.35,
                    zorder=1,
                )
                ax.add_patch(half)
                tx = hx0 + half_w / 2
                ty = y0 + cell_h / 2
                ax.text(
                    tx,
                    ty,
                    _pill_half_label(label, cell),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=_pill_text_color(quadrant),
                    zorder=3,
                    linespacing=1.35,
                )

    ax.set_xticks([j + 0.5 for j in range(n_cats)])
    ax.set_xticklabels(
        [f"{c}\n{CAT_META[c]}" for c in CATEGORIES],
        fontsize=10,
        linespacing=1.15,
    )
    ax.set_yticks([i + 0.5 for i in range(n_cities)])
    ylabels = []
    for _, row in wide.iterrows():
        reg = REGION_LABELS.get(row["region"], row["region"])
        ylabels.append(f"{row['city_name']} ({reg})")
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        mpatches.Patch(color=BCG_PILL_COLORS[k], label=BCG_PILL_LABELS[k])
        for k in BCG_PILL_LEGEND
    ]
    handles.append(
        mpatches.Patch(
            color=BCG_PILL_COLORS["low_volume"],
            label=(
                f"Xám: chưa đủ mẫu (tin<{PILL_MIN_N_LISTINGS} "
                f"hoặc explicit<{PILL_MIN_N_EXPLICIT})"
            ),
        )
    )
    handles.append(
        mpatches.Patch(color=BCG_PILL_COLORS["none"], label="Không có dữ liệu")
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=8,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )
    ax.set_title(
        "BCG toàn quốc — ma trận tỉnh × danh mục\n"
        "L/S = thuê/bán | Số = e÷n (CVR/tin) | (n,e)=tin & explicit",
        fontsize=12,
        pad=14,
    )
    for i, line in enumerate(_pill_matrix_methodology_text().split("\n")):
        fig.text(
            0.02,
            0.018 + i * 0.024,
            line,
            transform=fig.transFigure,
            fontsize=6.3,
            ha="left",
            va="bottom",
            color="#333333",
        )

    out_path = MAP_DIR / "fig_bcg_pill_matrix.png"
    _finish_fig(fig, out_path)
    return out_path


def plot_bcg_provinces_highlight_map(
    gdf: gpd.GeoDataFrame,
    bcg_nat: pd.DataFrame,
    aliases: dict[str, str],
    path: Path | None = None,
) -> Path:
    """
    Bản đồ Việt Nam: tô màu theo ô BCG dominant (tổng explicit theo tỉnh);
    TP.HCM viền tím + callout «xử lý riêng» (loại khỏi median BCG).
    """
    out_path = path or (MAP_DIR / "fig_bcg_provinces_coverage.png")
    summary = build_bcg_coverage_city_summary(bcg_nat)
    m = summary.copy()
    m["name_vi_geo"] = m["city_name"].map(lambda c: city_to_geo_name(c, aliases))
    merged = gdf.merge(m, on="name_vi_geo", how="left")
    merged["_fill"] = merged["bcg_quadrant"].map(
        lambda q: BCG_COLORS.get(str(q), BCG_COVERAGE_OTHER)
        if pd.notna(q)
        else BCG_COVERAGE_OTHER
    )
    merged.loc[merged["city_name"].isna(), "_fill"] = BCG_COVERAGE_OTHER

    fig, ax = plt.subplots(1, 1, figsize=(10, 11))
    xmin, ymin, xmax, ymax = merged.total_bounds
    ax.set_xlim(xmin - 0.25, xmax + 2.6)
    ax.set_ylim(ymin - 0.35, ymax + 0.35)

    base = merged[merged["city_name"].isna()]
    if len(base):
        base.plot(ax=ax, color=BCG_COVERAGE_OTHER, linewidth=0.35, edgecolor="#888888")

    provinces = merged[
        merged["city_name"].notna() & (merged["is_hcmc"].fillna(0) != 1)
    ]
    if len(provinces):
        provinces.plot(
            ax=ax,
            color=provinces["_fill"],
            linewidth=0.45,
            edgecolor="#444444",
        )

    hcm_rows = merged[merged["is_hcmc"].fillna(0) == 1]
    if len(hcm_rows):
        hcm_rows.plot(
            ax=ax,
            color=HCM_HIGHLIGHT_FILL,
            linewidth=3.0,
            edgecolor=HCM_HIGHLIGHT_EDGE,
            linestyle=(0, (6, 4)),
            hatch="///",
            zorder=4,
        )
        hcm_pt = hcm_rows.geometry.representative_point().iloc[0]
        hcm_note_x = xmax + 1.85
        hcm_note_y = ymin + 1.05
        ax.annotate(
            "★ TP.HCM — xử lý riêng\n(loại khỏi median BCG)",
            xy=(hcm_pt.x, hcm_pt.y),
            xytext=(hcm_note_x, hcm_note_y),
            fontsize=7.5,
            ha="left",
            va="center",
            color=HCM_HIGHLIGHT_EDGE,
            fontweight="bold",
            zorder=10,
            arrowprops=dict(
                arrowstyle="-|>",
                color=HCM_HIGHLIGHT_EDGE,
                lw=1.5,
                connectionstyle="arc3,rad=-0.12",
                shrinkA=4,
                shrinkB=4,
            ),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#FFF8E1",
                edgecolor=HCM_HIGHLIGHT_EDGE,
                linewidth=1.4,
                alpha=0.97,
            ),
        )

    label_rows = merged[
        merged["city_name"].notna() & (merged["is_hcmc"].fillna(0) != 1)
    ].copy()
    label_rows["_pt"] = label_rows.geometry.representative_point()
    anchors: list[tuple[float, float, str]] = []
    label_meta: list[dict] = []
    for _, row in label_rows.iterrows():
        pt = row["_pt"]
        city = row["city_name"]
        tx, ty = _bcg_label_anchor(city, pt.x, pt.y)
        anchors.append((tx, ty, city))
        label_meta.append(
            {
                "city": city,
                "pt": pt,
                "quadrant": str(row["bcg_quadrant"]),
            }
        )
    anchors = _repel_bcg_label_anchors(anchors)
    anchor_by_city = {c: (x, y) for x, y, c in anchors}
    for meta in label_meta:
        city = meta["city"]
        pt = meta["pt"]
        tx, ty = anchor_by_city[city]
        label = BCG_CITY_LABEL_SHORT.get(city, city)
        edge = BCG_COLORS.get(meta["quadrant"], "#888888")
        ax.annotate(
            label,
            xy=(pt.x, pt.y),
            xytext=(tx, ty),
            fontsize=6,
            ha="center",
            va="center",
            color="#1a1a1a",
            zorder=6,
            arrowprops=dict(
                arrowstyle="-",
                color="#aaaaaa",
                lw=0.55,
                shrinkA=2,
                shrinkB=2,
            ),
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor=edge,
                alpha=0.95,
                linewidth=0.7,
            ),
        )

    n_pill = int((summary["is_hcmc"] == 0).sum())
    handles = [
        mpatches.Patch(color=BCG_COLORS[k], label=BCG_LEGEND_LABELS[k])
        for k in BCG_MAP_LEGEND
        if (summary["bcg_quadrant"] == k).any()
    ]
    if (summary["bcg_quadrant"] == "low_volume").any():
        handles.append(
            mpatches.Patch(
                color=BCG_COLORS["low_volume"],
                label="Volume thấp (chưa đủ n/e)",
            )
        )
    handles.append(
        mpatches.Patch(
            facecolor=HCM_HIGHLIGHT_FILL,
            edgecolor=HCM_HIGHLIGHT_EDGE,
            hatch="///",
            linewidth=2,
            label="TP.HCM — xử lý riêng (ex median BCG)",
        )
    )
    handles.append(
        mpatches.Patch(color=BCG_COVERAGE_OTHER, label="Không trong ma trận BCG")
    )
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.42),
        fontsize=7,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.78)
    ax.set_title(
        "Việt Nam — tỉnh/thành trong phân tích BCG\n"
        f"{n_pill} tỉnh ma trận pill | Màu = ô BCG dominant (theo explicit) | ★ HCM riêng",
        fontsize=11,
    )
    ax.set_axis_off()
    _finish_fig(fig, out_path)
    return out_path


def plot_choropleth_categorical(
    merged: gpd.GeoDataFrame,
    cat_col: str,
    title: str,
    path: Path,
) -> None:
    plot_gdf = merged.copy()
    cats = [c for c in BCG_COLORS if c != "none"]
    plot_gdf["_cat"] = plot_gdf[cat_col].fillna("none")
    color_series = plot_gdf["_cat"].map(lambda x: BCG_COLORS.get(x, BCG_COLORS["none"]))

    fig, ax = plt.subplots(1, 1, figsize=(7, 10))
    plot_gdf.plot(ax=ax, color=color_series, linewidth=0.35, edgecolor="#666666")
    handles = [
        mpatches.Patch(color=BCG_COLORS[k], label=k.replace("_", " "))
        for k in cats
        if (plot_gdf["_cat"] == k).any()
    ]
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    _finish_fig(fig, path)


def plot_category_grid(
    rev: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    aliases: dict[str, str],
    ad_type: str,
    value_col: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Choropleth — explicit lead ({ad_type}) theo category",
        fontsize=13,
        y=1.02,
    )
    all_vals = []
    slices = []
    for cat in CAT_META:
        m = aggregate_slice(rev, category=cat, ad_type=ad_type)
        merged = merge_geo_metrics(gdf, m, aliases, value_col)
        slices.append((cat, merged))
        all_vals.append(merged[value_col].values)
    vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals else 1.0

    for ax, (cat, merged) in zip(axes.flat, slices):
        merged.plot(
            column=value_col,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            linewidth=0.25,
            edgecolor="#888",
            legend=False,
            missing_kwds={"color": "#eee"},
        )
        ax.set_title(f"{cat} — {CAT_META[cat]}", fontsize=10)
        ax.set_axis_off()

    axes.flat[-1].axis("off")
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="Sự kiện explicit (login)")
    _finish_fig(fig, path)


def export_merged_geojson(
    gdf: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
    aliases: dict[str, str],
    path: Path,
) -> None:
    m = metrics.copy()
    m["name_vi_geo"] = m["city_name"].map(lambda c: city_to_geo_name(c, aliases))
    out = gdf.merge(
        m[
            [
                "name_vi_geo",
                "city_name",
                "region",
                "is_hcmc",
                "n_explicit_events",
                "n_positive_events",
                "explicit_per_1k",
            ]
        ],
        on="name_vi_geo",
        how="left",
    )
    out.to_file(path, driver="GeoJSON")
    print("Saved", path)


def main() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    aliases = load_city_aliases()
    gdf = load_geo()
    rev = load_revenue()

    # --- 10 bản đồ BCG national (category × thuê/bán) ---
    bcg_nat = load_bcg_national()
    print("Plotting 10 national BCG choropleth maps …")
    plot_national_bcg_choropleth_maps(gdf, bcg_nat, aliases)
    print("Plotting BCG pill matrix …")
    plot_bcg_pill_matrix(bcg_nat)
    print("Plotting BCG province coverage map …")
    plot_bcg_provinces_highlight_map(gdf, bcg_nat, aliases)

    # --- Tổng hợp toàn quốc (tham khảo) ---
    total = aggregate_slice(rev)
    merged_total = merge_geo_metrics(gdf, total, aliases, "n_explicit_events")
    plot_choropleth(
        merged_total,
        "n_explicit_events",
        "Việt Nam — sự kiện explicit (lead, login)\nTổng 5 category × thuê/bán",
        MAP_DIR / "fig_choropleth_explicit_total.png",
        log_scale=True,
        legend_label="log1p(explicit events)",
    )
    plot_choropleth(
        merge_geo_metrics(gdf, total, aliases, "n_positive_events"),
        "n_positive_events",
        "Việt Nam — sự kiện positive (datathon)\nTổng 5 category × thuê/bán",
        MAP_DIR / "fig_choropleth_positive_total.png",
        log_scale=True,
        legend_label="log1p(positive events)",
    )
    plot_choropleth(
        merged_total,
        "explicit_per_1k",
        "Việt Nam — explicit / 1.000 tin đăng",
        MAP_DIR / "fig_choropleth_explicit_per_1k.png",
        cmap="PuBuGn",
    )

    # --- Theo ad_type ---
    for ad_type in ("let", "sell"):
        m = aggregate_slice(rev, ad_type=ad_type)
        merged = merge_geo_metrics(gdf, m, aliases, "n_explicit_events")
        plot_choropleth(
            merged,
            "n_explicit_events",
            f"Explicit lead — {ad_type} (5 category)",
            MAP_DIR / f"fig_choropleth_explicit_{ad_type}.png",
            log_scale=True,
        )
        plot_category_grid(
            rev, gdf, aliases, ad_type, "n_explicit_events",
            MAP_DIR / f"fig_choropleth_explicit_{ad_type}_by_category.png",
        )

    # --- Tỷ trọng ex-HCM (theo miền: normalize within region on map via facet) ---
    share_path = REV_DIR / "03_share_ex_hcmc.csv"
    if share_path.exists():
        share = pd.read_csv(share_path)
        share = share[share["is_hcmc"] == 0]
        for region in REGIONS:
            s = (
                share[(share["region"] == region) & (share["ad_type"] == "let")]
                .groupby("city_name", as_index=False)
                .agg(share_pct=("share_explicit_events_ex_hcmc_pct", "mean"))
            )
            if s.empty:
                continue
            s["region"] = region
            s["is_hcmc"] = 0
            merged = merge_geo_metrics(gdf, s, aliases, "share_pct")
            plot_choropleth(
                merged,
                "share_pct",
                f"Tỷ trọng explicit (let, avg category) — {REGION_LABELS[region]}\nDenominator: miền trừ HCM",
                MAP_DIR / f"fig_choropleth_share_let_{region}_ex_hcmc.png",
                cmap="Blues",
                exclude_hcmc=True,
                legend_label="% trong miền (ex-HCM)",
            )

    # --- BCG quadrant maps (Nam let/sell as primary; all combos in subfolder) ---
    bcg_path = REV_DIR / "05_bcg_quadrants.csv"
    if bcg_path.exists():
        bcg = pd.read_csv(bcg_path)
        hcmc_cities = set(
            pd.read_csv(CONFIG_DIR / "city_region_mapping.csv")
            .query("is_hcmc == 1")["city_name_raw"]
        )
        bcg = bcg[~bcg["city_name"].isin(hcmc_cities)]
        bcg_dir = MAP_DIR / "bcg"
        bcg_dir.mkdir(exist_ok=True)
        for region in REGIONS:
            for ad_type in ("let", "sell"):
                for cat in CAT_META:
                    sub = bcg[
                        (bcg["region"] == region)
                        & (bcg["ad_type"] == ad_type)
                        & (bcg["category"] == cat)
                    ]
                    if sub.empty:
                        continue
                    cols = sub[
                        ["city_name", "bcg_quadrant", "n_explicit_events", "relative_share"]
                    ].copy()
                    cols["name_vi_geo"] = cols["city_name"].map(
                        lambda c: city_to_geo_name(c, aliases)
                    )
                    mgeo = gdf.merge(cols, on="name_vi_geo", how="left")
                    mgeo["bcg_quadrant"] = mgeo["bcg_quadrant"].fillna("none")
                    plot_choropleth_categorical(
                        mgeo,
                        "bcg_quadrant",
                        f"BCG — {REGION_LABELS[region]} / {ad_type} / {cat} {CAT_META[cat]}",
                        bcg_dir / f"fig_bcg_map_{region}_{ad_type}_{cat}.png",
                    )

    # --- GeoJSON kèm metric ---
    export_merged_geojson(
        gdf, total, aliases, MAP_DIR / "vietnam_provinces_explicit_total.geojson"
    )

    # Unmapped cities check
    m = total.copy()
    m["name_vi_geo"] = m["city_name"].map(lambda c: city_to_geo_name(c, aliases))
    geo_names = set(gdf["name_vi_geo"])
    unmapped = m[~m["name_vi_geo"].isin(geo_names)]["city_name"].unique()
    if len(unmapped):
        pd.DataFrame({"city_name": unmapped}).to_csv(
            MAP_DIR / "unmapped_cities.csv", index=False
        )
        print("WARNING unmapped:", list(unmapped))

    print("Done →", MAP_DIR)


if __name__ == "__main__":
    main()
