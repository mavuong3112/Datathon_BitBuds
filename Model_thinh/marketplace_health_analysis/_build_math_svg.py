"""
Generates the math notation panel as SVG + PNG.

SVG uses `svg.fonttype = 'path'` so text is converted to vector paths.
This ensures the math notation renders IDENTICALLY on any system regardless
of installed fonts — no font fallback issues with Σ, α, β, λ, σ, Δ, ·, etc.

Output:
  slides/charts/math_notation.svg   (vector, universal rendering)
  slides/charts/math_notation.png   (high-res raster, PowerPoint-safe)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# CRITICAL: convert text → vector paths in SVG → no font dependency
matplotlib.rcParams['svg.fonttype'] = 'path'

OUT_DIR = Path('/Volumes/mavuong3112/Datathon_Data/marketplace_health_analysis/slides/charts')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors (match slide design) ───────────────────────────────────────────────
SLATE     = '#1B2538'
WHITE     = '#FFFFFF'
DARK_TEXT = '#102030'
BORDER    = '#ECEFF3'

# ── Math content (ASCII-safe, but Σ/α/β/λ/σ/Δ are universal) ──────────────────
math_body = """ALS [1]      L = Σ c_ui (p_ui - u_u^T v_i)^2 + λ(||U||^2 + ||V||^2)
             c_ui = 1 + α·r_ui ;  α=40 ; factors=512

EASE [2]     B = (G + λI)^-1 ;  B_ii <- 0 ;  G = X^T X ;  λ=200

ItemCF [3]   s(i,j) = Σ_u w(t)·1[u,i]·1[u,j] / sqrt(...)
             w(t) = exp(-β·Δt) ;  β=0.005 (~140d half-life)

SASRec [4]   SA(Q,K,V) = softmax(QK^T / sqrt(d))·V
             d=64, heads=2, blocks=2, L_max=50

CBF [8]      e_i = PhoBERT(title_i) in R^768 ;  cos(e_u, e_i)

LambdaRank [5,6]
   L = Σ log(1 + exp(-σ(s_i - s_j))) · |Δ NDCG_ij|

s_final = 0.65·s_LGBM_norm + 0.35·s_XGB_norm
          + β_f·1[fresh] + β_c·1[cat_match]"""

# ── Figure setup ──────────────────────────────────────────────────────────────
# Match aspect ratio of slide panel (5.0" × 3.10" in original deck)
FIG_W, FIG_H = 9.0, 5.8
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_axis_off()

# Rounded panel background (white with light border)
panel = mpatches.FancyBboxPatch(
    (0.005, 0.005), 0.99, 0.99,
    boxstyle='round,pad=0,rounding_size=0.012',
    facecolor=WHITE, edgecolor=BORDER, linewidth=1.0,
    transform=ax.transAxes
)
ax.add_patch(panel)

# Slate header bar (top ~10%)
HEADER_H = 0.10
header_bar = mpatches.FancyBboxPatch(
    (0.005, 1 - HEADER_H - 0.005), 0.99, HEADER_H,
    boxstyle='round,pad=0,rounding_size=0.012',
    facecolor=SLATE, edgecolor='none',
    transform=ax.transAxes
)
ax.add_patch(header_bar)

# Header title (white text on slate)
ax.text(0.025, 1 - HEADER_H/2 - 0.005,
        'MATH NOTATION  ·  5 RETRIEVERS + RANKER',
        fontsize=13, fontweight='bold', color=WHITE,
        ha='left', va='center',
        family='DejaVu Sans',
        transform=ax.transAxes)

# Math body text (monospace, DejaVu Sans Mono handles Σ/α/β/λ/σ/Δ universally)
ax.text(0.025, 0.85, math_body,
        fontsize=10.5, color=DARK_TEXT,
        ha='left', va='top',
        family='DejaVu Sans Mono',
        linespacing=1.32,
        transform=ax.transAxes)

# ── Save SVG (vector, text-as-paths → font-independent) ───────────────────────
svg_path = OUT_DIR / 'math_notation.svg'
fig.savefig(str(svg_path), format='svg',
            facecolor=WHITE, edgecolor='none')
print(f'✓ SVG saved: {svg_path}  ({svg_path.stat().st_size // 1024} KB)')

# ── Save PNG (high-res raster — PowerPoint-friendly) ─────────────────────────
png_path = OUT_DIR / 'math_notation.png'
fig.savefig(str(png_path), format='png', dpi=300,
            facecolor=WHITE, edgecolor='none')
print(f'✓ PNG saved: {png_path}  ({png_path.stat().st_size // 1024} KB)')

plt.close(fig)
print('\nDone. SVG renders identically on any system (text → vector paths).')
print('PNG is fallback if your viewer/PowerPoint version doesn\'t handle SVG.')
