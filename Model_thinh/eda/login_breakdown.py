import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

conn = duckdb.connect()
conn.execute("SET memory_limit='20GB'")
conn.execute("SET threads=4")
conn.execute("SET enable_progress_bar=false")

files = [f.replace('\\', '/') for f in glob.glob('d:/Datathon_Data/fact_user_events/*.parquet')]

CATEGORIES = {
    1010: "Phòng trọ\n/Cho thuê",
    1020: "Căn hộ\n/Chung cư",
    1030: "Nhà ở",
    1040: "Đất nền",
    1050: "Dự án mới",
}

print("Querying login vs non-login metrics per category …")

# ── Query: events, sessions, unique users (login) per category × is_login ──
df = conn.execute(f"""
    SELECT
        category,
        is_login,
        COUNT(*)                    AS total_events,
        APPROX_COUNT_DISTINCT(session_id) AS unique_sessions,
        APPROX_COUNT_DISTINCT(
            CASE WHEN is_login='login' THEN user_id END
        )                           AS unique_login_users
    FROM read_parquet({files})
    WHERE category IN (1010,1020,1030,1040,1050)
    GROUP BY category, is_login
    ORDER BY category, is_login
""").df()

conn.close()

print(df.to_string(index=False))

# ── Reshape ──────────────────────────────────────────────────────────────────
login_df    = df[df['is_login'] == 'login'].set_index('category')
nonlogin_df = df[df['is_login'] == 'non-login'].set_index('category')

cats = list(CATEGORIES.keys())
cat_labels = [CATEGORIES[c] for c in cats]

# Three metrics to compare
metrics = {
    'total_events':     ('Total Events',   'Tổng sự kiện'),
    'unique_sessions':  ('Unique Sessions', 'Unique Sessions\n(proxy user count)'),
}

# ── Figure: 2 rows × 1 plot each (stacked 100% bar) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Login vs Non-Login per Category (%)', fontsize=13, fontweight='bold', y=1.01)

COLOR_LOGIN    = '#2171b5'
COLOR_NONLOGIN = '#fd8d3c'

for ax, (col, (eng_title, vn_title)) in zip(axes, metrics.items()):
    login_vals    = [login_df.loc[c, col]    if c in login_df.index    else 0 for c in cats]
    nonlogin_vals = [nonlogin_df.loc[c, col] if c in nonlogin_df.index else 0 for c in cats]

    totals = [l + n for l, n in zip(login_vals, nonlogin_vals)]
    login_pct    = [l / t * 100 if t else 0 for l, t in zip(login_vals, totals)]
    nonlogin_pct = [n / t * 100 if t else 0 for n, t in zip(nonlogin_vals, totals)]

    x = np.arange(len(cats))
    bar_w = 0.55

    b1 = ax.bar(x, login_pct,    bar_w, label='Login',     color=COLOR_LOGIN,    alpha=0.88)
    b2 = ax.bar(x, nonlogin_pct, bar_w, bottom=login_pct,  label='Non-Login',    color=COLOR_NONLOGIN, alpha=0.88)

    # Annotate %
    for i, (lp, np_) in enumerate(zip(login_pct, nonlogin_pct)):
        if lp > 4:
            ax.text(x[i], lp / 2, f'{lp:.1f}%',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if np_ > 4:
            ax.text(x[i], lp + np_ / 2, f'{np_:.1f}%',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Raw numbers below x-axis as secondary info
    for i, (lv, nv, t) in enumerate(zip(login_vals, nonlogin_vals, totals)):
        def fmt(v):
            if v >= 1e6: return f'{v/1e6:.1f}M'
            if v >= 1e3: return f'{v/1e3:.0f}K'
            return str(int(v))
        ax.text(x[i], -7, f'L:{fmt(lv)}\nN:{fmt(nv)}',
                ha='center', va='top', fontsize=7, color='#444', linespacing=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylim(-18, 108)
    ax.set_ylabel('Tỷ lệ (%)')
    ax.set_title(f'{eng_title}\n({vn_title})', fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axhline(50, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.spines[['top','right']].set_visible(False)

fig.tight_layout()
out = 'd:/Datathon_Data/eda/outputs/login_breakdown_stacked.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')

# ── Print summary table ───────────────────────────────────────────────────────
print('\n=== Summary Table (%) ===')
rows = []
for c in cats:
    le = login_df.loc[c, 'total_events']    if c in login_df.index    else 0
    ne = nonlogin_df.loc[c, 'total_events'] if c in nonlogin_df.index else 0
    ls = login_df.loc[c, 'unique_sessions']    if c in login_df.index    else 0
    ns = nonlogin_df.loc[c, 'unique_sessions'] if c in nonlogin_df.index else 0
    lu = login_df.loc[c, 'unique_login_users'] if c in login_df.index    else 0
    te, ts = le + ne, ls + ns
    rows.append({
        'Category': f"{c} {CATEGORIES[c].replace(chr(10),'')}",
        'Login Events %':    f'{le/te*100:.1f}%',
        'NonLogin Events %': f'{ne/te*100:.1f}%',
        'Login Sessions %':  f'{ls/ts*100:.1f}%',
        'NonLogin Sess %':   f'{ns/ts*100:.1f}%',
        'Unique Login Users': f'{lu:,.0f}',
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
