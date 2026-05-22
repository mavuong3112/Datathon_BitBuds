time_cov = con.execute(f'''
    SELECT * FROM (
        SELECT 'dim_listing.posted_date' AS src,
               CAST(MIN(posted_date) AS DATE) AS t_min,
               CAST(MAX(posted_date) AS DATE) AS t_max,
               COUNT(*)::BIGINT AS n_rows
        FROM read_parquet('{DIM_GLOB}')
        UNION ALL
        SELECT 'fact_user_events.date', MIN(date), MAX(date), COUNT(*)::BIGINT
        FROM read_parquet('{EVENTS_GLOB}')
    ) ORDER BY 1
''').df()
show_df(time_cov, "Time coverage")
if EXPORT_CSV:
    time_cov.to_csv(OUT["concentration"] / "00_time_coverage.csv", index=False)

con.execute(f'''
    CREATE OR REPLACE TEMP TABLE explicit_ev AS
    SELECT category, user_id, CAST(item_id AS VARCHAR) AS item_id, event_type
    FROM read_parquet('{EVENTS_GLOB}')
    TABLESAMPLE {SAMPLE_PCT} PERCENT (SYSTEM)
    WHERE {LOGIN_WHERE} AND category IN ({CAT_IN})
      AND event_type IN ({EXPLICIT_SQL}) AND item_id IS NOT NULL
''')

listing_all = con.execute(
    "SELECT category, item_id, COUNT(*)::BIGINT AS contacts FROM explicit_ev GROUP BY 1, 2"
).df()
user_all = con.execute(
    "SELECT category, user_id, COUNT(*)::BIGINT AS contacts FROM explicit_ev GROUP BY 1, 2"
).df()
channel_all = con.execute(
    "SELECT category, event_type, COUNT(*)::BIGINT AS contacts FROM explicit_ev GROUP BY 1, 2"
).df()
channel_all["pct_of_explicit"] = channel_all.groupby("category")["contacts"].transform(
    lambda s: (100.0 * s / s.sum()).round(2)
)

summary_rows = []
for cat in CATEGORIES:
    lc = listing_all.loc[listing_all["category"] == cat, "contacts"].to_numpy()
    uc = user_all.loc[user_all["category"] == cat, "contacts"].to_numpy()
    cl = concentration_row("listing", lc)
    cu = concentration_row("user", uc)
    summary_rows.append({
        "category": cat,
        "category_label": CAT_META[cat],
        "listing_gini": cl["gini"],
        "listing_top_10pct_share": cl["top_10pct_share"],
        "user_gini": cu["gini"],
        "user_top_10pct_share": cu["top_10pct_share"],
        "total_explicit_contacts": cl["total_contacts"],
    })

summary_p1 = pd.DataFrame(summary_rows)
show_df(summary_p1, f"Contact concentration — summary (sample {SAMPLE_PCT}%)")
if EXPORT_CSV:
    summary_p1.to_csv(OUT["concentration"] / "01_summary_all_categories.csv", index=False)
    channel_all.to_csv(OUT["concentration"] / "02_channel_mix_by_category.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(summary_p1))
w = 0.35
axes[0].bar(x - w / 2, summary_p1["listing_gini"], width=w, label="Listing Gini")
axes[0].bar(x + w / 2, summary_p1["user_gini"], width=w, label="User Gini")
axes[0].set_xticks(x)
axes[0].set_xticklabels(summary_p1["category"].astype(str))
axes[0].legend(fontsize=8)
axes[1].bar(x - w / 2, summary_p1["listing_top_10pct_share"], width=w, label="Top 10% listings")
axes[1].bar(x + w / 2, summary_p1["user_top_10pct_share"], width=w, label="Top 10% users")
axes[1].set_xticks(x)
axes[1].set_xticklabels(summary_p1["category"].astype(str))
axes[1].legend(fontsize=8)
save_fig(OUT["concentration"] / "fig_01_gini_and_top10_share.png")

piv = channel_all.pivot(index="category", columns="event_type", values="pct_of_explicit").fillna(0)
piv = piv.reindex(CATEGORIES)
piv.plot(kind="bar", stacked=True, figsize=(9, 4.5), colormap="Set2", edgecolor="white")
plt.ylabel("% explicit")
plt.title("Mix kênh liên hệ")
plt.xticks(rotation=0)
plt.legend(title="event_type", bbox_to_anchor=(1.02, 1), fontsize=8)
save_fig(OUT["concentration"] / "fig_02_channel_mix.png")
