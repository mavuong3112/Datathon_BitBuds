print(f"Demand-side sample {SAMPLE_PCT}%...")
con.execute(f'''
    CREATE OR REPLACE TEMP TABLE base_events AS
    SELECT category, user_id, session_id, item_id, date, event_ts, event_type,
        CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END AS is_explicit,
        CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END AS is_pv,
        CASE WHEN event_type = 'pageview' AND query IS NOT NULL AND trim(CAST(query AS VARCHAR)) <> ''
             THEN regexp_replace(lower(trim(CAST(query AS VARCHAR))), '\\s+', ' ', 'g') ELSE NULL END AS qnorm
    FROM read_parquet('{EVENTS_GLOB}') TABLESAMPLE {SAMPLE_PCT} PERCENT (SYSTEM)
    WHERE {LOGIN_WHERE} AND category IN ({CAT_IN})
''')
con.execute('''
    CREATE OR REPLACE TEMP TABLE user_first_contact AS
    SELECT category, user_id, MIN(event_ts) AS t0, COUNT(DISTINCT item_id)::BIGINT AS contacted_listings
    FROM base_events WHERE is_explicit=1 AND item_id IS NOT NULL GROUP BY 1, 2
''')
con.execute('''
    CREATE OR REPLACE TEMP TABLE user_sessions AS
    WITH ss AS (
        SELECT category, user_id, session_id, MIN(event_ts) AS st FROM base_events
        WHERE session_id IS NOT NULL GROUP BY 1, 2, 3
    )
    SELECT s.category, s.user_id,
        SUM(CASE WHEN s.st < f.t0 THEN 1 ELSE 0 END)::BIGINT AS sessions_before_contact,
        COUNT(s.session_id)::BIGINT AS total_sessions
    FROM ss s JOIN user_first_contact f USING (category, user_id) GROUP BY 1, 2
''')
con.execute('''
    CREATE OR REPLACE TEMP TABLE user_pv_days AS
    SELECT e.category, e.user_id, COUNT(DISTINCT e.date)::BIGINT AS pv_days_before
    FROM base_events e JOIN user_first_contact f USING (category, user_id)
    WHERE e.is_pv=1 AND e.event_ts < f.t0 GROUP BY 1, 2
''')
con.execute('''
    CREATE OR REPLACE TEMP TABLE all_searches AS
    SELECT category, user_id, session_id, event_ts, qnorm,
        LAG(qnorm) OVER (PARTITION BY category, user_id, session_id ORDER BY event_ts) AS pq
    FROM base_events WHERE qnorm IS NOT NULL
''')
con.execute('''
    CREATE OR REPLACE TEMP TABLE user_refinements_before AS
    SELECT s.category, s.user_id,
        SUM(CASE WHEN s.pq IS NOT NULL AND s.qnorm <> s.pq THEN 1 ELSE 0 END)::BIGINT AS total_refinements
    FROM all_searches s JOIN user_first_contact f USING (category, user_id)
    WHERE s.event_ts < f.t0 GROUP BY 1, 2
''')

demand_df = con.execute('''
    SELECT c.category, COUNT(c.user_id)::BIGINT AS converting_users_sample,
        median(c.contacted_listings) AS median_contacted_listings,
        avg(c.contacted_listings) AS mean_contacted_listings,
        median(us.sessions_before_contact) AS median_sessions_before_contact,
        avg(CASE WHEN pd.pv_days_before >= 2 THEN 100.0 ELSE 0.0 END) AS pct_users_repeat_daily_2plus_pv_days,
        median(urb.total_refinements) AS median_total_refinements_before_contact
    FROM user_first_contact c
    LEFT JOIN user_sessions us USING (category, user_id)
    LEFT JOIN user_pv_days pd USING (category, user_id)
    LEFT JOIN user_refinements_before urb USING (category, user_id)
    GROUP BY c.category ORDER BY c.category
''').df()
demand_df["label"] = demand_df["category"].map(CAT_META)
show_df(demand_df, "Demand-side (sample)")
if EXPORT_CSV:
    demand_df.to_csv(OUT["demand"] / "01_demand_side_by_category_sampled.csv", index=False)

full_path = OUT["demand"] / "01_demand_side_by_category.csv"
if full_path.exists():
    show_df(pd.read_csv(full_path), "Demand-side FULL (tham chiếu)")

rank_metrics = [
    "median_contacted_listings",
    "median_sessions_before_contact",
    "pct_users_repeat_daily_2plus_pv_days",
    "median_total_refinements_before_contact",
]
rank_rows = [
    {
        "metric": m,
        "value_1020": demand_df.loc[demand_df["category"] == 1020, m].iloc[0],
        "rank_1020": int(demand_df.set_index("category")[m].rank(ascending=False).loc[1020]),
    }
    for m in rank_metrics
]
show_df(pd.DataFrame(rank_rows), "1020 rank (1 = cao nhất)")

merged = summary_p1.merge(demand_df, on="category", how="inner")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(merged))
w = 0.35
axes[0].bar(x - w / 2, merged["listing_gini"], width=w, label="Listing Gini")
axes[0].bar(x + w / 2, merged["user_gini"], width=w, label="User Gini")
axes[0].set_xticks(x)
axes[0].set_xticklabels(merged["category"].astype(str))
axes[0].set_title("Supply concentration (Part 1 sample)")
axes[0].legend(fontsize=8)
axes[1].bar(x - w / 2, merged["listing_top_10pct_share"], width=w, label="Top 10% listings")
axes[1].bar(x + w / 2, merged["user_top_10pct_share"], width=w, label="Top 10% users")
axes[1].set_xticks(x)
axes[1].set_xticklabels(merged["category"].astype(str))
axes[1].legend(fontsize=8)
save_fig(OUT["demand"] / "fig_gini_and_top10_from_part1.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(demand_df))
axes[0].bar(x, demand_df["pct_users_repeat_daily_2plus_pv_days"], color="#238b45")
axes[0].set_xticks(x)
axes[0].set_xticklabels(demand_df["category"].astype(str))
axes[0].set_ylabel("% users >=2 PV days")
axes[1].bar(x, demand_df["median_sessions_before_contact"], color="#2171b5")
axes[1].set_xticks(x)
axes[1].set_xticklabels(demand_df["category"].astype(str))
axes[1].set_ylabel("Median sessions before contact")
save_fig(OUT["demand"] / "fig_demand_signals.png")
