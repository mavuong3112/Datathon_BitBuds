if USE_FILTERED:
    ev_from = f"read_parquet('{FILTERED_FILE.as_posix()}')"
    ev_where = f"category = {CAT_1020}"
    print("1020: filtered_events.parquet")
else:
    ev_from = f"read_parquet('{EVENTS_GLOB}')"
    ev_where = f"category = {CAT_1020} AND {LOGIN_WHERE}"
    print("1020: raw events login")

ttc = con.execute(f'''
    WITH ev AS (
        SELECT user_id, event_ts,
            CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END AS is_explicit,
            CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END AS is_pv
        FROM {ev_from} WHERE {ev_where}
    ),
    bounds AS (
        SELECT user_id,
            MIN(CASE WHEN is_pv=1 THEN event_ts END) AS first_pv_ts,
            MIN(CASE WHEN is_explicit=1 THEN event_ts END) AS first_contact_ts
        FROM ev GROUP BY 1
        HAVING first_pv_ts IS NOT NULL AND first_contact_ts IS NOT NULL
           AND first_contact_ts >= first_pv_ts
    ),
    lag AS (
        SELECT date_diff('day', CAST(first_pv_ts AS DATE), CAST(first_contact_ts AS DATE)) AS days
        FROM bounds
    )
    SELECT COUNT(*)::BIGINT AS users,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY days),1) AS p50,
        ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY days),1) AS p75,
        ROUND(percentile_cont(0.9) WITHIN GROUP (ORDER BY days),1) AS p90,
        ROUND(100.0*SUM(CASE WHEN days=0 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_same_day,
        ROUND(100.0*SUM(CASE WHEN days>=3 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_3plus_days
    FROM lag
''').df()
show_df(ttc, "Time to first contact")
if EXPORT_CSV:
    ttc.to_csv(OUT["deep1020"] / "01_time_to_first_contact_days.csv", index=False)

fresh = con.execute(f'''
    WITH fc AS (
        SELECT user_id, CAST(item_id AS VARCHAR) AS item_id, MIN(CAST(date AS DATE)) AS contact_date
        FROM {ev_from}
        WHERE {ev_where} AND event_type IN ({EXPLICIT_SQL}) AND item_id IS NOT NULL
        GROUP BY 1, 2
    ),
    j AS (
        SELECT CASE WHEN s.listing_age_days IS NULL THEN 'unknown'
            WHEN s.listing_age_days <= 3 THEN '0-3d fresh'
            WHEN s.listing_age_days <= 14 THEN '4-14d'
            WHEN s.listing_age_days <= 30 THEN '15-30d' ELSE '31d+' END AS age_bucket,
            COUNT(*)::BIGINT AS n
        FROM fc
        LEFT JOIN read_parquet('{SNAP_GLOB}') s
            ON fc.item_id = CAST(s.item_id AS VARCHAR) AND fc.contact_date = CAST(s.date AS DATE)
        GROUP BY 1
    )
    SELECT *, ROUND(100.0*n/SUM(n) OVER(),2) AS pct FROM j ORDER BY n DESC
''').df()
show_df(fresh, "Listing age at contact")
if EXPORT_CSV:
    fresh.to_csv(OUT["deep1020"] / "02_contact_by_listing_age.csv", index=False)

repeat_pv = con.execute(f'''
    WITH ev AS (
        SELECT user_id, CAST(item_id AS VARCHAR) AS item_id, event_ts, event_type
        FROM {ev_from} WHERE {ev_where}
    ),
    fc AS (
        SELECT user_id, item_id, MIN(event_ts) AS t0
        FROM ev WHERE event_type IN ({EXPLICIT_SQL}) GROUP BY 1, 2
    ),
    pv AS (
        SELECT e.user_id, e.item_id, COUNT(*)::BIGINT AS n_pv
        FROM ev e JOIN fc f USING (user_id, item_id)
        WHERE e.event_type='pageview' AND e.event_ts < f.t0
        GROUP BY 1, 2
    )
    SELECT COUNT(*)::BIGINT AS pairs,
        ROUND(100.0*SUM(CASE WHEN n_pv>=2 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_2plus_pv,
        ROUND(100.0*SUM(CASE WHEN n_pv>=3 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_3plus_pv
    FROM pv
''').df()
show_df(repeat_pv, "Repeat PV same listing")
if EXPORT_CSV:
    repeat_pv.to_csv(OUT["deep1020"] / "03_repeat_pv_same_listing.csv", index=False)

split = con.execute(f'''
    WITH ev AS (
        SELECT e.user_id, e.session_id, CAST(e.item_id AS VARCHAR) AS item_id,
            e.date, e.event_ts, e.event_type,
            CASE WHEN e.event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END AS is_explicit,
            CASE WHEN e.event_type='pageview' THEN 1 ELSE 0 END AS is_pv
        FROM {ev_from} e
        WHERE e.category = {CAT_1020}
    ),
    ev_ad AS (
        SELECT ev.*, COALESCE(d.ad_type,'unknown') AS ad_type
        FROM ev
        LEFT JOIN read_parquet('{DIM_GLOB}') d
            ON ev.item_id = CAST(d.item_id AS VARCHAR) AND d.category = {CAT_1020}
    ),
    fc AS (
        SELECT ad_type, user_id, MIN(event_ts) AS t0
        FROM ev_ad WHERE is_explicit=1 GROUP BY 1, 2
    ),
    pu AS (
        SELECT e.ad_type, e.user_id,
            COUNT(DISTINCT CASE WHEN e.is_explicit=1 THEN e.item_id END)::BIGINT AS contacted,
            COUNT(DISTINCT CASE WHEN e.is_pv=1 AND e.event_ts<f.t0 THEN e.date END)::BIGINT AS pv_days
        FROM ev_ad e
        JOIN fc f ON e.ad_type=f.ad_type AND e.user_id=f.user_id
        GROUP BY 1, 2
    )
    SELECT ad_type, COUNT(*)::BIGINT AS users,
        ROUND(median(contacted),1) AS med_contact,
        ROUND(avg(contacted),2) AS mean_contact,
        ROUND(100.0*SUM(CASE WHEN pv_days>=2 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_2plus_days
    FROM pu GROUP BY 1 ORDER BY users DESC
''').df()
show_df(split, "1020 let vs sell")
if EXPORT_CSV:
    split.to_csv(OUT["deep1020"] / "04_metrics_by_ad_type.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
if not ttc.empty:
    ax.bar(["p50", "p75", "p90"], [ttc.iloc[0]["p50"], ttc.iloc[0]["p75"], ttc.iloc[0]["p90"]], color="#6a51a3")
    ax.set_ylabel("Days")
    ax.set_title("1020 time to first contact")
save_fig(OUT["deep1020"] / "fig_time_to_contact.png")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(fresh["age_bucket"], fresh["pct"], color="#fd8d3c")
plt.xticks(rotation=25, ha="right")
ax.set_ylabel("% contacts")
save_fig(OUT["deep1020"] / "fig_listing_age_at_contact.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(split["ad_type"], split["pct_2plus_days"], color="#2171b5")
ax.set_ylabel("% users >=2 PV days")
ax.set_title("1020 let vs sell")
save_fig(OUT["deep1020"] / "fig_let_vs_sell_repeat.png")
