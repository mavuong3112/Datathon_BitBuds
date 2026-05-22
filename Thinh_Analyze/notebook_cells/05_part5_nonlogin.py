cmp = con.execute(f'''
    SELECT is_login, COUNT(*)::BIGINT AS events,
        ROUND(100.0*SUM(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END)/COUNT(*),4) AS pct_explicit,
        ROUND(100.0*SUM(CASE WHEN event_type IN ({POSITIVE_SQL}) THEN 1 ELSE 0 END)/COUNT(*),4) AS pct_positive
    FROM read_parquet('{EVENTS_GLOB}') TABLESAMPLE {SAMPLE_PCT_QA} PERCENT (SYSTEM)
    WHERE is_login IN ('login','non-login') AND category IN ({CAT_IN})
    GROUP BY 1 ORDER BY 1
''').df()
show_df(cmp, f"Login vs non-login ({SAMPLE_PCT_QA}% sample)")

sess = con.execute(f'''
    WITH ev AS (
        SELECT session_id, is_login,
            MAX(CASE WHEN event_type IN ({EXPLICIT_SQL}) THEN 1 ELSE 0 END) AS has_explicit
        FROM read_parquet('{EVENTS_GLOB}') TABLESAMPLE {SAMPLE_PCT_QA} PERCENT (SYSTEM)
        WHERE session_id IS NOT NULL AND is_login IN ('login','non-login')
        GROUP BY 1, 2
    )
    SELECT is_login, COUNT(*)::BIGINT AS sessions,
        ROUND(100.0*SUM(has_explicit)/COUNT(*),2) AS pct_sess_explicit
    FROM ev GROUP BY 1
''').df()
show_df(sess, "Session explicit rate")
if EXPORT_CSV:
    cmp.to_csv(OUT["nonlogin"] / "01_login_vs_nonlogin_rates.csv", index=False)
    sess.to_csv(OUT["nonlogin"] / "02_session_explicit_rate.csv", index=False)

chat_cmp = con.execute(f'''
    WITH ev AS (
        SELECT is_login, CAST(user_id AS VARCHAR) AS user_id,
               CAST(item_id AS VARCHAR) AS item_id, CAST(date AS DATE) AS dt
        FROM read_parquet('{EVENTS_GLOB}') TABLESAMPLE {SAMPLE_PCT_QA} PERCENT (SYSTEM)
        WHERE event_type='contact_chat' AND is_contact=1 AND item_id IS NOT NULL
          AND is_login IN ('login','non-login')
    ),
    inter AS (
        SELECT CAST(user_id AS VARCHAR) AS user_id, CAST(item_id AS VARCHAR) AS item_id,
               CAST(date AS DATE) AS dt, chat_message_count
        FROM read_parquet('{INTER_GLOB}')
    ),
    joined AS (
        SELECT e.is_login,
            CASE WHEN i.user_id IS NULL THEN 'no_match'
                 WHEN COALESCE(i.chat_message_count,0)=0 THEN 'msg_zero'
                 ELSE 'msg_gt0' END AS bucket
        FROM ev e
        LEFT JOIN inter i ON e.user_id=i.user_id AND e.item_id=i.item_id AND e.dt=i.dt
    )
    SELECT is_login, bucket, COUNT(*)::BIGINT AS n FROM joined GROUP BY 1, 2 ORDER BY 1, 2
''').df()
pivot_chat = chat_cmp.pivot(index="bucket", columns="is_login", values="n").fillna(0)
show_df(pivot_chat, "contact_chat x chat_message_count")

mix = con.execute(f'''
    SELECT is_login, event_type, COUNT(*)::BIGINT AS n
    FROM read_parquet('{EVENTS_GLOB}') TABLESAMPLE {SAMPLE_PCT_QA} PERCENT (SYSTEM)
    WHERE is_login = 'non-login' AND category IN ({CAT_IN})
    GROUP BY 1, 2 ORDER BY n DESC LIMIT 12
''').df()
show_df(mix, "Top event types — non-login")

if EXPORT_CSV:
    chat_cmp.to_csv(OUT["nonlogin"] / "03_contact_chat_interaction_buckets.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(sess))
ax.bar(x, sess["pct_sess_explicit"], color=["#2171b5", "#cb181d"][: len(sess)])
ax.set_xticks(x)
ax.set_xticklabels(sess["is_login"])
ax.set_ylabel("% sessions explicit")
save_fig(OUT["nonlogin"] / "fig_session_explicit_rate.png")

if not pivot_chat.empty:
    pivot_chat.plot(kind="bar", figsize=(8, 4), colormap="Pastel1")
    plt.ylabel("contact_chat rows")
    plt.title("Join buckets")
    plt.xticks(rotation=0)
    plt.legend(title="is_login")
    save_fig(OUT["nonlogin"] / "fig_chat_message_buckets.png")

if not mix.empty:
    top = mix.head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["event_type"], top["n"], color="#636363")
    ax.invert_yaxis()
    ax.set_xlabel("events (non-login sample)")
    save_fig(OUT["nonlogin"] / "fig_nonlogin_event_mix.png")
