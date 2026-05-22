con.execute(f'''
    CREATE OR REPLACE TEMP TABLE dim_scoped AS
    SELECT CAST(item_id AS VARCHAR) AS item_id, seller_id, category, ad_type,
        title, regexp_replace(lower(trim(coalesce(title,''))), '\\s+', ' ', 'g') AS title_norm,
        regexp_replace(lower(trim(coalesce(city_name,''))), '\\s+', ' ', 'g') AS city_norm,
        regexp_replace(lower(trim(coalesce(district_name,''))), '\\s+', ' ', 'g') AS district_norm,
        regexp_replace(lower(trim(coalesce(ward_name,''))), '\\s+', ' ', 'g') AS ward_norm,
        regexp_replace(lower(trim(coalesce(price_bucket,''))), '\\s+', ' ', 'g') AS price_norm,
        area_sqm
    FROM read_parquet('{DIM_GLOB}')
    WHERE posted_date BETWEEN DATE '{EDA_MIN}' AND DATE '{EDA_MAX}'
      AND title IS NOT NULL AND trim(title) <> ''
''')

LEVELS = {
    "L1_title_category": "title_norm, category",
    "L2_title_geo_price": "title_norm, category, ad_type, city_norm, district_norm, price_norm",
    "L3_title_full_fingerprint": (
        "title_norm, category, ad_type, seller_id, city_norm, district_norm, ward_norm, "
        "price_norm, round(coalesce(area_sqm,-1),1)"
    ),
    "L4_title_seller": "title_norm, category, seller_id",
}
n_scoped = con.execute("SELECT COUNT(*) FROM dim_scoped").fetchone()[0]
level_rows = []
for level_name, group_cols in LEVELS.items():
    row = con.execute(
        f'''
        WITH grp AS (
            SELECT {group_cols}, COUNT(*)::BIGINT AS n_items,
                   COUNT(DISTINCT seller_id) AS n_sellers
            FROM dim_scoped GROUP BY ALL HAVING COUNT(*) > 1
        )
        SELECT '{level_name}' AS dup_level,
            COUNT(*)::BIGINT AS duplicate_groups,
            COALESCE(SUM(n_items),0)::BIGINT AS rows_in_dup_groups,
            ROUND(100.0*COALESCE(SUM(n_items),0)/{n_scoped},2) AS pct_rows_in_dup_groups,
            COALESCE(SUM(CASE WHEN n_sellers=1 THEN 1 ELSE 0 END),0)::BIGINT AS groups_same_seller,
            COALESCE(SUM(CASE WHEN n_sellers>1 THEN 1 ELSE 0 END),0)::BIGINT AS groups_multi_seller
        FROM grp
        '''
    ).df().iloc[0].to_dict()
    level_rows.append(row)

dup_summary = pd.DataFrame(level_rows)
show_df(dup_summary, "Duplicate levels")
if EXPORT_CSV:
    dup_summary.to_csv(OUT["duplicate"] / "01_duplicate_levels_summary.csv", index=False)

by_cat = con.execute(
    '''
    WITH grp AS (
        SELECT category, title_norm, ad_type, city_norm, district_norm, price_norm,
               COUNT(*)::BIGINT AS n_items, COUNT(DISTINCT seller_id) AS n_sellers
        FROM dim_scoped GROUP BY ALL HAVING COUNT(*) > 1
    )
    SELECT category, COUNT(*)::BIGINT AS duplicate_groups,
        SUM(n_items)::BIGINT AS rows_in_dup_groups,
        ROUND(100.0*SUM(n_items)/(SELECT COUNT(*) FROM dim_scoped d WHERE d.category=grp.category),2) AS pct_cat,
        SUM(CASE WHEN n_sellers=1 THEN 1 ELSE 0 END)::BIGINT AS same_seller,
        SUM(CASE WHEN n_sellers>1 THEN 1 ELSE 0 END)::BIGINT AS multi_seller
    FROM grp GROUP BY 1 ORDER BY 1
    '''
).df()
show_df(by_cat, "L2 by category")
if EXPORT_CSV:
    by_cat.to_csv(OUT["duplicate"] / "02_L2_duplicate_by_category.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(dup_summary["dup_level"], dup_summary["pct_rows_in_dup_groups"], color="#756bb1")
ax.set_xlabel("% listings in dup groups")
save_fig(OUT["duplicate"] / "fig_duplicate_levels.png")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(by_cat))
ax.bar(x - 0.2, by_cat["same_seller"], width=0.4, label="Cùng seller")
ax.bar(x + 0.2, by_cat["multi_seller"], width=0.4, label="Nhiều seller")
ax.set_xticks(x)
ax.set_xticklabels(by_cat["category"].astype(str))
ax.legend()
save_fig(OUT["duplicate"] / "fig_L2_same_vs_multi_seller.png")
