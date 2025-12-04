import duckdb
import time

start = time.time()
# duckdb.sql("DESCRIBE 'user_feature_matrix.parquet'").show(max_rows=100)
res = duckdb.sql("""
    SELECT
        persona,
        MAX(cutoff_ts_persona) AS cutoff_ts_persona,
        SUM(CASE WHEN churn_seed = 1 THEN 1 ELSE 0 END) AS churned,
        SUM(CASE WHEN churn_seed = 0 THEN 1 ELSE 0 END) AS unchurned,
        ROUND(
            100.0 * SUM(CASE WHEN churn_seed = 1 THEN 1 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN churn_seed IN (0,1) THEN 1 ELSE 0 END), 0)
        , 2) AS churn_percentage
    FROM 'seed_churn_labels_per_user.parquet'
    GROUP BY persona
    ORDER BY persona
""").write_csv("churn_seed_count.csv")

end = time.time()
print(f"Query executed in {end - start:.2f} seconds")
