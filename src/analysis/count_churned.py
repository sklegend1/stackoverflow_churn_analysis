import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import duckdb
import time

start = time.time()
# duckdb.sql("DESCRIBE '../../data/processed/user_feature_matrix.parquet'").show(max_rows=100)
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
    FROM '../../data/processed/seed_churn_labels_per_user.parquet'
    GROUP BY persona
    ORDER BY persona
""").write_csv("../../data/processed/churn_seed_count.csv")

end = time.time()
print(f"Query executed in {end - start:.2f} seconds")
