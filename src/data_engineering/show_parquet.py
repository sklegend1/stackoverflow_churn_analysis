import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import duckdb
import time

start = time.time()
# duckdb.sql("DESCRIBE '../../data/processed/user_feature_matrix.parquet'").show(max_rows=100)
res = duckdb.sql("""
    SELECT *
    FROM '../../data/processed/feature_matrix_v2.parquet' 
    limit 100
    
    
""").write_csv("../../data/processed/feature_matrix_sample.csv")

end = time.time()
print(f"Query executed in {end - start:.2f} seconds")
