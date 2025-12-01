import duckdb
import time

start = time.time()
# duckdb.sql("DESCRIBE 'user_feature_matrix.parquet'").show(max_rows=100)
res = duckdb.sql("""
    SELECT *
    FROM 'feature_matrix_v2.parquet' 
    limit 100
    
    
""").write_csv("feature_matrix_sample.csv")

end = time.time()
print(f"Query executed in {end - start:.2f} seconds")
