import duckdb
import time

start = time.time()
duckdb.sql("""
    SELECT *
    FROM 'eventlog_v1.parquet' 
    limit 10
    offset 500000
    
""").show()
end = time.time()
print(f"Query executed in {end - start:.2f} seconds")