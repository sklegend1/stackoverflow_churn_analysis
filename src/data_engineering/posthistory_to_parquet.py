import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import duckdb
import os
# from duckdb_extensions import import_extension
# import_extension("mysql")

import time

# Config: replace these with your actual values
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3307
MYSQL_USER = "root"
MYSQL_PASSWORD = "rootpass"
MYSQL_DATABASE = "xml_db"
MYSQL_TABLE = "posthistory"  # change if your table name is different

PARQUET_DIR = "../../data/processed/posthistory_parquet"
SINGLE_FILE_PATH = "../../data/processed/posthistory.parquet"

os.makedirs(PARQUET_DIR, exist_ok=True)

# Create DuckDB connection (file-based to persist metadata if needed)
con = duckdb.connect("mysql_export.duckdb")

# Install and load MySQL extension (needs internet on first INSTALL)
con.execute("INSTALL mysql;")
con.execute("LOAD mysql;")

print("MySQL extension loaded.")

# Build MySQL connection string
connection_string = (
    f"host={MYSQL_HOST} "
    f"port={MYSQL_PORT} "
    f"user={MYSQL_USER} "
    f"database={MYSQL_DATABASE} "
    f"password={MYSQL_PASSWORD}"
)
con.execute(f"ATTACH '{connection_string}' AS mysqldb (TYPE mysql);")
con.execute("USE mysqldb;")
show_tables = con.execute("SHOW ALL TABLES;").fetchall()
#print("Available tables in MySQL database:", show_tables)
# Get min and max ID from the posthistory table
min_max_query = f"""
SELECT 
    MIN(id) AS min_id,
    MAX(id) AS max_id
FROM {MYSQL_DATABASE}.{MYSQL_TABLE};
"""
ten_rows = f"""
SELECT * FROM {MYSQL_TABLE} LIMIT 1;
"""

print("Fetching min and max id...")
#min_id, max_id = con.execute(min_max_query).fetchone()
min_id = 6
max_id = 304883740
ten_result = con.execute(ten_rows).fetchall()
print(ten_result)
print(f"min_id={min_id}, max_id={max_id}")

CHUNK_SIZE = 10_000_000  # adjust based on your table size and laptop limits

current_start = 6
part_index = 0

while current_start <= max_id:
    start = time.time()
    current_end = current_start + CHUNK_SIZE - 1

    part_path = os.path.join(
        PARQUET_DIR,
        f"posthistory_part_{part_index:05d}.parquet"
    )

    chunk_query = f"""
    COPY (
        SELECT *
        FROM {MYSQL_TABLE}
        WHERE id BETWEEN {current_start} AND {current_end}
    )
    TO '{part_path}'
    (FORMAT PARQUET);
    """

    print(f"Exporting chunk {part_index} (id between {current_start} and {current_end})...")
    con.execute(chunk_query)
    print(f"Written: {part_path}")

    part_index += 1
    current_start = current_end + 1
    # Calculate time of execution for each batch
    end = time.time()
    print(f"Batch execution time: {end - start:.4f} seconds")

print("All chunks exported.")


