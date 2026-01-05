import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl

df = pl.read_parquet("../../data/processed/core_eventlog_v1.parquet")
print("Loaded core_eventlog_v1.parquet with", df.columns)
users = df["user_id"].unique().head(100)
# convert gap_time to integer seconds
df = df.with_columns([
    pl.col("gap_time_key_actions_days").cast(pl.Float32),
    pl.col("gap_time_key_actions_seconds").cast(pl.Int64)
])

df.filter(pl.col("user_id").is_in(users.implode())).write_csv("eventlog_sample_100_users.csv")