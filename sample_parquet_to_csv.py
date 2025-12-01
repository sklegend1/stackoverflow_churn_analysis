import polars as pl

df = pl.read_parquet("eventlog_v2.parquet")

users = df["user_id"].unique().head(100)
# convert gap_time to integer seconds
df = df.with_columns([
    pl.col("gap_time_actions_days").cast(pl.Float32),
    pl.col("gap_time_actions_seconds").cast(pl.Int64)
])

df.filter(pl.col("user_id").is_in(users.implode())).write_csv("eventlog_sample_100_users.csv")