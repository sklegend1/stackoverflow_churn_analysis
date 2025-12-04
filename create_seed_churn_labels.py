import polars as pl

# -------------------------------------------
# 0) Load core event log
# -------------------------------------------
elog = pl.scan_parquet("core_eventlog_v1.parquet")

# -------------------------------------------
# A) Persona-level truncation based on last_ts_user
#    Step A1: user-level last_ts_user (on full log)
# -------------------------------------------
user_last_full = (
    elog
    .group_by(["persona", "user_id"])
    .agg([
        pl.col("timestamp").max().alias("last_ts_user"),
    ])
)

# Step A2: persona-level truncation timestamp:
# trunc_ts_persona^p = q-quantile(last_ts_user)  (here q = 0.90)
persona_trunc = (
    user_last_full
    .group_by("persona")
    .agg([
        pl.col("last_ts_user")
          .quantile(0.90, interpolation="nearest")
          .alias("trunc_ts_persona")
    ])
)

# Step A3: truncate event log per persona at trunc_ts_persona
# elog_trunc = (
#     elog
#     .join(persona_trunc, on="persona", how="inner")
#     .filter(pl.col("timestamp") <= pl.col("trunc_ts_persona"))
    
#     # .drop("trunc_ts_persona")  # no longer needed inside elog
# )
elog_trunc = elog
# -------------------------------------------
# 1) Persona-level gap stats + Tp_seed + cutoff_ts_persona
#    On truncated event log:
#    gap_p95_sec^p = p95(gap_time_key_actions_seconds | persona p, gap>0)
#    Tp_seed_sec^p = 3 * gap_p95_sec^p
#    cutoff_ts_persona^p = trunc_ts_persona^p - Tp_seed_sec^p
# -------------------------------------------

# Compute gap stats first
gap_stats = (
    elog_trunc
    .filter(pl.col("gap_time_key_actions_seconds").is_not_null())
    .filter(pl.col("gap_time_key_actions_seconds") > 0)
    .group_by("persona")
    .agg([
        # 95th percentile of key-action gaps (seconds) on truncated log
        pl.col("gap_time_key_actions_seconds")
          .quantile(0.95, interpolation="nearest")
          .alias("gap_p95_sec"),
    ])
)

# Collect gap_stats to materialize it, then join with persona_trunc
persona_stats = (
    gap_stats
    .collect()  # Materialize the aggregation first
    .lazy()     # Convert back to lazy for further operations
    .join(persona_trunc, on="persona", how="left")
    .with_columns([
        # 3 * p95 for Tp_seed (in seconds)
        (pl.col("gap_p95_sec") * 3.0).alias("Tp_seed_sec"),
    ])
    .with_columns([
        # cutoff_ts = trunc_ts_persona - Tp_seed_sec (interpreted as seconds)
        (
            # pl.col("trunc_ts_persona")
            # - pl.duration(seconds=pl.col("Tp_seed_sec").cast(pl.Int64))
            pl.date(2019,1,1) 
        ).alias("cutoff_ts_persona"),
    ])
)

# -------------------------------------------
# 2) User-level last activity timestamp per persona (on truncated log)
# -------------------------------------------
user_last = (
    elog_trunc
    .group_by(["persona", "user_id"])
    .agg([
        pl.col("timestamp").max().alias("last_ts_user"),
    ])
)

# -------------------------------------------
# 3) Assign persona-level cutoff to each user and compute churn_seed
#    churn_seed = 1 if last_ts_user <= cutoff_ts_persona  else 0
# -------------------------------------------
user_with_cutoff = (
    user_last
    .join(
        persona_stats.select(["persona", "Tp_seed_sec", "cutoff_ts_persona"]),
        on="persona",
        how="left",
    )
    .with_columns([
        (pl.col("last_ts_user") <= pl.col("cutoff_ts_persona"))
        .cast(pl.Int8)
        .alias("churn_seed")
    ])
)

# -------------------------------------------
# 4) Collect and save
# -------------------------------------------
df_users = user_with_cutoff.collect()
df_users.write_parquet("seed_churn_labels_per_user.parquet")

# Cast any Duration/Timedelta columns to int before writing sample CSV
duration_cols = [
    name
    for name, dt in zip(df_users.columns, df_users.dtypes)
    if "Duration" in str(dt) or "timedelta" in str(dt)
]

df_users_for_csv = (
    df_users
    .with_columns([pl.col(c).cast(pl.Int64).alias(c) for c in duration_cols])
)

df_users_for_csv.head(100).write_csv("seed_churn_labels_per_user_sample.csv")

print("Saved seed_churn_labels_per_user.(parquet/csv)")