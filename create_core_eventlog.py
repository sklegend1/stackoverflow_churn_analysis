import polars as pl
import pandas as pd

# -------------------------------
# 1) Load core activities (from XGBoost output)
# -------------------------------
core_df = pl.read_csv("core_activity_per_persona_named.csv")
core = core_df.lazy()
# Keep only rows that are real actions (have action_type_id)
core = core.filter(pl.col("action_type_id").is_not_null())

# Ensure correct dtypes
core = core.with_columns([
    pl.col("persona").cast(pl.Int32),
    pl.col("action_type_id").cast(pl.Int32),
    pl.col("gain").cast(pl.Float64),
])

# For each persona, keep top 7 actions by gain
core_top = (
    core
    .sort(["persona", "gain"], descending=[False, True])
    .group_by("persona")
    .head(7)
    .select(["persona", "action_type_id", "gain"])
)

# -------------------------------
# 2) Load full eventlog (lazy)
# -------------------------------
elog = pl.scan_parquet("eventlog_v2.parquet")

# Expected columns in eventlog_v2:
# user_id, timestamp, action_type_id, post_id, post_type_id,
# persona, LastAccessDate, AccountCreationDate,
# gap_time_actions_days, gap_time_actions_seconds, id

# -------------------------------
# 3) Filter eventlog to core actions per persona
#    using join on (persona, action_type_id)
# -------------------------------
core_eventlog_lazy = (
    elog
    .join(
        core_top,
        on=["persona", "action_type_id"],
        how="inner"
    )
    .sort(["user_id", "timestamp"])
)

# -------------------------------
# 4) Compute gap_time_key_actions per user
#    (only between core events)
# -------------------------------
core_eventlog_lazy = core_eventlog_lazy.with_columns([
    # gap in seconds between consecutive core actions for each user
    (
        pl.col("timestamp")
        .diff()
        .over("user_id") / 1000000
    ).alias("gap_time_key_actions_seconds"),
    (
        pl.col("timestamp")
        .diff()
        .over("user_id") / (1000000 * 60 * 60 * 24)
    ).alias("gap_time_key_actions_days"),
])

core_eventlog_lazy = core_eventlog_lazy.drop(
    [c for c in core_eventlog_lazy.columns if c.startswith("gap_time_actions")]+["id"]
)

# Optionally add a new primary key for core log
core_eventlog_lazy = core_eventlog_lazy.with_row_index("id")

# -------------------------------
# 5) Collect & write to parquet
# -------------------------------
print("Collecting core_event_log with streaming engine...")
core_eventlog = core_eventlog_lazy.collect(engine="streaming")

print("Created core_event_log with", core_eventlog.height, "rows.")
core_eventlog.write_parquet("core_eventlog_v1.parquet")
print("Done.")
