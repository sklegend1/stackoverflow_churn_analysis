import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl
from datetime import datetime, timedelta

# ============================================================
# Load event log
# ============================================================
elog = pl.scan_parquet("../../data/processed/eventlog_v2.parquet")
# Remove invalid action rows
elog = elog.filter(pl.col("action_type_id").is_not_null())

# Recompute action IDs cleanly
ACTION_IDS = (
    elog
    .select("action_type_id")
    .unique()
    .collect()
    .to_series()
    .to_list()
)
# NOTE:
# eventlog_v2 MUST include:
#   user_id, timestamp, action_type_id,
#   gap_time_actions, persona,
#   AccountCreationDate, LastAccessDate
# ============================================================

# Ensure datetime casting
elog = elog.with_columns([
    pl.col("timestamp").cast(pl.Datetime),
    pl.col("LastAccessDate").cast(pl.Datetime),
    pl.col("AccountCreationDate").cast(pl.Date),
])

# ============================================================
# 1) Total Actions per User
# ============================================================
feature_total = (
    elog
    .group_by("user_id")
    .agg(pl.len().alias("total_actions"))
)

# ============================================================
# 2) Action Counts per ActionTypeId (Lazy pivot)
# ============================================================

# get unique action ids
ACTION_IDS = (
    elog.select(pl.col("action_type_id"))
    .unique()
    .collect()
    .to_series()
    .to_list()
)

user_action_counts = (
    elog
    .group_by(["user_id", "action_type_id"])
    .agg(pl.len().alias("freq"))
)

feature_action_counts = (
    user_action_counts
    .group_by("user_id")
    .agg([
        pl.col("freq")
        .filter(pl.col("action_type_id") == aid)
        .first()
        .fill_null(0)
        .alias(f"count_action_{aid}")
        for aid in ACTION_IDS
    ])
)

# ============================================================
# 3) Recency per user (relative to LastAccessDate of same user)
# ============================================================

recency = (
    elog
    .group_by("user_id")
    .agg([
        (pl.col("LastAccessDate") - pl.col("timestamp").max())
            .dt.total_days().first()
            .alias("days_since_last_action")
    ])
)

# ============================================================
# 4) Gap Time Features
# ============================================================

gap_stats = (
    elog
    .group_by("user_id")
    .agg([
        pl.col("gap_time_actions_seconds").mean().alias("gap_mean"),
        pl.col("gap_time_actions_seconds").std().alias("gap_std"),
        pl.col("gap_time_actions_seconds").max().alias("gap_max"),
        pl.col("gap_time_actions_seconds").min().alias("gap_min"),
        pl.col("gap_time_actions_seconds").quantile(0.95).alias("gap_p95"),
        pl.col("gap_time_actions_seconds").last().alias("gap_last"),
    ])
)

# ============================================================
# 5) Window features relative to each user's LastAccessDate
# ============================================================

def window_relative(days, colname):
    return (
        elog
        .filter(
            (pl.col("timestamp") <= pl.col("LastAccessDate")) &
            (pl.col("timestamp") >= (pl.col("LastAccessDate") - pl.duration(days=days)))
        )
        .group_by("user_id")
        .agg(pl.len().alias(colname))
    )

# window_30 = window_relative(30, "actions_last_30")
# window_90 = window_relative(90, "actions_last_90")
# window_180 = window_relative(180, "actions_last_180")

# ============================================================
# 6) User metadata
# ============================================================

users_meta = (
    elog
    .select(["user_id", "persona", "AccountCreationDate", "LastAccessDate"])
    .unique(subset=["user_id"])
    # .with_columns([
    #     (pl.col("LastAccessDate").cast(pl.Date) - pl.col("AccountCreationDate"))
    #         .dt.total_days()
    #         .alias("account_age_days")
    # ])
)

# ============================================================
# 7) Merge all features into Feature Matrix
# ============================================================

feature_matrix_lazy = (
    users_meta
    # .join(feature_total, on="user_id", how="left")
    .join(feature_action_counts, on="user_id", how="left")
    # .join(recency, on="user_id", how="left")
    # .join(gap_stats, on="user_id", how="left")
    # .join(window_30, on="user_id", how="left")
    # .join(window_90, on="user_id", how="left")
    # .join(window_180, on="user_id", how="left")
)

# ============================================================
# 8) Collect
# ============================================================

feature_matrix = feature_matrix_lazy.collect(engine="streaming")
feature_matrix.write_parquet("../../data/processed/feature_matrix_v2.parquet")

print("Feature Matrix ready:", feature_matrix.shape)
