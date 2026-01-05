import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl

# -------------------------------
# 1) Load core activities (from XGBoost output)
# -------------------------------
core_df = pl.read_csv("../../data/processed/core_activity_per_persona_named.csv")
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
elog = pl.scan_parquet("../../data/processed/eventlog_v2.parquet")

# -------------------------------
# 2.5) Deterministic 80/20 user split per persona
# -------------------------------
SEED = 42
EVAL_RATIO = 0.20

# Build one row per (persona, user_id)
users = (
    elog
    .select([pl.col("persona").cast(pl.Int32), pl.col("user_id").cast(pl.Int64)])
    .unique()
)

# Deterministic pseudo-random in [0,1) using a simple LCG-style formula
# This avoids needing a true RNG in Lazy mode and stays reproducible.
m = 2147483647  # 2^31-1 (prime)
a = 1103515245

users = users.with_columns([
    (
        (
            (
                (pl.col("user_id") + pl.col("persona").cast(pl.Int64) * 1000003 + pl.lit(SEED))
                * pl.lit(a)
            ) % pl.lit(m)
        ) / pl.lit(m)
    ).alias("rnd"),
])

users = users.with_columns([
    (pl.col("rnd") < pl.lit(EVAL_RATIO)).alias("is_eval")
]).drop("rnd")

# -------------------------------
# 3) Filter eventlog to core actions per persona
# -------------------------------
core_eventlog_lazy = (
    elog
    .join(core_top, on=["persona", "action_type_id"], how="inner")
    .join(users, on=["persona", "user_id"], how="inner")   # attach split flag
    .sort(["user_id", "timestamp"])
)

# -------------------------------
# 4) Helper: compute key gaps + cleanup
# -------------------------------
def finalize_core_log(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns([
        (
            pl.col("timestamp").diff().over("user_id") / 1_000_000
        ).alias("gap_time_key_actions_seconds"),
        (
            pl.col("timestamp").diff().over("user_id") / (1_000_000 * 60 * 60 * 24)
        ).alias("gap_time_key_actions_days"),
    ])

    lf = lf.drop(
        [c for c in lf.columns if c.startswith("gap_time_actions")] + ["id"]
    )

    lf = lf.with_row_index("id")
    return lf

# -------------------------------
# 5) Split into train/eval by user, then write two parquets
# -------------------------------
core_train_lazy = finalize_core_log(
    core_eventlog_lazy
    .filter(~pl.col("is_eval"))
    .drop("is_eval")
)

core_eval_lazy = finalize_core_log(
    core_eventlog_lazy
    .filter(pl.col("is_eval"))
    .drop("is_eval")
)

print("Collecting core_eventlog TRAIN with streaming engine...")
core_train = core_train_lazy.collect(engine="streaming")
print("Train rows:", core_train.height)
core_train.write_parquet("../../data/processed/core_eventlog_train_v1.parquet")

print("Collecting core_eventlog EVAL with streaming engine...")
core_eval = core_eval_lazy.collect(engine="streaming")
print("Eval rows:", core_eval.height)
core_eval.write_parquet("../../data/processed/core_eventlog_eval_v1.parquet")

print("Done.")
