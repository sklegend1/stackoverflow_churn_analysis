import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# -------------------------------------------------------
# Load core event log (only core activities)
# -------------------------------------------------------
elog = pl.read_parquet("../../data/processed/core_eventlog_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime)
])

# -------------------------------------------------------
# Load best Ap / Tp per persona
# -------------------------------------------------------
grid = pd.read_csv("../../results/best_params/tp_ap_xgb_joint_search.csv")

# For each persona, keep best (highest AUC)
best_params = (
    grid.sort_values(["persona", "auc"], ascending=[True, False])
        .groupby("persona")
        .head(1)
)

# Convert to dictionary for fast access
Ap_map = dict(zip(best_params["persona"], best_params["Ap"]))
Tp_map = dict(zip(best_params["persona"], best_params["Tp_sec"]))

# -------------------------------------------------------
# Choose cutoff for evaluation
# -------------------------------------------------------
cutoff = pl.datetime(2022, 1, 1)
grace_period_sec = 180 * 86400  # 90 days

# -------------------------------------------------------
# 1) Identify actual churn based on activity after cutoff
#    (global، بعداً در هر پرسونا subset می‌گیریم)
# -------------------------------------------------------
last_after_cutoff = (
    elog.filter(pl.col("timestamp") > (cutoff + pl.duration(seconds=grace_period_sec)))
        .group_by(["persona", "user_id"])
        .agg(pl.len().alias("cnt"))
)

# If cnt is null → user never appeared after cutoff+grace → churn
actual = (
    elog.group_by(["persona", "user_id"])
        .agg(pl.col("timestamp").max().alias("last_ts"))
        .join(last_after_cutoff, on=["persona", "user_id"], how="left")
        .with_columns([
            (pl.col("cnt").is_null()).cast(pl.Int8).alias("actual_churn")
        ])
        .select(["persona", "user_id", "actual_churn"])
)

# -------------------------------------------------------
# 2) Compute predicted churn using Ap, Tp
#    فقط روی یوزرهایی که در پنجره (cutoff - Tp, cutoff] اکتیو بوده‌اند
# -------------------------------------------------------
rows = []

for persona in Ap_map.keys():

    Ap = Ap_map[persona]
    Tp = Tp_map[persona]

    # Compute window start: cutoff - Tp seconds
    window_start = cutoff - pl.duration(seconds=int(Tp))

    # Events of this persona in the window
    df_p = (
        elog.filter(pl.col("persona") == persona)
            .filter(
                (pl.col("timestamp") >= window_start) &
                (pl.col("timestamp") <= cutoff)
            )
    )

    # --- NEW: restrict to active users in this window ---
    active_users = df_p.select("user_id").unique()["user_id"].to_list()
    if len(active_users) == 0:
        # nobody active in this window for this persona → skip
        continue

    # Re-filter df_p to make intention explicit (optional)
    df_p = df_p.filter(pl.col("user_id").is_in(active_users))

    # Weighted-activity per user (only active users)
    wa = (
        df_p.group_by("user_id")
            .agg(pl.col("gain").sum().alias("WA"))
    )

    # Actual churn labels for this persona, restricted to active users
    actual_p = (
        actual
        .filter(pl.col("persona") == persona)
        .filter(pl.col("user_id").is_in(active_users))
    )

    # Join labels & WA
    merged = (
        actual_p
            .join(wa, on="user_id", how="left")
            .with_columns([
                pl.col("WA").fill_null(0)
            ])
            .with_columns([
                (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
            ])
    )

    merged_pd = merged.to_pandas()

    if len(merged_pd) == 0:
        continue

    # Evaluation metrics
    y_true = merged_pd["actual_churn"]
    y_pred = merged_pd["pred_churn"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # If at least 2 classes exist, compute AUC
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = np.nan

    rows.append({
        "persona": persona,
        "Tp_sec": Tp,
        "Ap": Ap,
        "num_users": len(y_true),
        "actual_churn_rate": float(y_true.mean()),
        "pred_churn_rate": float(y_pred.mean()),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc
    })

# -------------------------------------------------------
# Save results
# -------------------------------------------------------
df_out = pd.DataFrame(rows)
df_out.to_csv("../../results/evaluations/ap_tp_evaluation_results_active_window_only.csv", index=False)

print("Saved → ap_tp_evaluation_results_active_window_only.csv")
