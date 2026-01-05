import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl
import pandas as pd
import numpy as np
import sys

# -----------------------------------------
# 1) Load core event log
# -----------------------------------------
elog = pl.read_parquet("../../data/processed/core_eventlog_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

# -----------------------------------------
# 2) Choose which method to use
# -----------------------------------------
METHOD_CHOICES = {
    "time_avg": "../../results/best_params/best_params_time_average.csv",
    "stability": "../../results/best_params/best_params_stability.csv", 
    "train_val": "../../results/best_params/best_params_validation.csv"
}

method = "train_val"  # Change to "stability" or "train_val" if needed

if len(sys.argv) > 1:
    method = sys.argv[1]
    if method not in METHOD_CHOICES:
        print(f"ERROR: Invalid method '{method}'")
        print(f"Choose from: {list(METHOD_CHOICES.keys())}")
        sys.exit(1)

params_file = METHOD_CHOICES[method]

try:
    best_params = pd.read_csv(params_file)
    print(f"Loaded params from: {params_file}")
    print(f"Method: {method}")
    print(f"Personas: {len(best_params)}")
except FileNotFoundError:
    print(f"ERROR: File not found: {params_file}")
    print("Run ap_tp_optimize.py first!")
    sys.exit(1)

# -----------------------------------------
# 3) Build Ap and Tp maps
# -----------------------------------------
if method == "time_avg":
    # For time_avg, we might have multiple rows per persona (top_n)
    # Take the first one (highest F1)
    best_params = best_params.sort_values(["persona", "avg_F1"], ascending=[True, False])
    best_params = best_params.groupby("persona").head(1)
    
    Ap_map = dict(zip(best_params["persona"], best_params["Ap"]))
    Tp_map = dict(zip(best_params["persona"], best_params["Tp_sec"]))
    
    print("\nUsing time-averaged params (avg across all periods):")
    print(best_params[["persona", "Ap", "Tp_sec", "avg_F1", "avg_recall", "avg_precision"]])
    
elif method == "stability":
    Ap_map = dict(zip(best_params["persona"], best_params["Ap"]))
    Tp_map = dict(zip(best_params["persona"], best_params["Tp_sec"]))
    
    print("\nUsing stability-based params (low variance):")
    print(best_params[["persona", "Ap", "Tp_sec", "avg_F1", "std_F1", "avg_recall"]])
    
elif method == "train_val":
    Ap_map = dict(zip(best_params["persona"], best_params["Ap"]))
    Tp_map = dict(zip(best_params["persona"], best_params["Tp_sec"]))
    
    print("\nUsing train/val split params:")
    print(best_params[["persona", "Ap", "Tp_sec", "train_F1", "val_F1", "val_recall"]])

# -----------------------------------------
# 4) Evaluation cutoff & grace period
# -----------------------------------------
cutoff = pl.datetime(2016, 1, 1)  # Use recent date for testing
grace_days = 60
grace_sec = grace_days * 86400

print(f"\nEvaluation cutoff: {cutoff}")
print(f"Grace period: {grace_days} days")

rows = []

# -----------------------------------------
# 5) Loop over personas with available params
# -----------------------------------------
for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]
    
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))
    
    win = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(
            (pl.col("timestamp") >= window_start) &
            (pl.col("timestamp") <= cutoff)
        )
    )
    
    active_users = win.select("user_id").unique()
    
    if active_users.height == 0:
        print(f"  Persona {persona}: No active users in window")
        continue
    
    wa = (
        win
        .group_by("user_id")
        .agg(pl.col("gain").sum().alias("WA"))
    )
    
    wa = wa.join(active_users, on="user_id", how="inner")
    
    wa = wa.with_columns([
        (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
    ])
    
    after = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec))
        .group_by("user_id")
        .agg(pl.len().alias("cnt_after"))
    )
    
    eval_df = (
        wa.join(after, on="user_id", how="left")
          .with_columns([
              (pl.col("cnt_after").is_null()).cast(pl.Int8).alias("actual_churn")
          ])
    )
    
    if eval_df.height == 0:
        continue
    
    pdf = eval_df.select(["user_id", "WA", "pred_churn", "actual_churn"]).to_pandas()
    
    y_true = pdf["actual_churn"].values
    y_pred = pdf["pred_churn"].values
    
    accuracy = float((y_true == y_pred).mean())
    actual_rate = float(y_true.mean())
    pred_rate = float(y_pred.mean())
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    FP = np.sum((y_true == 0) & (y_pred == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    rows.append({
        "persona": int(persona),
        "Tp_sec": float(Tp_sec),
        "Ap": float(Ap),
        "num_users": int(len(y_true)),
        "actual_churn_rate": actual_rate,
        "pred_churn_rate": pred_rate,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": F1,
    })
    
    print(f"  Persona {persona}: users={len(y_true)}, acc={accuracy:.3f}, F1={F1:.3f}")

# -----------------------------------------
# 6) Save results
# -----------------------------------------
if len(rows) == 0:
    print("\nERROR: No personas evaluated. Check your cutoff date and data range.")
    sys.exit(1)

df_out = pd.DataFrame(rows)
output_file = f"eval_{method}.csv"
df_out.to_csv(output_file, index=False)
print(f"\nSaved → {output_file}")

# Overall metrics
total_correct = sum(r["accuracy"] * r["num_users"] for r in rows)
total_users = sum(r["num_users"] for r in rows)
overall_acc = total_correct / total_users

weighted_precision = sum(r["precision"] * r["num_users"] for r in rows) / total_users
weighted_recall = sum(r["recall"] * r["num_users"] for r in rows) / total_users
weighted_F1 = sum(r["F1"] * r["num_users"] for r in rows) / total_users

print("\n" + "=" * 60)
print("OVERALL METRICS (weighted by users):")
print("=" * 60)
print(f"Total users evaluated: {total_users}")
print(f"Accuracy:  {overall_acc:.4f}")
print(f"Precision: {weighted_precision:.4f}")
print(f"Recall:    {weighted_recall:.4f}")
print(f"F1 Score:  {weighted_F1:.4f}")
print("=" * 60)

# Per-persona summary
print("\nPer-persona results:")
print(df_out.sort_values("F1", ascending=False).to_string(index=False))