import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("SLIDING WINDOW EVALUATION - MEDIAN F1 SELECTION")
print("=" * 80)

# Load core event log
elog = pl.read_parquet("core_eventlog_eval_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"Event log loaded: {elog.height:,} events")
print(f"Date range: {elog['timestamp'].min()} to {elog['timestamp'].max()}")

# Load all tested (Ap, Tp) combinations from grid search
grid = pd.read_csv("tp_ap_xgb_enhanced_all_results.csv")
print(f"\nGrid search results loaded: {len(grid)} rows")

# Get unique parameter combinations per persona
param_combos = (
    grid[["persona", "Tp_sec", "Ap"]]
    .drop_duplicates()
    .sort_values(["persona", "Tp_sec", "Ap"])
)

print(f"Unique (persona, Tp, Ap) combinations: {len(param_combos)}")
print(f"Personas: {sorted(param_combos['persona'].unique())}")

# Generate 6-month sliding windows from 2015 to 2023
def generate_windows_multigrace(start_year=2014, end_year=2023, graces_coef=[1,2,3]):
    windows = []
    for year in range(start_year, end_year + 1):
        for coef in graces_coef:
            
            cutoff = (datetime(year, 1, 1), coef)
            windows.append(cutoff)

    
    return windows

windows = generate_windows_multigrace(2014, 2023, [1,2,3])
print(f"\nEvaluation windows: {len(windows)}")
for w in windows:
    print(f"  {w[0].strftime('%Y-%m-%d')}")

grace_days = 90
grace_sec = grace_days * 86400

# Store all evaluation results
all_results = []

# Main evaluation loop
for idx, row in param_combos.iterrows():
    persona = int(row["persona"])
    Tp_sec = float(row["Tp_sec"])
    Ap = float(row["Ap"])
    
    print(f"\n[{idx+1}/{len(param_combos)}] Evaluating persona={persona}, Tp={Tp_sec:.0f}s, Ap={Ap:.4f}")
    
    window_f1_scores = []
    
    for cutoff_dt in windows:
        cutoff = cutoff_dt[0]
        grace_coef = cutoff_dt[1]
        window_start = cutoff - pl.duration(seconds=int(Tp_sec))
        
        # Activity window
        win = (
            elog
            .filter(pl.col("persona") == persona)
            .filter(
                (pl.col("timestamp") >= window_start) &
                (pl.col("timestamp") <= cutoff)
            )
        )
        
        active_users = win.select("user_id").unique()
        
        if active_users.height < 10:
            continue
        
        # Weighted activity
        wa_raw = (
            win
            .group_by("user_id")
            .agg(pl.col("gain").sum().alias("WA"))
        )
        
        wa = wa_raw.with_columns([
            (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
        ])
        
        # Future activity
        after = (
            elog
            .filter(pl.col("persona") == persona)
            .filter((pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec*grace_coef)) & (pl.col("timestamp") < cutoff +  pl.duration(seconds=365 * 86400) + pl.duration(seconds=grace_sec*grace_coef)))
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
        
        pdf = eval_df.select(["pred_churn", "actual_churn"]).to_pandas()
        y_true = pdf["actual_churn"].values
        y_pred = pdf["pred_churn"].values
        
        Ap_Calibrated = Ap * (y_true.mean() / y_pred.mean()) if y_pred.mean() > 0 else Ap

        wa = wa_raw.with_columns([
            (pl.col("WA") < Ap_Calibrated).cast(pl.Int8).alias("pred_churn")
        ])
        y_pred = pdf["pred_churn"].values

        # Need both classes
        if len(np.unique(y_true)) < 2:
            continue
        
        # Calculate metrics
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        window_f1_scores.append({
            "cutoff": cutoff.strftime('%Y-%m-%d'),
            "grace_days": grace_days * grace_coef,
            "num_users": len(y_true),
            "f1": f1,
            "precision": precision,
            "recall": recall
        })
    
    # Calculate statistics across windows
    if len(window_f1_scores) >= 3:
        f1_values = [w["f1"] for w in window_f1_scores]
        
        all_results.append({
            "persona": persona,
            "Tp_sec": Tp_sec,
            "Ap": Ap,
            "n_windows": len(window_f1_scores),
            "median_f1": np.median(f1_values),
            "mean_f1": np.mean(f1_values),
            "std_f1": np.std(f1_values),
            "min_f1": np.min(f1_values),
            "max_f1": np.max(f1_values),
            "median_precision": np.median([w["precision"] for w in window_f1_scores]),
            "median_recall": np.median([w["recall"] for w in window_f1_scores]),
            "total_users": sum([w["num_users"] for w in window_f1_scores])
        })
        
        print(f"  Evaluated on {len(window_f1_scores)} windows: median_F1={np.median(f1_values):.3f}")
    else:
        print(f"  Skipped: only {len(window_f1_scores)} valid windows")

# Save all results
if len(all_results) == 0:
    print("\nERROR: No valid evaluations completed!")
    exit(1)

df_all = pd.DataFrame(all_results)
df_all.to_csv("sliding_window_all_results.csv", index=False)
print(f"\nSaved all results → sliding_window_all_results.csv ({len(df_all)} rows)")

# Select best params per persona based on median F1
best_params = (
    df_all[df_all["n_windows"] >= 5]
    .sort_values(["persona", "median_f1"], ascending=[True, False])
    .groupby("persona")
    .head(1)
)

if len(best_params) == 0:
    print("\nWARNING: No params with >= 5 valid windows. Relaxing constraint...")
    best_params = (
        df_all[df_all["n_windows"] >= 3]
        .sort_values(["persona", "median_f1"], ascending=[True, False])
        .groupby("persona")
        .head(1)
    )

best_params.to_csv("best_params_sliding_median_f1.csv", index=False)
print(f"Saved best params → best_params_sliding_median_f1.csv ({len(best_params)} personas)")

# Display results
print("\n" + "=" * 80)
print("BEST PARAMETERS PER PERSONA (by median F1)")
print("=" * 80)
print(best_params[[
    "persona", "Tp_sec", "Ap", "n_windows", 
    "median_f1", "median_recall", "median_precision", "std_f1"
]].to_string(index=False))

# Overall statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total personas with best params: {len(best_params)}")
print(f"Average median F1: {best_params['median_f1'].mean():.3f}")
print(f"Average median recall: {best_params['median_recall'].mean():.3f}")
print(f"Average median precision: {best_params['median_precision'].mean():.3f}")
print(f"Average windows evaluated: {best_params['n_windows'].mean():.1f}")

# Identify stable vs unstable params
print("\n" + "=" * 80)
print("STABILITY ANALYSIS")
print("=" * 80)
stable = best_params[best_params["std_f1"] < 0.10]
unstable = best_params[best_params["std_f1"] >= 0.20]

print(f"Stable params (std < 0.10): {len(stable)} personas")
print(f"Unstable params (std >= 0.20): {len(unstable)} personas")

if len(unstable) > 0:
    print("\nUnstable personas (high variance):")
    print(unstable[["persona", "median_f1", "std_f1", "min_f1", "max_f1"]].to_string(index=False))

print("\n" + "=" * 80)
print("DONE! Use best_params_sliding_median_f1.csv for production.")
print("=" * 80)