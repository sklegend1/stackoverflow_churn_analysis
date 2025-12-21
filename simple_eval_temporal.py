"""
Temporal-Aware Churn Evaluation Script

This script addresses temporal concept drift by adjusting Ap thresholds
based on the year to maintain consistent performance across time periods.

Based on temporal_drift_analysis.py results, this uses year-specific multipliers.
"""

import polars as pl
import pandas as pd
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

# Which WA method to use (from comparison: WA_recency was best)
WA_METHOD = 'WA_recency'

# Evaluation parameters
CUTOFF_YEAR = 2016  # Change this to evaluate different years
GRACE_DAYS = 180

# =============================================================================
# Load Data
# =============================================================================

print("=" * 80)
print("TEMPORAL-AWARE CHURN EVALUATION")
print("=" * 80)
print(f"WA Method: {WA_METHOD}")
print(f"Cutoff Year: {CUTOFF_YEAR}")
print(f"Grace Period: {GRACE_DAYS} days")

elog = pl.read_parquet("core_eventlog_eval_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"\nEvent log loaded: {elog.height:,} events")

# Load best params for chosen WA method
# best_params_file = f"tp_ap_xgb_enhanced_best_params_{WA_METHOD}.csv"
best_params_file = "lstm_derived_ap_tp_2016.csv"
try:
    best_params = pd.read_csv(best_params_file)
    print(f"Loaded best params from: {best_params_file}")
except FileNotFoundError:
    print(f"WARNING: {best_params_file} not found, falling back to WA_trend")
    best_params = pd.read_csv("tp_ap_xgb_enhanced_best_params_WA_trend.csv")
    WA_METHOD = 'WA_trend'

Ap_map = dict(zip(best_params["persona"], best_params["Ap"]))
Tp_map = dict(zip(best_params["persona"], best_params["Tp_sec"]))

# Load temporal multipliers
try:
    year_multipliers_df = pd.read_csv("temporal_year_multipliers.csv")
    year_multipliers = dict(zip(year_multipliers_df["year"], year_multipliers_df["multiplier"]))
    print(f"Loaded temporal multipliers from: temporal_year_multipliers.csv")

    # Get multiplier for current year
    year_multiplier = year_multipliers.get(CUTOFF_YEAR, 1.0)
    print(f"Year multiplier for {CUTOFF_YEAR}: {year_multiplier:.4f}")

    # Adjust Ap values based on year
    Ap_map = {persona: Ap * year_multiplier for persona, Ap in Ap_map.items()}
    print(f"✓ Ap thresholds adjusted for temporal drift")

except FileNotFoundError:
    print("WARNING: temporal_year_multipliers.csv not found")
    print("Run temporal_drift_analysis.py first to generate multipliers")
    print("Proceeding without temporal adjustment...")
    year_multiplier = 1.0

# =============================================================================
# Evaluation Logic
# =============================================================================

cutoff = pl.datetime(CUTOFF_YEAR, 1, 1)
grace_sec = GRACE_DAYS * 86400

rows = []

print(f"\n{'='*80}")
print("EVALUATING PERSONAS")
print(f"{'='*80}")

for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

    # Window start: [cutoff - Tp, cutoff]
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))

    # Only events in the window for this persona
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
        continue

    # Calculate WA based on method
    mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

    features = (
        win
        .with_columns([
            (pl.col("timestamp") >= mid_point).alias("is_recent"),
            ((cutoff - pl.col("timestamp")).dt.total_seconds() / 86400).alias("days_ago"),
        ])
        .group_by("user_id")
        .agg([
            pl.col("gain").sum().alias("total_gain"),
            pl.col("gain").filter(pl.col("is_recent")).sum().alias("recent_gain"),
            pl.col("gain").filter(~pl.col("is_recent")).sum().alias("old_gain"),
            pl.col("days_ago").min().alias("days_since_last"),
        ])
    )

    features = features.with_columns([
        ((pl.col("recent_gain") + 1.0) / (pl.col("old_gain") + 1.0)).alias("activity_trend"),
        ((-pl.col("days_since_last") / 30.0).exp()).alias("recency_weight"),
    ])

    # Select WA calculation based on method
    if WA_METHOD == 'WA_simple':
        wa_expr = pl.col("total_gain")
    elif WA_METHOD == 'WA_trend':
        wa_expr = pl.col("total_gain") * pl.col("activity_trend").clip(0.5, 2.0)
    elif WA_METHOD == 'WA_recency':
        wa_expr = pl.col("total_gain") * pl.col("recency_weight")
    elif WA_METHOD == 'WA_enhanced':
        wa_expr = (
            pl.col("total_gain")
            * pl.col("activity_trend").clip(0.5, 2.0)
            * pl.col("recency_weight")
        )
    else:
        wa_expr = pl.col("total_gain")

    wa = features.with_columns([wa_expr.alias("WA")]).select(["user_id", "WA"])
    wa = wa.join(active_users, on="user_id", how="inner")

    # Predict: WA < Ap → churn = 1
    wa = wa.with_columns([
        (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
    ])

    # Actual churn after grace period
    after = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(
            (pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec)) &
            (pl.col("timestamp") < cutoff + pl.duration(seconds=grace_sec + 365 * 86400))
        )
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

    # Metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    actual_rate = float(y_true.mean())
    pred_rate = float(y_pred.mean())

    rows.append({
        "persona": int(persona),
        "Tp_sec": float(Tp_sec),
        "Tp_days": int(float(Tp_sec) / 86400.0),
        "Ap": float(Ap),
        "Ap_base": float(Ap / year_multiplier),  # Original Ap before adjustment
        "year_multiplier": float(year_multiplier),
        "num_users": int(len(y_true)),
        "actual_churn_rate": actual_rate,
        "pred_churn_rate": pred_rate,
        "rate_ratio": (pred_rate / actual_rate) if actual_rate > 0 else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    print(f"Persona {persona:2d}: {len(y_true):5d} users, F1={f1:.4f}, P={precision:.3f}, R={recall:.3f}")

# =============================================================================
# Results Summary
# =============================================================================

df_out = pd.DataFrame(rows)
output_file = f"ap_tp_simple_eval_temporal_{CUTOFF_YEAR}_{WA_METHOD}.csv"
df_out.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if len(rows) > 0:
    print(f"Evaluated {len(rows)} personas")
    print(f"Total users: {df_out['num_users'].sum():,}")
    print(f"\nPerformance Metrics:")
    print(f"  Median F1: {df_out['f1'].median():.4f}")
    print(f"  Mean F1: {df_out['f1'].mean():.4f}")
    print(f"  Median Precision: {df_out['precision'].median():.4f}")
    print(f"  Median Recall: {df_out['recall'].median():.4f}")
    print(f"\nChurn Rate Analysis:")
    print(f"  Avg Actual Churn Rate: {df_out['actual_churn_rate'].mean():.4f}")
    print(f"  Avg Predicted Churn Rate: {df_out['pred_churn_rate'].mean():.4f}")
    print(f"  Median Rate Ratio: {df_out['rate_ratio'].median():.4f}")

    print(f"\n{'='*80}")
    print(f"Saved → {output_file}")
    print(f"{'='*80}")
else:
    print("No personas had active users in the evaluation window.")
