import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Temporal Drift Analysis & Solution for Churn Prediction

This script addresses the temporal concept drift problem where user behavior
patterns changed significantly over time due to:
- COVID-19 pandemic (2020-2021)
- LLM revolution (ChatGPT/Copilot, 2022+)
- Platform evolution
- Changing user demographics

Strategy: Test multiple approaches and recommend the best one.
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================================
# Configuration
# =============================================================================

# Test years for temporal cross-validation
CUTOFF_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400

# Era definitions (based on Stack Overflow usage patterns)
ERAS = {
    'pre_covid': (2016, 2019),      # Normal times
    'covid': (2020, 2021),           # Remote work surge
    'post_llm': (2022, 2024),        # ChatGPT/Copilot era
}

# Which WA method to use (based on your comparison)
WA_METHOD = 'WA_recency'  # Winner from comparison: F1=0.5804

# =============================================================================
# Load Data
# =============================================================================

print("=" * 80)
print("TEMPORAL DRIFT ANALYSIS & SOLUTION")
print("=" * 80)
print(f"WA Method: {WA_METHOD}")
print(f"Grace Period: {GRACE_DAYS} days")
print(f"Cutoff Years: {CUTOFF_YEARS}")

elog = pl.read_parquet("../../data/processed/core_eventlog_train_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"Event log loaded: {elog.height:,} events")
print(f"Date range: {elog['timestamp'].min()} to {elog['timestamp'].max()}")

# Load best params (you can change this to any of the 4 WA methods)
best_params_file = f"tp_ap_xgb_enhanced_best_params_{WA_METHOD}.csv"
try:
    best_params = pd.read_csv(best_params_file)
    print(f"\nLoaded best params from: {best_params_file}")
except FileNotFoundError:
    print(f"\nWARNING: {best_params_file} not found, falling back to WA_trend")
    best_params = pd.read_csv("../../results/best_params/tp_ap_xgb_enhanced_best_params_WA_trend.csv")
    WA_METHOD = 'WA_trend'

Ap_map = dict(zip(best_params["persona"], best_params["Ap"]))
Tp_map = dict(zip(best_params["persona"], best_params["Tp_sec"]))

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_wa(win, Tp_sec, cutoff, wa_type='WA_recency'):
    """
    Calculate Weighted Activity based on method type
    """
    mid_point = cutoff - pl.duration(seconds=int(Tp_sec / 2))

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

    if wa_type == 'WA_simple':
        wa_expr = pl.col("total_gain")
    elif wa_type == 'WA_trend':
        wa_expr = pl.col("total_gain") * pl.col("activity_trend").clip(0.5, 2.0)
    elif wa_type == 'WA_recency':
        wa_expr = pl.col("total_gain") * pl.col("recency_weight")
    elif wa_type == 'WA_enhanced':
        wa_expr = (
            pl.col("total_gain")
            * pl.col("activity_trend").clip(0.5, 2.0)
            * pl.col("recency_weight")
        )
    else:
        raise ValueError(f"Unknown WA type: {wa_type}")

    wa = features.with_columns([wa_expr.alias("WA")]).select(["user_id", "WA"])

    return wa


def evaluate_persona_year(elog, persona, Ap, Tp_sec, cutoff_year, wa_type):
    """
    Evaluate a single persona for a single cutoff year
    """
    cutoff = pl.datetime(cutoff_year, 1, 1)
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
        return None

    # Calculate WA
    wa = calculate_wa(win, Tp_sec, cutoff, wa_type)
    wa = wa.join(active_users, on="user_id", how="inner")

    # Predict churn: WA < Ap
    wa = wa.with_columns([
        (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
    ])

    # Actual churn after grace period
    after = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(
            (pl.col("timestamp") > cutoff + pl.duration(seconds=GRACE_SEC)) &
            (pl.col("timestamp") < cutoff + pl.duration(seconds=GRACE_SEC + 365 * 86400))
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
        return None

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

    return {
        "persona": int(persona),
        "cutoff_year": cutoff_year,
        "Tp_days": int(Tp_sec / 86400),
        "Ap": float(Ap),
        "num_users": len(y_true),
        "actual_churn_rate": float(y_true.mean()),
        "pred_churn_rate": float(y_pred.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# APPROACH 1: Baseline (No Temporal Adjustment)
# =============================================================================

print("\n" + "=" * 80)
print("APPROACH 1: BASELINE (No Temporal Adjustment)")
print("=" * 80)
print("Using original Ap/Tp values across all years...")

baseline_results = []

for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

    for year in CUTOFF_YEARS:
        result = evaluate_persona_year(elog, persona, Ap, Tp_sec, year, WA_METHOD)
        if result:
            baseline_results.append(result)

df_baseline = pd.DataFrame(baseline_results)

print(f"\nResults per year:")
year_summary = df_baseline.groupby('cutoff_year').agg({
    'f1': ['mean', 'median', 'std'],
    'precision': 'mean',
    'recall': 'mean',
    'persona': 'count'
}).round(4)
print(year_summary)

print(f"\nOverall:")
print(f"  Mean F1: {df_baseline['f1'].mean():.4f}")
print(f"  Median F1: {df_baseline['f1'].median():.4f}")
print(f"  F1 Std: {df_baseline['f1'].std():.4f}")

# =============================================================================
# APPROACH 2: Year-Specific Threshold Adjustment
# =============================================================================

print("\n" + "=" * 80)
print("APPROACH 2: YEAR-SPECIFIC THRESHOLD ADJUSTMENT")
print("=" * 80)
print("Adjusting Ap threshold based on year to maintain consistent performance...")

# Calculate year-specific multipliers based on baseline performance
year_multipliers = {}
baseline_year_f1 = df_baseline.groupby('cutoff_year')['f1'].median()
reference_f1 = baseline_year_f1.max()  # Best year as reference

for year, year_f1 in baseline_year_f1.items():
    if year_f1 > 0:
        # If F1 is lower, we need to adjust threshold (lower Ap = more lenient)
        multiplier = np.sqrt(year_f1 / reference_f1)  # Square root for smoother adjustment
        year_multipliers[year] = multiplier
    else:
        year_multipliers[year] = 1.0

print(f"\nYear multipliers (based on F1 performance):")
for year, mult in sorted(year_multipliers.items()):
    print(f"  {year}: {mult:.4f}")

adjusted_results = []

for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

    for year in CUTOFF_YEARS:
        adjusted_Ap = Ap * year_multipliers.get(year, 1.0)
        result = evaluate_persona_year(elog, persona, adjusted_Ap, Tp_sec, year, WA_METHOD)
        if result:
            result['Ap_original'] = Ap
            result['year_multiplier'] = year_multipliers.get(year, 1.0)
            adjusted_results.append(result)

df_adjusted = pd.DataFrame(adjusted_results)

print(f"\nResults per year:")
year_summary_adj = df_adjusted.groupby('cutoff_year').agg({
    'f1': ['mean', 'median', 'std'],
    'precision': 'mean',
    'recall': 'mean',
    'persona': 'count'
}).round(4)
print(year_summary_adj)

print(f"\nOverall:")
print(f"  Mean F1: {df_adjusted['f1'].mean():.4f}")
print(f"  Median F1: {df_adjusted['f1'].median():.4f}")
print(f"  F1 Std: {df_adjusted['f1'].std():.4f}")

# =============================================================================
# APPROACH 3: Era-Based Models
# =============================================================================

print("\n" + "=" * 80)
print("APPROACH 3: ERA-BASED MODELS")
print("=" * 80)
print("Training separate thresholds for different eras...")
print(f"Eras: {ERAS}")

# Group years into eras
def get_era(year):
    for era_name, (start, end) in ERAS.items():
        if start <= year <= end:
            return era_name
    return 'unknown'

df_baseline['era'] = df_baseline['cutoff_year'].apply(get_era)

# Calculate era-specific multipliers
era_multipliers = {}
era_f1 = df_baseline.groupby('era')['f1'].median()
reference_era_f1 = era_f1.max()

for era, era_f1_val in era_f1.items():
    if era_f1_val > 0:
        multiplier = np.sqrt(era_f1_val / reference_era_f1)
        era_multipliers[era] = multiplier
    else:
        era_multipliers[era] = 1.0

print(f"\nEra multipliers (based on F1 performance):")
for era, mult in sorted(era_multipliers.items()):
    print(f"  {era}: {mult:.4f}")

era_results = []

for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

    for year in CUTOFF_YEARS:
        era = get_era(year)
        adjusted_Ap = Ap * era_multipliers.get(era, 1.0)
        result = evaluate_persona_year(elog, persona, adjusted_Ap, Tp_sec, year, WA_METHOD)
        if result:
            result['Ap_original'] = Ap
            result['era'] = era
            result['era_multiplier'] = era_multipliers.get(era, 1.0)
            era_results.append(result)

df_era = pd.DataFrame(era_results)

print(f"\nResults per year:")
year_summary_era = df_era.groupby('cutoff_year').agg({
    'f1': ['mean', 'median', 'std'],
    'precision': 'mean',
    'recall': 'mean',
    'persona': 'count'
}).round(4)
print(year_summary_era)

print(f"\nResults per era:")
era_summary = df_era.groupby('era').agg({
    'f1': ['mean', 'median', 'std'],
    'precision': 'mean',
    'recall': 'mean',
    'persona': 'count'
}).round(4)
print(era_summary)

print(f"\nOverall:")
print(f"  Mean F1: {df_era['f1'].mean():.4f}")
print(f"  Median F1: {df_era['f1'].median():.4f}")
print(f"  F1 Std: {df_era['f1'].std():.4f}")

# =============================================================================
# Comparison Summary
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

comparison = pd.DataFrame([
    {
        'Approach': 'Baseline (No Adjustment)',
        'Mean F1': df_baseline['f1'].mean(),
        'Median F1': df_baseline['f1'].median(),
        'F1 Std': df_baseline['f1'].std(),
        'F1 Range': df_baseline['f1'].max() - df_baseline['f1'].min(),
    },
    {
        'Approach': 'Year-Specific Adjustment',
        'Mean F1': df_adjusted['f1'].mean(),
        'Median F1': df_adjusted['f1'].median(),
        'F1 Std': df_adjusted['f1'].std(),
        'F1 Range': df_adjusted['f1'].max() - df_adjusted['f1'].min(),
    },
    {
        'Approach': 'Era-Based Models',
        'Mean F1': df_era['f1'].mean(),
        'Median F1': df_era['f1'].median(),
        'F1 Std': df_era['f1'].std(),
        'F1 Range': df_era['f1'].max() - df_era['f1'].min(),
    },
]).round(4)

print(comparison.to_string(index=False))

# Determine best approach
best_idx = comparison['Median F1'].idxmax()
best_approach = comparison.loc[best_idx, 'Approach']

print(f"\n{'='*80}")
print(f"RECOMMENDATION: {best_approach}")
print(f"{'='*80}")
print(f"Median F1: {comparison.loc[best_idx, 'Median F1']:.4f}")
print(f"F1 Stability (lower std is better): {comparison.loc[best_idx, 'F1 Std']:.4f}")

# =============================================================================
# Save Results
# =============================================================================

df_baseline.to_csv("../../results/temporal_baseline_results.csv", index=False)
df_adjusted.to_csv("../../results/temporal_year_adjusted_results.csv", index=False)
df_era.to_csv("../../results/temporal_era_based_results.csv", index=False)
comparison.to_csv("../../results/temporal_comparison_summary.csv", index=False)

print(f"\n{'='*80}")
print("OUTPUT FILES")
print(f"{'='*80}")
print("1. temporal_baseline_results.csv - No adjustment")
print("2. temporal_year_adjusted_results.csv - Year-specific thresholds")
print("3. temporal_era_based_results.csv - Era-based models")
print("4. temporal_comparison_summary.csv - Summary comparison")

# Save multipliers for production use
if best_approach == 'Year-Specific Adjustment':
    multipliers_df = pd.DataFrame(list(year_multipliers.items()),
                                  columns=['year', 'multiplier'])
    multipliers_df.to_csv("../../results/temporal_year_multipliers.csv", index=False)
    print("5. temporal_year_multipliers.csv - Year multipliers for production")
elif best_approach == 'Era-Based Models':
    multipliers_df = pd.DataFrame(list(era_multipliers.items()),
                                  columns=['era', 'multiplier'])
    multipliers_df.to_csv("temporal_era_multipliers.csv", index=False)
    print("5. temporal_era_multipliers.csv - Era multipliers for production")

print("\nDone!")
