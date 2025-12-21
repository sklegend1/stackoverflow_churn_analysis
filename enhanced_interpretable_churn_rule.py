import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

"""
ENHANCED INTERPRETABLE CHURN RULE: if WA(Tp) < Ap → Churn

Key Improvements:
1. Enhanced WA metric (not just sum of gains):
   - Activity trend (recent vs old activity)
   - Recency weighting (exponential decay)
   - Consistency metrics (regularity)

2. Systematic Tp grid search (60-270 days)

3. F-beta optimization (adjustable precision/recall trade-off)

4. Temporal cross-validation (multiple time windows)

5. Both simple and enhanced WA comparison
"""

print("=" * 80)
print("ENHANCED INTERPRETABLE CHURN RULE: WA(Tp) < Ap")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load event log
elog = pl.read_parquet("core_eventlog_train_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"\nEvent log loaded: {elog.height:,} events")
print(f"Date range: {elog['timestamp'].min()} to {elog['timestamp'].max()}")
print(f"Unique users: {elog['user_id'].n_unique():,}")
print(f"Personas: {sorted(elog['persona'].unique())}")

# Parameters
GRACE_DAYS = 180
GRACE_SEC = GRACE_DAYS * 86400
CHURN_CHECK_WINDOW_DAYS = 365
CHURN_CHECK_WINDOW_SEC = CHURN_CHECK_WINDOW_DAYS * 86400

# Tp range to test (in days)
TP_DAYS_RANGE = [60, 90, 120, 150, 180, 210, 240, 270]

# Beta for F-beta score (beta=1: F1, beta=2: favor recall, beta=0.5: favor precision)
BETA = 1.0

# Cutoff dates for temporal cross-validation
TRAIN_CUTOFFS = [
    datetime(2016, 1, 1),
    datetime(2017, 1, 1),
    datetime(2018, 1, 1),
    datetime(2019, 1, 1),
    datetime(2020, 1, 1),
    datetime(2021, 1, 1),
    datetime(2022, 1, 1),
    datetime(2023, 1, 1),
]

# How many Ap values to test (percentiles of WA distribution)
AP_PERCENTILES = np.arange(10, 91, 5)  # 10th, 15th, ..., 90th percentile

print(f"\nConfiguration:")
print(f"  Grace period: {GRACE_DAYS} days")
print(f"  Churn check window: {CHURN_CHECK_WINDOW_DAYS} days")
print(f"  Tp range: {TP_DAYS_RANGE}")
print(f"  F-beta (beta={BETA})")
print(f"  Training cutoffs: {len(TRAIN_CUTOFFS)} windows")
print(f"  Ap candidates: {len(AP_PERCENTILES)} percentiles")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_enhanced_wa(elog_window, cutoff, Tp_sec):
    """
    Calculate Enhanced Weighted Activity with multiple signals:

    1. WA_simple: sum(gain) - baseline

    2. WA_enhanced: combines:
       - Activity trend (recent vs old activity)
       - Recency weighting (exponential decay)
       - Consistency bonus (regular users)

    Returns: DataFrame with user_id and various WA metrics
    """
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))

    # Get events in window
    win = elog_window.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    )

    if win.height == 0:
        return None

    # Calculate temporal midpoint for trend analysis
    mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

    # Extract features per user
    features = (
        win
        .with_columns([
            (pl.col("timestamp") >= mid_point).alias("is_recent"),
            ((cutoff - pl.col("timestamp")).dt.total_seconds() / 86400).alias("days_ago"),
        ])
        .group_by("user_id")
        .agg([
            # Basic metrics
            pl.col("gain").sum().alias("total_gain"),
            pl.len().alias("event_count"),

            # Temporal split for trend detection
            pl.col("gain").filter(pl.col("is_recent")).sum().alias("recent_gain"),
            pl.col("gain").filter(~pl.col("is_recent")).sum().alias("old_gain"),
            pl.len().filter(pl.col("is_recent")).alias("recent_count"),
            pl.len().filter(~pl.col("is_recent")).alias("old_count"),

            # Recency
            pl.col("days_ago").min().alias("days_since_last_activity"),

            # Diversity
            pl.col("action_type_id").n_unique().alias("unique_actions"),

            # Consistency (inter-event time gaps)
            pl.col("timestamp").sort().diff().dt.total_seconds().drop_nulls().std().alias("gap_std"),
            pl.col("timestamp").sort().diff().dt.total_seconds().drop_nulls().mean().alias("gap_mean"),
        ])
    )

    # Calculate derived metrics
    features = features.with_columns([
        # Activity trend: ratio of recent to old activity
        # Values > 1 = growing, < 1 = declining
        ((pl.col("recent_gain") + 1.0) / (pl.col("old_gain") + 1.0)).alias("activity_trend"),

        # Recency weight: exponential decay
        # Recent activity → weight closer to 1, old activity → weight closer to 0
        ((-pl.col("days_since_last_activity") / 30.0).exp()).alias("recency_weight"),

        # Consistency: coefficient of variation (lower = more consistent)
        (pl.col("gap_std") / (pl.col("gap_mean") + 1.0)).alias("gap_cv"),
    ])

    # Create enhanced WA metrics
    features = features.with_columns([
        # WA_simple: traditional approach
        pl.col("total_gain").alias("WA_simple"),

        # WA_trend: incorporates activity trend
        # Boost for growing users, penalty for declining
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0)  # Limit to 0.5x - 2x multiplier
        ).alias("WA_trend"),

        # WA_recency: incorporates recency
        # Recent activity weighted higher
        (
            pl.col("total_gain") *
            pl.col("recency_weight")
        ).alias("WA_recency"),

        # WA_enhanced: combines trend + recency
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0) *
            pl.col("recency_weight")
        ).alias("WA_enhanced"),

        # WA_consistent: bonus for regular users
        (
            pl.col("total_gain") *
            (2.0 / (1.0 + pl.col("gap_cv")))  # Lower CV → higher multiplier
        ).alias("WA_consistent"),
    ])

    return features

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_ap_tp(elog_persona, persona, cutoff, Tp_sec):
    """
    For given persona, cutoff, and Tp:
    1. Calculate various WA metrics
    2. Get actual churn labels
    3. Test multiple Ap values (percentiles)
    4. Return best Ap for each WA type
    """
    # Calculate WA features
    features = calculate_enhanced_wa(elog_persona, cutoff, Tp_sec)

    if features is None or features.height < 20:
        return None

    # Get actual churn labels (no activity in grace+churn window)
    future_start = cutoff + pl.duration(seconds=GRACE_SEC)
    future_end = future_start + pl.duration(seconds=CHURN_CHECK_WINDOW_SEC)

    future_activity = (
        elog_persona
        .filter(
            (pl.col("timestamp") > future_start) &
            (pl.col("timestamp") <= future_end)
        )
        .group_by("user_id")
        .agg(pl.len().alias("future_events"))
    )

    # Join features with labels
    eval_df = (
        features
        .join(future_activity, on="user_id", how="left")
        .with_columns([
            (pl.col("future_events").is_null() | (pl.col("future_events") == 0))
            .cast(pl.Int8).alias("actual_churn")
        ])
    )

    pdf = eval_df.to_pandas()

    if len(pdf) < 20:
        return None

    y_true = pdf["actual_churn"].values

    # Need both classes
    if len(np.unique(y_true)) < 2:
        return None

    # Test different WA types
    wa_types = ["WA_simple", "WA_trend", "WA_recency", "WA_enhanced", "WA_consistent"]

    results = []

    for wa_type in wa_types:
        if wa_type not in pdf.columns:
            continue

        wa_scores = pdf[wa_type].values

        # Skip if all same value
        if wa_scores.std() < 1e-6:
            continue

        best_f_beta = 0
        best_ap = None
        best_metrics = None

        # Try different Ap values (percentiles of WA distribution)
        for percentile in AP_PERCENTILES:
            Ap = np.percentile(wa_scores, percentile)

            # Predict: churn if WA < Ap
            y_pred = (wa_scores < Ap).astype(int)

            # Calculate metrics
            TP = np.sum((y_true == 1) & (y_pred == 1))
            TN = np.sum((y_true == 0) & (y_pred == 0))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            FN = np.sum((y_true == 1) & (y_pred == 0))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            # F-beta score
            f_beta = (1 + BETA**2) * precision * recall / ((BETA**2 * precision) + recall) if (precision + recall) > 0 else 0

            # Track best
            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_ap = Ap

                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (TP + TN) / len(y_true)

                # Calculate AUC using WA as score (invert: low WA = high churn prob)
                try:
                    churn_prob = 1.0 / (1.0 + wa_scores + 1e-6)
                    auc = roc_auc_score(y_true, churn_prob)
                except:
                    auc = 0.5

                best_metrics = {
                    'wa_type': wa_type,
                    'Ap': best_ap,
                    'Ap_percentile': percentile,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'f_beta': f_beta,
                    'accuracy': accuracy,
                    'auc': auc,
                    'num_users': len(y_true),
                    'actual_churn_rate': float(y_true.mean()),
                    'pred_churn_rate': float(y_pred.mean()),
                    'TP': int(TP),
                    'TN': int(TN),
                    'FP': int(FP),
                    'FN': int(FN),
                }

        if best_metrics is not None:
            results.append(best_metrics)

    return results

# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

print("\n" + "=" * 80)
print("GRID SEARCH: Optimizing Ap for each (Persona, Tp, WA_type)")
print("=" * 80)

all_results = []
personas = sorted(elog["persona"].unique())

for persona in personas:
    print(f"\n{'='*80}")
    print(f"PERSONA {persona}")
    print(f"{'='*80}")

    elog_persona = elog.filter(pl.col("persona") == persona)

    for Tp_days in TP_DAYS_RANGE:
        Tp_sec = Tp_days * 86400
        print(f"\n  Tp = {Tp_days} days:")

        window_results = []

        for cutoff in TRAIN_CUTOFFS:
            results = evaluate_ap_tp(elog_persona, persona, cutoff, Tp_sec)

            if results is not None:
                for r in results:
                    window_results.append({
                        **r,
                        'persona': persona,
                        'Tp_days': Tp_days,
                        'Tp_sec': Tp_sec,
                        'cutoff': cutoff.strftime('%Y-%m-%d'),
                    })

        if len(window_results) > 0:
            all_results.extend(window_results)

            # Show progress: best result per WA type
            df_temp = pd.DataFrame(window_results)
            summary = df_temp.groupby('wa_type').agg({
                'f1': 'median',
                'precision': 'median',
                'recall': 'median',
                'f_beta': 'median',
            }).round(3)

            for wa_type, row in summary.iterrows():
                print(f"    {wa_type:15s}: F1={row['f1']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f}")

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================

df_all = pd.DataFrame(all_results)
df_all.to_csv("enhanced_wa_all_results.csv", index=False)
print(f"\n{'='*80}")
print(f"Saved all results → enhanced_wa_all_results.csv ({len(df_all)} rows)")
print(f"{'='*80}")

# ============================================================================
# SELECT BEST PARAMETERS
# ============================================================================

print("\n" + "=" * 80)
print("SELECTING BEST PARAMETERS PER PERSONA")
print("=" * 80)

best_params_list = []

for persona in personas:
    df_persona = df_all[df_all['persona'] == persona]

    if len(df_persona) == 0:
        print(f"Persona {persona}: No valid results")
        continue

    # For each (Tp, wa_type), calculate median metrics across cutoffs
    summary = (
        df_persona
        .groupby(['Tp_days', 'Tp_sec', 'wa_type'])
        .agg({
            'f1': ['median', 'std', 'min', 'max'],
            'f_beta': ['median', 'std'],
            'precision': 'median',
            'recall': 'median',
            'auc': 'median',
            'Ap': 'median',
            'Ap_percentile': 'median',
            'num_users': 'sum',
            'actual_churn_rate': 'mean',
            'pred_churn_rate': 'mean',
        })
        .reset_index()
    )

    # Flatten column names
    new_cols = []
    for col in summary.columns:
        if isinstance(col, tuple):
            new_cols.append('_'.join(str(c) for c in col).strip('_'))
        else:
            new_cols.append(str(col))
    summary.columns = new_cols

    # Select best based on median F-beta
    if 'f_beta_median' not in summary.columns:
        continue

    best_idx = summary['f_beta_median'].idxmax()
    best = summary.loc[best_idx]

    best_params_list.append({
        'persona': int(persona),
        'Tp_days': int(best['Tp_days']),
        'Tp_sec': float(best['Tp_sec']),
        'wa_type': best['wa_type'],
        'Ap': float(best['Ap_median']),
        'Ap_percentile': float(best['Ap_percentile_median']),
        'median_f1': float(best['f1_median']),
        'std_f1': float(best['f1_std']),
        'min_f1': float(best['f1_min']),
        'max_f1': float(best['f1_max']),
        'median_f_beta': float(best['f_beta_median']),
        'median_precision': float(best['precision_median']),
        'median_recall': float(best['recall_median']),
        'median_auc': float(best['auc_median']),
        'total_users': int(best['num_users_sum']),
        'avg_actual_churn_rate': float(best['actual_churn_rate_mean']),
        'avg_pred_churn_rate': float(best['pred_churn_rate_mean']),
    })

df_best = pd.DataFrame(best_params_list)

if len(df_best) == 0:
    print("\nERROR: No valid parameters found for any persona!")
    exit(1)

df_best = df_best.sort_values('persona')
df_best.to_csv("enhanced_wa_best_params.csv", index=False)

print(f"\nSaved best params → enhanced_wa_best_params.csv")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("BEST PARAMETERS PER PERSONA")
print("=" * 80)
print(df_best[[
    'persona', 'Tp_days', 'wa_type', 'Ap',
    'median_f1', 'median_precision', 'median_recall', 'std_f1'
]].to_string(index=False))

print("\n" + "=" * 80)
print("COMPARISON WITH BASELINE")
print("=" * 80)

# Compare WA types
wa_comparison = df_best.groupby('wa_type').agg({
    'median_f1': 'mean',
    'median_precision': 'mean',
    'median_recall': 'mean',
    'persona': 'count'
}).round(3)
wa_comparison.columns = ['Avg F1', 'Avg Precision', 'Avg Recall', 'N Personas']
print("\nPerformance by WA type:")
print(wa_comparison.to_string())

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Personas covered: {len(df_best)}/{len(personas)}")
print(f"Average median F1: {df_best['median_f1'].mean():.3f} (±{df_best['median_f1'].std():.3f})")
print(f"Average median precision: {df_best['median_precision'].mean():.3f}")
print(f"Average median recall: {df_best['median_recall'].mean():.3f}")
print(f"Average median AUC: {df_best['median_auc'].mean():.3f}")
print(f"\nF1 Score Distribution:")
print(f"  Min: {df_best['median_f1'].min():.3f}")
print(f"  25th percentile: {df_best['median_f1'].quantile(0.25):.3f}")
print(f"  Median: {df_best['median_f1'].median():.3f}")
print(f"  75th percentile: {df_best['median_f1'].quantile(0.75):.3f}")
print(f"  Max: {df_best['median_f1'].max():.3f}")

# Stability analysis
print("\n" + "=" * 80)
print("STABILITY ANALYSIS")
print("=" * 80)
stable = df_best[df_best['std_f1'] < 0.05]
moderate = df_best[(df_best['std_f1'] >= 0.05) & (df_best['std_f1'] < 0.10)]
unstable = df_best[df_best['std_f1'] >= 0.10]

print(f"Stable (std < 0.05): {len(stable)} personas")
print(f"Moderate (0.05 ≤ std < 0.10): {len(moderate)} personas")
print(f"Unstable (std ≥ 0.10): {len(unstable)} personas")

if len(unstable) > 0:
    print("\nUnstable personas:")
    print(unstable[['persona', 'median_f1', 'std_f1', 'min_f1', 'max_f1']].to_string(index=False))

print("\n" + "=" * 80)
print("INTERPRETABLE RULE SUMMARY")
print("=" * 80)
print(f"\nFor each persona, use the rule:")
print(f"  if WA_{{wa_type}}(Tp) < Ap → Predict CHURN")
print(f"\nWhere:")
print(f"  - Tp: Time period (lookback window in days)")
print(f"  - Ap: Activity threshold")
print(f"  - WA_{{wa_type}}(Tp): Weighted Activity calculated over Tp days")
print(f"\nOptimal parameters saved in: enhanced_wa_best_params.csv")

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
