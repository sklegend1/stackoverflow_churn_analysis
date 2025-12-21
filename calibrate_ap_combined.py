import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

"""
Combined Calibration Method: Optimize both F1 and Rate Ratio

This script combines the strengths of different calibration methods:
- Start with ratio_scaling Ap (good F1)
- Apply iterative adjustment to bring rate_ratio closer to 1.0
"""

print("=" * 80)
print("COMBINED CALIBRATION: Optimize F1 + Rate Ratio")
print("=" * 80)

# Load base parameters (from XGBoost)
base_params = pd.read_csv("tp_ap_xgb_enhanced_best_params.csv")
print(f"\nLoaded {len(base_params)} personas from base params")

# Load event logs
elog_train = pl.read_parquet("core_eventlog_train_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

elog_eval = pl.read_parquet("core_eventlog_eval_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

# Configuration
GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400
CHURN_CHECK_DAYS = 365
CHURN_CHECK_SEC = CHURN_CHECK_DAYS * 86400

# Training cutoffs
TRAIN_CUTOFFS = [
    pl.datetime(2016, 1, 1),
    pl.datetime(2017, 1, 1),
    pl.datetime(2018, 1, 1),
    pl.datetime(2019, 1, 1),
    pl.datetime(2020, 1, 1),
    pl.datetime(2021, 1, 1),
    pl.datetime(2022, 1, 1),
    pl.datetime(2023, 1, 1),
]

# Eval cutoffs
EVAL_CUTOFFS = [
    pl.datetime(2016, 7, 1),
    pl.datetime(2017, 7, 1),
    pl.datetime(2018, 7, 1),
    pl.datetime(2019, 7, 1),
    pl.datetime(2020, 7, 1),
    pl.datetime(2021, 7, 1),
    pl.datetime(2022, 7, 1),
    pl.datetime(2023, 7, 1),
]

def calculate_wa_enhanced(elog_window, cutoff, Tp_sec):
    """Calculate WA_enhanced for users in window."""
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))
    mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

    win = elog_window.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    )

    if win.height == 0:
        return None

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

    features = features.with_columns([
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0) *
            pl.col("recency_weight")
        ).alias("WA_enhanced")
    ])

    return features.select(["user_id", "WA_enhanced"])

def get_calibration_data(elog, persona, Tp_sec, cutoff):
    """Get (WA_scores, actual_churn_labels) for calibration."""
    elog_persona = elog.filter(pl.col("persona") == persona)

    # Calculate WA
    wa_features = calculate_wa_enhanced(elog_persona, cutoff, Tp_sec)

    if wa_features is None or wa_features.height < 20:
        return None, None

    # Get actual churn labels
    future_start = cutoff + pl.duration(seconds=GRACE_SEC)
    future_end = future_start + pl.duration(seconds=CHURN_CHECK_SEC)

    future_activity = (
        elog_persona
        .filter(
            (pl.col("timestamp") > future_start) &
            (pl.col("timestamp") <= future_end)
        )
        .group_by("user_id")
        .agg(pl.len().alias("future_events"))
    )

    eval_df = (
        wa_features
        .join(future_activity, on="user_id", how="left")
        .with_columns([
            (pl.col("future_events").is_null() | (pl.col("future_events") == 0))
            .cast(pl.Int8).alias("actual_churn")
        ])
    )

    pdf = eval_df.to_pandas()

    return pdf["WA_enhanced"].values, pdf["actual_churn"].values

def calibrate_combined(persona, Tp_sec, Ap_base):
    """
    Combined calibration method:
    1. Use precision-recall curve to find optimal Ap
    2. Apply iterative adjustment for rate_ratio
    """
    # Collect training data
    all_WA = []
    all_y = []

    for cutoff in TRAIN_CUTOFFS:
        WA_scores, y_true = get_calibration_data(elog_train, persona, Tp_sec, cutoff)
        if WA_scores is not None:
            all_WA.append(WA_scores)
            all_y.append(y_true)

    if len(all_WA) == 0:
        return Ap_base

    WA_combined = np.concatenate(all_WA)
    y_combined = np.concatenate(all_y)

    # Step 1: Use precision-recall curve to find Ap with good F1
    precision, recall, thresholds = precision_recall_curve(y_combined, 1 - WA_combined / WA_combined.max())

    # Target: High recall (catch churners) + reasonable precision
    target_recall = 0.75
    target_precision = 0.45

    if thresholds.size == 0:
        Ap_initial = Ap_base
    else:
        # Find threshold meeting targets
        cond = (recall[:-1] >= target_recall) & (precision[:-1] >= target_precision)
        idxs = np.where(cond)[0]

        if idxs.size > 0:
            # Convert back to WA threshold
            prob_threshold = thresholds[idxs[0]]
            Ap_initial = WA_combined.max() * (1 - prob_threshold)
        else:
            # Fallback: maximize F1
            f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
            best_idx = np.argmax(f1_scores)
            prob_threshold = thresholds[best_idx]
            Ap_initial = WA_combined.max() * (1 - prob_threshold)

    # Step 2: Adjust Ap to balance rate_ratio
    # Calculate current prediction rate
    y_pred_initial = (WA_combined < Ap_initial).astype(int)
    actual_rate = y_combined.mean()
    pred_rate_initial = y_pred_initial.mean()

    if pred_rate_initial == 0:
        return Ap_initial

    # If over-predicting, increase Ap (make it harder to be classified as churn)
    # If under-predicting, decrease Ap
    rate_ratio = pred_rate_initial / actual_rate

    # Apply gentle correction (don't overcorrect)
    if rate_ratio > 1.2:  # Over-predicting
        # Increase Ap to reduce predictions
        correction_factor = min(1.5, rate_ratio ** 0.5)  # Gentle correction
        Ap_calibrated = Ap_initial * correction_factor
    elif rate_ratio < 0.8:  # Under-predicting
        # Decrease Ap to increase predictions
        correction_factor = max(0.7, rate_ratio ** 0.5)
        Ap_calibrated = Ap_initial * correction_factor
    else:
        # Rate ratio is good, keep it
        Ap_calibrated = Ap_initial

    # Ensure reasonable bounds
    Ap_calibrated = np.clip(Ap_calibrated, WA_combined.min() * 1.1, WA_combined.max() * 0.9)

    return Ap_calibrated

# Main calibration loop
calibrated_params = []

for idx, row in base_params.iterrows():
    persona = int(row['persona'])
    Tp_sec = float(row['Tp_sec'])
    Ap_base = float(row['Ap'])

    print(f"\nCalibrating Persona {persona}...")
    print(f"  Base Ap: {Ap_base:.4f}, Tp: {Tp_sec/86400:.0f} days")

    Ap_calibrated = calibrate_combined(persona, Tp_sec, Ap_base)

    print(f"  Calibrated Ap: {Ap_calibrated:.4f} (change: {(Ap_calibrated/Ap_base - 1)*100:+.1f}%)")

    calibrated_params.append({
        'persona': persona,
        'Tp_sec': Tp_sec,
        'Tp_days': int(Tp_sec / 86400),
        'Ap_base': Ap_base,
        'Ap': Ap_calibrated,
    })

df_calibrated = pd.DataFrame(calibrated_params)
df_calibrated.to_csv("calibrated_ap_combined.csv", index=False)

print("\n" + "=" * 80)
print("Saved → calibrated_ap_combined.csv")
print("=" * 80)

# Evaluate on eval set
print("\n" + "=" * 80)
print("EVALUATING ON EVAL SET")
print("=" * 80)

eval_results = []

for idx, row in df_calibrated.iterrows():
    persona = row['persona']
    Tp_sec = row['Tp_sec']
    Ap_calibrated = row['Ap']

    for eval_cutoff in EVAL_CUTOFFS:
        WA_scores, y_true = get_calibration_data(elog_eval, persona, Tp_sec, eval_cutoff)

        if WA_scores is None:
            continue

        y_pred = (WA_scores < Ap_calibrated).astype(int)

        actual_rate = y_true.mean()
        pred_rate = y_pred.mean()

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        rate_ratio = pred_rate / actual_rate if actual_rate > 0 else np.nan

        eval_results.append({
            'persona': persona,
            'eval_cutoff': str(eval_cutoff),
            'Tp_sec': Tp_sec,
            'Ap': Ap_calibrated,
            'num_users': len(y_true),
            'actual_churn_rate': actual_rate,
            'pred_churn_rate': pred_rate,
            'rate_ratio': rate_ratio,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        })

df_eval = pd.DataFrame(eval_results)

# Summary
summary = df_eval.groupby('persona').agg({
    'f1': 'mean',
    'rate_ratio': 'mean',
    'precision': 'mean',
    'recall': 'mean',
}).round(3)

print("\nPER-PERSONA RESULTS:")
print(summary)

overall = df_eval.agg({
    'f1': 'mean',
    'rate_ratio': 'mean',
    'precision': 'mean',
    'recall': 'mean',
}).round(3)

print("\n" + "=" * 80)
print("OVERALL RESULTS:")
print(f"  F1:         {overall['f1']:.3f}")
print(f"  Rate Ratio: {overall['rate_ratio']:.3f} (target: 1.0)")
print(f"  Precision:  {overall['precision']:.3f}")
print(f"  Recall:     {overall['recall']:.3f}")
print("=" * 80)

print("\nDONE!")
