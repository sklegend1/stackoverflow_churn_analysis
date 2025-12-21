import polars as pl
import pandas as pd
import numpy as np

"""
Evaluation script using ADJUSTED Ap values to reduce over-prediction.
This compares results between original and adjusted Ap values.
"""

# -----------------------------------------
# 1) Load core event log
# -----------------------------------------
elog = pl.read_parquet("core_eventlog_eval_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

# -----------------------------------------
# 2) Load BOTH original and adjusted params
# -----------------------------------------
params_original = pd.read_csv("calibrated_ap_ratio_scaling.csv")
params_adjusted = pd.read_csv("calibrated_ap_ratio_scaling_adjusted.csv")

print("=" * 80)
print("COMPARING ORIGINAL vs ADJUSTED Ap VALUES")
print("=" * 80)
print("\nSample adjustments:")
comparison = pd.DataFrame({
    'persona': params_adjusted['persona'],
    'Tp_days': params_adjusted['Tp_days'],
    'Ap_original': params_adjusted['Ap_original'],
    'Ap_adjusted': params_adjusted['Ap'],
    'change_%': params_adjusted['pct_change']
})
print(comparison.head(10).to_string(index=False))

# Create maps for ADJUSTED values
Ap_map_adjusted = dict(zip(params_adjusted["persona"], params_adjusted["Ap"]))
Tp_map = dict(zip(params_adjusted["persona"], params_adjusted["Tp_sec"]))

# -----------------------------------------
# 3) Evaluation cutoff & grace period
# -----------------------------------------
cutoff = pl.datetime(2021, 3, 1)
grace_days = 180
grace_sec = grace_days * 86400

rows_adjusted = []

# -----------------------------------------
# 4) Loop over personas with ADJUSTED Ap
# -----------------------------------------
print("\n" + "=" * 80)
print("EVALUATING WITH ADJUSTED Ap VALUES")
print("=" * 80)

for persona, Ap_adj in Ap_map_adjusted.items():
    Tp_sec = Tp_map[persona]

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

    # Only users active in this window
    active_users = win.select("user_id").unique()

    if active_users.height == 0:
        continue

    # Calculate WA_enhanced (same as calibration script)
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

    wa = features.with_columns([
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0) *
            pl.col("recency_weight")
        ).alias("WA")
    ]).select(["user_id", "WA"])

    wa = wa.join(active_users, on="user_id", how="inner")

    # Predict with ADJUSTED Ap
    wa = wa.with_columns([
        (pl.col("WA") < Ap_adj).cast(pl.Int8).alias("pred_churn")
    ])

    # -------------------------------------
    # 5) Actual churn after cutoff + grace
    # -------------------------------------
    after = (
        elog
        .filter(pl.col("persona") == persona)
        .filter((pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec))
                & (pl.col("timestamp") < cutoff + pl.duration(seconds=grace_sec) + pl.duration(seconds=365 * 86400)))
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

    # Metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    rows_adjusted.append({
        "persona": int(persona),
        "Tp_sec": float(Tp_sec),
        "Tp_days": int(float(Tp_sec) / 86400.0),
        "Ap": float(Ap_adj),
        "num_users": int(len(y_true)),
        "actual_churn_rate": actual_rate,
        "pred_churn_rate": pred_rate,
        "pred/actual": (pred_rate / actual_rate) if actual_rate > 0 else np.nan,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

# -----------------------------------------
# 6) Save results and compare
# -----------------------------------------
df_adjusted = pd.DataFrame(rows_adjusted)
df_adjusted.to_csv("ap_tp_simple_eval_adjusted.csv", index=False)
print("\n✓ Saved → ap_tp_simple_eval_adjusted.csv")

# Load original results for comparison
df_original = pd.read_csv("ap_tp_simple_eval.csv")

# Calculate overall metrics
def calc_overall(df):
    return {
        'avg_pred/actual': df['pred/actual'].mean(),
        'avg_f1': df['f1'].mean(),
        'avg_precision': df['precision'].mean(),
        'avg_recall': df['recall'].mean(),
        'avg_accuracy': df['accuracy'].mean(),
    }

overall_original = calc_overall(df_original)
overall_adjusted = calc_overall(df_adjusted)

print("\n" + "=" * 80)
print("COMPARISON: ORIGINAL vs ADJUSTED")
print("=" * 80)

print(f"\n{'Metric':<20} {'Original':<15} {'Adjusted':<15} {'Change':<15}")
print("-" * 65)

for key in ['avg_pred/actual', 'avg_f1', 'avg_precision', 'avg_recall', 'avg_accuracy']:
    orig = overall_original[key]
    adj = overall_adjusted[key]
    change = ((adj / orig) - 1) * 100 if orig != 0 else 0

    label = key.replace('avg_', '').replace('pred/actual', 'Rate Ratio')
    print(f"{label:<20} {orig:<15.3f} {adj:<15.3f} {change:+.1f}%")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("✓ Rate Ratio should decrease (closer to 1.0 = better)")
print("✓ F1 may decrease slightly (acceptable trade-off)")
print("✓ Precision should increase (fewer false positives)")
print("✓ Recall may decrease (missing some churners, but more accurate overall)")
print("=" * 80)

if len(rows_adjusted) > 0:
    total_correct = sum(r["accuracy"] * r["num_users"] for r in rows_adjusted)
    total_users = sum(r["num_users"] for r in rows_adjusted)
    overall_acc = total_correct / total_users
    print(f"\nOverall accuracy (weighted by users): {overall_acc:.4f}")
else:
    print("\nNo personas had active users in the evaluation window.")