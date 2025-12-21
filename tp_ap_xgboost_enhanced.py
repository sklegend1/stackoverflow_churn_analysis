import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    accuracy_score, roc_curve
)
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

"""
ENHANCED XGBoost Ap/Tp Finder with Trend + Recency Features

Improvements over original:
1. Multiple WA metrics: simple, trend, recency, enhanced
2. Same XGBoost optimization framework
3. Tests all WA types and selects best per persona
"""

print("=" * 80)
print("ENHANCED XGBoost Ap/Tp Optimization with Trend + Recency")
print("=" * 80)

# -------------------------------------------------------
# Load core event log
# -------------------------------------------------------
elog = pl.read_parquet("core_eventlog_train_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])
elog = elog.with_columns([
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64)
])

print(f"Event log loaded: {elog.height:,} events")
print(f"Date range: {elog['timestamp'].min()} to {elog['timestamp'].max()}")

# -------------------------------------------------------
# p95 gap per persona (sec)
# -------------------------------------------------------
p95_df = (
    elog
    .filter(pl.col("gap_time_key_actions_seconds") > 0)
    .group_by("persona")
    .agg(
        pl.col("gap_time_key_actions_seconds")
        .quantile(0.95)
        .alias("p95_gap_sec")
    )
)

p95_map = dict(zip(p95_df["persona"], p95_df["p95_gap_sec"]))

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
cutoff_years = []
for y in range(2016, 2023):
    cutoff_years.append((y, 1))

grace_period_sec = 90 * 86400
alpha_values = np.arange(4, 20, 1.5).tolist()

# WA types to test
WA_TYPES = ['WA_simple', 'WA_trend', 'WA_recency', 'WA_enhanced']

print(f"\nConfiguration:")
print(f"  Cutoff years: {len(cutoff_years)}")
print(f"  Alpha values: {alpha_values}")
print(f"  WA types: {WA_TYPES}")
print(f"  Grace period: {grace_period_sec // 86400} days")

# -------------------------------------------------------
# Function to calculate enhanced WA metrics
# -------------------------------------------------------
def calculate_wa_metrics(win, cutoff, Tp_sec):
    """
    Calculate multiple WA metrics for users in window.

    Returns DataFrame with columns:
    - user_id
    - WA_simple: sum(gain)
    - WA_trend: sum(gain) * activity_trend
    - WA_recency: sum(gain) * recency_weight
    - WA_enhanced: sum(gain) * activity_trend * recency_weight
    """
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))
    mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

    # Calculate features per user
    features = (
        win
        .with_columns([
            (pl.col("timestamp") >= mid_point).alias("is_recent"),
            ((cutoff - pl.col("timestamp")).dt.total_seconds() / 86400).alias("days_ago"),
        ])
        .group_by("user_id")
        .agg([
            # Basic
            pl.col("gain").sum().alias("total_gain"),

            # Trend components
            pl.col("gain").filter(pl.col("is_recent")).sum().alias("recent_gain"),
            pl.col("gain").filter(~pl.col("is_recent")).sum().alias("old_gain"),

            # Recency
            pl.col("days_ago").min().alias("days_since_last"),
        ])
    )

    # Calculate WA variants
    features = features.with_columns([
        # WA_simple: traditional
        pl.col("total_gain").alias("WA_simple"),

        # Activity trend
        ((pl.col("recent_gain") + 1.0) / (pl.col("old_gain") + 1.0)).alias("activity_trend"),

        # Recency weight
        ((-pl.col("days_since_last") / 30.0).exp()).alias("recency_weight"),
    ])

    # Apply trend and recency
    features = features.with_columns([
        # WA_trend
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0)
        ).alias("WA_trend"),

        # WA_recency
        (
            pl.col("total_gain") *
            pl.col("recency_weight")
        ).alias("WA_recency"),

        # WA_enhanced: combination
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0) *
            pl.col("recency_weight")
        ).alias("WA_enhanced"),
    ])

    return features.select(["user_id", "WA_simple", "WA_trend", "WA_recency", "WA_enhanced"])

# -------------------------------------------------------
# Main optimization loop
# -------------------------------------------------------
rows = []

for persona, p95_gap in p95_map.items():
    print(f"\n{'='*60}")
    print(f"PERSONA {persona}")
    print(f"{'='*60}")

    # Filter logs for this persona once
    elog_p = elog.filter(pl.col("persona") == persona)

    if elog_p.height == 0 or not np.isfinite(p95_gap):
        continue

    for alpha in alpha_values:
        Tp_sec = float(alpha * p95_gap)
        Tp_days = Tp_sec / 86400

        # Skip if Tp out of reasonable range
        if Tp_sec > (86400 * 270) or Tp_sec < (86400 * 60):
            continue

        print(f"\n  Alpha={alpha}, Tp={Tp_days:.0f} days:")

        for year, month in cutoff_years:
            cutoff = pl.datetime(year, 1, 1)
            window_start = cutoff - pl.duration(seconds=int(Tp_sec))

            # Events inside training window
            win = (
                elog_p
                .filter(
                    (pl.col("timestamp") >= window_start) &
                    (pl.col("timestamp") <= cutoff)
                )
            )

            # Users with activity in window
            active_users = win.select("user_id").unique()["user_id"].to_list()
            if len(active_users) < 20:
                continue

            # Calculate all WA metrics
            wa_metrics = calculate_wa_metrics(win, cutoff, Tp_sec)

            # Future activity after cutoff + grace period
            future = (
                elog_p
                .filter(
                    (pl.col("timestamp") > (cutoff + pl.duration(seconds=grace_period_sec * month))) &
                    (pl.col("timestamp") < (cutoff + pl.duration(seconds=366 * 86400 + grace_period_sec * month)))
                )
                .group_by("user_id")
                .agg(pl.len().alias("cnt"))
            )

            # Join WA with future to build labels
            actual = (
                wa_metrics.join(future, on="user_id", how="left")
                  .with_columns([
                      (pl.col("cnt").is_null()).cast(pl.Int8).alias("actual_churn")
                  ])
            )

            pdf = actual.to_pandas()
            y = pdf["actual_churn"].values

            # Need both classes
            if np.unique(y).size < 2:
                continue

            # ---------------------------------------------------
            # Test each WA type with XGBoost
            # ---------------------------------------------------
            for wa_type in WA_TYPES:
                if wa_type not in pdf.columns:
                    continue

                X = pdf[[wa_type]].values

                # Train XGBoost on this WA metric
                model = XGBClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    n_estimators=60,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42
                )
                model.fit(X, y)

                pred_prob = model.predict_proba(X)[:, 1]

                # Find optimal Ap using precision-recall curve
                precision, recall, thresholds = precision_recall_curve(y, pred_prob)

                target_recall = 0.8
                target_precision = 0.65

                if thresholds.size == 0:
                    Ap = 0.5
                else:
                    cond = (recall[1:] >= target_recall) & (precision[1:] >= target_precision)
                    idxs = np.where(cond)[0]
                    if idxs.size:
                        Ap = float(thresholds[idxs[0]])
                    else:
                        cand = np.where(recall[1:] >= target_recall)[0]
                        if cand.size:
                            best = cand[np.argmax(precision[1:][cand])]
                            Ap = float(thresholds[best])
                        else:
                            best = np.argmax(recall[1:])
                            Ap = float(thresholds[best])

                # Binary prediction with Ap
                pred = (pred_prob >= Ap).astype(int)
                pred_rate = float(pred.mean())

                # Calibrate Ap
                # Ap_Calibrated = Ap * ((y.mean()) / pred_rate) if pred_rate > 0 else Ap
                # Ap = Ap_Calibrated

                if Ap < 0.05:
                    continue

                # Metrics
                try:
                    auc = float(roc_auc_score(y, pred_prob))
                except:
                    auc = 0.5

                acc = float(accuracy_score(y, pred))
                prec = float(precision_score(y, pred, zero_division=0))
                rec = float(recall_score(y, pred, zero_division=0))
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                cutoff_str = f"{year:04d}-01-01"

                rows.append({
                    "persona": int(persona),
                    "wa_type": wa_type,
                    "cutoff_date": cutoff_str,
                    "grace_period_days": (grace_period_sec * month) // 86400,
                    "alpha": float(alpha),
                    "Tp_sec": Tp_sec,
                    "Tp_days": Tp_days,
                    "p95_gap_sec": float(p95_gap),
                    "Ap": Ap,
                    "num_users": int(len(y)),
                    "actual_churn_rate": float(y.mean()),
                    "pred_churn_rate": float(pred.mean()),
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "auc": auc,
                    "f1": f1
                })

            print(f"    Year {year}: Tested {len(WA_TYPES)} WA types")

# -------------------------------------------------------
# Save all results
# -------------------------------------------------------
df_out = pd.DataFrame(rows)
df_out.sort_values(["persona", "wa_type", "cutoff_date", "f1"], inplace=True, ascending=[True, True, True, False])
df_out.to_csv("tp_ap_xgb_enhanced_all_results.csv", index=False)

print("\n" + "=" * 80)
print(f"Saved all results → tp_ap_xgb_enhanced_all_results.csv ({len(df_out)} rows)")
print("=" * 80)

# -------------------------------------------------------
# Select best parameters per persona FOR EACH WA TYPE
# -------------------------------------------------------
print("\n" + "=" * 80)
print("SELECTING BEST PARAMETERS PER PERSONA FOR EACH WA TYPE")
print("=" * 80)

personas = sorted(df_out["persona"].unique())

# Store best params for each WA type
all_wa_best_params = {}

for wa_type in WA_TYPES:
    print(f"\n--- Processing {wa_type} ---")
    best_params_list = []

    for persona in personas:
        df_persona = df_out[df_out["persona"] == persona]

        if len(df_persona) == 0:
            continue

        # FILTER: Only keep current WA type results
        df_persona_wa = df_persona[df_persona["wa_type"] == wa_type]

        if len(df_persona_wa) == 0:
            print(f"Warning: Persona {persona} has no {wa_type} results, skipping...")
            continue

        # For each (Tp), calculate median F1 across cutoffs
        summary = (
            df_persona_wa
            .groupby(['wa_type', 'Tp_days', 'Tp_sec', 'alpha'])
            .agg({
                'f1': ['median', 'std', 'min', 'max'],
                'precision': 'median',
                'recall': 'median',
                'auc': 'median',
                'Ap': 'median',
                'num_users': 'sum',
                'actual_churn_rate': 'mean',
                'pred_churn_rate': 'mean',
            })
            .reset_index()
        )

        # Flatten column names
        summary.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else str(col)
                           for col in summary.columns.values]

        # Select best based on median F1
        if 'f1_median' not in summary.columns:
            continue

        best_idx = summary['f1_median'].idxmax()
        best = summary.loc[best_idx]

        best_params_list.append({
            'persona': int(persona),
            'wa_type': best['wa_type'],
            'Tp_days': int(best['Tp_days']),
            'Tp_sec': float(best['Tp_sec']),
            'alpha': float(best['alpha']),
            'Ap': float(best['Ap_median']),
            'median_f1': float(best['f1_median']),
            'std_f1': float(best['f1_std']),
            'min_f1': float(best['f1_min']),
            'max_f1': float(best['f1_max']),
            'median_precision': float(best['precision_median']),
            'median_recall': float(best['recall_median']),
            'median_auc': float(best['auc_median']),
            'total_users': int(best['num_users_sum']),
            'avg_actual_churn_rate': float(best['actual_churn_rate_mean']),
            'avg_pred_churn_rate': float(best['pred_churn_rate_mean']),
        })

    if len(best_params_list) == 0:
        print(f"WARNING: No valid parameters found for {wa_type}!")
        continue

    df_best_wa = pd.DataFrame(best_params_list)
    df_best_wa = df_best_wa.sort_values('persona')

    # Save to file
    output_file = f"tp_ap_xgb_enhanced_best_params_{wa_type}.csv"
    df_best_wa.to_csv(output_file, index=False)
    print(f"Saved → {output_file} ({len(df_best_wa)} personas)")

    # Store for comparison
    all_wa_best_params[wa_type] = df_best_wa

# Also save the original WA_trend file for backward compatibility
if 'WA_trend' in all_wa_best_params:
    all_wa_best_params['WA_trend'].to_csv("tp_ap_xgb_enhanced_best_params.csv", index=False)
    print(f"\nSaved WA_trend → tp_ap_xgb_enhanced_best_params.csv (for backward compatibility)")

# -------------------------------------------------------
# Display results for each WA type
# -------------------------------------------------------
print("\n" + "=" * 80)
print("BEST PARAMETERS PER PERSONA FOR EACH WA TYPE")
print("=" * 80)

for wa_type, df_best_wa in all_wa_best_params.items():
    print(f"\n{'='*70}")
    print(f"{wa_type} - {len(df_best_wa)} personas")
    print(f"{'='*70}")
    print(df_best_wa[[
        'persona', 'Tp_days', 'Ap',
        'median_f1', 'median_precision', 'median_recall', 'std_f1'
    ]].to_string(index=False))

    # Summary stats for this WA type
    print(f"\n{wa_type} Summary:")
    print(f"  Personas covered: {len(df_best_wa)}/{len(personas)}")
    print(f"  Average median F1: {df_best_wa['median_f1'].mean():.3f} (±{df_best_wa['median_f1'].std():.3f})")
    print(f"  Average median precision: {df_best_wa['median_precision'].mean():.3f}")
    print(f"  Average median recall: {df_best_wa['median_recall'].mean():.3f}")
    print(f"  Average median AUC: {df_best_wa['median_auc'].mean():.3f}")

# -------------------------------------------------------
# WA types comparison summary
# -------------------------------------------------------
print("\n" + "=" * 80)
print("WA TYPE PERFORMANCE COMPARISON (Best Params)")
print("=" * 80)

comparison_data = []
for wa_type, df_best_wa in all_wa_best_params.items():
    comparison_data.append({
        'WA Type': wa_type,
        'Avg F1': df_best_wa['median_f1'].mean(),
        'Avg Precision': df_best_wa['median_precision'].mean(),
        'Avg Recall': df_best_wa['median_recall'].mean(),
        'Avg AUC': df_best_wa['median_auc'].mean(),
        'N Personas': len(df_best_wa)
    })

df_comparison = pd.DataFrame(comparison_data).round(3)
df_comparison = df_comparison.sort_values('Avg F1', ascending=False)
print(df_comparison.to_string(index=False))

print(f"\n** Generated separate best_params files for all {len(WA_TYPES)} WA types **")

print("\n" + "=" * 80)
print("DONE! Enhanced XGBoost Ap/Tp optimization complete.")
print("=" * 80)
print("\nOutput files:")
print("  1. tp_ap_xgb_enhanced_all_results.csv - All combinations tested")
print("  2. tp_ap_xgb_enhanced_best_params.csv - Best params (WA_trend, backward compatibility)")
print("  3. tp_ap_xgb_enhanced_best_params_WA_simple.csv - Best params for WA_simple")
print("  4. tp_ap_xgb_enhanced_best_params_WA_trend.csv - Best params for WA_trend")
print("  5. tp_ap_xgb_enhanced_best_params_WA_recency.csv - Best params for WA_recency")
print("  6. tp_ap_xgb_enhanced_best_params_WA_enhanced.csv - Best params for WA_enhanced")