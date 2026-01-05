import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    accuracy_score, f1_score
)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

print("=" * 80)
print("XGBoost Ap/Tp with Per-Persona Calibrated Thresholds")
print("=" * 80)

elog = pl.read_parquet("../../../data/processed/core_eventlog_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])
elog = elog.with_columns([
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64)
])

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

cutoff_years = []
for y in range(2017, 2023):
    cutoff_years.append((y, 1))

grace_period_sec = 180 * 86400
alpha_values = np.arange(2, 8, 1).tolist()

rows = []

for persona, p95_gap in p95_map.items():
    elog_p = elog.filter(pl.col("persona") == persona)

    if elog_p.height == 0 or not np.isfinite(p95_gap):
        continue

    print(f"\n{'='*60}")
    print(f"Persona {persona}")
    print(f"{'='*60}")

    for alpha in alpha_values:
        Tp_sec = float(alpha * p95_gap)
        Tp_days = Tp_sec / 86400
        
        if Tp_days > 270 or Tp_days < 60:
            continue

        print(f"\n  Alpha={alpha}, Tp={Tp_days:.0f} days")
        
        # Collect data from all time windows for this (persona, Tp)
        all_WA = []
        all_y = []
        all_window_ids = []
        
        for window_id, (year, month) in enumerate(cutoff_years):
            cutoff = pl.datetime(year, month, 1)
            window_start = cutoff - pl.duration(seconds=int(Tp_sec))

            win = (
                elog_p
                .filter(
                    (pl.col("timestamp") >= window_start) &
                    (pl.col("timestamp") <= cutoff)
                )
            )

            active_users = win.select("user_id").unique()["user_id"].to_list()
            if len(active_users) < 20:
                continue

            wa = (
                win
                .group_by("user_id")
                .agg(pl.col("gain").sum().alias("WA"))
            )

            future = (
                elog_p
                .filter(pl.col("timestamp") > (cutoff + pl.duration(seconds=grace_period_sec)))
                .group_by("user_id")
                .agg(pl.len().alias("cnt"))
            )

            actual = (
                wa.join(future, on="user_id", how="left")
                  .with_columns([
                      (pl.col("cnt").is_null()).cast(pl.Int8).alias("actual_churn")
                  ])
            )

            pdf = actual.to_pandas()
            y = pdf["actual_churn"].values
            WA = pdf["WA"].values

            if np.unique(y).size < 2 or len(y) < 10:
                continue

            all_WA.extend(WA)
            all_y.extend(y)
            all_window_ids.extend([window_id] * len(y))
        
        if len(all_y) < 100:
            continue
        
        all_WA = np.array(all_WA)
        all_y = np.array(all_y)
        all_window_ids = np.array(all_window_ids)
        
        actual_churn_rate = all_y.mean()
        
        # Train XGBoost on all data
        X = all_WA.reshape(-1, 1)
        
        scale_pos_weight = (len(all_y) - all_y.sum()) / all_y.sum()
        
        model = XGBClassifier(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X, all_y, verbose=False)
        
        pred_prob = model.predict_proba(X)[:, 1]
        
        # CALIBRATED Ap: Find threshold on WA that gives actual churn rate
        # Sort by WA ascending (low WA = high churn risk)
        sorted_indices = np.argsort(all_WA)
        WA_sorted = all_WA[sorted_indices]
        
        # Find Ap such that % of users with WA < Ap ≈ actual_churn_rate
        Ap_index = int(actual_churn_rate * len(WA_sorted))
        Ap_calibrated = float(WA_sorted[Ap_index])
        
        # Predict with calibrated Ap
        pred_calibrated = (all_WA < Ap_calibrated).astype(int)
        
        # Also find XGBoost-based Ap (for comparison)
        # Find threshold that maximizes F1 on probability
        best_f1 = 0
        best_Ap_xgb = 0.5
        
        for threshold_pct in np.linspace(10, 90, 20):
            threshold_wa = np.percentile(all_WA, threshold_pct)
            pred_temp = (all_WA < threshold_wa).astype(int)
            
            if pred_temp.sum() > 0 and (len(pred_temp) - pred_temp.sum()) > 0:
                f1_temp = f1_score(all_y, pred_temp, zero_division=0)
                if f1_temp > best_f1:
                    best_f1 = f1_temp
                    best_Ap_xgb = threshold_wa
        
        pred_xgb = (all_WA < best_Ap_xgb).astype(int)
        
        # Metrics for both approaches
        # Calibrated
        f1_cal = f1_score(all_y, pred_calibrated, zero_division=0)
        prec_cal = precision_score(all_y, pred_calibrated, zero_division=0)
        rec_cal = recall_score(all_y, pred_calibrated, zero_division=0)
        acc_cal = accuracy_score(all_y, pred_calibrated)
        
        # XGBoost-optimized
        f1_xgb = f1_score(all_y, pred_xgb, zero_division=0)
        prec_xgb = precision_score(all_y, pred_xgb, zero_division=0)
        rec_xgb = recall_score(all_y, pred_xgb, zero_division=0)
        acc_xgb = accuracy_score(all_y, pred_xgb)
        
        try:
            auc = roc_auc_score(all_y, pred_prob)
        except:
            auc = 0.5
        
        # Save both
        rows.append({
            "persona": int(persona),
            "alpha": float(alpha),
            "Tp_sec": int(Tp_sec),
            "Tp_days": int(Tp_days),
            "p95_gap_sec": float(p95_gap),
            "method": "calibrated",
            "Ap": float(Ap_calibrated),
            "num_users": int(len(all_y)),
            "num_windows": len(cutoff_years),
            "actual_churn_rate": float(actual_churn_rate),
            "pred_churn_rate": float(pred_calibrated.mean()),
            "accuracy": float(acc_cal),
            "precision": float(prec_cal),
            "recall": float(rec_cal),
            "f1": float(f1_cal),
            "auc": float(auc),
        })
        
        rows.append({
            "persona": int(persona),
            "alpha": float(alpha),
            "Tp_sec": int(Tp_sec),
            "Tp_days": int(Tp_days),
            "p95_gap_sec": float(p95_gap),
            "method": "xgb_optimized",
            "Ap": float(best_Ap_xgb),
            "num_users": int(len(all_y)),
            "num_windows": len(cutoff_years),
            "actual_churn_rate": float(actual_churn_rate),
            "pred_churn_rate": float(pred_xgb.mean()),
            "accuracy": float(acc_xgb),
            "precision": float(prec_xgb),
            "recall": float(rec_xgb),
            "f1": float(f1_xgb),
            "auc": float(auc),
        })
        
        print(f"    Calibrated:    Ap={Ap_calibrated:.3f}, F1={f1_cal:.3f}, "
              f"actual={actual_churn_rate:.2%}, pred={pred_calibrated.mean():.2%}")
        print(f"    XGB-optimized: Ap={best_Ap_xgb:.3f}, F1={f1_xgb:.3f}, "
              f"actual={actual_churn_rate:.2%}, pred={pred_xgb.mean():.2%}")

df_out = pd.DataFrame(rows)
df_out.to_csv("../../../results/xgb_ap_tp_calibrated_comparison.csv", index=False)

print("\n" + "=" * 80)
print("COMPARISON: Calibrated vs XGB-Optimized")
print("=" * 80)

# Best params per persona for each method
for method in ["calibrated", "xgb_optimized"]:
    df_method = df_out[df_out["method"] == method]
    best_params = (
        df_method
        .sort_values(["persona", "f1"], ascending=[True, False])
        .groupby("persona")
        .head(1)
    )
    
    print(f"\n{method.upper()}:")
    print(f"  Average F1: {best_params['f1'].mean():.3f}")
    print(f"  Average Recall: {best_params['recall'].mean():.3f}")
    print(f"  Average Precision: {best_params['precision'].mean():.3f}")
    print(f"  Avg pred rate error: {abs(best_params['pred_churn_rate'] - best_params['actual_churn_rate']).mean():.3f}")
    
    output_file = f"best_params_{method}.csv"
    best_params.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

print("\n" + "=" * 80)
print("Per-Persona Best Parameters (Calibrated):")
print("=" * 80)

best_cal = df_out[df_out["method"] == "calibrated"].sort_values(
    ["persona", "f1"], ascending=[True, False]
).groupby("persona").head(1)

print(best_cal[[
    "persona", "Tp_days", "Ap", "f1", "recall", "precision",
    "actual_churn_rate", "pred_churn_rate"
]].to_string(index=False))

print("\nSaved → xgb_ap_tp_calibrated_comparison.csv")