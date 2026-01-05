import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_auc_score,
    precision_score, recall_score, accuracy_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("F1/F2 OPTIMIZATION WITH TIME-WEIGHTED EVAL & CONSTRAINED Tp")
print("=" * 80)

elog = pl.read_parquet("../../../data/processed/core_eventlog_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64),
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

windows = []
for y in range(2015, 2023):
    for m in [1, 7]:
        time_weight = np.exp((y - 2015) / 5)
        windows.append((y, m, time_weight))

print(f"\nWindows with time weights:")
for y, m, w in windows[-5:]:
    print(f"  {y}-{m:02d}: weight={w:.2f}")

grace_period_sec = 180 * 86400

# CONSTRAINT: Tp should be between 30 and 180 days
MIN_TP_DAYS = 30
MAX_TP_DAYS = 180
MIN_TP_SEC = MIN_TP_DAYS * 86400
MAX_TP_SEC = MAX_TP_DAYS * 86400

print(f"\nTp constraints: {MIN_TP_DAYS} to {MAX_TP_DAYS} days")

# Generate Tp values directly in seconds (not alpha-based)
Tp_values_days = [30, 45, 60, 90, 120, 150, 180]
Tp_values_sec = [d * 86400 for d in Tp_values_days]

print(f"Tp values (days): {Tp_values_days}")

def extract_rich_features(win_df, Tp_sec):
    features = (
        win_df
        .group_by("user_id")
        .agg([
            pl.col("gain").sum().alias("total_gain"),
            pl.col("gain").mean().alias("avg_gain"),
            pl.col("gain").std().alias("std_gain"),
            pl.len().alias("num_actions"),
            pl.col("action_type_id").n_unique().alias("unique_actions"),
            pl.col("post_id").n_unique().alias("unique_posts"),
            pl.col("timestamp").min().alias("first_ts"),
            pl.col("timestamp").max().alias("last_ts"),
        ])
        .with_columns([
            ((pl.col("last_ts") - pl.col("first_ts")).dt.total_seconds() / 86400).alias("activity_span_days"),
            (pl.col("num_actions") / (Tp_sec / 86400)).alias("actions_per_day"),
            ((Tp_sec / 86400) - ((pl.col("last_ts") - pl.col("first_ts")).dt.total_seconds() / 86400)).alias("days_since_last"),
        ])
        .drop(["first_ts", "last_ts"])
        .fill_null(0)
    )
    return features

all_results = []

for persona in sorted(p95_map.keys()):
    p95_gap = p95_map[persona]
    
    if not np.isfinite(p95_gap):
        continue
    
    elog_p = elog.filter(pl.col("persona") == persona)
    
    if elog_p.height < 500:
        continue
    
    print(f"\n{'='*60}")
    print(f"Persona {persona} (p95_gap={p95_gap/86400:.1f} days)")
    print(f"{'='*60}")
    
    for Tp_sec in Tp_values_sec:
        Tp_days = Tp_sec / 86400
        
        print(f"\nTp={Tp_days:.0f} days ({Tp_sec:.0f} sec)")
        
        window_scores = []
        
        for year, month, time_weight in windows:
            cutoff = pl.datetime(year, month, 1)
            window_start = cutoff - pl.duration(seconds=int(Tp_sec))
            
            win = elog_p.filter(
                (pl.col("timestamp") >= window_start) &
                (pl.col("timestamp") <= cutoff)
            )
            
            if win.height < 50:
                continue
            
            features = extract_rich_features(win, Tp_sec)
            
            future = (
                elog_p
                .filter(pl.col("timestamp") > cutoff + pl.duration(seconds=grace_period_sec))
                .group_by("user_id")
                .agg(pl.len().alias("cnt"))
            )
            
            data = (
                features.join(future, on="user_id", how="left")
                .with_columns([
                    (pl.col("cnt").is_null()).cast(pl.Int8).alias("actual_churn")
                ])
            )
            
            if data.height < 20:
                continue
            
            pdf = data.to_pandas()
            y = pdf["actual_churn"].values
            
            if y.sum() < 3 or (len(y) - y.sum()) < 3:
                continue
            
            X_xgb = pdf[[
                "total_gain", "avg_gain", "std_gain", "num_actions",
                "unique_actions", "unique_posts", "activity_span_days",
                "actions_per_day", "days_since_last"
            ]].fillna(0).values
            
            WA = pdf["total_gain"].values
            
            scale_pos_weight = (len(y) - y.sum()) / y.sum()
            
            model = XGBClassifier(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42
            )
            
            try:
                model.fit(X_xgb, y, verbose=False)
                pred_prob = model.predict_proba(X_xgb)[:, 1]
            except:
                continue
            
            sorted_idx = np.argsort(WA)
            WA_sorted = WA[sorted_idx]
            y_sorted = y[sorted_idx]
            
            best_f1 = 0
            best_f2 = 0
            best_Ap_f1 = 0
            best_Ap_f2 = 0
            
            percentiles = np.linspace(10, 90, 30)
            for pct in percentiles:
                threshold = np.percentile(WA, pct)
                pred = (WA < threshold).astype(int)
                
                if pred.sum() > 0 and (len(pred) - pred.sum()) > 0:
                    precision = precision_score(y, pred, zero_division=0)
                    recall = recall_score(y, pred, zero_division=0)
                    
                    f1 = f1_score(y, pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_Ap_f1 = threshold
                    
                    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-10)
                    if f2 > best_f2:
                        best_f2 = f2
                        best_Ap_f2 = threshold
            
            if best_f1 > 0:
                metrics = {}
                for metric_name, Ap in [("f1", best_Ap_f1), ("f2", best_Ap_f2)]:
                    pred = (WA < Ap).astype(int)
                    
                    metrics[metric_name] = {
                        "Ap": Ap,
                        "f1": f1_score(y, pred, zero_division=0),
                        "precision": precision_score(y, pred, zero_division=0),
                        "recall": recall_score(y, pred, zero_division=0),
                    }
                
                try:
                    auc = roc_auc_score(y, pred_prob)
                except:
                    auc = 0.5
                
                window_scores.append({
                    "year": year,
                    "month": month,
                    "cutoff": f"{year}-{month:02d}-01",
                    "time_weight": time_weight,
                    "num_users": len(y),
                    "churn_rate": y.mean(),
                    "auc": auc,
                    **{f"{k}_{mk}": mv for k, m in metrics.items() for mk, mv in m.items()}
                })
        
        if len(window_scores) >= 5:
            weights = np.array([w["time_weight"] for w in window_scores])
            weights = weights / weights.sum()
            
            for metric in ["f1", "f2"]:
                weighted_f1 = np.average([w[f"{metric}_f1"] for w in window_scores], weights=weights)
                weighted_recall = np.average([w[f"{metric}_recall"] for w in window_scores], weights=weights)
                weighted_precision = np.average([w[f"{metric}_precision"] for w in window_scores], weights=weights)
                
                median_Ap = np.median([w[f"{metric}_Ap"] for w in window_scores])
                median_f1 = np.median([w[f"{metric}_f1"] for w in window_scores])
                
                recent_f1 = np.mean([w[f"{metric}_f1"] for w in window_scores[-3:]])
                old_f1 = np.mean([w[f"{metric}_f1"] for w in window_scores[:3]]) if len(window_scores) >= 6 else 0
                
                all_results.append({
                    "persona": int(persona),
                    "Tp_sec": int(Tp_sec),
                    "Tp_days": int(Tp_days),
                    "Ap": float(median_Ap),
                    "p95_gap_sec": float(p95_gap),
                    "optimization_metric": metric,
                    "n_windows": len(window_scores),
                    "weighted_f1": weighted_f1,
                    "weighted_recall": weighted_recall,
                    "weighted_precision": weighted_precision,
                    "median_f1": median_f1,
                    "median_precision": np.median([w[f"{metric}_precision"] for w in window_scores]),
                    "median_recall": np.median([w[f"{metric}_recall"] for w in window_scores]),
                    "median_auc": np.median([w["auc"] for w in window_scores]),
                    "recent_f1": recent_f1,
                    "old_f1": old_f1,
                    "stability": 1 - abs(recent_f1 - old_f1),
                })
            
            print(f"  Windows: {len(window_scores)}, Weighted F1={weighted_f1:.3f}")

df_all = pd.DataFrame(all_results)
df_all.to_csv("../../../results/f1_f2_optimization_results.csv", index=False)
print(f"\n\nSaved all results: {len(df_all)} rows")

for metric in ["f1", "f2"]:
    df_metric = df_all[df_all["optimization_metric"] == metric]
    
    best_params = (
        df_metric
        .sort_values(["persona", "weighted_f1"], ascending=[True, False])
        .groupby("persona")
        .head(1)
    )
    
    output_file = f"best_params_{metric}_optimized.csv"
    best_params.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print(f"BEST PARAMETERS - {metric.upper()} OPTIMIZED")
    print("=" * 80)
    print(best_params[[
        "persona", "Tp_days", "Ap", "weighted_f1", 
        "weighted_recall", "weighted_precision",
        "recent_f1", "old_f1", "stability"
    ]].to_string(index=False))
    
    print(f"\nSummary:")
    print(f"  Average weighted F1: {best_params['weighted_f1'].mean():.3f}")
    print(f"  Average weighted recall: {best_params['weighted_recall'].mean():.3f}")
    print(f"  Average weighted precision: {best_params['weighted_precision'].mean():.3f}")
    print(f"  Average recent F1: {best_params['recent_f1'].mean():.3f}")
    print(f"  Average old F1: {best_params['old_f1'].mean():.3f}")
    print(f"  Average stability: {best_params['stability'].mean():.3f}")
    print(f"  Performance gap (recent - old): {(best_params['recent_f1'].mean() - best_params['old_f1'].mean()):.3f}")
    print(f"\nSaved to: {output_file}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("F1: Balanced precision/recall (use when both matter equally)")
print("F2: Prioritizes recall 2x (use when catching churners is critical)")
print("\nLook at 'stability' column: values closer to 1.0 mean consistent")
print("performance across old and recent time periods.")