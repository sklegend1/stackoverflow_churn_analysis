import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("IMPROVED XGBOOST WITH RICH FEATURES + DIRECT Ap OPTIMIZATION")
print("=" * 80)

elog = pl.read_parquet("../../../data/processed/core_eventlog_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"Event log: {elog.height:,} events")

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

# Generate windows
windows = []
for y in range(2015, 2023):
    for m in [1, 7]:
        windows.append((y, m))

grace_period_sec = 60 * 86400
alpha_values = np.arange(2, 16, 2).tolist()

print(f"\nAlpha values: {alpha_values}")
print(f"Windows: {len(windows)}")

def extract_rich_features(win_df, Tp_sec):
    """
    Extract rich features beyond just sum of gain
    """
    features = (
        win_df
        .group_by("user_id")
        .agg([
            # Activity features
            pl.col("gain").sum().alias("total_gain"),
            pl.col("gain").mean().alias("avg_gain"),
            pl.col("gain").std().alias("std_gain"),
            pl.len().alias("num_actions"),
            
            # Diversity features
            pl.col("action_type_id").n_unique().alias("unique_actions"),
            pl.col("post_id").n_unique().alias("unique_posts"),
            
            # Temporal features
            pl.col("timestamp").min().alias("first_ts"),
            pl.col("timestamp").max().alias("last_ts"),
        ])
        .with_columns([
            # Activity span
            ((pl.col("last_ts") - pl.col("first_ts")).dt.total_seconds() / 86400).alias("activity_span_days"),
            # Activity intensity
            (pl.col("num_actions") / (Tp_sec / 86400)).alias("actions_per_day"),
            # Recency (days since last activity from window end)
            ((Tp_sec / 86400) - ((pl.col("last_ts") - pl.col("first_ts")).dt.total_seconds() / 86400)).alias("days_since_last"),
        ])
        .drop(["first_ts", "last_ts"])
        .fill_null(0)
    )
    
    return features

all_results = []

for persona, p95_gap in p95_map.items():
    if not np.isfinite((p95_gap)) :
        continue
    
    elog_p = elog.filter(pl.col("persona") == persona)
    
    if elog_p.height < 500:
        continue
    
    print(f"\n{'='*60}")
    print(f"Persona {persona}")
    print(f"{'='*60}")
    
    for alpha in alpha_values:
        Tp_sec = float(alpha * p95_gap)
        Tp_days = Tp_sec / 86400
        
        print(f"\nAlpha={alpha:.1f}, Tp={Tp_days:.1f} days")
        
        window_scores = []
        
        for year, month in windows:
            cutoff = pl.datetime(year, month, 1)
            window_start = cutoff - pl.duration(seconds=int(Tp_sec))
            
            win = elog_p.filter(
                (pl.col("timestamp") >= window_start) &
                (pl.col("timestamp") <= cutoff)
            )
            
            if win.height < 50:
                continue
            
            # Extract rich features
            features = extract_rich_features(win, Tp_sec)
            
            # Get labels
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
            
            # Prepare features for XGBoost
            X_xgb = pdf[[
                "total_gain", "avg_gain", "std_gain", "num_actions",
                "unique_actions", "unique_posts", "activity_span_days",
                "actions_per_day", "days_since_last"
            ]].fillna(0).values
            
            WA = pdf["total_gain"].values
            
            # Train XGBoost with class weights
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
            
            # Find optimal Ap directly on WA values
            # Sort by WA ascending (low WA = high churn risk)
            sorted_idx = np.argsort(WA)
            WA_sorted = WA[sorted_idx]
            y_sorted = y[sorted_idx]
            
            best_f1 = 0
            best_Ap = 0
            
            # Try different Ap thresholds based on WA percentiles
            percentiles = np.linspace(10, 90, 20)
            for pct in percentiles:
                threshold = np.percentile(WA, pct)
                pred = (WA < threshold).astype(int)
                
                if pred.sum() > 0 and (len(pred) - pred.sum()) > 0:
                    f1 = f1_score(y, pred, zero_division=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_Ap = threshold
            
            if best_f1 > 0:
                pred_optimal = (WA < best_Ap).astype(int)
                
                from sklearn.metrics import precision_score, recall_score, accuracy_score
                
                precision = precision_score(y, pred_optimal, zero_division=0)
                recall = recall_score(y, pred_optimal, zero_division=0)
                accuracy = accuracy_score(y, pred_optimal)
                
                try:
                    auc = roc_auc_score(y, pred_prob)
                except:
                    auc = 0.5
                
                window_scores.append({
                    "cutoff": f"{year}-{month:02d}-01",
                    "f1": best_f1,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "auc": auc,
                    "Ap": best_Ap,
                    "num_users": len(y),
                    "churn_rate": y.mean()
                })
        
        if len(window_scores) >= 3:
            median_f1 = np.median([w["f1"] for w in window_scores])
            median_Ap = np.median([w["Ap"] for w in window_scores])
            
            all_results.append({
                "persona": int(persona),
                "alpha": alpha,
                "Tp_sec": Tp_sec,
                "Tp_days": Tp_days,
                "Ap": median_Ap,
                "p95_gap_sec": p95_gap,
                "n_windows": len(window_scores),
                "median_f1": median_f1,
                "mean_f1": np.mean([w["f1"] for w in window_scores]),
                "std_f1": np.std([w["f1"] for w in window_scores]),
                "median_precision": np.median([w["precision"] for w in window_scores]),
                "median_recall": np.median([w["recall"] for w in window_scores]),
                "median_auc": np.median([w["auc"] for w in window_scores]),
            })
            
            print(f"  Windows: {len(window_scores)}, Median F1: {median_f1:.3f}, Ap: {median_Ap:.4f}")

# Save all results
df_all = pd.DataFrame(all_results)
df_all.to_csv("../../../results/xgb_rich_features_all_results.csv", index=False)
print(f"\nSaved all results: {len(df_all)} rows")

# Select best params per persona
best_params = (
    df_all
    .sort_values(["persona", "median_f1"], ascending=[True, False])
    .groupby("persona")
    .head(1)
)

best_params.to_csv("../../../results/best_params/best_params_xgb_rich.csv", index=False)

print("\n" + "=" * 80)
print("BEST PARAMETERS PER PERSONA")
print("=" * 80)
print(best_params[[
    "persona", "Tp_days", "Ap", "median_f1", 
    "median_recall", "median_precision", "n_windows"
]].to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total personas: {len(best_params)}")
print(f"Average median F1: {best_params['median_f1'].mean():.3f}")
print(f"Average median recall: {best_params['median_recall'].mean():.3f}")
print(f"Average median precision: {best_params['median_precision'].mean():.3f}")
print("\nSaved to: best_params_xgb_rich.csv")