import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import pandas as pd
import numpy as np

# Load grid search results
grid = pd.read_csv("../../../results/best_params/tp_ap_xgb_joint_search_sliding.csv")

print(f"Loaded {len(grid)} rows")
print(f"Personas: {sorted(grid['persona'].unique())}")
print(f"Date range: {grid['cutoff_date'].min()} to {grid['cutoff_date'].max()}")

# Calculate F1 first
if 'F1' not in grid.columns:
    grid["F1"] = 2 * (grid["precision"] * grid["recall"]) / (
        (grid["precision"] + grid["recall"]).replace(0, np.nan)
    )

print(f"\nMetrics summary:")
print(grid[['precision', 'recall', 'auc', 'F1']].describe())

# Method 1: Time-Averaged Performance
def select_params_time_average(df, min_recall=0.60, top_n=3):
    results = []
    
    for persona in df["persona"].unique():
        df_p = df[df["persona"] == persona].copy()
        
        print(f"\nProcessing persona {persona}: {len(df_p)} rows")
        
        avg_perf = (
            df_p.groupby(["alpha", "Tp_sec", "Ap"])
            .agg({
                "F1": "mean",
                "recall": "mean", 
                "precision": "mean",
                "auc": "mean",
                "num_users": "sum",
                "cutoff_date": "count"
            })
            .reset_index()
            .rename(columns={"cutoff_date": "n_periods"})
        )
        
        print(f"  Unique (alpha, Tp, Ap) combinations: {len(avg_perf)}")
        print(f"  Recall range: {avg_perf['recall'].min():.3f} to {avg_perf['recall'].max():.3f}")
        
        candidates = avg_perf[
            (avg_perf["recall"] >= min_recall) & 
            (avg_perf["n_periods"] >= 2)
        ].copy()
        
        print(f"  Candidates after filtering: {len(candidates)}")
        
        if len(candidates) == 0:
            print(f"  WARNING: No params meet criteria for persona {persona}")
            # Fallback: take best by F1 regardless of recall threshold
            candidates = avg_perf.nlargest(top_n, "F1")
            print(f"  Fallback: Using top {len(candidates)} by F1")
        else:
            candidates = candidates.sort_values("F1", ascending=False).head(top_n)
        
        for _, row in candidates.iterrows():
            results.append({
                "persona": int(persona),
                "alpha": row["alpha"],
                "Tp_sec": row["Tp_sec"],
                "Ap": row["Ap"],
                "avg_F1": row["F1"],
                "avg_recall": row["recall"],
                "avg_precision": row["precision"],
                "avg_auc": row["auc"],
                "n_periods": int(row["n_periods"]),
                "total_users": int(row["num_users"]),
                "selection_method": "time_average"
            })
    
    return pd.DataFrame(results)

# Method 2: Stability-Based Selection
def select_params_stability(df, min_recall=0.60):
    results = []
    
    for persona in df["persona"].unique():
        df_p = df[df["persona"] == persona].copy()
        
        stats = (
            df_p.groupby(["alpha", "Tp_sec", "Ap"])
            .agg({
                "F1": ["mean", "std"],
                "recall": ["mean", "std"],
                "precision": "mean",
                "auc": "mean",
                "cutoff_date": "count"
            })
            .reset_index()
        )
        
        stats.columns = [
            "alpha", "Tp_sec", "Ap",
            "F1_mean", "F1_std", 
            "recall_mean", "recall_std",
            "precision_mean", "auc_mean", "n_periods"
        ]
        
        candidates = stats[
            (stats["recall_mean"] >= min_recall) & 
            (stats["n_periods"] >= 2)
        ].copy()
        
        if len(candidates) == 0:
            candidates = stats.nlargest(1, "F1_mean")
        
        if candidates["F1_std"].max() > 0:
            candidates["F1_std_norm"] = (
                candidates["F1_std"] / candidates["F1_std"].max()
            )
        else:
            candidates["F1_std_norm"] = 0
            
        candidates["stability_score"] = (
            candidates["F1_mean"] * 0.7 - candidates["F1_std_norm"] * 0.3
        )
        
        best = candidates.nlargest(1, "stability_score").iloc[0]
        
        results.append({
            "persona": int(persona),
            "alpha": best["alpha"],
            "Tp_sec": best["Tp_sec"],
            "Ap": best["Ap"],
            "avg_F1": best["F1_mean"],
            "std_F1": best["F1_std"],
            "avg_recall": best["recall_mean"],
            "std_recall": best["recall_std"],
            "avg_precision": best["precision_mean"],
            "n_periods": int(best["n_periods"]),
            "stability_score": best["stability_score"],
            "selection_method": "stability"
        })
    
    return pd.DataFrame(results)

# Method 3: Train/Validation Split (with overlap check)
def select_params_validation_split(df, val_years=[2016, 2021], min_recall=0.60):
    results = []
    
    df["year"] = pd.to_datetime(df["cutoff_date"]).dt.year
    
    available_years = sorted(df["year"].unique())
    print(f"\nAvailable years: {available_years}")
    
    if len(available_years) < 3:
        print(f"WARNING: Not enough years for train/val split")
        return pd.DataFrame(results)
    
    val_years = available_years[-2:]
    train_years = available_years[:-2]
    print(f"Using train years: {train_years}, val years: {val_years}")
    
    for persona in df["persona"].unique():
        df_p = df[df["persona"] == persona].copy()
        
        train = df_p[df_p["year"].isin(train_years)]
        val = df_p[df_p["year"].isin(val_years)]
        
        if len(train) == 0 or len(val) == 0:
            print(f"  Persona {persona}: insufficient train/val data")
            continue
        
        # Find params tested in BOTH train and val
        train_params = set(train[["alpha", "Tp_sec", "Ap"]].itertuples(index=False, name=None))
        val_params = set(val[["alpha", "Tp_sec", "Ap"]].itertuples(index=False, name=None))
        common_params = train_params & val_params
        
        print(f"  Persona {persona}:")
        print(f"    Train params: {len(train_params)}")
        print(f"    Val params: {len(val_params)}")
        print(f"    Common params: {len(common_params)}")
        
        if len(common_params) == 0:
            print(f"    WARNING: No overlapping params - using alpha-only matching")
            # Fallback: match only by alpha (allow Ap to vary)
            train_alphas = set(train["alpha"].unique())
            val_alphas = set(val["alpha"].unique())
            common_alphas = train_alphas & val_alphas
            
            if len(common_alphas) == 0:
                print(f"    ERROR: No common alphas at all - skipping")
                continue
            
            # Filter to common alphas
            train = train[train["alpha"].isin(common_alphas)]
            val = val[val["alpha"].isin(common_alphas)]
            
            # Average by alpha only
            train_avg = (
                train.groupby("alpha")
                .agg({"F1": "mean", "recall": "mean", "Tp_sec": "first", "Ap": "mean"})
                .reset_index()
            )
            
            train_candidates = train_avg[train_avg["recall"] >= min_recall]
            if len(train_candidates) == 0:
                train_candidates = train_avg.nlargest(1, "F1")
            
            best_params = train_candidates.nlargest(1, "F1").iloc[0]
            
            # Evaluate on val with same alpha
            val_perf = val[val["alpha"] == best_params["alpha"]]
            
        else:
            # Normal path: exact param matching
            common_df = pd.DataFrame(list(common_params), columns=["alpha", "Tp_sec", "Ap"])
            
            train_common = train.merge(common_df, on=["alpha", "Tp_sec", "Ap"])
            
            train_avg = (
                train_common.groupby(["alpha", "Tp_sec", "Ap"])
                .agg({"F1": "mean", "recall": "mean"})
                .reset_index()
            )
            
            train_candidates = train_avg[train_avg["recall"] >= min_recall]
            if len(train_candidates) == 0:
                train_candidates = train_avg.nlargest(1, "F1")
            
            best_params = train_candidates.nlargest(1, "F1").iloc[0]
            
            val_perf = val[
                (val["alpha"] == best_params["alpha"]) &
                (val["Tp_sec"] == best_params["Tp_sec"]) &
                (val["Ap"] == best_params["Ap"])
            ]
        
        if len(val_perf) == 0:
            print(f"    ERROR: Still no validation data - skipping")
            continue
        
        results.append({
            "persona": int(persona),
            "alpha": best_params["alpha"],
            "Tp_sec": best_params["Tp_sec"],
            "Ap": best_params["Ap"],
            "train_F1": best_params["F1"],
            "train_recall": best_params["recall"],
            "val_F1": val_perf["F1"].mean(),
            "val_recall": val_perf["recall"].mean(),
            "val_precision": val_perf["precision"].mean(),
            "val_auc": val_perf["auc"].mean(),
            "selection_method": "train_val_split"
        })
        
        print(f"    SUCCESS: F1={best_params['F1']:.3f} (train) -> {val_perf['F1'].mean():.3f} (val)")
    
    return pd.DataFrame(results)

# Run all methods
print("\n" + "=" * 60)
print("Method 1: Time-Averaged Performance")
print("=" * 60)
best_time_avg = select_params_time_average(grid, min_recall=0.60, top_n=3)
print(f"\nResults: {len(best_time_avg)} rows")
if len(best_time_avg) > 0:
    print(best_time_avg.to_string(index=False))
    best_time_avg.to_csv("../../../results/best_params/best_params_time_average.csv", index=False)
    print("\nSaved to: best_params_time_average.csv")

print("\n" + "=" * 60)
print("Method 2: Stability-Based Selection")
print("=" * 60)
best_stable = select_params_stability(grid, min_recall=0.60)
print(f"\nResults: {len(best_stable)} rows")
if len(best_stable) > 0:
    print(best_stable.to_string(index=False))
    best_stable.to_csv("../../../results/best_params/best_params_stability.csv", index=False)
    print("\nSaved to: best_params_stability.csv")

print("\n" + "=" * 60)
print("Method 3: Train/Validation Split")
print("=" * 60)
best_val = select_params_validation_split(grid, min_recall=0.60)
print(f"\nResults: {len(best_val)} rows")
if len(best_val) > 0:
    print(best_val.to_string(index=False))
    best_val.to_csv("../../../results/best_params/best_params_validation.csv", index=False)
    print("\nSaved to: best_params_validation.csv")

print("\n" + "=" * 80)
print("DONE! Check the CSV files for results.")
print("=" * 80)