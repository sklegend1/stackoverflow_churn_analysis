import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle

# -------------------------------------------------------
# Load logs and labels
# -------------------------------------------------------
elog = pl.read_parquet("../../../data/processed/core_eventlog_v1.parquet")
elog = elog.with_columns([
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64)
])
labels = pl.read_parquet("../../../data/processed/seed_churn_labels_per_user.parquet")

# Join churn labels to the event log
elog = (
    elog.join(
        labels.select(["user_id", "persona", "churn_seed"]),
        on=["user_id", "persona"],
        how="inner"
    )
)

# -------------------------------------------------------
# Persona-level p95 gap values
# -------------------------------------------------------
gap_stats = (
    elog.filter(pl.col("gap_time_key_actions_seconds") > 0)
        .group_by("persona")
        .agg(
            pl.col("gap_time_key_actions_seconds")
              .quantile(0.95)
              .alias("p95_gap")
        )
        
)

p95_map = dict(zip(gap_stats["persona"], gap_stats["p95_gap"]))

# -------------------------------------------------------
# Training window cutoff (fixed at Jan 1, 2019)
# -------------------------------------------------------
train_cutoff = pl.datetime(2019, 1, 1)
train_elog = elog.filter(pl.col("timestamp") <= train_cutoff)

# Candidate α values
alpha_values = [1.5, 2.0, 2.5, 3.0]

results = []

for alpha in alpha_values:
    persona_results = []
    auc_scores = []

    for persona, p95_gap in p95_map.items():

        Tp_sec = alpha * p95_gap

        elog_p = train_elog.filter(pl.col("persona") == persona)
        if elog_p.height == 0:
            continue

        max_ts = elog_p.select(pl.max("timestamp")).item()
        window_start = max_ts - pl.duration(seconds=int(Tp_sec))

        df_w = elog_p.filter(pl.col("timestamp") >= window_start)

        # Pivot user × action_type counts
        feat = (
            df_w.group_by(["user_id", "action_type_id"])
                 .agg(pl.len().alias("count"))
                 .pivot(
                     values="count",
                     index="user_id",
                     on="action_type_id"
                 )
                 .fill_null(0)
                 
        )

        if feat.height == 0:
            continue

        # Align X and y by user_id
        feat_users = feat["user_id"].to_list()

        y_df = (
            labels
            .filter(pl.col("persona") == persona)
            .filter(pl.col("user_id").is_in(feat_users))
            .sort("user_id")
            
        )

        X_df = (
            feat
            .filter(pl.col("user_id").is_in(feat_users))
            .sort("user_id")
        )

        X = X_df.drop("user_id").to_pandas()
        y = y_df["churn_seed"].to_pandas()

        if y.nunique() < 2:
            continue

        # Train logistic model
        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

        pred = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred)

        # Compute Ap (threshold in weighted space)
        coef = model.coef_[0]
        intercept = model.intercept_[0]

        # Ap derived as decision boundary in weighted-activity space
        Ap = -intercept

        persona_results.append({
            "persona": persona,
            "Tp_sec": float(Tp_sec),
            "Ap": float(Ap),
            "model": model,
            "auc": auc
        })

        auc_scores.append(auc)

    results.append({
        "alpha": alpha,
        "mean_auc": float(np.mean(auc_scores)) if auc_scores else 0.0,
        "details": persona_results
    })

best = max(results, key=lambda d: d["mean_auc"])

print("Best alpha:", best["alpha"])
print("Mean AUC:", best["mean_auc"])

with open("../../../checkpoints/tp_ap_models.pkl", "wb") as f:
    pickle.dump(best, f)
# 2) Flat CSV for inspection
rows = []
for r in best["details"]:
    rows.append({
        "persona": r["persona"],
        "Tp_sec": r["Tp_sec"],
        "Ap": r["Ap"],
        "AUC": r["auc"],
        "alpha_used": best["alpha"]
    })

pd.DataFrame(rows).to_csv("../../../results/tp_ap_estimates.csv", index=False)
print("Saved → tp_ap_models.pkl")
