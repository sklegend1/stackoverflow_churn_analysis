import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ----------------------------------------
# Load data
# ----------------------------------------
core = pl.read_parquet("../../../data/processed/core_eventlog_v1.parquet")
core = core.with_columns([
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64)
])
labels = pl.read_parquet("../../../data/processed/seed_churn_labels_per_user.parquet")

# Ensure gain is float
core = core.with_columns([
    pl.col("gain").cast(pl.Float64)
])

# Join churn labels
core = (
    core.join(
        labels.select(["user_id", "persona", "churn_seed"]),
        on=["user_id", "persona"],
        how="inner",
    )
)

# Training cutoff: we only use events up to this time
train_cutoff = pl.datetime(2019, 1, 1)

core_train = core.filter(pl.col("timestamp") <= train_cutoff)

# ----------------------------------------
# Persona-level p95 gap (seconds)
# ----------------------------------------
gap_stats = (
    core_train
    .filter(pl.col("gap_time_key_actions_seconds") > 0)
    .group_by("persona")
    .agg(
        pl.col("gap_time_key_actions_seconds")
          .quantile(0.95)
          .alias("p95_gap_sec")
    )
)

p95_map = dict(zip(gap_stats["persona"], gap_stats["p95_gap_sec"]))

# Candidate alpha values for Tp = alpha * p95_gap
ALPHAS = [3.0, 5.0, 7.0, 10.0]

rows = []

for persona, p95_gap_sec in p95_map.items():
    core_p = core_train.filter(pl.col("persona") == persona)

    if core_p.height == 0:
        continue

    for alpha in ALPHAS:
        Tp_sec = alpha * p95_gap_sec

        # Time window: [train_cutoff - Tp_sec, train_cutoff]
        window_start = train_cutoff - pl.duration(seconds=int(Tp_sec))

        # Filter events inside the window
        win = core_p.filter(
            (pl.col("timestamp") >= window_start) &
            (pl.col("timestamp") <= train_cutoff)
        )

        if win.height == 0:
            continue

        # Aggregate weighted activity per user: sum(gain)
        wa = (
            win
            .group_by("user_id")
            .agg(pl.col("gain").sum().alias("weighted_activity"))
        )

        # Join labels for this persona
        y_df = (
            labels
            .filter(pl.col("persona") == persona)
            .join(wa, on="user_id", how="inner")
        )

        if y_df.height < 20:
            # Not enough samples for this combination
            continue

        # Prepare X, y
        y_df = y_df.select(["user_id", "weighted_activity", "churn_seed"])
        y_df = y_df.sort("user_id")

        X = y_df["weighted_activity"].to_pandas().values.reshape(-1, 1)
        y = y_df["churn_seed"].to_pandas().values

        if np.unique(y).shape[0] < 2:
            # Only one class -> can't train logistic
            continue

        # Logistic regression with a single feature
        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

        # AUC
        y_pred = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred)

        coef = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])

        # If coef == 0, the model is degenerate; skip Ap
        if coef == 0.0:
            Ap = np.nan
        else:
            Ap = -intercept / coef   # decision boundary in WA space

        rows.append({
            "persona": int(persona),
            "alpha": float(alpha),
            "Tp_sec": float(Tp_sec),
            "p95_gap_sec": float(p95_gap_sec),
            "num_users": int(len(y)),
            "num_churn_1": int(y.sum()),
            "num_churn_0": int((y == 0).sum()),
            "auc": float(auc),
            "Ap": float(Ap),
        })

# Save all results
df_res = pd.DataFrame(rows)
df_res.to_csv("../../../results/best_params/tp_ap_gridsearch_scalarWA.csv", index=False)
print("Saved -> tp_ap_gridsearch_scalarWA.csv")

# Optionally, print best alpha per persona
if not df_res.empty:
    best_per_persona = (
        df_res.sort_values(["persona", "auc"], ascending=[True, False])
              .groupby("persona")
              .head(1)
    )
    print("Best (persona, alpha, Tp_sec, Ap, AUC):")
    print(best_per_persona[["persona", "alpha", "Tp_sec", "Ap", "auc"]])
