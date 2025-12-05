import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------
# Load logs and churn labels
# -------------------------------------------------------
elog = pl.read_parquet("core_eventlog_v1.parquet")
elog = elog.with_columns([
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64)
])

labels = pl.read_parquet("seed_churn_labels_per_user.parquet")

# We assume labels has columns: user_id, persona, churn_seed

# -------------------------------------------------------
# Persona-level p95 gap values (key action gaps in seconds)
# -------------------------------------------------------
gap_stats = (
    elog
    .filter(pl.col("gap_time_key_actions_seconds") > 0)
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

# -------------------------------------------------------
# Candidate alpha values (Tp = alpha * p95_gap)
# -------------------------------------------------------
alpha_values = np.arange(1, 16, 4).tolist()

results_rows = []

for persona, p95_gap in p95_map.items():
    # Filter event log for this persona in training window
    elog_p = train_elog.filter(pl.col("persona") == persona)
    if elog_p.height == 0:
        continue

    # Last timestamp for this persona in training data
    max_ts = elog_p.select(pl.max("timestamp")).item()

    for alpha in alpha_values:
        Tp_sec = float(alpha * p95_gap)

        # Define time window [max_ts - Tp_sec, max_ts]
        window_start = max_ts - pl.duration(seconds=int(Tp_sec))
        df_w = elog_p.filter(pl.col("timestamp") >= window_start)
        # df_w = elog_p
        if df_w.height == 0:
            continue

        # ---------------------------------------------------
        # Build feature matrix: user_id x action_type_id counts
        # ---------------------------------------------------
        feat = (
            df_w
            .group_by(["user_id", "action_type_id"])
            .agg(
            (pl.col("gain").sum()).alias("gain")
            )
            .pivot(
            values="gain",
            index="user_id",
            on="action_type_id"
            )
            .fill_null(0)
        )
        # feat = (
        #     df_w
        #     .group_by(["user_id", "action_type_id"])
        #     .agg(
        #     pl.col("action_type_id").len().alias("count")
        #     )
        #     .pivot(
        #     values="count",
        #     index="user_id",
        #     on="action_type_id"
        #     )
        #     .fill_null(0)
        # )

        if feat.height == 0:
            continue

        feat_users = feat["user_id"].to_list()

        # Align churn labels with feature users
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

        if X_df.height == 0 or y_df.height == 0:
            continue

        X = X_df.drop("user_id").to_pandas()
        y = y_df["churn_seed"].to_pandas()

        # Need both classes to train logistic regression
        if y.nunique() < 2:
            continue

        # ---------------------------------------------------
        # Train logistic regression
        # ---------------------------------------------------
        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

        pred = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred)

        coef = model.coef_[0]
        intercept = model.intercept_[0]

        # Decision boundary in weighted-activity space:
        # WeightedActivity = sum_j w_j * x_j
        # churn if WeightedActivity < Ap
        # At decision boundary: intercept + WeightedActivity = 0
        # => Ap = -intercept
        Ap = float(-intercept)

        n_users = int(len(y))
        n_churn_1 = int(y.sum())
        n_churn_0 = n_users - n_churn_1

        results_rows.append(
            {
                "persona": int(persona),
                "alpha": float(alpha),
                "Tp_sec": Tp_sec,
                "p95_gap_sec": float(p95_gap),
                "num_users": n_users,
                "num_churn_1": n_churn_1,
                "num_churn_0": n_churn_0,
                "auc": float(auc),
                "Ap": Ap,
            }
        )

# -------------------------------------------------------
# Save results to CSV
# -------------------------------------------------------
if results_rows:
    df_res = pd.DataFrame(results_rows)
    df_res.sort_values(["persona", "auc"], inplace=True, ascending=[True, False])
    df_res.to_csv("tp_ap_gridsearch_results.csv", index=False)
    print("Saved → tp_ap_gridsearch_results.csv")
else:
    print("No valid results were computed (check data/labels).")
