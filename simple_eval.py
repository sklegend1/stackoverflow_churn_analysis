import polars as pl
import pandas as pd
import numpy as np

# -----------------------------------------
# 1) Load core event log
# -----------------------------------------
elog = pl.read_parquet("core_eventlog_eval_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

# -----------------------------------------
# 2) Load best Ap / Tp per persona
#   
# -----------------------------------------
# grid = pd.read_csv("tp_ap_xgb_joint_search_sliding.csv")

#  Select best row per persona based on AUC
# best_params_auc = (
#     grid.sort_values(["persona", "f1"], ascending=[True, False])
#         .groupby("persona")
#         .head(1)
# )
# Ap_map= dict(zip(best_params_auc["persona"], best_params_auc["Ap"] ))
# Tp_map= dict(zip(best_params_auc["persona"], best_params_auc["Tp_sec"] ))
# F1_map= dict(zip(best_params_auc["persona"], best_params_auc["f1"] ))
# date_map= dict(zip(best_params_auc["persona"], best_params_auc["cutoff_date"] ))
# grace_map= dict(zip(best_params_auc["persona"], best_params_auc["grace_period_days"] ))
# Compute F1 as harmonic mean of precision & recall
# grid["F1"] = 2 * (grid["precision"] * grid["recall"]) / (
#     (grid["precision"] + grid["recall"]).replace(0, np.nan)
# )

# # Select best row per persona based on F1
# best_params = (
#     grid.sort_values(["persona", "F1"], ascending=[True, False])
#         .groupby("persona")
#         .head(1)
# )
# print("Best params per persona based on F1:")
# print(best_params[["persona", "Ap", "Tp_sec", "precision", "recall"]])
# Ap_map = dict(zip(best_params["persona"], best_params["Ap"]))
# Tp_map = dict(zip(best_params["persona"], best_params["Tp_sec"]))

# pragmatic_best_params_file = pd.read_csv("best_params_sliding_median_f1.csv")
# pragmatic_best_params_file = pd.read_csv("best_params_sliding_median_f1.csv")
# pragmatic_best_params_file = pd.read_csv("sliding_window_all_results.csv")

#  Select best row per persona based on AUC
# pragmatic_best_params = (
#     pragmatic_best_params_file.sort_values(["persona", "median_f1"], ascending=[True, False])
#         .groupby("persona")
#         .head(1)
# )
# calibrated_params_file = pd.read_csv("calibration_evaluation_results.csv")

# Ensure numeric comparison works even if the CSV has strings
# calibrated_params_file["rate_ratio"] = pd.to_numeric(calibrated_params_file["rate_ratio"], errors="coerce")

#  Select best row per persona based on rate_ratio close to 1
# pragmatic_best_params = (
#     calibrated_params_file.loc[calibrated_params_file["rate_ratio"].between(0.7, 1.5) & (calibrated_params_file["Ap"] > 0.1)]
#         .sort_values(["persona", "f1"], ascending=[True, False])
#         .groupby("persona")
#         .head(1)
# )
# print(pragmatic_best_params)
pragmatic_best_params = pd.read_csv("lstm_derived_ap_tp_2016.csv")
Ap_map= dict(zip(pragmatic_best_params["persona"], pragmatic_best_params["Ap"]/1 ))
Tp_map= dict(zip(pragmatic_best_params["persona"], pragmatic_best_params["Tp_sec"] ))
# F1_map= dict(zip(pragmatic_best_params["persona"], pragmatic_best_params["median_f1"] ))
# -----------------------------------------
# 3) Evaluation cutoff & grace period
# -----------------------------------------
cutoff = pl.datetime(2016, 1, 1)      
grace_days = 120 
grace_sec = grace_days * 86400
# difficulty_factor = dict(zip(pragmatic_best_params["persona"], pragmatic_best_params["avg_actual_churn_rate"]/pragmatic_best_params["avg_pred_churn_rate"] ))

rows = []

# -----------------------------------------
# 4) Loop over personas
# -----------------------------------------
for persona, Ap  in Ap_map.items():
    Tp_sec = Tp_map[persona]
    
    # print(f"Evaluating persona={persona} with Ap={Ap:.4f}, Tp_sec={Tp_sec}, F1={F1_map[persona]:.4f}")
    # window start : [cutoff - Tp , cutoff]
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))

    # only events in the window for this persona
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
        # too few users → skip
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
            pl.col("total_gain") 
            #  * pl.col("activity_trend").clip(0.5, 2.0) 
               * pl.col("recency_weight") # optionally include recency weight
        ).alias("WA")
    ]).select(["user_id", "WA"])

    # فقط همون کاربرهای active رو نگه داریم (در عمل همون‌ها هستن)
    wa = wa.join(active_users, on="user_id", how="inner")
    Ap_old = Ap
    # Ap = Ap_old * difficulty_factor.get(persona, 1.0)
    # Ap = Ap  * ( wa["WA"].quantile(0.5) / Ap_old) 
    # پیش‌بینی: WA < Ap → churn = 1
    wa = wa.with_columns([
        (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
    ])

    # -------------------------------------
    # 5) Actual churn بعد از cutoff + grace
    # -------------------------------------
    after = (
        elog
        .filter(pl.col("persona") == persona)
        .filter((pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec)) 
                & (pl.col("timestamp") < cutoff + pl.duration(seconds=grace_sec)+  pl.duration(seconds=365 * 86400)) 
                                                                                   )
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
    # print(pdf.head(1))
    
    
    accuracy = float((y_true == y_pred).mean())
    actual_rate = float(y_true.mean())
    pred_rate = float(y_pred.mean())

    
    # Recall: TP / (TP + FN)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # Precision: TP / (TP + FP)
    FP = np.sum((y_true == 0) & (y_pred == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    rows.append({
        "persona": int(persona),
        "Tp_sec": float(Tp_sec),
        "Tp_days": int(float(Tp_sec) / 86400.0),
        "Ap": float(Ap),
        "Ap_old": float(Ap_old),
        "num_users": int(len(y_true)),
        "actual_churn_rate": actual_rate,
        "pred_churn_rate": pred_rate,
        "pred/actual" : (pred_rate / actual_rate) if actual_rate > 0 else np.nan,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0,
    })

# -----------------------------------------
# 6) Save per-persona results + overall acc
# -----------------------------------------
df_out = pd.DataFrame(rows)
df_out.to_csv("ap_tp_simple_eval.csv", index=False)
print("Saved → ap_tp_simple_eval.csv")

if len(rows) > 0:
    total_correct = sum(r["accuracy"] * r["num_users"] for r in rows)
    total_users = sum(r["num_users"] for r in rows)
    overall_acc = total_correct / total_users
    print(f"Median F1 across personas: {df_out['f1'].median():.4f}")
else:
    print("No personas had active users in the evaluation window.")
