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
grid = pd.read_csv("tp_ap_xgb_joint_search_sliding.csv")

#  Select best row per persona based on AUC
# best_params = (
#     grid.sort_values(["persona", "auc"], ascending=[True, False])
#         .groupby("persona")
#         .head(1)
# )

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
pragmatic_best_params_file = pd.read_csv("sliding_window_all_results.csv")

#  Select best row per persona based on AUC
pragmatic_best_params = (
    pragmatic_best_params_file.sort_values(["persona", "median_f1"], ascending=[True, False])
        .groupby("persona")
        .head(1)
)
Ap_map= dict(zip(pragmatic_best_params["persona"], pragmatic_best_params["Ap"]))
Tp_map= dict(zip(pragmatic_best_params["persona"], pragmatic_best_params["Tp_sec"]))

# -----------------------------------------
# 3) Evaluation cutoff & grace period
# -----------------------------------------
cutoff = pl.datetime(2016, 3, 1)      
grace_days = 180            
grace_sec = grace_days * 86400

rows = []

# -----------------------------------------
# 4) Loop over personas
# -----------------------------------------
for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

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

    # weighted activity per user (sum of gain)
    wa = (
        win
        .group_by("user_id")
        .agg(pl.col("gain").sum().alias("WA"))
    )

    # فقط همون کاربرهای active رو نگه داریم (در عمل همون‌ها هستن)
    wa = wa.join(active_users, on="user_id", how="inner")

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
        .filter(pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec))
        .group_by("user_id")
        .agg(pl.len().alias("cnt_after"))
    )

    eval_df = (
        wa.join(after, on="user_id", how="left")
          .with_columns([
              # اگر هیچ فعالیتی بعد از cutoff+grace نداشته → churn واقعی = 1
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
        "num_users": int(len(y_true)),
        "actual_churn_rate": actual_rate,
        "pred_churn_rate": pred_rate,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
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
    print(f"Overall accuracy (weighted by users): {overall_acc:.4f}")
else:
    print("No personas had active users in the evaluation window.")
