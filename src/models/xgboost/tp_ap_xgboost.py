import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    accuracy_score, roc_curve
)
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

# -------------------------------------------------------
# Load core event log
# -------------------------------------------------------
elog = pl.read_parquet("../../../data/processed/core_eventlog_train_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])
elog = elog.with_columns([
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64)
])

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
# sliding training cutoffs (you can adjust years)
# -------------------------------------------------------

# cutoff_years = list(range(2016, 2023))

cutoff_years = []
for y in range(2016, 2021):
    # cutoff_years.append((y, 1))
     cutoff_years.append((y, 2))
    # cutoff_years.append((y, 3))

# Note: update the loop to unpack (year, month), e.g.:
# for year, month in cutoff_years:
#     cutoff = pl.datetime(year, month, 1)
#     cutoff_str = f"{year:04d}-{month:02d}-01"
grace_period_sec = 90 * 86400          # 180 days

# -------------------------------------------------------
# alpha values to test (Tp = alpha * p95_gap_sec)
# -------------------------------------------------------
alpha_values = np.arange(3, 10,1).tolist()  # [1, 5, 9, 13]

rows = []

# -------------------------------------------------------
# Begin search with sliding window
# -------------------------------------------------------
for persona, p95_gap in p95_map.items():
    # filter logs for this persona once
    elog_p = elog.filter(pl.col("persona") == persona)

    if elog_p.height == 0 or not np.isfinite(p95_gap):
        continue

    for alpha in alpha_values:
        Tp_sec = float(alpha * p95_gap)

        for year,grace_coef in cutoff_years:
            # cutoff as literal datetime expression
            cutoff = pl.datetime(year,1, 1)

            # training window: [cutoff - Tp, cutoff]
            window_start = cutoff - pl.duration(seconds=int(Tp_sec))

            # events inside training window
            win = (
                elog_p
                .filter(
                    (pl.col("timestamp") >= window_start) &
                    (pl.col("timestamp") <= cutoff)
                )
            )

            # users with activity in window
            active_users = win.select("user_id").unique()["user_id"].to_list()
            if len(active_users) < 20:
                # too few users → skip
                continue

            # weighted activity per user (sum of gain)
            wa = (
                win
                .group_by("user_id")
                .agg(pl.col("gain").sum().alias("WA"))
            )

            # future activity after cutoff + grace period
            future = (
                elog_p
                .filter((pl.col("timestamp") > (cutoff + pl.duration(seconds=grace_period_sec*grace_coef))) & (pl.col("timestamp") < (cutoff + pl.duration(seconds=366 * 86400)) + pl.duration(seconds=grace_period_sec*grace_coef) ))
                .group_by("user_id")
                .agg(pl.len().alias("cnt"))
            )

            # join WA with future to build labels
            actual = (
                wa.join(future, on="user_id", how="left")
                  .with_columns([
                      (pl.col("cnt").is_null()).cast(pl.Int8).alias("actual_churn")
                  ])
            )

            pdf = actual.to_pandas()
            y = pdf["actual_churn"].values
            X = pdf["WA"].values.reshape(-1, 1)

            # need both classes
            if np.unique(y).size < 2:
                continue

            # ---------------------------------------------------
            # Train XGBoost on scalar WA
            # ---------------------------------------------------
            model = XGBClassifier(
                max_depth=3,
                learning_rate=0.05,
                n_estimators=60,
                subsample=0.9,
                colsample_bytree=0.8,
                eval_metric="logloss",
            )
            model.fit(X, y)

            pred_prob = model.predict_proba(X)[:, 1]

            # ---------------------------------------------------
            # Find optimal Ap using Youden's J (TPR - FPR)
            # ---------------------------------------------------
            # fpr, tpr, thresh = roc_curve(y, pred_prob)
            # j_scores = tpr - fpr
            # best_idx = np.argmax(j_scores)
            # Ap = float(thresh[best_idx])

            # ---------------------------------------------------
            # Alternatively, set Ap to achieve n% recall
            precision, recall, thresholds = precision_recall_curve(y, pred_prob)

            target_recall = 0.8
            target_precision = 0.65

            # thresholds correspond to precision[1:], recall[1:]
            if thresholds.size == 0:
                # no thresholds (all predictions identical) — pick a safe default
                Ap = 0.5
            else:
                cond = (recall[1:] >= target_recall) & (precision[1:] >= target_precision)
                idxs = np.where(cond)[0]
                if idxs.size:
                    Ap = float(thresholds[idxs[0]])   # first threshold meeting both targets
                else:
                    # fallback: among thresholds that meet recall, pick highest precision; if none, pick threshold with max recall
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
            Ap_Calibrated = Ap * ((y.mean()  ) / pred_rate) if pred_rate > 0 else Ap
            Ap = Ap_Calibrated
            if Ap < 0.05 :
                continue
            #pred = (pred_prob >= Ap).astype(int)
            # Metrics
            auc = float(roc_auc_score(y, pred_prob))
            acc = float(accuracy_score(y, pred))
            prec = float(precision_score(y, pred, zero_division=0))
            rec = float(recall_score(y, pred, zero_division=0))
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            # save row (cutoff in ISO format for readability)
            cutoff_str = f"{year:04d}-{'01'}-01"
            if Tp_sec > (86400 * 270) or Tp_sec < (86400 * 60):
                # skip Tp > 1 year
                continue

            rows.append({
                "persona": int(persona),
                "cutoff_date": cutoff_str,
                "grace_period_days": (grace_period_sec * grace_coef) // 86400,
                "alpha": float(alpha),
                "Tp_sec": Tp_sec,
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
            print(f"Done persona={persona} year={year} alpha={alpha}")

# -------------------------------------------------------
# Save results
# -------------------------------------------------------
df_out = pd.DataFrame(rows)
df_out.sort_values(["persona", "cutoff_date", "auc"], inplace=True, ascending=[True,True, False])
df_out.to_csv("../../../results/best_params/tp_ap_xgb_joint_search_sliding.csv", index=False)

print("Saved → tp_ap_xgb_joint_search_sliding.csv")
