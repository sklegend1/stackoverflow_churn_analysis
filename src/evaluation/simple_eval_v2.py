import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# simple_eval_v3_era_calib.py
# ---------------------------------------------------------
# Interpretable churn evaluation with era-aware calibration
# - Single Ap/Tp per persona (from one CSV)
# - Era-aware calibration adjusts Ap (not Tp) to reduce drift
# - Rule remains: WA < Ap_used => pred_churn = 1
# ---------------------------------------------------------

import os
import polars as pl
import pandas as pd
import numpy as np

# --------------------------
# Config (env overrides)
# --------------------------
EVENTLOG_PATH = os.getenv("EVENTLOG_PATH", "../../data/processed/core_eventlog_eval_v1.parquet")
APTP_CSV_PATH = os.getenv("APTP_CSV_PATH", "../../results/best_params/tp_ap_xgb_enhanced_best_params_WA_recency.csv")

CUTOFF_YEAR = int(os.getenv("CUTOFF_YEAR", "2016"))
CUTOFF_MONTH = int(os.getenv("CUTOFF_MONTH", "1"))
CUTOFF_DAY = int(os.getenv("CUTOFF_DAY", "1"))

# Keep consistent across pipelines
GRACE_DAYS = int(os.getenv("GRACE_DAYS", "90"))
FUTURE_DAYS = int(os.getenv("FUTURE_DAYS", "365"))

# WA type: "recency" is your current default in simple_eval.py
WA_TYPE = os.getenv("WA_TYPE", "recency").lower()
RECENCY_TAU_DAYS = float(os.getenv("RECENCY_TAU_DAYS", "30.0"))
TREND_CLIP_MIN = float(os.getenv("TREND_CLIP_MIN", "0.5"))
TREND_CLIP_MAX = float(os.getenv("TREND_CLIP_MAX", "2.0"))

# Calibration mode:
#   "none"              -> no calibration
#   "era_quantile"      -> Ap_used = quantile(WA, actual_rate)  (per persona, per cutoff/era)
#   "era_quantile_clip" -> same but clipped around Ap_base for stability
CALIBRATION_MODE = os.getenv("CALIBRATION_MODE", "era_quantile_clip").lower()
CALIB_CLIP_MIN = float(os.getenv("CALIB_CLIP_MIN", "0.7"))  # Ap_used >= 0.7 * Ap_base
CALIB_CLIP_MAX = float(os.getenv("CALIB_CLIP_MAX", "1.5"))  # Ap_used <= 1.5 * Ap_base

MIN_USERS = int(os.getenv("MIN_USERS", "20"))

# --------------------------
# Era helpers (same spirit as your LSTM)
# --------------------------
def get_era(year: int) -> str:
    if 2016 <= year <= 2019:
        return "pre_covid"
    if 2020 <= year <= 2021:
        return "covid"
    return "post_llm"

def safe_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

def compute_wa(win: pl.DataFrame, cutoff: pl.Expr, window_start: pl.Expr, tp_sec: float) -> pl.DataFrame:
    """Compute WA per user for the window, matching XGB WA definitions."""
    mid_point = window_start + pl.duration(seconds=int(tp_sec / 2))

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
        .with_columns([
            ((pl.col("recent_gain") + 1.0) / (pl.col("old_gain") + 1.0)).alias("activity_trend_raw"),
            ((-pl.col("days_since_last") / RECENCY_TAU_DAYS).exp()).alias("recency_weight"),
        ])
        .with_columns([
            pl.col("activity_trend_raw").clip(TREND_CLIP_MIN, TREND_CLIP_MAX).alias("activity_trend"),
        ])
    )

    if WA_TYPE == "simple":
        return features.select(["user_id", pl.col("total_gain").alias("WA")])
    if WA_TYPE == "trend":
        return features.select(["user_id", (pl.col("total_gain") * pl.col("activity_trend")).alias("WA")])
    if WA_TYPE == "recency":
        return features.select(["user_id", (pl.col("total_gain") * pl.col("recency_weight")).alias("WA")])
    if WA_TYPE == "enhanced":
        return features.select(["user_id", (pl.col("total_gain") * pl.col("activity_trend") * pl.col("recency_weight")).alias("WA")])

    raise ValueError("WA_TYPE must be one of: simple, trend, recency, enhanced")

# --------------------------
# Load data
# --------------------------
elog = pl.read_parquet(EVENTLOG_PATH).with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

params = pd.read_csv(APTP_CSV_PATH)
if not {"persona", "Ap", "Tp_sec"}.issubset(set(params.columns)):
    raise ValueError("APTP CSV must contain columns: persona, Ap, Tp_sec")

Ap_map = dict(zip(params["persona"].astype(int), params["Ap"].astype(float)))
Tp_map = dict(zip(params["persona"].astype(int), params["Tp_sec"].astype(float)))

# --------------------------
# Evaluation setup
# --------------------------
cutoff = pl.datetime(CUTOFF_YEAR, CUTOFF_MONTH, CUTOFF_DAY)
era = get_era(CUTOFF_YEAR)

grace_sec = GRACE_DAYS * 86400
future_sec = FUTURE_DAYS * 86400

rows = []

# --------------------------
# Main loop
# --------------------------
for persona, Ap_base in Ap_map.items():
    Tp_sec = Tp_map.get(persona, None)
    if Tp_sec is None or not np.isfinite(Tp_sec) or Tp_sec <= 0:
        continue

    window_start = cutoff - pl.duration(seconds=int(Tp_sec))

    # Window events for this persona
    win = (
        elog
        .filter(pl.col("persona") == int(persona))
        .filter((pl.col("timestamp") >= window_start) & (pl.col("timestamp") <= cutoff))
    )

    active_users = win.select("user_id").unique()
    if active_users.height < MIN_USERS:
        continue

    # Compute WA
    wa = compute_wa(win, cutoff, window_start, Tp_sec).join(active_users, on="user_id", how="inner")
    if wa.height < MIN_USERS:
        continue

    # Actual churn after cutoff + grace
    after = (
        elog
        .filter(pl.col("persona") == int(persona))
        .filter(
            (pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec)) &
            (pl.col("timestamp") < cutoff + pl.duration(seconds=grace_sec + future_sec))
        )
        .group_by("user_id")
        .agg(pl.len().alias("cnt_after"))
    )

    eval_df = (
        wa.join(after, on="user_id", how="left")
          .with_columns([(pl.col("cnt_after").is_null()).cast(pl.Int8).alias("actual_churn")])
    )

    pdf = eval_df.select(["user_id", "WA", "actual_churn"]).to_pandas()
    if len(pdf) < MIN_USERS:
        continue

    y_true = pdf["actual_churn"].values.astype(int)
    wa_vals = pdf["WA"].values.astype(float)

    actual_rate = float(np.mean(y_true))

    # --------------------------
    # Era-aware calibration (per persona, per cutoff/era)
    # --------------------------
    Ap_used = float(Ap_base)
    calib_factor = 1.0

    if CALIBRATION_MODE == "none":
        pass
    elif CALIBRATION_MODE in ("era_quantile", "era_quantile_clip"):
        # If actual_rate is 0 or 1, quantile is unstable; fall back to base.
        if 0.0 < actual_rate < 1.0:
            # We want pred_rate ~= actual_rate for rule (WA < Ap_used)
            # => choose Ap_used at the actual_rate quantile of WA distribution
            q = float(np.quantile(wa_vals, actual_rate))
            if np.isfinite(q) and q > 0:
                Ap_used = q
                calib_factor = Ap_used / float(Ap_base) if Ap_base != 0 else 1.0

                if CALIBRATION_MODE == "era_quantile_clip":
                    # Clip Ap_used around Ap_base for stability across years
                    lo = float(Ap_base) * CALIB_CLIP_MIN
                    hi = float(Ap_base) * CALIB_CLIP_MAX
                    Ap_used = float(np.clip(Ap_used, lo, hi))
                    calib_factor = Ap_used / float(Ap_base) if Ap_base != 0 else 1.0
    else:
        raise ValueError("CALIBRATION_MODE must be one of: none, era_quantile, era_quantile_clip")

    # Prediction
    y_pred = (wa_vals < Ap_used).astype(int)

    # Metrics
    accuracy = float(np.mean(y_true == y_pred))
    pred_rate = float(np.mean(y_pred))

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = safe_f1(precision, recall)

    rows.append({
        "persona": int(persona),
        "cutoff_date": f"{CUTOFF_YEAR:04d}-{CUTOFF_MONTH:02d}-{CUTOFF_DAY:02d}",
        "era": era,
        "WA_type": WA_TYPE,
        "Tp_sec": float(Tp_sec),
        "Tp_days": int(float(Tp_sec) / 86400.0),
        "grace_days": int(GRACE_DAYS),
        "Ap_base": float(Ap_base),
        "Ap_used": float(Ap_used),
        "calibration_mode": CALIBRATION_MODE,
        "calib_factor": float(calib_factor),
        "num_users": int(len(y_true)),
        "actual_churn_rate": float(actual_rate),
        "pred_churn_rate": float(pred_rate),
        "pred/actual": (pred_rate / actual_rate) if actual_rate > 0 else np.nan,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "wa_p10": float(np.quantile(wa_vals, 0.10)),
        "wa_p50": float(np.quantile(wa_vals, 0.50)),
        "wa_p90": float(np.quantile(wa_vals, 0.90)),
    })

# --------------------------
# Save results
# --------------------------
df_out = pd.DataFrame(rows)
df_out.to_csv("ap_tp_simple_eval.csv", index=False)
print("Saved → ap_tp_simple_eval.csv")

if len(df_out) > 0:
    print(f"Era: {era} | Cutoff: {df_out['cutoff_date'].iloc[0]}")
    print(f"Median F1 across personas: {df_out['f1'].median():.4f}")
else:
    print("No personas had enough active users in the evaluation window.")
