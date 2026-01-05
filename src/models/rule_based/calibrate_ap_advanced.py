import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

"""
ADVANCED Ap CALIBRATION METHODS

Goal: Synchronize pred_churn_rate with actual_churn_rate

Methods implemented:
1. Simple Ratio Scaling (baseline)
2. Platt Scaling (logistic calibration)
3. Isotonic Regression (non-parametric calibration)
4. Beta Calibration (handles both over/under-confidence)
5. Temperature Scaling (neural network calibration)
6. Quantile Matching (distribution alignment)
7. Empirical Bayes Calibration (persona-aware)

Each method learns Ap_calibrated from training data.
"""

print("=" * 80)
print("ADVANCED Ap CALIBRATION - Multiple Methods")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

# Load event log
elog_train = pl.read_parquet("../../../data/processed/core_eventlog_train_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

elog_eval = pl.read_parquet("../../../data/processed/core_eventlog_train_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

# Load base parameters (uncalibrated)
base_params = pd.read_csv("../../../results/best_params/tp_ap_xgb_enhanced_best_params_WA_enhanced.csv")

print(f"\nTraining events: {elog_train.height:,}")
print(f"Evaluation events: {elog_eval.height:,}")
print(f"Personas to calibrate: {len(base_params)}")

# Configuration
GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400
CHURN_CHECK_DAYS = 365
CHURN_CHECK_SEC = CHURN_CHECK_DAYS * 86400

# Calibration cutoffs (use training data)
CALIB_CUTOFFS = [
    pl.datetime(2016, 1, 1),
    pl.datetime(2017, 1, 1),
    pl.datetime(2018, 1, 1),
    pl.datetime(2019, 1, 1),
    pl.datetime(2020, 1, 1),
    pl.datetime(2021, 1, 1),
    pl.datetime(2022, 1, 1),
    pl.datetime(2023, 1, 1),
]

print(f"\nCalibration cutoffs: {len(CALIB_CUTOFFS)}")
print(f"Grace period: {GRACE_DAYS} days")
print(f"Churn check window: {CHURN_CHECK_DAYS} days")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_wa_enhanced(elog_window, cutoff, Tp_sec):
    """Calculate WA_enhanced for users in window."""
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))
    mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

    win = elog_window.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    )

    if win.height == 0:
        return None

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

    features = features.with_columns([
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0) 
            * pl.col("recency_weight") # Optionally include recency weight
        ).alias("WA_enhanced")
    ])

    return features.select(["user_id", "WA_enhanced"])

def get_calibration_data(elog, persona, Tp_sec, Ap_base, cutoff):
    """Get (WA_scores, actual_churn_labels) for calibration."""
    elog_persona = elog.filter(pl.col("persona") == persona)

    # Calculate WA
    wa_features = calculate_wa_enhanced(elog_persona, cutoff, Tp_sec)

    if wa_features is None or wa_features.height < 20:
        return None, None

    # Get actual churn labels
    future_start = cutoff + pl.duration(seconds=GRACE_SEC)
    future_end = future_start + pl.duration(seconds=CHURN_CHECK_SEC)

    future_activity = (
        elog_persona
        .filter(
            (pl.col("timestamp") > future_start) &
            (pl.col("timestamp") <= future_end)
        )
        .group_by("user_id")
        .agg(pl.len().alias("future_events"))
    )

    eval_df = (
        wa_features
        .join(future_activity, on="user_id", how="left")
        .with_columns([
            (pl.col("future_events").is_null() | (pl.col("future_events") == 0))
            .cast(pl.Int8).alias("actual_churn")
        ])
    )

    pdf = eval_df.to_pandas()

    return pdf["WA_enhanced"].values, pdf["actual_churn"].values

# ============================================================================
# CALIBRATION METHOD 1: Simple Ratio Scaling
# ============================================================================

def calibrate_ratio_scaling(WA_scores, y_true, Ap_base):
    """
    Ap_calibrated = Ap_base * (actual_churn_rate / pred_churn_rate)

    Simple but effective baseline.
    """
    y_pred_base = (WA_scores < Ap_base).astype(int)

    actual_rate = y_true.mean()
    pred_rate = y_pred_base.mean()

    if pred_rate == 0:
        return Ap_base

    # Scale Ap to match rates
    Ap_calibrated = Ap_base * (actual_rate / pred_rate)

    return Ap_calibrated

# ============================================================================
# CALIBRATION METHOD 2: Platt Scaling (Logistic)
# ============================================================================

def calibrate_platt_scaling(WA_scores, y_true, Ap_base):
    """
    Learn logistic function: P(churn) = 1 / (1 + exp(a*WA + b))
    Then find Ap where P(churn) = 0.5
    """
    # Reshape for sklearn
    X = WA_scores.reshape(-1, 1)

    # Fit logistic regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y_true)

    # Find WA threshold where P(churn) = 0.5
    # log(odds) = a*WA + b
    # odds = 1 when P = 0.5
    # 0 = a*WA + b
    # WA = -b/a

    a = lr.coef_[0][0]
    b = lr.intercept_[0]

    if a == 0:
        return Ap_base

    Ap_calibrated = -b / a

    # Ensure reasonable range
    Ap_calibrated = np.clip(Ap_calibrated, WA_scores.min(), WA_scores.max())

    return Ap_calibrated

# ============================================================================
# CALIBRATION METHOD 3: Isotonic Regression
# ============================================================================

def calibrate_isotonic(WA_scores, y_true, Ap_base):
    """
    Non-parametric calibration that learns monotonic mapping.
    More flexible than Platt scaling.
    """
    # Fit isotonic regression
    iso = IsotonicRegression(out_of_bounds='clip')

    # Map WA → calibrated probability
    calibrated_probs = iso.fit_transform(WA_scores, y_true)

    # Find WA where calibrated_prob ≈ 0.5
    # Sort by WA
    sorted_idx = np.argsort(WA_scores)
    sorted_WA = WA_scores[sorted_idx]
    sorted_probs = calibrated_probs[sorted_idx]

    # Find threshold closest to 0.5
    target_idx = np.argmin(np.abs(sorted_probs - 0.5))
    Ap_calibrated = sorted_WA[target_idx]

    return Ap_calibrated

# ============================================================================
# CALIBRATION METHOD 4: Beta Calibration
# ============================================================================

def calibrate_beta(WA_scores, y_true, Ap_base):
    """
    Beta calibration: more flexible than Platt, handles over/under-confidence.

    P_calibrated = 1 / (1 + exp(a + b*log(WA/(1-WA))))

    Reduces to Platt when b=1.
    """
    # Transform WA to (0,1) range
    WA_min, WA_max = WA_scores.min(), WA_scores.max()
    WA_normalized = (WA_scores - WA_min) / (WA_max - WA_min + 1e-8)
    WA_normalized = np.clip(WA_normalized, 1e-8, 1 - 1e-8)

    # Logit transform
    WA_logit = np.log(WA_normalized / (1 - WA_normalized))

    X = np.column_stack([np.ones_like(WA_logit), WA_logit])

    # Fit logistic regression on logit-transformed features
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y_true)

    # Find WA where P(churn) = 0.5
    # a + b*logit(WA) = 0
    # logit(WA) = -a/b
    # WA = sigmoid(-a/b)

    a = lr.intercept_[0]
    b = lr.coef_[0][1]

    if b == 0:
        return Ap_base

    logit_threshold = -a / b
    WA_threshold_norm = 1 / (1 + np.exp(-logit_threshold))

    # Transform back to original scale
    Ap_calibrated = WA_threshold_norm * (WA_max - WA_min) + WA_min

    return Ap_calibrated

# ============================================================================
# CALIBRATION METHOD 5: Temperature Scaling
# ============================================================================

def calibrate_temperature(WA_scores, y_true, Ap_base):
    """
    Learn temperature T such that:
    P_calibrated = sigmoid((Ap_base - WA) / T)

    Commonly used in neural network calibration.
    """
    # Define loss function: negative log-likelihood
    def nll_loss(T):
        if T <= 0:
            return 1e10

        # Scores: how far is WA from Ap_base
        scores = (Ap_base - WA_scores) / T
        probs = 1 / (1 + np.exp(-scores))
        probs = np.clip(probs, 1e-8, 1 - 1e-8)

        # Cross-entropy loss
        loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
        return loss

    # Optimize temperature
    result = minimize_scalar(nll_loss, bounds=(0.01, 10), method='bounded')
    T_opt = result.x

    # With optimal temperature, find new Ap where P = 0.5
    # sigmoid((Ap - WA) / T) = 0.5
    # (Ap - WA) / T = 0
    # Ap = WA at median

    # Actually, we keep Ap_base but adjust scores
    # For simplicity, return scaled Ap
    Ap_calibrated = Ap_base * T_opt

    return Ap_calibrated

# ============================================================================
# CALIBRATION METHOD 6: Quantile Matching
# ============================================================================

def calibrate_quantile_matching(WA_scores, y_true, Ap_base):
    """
    Match quantiles: find Ap such that fraction of users below Ap
    equals actual churn rate.

    If actual churn = 30%, set Ap at 30th percentile of WA distribution.
    """
    actual_churn_rate = y_true.mean()

    # Find WA value at (1 - churn_rate) percentile
    # If churn_rate = 0.3, we want 30% below threshold → 70th percentile
    percentile = (1 - actual_churn_rate) * 100

    Ap_calibrated = np.percentile(WA_scores, percentile)

    return Ap_calibrated

# ============================================================================
# CALIBRATION METHOD 7: Empirical Bayes
# ============================================================================

def calibrate_empirical_bayes(WA_scores, y_true, Ap_base, global_churn_rate):
    """
    Shrink Ap towards global mean weighted by confidence.

    Ap_calibrated = w * Ap_optimal + (1-w) * Ap_base

    where w depends on sample size (more data → less shrinkage).
    """
    n = len(WA_scores)

    # Find optimal Ap for this sample
    # Try grid of values
    best_f1 = 0
    Ap_optimal = Ap_base

    for percentile in range(10, 91, 5):
        Ap_candidate = np.percentile(WA_scores, percentile)
        y_pred = (WA_scores < Ap_candidate).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            Ap_optimal = Ap_candidate

    # Shrinkage weight: more data → less shrinkage
    w = n / (n + 100)  # Adjust 100 based on domain

    Ap_calibrated = w * Ap_optimal + (1 - w) * Ap_base

    return Ap_calibrated

# ============================================================================
# MAIN CALIBRATION LOOP
# ============================================================================

calibration_methods = {
    'ratio_scaling': calibrate_ratio_scaling,
    'platt_scaling': calibrate_platt_scaling,
    'isotonic': calibrate_isotonic,
    'beta': calibrate_beta,
    'temperature': calibrate_temperature,
    'quantile_matching': calibrate_quantile_matching,
    # 'empirical_bayes': calibrate_empirical_bayes,  # Needs global rate
}

print("\n" + "=" * 80)
print("CALIBRATING Ap FOR EACH PERSONA")
print("=" * 80)

all_calibrated_params = []

for idx, row in base_params.iterrows():
    persona = int(row['persona'])
    Tp_sec = float(row['Tp_sec'])
    Ap_base = float(row['Ap'])

    print(f"\nPersona {persona}:")
    print(f"  Base Ap: {Ap_base:.4f}, Tp: {Tp_sec/86400:.0f} days")

    # Collect calibration data from multiple cutoffs
    all_WA = []
    all_y = []

    for cutoff in CALIB_CUTOFFS:
        WA_scores, y_true = get_calibration_data(
            elog_train, persona, Tp_sec, Ap_base, cutoff
        )

        if WA_scores is not None:
            all_WA.append(WA_scores)
            all_y.append(y_true)

    if len(all_WA) == 0:
        print("  ⚠ No calibration data available, keeping base Ap")
        for method_name in calibration_methods.keys():
            all_calibrated_params.append({
                'persona': persona,
                'method': method_name,
                'Tp_sec': Tp_sec,
                'Tp_days': int(Tp_sec / 86400),
                'Ap_base': Ap_base,
                'Ap_calibrated': Ap_base,
            })
        continue

    # Concatenate all calibration data
    WA_combined = np.concatenate(all_WA)
    y_combined = np.concatenate(all_y)

    print(f"  Calibration samples: {len(y_combined)}, Churn rate: {y_combined.mean():.2%}")

    # Apply each calibration method
    for method_name, calibrate_func in calibration_methods.items():
        try:
            if method_name == 'empirical_bayes':
                Ap_calibrated = calibrate_func(
                    WA_combined, y_combined, Ap_base, global_churn_rate=0.35
                )
            else:
                Ap_calibrated = calibrate_func(WA_combined, y_combined, Ap_base)

            # Ensure reasonable bounds
            Ap_calibrated = np.clip(Ap_calibrated, 0.01, WA_combined.max() * 2)

            print(f"    {method_name:20s}: Ap = {Ap_calibrated:.4f} (change: {(Ap_calibrated/Ap_base - 1)*100:+.1f}%)")

            all_calibrated_params.append({
                'persona': persona,
                'method': method_name,
                'Tp_sec': Tp_sec,
                'Tp_days': int(Tp_sec / 86400),
                'Ap_base': Ap_base,
                'Ap_calibrated': Ap_calibrated,
            })
        except Exception as e:
            print(f"    {method_name:20s}: FAILED ({str(e)})")
            all_calibrated_params.append({
                'persona': persona,
                'method': method_name,
                'Tp_sec': Tp_sec,
                'Tp_days': int(Tp_sec / 86400),
                'Ap_base': Ap_base,
                'Ap_calibrated': Ap_base,
            })

# ============================================================================
# SAVE CALIBRATED PARAMETERS
# ============================================================================

df_calibrated = pd.DataFrame(all_calibrated_params)
df_calibrated.to_csv("../../../results/calibration/calibrated_ap_all_methods.csv", index=False)

print("\n" + "=" * 80)
print(f"Saved calibrated params → calibrated_ap_all_methods.csv ({len(df_calibrated)} rows)")
print("=" * 80)

# ============================================================================
# EVALUATE ALL METHODS ON EVAL SET
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING CALIBRATION METHODS ON EVAL SET")
print("=" * 80)

# Evaluation cutoffs - use multiple time periods to account for behavior changes
EVAL_CUTOFFS = [
    pl.datetime(2016, 7, 1),
    pl.datetime(2017, 7, 1),
    pl.datetime(2018, 7, 1),
    pl.datetime(2019, 7, 1),
    pl.datetime(2020, 7, 1),
    pl.datetime(2021, 7, 1),
    pl.datetime(2022, 7, 1),
    pl.datetime(2023, 7, 1),
]

print(f"Evaluating on {len(EVAL_CUTOFFS)} time periods to account for temporal changes")

evaluation_results = []

for idx, row in df_calibrated.iterrows():
    persona = row['persona']
    method = row['method']
    Tp_sec = row['Tp_sec']
    Ap_calibrated = row['Ap_calibrated']

    # Evaluate on multiple cutoffs to get robust metrics
    for eval_cutoff in EVAL_CUTOFFS:
        # Get eval data
        WA_scores, y_true = get_calibration_data(
            elog_eval, persona, Tp_sec, Ap_calibrated, eval_cutoff
        )

        if WA_scores is None:
            continue

        # Predict with calibrated Ap
        y_pred = (WA_scores < Ap_calibrated).astype(int)

        # Calculate metrics
        actual_churn_rate = y_true.mean()
        pred_churn_rate = y_pred.mean()

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / len(y_true)

        rate_ratio = pred_churn_rate / actual_churn_rate if actual_churn_rate > 0 else np.nan

        evaluation_results.append({
            'persona': persona,
            'method': method,
            'eval_cutoff': str(eval_cutoff),
            'Tp_sec': Tp_sec,
            'Tp_days': int(Tp_sec / 86400),
            'Ap': Ap_calibrated,
            'num_users': len(y_true),
            'actual_churn_rate': actual_churn_rate,
            'pred_churn_rate': pred_churn_rate,
            'rate_ratio': rate_ratio,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
        })

df_eval = pd.DataFrame(evaluation_results)
df_eval.to_csv("../../../results/calibration_evaluation_results.csv", index=False)

print(f"\nSaved evaluation → calibration_evaluation_results.csv")

# ============================================================================
# COMPARE METHODS
# ============================================================================

print("\n" + "=" * 80)
print("METHOD COMPARISON")
print("=" * 80)

method_comparison = df_eval.groupby('method').agg({
    'rate_ratio': 'mean',
    'f1': 'mean',
    'precision': 'mean',
    'recall': 'mean',
    'persona': 'count'
}).round(3)

method_comparison.columns = ['Avg Rate Ratio', 'Avg F1', 'Avg Precision', 'Avg Recall', 'N Personas']
method_comparison = method_comparison.sort_values('Avg F1', ascending=False)

print(method_comparison.to_string())

print("\n" + "=" * 80)
print("BEST METHOD (by F1):")
best_method = method_comparison.index[0]
print(f"  {best_method}")
print(f"  F1: {method_comparison.loc[best_method, 'Avg F1']:.3f}")
print(f"  Rate Ratio: {method_comparison.loc[best_method, 'Avg Rate Ratio']:.3f} (closer to 1.0 is better)")
print("=" * 80)

# Save best method params
df_best_method = df_calibrated[df_calibrated['method'] == best_method].copy()
df_best_method = df_best_method.rename(columns={'Ap_calibrated': 'Ap'})
df_best_method = df_best_method[['persona', 'Tp_sec', 'Tp_days', 'Ap']]
df_best_method.to_csv(f"calibrated_ap_{best_method}.csv", index=False)

print(f"\nBest method params saved → calibrated_ap_{best_method}.csv")

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)