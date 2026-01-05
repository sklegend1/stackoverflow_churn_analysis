import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import pandas as pd
import numpy as np

"""
Adjust Ap values from ratio_scaling to reduce over-prediction.

Problem: ratio_scaling gives good F1 (0.555) but bad rate_ratio (1.95)
Solution: Apply correction factor to Ap values to bring rate_ratio closer to 1.0

Logic:
- If we're over-predicting by 2x, we need to make the threshold stricter
- Increasing Ap means fewer users classified as churn (since churn = WA < Ap)
- Apply gentle correction to avoid overcorrecting
"""

print("=" * 80)
print("ADJUSTING Ap VALUES TO REDUCE OVER-PREDICTION")
print("=" * 80)

# Load ratio_scaling parameters
ratio_scaling = pd.read_csv("../../../results/calibration/calibrated_ap_ratio_scaling.csv")
print(f"\nLoaded {len(ratio_scaling)} personas from ratio_scaling")

# Load current evaluation results to calculate average rate_ratio
eval_results = pd.read_csv("ap_tp_simple_eval.csv")
avg_rate_ratio = eval_results['pred/actual'].mean()

print(f"\nCurrent average pred/actual ratio: {avg_rate_ratio:.3f}")
print(f"Target ratio: 1.0")

# Calculate correction factor
# If pred/actual = 2.0, we want to reduce predictions by ~2x
# Increasing Ap by sqrt(rate_ratio) is a gentle correction
correction_factor = np.sqrt(avg_rate_ratio)

print(f"\nApplying correction factor: {correction_factor:.3f}")
print(f"This will multiply all Ap values by {correction_factor:.3f}")

# Apply correction
ratio_scaling['Ap_original'] = ratio_scaling['Ap']
ratio_scaling['Ap'] = ratio_scaling['Ap'] * correction_factor
ratio_scaling['correction_factor'] = correction_factor

# Calculate percentage change
ratio_scaling['pct_change'] = ((ratio_scaling['Ap'] / ratio_scaling['Ap_original']) - 1) * 100

# Save adjusted parameters
output_file = "../../../results/calibration/calibrated_ap_ratio_scaling_adjusted.csv"
ratio_scaling.to_csv(output_file, index=False)

print(f"\n✓ Saved adjusted parameters → {output_file}")
print("\nSample adjustments:")
print(ratio_scaling[['persona', 'Tp_days', 'Ap_original', 'Ap', 'pct_change']].head(10))

print("\n" + "=" * 80)
print("NEXT STEP: Run simple_eval.py with adjusted parameters")
print("Update simple_eval.py line 70 to load '../../../results/calibration/calibrated_ap_ratio_scaling_adjusted.csv'")
print("=" * 80)
