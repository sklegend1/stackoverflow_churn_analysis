import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
"""
Comprehensive Analysis: LSTM Performance Across All Years (2016-2023)

Compares LSTM vs XGBoost temporal drift handling across the entire timeline.
"""

import pandas as pd
import numpy as np

print('='*80)
print('LSTM PERFORMANCE ANALYSIS: 2016-2023 (ALL YEARS)')
print('='*80)

# Load LSTM results (all years)
lstm_all = pd.read_csv('../../../results/evaluations/lstm_eval_2016_to_2023.csv')

# Load XGBoost baseline (from temporal analysis)
try:
    xgb_baseline = pd.read_csv('../../../results/temporal_baseline_results.csv')
    has_baseline = True
except FileNotFoundError:
    print('\n⚠️  Warning: XGBoost baseline not found')
    print('    Run temporal_drift_analysis.py to generate baseline')
    has_baseline = False

print('\n📊 YEAR-BY-YEAR LSTM PERFORMANCE')
print('-'*80)
print(f'{"Year":<8} {"Personas":<10} {"Users":<10} {"Median F1":<12} {"Mean F1":<12} {"Std F1":<10} {"Rate Ratio":<12}')
print('-'*80)

year_summary = []

for year in sorted(lstm_all['year'].unique()):
    df_yr = lstm_all[lstm_all['year'] == year]

    median_f1 = df_yr['f1'].median()
    mean_f1 = df_yr['f1'].mean()
    std_f1 = df_yr['f1'].std()
    rate_ratio = (df_yr['pred_rate'] / df_yr['actual_rate']).median()

    print(f'{year:<8} {len(df_yr):<10} {df_yr["num_users"].sum():<10,} {median_f1:<12.4f} {mean_f1:<12.4f} {std_f1:<10.4f} {rate_ratio:<12.4f}')

    year_summary.append({
        'year': year,
        'median_f1_lstm': median_f1,
        'mean_f1_lstm': mean_f1,
        'std_f1_lstm': std_f1,
        'personas': len(df_yr),
        'users': df_yr['num_users'].sum(),
        'rate_ratio': rate_ratio
    })

df_summary = pd.DataFrame(year_summary)

# Calculate trends
print('\n\n📈 TEMPORAL TRENDS')
print('-'*80)

f1_2016 = df_summary[df_summary['year'] == 2016]['median_f1_lstm'].values[0]
f1_2023 = df_summary[df_summary['year'] == 2023]['median_f1_lstm'].values[0]
f1_improvement = f1_2023 - f1_2016
f1_improvement_pct = (f1_improvement / f1_2016) * 100

print(f'F1 Evolution (2016 → 2023):')
print(f'  2016: {f1_2016:.4f}')
print(f'  2023: {f1_2023:.4f}')
print(f'  Improvement: +{f1_improvement:.4f} ({f1_improvement_pct:+.1f}%)')
print(f'  Trend: {"Improving" if f1_improvement > 0 else "Declining"}')

# Stability over time
print(f'\nStability Over Time:')
print(f'  Avg std (2016-2019): {df_summary[df_summary["year"] <= 2019]["std_f1_lstm"].mean():.4f}')
print(f'  Avg std (2020-2023): {df_summary[df_summary["year"] >= 2020]["std_f1_lstm"].mean():.4f}')

# Rate calibration
print(f'\nRate Calibration:')
print(f'  Avg rate ratio (2016-2019): {df_summary[df_summary["year"] <= 2019]["rate_ratio"].mean():.4f}')
print(f'  Avg rate ratio (2020-2023): {df_summary[df_summary["year"] >= 2020]["rate_ratio"].mean():.4f}')
print(f'  Overall avg: {df_summary["rate_ratio"].mean():.4f}')

# Compare with XGBoost if available
if has_baseline:
    print('\n\n🆚 LSTM vs XGBoost BASELINE (Year-by-Year)')
    print('-'*80)
    print(f'{"Year":<8} {"LSTM F1":<12} {"XGBoost F1":<12} {"Improvement":<15} {"% Gain":<10}')
    print('-'*80)

    comparison = []

    for year in sorted(df_summary['year'].unique()):
        lstm_f1 = df_summary[df_summary['year'] == year]['median_f1_lstm'].values[0]

        xgb_yr = xgb_baseline[xgb_baseline['cutoff_year'] == year]
        if len(xgb_yr) > 0:
            xgb_f1 = xgb_yr['f1'].median()
            improvement = lstm_f1 - xgb_f1
            pct_gain = (improvement / xgb_f1) * 100 if xgb_f1 > 0 else 0

            print(f'{year:<8} {lstm_f1:<12.4f} {xgb_f1:<12.4f} {improvement:+<15.4f} {pct_gain:+<10.1f}%')

            comparison.append({
                'year': year,
                'lstm_f1': lstm_f1,
                'xgb_f1': xgb_f1,
                'improvement': improvement,
                'pct_gain': pct_gain
            })
        else:
            print(f'{year:<8} {lstm_f1:<12.4f} {"N/A":<12} {"N/A":<15} {"N/A":<10}')

    if comparison:
        df_comp = pd.DataFrame(comparison)

        print('\n' + '='*80)
        print('OVERALL COMPARISON SUMMARY')
        print('='*80)

        print(f'\nAverage Improvements:')
        print(f'  Mean F1 improvement: +{df_comp["improvement"].mean():.4f}')
        print(f'  Median F1 improvement: +{df_comp["improvement"].median():.4f}')
        print(f'  Average % gain: +{df_comp["pct_gain"].mean():.1f}%')

        print(f'\nBest Year (LSTM advantage):')
        best_year = df_comp.loc[df_comp['pct_gain'].idxmax()]
        print(f'  Year: {int(best_year["year"])}')
        print(f'  LSTM F1: {best_year["lstm_f1"]:.4f}')
        print(f'  XGBoost F1: {best_year["xgb_f1"]:.4f}')
        print(f'  Gain: +{best_year["pct_gain"]:.1f}%')

        print(f'\nWorst Year (smallest advantage):')
        worst_year = df_comp.loc[df_comp['pct_gain'].idxmin()]
        print(f'  Year: {int(worst_year["year"])}')
        print(f'  LSTM F1: {worst_year["lstm_f1"]:.4f}')
        print(f'  XGBoost F1: {worst_year["xgb_f1"]:.4f}')
        print(f'  Gain: +{worst_year["pct_gain"]:.1f}%')

# Identify problematic years
print('\n\n⚠️  PROBLEM YEARS ANALYSIS')
print('-'*80)

low_threshold = 0.65
problem_years = df_summary[df_summary['median_f1_lstm'] < low_threshold]

if len(problem_years) > 0:
    print(f'Years with F1 < {low_threshold}:')
    for _, row in problem_years.iterrows():
        print(f'  {int(row["year"])}: F1 = {row["median_f1_lstm"]:.4f} (needs attention)')
else:
    print(f'✓ No problem years! All years have F1 ≥ {low_threshold}')

# Era-based analysis
print('\n\n🌍 ERA-BASED ANALYSIS')
print('-'*80)

def get_era(year):
    if year <= 2019:
        return 'pre_covid'
    elif year <= 2021:
        return 'covid'
    else:
        return 'post_llm'

df_summary['era'] = df_summary['year'].apply(get_era)

era_stats = df_summary.groupby('era').agg({
    'median_f1_lstm': ['mean', 'std'],
    'rate_ratio': 'mean',
    'users': 'sum'
}).round(4)

print(era_stats)

# Best and worst performing personas across all years
print('\n\n🏆 PERSONA PERFORMANCE (Across All Years)')
print('-'*80)

persona_avg = lstm_all.groupby('persona').agg({
    'f1': ['mean', 'std', 'min', 'max'],
    'num_users': 'sum'
}).round(4)

persona_avg.columns = ['_'.join(col).strip() for col in persona_avg.columns.values]
persona_avg = persona_avg.sort_values('f1_mean', ascending=False)

print('\nTop 5 Best Performing Personas:')
print(persona_avg.head(5)[['f1_mean', 'f1_std', 'f1_min', 'f1_max']])

print('\nBottom 5 Personas (Need Attention):')
print(persona_avg.tail(5)[['f1_mean', 'f1_std', 'f1_min', 'f1_max']])

# Save detailed summary
df_summary.to_csv('../../../results/lstm_year_by_year_summary.csv', index=False)
print(f'\n\n✓ Detailed summary saved to: lstm_year_by_year_summary.csv')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)

# Final verdict
print('\n🎯 FINAL VERDICT')
print('-'*80)

avg_f1 = df_summary['median_f1_lstm'].mean()
min_f1 = df_summary['median_f1_lstm'].min()
max_f1 = df_summary['median_f1_lstm'].max()
avg_stability = df_summary['std_f1_lstm'].mean()
avg_rate_ratio = df_summary['rate_ratio'].mean()

print(f'Average F1 (2016-2023): {avg_f1:.4f}')
print(f'F1 Range: [{min_f1:.4f}, {max_f1:.4f}]')
print(f'Average Stability: {avg_stability:.4f}')
print(f'Average Rate Ratio: {avg_rate_ratio:.4f}')

if avg_f1 > 0.75 and avg_stability < 0.10 and avg_rate_ratio < 1.3:
    print('\n✅ LSTM MODEL IS PRODUCTION READY!')
    print('   - High F1 across all years')
    print('   - Good stability')
    print('   - Acceptable rate calibration')
elif avg_f1 > 0.65:
    print('\n⚠️  LSTM MODEL IS ACCEPTABLE (with caveats)')
    print('   - Decent F1 but may need monitoring')
    print('   - Check problem years and personas')
else:
    print('\n❌ LSTM MODEL NEEDS IMPROVEMENT')
    print('   - F1 too low for production')
    print('   - Consider hyperparameter tuning or architecture changes')
