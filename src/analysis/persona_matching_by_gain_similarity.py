import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Persona Classification via Activity Gain Similarity

Idea: Match users to personas by comparing their activity patterns
to the XGBoost feature importance (gain) patterns for each persona.

Input:
- core_activity_per_persona_named.csv (persona gain patterns)
- feature_matrix_v2.parquet (user activity counts)

Algorithm:
1. For each persona, create a gain vector for all actions
2. For each user, create an activity count vector
3. Calculate similarity between user activity and each persona's gain pattern
4. Assign user to persona with highest similarity
"""

import pandas as pd
import polars as pl
import numpy as np
import time

# Custom implementations (sklearn not available)
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred, average='macro', pos_label=True, zero_division=0):
    if average == 'macro':
        # Multi-class macro F1
        classes = np.unique(y_true)
        f1_scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        return np.mean(f1_scores)
    else:
        # Binary F1
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else zero_division

def normalize(X, norm='l2'):
    if norm == 'l2':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return X / norms
    return X

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_cls) & (y_pred == pred_cls))
    return cm

def classification_report(y_true, y_pred, target_names=None):
    classes = np.unique(y_true)
    if target_names is None:
        target_names = [f"Class_{i}" for i in classes]

    report_lines = []
    report_lines.append(f"{'':20s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    report_lines.append("")

    for i, cls in enumerate(classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report_lines.append(f"{target_names[i]:20s} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10d}")

    # Add accuracy
    acc = accuracy_score(y_true, y_pred)
    report_lines.append("")
    report_lines.append(f"{'accuracy':20s} {'':<10s} {'':<10s} {acc:>10.2f} {len(y_true):>10d}")

    return "\n".join(report_lines)

print("=" * 80)
print("PERSONA CLASSIFICATION VIA ACTIVITY GAIN SIMILARITY")
print("=" * 80)

start = time.time()

# =============================================================================
# Step 1: Load Persona Gain Patterns
# =============================================================================

print("\n[1/5] Loading persona gain patterns from XGBoost feature importance...")

gain_df = pd.read_csv("../../data/processed/core_activity_per_persona_named.csv")

# Filter only action-related features (exclude gap stats, etc.)
action_gains = gain_df[gain_df["action_type_id"].notna()].copy()

# Pivot to create persona x action_type_id matrix
persona_gain_matrix = action_gains.pivot(
    index="persona",
    columns="action_type_id",
    values="gain"
).fillna(0)

print(f"Persona gain matrix shape: {persona_gain_matrix.shape}")
print(f"Personas: {sorted(persona_gain_matrix.index.tolist())}")
print(f"Action types: {len(persona_gain_matrix.columns)}")

# Get ordered list of action IDs
action_ids = sorted(persona_gain_matrix.columns.astype(int).tolist())

# =============================================================================
# Step 2: Load User Activity Counts
# =============================================================================

print("\n[2/5] Loading user activity counts from feature matrix...")

fm = pl.read_parquet("../../data/processed/feature_matrix_v2.parquet")

# Extract user_id, persona, and action count columns
action_cols = [col for col in fm.columns if col.startswith("count_action_")]

# Convert to pandas for easier manipulation
fm_pd = fm.select(["user_id", "persona"] + action_cols).to_pandas()

print(f"User activity matrix shape: {fm_pd.shape}")
print(f"Users: {len(fm_pd):,}")

# =============================================================================
# Step 3: Align User Activity Matrix with Persona Gain Matrix
# =============================================================================

print("\n[3/5] Aligning user activity counts with persona gain patterns...")

# Extract action_type_ids from column names (count_action_X -> X)
user_action_ids = [int(col.replace("count_action_", "")) for col in action_cols]

# Ensure both matrices have the same action types in the same order
common_actions = sorted(set(action_ids) & set(user_action_ids))

print(f"Common action types: {len(common_actions)}")

# Reorder persona gain matrix
persona_gain_aligned = persona_gain_matrix[common_actions].values  # Shape: (20, num_actions)

# Reorder user activity matrix
user_action_cols = [f"count_action_{aid}" for aid in common_actions]
user_activity_matrix = fm_pd[user_action_cols].values  # Shape: (92289, num_actions)

y_true = fm_pd["persona"].values
user_ids = fm_pd["user_id"].values

print(f"Persona gain matrix: {persona_gain_aligned.shape}")
print(f"User activity matrix: {user_activity_matrix.shape}")

# =============================================================================
# Step 4: Compute Similarity and Assign Personas
# =============================================================================

print("\n[4/5] Computing similarity between users and persona gain patterns...")

# Strategy 1: Cosine Similarity (normalized dot product)
print("\n--- Strategy 1: Cosine Similarity ---")

# Normalize user activity vectors (L2 norm)
user_activity_normalized = normalize(user_activity_matrix, norm='l2')

# Normalize persona gain vectors (L2 norm)
persona_gain_normalized = normalize(persona_gain_aligned, norm='l2')

# Compute cosine similarity: (92289, num_actions) @ (num_actions, 20)
similarity_matrix = user_activity_normalized @ persona_gain_normalized.T  # Shape: (92289, 20)

# Assign each user to persona with highest similarity
y_pred_cosine = np.argmax(similarity_matrix, axis=1)

acc_cosine = accuracy_score(y_true, y_pred_cosine)
f1_cosine = f1_score(y_true, y_pred_cosine, average='macro')

print(f"✓ Cosine Similarity Accuracy: {acc_cosine:.4f} ({acc_cosine * 100:.2f}%)")
print(f"✓ Cosine Similarity Macro F1: {f1_cosine:.4f} ({f1_cosine * 100:.2f}%)")

# Strategy 2: Weighted Dot Product (activity counts × gain values)
print("\n--- Strategy 2: Weighted Dot Product ---")

# Dot product: (92289, num_actions) @ (num_actions, 20)
weighted_scores = user_activity_matrix @ persona_gain_aligned.T  # Shape: (92289, 20)

# Assign each user to persona with highest weighted score
y_pred_weighted = np.argmax(weighted_scores, axis=1)

acc_weighted = accuracy_score(y_true, y_pred_weighted)
f1_weighted = f1_score(y_true, y_pred_weighted, average='macro')

print(f"✓ Weighted Dot Product Accuracy: {acc_weighted:.4f} ({acc_weighted * 100:.2f}%)")
print(f"✓ Weighted Dot Product Macro F1: {f1_weighted:.4f} ({f1_weighted * 100:.2f}%)")

# Strategy 3: Normalized Weighted Score (activity ratios × gain)
print("\n--- Strategy 3: Normalized Weighted (Activity Ratios × Gain) ---")

# Normalize user activities to proportions (sum to 1 per user)
user_activity_ratios = user_activity_matrix / (user_activity_matrix.sum(axis=1, keepdims=True) + 1e-10)

# Weighted score with ratios
ratio_weighted_scores = user_activity_ratios @ persona_gain_aligned.T

y_pred_ratio = np.argmax(ratio_weighted_scores, axis=1)

acc_ratio = accuracy_score(y_true, y_pred_ratio)
f1_ratio = f1_score(y_true, y_pred_ratio, average='macro')

print(f"✓ Ratio-Weighted Accuracy: {acc_ratio:.4f} ({acc_ratio * 100:.2f}%)")
print(f"✓ Ratio-Weighted Macro F1: {f1_ratio:.4f} ({f1_ratio * 100:.2f}%)")

# =============================================================================
# Step 5: Choose Best Strategy and Evaluate
# =============================================================================

print("\n[5/5] Selecting best strategy and generating detailed report...")

# Compare all strategies
strategies = {
    "Cosine Similarity": (y_pred_cosine, acc_cosine, f1_cosine),
    "Weighted Dot Product": (y_pred_weighted, acc_weighted, f1_weighted),
    "Ratio-Weighted": (y_pred_ratio, acc_ratio, f1_ratio),
}

best_strategy = max(strategies.items(), key=lambda x: x[1][1])
best_name, (y_pred_best, acc_best, f1_best) = best_strategy

print(f"\n🏆 BEST STRATEGY: {best_name}")
print(f"   Accuracy: {acc_best:.4f} ({acc_best * 100:.2f}%)")
print(f"   Macro F1: {f1_best:.4f} ({f1_best * 100:.2f}%)")

# =============================================================================
# Detailed Evaluation
# =============================================================================

print("\n" + "=" * 80)
print("DETAILED EVALUATION - BEST STRATEGY")
print("=" * 80)

# Classification report
print("\nPer-Persona Classification Report:")
print(classification_report(y_true, y_pred_best, target_names=[f"Persona_{i}" for i in range(20)]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_best)
print("\nConfusion Matrix (rows=true, cols=predicted):")
print(cm)

# Per-persona accuracy
per_persona_stats = []
for persona in range(20):
    mask = y_true == persona
    if mask.sum() == 0:
        continue

    persona_acc = accuracy_score(y_true[mask], y_pred_best[mask])
    persona_f1 = f1_score(y_true == persona, y_pred_best == persona, pos_label=True, zero_division=0)
    support = mask.sum()

    per_persona_stats.append({
        "persona": persona,
        "support": support,
        "accuracy": persona_acc,
        "f1_score": persona_f1,
    })

df_stats = pd.DataFrame(per_persona_stats).sort_values("f1_score", ascending=False)
print("\nPer-Persona Performance (sorted by F1 score):")
print(df_stats.to_string(index=False))

# =============================================================================
# Compare with Previous Best (XGBoost Enhanced: 46.08%)
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS APPROACHES")
print("=" * 80)

comparison = pd.DataFrame([
    {"Method": "Random Baseline", "Accuracy": 0.05, "F1 Score": 0.05, "Status": "Baseline"},
    {"Method": "K-NN Statistics", "Accuracy": 0.0186, "F1 Score": 0.0, "Status": "❌ Failed"},
    {"Method": "Neural Network Simple", "Accuracy": 0.1109, "F1 Score": 0.0779, "Status": "❌ Poor"},
    {"Method": "XGBoost Simple", "Accuracy": 0.2423, "F1 Score": 0.0944, "Status": "❌ Poor"},
    {"Method": "Neural Network Enhanced", "Accuracy": 0.2836, "F1 Score": 0.2209, "Status": "❌ Poor"},
    {"Method": "XGBoost Enhanced (Previous Best)", "Accuracy": 0.4608, "F1 Score": 0.3066, "Status": "✅ Best"},
    {"Method": f"Gain Similarity - {best_name}", "Accuracy": acc_best, "F1 Score": f1_best, "Status": "🆕 New"},
])

print(comparison.to_string(index=False))

if acc_best > 0.4608:
    print(f"\n🎉 NEW RECORD! Gain similarity beats XGBoost Enhanced by {(acc_best - 0.4608) * 100:.2f}%")
elif acc_best > 0.40:
    print(f"\n✅ COMPETITIVE! Gain similarity is close to XGBoost Enhanced (gap: {(0.4608 - acc_best) * 100:.2f}%)")
else:
    print(f"\n⚠️ LOWER: Gain similarity underperforms XGBoost Enhanced by {(0.4608 - acc_best) * 100:.2f}%")

# =============================================================================
# Save Results
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save predictions
results_df = pd.DataFrame({
    "user_id": user_ids,
    "true_persona": y_true,
    "predicted_persona": y_pred_best,
    "correct": y_true == y_pred_best,
})

results_df.to_csv("../../results/persona_gain_similarity_predictions.csv", index=False)
print("✓ Saved: persona_gain_similarity_predictions.csv")

# Save per-persona stats
df_stats.to_csv("../../data/processed/persona_gain_similarity_per_persona_stats.csv", index=False)
print("✓ Saved: persona_gain_similarity_per_persona_stats.csv")

# Save strategy comparison
strategy_comparison = pd.DataFrame([
    {"Strategy": name, "Accuracy": acc, "F1_Score": f1}
    for name, (_, acc, f1) in strategies.items()
])
strategy_comparison.to_csv("../../results/persona_gain_similarity_strategy_comparison.csv", index=False)
print("✓ Saved: persona_gain_similarity_strategy_comparison.csv")

end = time.time()
print(f"\n⏱️  Total time: {end - start:.2f} seconds")

# =============================================================================
# Final Verdict
# =============================================================================

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

if acc_best >= 0.60:
    print("✅ EXCELLENT: Accuracy >= 60% - Production ready!")
    print("   → Can use this approach to classify new users")
elif acc_best >= 0.50:
    print("✅ GOOD: Accuracy 50-60% - Better than XGBoost Enhanced")
    print("   → Promising approach, consider confidence filtering")
elif acc_best >= 0.40:
    print("⚠️ FAIR: Accuracy 40-50% - Competitive with XGBoost")
    print("   → Similar performance to supervised methods")
else:
    print("❌ POOR: Accuracy < 40%")
    print("   → Gain similarity approach doesn't work well")

print("\n🎯 Next Steps:")
if acc_best >= 0.50:
    print("   1. Implement confidence thresholding (only classify high-confidence users)")
    print("   2. Use this method to classify new users outside 92k dataset")
    print("   3. Create eventlog for classified users and evaluate LSTM")
else:
    print("   1. Proceed with held-out validation approach (recommended in docs)")
    print("   2. Consider ensemble: Gain similarity + XGBoost voting")
    print("   3. Investigate comment sentiment feature extraction")

print("\n" + "=" * 80)
