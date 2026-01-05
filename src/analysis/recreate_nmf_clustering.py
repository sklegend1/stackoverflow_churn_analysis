import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Recreate NMF Clustering on 92k Users

This script:
1. Applies NMF (Non-negative Matrix Factorization) with 20 components on 92k users
2. Matches NMF clusters to actual personas by finding best user overlap
3. Creates a mapping: NMF_cluster → actual_persona
4. Validates the mapping quality

This is the most mathematically sound approach - we recreate the original
clustering method and map the results to known personas.
"""

import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score
import pickle

print("=" * 80)
print("RECREATE NMF CLUSTERING ON 92K USERS")
print("=" * 80)

start = time.time()

# =============================================================================
# Step 1: Load Enhanced Features
# =============================================================================

print("\n[1/5] Loading enhanced user features (33 features)...")
df = pl.read_csv("../../data/processed/user_persona_features_enhanced_92k.csv")
print(f"Loaded {df.height:,} users with {df.width - 2} features")

# Convert to pandas
df_pd = df.to_pandas()

# Separate features and target
feature_cols = [col for col in df_pd.columns if col not in ["user_id", "persona"]]
X = df_pd[feature_cols].values
y_true = df_pd["persona"].values
user_ids = df_pd["user_id"].values

print(f"Feature matrix: {X.shape}")
print(f"True personas: {len(np.unique(y_true))} classes")

# =============================================================================
# Step 2: Standardize Features (required for NMF)
# =============================================================================

print("\n[2/5] Standardizing features for NMF...")

# NMF requires non-negative data, so we'll scale to [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled data range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

# =============================================================================
# Step 3: Apply NMF Clustering (20 components)
# =============================================================================

print("\n[3/5] Applying NMF clustering with 20 components...")

# NMF configuration (matching research approach)
nmf = NMF(
    n_components=20,
    init='nndsvda',  # Non-negative SVD initialization (better than random)
    max_iter=500,
    random_state=42,
    alpha_W=0.1,  # L1 regularization for W
    alpha_H=0.1,  # L1 regularization for H (sparsity)
    l1_ratio=0.5,  # Balance between L1 and L2
    verbose=1
)

print("\nNMF Configuration:")
print(f"  n_components: 20")
print(f"  init: nndsvda (deterministic)")
print(f"  max_iter: 500")
print(f"  alpha: 0.1 (L1/L2 regularization)")
print(f"  l1_ratio: 0.5")

print("\nFitting NMF (this may take 1-2 minutes)...")
W = nmf.fit_transform(X_scaled)  # User-topic matrix (92289 x 20)
H = nmf.components_  # Topic-feature matrix (20 x 33)

print(f"\n✓ NMF fitting complete")
print(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")
print(f"  Iterations: {nmf.n_iter_}")

# Assign users to clusters (highest component value)
nmf_clusters = np.argmax(W, axis=1)

print(f"\nNMF cluster distribution:")
unique_nmf, counts_nmf = np.unique(nmf_clusters, return_counts=True)
for cluster, count in zip(unique_nmf[:10], counts_nmf[:10]):
    print(f"  NMF Cluster {cluster}: {count:,} users")
print("  ...")

# =============================================================================
# Step 4: Match NMF Clusters to Actual Personas
# =============================================================================

print("\n[4/5] Matching NMF clusters to actual personas...")

# Create confusion matrix: rows = true personas, cols = NMF clusters
confusion = confusion_matrix(y_true, nmf_clusters, labels=range(20))

print("\nConfusion Matrix (first 10x10):")
print("Rows = True Personas, Cols = NMF Clusters")
print(confusion[:10, :10])

# For each NMF cluster, find the persona with most overlap
nmf_to_persona_mapping = {}
persona_to_nmf_mapping = {}

print("\n🔗 NMF Cluster → Persona Mapping (by maximum overlap):")
print("=" * 80)

# Greedy matching: assign each NMF cluster to the persona it overlaps with most
used_personas = set()
cluster_mapping_quality = []

for nmf_cluster in range(20):
    # Get counts for this NMF cluster across all personas
    cluster_distribution = confusion[:, nmf_cluster]

    # Find persona with maximum overlap (excluding already used)
    available_personas = [p for p in range(20) if p not in used_personas]

    if available_personas:
        best_persona = max(available_personas, key=lambda p: cluster_distribution[p])
        best_count = cluster_distribution[best_persona]
        total_in_cluster = cluster_distribution.sum()

        if total_in_cluster > 0:
            purity = best_count / total_in_cluster
        else:
            purity = 0.0

        nmf_to_persona_mapping[nmf_cluster] = best_persona
        persona_to_nmf_mapping[best_persona] = nmf_cluster
        used_personas.add(best_persona)

        cluster_mapping_quality.append({
            'nmf_cluster': nmf_cluster,
            'mapped_persona': best_persona,
            'overlap_count': best_count,
            'cluster_size': total_in_cluster,
            'purity': purity
        })

        print(f"NMF Cluster {nmf_cluster:2d} → Persona {best_persona:2d} "
              f"({best_count:5,} / {total_in_cluster:5,} users = {purity * 100:5.1f}% purity)")
    else:
        # Fallback: map to itself if no personas left
        nmf_to_persona_mapping[nmf_cluster] = nmf_cluster
        print(f"NMF Cluster {nmf_cluster:2d} → Persona {nmf_cluster:2d} (fallback)")

# =============================================================================
# Step 5: Evaluate Mapping Quality
# =============================================================================

print("\n[5/5] Evaluating mapping quality...")

# Apply mapping to get predicted personas
y_pred = np.array([nmf_to_persona_mapping[c] for c in nmf_clusters])

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✓ Mapping Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Calculate Adjusted Rand Index (measures clustering similarity)
ari = adjusted_rand_score(y_true, nmf_clusters)
print(f"✓ Adjusted Rand Index: {ari:.4f}")
print("  (1.0 = perfect clustering match, 0.0 = random)")

# Per-persona mapping quality
df_mapping_quality = pd.DataFrame(cluster_mapping_quality)
print("\n📊 MAPPING QUALITY PER CLUSTER:")
print("=" * 80)
print(df_mapping_quality.to_string(index=False))

# Summary statistics
avg_purity = df_mapping_quality['purity'].mean()
print(f"\n📈 Average Cluster Purity: {avg_purity:.4f} ({avg_purity * 100:.2f}%)")

# Find poorly mapped clusters (purity < 0.5)
poor_mappings = df_mapping_quality[df_mapping_quality['purity'] < 0.5]
if len(poor_mappings) > 0:
    print(f"\n⚠️  WARNING: {len(poor_mappings)} clusters have purity < 50%")
    print("Poor mappings:")
    print(poor_mappings.to_string(index=False))
else:
    print("\n✅ All clusters have purity >= 50%")

# =============================================================================
# Save Results
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save NMF model and mapping
with open('../../checkpoints/nmf_clustering_model.pkl', 'wb') as f:
    pickle.dump({
        'nmf_model': nmf,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'nmf_to_persona_mapping': nmf_to_persona_mapping,
        'persona_to_nmf_mapping': persona_to_nmf_mapping,
        'W': W,  # User-topic matrix
        'H': H,  # Topic-feature matrix
        'accuracy': accuracy,
        'ari': ari,
        'avg_purity': avg_purity,
    }, f)

# Save mapping quality
df_mapping_quality.to_csv("../../data/processed/nmf_cluster_mapping_quality.csv", index=False)

# Save user assignments
results_df = pd.DataFrame({
    'user_id': user_ids,
    'true_persona': y_true,
    'nmf_cluster': nmf_clusters,
    'predicted_persona': y_pred,
    'correct': y_true == y_pred
})
results_df.to_csv("../../data/processed/nmf_user_cluster_assignments.csv", index=False)

# Save NMF components (topic-feature matrix)
components_df = pd.DataFrame(
    H,
    columns=feature_cols,
    index=[f"NMF_Component_{i}" for i in range(20)]
)
components_df.to_csv("../../data/processed/nmf_components_features.csv")

end = time.time()
print(f"\n⏱️  Total time: {end - start:.2f} seconds")

# =============================================================================
# Decision Guideline
# =============================================================================

print("\n" + "=" * 80)
print("DECISION GUIDELINE - NMF APPROACH")
print("=" * 80)

if accuracy >= 0.80:
    print("✅ EXCELLENT: Accuracy >= 80% - NMF mapping is highly accurate!")
    print("   → This is the BEST approach for mapping new users")
    print("   → NMF captures the true persona structure")
    print("   ✓ Use this model to classify new users:")
    print("     1. Extract same 33 features for new users")
    print("     2. Transform with NMF: W_new = nmf.transform(X_new)")
    print("     3. Assign cluster: nmf_cluster = argmax(W_new, axis=1)")
    print("     4. Map to persona: persona = nmf_to_persona_mapping[nmf_cluster]")
elif accuracy >= 0.70:
    print("✅ GOOD: Accuracy 70-80% - NMF mapping works well!")
    print("   → Better than supervised classification (46%)")
    print("   → NMF partially recovers original persona structure")
    print("   ✓ Can use for new user classification with confidence thresholding")
elif accuracy >= 0.60:
    print("⚠️  FAIR: Accuracy 60-70% - NMF mapping is better than supervised")
    print("   → Still better than XGBoost (46%)")
    print("   → Consider: Use NMF + confidence filtering")
    print("   → Only classify users with high max(W) values (confident assignments)")
elif accuracy >= 0.50:
    print("⚠️  MARGINAL: Accuracy 50-60% - NMF slightly better than random")
    print("   → Not significantly better than supervised methods")
    print("   → Recommendation: Proceed with held-out validation (Option E)")
else:
    print("❌ POOR: Accuracy < 50% - NMF doesn't match original clustering")
    print("   → Possible reasons:")
    print("     • Features don't match original NMF (missing comment sentiment)")
    print("     • Original clustering used different features/preprocessing")
    print("     • Personas were manually adjusted after NMF")
    print("   → Recommendation: Use held-out validation approach")

if ari >= 0.5:
    print(f"\n✅ High clustering agreement (ARI = {ari:.3f})")
    print("   → NMF successfully rediscovered similar clusters")
elif ari >= 0.3:
    print(f"\n⚠️  Moderate clustering agreement (ARI = {ari:.3f})")
    print("   → NMF found related but different structure")
else:
    print(f"\n❌ Low clustering agreement (ARI = {ari:.3f})")
    print("   → NMF clustering very different from original personas")

print("=" * 80)

print("\n✅ Files created:")
print("   - nmf_clustering_model.pkl (NMF model + mapping)")
print("   - nmf_cluster_mapping_quality.csv (per-cluster purity)")
print("   - nmf_user_cluster_assignments.csv (92k user assignments)")
print("   - nmf_components_features.csv (20 NMF components × 33 features)")

print("\n🎯 Next Steps:")
if accuracy >= 0.70:
    print("   1. Use NMF model to classify NEW users (not in 92k)")
    print("   2. Create eventlog for NMF-classified users")
    print("   3. Evaluate LSTM on completely new users")
else:
    print("   1. Analyze NMF components to understand discovered patterns")
    print("   2. Compare with original persona statistics")
    print("   3. If accuracy insufficient, use held-out validation approach")

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("\n📊 What is 'Purity'?")
print("   - For each NMF cluster, purity = (most common persona) / (cluster size)")
print("   - High purity (>0.7): Cluster mostly contains one persona → good mapping")
print("   - Low purity (<0.5): Cluster is mixed → poor mapping")
print("\n📊 What is 'Adjusted Rand Index' (ARI)?")
print("   - Measures similarity between two clusterings")
print("   - ARI = 1.0: Perfect agreement (clusters exactly match personas)")
print("   - ARI = 0.0: Random agreement (no better than chance)")
print("   - ARI < 0.0: Worse than random")
print("\n📊 Why might accuracy be lower than expected?")
print("   1. Missing features from original NMF (e.g., comment sentiment)")
print("   2. Original personas were manually curated after NMF")
print("   3. Feature scaling/preprocessing differs from original")
print("   4. Personas evolved over time (2016-2023 behavioral shifts)")
