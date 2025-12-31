"""
Statistics-Based Persona Classifier

Uses persona statistics (centroids) from nmf_cluster_statistics_20.csv
to classify users by finding nearest persona via cosine similarity.

This approach is:
- Training-free (no model fitting needed)
- Interpretable (distance to archetypes)
- Consistent with original NMF clustering
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import time

print("=" * 80)
print("STATISTICS-BASED PERSONA CLASSIFIER")
print("=" * 80)

start = time.time()

# =============================================================================
# Load Persona Statistics (Centroids)
# =============================================================================

print("\n[1/5] Loading persona statistics...")
persona_stats = pd.read_csv("nmf_cluster_statistics_20.csv", index_col=0).T
persona_stats.index = persona_stats.index.str.replace('Cluster ', '').astype(int)
persona_stats = persona_stats.sort_index()

print(f"Loaded statistics for {len(persona_stats)} personas")
print(f"Features available: {len(persona_stats.columns)}")

# Select features to use for matching (use averages, not min/max)
FEATURE_COLUMNS = [
    'reputation_average',
    'upvotes_average',
    'downvotes_average',
    'views_average',
    'total_duration_average',
    'count_posthistory_average',
    'count_comments_average',
    'count_badges_average',
    'count_questions_average',
    'count_answers_average',
]

# Extract persona centroids
persona_centroids = persona_stats[FEATURE_COLUMNS].values
print(f"Using {len(FEATURE_COLUMNS)} features for matching")

# =============================================================================
# Load User Features (92k users)
# =============================================================================

print("\n[2/5] Loading user features from eventlog...")

# We need to extract richer features from eventlog that match the statistics
eventlog = pl.scan_parquet("eventlog_v2.parquet")

# Extract features per user
print("Aggregating user features...")
user_features = (
    eventlog
    .group_by("user_id")
    .agg([
        pl.col("persona").first().alias("persona"),

        # Count features (matching persona statistics)
        (pl.col("action_type_id") == 1).sum().alias("count_questions"),
        (pl.col("action_type_id") == 2).sum().alias("count_answers"),
        (pl.col("action_type_id") == 70).sum().alias("count_comments"),
        pl.col("action_type_id").is_in(list(range(13, 59))).sum().alias("count_posthistory"),

        # Temporal (duration in days)
        ((pl.col("timestamp").max() - pl.col("timestamp").min()) / (1_000_000 * 60 * 60 * 24)).alias("total_duration"),

        # Placeholders for features we don't have in eventlog
        # (These would come from users.parquet in production)
        pl.lit(None).alias("reputation"),
        pl.lit(None).alias("upvotes"),
        pl.lit(None).alias("downvotes"),
        pl.lit(None).alias("views"),
        pl.lit(None).alias("count_badges"),
    ])
    .collect()
)

print(f"Extracted features for {user_features.height:,} users")

# =============================================================================
# Enrich with Users Data (reputation, votes, views, badges)
# =============================================================================

print("\n[3/5] Enriching with user metadata from users.parquet...")

# Load users data
users_data = pl.scan_parquet("filtered_users_parquet/*.parquet").select([
    pl.col("Id").alias("user_id"),
    pl.col("Reputation").alias("reputation"),
    pl.col("UpVotes").alias("upvotes"),
    pl.col("DownVotes").alias("downvotes"),
    pl.col("Views").alias("views"),
])

# Join with user features
user_features = user_features.join(users_data.collect(), on="user_id", how="left")

# Handle missing values (fill with 0)
user_features = user_features.with_columns([
    pl.col("reputation").fill_null(0),
    pl.col("upvotes").fill_null(0),
    pl.col("downvotes").fill_null(0),
    pl.col("views").fill_null(0),
    pl.col("total_duration").cast(pl.Float64).fill_null(0.0),
])

# We don't have badges in filtered_users, so we'll estimate from other features
# (In production, this would come from badges table)
user_features = user_features.with_columns([
    # Estimate badges from reputation (rough correlation)
    (pl.col("reputation") / 100).cast(pl.Int32).alias("count_badges")
])

print("✓ User features enriched")

# =============================================================================
# Extract Feature Matrix
# =============================================================================

print("\n[4/5] Preparing feature matrix for classification...")

# Create feature matrix matching persona centroids
feature_mapping = {
    'reputation_average': 'reputation',
    'upvotes_average': 'upvotes',
    'downvotes_average': 'downvotes',
    'views_average': 'views',
    'total_duration_average': 'total_duration',
    'count_posthistory_average': 'count_posthistory',
    'count_comments_average': 'count_comments',
    'count_badges_average': 'count_badges',
    'count_questions_average': 'count_questions',
    'count_answers_average': 'count_answers',
}

# Extract features in correct order
user_feature_matrix = user_features.select([
    "user_id",
    "persona",
    "reputation",
    "upvotes",
    "downvotes",
    "views",
    "total_duration",
    "count_posthistory",
    "count_comments",
    "count_badges",
    "count_questions",
    "count_answers",
]).to_pandas()

X = user_feature_matrix[list(feature_mapping.values())].values
y_true = user_feature_matrix["persona"].values

print(f"Feature matrix shape: {X.shape}")

# =============================================================================
# Standardize Features (important for distance calculation)
# =============================================================================

print("\n[5/5] Classifying users by nearest persona...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
centroids_scaled = scaler.transform(persona_centroids)

# Classify each user
predictions = []
confidences = []

print("Computing distances to persona centroids...")
for i in range(len(X_scaled)):
    user_vec = X_scaled[i]

    # Calculate cosine distance to each persona
    distances = [cosine(user_vec, centroid) for centroid in centroids_scaled]

    # Find nearest persona
    nearest_persona = np.argmin(distances)
    min_distance = distances[nearest_persona]

    predictions.append(nearest_persona)
    confidences.append(1.0 - min_distance)  # Convert distance to confidence

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1:,} / {len(X_scaled):,} users...")

predictions = np.array(predictions)
confidences = np.array(confidences)

# =============================================================================
# Evaluate
# =============================================================================

accuracy = accuracy_score(y_true, predictions)
macro_f1 = f1_score(y_true, predictions, average='macro')

print("\n" + "=" * 80)
print("CLASSIFICATION RESULTS")
print("=" * 80)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Average Confidence: {confidences.mean():.4f}")

# Per-persona performance
print("\nPer-Persona Performance:")
report = classification_report(y_true, predictions, output_dict=True, zero_division=0)

per_persona_results = []
for persona in sorted(np.unique(y_true)):
    if str(persona) in report:
        per_persona_results.append({
            'persona': persona,
            'precision': report[str(persona)]['precision'],
            'recall': report[str(persona)]['recall'],
            'f1_score': report[str(persona)]['f1-score'],
            'support': report[str(persona)]['support']
        })

df_per_persona = pd.DataFrame(per_persona_results)
print(df_per_persona.to_string(index=False))

# Save results
df_per_persona.to_csv("persona_classifier_statistics_per_persona.csv", index=False)

# Save predictions with confidence
results_df = user_feature_matrix[['user_id', 'persona']].copy()
results_df['predicted_persona'] = predictions
results_df['confidence'] = confidences
results_df['correct'] = (results_df['persona'] == results_df['predicted_persona'])
results_df.to_csv("persona_classifier_statistics_predictions.csv", index=False)

end = time.time()
print(f"\n⏱️  Total time: {end - start:.2f} seconds")

# =============================================================================
# Decision Guideline
# =============================================================================

print("\n" + "=" * 80)
print("DECISION GUIDELINE")
print("=" * 80)

if accuracy >= 0.75:
    print("✅ EXCELLENT: Accuracy >= 75% - Statistics-based matching works!")
    print("   → Can use this approach for new user classification")
elif accuracy >= 0.60:
    print("✅ GOOD: Accuracy 60-75% - Acceptable for production")
    print("   → Consider adding more features or filtering low-confidence predictions")
elif accuracy >= 0.45:
    print("⚠️  FAIR: Accuracy 45-60% - Better than XGBoost but still needs improvement")
    print("   → Try Option B (deep learning) or Option C (richer features)")
else:
    print("❌ POOR: Accuracy < 45% - Statistics-based matching insufficient")
    print("   → Need different approach (ensemble, deep learning, or manual rules)")

min_f1 = df_per_persona['f1_score'].min()
if min_f1 >= 0.50:
    print(f"✅ All personas classified well (min F1 = {min_f1:.4f})")
elif min_f1 >= 0.30:
    print(f"⚠️  Some personas struggle (min F1 = {min_f1:.4f})")
else:
    print(f"❌ Many personas fail (min F1 = {min_f1:.4f})")

print("=" * 80)

print("\n✅ Files created:")
print("   - persona_classifier_statistics_per_persona.csv")
print("   - persona_classifier_statistics_predictions.csv")
