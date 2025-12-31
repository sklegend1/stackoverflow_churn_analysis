"""
Clean NMF Clustering Approach

1. Extract features from eventlog_v2.parquet and filtered_users_parquet
2. Apply NMF with 20 components
3. Match NMF clusters to actual personas by user ID overlap
4. Use the mapping to classify new users

This is the proper way: recreate clustering, then map discovered clusters to known personas.
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
print("CLEAN NMF CLUSTERING (eventlog + filtered_users only)")
print("=" * 80)

start = time.time()

# =============================================================================
# Step 1: Extract Features from Eventlog
# =============================================================================

print("\n[1/5] Loading eventlog and extracting features...")

eventlog = pl.scan_parquet("eventlog_v2.parquet")

# Action types
CREATE_QUESTION = 1
CREATE_ANSWER = 2
COMMENT = 70
EDIT_ACTIONS = list(range(13, 19))

# Extract post ownership for individualistic ratio
print("  - Identifying post ownership...")
post_ownership = (
    eventlog
    .filter(pl.col("action_type_id").is_in([CREATE_QUESTION, CREATE_ANSWER]))
    .select([
        pl.col("post_id"),
        pl.col("user_id").alias("owner_id"),
    ])
    .unique()
    .collect()
)

# Join ownership and aggregate per user
print("  - Aggregating user features...")
eventlog_with_ownership = eventlog.join(post_ownership.lazy(), on="post_id", how="left")

user_features = (
    eventlog_with_ownership
    .group_by("user_id")
    .agg([
        # Metadata
        pl.col("persona").first().alias("persona"),
        pl.col("timestamp").min().alias("first_event_time"),
        pl.col("timestamp").max().alias("last_event_time"),
        pl.col("AccountCreationDate").first().alias("AccountCreationDate"),
        pl.col("LastAccessDate").first().alias("LastAccessDate"),

        # Activity counts
        pl.len().alias("total_events"),
        (pl.col("action_type_id") == CREATE_QUESTION).sum().alias("total_questions"),
        (pl.col("action_type_id") == CREATE_ANSWER).sum().alias("total_answers"),
        (pl.col("action_type_id") == COMMENT).sum().alias("total_comments"),
        pl.col("action_type_id").is_in(EDIT_ACTIONS).sum().alias("total_edits"),

        # Behavioral features
        (pl.col("action_type_id") == CREATE_ANSWER).sum().alias("answers_given"),
        (
            (pl.col("action_type_id").is_in(EDIT_ACTIONS)) &
            (pl.col("user_id") == pl.col("owner_id"))
        ).sum().alias("edits_to_own_posts"),

        # Temporal
        (pl.col("timestamp").max() - pl.col("timestamp").min()).alias("activity_span_microseconds"),
        pl.col("gap_time_actions_days").dt.total_days().mean().alias("avg_gap_days"),
        pl.col("gap_time_actions_days").dt.total_days().std().alias("std_gap_days"),

        # Interaction
        pl.col("post_id").n_unique().alias("unique_posts_interacted"),
    ])
    .collect()
)

print(f"  ✓ Extracted features for {user_features.height:,} users")

# Calculate derived features
user_features = user_features.with_columns([
    # Activity span in days
    (pl.col("activity_span_microseconds").cast(pl.Float64) / (1_000_000 * 60 * 60 * 24)).alias("activity_span_days"),

    # Collectivistic ratio (community-helping)
    (pl.col("answers_given").cast(pl.Float64) / (pl.col("total_questions") + pl.col("total_answers") + 1)).alias("collectivistic_ratio"),

    # Individualistic ratio (self-focused)
    (pl.col("edits_to_own_posts").cast(pl.Float64) / (pl.col("total_edits") + 1)).alias("individualistic_ratio"),

    # Ratios
    (pl.col("total_questions").cast(pl.Float64) / (pl.col("total_questions") + pl.col("total_answers") + 1)).alias("question_ratio"),
    (pl.col("total_answers").cast(pl.Float64) / (pl.col("total_questions") + pl.col("total_answers") + 1)).alias("answer_ratio"),
    (pl.col("total_comments").cast(pl.Float64) / (pl.col("total_events") + 1)).alias("comment_ratio"),
    (pl.col("total_edits").cast(pl.Float64) / (pl.col("total_events") + 1)).alias("edit_ratio"),

    # Events per post
    (pl.col("total_events").cast(pl.Float64) / (pl.col("unique_posts_interacted") + 1)).alias("events_per_post"),
])

# Events per day
user_features = user_features.with_columns([
    (pl.col("total_events").cast(pl.Float64) / (pl.col("activity_span_days") + 1)).alias("events_per_day"),
])

# Fill nulls
user_features = user_features.with_columns([
    pl.col("avg_gap_days").fill_null(0.0),
    pl.col("std_gap_days").fill_null(0.0),
])

# =============================================================================
# Step 2: Add User Metadata (reputation, location, aboutme)
# =============================================================================

print("\n[2/5] Loading user metadata from filtered_users...")

users_data = pl.scan_parquet("filtered_users_parquet/*.parquet").select([
    pl.col("Id").alias("user_id"),
    pl.col("Reputation").alias("reputation"),
    pl.col("UpVotes").alias("upvotes"),
    pl.col("DownVotes").alias("downvotes"),
    pl.col("Views").alias("views"),
    pl.col("Location"),
    pl.col("AboutMe"),
]).collect()

print(f"  ✓ Loaded {users_data.height:,} user records")

# Join with user features
user_features = user_features.join(users_data, on="user_id", how="left")

# Fill missing values
user_features = user_features.with_columns([
    pl.col("reputation").fill_null(0),
    pl.col("upvotes").fill_null(0),
    pl.col("downvotes").fill_null(0),
    pl.col("views").fill_null(0),
])

# AboutMe length
user_features = user_features.with_columns([
    pl.col("AboutMe").str.len_chars().fill_null(0).alias("aboutme_length"),
])

# Parse geographic regions
print("  - Parsing geographic regions...")
user_features_pd = user_features.to_pandas()

def extract_region(location):
    if pd.isna(location):
        return "Unknown"
    location = str(location).lower()

    if any(k in location for k in ['usa', 'united states', 'canada', 'mexico', 'california', 'new york', 'texas']):
        return "North America"
    elif any(k in location for k in ['uk', 'united kingdom', 'england', 'germany', 'france', 'spain', 'italy', 'netherlands']):
        return "Europe"
    elif any(k in location for k in ['india', 'china', 'japan', 'singapore', 'korea', 'thailand', 'indonesia']):
        return "Asia"
    elif any(k in location for k in ['brazil', 'argentina', 'chile', 'colombia']):
        return "South America"
    elif any(k in location for k in ['australia', 'new zealand']):
        return "Oceania"
    elif any(k in location for k in ['south africa', 'nigeria', 'egypt']):
        return "Africa"
    else:
        return "Unknown"

user_features_pd['region'] = user_features_pd['Location'].apply(extract_region)

# One-hot encode regions
region_dummies = pd.get_dummies(user_features_pd['region'], prefix='region')
user_features_pd = pd.concat([user_features_pd, region_dummies], axis=1)

print(f"  ✓ Feature extraction complete")

# =============================================================================
# Step 3: Prepare Feature Matrix for NMF
# =============================================================================

print("\n[3/5] Preparing feature matrix for NMF...")

# Select numeric features only (exclude user_id, persona, timestamps, text fields)
feature_cols = [
    'collectivistic_ratio', 'individualistic_ratio',
    'reputation', 'upvotes', 'downvotes', 'views', 'aboutme_length',
    'total_events', 'total_questions', 'total_answers', 'total_comments', 'total_edits',
    'question_ratio', 'answer_ratio', 'comment_ratio', 'edit_ratio',
    'activity_span_days', 'events_per_day', 'avg_gap_days', 'std_gap_days',
    'unique_posts_interacted', 'events_per_post',
] + [col for col in user_features_pd.columns if col.startswith('region_')]

X = user_features_pd[feature_cols].values
y_true = user_features_pd['persona'].values
user_ids = user_features_pd['user_id'].values

print(f"Feature matrix: {X.shape} ({len(feature_cols)} features)")
print(f"True personas: {len(np.unique(y_true))} classes")

# Scale to [0, 1] for NMF (requires non-negative data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

# =============================================================================
# Step 4: Apply NMF Clustering
# =============================================================================

print("\n[4/5] Applying NMF with 20 components...")

nmf = NMF(
    n_components=20,
    init='nndsvda',  # Deterministic initialization
    max_iter=1000,
    random_state=42,
    alpha_W=0.0,  # No regularization (let it find natural structure)
    alpha_H=0.0,
    tol=1e-4,
    verbose=1
)

print("\nFitting NMF...")
W = nmf.fit_transform(X_scaled)  # User-component matrix (92289 x 20)
H = nmf.components_  # Component-feature matrix (20 x num_features)

print(f"\n✓ NMF complete")
print(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")
print(f"  Iterations: {nmf.n_iter_}")

# Assign users to clusters
nmf_clusters = np.argmax(W, axis=1)

unique_nmf, counts_nmf = np.unique(nmf_clusters, return_counts=True)
print(f"\nNMF cluster distribution:")
for cluster, count in zip(unique_nmf, counts_nmf):
    print(f"  NMF Cluster {cluster}: {count:,} users ({count/len(nmf_clusters)*100:.1f}%)")

# =============================================================================
# Step 5: Match NMF Clusters to Actual Personas
# =============================================================================

print("\n[5/5] Matching NMF clusters to actual personas by user overlap...")

# Confusion matrix: rows = true personas, cols = NMF clusters
confusion = confusion_matrix(y_true, nmf_clusters, labels=range(20))

print("\nConfusion Matrix (true personas × NMF clusters):")
print("Rows = True Personas (0-19), Cols = NMF Clusters (0-19)")
print(confusion)

# Greedy matching: assign each NMF cluster to persona with max overlap
nmf_to_persona_mapping = {}
persona_to_nmf_mapping = {}
used_personas = set()
cluster_mapping_quality = []

print("\n" + "=" * 80)
print("NMF CLUSTER → PERSONA MAPPING")
print("=" * 80)

for nmf_cluster in range(20):
    cluster_distribution = confusion[:, nmf_cluster]
    total_in_cluster = cluster_distribution.sum()

    if total_in_cluster == 0:
        print(f"NMF Cluster {nmf_cluster:2d}: EMPTY (0 users)")
        continue

    # Find best persona (excluding already mapped)
    available_personas = [p for p in range(20) if p not in used_personas]

    if available_personas:
        best_persona = max(available_personas, key=lambda p: cluster_distribution[p])
        best_count = cluster_distribution[best_persona]
        purity = best_count / total_in_cluster

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

        print(f"NMF Cluster {nmf_cluster:2d} → Persona {best_persona:2d} | "
              f"{best_count:5,} / {total_in_cluster:5,} users = {purity*100:5.1f}% purity")
    else:
        # Fallback if all personas used
        nmf_to_persona_mapping[nmf_cluster] = nmf_cluster
        print(f"NMF Cluster {nmf_cluster:2d} → Persona {nmf_cluster:2d} (fallback)")

# =============================================================================
# Evaluate Mapping Quality
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

# Apply mapping
y_pred = np.array([nmf_to_persona_mapping.get(c, c) for c in nmf_clusters])

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, nmf_clusters)

print(f"\n✓ Mapping Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"✓ Adjusted Rand Index: {ari:.4f}")

# Mapping quality summary
df_mapping_quality = pd.DataFrame(cluster_mapping_quality)
avg_purity = df_mapping_quality['purity'].mean()

print(f"\n📊 Average Cluster Purity: {avg_purity:.4f} ({avg_purity * 100:.2f}%)")

poor_mappings = df_mapping_quality[df_mapping_quality['purity'] < 0.5]
if len(poor_mappings) > 0:
    print(f"⚠️  {len(poor_mappings)} clusters have purity < 50%")
else:
    print("✅ All clusters have purity >= 50%")

# =============================================================================
# Save Results
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save NMF model
with open('nmf_model_clean.pkl', 'wb') as f:
    pickle.dump({
        'nmf': nmf,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'nmf_to_persona_mapping': nmf_to_persona_mapping,
        'persona_to_nmf_mapping': persona_to_nmf_mapping,
        'accuracy': accuracy,
        'ari': ari,
        'avg_purity': avg_purity,
    }, f)

# Save mapping quality
df_mapping_quality.to_csv("nmf_cluster_mapping_quality_clean.csv", index=False)

# Save user assignments
results_df = pd.DataFrame({
    'user_id': user_ids,
    'true_persona': y_true,
    'nmf_cluster': nmf_clusters,
    'predicted_persona': y_pred,
    'correct': y_true == y_pred
})
results_df.to_csv("nmf_user_assignments_clean.csv", index=False)

# Save component interpretation
components_df = pd.DataFrame(H, columns=feature_cols)
components_df.index = [f"NMF_Component_{i}" for i in range(20)]
components_df.to_csv("nmf_components_clean.csv")

end = time.time()
print(f"\n⏱️  Total time: {end - start:.2f} seconds")

# =============================================================================
# Decision Guideline
# =============================================================================

print("\n" + "=" * 80)
print("DECISION GUIDELINE")
print("=" * 80)

if accuracy >= 0.70:
    print("✅ EXCELLENT: NMF mapping works well!")
    print("   → Use this model to classify new users")
    print("   → For new user: extract features → transform with NMF → map cluster to persona")
elif accuracy >= 0.60:
    print("✅ GOOD: NMF better than supervised classification (46%)")
    print("   → Can use for new user classification")
elif accuracy >= 0.50:
    print("⚠️  FAIR: NMF marginally better than random")
    print("   → Consider held-out validation approach instead")
else:
    print("❌ POOR: NMF doesn't recover original personas")
    print("   → Original clustering may have used different features")
    print("   → Recommend held-out validation approach")

if ari >= 0.5:
    print(f"\n✅ High clustering agreement (ARI = {ari:.3f})")
elif ari >= 0.3:
    print(f"\n⚠️  Moderate clustering agreement (ARI = {ari:.3f})")
else:
    print(f"\n❌ Low clustering agreement (ARI = {ari:.3f})")

print("\n✅ Files created:")
print("   - nmf_model_clean.pkl")
print("   - nmf_cluster_mapping_quality_clean.csv")
print("   - nmf_user_assignments_clean.csv")
print("   - nmf_components_clean.csv")

print("\n🎯 To classify NEW users:")
print("   1. Extract same features from their eventlog")
print("   2. Scale: X_new_scaled = scaler.transform(X_new)")
print("   3. Transform: W_new = nmf.transform(X_new_scaled)")
print("   4. Cluster: cluster = argmax(W_new, axis=1)")
print("   5. Map: persona = nmf_to_persona_mapping[cluster]")
