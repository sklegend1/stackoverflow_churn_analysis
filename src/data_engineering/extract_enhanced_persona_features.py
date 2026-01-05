import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Enhanced Persona Feature Extraction - Reverse Engineering NMF Clustering

This script extracts features identified from the original research proposal:
1. Collectivistic ratio (community-helping behavior)
2. Individualistic ratio (self-focused behavior)
3. Geographic regions from Location field
4. AboutMe length
5. All original activity-based features

Goal: Recreate features used in original NMF clustering for better classification.
"""

import time
import polars as pl
import pandas as pd
import numpy as np
import re

print("=" * 80)
print("ENHANCED PERSONA FEATURE EXTRACTION (NMF Reverse Engineering)")
print("=" * 80)

start = time.time()

# =============================================================================
# Action Type Definitions
# =============================================================================

CREATE_QUESTION = 1
CREATE_ANSWER = 2
COMMENT = 70
EDIT_ACTIONS = list(range(13, 19))  # 13-18: Edit Title/Body/Tags, Rollback Title/Body/Tags

# =============================================================================
# Part 1: Load Event Log and User Data
# =============================================================================

print("\n[1/6] Loading eventlog_v2.parquet...")
eventlog = pl.scan_parquet("../../data/processed/eventlog_v2.parquet")

print("\n[2/6] Loading users metadata (Location, AboutMe, etc.)...")
users_data = pl.scan_parquet("filtered_users_parquet/*.parquet").select([
    pl.col("Id").alias("user_id"),
    pl.col("Reputation").alias("reputation"),
    pl.col("UpVotes").alias("upvotes"),
    pl.col("DownVotes").alias("downvotes"),
    pl.col("Views").alias("views"),
    pl.col("Location"),
    pl.col("AboutMe"),
]).collect()

print(f"Loaded {users_data.height:,} user records")

# =============================================================================
# Part 2: Extract Post Ownership (for Individualistic Ratio)
# =============================================================================

print("\n[3/6] Determining post ownership from eventlog...")

# Get post creators: user who performed CREATE_QUESTION or CREATE_ANSWER on a post
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

print(f"Identified ownership for {post_ownership.height:,} posts")

# =============================================================================
# Part 3: Calculate Behavioral Features
# =============================================================================

print("\n[4/6] Calculating collectivistic and individualistic ratios...")

# Join eventlog with post ownership to identify edits to own vs others' posts
eventlog_with_ownership = (
    eventlog
    .join(post_ownership.lazy(), on="post_id", how="left")
    .collect()
)

# Aggregate per user
user_features = (
    eventlog_with_ownership.lazy()
    .group_by("user_id")
    .agg([
        # === Metadata ===
        pl.col("persona").first().alias("persona"),
        pl.col("LastAccessDate").first().alias("LastAccessDate"),
        pl.col("AccountCreationDate").first().alias("AccountCreationDate"),
        pl.col("timestamp").min().alias("first_event_time"),
        pl.col("timestamp").max().alias("last_event_time"),

        # === Activity Volume ===
        pl.len().alias("total_events"),
        (pl.col("action_type_id") == CREATE_QUESTION).sum().alias("total_questions"),
        (pl.col("action_type_id") == CREATE_ANSWER).sum().alias("total_answers"),
        (pl.col("action_type_id") == COMMENT).sum().alias("total_comments"),
        pl.col("action_type_id").is_in(EDIT_ACTIONS).sum().alias("total_edits"),

        # === COLLECTIVISTIC RATIO (Community-helping behavior) ===
        # Answers given (helping others) divided by total Q&A activity
        (pl.col("action_type_id") == CREATE_ANSWER).sum().alias("answers_given"),

        # === INDIVIDUALISTIC RATIO (Self-focused behavior) ===
        # Edits to own posts divided by total edits
        (
            (pl.col("action_type_id").is_in(EDIT_ACTIONS)) &
            (pl.col("user_id") == pl.col("owner_id"))
        ).sum().alias("edits_to_own_posts"),

        # === Temporal Features ===
        (pl.col("timestamp").max() - pl.col("timestamp").min()).alias("activity_span_microseconds"),
        pl.col("gap_time_actions_days").dt.total_days().mean().alias("avg_gap_days"),
        pl.col("gap_time_actions_days").dt.total_days().median().alias("median_gap_days"),
        pl.col("gap_time_actions_days").dt.total_days().std().alias("std_gap_days"),

        # === Post Interaction ===
        pl.col("post_id").n_unique().alias("unique_posts_interacted"),
    ])
    .collect()
)

print(f"Aggregated {user_features.height:,} users")

# =============================================================================
# Part 4: Calculate Derived Features
# =============================================================================

print("\n[5/6] Computing derived behavioral features...")

user_features = user_features.with_columns([
    # Activity span in days
    (pl.col("activity_span_microseconds").cast(pl.Float64) / (1_000_000 * 60 * 60 * 24)).alias("activity_span_days"),

    # === COLLECTIVISTIC RATIO ===
    # Formula: answers_given / (total_questions + total_answers)
    # Higher value = more community-helping behavior
    (
        pl.col("answers_given").cast(pl.Float64) /
        (pl.col("total_questions") + pl.col("total_answers") + 1)
    ).alias("collectivistic_ratio"),

    # === INDIVIDUALISTIC RATIO ===
    # Formula: edits_to_own_posts / total_edits
    # Higher value = more self-focused behavior (editing own content)
    (
        pl.col("edits_to_own_posts").cast(pl.Float64) /
        (pl.col("total_edits") + 1)
    ).alias("individualistic_ratio"),

    # Original ratios
    (pl.col("total_questions").cast(pl.Float64) /
     (pl.col("total_questions") + pl.col("total_answers") + 1)).alias("question_ratio"),

    (pl.col("total_answers").cast(pl.Float64) /
     (pl.col("total_questions") + pl.col("total_answers") + 1)).alias("answer_ratio"),

    (pl.col("total_comments").cast(pl.Float64) / (pl.col("total_events") + 1)).alias("comment_ratio"),

    (pl.col("total_edits").cast(pl.Float64) / (pl.col("total_events") + 1)).alias("edit_ratio"),

    # Events per post
    (pl.col("total_events").cast(pl.Float64) / (pl.col("unique_posts_interacted") + 1)).alias("events_per_post"),

    # Edits per question
    (pl.col("total_edits").cast(pl.Float64) / (pl.col("total_questions") + 1)).alias("edits_per_question"),

    # Account age at first event (days)
    ((pl.col("first_event_time") - pl.col("AccountCreationDate")).dt.total_microseconds() /
     (1_000_000 * 60 * 60 * 24)).alias("account_age_at_first_event_days"),

    # Days from last event to last access
    ((pl.col("LastAccessDate") - pl.col("last_event_time")).dt.total_microseconds() /
     (1_000_000 * 60 * 60 * 24)).alias("days_after_last_event"),
])

# Events per day (calculated after activity_span_days exists)
user_features = user_features.with_columns([
    (pl.col("total_events").cast(pl.Float64) /
     (pl.col("activity_span_days") + 1)).alias("events_per_day"),
])

# Fill nulls in gap statistics (users with only 1 event)
user_features = user_features.with_columns([
    pl.col("avg_gap_days").fill_null(0.0),
    pl.col("median_gap_days").fill_null(0.0),
    pl.col("std_gap_days").fill_null(0.0),
])

# =============================================================================
# Part 5: Enrich with User Metadata (Geographic, AboutMe)
# =============================================================================

print("\n[6/6] Enriching with user metadata (Location, AboutMe, reputation)...")

user_features = user_features.join(users_data, on="user_id", how="left")

# === Calculate AboutMe Length ===
user_features = user_features.with_columns([
    pl.col("AboutMe").str.len_chars().fill_null(0).alias("aboutme_length"),
])

# === Parse Geographic Regions from Location Field ===
# Reference regions from proposal: North America, Europe, Asia, South America, Africa, Oceania

def extract_region(location_series):
    """
    Map Location strings to geographic regions.
    Based on common patterns in Stack Overflow Location field.
    """
    if location_series is None:
        return None

    location = str(location_series).lower()

    # North America
    if any(keyword in location for keyword in [
        'usa', 'united states', 'canada', 'mexico', 'america',
        'california', 'new york', 'texas', 'florida', 'washington'
    ]):
        return "North America"

    # Europe
    elif any(keyword in location for keyword in [
        'uk', 'united kingdom', 'england', 'london', 'germany', 'france',
        'spain', 'italy', 'netherlands', 'poland', 'sweden', 'belgium',
        'switzerland', 'austria', 'denmark', 'norway', 'finland', 'ireland'
    ]):
        return "Europe"

    # Asia
    elif any(keyword in location for keyword in [
        'india', 'china', 'japan', 'singapore', 'korea', 'thailand',
        'indonesia', 'malaysia', 'philippines', 'vietnam', 'taiwan',
        'hong kong', 'pakistan', 'bangladesh', 'israel'
    ]):
        return "Asia"

    # South America
    elif any(keyword in location for keyword in [
        'brazil', 'argentina', 'chile', 'colombia', 'peru', 'venezuela'
    ]):
        return "South America"

    # Oceania
    elif any(keyword in location for keyword in [
        'australia', 'new zealand', 'sydney', 'melbourne'
    ]):
        return "Oceania"

    # Africa
    elif any(keyword in location for keyword in [
        'south africa', 'nigeria', 'egypt', 'kenya', 'morocco'
    ]):
        return "Africa"

    else:
        return "Unknown"

# Apply region extraction
print("Parsing geographic regions from Location field...")
user_features_pd = user_features.to_pandas()
user_features_pd['region'] = user_features_pd['Location'].apply(extract_region)

# One-hot encode regions
region_dummies = pd.get_dummies(user_features_pd['region'], prefix='region')
user_features_pd = pd.concat([user_features_pd, region_dummies], axis=1)

# Convert back to Polars
user_features = pl.from_pandas(user_features_pd)

# Fill missing reputation/votes/views with 0
user_features = user_features.with_columns([
    pl.col("reputation").fill_null(0),
    pl.col("upvotes").fill_null(0),
    pl.col("downvotes").fill_null(0),
    pl.col("views").fill_null(0),
])

# =============================================================================
# Part 6: Select Final Feature Set
# =============================================================================

print("\nSelecting final features...")

FEATURE_COLUMNS = [
    "user_id",
    "persona",  # Target

    # === NEW: Behavioral Features (from NMF reverse engineering) ===
    "collectivistic_ratio",  # Community-helping behavior
    "individualistic_ratio",  # Self-focused behavior

    # === NEW: User Metadata ===
    "reputation",
    "upvotes",
    "downvotes",
    "views",
    "aboutme_length",

    # === NEW: Geographic Features ===
    "region_Africa",
    "region_Asia",
    "region_Europe",
    "region_North America",
    "region_Oceania",
    "region_South America",
    "region_Unknown",

    # === Activity Volume ===
    "total_events",
    "total_questions",
    "total_answers",
    "total_comments",
    "total_edits",

    # === Activity Ratios ===
    "question_ratio",
    "answer_ratio",
    "comment_ratio",
    "edit_ratio",

    # === Temporal Features ===
    "activity_span_days",
    "events_per_day",
    "avg_gap_days",
    "median_gap_days",
    "std_gap_days",

    # === Interaction Features ===
    "unique_posts_interacted",
    "events_per_post",
    "edits_per_question",

    # === Account Features ===
    "account_age_at_first_event_days",
    "days_after_last_event",
]

user_features_final = user_features.select(FEATURE_COLUMNS)

# =============================================================================
# Save to CSV
# =============================================================================

output_file = "../../data/processed/user_persona_features_enhanced_92k.csv"
print(f"\nSaving to {output_file}...")
user_features_final.write_csv(output_file)

end = time.time()
print(f"\n✅ Done! Extracted {len(FEATURE_COLUMNS) - 2} features for {user_features_final.height:,} users")
print(f"⏱️  Execution time: {end - start:.2f} seconds")

# =============================================================================
# Display Summary Statistics
# =============================================================================

print("\n" + "=" * 80)
print("ENHANCED FEATURE SUMMARY")
print("=" * 80)

# Persona distribution
persona_dist = user_features_final.group_by("persona").len().sort("persona")
print("\nPersona distribution:")
print(persona_dist)

# NEW behavioral features statistics
print("\n🆕 NEW BEHAVIORAL FEATURES:")
behavioral_features = user_features_final.select([
    "collectivistic_ratio",
    "individualistic_ratio"
]).to_pandas()

print(f"\nCollectivistic Ratio (community-helping behavior):")
print(f"  Mean: {behavioral_features['collectivistic_ratio'].mean():.4f}")
print(f"  Median: {behavioral_features['collectivistic_ratio'].median():.4f}")
print(f"  Std: {behavioral_features['collectivistic_ratio'].std():.4f}")

print(f"\nIndividualistic Ratio (self-focused behavior):")
print(f"  Mean: {behavioral_features['individualistic_ratio'].mean():.4f}")
print(f"  Median: {behavioral_features['individualistic_ratio'].median():.4f}")
print(f"  Std: {behavioral_features['individualistic_ratio'].std():.4f}")

# Geographic distribution
print("\n🌍 GEOGRAPHIC DISTRIBUTION:")
region_cols = [col for col in FEATURE_COLUMNS if col.startswith("region_")]
region_counts = user_features_final.select(region_cols).sum().to_pandas().T
region_counts.columns = ['count']
region_counts = region_counts.sort_values('count', ascending=False)
print(region_counts)

# AboutMe statistics
print("\n📝 ABOUTME LENGTH:")
aboutme_stats = user_features_final.select("aboutme_length").to_pandas()['aboutme_length']
print(f"  Users with AboutMe: {(aboutme_stats > 0).sum():,} ({(aboutme_stats > 0).sum() / len(aboutme_stats) * 100:.1f}%)")
print(f"  Mean length: {aboutme_stats.mean():.1f} characters")
print(f"  Median length: {aboutme_stats.median():.1f} characters")

# Reputation statistics
print("\n⭐ REPUTATION:")
reputation_stats = user_features_final.select("reputation").to_pandas()['reputation']
print(f"  Mean: {reputation_stats.mean():.1f}")
print(f"  Median: {reputation_stats.median():.1f}")
print(f"  Max: {reputation_stats.max():,}")

print("\n" + "=" * 80)
print("FEATURE COMPARISON")
print("=" * 80)
print(f"Original feature set: 19 features")
print(f"Enhanced feature set: {len(FEATURE_COLUMNS) - 2} features")
print(f"\n🆕 NEW FEATURES ADDED:")
print("  • collectivistic_ratio (behavioral)")
print("  • individualistic_ratio (behavioral)")
print("  • reputation, upvotes, downvotes, views (user metrics)")
print("  • aboutme_length (profile completeness)")
print("  • 7 geographic region indicators")
print(f"\nTotal improvement: +{len(FEATURE_COLUMNS) - 2 - 19} features")

print("\n✅ Ready to train enhanced persona classifier!")
print("   Next: Run train_persona_classifier_enhanced.py")
