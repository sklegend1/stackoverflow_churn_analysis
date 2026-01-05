import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
import polars as pl
import numpy as np

print("=" * 80)
print("PERSONA FEATURE EXTRACTION FROM EVENTLOG")
print("=" * 80)

start = time.time()

# ---------- Load eventlog ----------
print("\n[1/4] Loading eventlog_v2.parquet...")
eventlog = pl.scan_parquet("../../data/processed/eventlog_v2.parquet")

# Preview schema
print("Eventlog schema:")
sample = eventlog.fetch(5)
print(sample)
print(f"Total rows in eventlog: {eventlog.select(pl.len()).collect().item():,}")

# ---------- Define action types ----------
CREATE_QUESTION = 1
CREATE_ANSWER = 2
COMMENT = 70
EDIT_ACTIONS = list(range(13, 19))  # 13-18: Edit Title/Body/Tags, Rollback Title/Body/Tags

# ---------- Feature engineering per user ----------
print("\n[2/4] Aggregating features per user...")

user_features = (
    eventlog
    .group_by("user_id")
    .agg([
        # --- Activity Volume Features ---
        pl.len().alias("total_events"),
        (pl.col("action_type_id") == CREATE_QUESTION).sum().alias("total_questions"),
        (pl.col("action_type_id") == CREATE_ANSWER).sum().alias("total_answers"),
        (pl.col("action_type_id") == COMMENT).sum().alias("total_comments"),
        pl.col("action_type_id").is_in(EDIT_ACTIONS).sum().alias("total_edits"),

        # --- Temporal Features ---
        (pl.col("timestamp").max() - pl.col("timestamp").min()).alias("activity_span_microseconds"),
        # gap_time_actions_days is a duration type, convert to float days
        pl.col("gap_time_actions_days").dt.total_days().mean().alias("avg_gap_days"),
        pl.col("gap_time_actions_days").dt.total_days().median().alias("median_gap_days"),
        pl.col("gap_time_actions_days").dt.total_days().std().alias("std_gap_days"),

        # --- Post Interaction ---
        pl.col("post_id").n_unique().alias("unique_posts_interacted"),

        # --- Metadata ---
        pl.col("persona").first().alias("persona"),  # Target label
        pl.col("LastAccessDate").first().alias("LastAccessDate"),
        pl.col("AccountCreationDate").first().alias("AccountCreationDate"),
        pl.col("timestamp").min().alias("first_event_time"),
        pl.col("timestamp").max().alias("last_event_time"),
    ])
    .collect()
)

print(f"Aggregated {user_features.height:,} users")

# ---------- Calculate derived features ----------
print("\n[3/4] Computing derived features...")

user_features = user_features.with_columns([
    # Activity span in days (convert duration to float first)
    (pl.col("activity_span_microseconds").cast(pl.Float64) / (1_000_000 * 60 * 60 * 24)).alias("activity_span_days"),

    # Question ratio
    (pl.col("total_questions").cast(pl.Float64) /
     (pl.col("total_questions") + pl.col("total_answers") + 1)).alias("question_ratio"),

    # Answer ratio
    (pl.col("total_answers").cast(pl.Float64) /
     (pl.col("total_questions") + pl.col("total_answers") + 1)).alias("answer_ratio"),

    # Comment ratio
    (pl.col("total_comments").cast(pl.Float64) / (pl.col("total_events") + 1)).alias("comment_ratio"),

    # Edit ratio
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

# Calculate events_per_day separately after activity_span_days is created
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

# ---------- Select final feature set ----------
print("\n[4/4] Selecting final features...")

FEATURE_COLUMNS = [
    "user_id",
    "persona",  # Target
    # Activity volume
    "total_events",
    "total_questions",
    "total_answers",
    "total_comments",
    "total_edits",
    # Ratios
    "question_ratio",
    "answer_ratio",
    "comment_ratio",
    "edit_ratio",
    # Temporal
    "activity_span_days",
    "events_per_day",
    "avg_gap_days",
    "median_gap_days",
    "std_gap_days",
    # Interaction
    "unique_posts_interacted",
    "events_per_post",
    "edits_per_question",
    # Account
    "account_age_at_first_event_days",
    "days_after_last_event",
]

user_features_final = user_features.select(FEATURE_COLUMNS)

# ---------- Save to CSV ----------
output_file = "user_persona_features_92k.csv"
print(f"\nSaving to {output_file}...")
user_features_final.write_csv(output_file)

end = time.time()
print(f"\n✅ Done! Extracted {len(FEATURE_COLUMNS) - 2} features for {user_features_final.height:,} users")
print(f"⏱️  Execution time: {end - start:.2f} seconds")

# ---------- Display summary statistics ----------
print("\n" + "=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)

# Persona distribution
persona_dist = user_features_final.group_by("persona").len().sort("persona")
print("\nPersona distribution:")
print(persona_dist)

# Feature statistics
print("\nFeature statistics (first 10 features):")
feature_cols = [col for col in FEATURE_COLUMNS if col not in ["user_id", "persona"]]
stats = user_features_final.select(feature_cols[:10]).describe()
print(stats)

print("\n" + "=" * 80)
