"""
Test Pipeline: Evaluate LSTM model on NEW users (not in training set)

This script:
1. Extracts 100k users from users_parquet that are NOT in filtered_users
2. Predicts their personas using the trained persona prediction model
3. Creates their eventlog (like create_eventlog_v2.py)
4. Creates their core eventlog (like create_core_eventlog.py)
5. Runs LSTM evaluation (like lstm_eval_multi_year.py)
6. Compares results with lstm_eval_2016_to_2023.csv
"""

import time
import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support
import re

print("=" * 80)
print("TEST PIPELINE: EVALUATING LSTM ON NEW USERS")
print("=" * 80)

start_time = time.time()

# ============================================================================
# STEP 1: Extract 100k non-overlapping users
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Extracting 100k non-overlapping users")
print("=" * 80)

# Load filtered users (the 92k training users)
filtered_users = pl.read_parquet("filtered_users_parquet/filtered_users_part_00000.parquet")
filtered_user_ids = set(filtered_users["Id"].to_list())
print(f"Filtered users (training set): {len(filtered_user_ids):,}")

# Load all users from users_parquet
all_users = pl.scan_parquet("users_parquet/*.parquet")

# Filter out users that are in the training set
new_users_lazy = all_users.filter(~pl.col("Id").is_in(list(filtered_user_ids)))

# Get count first
print("Counting available new users...")
new_users_count = new_users_lazy.select(pl.count()).collect().item()
print(f"Available new users: {new_users_count:,}")

# Take 100k users (or all if less)
sample_size = min(100000, new_users_count)
print(f"Sampling {sample_size:,} users...")

new_users = (
    new_users_lazy
    .head(sample_size)
    .collect()
)
print(f"New users extracted: {len(new_users):,}")

# Verify no overlap
new_user_ids = set(new_users["Id"].to_list())
overlap = new_user_ids.intersection(filtered_user_ids)
print(f"Overlap check: {len(overlap)} users (should be 0)")

# ============================================================================
# STEP 2: Predict personas for new users
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Predicting personas for new users")
print("=" * 80)

# Load the persona prediction model
print("Loading persona prediction model...")
model_data = joblib.load('persona_model_region_based.joblib')

# Region extraction function
REGION_KEYWORDS = {
    'North America': ['usa', 'united states', 'us', 'canada', 'mexico', 'california', 
                      'new york', 'texas', 'washington', 'seattle', 'san francisco', 
                      'boston', 'chicago', 'toronto', 'vancouver', 'montreal'],
    'Europe': ['uk', 'united kingdom', 'england', 'germany', 'france', 'spain', 'italy',
               'netherlands', 'poland', 'sweden', 'norway', 'denmark', 'switzerland',
               'austria', 'belgium', 'london', 'berlin', 'paris', 'amsterdam', 'barcelona',
               'europe', 'ireland', 'scotland', 'portugal', 'czech', 'romania', 'hungary'],
    'South Asia': ['india', 'pakistan', 'bangladesh', 'sri lanka', 'nepal', 'mumbai',
                   'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata'],
    'East Asia': ['china', 'japan', 'korea', 'taiwan', 'hong kong', 'beijing', 'shanghai',
                  'tokyo', 'seoul', 'singapore', 'vietnam', 'thailand', 'malaysia'],
    'Oceania': ['australia', 'new zealand', 'sydney', 'melbourne', 'brisbane', 'auckland'],
    'South America': ['brazil', 'argentina', 'chile', 'colombia', 'peru', 'venezuela'],
    'Western Asia': ['israel', 'turkey', 'iran', 'uae', 'dubai', 'saudi', 'qatar', 'kuwait'],
    'North Africa': ['egypt', 'morocco', 'algeria', 'tunisia', 'libya', 'cairo'],
    'Africa': ['south africa', 'nigeria', 'kenya', 'ghana', 'ethiopia', 'johannesburg']
}

def extract_region(location_str):
    if pd.isna(location_str) or location_str == '':
        return 'Unknown'
    location_lower = str(location_str).lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw in location_lower for kw in keywords):
            return region
    return 'Unknown'

# Prepare features for prediction
print("Preparing user features...")

# Convert to pandas for easier manipulation
new_users_pd = new_users.to_pandas()

# Basic features
features = pd.DataFrame()
features['user_id'] = new_users_pd['Id']
features['reputation'] = new_users_pd['Reputation'].fillna(0)
features['upvotes'] = new_users_pd['UpVotes'].fillna(0)
features['downvotes'] = new_users_pd['DownVotes'].fillna(0)
features['views'] = new_users_pd['Views'].fillna(0)

# AboutMe length
def get_aboutme_length(x):
    if pd.isna(x) or x == '':
        return 0
    return len(re.sub(r'<[^>]+>', '', str(x)).strip())

features['aboutme_length'] = new_users_pd['AboutMe'].apply(get_aboutme_length)

# Account duration
new_users_pd['CreationDate'] = pd.to_datetime(new_users_pd['CreationDate'])
new_users_pd['LastAccessDate'] = pd.to_datetime(new_users_pd['LastAccessDate'])
features['total_duration'] = ((new_users_pd['LastAccessDate'] - new_users_pd['CreationDate']).dt.days / 30.44).clip(lower=0)

# Region
features['region'] = new_users_pd['Location'].apply(extract_region)

# For eventlog-based features, we need to load posts data
print("Loading activity data for features...")

# Since we don't have full posts data for all users, we'll use posthistory and comments
# which are available for all users
comments = pl.scan_parquet("comments_parquet/*.parquet")

# Get comment counts for new users
print("Getting comment counts...")
user_comment_counts = (
    comments
    .filter(pl.col("userid").is_in(list(new_user_ids)))
    .group_by("userid")
    .agg(pl.len().alias("count_comments"))
    .collect()
    .to_pandas()
)

# Get posthistory counts (which includes edits and other actions)
print("Getting posthistory counts...")
ph = pl.scan_parquet("posthistory_parquet/*.parquet")
user_ph_counts = (
    ph
    .filter(pl.col("userid").is_in(list(new_user_ids)))
    .group_by("userid")
    .agg([
        pl.len().alias("count_posthistory"),
        # Posts where user edited = rough proxy for their posts
        pl.col("postid").n_unique().alias("unique_posts_touched"),
    ])
    .collect()
    .to_pandas()
)

# For question/answer counts, we'll use a rough estimate based on posthistory
# Initial posts (type 1, 2, 3 = initial title, body, tags) indicate post creation
user_post_creation = (
    ph
    .filter(pl.col("userid").is_in(list(new_user_ids)))
    .filter(pl.col("posthistorytypeid").is_in([1, 2, 3]))  # Initial creation events
    .group_by("userid")
    .agg(pl.col("postid").n_unique().alias("estimated_posts_created"))
    .collect()
    .to_pandas()
)

# Merge activity features
features = features.merge(
    user_comment_counts.rename(columns={'userid': 'user_id'}),
    on='user_id', how='left'
)
features = features.merge(
    user_ph_counts.rename(columns={'userid': 'user_id'}),
    on='user_id', how='left'
)
features = features.merge(
    user_post_creation.rename(columns={'userid': 'user_id'}),
    on='user_id', how='left'
)

# Fill missing values - use estimated values from posthistory
features['count_comments'] = features['count_comments'].fillna(0)
features['count_posthistory'] = features['count_posthistory'].fillna(0)
features['unique_posts_touched'] = features['unique_posts_touched'].fillna(0)
features['estimated_posts_created'] = features['estimated_posts_created'].fillna(0)

# Estimate questions and answers (rough 40/60 split based on typical SO distribution)
features['total_posts'] = features['estimated_posts_created']
features['count_questions'] = (features['total_posts'] * 0.4).astype(int)
features['count_answers'] = (features['total_posts'] * 0.6).astype(int)

features['total_events'] = features['total_posts'] + features['count_comments']
features['count_posthistory'] = 0  # We don't have this for simplicity

# Derived ratios
total_qa = features['count_questions'] + features['count_answers'] + 1
features['collectivistic_ratio'] = features['count_answers'] / total_qa
features['individualistic_ratio'] = features['count_questions'] / total_qa
features['answer_question_ratio'] = features['count_answers'] / (features['count_questions'] + 1)
features['comments_per_qa'] = features['count_comments'] / total_qa
features['edits_per_qa'] = 0
features['reputation_per_answer'] = features['reputation'] / (features['count_answers'] + 1)
features['reputation_per_qa'] = features['reputation'] / total_qa
features['upvote_ratio'] = features['upvotes'] / (features['upvotes'] + features['downvotes'] + 1)
features['activity_intensity'] = features['total_events'] / (features['total_duration'] + 1)
features['activity_duration_months'] = features['total_duration']

# Log transforms
for col in ['reputation', 'views', 'upvotes', 'downvotes', 'total_events',
            'count_questions', 'count_answers', 'count_comments']:
    features[f'{col}_log'] = np.log1p(features[col])

# Replace inf
features = features.replace([np.inf, -np.inf], 0)

# Prepare feature matrix for prediction
all_regions = ['Unknown', 'North America', 'Europe', 'South Asia', 'East Asia',
               'Oceania', 'South America', 'Western Asia', 'North Africa', 'Africa']

feature_cols = model_data['feature_names']
X = pd.DataFrame()

for col in feature_cols:
    if col.startswith('region_'):
        region = col.replace('region_', '')
        X[col] = (features['region'] == region).astype(int)
    elif col in features.columns:
        X[col] = features[col]
    else:
        X[col] = 0

X = X.fillna(0).replace([np.inf, -np.inf], 0)
X = X[feature_cols]  # Ensure correct order

# Scale and predict
X_scaled = model_data['scaler'].transform(X)
predicted_personas = model_data['model'].predict(X_scaled)

# Add persona to features
features['persona'] = predicted_personas
features['ClusterNmf20'] = predicted_personas  # For compatibility

print(f"Personas predicted for {len(features):,} users")
print(f"Persona distribution:\n{pd.Series(predicted_personas).value_counts().sort_index()}")

# Add LastAccessDate for eventlog
features['LastAccessDate'] = new_users_pd['LastAccessDate']
features['AccountCreationDate'] = new_users_pd['CreationDate']

# Save for later use
new_users_with_personas = features.copy()
new_users_with_personas.to_parquet("test_new_users_with_personas.parquet", index=False)
print("Saved: test_new_users_with_personas.parquet")

# ============================================================================
# STEP 3: Create eventlog for new users
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Creating eventlog for new users")
print("=" * 80)

# Action type mappings (same as create_eventlog_v2.py)
PH_ACTION_MAP = {
    4: 13, 5: 14, 6: 15, 7: 16, 8: 17, 9: 18, 10: 30, 11: 31, 12: 32,
    13: 33, 14: 34, 15: 35, 16: 40, 17: 41, 18: 42, 19: 43, 20: 44,
    21: 45, 22: 46, 24: 47, 25: 48, 31: 49, 33: 50, 34: 51, 35: 52,
    36: 53, 37: 54, 38: 55, 50: 56, 52: 57, 53: 58, 66: 59,
}

CREATE_QUESTION_ACTION = 1
CREATE_ANSWER_ACTION = 2
COMMENT_ACTION = 70

# Prepare users lazy frame with persona info
users_for_eventlog = pl.DataFrame({
    'user_id': features['user_id'].values,
    'persona': features['persona'].values,
    'AccountCreationDate': features['AccountCreationDate'].values,
    'LastAccessDate': features['LastAccessDate'].values,
}).lazy()

# Load data sources
posts = pl.scan_parquet("posts_parquet/*.parquet")
ph = pl.scan_parquet("posthistory_parquet/*.parquet")
comments = pl.scan_parquet("comments_parquet/*.parquet")

# Events: Create Post
print("Creating post events...")
events_create_post = (
    posts
    .filter(pl.col("owneruserid").is_in(list(new_user_ids)))
    .join(users_for_eventlog, left_on="owneruserid", right_on="user_id", how="inner")
    .with_columns([
        pl.col("owneruserid").alias("user_id"),
        pl.col("creationdate").alias("timestamp"),
        pl.col("id").alias("post_id"),
        pl.col("posttypeid").alias("post_type_id"),
        pl.when(pl.col("posttypeid") == 1)
          .then(pl.lit(CREATE_QUESTION_ACTION))
          .when(pl.col("posttypeid") == 2)
          .then(pl.lit(CREATE_ANSWER_ACTION))
          .otherwise(pl.lit(None))
          .alias("action_type_id"),
    ])
    .select([
        "user_id", "timestamp", "action_type_id", "post_id", "post_type_id",
        "persona", "LastAccessDate", "AccountCreationDate",
    ])
)

# Events: PostHistory
print("Creating posthistory events...")
PH_VALID = list(PH_ACTION_MAP.keys())

ph_user = (
    ph
    .join(users_for_eventlog, left_on="userid", right_on="user_id", how="inner")
    .filter(pl.col("posthistorytypeid").is_in(PH_VALID))
)

ph_mapped = ph_user.with_columns([
    pl.col("userid").alias("user_id"),
    pl.col("creationdate").alias("timestamp"),
    pl.col("postid").alias("post_id"),
    pl.col("posthistorytypeid")
    .replace(PH_ACTION_MAP)
    .cast(pl.Int32)
    .alias("action_type_id"),
])

events_ph = (
    ph_mapped
    .join(
        posts.select([
            pl.col("id").alias("post_id"),
            pl.col("posttypeid").alias("post_type_id"),
        ]),
        on="post_id",
        how="left",
    )
    .select([
        "user_id", "timestamp", "action_type_id", "post_id", "post_type_id",
        "persona", "LastAccessDate", "AccountCreationDate",
    ])
)

# Events: Comments
print("Creating comment events...")
comments_user = (
    comments
    .join(users_for_eventlog, left_on="userid", right_on="user_id", how="inner")
)

events_comments = (
    comments_user
    .join(
        posts.select([
            pl.col("id").alias("post_id"),
            pl.col("posttypeid").alias("post_type_id"),
        ]),
        left_on="postid",
        right_on="post_id",
        how="left",
    )
    .with_columns([
        pl.col("userid").alias("user_id"),
        pl.col("creationdate").alias("timestamp"),
        pl.lit(COMMENT_ACTION).alias("action_type_id"),
        pl.col("postid").alias("post_id"),
    ])
    .select([
        "user_id", "timestamp", "action_type_id", "post_id", "post_type_id",
        "persona", "LastAccessDate", "AccountCreationDate",
    ])
)

# Combine all events
print("Combining all events...")
eventlog_lazy = pl.concat(
    [events_create_post, events_ph, events_comments],
    how="diagonal_relaxed"
)

# Sort and compute gap times
eventlog_lazy = eventlog_lazy.sort(["user_id", "timestamp"])
eventlog_lazy = eventlog_lazy.with_columns([
    (pl.col("timestamp").diff().over("user_id") / (1000000 * 60 * 60 * 24)).alias("gap_time_actions_days"),
    (pl.col("timestamp").diff().over("user_id") / 1000000).alias("gap_time_actions_seconds"),
])
eventlog_lazy = eventlog_lazy.with_row_index("id")

print("Collecting eventlog...")
eventlog = eventlog_lazy.collect(engine="streaming")
print(f"Created eventlog with {eventlog.height:,} rows")

eventlog.write_parquet("test_eventlog_new_users.parquet")
print("Saved: test_eventlog_new_users.parquet")

# ============================================================================
# STEP 4: Create core eventlog for new users
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Creating core eventlog for new users")
print("=" * 80)

# Load core activities
core_df = pl.read_csv("core_activity_per_persona_named.csv")
core = core_df.lazy()
core = core.filter(pl.col("action_type_id").is_not_null())
core = core.with_columns([
    pl.col("persona").cast(pl.Int32),
    pl.col("action_type_id").cast(pl.Int32),
    pl.col("gain").cast(pl.Float64),
])

# Top 7 actions per persona
core_top = (
    core
    .sort(["persona", "gain"], descending=[False, True])
    .group_by("persona")
    .head(7)
    .select(["persona", "action_type_id", "gain"])
)

# Load eventlog
elog = pl.scan_parquet("test_eventlog_new_users.parquet")

# Filter to core actions
core_eventlog_lazy = (
    elog
    .join(core_top, on=["persona", "action_type_id"], how="inner")
    .sort(["user_id", "timestamp"])
)

# Compute gap times
core_eventlog_lazy = core_eventlog_lazy.with_columns([
    (pl.col("timestamp").diff().over("user_id") / 1000000).alias("gap_time_key_actions_seconds"),
    (pl.col("timestamp").diff().over("user_id") / (1000000 * 60 * 60 * 24)).alias("gap_time_key_actions_days"),
])

# Drop old gap columns and add new id
cols_to_drop = [c for c in core_eventlog_lazy.columns if c.startswith("gap_time_actions")] + ["id"]
cols_to_drop = [c for c in cols_to_drop if c in core_eventlog_lazy.columns]
core_eventlog_lazy = core_eventlog_lazy.drop(cols_to_drop)
core_eventlog_lazy = core_eventlog_lazy.with_row_index("id")

print("Collecting core eventlog...")
core_eventlog = core_eventlog_lazy.collect(engine="streaming")
print(f"Created core eventlog with {core_eventlog.height:,} rows")

core_eventlog.write_parquet("test_core_eventlog_new_users.parquet")
print("Saved: test_core_eventlog_new_users.parquet")

# ============================================================================
# STEP 5: Run LSTM evaluation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Running LSTM evaluation on new users")
print("=" * 80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

EVAL_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400

# Load LSTM model metadata
with open('lstm_model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

SEQUENCE_LENGTH = metadata['sequence_length']
PERSONA_EMBED_DIM = metadata['persona_embed_dim']
LSTM_HIDDEN = metadata['lstm_hidden']
NUM_LAYERS = metadata['num_layers']
persona_features_matrix = metadata['persona_features_matrix']

# Define model
class TemporalChurnPredictor(nn.Module):
    def __init__(self, sequence_feat_dim, global_feat_dim, persona_feat_dim,
                 lstm_hidden, num_layers, persona_embed_dim):
        super().__init__()
        self.persona_embed = nn.Linear(persona_feat_dim, persona_embed_dim)
        self.lstm = nn.LSTM(
            input_size=sequence_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        combined_dim = lstm_hidden + global_feat_dim + persona_embed_dim
        self.fc_shared = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.churn_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.ap_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.tp_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU()
        )

    def forward(self, sequence, global_feat, persona_feat):
        persona_embed = torch.relu(self.persona_embed(persona_feat))
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        sequence_encoding = h_n[-1]
        combined = torch.cat([sequence_encoding, global_feat, persona_embed], dim=1)
        shared = self.fc_shared(combined)
        return self.churn_head(shared), self.ap_head(shared), self.tp_head(shared)

# Load model
model = TemporalChurnPredictor(
    sequence_feat_dim=2,
    global_feat_dim=4,
    persona_feat_dim=persona_features_matrix.shape[1],
    lstm_hidden=LSTM_HIDDEN,
    num_layers=NUM_LAYERS,
    persona_embed_dim=PERSONA_EMBED_DIM
).to(DEVICE)

model.load_state_dict(torch.load('best_temporal_lstm_model.pt', map_location=DEVICE, weights_only=False))
model.eval()
print("✓ LSTM Model loaded")

# Load core eventlog
elog = pl.read_parquet("test_core_eventlog_new_users.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])
print(f"✓ Data loaded: {elog.height:,} events")

ERAS = {'pre_covid': (2016, 2019), 'covid': (2020, 2021), 'post_llm': (2022, 2024)}

def get_era(year):
    for era_name, (start, end) in ERAS.items():
        if start <= year <= end:
            return era_name
    return 'post_llm'

def era_to_int(era):
    return {'pre_covid': 0, 'covid': 1, 'post_llm': 2}.get(era, 2)

persona_features_tensor = torch.FloatTensor(persona_features_matrix).to(DEVICE)

all_year_results = []

for EVAL_YEAR in EVAL_YEARS:
    print(f"\n--- Evaluating {EVAL_YEAR} ---")
    cutoff = datetime(EVAL_YEAR, 1, 1)
    era = get_era(EVAL_YEAR)
    era_int = era_to_int(era)
    
    year_results = []
    
    for persona in range(20):
        persona_events = (
            elog
            .filter(pl.col("persona") == persona)
            .filter(pl.col("timestamp") < pl.lit(cutoff))
            .sort("timestamp")
        )
        
        if persona_events.height == 0:
            continue
        
        users = persona_events.select("user_id").unique()["user_id"].to_list()
        if len(users) == 0:
            continue
        
        user_predictions = []
        
        for user_id in users:
            user_events = persona_events.filter(pl.col("user_id") == user_id).sort("timestamp")
            
            if user_events.height < 5:
                continue
            
            sequence = user_events.tail(SEQUENCE_LENGTH)
            timestamps = sequence["timestamp"].to_list()
            gains = sequence["gain"].to_list()
            
            if len(gains) < SEQUENCE_LENGTH:
                gains = [0.0] * (SEQUENCE_LENGTH - len(gains)) + gains
                timestamps = [timestamps[0]] * (SEQUENCE_LENGTH - len(timestamps)) + timestamps
            
            time_deltas = [0.0]
            for i in range(1, len(timestamps)):
                delta = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400.0
                time_deltas.append(delta)
            
            days_since_last = (cutoff - timestamps[-1]).total_seconds() / 86400.0
            
            # Check actual churn
            after_events = (
                elog
                .filter(pl.col("user_id") == user_id)
                .filter(pl.col("persona") == persona)
                .filter(
                    (pl.col("timestamp") > pl.lit(cutoff) + pl.duration(seconds=GRACE_SEC)) &
                    (pl.col("timestamp") < pl.lit(cutoff) + pl.duration(seconds=GRACE_SEC + 365 * 86400))
                )
            )
            churned = 1 if after_events.height == 0 else 0
            
            # Prepare tensors
            sequence_tensor = torch.FloatTensor(
                np.column_stack([gains, time_deltas])
            ).unsqueeze(0).to(DEVICE)
            
            global_features = torch.FloatTensor([
                days_since_last, era_int, cutoff.month / 12.0, (EVAL_YEAR - 2016) / 7.0,
            ]).unsqueeze(0).to(DEVICE)
            
            persona_feat = persona_features_tensor[persona].unsqueeze(0)
            
            with torch.no_grad():
                churn_prob, _, _ = model(sequence_tensor, global_features, persona_feat)
            
            pred_lstm = 1 if churn_prob.cpu().item() > 0.5 else 0
            user_predictions.append({
                'churned_actual': churned,
                'pred_lstm': pred_lstm,
                'churn_prob': churn_prob.cpu().item(),
            })
        
        if len(user_predictions) == 0:
            continue
        
        df_pred = pd.DataFrame(user_predictions)
        y_true = df_pred['churned_actual'].values
        y_pred = df_pred['pred_lstm'].values
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        year_results.append({
            'year': EVAL_YEAR,
            'persona': persona,
            'num_users': len(df_pred),
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'actual_rate': y_true.mean(),
            'pred_rate': y_pred.mean(),
        })
    
    df_year = pd.DataFrame(year_results)
    all_year_results.extend(year_results)
    
    if len(df_year) > 0:
        print(f"  Personas: {len(df_year)}, Users: {df_year['num_users'].sum():,}")
        print(f"  Median F1: {df_year['f1'].median():.4f}, Mean F1: {df_year['f1'].mean():.4f}")

# Save results
df_new_users = pd.DataFrame(all_year_results)
df_new_users.to_csv("lstm_eval_new_users_2016_to_2023.csv", index=False)
print(f"\n✓ Results saved to: lstm_eval_new_users_2016_to_2023.csv")

# ============================================================================
# STEP 6: Compare with original results
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Comparing with original training set results")
print("=" * 80)

# Load original results
df_original = pd.read_csv("lstm_eval_2016_to_2023.csv")

print("\n" + "-" * 80)
print("COMPARISON: Original (92k training users) vs New (100k test users)")
print("-" * 80)

comparison_results = []

for year in EVAL_YEARS:
    orig_year = df_original[df_original['year'] == year]
    new_year = df_new_users[df_new_users['year'] == year]
    
    if len(orig_year) > 0 and len(new_year) > 0:
        orig_f1 = orig_year['f1'].median()
        new_f1 = new_year['f1'].median()
        diff = new_f1 - orig_f1
        
        comparison_results.append({
            'year': year,
            'original_users': orig_year['num_users'].sum(),
            'new_users': new_year['num_users'].sum(),
            'original_f1': orig_f1,
            'new_f1': new_f1,
            'f1_diff': diff,
        })
        
        print(f"\n{year}:")
        print(f"  Original: {orig_year['num_users'].sum():,} users, Median F1: {orig_f1:.4f}")
        print(f"  New:      {new_year['num_users'].sum():,} users, Median F1: {new_f1:.4f}")
        print(f"  Difference: {diff:+.4f}")

df_comparison = pd.DataFrame(comparison_results)
df_comparison.to_csv("lstm_eval_comparison_original_vs_new.csv", index=False)
print(f"\n✓ Comparison saved to: lstm_eval_comparison_original_vs_new.csv")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if len(df_comparison) > 0:
    avg_orig_f1 = df_comparison['original_f1'].mean()
    avg_new_f1 = df_comparison['new_f1'].mean()
    avg_diff = df_comparison['f1_diff'].mean()
    
    print(f"""
    Overall Comparison:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Original (Training Set)  Average F1: {avg_orig_f1:.4f}                      │
    │ New Users (Test Set)     Average F1: {avg_new_f1:.4f}                      │
    │ Average Difference:                  {avg_diff:+.4f}                      │
    └─────────────────────────────────────────────────────────────────────┘
    
    Interpretation:
    - If difference is small (< 0.05): Model generalizes well!
    - If new F1 is lower: Some overfitting to training users
    - If new F1 is higher: New users may be easier to predict
    """)

end_time = time.time()
print(f"\nTotal execution time: {(end_time - start_time) / 60:.1f} minutes")
print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
