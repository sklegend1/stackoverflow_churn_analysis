import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
"""
Time-Series LSTM for Dynamic Ap/Tp Prediction

This model learns temporal patterns in user behavior to predict:
1. Churn probability
2. Optimal Ap threshold (personalized)
3. Optimal Tp window (personalized)

Key innovations:
- Persona embeddings from statistics
- Temporal features (year, era)
- Event sequence modeling
- Multi-task learning
"""

import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# =============================================================================
# Configuration
# =============================================================================

SEQUENCE_LENGTH = 50  # Number of recent events to consider
PERSONA_EMBED_DIM = 32  # Persona embedding dimension
LSTM_HIDDEN = 128  # LSTM hidden size
NUM_LSTM_LAYERS = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Temporal era definitions
ERAS = {
    'pre_covid': (2016, 2019),
    'covid': (2020, 2021),
    'post_llm': (2022, 2024),
}

GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400

print("=" * 80)
print("TEMPORAL LSTM FOR DYNAMIC AP/TP PREDICTION")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"LSTM Hidden: {LSTM_HIDDEN}")
print(f"Batch Size: {BATCH_SIZE}")

# =============================================================================
# Load Data
# =============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load event log
elog = pl.read_parquet("../../../data/processed/core_eventlog_train_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"Event log: {elog.height:,} events")
print(f"Date range: {elog['timestamp'].min()} to {elog['timestamp'].max()}")

# Load persona statistics
persona_stats = pd.read_csv("../../../data/processed/nmf_cluster_statistics_20.csv", index_col=0).T
persona_stats.index = persona_stats.index.str.replace('Cluster ', '').astype(int)

print(f"Loaded persona statistics for {len(persona_stats)} personas")

# Load best params (as reference for initial Ap/Tp)
best_params = pd.read_csv("../../../results/best_params/tp_ap_xgb_enhanced_best_params_WA_recency.csv")
Ap_ref = dict(zip(best_params["persona"], best_params["Ap"]))
Tp_ref = dict(zip(best_params["persona"], best_params["Tp_sec"]))

# =============================================================================
# Helper Functions
# =============================================================================

def get_era(year):
    """Map year to era"""
    for era_name, (start, end) in ERAS.items():
        if start <= year <= end:
            return era_name
    return 'post_llm'  # Default for future

def era_to_int(era):
    """Encode era as integer"""
    mapping = {'pre_covid': 0, 'covid': 1, 'post_llm': 2}
    return mapping.get(era, 2)

# =============================================================================
# Create Persona Embeddings from Statistics
# =============================================================================

print("\n" + "=" * 80)
print("CREATING PERSONA EMBEDDINGS")
print("=" * 80)

# Extract key features for persona embeddings
persona_features = []
persona_ids = []

for persona_id in range(20):
    if persona_id not in persona_stats.index:
        # If persona missing, use zeros
        persona_features.append(np.zeros(10))
        persona_ids.append(persona_id)
        continue

    stats = persona_stats.loc[persona_id]

    # Select discriminative features (normalized)
    features = [
        np.log1p(stats.get('reputation_average', 0)),
        np.log1p(stats.get('upvotes_average', 0)),
        np.log1p(stats.get('count_answers_average', 0)),
        np.log1p(stats.get('count_questions_average', 0)),
        stats.get('total_duration_average', 0) / 190.0,  # Normalize by max
        stats.get('collectivistic_ratio_average', 0),
        stats.get('individualistic_ratio_average', 0),
        # Persona type (H/C/T/I/SPECIALIST) - one-hot-ish
        # 1.0 if stats.get('H', 0) > 0 else 0.0,
        # 1.0 if stats.get('C', 0) > 0 else 0.0,
        # 1.0 if stats.get('T', 0) > 0 else 0.0,
        # 1.0 if stats.get('I', 0) > 0 else 0.0,
        # 1.0 if stats.get('SPECIALIST', 0) > 0 else 0.0,
        # aboutme length average
        # np.log1p(stats.get('aboutme_length_average', 0)) / 100,
        # comment sentiment average
        # stats.get('comment_sentiment_average', 0) ,
        # profile views average
        # np.log1p(stats.get('profile_views_average', 0)) / 1000
    ]

    persona_features.append(features)
    persona_ids.append(persona_id)

persona_features_matrix = np.array(persona_features, dtype=np.float32)
print(f"Persona feature matrix shape: {persona_features_matrix.shape}")
print(f"Feature ranges: min={persona_features_matrix.min():.2f}, max={persona_features_matrix.max():.2f}")

# =============================================================================
# Prepare Training Data
# =============================================================================

print("\n" + "=" * 80)
print("PREPARING TRAINING DATA")
print("=" * 80)

# Sample users and create sequences
SAMPLE_CUTOFFS = [2017, 2018, 2019, 2020, 2021, 2022]  # Training years
MAX_USERS_PER_PERSONA = 500  # Limit for memory

training_samples = []

for cutoff_year in SAMPLE_CUTOFFS:
    cutoff = datetime(cutoff_year, 1, 1)

    print(f"\nProcessing cutoff year: {cutoff_year}")

    for persona in range(20):
        # Get events before cutoff for this persona
        persona_events = (
            elog
            .filter(pl.col("persona") == persona)
            .filter(pl.col("timestamp") < pl.lit(cutoff))
            .sort("timestamp")
        )

        if persona_events.height == 0:
            continue

        # Get unique users
        users = persona_events.select("user_id").unique()["user_id"].to_list()

        # Sample users if too many
        if len(users) > MAX_USERS_PER_PERSONA:
            users = np.random.choice(users, MAX_USERS_PER_PERSONA, replace=False)

        for user_id in users:
            # Get user's event sequence before cutoff
            user_events = (
                persona_events
                .filter(pl.col("user_id") == user_id)
                .sort("timestamp")
            )

            if user_events.height < 5:  # Need minimum events
                continue

            # Get last N events
            sequence = user_events.tail(SEQUENCE_LENGTH)

            # Extract sequence features
            timestamps = sequence["timestamp"].to_list()
            gains = sequence["gain"].to_list()

            # Pad if needed
            if len(gains) < SEQUENCE_LENGTH:
                gains = [0.0] * (SEQUENCE_LENGTH - len(gains)) + gains
                timestamps = [timestamps[0]] * (SEQUENCE_LENGTH - len(timestamps)) + timestamps

            # Calculate time deltas (days between events)
            time_deltas = [0.0]
            for i in range(1, len(timestamps)):
                delta = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400.0
                time_deltas.append(delta)

            # Calculate recency (days since last event to cutoff)
            days_since_last = (cutoff - timestamps[-1]).total_seconds() / 86400.0

            # Calculate label: did user churn?
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

            # Get reference Ap/Tp for this persona
            ap_ref = Ap_ref.get(persona, 0.5)
            tp_ref = Tp_ref.get(persona, 30 * 86400)  # Default 30 days

            # Temporal features
            era = get_era(cutoff_year)
            era_int = era_to_int(era)

            training_samples.append({
                'user_id': user_id,
                'persona': persona,
                'cutoff_year': cutoff_year,
                'era': era_int,
                'month': cutoff.month,
                'gains': gains,
                'time_deltas': time_deltas,
                'days_since_last': days_since_last,
                'churned': churned,
                'ap_ref': ap_ref,
                'tp_ref_days': tp_ref / 86400.0,
            })

        print(f"  Persona {persona}: {len([s for s in training_samples if s['persona'] == persona and s['cutoff_year'] == cutoff_year])} samples")

print(f"\nTotal training samples: {len(training_samples)}")

# Calculate class balance
churn_rate = np.mean([s['churned'] for s in training_samples])
print(f"Churn rate: {churn_rate:.4f}")

# =============================================================================
# PyTorch Dataset
# =============================================================================

class ChurnSequenceDataset(Dataset):
    def __init__(self, samples, persona_features):
        self.samples = samples
        self.persona_features = torch.FloatTensor(persona_features)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Event sequence features: [gain, time_delta]
        sequence = np.column_stack([
            sample['gains'],
            sample['time_deltas']
        ]).astype(np.float32)

        # Global features: [days_since_last, era, month, year]
        global_features = np.array([
            sample['days_since_last'],
            sample['era'],
            sample['month'] / 12.0,  # Normalize month
            (sample['cutoff_year'] - 2016) / 7.0,  # Normalize year
        ], dtype=np.float32)

        # Persona features
        persona_idx = sample['persona']
        persona_feat = self.persona_features[persona_idx]

        # Labels
        churned = np.array([sample['churned']], dtype=np.float32)
        ap_ref = np.array([sample['ap_ref']], dtype=np.float32)
        tp_ref = np.array([sample['tp_ref_days']], dtype=np.float32)

        return {
            'sequence': torch.FloatTensor(sequence),
            'global_feat': torch.FloatTensor(global_features),
            'persona_feat': persona_feat,
            'churned': torch.FloatTensor(churned),
            'ap_ref': torch.FloatTensor(ap_ref),
            'tp_ref': torch.FloatTensor(tp_ref),
        }

# =============================================================================
# Model Architecture
# =============================================================================

class TemporalChurnPredictor(nn.Module):
    def __init__(self, sequence_feat_dim, global_feat_dim, persona_feat_dim,
                 lstm_hidden, num_layers, persona_embed_dim):
        super().__init__()

        # Persona embedding layer
        self.persona_embed = nn.Linear(persona_feat_dim, persona_embed_dim)

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=sequence_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Combine LSTM output + global features + persona embedding
        combined_dim = lstm_hidden + global_feat_dim + persona_embed_dim

        # Shared layers
        self.fc_shared = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Multi-task heads
        self.churn_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability
        )

        self.ap_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Ap in (0, 1) range
        )

        self.tp_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Tp in days (positive)
        )

    def forward(self, sequence, global_feat, persona_feat):
        # Process persona features
        persona_embed = self.persona_embed(persona_feat)
        persona_embed = torch.relu(persona_embed)

        # Process sequence with LSTM
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        # Take last hidden state
        sequence_encoding = h_n[-1]  # Last layer's hidden state

        # Concatenate all features
        combined = torch.cat([sequence_encoding, global_feat, persona_embed], dim=1)

        # Shared processing
        shared = self.fc_shared(combined)

        # Multi-task outputs
        churn_prob = self.churn_head(shared)
        ap_pred = self.ap_head(shared)
        tp_pred = self.tp_head(shared)

        return churn_prob, ap_pred, tp_pred

# =============================================================================
# Training
# =============================================================================

print("\n" + "=" * 80)
print("INITIALIZING MODEL")
print("=" * 80)

# Split data
train_samples, val_samples = train_test_split(training_samples, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")

# Create datasets
train_dataset = ChurnSequenceDataset(train_samples, persona_features_matrix)
val_dataset = ChurnSequenceDataset(val_samples, persona_features_matrix)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = TemporalChurnPredictor(
    sequence_feat_dim=2,  # gain, time_delta
    global_feat_dim=4,  # days_since_last, era, month, year
    persona_feat_dim=persona_features_matrix.shape[1],
    lstm_hidden=LSTM_HIDDEN,
    num_layers=NUM_LSTM_LAYERS,
    persona_embed_dim=PERSONA_EMBED_DIM
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss functions
criterion_churn = nn.BCELoss()
criterion_ap = nn.MSELoss()
criterion_tp = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_loss = float('inf')
patience_counter = 0
EARLY_STOP_PATIENCE = 10

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_churn_loss = 0.0
    train_ap_loss = 0.0
    train_tp_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        churn_pred, ap_pred, tp_pred = model(
            batch['sequence'].to(DEVICE),
            batch['global_feat'].to(DEVICE),
            batch['persona_feat'].to(DEVICE)
        )

        # Calculate losses
        loss_churn = criterion_churn(churn_pred, batch['churned'].to(DEVICE))
        loss_ap = criterion_ap(ap_pred, batch['ap_ref'].to(DEVICE))
        loss_tp = criterion_tp(tp_pred / 100.0, batch['tp_ref'].to(DEVICE) / 100.0)  # Scale down for stability

        # Combined loss (weighted)
        loss = 2.0 * loss_churn + loss_ap + 0.1 * loss_tp  # Prioritize churn prediction

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        train_loss += loss.item()
        train_churn_loss += loss_churn.item()
        train_ap_loss += loss_ap.item()
        train_tp_loss += loss_tp.item()

    train_loss /= len(train_loader)
    train_churn_loss /= len(train_loader)
    train_ap_loss /= len(train_loader)
    train_tp_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_churn_loss = 0.0
    val_ap_loss = 0.0
    val_tp_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            churn_pred, ap_pred, tp_pred = model(
                batch['sequence'].to(DEVICE),
                batch['global_feat'].to(DEVICE),
                batch['persona_feat'].to(DEVICE)
            )

            loss_churn = criterion_churn(churn_pred, batch['churned'].to(DEVICE))
            loss_ap = criterion_ap(ap_pred, batch['ap_ref'].to(DEVICE))
            loss_tp = criterion_tp(tp_pred / 100.0, batch['tp_ref'].to(DEVICE) / 100.0)

            loss = 2.0 * loss_churn + loss_ap + 0.1 * loss_tp

            val_loss += loss.item()
            val_churn_loss += loss_churn.item()
            val_ap_loss += loss_ap.item()
            val_tp_loss += loss_tp.item()

    val_loss /= len(val_loader)
    val_churn_loss /= len(val_loader)
    val_ap_loss /= len(val_loader)
    val_tp_loss /= len(val_loader)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} (Churn: {train_churn_loss:.4f}, Ap: {train_ap_loss:.4f}, Tp: {train_tp_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (Churn: {val_churn_loss:.4f}, Ap: {val_ap_loss:.4f}, Tp: {val_tp_loss:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '../../../checkpoints/best_temporal_lstm_model.pt')
        patience_counter = 0
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: best_temporal_lstm_model.pt")

# Save training metadata
metadata = {
    'sequence_length': SEQUENCE_LENGTH,
    'persona_embed_dim': PERSONA_EMBED_DIM,
    'lstm_hidden': LSTM_HIDDEN,
    'num_layers': NUM_LSTM_LAYERS,
    'persona_features_matrix': persona_features_matrix,
    'ap_ref': Ap_ref,
    'tp_ref': Tp_ref,
}

with open('../../../checkpoints/lstm_model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Metadata saved to: lstm_model_metadata.pkl")
