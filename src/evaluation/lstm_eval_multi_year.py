import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Multi-Year LSTM Evaluation: 2016 (old) vs 2021 (recent)

Tests LSTM performance on both problematic early years and good recent years.
"""

import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Load metadata and model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_YEARS = [2016,2017,2018,2019,2020, 2021,2022,2023]  # Old vs Recent
GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400

print("=" * 80)
print("LSTM MULTI-YEAR EVALUATION: 2016 to 2023 ")
print("=" * 80)

# Load metadata
with open('../../checkpoints/lstm_model_metadata.pkl', 'rb') as f:
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.ap_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.tp_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, sequence, global_feat, persona_feat):
        persona_embed = self.persona_embed(persona_feat)
        persona_embed = torch.relu(persona_embed)
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        sequence_encoding = h_n[-1]
        combined = torch.cat([sequence_encoding, global_feat, persona_embed], dim=1)
        shared = self.fc_shared(combined)
        churn_prob = self.churn_head(shared)
        ap_pred = self.ap_head(shared)
        tp_pred = self.tp_head(shared)
        return churn_prob, ap_pred, tp_pred

# Load model
model = TemporalChurnPredictor(
    sequence_feat_dim=2,
    global_feat_dim=4,
    persona_feat_dim=persona_features_matrix.shape[1],
    lstm_hidden=LSTM_HIDDEN,
    num_layers=NUM_LAYERS,
    persona_embed_dim=PERSONA_EMBED_DIM
).to(DEVICE)

model.load_state_dict(torch.load('../../checkpoints/best_temporal_lstm_model.pt', map_location=DEVICE, weights_only=False))
model.eval()

print("✓ Model loaded")

# Load data
elog = pl.read_parquet("../../data/processed/core_eventlog_eval_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"✓ Data loaded: {elog.height:,} events\n")

ERAS = {
    'pre_covid': (2016, 2019),
    'covid': (2020, 2021),
    'post_llm': (2022, 2024),
}

def get_era(year):
    for era_name, (start, end) in ERAS.items():
        if start <= year <= end:
            return era_name
    return 'post_llm'

def era_to_int(era):
    mapping = {'pre_covid': 0, 'covid': 1, 'post_llm': 2}
    return mapping.get(era, 2)

persona_features_tensor = torch.FloatTensor(persona_features_matrix).to(DEVICE)

all_year_results = []

# Evaluate each year
for EVAL_YEAR in EVAL_YEARS:
    print("=" * 80)
    print(f"EVALUATING YEAR {EVAL_YEAR}")
    print("=" * 80)

    cutoff = datetime(EVAL_YEAR, 1, 1)
    era = get_era(EVAL_YEAR)
    era_int = era_to_int(era)

    year_results = []

    for persona in range(20):
        # Get events before cutoff
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
            user_events = (
                persona_events
                .filter(pl.col("user_id") == user_id)
                .sort("timestamp")
            )

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
                days_since_last,
                era_int,
                cutoff.month / 12.0,
                (EVAL_YEAR - 2016) / 7.0,
            ]).unsqueeze(0).to(DEVICE)

            persona_feat = persona_features_tensor[persona].unsqueeze(0)

            # Predict
            with torch.no_grad():
                churn_prob, ap_pred, tp_pred = model(sequence_tensor, global_features, persona_feat)

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

    # Summary for this year
    df_year = pd.DataFrame(year_results)
    all_year_results.extend(year_results)

    print(f"\n{EVAL_YEAR} Summary:")
    print(f"  Personas evaluated: {len(df_year)}")
    print(f"  Total users: {df_year['num_users'].sum():,}")
    print(f"  Median F1: {df_year['f1'].median():.4f}")
    print(f"  Mean F1: {df_year['f1'].mean():.4f}")
    print(f"  F1 Std: {df_year['f1'].std():.4f}")
    print(f"  Median Precision: {df_year['precision'].median():.4f}")
    print(f"  Median Recall: {df_year['recall'].median():.4f}")
    print(f"  Rate Ratio: {(df_year['pred_rate'] / df_year['actual_rate']).median():.4f}")

# Overall comparison
print("\n" + "=" * 80)
print("COMPARISON: 2016 vs 2023")
print("=" * 80)

df_all = pd.DataFrame(all_year_results)

for year in [2016, 2023]:
    df_yr = df_all[df_all['year'] == year]
    print(f"\n{year}:")
    print(f"  Median F1: {df_yr['f1'].median():.4f}")
    print(f"  F1 Range: [{df_yr['f1'].min():.4f}, {df_yr['f1'].max():.4f}]")
    print(f"  Stability (std): {df_yr['f1'].std():.4f}")

# Save results
df_all.to_csv("../../results/evaluations/lstm_eval_2016_to_2023_only_stats.csv", index=False)
print(f"\n✓ Results saved to: ../../results/evaluations/lstm_eval_2016_to_2023_only_stats.csv")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
