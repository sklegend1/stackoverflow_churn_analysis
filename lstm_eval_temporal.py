"""
Evaluation Script for Temporal LSTM Model

Compares LSTM performance against XGBoost baseline on temporal churn prediction.
"""

import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_YEAR = 2023  # Unseen year for final evaluation
GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400

print("=" * 80)
print("TEMPORAL LSTM EVALUATION")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Evaluation Year: {EVAL_YEAR}")

# =============================================================================
# Load Model and Metadata
# =============================================================================

print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

# Load metadata
with open('lstm_model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

SEQUENCE_LENGTH = metadata['sequence_length']
PERSONA_EMBED_DIM = metadata['persona_embed_dim']
LSTM_HIDDEN = metadata['lstm_hidden']
NUM_LAYERS = metadata['num_layers']
persona_features_matrix = metadata['persona_features_matrix']
Ap_ref = metadata['ap_ref']
Tp_ref = metadata['tp_ref']

print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"LSTM Hidden: {LSTM_HIDDEN}")
print(f"Persona Embed Dim: {PERSONA_EMBED_DIM}")

# Define model architecture (same as training)
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

# Initialize model
model = TemporalChurnPredictor(
    sequence_feat_dim=2,
    global_feat_dim=4,
    persona_feat_dim=persona_features_matrix.shape[1],
    lstm_hidden=LSTM_HIDDEN,
    num_layers=NUM_LAYERS,
    persona_embed_dim=PERSONA_EMBED_DIM
).to(DEVICE)

# Load trained weights
model.load_state_dict(torch.load('best_temporal_lstm_model.pt', map_location=DEVICE))
model.eval()

print("✓ Model loaded successfully")

# =============================================================================
# Load Data
# =============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

elog = pl.read_parquet("core_eventlog_eval_v1.parquet")
elog = elog.with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"Event log: {elog.height:,} events")

# =============================================================================
# Helper Functions
# =============================================================================

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

# =============================================================================
# Evaluation
# =============================================================================

print("\n" + "=" * 80)
print(f"EVALUATING ON YEAR {EVAL_YEAR}")
print("=" * 80)

cutoff = datetime(EVAL_YEAR, 1, 1)
era = get_era(EVAL_YEAR)
era_int = era_to_int(era)

results = []

persona_features_tensor = torch.FloatTensor(persona_features_matrix).to(DEVICE)

for persona in range(20):
    print(f"\nPersona {persona}:")

    # Get events before cutoff for this persona
    persona_events = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(pl.col("timestamp") < pl.lit(cutoff))
        .sort("timestamp")
    )

    if persona_events.height == 0:
        print("  No events before cutoff, skipping")
        continue

    # Get unique users
    users = persona_events.select("user_id").unique()["user_id"].to_list()

    print(f"  Users: {len(users)}")

    user_predictions = []

    for user_id in users:
        # Get user's event sequence before cutoff
        user_events = (
            persona_events
            .filter(pl.col("user_id") == user_id)
            .sort("timestamp")
        )

        if user_events.height < 5:
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

        # Calculate time deltas
        time_deltas = [0.0]
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400.0
            time_deltas.append(delta)

        # Calculate recency
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

        # Prepare input tensors
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

        churn_prob = churn_prob.cpu().item()
        ap_pred = ap_pred.cpu().item()
        tp_pred = tp_pred.cpu().item()

        # Predict using both LSTM probability and LSTM-predicted Ap
        pred_lstm_prob = 1 if churn_prob > 0.5 else 0

        user_predictions.append({
            'user_id': user_id,
            'churned_actual': churned,
            'churn_prob_lstm': churn_prob,
            'ap_pred_lstm': ap_pred,
            'tp_pred_lstm': tp_pred,
            'pred_lstm': pred_lstm_prob,
        })

    if len(user_predictions) == 0:
        continue

    # Calculate metrics for this persona
    df_pred = pd.DataFrame(user_predictions)

    y_true = df_pred['churned_actual'].values
    y_pred_lstm = df_pred['pred_lstm'].values

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_lstm, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, df_pred['churn_prob_lstm'].values)
    except:
        auc = 0.0

    actual_rate = y_true.mean()
    pred_rate = y_pred_lstm.mean()

    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Actual churn: {actual_rate:.4f}, Predicted: {pred_rate:.4f}")
    print(f"  Avg Ap pred: {df_pred['ap_pred_lstm'].mean():.4f}")
    print(f"  Avg Tp pred: {df_pred['tp_pred_lstm'].mean():.1f} days")

    results.append({
        'persona': persona,
        'num_users': len(df_pred),
        'f1_lstm': f1,
        'precision_lstm': precision,
        'recall_lstm': recall,
        'auc_lstm': auc,
        'actual_churn_rate': actual_rate,
        'pred_churn_rate': pred_rate,
        'avg_ap_pred': df_pred['ap_pred_lstm'].mean(),
        'avg_tp_pred': df_pred['tp_pred_lstm'].mean(),
        'ap_ref': Ap_ref.get(persona, np.nan),
        'tp_ref_days': Tp_ref.get(persona, np.nan) / 86400.0,
    })

# =============================================================================
# Summary
# =============================================================================

df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("LSTM EVALUATION SUMMARY")
print("=" * 80)

print(f"\nPersonas evaluated: {len(df_results)}")
print(f"Total users: {df_results['num_users'].sum():,}")

print(f"\n**LSTM Performance:**")
print(f"  Median F1: {df_results['f1_lstm'].median():.4f}")
print(f"  Mean F1: {df_results['f1_lstm'].mean():.4f}")
print(f"  F1 Std: {df_results['f1_lstm'].std():.4f}")
print(f"  Median Precision: {df_results['precision_lstm'].median():.4f}")
print(f"  Median Recall: {df_results['recall_lstm'].median():.4f}")
print(f"  Median AUC: {df_results['auc_lstm'].median():.4f}")

print(f"\nChurn Rate Analysis:")
print(f"  Avg Actual: {df_results['actual_churn_rate'].mean():.4f}")
print(f"  Avg Predicted: {df_results['pred_churn_rate'].mean():.4f}")
print(f"  Median Rate Ratio: {(df_results['pred_churn_rate'] / df_results['actual_churn_rate']).median():.4f}")

print(f"\nAp/Tp Predictions:")
print(f"  Avg Ap predicted: {df_results['avg_ap_pred'].mean():.4f}")
print(f"  Avg Tp predicted: {df_results['avg_tp_pred'].mean():.1f} days")

# Save results
df_results.to_csv(f"lstm_eval_results_{EVAL_YEAR}.csv", index=False)
print(f"\n✓ Results saved to: lstm_eval_results_{EVAL_YEAR}.csv")

# =============================================================================
# Comparison with XGBoost
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON WITH XGBOOST BASELINE")
print("=" * 80)

# Load XGBoost results (if available)
try:
    xgb_results = pd.read_csv("ap_tp_simple_eval.csv")

    print(f"\n**XGBoost Baseline:**")
    print(f"  Median F1: {xgb_results['f1'].median():.4f}")
    print(f"  Mean F1: {xgb_results['f1'].mean():.4f}")
    print(f"  F1 Std: {xgb_results['f1'].std():.4f}")

    print(f"\n**Improvement:**")
    f1_improvement = df_results['f1_lstm'].median() - xgb_results['f1'].median()
    print(f"  F1 Improvement: {f1_improvement:+.4f} ({f1_improvement/xgb_results['f1'].median()*100:+.1f}%)")

    std_improvement = xgb_results['f1'].std() - df_results['f1_lstm'].std()
    print(f"  Stability Improvement: {std_improvement:+.4f} (lower std is better)")

except FileNotFoundError:
    print("  XGBoost baseline results not found (ap_tp_simple_eval.csv)")
    print("  Run simple_eval.py first to generate baseline")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
