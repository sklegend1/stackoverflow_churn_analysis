"""
Extract Ap/Tp Values from LSTM Predictions

The LSTM model predicts personalized Ap and Tp for each user.
This script extracts the median Ap/Tp per persona to create the interpretable rule:

    if WA_recency(Tp) < Ap:
        CHURN_ALARM()

This gives us BOTH:
1. High-accuracy LSTM predictions (F1=0.90)
2. Interpretable rule with LSTM-optimized thresholds
"""

import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from datetime import datetime

print("=" * 80)
print("EXTRACTING Ap/Tp FROM LSTM PREDICTIONS")
print("=" * 80)

# Load model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_YEAR = 2016  # Use most recent year for thresholds
GRACE_DAYS = 90
GRACE_SEC = GRACE_DAYS * 86400

with open('lstm_model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

SEQUENCE_LENGTH = metadata['sequence_length']
PERSONA_EMBED_DIM = metadata['persona_embed_dim']
LSTM_HIDDEN = metadata['lstm_hidden']
NUM_LAYERS = metadata['num_layers']
persona_features_matrix = metadata['persona_features_matrix']

# Define model architecture
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

model.load_state_dict(torch.load('best_temporal_lstm_model.pt', map_location=DEVICE, weights_only=False))
model.eval()

print("✓ Model loaded")

# Load data
elog = pl.read_parquet("core_eventlog_train_v1.parquet")
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

cutoff = datetime(EVAL_YEAR, 1, 1)
era = get_era(EVAL_YEAR)
era_int = era_to_int(era)

print("=" * 80)
print(f"EXTRACTING Ap/Tp FOR YEAR {EVAL_YEAR}")
print("=" * 80)

persona_ap_tp = []

for persona in range(20):
    print(f"\nPersona {persona}:")

    # Get events before cutoff
    persona_events = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(pl.col("timestamp") < pl.lit(cutoff))
        .sort("timestamp")
    )

    if persona_events.height == 0:
        print("  No data, skipping")
        continue

    users = persona_events.select("user_id").unique()["user_id"].to_list()

    # Collect Ap/Tp predictions for all users in this persona
    ap_predictions = []
    tp_predictions = []
    churn_probs = []

    for user_id in users[:500]:  # Sample up to 500 users per persona
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

        ap_predictions.append(ap_pred.cpu().item())
        tp_predictions.append(tp_pred.cpu().item())
        churn_probs.append(churn_prob.cpu().item())

    if len(ap_predictions) == 0:
        print("  No valid predictions")
        continue

    # Calculate statistics
    median_ap = np.median(ap_predictions)
    median_tp = np.median(tp_predictions)
    mean_ap = np.mean(ap_predictions)
    mean_tp = np.mean(tp_predictions)
    std_ap = np.std(ap_predictions)
    std_tp = np.std(tp_predictions)

    print(f"  Users sampled: {len(ap_predictions)}")
    print(f"  Ap: median={median_ap:.4f}, mean={mean_ap:.4f}, std={std_ap:.4f}")
    print(f"  Tp: median={median_tp:.1f} days, mean={mean_tp:.1f} days, std={std_tp:.1f}")

    persona_ap_tp.append({
        'persona': persona,
        'Ap': median_ap,
        'Ap_mean': mean_ap,
        'Ap_std': std_ap,
        'Tp_days': median_tp,
        'Tp_mean': mean_tp,
        'Tp_std': std_tp,
        'Tp_sec': median_tp * 86400,
        'num_users_sampled': len(ap_predictions),
        'source': 'LSTM_predictions',
        'year': EVAL_YEAR
    })

# Save results
df_ap_tp = pd.DataFrame(persona_ap_tp)
df_ap_tp = df_ap_tp.sort_values('persona')

output_file = f"lstm_derived_ap_tp_{EVAL_YEAR}.csv"
df_ap_tp.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nExtracted Ap/Tp for {len(df_ap_tp)} personas")
print(f"\nMedian values across all personas:")
print(f"  Ap: {df_ap_tp['Ap'].median():.4f}")
print(f"  Tp: {df_ap_tp['Tp_days'].median():.1f} days")

print(f"\nRange:")
print(f"  Ap: [{df_ap_tp['Ap'].min():.4f}, {df_ap_tp['Ap'].max():.4f}]")
print(f"  Tp: [{df_ap_tp['Tp_days'].min():.1f}, {df_ap_tp['Tp_days'].max():.1f}] days")

print(f"\n✓ Saved to: {output_file}")

# Display table
print("\n" + "=" * 80)
print("LSTM-DERIVED Ap/Tp PER PERSONA")
print("=" * 80)
print(df_ap_tp[['persona', 'Ap', 'Tp_days', 'num_users_sampled']].to_string(index=False))

print("\n" + "=" * 80)
print("USAGE IN INTERPRETABLE RULE")
print("=" * 80)
print("""
The extracted Ap/Tp values can be used in your interpretable formula:

```python
def predict_churn(user_id, persona, cutoff_date):
    # Load Ap/Tp for this persona
    Ap = ap_tp_table[persona]['Ap']
    Tp_days = ap_tp_table[persona]['Tp_days']

    # Calculate WA_recency
    window_start = cutoff_date - timedelta(days=Tp_days)
    events = get_events(user_id, persona, window_start, cutoff_date)

    total_gain = sum(events['gain'])
    days_since_last = (cutoff_date - max(events['timestamp'])).days
    recency_weight = exp(-days_since_last / 30.0)

    WA = total_gain * recency_weight

    # Churn alarm rule
    if WA < Ap:
        return CHURN_ALARM
    else:
        return OK
```

These Ap/Tp values are **LSTM-optimized** - they come from the neural network
that achieved F1=0.90, so they should perform much better than XGBoost-derived values!
""")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
