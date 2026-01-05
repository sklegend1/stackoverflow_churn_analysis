import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LSTM CHURN PREDICTION WITH PERSONA EMBEDDING")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

elog = pl.read_parquet("../../../data/processed/core_eventlog_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
    pl.col("persona").cast(pl.Int32),
])

print(f"Event log: {elog.height:,} events")
print(f"Personas: {sorted(elog['persona'].unique().to_list())}")

def create_sequences_with_persona(elog, cutoff, lookback_days, grace_period_sec):
    """
    Create sequences with persona information
    """
    window_start = cutoff - pl.duration(days=lookback_days)
    
    win = elog.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    )
    
    sequences = []
    labels = []
    personas = []
    user_ids = []
    
    for user_id in win["user_id"].unique():
        user_events = (
            win.filter(pl.col("user_id") == user_id)
            .sort("timestamp")
        )
        
        if user_events.height < 5:
            continue
        
        pdf = user_events.to_pandas()
        user_persona = int(pdf["persona"].iloc[0])
        
        time_deltas = pdf["timestamp"].diff().dt.total_seconds().fillna(0).values
        time_deltas_norm = np.log1p(time_deltas) / 10.0
        
        seq = np.column_stack([
            pdf["gain"].values,
            time_deltas_norm,
            pdf["action_type_id"].values / 10.0,
        ])
        
        max_len = 50
        if len(seq) > max_len:
            seq = seq[-max_len:]
        else:
            pad_len = max_len - len(seq)
            seq = np.vstack([np.zeros((pad_len, 3)), seq])
        
        future_activity = elog.filter(
            (pl.col("user_id") == user_id) &
            (pl.col("timestamp") > cutoff + pl.duration(seconds=grace_period_sec))
        ).height
        
        label = 1 if future_activity == 0 else 0
        
        sequences.append(seq)
        labels.append(label)
        personas.append(user_persona)
        user_ids.append(user_id)
    
    return np.array(sequences), np.array(labels), np.array(personas), user_ids

class ChurnDataset(Dataset):
    def __init__(self, sequences, labels, personas):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.personas = torch.LongTensor(personas)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.personas[idx], self.labels[idx]

class PersonaAwareLSTM(nn.Module):
    def __init__(
        self,
        input_size=3,
        hidden_size=128,
        num_personas=20,
        persona_embed_dim=32,
        num_layers=3,
        dropout=0.3
    ):
        super(PersonaAwareLSTM, self).__init__()
        
        self.persona_embedding = nn.Embedding(num_personas, persona_embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size + persona_embed_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + persona_embed_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    
    def forward(self, x, persona):
        batch_size = x.size(0)
        
        lstm_out, _ = self.lstm(x)
        
        persona_embed = self.persona_embedding(persona)
        persona_expand = persona_embed.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        
        combined = torch.cat([lstm_out, persona_expand], dim=2)
        
        attn_weights = self.attention(combined)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        final_features = torch.cat([context, persona_embed], dim=1)
        
        out = self.fc(final_features)
        return out

def train_model(model, train_loader, val_loader, class_weights, epochs=50, patience=10):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for sequences, personas, labels in train_loader:
            sequences = sequences.to(device)
            personas = personas.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, personas)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        val_personas = []
        
        with torch.no_grad():
            for sequences, personas, labels in val_loader:
                sequences = sequences.to(device)
                personas = personas.to(device)
                
                outputs = model(sequences, personas)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_preds.extend(preds)
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())
                val_personas.extend(personas.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_probs = np.array(val_probs)
        
        if val_labels.sum() > 0 and val_preds.sum() > 0:
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_precision = precision_score(val_labels, val_preds, zero_division=0)
            val_recall = recall_score(val_labels, val_preds, zero_division=0)
            val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        else:
            val_f1 = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_auc = 0.5
        
        avg_loss = train_loss / len(train_loader)
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), '../../../checkpoints/best_lstm_churn_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  F1: {val_f1:.3f}, P: {val_precision:.3f}, R: {val_recall:.3f}, AUC: {val_auc:.3f}")
            
            per_persona_f1 = {}
            for p in np.unique(val_personas):
                mask = np.array(val_personas) == p
                if mask.sum() > 0:
                    p_f1 = f1_score(
                        np.array(val_labels)[mask],
                        np.array(val_preds)[mask],
                        zero_division=0
                    )
                    per_persona_f1[int(p)] = p_f1
            
            print(f"  Per-persona F1 (top 5): {sorted(per_persona_f1.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_f1

grace_period_sec = 270 * 86400
lookback_days = 270

windows = [
    (2017, 1), (2017, 7),
    (2018, 1), (2018, 7),
    (2019, 1), (2019, 7),
    (2020, 1), (2020, 7),
    (2021, 1), (2021, 7),
    (2022, 1), (2022, 7),
    (2023, 1), (2023, 7),
    
]

print(f"\nTraining on {len(windows)} time windows")

all_sequences = []
all_labels = []
all_personas = []

for year, month in windows:
    cutoff = pl.datetime(year, month, 1)
    print(f"\nProcessing window: {year}-{month:02d}-01")
    
    X, y, p, _ = create_sequences_with_persona(elog, cutoff, lookback_days, grace_period_sec)
    
    print(f"  Samples: {len(X)}, Churners: {y.sum()} ({y.mean():.2%})")
    
    all_sequences.append(X)
    all_labels.append(y)
    all_personas.append(p)

all_sequences = np.vstack(all_sequences)
all_labels = np.concatenate(all_labels)
all_personas = np.concatenate(all_personas)

print(f"\nTotal dataset: {len(all_sequences)} samples")
print(f"Churn rate: {all_labels.mean():.2%}")
print(f"Persona distribution:")
unique, counts = np.unique(all_personas, return_counts=True)
for p, c in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  Persona {p}: {c} samples")

split_idx = int(0.8 * len(all_sequences))
X_train, X_val = all_sequences[:split_idx], all_sequences[split_idx:]
y_train, y_val = all_labels[:split_idx], all_labels[split_idx:]
p_train, p_val = all_personas[:split_idx], all_personas[split_idx:]

print(f"\nTrain: {len(X_train)} samples, {y_train.sum()} churners")
print(f"Val: {len(X_val)} samples, {y_val.sum()} churners")

class_weights_np = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights = torch.FloatTensor(class_weights_np)
print(f"\nClass weights: [non-churn: {class_weights[0]:.2f}, churn: {class_weights[1]:.2f}]")

train_dataset = ChurnDataset(X_train, y_train, p_train)
val_dataset = ChurnDataset(X_val, y_val, p_val)

from torch.utils.data import WeightedRandomSampler
train_targets = y_train
class_sample_counts = np.bincount(train_targets)
weights = 1. / class_sample_counts[train_targets]
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=64)

num_personas = len(np.unique(all_personas))
print(f"\nBuilding model for {num_personas} personas...")

model = PersonaAwareLSTM(
    input_size=3,
    hidden_size=128,
    num_personas=num_personas,
    persona_embed_dim=32,
    num_layers=3,
    dropout=0.3
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_f1 = train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    epochs=100,
    patience=10
)

print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

model.load_state_dict(torch.load('../../../checkpoints/best_lstm_churn_model.pt'))
model.eval()

final_preds = []
final_probs = []
final_labels = []
final_personas = []

with torch.no_grad():
    for sequences, personas, labels in val_loader:
        sequences = sequences.to(device)
        personas = personas.to(device)
        
        outputs = model(sequences, personas)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        final_preds.extend(preds)
        final_probs.extend(probs)
        final_labels.extend(labels.numpy())
        final_personas.extend(personas.cpu().numpy())

final_preds = np.array(final_preds)
final_labels = np.array(final_labels)
final_probs = np.array(final_probs)
final_personas = np.array(final_personas)

overall_f1 = f1_score(final_labels, final_preds)
overall_precision = precision_score(final_labels, final_preds)
overall_recall = recall_score(final_labels, final_preds)
overall_auc = roc_auc_score(final_labels, final_probs)

print(f"\nOverall Performance:")
print(f"  F1: {overall_f1:.3f}")
print(f"  Precision: {overall_precision:.3f}")
print(f"  Recall: {overall_recall:.3f}")
print(f"  AUC: {overall_auc:.3f}")

print(f"\nPer-Persona Performance:")
per_persona_results = []
for p in sorted(np.unique(final_personas)):
    mask = final_personas == p
    if mask.sum() > 10:
        p_f1 = f1_score(final_labels[mask], final_preds[mask], zero_division=0)
        p_recall = recall_score(final_labels[mask], final_preds[mask], zero_division=0)
        p_precision = precision_score(final_labels[mask], final_preds[mask], zero_division=0)
        
        per_persona_results.append({
            'persona': int(p),
            'samples': int(mask.sum()),
            'f1': p_f1,
            'real_churn_count': int(final_labels[mask].sum()),
            'predicted_churn_count': int(final_preds[mask].sum()),
            'precision': p_precision,
            'recall': p_recall
        })
        
        print(f"  Persona {p}: F1={p_f1:.3f}, P={p_precision:.3f}, R={p_recall:.3f} (n={mask.sum()})")

df_results = pd.DataFrame(per_persona_results)
df_results.to_csv("../../../results/lstm_persona_aware_results.csv", index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)
print("Model: best_lstm_churn_model.pt")
print("Results: lstm_persona_aware_results.csv")
