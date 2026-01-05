import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LSTM CHURN PREDICTION WITH CLASS BALANCING")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

elog = pl.read_parquet("../../../data/processed/core_eventlog_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
])

print(f"Event log: {elog.height:,} events")

def create_sequences(elog_p, cutoff, lookback_days, grace_period_sec):
    window_start = cutoff - pl.duration(days=lookback_days)
    
    win = elog_p.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    )
    
    sequences = []
    labels = []
    user_ids = []
    
    for user_id in win["user_id"].unique():
        user_events = (
            win.filter(pl.col("user_id") == user_id)
            .sort("timestamp")
        )
        
        if user_events.height < 3:
            continue
        
        pdf = user_events.to_pandas()
        
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
        
        future_activity = elog_p.filter(
            (pl.col("user_id") == user_id) &
            (pl.col("timestamp") > cutoff + pl.duration(seconds=grace_period_sec))
        ).height
        
        label = 1 if future_activity == 0 else 0
        
        sequences.append(seq)
        labels.append(label)
        user_ids.append(user_id)
    
    return np.array(sequences), np.array(labels), user_ids

class ChurnDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class ChurnLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.3):
        super(ChurnLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        out = self.fc(context)
        return out

def train_model(model, train_loader, val_loader, class_weights, epochs=30):
    # Use weighted loss for imbalanced classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_f1 = 0
    patience_counter = 0
    max_patience = 7
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_preds.extend(preds)
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        if val_labels.sum() > 0 and val_preds.sum() > 0:
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_precision = precision_score(val_labels, val_preds, zero_division=0)
            val_recall = recall_score(val_labels, val_preds, zero_division=0)
        else:
            val_f1 = 0.0
            val_precision = 0.0
            val_recall = 0.0
        
        avg_loss = train_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, F1={val_f1:.3f}, P={val_precision:.3f}, R={val_recall:.3f}")
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_f1

results = []
grace_period_sec = 180 * 86400
lookback_days = 270

windows = [(2019, 1), (2019, 7), (2020, 1), (2020, 7), (2021, 1), (2021, 7), (2022, 1)]

personas = sorted(elog["persona"].unique().to_list())

for persona in personas:
    print(f"\n{'='*60}")
    print(f"Training LSTM for Persona {persona}")
    print(f"{'='*60}")
    
    elog_p = elog.filter(pl.col("persona") == persona)
    
    if elog_p.height < 1000:
        print(f"Insufficient data")
        continue
    
    f1_scores = []
    
    for i, (year, month) in enumerate(windows[:-2]):
        cutoff = pl.datetime(year, month, 1)
        
        print(f"\nWindow {i+1}: {year}-{month:02d}-01")
        
        X, y, _ = create_sequences(elog_p, cutoff, lookback_days, grace_period_sec)
        
        churn_count = y.sum()
        non_churn_count = len(y) - churn_count
        
        if len(X) < 50 or churn_count < 5:
            print(f"  Insufficient samples (churners: {churn_count})")
            continue
        
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"  Churn rate: {y.mean():.2%} (churners: {churn_count}, non-churners: {non_churn_count})")
        
        # Compute class weights
        class_weights_np = compute_class_weight(
            'balanced',
            classes=np.array([0, 1]),
            y=y_train
        )
        class_weights = torch.FloatTensor(class_weights_np)
        
        print(f"  Class weights: [non-churn: {class_weights[0]:.2f}, churn: {class_weights[1]:.2f}]")
        
        train_dataset = ChurnDataset(X_train, y_train)
        val_dataset = ChurnDataset(X_val, y_val)
        
        # Create weighted sampler for balanced batches
        train_targets = y_train
        class_sample_counts = np.bincount(train_targets)
        weights = 1. / class_sample_counts[train_targets]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        model = ChurnLSTM().to(device)
        best_f1 = train_model(model, train_loader, val_loader, class_weights, epochs=30)
        
        f1_scores.append(best_f1)
        print(f"  Best F1: {best_f1:.3f}")
    
    if len(f1_scores) >= 3:
        results.append({
            "persona": int(persona),
            "median_f1": np.median(f1_scores),
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
            "n_windows": len(f1_scores),
            "method": "LSTM_balanced"
        })
        
        print(f"\nPersona {persona} Summary: Median F1 = {np.median(f1_scores):.3f}")

df_results = pd.DataFrame(results)
df_results.to_csv("../../../results/lstm_churn_results.csv", index=False)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(df_results.to_string(index=False))
print(f"\nSaved to: lstm_churn_results.csv")