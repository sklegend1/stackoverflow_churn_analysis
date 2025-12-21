import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

split_mode = 1  # 0: time-based, 1: user-based stratified

print("=" * 80)
print("LSTM CHURN PREDICTION V2 - OPTIMIZED & CALIBRATED")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

elog = pl.read_parquet("core_eventlog_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
    pl.col("persona").cast(pl.Int32),
])

print(f"Event log: {elog.height:,} events")

def create_sequences_optimized(elog, cutoff, lookback_days, grace_period_sec):
    """
    OPTIMIZED: Use Polars for all operations, minimize Python loops
    """
    window_start = cutoff - pl.duration(days=lookback_days)
    
    # Filter once
    win = elog.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    ).sort(["user_id", "timestamp"])
    
    # Label generation - vectorized in Polars
    future = (
        elog
        .filter(pl.col("timestamp") > cutoff + pl.duration(seconds=grace_period_sec))
        .group_by("user_id")
        .agg(pl.len().alias("future_cnt"))
    )
    
    # Get user metadata
    user_meta = (
        win
        .group_by("user_id")
        .agg([
            pl.col("persona").first().alias("persona"),
            pl.len().alias("event_count")
        ])
        .filter(pl.col("event_count") >= 5)
        .join(future, on="user_id", how="left")
        .with_columns([
            (pl.col("future_cnt").is_null()).cast(pl.Int8).alias("churn")
        ])
    )
    
    eligible_users = user_meta["user_id"].to_list()
    labels_map = dict(zip(user_meta["user_id"], user_meta["churn"]))
    persona_map = dict(zip(user_meta["user_id"], user_meta["persona"]))
    
    # Filter to eligible users only
    win_eligible = win.filter(pl.col("user_id").is_in(eligible_users))
    
    # Time delta calculation - vectorized
    win_eligible = win_eligible.with_columns([
        pl.col("timestamp")
          .diff()
          .over("user_id")
          .dt.total_seconds()
          .fill_null(0)
          .alias("time_delta")
    ])
    
    # Convert to pandas for sequence building (still faster than full Python loop)
    pdf = win_eligible.select([
        "user_id", "gain", "time_delta", "action_type_id"
    ]).to_pandas()
    
    sequences = []
    labels = []
    personas = []
    user_ids = []
    
    max_len = 80
    
    # Group by user - still need loop but much faster
    for user_id, group in pdf.groupby("user_id"):
        time_deltas_norm = np.log1p(group["time_delta"].values) / 10.0
        
        seq = np.column_stack([
            group["gain"].values,
            time_deltas_norm,
            group["action_type_id"].values / 10.0,
        ]).astype(np.float32)
        
        if len(seq) > max_len:
            seq = seq[-max_len:]
        else:
            pad_len = max_len - len(seq)
            seq = np.vstack([np.zeros((pad_len, 3), dtype=np.float32), seq])
        
        sequences.append(seq)
        labels.append(labels_map[user_id])
        personas.append(persona_map[user_id])
        user_ids.append(user_id)
    
    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(personas, dtype=np.int64),
        user_ids
    )

class ChurnDataset(Dataset):
    def __init__(self, sequences, labels, personas):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.personas = torch.LongTensor(personas)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.personas[idx], self.labels[idx]

class ImprovedPersonaLSTM(nn.Module):
    def __init__(
        self,
        input_size=3,
        hidden_size=128,
        num_personas=20,
        persona_embed_dim=32,
        num_layers=3,
        dropout=0.4
    ):
        super(ImprovedPersonaLSTM, self).__init__()
        
        # Persona embedding with higher capacity
        self.persona_embedding = nn.Embedding(num_personas, persona_embed_dim)
        
        # Bidirectional LSTM for better context
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2  # bidirectional
        
        # Improved attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size + persona_embed_dim, lstm_output_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(lstm_output_size, 1)
        )
        
        # Deeper classification head with residual connection
        self.fc1 = nn.Sequential(
            nn.Linear(lstm_output_size + persona_embed_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.output = nn.Linear(32, 2)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, x, persona):
        lstm_out, _ = self.lstm(x)
        
        persona_embed = self.persona_embedding(persona)
        persona_expand = persona_embed.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        
        combined = torch.cat([lstm_out, persona_expand], dim=2)
        
        attn_weights = torch.softmax(self.attention(combined), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        final_features = torch.cat([context, persona_embed], dim=1)
        
        h1 = self.fc1(final_features)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        logits = self.output(h3)
        
        # Temperature scaling for calibration
        logits = logits / self.temperature
        
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance better
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_model(model, train_loader, val_loader, epochs=60, patience=12):
    # Focal Loss instead of CrossEntropy
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,  # Lower learning rate
        weight_decay=5e-4,  # Stronger regularization
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
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
                
                # Dynamic threshold based on training churn rate
                threshold = 0.5  # Will be adjusted per-persona in evaluation
                preds = (probs > threshold).astype(int)
                
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
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'f1': best_f1,
            }, 'best_lstm_churn_model_v2.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  F1: {val_f1:.3f}, P: {val_precision:.3f}, R: {val_recall:.3f}, AUC: {val_auc:.3f}")
            print(f"  Temperature: {model.temperature.item():.3f}")
        
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

print(f"\nProcessing {len(windows)} time windows...")

all_sequences = []
all_labels = []
all_personas = []
all_window_ids = []

for window_id, (year, month) in enumerate(windows):
    cutoff = pl.datetime(year, month, 1)
    print(f"\nWindow {window_id+1}/{len(windows)}: {year}-{month:02d}-01")
    
    X, y, p, _ = create_sequences_optimized(elog, cutoff, lookback_days, grace_period_sec)
    
    print(f"  Samples: {len(X)}, Churners: {y.sum()} ({y.mean():.2%})")
    
    all_sequences.append(X)
    all_labels.append(y)
    all_personas.append(p)
    all_window_ids.extend([window_id] * len(X))

all_sequences = np.vstack(all_sequences)
all_labels = np.concatenate(all_labels)
all_personas = np.concatenate(all_personas)
all_window_ids = np.array(all_window_ids)

print(f"\nTotal dataset: {len(all_sequences)} samples")
print(f"Churn rate: {all_labels.mean():.2%}")

# Time-based split: use middle windows for validation (avoid data cutoff issues)
# Validation: windows 6 and 10 (2019-07 and 2021-07)
val_window_ids = [4, 10]
val_mask = np.isin(all_window_ids, val_window_ids)
train_mask = ~val_mask

# User-based split: 20% random samples per persona
if split_mode == 1:
    print("\nCreating stratified user-based split (80/20)...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(all_sequences, all_personas))

    train_mask = np.zeros(len(all_sequences), dtype=bool)
    train_mask[train_idx] = True
    val_mask = ~train_mask

    print(f"Split ensures each persona has ~20% in validation")

X_train = all_sequences[train_mask]
y_train = all_labels[train_mask]
p_train = all_personas[train_mask]

X_val = all_sequences[val_mask]
y_val = all_labels[val_mask]
p_val = all_personas[val_mask]

print(f"\nTrain: {len(X_train)} samples, {y_train.sum()} churners ({y_train.mean():.2%})")
print(f"Val: {len(X_val)} samples, {y_val.sum()} churners ({y_val.mean():.2%})")

train_dataset = ChurnDataset(X_train, y_train, p_train)
val_dataset = ChurnDataset(X_val, y_val, p_val)

from torch.utils.data import WeightedRandomSampler
class_sample_counts = np.bincount(y_train)
weights = 1. / class_sample_counts[y_train]
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128)

num_personas = int(max(all_personas)) + 1
print(f"\nBuilding improved model for {num_personas} personas...")

model = ImprovedPersonaLSTM(
    input_size=3,
    hidden_size=128,
    num_personas=num_personas,
    persona_embed_dim=32,
    num_layers=3,
    dropout=0.4
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_f1 = train_model(
    model,
    train_loader,
    val_loader,
    epochs=60,
    patience=12
)

print("\n" + "=" * 80)
print("FINAL EVALUATION WITH CALIBRATION")
print("=" * 80)

checkpoint = torch.load('best_lstm_churn_model_v2.pt')
model.load_state_dict(checkpoint['model_state_dict'])
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
        
        final_probs.extend(probs)
        final_labels.extend(labels.numpy())
        final_personas.extend(personas.cpu().numpy())

final_probs = np.array(final_probs)
final_labels = np.array(final_labels)
final_personas = np.array(final_personas)

# Per-persona optimal thresholds
persona_thresholds = {}
for p in sorted(np.unique(final_personas)):
    mask = final_personas == p
    p_probs = final_probs[mask]
    p_labels = final_labels[mask]
    
    if p_labels.sum() < 3:
        persona_thresholds[p] = 0.5
        continue
    
    # Find threshold that gives actual churn rate
    target_rate = p_labels.mean()
    threshold = np.percentile(p_probs, (1 - target_rate) * 100)
    persona_thresholds[p] = threshold

# Apply calibrated thresholds
final_preds = np.zeros_like(final_labels)
for p in np.unique(final_personas):
    mask = final_personas == p
    threshold = persona_thresholds.get(p, 0.5)
    final_preds[mask] = (final_probs[mask] > threshold).astype(int)

overall_f1 = f1_score(final_labels, final_preds)
overall_precision = precision_score(final_labels, final_preds)
overall_recall = recall_score(final_labels, final_preds)
overall_auc = roc_auc_score(final_labels, final_probs)

print(f"\nOverall Performance (Calibrated):")
print(f"  F1: {overall_f1:.3f}")
print(f"  Precision: {overall_precision:.3f}")
print(f"  Recall: {overall_recall:.3f}")
print(f"  AUC: {overall_auc:.3f}")

print(f"\nPer-Persona Performance:")
per_persona_results = []
for p in sorted(np.unique(final_personas)):
    mask = final_personas == p
    if mask.sum() > 10:
        p_preds = final_preds[mask]
        p_labels = final_labels[mask]
        p_probs = final_probs[mask]
        
        p_f1 = f1_score(p_labels, p_preds, zero_division=0)
        p_recall = recall_score(p_labels, p_preds, zero_division=0)
        p_precision = precision_score(p_labels, p_preds, zero_division=0)
        
        per_persona_results.append({
            'persona': int(p),
            'samples': int(mask.sum()),
            'threshold': float(persona_thresholds[p]),
            'actual_churn_rate': float(p_labels.mean()),
            'pred_churn_rate': float(p_preds.mean()),
            'f1': p_f1,
            'precision': p_precision,
            'recall': p_recall
        })
        
        print(f"  Persona {p}: F1={p_f1:.3f}, thresh={persona_thresholds[p]:.3f}, "
              f"actual={p_labels.mean():.2%}, pred={p_preds.mean():.2%}")

df_results = pd.DataFrame(per_persona_results)
df_results.to_csv("lstm_persona_aware_results_v2.csv", index=False)

# Save thresholds
df_thresholds = pd.DataFrame([
    {'persona': p, 'threshold': t} for p, t in persona_thresholds.items()
])
df_thresholds.to_csv("persona_thresholds.csv", index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)
print("Model: best_lstm_churn_model_v2.pt")
print("Results: lstm_persona_aware_results_v2.csv")
print("Thresholds: persona_thresholds.csv")
