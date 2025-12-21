import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import datetime as dt

print("=" * 80)
print("LSTM DIRECT Ap/Tp PREDICTOR WITH ACTIVITY DECLINE DETECTION")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load event log
elog = pl.read_parquet("core_eventlog_v1.parquet").with_columns([
    pl.col("gain").cast(pl.Float64),
    pl.col("timestamp").cast(pl.Datetime),
    pl.col("persona").cast(pl.Int32),
])

print(f"Event log: {elog.height:,} events")

# Configuration
LOOKBACK_DAYS = 365  # Long lookback to detect activity decline
GRACE_PERIOD_SEC = 180 * 86400
CHURN_CHECK_WINDOW_SEC = 365 * 86400  # Check only 1 year after grace
MAX_SEQ_LEN = 100  # Longer sequence for better pattern detection
MIN_EVENTS = 10

def create_temporal_features(elog_window):
    """
    Extract temporal features to detect activity decline
    """
    # Calculate activity per month in the window
    monthly_activity = (
        elog_window
        .with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month")
        ])
        .group_by(["user_id", "year", "month"])
        .agg([
            pl.col("gain").sum().alias("monthly_gain"),
            pl.len().alias("monthly_count")
        ])
        .sort(["user_id", "year", "month"])
    )
    
    return monthly_activity

def create_sequences_with_decline(elog, cutoff, lookback_days, grace_sec, churn_window_sec):
    """
    Create sequences with:
    - Long lookback to capture activity decline
    - Cutoff year as feature
    - Activity trend features
    - 1-year churn window after grace period
    """
    window_start = cutoff - pl.duration(days=lookback_days)
    churn_check_end = cutoff + pl.duration(seconds=grace_sec + churn_window_sec)
    
    # Activity in observation window
    win = elog.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    ).sort(["user_id", "timestamp"])
    
    # Future activity: between [cutoff + grace, cutoff + grace + 1year]
    future = (
        elog
        .filter(
            (pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec)) &
            (pl.col("timestamp") <= churn_check_end)
        )
        .group_by("user_id")
        .agg(pl.len().alias("future_cnt"))
    )
    
    # User metadata with event count
    user_meta = (
        win
        .group_by("user_id")
        .agg([
            pl.col("persona").first().alias("persona"),
            pl.len().alias("event_count"),
            pl.col("gain").sum().alias("total_gain")
        ])
        .filter(pl.col("event_count") >= MIN_EVENTS)
        .join(future, on="user_id", how="left")
        .with_columns([
            (pl.col("future_cnt").is_null() | (pl.col("future_cnt") == 0))
            .cast(pl.Int8).alias("churn")
        ])
    )
    
    eligible_users = user_meta["user_id"].to_list()
    labels_map = dict(zip(user_meta["user_id"], user_meta["churn"]))
    persona_map = dict(zip(user_meta["user_id"], user_meta["persona"]))
    total_gain_map = dict(zip(user_meta["user_id"], user_meta["total_gain"]))
    
    # Filter to eligible users
    win_eligible = win.filter(pl.col("user_id").is_in(eligible_users))
    
    # Calculate time deltas
    win_eligible = win_eligible.with_columns([
        pl.col("timestamp")
          .diff()
          .over("user_id")
          .dt.total_seconds()
          .fill_null(0)
          .alias("time_delta")
    ])
    
    # Calculate activity trend (recent vs old)
    cutoff_ts = cutoff.timestamp()
    win_eligible = win_eligible.with_columns([
        ((cutoff_ts - pl.col("timestamp").dt.timestamp()) / (86400 * 30)).alias("months_before_cutoff")
    ])
    
    pdf = win_eligible.select([
        "user_id", "gain", "time_delta", "action_type_id", "months_before_cutoff"
    ]).to_pandas()
    
    sequences = []
    labels = []
    personas = []
    user_ids_list = []
    cutoff_years = []
    WA_values = []
    
    cutoff_year = cutoff.year
    
    for user_id, group in pdf.groupby("user_id"):
        time_deltas_norm = np.log1p(group["time_delta"].values) / 10.0
        
        # Features: [gain, time_delta, action_type, recency]
        seq = np.column_stack([
            group["gain"].values,
            time_deltas_norm,
            group["action_type_id"].values / 10.0,
            group["months_before_cutoff"].values / 12.0,  # Normalize to years
        ]).astype(np.float32)
        
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
        else:
            pad_len = MAX_SEQ_LEN - len(seq)
            seq = np.vstack([np.zeros((pad_len, 4), dtype=np.float32), seq])
        
        sequences.append(seq)
        labels.append(labels_map[user_id])
        personas.append(persona_map[user_id])
        user_ids_list.append(user_id)
        cutoff_years.append(cutoff_year)
        WA_values.append(total_gain_map[user_id])
    
    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(personas, dtype=np.int64),
        np.array(cutoff_years, dtype=np.int32),
        np.array(WA_values, dtype=np.float32),
        user_ids_list
    )

class ChurnDatasetWithMetadata(Dataset):
    def __init__(self, sequences, labels, personas, cutoff_years, WA_values):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.personas = torch.LongTensor(personas)
        self.cutoff_years = torch.FloatTensor(cutoff_years)
        self.WA_values = torch.FloatTensor(WA_values)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.personas[idx],
            self.cutoff_years[idx],
            self.WA_values[idx],
            self.labels[idx]
        )

class ActivityDeclineLSTM(nn.Module):
    """
    LSTM that predicts optimal Ap and Tp based on activity patterns
    """
    def __init__(
        self,
        input_size=4,  # [gain, time_delta, action_type, recency]
        hidden_size=128,
        num_personas=20,
        persona_embed_dim=32,
        num_layers=2,
        dropout=0.3
    ):
        super(ActivityDeclineLSTM, self).__init__()
        
        # Embeddings
        self.persona_embedding = nn.Embedding(num_personas, persona_embed_dim)
        
        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2
        
        # Attention for focusing on decline patterns
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size + persona_embed_dim, lstm_output_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(lstm_output_size, 1)
        )
        
        # Feature fusion
        fusion_input_size = lstm_output_size + persona_embed_dim + 2  # +2 for cutoff_year and WA
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
        )
        
        # Separate heads for Ap and Tp prediction
        self.ap_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive Ap
        )
        
        self.tp_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive Tp
        )
        
        # Churn classification head
        self.churn_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    
    def forward(self, sequences, personas, cutoff_years, WA_values):
        batch_size = sequences.size(0)
        
        # LSTM processing
        lstm_out, _ = self.lstm(sequences)
        
        # Persona embedding
        persona_embed = self.persona_embedding(personas)
        persona_expand = persona_embed.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        
        # Attention over sequence
        combined = torch.cat([lstm_out, persona_expand], dim=2)
        attn_weights = torch.softmax(self.attention(combined), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Normalize cutoff_year (2015-2023 → 0-1)
        cutoff_norm = (cutoff_years - 2015) / 8.0
        cutoff_norm = cutoff_norm.unsqueeze(1)
        
        # Log-scale WA
        WA_log = torch.log1p(WA_values).unsqueeze(1)
        
        # Fusion
        fused = torch.cat([context, persona_embed, cutoff_norm, WA_log], dim=1)
        
        # Encode
        encoded = self.encoder(fused)
        
        # Predict Ap, Tp, and churn
        Ap_pred = self.ap_head(encoded).squeeze(1)  # Range: [0, inf)
        Tp_pred = self.tp_head(encoded).squeeze(1)  # Range: [0, inf), scale to days later
        churn_logits = self.churn_head(encoded)
        
        return Ap_pred, Tp_pred, churn_logits

def custom_loss(Ap_pred, Tp_pred, churn_logits, WA_values, labels, churn_criterion):
    """
    Multi-task loss:
    1. Churn classification loss (primary)
    2. Ap constraint: users with WA < Ap should be churners
    3. Tp constraint: reasonable range (60-270 days)
    """
    # Churn classification loss
    churn_loss = churn_criterion(churn_logits, labels)
    
    # Ap constraint loss: encourage Ap > WA for churners, Ap < WA for non-churners
    Ap_constraint = torch.where(
        labels == 1,
        torch.relu(WA_values - Ap_pred),  # Churners: WA should be < Ap
        torch.relu(Ap_pred - WA_values)   # Non-churners: WA should be > Ap
    ).mean()
    
    # Tp constraint: penalize if outside reasonable range (60-270 days in seconds)
    Tp_min = 60 * 86400
    Tp_max = 270 * 86400
    Tp_penalty = torch.relu(Tp_min - Tp_pred).mean() + torch.relu(Tp_pred - Tp_max).mean()
    
    # Combined loss
    total_loss = churn_loss + 0.3 * Ap_constraint + 0.1 * Tp_penalty
    
    return total_loss, churn_loss, Ap_constraint, Tp_penalty

def train_model(model, train_loader, val_loader, epochs=50, patience=10):
    churn_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_churn_loss = 0
        
        for sequences, personas, cutoff_years, WA_values, labels in train_loader:
            sequences = sequences.to(device)
            personas = personas.to(device)
            cutoff_years = cutoff_years.to(device)
            WA_values = WA_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            Ap_pred, Tp_pred, churn_logits = model(sequences, personas, cutoff_years, WA_values)
            
            loss, c_loss, ap_loss, tp_loss = custom_loss(
                Ap_pred, Tp_pred, churn_logits, WA_values, labels, churn_criterion
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_total_loss += loss.item()
            train_churn_loss += c_loss.item()
        
        # Validation
        model.eval()
        val_total_loss = 0
        val_churn_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, personas, cutoff_years, WA_values, labels in val_loader:
                sequences = sequences.to(device)
                personas = personas.to(device)
                cutoff_years = cutoff_years.to(device)
                WA_values = WA_values.to(device)
                labels = labels.to(device)
                
                Ap_pred, Tp_pred, churn_logits = model(sequences, personas, cutoff_years, WA_values)
                
                loss, _, _, _ = custom_loss(
                    Ap_pred, Tp_pred, churn_logits, WA_values, labels, churn_criterion
                )
                
                val_total_loss += loss.item()
                
                preds = torch.argmax(churn_logits, dim=1)
                val_churn_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        avg_train_loss = train_total_loss / len(train_loader)
        avg_val_loss = val_total_loss / len(val_loader)
        val_acc = val_churn_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, 'best_lstm_ap_tp_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.3f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss

# Main execution
print("\nGenerating dataset from time windows...")

windows = [
    (2017, 1), (2018, 1), (2019, 1),
    (2020, 1), (2021, 1), (2022, 1), (2023, 1)
]

all_sequences = []
all_labels = []
all_personas = []
all_cutoff_years = []
all_WA_values = []
all_user_ids = []

for year, month in windows:
    cutoff = dt.datetime(year, month, 1)
    print(f"Processing window: {year}-{month:02d}-01")
    
    X, y, p, cy, wa, uids = create_sequences_with_decline(
        elog, cutoff, LOOKBACK_DAYS, GRACE_PERIOD_SEC, CHURN_CHECK_WINDOW_SEC
    )
    
    print(f"  Samples: {len(X)}, Churn rate: {y.mean():.2%}")
    
    all_sequences.append(X)
    all_labels.append(y)
    all_personas.append(p)
    all_cutoff_years.append(cy)
    all_WA_values.append(wa)
    all_user_ids.extend(uids)

all_sequences = np.vstack(all_sequences)
all_labels = np.concatenate(all_labels)
all_personas = np.concatenate(all_personas)
all_cutoff_years = np.concatenate(all_cutoff_years)
all_WA_values = np.concatenate(all_WA_values)

print(f"\nTotal dataset: {len(all_sequences)} samples")
print(f"Overall churn rate: {all_labels.mean():.2%}")

# User-based stratified split
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(all_sequences, all_personas))

X_train = all_sequences[train_idx]
y_train = all_labels[train_idx]
p_train = all_personas[train_idx]
cy_train = all_cutoff_years[train_idx]
wa_train = all_WA_values[train_idx]

X_val = all_sequences[val_idx]
y_val = all_labels[val_idx]
p_val = all_personas[val_idx]
cy_val = all_cutoff_years[val_idx]
wa_val = all_WA_values[val_idx]

print(f"\nTrain: {len(X_train)} samples ({y_train.mean():.2%} churn)")
print(f"Val: {len(X_val)} samples ({y_val.mean():.2%} churn)")

# Create datasets
train_dataset = ChurnDatasetWithMetadata(X_train, y_train, p_train, cy_train, wa_train)
val_dataset = ChurnDatasetWithMetadata(X_val, y_val, p_val, cy_val, wa_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Initialize model
num_personas = int(max(all_personas)) + 1
model = ActivityDeclineLSTM(
    input_size=4,
    hidden_size=128,
    num_personas=num_personas,
    persona_embed_dim=32,
    num_layers=2,
    dropout=0.3
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

train_model(model, train_loader, val_loader, epochs=100, patience=10)

# Extract optimal Ap/Tp per persona
print("\n" + "=" * 80)
print("EXTRACTING OPTIMAL Ap/Tp PER PERSONA")
print("=" * 80)

checkpoint = torch.load('best_lstm_ap_tp_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

persona_results = []

with torch.no_grad():
    for sequences, personas, cutoff_years, WA_values, labels in val_loader:
        sequences = sequences.to(device)
        personas = personas.to(device)
        cutoff_years = cutoff_years.to(device)
        WA_values = WA_values.to(device)
        
        Ap_pred, Tp_pred, _ = model(sequences, personas, 
                                    cutoff_years,
                                      WA_values)
        
        # Move to CPU
        personas_cpu = personas.cpu().numpy()
        Ap_cpu = Ap_pred.cpu().numpy()
        Tp_cpu = Tp_pred.cpu().numpy()
        
        for p, ap, tp in zip(personas_cpu, Ap_cpu, Tp_cpu):
            persona_results.append({
                'persona': int(p),
                'Ap': float(ap),
                'Tp_sec': float(tp),
                'Tp_days': float(tp / 86400)
            })

# Aggregate per persona (median values)
df_results = pd.DataFrame(persona_results)
optimal_params = (
    df_results
    .groupby('persona')
    .agg({
        'Ap': 'median',
        'Tp_sec': 'median',
        'Tp_days': 'median'
    })
    .reset_index()
)

optimal_params.to_csv("lstm_optimal_ap_tp_per_persona.csv", index=False)

print("\nOptimal Ap/Tp per Persona:")
print(optimal_params.to_string(index=False))

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)
print("Model: best_lstm_ap_tp_model.pt")
print("Parameters: lstm_optimal_ap_tp_per_persona.csv")
