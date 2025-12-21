import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "best_lstm_churn_model_v2.pt"
THRESHOLDS_PATH = "persona_thresholds.csv"
EVENTLOG_PATH = "core_eventlog_v1.parquet"

CUTOFF = pl.datetime(2023, 3, 1)
LOOKBACK_DAYS = 270
GRACE_SEC = 270 * 86400

MAX_LEN = 80
MIN_EVENTS = 5
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model (must match training V2)
# ----------------------------
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
        super().__init__()

        self.persona_embedding = nn.Embedding(num_personas, persona_embed_dim)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_size = hidden_size * 2

        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size + persona_embed_dim, lstm_output_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(lstm_output_size, 1)
        )

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

        logits = logits / self.temperature

        return logits


# ----------------------------
# Optimized sequence builder
# ----------------------------
def build_eval_dataset_optimized(elog: pl.DataFrame, cutoff, lookback_days, grace_sec):
    window_start = cutoff - pl.duration(days=lookback_days)

    win = elog.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    ).sort(["user_id", "timestamp"])

    future = (
        elog
        .filter(pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec))
        .group_by("user_id")
        .agg(pl.len().alias("future_cnt"))
    )

    user_meta = (
        win
        .group_by("user_id")
        .agg([
            pl.col("persona").first().alias("persona"),
            pl.len().alias("event_count")
        ])
        .filter(pl.col("event_count") >= MIN_EVENTS)
        .join(future, on="user_id", how="left")
        .with_columns([
            (pl.col("future_cnt").is_null()).cast(pl.Int8).alias("churn")
        ])
    )

    eligible_users = user_meta["user_id"].to_list()
    labels_map = dict(zip(user_meta["user_id"], user_meta["churn"]))
    persona_map = dict(zip(user_meta["user_id"], user_meta["persona"]))

    win_eligible = win.filter(pl.col("user_id").is_in(eligible_users))

    win_eligible = win_eligible.with_columns([
        pl.col("timestamp")
          .diff()
          .over("user_id")
          .dt.total_seconds()
          .fill_null(0)
          .alias("time_delta")
    ])

    pdf = win_eligible.select([
        "user_id", "gain", "time_delta", "action_type_id"
    ]).to_pandas()

    sequences = []
    labels = []
    personas = []
    user_ids = []

    for user_id, group in pdf.groupby("user_id"):
        time_deltas_norm = np.log1p(group["time_delta"].values) / 10.0

        seq = np.column_stack([
            group["gain"].values,
            time_deltas_norm,
            group["action_type_id"].values / 10.0,
        ]).astype(np.float32)

        if len(seq) > MAX_LEN:
            seq = seq[-MAX_LEN:]
        else:
            pad_len = MAX_LEN - len(seq)
            seq = np.vstack([np.zeros((pad_len, 3), dtype=np.float32), seq])

        sequences.append(seq)
        labels.append(labels_map[user_id])
        personas.append(persona_map[user_id])
        user_ids.append(int(user_id))

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    p = np.array(personas, dtype=np.int64)
    u = np.array(user_ids, dtype=np.int64)
    return X, y, p, u


def batched_predict(model, X, p, batch_size=256):
    model.eval()
    probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            pb = torch.from_numpy(p[i:i+batch_size]).to(device)

            logits = model(xb, pb)
            pr = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(pr)

    return np.concatenate(probs)


def main():
    print("=" * 80)
    print(f"LSTM V2 EVALUATION - Cutoff: {CUTOFF}")
    print("=" * 80)
    
    elog = pl.read_parquet(EVENTLOG_PATH).with_columns([
        pl.col("gain").cast(pl.Float64),
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("persona").cast(pl.Int32),
    ])

    print("\nBuilding evaluation dataset...")
    X, y_true, personas, user_ids = build_eval_dataset_optimized(
        elog, CUTOFF, LOOKBACK_DAYS, GRACE_SEC
    )

    if len(X) == 0:
        print("No eligible users found for this cutoff/window.")
        return

    print(f"Total users: {len(X)}")
    print(f"Actual churn rate: {y_true.mean():.2%}")
    
    # Load persona thresholds
    try:
        df_thresholds = pd.read_csv(THRESHOLDS_PATH)
        persona_thresholds = dict(zip(df_thresholds["persona"], df_thresholds["threshold"]))
        print(f"Loaded thresholds for {len(persona_thresholds)} personas")
    except:
        print("WARNING: Could not load persona_thresholds.csv, using default 0.5")
        persona_thresholds = {p: 0.5 for p in np.unique(personas)}

    # Load model
    num_personas = int(np.max(personas)) + 1
    model = ImprovedPersonaLSTM(num_personas=num_personas).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")

    # Predict
    print("\nPredicting...")
    prob = batched_predict(model, X, personas, batch_size=BATCH_SIZE)
    
    # Apply persona-specific thresholds
    y_pred = np.zeros_like(y_true)
    for p in np.unique(personas):
        mask = personas == p
        threshold = persona_thresholds.get(int(p), 0.5)
        y_pred[mask] = (prob[mask] >= threshold).astype(int)

    # Overall metrics
    overall_acc = float((y_true == y_pred).mean())
    overall_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    overall_precision = float(precision_score(y_true, y_pred, zero_division=0))
    overall_recall = float(recall_score(y_true, y_pred, zero_division=0))
    overall_auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {overall_acc:.4f}")
    print(f"F1:        {overall_f1:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall:    {overall_recall:.4f}")
    print(f"AUC:       {overall_auc:.4f}")
    print(f"Predicted churn rate: {y_pred.mean():.2%}")

    # Per-persona metrics
    rows = []
    print("\n" + "=" * 80)
    print("PER-PERSONA RESULTS")
    print("=" * 80)
    
    for p in sorted(np.unique(personas)):
        mask = personas == p
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        pr = prob[mask]
        
        threshold = persona_thresholds.get(int(p), 0.5)

        accuracy = float((yt == yp).mean())
        f1 = float(f1_score(yt, yp, zero_division=0))
        precision = float(precision_score(yt, yp, zero_division=0))
        recall = float(recall_score(yt, yp, zero_division=0))
        auc = float(roc_auc_score(yt, pr)) if len(np.unique(yt)) > 1 else float("nan")

        rows.append({
            "persona": int(p),
            "threshold": float(threshold),
            "num_users": int(mask.sum()),
            "actual_churn_rate": float(yt.mean()),
            "pred_churn_rate": float(yp.mean()),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        })
        
        print(f"Persona {p:2d}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}, "
              f"actual={yt.mean():.2%}, pred={yp.mean():.2%}, thresh={threshold:.3f}")

    df = pd.DataFrame(rows)
    output_file = f"lstm_v2_eval_results.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved → {output_file}")


if __name__ == "__main__":
    main()
