import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import precision_score, recall_score, roc_auc_score

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "../../checkpoints/best_lstm_churn_model.pt"
EVENTLOG_PATH = "../../data/processed/core_eventlog_v1.parquet"

CUTOFF = pl.datetime(2022, 9, 1)     # change as needed
LOOKBACK_DAYS = 270                  # must match training
GRACE_SEC = 270 * 86400              # change as needed

MAX_LEN = 50
MIN_EVENTS = 5
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model (must match training)
# ----------------------------
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
        super().__init__()

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
        lstm_out, _ = self.lstm(x)

        persona_embed = self.persona_embedding(persona)
        persona_expand = persona_embed.unsqueeze(1).expand(-1, lstm_out.size(1), -1)

        combined = torch.cat([lstm_out, persona_expand], dim=2)

        attn_weights = self.attention(combined)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        final_features = torch.cat([context, persona_embed], dim=1)
        out = self.fc(final_features)
        return out


# ----------------------------
# Sequence builder (same logic as training)
# ----------------------------
def build_eval_dataset(elog: pl.DataFrame, cutoff, lookback_days, grace_sec):
    window_start = cutoff - pl.duration(days=lookback_days)

    win = elog.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    )

    user_ids = win.select("user_id").unique().to_series().to_list()

    sequences = []
    labels = []
    personas = []
    kept_user_ids = []

    for user_id in user_ids:
        user_events = (
            win.filter(pl.col("user_id") == user_id)
               .sort("timestamp")
        )

        if user_events.height < MIN_EVENTS:
            continue

        pdf = user_events.select(["timestamp", "gain", "action_type_id", "persona"]).to_pandas()
        persona = int(pdf["persona"].iloc[0])

        time_deltas = pdf["timestamp"].diff().dt.total_seconds().fillna(0).values
        time_deltas_norm = np.log1p(time_deltas) / 10.0

        seq = np.column_stack([
            pdf["gain"].values,
            time_deltas_norm,
            (pdf["action_type_id"].values / 10.0),
        ]).astype(np.float32)

        if len(seq) > MAX_LEN:
            seq = seq[-MAX_LEN:]
        else:
            pad_len = MAX_LEN - len(seq)
            seq = np.vstack([np.zeros((pad_len, 3), dtype=np.float32), seq])

        future_cnt = elog.filter(
            (pl.col("user_id") == user_id) &
            (pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec))
        ).height

        label = 1 if future_cnt == 0 else 0

        sequences.append(seq)
        labels.append(label)
        personas.append(persona)
        kept_user_ids.append(int(user_id))

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    p = np.array(personas, dtype=np.int64)
    u = np.array(kept_user_ids, dtype=np.int64)
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
    elog = pl.read_parquet(EVENTLOG_PATH).with_columns([
        pl.col("gain").cast(pl.Float64),
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("persona").cast(pl.Int32),
    ])

    # Build dataset (eligibility is enforced here by MIN_EVENTS inside window)
    X, y_true, personas, user_ids = build_eval_dataset(elog, CUTOFF, LOOKBACK_DAYS, GRACE_SEC)

    if len(X) == 0:
        print("No eligible users found for this cutoff/window.")
        return

    num_personas = int(np.max(personas)) + 1
    model = PersonaAwareLSTM(num_personas=num_personas).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    prob = batched_predict(model, X, personas, batch_size=BATCH_SIZE)
    y_pred = (prob >= 0.5).astype(int)

    rows = []
    for p in sorted(np.unique(personas)):
        mask = personas == p
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        pr = prob[mask]

        accuracy = float((yt == yp).mean())
        precision = float(precision_score(yt, yp, zero_division=0))
        recall = float(recall_score(yt, yp, zero_division=0))
        auc = float(roc_auc_score(yt, pr)) if len(np.unique(yt)) > 1 else float("nan")

        rows.append({
            "persona": int(p),
            "num_users": int(mask.sum()),
            "actual_churn_rate": float(yt.mean()),
            "pred_churn_rate": float(yp.mean()),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        })

    df = pd.DataFrame(rows)
    df.to_csv("../../results/evaluations/lstm_simple_eval_results.csv", index=False)

    overall_acc = float((y_true == y_pred).mean())
    overall_precision = float(precision_score(y_true, y_pred, zero_division=0))
    overall_recall = float(recall_score(y_true, y_pred, zero_division=0))
    overall_auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")

    print("Saved -> lstm_simple_eval_results.csv")
    print(f"Overall: acc={overall_acc:.4f}, precision={overall_precision:.4f}, recall={overall_recall:.4f}, auc={overall_auc:.4f}")


if __name__ == "__main__":
    main()
