import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# lstm_churn_with_persona_v2.py
# ------------------------------------------------------------
# Improvements over v1:
# 1) Persona-wise calibration (Platt scaling) to fix over-prediction
# 2) Explicit eligibility filtering (only users active in lookback window)
# 3) Dynamic lookback based on persona p95_gap
# 4) Class imbalance handling via pos_weight
# 5) Clear separation: train / inference / evaluation-ready outputs
# ------------------------------------------------------------

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CORE_EVENTLOG = "../../../data/processed/core_eventlog_v1.parquet"
CUTOFF_DATE = datetime(2019, 3, 1)
GRACE_DAYS = 270
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
elog = pl.read_parquet(CORE_EVENTLOG)
elog = elog.with_columns([
    pl.col("timestamp").cast(pl.Datetime),
    pl.col("gain").cast(pl.Float32),
])

# ------------------------------------------------------------
# Persona-level p95 gap → dynamic lookback
# ------------------------------------------------------------
p95_gap = (
    elog.filter(pl.col("gap_time_key_actions_seconds") > 0)
        .group_by("persona")
        .agg(pl.col("gap_time_key_actions_seconds").quantile(0.95).alias("p95_gap").cast(pl.Float32))
)

LOOKBACK_MULT = 4.0
LOOKBACK_MAP = {
    int(r["persona"]): int(r["p95_gap"] * LOOKBACK_MULT)
    for r in p95_gap.to_dicts()
}

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class PersonaSequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame, persona: int, cutoff: datetime):
        self.persona = persona
        self.lookback = LOOKBACK_MAP[persona]

        start = cutoff - pd.to_timedelta(self.lookback, unit="s")
        end = cutoff

        # Eligibility: only users active in lookback window
        df_p = df.filter((pl.col("persona") == persona))
        active_users = (
            df_p.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
                .select("user_id").unique()
        )

        df_p = df_p.join(active_users, on="user_id", how="inner")

        self.users = df_p.select("user_id").unique().to_series().to_list()
        self.df = df_p
        self.cutoff = cutoff

        # Build labels (real churn)
        after = df_p.filter(pl.col("timestamp") > cutoff + pd.Timedelta(days=GRACE_DAYS))
        self.churned = set(self.users) - set(after.select("user_id").to_series().to_list())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        hist = (
            self.df.filter(pl.col("user_id") == uid)
                   .sort("timestamp")
        )

        x = hist.select("gain").to_numpy().astype(np.float32)
        y = 1 if uid in self.churned else 0
        return torch.tensor(x), torch.tensor(y)

# ------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------
class LSTMChurn(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B,T,1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

# ------------------------------------------------------------
# Training + Evaluation per persona
# ------------------------------------------------------------
rows = []

for persona in LOOKBACK_MAP.keys():
    print(f"=== Persona {persona} ===")

    ds = PersonaSequenceDataset(elog, persona, CUTOFF_DATE)
    if len(ds) < 50:
        continue

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Class imbalance handling
    labels = [ds[i][1].item() for i in range(len(ds))]
    pos_weight = torch.tensor([(len(labels) - sum(labels)) / (sum(labels) + 1e-6)]).to(DEVICE)

    model = LSTMChurn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ----------------- TRAIN -----------------
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.float().to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    # ----------------- INFERENCE -----------------
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            logit = model(x.unsqueeze(0).to(DEVICE))
            prob = torch.sigmoid(logit).item()
            probs.append(prob)
            ys.append(y.item())

    probs = np.array(probs)
    ys = np.array(ys)

    # Persona-wise calibration (Platt scaling)
    calib = LogisticRegression(max_iter=1000)
    calib.fit(probs.reshape(-1, 1), ys)
    probs_cal = calib.predict_proba(probs.reshape(-1, 1))[:, 1]
    # ----------------- SAVE MODEL -----------------
    model_path = f"saved_models/lstm_persona_{persona}.pt"
    calib_path = f"saved_models/calib_persona_{persona}.pkl"

    torch.save({
        "persona": persona,
        "model_state": model.state_dict(),
        "lookback_sec": LOOKBACK_MAP[persona],
        "cutoff": CUTOFF_DATE,
        "grace_days": GRACE_DAYS,
    }, model_path)

    import pickle
    with open(calib_path, "wb") as f:
        pickle.dump(calib, f)

    

    # Metrics
    auc = roc_auc_score(ys, probs_cal) if len(np.unique(ys)) > 1 else np.nan
    prec = precision_score(ys, probs_cal > 0.5, zero_division=0)
    rec = recall_score(ys, probs_cal > 0.5, zero_division=0)

    rows.append({
        "persona": persona,
        "num_users": len(ys),
        "churn_rate": ys.mean(),
        "pred_rate": (probs_cal > 0.5).mean(),
        "auc": auc,
        "precision": prec,
        "recall": rec,
    })

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
out = pd.DataFrame(rows)
out.to_csv("../../../results/evaluations/lstm_simple_eval_results_v2.csv", index=False)
print("Saved → lstm_simple_eval_results_v2.csv")
