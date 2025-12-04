# train_tp_ap_pytorch.py

import polars as pl
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# -----------------------------
# Config
# -----------------------------
CORE_EVENTLOG_PATH = "core_eventlog_v1.parquet"
SEED_LABELS_PATH = "seed_churn_labels_per_user.parquet"
EVAL_TIME = pl.datetime(2019, 1, 1)  # evaluation cutoff
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-3
TAU = 86400.0  # softness (in seconds) for the time window mask


# -----------------------------
# Data preparation with Polars
# -----------------------------
def build_dataset():
    # Load core event log (already filtered to 7 core activities per persona)
    elog = pl.read_parquet(CORE_EVENTLOG_PATH)
    elog = elog.with_columns([
    pl.col("gap_time_key_actions_seconds").cast(pl.Float64)
    ])
    # Load churn seeds per (user, persona)
    labels = pl.read_parquet(SEED_LABELS_PATH)

    # Keep only events before evaluation time and join labels
    elog = (
        elog
        .filter(pl.col("timestamp") <= EVAL_TIME)
        .join(
            labels.select(["user_id", "persona", "churn_seed"]),
            on=["user_id", "persona"],
            how="inner",
        )
    )

    # Compute time distance to evaluation time (in seconds)
    elog = elog.with_columns(
        (EVAL_TIME - pl.col("timestamp"))
        .dt.total_seconds()
        .cast(pl.Float32)
        .alias("dt_sec")
    ).filter(pl.col("dt_sec") >= 0)

    # Build persona and action dictionaries
    persona_ids = (
        elog.select("persona").unique().sort("persona")["persona"].to_list()
    )
    action_ids = (
        elog.select("action_type_id").unique().sort("action_type_id")["action_type_id"].to_list()
    )

    persona_to_idx = {p: i for i, p in enumerate(persona_ids)}
    action_to_idx = {a: i for i, a in enumerate(action_ids)}

    # Warm-start weights from mean gain per (persona, action_type_id)
    gain_stats = (
        elog
        .group_by(["persona", "action_type_id"])
        .agg(pl.col("gain").mean().alias("mean_gain"))
        .to_pandas()
    )

    P = len(persona_ids)
    A = len(action_ids)
    W_init = np.zeros((P, A), dtype=np.float32)

    for _, row in gain_stats.iterrows():
        p = row["persona"]
        a = row["action_type_id"]
        pi = persona_to_idx[p]
        ai = action_to_idx[a]
        W_init[pi, ai] = 0.05 #row["mean_gain"]

    if W_init.max() > 0:
        W_init = W_init / (W_init.max() + 1e-8)

    # Aggregate events per (persona, user) into lists
    agg = (
        elog
        .select(
            "persona",
            "user_id",
            "churn_seed",
            "dt_sec",
            "action_type_id",
        )
        .group_by(["persona", "user_id", "churn_seed"])
        .agg([
            pl.col("dt_sec").alias("dt_list"),
            pl.col("action_type_id").alias("action_list"),
        ])
        .to_pandas()
    )

    return agg, persona_ids, action_ids, persona_to_idx, action_to_idx, W_init


# -----------------------------
# PyTorch Dataset
# -----------------------------
class CoreEventDataset(Dataset):
    def __init__(self, df, persona_to_idx, action_to_idx):
        self.samples = []
        for _, row in df.iterrows():
            persona = row["persona"]
            user_id = row["user_id"]
            y = int(row["churn_seed"])

            dt_list = np.array(row["dt_list"], dtype=np.float32)
            act_raw = row["action_list"]
            act_idx = np.array(
                [action_to_idx[a] for a in act_raw],
                dtype=np.int64
            )

            p_idx = persona_to_idx[persona]
            self.samples.append((p_idx, dt_list, act_idx, y, persona, user_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_idx, dt_list, act_idx, y, persona, user_id = self.samples[idx]
        return p_idx, dt_list, act_idx, y, persona, user_id


def collate_fn(batch):
    persona_idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    y = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    personas = [b[4] for b in batch]
    user_ids = [b[5] for b in batch]

    dt_list = [torch.from_numpy(b[1]).float() for b in batch]
    act_list = [torch.from_numpy(b[2]).long() for b in batch]

    return persona_idx, dt_list, act_list, y, personas, user_ids


# -----------------------------
# Model: Tp & Ap & w_{p,a}
# -----------------------------
class TpApModel(nn.Module):
    def __init__(self, num_personas, num_actions, w_init, tau=86400.0):
        super().__init__()
        self.tau = tau

        # Raw parameters (unconstrained). We'll map them to positive in forward.
        # Initialize T with something like 7 days (in seconds)
        init_T = np.full((num_personas,), 7 * 86400.0, dtype=np.float32)
        self.log_T_raw = nn.Parameter(torch.log(torch.from_numpy(init_T)))

        # Initialize Ap with small positive values
        init_A = np.full((num_personas,), 1.0, dtype=np.float32)
        self.A_raw = nn.Parameter(torch.from_numpy(init_A))

        # Warm-start action weights from w_init; enforce positivity via softplus
        # To invert softplus approximately for initialization, use log(expm1(x))
        w_init_clipped = np.clip(w_init, 1e-6, None)
        w_raw_init = np.log(np.expm1(w_init_clipped))
        self.W_raw = nn.Parameter(torch.from_numpy(w_raw_init))

    def forward(self, persona_idx, dt_list, act_list):
        """
        persona_idx: (B,)
        dt_list: list of length B, each tensor (Ni,) seconds since eval time
        act_list: list of length B, each tensor (Ni,) action indices
        """
        B = len(dt_list)
        device = self.W_raw.device

        persona_idx = persona_idx.to(device)

        Tp_all = torch.nn.functional.softplus(self.log_T_raw)   # (P,)
        Ap_all = torch.nn.functional.softplus(self.A_raw)       # (P,)
        W_all = torch.nn.functional.softplus(self.W_raw)        # (P, A)

        logits = []

        for b in range(B):
            p = persona_idx[b]

            T_p = Tp_all[p]          # scalar
            A_p = Ap_all[p]          # scalar
            W_p = W_all[p]           # (A,)

            dt = dt_list[b].to(device)   # (Ni,)
            acts = act_list[b].to(device)  # (Ni,)

            # Soft time window mask: approx 1 if dt < T_p, approx 0 if dt >> T_p
            m = torch.sigmoid((T_p - dt) / self.tau)   # (Ni,)

            w_events = W_p[acts]      # (Ni,)
            WA = torch.sum(w_events * m)  # scalar

            # logit = Ap - WeightedActivity  (churn if WA < Ap)
            logit = A_p - WA
            logits.append(logit)

        logits = torch.stack(logits)  # (B,)
        return logits, Tp_all, Ap_all


# -----------------------------
# Training loop
# -----------------------------
def train():
    df, persona_ids, action_ids, persona_to_idx, action_to_idx, W_init = build_dataset()

    num_personas = len(persona_ids)
    num_actions = len(action_ids)

    dataset = CoreEventDataset(df, persona_to_idx, action_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TpApModel(num_personas, num_actions, W_init, tau=TAU).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(NUM_EPOCHS):
        all_logits = []
        all_targets = []
        epoch_loss = 0.0

        for persona_idx, dt_list, act_list, y, _, _ in loader:
            optimizer.zero_grad()
            logits, _, _ = model(persona_idx, dt_list, act_list)
            y = y.to(logits.device)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

        all_logits = torch.cat(all_logits).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # AUC for monitoring (on train set)
        try:
            auc = roc_auc_score(all_targets, 1 / (1 + np.exp(-all_logits)))
        except ValueError:
            auc = float("nan")

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss={epoch_loss:.4f} - AUC={auc:.4f}")

    # After training, extract Tp and Ap per persona and save CSV
    model.eval()
    with torch.no_grad():
        Tp_all = torch.nn.functional.softplus(model.log_T_raw).cpu().numpy()
        Ap_all = torch.nn.functional.softplus(model.A_raw).cpu().numpy()

    out_rows = []
    for i, persona in enumerate(persona_ids):
        out_rows.append({
            "persona": persona,
            "Tp_sec": float(Tp_all[i]),
            "Ap": float(Ap_all[i]),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("tp_ap_estimates_pytorch.csv", index=False)
    print("Saved → tp_ap_estimates_pytorch.csv")


if __name__ == "__main__":
    train()
