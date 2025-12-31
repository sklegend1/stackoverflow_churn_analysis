# LSTM Temporal Ap/Tp Prediction - Architecture Diagram

## Overview

This document visualizes the architecture of the Temporal LSTM model for Dynamic Ap/Tp Prediction as implemented in [lstm_temporal_ap_tp.py](lstm_temporal_ap_tp.py).

---

## Complete Model Architecture

```mermaid
flowchart TD
    %% Input Layer
    subgraph INPUT["🔵 INPUT LAYER"]
        SEQ["Event Sequence<br/>(50 timesteps)<br/>- Gain values<br/>- Time deltas"]
        GLOBAL["Global Features (4D)<br/>- Days since last event<br/>- Era (pre-COVID/COVID/post-LLM)<br/>- Month (normalized)<br/>- Year (normalized)"]
        PERSONA["Persona Features (10D)<br/>- Log reputation<br/>- Log upvotes<br/>- Log answers/questions<br/>- Duration ratio<br/>- Collectivistic/Individualistic<br/>- H/C/T flags"]
    end

    %% Embedding Layer
    subgraph EMBED["🟢 EMBEDDING LAYER"]
        PE["Persona Embedding<br/>Linear(10 → 32)<br/>+ ReLU"]
    end

    %% LSTM Processing
    subgraph LSTM_LAYER["🟡 SEQUENCE PROCESSING"]
        LSTM["Bi-Directional LSTM<br/>Input: 2D (gain, time_delta)<br/>Hidden: 128<br/>Layers: 2<br/>Dropout: 0.2"]
        LSTM_OUT["LSTM Hidden State<br/>(128D)"]
    end

    %% Concatenation
    subgraph CONCAT["🟠 FEATURE FUSION"]
        CAT["Concatenate<br/>LSTM (128) + Global (4) + Persona Embed (32)<br/>= 164D"]
    end

    %% Shared Layers
    subgraph SHARED["🔴 SHARED PROCESSING"]
        FC1["FC Layer 1<br/>Linear(164 → 256)<br/>+ ReLU<br/>+ Dropout(0.3)"]
        FC2["FC Layer 2<br/>Linear(256 → 128)<br/>+ ReLU<br/>+ Dropout(0.2)"]
    end

    %% Multi-Task Heads
    subgraph HEADS["🟣 MULTI-TASK HEADS"]
        subgraph CHURN_HEAD["Churn Prediction Head"]
            CH1["Linear(128 → 64) + ReLU"]
            CH2["Linear(64 → 1) + Sigmoid"]
            CHURN_OUT["Churn Probability<br/>[0, 1]"]
        end

        subgraph AP_HEAD["Ap Threshold Head"]
            AP1["Linear(128 → 64) + ReLU"]
            AP2["Linear(64 → 1) + Sigmoid"]
            AP_OUT["Ap Threshold<br/>[0, 1]"]
        end

        subgraph TP_HEAD["Tp Window Head"]
            TP1["Linear(128 → 64) + ReLU"]
            TP2["Linear(64 → 1) + ReLU"]
            TP_OUT["Tp Window (days)<br/>[0, ∞)"]
        end
    end

    %% Loss Functions
    subgraph LOSS["⚫ LOSS FUNCTIONS"]
        L_CHURN["BCE Loss<br/>(Weight: 2.0)"]
        L_AP["MSE Loss<br/>(Weight: 1.0)"]
        L_TP["MSE Loss<br/>(Weight: 0.1)"]
        L_TOTAL["Combined Loss<br/>2.0×Churn + Ap + 0.1×Tp"]
    end

    %% Flow
    SEQ --> LSTM
    LSTM --> LSTM_OUT
    PERSONA --> PE
    LSTM_OUT --> CAT
    GLOBAL --> CAT
    PE --> CAT
    CAT --> FC1
    FC1 --> FC2

    FC2 --> CH1
    CH1 --> CH2
    CH2 --> CHURN_OUT
    CHURN_OUT --> L_CHURN

    FC2 --> AP1
    AP1 --> AP2
    AP2 --> AP_OUT
    AP_OUT --> L_AP

    FC2 --> TP1
    TP1 --> TP2
    TP2 --> TP_OUT
    TP_OUT --> L_TP

    L_CHURN --> L_TOTAL
    L_AP --> L_TOTAL
    L_TP --> L_TOTAL

    style INPUT fill:#e3f2fd
    style EMBED fill:#c8e6c9
    style LSTM_LAYER fill:#fff9c4
    style CONCAT fill:#ffe0b2
    style SHARED fill:#ffcdd2
    style HEADS fill:#e1bee7
    style LOSS fill:#cfd8dc
```

---

## Data Flow (Training Pipeline)

```mermaid
flowchart LR
    subgraph DATA["📊 DATA PREPARATION"]
        ELOG["Event Log<br/>core_eventlog_train_v1.parquet"]
        PSTATS["Persona Statistics<br/>nmf_cluster_statistics_20.csv"]
        PARAMS["Best Params<br/>tp_ap_xgb_enhanced_best_params_WA_recency.csv"]
    end

    subgraph PROCESS["⚙️ PREPROCESSING"]
        SAMPLE["Sample Users<br/>Max 500/persona<br/>Years: 2017-2022"]
        SEQ_BUILD["Build Sequences<br/>Last 50 events<br/>Calculate labels"]
        SPLIT["Train/Val Split<br/>80% / 20%"]
    end

    subgraph TRAIN["🎯 TRAINING"]
        MODEL["Temporal LSTM Model<br/>~1.5M parameters"]
        OPT["Adam Optimizer<br/>LR: 0.001<br/>+ ReduceLROnPlateau"]
        ES["Early Stopping<br/>Patience: 10 epochs"]
    end

    subgraph OUTPUT["💾 OUTPUT"]
        BEST["best_temporal_lstm_model.pt"]
        META["lstm_model_metadata.pkl"]
    end

    ELOG --> SAMPLE
    PSTATS --> SAMPLE
    PARAMS --> SAMPLE
    SAMPLE --> SEQ_BUILD
    SEQ_BUILD --> SPLIT
    SPLIT --> MODEL
    MODEL --> OPT
    OPT --> ES
    ES --> BEST
    ES --> META
```

---

## Temporal Feature Engineering

```mermaid
flowchart TD
    subgraph USER_EVENTS["👤 User Event Sequence (Before Cutoff)"]
        E1["Event 1<br/>timestamp, gain"]
        E2["Event 2<br/>timestamp, gain"]
        DOTS["..."]
        E50["Event 50<br/>timestamp, gain"]
    end

    subgraph FEATURES["🔧 Derived Features"]
        GAINS["Gain Sequence<br/>[g₁, g₂, ..., g₅₀]"]
        DELTAS["Time Deltas<br/>[Δt₁, Δt₂, ..., Δt₅₀]<br/>(days between events)"]
        RECENCY["Recency<br/>Days from last event to cutoff"]
        ERA["Temporal Era<br/>0: pre-COVID (2016-2019)<br/>1: COVID (2020-2021)<br/>2: post-LLM (2022-2024)"]
    end

    subgraph LABEL["🎯 Label Generation"]
        CUTOFF["Cutoff Date"]
        GRACE["Grace Period<br/>90 days"]
        WINDOW["Observation Window<br/>90 days + 365 days"]
        CHECK{"Has events in<br/>observation window?"}
        CHURN_YES["Churned = 1"]
        CHURN_NO["Churned = 0"]
    end

    E1 --> GAINS
    E2 --> GAINS
    E50 --> GAINS

    E1 --> DELTAS
    E2 --> DELTAS
    E50 --> DELTAS

    E50 --> RECENCY
    CUTOFF --> RECENCY

    CUTOFF --> ERA

    CUTOFF --> GRACE
    GRACE --> WINDOW
    WINDOW --> CHECK
    CHECK -->|No events| CHURN_YES
    CHECK -->|Has events| CHURN_NO

    style USER_EVENTS fill:#e1f5fe
    style FEATURES fill:#fff3e0
    style LABEL fill:#fce4ec
```

---

## Model Input/Output Summary

### Inputs

| Input Type | Dimension | Description | Source |
|------------|-----------|-------------|--------|
| **Event Sequence** | `(batch, 50, 2)` | Gain values + time deltas for last 50 events | Event log |
| **Global Features** | `(batch, 4)` | Days since last event, era, month, year | Temporal context |
| **Persona Features** | `(batch, 10)` | Persona statistics (reputation, upvotes, etc.) | NMF cluster stats |

### Outputs

| Output | Dimension | Range | Activation | Loss Function |
|--------|-----------|-------|------------|---------------|
| **Churn Probability** | `(batch, 1)` | [0, 1] | Sigmoid | BCE (weight: 2.0) |
| **Ap Threshold** | `(batch, 1)` | [0, 1] | Sigmoid | MSE (weight: 1.0) |
| **Tp Window (days)** | `(batch, 1)` | [0, ∞) | ReLU | MSE (weight: 0.1) |

---

## Key Hyperparameters

```python
SEQUENCE_LENGTH = 50        # Number of recent events
PERSONA_EMBED_DIM = 32      # Persona embedding dimension
LSTM_HIDDEN = 128           # LSTM hidden size
NUM_LSTM_LAYERS = 2         # Number of stacked LSTM layers
BATCH_SIZE = 256            # Training batch size
LEARNING_RATE = 0.001       # Initial learning rate
NUM_EPOCHS = 50             # Maximum epochs
GRACE_DAYS = 90             # Grace period for churn label
```

---

## Training Strategy

### Multi-Task Loss Weighting

The model uses a weighted combination of losses:

```
Total Loss = 2.0 × BCE(Churn) + 1.0 × MSE(Ap) + 0.1 × MSE(Tp)
```

**Rationale:**
- **Churn prediction** (weight 2.0): Primary task, most important
- **Ap threshold** (weight 1.0): Secondary task, important for threshold optimization
- **Tp window** (weight 0.1): Tertiary task, less critical (scaled down for numerical stability)

### Regularization

- **Dropout**: 0.2 (LSTM), 0.3 (FC1), 0.2 (FC2)
- **Gradient Clipping**: Max norm = 1.0
- **Early Stopping**: Patience = 10 epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)

---

## How to Customize This Diagram

This diagram is written in **Mermaid** syntax, which you can edit as plain text:

### 1. Edit in VSCode with Mermaid Extension
```bash
# Install extension
code --install-extension bierner.markdown-mermaid
```

Then open this file and preview with `Ctrl+Shift+V` (Windows) or `Cmd+Shift+V` (Mac).

### 2. Edit Online
- Visit [Mermaid Live Editor](https://mermaid.live/)
- Copy any diagram code block
- Edit and download as SVG/PNG

### 3. Common Customizations

#### Change Node Colors
```mermaid
style NODE_ID fill:#COLOR_CODE
```

#### Add/Remove Nodes
```mermaid
NEW_NODE["Node Label<br/>Description"]
NODE_A --> NEW_NODE
```

#### Change Flow Direction
```mermaid
flowchart TD   %% Top to Down
flowchart LR   %% Left to Right
flowchart BT   %% Bottom to Top
```

#### Modify Connections
```mermaid
A --> B      %% Arrow
A --- B      %% Line
A -.-> B     %% Dotted arrow
A ==> B      %% Thick arrow
```

---

## References

- **Implementation**: [lstm_temporal_ap_tp.py](lstm_temporal_ap_tp.py)
- **Model Class**: `TemporalChurnPredictor` (lines 311-384)
- **Training Loop**: Lines 437-529
- **Feature Engineering**: Lines 155-253

---

## Model Architecture Summary (Text)

```
Input Layer:
├─ Event Sequence (50 × 2)
├─ Global Features (4D)
└─ Persona Features (10D)
   ↓
Embedding Layer:
└─ Persona Embedding: Linear(10 → 32) + ReLU
   ↓
LSTM Processing:
└─ LSTM(input=2, hidden=128, layers=2, dropout=0.2) → 128D
   ↓
Feature Fusion:
└─ Concatenate[LSTM(128) + Global(4) + Persona(32)] → 164D
   ↓
Shared Processing:
├─ FC1: Linear(164 → 256) + ReLU + Dropout(0.3)
└─ FC2: Linear(256 → 128) + ReLU + Dropout(0.2)
   ↓
Multi-Task Heads:
├─ Churn Head: FC(128→64→1) + Sigmoid → Churn Probability [0,1]
├─ Ap Head: FC(128→64→1) + Sigmoid → Ap Threshold [0,1]
└─ Tp Head: FC(128→64→1) + ReLU → Tp Window (days) [0,∞)
   ↓
Loss Functions:
└─ Combined Loss = 2.0×BCE(Churn) + 1.0×MSE(Ap) + 0.1×MSE(Tp)
```

**Total Parameters**: ~1.5 million trainable parameters

---

## Questions or Modifications?

To modify these diagrams:
1. Edit the Mermaid code blocks directly in this file
2. Use the [Mermaid Live Editor](https://mermaid.live/) for visual editing
3. Refer to [Mermaid Documentation](https://mermaid.js.org/intro/) for syntax

For questions about the model implementation, see [lstm_temporal_ap_tp.py](lstm_temporal_ap_tp.py:311-384).
