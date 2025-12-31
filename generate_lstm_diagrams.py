"""
ASCII & HTML Neural Network Diagram Generator for LSTM Model
Creates visual representations that can be used in documentation or as reference for manual diagrams
No external dependencies required (no matplotlib/graphviz needed)
"""

print("=" * 100)
print("GENERATING LSTM ARCHITECTURE DIAGRAMS")
print("=" * 100)

# Full ASCII diagram
diagram_ascii = r"""
════════════════════════════════════════════════════════════════════════════════════════════════════
                            LSTM TEMPORAL MODEL FOR AP/TP PREDICTION
                              Multi-Task Deep Learning Architecture
════════════════════════════════════════════════════════════════════════════════════════════════════

                                         ┌─ INPUT LAYER ─┐

         ┌────────────────┐         ┌────────────────┐         ┌────────────────┐
         │ Event Sequence │         │Global Features │         │Persona Features│
         │  [batch,50,2]  │         │   [batch, 4]   │         │  [batch, 10]   │
         │ ━━━━━━━━━━━━━━ │         │ ━━━━━━━━━━━━━━ │         │ ━━━━━━━━━━━━━━ │
         │ • Gain values  │         │ • Days since   │         │ • Reputation   │
         │ • Time deltas  │         │ • Era (0/1/2)  │         │ • Upvotes      │
         │                │         │ • Month        │         │ • Answers/Qs   │
         │ ○ ○            │         │ • Year         │         │ • Duration     │
         └────────┬───────┘         └───────┬────────┘         │ • Collect/Indv │
                  │                         │                  │ • H/C/T flags  │
                  │                         │                  │ ○ ○ ○ ○ ○     │
                  │                         │                  └───────┬────────┘
                  │                         │                          │
                  │                         │                          ▼
                  │                         │                  ┌────────────────┐
                  │                         │                  │Persona Embed   │
                  │                         │                  │  [batch, 32]   │
                  │                         │                  │ ━━━━━━━━━━━━━━ │
                  │                         │                  │Linear(10→32)   │
                  │                         │                  │    + ReLU      │
                  │                         │                  │ ○○○○○          │
                  ▼                         │                  └───────┬────────┘
         ┌────────────────┐                 │                          │
         │      LSTM      │                 │                          │
         │ [batch,50,128] │                 │                          │
         │ ━━━━━━━━━━━━━━ │                 │                          │
         │ • Input: 2D    │                 │                          │
         │ • Hidden: 128  │                 │                          │
         │ • Layers: 2    │                 │                          │
         │ • Dropout: 0.2 │                 │                          │
         │                │                 │                          │
         │ ┌─────────┐    │                 │                          │
         │ │ h₀ c₀   │    │                 │                          │
         │ │   ↓     │    │                 │                          │
         │ │ [LSTM]  │    │ Layer 1         │                          │
         │ │   ↓     │    │                 │                          │
         │ │ h₁ c₁   │    │                 │                          │
         │ │   ↓     │    │                 │                          │
         │ │ [LSTM]  │    │ Layer 2         │                          │
         │ │   ↓     │    │                 │                          │
         │ │ h₂ c₂   │    │                 │                          │
         │ └─────────┘    │                 │                          │
         │ ○○○○○○○○       │                 │                          │
         └────────┬───────┘                 │                          │
                  │                         │                          │
                  ▼                         │                          │
         ┌────────────────┐                 │                          │
         │ LSTM Hidden    │                 │                          │
         │  [batch, 128]  │                 │                          │
         │ ━━━━━━━━━━━━━━ │                 │                          │
         │ Extract h_n[-1]│                 │                          │
         │ (last output)  │                 │                          │
         │ ○○○○○○○○       │                 │                          │
         └────────┬───────┘                 │                          │
                  │                         │                          │
                  │            128D         │  4D                 32D  │
                  └──────────┬──────────────┴──────────┬────────────────┘
                             │                         │
                             ▼                         │
                    ┌────────────────────┐             │
                    │  CONCATENATE       │◄────────────┘
                    │   [batch, 164]     │
                    │  ━━━━━━━━━━━━━━━━  │
                    │ 128 + 4 + 32 = 164 │
                    │ ○○○○○○○○           │
                    └─────────┬──────────┘
                              │
                              │ 164D
                              ▼
                    ┌────────────────────┐
                    │ Fully Connected 1  │
                    │   [batch, 256]     │
                    │  ━━━━━━━━━━━━━━━━  │
                    │ Linear(164 → 256)  │
                    │      + ReLU        │
                    │  + Dropout(0.3)    │
                    │                    │
                    │ ○ ○ ○ ○ ○ ○ ○ ○   │
                    │ ○ ○ ○ ○ ○ ○ ○ ○   │
                    └─────────┬──────────┘
                              │ 256D
                              ▼
                    ┌────────────────────┐
                    │ Fully Connected 2  │
                    │   [batch, 128]     │
                    │  ━━━━━━━━━━━━━━━━  │
                    │ Linear(256 → 128)  │
                    │      + ReLU        │
                    │  + Dropout(0.2)    │
                    │                    │
                    │ ○ ○ ○ ○ ○ ○ ○ ○   │
                    └─────────┬──────────┘
                              │
                              │ 128D (Shared Representation)
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                │             │             │
        ┌───────┴─────┐ ┌────┴────┐ ┌──────┴──────┐
        │             │ │         │ │             │
        ▼             ▼ ▼         ▼ ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Churn FC │   │  Ap FC  │   │  Tp FC  │      ◄─── Multi-Task Heads
   │[btch,64]│   │[btch,64]│   │[btch,64]│
   │ ━━━━━━━ │   │ ━━━━━━━ │   │ ━━━━━━━ │
   │Lin+ReLU │   │Lin+ReLU │   │Lin+ReLU │
   │ ○○○○○○  │   │ ○○○○○○  │   │ ○○○○○○  │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Churn   │   │   Ap    │   │   Tp    │      ◄─── Output Layers
   │ Output  │   │ Output  │   │ Output  │
   │[btch, 1]│   │[btch, 1]│   │[btch, 1]│
   │ ━━━━━━━ │   │ ━━━━━━━ │   │ ━━━━━━━ │
   │ Sigmoid │   │ Sigmoid │   │  ReLU   │
   │    ○    │   │    ○    │   │    ○    │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │  🎯     │   │  🎯     │   │  🎯     │      ◄─── Predictions
   │ Churn   │   │   Ap    │   │   Tp    │
   │  Prob   │   │Threshold│   │ Window  │
   │  [0,1]  │   │  [0,1]  │   │ [0,∞)   │
   │         │   │         │   │  days   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │BCE Loss │   │MSE Loss │   │MSE Loss │      ◄─── Loss Functions
   │Weight:  │   │Weight:  │   │Weight:  │
   │  2.0    │   │  1.0    │   │  0.1    │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
             ┌────────────────┐
             │ Combined Loss  │
             │ ━━━━━━━━━━━━━━ │
             │ L = 2.0×BCE +  │
             │     1.0×MSE +  │
             │     0.1×MSE    │
             └────────────────┘
                      │
                      ▼
                [ Backpropagation ]
                      ↓
                 Adam Optimizer
              (LR=0.001, β₁=0.9)

════════════════════════════════════════════════════════════════════════════════════════════════════
PARAMETERS: ~1.5M  |  FRAMEWORK: PyTorch  |  BATCH: 256  |  EPOCHS: 50
════════════════════════════════════════════════════════════════════════════════════════════════════
"""

# Simplified version
diagram_simple = r"""
LSTM TEMPORAL MODEL - SIMPLIFIED VIEW
═════════════════════════════════════

Inputs (3 branches)
├─ Event Sequence [batch, 50, 2]
├─ Global Features [batch, 4]
└─ Persona Features [batch, 10]
   ↓
Persona Embedding [batch, 32]
   ↓
LSTM [batch, 50, 128] → Extract h_n[-1] [batch, 128]
   ↓
Concatenate [batch, 164] (128 + 4 + 32)
   ↓
FC1 [batch, 256] + ReLU + Dropout(0.3)
   ↓
FC2 [batch, 128] + ReLU + Dropout(0.2)
   ↓
┌──────────────┬──────────────┬──────────────┐
│ Churn Head   │  Ap Head     │  Tp Head     │
│ FC [64]      │  FC [64]     │  FC [64]     │
│ Out [1]      │  Out [1]     │  Out [1]     │
│ Sigmoid      │  Sigmoid     │  ReLU        │
│ 🎯 [0,1]     │ 🎯 [0,1]     │ 🎯 [0,∞)     │
│ BCE×2.0      │ MSE×1.0      │ MSE×0.1      │
└──────────────┴──────────────┴──────────────┘
                     ↓
              Combined Loss
                     ↓
           Adam Optimizer (LR=0.001)
"""

# HTML version with interactive styling
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LSTM Temporal Model - Architecture Diagram</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #1976D2;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            border-bottom: 4px solid #1976D2;
            padding-bottom: 15px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2em;
            margin-bottom: 40px;
            font-style: italic;
        }
        .layer {
            border: 3px solid #333;
            padding: 20px;
            margin: 15px 0;
            border-radius: 12px;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .layer:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .layer-title {
            font-weight: bold;
            font-size: 1.3em;
            margin-bottom: 10px;
        }
        .layer-spec {
            font-size: 0.95em;
            color: #555;
            line-height: 1.8;
        }
        .input-layer { border-color: #2196F3; background: linear-gradient(135deg, #E3F2FD, #BBDEFB); }
        .embed-layer { border-color: #4CAF50; background: linear-gradient(135deg, #C8E6C9, #A5D6A7); }
        .lstm-layer { border-color: #FF9800; background: linear-gradient(135deg, #FFF9C4, #FFF59D); }
        .concat-layer { border-color: #FF5722; background: linear-gradient(135deg, #FFE0B2, #FFCC80); }
        .fc-layer { border-color: #F44336; background: linear-gradient(135deg, #FFCDD2, #EF9A9A); }
        .head-layer { border-color: #9C27B0; background: linear-gradient(135deg, #E1BEE7, #CE93D8); }
        .loss-layer { border-color: #607D8B; background: linear-gradient(135deg, #CFD8DC, #B0BEC5); }
        .arrow {
            text-align: center;
            font-size: 2.5em;
            color: #666;
            margin: 10px 0;
            animation: bounce 2s infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        .task-heads {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        .task-head {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
        }
        .spec-table {
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .spec-table th {
            background: linear-gradient(135deg, #1976D2, #1565C0);
            color: white;
            padding: 15px;
            font-size: 1.1em;
            text-align: left;
        }
        .spec-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }
        .spec-table tr:hover {
            background: #f5f5f5;
        }
        .note {
            margin: 30px 0;
            padding: 20px;
            background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
            border-left: 6px solid #FF9800;
            border-radius: 8px;
        }
        .neurons {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 10px;
        }
        .neuron {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: white;
            border: 2px solid #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 LSTM Temporal Model</h1>
        <div class="subtitle">Multi-Task Deep Learning for Ap/Tp Prediction</div>

        <div class="layer input-layer">
            <div class="layer-title">INPUT LAYER (3 branches)</div>
            <div class="layer-spec">
                <strong>Event Sequence:</strong> [batch, 50, 2] → Gain values + Time deltas<br>
                <strong>Global Features:</strong> [batch, 4] → Recency, Era, Month, Year<br>
                <strong>Persona Features:</strong> [batch, 10] → NMF cluster statistics
            </div>
            <div class="neurons">
                <span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span>
            </div>
        </div>

        <div class="arrow">↓</div>

        <div class="layer embed-layer">
            <div class="layer-title">PERSONA EMBEDDING</div>
            <div class="layer-spec">
                <strong>Dimension:</strong> [batch, 32]<br>
                <strong>Operation:</strong> Linear(10 → 32) + ReLU
            </div>
            <div class="neurons">
                <span class="neuron"></span><span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span>
            </div>
        </div>

        <div class="arrow">↓</div>

        <div class="layer lstm-layer">
            <div class="layer-title">LSTM (Sequence Processing)</div>
            <div class="layer-spec">
                <strong>Input:</strong> [batch, 50, 2] → 50 timesteps, 2 features<br>
                <strong>Hidden Size:</strong> 128<br>
                <strong>Layers:</strong> 2 (stacked)<br>
                <strong>Dropout:</strong> 0.2<br>
                <strong>Output:</strong> Extract h_n[-1] → [batch, 128]
            </div>
            <div class="neurons">
                <span class="neuron"></span><span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span>
            </div>
        </div>

        <div class="arrow">↓</div>

        <div class="layer concat-layer">
            <div class="layer-title">CONCATENATE</div>
            <div class="layer-spec">
                <strong>Dimension:</strong> [batch, 164]<br>
                <strong>Composition:</strong> LSTM(128) + Global(4) + Persona(32) = 164
            </div>
        </div>

        <div class="arrow">↓</div>

        <div class="layer fc-layer">
            <div class="layer-title">FULLY CONNECTED 1</div>
            <div class="layer-spec">
                <strong>Dimension:</strong> [batch, 256]<br>
                <strong>Operation:</strong> Linear(164 → 256) + ReLU + Dropout(0.3)
            </div>
            <div class="neurons">
                <span class="neuron"></span><span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span>
            </div>
        </div>

        <div class="arrow">↓</div>

        <div class="layer fc-layer">
            <div class="layer-title">FULLY CONNECTED 2</div>
            <div class="layer-spec">
                <strong>Dimension:</strong> [batch, 128]<br>
                <strong>Operation:</strong> Linear(256 → 128) + ReLU + Dropout(0.2)
            </div>
            <div class="neurons">
                <span class="neuron"></span><span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span><span class="neuron"></span>
                <span class="neuron"></span><span class="neuron"></span>
            </div>
        </div>

        <div class="arrow">↓ Split to 3 Heads</div>

        <div class="task-heads">
            <div class="task-head head-layer">
                <div class="layer-title">🎯 Churn Head</div>
                <div class="layer-spec">
                    FC [64] → Linear + ReLU<br>
                    Output [1] → Sigmoid<br>
                    <strong>Prediction: Churn Prob [0,1]</strong><br>
                    Loss: BCE (weight: 2.0)
                </div>
                <div class="neurons">
                    <span class="neuron"></span>
                </div>
            </div>

            <div class="task-head head-layer">
                <div class="layer-title">🎯 Ap Head</div>
                <div class="layer-spec">
                    FC [64] → Linear + ReLU<br>
                    Output [1] → Sigmoid<br>
                    <strong>Prediction: Ap Threshold [0,1]</strong><br>
                    Loss: MSE (weight: 1.0)
                </div>
                <div class="neurons">
                    <span class="neuron"></span>
                </div>
            </div>

            <div class="task-head head-layer">
                <div class="layer-title">🎯 Tp Head</div>
                <div class="layer-spec">
                    FC [64] → Linear + ReLU<br>
                    Output [1] → ReLU<br>
                    <strong>Prediction: Tp Window [0,∞) days</strong><br>
                    Loss: MSE (weight: 0.1)
                </div>
                <div class="neurons">
                    <span class="neuron"></span>
                </div>
            </div>
        </div>

        <div class="arrow">↓</div>

        <div class="layer loss-layer">
            <div class="layer-title">COMBINED LOSS</div>
            <div class="layer-spec">
                <strong>Formula:</strong> L = 2.0 × BCE(Churn) + 1.0 × MSE(Ap) + 0.1 × MSE(Tp)<br>
                <strong>Optimization:</strong> Adam (lr=0.001, betas=(0.9, 0.999))<br>
                <strong>Gradient Clipping:</strong> Max norm = 1.0
            </div>
        </div>

        <h2 style="margin-top: 50px; color: #1976D2;">📊 Model Specifications</h2>
        <table class="spec-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Total Parameters</td><td>~1,500,000</td></tr>
            <tr><td>Framework</td><td>PyTorch</td></tr>
            <tr><td>Optimizer</td><td>Adam (lr=0.001)</td></tr>
            <tr><td>Batch Size</td><td>256</td></tr>
            <tr><td>Max Epochs</td><td>50</td></tr>
            <tr><td>Early Stopping</td><td>Patience: 10 epochs</td></tr>
            <tr><td>LR Scheduler</td><td>ReduceLROnPlateau (factor=0.5, patience=5)</td></tr>
            <tr><td>Gradient Clipping</td><td>Max norm: 1.0</td></tr>
        </table>

        <h2 style="margin-top: 30px; color: #1976D2;">📈 Training Data</h2>
        <table class="spec-table">
            <tr><th>Aspect</th><th>Details</th></tr>
            <tr><td>Sequence Length</td><td>50 recent events per user</td></tr>
            <tr><td>Temporal Coverage</td><td>2017-2022 (6 training years)</td></tr>
            <tr><td>Personas</td><td>20 NMF clusters</td></tr>
            <tr><td>Grace Period</td><td>90 days (churn definition)</td></tr>
            <tr><td>Training Samples</td><td>~50,000 user sequences</td></tr>
            <tr><td>Train/Val Split</td><td>80% / 20%</td></tr>
        </table>

        <div class="note">
            <strong>📝 Note:</strong> This model employs a multi-task learning approach where shared layers
            learn common representations from user behavior sequences, while task-specific heads specialize
            in predicting churn probability, optimal Ap threshold, and Tp window simultaneously. The weighted
            loss prioritizes churn prediction (weight 2.0) while learning auxiliary Ap/Tp parameters.
        </div>

        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f5f5f5; border-radius: 10px;">
            <p style="color: #666; font-size: 0.9em;">
                <strong>Implementation:</strong> lstm_temporal_ap_tp.py<br>
                <strong>Model Class:</strong> TemporalChurnPredictor (lines 311-384)<br>
                <strong>Generated:</strong> Neural Network Architecture Diagram Tool
            </p>
        </div>
    </div>
</body>
</html>
"""

# Save all files
with open('lstm_architecture_ascii.txt', 'w', encoding='utf-8') as f:
    f.write(diagram_ascii)
print("✓ Saved: lstm_architecture_ascii.txt")

with open('lstm_architecture_simple.txt', 'w', encoding='utf-8') as f:
    f.write(diagram_simple)
print("✓ Saved: lstm_architecture_simple.txt")

with open('lstm_architecture_interactive.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
print("✓ Saved: lstm_architecture_interactive.html")

print("\n" + "=" * 100)
print("DIAGRAM GENERATION COMPLETE!")
print("=" * 100)
print("\n📁 Generated 3 Files:")
print("   1. lstm_architecture_ascii.txt          - Full ASCII diagram (copy-paste ready)")
print("   2. lstm_architecture_simple.txt         - Simplified tree view")
print("   3. lstm_architecture_interactive.html   - Interactive HTML (open in browser!)")
print("\n💡 How to Use:")
print("   • ASCII/Simple: Copy into documentation, markdown files, code comments")
print("   • HTML: Open in browser → Beautiful interactive diagram → Print to PDF or screenshot")
print("   • Edit in text editor to customize colors, labels, or structure")
print("\n🎨 To Create a Graphical Diagram:")
print("   • Use these as reference templates")
print("   • Recreate in PowerPoint, Draw.io, Lucidchart, or Figma")
print("   • Or: pip install matplotlib && run visualize_lstm_nn_diagram.py")
print("\n" + "=" * 100)
