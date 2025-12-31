"""
Deep Learning Persona Classifier

Uses a neural network with:
- Multiple hidden layers for complex pattern learning
- Dropout for regularization
- Class weights to handle imbalance
- Batch normalization for stable training
"""

import time
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

print("=" * 80)
print("DEEP LEARNING PERSONA CLASSIFIER")
print("=" * 80)

start = time.time()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# =============================================================================
# Load Data
# =============================================================================

print("\n[1/5] Loading user features...")
df = pl.read_csv("user_persona_features_92k.csv")
print(f"Loaded {df.height:,} users with {df.width - 2} features")

# Convert to pandas
df_pd = df.to_pandas()

# Separate features and target
feature_cols = [col for col in df_pd.columns if col not in ["user_id", "persona"]]
X = df_pd[feature_cols].values
y = df_pd["persona"].values

print(f"Feature matrix: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# =============================================================================
# Train/Val/Test Split
# =============================================================================

print("\n[2/5] Splitting data (70/15/15)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"Train set: {X_train.shape[0]:,} users")
print(f"Val set: {X_val.shape[0]:,} users")
print(f"Test set: {X_test.shape[0]:,} users")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# Calculate Class Weights (handle imbalance)
# =============================================================================

print("\n[3/5] Calculating class weights...")
unique_classes, class_counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
class_weights = total_samples / (len(unique_classes) * class_counts)
class_weights_dict = dict(zip(unique_classes, class_weights))

print("Class distribution (train):")
for cls, count in zip(unique_classes, class_counts):
    print(f"  Persona {cls}: {count:,} samples (weight: {class_weights_dict[cls]:.3f})")

# Convert to tensor
class_weights_tensor = torch.FloatTensor([class_weights_dict[i] for i in range(20)]).to(DEVICE)

# =============================================================================
# PyTorch Dataset
# =============================================================================

class PersonaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PersonaDataset(X_train_scaled, y_train)
val_dataset = PersonaDataset(X_val_scaled, y_val)
test_dataset = PersonaDataset(X_test_scaled, y_test)

# Create data loaders
BATCH_SIZE = 512
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# Neural Network Architecture
# =============================================================================

class PersonaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize model
INPUT_DIM = X_train_scaled.shape[1]
HIDDEN_DIMS = [256, 128, 64]  # 3 hidden layers
NUM_CLASSES = 20
DROPOUT_RATE = 0.4

model = PersonaClassifier(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, DROPOUT_RATE).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n[4/5] Model initialized")
print(f"Architecture: {INPUT_DIM} -> {' -> '.join(map(str, HIDDEN_DIMS))} -> {NUM_CLASSES}")
print(f"Total parameters: {total_params:,}")

# =============================================================================
# Training Configuration
# =============================================================================

LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# =============================================================================
# Training Loop
# =============================================================================

print("\n[5/5] Training model...")
print("=" * 80)

best_val_acc = 0.0
patience_counter = 0
train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(batch_y.numpy())

    val_acc = accuracy_score(val_true, val_preds)
    val_f1 = f1_score(val_true, val_preds, average='macro')
    val_accuracies.append(val_acc)

    # Learning rate scheduling
    scheduler.step(val_acc)

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'persona_classifier_nn_best.pt')
        patience_counter = 0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  ✓ Best model saved (val_acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

# =============================================================================
# Load Best Model and Evaluate
# =============================================================================

print("\n[6/6] Evaluating best model on test set...")
model.load_state_dict(torch.load('persona_classifier_nn_best.pt', weights_only=True))
model.eval()

# Test set evaluation
test_preds = []
test_true = []
test_probs = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(DEVICE)
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        test_preds.extend(preds)
        test_true.extend(batch_y.numpy())
        test_probs.extend(probs.cpu().numpy())

test_acc = accuracy_score(test_true, test_preds)
test_f1_macro = f1_score(test_true, test_preds, average='macro')
test_f1_weighted = f1_score(test_true, test_preds, average='weighted')

print("\n" + "=" * 80)
print("TEST SET RESULTS")
print("=" * 80)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"Test Macro F1: {test_f1_macro:.4f}")
print(f"Test Weighted F1: {test_f1_weighted:.4f}")

# Per-persona performance
print("\nPer-Persona Performance:")
report = classification_report(test_true, test_preds, output_dict=True, zero_division=0)

per_persona_results = []
for persona in sorted(np.unique(test_true)):
    if str(persona) in report:
        per_persona_results.append({
            'persona': persona,
            'precision': report[str(persona)]['precision'],
            'recall': report[str(persona)]['recall'],
            'f1_score': report[str(persona)]['f1-score'],
            'support': report[str(persona)]['support']
        })

df_per_persona = pd.DataFrame(per_persona_results)
print(df_per_persona.to_string(index=False))

# Check struggling personas
low_f1_personas = df_per_persona[df_per_persona['f1_score'] < 0.5]
if len(low_f1_personas) > 0:
    print(f"\n⚠️  WARNING: {len(low_f1_personas)} personas have F1 < 0.5")
else:
    print("\n✅ All personas have F1 >= 0.5")

# =============================================================================
# Save Results
# =============================================================================

df_per_persona.to_csv("persona_classifier_nn_per_persona_f1.csv", index=False)

# Save model metadata
metadata = {
    'model_state': 'persona_classifier_nn_best.pt',
    'scaler': scaler,
    'feature_columns': feature_cols,
    'input_dim': INPUT_DIM,
    'hidden_dims': HIDDEN_DIMS,
    'num_classes': NUM_CLASSES,
    'dropout_rate': DROPOUT_RATE,
    'test_accuracy': test_acc,
    'test_f1_macro': test_f1_macro,
    'test_f1_weighted': test_f1_weighted,
}

with open('persona_classifier_nn_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

end = time.time()
print(f"\n⏱️  Total time: {end - start:.2f} seconds")

# =============================================================================
# Decision Guideline
# =============================================================================

print("\n" + "=" * 80)
print("DECISION GUIDELINE")
print("=" * 80)

if test_acc >= 0.75:
    print("✅ EXCELLENT: Accuracy >= 75% - Neural network works!")
    print("   → Can use this for new user classification")
elif test_acc >= 0.60:
    print("✅ GOOD: Accuracy 60-75% - Acceptable for production")
    print("   → Consider confidence thresholding for predictions")
elif test_acc >= 0.45:
    print("⚠️  FAIR: Accuracy 45-60% - Better than random, but needs caution")
    print("   → Use only for high-confidence predictions")
elif test_acc >= 0.24:
    print("⚠️  IMPROVEMENT: Accuracy > 24% - Better than XGBoost!")
    print("   → Still not sufficient, but shows neural networks help")
else:
    print("❌ POOR: Accuracy <= XGBoost - Neural network didn't help")

min_f1 = df_per_persona['f1_score'].min()
if min_f1 >= 0.50:
    print(f"✅ All personas classified well (min F1 = {min_f1:.4f})")
elif min_f1 >= 0.30:
    print(f"⚠️  Some personas struggle (min F1 = {min_f1:.4f})")
else:
    print(f"❌ Many personas fail (min F1 = {min_f1:.4f})")

print("=" * 80)

print("\n✅ Files created:")
print("   - persona_classifier_nn_best.pt")
print("   - persona_classifier_nn_metadata.pkl")
print("   - persona_classifier_nn_per_persona_f1.csv")
