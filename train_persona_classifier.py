import time
import numpy as np
import pandas as pd
import polars as pl
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("PERSONA CLASSIFIER TRAINING (XGBoost 20-Class)")
print("=" * 80)

start = time.time()

# ---------- Load features ----------
print("\n[1/6] Loading user features...")
df = pl.read_csv("user_persona_features_92k.csv")
print(f"Loaded {df.height:,} users with {df.width - 2} features")

# Convert to pandas for sklearn/xgboost
df_pd = df.to_pandas()

# Separate features and target
feature_cols = [col for col in df_pd.columns if col not in ["user_id", "persona"]]
X = df_pd[feature_cols].values
y = df_pd["persona"].values

print(f"Feature matrix: {X.shape}")
print(f"Target classes: {np.unique(y)}")
print(f"Number of classes: {len(np.unique(y))}")

# ---------- Train/Val/Test split ----------
print("\n[2/6] Splitting data (70/15/15)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"Train set: {X_train.shape[0]:,} users")
print(f"Val set: {X_val.shape[0]:,} users")
print(f"Test set: {X_test.shape[0]:,} users")

# ---------- Train XGBoost classifier ----------
print("\n[3/6] Training XGBoost multi-class classifier...")

model = XGBClassifier(
    objective='multi:softmax',
    num_class=20,
    max_depth=8,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

train_time = time.time() - start
print(f"\n✅ Training completed in {train_time:.2f} seconds")

# ---------- Evaluate on all sets ----------
print("\n[4/6] Evaluating model...")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred, average='macro')
val_f1 = f1_score(y_val, y_val_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')

print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)
print(f"Train Accuracy: {train_acc:.4f} | Macro F1: {train_f1:.4f}")
print(f"Val   Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f}")
print(f"Test  Accuracy: {test_acc:.4f} | Macro F1: {test_f1:.4f}")
print("=" * 80)

# ---------- Per-persona F1 scores ----------
print("\n[5/6] Computing per-persona metrics...")

report = classification_report(y_test, y_test_pred, output_dict=True)
per_persona_f1 = []
for persona in sorted(np.unique(y)):
    if str(persona) in report:
        f1 = report[str(persona)]['f1-score']
        support = report[str(persona)]['support']
        per_persona_f1.append({
            'persona': persona,
            'f1_score': f1,
            'support': support
        })

df_per_persona = pd.DataFrame(per_persona_f1)
df_per_persona = df_per_persona.sort_values('f1_score')

print("\nPer-Persona F1 Scores (Test Set):")
print(df_per_persona.to_string(index=False))

# Check if any persona has very low F1
low_f1_personas = df_per_persona[df_per_persona['f1_score'] < 0.5]
if len(low_f1_personas) > 0:
    print(f"\n⚠️  WARNING: {len(low_f1_personas)} personas have F1 < 0.5:")
    print(low_f1_personas.to_string(index=False))
else:
    print("\n✅ All personas have F1 >= 0.5")

# ---------- Feature importance ----------
print("\n[6/6] Analyzing feature importance...")

feature_importance = model.feature_importances_
feature_names = feature_cols
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# Save full importance
importance_df.to_csv("persona_classifier_feature_importance.csv", index=False)
print("\n✅ Saved feature importance to persona_classifier_feature_importance.csv")

# ---------- Confusion matrix ----------
print("\nComputing confusion matrix...")
cm = confusion_matrix(y_test, y_test_pred)

# Save confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(np.unique(y)),
            yticklabels=sorted(np.unique(y)))
plt.title('Persona Classifier - Confusion Matrix (Test Set)')
plt.xlabel('Predicted Persona')
plt.ylabel('True Persona')
plt.tight_layout()
plt.savefig('persona_classifier_confusion_matrix.png', dpi=150)
print("✅ Saved confusion matrix to persona_classifier_confusion_matrix.png")
plt.close()

# ---------- Save model ----------
print("\nSaving model...")
with open('persona_classifier_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'feature_columns': feature_cols,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'num_classes': 20,
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
    }, f)

print("✅ Saved model to persona_classifier_model.pkl")

# ---------- Save per-persona results ----------
df_per_persona.to_csv("persona_classifier_per_persona_f1.csv", index=False)
print("✅ Saved per-persona F1 to persona_classifier_per_persona_f1.csv")

# ---------- Summary ----------
end = time.time()
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"✅ Model trained successfully")
print(f"⏱️  Total time: {end - start:.2f} seconds")
print(f"📊 Test Accuracy: {test_acc:.4f}")
print(f"📊 Test Macro F1: {test_f1:.4f}")
print(f"📁 Files created:")
print(f"   - persona_classifier_model.pkl")
print(f"   - persona_classifier_feature_importance.csv")
print(f"   - persona_classifier_per_persona_f1.csv")
print(f"   - persona_classifier_confusion_matrix.png")
print("=" * 80)

# ---------- Decision guideline ----------
print("\n" + "=" * 80)
print("DECISION GUIDELINE")
print("=" * 80)
if test_acc >= 0.85:
    print("✅ EXCELLENT: Accuracy >= 85% - Model is ready for production!")
elif test_acc >= 0.75:
    print("✅ GOOD: Accuracy >= 75% - Model is acceptable for persona mapping")
elif test_acc >= 0.65:
    print("⚠️  FAIR: Accuracy 65-75% - Consider adding more features or tuning")
else:
    print("❌ POOR: Accuracy < 65% - Model needs significant improvement")

if test_f1 >= 0.80:
    print("✅ EXCELLENT: Macro F1 >= 0.80 - Balanced across all personas")
elif test_f1 >= 0.70:
    print("✅ GOOD: Macro F1 >= 0.70 - Reasonably balanced")
elif test_f1 >= 0.60:
    print("⚠️  FAIR: Macro F1 60-70% - Some persona imbalance")
else:
    print("❌ POOR: Macro F1 < 0.60 - Significant persona imbalance")

min_f1 = df_per_persona['f1_score'].min()
if min_f1 >= 0.60:
    print(f"✅ EXCELLENT: Min persona F1 = {min_f1:.4f} - All personas well-classified")
elif min_f1 >= 0.50:
    print(f"✅ GOOD: Min persona F1 = {min_f1:.4f} - Acceptable minimum performance")
else:
    print(f"⚠️  WARNING: Min persona F1 = {min_f1:.4f} - Some personas struggle")

print("=" * 80)
