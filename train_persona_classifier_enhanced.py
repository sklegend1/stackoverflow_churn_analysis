"""
Enhanced Persona Classifier with Behavioral Features

Uses the enhanced feature set including:
- Collectivistic ratio (community-helping behavior)
- Individualistic ratio (self-focused behavior)
- Geographic regions
- Reputation metrics
- AboutMe length

Goal: Improve persona classification accuracy from 24% to >60%
"""

import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import pickle

print("=" * 80)
print("ENHANCED PERSONA CLASSIFIER (XGBoost + Behavioral Features)")
print("=" * 80)

start = time.time()

# =============================================================================
# Load Enhanced Features
# =============================================================================

print("\n[1/5] Loading enhanced user features...")
df = pl.read_csv("user_persona_features_enhanced_92k.csv")
print(f"Loaded {df.height:,} users with {df.width - 2} features")

# Convert to pandas for XGBoost
df_pd = df.to_pandas()

# Separate features and target
feature_cols = [col for col in df_pd.columns if col not in ["user_id", "persona"]]
X = df_pd[feature_cols].values
y = df_pd["persona"].values

print(f"Feature matrix: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Display feature categories
print("\n📊 FEATURE BREAKDOWN:")
behavioral_features = [col for col in feature_cols if 'ratio' in col]
geographic_features = [col for col in feature_cols if col.startswith('region_')]
reputation_features = ['reputation', 'upvotes', 'downvotes', 'views']
temporal_features = [col for col in feature_cols if 'days' in col or 'gap' in col]
activity_features = [col for col in feature_cols if col.startswith('total_') or col.startswith('events_')]

print(f"  Behavioral features: {len(behavioral_features)} (NEW!)")
print(f"    → {behavioral_features}")
print(f"  Geographic features: {len(geographic_features)} (NEW!)")
print(f"  Reputation features: {len(reputation_features)} (NEW!)")
print(f"  Temporal features: {len(temporal_features)}")
print(f"  Activity features: {len(activity_features)}")

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
sample_weights = np.array([class_weights[y] for y in y_train])

print("Class distribution (train):")
for cls, count in zip(unique_classes[:5], class_counts[:5]):
    print(f"  Persona {cls}: {count:,} samples")
print("  ...")

# =============================================================================
# Train Enhanced XGBoost Model
# =============================================================================

print("\n[4/5] Training Enhanced XGBoost model...")

model = XGBClassifier(
    objective='multi:softmax',
    num_class=20,
    max_depth=10,  # Increased depth for richer features
    learning_rate=0.05,  # Lower LR for better convergence
    n_estimators=500,  # More trees
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,  # Regularization
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    early_stopping_rounds=30,
)

print("\nModel configuration:")
print(f"  Max depth: {model.max_depth}")
print(f"  Learning rate: {model.learning_rate}")
print(f"  N estimators: {model.n_estimators}")
print(f"  Using class weights: YES")

print("\nTraining...")
model.fit(
    X_train_scaled, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
    verbose=50
)

# Get predictions
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

print("\n" + "=" * 80)
print("TRAINING RESULTS")
print("=" * 80)
print(f"Train Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"Val Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"\nTrain Macro F1: {train_f1_macro:.4f}")
print(f"Val Macro F1: {val_f1_macro:.4f}")
print(f"Test Macro F1: {test_f1_macro:.4f}")

# =============================================================================
# Feature Importance Analysis
# =============================================================================

print("\n[5/5] Analyzing feature importance...")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 TOP 15 MOST IMPORTANT FEATURES:")
print(feature_importance.head(15).to_string(index=False))

# Highlight new behavioral features
print("\n🆕 BEHAVIORAL FEATURE IMPORTANCE:")
behavioral_importance = feature_importance[feature_importance['feature'].str.contains('collectivistic|individualistic')]
print(behavioral_importance.to_string(index=False))

print("\n🌍 GEOGRAPHIC FEATURE IMPORTANCE:")
geographic_importance = feature_importance[feature_importance['feature'].str.contains('region_')]
print(geographic_importance.to_string(index=False))

# =============================================================================
# Per-Persona Performance
# =============================================================================

print("\n" + "=" * 80)
print("PER-PERSONA PERFORMANCE (Test Set)")
print("=" * 80)

report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

per_persona_results = []
for persona in sorted(np.unique(y_test)):
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

# =============================================================================
# Save Model and Results
# =============================================================================

print("\n" + "=" * 80)
print("SAVING MODEL AND RESULTS")
print("=" * 80)

# Save model
with open('persona_classifier_enhanced_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'class_weights': class_weights,
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1_macro,
    }, f)

# Save feature importance
feature_importance.to_csv("persona_classifier_enhanced_feature_importance.csv", index=False)

# Save per-persona results
df_per_persona.to_csv("persona_classifier_enhanced_per_persona_f1.csv", index=False)

end = time.time()
print(f"\n⏱️  Total training time: {end - start:.2f} seconds")

# =============================================================================
# Performance Comparison
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON: Original vs Enhanced Model")
print("=" * 80)

print("\n📊 ORIGINAL MODEL (19 simple features):")
print("  Test Accuracy: 24.23%")
print("  Test Macro F1: 9.44%")
print("  Features: Activity counts, ratios, temporal")

print(f"\n🚀 ENHANCED MODEL (33 features with behavioral):")
print(f"  Test Accuracy: {test_acc * 100:.2f}%")
print(f"  Test Macro F1: {test_f1_macro * 100:.2f}%")
print("  Features: + collectivistic_ratio, individualistic_ratio,")
print("            reputation, geographic regions, aboutme_length")

improvement_acc = (test_acc - 0.2423) / 0.2423 * 100
improvement_f1 = (test_f1_macro - 0.0944) / 0.0944 * 100
print(f"\n📈 IMPROVEMENT:")
print(f"  Accuracy: {improvement_acc:+.1f}%")
print(f"  Macro F1: {improvement_f1:+.1f}%")

# =============================================================================
# Decision Guideline
# =============================================================================

print("\n" + "=" * 80)
print("DECISION GUIDELINE")
print("=" * 80)

if test_acc >= 0.75:
    print("✅ EXCELLENT: Accuracy >= 75% - Model is production-ready!")
    print("   → Can confidently map new users to personas")
    print("   → Proceed with LSTM validation on new users")
elif test_acc >= 0.60:
    print("✅ GOOD: Accuracy 60-75% - Acceptable for production")
    print("   → Consider confidence thresholding for predictions")
    print("   → Can proceed with LSTM validation")
elif test_acc >= 0.45:
    print("⚠️  FAIR: Accuracy 45-60% - Better than baseline but needs caution")
    print("   → Use only for high-confidence predictions")
    print("   → Consider ensemble methods or held-out validation")
elif test_acc >= 0.30:
    print("⚠️  IMPROVEMENT: Accuracy 30-45% - Progress but not sufficient")
    print("   → Consider: Neural network, ensemble, or held-out validation")
    print("   → May need comment sentiment analysis")
else:
    print("❌ STILL POOR: Accuracy < 30% - Need different approach")
    print("   → Consider held-out validation (Option E)")
    print("   → May need access to comment text for sentiment analysis")

min_f1 = df_per_persona['f1_score'].min()
max_f1 = df_per_persona['f1_score'].max()
print(f"\n📊 Per-Persona F1 Range: {min_f1:.4f} - {max_f1:.4f}")

if min_f1 >= 0.50:
    print(f"✅ All personas classified well (min F1 = {min_f1:.4f})")
elif min_f1 >= 0.30:
    print(f"⚠️  Some personas struggle (min F1 = {min_f1:.4f})")
else:
    print(f"❌ Many personas fail (min F1 = {min_f1:.4f})")

print("=" * 80)

print("\n✅ Files created:")
print("   - persona_classifier_enhanced_model.pkl")
print("   - persona_classifier_enhanced_feature_importance.csv")
print("   - persona_classifier_enhanced_per_persona_f1.csv")

print("\n🎯 Next Steps:")
if test_acc >= 0.60:
    print("   1. Use this model to classify new users (not in 92k)")
    print("   2. Create event logs for classified new users")
    print("   3. Evaluate LSTM on completely new users")
else:
    print("   1. Try neural network with these enhanced features")
    print("   2. Consider held-out validation approach (Option E)")
    print("   3. Investigate if comment sentiment data is accessible")
