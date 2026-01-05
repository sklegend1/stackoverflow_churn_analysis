# Persona Classification - Final Summary & Recommendation

**Date**: 2025-12-22
**Task**: Classify 92,289 Stack Overflow users into 20 personas for LSTM churn prediction validation
**Contractual Requirement**: Must use the 20 personas (cannot be removed)

---

## 🎯 Executive Summary

After 5 classification attempts with progressively enhanced features, we achieved:
- **Best accuracy**: 46.08% (XGBoost with 33 enhanced features)
- **Original baseline**: 24.23% (XGBoost with 19 simple features)
- **Improvement**: +90% accuracy increase
- **Conclusion**: **Cannot achieve production-ready accuracy (>60%)**

**RECOMMENDATION**: Proceed with **Held-Out Validation Approach** (Option E) instead of attempting to map new users to personas.

---

## 📊 All Classification Attempts

| Attempt | Method | Features | Test Acc | Test F1 | Status |
|---------|--------|----------|----------|---------|--------|
| **1** | XGBoost | 19 simple | 24.23% | 9.44% | ❌ Poor |
| **2** | K-NN Statistics | 10 basic | 1.86% | N/A | ❌ Failed |
| **3** | Neural Network | 19 simple | 11.09% | 7.79% | ❌ Poor |
| **4** | **XGBoost** | **33 enhanced** | **46.08%** | **30.66%** | ⚠️ **Best** |
| **5** | Neural Network | 33 enhanced | 28.36% | 22.09% | ❌ Worse |

---

## 🔍 What We Tried

### Attempt 1: XGBoost with Simple Features (24.23%)
**File**: [train_persona_classifier.py](train_persona_classifier.py)

**Features (19)**:
- Activity counts: questions, answers, comments, edits
- Activity ratios: question_ratio, answer_ratio, comment_ratio, edit_ratio
- Temporal: activity_span_days, avg_gap_days, median_gap_days
- Interaction: unique_posts_interacted, events_per_post

**Result**: 24.23% accuracy (barely better than random 5%)
**Problem**: Features too simple to distinguish 20 personas

---

### Attempt 2: K-NN Statistics-Based (1.86%)
**File**: [persona_classifier_statistics_based.py](persona_classifier_statistics_based.py)

**Method**: Cosine similarity to persona centroids from `nmf_cluster_statistics_20.csv`

**Features (10)**:
- Used persona statistics: reputation_avg, upvotes_avg, downvotes_avg, views_avg
- Counted user activities: questions, answers, comments, badges

**Result**: 1.86% accuracy (WORSE than random!)
**Problem**: Feature mismatch - statistics from original NMF don't align with event log features

---

### Attempt 3: Neural Network with Simple Features (11.09%)
**File**: [train_persona_classifier_nn.py](train_persona_classifier_nn.py)

**Architecture**: 19 → 256 → 128 → 64 → 20
- Batch normalization, dropout (0.4), class weights
- Early stopping at epoch 43

**Result**: 11.09% accuracy (worse than XGBoost!)
**Problem**: Deep learning can't learn from insufficient features

---

### Attempt 4: XGBoost with Enhanced Features (46.08%) ✅ BEST
**File**: [train_persona_classifier_enhanced.py](train_persona_classifier_enhanced.py)

**Features (33)** - Added 14 new features:

#### 🆕 Behavioral Features (from NMF reverse engineering)
1. **collectivistic_ratio** = `answers_given / (questions + answers)`
   - Measures community-helping behavior
   - Mean: 0.6716 (67% answer-focused)

2. **individualistic_ratio** = `edits_to_own_posts / total_edits`
   - Measures self-focused behavior
   - Mean: 0.6369 (64% edit own content)

#### 🌍 Geographic Features (7 regions)
- Europe, Asia, North America, South America, Oceania, Africa, Unknown
- **KEY FINDING**: Geographic features = 79.91% total importance!
- Top 7 most important features are ALL geographic

#### ⭐ Reputation Features
- reputation, upvotes, downvotes, views

#### 📝 Profile Feature
- aboutme_length (profile completeness)

**Result**: 46.08% accuracy (+90% improvement!)
**Top Features**:
1. region_Europe (25.25%)
2. region_Asia (17.81%)
3. region_North America (13.48%)

**Problem**: Still not production-ready (need >60%)

---

### Attempt 5: Neural Network with Enhanced Features (28.36%)
**File**: [train_persona_classifier_nn_enhanced.py](train_persona_classifier_nn_enhanced.py)

**Architecture**: 33 → 512 → 256 → 128 → 64 → 20 (4 hidden layers, 193k parameters)
- Deeper network (4 layers vs 3)
- More capacity (512 neurons vs 256)
- Same regularization (dropout, batch norm, class weights)

**Result**: 28.36% accuracy (38% WORSE than XGBoost!)
**Problem**: Neural network overfits to geographic features, doesn't generalize

---

## 🔬 Root Cause Analysis

### Why Can't We Achieve >60% Accuracy?

#### 1. Geographic Features Dominate (79.91% importance)
- Model essentially predicts location, not behavior
- Personas ARE correlated with geography (time zones, culture)
- But geography alone isn't sufficient for accurate classification

#### 2. Missing Critical Features from Original NMF Clustering

From [proposal.pdf](proposal.pdf) analysis, the original clustering used:

| Feature | Availability | Impact |
|---------|--------------|--------|
| **comment_sentiment** | ❌ NOT available (need comment text) | **HIGH** |
| collectivistic_ratio | ✅ Calculated | LOW (0.73% importance) |
| individualistic_ratio | ✅ Calculated | LOW (0.72% importance) |
| reputation | ✅ Available | LOW (1.12% importance) |
| Geographic data | ✅ Extracted from Location | **HIGH** (80% importance) |
| aboutme_length | ✅ Available | NEGLIGIBLE |
| Age demographics | ❌ NOT available | UNKNOWN |
| Persona type labels (H/C/T/I) | ❌ Outcomes, not features | N/A |

**Key Missing Feature**: **comment_sentiment** (NLP sentiment analysis on comment text)
- This was explicitly mentioned in the original research
- Requires access to comment text (not in eventlog_v2.parquet)
- Likely has HIGH discriminative power

#### 3. Class Imbalance Persists
Even with class weights, small personas fail:
- Persona 14: 83 samples → F1 = 0.019
- Persona 10: 185 samples → F1 = 0.042
- Persona 13: 117 samples → F1 = 0.152

#### 4. Behavioral Ratios Insufficient
- collectivistic_ratio: Only 0.73% importance (ranked 16th)
- individualistic_ratio: Only 0.72% importance (ranked 17th)
- **Without comment sentiment**, these ratios don't capture full behavioral patterns

---

## 🎯 Recommended Solution: Held-Out Validation (Option E)

### Why This Approach?

1. **Avoids persona mapping problem** - Uses only users with correct persona labels
2. **Validates true generalization** - Tests on unseen users with known ground truth
3. **Contractually compliant** - Still uses the 20 personas, just not for new users
4. **Most pragmatic** - Doesn't require solving unsolvable classification problem

### Implementation Steps

#### Step 1: Split 92k Users (BEFORE any training)
```python
# Hold out 20% of users for final validation
users_92k = load_eventlog_users()  # 92,289 users
train_users, holdout_users = train_test_split(users_92k, test_size=0.20, stratify=persona)

# train_users: 73,831 (80%)
# holdout_users: 18,458 (20%)
```

#### Step 2: Train XGBoost Ap/Tp Models on Train Set Only
```python
# Train separate XGBoost models for each persona ON TRAIN SET ONLY
for persona in range(20):
    persona_train_data = train_users[train_users['persona'] == persona]
    xgb_model = train_xgboost_ap_tp(persona_train_data)
    save_model(f"xgb_persona_{persona}_held_out.pkl")
```

#### Step 3: Train LSTM on Train Set Only
```python
# Create core_eventlog for train users only
create_core_eventlog(train_users, xgb_models, output="core_eventlog_train_only.parquet")

# Train LSTM on train set
lstm_model = train_lstm(core_eventlog_train_only)
```

#### Step 4: Evaluate LSTM on Held-Out Set
```python
# Create core_eventlog for held-out users (using their KNOWN personas)
create_core_eventlog(holdout_users, xgb_models, output="core_eventlog_holdout.parquet")

# Evaluate LSTM on held-out users
holdout_results = evaluate_lstm(lstm_model, core_eventlog_holdout)

# Calculate F1 scores per persona and overall
print_per_persona_f1(holdout_results)
```

### Expected Outcomes

| Metric | Train Set (74k) | Held-Out Set (18k) | Interpretation |
|--------|----------------|-------------------|----------------|
| F1 Score | 0.78 (current) | **0.70-0.75** | ✅ Good generalization |
| F1 Score | 0.78 (current) | **0.60-0.70** | ⚠️ Acceptable, some overfitting |
| F1 Score | 0.78 (current) | **<0.60** | ❌ Poor generalization |

### Advantages
1. **No persona mapping needed** - Uses known labels
2. **True unseen validation** - Held-out users never used in any training
3. **Validates temporal generalization** - Tests 2016-2023 patterns
4. **Persona-specific insights** - Can see which personas generalize well
5. **Production-ready metric** - Demonstrates actual deployment performance

### Disadvantages
1. **Doesn't validate on completely new users** - Held-out users are still from original 92k
2. **Smaller training set** - Reduces from 92k to 74k (but still plenty)
3. **Doesn't test "new user onboarding"** - Can't predict churn for users outside 92k

---

## 📈 Per-Persona Classification Results (Best Model: XGBoost Enhanced)

| Persona | F1 Score | Support | Status | Comment |
|---------|----------|---------|--------|---------|
| **1** | 0.686 | 2,027 | ✅ Good | Largest persona, well-classified |
| **5** | 0.576 | 2,720 | ✅ Good | Second largest, acceptable |
| **9** | 0.531 | 788 | ✅ Good | Medium-sized, acceptable |
| **15** | 0.524 | 1,670 | ✅ Good | Large persona, acceptable |
| **4** | 0.477 | 1,461 | ⚠️ Fair | Large but borderline |
| **16** | 0.383 | 530 | ⚠️ Fair | Medium-sized, struggling |
| **3** | 0.360 | 574 | ⚠️ Fair | Moderate performance |
| **0** | 0.320 | 757 | ⚠️ Fair | Large but poor F1 |
| **17** | 0.276 | 448 | ❌ Poor | Geographic confusion? |
| **8** | 0.188 | 417 | ❌ Poor | Cannot reliably classify |
| **7** | 0.144 | 221 | ❌ Poor | Small and confused |
| **6** | 0.118 | 459 | ❌ Poor | Medium but very poor F1 |
| **13** | 0.152 | 117 | ❌ Poor | Small sample, class imbalance |
| **2** | 0.052 | 392 | ❌ Very poor | Essentially unclassifiable |
| **10** | 0.042 | 185 | ❌ Very poor | Small and unclassifiable |
| **14** | 0.019 | 83 | ❌ Cannot classify | Smallest persona, no signal |

**Summary**:
- **4 personas classifiable** (F1 > 0.5): 1, 5, 9, 15
- **8 personas borderline** (0.3 < F1 < 0.5): 0, 3, 4, 11, 12, 16, 18, 19
- **8 personas fail** (F1 < 0.3): 2, 6, 7, 8, 10, 13, 14, 17

---

## 💡 Alternative Future Approaches (If More Resources Available)

### Option F: Comment Sentiment Analysis (High Effort, High Potential)
If comment text is accessible:
1. Load `comments.parquet` or join with `posthistory.parquet`
2. Extract comment text for 92k users
3. Run sentiment analysis:
   - VADER (simple, fast)
   - TextBlob (moderate)
   - BERT sentiment (state-of-the-art)
4. Calculate `comment_sentiment` feature (positive/negative/neutral ratio)
5. Retrain XGBoost with 34 features

**Expected Impact**: Could push accuracy from 46% to 60-65%

### Option G: Ensemble Method
1. Train multiple models: XGBoost, Random Forest, LightGBM
2. Use voting or stacking ensemble
3. Apply geographic-specific models (separate model per region)

**Expected Impact**: +3-5% accuracy (49-51%)

### Option H: Accept Lower Accuracy with Confidence Filtering
1. Use XGBoost (46% accuracy) for classification
2. Only map users with high confidence (prediction probability >0.8)
3. Discard low-confidence predictions (40-50% of users)
4. Validate LSTM only on high-confidence classified users

**Expected Impact**: Effective accuracy 70-80% on filtered subset

---

## 📁 Files Generated

### Feature Extraction
- [extract_persona_features.py](extract_persona_features.py) - Original 19 features
- [extract_enhanced_persona_features.py](extract_enhanced_persona_features.py) - Enhanced 33 features
- `user_persona_features_92k.csv` - 92,289 users × 19 features
- `user_persona_features_enhanced_92k.csv` - 92,289 users × 33 features

### Models
- [train_persona_classifier.py](train_persona_classifier.py) - XGBoost 19 features (24% acc)
- [train_persona_classifier_nn.py](train_persona_classifier_nn.py) - NN 19 features (11% acc)
- [train_persona_classifier_enhanced.py](train_persona_classifier_enhanced.py) - XGBoost 33 features (46% acc) ✅ BEST
- [train_persona_classifier_nn_enhanced.py](train_persona_classifier_nn_enhanced.py) - NN 33 features (28% acc)
- [persona_classifier_statistics_based.py](persona_classifier_statistics_based.py) - K-NN (1.86% acc)

### Results
- [PERSONA_CLASSIFIER_RESULTS.md](PERSONA_CLASSIFIER_RESULTS.md) - Original attempt analysis
- [PERSONA_CLASSIFIER_ENHANCED_RESULTS.md](PERSONA_CLASSIFIER_ENHANCED_RESULTS.md) - Enhanced attempt analysis
- `persona_classifier_enhanced_model.pkl` - Best model (46% accuracy)
- `persona_classifier_enhanced_feature_importance.csv`
- `persona_classifier_enhanced_per_persona_f1.csv`

---

## 🎯 Final Recommendation

### Proceed with Held-Out Validation (Option E)

**Rationale**:
1. **Cannot achieve production accuracy** (46% < 60% target) despite 5 attempts
2. **Missing critical features** (comment sentiment) that we cannot easily obtain
3. **Geographic dominance** suggests personas are location-specific, not purely behavioral
4. **Contractual requirement met** - Still uses 20 personas, validates LSTM properly
5. **Most pragmatic** - Avoids unsolvable problem, provides real validation

**Implementation Time**: 2-3 hours
**Expected LSTM F1**: 0.70-0.75 on held-out set (vs 0.78 on train)

### Next Steps
1. Create data split script: `create_held_out_split.py`
2. Retrain XGBoost Ap/Tp on 80% of users only
3. Retrain LSTM on 80% of users only
4. Evaluate LSTM on 20% held-out users
5. Report per-persona F1 scores and overall generalization

This approach validates your LSTM model's generalization to unseen users **without** needing to solve the intractable persona classification problem.

---

## 📊 Key Statistics Summary

| Metric | Value |
|--------|-------|
| Users in dataset | 92,289 |
| Number of personas | 20 |
| Best classification accuracy | 46.08% (XGBoost enhanced) |
| Original baseline | 24.23% (XGBoost simple) |
| Improvement achieved | +90% |
| Target for production | ≥60% (not achieved) |
| Features tried | 19 → 33 (+14 features) |
| Most important feature | region_Europe (25.25%) |
| Geographic feature importance | 79.91% (top 7 features) |
| Behavioral feature importance | 1.45% (collectivistic + individualistic) |
| Personas well-classified (F1>0.5) | 4 / 20 (20%) |
| Personas failing (F1<0.3) | 8 / 20 (40%) |

---

**Conclusion**: While we significantly improved classification (+90%), we cannot achieve production-ready accuracy without comment sentiment analysis. The held-out validation approach is the most practical solution for validating the LSTM model while respecting the contractual persona requirement.
