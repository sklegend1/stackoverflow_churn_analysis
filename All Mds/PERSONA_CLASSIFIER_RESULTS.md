# Persona Classifier - Initial Results

**Date**: 2025-12-22
**Model**: XGBoost 20-Class Classifier
**Data**: 92,289 users with known personas from eventlog_v2.parquet

---

## Results Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| **Accuracy** | 45.85% | 24.48% | **24.23%** |
| **Macro F1** | 39.77% | 9.26% | **9.44%** |

⏱️ **Training Time**: 9.50 seconds
📊 **Features Used**: 19 activity-based features

---

## Per-Persona Performance (Test Set)

| Persona | F1 Score | Support | Status |
|---------|----------|---------|--------|
| 2 | 0.000 | 392 | ❌ Cannot classify |
| 6 | 0.000 | 459 | ❌ Cannot classify |
| 13 | 0.000 | 117 | ❌ Cannot classify |
| 10 | 0.000 | 185 | ❌ Cannot classify |
| 7 | 0.009 | 221 | ❌ Nearly impossible |
| 9 | 0.014 | 788 | ❌ Nearly impossible |
| 8 | 0.021 | 417 | ❌ Nearly impossible |
| ... | ... | ... | ... |
| 1 | 0.340 | 2027 | ⚠️ Poor but usable |
| 5 | 0.369 | 2720 | ⚠️ Poor but usable |

**Best performing personas**: 5 (F1=0.37), 1 (F1=0.34)
**Worst performing personas**: 2, 6, 13, 10 (F1=0.00)

---

## Feature Importance (Top 10)

1. `total_answers` - 14.15%
2. `unique_posts_interacted` - 10.93%
3. `total_comments` - 10.54%
4. `activity_span_days` - 6.11%
5. `total_questions` - 5.63%
6. `total_events` - 5.45%
7. `question_ratio` - 5.11%
8. `comment_ratio` - 5.07%
9. `answer_ratio` - 4.91%
10. `days_after_last_event` - 4.80%

---

## Problem Analysis

### Why is accuracy so low (24%)?

1. **Feature Set Too Simple**: The 19 activity-based features (counts, ratios, temporal) are not distinctive enough to separate 20 personas.

2. **Severe Overfitting**: Train accuracy (45.85%) drops to test accuracy (24.23%), indicating the model memorizes training data but doesn't generalize.

3. **Personas May Not Be Activity-Based**: The original NMF clustering likely used more sophisticated features:
   - Post content (text embeddings, topics)
   - Vote patterns (received upvotes/downvotes)
   - Tag preferences
   - Reputation growth curves
   - Badge types
   - Community interactions

4. **Class Imbalance**: Some personas have very few samples (persona 13: 117 users, persona 14: 83 users), making them hard to learn.

5. **Persona Overlap**: Many personas may have similar activity patterns (e.g., both "answer-focused" personas would have high answer ratios).

---

## What This Means for the Validation Plan

### ❌ Current Approach is NOT Viable

With 24% accuracy, we **cannot reliably map new users to personas** using this classifier. The predictions would be essentially random guessing (20 classes = 5% baseline).

### Why We Need Persona Labels

The original 92k users were clustered using **NMF (Non-negative Matrix Factorization)** on features we don't have access to. Without those features or the original NMF model, we cannot reproduce the persona assignments.

---

## Alternative Approaches

### **Option A: Use Original NMF Model (BEST)**
- **If available**: Load the original NMF model used to cluster the 92k users
- Apply it directly to new users using the same feature transformation
- **Pros**: Mathematically consistent, preserves original clustering
- **Cons**: Requires original model file

### **Option B: Retrain NMF from Scratch**
- Extract same features used in original research (if documented)
- Retrain NMF with 20 components on 92k users
- Apply to new users
- **Pros**: Reproducible, doesn't require original model
- **Cons**: Need to know exact original features

### **Option C: Deep Features + Better Classifier**
- Extract richer features:
  - Post text embeddings (BERT/Word2Vec on post bodies)
  - Tag frequency vectors
  - Vote patterns (upvote/downvote ratios per post type)
  - Reputation trajectory (growth curve)
  - Badge acquisition patterns
  - Time-of-day activity patterns
- Train neural network or ensemble model
- **Pros**: Might work with enough features
- **Cons**: Computationally expensive, still uncertain

### **Option D: Skip Persona Mapping for New Users**
- Use LSTM model ONLY on the existing 92k users
- Validate temporal generalization (2016-2023) on known-persona users
- Don't attempt to predict on truly new users
- **Pros**: Avoids persona mapping problem entirely
- **Cons**: Doesn't validate on completely unseen users

### **Option E: Simplified Validation Strategy**
- Hold out 20% of the 92k users BEFORE any model training
- Never use them in XGBoost Ap/Tp finding OR LSTM training
- Evaluate LSTM on this held-out set (known personas)
- **Pros**: True validation on unseen users with correct persona labels
- **Cons**: Reduces training data, but 74k users still plenty

---

## Recommended Next Steps

### **Immediate Action: Ask User**

Before proceeding, we need to know:

1. **Do we have the original NMF model file?**
   - If YES → Use Option A (load and apply)
   - If NO → Continue to next question

2. **What features were used in the original NMF clustering?**
   - Post statistics? (scores, views, votes)
   - Text features? (topic distributions, embeddings)
   - Tag distributions?
   - Temporal patterns?
   - Badge/reputation features?

3. **Is there documentation of the original persona research?**
   - Papers, notebooks, scripts that describe the clustering method?

4. **Would Option E (held-out validation) be acceptable?**
   - Hold out 20k users from 92k, evaluate LSTM on them
   - Validates model without needing new users

### **If No Original Model/Features Available**

**Option E (Held-Out Validation)** is the most pragmatic:
1. Split 92k users: 70k train, 22k test (hold-out)
2. Train XGBoost Ap/Tp on 70k only
3. Train LSTM on 70k only
4. Evaluate LSTM on 22k held-out users (with correct persona labels)
5. Compare F1 scores to validate generalization

This gives us true validation on unseen users WITHOUT needing to solve the persona mapping problem.

---

## Files Generated

- `user_persona_features_92k.csv` - 92,289 users × 19 features
- `persona_classifier_model.pkl` - XGBoost model (24% accuracy)
- `persona_classifier_feature_importance.csv` - Feature rankings
- `persona_classifier_per_persona_f1.csv` - Per-persona metrics
- `persona_classifier_confusion_matrix.png` - Confusion matrix visualization

---

## Conclusion

**The simple activity-based classifier cannot distinguish between personas (24% accuracy).**

We need to either:
1. **Find the original NMF model/features** to properly map new users
2. **Use held-out validation** on the existing 92k users instead
3. **Extract much richer features** and retrain (uncertain success)

**Recommendation**: Ask user about original NMF model availability, then proceed with Option E (held-out validation) as the most practical approach.
