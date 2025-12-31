# Persona Clustering Model for New Users

## 📋 Overview

This document describes the development of a machine learning model to predict user personas (clusters) for new Stack Overflow users. The original research clustered 92,289 users into **20 distinct personas** using Non-negative Matrix Factorization (NMF). Since we don't have access to the original clustering model, we developed a supervised learning approach to map new users to these existing personas.

---

## 🎯 Problem Statement

### The Challenge
- Original NMF clustering model is **not available**
- Some features used in original clustering are **not accessible** for new users (e.g., age, persona type labels)
- Need to predict personas for users who weren't part of the original clustering
- Must work with available data: user profiles and activity logs

### Available Data Sources
| File | Description | Records |
|------|-------------|---------|
| `filtered_users_parquet/filtered_users_part_00000.parquet` | User profiles with cluster labels | 92,289 users |
| `eventlog_v2.parquet` | User activity events | 40.6M events |
| `nmf_cluster_statistics_20.csv` | Statistics for each cluster | 20 clusters × 105 features |

---

## 🔬 Research Journey

### Step 1: Data Exploration

We analyzed the cluster statistics to understand what features were used in the original clustering:

```
Key Features Identified:
├── User Profile Features
│   ├── Reputation (min, max, avg, median, std)
│   ├── UpVotes, DownVotes
│   ├── Views
│   └── AboutMe length
│
├── Activity Features
│   ├── Count of questions, answers, comments
│   ├── Post history (edits)
│   ├── Total activity duration
│   └── Collectivistic/Individualistic ratios
│
├── Demographic Features
│   ├── Geographic region (North America, Europe, South Asia, etc.)
│   ├── Age groups (18-24, 25-34, 35-44, etc.)
│   └── Persona types (H=Helper, C=Contributor, T=Teacher, I=Inquirer)
│
└── Behavioral Features
    └── Comment sentiment
```

### Step 2: Feature Engineering

Since we don't have direct access to age and persona type labels for new users, we focused on **derivable features**:

#### User Profile Features
- `reputation`, `upvotes`, `downvotes`, `views`
- `aboutme_length` (HTML-cleaned text length)
- `total_duration` (account age in months)

#### Activity Features (from eventlog)
- `count_questions` (action_type_id = 1)
- `count_answers` (action_type_id = 2)
- `count_comments` (action_type_id = 70)
- `count_posthistory` (action_type_id = 13, 14, 15)
- `total_events` (all activities)

#### Derived Ratios
- `collectivistic_ratio` = answers / (answers + questions)
- `individualistic_ratio` = questions / (answers + questions)
- `answer_question_ratio` = answers / questions
- `reputation_per_answer`, `comments_per_qa`, etc.

#### Geographic Region
Extracted from `Location` field using keyword matching:
- North America, Europe, South Asia, East Asia
- Oceania, South America, Western Asia
- North Africa, Africa, Unknown

### Step 3: Model Development

We iteratively developed and tested multiple approaches:

| Iteration | Model | Accuracy | Notes |
|-----------|-------|----------|-------|
| v1 | XGBoost Classifier | 57.9% | Basic features |
| v2 | Random Forest | 56.7% | Similar performance |
| v3 | Hybrid (XGBoost + Centroid) | 51.5% | Centroid matching didn't help |
| **v4** | **XGBoost (Class-balanced)** | **50.0%** | Production model with robust features |

### Step 4: Key Insights

#### Feature Importance Analysis
The top features driving cluster predictions:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | region_South Asia | 20.7% |
| 2 | region_Europe | 17.9% |
| 3 | region_North America | 9.2% |
| 4 | region_Unknown | 5.2% |
| 5 | region_East Asia | 3.7% |
| 6 | count_answers | 1.7% |
| 7 | count_comments | 1.2% |
| 8 | action_70 (comments) | 1.8% |
| 9 | downvotes | 0.9% |
| 10 | reputation | 0.7% |

**Key Finding**: Geographic region is the **dominant factor** in cluster assignment, accounting for ~60% of prediction importance.

#### Per-Cluster Performance
Some clusters are predicted with high accuracy, others are challenging:

| Cluster | Accuracy | Size | Dominant Characteristics |
|---------|----------|------|--------------------------|
| 1 | 80.6% | 13,515 | East Asia, 25-34 years, Helper |
| 9 | 65.7% | 5,254 | South Asia, 25-34 years |
| 4 | 62.1% | 9,739 | North America, 35-44 years |
| 5 | 48.8% | 18,131 | Europe, 25-34 years, Helper |
| 10 | 4.9% | 1,236 | South Asia, 25-34 years |

---

## 📦 Files Created

| File | Size | Purpose |
|------|------|---------|
| `persona_predictor_production.joblib` | 47.2 MB | **Trained model** (use this) |
| `persona_features_production.parquet` | 10.7 MB | Feature matrix for 92k users |
| `persona_feature_importance_production.csv` | 717 B | Feature importance ranking |
| `persona_predictor_production.py` | 23.1 KB | Training script & model class |
| `predict_new_user_personas.py` | 8.3 KB | Prediction utility script |

---

## 🚀 How to Use the Model

### Option 1: Python API

```python
from predict_new_user_personas import predict_personas_for_users, get_cluster_description
import pandas as pd

# Load your new users data
new_users = pd.read_parquet('your_new_users.parquet')

# Optional: Load eventlog for activity features
eventlog = pd.read_parquet('eventlog_v2.parquet')

# Predict personas
results = predict_personas_for_users(new_users, eventlog)

# Results include:
# - user_id
# - predicted_cluster (0-19)
# - confidence (0-1)
# - top_3_clusters (list of most likely clusters)
# - prob_cluster_X (probability for each cluster)

print(results.head())
```

### Option 2: Command Line

```bash
python predict_new_user_personas.py \
    --users new_users.parquet \
    --eventlog eventlog_v2.parquet \
    --output predictions.csv
```

### Option 3: Single User Prediction

```python
from predict_new_user_personas import predict_persona_single_user

user_data = {
    'Id': 12345,
    'Reputation': 5000,
    'UpVotes': 100,
    'DownVotes': 5,
    'Views': 1000,
    'Location': 'London, UK',
    'AboutMe': 'Software developer with 5 years of experience.',
    'CreationDate': '2020-01-01',
    'LastAccessDate': '2024-01-01'
}

result = predict_persona_single_user(user_data)
print(f"Predicted cluster: {result['predicted_cluster']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 3 clusters: {result['top_3_clusters']}")
```

### Option 4: Get Cluster Descriptions

```python
from predict_new_user_personas import get_cluster_description

# Get characteristics of cluster 1
desc = get_cluster_description(1)
print(f"Dominant region: {desc['dominant_region']}")
print(f"Dominant age: {desc['dominant_age']}")
print(f"Median reputation: {desc['reputation_median']}")
```

---

## 📊 Understanding Predictions

### Confidence Scores

The model outputs a confidence score (0-1) for each prediction:

| Confidence | Interpretation | Recommendation |
|------------|----------------|----------------|
| ≥ 0.7 | High confidence | Trust the prediction |
| 0.5 - 0.7 | Moderate confidence | Prediction likely correct |
| 0.3 - 0.5 | Low confidence | Consider top-3 clusters |
| < 0.3 | Very low | Manual review needed |

### Using Top-3 Clusters

When confidence is low, use the `top_3_clusters` field:

```python
# For low-confidence predictions, consider all likely clusters
if result['confidence'] < 0.5:
    print(f"User could belong to clusters: {result['top_3_clusters']}")
```

---

## 🔧 Ways to Improve the Model

### 1. **Add More Features**

#### Temporal Activity Patterns
```python
# Extract time-based patterns from eventlog
- Hour of day distribution (working hours vs. off-hours)
- Day of week patterns (weekday vs. weekend activity)
- Activity trends over time (increasing, decreasing, stable)
- Response time to questions
```

#### Content Analysis
```python
# Analyze post/comment content
- Primary programming language/technology tags
- Question vs. answer quality metrics
- Topic clustering (ML, web dev, databases, etc.)
- Writing style complexity
```

#### Network Features
```python
# User interaction patterns
- Number of distinct users interacted with
- Community participation breadth
- Badge diversity and types
```

### 2. **Improve Geographic Detection**

Current region extraction uses keyword matching. Improvements:

```python
# Use a geocoding library
from geopy.geocoders import Nominatim

def get_region_advanced(location):
    geolocator = Nominatim(user_agent="persona_model")
    location = geolocator.geocode(location)
    if location:
        return map_country_to_region(location.address)
    return 'Unknown'
```

### 3. **Handle Class Imbalance Better**

The 20 clusters have highly imbalanced sizes (552 to 18,131 users):

```python
# Options:
- SMOTE oversampling for minority clusters
- Focal loss for XGBoost
- Cluster hierarchical predictions (first predict region, then sub-cluster)
```

### 4. **Hierarchical Clustering Approach**

Given the importance of geography, consider a two-stage model:

```
Stage 1: Predict Geographic Region (high accuracy)
    ↓
Stage 2: Within-Region Cluster Prediction (more refined)
```

### 5. **Semi-Supervised Learning**

Use the cluster statistics to create pseudo-labels:

```python
# For users close to cluster centroids, assign high-confidence labels
# Use these for semi-supervised training
from sklearn.semi_supervised import LabelPropagation
```

### 6. **Ensemble Methods**

Combine multiple models:

```python
models = [
    XGBClassifier(...),
    LGBMClassifier(...),
    CatBoostClassifier(...),
    RandomForestClassifier(...)
]

# Voting or stacking ensemble
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=models, voting='soft')
```

### 7. **Deep Learning Approach**

For activity sequences from eventlog:

```python
# LSTM/Transformer for activity sequence modeling
- Input: Sequence of (action_type, timestamp, post_type)
- Output: Cluster prediction

# This could capture behavioral patterns better than aggregate features
```

### 8. **Active Learning**

For continuous improvement:

```python
# Identify uncertain predictions for human labeling
uncertain_users = results[results['confidence'] < 0.4]

# After manual labeling, retrain model
model.fit(updated_features, updated_labels)
```

---

## 📈 Performance Metrics Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | 50.0% | 20 classes, random = 5% |
| F1 (weighted) | 51.2% | Accounts for class imbalance |
| F1 (macro) | 37.2% | Equal weight per class |
| CV Accuracy | 50.0% ± 0.25% | Stable across folds |
| Best Cluster Accuracy | 80.6% | Cluster 1 |
| Worst Cluster Accuracy | 4.5% | Cluster 14 |

---

## 🧪 Feature Importance Experiments

### The Question: Is Region Really Dominant?

Initially, we observed that region features accounted for ~67% of the XGBoost model's feature importance. This raised a concern: is the model just learning geography, or are behavioral features meaningful?

We conducted a systematic comparison using three different models:

### Model Comparison Results

| Model | Clusters | Accuracy | Features Used |
|-------|----------|----------|---------------|
| **Region-ONLY** | 20 | **14.2%** | 10 (only region one-hot) |
| **Region + Behavior** | 20 | **48.9%** | ~60 (all features) |
| **Behavior-ONLY** | 8 super-clusters | 30.8% | ~50 (no region) |

### Key Insight

```
┌─────────────────────────────────────────────────────────────────────┐
│ Region-ONLY accuracy:     14.2%                                     │
│ Region+Behavior accuracy: 48.9%                                     │
│ Behavior contribution:    +34.7% (HELPFUL!)                         │
└─────────────────────────────────────────────────────────────────────┘
```

### What This Means

1. **Region alone is NOT sufficient** (14.2% is barely above random for 20 classes = 5%)
2. **Behavior features are critical** - they add 34.7% accuracy!
3. **The 67% region importance was misleading** - XGBoost uses region as an *interaction term* with behavior
4. The original NMF clustering captured **Region × Behavior interactions**, not just geography

### Why Region Shows High "Importance" But Low Solo Accuracy

XGBoost's feature importance measures how often a feature is used for splits, not its standalone predictive power. The model learns patterns like:

- "Users from South Asia who answer frequently" → Cluster 9
- "Users from Europe who answer frequently" → Cluster 5
- "Users from South Asia who ask questions" → Cluster 10

Without behavior features, the model cannot distinguish between different user types within the same region.

### Available Models

| Model File | Description | When to Use |
|------------|-------------|-------------|
| `persona_model_region_only.joblib` | Only geography | Testing/comparison |
| `persona_model_region_based.joblib` | Full features | **Production (recommended)** |
| `persona_model_behavior_based.joblib` | No region, 8 super-clusters | Cross-regional comparison |

### Super-Cluster Mapping (Behavior-Only Model)

When using the behavior-only model, the 20 clusters are mapped to 8 behavioral super-clusters:

| Super-Cluster | Name | Original Clusters |
|---------------|------|-------------------|
| 0 | Moderate Contributors | [14, 8] |
| 1 | Active Mid-Level Users | [3, 18] |
| 2 | Casual Helpers | [15, 7, 5] |
| 3 | Regular Helpers | [10, 1, 6] |
| 4 | Elite Contributors | [12, 2] |
| 5 | Active Specialists | [16, 11] |
| 6 | Mid-Level Regulars | [0, 4, 9, 13, 19] |
| 7 | Newcomers | [17] |

---

## 🏁 Conclusion

We successfully built a persona prediction model that:

✅ Maps new users to existing 20 personas  
✅ Achieves 50% accuracy (10x better than random)  
✅ Provides confidence scores for prediction quality  
✅ Is easy to use via Python API or command line  

### Key Findings from Experiments

✅ **Behavior is NOT noise** - it contributes +34.7% accuracy  
✅ **Region × Behavior interaction** - the model learns combined patterns  
✅ **Best model uses both** - Region+Behavior achieves 48.9% accuracy  

### Key Limitations

⚠️ Cannot recover age/persona-type features from original clustering  
⚠️ Some clusters (10, 14) are poorly predicted due to overlap  
⚠️ Behavior-only model achieves only 30.8% on 8 super-clusters  

### Recommended Usage

For churn prediction integration:
1. Predict personas for new users using this model
2. Filter by confidence ≥ 0.5 for reliable assignments
3. For low-confidence users, consider using top-3 clusters
4. Periodically validate predictions against user behavior

---

## 📚 References

- Training data: 92,289 Stack Overflow users (2008-2024)
- Original clustering: NMF with 20 components
- Activity data: 40.6M events across 26 action types
- Model: XGBoost Classifier with balanced class weights

---

*Document generated: December 23, 2025*
