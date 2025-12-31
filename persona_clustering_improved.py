"""
Improved Persona Clustering Model with Centroid Matching

This script creates an improved clustering model that:
1. Uses the cluster statistics to create synthetic centroids
2. Combines classification with distance-based matching
3. Handles cluster imbalance better
4. Uses more sophisticated feature engineering

Input files:
- filtered_users_parquet/filtered_users_part_00000.parquet
- nmf_cluster_statistics_20.csv
- eventlog_v2.parquet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
import xgboost as xgb
import joblib
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED FEATURE EXTRACTION
# ============================================================================

def extract_region(location_str):
    """Extract region from location string"""
    if pd.isna(location_str) or location_str == '':
        return 'Unknown'
    
    location_lower = str(location_str).lower()
    
    # More comprehensive region detection
    region_keywords = {
        'North America': ['usa', 'united states', 'us', 'canada', 'mexico', 'california', 
                          'new york', 'texas', 'washington', 'seattle', 'san francisco', 
                          'boston', 'chicago', 'toronto', 'vancouver', 'montreal', 'ontario',
                          'florida', 'colorado', 'oregon', 'arizona', 'michigan', 'ohio',
                          'massachusetts', 'pennsylvania', 'illinois', 'georgia', 'north carolina'],
        'Europe': ['uk', 'united kingdom', 'england', 'germany', 'france', 'spain', 'italy',
                   'netherlands', 'poland', 'sweden', 'norway', 'denmark', 'switzerland',
                   'austria', 'belgium', 'london', 'berlin', 'paris', 'amsterdam', 'barcelona',
                   'europe', 'ireland', 'scotland', 'portugal', 'czech', 'romania', 'hungary',
                   'finland', 'greece', 'ukraine', 'russia', 'moscow', 'kiev', 'warsaw',
                   'prague', 'vienna', 'munich', 'zurich', 'dublin', 'brussels', 'copenhagen'],
        'South Asia': ['india', 'pakistan', 'bangladesh', 'sri lanka', 'nepal', 'mumbai',
                       'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata',
                       'pune', 'kerala', 'karachi', 'lahore', 'gujarat', 'maharashtra',
                       'tamil nadu', 'andhra pradesh', 'telangana', 'west bengal'],
        'East Asia': ['china', 'japan', 'korea', 'taiwan', 'hong kong', 'beijing', 'shanghai',
                      'tokyo', 'seoul', 'singapore', 'vietnam', 'thailand', 'malaysia',
                      'indonesia', 'philippines', 'shenzhen', 'hangzhou', 'osaka', 'manila'],
        'Oceania': ['australia', 'new zealand', 'sydney', 'melbourne', 'brisbane', 'auckland',
                    'perth', 'adelaide', 'wellington', 'queensland', 'victoria'],
        'South America': ['brazil', 'argentina', 'chile', 'colombia', 'peru', 'venezuela',
                          'sao paulo', 'rio', 'buenos aires', 'santiago', 'bogota', 'lima'],
        'Western Asia': ['israel', 'turkey', 'iran', 'uae', 'dubai', 'saudi', 'qatar', 'kuwait',
                         'jordan', 'lebanon', 'tel aviv', 'istanbul', 'tehran', 'riyadh',
                         'abu dhabi', 'beirut', 'amman', 'jerusalem'],
        'North Africa': ['egypt', 'morocco', 'algeria', 'tunisia', 'libya', 'cairo',
                         'casablanca', 'alexandria'],
        'Africa': ['south africa', 'nigeria', 'kenya', 'ghana', 'ethiopia', 'johannesburg',
                   'cape town', 'nairobi', 'lagos', 'accra', 'addis ababa']
    }
    
    for region, keywords in region_keywords.items():
        if any(kw in location_lower for kw in keywords):
            return region
    
    return 'Unknown'


def extract_aboutme_length(aboutme_str):
    """Extract length of AboutMe field (cleaned from HTML)"""
    if pd.isna(aboutme_str) or aboutme_str == '':
        return 0
    
    clean_text = re.sub(r'<[^>]+>', '', str(aboutme_str))
    clean_text = clean_text.strip()
    return len(clean_text)


def compute_comprehensive_activity_features(eventlog_df, user_ids):
    """Compute comprehensive activity-based features from eventlog"""
    print("  Computing comprehensive activity features...")
    
    # Filter and prepare
    user_ids_set = set(user_ids)
    eventlog_filtered = eventlog_df[eventlog_df['user_id'].isin(user_ids_set)].copy()
    
    print(f"    Filtered eventlog: {len(eventlog_filtered)} events for {eventlog_filtered['user_id'].nunique()} users")
    
    # Basic aggregations
    user_stats = eventlog_filtered.groupby('user_id').agg({
        'id': 'count',
        'timestamp': ['min', 'max', 'count'],
        'gap_time_actions_days': ['mean', 'std', 'max']
    })
    
    user_stats.columns = ['total_events', 'first_activity', 'last_activity', 'event_count',
                          'gap_mean_days', 'gap_std_days', 'gap_max_days']
    user_stats = user_stats.reset_index()
    
    # Action type counts
    action_counts = eventlog_filtered.groupby(['user_id', 'action_type_id']).size().unstack(fill_value=0)
    action_counts.columns = [f'action_{int(c)}' for c in action_counts.columns]
    action_counts = action_counts.reset_index()
    
    # Merge
    user_stats = user_stats.merge(action_counts, on='user_id', how='left')
    
    # Derived features
    # Questions (action_type_id = 1)
    user_stats['count_questions'] = user_stats.get('action_1', 0)
    
    # Answers (action_type_id = 2)
    user_stats['count_answers'] = user_stats.get('action_2', 0)
    
    # Comments (action_type_id = 70)
    user_stats['count_comments'] = user_stats.get('action_70', 0)
    
    # Post history (edits)
    edit_cols = ['action_13', 'action_14', 'action_15']
    for col in edit_cols:
        if col not in user_stats.columns:
            user_stats[col] = 0
    user_stats['count_posthistory'] = user_stats[edit_cols].sum(axis=1)
    
    # Duration in months
    user_stats['first_activity'] = pd.to_datetime(user_stats['first_activity'], errors='coerce')
    user_stats['last_activity'] = pd.to_datetime(user_stats['last_activity'], errors='coerce')
    user_stats['activity_duration_months'] = (
        (user_stats['last_activity'] - user_stats['first_activity']).dt.total_seconds() / (30.44 * 24 * 3600)
    ).clip(lower=0)
    
    # Post type analysis
    post_types = eventlog_filtered.groupby(['user_id', 'post_type_id']).size().unstack(fill_value=0)
    post_types.columns = [f'post_type_{int(c) if pd.notna(c) else 0}' for c in post_types.columns]
    post_types = post_types.reset_index()
    
    user_stats = user_stats.merge(post_types, on='user_id', how='left')
    
    return user_stats


def extract_all_features(users_df, eventlog_df):
    """Extract all features from users and eventlog"""
    print("Extracting all features...")
    
    # User-based features
    print("  Extracting user features...")
    features = pd.DataFrame()
    features['user_id'] = users_df['Id']
    features['reputation'] = users_df['Reputation']
    features['upvotes'] = users_df['UpVotes']
    features['downvotes'] = users_df['DownVotes']
    features['views'] = users_df['Views']
    features['aboutme_length'] = users_df['AboutMe'].apply(extract_aboutme_length)
    
    # Account duration
    users_df = users_df.copy()
    users_df['CreationDate'] = pd.to_datetime(users_df['CreationDate'])
    users_df['LastAccessDate'] = pd.to_datetime(users_df['LastAccessDate'])
    features['total_duration'] = ((users_df['LastAccessDate'] - users_df['CreationDate']).dt.days / 30.44).clip(lower=0)
    
    # Region
    features['region'] = users_df['Location'].apply(extract_region)
    
    # Cluster label
    features['cluster'] = users_df['ClusterNmf20']
    
    # Activity features
    activity_features = compute_comprehensive_activity_features(eventlog_df, users_df['Id'].tolist())
    
    # Merge
    features = features.merge(activity_features, left_on='user_id', right_on='user_id', how='left')
    
    # Fill NaN
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)
    
    return features


def compute_derived_features(df):
    """Compute ratio and derived features"""
    print("  Computing derived features...")
    
    df = df.copy()
    
    # Total posts
    df['total_posts'] = df['count_questions'] + df['count_answers']
    df['total_posts'] = df['total_posts'].replace(0, 1)  # Avoid division by zero
    
    # Ratios
    df['collectivistic_ratio'] = df['count_answers'] / (df['count_answers'] + df['count_questions'] + 1)
    df['individualistic_ratio'] = df['count_questions'] / (df['count_answers'] + df['count_questions'] + 1)
    
    # Activity patterns
    df['comments_per_post'] = df['count_comments'] / df['total_posts']
    df['edits_per_post'] = df['count_posthistory'] / df['total_posts']
    df['reputation_per_post'] = df['reputation'] / df['total_posts']
    df['reputation_per_answer'] = df['reputation'] / (df['count_answers'] + 1)
    
    # Engagement intensity
    total_events = df['total_events'].replace(0, 1)
    df['activity_intensity'] = total_events / (df['total_duration'] + 1)
    
    # Vote ratios
    total_votes = df['upvotes'] + df['downvotes'] + 1
    df['upvote_ratio'] = df['upvotes'] / total_votes
    df['downvote_ratio'] = df['downvotes'] / total_votes
    
    # Answer-to-question ratio
    df['answer_question_ratio'] = df['count_answers'] / (df['count_questions'] + 1)
    
    # Views efficiency
    df['views_per_post'] = df['views'] / df['total_posts']
    
    # Log transforms for skewed features
    for col in ['reputation', 'views', 'upvotes', 'downvotes', 'total_events', 
                'count_questions', 'count_answers', 'count_comments']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


# ============================================================================
# CENTROID-BASED MATCHING
# ============================================================================

def build_cluster_centroids(cluster_stats):
    """Build cluster centroids from statistics"""
    print("Building cluster centroids from statistics...")
    
    # Extract median values for key features (more robust than mean)
    feature_rows = [
        'reputation_median', 'downvotes_median', 'upvotes_median', 'views_median',
        'total_duration_median', 'count_posthistory_median', 'count_comments_median',
        'count_badges_median', 'count_questions_median', 'count_answers_median',
        'aboutme_length_median', 'collectivistic_ratio_median', 'individualistic_ratio_median'
    ]
    
    # Get available features
    available_features = [f for f in feature_rows if f in cluster_stats.index]
    
    centroids = cluster_stats.loc[available_features].T
    centroids.index = [int(c.replace('Cluster ', '')) for c in centroids.index]
    
    # Rename columns to match our features
    centroids.columns = [c.replace('_median', '') for c in centroids.columns]
    
    return centroids


def compute_centroid_distances(user_features, centroids, feature_mapping):
    """Compute distance from each user to each cluster centroid"""
    
    # Map user features to centroid feature names
    user_matrix = []
    for centroid_feat in centroids.columns:
        if centroid_feat in feature_mapping:
            user_feat = feature_mapping[centroid_feat]
            if user_feat in user_features.columns:
                user_matrix.append(user_features[user_feat].values)
            else:
                user_matrix.append(np.zeros(len(user_features)))
        else:
            user_matrix.append(np.zeros(len(user_features)))
    
    user_matrix = np.array(user_matrix).T
    
    # Standardize
    scaler = RobustScaler()
    user_matrix_scaled = scaler.fit_transform(user_matrix)
    centroid_matrix_scaled = scaler.transform(centroids.values)
    
    # Compute distances
    distances = cdist(user_matrix_scaled, centroid_matrix_scaled, metric='euclidean')
    
    return distances, centroids.index.tolist()


# ============================================================================
# HYBRID MODEL
# ============================================================================

class HybridPersonaModel:
    """
    Hybrid model combining:
    1. XGBoost classifier for pattern recognition
    2. Centroid distance for boundary cases
    3. Weighted ensemble
    """
    
    def __init__(self, classifier=None, scaler=None, feature_names=None, 
                 centroids=None, feature_mapping=None, centroid_weight=0.3):
        self.classifier = classifier
        self.scaler = scaler
        self.feature_names = feature_names
        self.centroids = centroids
        self.feature_mapping = feature_mapping
        self.centroid_weight = centroid_weight
        self.cluster_labels = None
    
    def fit(self, X, y, centroids=None, feature_mapping=None):
        """Fit the hybrid model"""
        self.cluster_labels = sorted(np.unique(y))
        self.centroids = centroids
        self.feature_mapping = feature_mapping
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost with class weights
        class_counts = Counter(y)
        class_weights = {c: len(y) / (len(class_counts) * count) for c, count in class_counts.items()}
        sample_weights = np.array([class_weights[yi] for yi in y])
        
        self.classifier = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        self.classifier.fit(X_scaled, y, sample_weight=sample_weights)
        
        return self
    
    def predict(self, X, user_features_df=None):
        """Predict with hybrid approach"""
        X_scaled = self.scaler.transform(X)
        
        # Classifier predictions and probabilities
        clf_probs = self.classifier.predict_proba(X_scaled)
        
        if self.centroids is not None and user_features_df is not None and self.feature_mapping is not None:
            # Compute centroid distances
            distances, cluster_order = compute_centroid_distances(
                user_features_df, self.centroids, self.feature_mapping
            )
            
            # Convert distances to similarity scores
            similarities = 1 / (1 + distances)
            similarities = similarities / similarities.sum(axis=1, keepdims=True)
            
            # Create mapping from cluster to probability index
            cluster_to_idx = {c: i for i, c in enumerate(self.classifier.classes_)}
            
            # Reorder similarities to match classifier class order
            reordered_similarities = np.zeros_like(clf_probs)
            for i, cluster in enumerate(cluster_order):
                if cluster in cluster_to_idx:
                    reordered_similarities[:, cluster_to_idx[cluster]] = similarities[:, i]
            
            # Combine predictions
            combined_probs = (1 - self.centroid_weight) * clf_probs + self.centroid_weight * reordered_similarities
        else:
            combined_probs = clf_probs
        
        predictions = self.classifier.classes_[np.argmax(combined_probs, axis=1)]
        
        return predictions, combined_probs
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'centroids': self.centroids,
            'feature_mapping': self.feature_mapping,
            'centroid_weight': self.centroid_weight,
            'cluster_labels': self.cluster_labels
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model"""
        data = joblib.load(path)
        model = cls(
            classifier=data['classifier'],
            scaler=data['scaler'],
            feature_names=data['feature_names'],
            centroids=data['centroids'],
            feature_mapping=data['feature_mapping'],
            centroid_weight=data['centroid_weight']
        )
        model.cluster_labels = data['cluster_labels']
        return model


def prepare_training_data(df, include_log_features=True):
    """Prepare feature matrix for training"""
    print("Preparing training data...")
    
    # Define numeric features
    base_features = [
        'reputation', 'upvotes', 'downvotes', 'views', 'aboutme_length',
        'total_duration', 'total_events', 'count_questions', 'count_answers',
        'count_comments', 'count_posthistory', 'collectivistic_ratio',
        'individualistic_ratio', 'activity_intensity', 'comments_per_post',
        'edits_per_post', 'reputation_per_post', 'reputation_per_answer',
        'upvote_ratio', 'downvote_ratio', 'answer_question_ratio',
        'views_per_post', 'activity_duration_months'
    ]
    
    # Add log features if available
    if include_log_features:
        log_features = [f'{c}_log' for c in ['reputation', 'views', 'upvotes', 'downvotes', 
                                              'total_events', 'count_questions', 'count_answers', 
                                              'count_comments']]
        base_features.extend(log_features)
    
    # Add action features
    action_cols = [c for c in df.columns if c.startswith('action_') and not c.endswith('_types')]
    base_features.extend(action_cols)
    
    # Add post type features
    post_type_cols = [c for c in df.columns if c.startswith('post_type_')]
    base_features.extend(post_type_cols)
    
    # Filter to available
    available_features = [f for f in base_features if f in df.columns]
    
    # One-hot encode region
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    
    # Combine
    X = df[available_features].copy()
    X = pd.concat([X, region_dummies], axis=1)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    feature_names = available_features + list(region_dummies.columns)
    
    return X, feature_names


def train_and_evaluate(X, y, centroids, feature_mapping, user_features_df):
    """Train and evaluate the hybrid model"""
    print("\nTraining hybrid model...")
    
    # Split data
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    # Get corresponding user features for centroid matching
    test_user_features = user_features_df.iloc[idx_test].reset_index(drop=True)
    
    # Train model
    model = HybridPersonaModel(feature_names=X.columns.tolist())
    model.fit(X_train, y_train, centroids=centroids, feature_mapping=feature_mapping)
    
    # Evaluate
    predictions, probs = model.predict(X_test, test_user_features)
    
    accuracy = accuracy_score(y_test, predictions)
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    f1_macro = f1_score(y_test, predictions, average='macro')
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        val_user_features = user_features_df.iloc[val_idx].reset_index(drop=True)
        
        fold_model = HybridPersonaModel()
        fold_model.fit(X_tr, y_tr, centroids=centroids, feature_mapping=feature_mapping)
        preds, _ = fold_model.predict(X_val, val_user_features)
        cv_scores.append(accuracy_score(y_val, preds))
    
    print(f"CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    
    return model, accuracy, f1_weighted


def main():
    print("=" * 70)
    print("IMPROVED PERSONA CLUSTERING MODEL")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    users_df = pd.read_parquet('filtered_users_parquet/filtered_users_part_00000.parquet')
    print(f"   Users: {len(users_df)}")
    
    eventlog_df = pd.read_parquet('eventlog_v2.parquet')
    print(f"   Events: {len(eventlog_df)}")
    
    cluster_stats = pd.read_csv('nmf_cluster_statistics_20.csv', index_col=0)
    print(f"   Cluster stats: {cluster_stats.shape}")
    
    # Extract features
    print("\n2. Extracting features...")
    all_features = extract_all_features(users_df, eventlog_df)
    all_features = compute_derived_features(all_features)
    
    print(f"   Feature matrix shape: {all_features.shape}")
    
    # Save feature matrix
    all_features.to_parquet('persona_feature_matrix_v2.parquet', index=False)
    print("   Saved to persona_feature_matrix_v2.parquet")
    
    # Build centroids
    print("\n3. Building cluster centroids...")
    centroids = build_cluster_centroids(cluster_stats)
    print(f"   Centroids shape: {centroids.shape}")
    
    # Feature mapping (centroid feature -> our feature)
    feature_mapping = {
        'reputation': 'reputation',
        'downvotes': 'downvotes',
        'upvotes': 'upvotes',
        'views': 'views',
        'total_duration': 'total_duration',
        'count_posthistory': 'count_posthistory',
        'count_comments': 'count_comments',
        'count_questions': 'count_questions',
        'count_answers': 'count_answers',
        'aboutme_length': 'aboutme_length',
        'collectivistic_ratio': 'collectivistic_ratio',
        'individualistic_ratio': 'individualistic_ratio'
    }
    
    # Prepare training data
    print("\n4. Preparing training data...")
    X, feature_names = prepare_training_data(all_features)
    y = all_features['cluster'].values
    
    print(f"   Feature matrix: {X.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Clusters: {len(np.unique(y))}")
    
    # Train and evaluate
    print("\n5. Training and evaluating model...")
    model, accuracy, f1_score_weighted = train_and_evaluate(
        X, y, centroids, feature_mapping, all_features
    )
    
    # Save model
    print("\n6. Saving model...")
    model.feature_names = feature_names
    model.save('persona_predictor_hybrid.joblib')
    
    # Feature importance
    print("\n7. Feature importance:")
    if hasattr(model.classifier, 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': model.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feat_imp.head(25).to_string(index=False))
        feat_imp.to_csv('persona_feature_importance.csv', index=False)
    
    # Cluster analysis
    print("\n8. Cluster prediction accuracy:")
    X_all = X.copy()
    preds, probs = model.predict(X_all, all_features)
    
    cluster_acc = {}
    for cluster in sorted(np.unique(y)):
        mask = y == cluster
        cluster_preds = preds[mask]
        cluster_acc[cluster] = (cluster_preds == cluster).mean()
    
    cluster_acc_df = pd.DataFrame([
        {'cluster': k, 'accuracy': v, 'count': (y == k).sum()}
        for k, v in cluster_acc.items()
    ]).sort_values('accuracy', ascending=False)
    
    print(cluster_acc_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nFiles saved:")
    print("  - persona_feature_matrix_v2.parquet")
    print("  - persona_predictor_hybrid.joblib")
    print("  - persona_feature_importance.csv")
    
    return model, all_features


if __name__ == "__main__":
    model, features = main()
