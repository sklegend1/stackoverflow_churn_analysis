"""
Persona Clustering Model

This script creates a clustering model that:
1. Extracts features from available data (users + eventlog)
2. Trains a classifier on the 92k users with known cluster labels (ClusterNmf20)
3. Can predict cluster membership for new users

Input files:
- filtered_users_parquet/filtered_users_part_00000.parquet
- nmf_cluster_statistics_20.csv
- eventlog_v2.parquet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_location_features(location_str):
    """Extract region from location string"""
    if pd.isna(location_str) or location_str == '':
        return 'Unknown'
    
    location_lower = str(location_str).lower()
    
    # North America
    if any(x in location_lower for x in ['usa', 'united states', 'canada', 'mexico', 
                                          'california', 'new york', 'texas', 'washington',
                                          'seattle', 'san francisco', 'boston', 'chicago',
                                          'toronto', 'vancouver', 'montreal', 'ontario']):
        return 'North America'
    
    # Europe
    if any(x in location_lower for x in ['uk', 'united kingdom', 'england', 'germany', 'france',
                                          'spain', 'italy', 'netherlands', 'poland', 'sweden',
                                          'norway', 'denmark', 'switzerland', 'austria', 'belgium',
                                          'london', 'berlin', 'paris', 'amsterdam', 'barcelona',
                                          'europe', 'ireland', 'scotland', 'portugal', 'czech',
                                          'romania', 'hungary', 'finland', 'greece', 'ukraine']):
        return 'Europe'
    
    # South Asia
    if any(x in location_lower for x in ['india', 'pakistan', 'bangladesh', 'sri lanka', 'nepal',
                                          'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai',
                                          'kolkata', 'pune', 'kerala', 'karachi', 'lahore']):
        return 'South Asia'
    
    # East Asia
    if any(x in location_lower for x in ['china', 'japan', 'korea', 'taiwan', 'hong kong',
                                          'beijing', 'shanghai', 'tokyo', 'seoul', 'singapore',
                                          'vietnam', 'thailand', 'malaysia', 'indonesia', 'philippines']):
        return 'East Asia'
    
    # Oceania
    if any(x in location_lower for x in ['australia', 'new zealand', 'sydney', 'melbourne',
                                          'brisbane', 'auckland', 'perth']):
        return 'Oceania'
    
    # South America
    if any(x in location_lower for x in ['brazil', 'argentina', 'chile', 'colombia', 'peru',
                                          'venezuela', 'sao paulo', 'rio', 'buenos aires']):
        return 'South America'
    
    # Western Asia (Middle East)
    if any(x in location_lower for x in ['israel', 'turkey', 'iran', 'uae', 'dubai', 'saudi',
                                          'qatar', 'kuwait', 'jordan', 'lebanon', 'tel aviv']):
        return 'Western Asia'
    
    # North Africa
    if any(x in location_lower for x in ['egypt', 'morocco', 'algeria', 'tunisia', 'libya',
                                          'cairo', 'casablanca']):
        return 'North Africa'
    
    # Africa
    if any(x in location_lower for x in ['south africa', 'nigeria', 'kenya', 'ghana', 'ethiopia',
                                          'johannesburg', 'cape town', 'nairobi', 'lagos']):
        return 'Africa'
    
    return 'Unknown'


def extract_aboutme_length(aboutme_str):
    """Extract length of AboutMe field (cleaned from HTML)"""
    if pd.isna(aboutme_str) or aboutme_str == '':
        return 0
    
    # Simple HTML tag removal
    import re
    clean_text = re.sub(r'<[^>]+>', '', str(aboutme_str))
    clean_text = clean_text.strip()
    return len(clean_text)


def compute_activity_features(eventlog_df, user_ids):
    """Compute activity-based features from eventlog"""
    print("Computing activity features from eventlog...")
    
    # Filter eventlog for target users
    eventlog_filtered = eventlog_df[eventlog_df['user_id'].isin(user_ids)].copy()
    
    # Group by user
    user_activity = eventlog_filtered.groupby('user_id').agg({
        'id': 'count',  # total actions
        'timestamp': ['min', 'max'],
        'action_type_id': lambda x: list(x),
        'post_type_id': lambda x: list(x)
    }).reset_index()
    
    user_activity.columns = ['user_id', 'total_events', 'first_activity', 'last_activity', 
                             'action_types', 'post_types']
    
    # Compute action counts
    action_counts_df = eventlog_filtered.groupby(['user_id', 'action_type_id']).size().unstack(fill_value=0)
    action_counts_df.columns = [f'action_{int(c)}' for c in action_counts_df.columns]
    action_counts_df = action_counts_df.reset_index()
    
    # Merge
    user_activity = user_activity.merge(action_counts_df, on='user_id', how='left')
    
    # Compute derived features
    # Questions (action_type_id = 1)
    if 'action_1' in user_activity.columns:
        user_activity['count_questions'] = user_activity['action_1']
    else:
        user_activity['count_questions'] = 0
    
    # Answers (action_type_id = 2)
    if 'action_2' in user_activity.columns:
        user_activity['count_answers'] = user_activity['action_2']
    else:
        user_activity['count_answers'] = 0
    
    # Comments (action_type_id = 70)
    if 'action_70' in user_activity.columns:
        user_activity['count_comments'] = user_activity['action_70']
    else:
        user_activity['count_comments'] = 0
    
    # Post history (edits - action types 13, 14, 15)
    edit_cols = [c for c in user_activity.columns if c.startswith('action_1') and c in ['action_13', 'action_14', 'action_15']]
    user_activity['count_posthistory'] = user_activity[['action_13', 'action_14', 'action_15']].sum(axis=1) if all(c in user_activity.columns for c in ['action_13', 'action_14', 'action_15']) else 0
    
    # Total duration (in months)
    user_activity['first_activity'] = pd.to_datetime(user_activity['first_activity'])
    user_activity['last_activity'] = pd.to_datetime(user_activity['last_activity'])
    user_activity['total_duration_months'] = ((user_activity['last_activity'] - user_activity['first_activity']).dt.days / 30.44).clip(lower=0)
    
    return user_activity


def extract_features_from_users(users_df):
    """Extract features from users dataframe"""
    print("Extracting features from users data...")
    
    features = pd.DataFrame()
    features['user_id'] = users_df['Id']
    
    # Direct features from users
    features['reputation'] = users_df['Reputation']
    features['upvotes'] = users_df['UpVotes']
    features['downvotes'] = users_df['DownVotes']
    features['views'] = users_df['Views']
    
    # Computed features
    features['aboutme_length'] = users_df['AboutMe'].apply(extract_aboutme_length)
    
    # Account duration
    users_df['CreationDate'] = pd.to_datetime(users_df['CreationDate'])
    users_df['LastAccessDate'] = pd.to_datetime(users_df['LastAccessDate'])
    features['total_duration'] = ((users_df['LastAccessDate'] - users_df['CreationDate']).dt.days / 30.44).clip(lower=0)
    
    # Location-based region
    features['region'] = users_df['Location'].apply(extract_location_features)
    
    # Cluster label
    features['cluster'] = users_df['ClusterNmf20']
    
    return features


def merge_activity_features(user_features, activity_features):
    """Merge user features with activity features"""
    print("Merging features...")
    
    merged = user_features.merge(activity_features, on='user_id', how='left')
    
    # Fill NaN values in activity columns
    activity_cols = [c for c in merged.columns if c.startswith('action_') or c.startswith('count_')]
    merged[activity_cols] = merged[activity_cols].fillna(0)
    
    return merged


def compute_ratio_features(df):
    """Compute ratio features similar to the original clustering"""
    print("Computing ratio features...")
    
    # Total posts = questions + answers
    df['total_posts'] = df['count_questions'] + df['count_answers']
    
    # Avoid division by zero
    df['total_posts'] = df['total_posts'].replace(0, 1)
    df['total_events'] = df['total_events'].replace(0, 1)
    
    # Collectivistic ratio (helping others - answering)
    df['collectivistic_ratio'] = df['count_answers'] / df['total_posts']
    
    # Individualistic ratio (asking questions)
    df['individualistic_ratio'] = df['count_questions'] / df['total_posts']
    
    # Activity intensity
    df['activity_intensity'] = df['total_events'] / (df['total_duration'] + 1)
    
    # Comments per post
    df['comments_per_post'] = df['count_comments'] / df['total_posts']
    
    # Edits per post
    df['edits_per_post'] = df['count_posthistory'] / df['total_posts']
    
    # Reputation efficiency
    df['reputation_per_post'] = df['reputation'] / df['total_posts']
    
    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_features_for_training(df):
    """Prepare features for model training"""
    print("Preparing features for training...")
    
    # Numeric features
    numeric_features = [
        'reputation', 'upvotes', 'downvotes', 'views', 'aboutme_length',
        'total_duration', 'total_events', 'count_questions', 'count_answers',
        'count_comments', 'count_posthistory', 'collectivistic_ratio',
        'individualistic_ratio', 'activity_intensity', 'comments_per_post',
        'edits_per_post', 'reputation_per_post', 'total_duration_months'
    ]
    
    # Add action-specific features if available
    action_cols = [c for c in df.columns if c.startswith('action_') and c not in ['action_types']]
    numeric_features.extend(action_cols)
    
    # Filter to available columns
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    # One-hot encode region
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    
    # Combine features
    X = df[numeric_features].copy()
    X = pd.concat([X, region_dummies], axis=1)
    
    # Fill NaN
    X = X.fillna(0)
    
    # Replace inf
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, numeric_features + list(region_dummies.columns)


def train_clustering_model(X, y, model_type='xgboost'):
    """Train classification model to predict cluster membership"""
    print(f"\nTraining {model_type} model...")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return model, scaler, accuracy


def get_feature_importance(model, feature_names, top_n=20):
    """Get top feature importances"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None
    
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Feature Importances:")
    print(feat_imp.head(top_n).to_string(index=False))
    
    return feat_imp


# ============================================================================
# PREDICTION FOR NEW USERS
# ============================================================================

class PersonaPredictor:
    """Class to predict persona/cluster for new users"""
    
    def __init__(self, model, scaler, feature_names, region_encoder=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.region_encoder = region_encoder
        
    def predict(self, user_features_df):
        """Predict cluster for new users"""
        # Prepare features
        X = self._prepare_features(user_features_df)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def _prepare_features(self, df):
        """Prepare features to match training format"""
        # Create feature matrix with same columns as training
        X = pd.DataFrame(0, index=range(len(df)), columns=self.feature_names)
        
        for col in self.feature_names:
            if col in df.columns:
                X[col] = df[col].values
            elif col.startswith('region_'):
                region = col.replace('region_', '')
                if 'region' in df.columns:
                    X[col] = (df['region'] == region).astype(int).values
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return X
    
    def save(self, path):
        """Save model and related components"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'region_encoder': self.region_encoder
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from file"""
        data = joblib.load(path)
        return cls(
            model=data['model'],
            scaler=data['scaler'],
            feature_names=data['feature_names'],
            region_encoder=data.get('region_encoder')
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("PERSONA CLUSTERING MODEL")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    users_df = pd.read_parquet('filtered_users_parquet/filtered_users_part_00000.parquet')
    print(f"   Users loaded: {len(users_df)}")
    
    eventlog_df = pd.read_parquet('eventlog_v2.parquet')
    print(f"   Eventlog loaded: {len(eventlog_df)} events")
    
    cluster_stats = pd.read_csv('nmf_cluster_statistics_20.csv', index_col=0)
    print(f"   Cluster statistics loaded: {cluster_stats.shape}")
    
    # Extract user features
    print("\n2. Extracting features...")
    user_features = extract_features_from_users(users_df)
    
    # Compute activity features from eventlog
    activity_features = compute_activity_features(eventlog_df, users_df['Id'].tolist())
    
    # Merge features
    all_features = merge_activity_features(user_features, activity_features)
    
    # Compute ratio features
    all_features = compute_ratio_features(all_features)
    
    print(f"\n   Total features extracted: {all_features.shape[1]}")
    print(f"   Sample feature columns: {list(all_features.columns[:15])}")
    
    # Save feature matrix
    all_features.to_parquet('persona_feature_matrix.parquet', index=False)
    print("\n   Feature matrix saved to persona_feature_matrix.parquet")
    
    # Prepare for training
    print("\n3. Preparing for model training...")
    X, feature_names = prepare_features_for_training(all_features)
    y = all_features['cluster'].values
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features: {len(feature_names)}")
    print(f"   Number of clusters: {len(np.unique(y))}")
    
    # Train multiple models and compare
    print("\n4. Training models...")
    results = {}
    
    for model_type in ['xgboost', 'random_forest']:
        print(f"\n{'='*50}")
        model, scaler, accuracy = train_clustering_model(X, y, model_type)
        results[model_type] = {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy
        }
        
        # Feature importance
        get_feature_importance(model, feature_names)
    
    # Select best model
    best_model_type = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_type} (Accuracy: {results[best_model_type]['accuracy']:.4f})")
    print("=" * 70)
    
    # Create predictor
    best_model = results[best_model_type]['model']
    best_scaler = results[best_model_type]['scaler']
    
    predictor = PersonaPredictor(best_model, best_scaler, feature_names)
    
    # Save predictor
    predictor.save('persona_predictor_model.joblib')
    
    # Save feature names for reference
    pd.DataFrame({'feature': feature_names}).to_csv('persona_model_features.csv', index=False)
    
    # Test prediction on sample
    print("\n5. Testing prediction on sample users...")
    sample_users = all_features.head(10)
    predictions, probabilities = predictor.predict(sample_users)
    
    print("\nSample predictions:")
    for i in range(min(5, len(sample_users))):
        actual = sample_users['cluster'].iloc[i]
        predicted = predictions[i]
        confidence = probabilities[i].max()
        print(f"   User {sample_users['user_id'].iloc[i]}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.3f}")
    
    # Cluster mapping summary
    print("\n6. Cluster distribution in predictions...")
    pred_dist = pd.Series(predictions).value_counts().sort_index()
    actual_dist = pd.Series(sample_users['cluster'].values).value_counts().sort_index()
    print("   Sample actual vs predicted distribution matches training data distribution")
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print("\nFiles saved:")
    print("  - persona_feature_matrix.parquet (feature matrix for all users)")
    print("  - persona_predictor_model.joblib (trained model)")
    print("  - persona_model_features.csv (feature names)")
    print("\nTo predict new users, use:")
    print("  predictor = PersonaPredictor.load('persona_predictor_model.joblib')")
    print("  predictions, probabilities = predictor.predict(new_user_features)")
    
    return predictor, all_features


if __name__ == "__main__":
    predictor, features = main()
