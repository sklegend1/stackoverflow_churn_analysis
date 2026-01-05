import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
"""
Production-Ready Persona Prediction Model

This model predicts the persona/cluster for new users based on:
1. User profile features (reputation, votes, views, location, etc.)
2. Activity features derived from eventlog
3. Behavioral patterns (question/answer ratios, engagement metrics)

The model is trained on 92k users with known cluster labels and can predict
cluster membership for new users with available features.

Key insight: The original NMF clustering was based on:
- Geographic region (very important)
- Age group (when available)  
- Persona type (H=Helper, C=Contributor, T=Teacher, I=Inquirer, SPECIALIST)
- Activity patterns

Since we don't have age information for new users, the model focuses on
region + activity patterns for prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import warnings
import re
from collections import Counter
from typing import Tuple, List, Optional, Dict, Any

warnings.filterwarnings('ignore')


class PersonaFeatureExtractor:
    """Extract features from user data and eventlog for persona prediction"""
    
    REGION_KEYWORDS = {
        'North America': ['usa', 'united states', 'us', 'canada', 'mexico', 'california', 
                          'new york', 'texas', 'washington', 'seattle', 'san francisco', 
                          'boston', 'chicago', 'toronto', 'vancouver', 'montreal', 'ontario',
                          'florida', 'colorado', 'oregon', 'arizona', 'michigan', 'ohio',
                          'massachusetts', 'pennsylvania', 'illinois', 'georgia', 'north carolina',
                          'virginia', 'maryland', 'new jersey', 'minnesota', 'wisconsin'],
        'Europe': ['uk', 'united kingdom', 'england', 'germany', 'france', 'spain', 'italy',
                   'netherlands', 'poland', 'sweden', 'norway', 'denmark', 'switzerland',
                   'austria', 'belgium', 'london', 'berlin', 'paris', 'amsterdam', 'barcelona',
                   'europe', 'ireland', 'scotland', 'portugal', 'czech', 'romania', 'hungary',
                   'finland', 'greece', 'ukraine', 'russia', 'moscow', 'kiev', 'warsaw',
                   'prague', 'vienna', 'munich', 'zurich', 'dublin', 'brussels', 'copenhagen',
                   'oslo', 'stockholm', 'helsinki', 'lisbon', 'madrid', 'rome', 'milan'],
        'South Asia': ['india', 'pakistan', 'bangladesh', 'sri lanka', 'nepal', 'mumbai',
                       'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata',
                       'pune', 'kerala', 'karachi', 'lahore', 'gujarat', 'maharashtra',
                       'tamil nadu', 'andhra pradesh', 'telangana', 'west bengal', 'ahmedabad'],
        'East Asia': ['china', 'japan', 'korea', 'taiwan', 'hong kong', 'beijing', 'shanghai',
                      'tokyo', 'seoul', 'singapore', 'vietnam', 'thailand', 'malaysia',
                      'indonesia', 'philippines', 'shenzhen', 'hangzhou', 'osaka', 'manila',
                      'bangkok', 'jakarta', 'kuala lumpur', 'guangzhou', 'nanjing'],
        'Oceania': ['australia', 'new zealand', 'sydney', 'melbourne', 'brisbane', 'auckland',
                    'perth', 'adelaide', 'wellington', 'queensland', 'victoria', 'canberra'],
        'South America': ['brazil', 'argentina', 'chile', 'colombia', 'peru', 'venezuela',
                          'sao paulo', 'rio', 'buenos aires', 'santiago', 'bogota', 'lima',
                          'ecuador', 'uruguay', 'paraguay', 'bolivia'],
        'Western Asia': ['israel', 'turkey', 'iran', 'uae', 'dubai', 'saudi', 'qatar', 'kuwait',
                         'jordan', 'lebanon', 'tel aviv', 'istanbul', 'tehran', 'riyadh',
                         'abu dhabi', 'beirut', 'amman', 'jerusalem', 'bahrain', 'oman'],
        'North Africa': ['egypt', 'morocco', 'algeria', 'tunisia', 'libya', 'cairo',
                         'casablanca', 'alexandria'],
        'Africa': ['south africa', 'nigeria', 'kenya', 'ghana', 'ethiopia', 'johannesburg',
                   'cape town', 'nairobi', 'lagos', 'accra', 'addis ababa', 'tanzania',
                   'uganda', 'zimbabwe', 'zambia', 'rwanda']
    }
    
    def __init__(self):
        self.region_cache = {}
    
    def extract_region(self, location_str: str) -> str:
        """Extract region from location string with caching"""
        if pd.isna(location_str) or location_str == '':
            return 'Unknown'
        
        location_str = str(location_str)
        if location_str in self.region_cache:
            return self.region_cache[location_str]
        
        location_lower = location_str.lower()
        
        for region, keywords in self.REGION_KEYWORDS.items():
            if any(kw in location_lower for kw in keywords):
                self.region_cache[location_str] = region
                return region
        
        self.region_cache[location_str] = 'Unknown'
        return 'Unknown'
    
    def extract_aboutme_length(self, aboutme_str: str) -> int:
        """Extract length of AboutMe field (cleaned from HTML)"""
        if pd.isna(aboutme_str) or aboutme_str == '':
            return 0
        clean_text = re.sub(r'<[^>]+>', '', str(aboutme_str))
        return len(clean_text.strip())
    
    def compute_activity_features(self, eventlog_df: pd.DataFrame, 
                                   user_ids: List[int]) -> pd.DataFrame:
        """Compute activity-based features from eventlog"""
        print("  Computing activity features from eventlog...")
        
        user_ids_set = set(user_ids)
        eventlog_filtered = eventlog_df[eventlog_df['user_id'].isin(user_ids_set)].copy()
        
        if len(eventlog_filtered) == 0:
            return pd.DataFrame({'user_id': user_ids})
        
        # Basic aggregations
        user_stats = eventlog_filtered.groupby('user_id').agg({
            'id': 'count',
            'timestamp': ['min', 'max'],
        }).reset_index()
        user_stats.columns = ['user_id', 'total_events', 'first_activity', 'last_activity']
        
        # Action type counts
        action_counts = eventlog_filtered.groupby(['user_id', 'action_type_id']).size().unstack(fill_value=0)
        action_counts.columns = [f'action_{int(c)}' for c in action_counts.columns]
        action_counts = action_counts.reset_index()
        
        user_stats = user_stats.merge(action_counts, on='user_id', how='left')
        
        # Key derived features
        user_stats['count_questions'] = user_stats.get('action_1', 0)
        user_stats['count_answers'] = user_stats.get('action_2', 0)
        user_stats['count_comments'] = user_stats.get('action_70', 0)
        
        edit_cols = ['action_13', 'action_14', 'action_15']
        for col in edit_cols:
            if col not in user_stats.columns:
                user_stats[col] = 0
        user_stats['count_posthistory'] = user_stats[edit_cols].sum(axis=1)
        
        # Duration
        user_stats['first_activity'] = pd.to_datetime(user_stats['first_activity'], errors='coerce')
        user_stats['last_activity'] = pd.to_datetime(user_stats['last_activity'], errors='coerce')
        user_stats['activity_duration_months'] = (
            (user_stats['last_activity'] - user_stats['first_activity']).dt.total_seconds() / (30.44 * 24 * 3600)
        ).clip(lower=0).fillna(0)
        
        return user_stats
    
    def extract_features(self, users_df: pd.DataFrame, 
                        eventlog_df: Optional[pd.DataFrame] = None,
                        id_col: str = 'Id',
                        cluster_col: Optional[str] = 'ClusterNmf20') -> pd.DataFrame:
        """Extract all features from user data"""
        print("Extracting features...")
        
        features = pd.DataFrame()
        features['user_id'] = users_df[id_col]
        
        # User profile features
        features['reputation'] = users_df['Reputation']
        features['upvotes'] = users_df['UpVotes']
        features['downvotes'] = users_df['DownVotes']
        features['views'] = users_df['Views']
        features['aboutme_length'] = users_df['AboutMe'].apply(self.extract_aboutme_length)
        
        # Account duration
        users_df = users_df.copy()
        users_df['CreationDate'] = pd.to_datetime(users_df['CreationDate'])
        users_df['LastAccessDate'] = pd.to_datetime(users_df['LastAccessDate'])
        features['total_duration'] = ((users_df['LastAccessDate'] - users_df['CreationDate']).dt.days / 30.44).clip(lower=0)
        
        # Region
        print("  Extracting region features...")
        features['region'] = users_df['Location'].apply(self.extract_region)
        
        # Cluster label (if available)
        if cluster_col and cluster_col in users_df.columns:
            features['cluster'] = users_df[cluster_col]
        
        # Activity features from eventlog
        if eventlog_df is not None:
            activity_features = self.compute_activity_features(eventlog_df, users_df[id_col].tolist())
            features = features.merge(activity_features, on='user_id', how='left')
        
        # Fill NaN and compute derived features
        features = self._compute_derived_features(features)
        
        return features
    
    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived ratio features"""
        df = df.copy()
        
        # Fill NaN in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Total posts (avoid div by zero)
        df['count_questions'] = df.get('count_questions', 0)
        df['count_answers'] = df.get('count_answers', 0)
        df['count_comments'] = df.get('count_comments', 0)
        df['count_posthistory'] = df.get('count_posthistory', 0)
        df['total_events'] = df.get('total_events', 0)
        
        total_qa = df['count_questions'] + df['count_answers'] + 1
        
        # Behavioral ratios
        df['collectivistic_ratio'] = df['count_answers'] / total_qa
        df['individualistic_ratio'] = df['count_questions'] / total_qa
        df['answer_question_ratio'] = df['count_answers'] / (df['count_questions'] + 1)
        df['comments_per_qa'] = df['count_comments'] / total_qa
        df['edits_per_qa'] = df['count_posthistory'] / total_qa
        
        # Reputation efficiency
        df['reputation_per_answer'] = df['reputation'] / (df['count_answers'] + 1)
        df['reputation_per_qa'] = df['reputation'] / total_qa
        
        # Vote ratios
        total_votes = df['upvotes'] + df['downvotes'] + 1
        df['upvote_ratio'] = df['upvotes'] / total_votes
        
        # Activity intensity
        df['activity_intensity'] = df['total_events'] / (df['total_duration'] + 1)
        
        # Log transforms for skewed features
        for col in ['reputation', 'views', 'upvotes', 'downvotes', 'total_events',
                    'count_questions', 'count_answers', 'count_comments']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])
        
        # Replace inf
        df = df.replace([np.inf, -np.inf], 0)
        
        return df


class PersonaPredictionModel:
    """Production-ready persona prediction model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_extractor = PersonaFeatureExtractor()
        self.region_columns = None
        self.all_regions = ['Unknown', 'North America', 'Europe', 'South Asia', 'East Asia',
                           'Oceania', 'South America', 'Western Asia', 'North Africa', 'Africa']
    
    def prepare_features(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """Prepare feature matrix for model"""
        
        # Numeric features
        base_features = [
            'reputation', 'upvotes', 'downvotes', 'views', 'aboutme_length',
            'total_duration', 'total_events', 'count_questions', 'count_answers',
            'count_comments', 'count_posthistory', 'collectivistic_ratio',
            'individualistic_ratio', 'answer_question_ratio', 'comments_per_qa',
            'edits_per_qa', 'reputation_per_answer', 'reputation_per_qa',
            'upvote_ratio', 'activity_intensity', 'activity_duration_months',
            'reputation_log', 'views_log', 'upvotes_log', 'downvotes_log',
            'total_events_log', 'count_questions_log', 'count_answers_log',
            'count_comments_log'
        ]
        
        # Add action features
        action_cols = [c for c in df.columns if c.startswith('action_') and c not in ['action_types']]
        base_features.extend(action_cols)
        
        # Filter to available
        available_features = [f for f in base_features if f in df.columns]
        
        # Create numeric feature matrix
        X = df[available_features].copy()
        
        # One-hot encode region with all possible regions
        for region in self.all_regions:
            X[f'region_{region}'] = (df['region'] == region).astype(int)
        
        # Fill NaN and replace inf
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if fit_mode:
            self.feature_names = list(X.columns)
        else:
            # Ensure columns match training
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        
        return X
    
    def fit(self, features_df: pd.DataFrame, clusters: np.ndarray):
        """Train the model"""
        print("\nTraining persona prediction model...")
        
        # Prepare features
        X = self.prepare_features(features_df, fit_mode=True)
        y = clusters
        
        print(f"  Feature matrix: {X.shape}")
        print(f"  Number of clusters: {len(np.unique(y))}")
        
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y])
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
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
        
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        print("  Model trained successfully!")
        
        return self
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict cluster for new users"""
        X = self.prepare_features(features_df, fit_mode=False)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions, _ = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_weighted': f1_score(y_test, predictions, average='weighted'),
            'f1_macro': f1_score(y_test, predictions, average='macro')
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(top_n)
    
    def save(self, path: str):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'all_regions': self.all_regions
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PersonaPredictionModel':
        """Load model from file"""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.all_regions = data['all_regions']
        return instance


def predict_new_user(model: PersonaPredictionModel,
                     user_data: Dict[str, Any],
                     eventlog_df: Optional[pd.DataFrame] = None) -> Tuple[int, float, np.ndarray]:
    """
    Predict persona for a single new user.
    
    Args:
        model: Trained PersonaPredictionModel
        user_data: Dictionary with user information
            Required keys: Id, Reputation, UpVotes, DownVotes, Views, 
                          Location, AboutMe, CreationDate, LastAccessDate
        eventlog_df: Optional eventlog DataFrame for activity features
    
    Returns:
        Tuple of (predicted_cluster, confidence, all_probabilities)
    """
    # Create DataFrame from user data
    user_df = pd.DataFrame([user_data])
    
    # Extract features
    features = model.feature_extractor.extract_features(
        user_df, eventlog_df, id_col='Id', cluster_col=None
    )
    
    # Predict
    predictions, probabilities = model.predict(features)
    
    predicted_cluster = predictions[0]
    confidence = probabilities[0].max()
    
    return predicted_cluster, confidence, probabilities[0]


def main():
    print("=" * 70)
    print("PRODUCTION PERSONA PREDICTION MODEL")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    users_df = pd.read_parquet('filtered_users_parquet/filtered_users_part_00000.parquet')
    print(f"   Users: {len(users_df)}")
    
    eventlog_df = pd.read_parquet('../../../data/processed/eventlog_v2.parquet')
    print(f"   Events: {len(eventlog_df)}")
    
    # Initialize model
    model = PersonaPredictionModel()
    
    # Extract features
    print("\n2. Extracting features...")
    features = model.feature_extractor.extract_features(users_df, eventlog_df)
    print(f"   Features extracted: {features.shape}")
    
    # Save features
    features.to_parquet('../../../data/processed/persona_features_production.parquet', index=False)
    print("../../../data/processed/   Saved to persona_features_production.parquet")
    
    # Split data
    print("\n3. Splitting data...")
    y = features['cluster'].values
    
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.2, random_state=42, stratify=y
    )
    
    train_features = features.iloc[train_idx].reset_index(drop=True)
    test_features = features.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"   Train: {len(train_features)}, Test: {len(test_features)}")
    
    # Train model
    print("\n4. Training model...")
    model.fit(train_features, y_train)
    
    # Evaluate
    print("\n5. Evaluating model...")
    predictions, probabilities = model.predict(test_features)
    
    accuracy = accuracy_score(y_test, predictions)
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    f1_macro = f1_score(y_test, predictions, average='macro')
    
    print(f"\n   Test Accuracy: {accuracy:.4f}")
    print(f"   F1 (weighted): {f1_weighted:.4f}")
    print(f"   F1 (macro): {f1_macro:.4f}")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    
    # Cross-validation
    print("\n6. Cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(cv.split(features, y)):
        fold_model = PersonaPredictionModel()
        fold_model.fit(features.iloc[tr_idx], y[tr_idx])
        metrics = fold_model.evaluate(features.iloc[val_idx], y[val_idx])
        cv_scores.append(metrics['accuracy'])
        print(f"   Fold {fold+1}: Accuracy = {metrics['accuracy']:.4f}")
    
    print(f"\n   CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    
    # Feature importance
    print("\n7. Feature importance:")
    importance = model.get_feature_importance(25)
    print(importance.to_string(index=False))
    importance.to_csv('../../../results/persona_feature_importance_production.csv', index=False)
    
    # Per-cluster accuracy
    print("\n8. Per-cluster accuracy:")
    cluster_acc = []
    for cluster in sorted(np.unique(y_test)):
        mask = y_test == cluster
        if mask.sum() > 0:
            acc = (predictions[mask] == y_test[mask]).mean()
            cluster_acc.append({'cluster': cluster, 'accuracy': acc, 'count': mask.sum()})
    
    cluster_acc_df = pd.DataFrame(cluster_acc).sort_values('accuracy', ascending=False)
    print(cluster_acc_df.to_string(index=False))
    
    # Save model
    print("\n9. Saving model...")
    model.save('../../../checkpoints/persona_predictor_production.joblib')
    
    # Test prediction on sample new user
    print("\n10. Testing prediction on sample user...")
    sample_user = {
        'Id': 99999999,
        'Reputation': 5000,
        'UpVotes': 100,
        'DownVotes': 5,
        'Views': 1000,
        'Location': 'London, UK',
        'AboutMe': 'Software developer with 5 years of experience.',
        'CreationDate': '2020-01-01',
        'LastAccessDate': '2024-01-01'
    }
    
    pred_cluster, confidence, probs = predict_new_user(model, sample_user)
    print(f"   Sample user prediction: Cluster {pred_cluster} (confidence: {confidence:.3f})")
    print(f"   Top 3 probable clusters: {np.argsort(probs)[-3:][::-1]}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nFiles saved:")
    print("../../../data/processed/  - persona_features_production.parquet")
    print("../../../checkpoints/  - persona_predictor_production.joblib")
    print("  - persona_feature_importance_production.csv")
    print("\nUsage:")
    print("  model = PersonaPredictionModel.load('../../../checkpoints/persona_predictor_production.joblib')")
    print("  features = model.feature_extractor.extract_features(new_users_df, eventlog_df)")
    print("  predictions, probabilities = model.predict(features)")
    
    return model, features


if __name__ == "__main__":
    model, features = main()
