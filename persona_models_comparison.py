"""
Persona Prediction Models: Region-Based vs Behavior-Based

This script provides two different approaches to persona prediction:

1. REGION-BASED MODEL (Original 20 clusters)
   - Uses geographic region as a key feature
   - ~50% accuracy on original 20 clusters
   - Best when region information is reliable

2. BEHAVIOR-BASED MODEL (8 super-clusters, NO region)
   - Ignores geographic region completely
   - Groups original 20 clusters into 8 behavior-based super-clusters
   - ~30% accuracy but based purely on user behavior
   - Better for comparing users across regions

The original NMF clustering was heavily region-based (60% importance),
so a pure behavior model has lower accuracy but may be more meaningful
for cross-regional analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import AgglomerativeClustering
import xgboost as xgb
import joblib
import warnings
import re

warnings.filterwarnings('ignore')


# ============================================================================
# FEATURE EXTRACTION (shared)
# ============================================================================

REGION_KEYWORDS = {
    'North America': ['usa', 'united states', 'us', 'canada', 'mexico', 'california', 
                      'new york', 'texas', 'washington', 'seattle', 'san francisco', 
                      'boston', 'chicago', 'toronto', 'vancouver', 'montreal'],
    'Europe': ['uk', 'united kingdom', 'england', 'germany', 'france', 'spain', 'italy',
               'netherlands', 'poland', 'sweden', 'norway', 'denmark', 'switzerland',
               'austria', 'belgium', 'london', 'berlin', 'paris', 'amsterdam', 'barcelona',
               'europe', 'ireland', 'scotland', 'portugal', 'czech', 'romania', 'hungary'],
    'South Asia': ['india', 'pakistan', 'bangladesh', 'sri lanka', 'nepal', 'mumbai',
                   'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata'],
    'East Asia': ['china', 'japan', 'korea', 'taiwan', 'hong kong', 'beijing', 'shanghai',
                  'tokyo', 'seoul', 'singapore', 'vietnam', 'thailand', 'malaysia'],
    'Oceania': ['australia', 'new zealand', 'sydney', 'melbourne', 'brisbane', 'auckland'],
    'South America': ['brazil', 'argentina', 'chile', 'colombia', 'peru', 'venezuela'],
    'Western Asia': ['israel', 'turkey', 'iran', 'uae', 'dubai', 'saudi', 'qatar', 'kuwait'],
    'North Africa': ['egypt', 'morocco', 'algeria', 'tunisia', 'libya', 'cairo'],
    'Africa': ['south africa', 'nigeria', 'kenya', 'ghana', 'ethiopia', 'johannesburg']
}

def extract_region(location_str):
    """Extract region from location string"""
    if pd.isna(location_str) or location_str == '':
        return 'Unknown'
    location_lower = str(location_str).lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw in location_lower for kw in keywords):
            return region
    return 'Unknown'


def extract_features_from_data(users_df, eventlog_df=None, include_region=True):
    """Extract features from user data and eventlog"""
    features = pd.DataFrame()
    features['user_id'] = users_df['Id']
    
    # User profile features
    features['reputation'] = users_df['Reputation']
    features['upvotes'] = users_df['UpVotes']
    features['downvotes'] = users_df['DownVotes']
    features['views'] = users_df['Views']
    
    # AboutMe length
    def get_aboutme_length(x):
        if pd.isna(x) or x == '':
            return 0
        return len(re.sub(r'<[^>]+>', '', str(x)).strip())
    features['aboutme_length'] = users_df['AboutMe'].apply(get_aboutme_length)
    
    # Account duration
    users_df = users_df.copy()
    users_df['CreationDate'] = pd.to_datetime(users_df['CreationDate'])
    users_df['LastAccessDate'] = pd.to_datetime(users_df['LastAccessDate'])
    features['total_duration'] = ((users_df['LastAccessDate'] - users_df['CreationDate']).dt.days / 30.44).clip(lower=0)
    
    # Region (optional)
    if include_region:
        features['region'] = users_df['Location'].apply(extract_region)
    
    # Cluster label if available
    if 'ClusterNmf20' in users_df.columns:
        features['cluster'] = users_df['ClusterNmf20']
    
    # Activity features from eventlog
    if eventlog_df is not None:
        activity = compute_activity_features(eventlog_df, users_df['Id'].tolist())
        features = features.merge(activity, on='user_id', how='left')
    
    # Derived features
    features = compute_derived_features(features)
    
    return features


def compute_activity_features(eventlog_df, user_ids):
    """Compute activity features from eventlog"""
    user_ids_set = set(user_ids)
    eventlog_filtered = eventlog_df[eventlog_df['user_id'].isin(user_ids_set)].copy()
    
    if len(eventlog_filtered) == 0:
        return pd.DataFrame({'user_id': user_ids})
    
    # Basic stats
    user_stats = eventlog_filtered.groupby('user_id').agg({
        'id': 'count',
        'timestamp': ['min', 'max'],
    }).reset_index()
    user_stats.columns = ['user_id', 'total_events', 'first_activity', 'last_activity']
    
    # Action counts
    action_counts = eventlog_filtered.groupby(['user_id', 'action_type_id']).size().unstack(fill_value=0)
    action_counts.columns = [f'action_{int(c)}' for c in action_counts.columns]
    action_counts = action_counts.reset_index()
    
    user_stats = user_stats.merge(action_counts, on='user_id', how='left')
    
    # Key counts
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


def compute_derived_features(df):
    """Compute ratio and derived features"""
    df = df.copy()
    
    # Fill NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Ensure key columns exist
    for col in ['count_questions', 'count_answers', 'count_comments', 'count_posthistory', 'total_events']:
        if col not in df.columns:
            df[col] = 0
    
    total_qa = df['count_questions'] + df['count_answers'] + 1
    
    # Behavioral ratios
    df['collectivistic_ratio'] = df['count_answers'] / total_qa
    df['individualistic_ratio'] = df['count_questions'] / total_qa
    df['answer_question_ratio'] = df['count_answers'] / (df['count_questions'] + 1)
    df['comments_per_qa'] = df['count_comments'] / total_qa
    df['edits_per_qa'] = df['count_posthistory'] / total_qa
    df['reputation_per_answer'] = df['reputation'] / (df['count_answers'] + 1)
    df['reputation_per_qa'] = df['reputation'] / total_qa
    
    # Vote ratios
    total_votes = df['upvotes'] + df['downvotes'] + 1
    df['upvote_ratio'] = df['upvotes'] / total_votes
    
    # Activity intensity
    df['activity_intensity'] = df['total_events'] / (df['total_duration'] + 1)
    
    # Log transforms
    for col in ['reputation', 'views', 'upvotes', 'downvotes', 'total_events',
                'count_questions', 'count_answers', 'count_comments']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


# ============================================================================
# MODEL 0: REGION-ONLY (Only geography, no behavior)
# ============================================================================

class RegionOnlyPersonaModel:
    """
    Predicts original 20 NMF clusters using ONLY region features.
    This tests the hypothesis that behavior features are just noise.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.all_regions = ['Unknown', 'North America', 'Europe', 'South Asia', 'East Asia',
                           'Oceania', 'South America', 'Western Asia', 'North Africa', 'Africa']
    
    def prepare_features(self, df, fit_mode=False):
        """Prepare feature matrix - ONLY region features"""
        X = pd.DataFrame()
        
        # Only region one-hot encoding
        for region in self.all_regions:
            X[f'region_{region}'] = (df['region'] == region).astype(int) if 'region' in df.columns else 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if fit_mode:
            self.feature_names = list(X.columns)
        else:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        
        return X
    
    def fit(self, features_df, clusters):
        """Train the model"""
        X = self.prepare_features(features_df, fit_mode=True)
        y = clusters
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y])
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='mlogloss'
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        return self
    
    def predict(self, features_df):
        """Predict clusters"""
        X = self.prepare_features(features_df, fit_mode=False)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def save(self, path):
        joblib.dump({
            'model': self.model, 'scaler': self.scaler,
            'feature_names': self.feature_names, 'all_regions': self.all_regions
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.all_regions = data['all_regions']
        return instance


# ============================================================================
# MODEL 1: REGION-BASED (Region + Behavior)
# ============================================================================

class RegionBasedPersonaModel:
    """
    Predicts original 20 NMF clusters using region + behavior features.
    Higher accuracy (~50%) but heavily influenced by geography.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.all_regions = ['Unknown', 'North America', 'Europe', 'South Asia', 'East Asia',
                           'Oceania', 'South America', 'Western Asia', 'North Africa', 'Africa']
    
    def prepare_features(self, df, fit_mode=False):
        """Prepare feature matrix"""
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
        
        action_cols = [c for c in df.columns if c.startswith('action_')]
        base_features.extend(action_cols)
        available = [f for f in base_features if f in df.columns]
        
        X = df[available].copy()
        
        # Add region one-hot encoding
        for region in self.all_regions:
            X[f'region_{region}'] = (df['region'] == region).astype(int) if 'region' in df.columns else 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if fit_mode:
            self.feature_names = list(X.columns)
        else:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        
        return X
    
    def fit(self, features_df, clusters):
        """Train the model"""
        X = self.prepare_features(features_df, fit_mode=True)
        y = clusters
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y])
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='mlogloss'
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        return self
    
    def predict(self, features_df):
        """Predict clusters"""
        X = self.prepare_features(features_df, fit_mode=False)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def save(self, path):
        joblib.dump({
            'model': self.model, 'scaler': self.scaler,
            'feature_names': self.feature_names, 'all_regions': self.all_regions
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.all_regions = data['all_regions']
        return instance


# ============================================================================
# MODEL 2: BALANCED HYBRID (Uses both but normalizes weights)
# ============================================================================

class BalancedHybridPersonaModel:
    """
    Predicts original 20 NMF clusters using BOTH region + behavior features,
    but applies feature weight constraints to achieve balanced importance.
    
    Approach:
    - Separate features into region and behavior groups
    - Apply multiplicative weights during training to constrain region influence
    - Use iterative training to achieve target balance
    """
    
    def __init__(self, target_region_weight=0.5):
        """
        Args:
            target_region_weight: Target weight for region features (0-1).
                                  0.5 = equal importance, 0.3 = 30% region influence
        """
        self.model = None
        self.scaler = None
        self.behavior_features = None
        self.region_features = None
        self.feature_names = None
        self.target_region_weight = target_region_weight
        self.region_scale_factor = 0.3  # Start with reduced region influence
        self.all_regions = ['Unknown', 'North America', 'Europe', 'South Asia', 'East Asia',
                           'Oceania', 'South America', 'Western Asia', 'North Africa', 'Africa']
    
    def prepare_features(self, df, fit_mode=False):
        """Prepare feature matrix with downweighted region features"""
        # Behavior features
        base_behavior = [
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
        action_cols = [c for c in df.columns if c.startswith('action_')]
        base_behavior.extend(action_cols)
        
        available_behavior = [f for f in base_behavior if f in df.columns]
        X_behavior = df[available_behavior].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Region features (one-hot) - downscaled
        X_region = pd.DataFrame()
        for region in self.all_regions:
            val = (df['region'] == region).astype(float) if 'region' in df.columns else 0.0
            X_region[f'region_{region}'] = val * self.region_scale_factor
        
        if fit_mode:
            self.behavior_features = list(X_behavior.columns)
            self.region_features = list(X_region.columns)
            
            # Combine and scale
            X_combined = pd.concat([X_behavior, X_region], axis=1)
            self.feature_names = list(X_combined.columns)
            
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_combined)
            
        else:
            for col in self.behavior_features:
                if col not in X_behavior.columns:
                    X_behavior[col] = 0
            X_behavior = X_behavior[self.behavior_features]
            
            for col in self.region_features:
                if col not in X_region.columns:
                    X_region[col] = 0
            X_region = X_region[self.region_features]
            
            X_combined = pd.concat([X_behavior, X_region], axis=1)
            X_combined = X_combined[self.feature_names]
            X_scaled = self.scaler.transform(X_combined)
        
        return X_scaled
    
    def fit(self, features_df, clusters):
        """Train the model with iterative balancing"""
        y = clusters
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y])
        
        # Iterative training to find optimal region scale factor
        best_balance = None
        best_model = None
        best_scaler = None
        best_accuracy = 0
        
        print("\n   Searching for optimal region weight balance...")
        
        for scale in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            self.region_scale_factor = scale
            X_scaled = self.prepare_features(features_df, fit_mode=True)
            
            # Split for internal validation
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                X_scaled, y, sample_weights, test_size=0.15, random_state=42, stratify=y
            )
            
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.2, reg_lambda=1.5, random_state=42, n_jobs=-1,
                use_label_encoder=False, eval_metric='mlogloss'
            )
            model.fit(X_train, y_train, sample_weight=sw_train)
            
            # Check importance balance
            importances = model.feature_importances_
            n_behavior = len(self.behavior_features)
            behavior_imp = importances[:n_behavior].sum()
            region_imp = importances[n_behavior:].sum()
            
            # Check accuracy
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            
            print(f"     Scale={scale:.2f}: Behavior={behavior_imp:.2%}, Region={region_imp:.2%}, Acc={acc:.1%}")
            
            # Prefer models closer to target balance with reasonable accuracy
            balance_error = abs(region_imp - self.target_region_weight)
            score = acc - balance_error * 0.3  # Penalize imbalance
            
            if best_balance is None or score > best_balance:
                best_balance = score
                best_model = model
                best_scaler = self.scaler
                best_accuracy = acc
                best_scale = scale
                best_behavior_imp = behavior_imp
                best_region_imp = region_imp
        
        # Use best settings and retrain on full data
        self.region_scale_factor = best_scale
        X_scaled = self.prepare_features(features_df, fit_mode=True)
        
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='mlogloss'
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        
        print(f"\n   Best scale factor: {best_scale:.2f}")
        
        return self
    
    def predict(self, features_df):
        """Predict clusters"""
        X_scaled = self.prepare_features(features_df, fit_mode=False)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def get_feature_importance_by_group(self):
        """Get feature importance split by behavior vs region"""
        importances = self.model.feature_importances_
        n_behavior = len(self.behavior_features)
        
        behavior_imp = importances[:n_behavior].sum()
        region_imp = importances[n_behavior:].sum()
        
        # Get top features from each group
        behavior_df = pd.DataFrame({
            'feature': self.behavior_features,
            'importance': importances[:n_behavior]
        }).sort_values('importance', ascending=False)
        
        region_df = pd.DataFrame({
            'feature': self.region_features,
            'importance': importances[n_behavior:]
        }).sort_values('importance', ascending=False)
        
        return {
            'behavior_total': behavior_imp,
            'region_total': region_imp,
            'behavior_top': behavior_df.head(10),
            'region_top': region_df.head(5),
            'scale_factor_used': self.region_scale_factor
        }
    
    def save(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'behavior_features': self.behavior_features,
            'region_features': self.region_features,
            'feature_names': self.feature_names,
            'target_region_weight': self.target_region_weight,
            'region_scale_factor': self.region_scale_factor,
            'all_regions': self.all_regions
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(target_region_weight=data['target_region_weight'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.behavior_features = data['behavior_features']
        instance.region_features = data['region_features']
        instance.feature_names = data['feature_names']
        instance.region_scale_factor = data['region_scale_factor']
        instance.all_regions = data['all_regions']
        return instance


# ============================================================================
# MODEL 3: BEHAVIOR-BASED (8 super-clusters, NO region)
# ============================================================================

# Super-cluster mapping based on behavioral similarity
CLUSTER_TO_SUPER = {
    14: 0, 8: 0,              # Super-0: Moderate contributors
    3: 1, 18: 1,              # Super-1: Active mid-level
    15: 2, 7: 2, 5: 2,        # Super-2: Casual helpers (low activity)
    10: 3, 1: 3, 6: 3,        # Super-3: Regular helpers
    12: 4, 2: 4,              # Super-4: Elite contributors
    16: 5, 11: 5,             # Super-5: Active specialists
    0: 6, 4: 6, 9: 6, 13: 6, 19: 6,  # Super-6: Mid-level regulars
    17: 7                      # Super-7: Newcomers/low activity
}

SUPER_CLUSTER_NAMES = {
    0: 'Moderate Contributors',
    1: 'Active Mid-Level Users',
    2: 'Casual Helpers',
    3: 'Regular Helpers',
    4: 'Elite Contributors',
    5: 'Active Specialists',
    6: 'Mid-Level Regulars',
    7: 'Newcomers'
}

SUPER_TO_CLUSTERS = {
    0: [14, 8],
    1: [3, 18],
    2: [15, 7, 5],
    3: [10, 1, 6],
    4: [12, 2],
    5: [16, 11],
    6: [0, 4, 9, 13, 19],
    7: [17]
}


class BehaviorBasedPersonaModel:
    """
    Predicts 8 behavioral super-clusters WITHOUT using region.
    Lower accuracy (~30%) but based purely on user behavior.
    Better for cross-regional comparisons.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.cluster_to_super = CLUSTER_TO_SUPER
        self.super_to_clusters = SUPER_TO_CLUSTERS
        self.super_cluster_names = SUPER_CLUSTER_NAMES
    
    def prepare_features(self, df, fit_mode=False):
        """Prepare feature matrix (NO region features)"""
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
        
        action_cols = [c for c in df.columns if c.startswith('action_')]
        base_features.extend(action_cols)
        available = [f for f in base_features if f in df.columns]
        
        X = df[available].fillna(0).replace([np.inf, -np.inf], 0)
        
        if fit_mode:
            self.feature_names = list(X.columns)
        else:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        
        return X
    
    def fit(self, features_df, clusters):
        """Train the model on super-clusters"""
        # Convert original clusters to super-clusters
        super_clusters = np.array([self.cluster_to_super.get(c, 6) for c in clusters])
        
        X = self.prepare_features(features_df, fit_mode=True)
        y = super_clusters
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y])
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='mlogloss'
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        return self
    
    def predict(self, features_df):
        """Predict super-clusters"""
        X = self.prepare_features(features_df, fit_mode=False)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def get_super_cluster_name(self, super_cluster_id):
        """Get human-readable name for super-cluster"""
        return self.super_cluster_names.get(super_cluster_id, f'Super-{super_cluster_id}')
    
    def get_original_clusters(self, super_cluster_id):
        """Get original cluster IDs that belong to this super-cluster"""
        return self.super_to_clusters.get(super_cluster_id, [])
    
    def save(self, path):
        joblib.dump({
            'model': self.model, 'scaler': self.scaler,
            'feature_names': self.feature_names,
            'cluster_to_super': self.cluster_to_super,
            'super_to_clusters': self.super_to_clusters,
            'super_cluster_names': self.super_cluster_names
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.cluster_to_super = data['cluster_to_super']
        instance.super_to_clusters = data['super_to_clusters']
        instance.super_cluster_names = data['super_cluster_names']
        return instance


# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def train_all_models():
    """Train all models for comparison"""
    print("=" * 70)
    print("TRAINING PERSONA PREDICTION MODELS")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    users_df = pd.read_parquet('filtered_users_parquet/filtered_users_part_00000.parquet')
    eventlog_df = pd.read_parquet('eventlog_v2.parquet')
    print(f"   Users: {len(users_df)}, Events: {len(eventlog_df)}")
    
    # Extract features
    print("\n2. Extracting features...")
    features = extract_features_from_data(users_df, eventlog_df, include_region=True)
    print(f"   Features shape: {features.shape}")
    
    y = features['cluster'].values
    
    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.2, random_state=42, stratify=y
    )
    train_features = features.iloc[train_idx].reset_index(drop=True)
    test_features = features.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # ---- MODEL 0: Region-ONLY ----
    print("\n" + "=" * 70)
    print("MODEL 0: REGION-ONLY (Only Geography, No Behavior)")
    print("=" * 70)
    
    region_only_model = RegionOnlyPersonaModel()
    region_only_model.fit(train_features, y_train)
    
    preds_ro, _ = region_only_model.predict(test_features)
    acc_region_only = accuracy_score(y_test, preds_ro)
    f1_region_only = f1_score(y_test, preds_ro, average='weighted')
    
    print(f"\n   Accuracy: {acc_region_only:.4f} ({acc_region_only*100:.1f}%)")
    print(f"   F1 (weighted): {f1_region_only:.4f}")
    print(f"   Features used: {len(region_only_model.feature_names)} (only region one-hot)")
    
    region_only_model.save('persona_model_region_only.joblib')
    print("   Saved to: persona_model_region_only.joblib")
    
    # ---- MODEL 1: Region + Behavior ----
    print("\n" + "=" * 70)
    print("MODEL 1: REGION + BEHAVIOR (Full Features)")
    print("=" * 70)
    
    region_model = RegionBasedPersonaModel()
    region_model.fit(train_features, y_train)
    
    preds, probs = region_model.predict(test_features)
    acc_region = accuracy_score(y_test, preds)
    f1_region = f1_score(y_test, preds, average='weighted')
    
    print(f"\n   Accuracy: {acc_region:.4f} ({acc_region*100:.1f}%)")
    print(f"   F1 (weighted): {f1_region:.4f}")
    
    # Feature importance
    feat_imp = pd.DataFrame({
        'feature': region_model.feature_names,
        'importance': region_model.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    region_importance = feat_imp[feat_imp['feature'].str.startswith('region_')]['importance'].sum()
    print(f"   Region features importance: {region_importance:.4f} ({region_importance*100:.1f}%)")
    
    region_model.save('persona_model_region_based.joblib')
    print("   Saved to: persona_model_region_based.joblib")
    
    # ---- MODEL 2: Behavior-Based ----
    print("\n" + "=" * 70)
    print("MODEL 2: BEHAVIOR-BASED (8 Super-Clusters, NO Region)")
    print("=" * 70)
    
    behavior_model = BehaviorBasedPersonaModel()
    behavior_model.fit(train_features, y_train)
    
    # Evaluate on super-clusters
    y_test_super = np.array([CLUSTER_TO_SUPER.get(c, 6) for c in y_test])
    preds_super, probs_super = behavior_model.predict(test_features)
    acc_super = accuracy_score(y_test_super, preds_super)
    f1_super = f1_score(y_test_super, preds_super, average='weighted')
    
    print(f"\n   Accuracy: {acc_super:.4f} ({acc_super*100:.1f}%)")
    print(f"   F1 (weighted): {f1_super:.4f}")
    
    print("\n   Super-Cluster Mapping:")
    for super_id, name in SUPER_CLUSTER_NAMES.items():
        clusters = SUPER_TO_CLUSTERS[super_id]
        print(f"     {super_id}: {name} <- Clusters {clusters}")
    
    behavior_model.save('persona_model_behavior_based.joblib')
    print("\n   Saved to: persona_model_behavior_based.joblib")
    
    # ============== SUMMARY ==============
    print("\n" + "=" * 70)
    print("SUMMARY: MODEL COMPARISON")
    print("=" * 70)
    
    # Calculate behavior contribution
    behavior_contribution = acc_region - acc_region_only
    
    print(f"""
    Model                    | Clusters | Accuracy | Features Used
    -------------------------|----------|----------|---------------
    0. Region-ONLY           |    20    |  {acc_region_only*100:.1f}%   | 10 (region only)
    1. Region + Behavior     |    20    |  {acc_region*100:.1f}%   | ~60 (region + behavior)
    2. Behavior-ONLY         |     8    |  {acc_super*100:.1f}%   | ~50 (behavior only)
    
    KEY INSIGHT:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Region-ONLY accuracy:     {acc_region_only*100:.1f}%                                   │
    │ Region+Behavior accuracy: {acc_region*100:.1f}%                                   │
    │ Behavior contribution:    {behavior_contribution*100:+.1f}% ({"NOISE" if behavior_contribution < 0.01 else "HELPFUL"})                       │
    └─────────────────────────────────────────────────────────────────────┘
    
    CONCLUSION:
    {"→ Behavior features add little value. Original clustering is purely geography-based!" if behavior_contribution < 0.02 else "→ Behavior features provide some additional signal."}
    """)
    
    return region_only_model, region_model, behavior_model


if __name__ == "__main__":
    region_only_model, region_model, behavior_model = train_all_models()
