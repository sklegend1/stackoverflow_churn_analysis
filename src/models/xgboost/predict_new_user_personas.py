import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
"""
Utility Script: Predict Personas for New Users

This script applies the trained persona prediction model to new users
who weren't part of the original NMF clustering.

Usage:
    python predict_new_user_personas.py --users new_users.parquet --output predictions.csv
    
Or import and use programmatically:
    from predict_new_user_personas import predict_personas_for_users
    predictions = predict_personas_for_users(new_users_df, eventlog_df)
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Import the model class
from persona_predictor_production import PersonaPredictionModel, PersonaFeatureExtractor


def predict_personas_for_users(users_df: pd.DataFrame,
                                eventlog_df: pd.DataFrame = None,
                                model_path: str = '../../../checkpoints/persona_predictor_production.joblib',
                                id_col: str = 'Id') -> pd.DataFrame:
    """
    Predict personas for a batch of users.
    
    Args:
        users_df: DataFrame with user data
            Required columns: Id, Reputation, UpVotes, DownVotes, Views, 
                            Location, AboutMe, CreationDate, LastAccessDate
        eventlog_df: Optional eventlog DataFrame for activity features
        model_path: Path to the trained model
        id_col: Name of the user ID column
    
    Returns:
        DataFrame with user_id, predicted_cluster, confidence, and top_3_clusters
    """
    print(f"Loading model from {model_path}...")
    model = PersonaPredictionModel.load(model_path)
    
    print(f"Extracting features for {len(users_df)} users...")
    features = model.feature_extractor.extract_features(
        users_df, eventlog_df, id_col=id_col, cluster_col=None
    )
    
    print("Making predictions...")
    predictions, probabilities = model.predict(features)
    
    # Build results DataFrame
    results = pd.DataFrame({
        'user_id': features['user_id'],
        'predicted_cluster': predictions,
        'confidence': probabilities.max(axis=1),
        'top_3_clusters': [
            str(np.argsort(probs)[-3:][::-1].tolist())
            for probs in probabilities
        ]
    })
    
    # Add probability columns for each cluster
    for i in range(probabilities.shape[1]):
        cluster_id = model.model.classes_[i]
        results[f'prob_cluster_{cluster_id}'] = probabilities[:, i]
    
    return results


def predict_persona_single_user(user_data: dict,
                                 eventlog_df: pd.DataFrame = None,
                                 model_path: str = '../../../checkpoints/persona_predictor_production.joblib') -> dict:
    """
    Predict persona for a single user.
    
    Args:
        user_data: Dictionary with user information
        eventlog_df: Optional eventlog DataFrame
        model_path: Path to the trained model
    
    Returns:
        Dictionary with prediction results
    """
    users_df = pd.DataFrame([user_data])
    results = predict_personas_for_users(users_df, eventlog_df, model_path)
    
    return {
        'user_id': user_data.get('Id'),
        'predicted_cluster': int(results['predicted_cluster'].iloc[0]),
        'confidence': float(results['confidence'].iloc[0]),
        'top_3_clusters': eval(results['top_3_clusters'].iloc[0])
    }


def get_cluster_description(cluster_id: int) -> dict:
    """
    Get description of a cluster based on the statistics.
    
    Returns key characteristics of the cluster.
    """
    # Load cluster statistics
    try:
        stats = pd.read_csv('../../../data/processed/nmf_cluster_statistics_20.csv', index_col=0)
        cluster_col = f'Cluster {cluster_id}'
        
        if cluster_col not in stats.columns:
            return {'error': f'Cluster {cluster_id} not found'}
        
        cluster_stats = stats[cluster_col]
        
        # Get key characteristics
        description = {
            'cluster_id': cluster_id,
            'reputation_median': cluster_stats.get('reputation_median', 0),
            'reputation_average': cluster_stats.get('reputation_average', 0),
            'count_answers_median': cluster_stats.get('count_answers_median', 0),
            'count_questions_median': cluster_stats.get('count_questions_median', 0),
            'count_comments_median': cluster_stats.get('count_comments_median', 0),
            'collectivistic_ratio_median': cluster_stats.get('collectivistic_ratio_median', 0),
            'total_duration_median': cluster_stats.get('total_duration_median', 0),
        }
        
        # Get dominant region
        regions = ['North America', 'Europe', 'South Asia', 'East Asia', 'Oceania',
                  'South America', 'Western Asia', 'North Africa', 'Africa']
        region_counts = {r: cluster_stats.get(r, 0) for r in regions}
        dominant_region = max(region_counts, key=region_counts.get)
        description['dominant_region'] = dominant_region
        description['dominant_region_count'] = region_counts[dominant_region]
        
        # Get dominant age group
        ages = ['18-24 years old', '25-34 years old', '35-44 years old', 
               '45-54 years old', '55-64 years old', '65 years or older']
        age_counts = {a: cluster_stats.get(a, 0) for a in ages}
        dominant_age = max(age_counts, key=age_counts.get)
        description['dominant_age'] = dominant_age
        description['dominant_age_count'] = age_counts[dominant_age]
        
        # Get persona type (H, C, T, I, SPECIALIST)
        personas = ['H', 'C', 'T', 'I', 'SPECIALIST']
        persona_counts = {p: cluster_stats.get(p, 0) for p in personas}
        dominant_persona = max(persona_counts, key=persona_counts.get)
        description['dominant_persona_type'] = dominant_persona
        description['dominant_persona_count'] = persona_counts[dominant_persona]
        
        return description
        
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Predict personas for new users')
    parser.add_argument('--users', type=str, required=True, 
                       help='Path to users parquet file')
    parser.add_argument('--eventlog', type=str, default=None,
                       help='Path to eventlog parquet file (optional)')
    parser.add_argument('--model', type=str, default='../../../checkpoints/persona_predictor_production.joblib',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='../../../results/persona_predictions.csv',
                       help='Output file path')
    parser.add_argument('--id-col', type=str, default='Id',
                       help='Name of user ID column')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PERSONA PREDICTION FOR NEW USERS")
    print("=" * 60)
    
    # Load users
    print(f"\nLoading users from {args.users}...")
    users_df = pd.read_parquet(args.users)
    print(f"  Loaded {len(users_df)} users")
    
    # Load eventlog if provided
    eventlog_df = None
    if args.eventlog:
        print(f"Loading eventlog from {args.eventlog}...")
        eventlog_df = pd.read_parquet(args.eventlog)
        print(f"  Loaded {len(eventlog_df)} events")
    
    # Predict
    results = predict_personas_for_users(
        users_df, eventlog_df, args.model, args.id_col
    )
    
    # Save results
    print(f"\nSaving predictions to {args.output}...")
    results.to_csv(args.output, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\nTotal users predicted: {len(results)}")
    print(f"\nCluster distribution:")
    print(results['predicted_cluster'].value_counts().sort_index())
    print(f"\nAverage confidence: {results['confidence'].mean():.3f}")
    print(f"Confidence distribution:")
    print(f"  < 0.3: {(results['confidence'] < 0.3).sum()}")
    print(f"  0.3-0.5: {((results['confidence'] >= 0.3) & (results['confidence'] < 0.5)).sum()}")
    print(f"  0.5-0.7: {((results['confidence'] >= 0.5) & (results['confidence'] < 0.7)).sum()}")
    print(f"  >= 0.7: {(results['confidence'] >= 0.7).sum()}")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
