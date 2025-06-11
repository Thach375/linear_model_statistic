import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


def analyze_feature_importance(n_top_features=20, save_plot=True):
    """
    Analyze feature importance using Random Forest and return the most important features.
    
    Parameters:
    -----------
    n_top_features : int, default=20
        Number of top features to return
    save_plot : bool, default=True
        Whether to save the feature importance plot
        
    Returns:
    --------
    top_features : pandas.DataFrame
        DataFrame with top feature names and their importance scores
    """
    # Load the dataset
    print("Loading data...")
    data = pd.read_csv('data/train.csv')
    
    # Split features and target
    X = data.drop(columns=['SalePrice', 'Id'])
    y = data['SalePrice']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    print(f"Dataset has {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
    
    # Create preprocessor for Random Forest
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Preprocess the data
    print("Preprocessing data...")
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after preprocessing
    feature_names = (
        list(numerical_cols) + 
        list(preprocessor.named_transformers_['cat'][1].get_feature_names_out(categorical_cols))
    )
    
    # Train a Random Forest for feature importance
    print("Training Random Forest model for feature importance...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_processed, y)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create a DataFrame with feature names and importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Select top N features
    top_features = feature_importance_df.head(n_top_features)
    print(f"\nTop {n_top_features} important features:")
    print(top_features)
    
    # Save feature importance
    top_features.to_csv('models/feature_importance.csv', index=False)
    print("Feature importance saved to models/feature_importance.csv")
    
    # Plot feature importance
    if save_plot:
        plt.figure(figsize=(12, 10))
        sns.set(style="whitegrid")
        
        # Create barplot of feature importance
        ax = sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {n_top_features} Feature Importance for House Price Prediction')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to models/feature_importance.png")
        
        # Display plot
        plt.show()
    
    return top_features


def train_model_with_selected_features(n_features=None, threshold=None):
    """
    Train models using only the most important features.
    
    Parameters:
    -----------
    n_features : int, optional
        Number of top features to use
    threshold : float, optional
        Minimum importance threshold for features
    
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    # Make sure we have feature importance data
    if not os.path.exists('models/feature_importance.csv'):
        print("Feature importance not found. Generating it first...")
        analyze_feature_importance()
    
    # Load feature importance
    feature_importance = pd.read_csv('models/feature_importance.csv')
    
    # Select features based on either count or threshold
    if n_features is not None:
        selected_features = feature_importance.head(n_features)['Feature'].tolist()
        print(f"Selected {len(selected_features)} top features by count")
    elif threshold is not None:
        selected_features = feature_importance[feature_importance['Importance'] >= threshold]['Feature'].tolist()
        print(f"Selected {len(selected_features)} features with importance >= {threshold}")
    else:
        # Default to top 20 if neither is specified
        selected_features = feature_importance.head(20)['Feature'].tolist()
        print(f"Selected top 20 features by default")
    
    print("Selected features:", selected_features)
    return selected_features


def get_feature_groups():
    """
    Group features by their type and importance.
    This is useful for understanding which categories of features are most important.
    
    Returns:
    --------
    feature_groups : dict
        Dictionary with feature groups and their average importance
    """
    # Make sure we have feature importance data
    if not os.path.exists('models/feature_importance.csv'):
        print("Feature importance not found. Generating it first...")
        analyze_feature_importance()
    
    # Load feature importance and original data
    feature_importance = pd.read_csv('models/feature_importance.csv')
    data = pd.read_csv('data/train.csv')
    
    # Identify categorical features from the actual data
    categorical_cols = data.drop(columns=['SalePrice', 'Id']).select_dtypes(include=['object']).columns.tolist()
    
    # Create groups of features
    feature_groups = {
        'Location': [col for col in feature_importance['Feature'] if any(x in col for x in ['Neighborhood', 'Condition1', 'Condition2', 'MSZoning'])],
        'Size': [col for col in feature_importance['Feature'] if any(x in col for x in ['Area', 'SF', 'Footage', 'Size', 'LotArea'])],
        'Quality': [col for col in feature_importance['Feature'] if any(x in col for x in ['Qual', 'Quality', 'Class', 'OverallQual', 'OverallCond'])],
        'Year': [col for col in feature_importance['Feature'] if any(x in col for x in ['Year', 'Yr', 'YrSold', 'YearBuilt'])],
        'Bathroom': [col for col in feature_importance['Feature'] if any(x in col for x in ['Bath', 'Toilet', 'Shower'])],
        'Bedroom': [col for col in feature_importance['Feature'] if any(x in col for x in ['Bed', 'Room', 'BR'])],
        'Garage': [col for col in feature_importance['Feature'] if 'Garage' in col],
        'Basement': [col for col in feature_importance['Feature'] if 'Bsmt' in col or 'Basement' in col],
        'External': [col for col in feature_importance['Feature'] if any(x in col for x in ['Exter', 'Out', 'Fence', 'Pool'])]
    }
    
    # Calculate average importance for each group
    group_importance = {}
    for group, features in feature_groups.items():
        if features:
            # Filter feature importance dataframe to only include features in this group
            group_features = feature_importance[feature_importance['Feature'].isin(features)]
            avg_importance = group_features['Importance'].mean()
            group_importance[group] = {
                'avg_importance': avg_importance,
                'count': len(features),
                'top_features': group_features.head(3)['Feature'].tolist()
            }
    
    # Sort groups by importance
    sorted_groups = {k: v for k, v in sorted(group_importance.items(), 
                                            key=lambda item: item[1]['avg_importance'], 
                                            reverse=True)}
    
    # Print group summary
    print("\nFeature Group Importance Summary:")
    for group, data in sorted_groups.items():
        print(f"{group}: {data['count']} features, Avg Importance: {data['avg_importance']:.4f}")
        print(f"  Top features: {', '.join(data['top_features'])}")
    
    return sorted_groups


if __name__ == "__main__":
    # Analyze feature importance and get top features
    top_features = analyze_feature_importance(n_top_features=30, save_plot=True)
    
    # Get feature groups
    feature_groups = get_feature_groups()
    
    print("\nYou can use these top features to train a more efficient model.")
    print("Example usage in train_model.py:")
    print("from feature_important import train_model_with_selected_features")
    print("selected_features = train_model_with_selected_features(n_features=20)")
    print("# Then use selected_features to filter your training data")