import pandas as pd
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform
import joblib
import json
import time
import os

# Load feature importance data
print("Loading important features from feature_importance.csv...")
try:
    feature_importance = pd.read_csv('models/feature_importance.csv')
    # Get top features (you can adjust the number)
    important_features = feature_importance['Feature'].tolist()
    print(f"Using features: {important_features}")
except FileNotFoundError:
    print("Warning: feature_importance.csv not found. Using all features.")
    important_features = None

# Load the dataset
data = pd.read_csv('data/train.csv')

# Split features and target
X_full = data.drop(columns=['SalePrice', 'Id'])
y = data['SalePrice']

# Filter for important features if available
if important_features:
    # Some features in the importance file might be one-hot encoded names
    # We need to extract original column names before one-hot encoding
    original_feature_names = []
    for feature in important_features:
        # Check if feature exists directly in the dataset
        if feature in X_full.columns:
            original_feature_names.append(feature)
        else:
            # This might be a one-hot encoded feature, extract the column name
            for col in X_full.columns:
                if col in feature:
                    if col not in original_feature_names:
                        original_feature_names.append(col)
                        
    X = X_full[original_feature_names]
else:
    X = X_full

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Define K-fold cross validation strategy
k_folds = 5  # Number of folds
cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Phân loại biến categorical thành có thứ tự và không có thứ tự
ordinal_features = ['OverallQual', 'BsmtQual', 'KitchenQual']

nominal_features = [col for col in categorical_cols if col not in ordinal_features]

# Tạo transformers riêng cho từng loại dữ liệu
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # drop='first' để giảm đa cộng tuyến
])

# Kết hợp các transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_features),
        ('nom', nominal_transformer, nominal_features)
    ])

# Define models and hyperparameter distributions
def new_func(preprocessor):
    models = {
    "LinearRegression": (
        LinearRegression(),
        {
            "model__fit_intercept": [True, False]
        },
        preprocessor, 
        {}
    ),
    "Ridge": (
        Ridge(random_state=42),
        {
            "model__alpha": uniform(0.01, 100),
            "model__fit_intercept": [True, False]
        },
        preprocessor, 
        {}
    ),
    "Lasso": (
        Lasso(random_state=42, max_iter=10000),
        {
            "model__alpha": uniform(0.01, 100),
            "model__fit_intercept": [True, False]
        },
        preprocessor, 
        {}
    ),
    "ElasticNet": (
        ElasticNet(random_state=42, max_iter=10000),
        {
            "model__alpha": uniform(0.01, 100),
            "model__l1_ratio": uniform(0.0, 1.0),
            "model__fit_intercept": [True, False]
        },
        preprocessor,
        {}
    ),
    "PolynomialRegression": (
        LinearRegression(),
        {
            "prep__poly__degree": [1, 2, 3],  # Fix the parameter path
            "model__fit_intercept": [True, False]
        },
        Pipeline([
            ('preprocess', preprocessor),
            ('poly', PolynomialFeatures())
        ]),
        {}
    )
}
    
    return models

models = new_func(preprocessor)

# Train and validate models
best_pipe, best_score, best_name = None, float('inf'), ""

os.makedirs("models", exist_ok=True)

for name, (est, dist, prep, fit_kw) in models.items():
    n_iter = 50
    print(f"\n▶▶ {name}: Randomized search ({n_iter} configs, 3-fold)")

    # Create the pipeline with prep only if it's provided
    if prep is None:
        pipe = Pipeline([('model', est)])
    else:
        pipe = Pipeline([('prep', prep), ('model', est)])
        
    search = RandomizedSearchCV(
        pipe, dist, n_iter=n_iter,
        scoring='neg_root_mean_squared_error', n_jobs=4, random_state=42, verbose=1, refit=True
    )
    tic = time.time()
    try:
        search.fit(X, y, **fit_kw)
    except Exception as e:
        print(f"↳ Error in {name}: {str(e)}")
        continue
    toc = time.time()
    print(f"↳ Done in {(toc-tic)/60:.1f} min — best RMSE={-search.best_score_:.4f}")

    # Save best hyperparameters
    with open(f"models/{name}_best.json", "w") as fp:
        json.dump(search.best_params_, fp, indent=2, default=str)

    # Print top-5 configurations
    top5 = (pd.DataFrame(search.cv_results_)
            .sort_values("rank_test_score")
            .head(5)[["mean_test_score", "params"]])
    top5['mean_test_score'] = -top5['mean_test_score']  # Convert to positive RMSE
    print(top5.to_string(index=False))

    # K-fold validation is already handled in RandomizedSearchCV
    # We'll use the best CV score as our validation metric
    cv_rmse = -search.best_score_  # Convert back to positive RMSE
    print(f"Cross-Validation RMSE: {cv_rmse:.4f}")    # Update best model
    if cv_rmse < best_score:
        best_score, best_pipe, best_name = cv_rmse, search.best_estimator_, name

    # Save checkpoint
    joblib.dump(search.best_estimator_, f"models/{name}_best.pkl")

# Final report
if best_pipe is None:
    raise RuntimeError("No model trained successfully!")

print(f"\nBest model = {best_name}  (val RMSE={best_score:.4f})")