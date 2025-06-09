import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform
import joblib
import json
import time
import os

# Load the dataset
data = pd.read_csv('data/train.csv')

# Split features and target
X = data.drop(columns=['SalePrice', 'Id'])
y = data['SalePrice']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# After identifying categorical and numerical columns, create preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
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
        search.fit(X_train, y_train, **fit_kw)
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

    # Evaluate on validation set
    y_pred = search.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f"Validation RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Update best model
    if rmse < best_score:
        best_score, best_pipe, best_name = rmse, search.best_estimator_, name

    # Save checkpoint
    joblib.dump(search.best_estimator_, f"models/{name}_best.pkl")

# Evaluate on test set
if best_pipe is None:
    raise RuntimeError("No model trained successfully!")

print(f"\nBest model = {best_name}  (val RMSE={best_score:.4f})")
y_test_pred = best_pipe.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")