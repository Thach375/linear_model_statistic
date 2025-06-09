import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Determine model directory path
script_dir = Path(__file__).parent
models_dir = script_dir / "models"
if not models_dir.exists():
    models_dir = Path("models")  # Try the root models directory

# Load the test dataset
print("Loading test data...")
test_data = pd.read_csv('data/test.csv')
test_ids = test_data['Id']

# Try to load sample submission to compare with - this could have "target" values
try:
    sample_submission = pd.read_csv('data/sample_submission.csv')
    has_sample_targets = True
    print("Loaded sample submission file for comparison")
except:
    has_sample_targets = False
    print("No sample submission file found for comparison")

# Remove Id column for prediction
X_test = test_data.drop(columns=['Id'])

# Load the best model
best_models = {}
best_model_name = None
best_rmse = float('inf')

# Find available models and their metrics
for model_json in models_dir.glob('*_best.json'):
    model_name = model_json.stem.replace('_best', '')
    model_path = models_dir / f"{model_name}_best.pkl"
    
    if model_path.exists():
        # Load model performance metrics
        with open(model_json, 'r') as f:
            params = json.load(f)
        
        best_models[model_name] = {
            'path': model_path,
            'params': params
        }

model_names = list(best_models.keys())
all_predictions = {}

# Evaluate each model and save predictions
for i in range(len(model_names)):
    selected_model = model_names[i]
    # Load the selected model
    print(f"\n{selected_model} model...")
    model = joblib.load(best_models[selected_model]['path'])

    # Make predictions
    predictions = model.predict(X_test)
    all_predictions[selected_model] = predictions

    # Create submission file
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })

    # Ensure predictions are non-negative (house prices can't be negative)
    submission['SalePrice'] = submission['SalePrice'].clip(lower=0)

    # Save submission file
    submission_path = f"submission_{selected_model}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Model parameters: {best_models[selected_model]['params']}")
    
    # Compare with sample submission if available
    if has_sample_targets:
        mse = mean_squared_error(sample_submission['SalePrice'], submission['SalePrice'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(sample_submission['SalePrice'], submission['SalePrice'])
        print(f"Metrics compared to sample submission:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

# Compare models with each other
print("\n=== Model Comparison ===")
if len(model_names) > 1:
    print("\nMSE between model predictions:")
    mse_matrix = {}
    for model1 in model_names:
        mse_matrix[model1] = {}
        for model2 in model_names:
            if model1 != model2:
                mse = mean_squared_error(all_predictions[model1], all_predictions[model2])
                mse_matrix[model1][model2] = mse
    
    # Display MSE matrix as a table
    mse_df = pd.DataFrame(mse_matrix)
    print(mse_df)
    
    # Find models with most similar predictions
    min_mse = float('inf')
    similar_models = (None, None)
    for model1 in model_names:
        for model2 in model_names:
            if model1 != model2:
                if mse_matrix[model1][model2] < min_mse:
                    min_mse = mse_matrix[model1][model2]
                    similar_models = (model1, model2)
    
    print(f"\nMost similar models: {similar_models[0]} and {similar_models[1]} (MSE: {min_mse:.2f})")

print("\nEvaluation complete!")
