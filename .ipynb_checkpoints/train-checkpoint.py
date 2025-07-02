# scripts/train.py 

import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Import our preprocessing function
from preprocess import get_preprocessed_data

# 1. Define mlflow's experiment name
mlflow.set_experiment("Car Price Prediction")

# 2. Load the data
print("Load preprocessed data...")
df, metadata = get_preprocessed_data()

# 3. Separate the target and the other features
X = df.drop(['price', 'price_log'], axis=1)
y = df['price_log']

# Split the data set in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing pipeline for numerical and categorical columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Scaling numerical columns and one hot encoding for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- 5. Training based models ---
print("\n--- 2. Training based models ---")

models_base = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest Base": RandomForestRegressor(random_state=42, n_jobs=-1),
    "XGBoost Base": XGBRegressor(random_state=42, n_jobs=-1)
}

best_mae = float('inf')
best_run_id = None

# 6. train loop
baseline_results = {}

for model_name, model in models_base.items():
    with mlflow.start_run(run_name=f"Base_{model_name}"):
        mlflow.set_tag("model_type", "baseline")
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        #Evaluation on train set
        predictions_log_train=pipeline.predict(X_train)
        y_train_orig=np.expm1(y_train)
        predictions_train_orig=np.expm1(predictions_log_train)

        r2_train = r2_score(y_train_orig, predictions_train_orig)
        mae_train = mean_absolute_error(y_train_orig, predictions_train_orig)
        mse_train=mean_squared_error(y_train_orig, predictions_train_orig)
    
        # Evaluation on test set
        predictions_log_test = pipeline.predict(X_test)
        y_test_orig = np.expm1(y_test)
        predictions_orig = np.expm1(predictions_log_test)
        
        r2_test = r2_score(y_test_orig, predictions_orig)
        mae_test = mean_absolute_error(y_test_orig, predictions_orig)
        mse_test= mean_squared_error(y_test_orig, predictions_orig)
        
        print(f"{model_name} -> R2_train: {r2_train:.4f}, MAE_train: {mae_train:.2f},MSE_train:{mse_train:.2f} , R2_test: {r2_test:.4f}, MAE_test: {mae_test:.2f}, MSE_test:{mse_test:.2f}")
        
        #Log the metrics for train and test set
        mlflow.log_metric("r2_score_train", r2_train)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("mse_train",mse_train)
        mlflow.log_metric("r2_score_test",r2_test)
        mlflow.log_metric("mae_test",mae_test)
        mlflow.log_metric("mse_test",mse_test)
        
        baseline_results[model_name] =r2_test

# Select the 2 best models for optimization
print('Select the 2 best models')
top_models_names = sorted(baseline_results, key=baseline_results.get, reverse=True)[:2]
print(f"\nBest models to optimize : {top_models_names}")


# ---  Hyper parameters optimization for RandomizedSearchCV ---
print("\n--- 7. Hyper parameters optimization ---")

# Paremeters grids for RandomizedSearchCV
param_grids = {
    "Random Forest": {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    },
    "XGBoost": {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_child_weight': [1, 3, 5]
    }
}

# Renaming models
model_map = { "Random Forest Base": "Random Forest", "XGBoost Base": "XGBoost" }
best_mae_overall = float('inf')
best_run_id_overall = None

for model_base_name in top_models_names:
    model_opt_name = model_map.get(model_base_name)
    if not model_opt_name:
        continue # if the model is not the grid

    print(f"\n--- Optimization for : {model_opt_name} ---")
    
    # Run MLflow parent for Optimization
    with mlflow.start_run(run_name=f"Opt_{model_opt_name}", nested=True) as parent_run:
        mlflow.set_tag("model_type", "optimized")
        
        model = models_base[model_base_name] # Récupérer l'instance du modèle de base
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        
        # Setting RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[model_opt_name],
            n_iter=10,  
            cv=3,       
            scoring='neg_mean_absolute_error', 
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        # Reseach of the best parameters
        print('best parametres found')
        random_search.fit(X_train, y_train)

        # Log the best parameters
        mlflow.log_params(random_search.best_params_)
        
        # Take back the best parameters
        best_pipeline = random_search.best_estimator_

        # Final evaluation of overfitting
        print("\nOptimized model evaluation...")
        
        # Score on train set
        train_preds_log = best_pipeline.predict(X_train)
        train_preds_orig = np.expm1(train_preds_log)
        train_mae = mean_absolute_error(np.expm1(y_train), train_preds_orig)
        
        # Score on test set
        test_preds_log = best_pipeline.predict(X_test)
        test_preds_orig = np.expm1(y_test)
        test_mae = mean_absolute_error(np.expm1(y_test), test_preds_orig)
        test_r2 = r2_score(np.expm1(y_test), test_preds_orig)

        print(f"  MAE (Train): {train_mae:.2f}")
        print(f"  MAE (Test):  {test_mae:.2f}")
        print(f"  R2 (Test):   {test_r2:.4f}")
        
        overfitting_gap = ((train_mae - test_mae) / test_mae) * 100
        print(f"  Overfitting (gap MAE Train-Test): {overfitting_gap:.2f}%")

        # Logger the final metrics
        mlflow.log_metric("final_mae_train", train_mae)
        mlflow.log_metric("final_mae_test", test_mae)
        mlflow.log_metric("final_r2_test", test_r2)
        mlflow.log_metric("overfitting_gap_percent", overfitting_gap)
        
        # Logger the best pipeline found
        mlflow.sklearn.log_model(best_pipeline, "optimized_model_pipeline")
        
        # Compare with the best overall model found so far (based on MAE)
        if test_mae < best_mae_overall:
            best_mae_overall = test_mae
            best_run_id_overall = parent_run.info.run_id
            print(f"*** New best model found : {model_opt_name} (MAE={test_mae:.2f}) ***")


# --- 8. Save the best model ---
print(f"\n--- 8. Save the best model ---")
print(f"the best run ID is : {best_run_id_overall}")
print(f"MAE on test set is : {best_mae_overall:.2f}")

# Load the pipeline from the MLflow artefacts of the best run
best_model_uri = f"runs:/{best_run_id_overall}/optimized_model_pipeline"
final_best_pipeline = mlflow.sklearn.load_model(best_model_uri)

# Save the final pipeline
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(final_best_pipeline, os.path.join(models_dir, 'best_model.pkl'))

print(f"\nBest pipeline successfully saved in '{os.path.join(models_dir, 'best_model.pkl')}'")

