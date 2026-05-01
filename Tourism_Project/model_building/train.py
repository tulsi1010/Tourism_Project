
import pandas as pd
import numpy as np
# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV # Changed from GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Ensure any active MLflow run is terminated before starting a new one
if mlflow.active_run():
    mlflow.end_run()

# Set up MLflow tracking from environment variables
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

api = HfApi(token=os.getenv("HF_TOKEN")) # Assumes HF_TOKEN is set as an environment variable

print("Loading preprocessed data...")
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv")
ytest = pd.read_csv("ytest.csv")

print(f"Training set shape: {Xtrain.shape}")
print(f"Test set shape: {Xtest.shape}")

# Identify numeric features (all features after encoding)
numeric_features = Xtrain.columns.tolist()

# Preprocessor - StandardScaler for all numeric features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)


# Define base XGBoost Classifier
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    # Removed 'use_label_encoder=False' as it's deprecated and causes a warning
    random_state=42
)

# Hyperparameter grid for classification (can also be a distribution for RandomizedSearchCV)
param_grid = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__scale_pos_weight': [1, 2, 3]  # Handle class imbalance
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, base_model)

# Start MLflow run

print("\nStarting MLflow experiment...")
with mlflow.start_run():
    print("Performing Randomized Search with Cross-Validation...") # Updated message
    # Randomized Search
    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_grid, # param_distributions instead of param_grid
        n_iter=50, # Set the number of iterations here (e.g., 50, 100, etc.)
        cv=3,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1,
        random_state=42
    ) # Added random_state for reproducibility
    random_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = random_search.cv_results_
    print(f"\nEvaluated {len(results['params'])} parameter combinations")

    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

# Best model

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best mean ROC-AUC: {random_search.best_score_}")

best_model = random_search.best_estimator_

# predictions
print("\nMaking predictions...")
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Probability predictions for ROC-AUC
y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

# Calculate metrics
print("\nCalculating metrics...")
train_accuracy = accuracy_score(ytrain, y_pred_train)
test_accuracy = accuracy_score(ytest, y_pred_test)

train_precision = precision_score(ytrain, y_pred_train)
test_precision = precision_score(ytest, y_pred_test)

train_recall = recall_score(ytrain, y_pred_train)
test_recall = recall_score(ytest, y_pred_test)

train_f1 = f1_score(ytrain, y_pred_train)
test_f1 = f1_score(ytest, y_pred_test)

train_roc_auc = roc_auc_score(ytrain, y_pred_train_proba)
test_roc_auc = roc_auc_score(ytest, y_pred_test_proba)

# Log metrics

mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1_score": train_f1,
        "test_f1_score": test_f1,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc
    })


# Print results
print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Train Precision: {train_precision}")
print(f"Test Precision: {test_precision}")
print(f"Train Recall: {train_recall}")
print(f"Test Recall: {test_recall}")
print(f"Train F1 Score: {train_f1}")
print(f"Test F1 Score: {test_f1}")
print(f"Train ROC-AUC: {train_roc_auc}")
print(f"Test ROC-AUC: {test_roc_auc}")
print("="*50)

print("\nClassification Report - Train:")
print(classification_report(ytrain, y_pred_train))

print("\nClassification Report - Test:")
print(classification_report(ytest, y_pred_test))

# save the model locally

print("\nSaving the model locally...")
model_path = "best_model.joblib"
joblib.dump(best_model, model_path)
print(f"\nModel saved locally as: {model_path}")

 # Log the model artifact

print("\nLogging the model artifact...")
mlflow.log_artifact("best_model.joblib")
print("Model artifact logged successfully.")

# Step 1: Check if the repository exists, if not create it
repo_id_model = "Tulsi10/Tourism" # Define repo_id for the model
repo_type_model = "model" # Define repo_type for the model

print("\nChecking if the model repository exists...")
try:
    api.repo_info(repo_id=repo_id_model, repo_type=repo_type_model)
    print(f"Repository '{repo_id_model}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id_model}' not found. Creating new repository...")
    create_repo(repo_id=repo_id_model, repo_type=repo_type_model, private=False)
    print(f"Repository '{repo_id_model}' created.")

# Upload model to Hugging Face
print("\nUploading the model to Hugging Face...")
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path.split("/")[-1],
    repo_id=repo_id_model,
    repo_type=repo_type_model,
)
print(f"Model uploaded to Hugging Face: {repo_id_model}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
