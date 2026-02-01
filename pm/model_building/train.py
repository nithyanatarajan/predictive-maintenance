"""
Production Training Script for Predictive Maintenance

This script:
1. Loads processed data from HuggingFace
2. Trains Random Forest with best hyperparameters
3. Logs experiment to MLflow
4. Uploads model to HuggingFace Hub
"""
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, fbeta_score, f1_score, precision_score, accuracy_score
import joblib
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

def slugify(name: str) -> str:
    """Convert name to HF-compatible slug (underscores to hyphens)."""
    return name.replace("_", "-")

# Configuration
RANDOM_STATE = 42
THRESHOLD = 0.4  # Production threshold for recall optimization

# HuggingFace configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
HF_DATASET_NAME = slugify(os.getenv("HF_DATASET_NAME", "predictive-maintenance-data"))
HF_MODEL_NAME = slugify(os.getenv("HF_MODEL_NAME", "predictive-maintenance-model"))

# Initialize API
api = HfApi(token=HF_TOKEN)
dataset_repo = f"{HF_USERNAME}/{HF_DATASET_NAME}"

# Set MLflow tracking (SQLite - no server needed)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("predictive-maintenance-training")
print(f"MLflow tracking: {mlflow.get_tracking_uri()}")

# Load processed data from HuggingFace
print(f"Loading data from: https://huggingface.co/datasets/{dataset_repo}")

train_path = hf_hub_download(repo_id=dataset_repo, filename="train.csv", repo_type="dataset", token=HF_TOKEN)
val_path = hf_hub_download(repo_id=dataset_repo, filename="val.csv", repo_type="dataset", token=HF_TOKEN)
test_path = hf_hub_download(repo_id=dataset_repo, filename="test.csv", repo_type="dataset", token=HF_TOKEN)

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Separate features and target
feature_cols = ['engine_rpm', 'lub_oil_pressure', 'fuel_pressure', 'coolant_pressure',
                'lub_oil_temp', 'coolant_temp', 'rpm_x_fuel_pressure', 'rpm_bins',
                'oil_health_index']
target_col = 'engine_condition'

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# Best hyperparameters from notebook experimentation
best_params = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}

print(f"\nTraining Random Forest with best parameters: {best_params}")

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(best_params)
    mlflow.log_param("threshold", THRESHOLD)

    # Create pipeline with scaling + classifier
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Evaluate on validation set with threshold
    y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= THRESHOLD).astype(int)

    val_metrics = {
        'val_recall': recall_score(y_val, y_val_pred),
        'val_f2': fbeta_score(y_val, y_val_pred, beta=2),
        'val_f1': f1_score(y_val, y_val_pred),
        'val_precision': precision_score(y_val, y_val_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred)
    }

    # Evaluate on test set with threshold
    y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= THRESHOLD).astype(int)

    test_metrics = {
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f2': fbeta_score(y_test, y_test_pred, beta=2),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred)
    }

    # Log metrics
    mlflow.log_metrics(val_metrics)
    mlflow.log_metrics(test_metrics)

    print(f"\nValidation Metrics (threshold={THRESHOLD}):")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nTest Metrics (threshold={THRESHOLD}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model locally
    model_path = "best_engine_maintenance_model.joblib"
    joblib.dump(model_pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"\nModel saved: {model_path}")

    # Upload to HuggingFace Model Hub
    model_repo_id = f"{HF_USERNAME}/{HF_MODEL_NAME}"
    repo_type = "model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Model repo '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{model_repo_id}' not found. Creating...")
        create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=model_repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to: https://huggingface.co/{model_repo_id}")

print("\nTraining complete!")
