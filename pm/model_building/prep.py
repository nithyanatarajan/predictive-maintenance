"""
Data Preparation Script for Predictive Maintenance

This script:
1. Downloads raw data from HuggingFace
2. Applies feature engineering (rule-based transformations)
3. Performs stratified train/val/test split (75/10/15)
4. Uploads processed splits to HuggingFace
"""
import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

def slugify(name: str) -> str:
    """Convert name to HF-compatible slug."""
    return name.replace("_", "-")

# Configuration
HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_NAME = slugify(os.getenv("HF_DATASET_NAME", "predictive-maintenance-data"))
RANDOM_STATE = 42

# Initialize API client
api = HfApi(token=HF_TOKEN)

# Load raw data from HuggingFace
repo_id = f"{HF_USERNAME}/{HF_DATASET_NAME}"
print(f"Loading raw data from: https://huggingface.co/datasets/{repo_id}")
raw_data_path = hf_hub_download(repo_id=repo_id, filename="engine_data.csv", repo_type="dataset", token=HF_TOKEN)
df = pd.read_csv(raw_data_path)
print(f"Loaded {len(df):,} rows")

# Column Standardization (snake_case)
print("\nStandardizing column names to snake_case...")
column_mapping = {
    'Engine rpm': 'engine_rpm',
    'Lub oil pressure': 'lub_oil_pressure',
    'Fuel pressure': 'fuel_pressure',
    'Coolant pressure': 'coolant_pressure',
    'lub oil temp': 'lub_oil_temp',
    'Coolant temp': 'coolant_temp',
    'Engine Condition': 'engine_condition'
}
df = df.rename(columns=column_mapping)
print(f"Columns: {list(df.columns)}")

# Feature Engineering (Rule-Based - safe before split)
print("\nApplying feature engineering...")

# 1. RPM × Fuel Pressure interaction
df['rpm_x_fuel_pressure'] = df['engine_rpm'] * df['fuel_pressure']

# 2. RPM Operating Bins (ordinal: 0=Idle, 1=Normal, 2=High-Load)
def categorize_rpm(rpm):
    if rpm < 300:
        return 0  # Idle
    elif rpm <= 1500:
        return 1  # Normal
    else:
        return 2  # High-Load

df['rpm_bins'] = df['engine_rpm'].apply(categorize_rpm)

# 3. Oil Health Index (viscosity degradation indicator)
df['oil_health_index'] = df['lub_oil_pressure'] / df['lub_oil_temp']

print(f"Created features: rpm_x_fuel_pressure, rpm_bins, oil_health_index")
print(f"Total features: {len(df.columns) - 1}")

# Define features and target
feature_cols = ['engine_rpm', 'lub_oil_pressure', 'fuel_pressure', 'coolant_pressure',
                'lub_oil_temp', 'coolant_temp', 'rpm_x_fuel_pressure', 'rpm_bins',
                'oil_health_index']
target_col = 'engine_condition'

X = df[feature_cols]
y = df[target_col]

# Stratified Split: 75/10/15
print("\nPerforming stratified split (75/10/15)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.60, stratify=y_temp, random_state=RANDOM_STATE
)

# Create DataFrames with target
train_df = X_train.copy()
train_df[target_col] = y_train

val_df = X_val.copy()
val_df[target_col] = y_val

test_df = X_test.copy()
test_df[target_col] = y_test

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Save locally
os.makedirs("pm/data/processed", exist_ok=True)
train_df.to_csv("pm/data/processed/train.csv", index=False)
val_df.to_csv("pm/data/processed/val.csv", index=False)
test_df.to_csv("pm/data/processed/test.csv", index=False)
print("\nSaved to pm/data/processed/")

# Upload processed splits to HuggingFace
print("\nUploading processed splits to HuggingFace...")
for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    path = f"pm/data/processed/{name}.csv"
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f"{name}.csv",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {name}.csv")

print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
print("Data preparation complete!")
