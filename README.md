# Predictive Maintenance - MLOps Pipeline

[![Predictive Maintenance MLOps Pipeline](https://github.com/nithyanatarajan/predictive-maintenance/actions/workflows/pipeline.yml/badge.svg)](https://github.com/nithyanatarajan/predictive-maintenance/actions/workflows/pipeline.yml)

An end-to-end MLOps pipeline that predicts engine maintenance needs before failures occur, enabling proactive intervention and reducing unplanned downtime.

## Project Overview

> **Note:** This project was developed as part of the [PGP-AIML](https://www.mygreatlearning.com/pg-program-artificial-intelligence-course) Capstone Project.

Engine failures lead to significant financial losses for both individual owners and fleet operators - unexpected breakdowns cause expensive repairs, operational downtime, and safety risks. This ML-powered system analyzes sensor data to identify engines requiring maintenance, enabling a shift from reactive repairs to proactive scheduling.

**Objective**: Build a predictive maintenance model that classifies whether an engine requires maintenance or is operating normally, by:
- Analyzing historical engine performance data to identify patterns
- Building a classification model to predict maintenance needs
- Implementing the model in a production-ready format

## Pipeline Architecture

```
GitHub Actions Pipeline
├── register-dataset   → Upload raw sensor data to HuggingFace Dataset
├── data-prep          → Feature engineering, preprocessing, and train/val/test split
├── model-training     → Train model with MLflow tracking, upload best model
└── deploy-hosting     → Deploy Streamlit app to HuggingFace Space
```

## Features

- **Automated CI/CD**: GitHub Actions triggers on push to main/master
- **Experiment Tracking**: MLflow logs parameters, metrics, and artifacts
- **Model Registry**: Best model stored on HuggingFace Model Hub
- **Data Versioning**: Raw and processed datasets on HuggingFace Datasets
- **Web Deployment**: Streamlit app hosted on HuggingFace Spaces (Docker)

## Model Performance

| Metric | Score |
|--------|-------|
| **Recall** | **94.21%** |
| F2 Score | 86.66% |
| Precision | 65.62% |

- **Algorithm**: Random Forest Classifier with threshold tuning (0.4)
- **Target**: Engine Condition (0: Normal, 1: Maintenance Required)
- **Top Features**: Engine RPM (18.21%), Fuel Pressure (12.94%), Lub Oil Temp (12.77%)

## Links

| Resource | URL |
|----------|-----|
| **Live App** | [HuggingFace Space](https://huggingface.co/spaces/nithyanatarajan/predictive-maintenance-app) |
| **Dataset** | [HuggingFace Dataset](https://huggingface.co/datasets/nithyanatarajan/predictive-maintenance-data) |
| **Model** | [HuggingFace Model](https://huggingface.co/nithyanatarajan/predictive-maintenance-model) |

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML** | scikit-learn, XGBoost, MLflow |
| **Data** | pandas, numpy |
| **Web** | Streamlit |
| **MLOps** | GitHub Actions, HuggingFace Hub |
| **Container** | Docker |

## Project Structure

```
predictive-maintenance/
├── .github/workflows/
│   └── pipeline.yml          # CI/CD workflow
├── pm/
│   ├── data/
│   │   └── engine_data.csv   # Raw sensor data
│   ├── data_registration/
│   │   └── register.py       # Upload raw data to HuggingFace
│   ├── model_building/
│   │   ├── prep.py           # Data preprocessing
│   │   └── train.py          # Model training with MLflow
│   ├── deployment/
│   │   ├── app.py            # Streamlit application
│   │   ├── Dockerfile        # Container configuration
│   │   └── requirements.txt  # App dependencies
│   └── hosting/
│       └── hosting.py        # Deploy to HuggingFace Space
└── requirements.txt          # Pipeline dependencies
```

## Local Development

```bash
# Clone the repository
git clone https://github.com/nithyanatarajan/predictive-maintenance.git
cd predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export HF_USERNAME="your_username"

# Run individual scripts
python pm/data_registration/register.py
python pm/model_building/prep.py
python pm/model_building/train.py
python pm/hosting/hosting.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace write token (secret) |
| `HF_USERNAME` | HuggingFace username |
| `HF_DATASET_NAME` | Dataset repository name |
| `HF_MODEL_NAME` | Model repository name |
| `HF_SPACE_NAME` | Space repository name |
