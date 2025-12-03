# FQDN Orphan Detection ML System

A production-ready machine learning system for classifying orphan FQDNs (Fully Qualified Domain Names) to their corresponding application IDs.

## Overview

This system addresses the challenge of orphan DNS records - FQDNs that exist in DNS infrastructure but lack proper ownership attribution (appId = 0). Using machine learning, we predict the most likely application owner based on:

- FQDN structural patterns
- Domain components (TLD, subdomain, etc.)
- Text similarity (TF-IDF)
- Historical ownership patterns
- Organizational hierarchy signals

## Features

- **Multi-class classification** for ~2000 unique application IDs
- **Ensemble models** combining XGBoost, LightGBM, and CatBoost
- **Confidence scoring** with uncertainty detection
- **Top-k predictions** for manual review
- **Open-set recognition** for truly unknown FQDNs
- **Experiment tracking** with MLflow
- **Production API** with FastAPI
- **Docker containerization** for deployment

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd fqdn-orphan-detection

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Training

```bash
# Basic training
uv run python scripts/train.py --labeled-data data/raw/labeled.csv

# With cross-validation and MLflow tracking
uv run python scripts/train.py \
    --labeled-data data/raw/labeled.csv \
    --model-type ensemble \
    --cross-validate \
    --experiment-name "fqdn_v1"

# With hyperparameter optimization
uv run python scripts/train.py \
    --labeled-data data/raw/labeled.csv \
    --optimize \
    --n-trials 100
```

### Prediction

```bash
# Batch prediction
uv run python scripts/predict.py \
    --input data/raw/orphans.csv \
    --output predictions.csv \
    --model-name fqdn_classifier

# With confidence filtering
uv run python scripts/predict.py \
    --input data/raw/orphans.csv \
    --output predictions.csv \
    --confidence-threshold 0.5
```

### API Server

```bash
# Start API server
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up api
```

## Project Structure

```
fqdn-orphan-detection/
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   ├── model_config.yaml      # Model hyperparameters
│   └── feature_config.yaml    # Feature engineering config
├── data/
│   ├── raw/                   # Raw input data
│   ├── processed/             # Processed datasets
│   └── external/              # External data sources
├── models/
│   ├── trained/               # Saved models
│   └── experiments/           # Experiment artifacts
├── src/
│   ├── config/               # Configuration management
│   ├── data/                 # Data loading & preprocessing
│   ├── features/             # Feature engineering
│   ├── models/               # Model implementations
│   ├── training/             # Training pipeline
│   ├── evaluation/           # Evaluation metrics
│   ├── inference/            # Prediction pipeline
│   ├── api/                  # FastAPI application
│   └── utils/                # Utilities
├── scripts/                   # CLI scripts
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks
└── docs/                      # Documentation
```

## Configuration

### Main Configuration (`config/config.yaml`)

```yaml
data:
  primary_source: csv # or "postgres"
  csv:
    labeled_path: data/raw/labeled_data.csv
    unlabeled_path: data/raw/unlabeled_data.csv

  split:
    train_ratio: 0.70
    val_ratio: 0.15
    test_ratio: 0.15

target:
  column: appid
  orphan_value: 0
```

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  primary: ensemble

  ensemble:
    models:
      - name: xgboost
        weight: 0.4
      - name: lightgbm
        weight: 0.4
      - name: catboost
        weight: 0.2
```

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"fqdn": "api.dev.example.com", "record_type": "A"}'
```

Response:

```json
{
  "fqdn": "api.dev.example.com",
  "predicted_appid": 1234,
  "confidence": 0.87,
  "top_k_predictions": [
    { "appid": 1234, "probability": 0.87 },
    { "appid": 5678, "probability": 0.08 },
    { "appid": 9012, "probability": 0.03 }
  ],
  "is_uncertain": false
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "records": [
            {"fqdn": "api.dev.example.com"},
            {"fqdn": "www.production.app.co.uk"}
        ]
    }'
```

## Docker Deployment

```bash
# Build and run training
docker-compose up training

# Build and run API
docker-compose up api

# Run with monitoring stack
docker-compose --profile monitoring up
```

## Experiment Tracking

The system integrates with MLflow for experiment tracking:

```bash
# Set MLflow tracking URI (optional, defaults to ./mlruns)
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run training with tracking
uv run python scripts/train.py --experiment-name "my_experiment"
```

View experiments by running:

```bash
mlflow ui
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_feature_engineer.py -v
```

## Performance Metrics

Expected performance on production data:

| Metric            | Value        |
| ----------------- | ------------ |
| Accuracy          | 85-90%       |
| F1 (macro)        | 0.75-0.85    |
| Top-3 Accuracy    | 95%+         |
| Inference Latency | <10ms/record |

## Data Requirements

### Labeled Data Schema

| Column      | Type   | Description                      |
| ----------- | ------ | -------------------------------- |
| fqdn        | string | Fully qualified domain name      |
| record_type | string | DNS record type (A, CNAME, etc.) |
| record_data | string | DNS record value                 |
| appid       | int    | Application ID (0 for orphans)   |
| brand       | string | Brand identifier                 |
| product     | string | Product identifier               |
| market      | string | Market/region                    |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

Proprietary - Internal Use Only
