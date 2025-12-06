# FQDN AppID Detection ML System

A production-ready machine learning system for classifying FQDNs (Fully Qualified Domain Names) to their corresponding application IDs.

## Overview

This system addresses the challenge of orphan DNS records - FQDNs that exist in DNS infrastructure but lack proper ownership attribution (appId = 0). Using machine learning, we predict the most likely application owner based on:

- FQDN structural patterns (length, dots, hyphens, depth)
- Domain components (TLD, subdomain, registered domain)
- Text features (TF-IDF, n-grams)
- DNS record data analysis (cloud provider detection, internal IPs)
- Business context (brand, product, market)
- Organizational hierarchy signals (ITSO, business levels)

## Features

- **Multi-class classification** for application IDs
- **Ensemble models** combining XGBoost, LightGBM, and CatBoost
- **Weighted soft voting** for prediction aggregation
- **Confidence scoring** with uncertainty detection
- **Top-k predictions** for manual review
- **Open-set recognition** for unknown FQDNs
- **Experiment tracking** with MLflow
- **Hyperparameter optimization** with Optuna
- **Production API** with FastAPI
- **Docker containerization** for deployment

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd fqdn-appid-detection-1

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with optional deep learning support
uv sync --extra deep

# Install with monitoring support
uv sync --extra monitoring
```

### Training

```bash
# Basic training with ensemble model
uv run python scripts/train.py --labeled-data data/raw/labeled_fqdns.csv

# With cross-validation and MLflow tracking
uv run python scripts/train.py \
    --labeled-data data/raw/labeled_fqdns.csv \
    --model-type ensemble \
    --cross-validate \
    --cv-folds 5 \
    --experiment-name "fqdn_v1"

# Train specific model type
uv run python scripts/train.py \
    --labeled-data data/raw/labeled_fqdns.csv \
    --model-type xgboost

# With hyperparameter optimization
uv run python scripts/train.py \
    --labeled-data data/raw/labeled_fqdns.csv \
    --optimize \
    --n-trials 100 \
    --model-type lightgbm
```

### Prediction

```bash
# Batch prediction
uv run python scripts/predict.py \
    --input data/raw/orphan_fqdns.csv \
    --output predictions.csv \
    --model-name fqdn_classifier

# With confidence filtering
uv run python scripts/predict.py \
    --input data/raw/orphan_fqdns.csv \
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
fqdn-appid-detection-1/
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   ├── model_config.yaml      # Model hyperparameters
│   └── feature_config.yaml    # Feature engineering config
├── data/
│   ├── raw/                   # Raw input data
│   └── processed/             # Processed datasets
├── models/                    # Saved models
├── mlruns/                    # MLflow experiment tracking
├── reports/                   # Evaluation reports
├── src/
│   ├── config/               # Configuration management
│   ├── data/                 # Data loading & preprocessing
│   ├── features/             # Feature engineering
│   │   ├── feature_engineer.py
│   │   ├── feature_pipeline.py
│   │   ├── feature_selector.py
│   │   └── record_data_extractor.py
│   ├── models/               # Model implementations
│   │   ├── base_model.py
│   │   ├── ensemble.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── catboost_model.py
│   ├── training/             # Training pipeline
│   ├── evaluation/           # Evaluation metrics
│   ├── inference/            # Prediction pipeline
│   ├── api/                  # FastAPI application
│   └── utils/                # Utilities & logging
├── scripts/                   # CLI scripts
│   ├── train.py              # Training entry point
│   ├── predict.py            # Batch prediction
│   ├── evaluate.py           # Model evaluation
│   └── generate_dummy_data.py
├── tests/                     # Test suite
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Multi-service orchestration
├── Makefile                   # Common commands
└── pyproject.toml             # Project dependencies
```

## Configuration

### Main Configuration (`config/config.yaml`)

```yaml
data:
  primary_source: csv # or "postgres"
  csv:
    labeled_data: data/raw/labeled_fqdns.csv
    unlabeled_data: data/raw/orphan_fqdns.csv

  split:
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
    stratify_column: appid

  class_config:
    min_samples_per_class: 3
    handle_rare_classes: group

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
    voting: soft
```

## Training Pipeline Architecture

The ensemble training pipeline combines three gradient boosting models with weighted soft voting:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DATA LOADING                                                            │
│     ├── Load labeled_fqdns.csv                                              │
│     └── Load orphan_fqdns.csv (optional)                                    │
│                       ↓                                                     │
│  2. PREPROCESSING                                                           │
│     ├── Filter by target column (appid)                                     │
│     ├── Handle rare classes (min 3 samples/class)                           │
│     └── Create label encoding (ClassMapping)                                │
│                       ↓                                                     │
│  3. FEATURE ENGINEERING                                                     │
│     ├── FQDN Features (length, dots, TLD, char distribution)                │
│     ├── TF-IDF (n-grams from domain tokens)                                 │
│     ├── Record Data Features (cloud provider, IP analysis)                  │
│     ├── Categorical Encoding (record_type, brand, market)                   │
│     ├── App Description Vectorization                                       │
│     ├── Scan Result Extraction                                              │
│     └── Feature Selection (variance-based)                                  │
│                       ↓                                                     │
│  4. DATA SPLITTING                                                          │
│     ├── Train: 70%                                                          │
│     ├── Validation: 15%                                                     │
│     └── Test: 15%                                                           │
│                       ↓                                                     │
│  5. ENSEMBLE TRAINING                                                       │
│     ┌────────────────────────────────────────────────────────┐              │
│     │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │              │
│     │  │  XGBoost    │   │  LightGBM   │   │  CatBoost   │   │              │
│     │  │  (0.4)      │   │  (0.4)      │   │  (0.2)      │   │              │
│     │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   │              │
│     │         │                 │                 │          │              │
│     │         └─────────────────┼─────────────────┘          │              │
│     │                           ↓                            │              │
│     │              ┌─────────────────────┐                   │              │
│     │              │  Weighted Soft      │                   │              │
│     │              │  Voting (prob avg)  │                   │              │
│     │              └─────────────────────┘                   │              │
│     └────────────────────────────────────────────────────────┘              │
│                       ↓                                                     │
│  6. EVALUATION                                                              │
│     ├── Accuracy, F1 Score (macro/weighted)                                 │
│     ├── Precision/Recall                                                    │
│     └── Confusion Matrix                                                    │
│                       ↓                                                     │
│  7. ARTIFACT SAVING                                                         │
│     ├── Model (ensemble + base models)                                      │
│     ├── Preprocessor & Feature Pipeline                                     │
│     └── MLflow Experiment Artifacts                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feature Engineering

### Feature Groups

| Group                | Description          | Examples                                |
| -------------------- | -------------------- | --------------------------------------- |
| **Primary**          | Core FQDN data       | fqdn, record_type, record_data          |
| **Domain Derived**   | Extracted from FQDN  | bus_domain, country_code, num_dots      |
| **Business Context** | Business metadata    | brand, product, market, client_channel  |
| **Technical**        | Infrastructure info  | tech_environment, category, nameserver  |
| **Organizational**   | Hierarchy signals    | itso_id, buslevel4/5/6                  |
| **Record Data**      | DNS target analysis  | cloud_provider, is_internal_ip, is_gslb |
| **Scan Results**     | Scan data extraction | http_unreachable, ssl_cert_error        |

### FQDN Text Features

- **Structural**: length, num_dots, num_hyphens, max_label_length
- **Domain Extraction**: TLD, subdomain, registered domain (via tldextract)
- **Pattern Matching**: environment (dev/staging/prod), service type (api/web/cdn)
- **N-grams**: character n-grams (2-4) and word n-grams
- **TF-IDF**: vectorization on full FQDN, subdomain, and domain components

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

The system uses MLflow for experiment tracking:

```bash
# Set MLflow tracking URI (optional, defaults to ./mlruns)
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run training with tracking
uv run python scripts/train.py --experiment-name "my_experiment"

# View experiments
uv run mlflow ui
```

Then open http://localhost:5000 in your browser.

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

### Input Data Schema

| Column           | Type   | Required | Description                      |
| ---------------- | ------ | -------- | -------------------------------- |
| fqdn             | string | Yes      | Fully qualified domain name      |
| record_type      | string | Yes      | DNS record type (A, CNAME, etc.) |
| record_data      | string | Yes      | DNS record value                 |
| appid            | int    | Yes      | Application ID (0 for orphans)   |
| brand            | string | No       | Brand identifier                 |
| product          | string | No       | Product identifier               |
| market           | string | No       | Market/region                    |
| fqdn_source      | string | No       | Source system                    |
| fqdn_status      | string | No       | Status (Active, Demised)         |
| tech_environment | string | No       | Environment (prod, dev, etc.)    |
| itso_id          | string | No       | ITSO identifier                  |

## Dependencies

Core dependencies (Python ≥3.11):

- **ML**: xgboost, lightgbm, catboost, scikit-learn
- **Data**: polars, pyarrow, connectorx
- **API**: fastapi, uvicorn, pydantic
- **Tracking**: mlflow, optuna
- **Text**: tldextract, rapidfuzz
- **Logging**: loguru, rich

Optional:

- **Deep Learning**: torch, transformers (install with `uv sync --extra deep`)
- **Monitoring**: evidently, prometheus-client (install with `uv sync --extra monitoring`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run tests: `uv run pytest tests/`
5. Run linting: `uv run ruff check src/`
6. Submit a pull request

## License

Proprietary - Internal Use Only
