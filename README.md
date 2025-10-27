# ChronoBridge Service

---

## Overview

The **ChronoBridge Service** extracts trained fused embeddings per asset from a neural fusion model and stores them in **MongoDB** and **Redis** for downstream use.  
It integrates multi-modal data, including **OHLCV market data** and **news embeddings**, leveraging the **NeuralFusionCore** model.

---

## Features

- Run **data ingestion** and **feature extraction** pipelines.
- Load and evaluate **trained model weights**.
- Generate **fused embeddings** for assets.
- Save results to **MongoDB** and **Redis**.
- Supports **sliding window inference** for continuous predictions.

---

## Architecture

1. **Data Ingest Service**: Pulls recent market and news data for the specified window.
2. **Feature Service**: Prepares features for model inference.
3. **NeuralFusionCore Model**: Fuses OHLCV and news embeddings to generate asset-level embeddings.
4. **Prediction Saving**:  
   - Converts NumPy types to native Python types.
   - Saves fused embeddings to **MongoDB** and **Redis**.
5. **Sliding Window Inference**: Supports continuous streaming inference across time-series data.

---

## Setup

### Dependencies

- Python 3.10+
- PyTorch 2.x
- NumPy, pandas
- MongoDB (Python driver: `pymongo`)
- Redis (`redis` Python library)
- NeuralFusionCore module (`apps.NeuralFusionCore`)

Install dependencies:

```bash
pip install torch numpy pandas pymongo redis
```
---

## Configuration

Update the paths and database settings in the script:

# Model checkpoint path
MODEL_CHECKPOINT = "data/outputs/model_weights.pt"

# MongoDB setup

- from pymongo import MongoClient
- mongo_client = MongoClient("mongodb://localhost:27017/")
- mongo_db = mongo_client["portfolio_db"]
- mongo_col = mongo_db["chrono_bridge"]

## Usage

Run the service for the last N hours:
```bash
python chronobridge_service.py --hours 4 --device cpu
```

# Arguments:

--hours: Number of past hours of data to process (default: 4)

--device: Device for model inference (cpu or cuda)


## Workflow

- Clear previous MongoDB entries.

- Run data ingestion service for the specified time window.

- Run feature service in bridge mode.

- Load model weights from checkpoint.

- Preprocess and pad data for sliding window inference.

- Compute fused embeddings per asset.

- Save predictions to Redis and MongoDB.

---

## Outputs

MongoDB collection: chrono_bridge

Stores per-asset fused embeddings with OHLCV features.

Redis cache: chrono_bridge key for fast access to latest embeddings.

# Each record includes:

* date

* symbol (asset)

* fused_embedding

* open, high, low, close, volume
---

## Notes

Model weights should be trained and saved in NeuralFusionCore before running this service.

Sliding window ensures historical context is used for each prediction.

Designed for real-time or batch inference in portfolio pipelines.