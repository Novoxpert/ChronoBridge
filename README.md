# ChronoBridge: Multi-Modal Embedding Fusion & Serving Pipeline

ChronoBridge extracts and distributes synchronized fused embeddings per asset from multi-modal data, including OHLCV and news, enabling real-time downstream services to access rich, pre-processed asset features for inference, analytics, and portfolio pipelines.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Layout](#-repository-layout)
- [Setup](#️-setup)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Arguments](#arguments)
- [Workflow](#workflow)
- [Outputs](#outputs)
- [Notes](#notes)
- [Authors & Citation](#-authors--citation)
- [Support](#-support)
---
## Overview

The **ChronoBridge Service** extracts trained fused embeddings per asset from a  **NeuralFusionCore** model and stores them in MongoDB and Redis for downstream use.  
It integrates multi-modal data, including **OHLCV market data** and **news embeddings**, leveraging the **NeuralFusionCore** model.

---

## Key Features

- Run **data ingestion** and **feature extraction** pipelines.
- Load and evaluate **trained model**.
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
## 📁 Repository Layout

```
ChronoBridge/
    ├── scripts/
    │        ├── chronobridge_api_service.py 
    │        └── chronobridge_service.py
    ├──apps/
    │      └──NeuralFusionCore/
    │        ├── data/
    │        │   ├── outputs/
    │        │   │   └── model_weights.pt        
    │        │   │   └── model_weights.pt        
    │        │   │   └── model_weights.pt        
    │        │   │   └── model_weights.pt        
    │        │   │   └── model_weights.pt        
    │        │   └── processed/
    │        │       └── show_files.py                   
    │        │   
    │        ├── lib/
    │        │   ├── backtest.py
    │        │   ├── backtest_weights.py        
    │        │   ├── dataset.py
    │        │   ├── features.py
    │        │   ├── loss_weights.py            
    │        │   ├── market.py
    │        │   ├── model.py
    │        │   ├── news.py
    │        │   ├── redis_utils.py
    │        │   ├── train.py
    │        │   └── utils.py
    │        ├──_init__.py
    │        ├── README.md
    │        ├── requirements.txt
    │        ├── config.py
    │        └── scripts/
    │              ├── data_ingest_service.py
    │              ├── features_service.py
    │              ├── train_service.py
    │              ├── finetune_service.py
    │              ├── prediction_service.py 
    │              └── api_service.py
    └── README.md
```
## ⚙️ Setup

### Dependencies

- Python 3.12+
- PyTorch 2.x
- MongoDB (Python driver: `pymongo`)
- Redis (`redis` Python library)
- NeuralFusionCore module (`apps.NeuralFusionCore`)

Install dependencies:

```bash

# Clone repository
git clone https://github.com/Novoxpert/ChronoBridge.git
cd ChronoBridge


# (optional) create a virtual environment
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows (PowerShell):
 .\.venv\Scripts\Activate.ps1

# install exact dependencies
pip install -r requirements.txt
```
---


## Usage

Run the service for the last N hours:

🔧 Usage Example:
```bash
python chronobridge_service.py --hours 4 --device cpu
```
```bash
python chronobridge_api_service.py 
```

#### Arguments:

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

#### Each record includes:

* date

* symbol (asset)

* fused_embedding

* open, high, low, close, volume
---

## Notes

- Model weights should be trained and saved in NeuralFusionCore before running this service.

- Sliding window ensures historical context is used for each prediction.

- Designed for real-time or batch inference in portfolio pipelines.

---
## 👥 Authors & Citation

**Developed by the [Novoxpert Research Team](https://github.com/Novoxpert)**  
Lead Contributors:
 - [Elham Esmaeilnia](https://github.com/Elham-Esmaeilnia)
 

If you use this repository or build upon our work, please cite:

> Novoxpert Research (2025). *NeuralFusionCore: Direct Portfolio Weight Forecasting with Cross-Gated Attention Fusion.*  
> GitHub: [https://github.com/Novoxpert/NeuralFusionCore](https://github.com/Novoxpert/NeuralFusionCore)

```bibtex
@software{novoxpert_neuralfusioncore_2025,
  author       = {Elham Esmaeilnia},
  title        = {NeuralFusionCore: Direct Portfolio Weight Forecasting with Cross-Gated Attention Fusion},
  organization = {Novoxpert Research},
  year         = {2025},
  url          = {https://github.com/Novoxpert/NeuralFusionCore}
}
```
---
## 📞 Support

- **Issues & Bugs**: [Open on GitHub](https://github.com/Novoxpert/neuralfusioncore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Novoxpert/neuralfusioncore/discussions)
- **Feature Requests**: Open a feature request issue
---