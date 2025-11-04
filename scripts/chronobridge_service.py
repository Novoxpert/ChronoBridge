"""
chronobridge_service.py
===================
Extracts fused multimodal embeddings (market data + news) for each asset
from a trained NeuralFusionCore model and stores them for downstream
temporal research and portfolio analytics.

----------------------------------------------------------------------------
Pipeline Overview
----------------------------------------------------------------------------
1. Fetch latest raw market & news data (`data_ingest_service`)
2. Generate lagged features + embeddings in *bridge mode*
3. Slide through the time series row-by-row:
     ▸ build rolling window of features and news embeddings
     ▸ run model forward pass with `return_embeddings=True`
     ▸ extract per-asset fused representations
4. Persist fused embeddings, OHLCV, and timestamp to:
     ▸ MongoDB  (collection: `chrono_bridge`)
     ▸ Redis    (key: `chrono_bridge`)
----------------------------------------------------------------------------
CLI Usage
----------------------------------------------------------------------------
Default mode runs the full chrono-bridge pipeline:

    python chronobridge_service.py

Optional arguments:

    --hours <N>     Number of past hours to ingest (default: 4)
    --mode <mode>   Feature generation mode (default: "synchronize")
                    Modes:
                        synchronize  → full sync pipeline for latest window
                        bridge    
    --device cpu|cuda   Torch device override (default: cpu)

Examples:

    # Full refresh (ingest + features + fused embeddings)
    python chronobridge_service.py --hours 6 --mode synchronize

----------------------------------------------------------------------------
Notes
----------------------------------------------------------------------------
• Clears previous MongoDB entries at start of run.
• Automatically falls back to CPU if CUDA is unavailable.
• Converts NumPy types to Python primitives for MongoDB compliance.
• Requires a trained NeuralFusionCore weights file.

----------------------------------------------------------------------------
Author: Elham Esmaeilnia
Date: 2025 oct 19
Version: 1.2.1
"""

import os, sys, subprocess, logging, pickle, torch, numpy as np, pandas as pd, json
from pymongo import MongoClient
from apps.NeuralFusionCore.lib.model import MarketNewsFusionWeightModel
from apps.NeuralFusionCore.config import Paths, FeatureCfg, MarketCfg, TrainCfg, BacktestCfg
import time
from apps.NeuralFusionCore.lib.redis_utils import redis_client
from dotenv import load_dotenv

# ---------------- Config ----------------
P = Paths(); F = FeatureCfg(); MC = MarketCfg(); T = TrainCfg(); B = BacktestCfg()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

MODEL_CHECKPOINT = "data/outputs/model_weights.pt"

# --------------------------- MongoDB setup ---------------------------
load_dotenv()
NOVO_MONGO_USER = os.getenv("NOVO_MONGO_USER")
NOVO_MONGO_PASS = os.getenv("NOVO_MONGO_PASS")
NOVO_MONGO_HOST = os.getenv("NOVO_MONGO_HOST")
NOVO_MONGO_PORT = os.getenv("NOVO_MONGO_PORT")
NOVO_MONGO_AUTH_DB = os.getenv("NOVO_MONGO_AUTH_DB")
NOVO_MONGO_DB = os.getenv("NOVO_MONGO_DB")

# Connect to MongoDB using the credentials
client = MongoClient(f"mongodb://{NOVO_MONGO_USER}:{NOVO_MONGO_PASS}@{NOVO_MONGO_HOST}:{NOVO_MONGO_PORT}/?authSource={NOVO_MONGO_AUTH_DB}")

# Access your database and collection
db = client[NOVO_MONGO_DB]
mongo_col = db["chrono_bridge"]
# --------------------------- Helper: convert NumPy to Python types ---------------------------
def to_python_types(obj):
    """Recursively convert NumPy scalar types to native Python types for MongoDB compatibility."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# --------------------------- Data ingest & feature service ---------------------------
def run_data_ingest(hours):
    logging.info(f"Running data_ingest_service to fetch last {hours} hour(s) of data")
    subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.data_ingest_service', '--mode', 'latest', '--hours', str(hours)], check=True)

def run_feature_service(hours, mode="synchronize"):
    logging.info(f"Running features_service in INFERENCE mode for last {hours} hour(s)")
    if mode=="bridge":
        subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.features_service', '--mode', 'bridge', '--latest_hours', str(hours)], check=True)
    elif mode=="synchronize":    
        subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.features_service', '--mode', 'synchronize', '--latest_hours', str(hours)], check=True)
    else:
        logging.error(f"Mode {mode} not recognized.")

# --------------------------- Model loader ---------------------------
def load_model(configs, feat_cols_len, stock_list_len, count_dim, device='cpu'):
    model = MarketNewsFusionWeightModel(
        configs=configs,
        ts_input_dim=feat_cols_len,
        num_stocks=stock_list_len,
        d_model=T.d_model,
        nhead=T.nhead,
        num_layers=T.num_layers,
        news_embed_dim=768,
        hidden_dim=T.hidden_dim,
        count_dim=count_dim,
        max_len=F.seq_len
    ).to(device)

    weights_path = getattr(P, "weights_pt", MODEL_CHECKPOINT)

    try:
        print(f"⚙️ [load_model] weights_path={weights_path}")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)
            logging.info(f"✅ Loaded model weights from {weights_path}")
        else:
            logging.warning(f"⚠️ No model weights found at {weights_path}, using untrained model.")
    except Exception as e:
        logging.error(f"❌ Failed to load weights: {e}")

    model.eval()
    return model

# --------------------------- Save predictions ---------------------------
def save_fused_embedding_predictions(date, fused, df_row, stock_list):
    records = []
    for stock_idx, stock in enumerate(stock_list):

        open_col = f"{stock}_open"
        open_val = df_row[open_col] if open_col in df_row else None
        high_col = f"{stock}_high"
        high_val = df_row[high_col] if high_col in df_row else None
        low_col = f"{stock}_low"
        low_val = df_row[low_col] if low_col in df_row else None
        close_col = f"{stock}_close"
        close_val = df_row[close_col] if close_col in df_row else None
        volume_col = f"{stock}_volume"
        volume_val = df_row[volume_col] if volume_col in df_row else None
        records.append({
            "date": str(date),
            "symbol": stock,
            "fused_embedding": fused[stock_idx].tolist(),
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val
        })

    # Save to Redis
    redis_client.set("chrono_bridge", pickle.dumps(records))
    logging.info("Saved bridge results to Redis")

    # Convert NumPy types → native Python before inserting into MongoDB
    clean_records = [to_python_types(r) for r in records]
    mongo_col.insert_many(clean_records)
    logging.info("Saved bridge results to MongoDB")


# --------------------------- Inference with sliding window ---------------------------
def run_inference(df_not_norm_te, df_te, feat_cols, data_stamp_cols, stock_list, cnt_cols, device='cpu'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = F.seq_len
    count_dim = len(cnt_cols) if cnt_cols else 0

    configs = {
        'task_name': 'classification',
        'seq_len': seq_len,
        'enc_in': len(feat_cols),
        'd_model': 64,
        'c_out': 2,
        'd_ff': 128,
        'num_kernels': 3,
        'dropout': 0.1,
        'e_layers': 2,
        'top_k': 3,
        'num_class': 2,
        'label_len': 30,
        'pred_len': 1,
        'embed': 'timeF',
        'freq': 't'
    }

    model = load_model(configs, len(feat_cols), len(stock_list), count_dim, device)

    # Convert news embeddings column to np.array
    df_te["embedding"] = df_te["embedding"].apply(lambda x: np.asarray(x, dtype=np.float32))
    date_col = "date" if "date" in df_te.columns else "dateTime"
    num_rows = len(df_te)

    # Prepend first row seq_len times to pad history for first rows
    pad = pd.concat([df_te.iloc[[0]]] * seq_len, ignore_index=True)
    df_padded = pd.concat([pad, df_te], ignore_index=True)

    for idx in range(num_rows):
        # previous seq_len rows only
        window = df_padded.iloc[idx : idx + seq_len]

        ts_input = torch.tensor(window[feat_cols].values.astype(np.float32)).unsqueeze(0).to(device)
        news_input = torch.tensor(np.stack(window["embedding"].values), dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(window[data_stamp_cols].values.astype(np.float32)).unsqueeze(0).to(device)
        count_input = torch.tensor(window[cnt_cols].values.astype(np.float32)).unsqueeze(0).to(device) if count_dim > 0 else torch.zeros((1, seq_len, 1), dtype=torch.float32).to(device)

        with torch.no_grad():
            fused = model(ts_input, mask, count_input, news_input, return_embeddings=True)

        fused = fused[0].cpu().numpy()

        # Save fused embedding for the current row
        save_fused_embedding_predictions(df_te.iloc[idx][date_col], fused, df_not_norm_te.iloc[idx], stock_list)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{num_rows} rows")

# --------------------------------------------------------------------------------------
def main():
    start_service_time = time.time()
    torch.cuda.empty_cache()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=4)
    parser.add_argument("--mode", type=int, default="synchronize")
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

    # ---------- Clear previous MongoDB data ----------
    logging.info("Clearing previous fused embeddings in MongoDB...")
    mongo_col.delete_many({})
    logging.info("Previous MongoDB data cleared.")
    
    run_data_ingest(args.hours)
    run_feature_service(args.hours, args.mode)

    online_merged_path = os.path.join(P.processed_dir, "online_bridge.parquet")
    if not os.path.exists(online_merged_path):
        logging.error(f"{online_merged_path} not found. Exiting.")
        return
    df_te = pd.read_parquet(online_merged_path)

    meta_path = os.path.join(P.processed_dir, 'meta.json')
    meta = json.load(open(meta_path))
    feat_cols = meta['feature_cols']
    data_stamp_cols = meta['data_stamp_cols']
    stock_list = meta['stock_list']
    cnt_cols = meta.get('count_cols', [])

    online_not_norm_merged_path = os.path.join(P.processed_dir, "online_bridge_not_norm.parquet")
    if not os.path.exists(online_not_norm_merged_path):
        logging.error(f"{online_not_norm_merged_path} not found. Exiting.")
        return
    df_te_not_norm = pd.read_parquet(online_not_norm_merged_path)

    run_inference(df_te_not_norm, df_te, feat_cols, data_stamp_cols, stock_list, cnt_cols, device=args.device)
    logging.info("Fused embedding prediction cycle complete.")
    print(f"Time elapsed: {time.time() - start_service_time:.2f} seconds")

if __name__ == "__main__":
    main()
