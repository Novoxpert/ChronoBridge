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

import os, sys, subprocess, logging, torch, time
from pymongo import MongoClient
from dotenv import load_dotenv

# --- Universal import fix (works standalone or as submodule) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHRONO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PARENT = os.path.basename(os.path.dirname(CHRONO_DIR))

if PARENT == "apps":  # running inside AlphaFusionNet/apps/
    ROOT = os.path.abspath(os.path.join(CHRONO_DIR, "..", ".."))
    sys.path.insert(0, ROOT)
else:  # running as standalone ChronoBridge repo
    sys.path.insert(0, CHRONO_DIR)

try:
    from apps.ChronoBridge.config import Paths, FeatureCfg, MarketCfg, TrainCfg, BacktestCfg
    from apps.ChronoBridge.src.inference import NeuralFusionCore_infer
except ImportError:
    from ..config import Paths, FeatureCfg, MarketCfg, TrainCfg, BacktestCfg
    from src.inference import NeuralFusionCore_infer


# ---------------- Config ----------------
P = Paths(); F = FeatureCfg(); MC = MarketCfg(); T = TrainCfg(); B = BacktestCfg()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

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

# --------------------------- Data ingest & feature service ---------------------------
def _resolve_module_path(submodule: str) -> str:
    """
    Dynamically resolve correct module import path whether running from
    Alphafusionnet root or directly inside ChronoBridge.
    """
    # Detect if "apps" package is available in sys.modules (means running from AlphaFusionNet)
    if "apps" in sys.modules or os.path.basename(os.getcwd()) == "AlphaFusionNet":
        base_module = "apps.ChronoBridge.scripts"
    else:
        # Local fallback (running inside ChronoBridge folder directly)
        base_module = "ChronoBridge.scripts"
    return f"{base_module}.{submodule}"

def run_data_ingest(hours: int):
    """
    Launch the data ingestion subprocess to fetch latest data.
    Works both inside ChronoBridge and from Alphafusionnet root.
    """
    logging.info(f"Running data_ingest_service to fetch last {hours} hour(s) of data")
    target_module = _resolve_module_path("data_ingest_service")
    subprocess.run(
        [sys.executable, "-m", target_module, "--mode", "latest", "--hours", str(hours)],
        check=True
    )


def run_feature_service(hours: int, mode: str = "synchronize"):
    """
    Launch the feature service subprocess in bridge or synchronize mode.
    Works both inside ChronoBridge and from Alphafusionnet root.
    """
    logging.info(f"Running features_service in INFERENCE mode for last {hours} hour(s)")
    target_module = _resolve_module_path("features_service")

    if mode not in ("bridge", "synchronize"):
        logging.error(f"Mode '{mode}' not recognized.")
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            target_module,
            "--mode",
            mode,
            "--latest_hours",
            str(hours),
        ],
        check=True,
    )

# --------------------------------------------------------------------------------------
def main():
    start_service_time = time.time()
    torch.cuda.empty_cache()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=None, help="Number of hours to fetch data for.")
    parser.add_argument("--history_days", type=int, default=None, help="Number of days of history to convert to hours.")
    parser.add_argument("--mode", type=str, default="synchronize")
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

     # ---------- Convert history_days to hours if provided ----------
    if args.history_days is not None:
        args.hours = args.history_days * 24
        logging.info(f"Converted history_days={args.history_days} to hours={args.hours}.")
    elif args.hours is None:
        # fallback default (if neither provided)
        args.hours = 6
        logging.info("No history_days or hours specified, using default 6 hours.")

    # ---------- Fetch data and Feature extraction ---------- 
    run_data_ingest(args.hours)
    run_feature_service(args.hours, args.mode)
    
    #---------- Get FusedEmbedding from NeuralFusionCore --------
    #logging.info("Clearing previous fused embeddings in MongoDB...")
    #mongo_col.delete_many({})
    #logging.info("Previous MongoDB data cleared.")

    NFC_infer = NeuralFusionCore_infer()
    model_checkpoint_path = "apps/NeuralFusionCore/data/outputs/model_weights.pt"
    NFC_infer.FusedEmbedding(model_checkpoint= model_checkpoint_path, mongo_collection = mongo_col, device=args.device)
    logging.info("Fused embedding prediction cycle complete.")

    print(f"Time elapsed: {time.time() - start_service_time:.2f} seconds")

if __name__ == "__main__":
    main()
