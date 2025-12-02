"""
chronobridge_service.py
===================
Extracts fused multimodal embeddings (market data + news) for each asset
from a trained NeuralFusionCore model and stores them for downstream
temporal research and portfolio analytics.

----------------------------------------------------------------------------
Pipeline Overview
----------------------------------------------------------------------------
1. Fetch latest/raw or custom-range market & news data (`data_ingest_service`)
2. Generate lagged features + embeddings in *bridge/synchronize mode*
3. Slide through the time series row-by-row:
     ▸ build rolling window of features and news embeddings
     ▸ run model forward pass with `return_embeddings=True`
     ▸ extract per-asset fused representations
4. Persist fused embeddings, OHLCV, and timestamp to:
     ▸ MongoDB  (collection: `chrono_bridge`)
----------------------------------------------------------------------------
CLI Usage
----------------------------------------------------------------------------
Default mode runs the full chrono-bridge pipeline over the last N hours:

    python chronobridge_service.py --hours 6 --mode synchronize

History in days (internally converted to hours):

    python chronobridge_service.py --history_days 3 --mode synchronize

Custom UTC time range (start_date / end_date):

    python chronobridge_service.py \
        --start_date "2025-02-26 00:00:00" \
        --end_date   "2025-02-27 00:00:00" \
        --mode synchronize

Options:

    --hours <N>          Number of past hours to ingest (default: 6 if nothing else given)
    --history_days <N>   Number of days of history (converted to hours)
    --start_date <str>   Custom UTC start datetime  "YYYY-MM-DD HH:MM:SS"
    --end_date   <str>   Custom UTC end   datetime  "YYYY-MM-DD HH:MM:SS"
    --mode <mode>        Feature generation mode (default: "synchronize")
                         Modes:
                            synchronize  → synced features for ChronoBridge + NFC
                            bridge       → bridge-only features
    --device cpu|cuda    Torch device override (default: cpu)

Notes:
• If start_date & end_date are provided, they override hours/history_days.
• Internally uses:
    - data_ingest_service:
        * latest mode for hours/history_days
        * custom mode for start_date/end_date
    - features_service:
        * --latest_hours for hours/history_days
        * --start_time/--end_time for start_date/end_date

----------------------------------------------------------------------------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 19
Version: 1.3.0  
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
client = MongoClient(
    f"mongodb://{NOVO_MONGO_USER}:{NOVO_MONGO_PASS}"
    f"@{NOVO_MONGO_HOST}:{NOVO_MONGO_PORT}/?authSource={NOVO_MONGO_AUTH_DB}"
)

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


def run_data_ingest(hours: int = None, start_date: str = None, end_date: str = None):
    """
    Launch the data ingestion subprocess to fetch data.

    - If start_date & end_date are provided → use data_ingest_service --mode custom
    - Else → use data_ingest_service --mode latest with --hours
    """
    target_module = _resolve_module_path("data_ingest_service")

    # Custom time range
    if start_date is not None and end_date is not None:
        logging.info(
            f"Running data_ingest_service in CUSTOM mode for range "
            f"{start_date} → {end_date} (UTC)"
        )
        cmd = [
            sys.executable,
            "-m",
            target_module,
            "--mode",
            "custom",
            "--start_time",
            start_date,
            "--end_time",
            end_date,
        ]
    else:
        # Latest N hours (existing behavior)
        if hours is None:
            raise ValueError("run_data_ingest: hours must be provided when no custom dates are given.")
        logging.info(f"Running data_ingest_service to fetch last {hours} hour(s) of data")
        cmd = [
            sys.executable,
            "-m",
            target_module,
            "--mode",
            "latest",
            "--hours",
            str(hours),
        ]

    subprocess.run(cmd, check=True)


def run_feature_service(
    hours: int = None,
    mode: str = "synchronize",
    start_date: str = None,
    end_date: str = None,
):
    """
    Launch the feature service subprocess in bridge/synchronize/etc. mode.

    - If start_date & end_date are provided → pass as --start_time/--end_time
    - Else → pass --latest_hours with 'hours'

    'mode' here is the FEATURES mode (synchronize, bridge, train, ...),
    not the time-range type (which is implied by presence of start_date/end_date).
    """
    if mode not in ("bridge", "synchronize", "train", "finetune", "inference", "backtesting", "future_testing"):
        logging.error(f"Mode '{mode}' not recognized for features_service.")
        return

    target_module = _resolve_module_path("features_service")

    # Custom time range
    if start_date is not None and end_date is not None:
        logging.info(
            f"Running features_service in {mode.upper()} mode "
            f"for custom range {start_date} → {end_date} (UTC)"
        )
        cmd = [
            sys.executable,
            "-m",
            target_module,
            "--mode",
            mode,
            "--start_time",
            start_date,
            "--end_time",
            end_date,
        ]
    else:
        # Latest N hours 
        if hours is None:
            raise ValueError("run_feature_service: hours must be provided when no custom dates are given.")
        logging.info(
            f"Running features_service in {mode.upper()} mode for last {hours} hour(s)"
        )
        cmd = [
            sys.executable,
            "-m",
            target_module,
            "--mode",
            mode,
            "--latest_hours",
            str(hours),
        ]

    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------------------
def main():
    start_service_time = time.time()
    torch.cuda.empty_cache()

    import argparse
    parser = argparse.ArgumentParser()

    # Time-range options
    parser.add_argument("--hours", type=int, default=None,
                        help="Number of hours to fetch data for (latest range).")
    parser.add_argument("--history_days", type=int, default=None,
                        help="Number of days of history (converted to hours).")

    # custom time range 
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help='UTC start datetime, format: "YYYY-MM-DD HH:MM:SS"',
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help='UTC end datetime, format: "YYYY-MM-DD HH:MM:SS"',
    )

    # Features mode 
    parser.add_argument("--mode", type=str, default="synchronize",
                        help="Feature-service mode: synchronize | bridge | train | finetune | inference | backtesting | future_testing")
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    # ---------- Decide between custom range vs latest-hours ----------
    use_custom_range = args.start_date is not None and args.end_date is not None

    if use_custom_range:
        logging.info(
            f"Using CUSTOM time range: start_date={args.start_date}, "
            f"end_date={args.end_date} (UTC)"
        )
        # No need to compute hours/history_days; they are ignored when custom is used
        hours_for_latest = None
    else:
        # ---------- Convert history_days to hours if provided ----------
        if args.history_days is not None:
            args.hours = args.history_days * 24
            logging.info(
                f"Converted history_days={args.history_days} to hours={args.hours}."
            )
        elif args.hours is None:
            # fallback default (if neither provided)
            args.hours = 6
            logging.info("No history_days or hours specified, using default 6 hours.")
        hours_for_latest = args.hours

    # ---------- Fetch data and Feature extraction ----------
    if use_custom_range:
        # custom range path
        run_data_ingest(
            hours=None,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        run_feature_service(
            hours=None,
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        # latest-hours path 
        run_data_ingest(hours=hours_for_latest)
        run_feature_service(hours=hours_for_latest, mode=args.mode)

    #---------- Get FusedEmbedding from NeuralFusionCore --------
    NFC_infer = NeuralFusionCore_infer()
    model_checkpoint_path = "apps/NeuralFusionCore/data/outputs/model_weights.pt"
    NFC_infer.FusedEmbedding(
        model_checkpoint=model_checkpoint_path,
        mongo_collection=mongo_col,
        device=args.device
    )
    logging.info("Fused embedding prediction cycle complete.")

    print(f"Time elapsed: {time.time() - start_service_time:.2f} seconds")


if __name__ == "__main__":
    main()
