"""
data_ingest_service.py
Stateless fetcher: fetch OHLCV from ClickHouse and news from Mongo for the given interval,
and push results (per-symbol ohlcv DataFrame pickles and news DataFrame) to Redis.

Modes:
  - latest:       fetch last hours data from DBs and save to redis
  - historical:   fetch historical days data from DBs and save  to redis
  - custom:       fetch data from DBs in custom time period

Usage examples:
  # historical: fetch last 30 days and save to disk (also push latest window to Redis)
  python data_ingest_service.py --mode historical --days 30 --save_dir data/ohlcv_history
  python data_ingest_service.py --mode custom  --start_time "2025-02-26 00:00:00"  --end_time "2025-02-27 00:00:00"

  # one-shot latest 4h (use scheduler to run every 4h)
  python data_ingest_service.py --mode latest --hours 4
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Sep 30
Version: 1.0.2
"""
import argparse, logging, os, pickle, sys, zlib
from datetime import datetime, timedelta, timezone
import pandas as pd
from clickhouse_driver import Client as CHClient
from pymongo import MongoClient
from redis import Redis
import time
from dotenv import load_dotenv
load_dotenv()

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
    from apps.ChronoBridge.config import ClickhouseCfg, MongoCfg, RedisCfg, Paths, MarketCfg, FeatureCfg
    from apps.ChronoBridge.lib import market as M
except ImportError:
    from ..config import ClickhouseCfg, MongoCfg, RedisCfg, Paths, MarketCfg, FeatureCfg
    from ..lib import market as M

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

CH = ClickhouseCfg(); MO = MongoCfg(); RE = RedisCfg(); P = Paths(); MC = MarketCfg(); FC = FeatureCfg()

ch_client = CHClient(host=CH.CH_HOST, port=CH.CH_PORT, user=CH.CH_USER,
                     password=CH.CH_PASS, database=CH.CH_DB)

mongo_kwargs = {
    "host": MO.MONGO_HOST,
    "port": MO.MONGO_PORT
}
if MO.MONGO_USER and MO.MONGO_PASS:
    mongo_kwargs.update({
        "username": MO.MONGO_USER,
        "password": MO.MONGO_PASS,
        "authSource": MO.MONGO_AUTHSOURCE or MO.MONGO_DB
    })
mongo_client = MongoClient(**mongo_kwargs)

# Safer Redis client: fail fast instead of "hanging" on giant payloads
redis_client = Redis(
    host=RE.host,
    port=RE.port,
    db=RE.db,
    socket_timeout=30,           # read/write timeout
    socket_connect_timeout=5,    # connect timeout
)

# ------------------------
# Helpers: compression for news payloads
# ------------------------
def _dumps_df_compressed(df: pd.DataFrame) -> bytes:
    """Pickle + compress a DataFrame for efficient Redis storage."""
    raw = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, level=6)

def _loads_df_compressed(b: bytes) -> pd.DataFrame:
    return pickle.loads(zlib.decompress(b))

# ------------------------
# ClickHouse OHLCV
# ------------------------
def fetch_ohlcv_range(symbol, start_utc, end_utc):
    start_str = start_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_str = end_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    q = f"""
        SELECT *
        FROM {CH.CH_TABLE}
        WHERE symbol = '{symbol}'
          AND candle_time >= '{start_str}'
          AND candle_time <= '{end_str}'
        ORDER BY candle_time ASC
    """
    data = ch_client.execute(q)
    if not data:
        return pd.DataFrame()
    cols = [c[0] for c in ch_client.execute(f"DESCRIBE TABLE {CH.CH_TABLE}")]
    df = pd.DataFrame(data, columns=cols)
    df['dateTime'] = pd.to_datetime(df['candle_time'], utc=True)
    df3m = M.resample_to_3m(df, FC.agg_cols)
    return df3m

# ------------------------
# Mongo News (index-friendly fetch)
# ------------------------
def ensure_mongo_indexes():
    """Idempotently ensure we have an index on releasedAt to speed up range scans."""
    try:
        col = mongo_client[MO.MONGO_DB][MO.MONGO_COLLECTION]
        col.create_index([("releasedAt", 1)], background=True)
        # If you often filter by symbol too, uncomment:
        # col.create_index([("releasedAt", 1), ("symbol", 1)], background=True)
    except Exception as e:
        logging.warning("Could not create Mongo index on releasedAt: %s", e)

def fetch_news_range(start_utc, end_utc):
    """
    Fetch news documents from Mongo in [start_utc, end_utc],
    logging progress as the cursor is consumed.
    """
    col = mongo_client[MO.MONGO_DB][MO.MONGO_COLLECTION]
    projection = {
        "_id": 0,
        "releasedAt": 1,
        "assets": 1,
        "content": 1,
        "news_count": 1
    }

    logging.info(
        f"Starting Mongo fetch for news from {start_utc} to {end_utc} ..."
    )

    query = {"releasedAt": {"$gte": start_utc, "$lte": end_utc}}
    cursor = col.find(query, projection=projection).sort("releasedAt", 1).batch_size(10_000)

    # --- iterative read with progress ---
    t0 = time.time()
    docs = []
    count = 0
    log_every = 10_000  # adjust depending on typical volume

    for doc in cursor:
        docs.append(doc)
        count += 1
        if count % log_every == 0:
            elapsed = time.time() - t0
            logging.info(f"Fetched {count:,} news docs so far... ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    logging.info(f"Mongo fetch complete. Total docs: {count:,} | Time: {elapsed:.1f}s")

    # --- convert to DataFrame ---
    if not docs:
        logging.info("No news found in the given time range.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["releasedAt"] = pd.to_datetime(df["releasedAt"], utc=True)
    logging.info(
        f"Converted to DataFrame with shape {df.shape} | Columns: {list(df.columns)}"
    )
    return df

# ------------------------
# Redis pushers
# ------------------------
def push_ohlcv_to_redis(sym, df):
    if df is None or df.empty:
        return
    key = f"ohlcv:{sym}"
    redis_client.set(key, pickle.dumps(df))
    logging.info("Pushed to redis: %s (%d rows)", key, len(df))

def push_news_chunked(df: pd.DataFrame, freq: str = 'D'):
    """
    Push news to Redis as multiple smaller keys (e.g., one per day) using a pipeline.
    Keys: news:YYYY-MM-DD  (if freq='D')
    Payload: pickled+compressed pandas DataFrame for that chunk.
    """
    if df is None or df.empty:
        return

    # Ensure tz-aware, normalized boundaries for grouping
    dfx = df.copy()
    if dfx['releasedAt'].dt.tz is None:
        dfx['releasedAt'] = dfx['releasedAt'].dt.tz_localize('UTC')
    else:
        dfx['releasedAt'] = dfx['releasedAt'].dt.tz_convert('UTC')

    # Group by day (or week if freq='W')
    dfx['__bucket__'] = dfx['releasedAt'].dt.floor(freq)

    pipe = redis_client.pipeline(transaction=False)
    total_rows = 0
    chunk_count = 0

    for bucket, g in dfx.groupby('__bucket__', sort=True):
        key = f"news:{bucket.strftime('%Y-%m-%d')}" if freq == 'D' else f"news:{bucket.strftime('%Y-%m-%d_%s')}"
        payload_df = g.drop(columns='__bucket__')
        payload = _dumps_df_compressed(payload_df)
        pipe.set(key, payload)
        total_rows += len(payload_df)
        chunk_count += 1
        logging.info("Prepared news chunk %s rows=%d bytes=%d", key, len(payload_df), len(payload))

    pipe.execute()
    logging.info("Pushed news to redis in %s chunks: %d chunks, %d rows total.", freq, chunk_count, total_rows)

# ------------------------
# Main
# ------------------------
def main():
    start_service_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="latest", choices=["historical", "latest", "custom"])
    parser.add_argument("--days", type=int, default=30, help="for historical mode")
    parser.add_argument("--hours", type=int, default=4, help="for latest mode")
    parser.add_argument("--start_time", type=str, help='UTC start time format: "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("--end_time", type=str, help='UTC end time format: "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("--save_dir", type=str, default=None, help="optional dir to save historical files")
    parser.add_argument("--news_freq", type=str, default="D", choices=["D", "W"], help="bucket size for news Redis keys")
    args = parser.parse_args()

    ensure_mongo_indexes()

    now = datetime.now(timezone.utc)

    if args.mode == "historical":
        start = now - timedelta(days=args.days)
        logging.info(f"Fetching historical data from {start} to {now}")

        symbols_ch = [f"{s}" for s in MC.symbols_usdt]
        for sym in symbols_ch:
            df_all = fetch_ohlcv_range(sym, start, now)
            if not df_all.empty:
                push_ohlcv_to_redis(sym, df_all)

        # fetch & push news (chunked)
        df_news = fetch_news_range(start, now)
        if not df_news.empty:
            push_news_chunked(df_news, freq=args.news_freq)

        logging.info("Historical ingest complete.")
        end_service_time = time.time()
        print(f"Time elapsed for data ingestion service: {end_service_time - start_service_time:.2f} seconds")
        return

    if args.mode == "latest":
        start = now - timedelta(hours=args.hours)
        symbols_ch = [f"{s}" for s in MC.symbols_usdt]
        for sym in symbols_ch:
            df = fetch_ohlcv_range(sym, start, now)
            if not df.empty:
                push_ohlcv_to_redis(sym, df)

        # latest windows are usually small; still use chunked for consistency
        df_news = fetch_news_range(start, now)
        if not df_news.empty:
            push_news_chunked(df_news, freq=args.news_freq)

        logging.info("Latest ingest complete.")
        end_service_time = time.time()
        print(f"Time elapsed for data ingestion service: {end_service_time - start_service_time:.2f} seconds")
        return

    if args.mode == "custom":
        start = datetime.fromisoformat(args.start_time).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end_time).replace(tzinfo=timezone.utc)

        logging.info(f"Fetching custom data range: {start} to {end}")

        symbols_ch = [f"{s}" for s in MC.symbols_usdt]
        for sym in symbols_ch:
            df = fetch_ohlcv_range(sym, start, end)
            if not df.empty:
                push_ohlcv_to_redis(sym, df)

        df_news = fetch_news_range(start, end)
        if not df_news.empty:
            push_news_chunked(df_news, freq=args.news_freq)

        logging.info("Custom ingest complete.")
        end_service_time = time.time()
        print(f"Time elapsed for data ingestion service: {end_service_time - start_service_time:.2f} seconds")
        return

if __name__ == "__main__":
    main()
