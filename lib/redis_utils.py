# lib/redis_utils.py
"""
Description: redis helpers.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Sep 30
Version: 1.0.1 
"""
import pickle
from redis import Redis
from apps.NeuralFusionCore.config import RedisCfg, Paths
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
import zlib

RE = RedisCfg(); P = Paths()

# ---------------------------
# Redis client (match ingest)
# ---------------------------
redis_client = Redis(
    host=RE.host, port=RE.port, db=RE.db,
    socket_timeout=30, socket_connect_timeout=5
)

# ---------------------------
# Serialization helpers
# ---------------------------
def _loads_df(b: bytes) -> pd.DataFrame:
    """Plain pickle (used for OHLCV)."""
    return pickle.loads(b)

def _loads_df_compressed(b: bytes) -> pd.DataFrame:
    """zlib + pickle (used for chunked news)."""
    return pickle.loads(zlib.decompress(b))

# ---------------------------
# Data loaders from Redis
# ---------------------------
def load_ohlcv_from_redis(symbols, start_time=None, end_time=None):
    """
    Load per-symbol OHLCV DataFrames from Redis keys: ohlcv:{symbol}
    Then time-filter to [start_time, end_time] if provided.
    """
    out = {}
    for sym in symbols:
        key = f"ohlcv:{sym}"
        raw = redis_client.get(key)
        if not raw:
            continue
        try:
            df = _loads_df(raw)
        except Exception as e:
            logging.warning("Failed to unpickle OHLCV for %s: %s", sym, e)
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # Ensure datetime column is tz-aware
        if 'dateTime' in df.columns:
            df['dateTime'] = pd.to_datetime(df['dateTime'], utc=True)
            if start_time is not None:
                df = df[df['dateTime'] >= pd.to_datetime(start_time, utc=True)]
            if end_time is not None:
                df = df[df['dateTime'] <= pd.to_datetime(end_time, utc=True)]
        out[sym] = df
    return out

def _iter_dates_inclusive(start_utc: datetime, end_utc: datetime):
    d = start_utc.date()
    while d <= end_utc.date():
        yield d
        d += timedelta(days=1)

def load_news_range_from_redis(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Load chunked news from Redis keys: news:YYYY-MM-DD (zlib-compressed pickled DataFrames),
    then concat and time-filter to [start_time, end_time].
    """
    if start_time is None or end_time is None:
        logging.info("News load skipped (no time window).")
        return pd.DataFrame()

    start_utc = pd.to_datetime(start_time, utc=True).to_pydatetime()
    end_utc = pd.to_datetime(end_time, utc=True).to_pydatetime()

    keys = [f"news:{d.strftime('%Y-%m-%d')}" for d in _iter_dates_inclusive(start_utc, end_utc)]
    # Batch GET with pipeline
    pipe = redis_client.pipeline()
    for k in keys: pipe.get(k)
    raw_list = pipe.execute()

    dfs = []
    for k, raw in zip(keys, raw_list):
        if not raw: 
            continue
        try:
            df = _loads_df_compressed(raw)
        except Exception as e:
            logging.warning("Failed to load news chunk %s: %s", k, e)
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # normalize timestamp column
        if 'releasedAt' in df.columns:
            df['releasedAt'] = pd.to_datetime(df['releasedAt'], utc=True)
            mask = (df['releasedAt'] >= start_utc) & (df['releasedAt'] <= end_utc)
            df = df.loc[mask]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def empty_current_database():
    redis_client.flushdb() 
    return "ok"

