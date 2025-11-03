#!/usr/bin/env python3
"""
chronobridge_api_service.py
--------------
Description: FastAPI service to serve chronobridge predictions from MongoDB.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 19
Version: 1.1.0
"""
from fastapi import FastAPI, Query
from typing import Optional, List
from pymongo import MongoClient
from datetime import datetime
import uvicorn
import json, os
from dotenv import load_dotenv
# ---------------- MongoDB setup ----------------
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
COLLECTION_NAME = "chrono_bridge"
mongo_col = db[COLLECTION_NAME]

# ---------------- FastAPI app ----------------
app = FastAPI(title="ChronoBridge API", version="1.0")

# Helper function to convert MongoDB documents to JSON-serializable dicts
def serialize_doc(doc):
    doc['_id'] = str(doc['_id'])
    return doc

# ---------------- GET endpoints ----------------
@app.get("/fused_embeddings")
def get_fused_embeddings(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    stocks: Optional[List[str]] = Query(None, description="List of stock symbols")
):
    """
    Get fused embeddings from MongoDB.
    You can filter by date range, stock symbols, or both.
    """

    query = {}

    # Filter by date range
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        query["date"] = date_filter

    # Filter by stock symbols
    if stocks:
        query["symbol"] = {"$in": stocks}

    results = list(mongo_col.find(query))
    results = [serialize_doc(r) for r in results]

    return {"count": len(results), "data": results}

# Optional: health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------------- Run app ----------------
if __name__ == "__main__":
    uvicorn.run("apps.ChronoBridge.scripts.chronobridge_api_service:app", host="0.0.0.0", port=8001, reload=True)
