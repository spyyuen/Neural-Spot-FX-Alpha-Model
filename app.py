from fastapi import FastAPI, Query
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

app = FastAPI(title="FX Tick Data API")

# -----------------------------
# Synthetic Tick Generator
# -----------------------------
def generate_ticks(start: datetime, end: datetime, symbol: str):
    np.random.seed(42)

    timestamps = []
    bids = []
    asks = []

    current_time = start
    price = 1.0800  # starting EURUSD price

    while current_time < end:
        # simulate tick arrival every 50-200 ms
        delta_ms = np.random.randint(50, 200)
        current_time += timedelta(milliseconds=delta_ms)

        # random walk
        price += np.random.normal(0, 0.00005)

        bid = price
        ask = price + 0.0001  # fixed spread assumption

        timestamps.append(current_time.isoformat())
        bids.append(round(bid, 5))
        asks.append(round(ask, 5))

    return pd.DataFrame({
        "timestamp": timestamps,
        "bid": bids,
        "ask": asks
    })


# -----------------------------
# Placeholder for Real Data API
# -----------------------------
def fetch_real_ticks(symbol: str, start: datetime, end: datetime):
    """
    Replace this with a real provider:
    - Dukascopy
    - FXCM
    - Refinitiv
    - etc.

    For now, we fallback to synthetic ticks.
    """
    return generate_ticks(start, end, symbol)


# -----------------------------
# API Endpoint
# -----------------------------
@app.get("/ticks")
def get_ticks(
    symbol: str = Query("EURUSD"),
    start: str = Query(...),
    end: str = Query(...)
):
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    df = fetch_real_ticks(symbol, start_dt, end_dt)

    return df.to_dict(orient="records")


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def root():
    return {"status": "FX Tick API running"}
