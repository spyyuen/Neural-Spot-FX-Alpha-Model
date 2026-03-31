import argparse
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# ----------------------------
# Config
# ----------------------------
BASE_URL = "http://127.0.0.1:8000/ticks"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syms", required=False, type=str, help="Comma-delimited list of syms, e.g. EURUSD,GBPUSD", default="EURUSD")
    parser.add_argument("--start", required=False, type=str, help="Start datetime (YYYY-MM-DD)", default="2026-02-01")
    parser.add_argument("--end", required=False, type=str, help="End datetime (YYYY-MM-DD)", default='2026-02-05')
    parser.add_argument("--backfill", action="store_true", help="Enable backfilling of data")
    args = parser.parse_args()
    return vars(args)


# ----------------------------
# Cache logic
# ----------------------------
def get_cache_path(symbol, start, end):
    return os.path.join(
        DATA_DIR,
        f"{symbol}_{start.date()}_{end.date()}.parquet"
    )


def load_or_fetch(symbol, start, end, backfill=False):
    cache_file = get_cache_path(symbol, start, end)
    if not backfill:
        if os.path.exists(cache_file):
            print(f"[CACHE HIT] {cache_file}")
            print('cache_file ', pd.read_parquet(cache_file))
            return pd.read_parquet(cache_file)

    print(f"[FETCH] {symbol} {start} -> {end}")
    df = fetch_ticks(symbol, start, end)

    df.to_parquet(cache_file)
    return df


# ----------------------------
# Main data loader
# ----------------------------
def load_data_via_api(symbol, start, end, backfill):
    dfs = []

    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)
    for chunk_start, chunk_end in generate_weekly_ranges(start, end):
        df_chunk = load_or_fetch(symbol, chunk_start, chunk_end, backfill)
        dfs.append(df_chunk)

    df = pd.concat(dfs).sort_index()
    return df


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--syms",
        nargs="+",
        required=True,
        help="List of syms, e.g. EURUSD GBPUSD"
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End datetime (YYYY-MM-DD)"
    )

    return parser.parse_args()


# ----------------------------
# API fetch (one chunk)
# ----------------------------
def fetch_ticks(symbol, start, end):
    params = {
        "symbol": symbol,
        "start": start.isoformat(),
        "end": end.isoformat()
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    df = pd.DataFrame(response.json())
    #df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    return df


# ----------------------------
# Weekly chunk generator
# ----------------------------
def generate_weekly_ranges(start, end):
    current = start
    while current < end:
        next_week = min(current + timedelta(days=7), end)
        yield current, next_week
        current = next_week


# ----------------------------
# Main
# ----------------------------
def run(syms, start, end, backfill):

    for symbol in syms.split(","):
        print(f"\nProcessing {symbol}...")

        df = load_data_via_api(symbol, start, end, backfill)

        print(df.head())
        print(df.tail())


if __name__ == "__main__":
    kwargs = parse_args()
    run(**kwargs)

