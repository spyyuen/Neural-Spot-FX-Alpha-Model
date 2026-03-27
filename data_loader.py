import pandas as pd
import requests

def load_fx_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    df['return'] = df['Close'].pct_change()
    return df.dropna()

def load_data_via_api():

    # API endpoint (local)
    BASE_URL = "http://127.0.0.1:8000/ticks"

    params = {
        "symbol": "EURUSD",
        "start": "2025-01-01T00:00:00",
        "end": "2025-03-01T00:01:00"
    }

    response = requests.get(BASE_URL, params=params)

    # Check request success
    response.raise_for_status()

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Set index
    df.set_index("timestamp", inplace=True)

    print("Fetched data of length len(df)")
    print(df.head())
    return df

if __name__ == "__main__":
    load_data_via_api()
