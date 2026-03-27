import pandas as pd

def create_features(df):
    df = df.copy()

    # Ensure mid + return exist
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2

    if "return" not in df.columns:
        df["return"] = df["mid"].pct_change()

    df = df.dropna()

    X = pd.DataFrame(index=df.index)

    # Lagged returns
    for lag in range(1, 6):
        X[f"return_lag_{lag}"] = df["return"].shift(lag)

    # Rolling volatility
    X["vol_5"] = df["return"].rolling(5).std()
    X["vol_10"] = df["return"].rolling(10).std()

    # Momentum (using mid price)
    X["momentum_5"] = df["mid"] / df["mid"].shift(5) - 1
    X["momentum_10"] = df["mid"] / df["mid"].shift(10) - 1

    # Moving average signal
    ma5 = df["mid"].rolling(5).mean()
    ma20 = df["mid"].rolling(20).mean()
    X["ma_signal"] = (ma5 - ma20) / df["mid"]

    # Target: next return
    y = df["return"].shift(-1)

    # Align and clean
    X = X.dropna()
    y = y.loc[X.index]

    return X, y
