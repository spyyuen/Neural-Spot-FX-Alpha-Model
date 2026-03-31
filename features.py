import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()

    # ----------------------------
    # Core price construction
    # ----------------------------
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2

    if "return" not in df.columns:
        df["return"] = df["mid"].pct_change()

    # Spread (important in FX)
    df["spread"] = df["ask"] - df["bid"]

    df = df.dropna()

    X = pd.DataFrame(index=df.index)

    # ----------------------------
    # 1. Lagged returns
    # ----------------------------
    for lag in range(1, 6):
        X[f"return_lag_{lag}"] = df["return"].shift(lag)

    # ----------------------------
    # 2. Rolling features
    # ----------------------------
    X["vol_5"] = df["return"].rolling(5).std()
    X["vol_10"] = df["return"].rolling(10).std()

    X["momentum_5"] = df["mid"] / df["mid"].shift(5) - 1
    X["momentum_10"] = df["mid"] / df["mid"].shift(10) - 1

    ma5 = df["mid"].rolling(5).mean()
    ma20 = df["mid"].rolling(20).mean()
    X["ma_signal"] = (ma5 - ma20) / df["mid"]

    # ----------------------------
    # 3. Expanding features (NEW)
    # ----------------------------
    print('calculating expanding features')
    # Expanding mean return (drift)
    X["expanding_mean_return"] = df["return"].expanding().mean()

    # Expanding volatility
    X["expanding_vol"] = df["return"].expanding().std()

    # Z-score of return relative to expanding stats
    X["return_zscore"] = (
        (df["return"] - X["expanding_mean_return"]) /
        X["expanding_vol"]
    )

    # Expanding Sharpe-like ratio
    X["expanding_sharpe"] = (
        X["expanding_mean_return"] / X["expanding_vol"]
    )

    # ----------------------------
    # 4. Microstructure features (FX specific)
    # ----------------------------

    # Spread dynamics
    X["spread"] = df["spread"]
    X["spread_change"] = df["spread"].diff()


    # Price acceleration
    X["return_acceleration"] = df["return"].diff()

    # Signed returns (trend persistence)
    X["signed_return"] = np.sign(df["return"]) * df["return"].abs()

    # ----------------------------
    # 5. Normalized features
    # ----------------------------

    # Rolling z-score (stationarization)
    rolling_mean = df["return"].rolling(20).mean()
    rolling_std = df["return"].rolling(20).std()

    X["rolling_zscore"] = (df["return"] - rolling_mean) / rolling_std

    # ----------------------------
    # Target
    # ----------------------------
    print('defining target')
    y = df["return"].shift(-1)

    # ----------------------------
    # Clean up
    # ----------------------------
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]

    return X, y
