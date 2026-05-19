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
    # Equity features
    # ----------------------------

    df["spx_ret"] = df["spx"].pct_change()
    df["eustoxx_ret"] = df["eustoxx"].pct_change()

    # Relative performance (KEY signal)
    X["equity_relative"] = df["eustoxx_ret"] - df["spx_ret"]

    # Risk regime
    X["spx_momentum_20"] = df["spx"] / df["spx"].shift(20) - 1

    # Volatility spillover
    X["spx_vol_20"] = df["spx_ret"].rolling(20).std()

    # Correlation regime
    X["fx_spx_corr_50"] = df["return"].rolling(50).corr(df["spx_ret"])

    # Divergence signal
    X["equity_fx_divergence"] = df["spx_ret"] - df["return"]

    # ----------------------------
    # Equities often close after FX moves, so lag them.
    # ----------------------------
    equity_cols = [
        "equity_relative",
        "fx_spx_corr_50",
        "spx_momentum_20",
        "spx_vol_20",
        "equity_fx_divergence"
    ]
    # Prevent lookahead bias
    for col in equity_cols:
        X[col] = X[col].shift(1)


    # ----------------------------
    # Add microstructure features
    # ----------------------------
    X["spread"] = df["ask"] - df["bid"]

    # spread regime
    X["spread_zscore"] = (
            (X["spread"] - X["spread"].rolling(50).mean())
            / X["spread"].rolling(50).std()
    )

    # tick direction
    X["tick_direction"] = np.sign(df["ret_1"])

    # signed flow proxy
    X["signed_flow"] = (
            np.sign(df["ret_1"]) * abs(df["ret_1"])
    )

    # order-flow persistence
    X["flow_persistence"] = (
        X["tick_direction"].rolling(20).mean()
    )

    # ----------------------------
    # Add time of day features
    # ----------------------------
    df["hour"] = df["timestamp"].dt.hour

    X["london_session"] = (

            (df["hour"] >= 7) &

            (df["hour"] <= 16)

    ).astype(int)

    X["ny_session"] = (

            (df["hour"] >= 13) &

            (df["hour"] <= 21)

    ).astype(int)

    X["session_overlap"] = (

            X["london_session"] &

            X["ny_session"]

    ).astype(int)

    # ----------------------------
    # Target
    # ----------------------------
    print('defining target')
    y = df["return"].shift(-1)

    # ----------------------------
    # Clean up
    # ----------------------------
    X = X.replace([np.inf, -np.inf], np.nan)

    dataset = pd.concat([X, y.rename("target")], axis=1)

    dataset = dataset.dropna().reset_index(drop=True)

    X = dataset.drop(columns="target")
    y = dataset["target"]

    # ----------------------------
    # Sanity check: If equities show zero signal, something is misaligned.
    # ----------------------------
    print(X[["equity_relative", "spx_momentum_20"]].describe())

    print("Correlation with target:")
    print(X.corrwith(y).sort_values().tail(10))

    return X, y

