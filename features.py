import pandas as pd

def create_features(df):
    X = pd.DataFrame(index=df.index)

    # Lagged returns
    for lag in range(1, 6):
        X[f'return_lag_{lag}'] = df['return'].shift(lag)

    # Rolling volatility
    X['vol_5'] = df['return'].rolling(5).std()
    X['vol_10'] = df['return'].rolling(10).std()

    # Momentum
    X['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    X['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1

    # Moving average signal
    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    X['ma_signal'] = (ma5 - ma20) / df['Close']

    y = df['return'].shift(-1)

    return X.dropna(), y.loc[X.index]
