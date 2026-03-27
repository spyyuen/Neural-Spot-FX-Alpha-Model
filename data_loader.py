import pandas as pd

def load_fx_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    df['return'] = df['Close'].pct_change()
    return df.dropna()
