from load_data import *
from features import create_features
from train import train_model
from backtest import backtest
import argparse
from plot_results import *

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syms", required=False, type=str, help="Comma-delimited list of syms, e.g. EURUSD,GBPUSD", default="EURUSD")
    parser.add_argument("--start", required=False, type=str, help="Start datetime (YYYY-MM-DD)", default="2026-02-01")
    parser.add_argument("--end", required=False, type=str, help="End datetime (YYYY-MM-DD)", default='2026-03-05')
    parser.add_argument("--backfill", action="store_true", help="Enable backfilling of data")
    args = parser.parse_args()

    return vars(args)

def run(syms, start, end, backfill):
    df = load_data_via_api(syms, start, end, backfill)
    print(df.head())
    
    #df = load_fx_data("data/eurusd.csv")
    print('creating features ')
    X, y = create_features(df)
    print('X', X)

    print('training model')
    model, scaler = train_model(X, y)
    print('model ', model)
    backtest(model, scaler, X, y)

    plot_results(df, X, y, model, scaler)

if __name__ == "__main__":
    kwargs = parse_args()
    run(**kwargs)

