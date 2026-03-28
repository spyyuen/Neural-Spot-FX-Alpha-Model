from load_data import load_fx_data, load_data_via_api
from features import create_features
from train import train_model
from backtest import backtest


def main():
    df = load_data_via_api()
    print(df.head())
    
    #df = load_fx_data("data/eurusd.csv")
    X, y = create_features(df)

    model, scaler = train_model(X, y)
    backtest(model, scaler, X, y)

if __name__ == "__main__":
    main()

