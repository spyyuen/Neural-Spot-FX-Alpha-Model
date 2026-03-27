from data_loader import load_fx_data
from features import create_features
from train import train_model
from backtest import backtest

df = load_fx_data("data/eurusd.csv")
X, y = create_features(df)

model, scaler = train_model(X, y)
backtest(model, scaler, X, y)
