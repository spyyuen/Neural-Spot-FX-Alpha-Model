# Neural-Spot-FX-Alpha-Model

This project implements a neural network–based alpha model for predicting short-term returns in spot FX (EUR/USD).

## Strategy

I model:

    r_{t+1} = f(X_t)

Where:
- X_t = engineered features from past price data
- r_{t+1} = next-period return

The model is a feedforward neural network trained to predict returns.

## Features

- Lagged returns
- Rolling volatility
- Momentum indicators
- Moving averages

## Pipeline

1. Load FX data
2. Engineer features
3. Train neural network
4. Generate signals
5. Backtest strategy

## Results

Sharpe ratio, cumulative returns, and drawdowns are evaluated.

## Future Improvements

- LSTM / sequence models
- Cross-asset features (rates, equities)
- Transaction cost modeling


## To run app and get tick data:
1.  $uvicorn api.app:app --reload
2.  In browser, go to http://127.0.0.1:8000
3.  You should see {"status":"FX Tick API running"}
4.  $curl "http://127.0.0.1:8000/ticks?symbol=EURUSD&start=2025-01-01T00:00:00&end=2025-01-01T00:00:05"

