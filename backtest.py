import numpy as np
import matplotlib.pyplot as plt
import torch

def backtest(model, scaler, X, y):
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    preds = model(X_tensor).detach().numpy().flatten()

    # Signal: sign of prediction
    signals = np.sign(preds)

    strategy_returns = signals * y.values

    cumulative = (1 + strategy_returns).cumprod()

    sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)

    print(f"Sharpe Ratio: {sharpe:.2f}")

    plt.plot(cumulative)
    plt.title("Strategy Performance")
    plt.show()
