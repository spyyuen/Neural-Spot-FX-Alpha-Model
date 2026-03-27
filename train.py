import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import FXModel

def train_model(X, y, epochs=20):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    model = FXModel(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model, scaler
