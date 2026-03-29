import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import os


def plot_results(df, X, y, model, scaler, output_path="reports/report.html"):
    os.makedirs("reports", exist_ok=True)

    # ----------------------------
    # Prepare data
    # ----------------------------
    X_scaled = scaler.transform(X)

    import torch
    model.eval()
    preds = model(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().flatten()

    # Align
    df_plot = pd.DataFrame(index=X.index)
    df_plot["return"] = y
    df_plot["prediction"] = preds

    # Strategy
    df_plot["signal"] = np.sign(df_plot["prediction"])
    df_plot["strategy_return"] = df_plot["signal"] * df_plot["return"]
    df_plot["cumulative"] = (1 + df_plot["strategy_return"]).cumprod()

    # ----------------------------
    # Create subplots
    # ----------------------------
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Mid Price",
            "Returns vs Predictions",
            "Feature Example (return_lag_1)",
            "Strategy Performance"
        ]
    )

    # ----------------------------
    # 1. Price
    # ----------------------------
    if "mid" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["mid"], name="Mid Price"),
            row=1, col=1
        )

    # ----------------------------
    # 2. Returns vs Predictions
    # ----------------------------
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["return"], name="Return"),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["prediction"], name="Prediction"),
        row=2, col=1
    )

    # ----------------------------
    # 3. Example Feature
    # ----------------------------
    feature_name = X.columns[0]
    fig.add_trace(
        go.Scatter(x=X.index, y=X[feature_name], name=feature_name),
        row=3, col=1
    )

    # ----------------------------
    # 4. Strategy Performance
    # ----------------------------
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["cumulative"], name="Cumulative Return"),
        row=4, col=1
    )

    # ----------------------------
    # Layout
    # ----------------------------
    fig.update_layout(
        height=1200,
        title="FX Alpha Model Report",
        showlegend=True
    )

    # ----------------------------
    # Save HTML
    # ----------------------------
    fig.write_html(output_path)

    print(f"Saved report to {output_path}")