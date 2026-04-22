#Garfico de velas Plotly

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def plot_crypto_candlestick(
    df_plot: pd.DataFrame,
    title: str = "Candlestick Chart + Prediction",
    prediction: float = None,
    show_rsi: bool = True,
) -> go.Figure:

    df_plot = df_plot.tail(120).copy()

    rows        = 2 if show_rsi else 1
    row_heights = [0.75, 0.25] if show_rsi else [1.0]
    subtitles   = ("Precio y Medias Móviles", "RSI (14)") if show_rsi else None

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=row_heights,
        subplot_titles=subtitles,
    )

    # Velas
    fig.add_trace(
        go.Candlestick(
            x=df_plot["timestamp"],
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="Velas",
        ),
        row=1, col=1,
    )

    for col, color, name in [
        ("sma_7",  "orange", "SMA 7"),
        ("sma_25", "blue",   "SMA 25"),
        ("sma_50", "purple", "SMA 50"),
    ]:
        if col in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot["timestamp"],
                    y=df_plot[col],
                    line=dict(color=color, width=2),
                    name=name,
                ),
                row=1, col=1,
            )

    # Predicción
    if prediction is not None:
        interval_val = df_plot["interval"].iloc[0] if "interval" in df_plot.columns else "1h"
        delta_map    = {"1h": 1, "4h": 4, "24h": 24}
        delta_hours  = delta_map.get(interval_val, 1)

        last_time    = df_plot["timestamp"].iloc[-1]
        next_time    = last_time + pd.Timedelta(hours=delta_hours)

        fig.add_trace(
            go.Scatter(
                x=[last_time, next_time],
                y=[df_plot["close"].iloc[-1], prediction],
                mode="lines+markers",
                line=dict(color="red", width=3, dash="dash"),
                marker=dict(size=10, color="red"),
                name="Predicción",
            ),
            row=1, col=1,
        )

    if show_rsi and "rsi_14" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["rsi_14"],
                line=dict(color="green", width=2),
                name="RSI 14",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False)

    return fig

if __name__ == "__main__":
    import pandas as pd
    from config import PICKLE_PROCESSED
    from predictor import predict_next_price

    df  = pd.read_pickle(PICKLE_PROCESSED)
    btc = df[df["symbol"] == "BTC"].copy()

    pred  = predict_next_price(btc, model_type="ridge")
    title = f"BTC — Predicción siguiente vela: ${pred:,.2f}"
    fig   = plot_crypto_candlestick(btc, title=title, prediction=pred)
    fig.show()
