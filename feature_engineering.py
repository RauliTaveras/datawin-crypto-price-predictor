import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH, DB_TABLE_PROCESSED, PICKLE_PROCESSED
from utils.logger import logger

def add_time_series_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy().sort_values("timestamp")

    for lag in [1, 2, 3, 5]:
        g[f"close_log_{lag}"] = g["close"].shift(lag)

    # Medias móviles simples
    g["sma_7"]  = g["close"].rolling(window=7).mean()
    g["sma_25"] = g["close"].rolling(window=25).mean()
    g["sma_50"] = g["close"].rolling(window=50).mean()

    # Media móvil exponencial
    g["ema_12"] = g["close"].ewm(span=12, adjust=False).mean()

    # Volatilidad
    g["volatily_7"]  = g["close"].rolling(window=7).std()
    g["volatily_25"] = g["close"].rolling(window=14).std()

    # RSI (14 períodos)
    delta = g["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / loss
    g["rsi_14"] = 100 - (100 / (1 + rs))

    # Bandas de Bollinger
    bb_mid          = g["close"].rolling(window=20).mean()
    bb_std          = g["close"].rolling(window=20).std()
    g["bb_upper"]   = bb_mid + 2 * bb_std
    g["bb_lower"]   = bb_mid - 2 * bb_std
    g["bb_width"]   = (g["bb_upper"] - g["bb_lower"]) / bb_mid

    # Volumen relativo
    g["volume_sma_7"] = g["volume"].rolling(window=7).mean()
    g["volume_ratio"] = g["volume"] / g["volume_sma_7"]

    # Features de tiempo
    g["hour"]       = g["timestamp"].dt.hour
    g["dayofweek"]  = g["timestamp"].dt.dayofweek
    g["is_weekend"] = g["dayofweek"].isin([5, 6]).astype(int)

    return g

def process(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)
    print("COLUMNAS:", df.columns.tolist()) 
    print("INDEX:", df.index.name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "interval", "timestamp"]).reset_index(drop=True)

    logger.info(f"[engineering] Aplicando features a {len(df):,} filas...")


    df_processed = (
          df.groupby(["symbol", "interval"], group_keys=False)
            .apply(add_time_series_features)
            .reset_index(drop=True)
        )

# Dolor de cabeza - Stevens 
#Esta era la parte que te decia que symbol y Interval se salian del groupby
#tuve que buscar la función process 
    if "symbol" not in df_processed.columns:
        df_processed["symbol"]   = df["symbol"].values
        df_processed["interval"] = df["interval"].values

# Target: precio de cierre del siguiente período
    df_processed["target"] = (
        df_processed.groupby(["symbol", "interval"])["close"]
                    .shift(-1)
        )


    # Eliminar filas sin target
    df_processed = df_processed.dropna(subset=["target"])

    # Rellenar NaN restantes en features con forward-fill
    feature_cols = [
        c for c in df_processed.columns
        if c not in ["timestamp", "symbol", "interval", "target"]
    ]
    df_processed[feature_cols] = df_processed[feature_cols].ffill()

    # Optimización de memoria: float64 → float32
    for col in df_processed.select_dtypes(include="float64").columns:
        df_processed[col] = pd.to_numeric(df_processed[col], downcast="float")

    logger.info(
        f"[engineering] Procesado listo — shape: {df_processed.shape}  "
        f"| memoria: {df_processed.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )
    return df_processed


def save_processed(df: pd.DataFrame) -> None:
    os.makedirs("data", exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql(DB_TABLE_PROCESSED, engine, if_exists="replace", index=False)
    df.to_pickle(PICKLE_PROCESSED)
    logger.info(f"[engineering] Guardado en '{DB_TABLE_PROCESSED}' y '{PICKLE_PROCESSED}'")

if __name__ == "__main__":
    from data_collector import collect_all
    raw       = collect_all()
    processed = process(raw)
    save_processed(processed)
    print(processed[["symbol", "interval", "timestamp", "close", "target", "rsi_14"]].tail())
