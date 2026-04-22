
import time
import requests
import pandas as pd
from sqlalchemy import create_engine

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CRYPTOS, INTERVALS, DAYS_BACK,
    COINGECKO_IDS, DB_PATH, DB_TABLE_RAW,
)
from utils.logger import logger

def fetch_from_coingecko(symbol: str, days: int = DAYS_BACK) -> pd.DataFrame | None:
    """Descarga datos diarios de CoinGecko y retorna un DataFrame OHLCV."""
    coin_id = COINGECKO_IDS.get(symbol)
    if coin_id is None:
        logger.warning(f"[CoinGecko] Símbolo desconocido: {symbol}")
        return None

    url    = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        prices  = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"]      = df["close"].shift(1)
        df["high"]      = df["close"]
        df["low"]       = df["close"]
        df["volume"]    = [v[1] for v in volumes]
        df = df.dropna()

        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.astype({"open": float, "high": float, "low": float,
                        "close": float, "volume": float})
        df.set_index("timestamp", inplace=True)
        df["symbol"]   = symbol
        df["interval"] = "24h"

        logger.info(f"[CoinGecko] {symbol} — {len(df)} filas descargadas")
        return df

    except Exception as exc:
        logger.error(f"[CoinGecko] Error en {symbol}: {exc}")
        return None


def fetch_from_cryptocompare(symbol: str, interval: str,
                             days: int = DAYS_BACK) -> pd.DataFrame | None:
    """Descarga datos horarios / 4h / diarios de CryptoCompare."""
    base_url     = "https://min-api.cryptocompare.com/data/v2"
    interval_map = {"1h": "histohour", "4h": "histohour", "24h": "histoday"}
    aggregate    = 4 if interval == "4h" else 1
    limit        = min(days * (24 if interval != "24h" else 1), 2000)

    url    = f"{base_url}/{interval_map[interval]}"
    params = {"fsym": symbol, "tsym": "USD",
              "limit": limit, "aggregate": aggregate}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data_json = resp.json()

        if data_json.get("Response") == "Error":
            raise ValueError(data_json.get("Message"))

        df = pd.DataFrame(data_json["Data"]["Data"])
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"volumeto": "volume"})
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.astype({"open": float, "high": float, "low": float,
                        "close": float, "volume": float})
        df.set_index("timestamp", inplace=True)
        df["symbol"]   = symbol
        df["interval"] = interval

        logger.info(f"[CryptoCompare] {symbol}/{interval} — {len(df)} filas descargadas")
        return df

    except Exception as exc:
        logger.error(f"[CryptoCompare] Error en {symbol}/{interval}: {exc}")
        return None


#función principal

def collect_all(cryptos: list = CRYPTOS,
                intervals: list = INTERVALS,
                days: int = DAYS_BACK) -> pd.DataFrame:
    """
    Itera todas las criptos e intervalos, combina fuentes y
    devuelve un único DataFrame crudo.
    """
    frames: list[pd.DataFrame] = []

    for symbol in cryptos:
        for interval in intervals:

            if interval == "24h":
                df = fetch_from_coingecko(symbol, days)
                if df is None:
                    df = fetch_from_cryptocompare(symbol, interval, days)
            else:
                df = fetch_from_cryptocompare(symbol, interval, days)

            if df is not None:
                df["interval"] = interval
                frames.append(df)

            time.sleep(1.5) 

    if not frames:
        raise RuntimeError("No se pudo obtener datos de ninguna fuente.")

    combined = pd.concat(frames)
    logger.info(f"[collector] Total filas recolectadas: {len(combined):,}")
    return combined

def save_raw(df: pd.DataFrame) -> None:
    """Persiste el DataFrame crudo en SQLite."""
    os.makedirs("data", exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql(DB_TABLE_RAW, engine, if_exists="replace")
    logger.info(f"[collector] Datos crudos guardados en {DB_PATH} → tabla '{DB_TABLE_RAW}'")

if __name__ == "__main__":
    raw = collect_all()
    save_raw(raw)
    print(raw.info())
