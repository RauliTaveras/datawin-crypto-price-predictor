#Rauli
# config.py — Configuración global del proyecto

CRYPTOS   = ["BTC", "ETH", "SOL", "BNB", "XRP"]
INTERVALS = ["1h", "4h", "24h"]
DAYS_BACK = 90

# Mapeo de símbolo → ID en CoinGecko
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
}

# Rutas de los archivos
DB_PATH             = "data/crypto_data.db"
DB_TABLE_RAW        = "crypto_raw"
DB_TABLE_PROCESSED  = "crypto_processed"
PICKLE_PROCESSED    = "data/crypto_processed.pkl"
MODEL_RIDGE         = "models/best_ridge_model.pkl"
MODEL_SARIMAX       = "models/sarimax_model.pkl"
FEATURE_COLS_PATH   = "models/feature_cols.pkl"
LOG_FILE            = "logs/pipeline.log"

# Hiperparámetros para el modelo Ridge
RIDGE_ALPHAS = [0.1, 1.0, 10.0]

# Parámetros para el modelo SARIMAX
SARIMAX_ORDER          = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 24)

# Intervalo ETL automático (en segundos)
ETL_INTERVAL_SECONDS = 3600  # 1 hora
