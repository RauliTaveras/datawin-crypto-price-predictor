# 📈 Crypto Predictor

Pipeline completo de predicción de precios de criptomonedas con dashboard interactivo.

---

## Estructura del proyecto

```
crypto_predictor/
│
├── config.py               # Constantes globales (criptos, rutas, hiperparámetros)
├── data_collector.py       # Paso 1 — Extracción OHLCV (CoinGecko + CryptoCompare)
├── feature_engineering.py  # Pasos 2+3 — Limpieza y features técnicas
├── model_trainer.py        # Paso 4 — Entrenamiento Ridge + SARIMAX
├── predictor.py            # Paso 5 — Predicción del siguiente precio
├── visualizer.py           # Gráficos de velas con Plotly
├── pipeline_etl.py         # Orquestador completo + scheduler automático
├── dashboard.py            # Dashboard Streamlit
├── requirements.txt
│
├── data/                   # Generado automáticamente
│   ├── crypto_data.db
│   └── crypto_processed.pkl
│
├── models/                 # Generado automáticamente
│   ├── best_ridge_model.pkl
│   ├── sarimax_model.pkl
│   └── feature_cols.pkl
│
└── logs/                   # Generado automáticamente
    └── pipeline.log
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Ejecutar el pipeline una sola vez
```bash
python pipeline_etl.py
```

### 2. Ejecutar en loop automático (cada hora)
```bash
python pipeline_etl.py --loop
```

### 3. Solo actualizar datos sin re-entrenar
```bash
python pipeline_etl.py --no-retrain
```

### 4. Lanzar el dashboard
```bash
streamlit run dashboard.py
```

---

## Flujo del Pipeline ETL

```
Extraer          Limpiar + Features      Entrenar        Predecir
CoinGecko    →   Lags, SMA, EMA,    →   Ridge +     →   Próxima
CryptoCompare    RSI, Bollinger Bands    SARIMAX         vela
                 Volumen, Temporales
```

---

## APIs utilizadas

| API            | Uso                                 | Restricción          |
|----------------|-------------------------------------|----------------------|
| CoinGecko      | Datos diarios (fuente principal)    | 30 req/min (free)    |
| CryptoCompare  | Datos 1h / 4h / fallback diario     | 100 req/min (free)   |
| Binance        | ❌ No usada (error 451 por región)  | —                    |

---

## Modelos

| Modelo   | Propósito                         | Métricas típicas           |
|----------|-----------------------------------|----------------------------|
| Ridge    | Predicción con features técnicas  | R² ≈ 0.9999, RMSE ≈ $300  |
| SARIMAX  | Modelo de series de tiempo puro   | Complementario al Ridge     |

## Modelo de IA: phi4-mini
