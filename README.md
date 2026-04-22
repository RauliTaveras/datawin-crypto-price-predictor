# 📈 Crypto Predictor
# Complete cryptocurrency price prediction pipeline with an interactive dashboard and chatbot.

Pipeline completo de predicción de precios de criptomonedas con dashboard interactivo y un chatbot.
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

## Use | Uso

### 1. Run the piplene once only | Ejecutar el pipeline una sola vez
```bash
python pipeline_etl.py
```

### 2. Run in automatic loop (hourly) | Ejecutar en loop automático (cada hora)
```bash
python pipeline_etl.py --loop
```

### 3. Update data only (no retraining) | Solo actualizar datos sin re-entrenar
```bash
python pipeline_etl.py --no-retrain
```

### 4. Launch the dashboard | Lanzar el dashboard
```bash
streamlit run dashboard.py
```

---

## ETL Pipeline Flow | Flujo del Pipeline ETL

```
Extract          Clean+ Features      Train        Predict
CoinGecko    →   Lags, SMA, EMA,    →   Ridge +     →   Next
CryptoCompare    RSI, Bollinger Bands    SARIMAX         candle
                 Volume, Time features
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
Used as an analytics assistant inside the dashboard to interpret metrics, chartss, charts, and model results in real time.

Utilizado como asistente de análisis dentro del dashboard para interpretar métricas, gráficos y resultados del modelo en tiempo real.

## Important Notes | Notas importantes
Trained models (.pkl) are not included in the repository due to size limitations.

Los modelos entrenados (.pkl) no se incluyen en el repositorio por limitaciones de tamaño.

Models are automatically generated when running pipeline_etl.py.

Los modelos se generan automáticamente al ejecutar pipeline_etl.py.

