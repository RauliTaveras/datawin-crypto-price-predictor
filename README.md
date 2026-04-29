# 📈 Crypto Predictor
# Complete cryptocurrency price prediction pipeline with an interactive dashboard and chatbot.

Pipeline completo de predicción de precios de criptomonedas con dashboard interactivo y un chatbot.
---

## Estructura del proyecto

```
crypto_predictor/
│
├── config.py               # Global constants (cryptos, routes, hyperparameters) | Constantes globales (criptos, rutas, hiperparámetros)
├── data_collector.py       # Step 1 — OHLCV Extraction (CoinGecko + CryptoCompare) | Paso 1 — Extracción OHLCV (CoinGecko + CryptoCompare)
├── feature_engineering.py  # Steps 2+3 — Cleaning and technical features | Pasos 2+3 — Limpieza y features técnicas
├── model_trainer.py        # Step 4 — Ridge + SARIMAX Training |  Entrenamiento Ridge + SARIMAX
├── predictor.py            # Step 5 Prediction of the next price | Predicción del siguiente precio
├── visualizer.py           # Candlestick charts with Plotly | Gráficos de velas con Plotly
├── pipeline_etl.py         # Complete orchestrator + automatic scheduler | Orquestador completo + scheduler automático
├── dashboard.py            # Streamlit Dashboard | Dashboard Streamlit
├── requirements.txt
│
├── data/                   # Automatically generated | Generado automáticamente
│   ├── crypto_data.db
│   └── crypto_processed.pkl
│
├── models/                 #Automatically generated | Generado automáticamente
│   ├── best_ridge_model.pkl
│   ├── sarimax_model.pkl
│   └── feature_cols.pkl
│
└── logs/                   #Automatically generated | Generado automáticamente
    └── pipeline.log
```

---

## installation | Instalación

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
| CoinGecko      | Daily data (main source)            | 30 req/min (free)    |
| CryptoCompare  | Data 1h / 4h / daily fallback       | 100 req/min (free)   |


---

## Models | Modelos
| Model    | Por   puse                        | Typical metrics           |
|----------|-----------------------------------|----------------------------|
| Ridge    | Prediction with technical features  | R² ≈ 0.9999, RMSE ≈ $300  |
| SARIMAX  | Prediction with technical features   | Complementary to the Ridge     |

| Modelos    | Propósito                         | Métricas típicas           |
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

