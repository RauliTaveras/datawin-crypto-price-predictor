# Pipeline automatiado con 
#1. Extraer → 2. Limpiar → 3. Feature engineering → 4. Entrenar → 5. Predecir
#ejecución cada 1hora


import sys
import time
import argparse
import schedule
import pandas as pd
from datetime import datetime

from data_collector    import collect_all, save_raw
from feature_engineering import process, save_processed
from model_trainer     import train_ridge, train_sarimax, save_models
from predictor         import predict_next_price
from utils.logger      import logger
from config            import PICKLE_PROCESSED, ETL_INTERVAL_SECONDS


# pipeline completo

def run_full_pipeline(retrain: bool = True) -> dict:

    start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"PIPELINE iniciado: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Paso 1: Extracción de datos
    logger.info("► Paso 1/5 — Extracción de datos")
    df_raw = collect_all()
    save_raw(df_raw)

    # Paso 2 + 3: Limpieza y Feature Engineering 
    logger.info("► Paso 2+3/5 — Limpieza y Feature Engineering")
    df_processed = process(df_raw)
    save_processed(df_processed)

    # Paso 4: Entrenamiento de modelos
    metrics_ridge = {}
    if retrain:
        logger.info("► Paso 4/5 — Entrenamiento de modelos")
        ridge_model, feature_cols, metrics_ridge = train_ridge(df_processed)
        sarimax_model = train_sarimax(df_processed)
        save_models(ridge_model, feature_cols, sarimax_model)
    else:
        logger.info("► Paso 4/5 — Entrenamiento omitido (retrain=False)")

    # Paso 5: Predicciones
    logger.info("► Paso 5/5 — Generando predicciones")
    predictions = {}

    for symbol in df_processed["symbol"].unique():
        predictions[symbol] = {}
        for interval in df_processed["interval"].unique():
            subset = df_processed[
                (df_processed["symbol"]   == symbol) &
                (df_processed["interval"] == interval)
            ].copy()

            if subset.empty:
                continue

            try:
                pred = predict_next_price(subset, model_type="ridge")
                predictions[symbol][interval] = pred
                logger.info(
                    f"    {symbol:4s} / {interval:3s} → ${pred:>12,.2f}"
                )
            except Exception as exc:
                logger.warning(f"    {symbol}/{interval} — no se pudo predecir: {exc}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"PIPELINE completado en {elapsed:.1f}s")
    logger.info("=" * 60)

    return {"metrics_ridge": metrics_ridge, "predictions": predictions}

#Programa

def run_pipeline_job():
    """Wrapper sin argumentos para schedule."""
    run_full_pipeline(retrain=True)


def start_scheduler(interval_seconds: int = ETL_INTERVAL_SECONDS):

    interval_minutes = interval_seconds // 60
    logger.info(
        f"Scheduler activo — el pipeline se ejecutará cada "
        f"{interval_minutes} minutos."
    )
    run_pipeline_job()

    # Sigueientes ejecuciones
    schedule.every(interval_minutes).minutes.do(run_pipeline_job)

    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline ETL de predicción de criptomonedas."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Ejecuta el pipeline en un bucle cada hora (ETL_INTERVAL_SECONDS).",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Omite el re-entrenamiento; solo actualiza datos y predice.",
    )
    args = parser.parse_args()

    if args.loop:
        start_scheduler()
    else:
        result = run_full_pipeline(retrain=not args.no_retrain)
        print("\n── Predicciones ──")
        for sym, intervals in result["predictions"].items():
            for itv, price in intervals.items():
                print(f"  {sym:4s} / {itv:3s} → ${price:>12,.2f}")
