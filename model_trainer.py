# Entrenamiento del modelo

import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DB_PATH, PICKLE_PROCESSED,
    MODEL_RIDGE, MODEL_SARIMAX, FEATURE_COLS_PATH,
    RIDGE_ALPHAS, SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER,
)
from utils.logger import logger


# Ridge

def train_ridge(df: pd.DataFrame) -> tuple[object, list[str], dict]:
    exclude = ["timestamp", "symbol", "interval", "target"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].select_dtypes(include="number")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["target"]

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge()),
    ])

    tscv        = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        pipeline,
        param_grid={"model__alpha": RIDGE_ALPHAS},
        cv=tscv,
        scoring="neg_mean_squared_error",
    )

    logger.info("[trainer] Entrenando Ridge con GridSearchCV...")
    grid_search.fit(X, y)

    best   = grid_search.best_estimator_
    y_pred = best.predict(X)

    metrics = {
        "alpha": grid_search.best_params_["model__alpha"],
        "MAE":   mean_absolute_error(y, y_pred),
        "RMSE":  np.sqrt(mean_squared_error(y, y_pred)),
        "R2":    r2_score(y, y_pred),
    }

    logger.info(
        f"[trainer] Ridge — alpha={metrics['alpha']}  "
        f"MAE=${metrics['MAE']:,.2f}  RMSE=${metrics['RMSE']:.2f}  R²={metrics['R2']:.4f}"
    )
    return best, feature_cols, metrics


# SARIMAX 

def train_sarimax(df: pd.DataFrame) -> object:
    price_series = (
        df.set_index("timestamp")["close"]
          .sort_index()
          .dropna()
    )

    logger.info("[trainer] Entrenando SARIMAX...")
    model = SARIMAX(
        price_series,
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    logger.info("[trainer] SARIMAX listo.")
    return model

def save_models(ridge_model, feature_cols: list, sarimax_model) -> None:
    os.makedirs("models", exist_ok=True)
    joblib.dump(ridge_model,   MODEL_RIDGE)
    joblib.dump(sarimax_model, MODEL_SARIMAX)
    with open(FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)
    logger.info(f"[trainer] Modelos guardados en models/")
    
if __name__ == "__main__":
    df = pd.read_pickle(PICKLE_PROCESSED)

    ridge, feat_cols, metrics = train_ridge(df)
    sarimax                   = train_sarimax(df)

    save_models(ridge, feat_cols, sarimax)
    print("\nMétricas Ridge:", metrics)
    print(sarimax.summary().tables[1])
