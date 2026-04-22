
#Predicción del siguiente precio


import pickle
import joblib
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_RIDGE, MODEL_SARIMAX, FEATURE_COLS_PATH
from utils.logger import logger


def predict_next_price(df: pd.DataFrame,
                       model_type: str = "ridge") -> float:
    
    if model_type == "ridge":
        model = joblib.load(MODEL_RIDGE)
        with open(FEATURE_COLS_PATH, "rb") as f:
            cols = pickle.load(f)
        X_next = df[cols].iloc[[-1]]
        pred   = model.predict(X_next)[0]

    elif model_type == "sarimax":
        model = joblib.load(MODEL_SARIMAX)
        pred  = model.forecast(steps=1).iloc[0]

    else:
        raise ValueError(f"model_type desconocido: {model_type!r}")

    logger.info(f"[predictor] Predicción ({model_type}): ${float(pred):,.2f}")
    return float(pred)


# ejecución directa

if __name__ == "__main__":
    from config import PICKLE_PROCESSED
    df  = pd.read_pickle(PICKLE_PROCESSED)
    btc = df[df["symbol"] == "BTC"].copy()

    ridge_pred   = predict_next_price(btc, model_type="ridge")
    sarimax_pred = predict_next_price(btc, model_type="sarimax")

    print(f"Ridge   → ${ridge_pred:,.2f}")
    print(f"SARIMAX → ${sarimax_pred:,.2f}")
