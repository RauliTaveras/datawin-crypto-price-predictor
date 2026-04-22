# ============================================================
# utils/logger.py — Logger centralizado
# ============================================================

import logging
import os
from config import LOG_FILE

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("crypto_predictor")
