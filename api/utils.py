import requests
import time
from loguru import logger
from typing import List, Dict, Any
import os

ML_URL = os.getenv("ML_URL", "http://ml:8001/predict")
RETRY_ATTEMPTS = int(os.getenv("ML_RETRY_ATTEMPTS", 3))
RETRY_DELAY = float(os.getenv("ML_RETRY_DELAY", 2.0))  # seconds

def call_ml_worker(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Call ML worker with retries. If all retries fail, return placeholder scores.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(f"{ML_URL}", json=transactions, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning("ML worker returned status {} on attempt {}", resp.status_code, attempt)
        except requests.RequestException as e:
            logger.warning("Attempt {}: ML worker call failed: {}", attempt, e)
        time.sleep(RETRY_DELAY)
    
    # If all retries fail, log and return placeholder scores
    logger.error("ML worker unavailable after {} attempts, returning default scores", RETRY_ATTEMPTS)
    placeholder = []
    for tx in transactions:
        placeholder.append({
            "transaction_id": tx.get("transaction_id"),
            "iforest": 0.0,
            "lof": 0.0,
            "autoencoder": 0.0,
            "model_version": "unavailable"
        })
    return placeholder
