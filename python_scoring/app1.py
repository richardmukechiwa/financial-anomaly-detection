from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from loguru import logger

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/ensemble_model_v1.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
SCORE_THRESHOLD_HIGH = float(os.getenv("SCORE_THRESHOLD_HIGH", "-0.3"))
SCORE_THRESHOLD_MED = float(os.getenv("SCORE_THRESHOLD_MED", "-0.1"))
AMOUNT_HIGH_RISK = float(os.getenv("AMOUNT_HIGH_RISK", "10000.0"))

# Load bundle
bundle = joblib.load(MODEL_PATH)
preproc = bundle["preproc"]
iforest = bundle["iforest"]
lof = bundle["lof"]

# Fix autoencoder path for container
ae_path = bundle.get("autoencoder_path", None)
if ae_path:
    # Normalize to container path if it still has Windows separators
    ae_path = ae_path.replace("\\", "/")
    # Prepend /app/models if path is relative
    if not ae_path.startswith("/"):
        ae_path = os.path.join("/app/models", ae_path)
    
# Load autoencoder safely
autoencoder = None
ae_threshold = float(bundle.get("ae_threshold", 1.0))
if ae_path and os.path.exists(ae_path):
    autoencoder = tf.keras.models.load_model(ae_path)
    logger.info("Loaded autoencoder from {}", ae_path)
else:
    logger.warning("Autoencoder path not found, skipping autoencoder loading: {}", ae_path)

logger.info("Loaded ML bundle from {}", MODEL_PATH)

app = FastAPI(title="ML Scoring API", version=MODEL_VERSION)

class TransactionIn(BaseModel):
    transaction_id: str
    timestamp: Optional[str]
    amount: float
    customer_id: str
    merchant_id: str
    channel: Optional[str] = ""
    location: Optional[str] = ""
    cust_avg_amount: Optional[float] = 0.0
    cust_txn_count: Optional[float] = 0.0
    merch_avg_amount: Optional[float] = 0.0

@app.post("/score")
def score_transaction(tx: TransactionIn):
    try:
        # prepare dataframe
        df = pd.DataFrame([{
            "amount": tx.amount,
            "cust_avg_amount": tx.cust_avg_amount,
            "cust_txn_count": tx.cust_txn_count,
            "merch_avg_amount": tx.merch_avg_amount,
            "channel": tx.channel,
            "location": tx.location
        }])
        X = preproc.transform(df)
        X_arr = np.asarray(X)

        # model scores
        s_if = iforest.decision_function(X_arr)[0]
        try:
            s_lof = lof._decision_function(X_arr)[0]
        except Exception:
            s_lof = -lof.negative_outlier_factor_[0] if hasattr(lof, "negative_outlier_factor_") else 0.0

        s_ae = 0.0
        if autoencoder:
            s_ae = float(np.mean(np.square(X_arr - autoencoder.predict(X_arr, verbose=0)), axis=1)[0])

        # normalize
        def inv_sig(x, center=SCORE_THRESHOLD_MED, scale=0.1):
            return 1.0 / (1.0 + np.exp((x - center) / scale))
        norm_if = float(inv_sig(s_if))
        norm_lof = float(inv_sig(s_lof))
        norm_ae = float(min(max(s_ae / (ae_threshold + 1e-9), 0.0), 1.0))

        # simple rule
        rule_score = 0.6 if tx.amount > 1000 else 0.0
        agg = 0.4 * norm_if + 0.3 * norm_lof + 0.3 * norm_ae + 0.45 * rule_score
        agg = max(agg, 0.85) if tx.amount >= AMOUNT_HIGH_RISK else agg
        risk_level = "High" if agg >= 0.7 else "Medium" if agg >= 0.4 else "Low"

        return {
            "transaction_id": tx.transaction_id,
            "anomaly_score_iforest": float(s_if),
            "anomaly_score_lof": float(s_lof),
            "anomaly_score_ae": float(s_ae),
            "rule_score": rule_score,
            "aggregated_score": float(agg),
            "risk_level": risk_level,
            "model_version": MODEL_VERSION
        }
    except Exception as e:
        logger.exception("Scoring failed: {}", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}
