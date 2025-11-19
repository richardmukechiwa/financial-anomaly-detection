from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
from loguru import logger
from model_wrapper import EnsembleModel
import mysql.connector

# ----------------------
# Environment
# ----------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/ensemble_model_v1.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DB_HOST = os.getenv("DB_HOST", "n8n-mysql")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_DATABASE = os.getenv("DB_DATABASE", "n8n_workflows")
SCORE_THRESHOLD_HIGH = float(os.getenv("SCORE_THRESHOLD_HIGH", -0.3))
SCORE_THRESHOLD_MED = float(os.getenv("SCORE_THRESHOLD_MED", -0.1))
AMOUNT_HIGH_RISK = float(os.getenv("AMOUNT_HIGH_RISK", 10000.0))

# ----------------------
# Load ensemble
# ----------------------
try:
    ensemble: EnsembleModel
    bundle = joblib.load(MODEL_PATH)
    ensemble = EnsembleModel(
        preproc=bundle["preproc"],
        iforest=bundle["iforest"],
        lof=bundle["lof"],
        autoencoder=bundle.get("autoencoder_path"),
        ae_threshold=bundle.get("ae_threshold", 1.0)
    )
    # load autoencoder if path exists
    if ensemble.autoencoder and os.path.exists(ensemble.autoencoder):
        import tensorflow as tf
        ensemble.autoencoder = tf.keras.models.load_model(ensemble.autoencoder)
        logger.info("Loaded autoencoder from {}", ensemble.autoencoder)
    logger.info("Ensemble loaded from {}", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load ensemble: {}", e)
    raise

# ----------------------
# App
# ----------------------
app = FastAPI(title="ML Scoring API", version=MODEL_VERSION)

# ----------------------
# Schemas
# ----------------------
class TransactionIn(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float
    customer_id: str
    merchant_id: str
    channel: Optional[str] = ""
    location: Optional[str] = ""
    customer_avg_amount: Optional[float] = 0.0
    customer_frequency: Optional[float] = 0.0
    merchant_avg_amount: Optional[float] = 0.0

    @validator("amount")
    def amount_positive(cls, v):
        if v < 0:
            raise ValueError("amount must be non-negative")
        return v

    @validator("transaction_id")
    def id_numeric(cls, v):
        if not str(v).isdigit():
            raise ValueError("transaction_id must be numeric")
        return v

    def compute_time_features(self):
        ts = pd.to_datetime(self.timestamp, utc=True)
        return {
            "year": ts.year,
            "month": ts.month,
            "day_of_week": ts.dayofweek,
            "hour": ts.hour
        }

class ScoreOut(BaseModel):
    transaction_id: str
    anomaly_score_iforest: float
    anomaly_score_lof: float
    anomaly_score_ae: float
    rule_score: float
    aggregated_score: float
    risk_level: str
    model_version: str

# ----------------------
# DB connection
# ----------------------
def get_db_conn():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_DATABASE,
        autocommit=True
    )

# ----------------------
# Profile features
# ----------------------
def compute_profile_features(tx: TransactionIn) -> Dict[str, float]:
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        # Customer
        cur.execute("""
            SELECT AVG(amount) AS cust_avg, COUNT(*)/90.0 AS cust_freq
            FROM transactions
            WHERE customer_id=%s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (tx.customer_id,))
        row = cur.fetchone() or {}
        cust_avg = float(row.get("cust_avg") or 0.0)
        cust_freq = float(row.get("cust_freq") or 0.0)

        # Merchant
        cur.execute("""
            SELECT AVG(amount) AS merch_avg
            FROM transactions
            WHERE merchant_id=%s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (tx.merchant_id,))
        row2 = cur.fetchone() or {}
        merch_avg = float(row2.get("merch_avg") or 0.0)

        cur.close()
        conn.close()

        deviation = (tx.amount - cust_avg) / (cust_avg + 1e-9) if cust_avg > 0 else 0.0
        return {
            "customer_avg_amount": cust_avg,
            "customer_frequency": cust_freq,
            "merchant_avg_amount": merch_avg,
            "deviation_score": deviation
        }
    except Exception as e:
        logger.exception("Profile compute failed: {}", e)
        return {"customer_avg_amount": 0.0, "customer_frequency": 0.0, "merchant_avg_amount": 0.0, "deviation_score": 0.0}

# ----------------------
# Rules
# ----------------------
def evaluate_rules(tx: TransactionIn, hour:int) -> float:
    score = 0.0
    if tx.amount > 1000:
        score += 0.6
    if tx.channel.lower() == "card" and tx.location.lower() not in ("harare","bulawayo"):
        score += 0.3
    if hour < 5:
        score += 0.2
    return min(score, 1.0)

# ----------------------
# Prepare features
# ----------------------
def prepare_features(tx: TransactionIn) -> pd.DataFrame:
    time_feats = tx.compute_time_features()
    df = pd.DataFrame([{
        "amount": tx.amount,
        "year": time_feats["year"],
        "month": time_feats["month"],
        "day_of_week": time_feats["day_of_week"],
        "hour": time_feats["hour"],
        "cust_avg_amount": tx.customer_avg_amount,
        "cust_txn_count": tx.customer_frequency,
        "merch_avg_amount": tx.merchant_avg_amount,
        "channel": tx.channel,
        "location": tx.location
    }])
    return df

# ----------------------
# Aggregate scores
# ----------------------
def aggregate_scores(s_if, s_lof, s_ae, rule_score, tx_amount):
    def inv_sig(x, center=SCORE_THRESHOLD_MED, scale=0.1):
        return 1.0 / (1.0 + np.exp((x - center) / scale))
    norm_if = float(inv_sig(s_if))
    norm_lof = float(inv_sig(s_lof))
    norm_ae = float(np.clip(s_ae / (ensemble.ae_threshold + 1e-9), 0.0, 1.0))
    agg = 0.4*norm_if + 0.3*norm_lof + 0.3*norm_ae + 0.45*rule_score
    if tx_amount >= AMOUNT_HIGH_RISK:
        agg = max(agg, 0.85)
    if agg >= 0.7:
        lvl = "High"
    elif agg >= 0.4:
        lvl = "Medium"
    else:
        lvl = "Low"
    return agg, lvl

# ----------------------
# Persist
# ----------------------
def persist_score(tx: TransactionIn, payload: Dict[str, Any]):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO anomalies_log
            (transaction_id, anomaly_score_iforest, anomaly_score_lof, anomaly_score_ae, rule_score, aggregated_score, risk_level, model_version, timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        """, (
            tx.transaction_id,
            payload["anomaly_score_iforest"],
            payload["anomaly_score_lof"],
            payload["anomaly_score_ae"],
            payload["rule_score"],
            payload["aggregated_score"],
            payload["risk_level"],
            MODEL_VERSION
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.exception("Persist failed: {}", e)

# ----------------------
# Endpoints
# ----------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}

@app.post("/score", response_model=ScoreOut)
def score_single(tx: TransactionIn):
    try:
        prof = compute_profile_features(tx)
        tx.customer_avg_amount = prof["customer_avg_amount"]
        tx.customer_frequency = prof["customer_frequency"]
        tx.merchant_avg_amount = prof["merchant_avg_amount"]

        df = prepare_features(tx)
        s_if = ensemble.score_iforest(df)[0]
        s_lof = ensemble.score_lof(df)[0]
        s_ae = ensemble.score_autoencoder(df)[0] if ensemble.autoencoder else 0.0

        rule_score = evaluate_rules(tx, df['hour'][0])
        agg, lvl = aggregate_scores(s_if, s_lof, s_ae, rule_score, tx.amount)

        out = {
            "transaction_id": tx.transaction_id,
            "anomaly_score_iforest": s_if,
            "anomaly_score_lof": s_lof,
            "anomaly_score_ae": s_ae,
            "rule_score": rule_score,
            "aggregated_score": agg,
            "risk_level": lvl,
            "model_version": MODEL_VERSION
        }

        persist_score(tx, out)
        return out
    except Exception as e:
        logger.exception("Scoring failed: {}", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/score/batch")
def score_batch(txs: List[TransactionIn]):
    results = []
    for tx in txs:
        try:
            results.append(score_single(tx))
        except Exception as e:
            results.append({"transaction_id": getattr(tx, "transaction_id", None), "error": str(e)})
    return results
