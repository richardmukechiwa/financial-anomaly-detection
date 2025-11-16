from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
import os
import mysql.connector

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/iforest_model_v1.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DB_HOST = os.getenv("DB_HOST", "n8n-mysql")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_DATABASE = os.getenv("DB_DATABASE", "n8n_workflows")
SCORE_THRESHOLD_HIGH = float(os.getenv("SCORE_THRESHOLD_HIGH", -0.3))
SCORE_THRESHOLD_MED = float(os.getenv("SCORE_THRESHOLD_MED", -0.1))

# Load trained pipeline
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from {}", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model: {}", e)
    raise

app = FastAPI(title="ML Scoring Service", version=MODEL_VERSION)

# ----------------------
# Pydantic schemas
# ----------------------
class TransactionIn(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float
    customer_id: str
    merchant_id: str
    channel: Optional[str] = None
    location: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day_of_week: Optional[int] = None
    hour: Optional[int] = None

    @validator("amount")
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("amount must be non-negative")
        return v
    
    @validator("transaction_id")
    def id_must_be_int(cls, v):
        if not str(v).isdigit():
            raise ValueError("transaction_id must be numeric")
        return v

    def compute_time_features(self):
        ts = pd.to_datetime(self.timestamp, utc=True)
        self.year = ts.year
        self.month = ts.month
        self.day_of_week = ts.dayofweek
        self.hour = ts.hour
        return self

class ScoreOut(BaseModel):
    transaction_id: str
    anomaly_score: float
    deviation_score: float
    rule_score: float
    aggregated_score: float
    risk_level: str
    model_version: str

# ----------------------
# DB helper
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
# Feature helpers
# ----------------------
def prepare_features(tx: TransactionIn) -> pd.DataFrame:
    tx.compute_time_features()
    tx.customer_id = str(tx.customer_id)
    tx.merchant_id = str(tx.merchant_id)
    df = pd.DataFrame([{
        "amount": tx.amount,
        "year": tx.year,
        "month": tx.month,
        "day_of_week": tx.day_of_week,
        "hour": tx.hour,
        "channel": tx.channel or "",
        "customer_id": tx.customer_id,
        "merchant_id": tx.merchant_id,
        "location": tx.location or ""
    }])
    return df

# ----------------------
# Historical deviation
# ----------------------
def compute_profile_features(tx: TransactionIn) -> Dict[str, float]:
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT AVG(amount) AS cust_avg, COUNT(*)/90.0 AS cust_freq
            FROM transactions
            WHERE customer_id=%s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (tx.customer_id,))
        row = cur.fetchone() or {}
        cust_avg = float(row.get("cust_avg") or 0.0)
        cust_freq = float(row.get("cust_freq") or 0.0)

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
# Rule evaluation
# ----------------------
def evaluate_rules(tx: TransactionIn) -> Dict[str, float]:
    score = 0.0
    if tx.amount > 1000:
        score += 0.6
    if tx.channel and tx.channel.lower() == "card" and tx.location and tx.location.lower() not in ("harare","bulawayo"):
        score += 0.3
    if tx.hour < 5:
        score += 0.2
    if score > 1.0:
        score = 1.0
    return {"rule_score": score}

# ----------------------
# Aggregation
# ----------------------
def aggregate_scores(anomaly_score: float, deviation_score: float, rule_score: float) -> Dict[str, Any]:
    if anomaly_score <= SCORE_THRESHOLD_HIGH:
        anomaly_risk = 1.0
    elif anomaly_score <= SCORE_THRESHOLD_MED:
        anomaly_risk = 0.7
    else:
        anomaly_risk = 0.2
    deviation_risk = min(max(deviation_score / 2.0, 0.0), 1.0)
    agg = 0.6*anomaly_risk + 0.25*rule_score + 0.15*deviation_risk
    if agg >= 0.7:
        lvl = "High"
    elif agg >= 0.4:
        lvl = "Medium"
    else:
        lvl = "Low"
    return {"aggregated_score": agg, "risk_level": lvl}

# ----------------------
# Persist
# ----------------------
def persist_score(tx: TransactionIn, payload: Dict[str, Any]):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO anomalies_log (transaction_id, anomaly_score, deviation_score, rule_score, aggregated_score, risk_level, model_version, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            tx.transaction_id,
            payload["anomaly_score"],
            payload["deviation_score"],
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
def score_transaction(tx: TransactionIn):
    try:
        X = prepare_features(tx)
        try:
            s = float(model.decision_function(X)[0])
        except Exception as e:
            logger.exception("Model scoring failed: {}", e)
            raise HTTPException(status_code=500, detail="Model scoring failed")

        prof = compute_profile_features(tx)
        rules = evaluate_rules(tx)
        agg = aggregate_scores(anomaly_score=s, deviation_score=prof["deviation_score"], rule_score=rules["rule_score"])

        out = {
            "transaction_id": tx.transaction_id,
            "anomaly_score": s,
            "deviation_score": prof["deviation_score"],
            "rule_score": rules["rule_score"],
            "aggregated_score": agg["aggregated_score"],
            "risk_level": agg["risk_level"],
            "model_version": MODEL_VERSION
        }

        persist_score(tx, out)
        return out
    except Exception as e:
        logger.exception("Scoring path failed: {}", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/score/batch")
def score_batch(txs: List[TransactionIn]):
    results = []
    for tx in txs:
        try:
            res = score_transaction(tx)
            results.append(res)
        except Exception as e:
            results.append({"transaction_id": getattr(tx, "transaction_id", None), "error": str(e)})
    return results
