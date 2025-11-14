from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
from loguru import logger
import os
import mysql.connector
from datetime import datetime, timezone

# CONFIG: environment-driven where appropriate
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/iforest_model_v1.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DB_HOST = os.getenv("DB_HOST", "n8n-mysql")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_DATABASE = os.getenv("DB_DATABASE", "n8n_workflows")
SCORE_THRESHOLD_HIGH = float(os.getenv("SCORE_THRESHOLD_HIGH", "-0.3"))  # example: lower -> more anomalous
SCORE_THRESHOLD_MED = float(os.getenv("SCORE_THRESHOLD_MED", "-0.1"))

# Load model once
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from {}", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model: {}", e)
    raise

app = FastAPI(title="ML Scoring Service", version=MODEL_VERSION)

# Pydantic schemas
class TransactionIn(BaseModel):
    transaction_id: int
    year: int
    month: int
    day_of_week: int
    hour: int
    amount: float
    customer_id: str
    merchant_id: str
    channel: Optional[str] = None
    location: Optional[str] = None

    @validator("amount")
    def amount_must_be_positive(cls, v):
        if v is None or not isinstance(v, (int, float)) or v < 0:
            raise ValueError("amount must be non-negative number")
        return float(v)

class ScoreOut(BaseModel):
    transaction_id: int
    anomaly_score: float
    deviation_score: Optional[float] = None
    rule_score: Optional[float] = None
    aggregated_score: float
    risk_level: str
    model_version: str

# DB helper (simple connection per request; replace with pool in heavy load)
def get_db_conn():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_DATABASE,
        autocommit=True,
    )

# Historical profiling helper
def compute_profile_features(tx: TransactionIn) -> Dict[str, Any]:
    """Query DB for simple historical stats for customer and merchant."""
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        # customer avg amount and frequency (last 90 days)
        cur.execute("""
            SELECT AVG(amount) AS cust_avg, COUNT(*)/90.0 AS cust_freq_per_day
            FROM transactions
            WHERE customer_id = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (tx.customer_id,))
        row = cur.fetchone() or {}
        cust_avg = float(row.get("cust_avg") or 0.0)
        cust_freq = float(row.get("cust_freq_per_day") or 0.0)

        # merchant avg for context
        cur.execute("""
            SELECT AVG(amount) AS merch_avg
            FROM transactions
            WHERE merchant_id = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (tx.merchant_id,))
        row2 = cur.fetchone() or {}
        merch_avg = float(row2.get("merch_avg") or 0.0)

        cur.close()
        conn.close()
        # deviation_score: how many std devs above customer's avg
        deviation = 0.0
        if cust_avg > 0:
            deviation = (tx.amount - cust_avg) / (cust_avg + 1e-9)
        return {"customer_avg_amount": cust_avg, "customer_frequency": cust_freq, "merchant_avg_amount": merch_avg, "deviation_score": deviation}
    except Exception as e:
        logger.exception("Profile compute failure: {}", e)
        return {"customer_avg_amount": 0.0, "customer_frequency": 0.0, "merchant_avg_amount": 0.0, "deviation_score": 0.0}

# Rule evaluation: returns a rule score [0..1] (higher = more suspicious) and flags
def evaluate_rules(tx: TransactionIn) -> Dict[str, Any]:
    flags = []
    score = 0.0
    # example rules; tune to business
    if tx.amount > 1000:  # high-value
        flags.append("high_value")
        score += 0.6
    if tx.channel and tx.channel.lower() == "card" and tx.location and tx.location.lower() not in ("harare","bulawayo"):
        flags.append("card_unusual_location")
        score += 0.3
    # time of day suspicious example: midnight
    if tx.timestamp:
        hour = tx.timestamp.astimezone(timezone.utc).hour
        if hour < 5:
            flags.append("odd_hour")
            score += 0.2
    # cap rule_score to 1
    if score > 1:
        score = 1.0
    return {"rule_score": float(score), "rule_flags": flags}

def aggregate_scores(anomaly_score: float, deviation_score: float, rule_score: float) -> Dict[str, Any]:
    # Normalize anomaly_score (model.decision_function: larger -> normal; smaller -> anomalous)
    # We map anomaly_score -> anomaly_risk in [0,1] where higher means more risky.
    # Example transform: inverse-sigmoid-ish scaling (simple linear clamp here, tune later)
    # Use negative thresholds because isolation forest returns negative for anomalies commonly.
    anomaly_risk = 0.0
    # make a heuristic mapping; tune per your model
    if anomaly_score <= SCORE_THRESHOLD_HIGH:
        anomaly_risk = 1.0
    elif anomaly_score <= SCORE_THRESHOLD_MED:
        anomaly_risk = 0.7
    else:
        anomaly_risk = 0.2

    # deviation_score: if deviation > 1 (100% above customer avg) increase risk
    deviation_risk = min(max(deviation_score / 2.0, 0.0), 1.0)  # simple scaling

    # Combine with weights: anomaly (0.6), rules (0.25), deviation (0.15)
    agg = 0.6 * anomaly_risk + 0.25 * float(rule_score) + 0.15 * deviation_risk
    # risk level thresholds
    if agg >= 0.7:
        lvl = "High"
    elif agg >= 0.4:
        lvl = "Medium"
    else:
        lvl = "Low"
    return {"aggregated_score": float(agg), "risk_level": lvl, "anomaly_risk": anomaly_risk, "deviation_risk": deviation_risk}

def persist_score(tx: TransactionIn, score_payload: Dict[str, Any]):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO anomalies_log (transaction_id, anomaly_score, deviation_score, rule_score, aggregated_score, risk_level, model_version, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            tx.transaction_id,
            float(score_payload.get("anomaly_score", 0.0)),
            float(score_payload.get("deviation_score", 0.0)),
            float(score_payload.get("rule_score", 0.0)),
            float(score_payload.get("aggregated_score", 0.0)),
            score_payload.get("risk_level", "Low"),
            MODEL_VERSION
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.exception("Failed to persist score: {}", e)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}

@app.get("/version")
def version():
    return {"model_version": MODEL_VERSION, "model_path": MODEL_PATH}

@app.post("/score", response_model=ScoreOut)
def score_transaction(tx: TransactionIn):
    # Validation is already performed by pydantic schema (amount, types)
    try:
        prof = compute_profile_features(tx)
        rules = evaluate_rules(tx)
        # prepare features for model; adapt to your model's expected features
        # We'll use amount, customer_id, merchant_id as numerical features --- replicate training features
        X = np.array([[tx.amount, tx.customer_id, tx.merchant_id]])
        try:
            s = float(model.decision_function(X)[0])
        except Exception as e:
            logger.exception("Model scoring failure: {}", e)
            raise HTTPException(status_code=500, detail="Model scoring failure")
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
        # persist asynchronous-ish (best-effort)
        try:
            persist_score(tx, out)
        except Exception:
            logger.exception("Persisting failed, continuing.")

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
