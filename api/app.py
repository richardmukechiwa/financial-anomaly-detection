# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger
import os
import mysql.connector
from datetime import datetime
import requests
import time

# CONFIG
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/ensemble_model_v1.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
ML_URL = os.getenv("ML_URL", "http://ml:8001/predict")
ML_RETRY_ATTEMPTS = int(os.getenv("ML_RETRY_ATTEMPTS", 3))
ML_RETRY_DELAY = float(os.getenv("ML_RETRY_DELAY", 2.0))

DB_HOST = os.getenv("DB_HOST", "n8n-mysql")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_DATABASE = os.getenv("DB_DATABASE", "n8n_workflows")

SCORE_THRESHOLD_HIGH = float(os.getenv("SCORE_THRESHOLD_HIGH", "-0.3"))
SCORE_THRESHOLD_MED = float(os.getenv("SCORE_THRESHOLD_MED", "-0.1"))
AMOUNT_HIGH_RISK = float(os.getenv("AMOUNT_HIGH_RISK", "10000.0"))

app = FastAPI(title="Hybrid ML Scoring Service", version=MODEL_VERSION)

# ---------------------------
# Models
# ---------------------------
class TransactionIn(BaseModel):
    transaction_id: str
    timestamp: Optional[str] = None
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
    def amount_non_negative(cls, v):
        if v < 0:
            raise ValueError("amount must be non-negative")
        return v

    def compute_time_features(self):
        if self.timestamp:
            ts = pd.to_datetime(self.timestamp, utc=True)
            self.year = int(ts.year)
            self.month = int(ts.month)
            self.day_of_week = int(ts.dayofweek)
            self.hour = int(ts.hour)
        return self

class ScoreOut(BaseModel):
    transaction_id: str
    anomaly_score_iforest: float
    anomaly_score_lof: float
    anomaly_score_ae: float
    deviation_score: float
    rule_score: float
    aggregated_score: float
    risk_level: str
    model_version: str

# ---------------------------
# DB helper
# ---------------------------
def get_db_conn():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_DATABASE,
        autocommit=True
    )

# ---------------------------
# ML Worker failover
# ---------------------------
def call_ml_worker(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for attempt in range(1, ML_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(f"{ML_URL}", json=transactions, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning("ML worker returned status {} on attempt {}", resp.status_code, attempt)
        except requests.RequestException as e:
            logger.warning("Attempt {}: ML worker call failed: {}", attempt, e)
        time.sleep(ML_RETRY_DELAY)

    # Fallback if ML worker unavailable
    logger.error("ML worker unavailable after {} attempts, returning default scores", ML_RETRY_ATTEMPTS)
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

# ---------------------------
# Scoring helpers
# ---------------------------
def compute_histories(tx: TransactionIn):
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT customer_id, AVG(amount) AS avg_amt, COUNT(*) AS cnt
            FROM transactions
            WHERE customer_id = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
            GROUP BY customer_id
        """, (tx.customer_id,))
        row = cur.fetchone()
        cust_hist = {str(tx.customer_id): {"avg": float(row.get("avg_amt") or 0.0), "count": float(row.get("cnt") or 0.0)}} if row else {str(tx.customer_id): {"avg":0.0, "count":0.0}}

        cur.execute("""
            SELECT merchant_id, AVG(amount) AS avg_amt
            FROM transactions
            WHERE merchant_id = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
            GROUP BY merchant_id
        """, (tx.merchant_id,))
        row2 = cur.fetchone()
        merch_hist = {str(tx.merchant_id): {"avg": float(row2.get("avg_amt") or 0.0)}} if row2 else {str(tx.merchant_id): {"avg": 0.0}}

        cur.close()
        conn.close()
        return cust_hist, merch_hist
    except Exception as e:
        logger.exception("Failed to fetch histories: {}", e)
        return {str(tx.customer_id): {"avg":0.0, "count":0.0}}, {str(tx.merchant_id): {"avg":0.0}}

def prepare_features_for_model(tx: TransactionIn, cust_hist: Dict[str, Any], merch_hist: Dict[str, Any]) -> Dict[str, Any]:
    tx.compute_time_features()
    cust_avg = cust_hist.get(str(tx.customer_id), {}).get("avg", 0.0)
    cust_count = cust_hist.get(str(tx.customer_id), {}).get("count", 0.0)
    merch_avg = merch_hist.get(str(tx.merchant_id), {}).get("avg", 0.0)
    return {
        "transaction_id": tx.transaction_id,
        "amount": float(tx.amount),
        "year": int(tx.year) if tx.year is not None else 0,
        "month": int(tx.month) if tx.month is not None else 0,
        "day_of_week": int(tx.day_of_week) if tx.day_of_week is not None else 0,
        "hour": int(tx.hour) if tx.hour is not None else 0,
        "cust_avg_amount": float(cust_avg),
        "cust_txn_count": float(cust_count),
        "merch_avg_amount": float(merch_avg),
        "channel": tx.channel or "",
        "location": tx.location or ""
    }

def evaluate_rules(tx: TransactionIn) -> Dict[str, Any]:
    score = 0.0
    flags = []
    if tx.amount > 1000:
        score += 0.6
        flags.append("large_amount")
    if tx.channel and tx.channel.lower() == "card" and tx.location and tx.location.lower() not in ("harare","bulawayo"):
        score += 0.3
        flags.append("card_unusual_location")
    if tx.hour is not None and tx.hour < 5:
        score += 0.2
        flags.append("odd_hour")
    return {"rule_score": min(score, 1.0), "rule_flags": flags}

def aggregate_scores(normalized_model_risks: Dict[str, float], deviation_score: float, rule_score: float, amount: float) -> Dict[str, Any]:
    model_risk = 0.4*normalized_model_risks["iforest"] + 0.3*normalized_model_risks["lof"] + 0.3*normalized_model_risks["ae"]
    agg = 0.45*rule_score + 0.4*model_risk + 0.15*min(max(deviation_score/2.0,0.0),1.0)
    if rule_score >= 0.6 and agg < 0.4:
        agg = 0.5
    if amount >= AMOUNT_HIGH_RISK:
        agg = max(agg,0.85)
    if agg >= 0.7:
        lvl = "High"
    elif agg >= 0.4:
        lvl = "Medium"
    else:
        lvl = "Low"
    return {"aggregated_score": float(agg), "risk_level": lvl, "model_risk": float(model_risk)}

def persist_score(tx: TransactionIn, payload: Dict[str, Any]):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO anomalies_log (transaction_id, anomaly_score_iforest, anomaly_score_lof, anomaly_score_ae,
                                     deviation_score, rule_score, aggregated_score, risk_level, model_version, timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        """, (
            tx.transaction_id,
            payload.get("anomaly_score_iforest"),
            payload.get("anomaly_score_lof"),
            payload.get("anomaly_score_ae"),
            payload.get("deviation_score"),
            payload.get("rule_score"),
            payload.get("aggregated_score"),
            payload.get("risk_level"),
            payload.get("model_version")
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.exception("Persist failed: {}", e)

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"status":"ok","model_loaded":True,"model_version":MODEL_VERSION}

@app.post("/score/batch")
def score_batch(txs: List[TransactionIn]):
    tx_dicts = [prepare_features_for_model(tx, *compute_histories(tx)) for tx in txs]
    ml_results = call_ml_worker(tx_dicts)
    results = []

    for tx, ml_res in zip(txs, ml_results):
        try:
            norm = {
                "iforest": ml_res.get("iforest",0.0),
                "lof": ml_res.get("lof",0.0),
                "ae": ml_res.get("autoencoder",0.0)
            }
            cust_hist, _ = compute_histories(tx)
            cust_avg = cust_hist.get(str(tx.customer_id), {}).get("avg",0.0)
            deviation = (tx.amount - cust_avg)/(cust_avg+1e-9) if cust_avg>0 else 0.0

            rules = evaluate_rules(tx)
            agg = aggregate_scores(norm, deviation, rules["rule_score"], tx.amount)

            out = {
                "transaction_id": tx.transaction_id,
                "anomaly_score_iforest": ml_res.get("iforest",0.0),
                "anomaly_score_lof": ml_res.get("lof",0.0),
                "anomaly_score_ae": ml_res.get("autoencoder",0.0),
                "deviation_score": deviation,
                "rule_score": rules["rule_score"],
                "aggregated_score": agg["aggregated_score"],
                "risk_level": agg["risk_level"],
                "model_version": ml_res.get("model_version","unavailable")
            }

            persist_score(tx,out)
            results.append(out)
        except Exception as e:
            results.append({"transaction_id": getattr(tx,"transaction_id",None),"error":str(e)})

    return results
