# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from loguru import logger
import os
import mysql.connector
from datetime import datetime, timezone
import tensorflow as tf

# CONFIG
MODEL_BUNDLE_PATH = os.getenv("MODEL_PATH", "/app/models/ensemble_model_v1.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DB_HOST = os.getenv("DB_HOST", "n8n-mysql")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_DATABASE = os.getenv("DB_DATABASE", "n8n_workflows")
SCORE_THRESHOLD_HIGH = float(os.getenv("SCORE_THRESHOLD_HIGH", "-0.3"))
SCORE_THRESHOLD_MED = float(os.getenv("SCORE_THRESHOLD_MED", "-0.1"))
AMOUNT_HIGH_RISK = float(os.getenv("AMOUNT_HIGH_RISK", "10000.0"))

# Load bundle
try:
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    preproc = bundle["preproc"]
    iforest = bundle["iforest"]
    lof = bundle["lof"]
    ae_path = bundle["autoencoder_path"]
    ae_threshold = float(bundle.get("ae_threshold", 1.0))
    # load autoencoder
    autoencoder = tf.keras.models.load_model(ae_path)
    logger.info("Loaded ensemble from {}", MODEL_BUNDLE_PATH)
except Exception as e:
    logger.exception("Failed to load model bundle: {}", e)
    raise

app = FastAPI(title="Hybrid ML Scoring Service", version=MODEL_VERSION)

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
        # if timestamp missing, keep provided fields or None
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

def get_db_conn():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_DATABASE,
        autocommit=True
    )

def prepare_features_for_model(tx: TransactionIn, cust_hist: Dict[str, Any], merch_hist: Dict[str, Any]) -> pd.DataFrame:
    # fill features as in training
    tx.compute_time_features()
    cust_avg = cust_hist.get(str(tx.customer_id), {}).get("avg", 0.0)
    cust_count = cust_hist.get(str(tx.customer_id), {}).get("count", 0.0)
    merch_avg = merch_hist.get(str(tx.merchant_id), {}).get("avg", 0.0)
    df = pd.DataFrame([{
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
    }])
    return df

def compute_histories(tx: TransactionIn):
    # fetch simple customer and merchant history from DB
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
        cust_hist = {}
        if row:
            cust_hist[str(tx.customer_id)] = {"avg": float(row.get("avg_amt") or 0.0), "count": float(row.get("cnt") or 0.0)}
        else:
            cust_hist[str(tx.customer_id)] = {"avg": 0.0, "count": 0.0}

        cur.execute("""
            SELECT merchant_id, AVG(amount) AS avg_amt
            FROM transactions
            WHERE merchant_id = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 90 DAY)
            GROUP BY merchant_id
        """, (tx.merchant_id,))
        row2 = cur.fetchone()
        merch_hist = {}
        if row2:
            merch_hist[str(tx.merchant_id)] = {"avg": float(row2.get("avg_amt") or 0.0)}
        else:
            merch_hist[str(tx.merchant_id)] = {"avg": 0.0}
        cur.close()
        conn.close()
        return cust_hist, merch_hist
    except Exception as e:
        logger.exception("Failed to fetch histories: {}", e)
        return {str(tx.customer_id): {"avg": 0.0, "count": 0.0}}, {str(tx.merchant_id): {"avg": 0.0}}

def model_scores_from_bundle(df_features: pd.DataFrame):
    # apply preproc
    X = preproc.transform(df_features)
    X_arr = np.asarray(X)
    s_if = iforest.decision_function(X_arr)[0]  # larger -> normal
    # convert LOF: with novelty=True LOF has _decision_function or negative_outlier_factor_
    try:
        s_lof = lof._decision_function(X_arr)[0]
    except Exception:
        # fallback, use negative_outlier_factor_ heuristic (neg flow)
        s_lof = -lof.negative_outlier_factor_[0] if hasattr(lof, "negative_outlier_factor_") else 0.0
    # AE score: MSE
    recon = autoencoder.predict(X_arr, verbose=0)
    s_ae = float(np.mean(np.square(X_arr - recon), axis=1)[0])
    return {"iforest": float(s_if), "lof": float(s_lof), "autoencoder": s_ae}

def normalize_model_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw model outputs into a 0..1 anomaly risk where 1 is most anomalous.
    Strategy:
    - for iforest: decision_function higher -> normal, so invert and scale via logistic-ish transform
    - lof: higher -> more normal in our usage, invert similarly
    - autoencoder: higher reconstruction error -> more anomalous -> scale (use threshold)
    """
    s_if = raw_scores["iforest"]
    s_lof = raw_scores["lof"]
    s_ae = raw_scores["autoencoder"]

    # iforest: typical decision_function range depends on data; use a robust transform:
    # map using sigmoid centered at SCORE_THRESHOLD_MED
    def inv_sig(x, center=SCORE_THRESHOLD_MED, scale=0.1):
        return 1.0 / (1.0 + np.exp((x - center) / scale))
    risk_if = float(inv_sig(s_if))

    # LOF: negative_outlier_factor is negative; we invert similarly
    risk_lof = float(inv_sig(s_lof))

    # AE: normalize by ae_threshold
    risk_ae = float(min(max(s_ae / (ae_threshold + 1e-9), 0.0), 1.0))

    return {"iforest": risk_if, "lof": risk_lof, "ae": risk_ae}

def evaluate_rules(tx: TransactionIn) -> Dict[str, Any]:
    score = 0.0
    flags = []
    # rule 1: large amount
    if tx.amount > 1000:
        score += 0.6
        flags.append("large_amount")
    # rule 2: card unusual location
    if tx.channel and tx.channel.lower() == "card" and tx.location and tx.location.lower() not in ("harare", "bulawayo"):
        score += 0.3
        flags.append("card_unusual_location")
    # rule 3: odd hour
    if tx.hour is not None and tx.hour < 5:
        score += 0.2
        flags.append("odd_hour")
    if score > 1.0:
        score = 1.0
    return {"rule_score": float(score), "rule_flags": flags}

def aggregate_scores(normalized_model_risks: Dict[str, float], deviation_score: float, rule_score: float, amount: float) -> Dict[str, Any]:
    # weights: rule (strong), ensemble (iforest+lof+ae), deviation small
    model_risk = 0.4 * normalized_model_risks["iforest"] + 0.3 * normalized_model_risks["lof"] + 0.3 * normalized_model_risks["ae"]
    agg = 0.45 * rule_score + 0.4 * model_risk + 0.15 * min(max(deviation_score / 2.0, 0.0), 1.0)

    # force medium if rule strong
    if rule_score >= 0.6 and agg < 0.4:
        agg = 0.5
    # hard high override for extreme amounts
    if amount >= AMOUNT_HIGH_RISK:
        agg = max(agg, 0.85)

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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            tx.transaction_id,
            payload.get("anomaly_score_iforest"),
            payload.get("anomaly_score_lof"),
            payload.get("anomaly_score_ae"),
            payload.get("deviation_score"),
            payload.get("rule_score"),
            payload.get("aggregated_score"),
            payload.get("risk_level"),
            MODEL_VERSION
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.exception("Persist failed: {}", e)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}

@app.post("/score", response_model=ScoreOut)
def score_transaction(tx: TransactionIn):
    try:
        # fetch customer and merchant history
        cust_hist, merch_hist = compute_histories(tx)
        df_feat = prepare_features_for_model(tx, cust_hist, merch_hist)
        raw_scores = model_scores_from_bundle(df_feat)
        norm = normalize_model_scores(raw_scores)
        prof = {"deviation_score": 0.0}
        # compute deviation based on cust_hist
        try:
            cust_avg = cust_hist.get(str(tx.customer_id), {}).get("avg", 0.0)
            if cust_avg > 0:
                prof["deviation_score"] = (tx.amount - cust_avg) / (cust_avg + 1e-9)
        except Exception:
            prof["deviation_score"] = 0.0

        rules = evaluate_rules(tx)
        agg = aggregate_scores(norm, prof["deviation_score"], rules["rule_score"], tx.amount)

        out = {
            "transaction_id": tx.transaction_id,
            "anomaly_score_iforest": raw_scores["iforest"],
            "anomaly_score_lof": raw_scores["lof"],
            "anomaly_score_ae": raw_scores["autoencoder"],
            "deviation_score": prof["deviation_score"],
            "rule_score": rules["rule_score"],
            "aggregated_score": agg["aggregated_score"],
            "risk_level": agg["risk_level"],
            "model_version": MODEL_VERSION
        }

        # persist
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
