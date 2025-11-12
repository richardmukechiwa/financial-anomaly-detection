# src/app/main.py (excerpt)
import pandas as pd
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import joblib
import numpy as np

app = FastAPI(title="Anomaly Scoring API")

# ---------------------------
# Model loading
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(BASE_DIR, "..", "..", "models", "iforest_model_v1.pkl")
MODEL_PATH = os.path.normpath(MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# ---------------------------
# Request schema
# ---------------------------
class BatchRequest(BaseModel):
    transaction_ids: List[int]  # optional: fetch by IDs

# ---------------------------
# MySQL connection helper
# ---------------------------
def get_mysql_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        database=os.environ.get("MYSQL_DB", "crm")
    )

# ---------------------------
# Scoring endpoint
# ---------------------------
@app.post("/score")
def score_batch(req: BatchRequest) -> Dict[str, List[Dict]]:
    try:
        conn = get_mysql_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MySQL connection error: {e}")

    # Fetch transactions from MySQL
    try:
        ids_tuple = tuple(req.transaction_ids)
        query = f"""
        SELECT transaction_id, amount, merchant_id, customer_id, channel, location, processed
        FROM transactions
        WHERE transaction_id IN ({','.join(['%s']*len(ids_tuple))})
        """
        df = pd.read_sql(query, conn, params=ids_tuple)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MySQL fetch error: {e}")
    finally:
        conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail="No transactions found")

    # ---------------------------
    # Type conversion
    # ---------------------------
    expected_types = {
        'amount': float,
        'merchant_id': str,
        'customer_id': str,
        'channel': str,
        'location': str,
        'processed': int
    }

    for col, typ in expected_types.items():
        if col in df.columns:
            df[col] = df[col].astype(typ)

    # ---------------------------
    # Model prediction
    # ---------------------------
    expected_features = list(expected_types.keys())
    try:
        preds = model.predict(df[expected_features])
        scores = model.decision_function(df[expected_features])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    # Scale scores 0-1 (higher = more anomalous)
    inv = -np.asarray(scores)
    scaled = (inv - np.min(inv)) / np.ptp(inv) if np.ptp(inv) != 0 else np.zeros_like(inv)

    # Prepare results
    results = []
    for i, row in df.iterrows():
        is_anomaly = int(preds[i] == -1)
        anomaly_score = float(scaled[i])
        reason = []

        if row['amount'] > df['amount'].mean() + 3 * df['amount'].std():
            reason.append("amount >> historical mean")
        if row['amount'] > 10000:
            reason.append("high absolute amount")
        reason_text = "; ".join(reason) if reason else "model pattern match"

        results.append({
            "transaction_id": row['transaction_id'],
            "anomaly_label": is_anomaly,
            "anomaly_score": anomaly_score,
            "confidence_level": round(1.0 - anomaly_score, 3),
            "reason": reason_text
        })

    return {"results": results}
