# src/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import joblib
import pandas as pd
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Anomaly Scoring API")

# ---------------------------
# Static files / favicon
# ---------------------------
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("src/app/static/favicon.ico")

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
# Request schemas
# ---------------------------
class Transaction(BaseModel):
    transaction_id: str
    amount: float
    merchant_id: str
    customer_id: str
    channel: str = None
    location: str = None
    processed: int = 0

class BatchRequest(BaseModel):
    transactions: List[Transaction]

# ---------------------------
# Health endpoint
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

# ---------------------------
# Scoring endpoint
# ---------------------------
@app.post("/score")
def score_batch(req: BatchRequest) -> Dict[str, List[Dict]]:
    if not req.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    df = pd.DataFrame([t.dict() for t in req.transactions])

    # Features expected by the pipeline (must match train.py)
    expected_features = ['amount', 'merchant_id', 'customer_id', 'channel', 'location', 'processed']
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    try:
        # Pass dataframe directly to trained pipeline
        preds = model.predict(df[expected_features])       # -1 = anomaly, 1 = normal
        scores = model.decision_function(df[expected_features])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    # Scale scores to 0-1 (higher = more anomalous)
    inv = -np.asarray(scores)
    scaled = (inv - np.min(inv)) / np.ptp(inv) if np.ptp(inv) != 0 else np.zeros_like(inv)

    # Prepare results
    results = []
    for i, row in df.iterrows():
        is_anomaly = int(preds[i] == -1)
        anomaly_score = float(scaled[i])
        reason = []

        # Optional human-readable explanations
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

