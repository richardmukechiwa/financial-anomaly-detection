# src/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Anomaly Scoring API")

# Model path relative to project root when containerized
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/iforest_model_v1.pkl")

# Load model at import time so it is available for all requests
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # raise at startup to make failure visible in logs
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    merchant_id: str
    customer_id: str

class BatchRequest(BaseModel):
    transactions: List[Transaction]

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post("/score")
def score_batch(req: BatchRequest):
    if not req.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    df = pd.DataFrame([t.dict() for t in req.transactions])
    # Ensure same features used in training
    expected_features = ['amount', 'merchant_id', 'customer_id']
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Use full pipeline if persisted as Pipeline(preproc + model)
    # If model is the pipeline, call predict and decision_function directly on raw X
    X = df[expected_features]

    # For pipeline objects, calling decision_function may require accessing the final estimator.
    try:
        # If persisted object is a Pipeline with last step 'iforest', use it
        if hasattr(model, 'predict') and hasattr(model, 'decision_function'):
            preds = model.predict(X)              # -1 anomaly, 1 normal
            scores = model.decision_function(X)   # higher means more normal
        else:
            # if model is a wrapper, attempt to access 'iforest' step
            preds = model['iforest'].predict(model['preproc'].transform(X))
            scores = model['iforest'].decision_function(model['preproc'].transform(X))
    except Exception as e:
        # If transform methods required, try transforming first
        try:
            X_proc = model.named_steps['preproc'].transform(X)
            preds = model.named_steps['iforest'].predict(X_proc)
            scores = model.named_steps['iforest'].decision_function(X_proc)
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Scoring error: {e}; fallback error: {e2}")

    # Convert score to anomaly score between 0 and 1 where higher means more anomalous.
    # IsolationForest.decision_function: higher means more normal. We invert and scale.
    scores = np.asarray(scores)
    inv = -scores                      # higher = more anomalous
    # Min-max scale
    if np.ptp(inv) == 0:
        scaled = np.zeros_like(inv)
    else:
        scaled = (inv - np.min(inv)) / np.ptp(inv)

    results = []
    for i, row in df.iterrows():
        is_anomaly = int(preds[i] == -1)
        anomaly_score = float(scaled[i])
        # Simple reason heuristic
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
