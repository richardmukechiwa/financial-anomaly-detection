from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model once at startup
model = joblib.load("/app/models/iforest_model_v1.pkl")

app = FastAPI()

# Define transaction schema
class Transaction(BaseModel):
    transaction_id: int
    amount: float
    merchant_id: int
    customer_id: int
    channel: str
    location: str
    processed: int = 0

@app.get("/health")
def health():
    return {"status": "ok", "model_path": "/app/models/iforest_model_v1.pkl"}

@app.post("/score")
def score(transaction: Transaction):
    # Convert transaction to model input
    # Adjust features to match training
    features = np.array([[transaction.amount, transaction.merchant_id, transaction.customer_id]])
    
    # Predict anomaly (1 = normal, -1 = anomaly for IsolationForest)
    prediction = model.predict(features)[0]
    
    return {"transaction_id": transaction.transaction_id, "anomaly": int(prediction)}
