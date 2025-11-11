# src/train.py
import os
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import build_pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def main(csv_path, model_name='iforest_model_v1.pkl', contamination=0.01, random_state=42):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Basic sanity check
    if df.empty:
        raise ValueError("Input CSV is empty")

    # Define features used for the model. Edit to match your dataset.
    numeric_cols = ['amount']                     # add more numeric features if available
    categorical_cols = ['merchant_id', 'customer_id']  # low-cardinality preferred

    features = numeric_cols + categorical_cols
    X = df[features]

    # Build pipeline: preprocessing + model
    preproc = build_pipeline(numeric_cols, categorical_cols)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)

    pipeline = Pipeline([
        ('preproc', preproc),
        ('iforest', model)
    ])

    # Fit
    pipeline.fit(X)

    # Optional: evaluation if labels exist
    if 'is_anomaly' in df.columns:
        y = df['is_anomaly']
        preds = pipeline.predict(X)     # -1 anomaly, 1 normal
        # map preds to 1/0 for metrics
        preds_bin = (preds == -1).astype(int)
        print("Classification report (if labels present):")
        print(classification_report(y, preds_bin))

    # Save
    out_path = os.path.join(MODEL_DIR, model_name)
    joblib.dump(pipeline, out_path, compress=3)
    print(f"Saved model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to transactions CSV")
    parser.add_argument("--model-name", default="iforest_model_v1.pkl")
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()
    main(args.csv, model_name=args.model_name, contamination=args.contamination)

