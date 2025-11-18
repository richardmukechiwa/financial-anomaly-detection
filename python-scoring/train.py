# train_pipeline.py
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from preprocessing import build_pipeline
from model_wrapper import save_ensemble
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
AUTOENC_DIR = os.path.join(MODEL_DIR, 'autoencoder')

def build_autoencoder(input_dim, latent_dim=32):
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(encoder_input)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    autoencoder = models.Model(encoder_input, decoded)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse')
    return autoencoder

def main(csv_path, model_name='ensemble_model_v1.pkl', contamination=0.01, random_state=42):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(AUTOENC_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV empty")
    # parse timestamp and create time features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour

    # Basic engineered features for IDs: avg amount per customer/merchant (90 days window)
    # For training, compute simple aggregate per customer/merchant from the CSV itself.
    cust_avg = df.groupby('customer_id')['amount'].transform('mean')
    cust_count = df.groupby('customer_id')['amount'].transform('count')
    df['cust_avg_amount'] = cust_avg.fillna(0)
    df['cust_txn_count'] = cust_count.fillna(0)

    merch_avg = df.groupby('merchant_id')['amount'].transform('mean')
    df['merch_avg_amount'] = merch_avg.fillna(0)

    # Features
    numeric_cols = ['amount', 'year', 'month', 'day_of_week', 'hour', 'cust_avg_amount', 'cust_txn_count', 'merch_avg_amount']
    categorical_cols = ['channel', 'location']  # don't OHE high-cardinality ids

    # drop rows with missing amount
    df = df.dropna(subset=['amount'])

    X_raw = df[numeric_cols + categorical_cols].copy()

    preproc = build_pipeline(numeric_cols, categorical_cols)
    # Fit preproc
    preproc.fit(X_raw)
    X = preproc.transform(X_raw)
    # force numpy array
    X = np.asarray(X)

    # Isolation Forest
    iforest = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    iforest.fit(X)

    # LOF (fit_predict style — we use fit for neighborhood info)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(X)  # novelty=True allows calling predict on new data

    # Autoencoder
    input_dim = X.shape[1]
    ae = build_autoencoder(input_dim, latent_dim=min(32, input_dim//2 if input_dim//2>1 else 2))
    # split for training autoencoder to avoid overfitting
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=random_state)
    ae.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_val, X_val), verbose=1)
    # compute AE reconstruction errors on validation for threshold
    recon = ae.predict(X_val, verbose=0)
    mse = np.mean(np.square(X_val - recon), axis=1)
    ae_threshold = float(np.percentile(mse, 97.5))  # choose 97.5 percentile

    # Save autoencoder model separately
    ae_path = os.path.join(AUTOENC_DIR, 'autoencoder_model')
    ae.save(ae_path, save_format='tf')
    # Save pipeline and iforest and lof bundled. For AE we store path and threshold
    bundle = {
        "preproc": preproc,
        "iforest": iforest,
        "lof": lof,
        "autoencoder_path": ae_path,
        "ae_threshold": ae_threshold
    }
    out_path = os.path.join(MODEL_DIR, model_name)
    joblib.dump(bundle, out_path, compress=3)
    print(f"Saved ensemble at {out_path}, autoencoder at {ae_path} with threshold {ae_threshold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-name", default="ensemble_model_v1.pkl")
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()
    main(args.csv, model_name=args.model_name, contamination=args.contamination)
