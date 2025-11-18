# model_wrapper.py
import joblib
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf

class EnsembleModel:
    """
    Holds preprocessor + three models:
    - iforest (pipeline: preproc -> IsolationForest)
    - lof (uses same preproc; LOF has no decision_function in same sense; use negative_outlier_factor_)
    - autoencoder: trained on preprocessed numeric arrays (reconstruction error)
    """
    def __init__(self, preproc, iforest, lof, autoencoder, ae_threshold=None):
        self.preproc = preproc
        self.iforest = iforest
        self.lof = lof
        self.autoencoder = autoencoder
        self.ae_threshold = ae_threshold

    def transform(self, df):
        return self.preproc.transform(df)

    def score_iforest(self, X):
        # IsolationForest.decision_function: higher means more normal
        # convert to anomaly score where lower -> more anomalous => invert
        raw = self.iforest.decision_function(X)  # larger -> normal
        # map to -1..1 or keep raw, conversion to risk later
        return raw

    def score_lof(self, X):
        # LOF: negative_outlier_factor_ : lower = more outlier
        lof_vals = self.lof._decision_function(X) if hasattr(self.lof, "_decision_function") else self.lof.negative_outlier_factor_
        # if using predict then negative_outlier_factor_ is available after fit
        if hasattr(self.lof, "negative_outlier_factor_"):
            raw = -self.lof.negative_outlier_factor_
            return raw
        else:
            # fallback
            return self.lof._decision_function(X)

    def score_autoencoder(self, X):
        # X must be numeric numpy array
        # autoencoder expects shape (n_samples, n_features)
        recon = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.square(X - recon), axis=1)
        return mse

    def decision_scores(self, df):
        """
        Accepts a DataFrame (raw features with the same columns used for preproc).
        Returns dict with raw model outputs.
        """
        X = self.transform(df)
        # Ensure X is 2D numpy
        X_arr = X if isinstance(X, (np.ndarray,)) else X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        s_if = self.score_iforest(X_arr)
        s_lof = self.score_lof(X_arr)
        s_ae = self.score_autoencoder(X_arr)
        return {"iforest": s_if, "lof": s_lof, "autoencoder": s_ae}

def save_ensemble(path: str, preproc, iforest, lof, autoencoder, ae_threshold=None):
    bundle = {
        "preproc": preproc,
        "iforest": iforest,
        "lof": lof,
        "autoencoder": autoencoder,
        "ae_threshold": ae_threshold
    }
    joblib.dump(bundle, path, compress=3)

def load_ensemble(path: str) -> EnsembleModel:
    b = joblib.load(path)
    from sklearn.base import clone
    return EnsembleModel(b["preproc"], b["iforest"], b["lof"], b["autoencoder"], b.get("ae_threshold", None))
