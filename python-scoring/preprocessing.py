# preprocessing.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List

def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]):
    """
    Returns a ColumnTransformer instance.
    Ensure this returns the transformer object (not a tuple).
    """
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ], remainder='drop')
    return preproc
