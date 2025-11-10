# src/train.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import joblib
from preprocessing import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load
df = pd.read_csv('data/transactions.csv')

# choose features
numeric_cols = ['amount', 'some_numeric_feature']
categorical_cols = ['merchant_id', 'customer_id']

X = df[numeric_cols + categorical_cols]
# For unsupervised IF, labels may be absent. But if you have labeled anomalies, keep for eval.
y = df.get('is_anomaly')  # optional

# build pipeline
preproc = build_pipeline(numeric_cols, categorical_cols)
model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)

pipe = Pipeline([
    ('preproc', preproc),
    ('iforest', model)
])

# fit
pipe.fit(X)

# optional evaluation: use decision_function and a labeled test set if available
if y is not None:
    preds = pipe.predict(X)
    print(classification_report(y, preds))

# persist
joblib.dump(pipe, 'models/iforest_model_v1.pkl', compress=3)
print('saved models/iforest_model_v1.pkl')
