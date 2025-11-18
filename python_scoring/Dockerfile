FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn tensorflow loguru mysql-connector-python

# Copy application code
COPY app.py .
COPY models ./models

# Env
ENV MODEL_PATH=/app/models/ensemble_model_v1.pkl

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
