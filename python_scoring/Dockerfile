FROM python:3.11-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML API code and models
COPY app.py .
COPY models ./models

# Env
ENV MODEL_PATH=/app/models/ensemble_model_v1.pkl
ENV MODEL_VERSION=v1
ENV SCORE_THRESHOLD_HIGH=-0.3
ENV SCORE_THRESHOLD_MED=-0.1
ENV AMOUNT_HIGH_RISK=10000.0

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
