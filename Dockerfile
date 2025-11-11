# Dockerfile - builds FastAPI scoring service
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/ /app
# Copy models if building locally. When using docker-compose, mount models folder.
COPY models/ /app/models/

EXPOSE 8000

ENV MODEL_PATH=/app/models/iforest_model_v1.pkl
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
