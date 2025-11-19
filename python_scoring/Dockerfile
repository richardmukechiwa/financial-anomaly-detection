# Use Python 3.11 slim for best compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install all dependencies in one step
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY models ./models

# Set environment variable for model path
ENV MODEL_PATH=/app/models/ensemble_model_v1.pkl

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
