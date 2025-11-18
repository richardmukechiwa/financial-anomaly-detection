FROM python:3.11-slim

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# System deps for joblib/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package
COPY python_scoring/ python_scoring/

EXPOSE 8000

# Start Uvicorn using module notation
CMD ["uvicorn", "python_scoring.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
