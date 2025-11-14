# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    gcc \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages if needed
RUN pip install --no-cache-dir mysql-connector-python pandas requests uvicorn fastapi joblib

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Healthcheck using Python requests
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
 CMD python -c "import requests; r=requests.get('http://127.0.0.1:8000/health'); exit(0 if r.status_code==200 else 1)"

# Run the API server
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

