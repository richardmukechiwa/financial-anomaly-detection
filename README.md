# financial-anomaly-detection

# AI Anomaly - quick start

1. Put historical transactions CSV at `data/transactions.csv`.
   Required fields: transaction_id, amount, merchant_id, customer_id
   Optional: is_anomaly (0/1) for evaluation.

2. Train model locally:
   $ cd src
   $ python train.py --csv ../data/transactions.csv

   This writes models/iforest_model_v1.pkl

3. Run service with Docker Compose:
   $ docker-compose up --build

   FastAPI will be available at http://localhost:8000
   Health: GET http://localhost:8000/health
   Score: POST http://localhost:8000/score

4. Import `n8n-workflow.json` into n8n. Update HTTP Request node URL if needed.

5. Production notes:
   - Build CI to produce model artifact and push to artifact storage.
   - Deploy scoring service behind HTTPS and an API gateway.
   - Implement auth on scoring endpoint.

