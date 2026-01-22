# MLflow Quick Setup (local, minimal)

This guide shows how to run a minimal MLflow stack locally for development/testing.
It uses Postgres as the tracking backend and MinIO (S3-compatible) for artifacts.

Files provided:
- `docker-compose.mlflow.yml` - docker-compose config that brings up Postgres, MinIO, and a simple MLflow server

Start the stack:

1. Start docker compose

   docker-compose -f docker-compose.mlflow.yml up -d

2. Confirm services:
   - MLflow UI: http://localhost:5000
   - MinIO console: http://localhost:9000
     - Access key: minioadmin
     - Secret key: minioadmin

3. Set environment variables for clients/workers

   export MLFLOW_TRACKING_URI=http://localhost:5000
   export AWS_ACCESS_KEY_ID=minioadmin
   export AWS_SECRET_ACCESS_KEY=minioadmin
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

4. Use as normal in your code (example):

   import mlflow
   mlflow.set_experiment('my-exp')
   with mlflow.start_run():
       mlflow.log_param('alpha', 0.1)
       mlflow.log_artifact('outputs/workbench_plots/fusion_demo_hcrl_sa.png')

Notes:
- This is a minimal dev setup. For production use you should:
  - Use managed/Postgres + secure MinIO or an S3 provider
  - Place MLflow behind an authenticated reverse-proxy (Nginx + OAuth/OIDC)
  - Use TLS/HTTPS for the MLflow UI

If you'd like, I can add a `make mlflow-up` / `make mlflow-down` in the repo Makefile and a small health-check script.
