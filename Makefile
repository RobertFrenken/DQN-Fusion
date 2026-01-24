# Helper make targets for MLflow dev stack
.PHONY: mlflow-up mlflow-down mlflow-check

mlflow-up:
	@echo "Starting MLflow stack via docker-compose"
	docker-compose -f docker-compose.mlflow.yml up -d

mlflow-down:
	@echo "Stopping MLflow stack"
	docker-compose -f docker-compose.mlflow.yml down

mlflow-check:
	@echo "Checking MLflow UI (http://localhost:5000)"
	@curl --silent --fail http://localhost:5000 || echo "MLflow UI not reachable"
