# Makefile - Simple wrapper for KD-GAT development tasks
#
# This project uses 'just' (https://github.com/casey/just) as the primary task runner.
# This Makefile provides basic compatibility and MLflow docker-compose integration.
#
# INSTALLATION:
#   Install just: cargo install just  (or via package manager)
#   See: https://just.systems/man/en/chapter_4.html
#
# USAGE:
#   make help          - Show available tasks
#   just --list        - Show all just tasks
#   just <task>        - Run a just task
#

.PHONY: help
.DEFAULT_GOAL := help

# ============================================================================
# Help and Documentation
# ============================================================================

help: ## Show this help message
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║          KD-GAT Development Task Runner                   ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "This project uses 'just' for task automation."
	@echo ""
	@echo "📦 INSTALLATION:"
	@echo "   cargo install just              (if you have Rust/Cargo)"
	@echo "   brew install just               (macOS)"
	@echo "   sudo apt install just           (Ubuntu/Debian)"
	@echo "   conda install -c conda-forge just  (Conda)"
	@echo ""
	@echo "📖 USAGE:"
	@echo "   just --list                     List all available tasks"
	@echo "   just <task>                     Run a specific task"
	@echo "   just                            Show just help"
	@echo ""
	@echo "🔧 COMMON TASKS (via just):"
	@echo "   just check-env                  Check environment setup"
	@echo "   just smoke-synthetic            Quick smoke test (no data needed)"
	@echo "   just mlflow                     Start MLflow UI"
	@echo "   just pre-submit                 Pre-submission checks"
	@echo ""
	@echo "🐳 DOCKER TASKS (via make):"
	@echo "   make mlflow-up                  Start MLflow in Docker"
	@echo "   make mlflow-down                Stop MLflow Docker"
	@echo "   make mlflow-check               Check MLflow status"
	@echo ""
	@echo "📚 DOCUMENTATION:"
	@echo "   docs/GETTING_STARTED.md         Setup guide"
	@echo "   docs/PIPELINE_USER_GUIDE.md     Usage guide"
	@echo "   Justfile                        Full task definitions"
	@echo ""
	@if command -v just >/dev/null 2>&1; then \
		echo "✓ 'just' is installed - you're good to go!"; \
		echo ""; \
		echo "Run 'just --list' to see all available tasks."; \
	else \
		echo "⚠️  'just' is not installed."; \
		echo "   Install it to use the full task automation."; \
		echo "   See installation instructions above."; \
	fi
	@echo ""

# ============================================================================
# MLflow Docker Integration
# ============================================================================

mlflow-up: ## Start MLflow tracking server via Docker
	@echo "🐳 Starting MLflow stack via docker-compose"
	docker-compose -f docker-compose.mlflow.yml up -d
	@echo "✓ MLflow UI available at http://localhost:5000"

mlflow-down: ## Stop MLflow Docker containers
	@echo "🐳 Stopping MLflow stack"
	docker-compose -f docker-compose.mlflow.yml down

mlflow-check: ## Check if MLflow UI is accessible
	@echo "Checking MLflow UI (http://localhost:5000)"
	@curl --silent --fail http://localhost:5000 >/dev/null 2>&1 && \
		echo "✓ MLflow UI is running" || \
		echo "✗ MLflow UI is not reachable"

# ============================================================================
# Wrapper Tasks (delegate to just if installed)
# ============================================================================

check-env: ## Check environment (delegates to just)
	@command -v just >/dev/null 2>&1 && just check-env || \
		(echo "Error: 'just' not found. Install it first." && exit 1)

smoke: ## Run smoke test (delegates to just)
	@command -v just >/dev/null 2>&1 && just smoke-synthetic || \
		(echo "Error: 'just' not found. Install it first." && exit 1)

test: ## Run pytest (basic command, not delegated)
	pytest tests/ -v

# ============================================================================
# Installation Helpers
# ============================================================================

install-conda: ## Create conda environment (delegates to just)
	@command -v just >/dev/null 2>&1 && just install-conda || \
		conda env create -f environment.yml -n gnn-gpu || \
		conda env update -f environment.yml -n gnn-gpu

install-just: ## Install just command (tries multiple methods)
	@echo "Attempting to install 'just'..."
	@if command -v cargo >/dev/null 2>&1; then \
		echo "Installing via cargo..."; \
		cargo install just; \
	elif command -v brew >/dev/null 2>&1; then \
		echo "Installing via brew..."; \
		brew install just; \
	elif command -v apt >/dev/null 2>&1; then \
		echo "Installing via apt..."; \
		sudo apt install just; \
	elif command -v conda >/dev/null 2>&1; then \
		echo "Installing via conda..."; \
		conda install -c conda-forge just; \
	else \
		echo "No package manager found."; \
		echo "Please install just manually:"; \
		echo "  https://just.systems/man/en/chapter_4.html"; \
		exit 1; \
	fi
