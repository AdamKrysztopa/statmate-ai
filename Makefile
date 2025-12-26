.PHONY: help install dev prod api ui db-init db-seed clean test lint format kill list-models frontend-install frontend-dev frontend-build frontend-preview node-install

# Colors for terminal output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
RESET  := \033[0m

# Local Node.js (for frontend) - installs into ~/.local (no sudo)
NODE_VERSION ?= 20.18.1
NODE_ARCH    := $(shell uname -m)
NODE_DIST    := $(if $(filter x86_64,$(NODE_ARCH)),x64,$(if $(filter aarch64 arm64,$(NODE_ARCH)),arm64,$(NODE_ARCH)))
NODE_HOME    := $(HOME)/.local/node-v$(NODE_VERSION)-linux-$(NODE_DIST)
NODE_BIN     := $(NODE_HOME)/bin

help: ## Show this help message
	@echo '$(GREEN)StatmateAI - Makefile Commands$(RESET)'
	@echo ''
	@echo 'Usage:'
	@echo '  make $(YELLOW)<target>$(RESET)'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(RESET) %s\n", $$1, $$2}'

# =============================================================================
# Installation & Setup
# =============================================================================

install: ## Install all dependencies
	@echo '$(GREEN)Installing dependencies...$(RESET)'
	uv sync
	@echo '$(GREEN)✓ Installation complete!$(RESET)'

install-dev: ## Install with development dependencies
	@echo '$(GREEN)Installing with dev dependencies...$(RESET)'
	uv sync --all-extras
	@echo '$(GREEN)✓ Dev installation complete!$(RESET)'

setup-env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		echo '$(YELLOW)Creating .env file...$(RESET)'; \
		cp .env.example .env; \
		echo '$(GREEN)✓ .env created! Please edit it with your API keys.$(RESET)'; \
	else \
		echo '$(RED)✗ .env already exists. Not overwriting.$(RESET)'; \
	fi

# =============================================================================
# Database
# =============================================================================

db-init: ## Initialize database
	@echo '$(GREEN)Initializing database...$(RESET)'
	@export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python scripts/init_db.py
	@echo '$(GREEN)✓ Database initialized!$(RESET)'

db-seed: ## Seed database with sample data
	@echo '$(GREEN)Seeding database...$(RESET)'
	@export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python scripts/seed_db.py
	@echo '$(GREEN)✓ Database seeded!$(RESET)'

db-reset: ## Reset database (WARNING: deletes all data!)
	@echo '$(RED)⚠️  Resetting database (all data will be lost)...$(RESET)'
	rm -f database/statmate.db
	@export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python scripts/init_db.py --seed
	@echo '$(GREEN)✓ Database reset complete!$(RESET)'

# =============================================================================
# Development Mode (Uses .env API keys)
# =============================================================================

dev: ## Run in DEVELOPMENT mode (API keys from .env)
	@echo '$(GREEN)╔════════════════════════════════════════════════════╗$(RESET)'
	@echo '$(GREEN)║      DEVELOPMENT MODE - Using .env API Keys        ║$(RESET)'
	@echo '$(GREEN)╚════════════════════════════════════════════════════╝$(RESET)'
	@if [ ! -f .env ]; then \
		echo '$(RED)✗ .env file not found!$(RESET)'; \
		echo '$(YELLOW)Run: make setup-env$(RESET)'; \
		exit 1; \
	fi
	@if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null && ! grep -q "ANTHROPIC_API_KEY=sk-ant-" .env 2>/dev/null && ! grep -q "OLLAMA_ENABLED=true" .env 2>/dev/null; then \
		echo '$(RED)✗ No API keys configured in .env!$(RESET)'; \
		echo '$(YELLOW)Please edit .env and add at least one API key$(RESET)'; \
		exit 1; \
	fi
	@if [ ! -f database/statmate.db ]; then \
		echo '$(YELLOW)Database not found, initializing...$(RESET)'; \
		export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python scripts/init_db.py --seed; \
	fi
	@mkdir -p data/uploads data/results data/logs
	@export ENVIRONMENT=development && bash scripts/run_dev.sh

api: ## Run only the FastAPI backend
	@echo '$(GREEN)Starting FastAPI backend...$(RESET)'
	@export ENVIRONMENT=development PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python statmate/api/main.py

ui: ## Run only the Streamlit UI
	@echo '$(GREEN)Starting Streamlit UI...$(RESET)'
	@export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run streamlit run statmate/ui/app.py

# =============================================================================
# Production Mode (Users provide credentials via UI)
# =============================================================================

prod: ## Run in PRODUCTION mode (users provide their own credentials)
	@echo '$(GREEN)╔════════════════════════════════════════════════════╗$(RESET)'
	@echo '$(GREEN)║   PRODUCTION MODE - User Credentials Required      ║$(RESET)'
	@echo '$(GREEN)╚════════════════════════════════════════════════════╝$(RESET)'
	@echo '$(YELLOW)Users will be prompted to enter their API keys in the UI$(RESET)'
	@if [ ! -f database/statmate.db ]; then \
		echo '$(YELLOW)Database not found, initializing...$(RESET)'; \
		export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python scripts/init_db.py; \
	fi
	@mkdir -p data/uploads data/results data/logs
	@export ENVIRONMENT=production DEBUG=false PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python statmate/api/main.py &
	@sleep 2
	@export ENVIRONMENT=production PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run streamlit run statmate/ui/app.py

prod-api: ## Run API in production mode
	@echo '$(GREEN)Starting API in PRODUCTION mode...$(RESET)'
	@export ENVIRONMENT=production DEBUG=false PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python statmate/api/main.py

prod-ui: ## Run UI in production mode
	@echo '$(GREEN)Starting UI in PRODUCTION mode...$(RESET)'
	@export ENVIRONMENT=production PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run streamlit run statmate/ui/app.py

# =============================================================================
# Testing & Quality
# =============================================================================

test: ## Run tests
	@echo '$(GREEN)Running tests...$(RESET)'
	uv run pytest tests/ -v

test-coverage: ## Run tests with coverage report
	@echo '$(GREEN)Running tests with coverage...$(RESET)'
	uv run pytest tests/ --cov=statmate --cov-report=html --cov-report=term

lint: ## Run linter (ruff)
	@echo '$(GREEN)Running linter...$(RESET)'
	uv run ruff check .

format: ## Format code (ruff)
	@echo '$(GREEN)Formatting code...$(RESET)'
	uv run ruff format .

type-check: ## Run type checker (mypy)
	@echo '$(GREEN)Running type checker...$(RESET)'
	uv run mypy statmate

# =============================================================================
# Frontend (React + Vite)
# =============================================================================

node-install: ## Install local Node.js (no sudo) to ~/.local
	@echo '$(GREEN)Installing Node.js $(NODE_VERSION) for $(NODE_DIST)...$(RESET)'
	@mkdir -p $(HOME)/.local
	@curl -fsSL "https://nodejs.org/dist/v$(NODE_VERSION)/node-v$(NODE_VERSION)-linux-$(NODE_DIST).tar.xz" -o /tmp/node.tar.xz
	@tar -xf /tmp/node.tar.xz -C /tmp
	@rm -f /tmp/node.tar.xz
	@rm -rf $(NODE_HOME)
	@mv /tmp/node-v$(NODE_VERSION)-linux-$(NODE_DIST) $(NODE_HOME)
	@echo 'export PATH=$(NODE_BIN):$$PATH' >> $(HOME)/.profile
	@echo '$(GREEN)✓ Node installed to $(NODE_HOME) (add to PATH if not already)$(RESET)'
	@$(NODE_BIN)/node -v
	@$(NODE_BIN)/npm -v

frontend-install: ## Install frontend deps (npm)
	@echo '$(GREEN)Installing frontend dependencies...$(RESET)'
	@command -v npm >/dev/null 2>&1 || { echo '$(YELLOW)npm not found, installing local Node...$(RESET)'; $(MAKE) node-install; }
	@cd statmate/frontend && PATH=$(NODE_BIN):$$PATH npm install
	@echo '$(GREEN)✓ Frontend deps ready$(RESET)'

frontend-dev: ## Run React dev server (Vite on :3000)
	@cd statmate/frontend && PATH=$(NODE_BIN):$$PATH npm run dev -- --host --port 3000

frontend-build: ## Build production assets
	@cd statmate/frontend && PATH=$(NODE_BIN):$$PATH npm run build

frontend-preview: ## Preview production build locally
	@cd statmate/frontend && PATH=$(NODE_BIN):$$PATH npm run preview -- --host --port 3000

# =============================================================================
# Utilities
# =============================================================================

kill: ## Stop all running StatmateAI processes (API & UI)
	@echo '$(YELLOW)Stopping StatmateAI processes...$(RESET)'
	@PIDS=$$(ps -eo pid=,cmd= -ww | grep -E '[u]vicorn.*main:app|[s]tatmate/api/main.py|[s]treamlit run.*statmate/ui/app.py' | awk '{print $$1}' | paste -sd' ' -); \
	PIDS_CSV=$$(echo "$$PIDS" | tr ' ' ',' | sed 's/^,//;s/,$$//'); \
	if [ -n "$$PIDS" ]; then \
		echo '$(YELLOW)Found processes to stop:$(RESET)'; \
		ps -p $$PIDS_CSV -o pid,cmd --no-headers; \
		kill $$PIDS 2>/dev/null || true; \
		sleep 1; \
	fi; \
	REMAIN=$$(ps -eo pid=,cmd= -ww | grep -E '[u]vicorn.*main:app|[s]tatmate/api/main.py|[s]treamlit run.*statmate/ui/app.py' | awk '{print $$1}' | paste -sd' ' -); \
	REMAIN_CSV=$$(echo "$$REMAIN" | tr ' ' ',' | sed 's/^,//;s/,$$//'); \
	if [ -n "$$REMAIN" ]; then \
		echo '$(RED)⚠️  Some processes may still be running. Force killing...$(RESET)'; \
		ps -p $$REMAIN_CSV -o pid,cmd --no-headers; \
		kill -9 $$REMAIN 2>/dev/null || true; \
	fi; \
	if [ -z "$$PIDS" ] && [ -z "$$REMAIN" ]; then \
		echo '$(GREEN)✓ No StatmateAI processes were running.$(RESET)'; \
	else \
		echo '$(GREEN)✓ All processes stopped!$(RESET)'; \
	fi

clean: ## Clean up temporary files
	@echo '$(YELLOW)Cleaning up...$(RESET)'
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.log' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	@echo '$(GREEN)✓ Cleanup complete!$(RESET)'

clean-all: clean ## Clean everything including database and data
	@echo '$(RED)⚠️  Removing database and data files...$(RESET)'
	rm -rf database/statmate.db data/uploads/* data/results/* data/logs/*
	@echo '$(GREEN)✓ Deep cleanup complete!$(RESET)'

logs: ## Show recent logs
	@echo '$(GREEN)Recent logs:$(RESET)'
	@ls -lt data/logs/*.log 2>/dev/null | head -5 || echo 'No logs found'

list-models: ## List all available models from configured providers
	@echo '$(GREEN)Checking available models from providers...$(RESET)'
	@export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}$$(pwd) && uv run python scripts/list_available_models.py

status: ## Check system status
	@echo '$(GREEN)System Status:$(RESET)'
	@echo ''
	@echo 'Environment:'
	@if [ -f .env ]; then echo '  ✓ .env file exists'; else echo '  ✗ .env file missing'; fi
	@echo ''
	@echo 'Database:'
	@if [ -f database/statmate.db ]; then echo '  ✓ Database exists'; else echo '  ✗ Database not initialized'; fi
	@echo ''
	@echo 'API Keys Configured:'
	@if [ -f .env ]; then \
		grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null && echo '  ✓ OpenAI' || echo '  ✗ OpenAI'; \
		grep -q "ANTHROPIC_API_KEY=sk-ant-" .env 2>/dev/null && echo '  ✓ Anthropic' || echo '  ✗ Anthropic'; \
		grep -q "GOOGLE_API_KEY=.*" .env 2>/dev/null && echo '  ✓ Google/Gemini' || echo '  ✗ Google/Gemini'; \
		grep -q "GROQ_API_KEY=.*" .env 2>/dev/null && echo '  ✓ Groq' || echo '  ✗ Groq'; \
		grep -q "OLLAMA_ENABLED=true" .env 2>/dev/null && echo '  ✓ Ollama (Local)' || echo '  ✗ Ollama'; \
	fi

# =============================================================================
# Docker (Future)
# =============================================================================

docker-build: ## Build Docker image
	@echo '$(GREEN)Building Docker image...$(RESET)'
	docker build -t statmate-ai:latest .

docker-run: ## Run Docker container
	@echo '$(GREEN)Running Docker container...$(RESET)'
	docker run -p 8000:8000 -p 8501:8501 statmate-ai:latest

# =============================================================================
# Quick Start Guide
# =============================================================================

quickstart: ## First-time setup and run
	@echo '$(GREEN)╔════════════════════════════════════════════════════╗$(RESET)'
	@echo '$(GREEN)║           StatmateAI Quick Start Setup             ║$(RESET)'
	@echo '$(GREEN)╚════════════════════════════════════════════════════╝$(RESET)'
	@echo ''
	@echo '$(YELLOW)Step 1/4: Installing dependencies...$(RESET)'
	@make install
	@echo ''
	@echo '$(YELLOW)Step 2/4: Creating .env file...$(RESET)'
	@make setup-env
	@echo ''
	@echo '$(YELLOW)Step 3/4: Initializing database...$(RESET)'
	@make db-init db-seed
	@echo ''
	@echo '$(GREEN)╔════════════════════════════════════════════════════╗$(RESET)'
	@echo '$(GREEN)║                Setup Complete! 🎉                  ║$(RESET)'
	@echo '$(GREEN)╚════════════════════════════════════════════════════╝$(RESET)'
	@echo ''
	@echo '$(YELLOW)Next steps:$(RESET)'
	@echo '  1. Edit .env file and add your API keys'
	@echo '  2. Run: $(GREEN)make dev$(RESET)'
	@echo ''
