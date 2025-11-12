.PHONY: help up down build logs clean clean-images clean-volumes status init-db db-shell backend-shell frontend-shell restart health backup-db restore-db reset-db-data setup test test-backend test-backend-file test-query-engine test-document-processor test-frontend docs info

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

up: ## Start all services
	@echo "Starting all services..."
	docker compose up -d
	@echo "Services started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

down: ## Stop all services
	@echo "Stopping all services..."
	docker compose down
	@echo "Services stopped!"

build: ## Build all images
	@echo "Building all images..."
	docker compose build
	@echo "Images built successfully!"

build-backend: ## Build backend image
	docker compose build backend

build-frontend: ## Build frontend image
	docker compose build frontend

logs: ## Show logs for all services
	@echo "Showing logs for all services..."
	docker compose logs -f

logs-backend: ## Show backend logs
	@echo "Showing backend logs..."
	docker compose logs -f backend

logs-frontend: ## Show frontend logs
	@echo "Showing frontend logs..."
	docker compose logs -f frontend

logs-postgres: ## Show PostgreSQL logs
	@echo "Showing PostgreSQL logs..."
	docker compose logs -f postgres

logs-redis: ## Show Redis logs
	@echo "Showing Redis logs..."
	docker compose logs -f redis

logs-celery: ## Show Celery worker logs
	@echo "Showing Celery worker logs..."
	docker compose logs -f celery_worker

status: ## Show status of all containers
	@echo "Container Status:"
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" || echo "No services running"

init-db: ## Initialize database
	@echo "Initializing database..."
	docker compose exec backend python -m app.db.init_db

db-shell: ## Connect to PostgreSQL shell
	@echo "Connecting to PostgreSQL..."
	docker compose exec postgres psql -U funduser -d funddb

backend-shell: ## Open shell in backend container
	@echo "Opening shell in backend container..."
	docker compose exec backend /bin/bash

frontend-shell: ## Open shell in frontend container
	@echo "Opening shell in frontend container..."
	docker compose exec frontend /bin/sh

restart: ## Restart all services
	@echo "Restarting all services..."
	docker compose restart

restart-backend: ## Restart backend service
	@echo "Restarting backend..."
	docker compose restart backend

restart-frontend: ## Restart frontend service
	@echo "Restarting frontend..."
	docker compose restart frontend

restart-celery: ## Restart Celery worker service
	@echo "Restarting Celery worker..."
	docker compose restart celery_worker

health: ## Check service health
	@echo "Checking service health..."
	@docker compose ps --format "table {{.Name}}\t{{.Status}}" | grep -E "(backend|frontend|postgres|redis|celery)" || echo "Services not running"
	@echo ""
	@echo "Backend health check:"
	@curl -s http://localhost:8000/health 2>/dev/null || echo "Backend not responding"

backup-db: ## Backup database to file
	@echo "Backing up database..."
	docker compose exec -T postgres pg_dump -U funduser funddb > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backup created: backup_$(shell date +%Y%m%d_%H%M%S).sql"

restore-db: ## Restore database from backup file (usage: make restore-db FILE=backup.sql)
	@if [ -z "$(FILE)" ]; then echo "Error: FILE is required. Usage: make restore-db FILE=backup.sql"; exit 1; fi
	@echo "Restoring database from $(FILE)..."
	docker compose exec -T postgres psql -U funduser funddb < $(FILE)
	@echo "Database restored from $(FILE)"

reset-db-data: ## Clear all database data (keeps tables and structure)
	@echo "Clearing all database data while preserving table structure..."
	docker compose exec postgres psql -U funduser -d funddb -c "TRUNCATE TABLE funds, capital_calls, distributions, adjustments, documents, document_embeddings RESTART IDENTITY CASCADE;"
	@echo "Database data cleared! Tables are now empty."

clean: ## Stop all services and clean containers
	@echo "Stopping all services..."
	docker compose down
	@echo "Services cleaned up!"

clean-images: ## Remove unused images
	@echo "Removing unused images..."
	docker image prune -f
	@echo "Unused images cleaned up!"

clean-volumes: ## Remove unused volumes (WARNING: destroys data)
	@echo "WARNING: This will destroy all volume data!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker volume prune -f; \
		echo "Volumes cleaned up!"; \
	else \
		echo "Operation cancelled."; \
	fi

setup: ## Initial project setup
	@echo "Setting up project..."
	@cp .env.example .env
	@echo "Please edit .env file and add your OPENAI_API_KEY"
	@echo "Then run 'make up' to start the environment"

test: ## Run all tests
	@echo "Running backend tests..."
	docker compose exec backend pytest app/services/test/ -v --cov=app
	@echo "Running frontend tests..."
	docker compose exec frontend npm test

test-backend: ## Run backend tests
	@echo "Running backend tests..."
	docker compose exec backend pytest app/services/test/ -v --cov=app

test-backend-file: ## Run specific backend test file (usage: make test-backend-file FILE=test_query_engine.py)
	@if [ -z "$(FILE)" ]; then echo "Error: FILE is required. Usage: make test-backend-file FILE=test_query_engine.py"; exit 1; fi
	@echo "Running backend test file: $(FILE)"
	docker compose exec backend pytest app/services/test/$(FILE) -v

test-query-engine: ## Run query engine tests
	@echo "Running query engine tests..."
	docker compose exec backend pytest backend/app/services/test/test_query_engine.py -v

test-document-processor: ## Run document processor tests
	@echo "Running document processor tests..."
	docker compose exec backend pytest backend/app/services/test/test_document_processor.py -v

test-frontend: ## Run frontend tests
	@echo "Running frontend tests..."
	docker compose exec frontend npm test

docs: ## Open API documentation
	@echo "Opening API documentation..."
	@echo "Please start the backend service first with: make up"
	@echo "Then visit: http://localhost:8000/docs"

info: ## Show environment information
	@echo "=== Environment Information ==="
	@echo "Current directory: $(PWD)"
	@echo "Docker version: $(shell docker --version)"
	@echo "Docker Compose version: $(shell docker compose version)"
	@echo ""
	@echo "=== Service URLs ==="
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "PostgreSQL: localhost:5432 (user: funduser, password: fundpass)"
	@echo "Redis: localhost:6379"
	@echo ""
	@echo "=== Running Services ==="
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No services running"