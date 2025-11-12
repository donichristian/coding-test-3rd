.PHONY: help dev dev-build dev-up dev-down prod prod-build prod-up prod-deploy clean clean-images clean-volumes clean-fund-images

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Development commands
dev: ## Start development environment (default)
	@echo "Starting development environment..."
	docker compose --profile dev up -d postgres redis backend frontend celery_worker
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

dev-build: ## Build development images
	@echo "Building development images..."
	docker compose build
	@echo "Cleaning up dangling images..."
	docker image prune -f
	@echo "Development images built successfully!"

dev-build-backend: ## Build only backend development image
	docker compose build backend

dev-build-frontend: ## Build only frontend development image
	docker compose build frontend

dev-up: ## Start development services
	@echo "Starting development services..."
	docker compose up -d
	@echo "Development services started!"

dev-down: ## Stop development services
	@echo "Stopping development services..."
	docker compose down
	@echo "Development services stopped!"

stop: ## Gracefully stop all fund containers
	@echo "Gracefully stopping all fund containers..."
	@containers=$$(docker ps -q --filter "name=fund-"); \
	if [ -n "$$containers" ]; then \
		echo "Stopping containers..."; \
		docker stop $$containers; \
		echo "Removing containers..."; \
		docker rm $$containers; \
		echo "All fund containers stopped and removed."; \
	else \
		echo "No fund containers running."; \
	fi

stop-dev: ## Stop only development containers
	@echo "Stopping development containers..."
	@containers=$$(docker ps -q --filter "name=fund-" | grep -v "prod"); \
	if [ -n "$$containers" ]; then \
		docker stop $$containers && docker rm $$containers; \
		echo "Development containers stopped."; \
	else \
		echo "No development containers running."; \
	fi

stop-prod: ## Stop only production containers
	@echo "Stopping production containers..."
	@containers=$$(docker ps -q --filter "name=fund-prod"); \
	if [ -n "$$containers" ]; then \
		docker stop $$containers && docker rm $$containers; \
		echo "Production containers stopped."; \
	else \
		echo "No production containers running."; \
	fi

# Production commands
prod: ## Start production environment
	@echo "Starting production environment..."
	docker compose --profile prod up -d postgres redis backend-prod frontend-prod celery_worker_prod
	@echo "Production environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"

prod-build: ## Build production images with cache-busting
	@echo "Building production images..."
	docker compose --profile prod build --no-cache
	@echo "Cleaning up dangling images..."
	docker image prune -f
	@echo "Production images built successfully!"

prod-build-backend: ## Build only backend production image with cache-busting
	docker compose --profile prod build backend --no-cache

prod-build-frontend: ## Build only frontend production image with cache-busting
	docker compose --profile prod build frontend --no-cache

prod-up: ## Start production services
	@echo "Starting production services..."
	docker compose --profile prod up -d
	@echo "Production services started!"

prod-deploy: ## Deploy production (build + up)
	@echo "Building production images..."
	docker compose --profile prod build --no-cache
	@echo "Starting production services..."
	docker compose --profile prod up -d
	@echo "Cleaning up dangling images and unused images..."
	docker image prune -f
	docker image prune -a --filter type=exec -f
	@echo "Production deployment complete!"

# Tagging commands
tag-prod: ## Tag production images with version (usage: make tag-prod TAG=v1.0.0)
	@if [ -z "$(TAG)" ]; then echo "Error: TAG is required. Usage: make tag-prod TAG=v1.0.0"; exit 1; fi
	docker tag fund-backend:latest fund-backend:$(TAG)
	docker tag fund-frontend:latest fund-frontend:$(TAG)
	@echo "Tagged images with $(TAG)"

push-prod: ## Push tagged production images (usage: make push-prod TAG=v1.0.0)
	@if [ -z "$(TAG)" ]; then echo "Error: TAG is required. Usage: make push-prod TAG=v1.0.0"; exit 1; fi
	docker push fund-backend:$(TAG)
	docker push fund-frontend:$(TAG)
	@echo "Pushed images with tag $(TAG)"

# Cleanup commands
clean: ## Stop all services and clean containers
	@echo "Stopping all services..."
	docker compose down
	@echo "Services cleaned up!"

clean-images: ## Remove unused images
	@echo "Removing unused images..."
	docker image prune -f
	@echo "Removing fund-related dangling images..."
	@docker images --filter "dangling=true" --filter "reference=fund-*" -q | xargs -r docker rmi || true
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

clean-all: ## Complete cleanup (WARNING: destroys all data)
	@echo "WARNING: This will destroy ALL Docker data!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker compose down -v; \
		docker system prune -a -f --volumes; \
		echo "Complete cleanup finished!"; \
	else \
		echo "Operation cancelled."; \
	fi

clean-fund-images: ## Remove all fund-related images (keeps other images)
	@echo "Removing all fund-related images..."
	@docker images --filter "reference=fund-*" -q | xargs -r docker rmi -f || true
	@docker images --filter "dangling=true" --filter "reference=fund-*" -q | xargs -r docker rmi || true
	@echo "Fund-related images cleaned up!"

# Utility commands
logs: ## Show logs for all services
	@echo "Showing logs for all services..."
	docker compose logs -f

logs-dev: ## Show logs for development services
	@echo "Showing logs for development services..."
	docker compose --profile dev logs -f

logs-prod: ## Show logs for production services
	@echo "Showing logs for production services..."
	docker compose --profile prod logs -f

logs-backend: ## Show backend logs
	@echo "Showing backend logs..."
	docker compose logs -f backend 2>/dev/null || docker compose --profile prod logs -f backend-prod 2>/dev/null || echo "Backend service not running"

logs-frontend: ## Show frontend logs
	@echo "Showing frontend logs..."
	docker compose logs -f frontend 2>/dev/null || docker compose --profile prod logs -f frontend-prod 2>/dev/null || echo "Frontend service not running"

logs-postgres: ## Show PostgreSQL logs
	@echo "Showing PostgreSQL logs..."
	docker compose logs -f postgres

logs-redis: ## Show Redis logs
	@echo "Showing Redis logs..."
	docker compose logs -f redis

logs-celery: ## Show Celery worker logs
	@echo "Showing Celery worker logs..."
	docker compose logs -f celery_worker 2>/dev/null || docker compose --profile prod logs -f celery_worker_prod 2>/dev/null || echo "Celery worker not running"

status: ## Show status of all containers
	@echo "Container Status:"
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" || echo "No services running"
	@echo ""
	@echo "All Containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

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

restart-dev: ## Restart development services
	@echo "Restarting development services..."
	docker compose --profile dev restart

restart-prod: ## Restart production services
	@echo "Restarting production services..."
	docker compose --profile prod restart

restart-backend: ## Restart backend service
	@echo "Restarting backend..."
	docker compose restart backend 2>/dev/null || docker compose --profile prod restart backend-prod

restart-frontend: ## Restart frontend service
	@echo "Restarting frontend..."
	docker compose restart frontend 2>/dev/null || docker compose --profile prod restart frontend-prod

restart-celery: ## Restart Celery worker service
	@echo "Restarting Celery worker..."
	docker compose restart celery_worker 2>/dev/null || docker compose --profile prod restart celery_worker_prod

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

# Cache commands
cache-clean: ## Clean Docker build cache
	docker builder prune -a -f

# Setup commands
setup: ## Initial project setup
	@echo "Setting up project..."
	@cp .env.example .env
	@echo "Please edit .env file and add your OPENAI_API_KEY"
	@echo "Then run 'make dev' to start development environment"

setup-dev: ## Setup development environment
	@echo "Setting up development environment..."
	@cp .env.example .env
	@echo "Please edit .env file and add your OPENAI_API_KEY"
	@make dev-build
	@make dev-up
	@echo "Development environment setup complete!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"

setup-prod: ## Setup production environment
	@echo "Setting up production environment..."
	@cp .env.example .env
	@echo "Please edit .env file and add your OPENAI_API_KEY"
	@make prod-build
	@make prod-up
	@echo "Production environment setup complete!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"

# Testing commands
test: ## Run all tests
	@echo "Running backend tests..."
	docker compose exec backend pytest app/services/test/ -v --cov=app
	@echo "Running frontend tests..."
	docker compose exec frontend npm test

test-backend: ## Run backend tests
	@echo "Running backend tests..."
	docker compose exec backend pytest app/services/test/ -v --cov=app

test-backend-unit: ## Run backend unit tests only
	@echo "Running backend unit tests..."
	docker compose exec backend pytest app/services/test/ -v

test-backend-cov: ## Run backend tests with coverage
	@echo "Running backend tests with coverage..."
	docker compose exec backend pytest app/services/test/ --cov=app --cov-report=html --cov-report=term

test-document-processor: ## Run document processor unit tests
	@echo "Running document processor unit tests..."
	docker compose exec backend pytest app/services/test/test_document_processor.py -v

test-query-engine: ## Run query engine unit tests
	@echo "Running query engine unit tests..."
	docker compose exec backend pytest app/services/test/test_query_engine.py -v

test-frontend: ## Run frontend tests
	@echo "Running frontend tests..."
	docker compose exec frontend npm test

# Documentation commands
docs: ## Open API documentation
	@echo "Opening API documentation..."
	@echo "Please start the backend service first with: make dev"
	@echo "Then visit: http://localhost:8000/docs"

# Quick commands
up: dev-up ## Alias for dev-up
down: dev-down ## Alias for dev-down
build: dev-build ## Alias for dev-build
rebuild: ## Rebuild and restart development environment
	@echo "Rebuilding development environment..."
	@make dev-down
	@make dev-build
	@make dev-up

# Individual service commands
start-postgres: ## Start PostgreSQL service only
	@echo "Starting PostgreSQL..."
	docker compose up -d postgres

start-redis: ## Start Redis service only
	@echo "Starting Redis..."
	docker compose up -d redis

start-backend: ## Start backend service only
	@echo "Starting backend..."
	docker compose --profile dev up -d backend

start-frontend: ## Start frontend service only
	@echo "Starting frontend..."
	docker compose --profile dev up -d frontend

start-celery: ## Start Celery worker service only
	@echo "Starting Celery worker..."
	docker compose --profile dev up -d celery_worker

stop-postgres: ## Stop PostgreSQL service only
	@echo "Stopping PostgreSQL..."
	docker compose stop postgres

stop-redis: ## Stop Redis service only
	@echo "Stopping Redis..."
	docker compose stop redis

stop-backend: ## Stop backend service only
	@echo "Stopping backend..."
	docker compose stop backend

stop-frontend: ## Stop frontend service only
	@echo "Stopping frontend..."
	docker compose stop frontend

stop-celery: ## Stop Celery worker service only
	@echo "Stopping Celery worker..."
	docker compose stop celery_worker

restart-postgres: ## Restart PostgreSQL service only
	@echo "Restarting PostgreSQL..."
	docker compose restart postgres

restart-redis: ## Restart Redis service only
	@echo "Restarting Redis..."
	docker compose restart redis

celery-status: ## Show Celery worker status
	@echo "Checking Celery worker status..."
	@docker compose exec celery_worker celery -A app.core.celery_app inspect active 2>/dev/null || docker compose --profile prod exec celery_worker_prod celery -A app.core.celery_app inspect active 2>/dev/null || echo "Celery worker not running"

celery-purge: ## Purge all Celery tasks
	@echo "Purging all Celery tasks..."
	@docker compose exec celery_worker celery -A app.core.celery_app purge -f 2>/dev/null || docker compose --profile prod exec celery_worker_prod celery -A app.core.celery_app purge -f 2>/dev/null || echo "Celery worker not running"

reset-db: ## Reset database data (keeps tables, removes data only)
	@echo "Resetting database data while preserving table structure..."
	docker compose down
	docker volume rm coding-test-3rd_postgres_data
	docker compose up -d postgres redis
	@echo "Waiting for database to be ready..."
	@sleep 5
	docker compose exec backend python -m app.db.init_db
	docker compose up -d backend frontend
	@echo "Database reset complete! Tables recreated and services restarted."

rebuild-backend-db: ## Rebuild backend, start it, and initialize database
	@echo "Rebuilding backend and reinitializing database..."
	make dev-build-backend
	make start-backend
	sleep 3
	make init-db
	@echo "Backend rebuilt and database initialized!"

# Environment info
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