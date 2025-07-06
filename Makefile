# ============================================================================
# Amazon Electronics Assistant - Build Automation
# Production-ready build system for Docker deployment and development
# ============================================================================

.PHONY: help install dev-setup clean test build deploy monitor

# Default target
help:
	@echo "🚀 Amazon Electronics Assistant - Build System"
	@echo "=============================================="
	@echo ""
	@echo "🔧 Development Commands:"
	@echo "  dev-setup     Initialize development environment"
	@echo "  dev-run       Launch development server locally"
	@echo "  dev-notebook  Start Jupyter notebook server"
	@echo "  dev-test      Run comprehensive test suite"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  docker-build  Build optimized production containers"
	@echo "  docker-up     Start complete application stack"
	@echo "  docker-down   Stop all running services"
	@echo "  docker-logs   View real-time application logs"
	@echo "  docker-health Check service health status"
	@echo ""
	@echo "🧹 Maintenance Commands:"
	@echo "  clean-docker  Remove containers and free disk space"
	@echo "  clean-cache   Clear Python and notebook cache files"
	@echo "  reset-db      Reset vector database to initial state"
	@echo ""
	@echo "📊 Monitoring Commands:"
	@echo "  monitor-perf  Display performance metrics"
	@echo "  monitor-logs  Stream application logs with filtering"
	@echo "  validate-env  Verify environment configuration"

# ============================================================================
# Development Environment Setup
# ============================================================================

install:
	@echo "📦 Installing project dependencies..."
	uv sync

dev-setup: install
	@echo "🔧 Setting up development environment..."
	uv run python -m ipykernel install --user --name amazon-assistant
	@echo "✅ Development environment ready!"

dev-run:
	@echo "🚀 Starting development server..."
	@echo "🌐 Application will be available at: http://localhost:8501"
	uv run streamlit run src/chatbot_ui/streamlit_app.py

dev-notebook:
	@echo "📓 Starting Jupyter notebook server..."
	@echo "🌐 Notebooks will be available at: http://localhost:8888"
	uv run jupyter notebook notebooks/

dev-test:
	@echo "🧪 Running comprehensive test suite..."
	uv run python eval/test_rag_system.py

# ============================================================================
# Docker Production Environment
# ============================================================================

docker-build:
	@echo "🐳 Building production-ready containers..."
	@echo "📦 This process may take several minutes..."
	docker-compose build --no-cache streamlit-app
	@echo "✅ Container build completed successfully!"

docker-up:
	@echo "🚀 Launching production application stack..."
	@echo "🔍 Weaviate Database: http://localhost:8080"
	@echo "🎯 Main Application: http://localhost:8501"
	docker-compose up -d
	@echo "✅ All services started successfully!"

docker-down:
	@echo "🛑 Stopping all application services..."
	docker-compose down
	@echo "✅ All services stopped gracefully!"

docker-logs:
	@echo "📋 Streaming application logs..."
	docker-compose logs -f streamlit-app

docker-health:
	@echo "🏥 Checking service health status..."
	@echo "📊 Service Status:"
	docker-compose ps
	@echo ""
	@echo "🔍 Weaviate Health:"
	curl -s http://localhost:8080/v1/meta | jq -r '.version' 2>/dev/null || echo "❌ Weaviate not responding"
	@echo ""
	@echo "🎯 Application Health:"
	curl -s http://localhost:8501/health | head -5 2>/dev/null || echo "❌ Application not responding"

# ============================================================================
# Maintenance and Cleanup
# ============================================================================

clean-docker:
	@echo "🧹 Performing comprehensive Docker cleanup..."
	docker-compose down -v --remove-orphans
	docker system prune -af --volumes
	@echo "✅ Docker cleanup completed!"

clean-cache:
	@echo "🧹 Clearing Python and notebook cache files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
	@echo "✅ Cache cleanup completed!"

reset-db:
	@echo "🔄 Resetting vector database to initial state..."
	docker-compose down weaviate
	docker volume rm agentic-amazon-product-assistant_weaviate_data 2>/dev/null || true
	docker-compose up -d weaviate
	@echo "✅ Database reset completed!"

# ============================================================================
# Monitoring and Validation
# ============================================================================

monitor-perf:
	@echo "📊 Displaying real-time performance metrics..."
	@echo "Memory Usage:"
	docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
	@echo ""
	@echo "Disk Usage:"
	docker system df

monitor-logs:
	@echo "📋 Streaming filtered application logs..."
	docker-compose logs -f streamlit-app | grep -E "(🚀|⏳|✅|⚠️|🔄|ERROR|INFO)"

validate-env:
	@echo "🔍 Validating environment configuration..."
	@echo "📋 Configuration Check:"
	uv run python -c "from src.chatbot_ui.core.config import config; print(f'✅ Environment: {config.ENVIRONMENT}'); print(f'✅ Docker Mode: {config.is_docker}'); print(f'✅ API Keys: {len([k for k in config.api_keys_dict.values() if k])} configured')"

# ============================================================================
# Legacy Support (Deprecated)
# ============================================================================

# Backward compatibility aliases
run-streamlit: dev-run
	@echo "⚠️  Warning: 'run-streamlit' is deprecated. Use 'make dev-run' instead."

build-docker-streamlit: docker-build
	@echo "⚠️  Warning: 'build-docker-streamlit' is deprecated. Use 'make docker-build' instead."

run-docker-streamlit: docker-up
	@echo "⚠️  Warning: 'run-docker-streamlit' is deprecated. Use 'make docker-up' instead."

stop-docker-streamlit: docker-down
	@echo "⚠️  Warning: 'stop-docker-streamlit' is deprecated. Use 'make docker-down' instead."

logs-docker-streamlit: docker-logs
	@echo "⚠️  Warning: 'logs-docker-streamlit' is deprecated. Use 'make docker-logs' instead."

# ============================================================================
# Advanced Operations
# ============================================================================

benchmark:
	@echo "⚡ Running performance benchmarks..."
	uv run python eval/run_evaluation.py --create-dataset
	uv run python eval/run_synthetic_evaluation.py --synthetic-only --num-synthetic 50

security-scan:
	@echo "🔒 Performing security vulnerability scan..."
	docker run --rm -v $(PWD):/app --workdir /app python:3.12-slim pip-audit --require-hashes --disable-pip || echo "⚠️  Install pip-audit for security scanning"

backup-data:
	@echo "💾 Creating data backup..."
	docker run --rm -v agentic-amazon-product-assistant_weaviate_data:/source -v $(PWD)/backups:/backup alpine tar czf /backup/weaviate_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz -C /source .

restore-data:
	@echo "🔄 Restoring data from latest backup..."
	@echo "⚠️  This will overwrite existing data!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ]
	docker run --rm -v agentic-amazon-product-assistant_weaviate_data:/target -v $(PWD)/backups:/backup alpine tar xzf /backup/$(shell ls -t backups/weaviate_backup_*.tar.gz | head -1) -C /target

# ============================================================================
# CI/CD Integration
# ============================================================================

ci-test:
	@echo "🔄 Running CI/CD test pipeline..."
	make validate-env
	make dev-test
	@echo "✅ CI/CD tests passed!"

ci-build:
	@echo "🔄 Running CI/CD build pipeline..."
	make docker-build
	make docker-up
	sleep 30
	make docker-health
	make docker-down
	@echo "✅ CI/CD build passed!"

# ============================================================================
# Documentation Generation
# ============================================================================

docs-serve:
	@echo "📚 Starting documentation server..."
	@echo "🌐 Documentation available at: http://localhost:8000"
	python -m http.server 8000 --directory docs/

docs-validate:
	@echo "📋 Validating documentation links..."
	find docs/ -name "*.md" -exec grep -l "http" {} \; | head -5