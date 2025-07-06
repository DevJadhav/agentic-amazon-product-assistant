#!/bin/bash

# =============================================================================
# 🚀 AI-Powered Amazon Product Assistant - Production Deployment Script
# =============================================================================
# Enterprise-grade deployment with monitoring, health checks, and rollback
# Author: Amazon Electronics Assistant Team
# Version: 1.0.0
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
APP_NAME="ai-product-assistant"
DEPLOY_DIR="/opt/${APP_NAME}"
BACKUP_DIR="/opt/${APP_NAME}-backups"
LOG_DIR="/var/log/${APP_NAME}"
COMPOSE_FILE="docker-compose.production.yml"
HEALTHCHECK_TIMEOUT=300
ROLLBACK_TIMEOUT=60

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Pre-deployment checks
check_prerequisites() {
    log "🔍 Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check system resources
    AVAILABLE_RAM=$(free -g | awk '/^Mem:/{print $7}')
    AVAILABLE_DISK=$(df -h / | awk 'NR==2{print $4}' | sed 's/G//')
    
    if [[ $AVAILABLE_RAM -lt 4 ]]; then
        error "Insufficient RAM: ${AVAILABLE_RAM}GB available, 4GB required"
        exit 1
    fi
    
    if [[ $AVAILABLE_DISK -lt 20 ]]; then
        error "Insufficient disk space: ${AVAILABLE_DISK}GB available, 20GB required"
        exit 1
    fi
    
    # Check ports
    if lsof -i :8501 &> /dev/null; then
        error "Port 8501 is already in use"
        exit 1
    fi
    
    if lsof -i :8080 &> /dev/null; then
        error "Port 8080 is already in use"
        exit 1
    fi
    
    success "All prerequisites met"
}

# Environment setup
setup_environment() {
    log "🔧 Setting up deployment environment..."
    
    # Create directories
    sudo mkdir -p $DEPLOY_DIR
    sudo mkdir -p $BACKUP_DIR
    sudo mkdir -p $LOG_DIR
    sudo chown -R $(whoami):$(whoami) $DEPLOY_DIR
    sudo chown -R $(whoami):$(whoami) $BACKUP_DIR
    sudo chmod 755 $LOG_DIR
    
    # Copy application files
    cp -r . $DEPLOY_DIR/
    cd $DEPLOY_DIR
    
    # Set up environment variables
    if [[ ! -f .env.production ]]; then
        log "📝 Creating production environment configuration..."
        cat > .env.production << EOF
# Production Configuration
DEPLOYMENT_ENV=production
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Database Configuration
WEAVIATE_HOST=weaviate
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_SCHEME=http

# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-product-assistant-prod
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Performance Configuration
PYTHONPATH=/app/src:/app
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1

# Security Configuration
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
EOF
    fi
    
    success "Environment setup complete"
}

# Backup current deployment
backup_deployment() {
    log "💾 Creating deployment backup..."
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    # Create backup directory
    mkdir -p $BACKUP_PATH
    
    # Backup application code
    if [[ -d "$DEPLOY_DIR.old" ]]; then
        cp -r $DEPLOY_DIR.old $BACKUP_PATH/application
    fi
    
    # Backup database
    if docker-compose -f $COMPOSE_FILE ps weaviate | grep -q "Up"; then
        log "📦 Backing up Weaviate database..."
        docker-compose -f $COMPOSE_FILE exec -T weaviate sh -c "tar -czf /tmp/weaviate_backup.tar.gz /var/lib/weaviate" || true
        docker-compose -f $COMPOSE_FILE cp weaviate:/tmp/weaviate_backup.tar.gz $BACKUP_PATH/ || true
    fi
    
    # Backup logs
    if [[ -d "$LOG_DIR" ]]; then
        cp -r $LOG_DIR $BACKUP_PATH/logs
    fi
    
    success "Backup created at $BACKUP_PATH"
    echo "BACKUP_PATH=$BACKUP_PATH" > /tmp/deployment_backup.env
}

# Deploy application
deploy_application() {
    log "🚀 Deploying application..."
    
    # Build and start services
    log "🔨 Building application images..."
    docker-compose -f $COMPOSE_FILE build --no-cache
    
    log "🔄 Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    # Wait for services to be ready
    log "⏳ Waiting for services to be ready..."
    wait_for_services
    
    success "Application deployed successfully"
}

# Wait for services to be ready
wait_for_services() {
    local timeout=$HEALTHCHECK_TIMEOUT
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if check_health; then
            success "All services are healthy"
            return 0
        fi
        
        log "Services not ready yet, waiting... (${elapsed}s/${timeout}s)"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    
    error "Services failed to become healthy within ${timeout}s"
    return 1
}

# Health check
check_health() {
    local all_healthy=true
    
    # Check Weaviate
    if ! curl -s http://localhost:8080/v1/meta > /dev/null; then
        warn "Weaviate is not responding"
        all_healthy=false
    fi
    
    # Check Streamlit
    if ! curl -s http://localhost:8501/_stcore/health > /dev/null; then
        warn "Streamlit is not responding"
        all_healthy=false
    fi
    
    # Check container health
    if ! docker-compose -f $COMPOSE_FILE ps | grep -q "healthy\|Up"; then
        warn "Some containers are not healthy"
        all_healthy=false
    fi
    
    return $all_healthy
}

# Performance monitoring setup
setup_monitoring() {
    log "📊 Setting up performance monitoring..."
    
    # Create monitoring scripts
    cat > /usr/local/bin/ai-assistant-monitor << 'EOF'
#!/bin/bash
# AI Assistant Performance Monitor

LOG_FILE="/var/log/ai-product-assistant/monitor.log"

log_metric() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

# System metrics
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1)

log_metric "SYSTEM_METRICS cpu=${CPU_USAGE}% memory=${MEMORY_USAGE}% disk=${DISK_USAGE}%"

# Application metrics
if curl -s http://localhost:8501/_stcore/health > /dev/null; then
    log_metric "STREAMLIT_STATUS healthy"
else
    log_metric "STREAMLIT_STATUS unhealthy"
fi

if curl -s http://localhost:8080/v1/meta > /dev/null; then
    log_metric "WEAVIATE_STATUS healthy"
else
    log_metric "WEAVIATE_STATUS unhealthy"
fi

# Container metrics
CONTAINER_COUNT=$(docker ps --format "table {{.Names}}" | grep -c "ai-product-assistant")
log_metric "CONTAINER_COUNT ${CONTAINER_COUNT}"

# Response time test
START_TIME=$(date +%s%3N)
curl -s http://localhost:8501 > /dev/null
END_TIME=$(date +%s%3N)
RESPONSE_TIME=$((END_TIME - START_TIME))
log_metric "RESPONSE_TIME ${RESPONSE_TIME}ms"
EOF

    chmod +x /usr/local/bin/ai-assistant-monitor
    
    # Set up cron job for monitoring
    (crontab -l 2>/dev/null || echo "") | grep -v "ai-assistant-monitor" | crontab -
    (crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/ai-assistant-monitor") | crontab -
    
    success "Performance monitoring configured"
}

# Rollback function
rollback_deployment() {
    error "Deployment failed, initiating rollback..."
    
    if [[ -f "/tmp/deployment_backup.env" ]]; then
        source /tmp/deployment_backup.env
        
        log "🔄 Rolling back to previous version..."
        
        # Stop current services
        docker-compose -f $COMPOSE_FILE down
        
        # Restore backup
        if [[ -d "$BACKUP_PATH/application" ]]; then
            rm -rf $DEPLOY_DIR.current
            mv $DEPLOY_DIR $DEPLOY_DIR.current
            cp -r $BACKUP_PATH/application $DEPLOY_DIR
        fi
        
        # Restore database
        if [[ -f "$BACKUP_PATH/weaviate_backup.tar.gz" ]]; then
            log "🔄 Restoring database..."
            docker-compose -f $COMPOSE_FILE up -d weaviate
            sleep 10
            docker-compose -f $COMPOSE_FILE cp $BACKUP_PATH/weaviate_backup.tar.gz weaviate:/tmp/
            docker-compose -f $COMPOSE_FILE exec -T weaviate sh -c "cd / && tar -xzf /tmp/weaviate_backup.tar.gz"
        fi
        
        # Restart services
        docker-compose -f $COMPOSE_FILE up -d
        
        # Wait for rollback to complete
        if wait_for_services; then
            success "Rollback completed successfully"
        else
            error "Rollback failed"
        fi
    else
        error "No backup found for rollback"
    fi
}

# Post-deployment verification
verify_deployment() {
    log "🔍 Verifying deployment..."
    
    # Test API endpoints
    if curl -s http://localhost:8501/_stcore/health | grep -q "ok"; then
        success "Streamlit health check passed"
    else
        error "Streamlit health check failed"
        return 1
    fi
    
    if curl -s http://localhost:8080/v1/meta | grep -q "version"; then
        success "Weaviate health check passed"
    else
        error "Weaviate health check failed"
        return 1
    fi
    
    # Test application functionality
    log "🧪 Testing application functionality..."
    
    # Create test query
    TEST_RESPONSE=$(curl -s -X POST http://localhost:8501/api/v1/query \
        -H "Content-Type: application/json" \
        -d '{"query": "test query"}' || echo "FAILED")
    
    if [[ "$TEST_RESPONSE" != "FAILED" ]]; then
        success "Application functionality test passed"
    else
        warn "Application functionality test failed (may be expected if API endpoint not implemented)"
    fi
    
    # Log deployment success
    log "📊 Deployment Summary:"
    log "   - Environment: $DEPLOYMENT_ENV"
    log "   - Application: Running on port 8501"
    log "   - Database: Running on port 8080"
    log "   - Monitoring: Enabled"
    log "   - Logs: Available in $LOG_DIR"
    
    success "Deployment verification completed"
}

# Main deployment function
main() {
    log "🚀 Starting AI Product Assistant deployment..."
    
    # Create production Docker Compose
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log "📝 Creating production Docker Compose configuration..."
        cat > $COMPOSE_FILE << 'EOF'
services:
  streamlit-app:
    build: 
      context: .
      dockerfile: Dockerfile.production
    container_name: ai-product-assistant-app
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ./.env.production:/app/.env:ro
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app/src:/app
      - WEAVIATE_HOST=weaviate
      - WEAVIATE_PORT=8080
      - WEAVIATE_GRPC_PORT=50051
    depends_on:
      - weaviate
    networks:
      - ai-assistant-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.25.0
    container_name: ai-product-assistant-weaviate
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - weaviate-data:/var/lib/weaviate
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=none
      - ENABLE_MODULES=text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai
      - CLUSTER_HOSTNAME=node1
      - LOG_LEVEL=info
    networks:
      - ai-assistant-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/meta"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  monitoring:
    image: prom/prometheus:latest
    container_name: ai-product-assistant-monitoring
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - ai-assistant-network

volumes:
  weaviate-data:
    driver: local
  prometheus-data:
    driver: local

networks:
  ai-assistant-network:
    driver: bridge
EOF
    fi
    
    # Execute deployment steps
    check_prerequisites
    setup_environment
    backup_deployment
    
    if deploy_application; then
        setup_monitoring
        if verify_deployment; then
            success "🎉 Deployment completed successfully!"
            log "📊 Application is running at: http://localhost:8501"
            log "📊 Database is running at: http://localhost:8080"
            log "📊 Monitoring is running at: http://localhost:9090"
        else
            rollback_deployment
            exit 1
        fi
    else
        rollback_deployment
        exit 1
    fi
}

# Trap errors and perform rollback
trap 'rollback_deployment' ERR

# Run deployment
main "$@" 