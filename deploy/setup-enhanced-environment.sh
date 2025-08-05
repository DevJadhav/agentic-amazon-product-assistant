#!/bin/bash

# =============================================================================
# Enhanced AI Product Assistant Environment Setup Script
# =============================================================================
# Sets up environment for shopping cart and router functionality
# Author: Enhanced System Team
# Version: 2.0.0
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
PROD_ENV_FILE="$PROJECT_ROOT/.env.production"

# Logging functions
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

# Check system requirements
check_system_requirements() {
    log "🔍 Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3.12+ first."
        exit 1
    fi
    
    # Check PostgreSQL client
    if ! command -v psql &> /dev/null; then
        warn "PostgreSQL client not found. Installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y postgresql-client
        elif command -v brew &> /dev/null; then
            brew install postgresql
        else
            error "Please install PostgreSQL client manually"
            exit 1
        fi
    fi
    
    # Check system resources
    AVAILABLE_RAM=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}' || echo "8")
    AVAILABLE_DISK=$(df -h . 2>/dev/null | awk 'NR==2{print $4}' | sed 's/G//' || echo "50")
    
    if [[ $AVAILABLE_RAM -lt 4 ]]; then
        warn "Low RAM detected: ${AVAILABLE_RAM}GB available, 8GB recommended"
    fi
    
    if [[ $AVAILABLE_DISK -lt 20 ]]; then
        warn "Low disk space: ${AVAILABLE_DISK}GB available, 50GB recommended"
    fi
    
    success "System requirements check completed"
}

# Generate secure passwords
generate_secure_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Setup environment files
setup_environment_files() {
    log "📝 Setting up environment configuration..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        log "Creating development environment file..."
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        
        # Generate secure passwords
        POSTGRES_PASSWORD=$(generate_secure_password)
        REDIS_PASSWORD=$(generate_secure_password)
        
        # Update passwords in .env file
        sed -i.bak "s/your_postgres_password_here/$POSTGRES_PASSWORD/g" "$ENV_FILE"
        sed -i.bak "s/your_redis_password_here/$REDIS_PASSWORD/g" "$ENV_FILE"
        rm -f "$ENV_FILE.bak"
        
        success "Development environment file created"
    else
        log "Development environment file already exists"
    fi
    
    # Create production environment file
    if [[ ! -f "$PROD_ENV_FILE" ]]; then
        log "Creating production environment file..."
        
        # Generate secure passwords for production
        PROD_POSTGRES_PASSWORD=$(generate_secure_password)
        PROD_REDIS_PASSWORD=$(generate_secure_password)
        GRAFANA_PASSWORD=$(generate_secure_password)
        
        cat > "$PROD_ENV_FILE" << EOF
# =============================================================================
# Enhanced AI Product Assistant - Production Environment Configuration
# =============================================================================

# Environment
ENVIRONMENT=production
LOG_LEVEL=WARNING

# API Keys (REQUIRED - Set these before deployment)
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Database Configuration
POSTGRES_DB=langgraph_assistant
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$PROD_POSTGRES_PASSWORD
REDIS_PASSWORD=$PROD_REDIS_PASSWORD

# Shopping Cart Configuration
SHOPPING_CART_ENABLED=true
SHOPPING_CART_SESSION_TIMEOUT=86400
SHOPPING_CART_MAX_ITEMS=100
SHOPPING_CART_CLEANUP_INTERVAL=3600
SHOPPING_CART_DB_POOL_SIZE=20
SHOPPING_CART_DB_MAX_OVERFLOW=30
SHOPPING_CART_DB_TIMEOUT=30

# Router Configuration
ROUTER_INTENT_CONFIDENCE_THRESHOLD=0.7
ROUTER_CACHE_ENABLED=true
ROUTER_CACHE_TTL=300
ROUTER_MAX_CLARIFICATION_ATTEMPTS=3
ROUTER_CLASSIFICATION_TIMEOUT=5
ROUTER_ENABLE_FALLBACK=true
ROUTER_FALLBACK_AGENT=qa

# LangGraph Configuration
LANGGRAPH_MAX_SESSIONS=2000
LANGGRAPH_CACHE_TTL=300

# Monitoring Configuration
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_CART_METRICS=true
ENABLE_ROUTER_METRICS=true
ENABLE_INTENT_CLASSIFICATION_METRICS=true
METRICS_COLLECTION_INTERVAL=60
METRICS_RETENTION_DAYS=90

# Grafana Configuration
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# Security Settings
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
EOF
        
        success "Production environment file created"
        warn "Please update API keys in $PROD_ENV_FILE before production deployment"
    else
        log "Production environment file already exists"
    fi
}

# Create necessary directories
create_directories() {
    log "📁 Creating necessary directories..."
    
    local directories=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/data/processed"
        "$PROJECT_ROOT/data/weaviate_db"
        "$PROJECT_ROOT/monitoring/grafana/dashboards"
        "$PROJECT_ROOT/monitoring/grafana/datasources"
        "$PROJECT_ROOT/deploy/ssl"
        "/tmp/ai_assistant_backups"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 755 "$PROJECT_ROOT/data"
    
    success "Directory structure created"
}

# Setup SSL certificates (self-signed for development)
setup_ssl_certificates() {
    log "🔐 Setting up SSL certificates..."
    
    local ssl_dir="$PROJECT_ROOT/deploy/ssl"
    local cert_file="$ssl_dir/cert.pem"
    local key_file="$ssl_dir/key.pem"
    
    if [[ ! -f "$cert_file" ]] || [[ ! -f "$key_file" ]]; then
        log "Generating self-signed SSL certificates for development..."
        
        openssl req -x509 -newkey rsa:4096 -keyout "$key_file" -out "$cert_file" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
            2>/dev/null
        
        chmod 600 "$key_file"
        chmod 644 "$cert_file"
        
        success "Self-signed SSL certificates generated"
        warn "For production, replace with proper SSL certificates"
    else
        log "SSL certificates already exist"
    fi
}

# Setup Grafana dashboards
setup_grafana_dashboards() {
    log "📊 Setting up Grafana dashboards..."
    
    local dashboards_dir="$PROJECT_ROOT/monitoring/grafana/dashboards"
    local datasources_dir="$PROJECT_ROOT/monitoring/grafana/datasources"
    
    # Create dashboard provisioning config
    cat > "$dashboards_dir/dashboard.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    # Create datasource config
    cat > "$datasources_dir/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    success "Grafana configuration created"
}

# Validate environment setup
validate_environment() {
    log "🔍 Validating environment setup..."
    
    local validation_passed=true
    
    # Check environment files
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Development environment file missing"
        validation_passed=false
    fi
    
    if [[ ! -f "$PROD_ENV_FILE" ]]; then
        error "Production environment file missing"
        validation_passed=false
    fi
    
    # Check API keys in development environment
    if grep -q "your_openai_api_key_here" "$ENV_FILE" 2>/dev/null; then
        warn "OpenAI API key not set in development environment"
    fi
    
    # Check directory structure
    local required_dirs=(
        "$PROJECT_ROOT/src/langgraph_integration"
        "$PROJECT_ROOT/src/chatbot_ui"
        "$PROJECT_ROOT/monitoring"
        "$PROJECT_ROOT/deploy"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            error "Required directory missing: $dir"
            validation_passed=false
        fi
    done
    
    # Check Docker Compose files
    if [[ ! -f "$PROJECT_ROOT/docker-compose.langgraph.yml" ]]; then
        error "LangGraph Docker Compose file missing"
        validation_passed=false
    fi
    
    if [[ ! -f "$PROJECT_ROOT/docker-compose.production.yml" ]]; then
        error "Production Docker Compose file missing"
        validation_passed=false
    fi
    
    if [[ "$validation_passed" == true ]]; then
        success "Environment validation passed"
        return 0
    else
        error "Environment validation failed"
        return 1
    fi
}

# Display setup summary
display_setup_summary() {
    log "📋 Environment Setup Summary:"
    echo ""
    echo "✅ System requirements checked"
    echo "✅ Environment files created"
    echo "✅ Directory structure created"
    echo "✅ SSL certificates generated"
    echo "✅ Grafana dashboards configured"
    echo ""
    echo "📁 Key Files Created:"
    echo "   - $ENV_FILE (development environment)"
    echo "   - $PROD_ENV_FILE (production environment)"
    echo "   - $PROJECT_ROOT/deploy/ssl/ (SSL certificates)"
    echo "   - $PROJECT_ROOT/monitoring/grafana/ (Grafana config)"
    echo ""
    echo "🚀 Next Steps:"
    echo "   1. Update API keys in environment files"
    echo "   2. For development: docker-compose -f docker-compose.langgraph.yml up -d"
    echo "   3. For production: ./deploy/production-deploy.sh"
    echo ""
    echo "📚 Documentation:"
    echo "   - Deployment Guide: docs/DEPLOYMENT_GUIDE_ENHANCED.md"
    echo "   - Shopping Cart Guide: docs/SHOPPING_CART_USER_GUIDE.md"
    echo "   - Troubleshooting: docs/TROUBLESHOOTING_GUIDE.md"
}

# Main setup function
main() {
    log "🚀 Starting Enhanced AI Product Assistant environment setup..."
    
    cd "$PROJECT_ROOT"
    
    check_system_requirements
    setup_environment_files
    create_directories
    setup_ssl_certificates
    setup_grafana_dashboards
    
    if validate_environment; then
        success "🎉 Environment setup completed successfully!"
        display_setup_summary
    else
        error "Environment setup failed validation"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --check)
        validate_environment
        ;;
    --directories)
        create_directories
        ;;
    --ssl)
        setup_ssl_certificates
        ;;
    --grafana)
        setup_grafana_dashboards
        ;;
    *)
        main "$@"
        ;;
esac