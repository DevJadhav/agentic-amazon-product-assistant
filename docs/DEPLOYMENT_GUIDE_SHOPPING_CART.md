# Shopping Cart Agent System - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the enhanced AI Product Assistant system with shopping cart functionality and intelligent routing capabilities. The deployment includes database schema migrations, environment configuration, and monitoring setup.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **Storage**: Minimum 50GB free disk space
- **CPU**: 4+ cores recommended
- **Network**: Stable internet connection for API access

### Software Dependencies

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **PostgreSQL**: Version 13+ (for database migrations)
- **Python**: Version 3.12+ (for migration scripts)
- **Git**: For code deployment

### API Keys Required

```bash
# Required API keys (set in .env file)
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
LANGSMITH_API_KEY=your_langsmith_key

# Database configuration
POSTGRES_PASSWORD=secure_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=langgraph_assistant
POSTGRES_USER=postgres

# Shopping cart configuration
CART_SESSION_TIMEOUT=86400  # 24 hours
CART_MAX_ITEMS=100
CART_CLEANUP_INTERVAL=3600  # 1 hour

# Router configuration
ROUTER_CONFIDENCE_THRESHOLD=0.7
ROUTER_MAX_CLARIFICATION_ATTEMPTS=3
ROUTER_ENABLE_CACHING=true
ROUTER_CACHE_TTL=300  # 5 minutes
```

## Pre-Deployment Checklist

### 1. Environment Preparation

```bash
# Create deployment directory
sudo mkdir -p /opt/ai-product-assistant
sudo chown $(whoami):$(whoami) /opt/ai-product-assistant
cd /opt/ai-product-assistant

# Clone repository
git clone <repository-url> .
git checkout main

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

### 2. Database Setup

```bash
# Start PostgreSQL container for initial setup
docker-compose -f docker-compose.langgraph.yml up -d postgres

# Wait for PostgreSQL to be ready
sleep 30

# Verify database connection
docker-compose -f docker-compose.langgraph.yml exec postgres psql -U postgres -d langgraph_assistant -c "SELECT version();"
```

### 3. Run Database Migrations

```bash
# Install Python dependencies for migration
python -m pip install -e .

# Run shopping cart schema migrations
python -m src.langgraph_integration.state.migrations

# Verify migration success
python -c "
from src.langgraph_integration.state.migrations import check_migration_status
status = check_migration_status()
print(f'Migrations: {status[\"successful_migrations\"]}/{status[\"total_migrations\"]} successful')
assert status['failed_migrations'] == 0, 'Some migrations failed'
print('✅ All migrations completed successfully')
"
```

## Deployment Steps

### Step 1: Build Application Images

```bash
# Build all images
docker-compose -f docker-compose.langgraph.yml build --no-cache

# Verify images are built
docker images | grep langgraph
```

### Step 2: Deploy Services

```bash
# Start all services
docker-compose -f docker-compose.langgraph.yml up -d

# Check service status
docker-compose -f docker-compose.langgraph.yml ps
```

### Step 3: Verify Deployment

```bash
# Wait for services to be ready
sleep 60

# Health checks
curl -f http://localhost:8000/health || echo "API health check failed"
curl -f http://localhost:8502/_stcore/health || echo "Streamlit health check failed"

# Test database connectivity
docker-compose -f docker-compose.langgraph.yml exec postgres psql -U postgres -d langgraph_assistant -c "SELECT COUNT(*) FROM shopping_cart;"

# Test cart functionality
python -c "
from src.langgraph_integration.state.shopping_cart_manager import ShoppingCartManager
from src.langgraph_integration.state.database import get_database_manager

cart_manager = ShoppingCartManager(get_database_manager())
result = cart_manager.add_item('deployment_test', 'test_product', 'Test Product', 1)
assert result['success'], f'Cart test failed: {result}'
print('✅ Cart functionality test passed')
"
```

## Configuration Management

### Environment Variables

Create production environment configuration:

```bash
# /opt/ai-product-assistant/.env.production
DEPLOYMENT_ENV=production

# Application settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Database configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=langgraph_assistant
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password

# Shopping cart settings
CART_SESSION_TIMEOUT=86400
CART_MAX_ITEMS=100
CART_CLEANUP_INTERVAL=3600
CART_ENABLE_PERSISTENCE=true

# Router settings
ROUTER_CONFIDENCE_THRESHOLD=0.7
ROUTER_MAX_CLARIFICATION_ATTEMPTS=3
ROUTER_ENABLE_CACHING=true
ROUTER_CACHE_TTL=300

# Performance settings
LANGGRAPH_CACHE_TTL=300
LANGGRAPH_MAX_SESSIONS=1000
LANGGRAPH_ENABLE_PERSISTENCE=true

# Security settings
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Monitoring
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-product-assistant-prod
PROMETHEUS_ENABLED=true
```

### Docker Compose Production Configuration

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: shopping_cart_postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./src/langgraph_integration/state/init.sql:/docker-entrypoint-initdb.d/01-init.sql
      - ./src/langgraph_integration/state/shopping_cart_schema.sql:/docker-entrypoint-initdb.d/02-shopping-cart.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - shopping_cart_network

  redis:
    image: redis:7-alpine
    container_name: shopping_cart_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - shopping_cart_network

  app:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: shopping_cart_app
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
    ports:
      - "8501:8501"
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - shopping_cart_network

  prometheus:
    image: prom/prometheus:latest
    container_name: shopping_cart_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    networks:
      - shopping_cart_network

  grafana:
    image: grafana/grafana:latest
    container_name: shopping_cart_grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - shopping_cart_network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  shopping_cart_network:
    driver: bridge
```

## Database Migration Procedures

### Automated Migration

```bash
#!/bin/bash
# migrate_shopping_cart.sh

set -e

echo "🔄 Starting shopping cart database migration..."

# Check database connection
docker-compose -f docker-compose.production.yml exec postgres pg_isready -U ${POSTGRES_USER}

# Create backup
BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
docker-compose -f docker-compose.production.yml exec postgres pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} > ${BACKUP_FILE}
echo "✅ Database backup created: ${BACKUP_FILE}"

# Run migrations
python -m src.langgraph_integration.state.migrations

# Verify migration
python -c "
from src.langgraph_integration.state.migrations import validate_shopping_cart_schema
validation = validate_shopping_cart_schema()
assert all(validation.values()), f'Schema validation failed: {validation}'
print('✅ Schema validation passed')
"

echo "✅ Shopping cart migration completed successfully"
```

### Manual Migration Steps

If automated migration fails, follow these manual steps:

```sql
-- Connect to database
psql -h localhost -U postgres -d langgraph_assistant

-- Create shopping cart tables
\i src/langgraph_integration/state/shopping_cart_schema.sql

-- Verify tables created
\dt shopping_cart*

-- Test basic operations
INSERT INTO shopping_cart (session_id, product_id, product_title, quantity) 
VALUES ('test_session', 'test_product', 'Test Product', 1);

SELECT * FROM shopping_cart WHERE session_id = 'test_session';

-- Clean up test data
DELETE FROM shopping_cart WHERE session_id = 'test_session';
```

## Monitoring and Alerting Setup

### Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'shopping-cart-app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

Create monitoring dashboards for:

1. **Application Metrics**
   - Request rates and response times
   - Error rates by endpoint
   - Cart operation success rates
   - Router classification accuracy

2. **Database Metrics**
   - Connection pool usage
   - Query performance
   - Cart table sizes
   - Migration status

3. **System Metrics**
   - CPU and memory usage
   - Disk space utilization
   - Network traffic

### Health Check Endpoints

The application provides several health check endpoints:

```bash
# Application health
curl http://localhost:8501/_stcore/health

# API health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# Cart functionality health
curl http://localhost:8000/health/cart

# Router health
curl http://localhost:8000/health/router
```

## Performance Optimization

### Database Optimization

```sql
-- Optimize PostgreSQL for cart operations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Reload configuration
SELECT pg_reload_conf();

-- Update table statistics
ANALYZE shopping_cart;
ANALYZE cart_sessions;
ANALYZE intent_classifications;
```

### Application Optimization

```bash
# Set environment variables for performance
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Configure connection pooling
export POSTGRES_POOL_SIZE=20
export POSTGRES_MAX_OVERFLOW=30
export POSTGRES_POOL_TIMEOUT=30

# Configure caching
export ROUTER_CACHE_SIZE=1000
export CART_CACHE_TTL=300
export INTENT_CACHE_SIZE=500
```

## Security Configuration

### Database Security

```sql
-- Create application-specific user
CREATE USER cart_app WITH PASSWORD 'secure_app_password';

-- Grant minimal required permissions
GRANT CONNECT ON DATABASE langgraph_assistant TO cart_app;
GRANT USAGE ON SCHEMA public TO cart_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON shopping_cart TO cart_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON cart_sessions TO cart_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON intent_classifications TO cart_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cart_app;
```

### Application Security

```bash
# Configure secure headers
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10

# Configure session security
export SESSION_COOKIE_SECURE=true
export SESSION_COOKIE_HTTPONLY=true
export SESSION_COOKIE_SAMESITE=strict
```

### Network Security

```yaml
# docker-compose.production.yml security additions
services:
  app:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
    user: "1000:1000"
```

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup_shopping_cart.sh

BACKUP_DIR="/opt/backups/shopping-cart"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/shopping_cart_backup_${DATE}.sql"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Database backup
docker-compose -f docker-compose.production.yml exec -T postgres pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} > ${BACKUP_FILE}

# Compress backup
gzip ${BACKUP_FILE}

# Remove old backups (keep last 7 days)
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +7 -delete

echo "✅ Backup completed: ${BACKUP_FILE}.gz"
```

### Recovery Procedures

```bash
#!/bin/bash
# restore_shopping_cart.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    exit 1
fi

# Stop application
docker-compose -f docker-compose.production.yml stop app

# Restore database
gunzip -c ${BACKUP_FILE} | docker-compose -f docker-compose.production.yml exec -T postgres psql -U ${POSTGRES_USER} ${POSTGRES_DB}

# Start application
docker-compose -f docker-compose.production.yml start app

echo "✅ Recovery completed from: ${BACKUP_FILE}"
```

## Troubleshooting

### Common Issues

1. **Migration Failures**
   ```bash
   # Check migration status
   python -c "from src.langgraph_integration.state.migrations import check_migration_status; print(check_migration_status())"
   
   # Retry failed migrations
   python -m src.langgraph_integration.state.migrations --retry-failed
   ```

2. **Database Connection Issues**
   ```bash
   # Check database connectivity
   docker-compose -f docker-compose.production.yml exec postgres pg_isready -U ${POSTGRES_USER}
   
   # Check connection pool
   docker-compose -f docker-compose.production.yml logs app | grep -i "connection"
   ```

3. **Cart Operation Failures**
   ```bash
   # Test cart functionality
   python -c "
   from src.langgraph_integration.state.shopping_cart_manager import ShoppingCartManager
   from src.langgraph_integration.state.database import get_database_manager
   
   cart_manager = ShoppingCartManager(get_database_manager())
   result = cart_manager.add_item('debug_session', 'debug_product', 'Debug Product', 1)
   print(f'Cart test result: {result}')
   "
   ```

### Log Analysis

```bash
# Application logs
docker-compose -f docker-compose.production.yml logs -f app

# Database logs
docker-compose -f docker-compose.production.yml logs -f postgres

# Filter for cart-related logs
docker-compose -f docker-compose.production.yml logs app | grep -i "cart\|shopping"

# Filter for router-related logs
docker-compose -f docker-compose.production.yml logs app | grep -i "router\|intent"
```

## Maintenance Procedures

### Regular Maintenance Tasks

```bash
#!/bin/bash
# maintenance.sh

# Clean up old intent classifications
docker-compose -f docker-compose.production.yml exec postgres psql -U ${POSTGRES_USER} ${POSTGRES_DB} -c "SELECT cleanup_old_intent_classifications(7);"

# Update table statistics
docker-compose -f docker-compose.production.yml exec postgres psql -U ${POSTGRES_USER} ${POSTGRES_DB} -c "ANALYZE shopping_cart; ANALYZE cart_sessions; ANALYZE intent_classifications;"

# Clean up old cart sessions (inactive for 30 days)
docker-compose -f docker-compose.production.yml exec postgres psql -U ${POSTGRES_USER} ${POSTGRES_DB} -c "
DELETE FROM shopping_cart 
WHERE session_id IN (
    SELECT session_id FROM cart_sessions 
    WHERE last_updated < CURRENT_TIMESTAMP - INTERVAL '30 days'
);
DELETE FROM cart_sessions 
WHERE last_updated < CURRENT_TIMESTAMP - INTERVAL '30 days';
"

# Restart application to clear memory
docker-compose -f docker-compose.production.yml restart app

echo "✅ Maintenance tasks completed"
```

### Performance Monitoring

```bash
#!/bin/bash
# monitor_performance.sh

# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8501/_stcore/health

# Check database performance
docker-compose -f docker-compose.production.yml exec postgres psql -U ${POSTGRES_USER} ${POSTGRES_DB} -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_tup_hot_upd as hot_updates
FROM pg_stat_user_tables 
WHERE tablename IN ('shopping_cart', 'cart_sessions', 'intent_classifications');
"

# Check cache hit rates
docker-compose -f docker-compose.production.yml exec redis redis-cli info stats | grep -E "keyspace_hits|keyspace_misses"
```

## Scaling Considerations

### Horizontal Scaling

For high-traffic deployments, consider:

1. **Load Balancer Configuration**
   ```yaml
   # nginx.conf
   upstream shopping_cart_app {
       server app1:8501;
       server app2:8501;
       server app3:8501;
   }
   ```

2. **Database Read Replicas**
   ```yaml
   # docker-compose.production.yml
   postgres_replica:
     image: postgres:15-alpine
     environment:
       POSTGRES_MASTER_SERVICE: postgres
       POSTGRES_REPLICA_USER: replica
   ```

3. **Redis Cluster**
   ```yaml
   redis_cluster:
     image: redis:7-alpine
     command: redis-server --cluster-enabled yes
   ```

### Vertical Scaling

```yaml
# docker-compose.production.yml resource limits
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
  
  postgres:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

## Conclusion

This deployment guide provides comprehensive instructions for deploying the shopping cart agent system in production. Follow the steps carefully and ensure all health checks pass before considering the deployment complete.

For additional support or troubleshooting, refer to the troubleshooting guide or contact the development team.

## Quick Reference

### Essential Commands

```bash
# Deploy
docker-compose -f docker-compose.production.yml up -d

# Health check
curl http://localhost:8501/_stcore/health

# View logs
docker-compose -f docker-compose.production.yml logs -f app

# Backup database
./backup_shopping_cart.sh

# Run maintenance
./maintenance.sh

# Scale application
docker-compose -f docker-compose.production.yml up -d --scale app=3
```

### Important URLs

- Application: http://localhost:8501
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Database: localhost:5432