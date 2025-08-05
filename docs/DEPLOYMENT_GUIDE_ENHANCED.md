# Enhanced AI Product Assistant Deployment Guide

## Overview

This guide covers the deployment of the enhanced AI Product Assistant with shopping cart functionality and intelligent routing capabilities. The system now includes:

- **Router Node**: Intelligent intent classification and routing
- **Shopping Cart Agent**: Specialized cart management functionality
- **Dual Tool Architecture**: MCP tools for QA Agent, function calling for Shopping Cart Agent
- **Enhanced Database Schema**: PostgreSQL with shopping cart tables
- **Real-time Frontend Updates**: Cart display with live updates
- **Performance Monitoring**: Enhanced metrics for cart and routing operations

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **Storage**: Minimum 50GB free disk space
- **CPU**: 4+ cores recommended
- **Network**: Stable internet connection for API calls

### Software Dependencies

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **PostgreSQL Client**: Version 13+ (for database operations)
- **Python**: Version 3.12+ (for migration scripts)

### API Keys Required

- **OpenAI API Key**: For LLM functionality
- **LangSmith API Key**: For tracing and monitoring (optional)
- **Groq API Key**: For alternative LLM provider (optional)
- **Google API Key**: For Gemini models (optional)

## Environment Configuration

### 1. Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Configure the following essential variables:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here

# Database Configuration
POSTGRES_PASSWORD=your_secure_postgres_password
REDIS_PASSWORD=your_secure_redis_password

# Shopping Cart Configuration
SHOPPING_CART_ENABLED=true
SHOPPING_CART_SESSION_TIMEOUT=86400
SHOPPING_CART_MAX_ITEMS=100

# Router Configuration
ROUTER_INTENT_CONFIDENCE_THRESHOLD=0.7
ROUTER_CACHE_ENABLED=true
ROUTER_MAX_CLARIFICATION_ATTEMPTS=3

# Monitoring
ENABLE_CART_METRICS=true
ENABLE_ROUTER_METRICS=true
LOG_LEVEL=INFO
```

### 2. Production Environment Variables

For production deployments, create `.env.production`:

```bash
# Production Configuration
ENVIRONMENT=production
LOG_LEVEL=WARNING

# Security Settings
POSTGRES_PASSWORD=your_very_secure_postgres_password
REDIS_PASSWORD=your_very_secure_redis_password

# Performance Settings
SHOPPING_CART_DB_POOL_SIZE=20
SHOPPING_CART_DB_MAX_OVERFLOW=30
LANGGRAPH_MAX_SESSIONS=2000

# Monitoring
ENABLE_PERFORMANCE_MONITORING=true
METRICS_RETENTION_DAYS=90
```

## Deployment Methods

### Method 1: Development Deployment

For development and testing:

```bash
# Clone the repository
git clone <repository-url>
cd agentic-amazon-product-assistant

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose -f docker-compose.langgraph.yml up -d

# Wait for services to be ready
./deploy/migrate-database.sh --check

# Run database migrations
./deploy/migrate-database.sh

# Verify deployment
curl http://localhost:8501/_stcore/health
curl http://localhost:8000/health
```

### Method 2: Production Deployment

For production environments:

```bash
# Use the production deployment script
chmod +x deploy/production-deploy.sh
sudo ./deploy/production-deploy.sh
```

### Method 3: Manual Production Deployment

For custom production setups:

#### Step 1: Prepare Environment

```bash
# Create deployment directory
sudo mkdir -p /opt/ai-product-assistant
sudo chown $(whoami):$(whoami) /opt/ai-product-assistant
cd /opt/ai-product-assistant

# Copy application files
cp -r /path/to/source/* .

# Set up production environment
cp .env.example .env.production
# Edit .env.production with production values
```

#### Step 2: Database Setup

```bash
# Start PostgreSQL and Redis
docker-compose -f docker-compose.langgraph.yml up -d postgres redis

# Wait for database to be ready
./deploy/migrate-database.sh --check

# Run migrations
./deploy/migrate-database.sh
```

#### Step 3: Application Deployment

```bash
# Build and start application services
docker-compose -f docker-compose.langgraph.yml up -d

# Verify all services are healthy
docker-compose -f docker-compose.langgraph.yml ps
```

## Database Migration

### Automatic Migration

The deployment includes automatic database migration:

```bash
# Run all migrations
./deploy/migrate-database.sh

# Check migration status
./deploy/migrate-database.sh --validate

# Check database connection
./deploy/migrate-database.sh --check
```

### Manual Migration

If automatic migration fails:

```bash
# Connect to PostgreSQL
PGPASSWORD=your_password psql -h localhost -p 5432 -U postgres -d langgraph_assistant

# Run schema manually
\i src/langgraph_integration/state/shopping_cart_schema.sql

# Verify tables
\dt
```

### Migration Rollback

If migration fails:

```bash
# Rollback to previous state
./deploy/migrate-database.sh --rollback
```

## Service Configuration

### PostgreSQL Configuration

The system requires PostgreSQL with the following tables:

- `conversations`: Conversation state management
- `conversation_messages`: Message history
- `agent_states`: Agent state persistence
- `shopping_cart`: Shopping cart items
- `cart_sessions`: Cart session summaries
- `intent_classifications`: Intent classification cache
- `schema_migrations`: Migration tracking

### Redis Configuration

Redis is used for:

- Session caching
- Intent classification cache
- Performance metrics temporary storage

### Monitoring Setup

#### Prometheus Configuration

Prometheus collects metrics from:

- Application endpoints (`/metrics`)
- Shopping cart metrics (`/metrics/cart`)
- Router metrics (`/metrics/router`)
- Database metrics
- System metrics

#### Grafana Dashboards

Access Grafana at `http://localhost:3000`:

- Default credentials: `admin/admin`
- Pre-configured dashboards for:
  - Application performance
  - Shopping cart metrics
  - Router performance
  - Database health

## Health Checks and Monitoring

### Application Health Checks

```bash
# Streamlit health check
curl http://localhost:8501/_stcore/health

# FastAPI health check
curl http://localhost:8000/health

# Database health check
curl http://localhost:8000/health/database

# Shopping cart health check
curl http://localhost:8000/health/cart
```

### Service Status Monitoring

```bash
# Check all services
docker-compose -f docker-compose.langgraph.yml ps

# Check logs
docker-compose -f docker-compose.langgraph.yml logs -f langgraph_app

# Check database logs
docker-compose -f docker-compose.langgraph.yml logs -f postgres
```

### Performance Monitoring

Monitor key metrics:

- **Response Times**: API and UI response times
- **Cart Operations**: Add/remove/list operations per minute
- **Router Performance**: Intent classification accuracy and speed
- **Database Performance**: Query execution times
- **Memory Usage**: Application and database memory consumption

## Security Considerations

### Database Security

```bash
# Use strong passwords
POSTGRES_PASSWORD=your_very_secure_password_here

# Limit database connections
# Configure in docker-compose.langgraph.yml:
# POSTGRES_MAX_CONNECTIONS=100
```

### API Security

```bash
# Enable CORS protection
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Use secure API keys
# Store in environment variables, not in code
```

### Network Security

```bash
# Use internal Docker networks
# Configure firewall rules for production
# Use HTTPS in production (configure reverse proxy)
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose -f docker-compose.langgraph.yml exec postgres pg_dump -U postgres langgraph_assistant > backup.sql

# Restore backup
docker-compose -f docker-compose.langgraph.yml exec -T postgres psql -U postgres langgraph_assistant < backup.sql
```

### Application Data Backup

```bash
# Backup application data
tar -czf app_backup.tar.gz data/ logs/ .env.production

# Backup Docker volumes
docker run --rm -v langgraph_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_data.tar.gz /data
```

## Scaling and Performance

### Horizontal Scaling

For high-traffic deployments:

```bash
# Scale application containers
docker-compose -f docker-compose.langgraph.yml up -d --scale langgraph_app=3

# Use load balancer (nginx, HAProxy)
# Configure database connection pooling
```

### Performance Optimization

```bash
# Optimize PostgreSQL
# Edit postgresql.conf:
# shared_buffers = 256MB
# effective_cache_size = 1GB
# work_mem = 4MB

# Optimize Redis
# Configure maxmemory and eviction policies
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check PostgreSQL status
docker-compose -f docker-compose.langgraph.yml logs postgres

# Test connection
PGPASSWORD=your_password pg_isready -h localhost -p 5432 -U postgres
```

#### Shopping Cart Issues

```bash
# Check cart table exists
docker-compose -f docker-compose.langgraph.yml exec postgres psql -U postgres -d langgraph_assistant -c "\dt shopping_cart"

# Check cart functionality
curl -X POST http://localhost:8000/api/cart/test
```

#### Router Issues

```bash
# Check intent classification
curl -X POST http://localhost:8000/api/router/classify -H "Content-Type: application/json" -d '{"message": "add to cart"}'

# Check router metrics
curl http://localhost:8000/metrics/router
```

### Log Analysis

```bash
# Application logs
docker-compose -f docker-compose.langgraph.yml logs -f langgraph_app

# Database logs
docker-compose -f docker-compose.langgraph.yml logs -f postgres

# System logs
tail -f /var/log/ai-product-assistant/monitor.log
```

## Maintenance

### Regular Maintenance Tasks

```bash
# Clean up old intent classifications
docker-compose -f docker-compose.langgraph.yml exec postgres psql -U postgres -d langgraph_assistant -c "SELECT cleanup_old_intent_classifications(7);"

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Update Docker images
docker-compose -f docker-compose.langgraph.yml pull
docker-compose -f docker-compose.langgraph.yml up -d
```

### Database Maintenance

```bash
# Vacuum and analyze tables
docker-compose -f docker-compose.langgraph.yml exec postgres psql -U postgres -d langgraph_assistant -c "VACUUM ANALYZE;"

# Check database size
docker-compose -f docker-compose.langgraph.yml exec postgres psql -U postgres -d langgraph_assistant -c "SELECT pg_size_pretty(pg_database_size('langgraph_assistant'));"
```

## Support and Documentation

### Additional Resources

- [Shopping Cart User Guide](SHOPPING_CART_USER_GUIDE.md)
- [Router Architecture Documentation](ROUTER_ARCHITECTURE.md)
- [API Documentation](SHOPPING_CART_API.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

### Getting Help

1. Check the troubleshooting guide
2. Review application logs
3. Check GitHub issues
4. Contact the development team

## Version Information

- **Enhanced System Version**: 2.0.0
- **Shopping Cart Agent**: 1.0.0
- **Router Node**: 1.0.0
- **Database Schema Version**: 2.0.0
- **Docker Compose Version**: 2.0+

---

**Note**: This deployment guide covers the enhanced system with shopping cart and routing functionality. For the basic system deployment, refer to the original deployment guide.