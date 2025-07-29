# LangGraph Integration Deployment Guide

This guide covers deploying the LangGraph-enhanced Amazon Electronics Assistant with persistent state management and agent workflows.

## Overview

The LangGraph integration adds sophisticated agent workflows, persistent conversation memory, and advanced tool-based interactions to the existing RAG system. This deployment includes:

- **Agent Workflows**: Multi-step reasoning with specialized agents
- **Persistent State**: PostgreSQL-backed conversation memory
- **Tool Integration**: Vector search and product analysis tools
- **Performance Monitoring**: Health checks and performance metrics
- **Error Handling**: Graceful degradation and fallback mechanisms

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  LangGraph      │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│  Agents         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Weaviate      │    │   Tools Layer   │
│   (State)       │    │   (Vector DB)   │    │   (Search/Analysis)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB+ available space
- **Network**: Stable internet connection for LLM API calls

### Software Dependencies

- Docker & Docker Compose
- Python 3.12+
- PostgreSQL 15+ (if not using Docker)
- Git

### API Keys Required

- **OpenAI API Key** (required)
- **Groq API Key** (optional)
- **Google API Key** (optional)
- **LangSmith API Key** (optional, for tracing)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-powered-amazon-product-assistant

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Environment Configuration

Edit `.env` file with your configuration:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Database Configuration
POSTGRES_PASSWORD=secure_password_here
POSTGRES_DB=langgraph_assistant
POSTGRES_USER=postgres

# Performance Settings
LANGGRAPH_MAX_SESSIONS=1000
LANGGRAPH_CACHE_TTL=300
LANGGRAPH_TIMEOUT=30

# Monitoring
GRAFANA_PASSWORD=admin_password_here
```

### 3. Deploy with Docker Compose

```bash
# Deploy with LangGraph support
docker-compose -f docker-compose.yml -f docker-compose.langgraph.yml up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f app
```

### 4. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Check LangGraph system health
curl http://localhost:8000/health/system

# Check agent capabilities
curl http://localhost:8000/agent/capabilities

# Access Streamlit UI
open http://localhost:8501
```

## Configuration Options

### Agent Configuration

Configure agent behavior in your environment:

```bash
# Agent Types Available
LANGGRAPH_DEFAULT_AGENT=ambient
LANGGRAPH_ENABLE_MEMORY=true
LANGGRAPH_MAX_RETRIES=3

# LLM Provider Settings
LANGGRAPH_DEFAULT_PROVIDER=openai
LANGGRAPH_DEFAULT_MODEL=gpt-4o-mini
LANGGRAPH_DEFAULT_TEMPERATURE=0.7
```

### Database Configuration

PostgreSQL settings for state persistence:

```bash
# Connection Settings
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=langgraph_assistant
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password

# Connection Pool Settings
POSTGRES_MIN_CONN=1
POSTGRES_MAX_CONN=20
```

### Performance Tuning

Optimize performance for your deployment:

```bash
# Cache Settings
LANGGRAPH_CACHE_TTL=300          # 5 minutes
LANGGRAPH_MAX_CACHE_SIZE=1000    # Max cached items

# Session Management
LANGGRAPH_MAX_SESSIONS=1000      # Max concurrent sessions
LANGGRAPH_SESSION_TIMEOUT=3600   # 1 hour timeout

# Workflow Settings
LANGGRAPH_MAX_WORKFLOW_TIME=30   # Max workflow execution time
LANGGRAPH_MAX_TOOL_RETRIES=3     # Tool retry attempts
```

## API Endpoints

### Traditional RAG Endpoints

- `GET /health` - Basic health check
- `POST /query` - Traditional RAG query
- `POST /query/structured` - Structured RAG query

### LangGraph Agent Endpoints

- `POST /agent/query` - Agent-based query processing
- `GET /agent/status/{session_id}` - Get agent status
- `GET /agent/history/{session_id}` - Get conversation history
- `DELETE /agent/conversation/{session_id}` - Clear conversation
- `GET /agent/capabilities` - Get agent capabilities

### System Monitoring Endpoints

- `GET /health/system` - Comprehensive system health
- `GET /health/history` - Health check history
- `GET /health/component/{component}` - Component-specific health

## Usage Examples

### Basic Agent Query

```bash
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best wireless headphones under $200?",
    "session_id": "user_123",
    "agent_type": "ambient",
    "max_products": 5,
    "max_reviews": 3
  }'
```

### Multi-turn Conversation

```bash
# First query
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are good gaming laptops?",
    "session_id": "conversation_456",
    "enable_memory": true
  }'

# Follow-up query (remembers context)
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about under $1000?",
    "session_id": "conversation_456",
    "enable_memory": true
  }'
```

### Product Comparison

```bash
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare iPhone 14 vs Samsung Galaxy S23",
    "agent_type": "comparison",
    "max_products": 4
  }'
```

## Monitoring and Observability

### Health Monitoring

The system provides comprehensive health monitoring:

```bash
# Quick health check
curl http://localhost:8000/health/system?quick=true

# Full system health check
curl http://localhost:8000/health/system

# Component-specific health
curl http://localhost:8000/health/component/database
curl http://localhost:8000/health/component/vector_search
```

### Performance Monitoring

Access performance metrics:

- **Grafana Dashboard**: http://localhost:3000
- **Prometheus Metrics**: http://localhost:9090

### Logging

Logs are available through Docker:

```bash
# Application logs
docker-compose logs -f app

# Database logs
docker-compose logs -f postgres

# All services
docker-compose logs -f
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

```bash
# Check PostgreSQL status
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Test database connection
docker-compose exec postgres psql -U postgres -d langgraph_assistant -c "SELECT 1;"
```

#### 2. Agent Workflow Failures

```bash
# Check agent capabilities
curl http://localhost:8000/agent/capabilities

# Check system health
curl http://localhost:8000/health/system

# Review application logs
docker-compose logs app | grep -i error
```

#### 3. Memory Issues

```bash
# Check container resource usage
docker stats

# Check PostgreSQL connections
docker-compose exec postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Clear agent cache
curl -X DELETE http://localhost:8000/agent/cache
```

#### 4. Performance Issues

```bash
# Check performance metrics
curl http://localhost:8000/health/system

# Review slow operations
docker-compose logs app | grep -i "slow operation"

# Check database performance
docker-compose exec postgres psql -U postgres -d langgraph_assistant -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

### Error Recovery

The system includes automatic error recovery:

- **Tool Failures**: Graceful degradation to simpler responses
- **LLM Errors**: Automatic provider switching
- **Database Issues**: Fallback to in-memory state
- **Network Issues**: Retry with exponential backoff

### Debugging

Enable debug logging:

```bash
# Set debug environment
export LOG_LEVEL=DEBUG

# Restart with debug logging
docker-compose restart app

# View debug logs
docker-compose logs -f app
```

## Production Deployment

### Security Considerations

1. **API Keys**: Use secure key management (AWS Secrets Manager, etc.)
2. **Database**: Use strong passwords and connection encryption
3. **Network**: Configure firewalls and VPNs appropriately
4. **HTTPS**: Use reverse proxy with SSL certificates

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple app instances behind load balancer
2. **Database**: Consider PostgreSQL clustering for high availability
3. **Caching**: Implement Redis clustering for distributed caching
4. **Monitoring**: Set up comprehensive monitoring and alerting

### Backup Strategy

```bash
# Database backup
docker-compose exec postgres pg_dump -U postgres langgraph_assistant > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres langgraph_assistant < backup.sql
```

## Migration from Traditional RAG

The LangGraph integration maintains backward compatibility:

1. **Existing APIs**: All traditional RAG endpoints continue to work
2. **Gradual Migration**: Use feature flags to gradually migrate users
3. **Data Migration**: Existing data remains accessible
4. **Rollback**: Easy rollback to traditional RAG if needed

### Migration Steps

1. Deploy LangGraph integration alongside existing system
2. Test agent endpoints with sample queries
3. Gradually migrate users to agent-based queries
4. Monitor performance and adjust configuration
5. Fully migrate when confident in stability

## Support and Maintenance

### Regular Maintenance

- **Database Cleanup**: Remove old conversation data periodically
- **Log Rotation**: Implement log rotation to manage disk space
- **Performance Review**: Regular performance analysis and optimization
- **Security Updates**: Keep dependencies and base images updated

### Monitoring Alerts

Set up alerts for:
- High error rates (>5%)
- Slow response times (>10s)
- Database connection issues
- Memory usage (>80%)
- Disk space (>90%)

For additional support, refer to the project documentation or create an issue in the repository.