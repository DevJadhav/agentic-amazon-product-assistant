# 🚀 Production Deployment Guide

## 🌟 Overview

Deploy your Amazon Electronics Assistant to production with confidence using this comprehensive deployment guide. Supports local development, Docker containers, and cloud platforms with enterprise-grade reliability.

### **🎯 Deployment Options**

- **🖥️ Local Development**: Quick setup for development and testing
- **🐳 Docker Container**: Production-ready containerized deployment
- **☁️ Cloud Platforms**: AWS, Google Cloud, Azure deployment ready
- **🔄 CI/CD Integration**: Automated deployment pipeline support

---

## 🏃‍♂️ Quick Start

### **Prerequisites**

- Python 3.12+
- Docker (for containerized deployment)
- Git
- 4GB+ RAM recommended

### **1. Clone and Install**

```bash
# Clone the repository
git clone <your-repository-url>
cd agentic-amazon-product-assistant

# Install dependencies
pip install -e .
```

### **2. Environment Configuration**

```bash
# Copy example environment file
cp .env.example .env

# Configure your API keys
echo "OPENAI_API_KEY=your_openai_key" >> .env
echo "LANGSMITH_API_KEY=your_langsmith_key" >> .env
echo "LANGSMITH_PROJECT=amazon-electronics-assistant" >> .env
```

### **3. Validate Installation**

```bash
# Run production validation tests
python scripts/simplified_production_test.py

# Expected output: 100% test success rate
```

### **4. Launch Application**

```bash
# Start the assistant
streamlit run src/chatbot_ui/streamlit_app.py

# Access at: http://localhost:8501
```

---

## 🐳 Docker Deployment

### **Production Container Build**

```bash
# Build production image
docker build -f Dockerfile.production -t amazon-assistant:latest .

# Run with proper configuration
docker run -d \
  --name amazon-assistant \
  -p 8501:8501 \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e LANGSMITH_API_KEY=${LANGSMITH_API_KEY} \
  -e LANGSMITH_PROJECT=amazon-electronics-assistant \
  -v $(pwd)/data:/app/data \
  amazon-assistant:latest
```

### **Docker Compose Deployment**

```yaml
# docker-compose.yml
version: '3.8'

services:
  amazon-assistant:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=amazon-electronics-assistant
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Deploy with Docker Compose
docker-compose up -d

# Monitor logs
docker-compose logs -f amazon-assistant
```

---

## ☁️ Cloud Platform Deployment

### **AWS ECS Deployment**

```bash
# Create ECS task definition
aws ecs register-task-definition \
  --family amazon-assistant \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 1024 \
  --memory 2048 \
  --execution-role-arn arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole \
  --container-definitions '[{
    "name": "amazon-assistant",
    "image": "your-registry/amazon-assistant:latest",
    "portMappings": [{"containerPort": 8501}],
    "environment": [
      {"name": "OPENAI_API_KEY", "value": "your-key"},
      {"name": "LANGSMITH_API_KEY", "value": "your-key"}
    ]
  }]'

# Create ECS service
aws ecs create-service \
  --cluster your-cluster \
  --service-name amazon-assistant \
  --task-definition amazon-assistant \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}'
```

### **Google Cloud Run**

```bash
# Build and push to Google Container Registry
docker build -f Dockerfile.production -t gcr.io/PROJECT-ID/amazon-assistant .
docker push gcr.io/PROJECT-ID/amazon-assistant

# Deploy to Cloud Run
gcloud run deploy amazon-assistant \
  --image gcr.io/PROJECT-ID/amazon-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars OPENAI_API_KEY=your-key,LANGSMITH_API_KEY=your-key
```

### **Azure Container Instances**

```bash
# Create container group
az container create \
  --resource-group your-resource-group \
  --name amazon-assistant \
  --image your-registry/amazon-assistant:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8501 \
  --environment-variables \
    OPENAI_API_KEY=your-key \
    LANGSMITH_API_KEY=your-key \
  --restart-policy Always
```

---

## 🔧 Configuration Management

### **Environment Variables**

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT models | - |
| `LANGSMITH_API_KEY` | No | LangSmith tracing API key | - |
| `LANGSMITH_PROJECT` | No | LangSmith project name | `amazon-electronics-assistant` |
| `GROQ_API_KEY` | No | Groq API key for fast inference | - |
| `GOOGLE_API_KEY` | No | Google AI API key | - |
| `LOG_LEVEL` | No | Logging level | `INFO` |

### **Production Settings**

```bash
# Production environment variables
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export LANGSMITH_TRACING=true
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### **Security Configuration**

```bash
# Secure deployment settings
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
export STREAMLIT_CLIENT_TOOLBAR_MODE=minimal
```

---

## 📊 Monitoring & Health Checks

### **Health Check Endpoints**

The application provides built-in health monitoring:

```bash
# Application health
curl http://localhost:8501/_stcore/health

# Database connectivity
curl http://localhost:8501/api/health/database

# System metrics
curl http://localhost:8501/api/health/metrics
```

### **Production Monitoring Setup**

```python
# Health monitoring script
#!/usr/bin/env python3
import requests
import sys
import time

def check_health():
    try:
        response = requests.get('http://localhost:8501/_stcore/health', timeout=10)
        if response.status_code == 200:
            print("✅ Application healthy")
            return True
        else:
            print(f"❌ Application unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
```

### **Performance Monitoring**

```bash
# Set up log aggregation
docker run -d \
  --name log-monitor \
  -v /var/log/amazon-assistant:/logs \
  fluent/fluent-bit

# Monitor resource usage
docker stats amazon-assistant

# Application metrics
curl http://localhost:8501/api/metrics | jq
```

---

## 🔄 CI/CD Pipeline

### **GitHub Actions Example**

```yaml
# .github/workflows/deploy.yml
name: Deploy Amazon Assistant

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -e .
      - name: Run tests
        run: python scripts/simplified_production_test.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        run: |
          docker build -f Dockerfile.production -t amazon-assistant:$GITHUB_SHA .
          docker push your-registry/amazon-assistant:$GITHUB_SHA
      - name: Deploy to production
        run: |
          # Your deployment script here
          ./deploy/production-deploy.sh
```

---

## 🚨 Troubleshooting

### **Common Issues**

#### **Application Won't Start**
```bash
# Check logs
docker logs amazon-assistant

# Verify environment variables
docker exec amazon-assistant env | grep API_KEY

# Test configuration
docker exec amazon-assistant python -c "from src.chatbot_ui.core.config import config; print('Config OK')"
```

#### **Database Connection Issues**
```bash
# Test vector database
docker exec amazon-assistant python -c "
from src.rag.query_processor import create_rag_processor
processor = create_rag_processor()
print('Database:', 'Connected' if processor.vector_db else 'Failed')
"
```

#### **Performance Issues**
```bash
# Monitor resource usage
docker stats amazon-assistant

# Check system health
python scripts/simplified_production_test.py

# Review performance metrics
curl http://localhost:8501/api/metrics
```

### **Debug Mode**

```bash
# Enable debug mode
docker run -d \
  -e LOG_LEVEL=DEBUG \
  -e STREAMLIT_LOGGER_LEVEL=debug \
  amazon-assistant:latest

# Access debug information
docker logs amazon-assistant | grep DEBUG
```

---

## 📈 Scaling Considerations

### **Horizontal Scaling**

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amazon-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: amazon-assistant
  template:
    metadata:
      labels:
        app: amazon-assistant
    spec:
      containers:
      - name: amazon-assistant
        image: your-registry/amazon-assistant:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### **Load Balancing**

```nginx
# nginx.conf
upstream amazon_assistant {
    server container1:8501;
    server container2:8501;
    server container3:8501;
}

server {
    listen 80;
    location / {
        proxy_pass http://amazon_assistant;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## ✅ Production Checklist

### **Pre-Deployment**
- [ ] Environment variables configured
- [ ] API keys validated
- [ ] Database connectivity tested
- [ ] Health checks passing
- [ ] Performance benchmarks met
- [ ] Security settings applied

### **Post-Deployment**
- [ ] Application accessible
- [ ] Monitoring dashboard active
- [ ] Log aggregation working
- [ ] Health checks automated
- [ ] Backup procedures verified
- [ ] Incident response plan ready

---

**Deploy your Amazon Electronics Assistant with confidence using this production-ready deployment guide.** 