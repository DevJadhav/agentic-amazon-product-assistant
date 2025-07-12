# 🛒 AI-Powered Amazon Electronics Assistant

**Production-ready AI assistant for electronics product discovery, comparison, and recommendations**

[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen)]()
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)]()
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue)]()

---

## 🌟 Features

### **🧠 Intelligent Query Processing**
- **6 Query Types**: Recommendations, comparisons, complaints, use-cases, general search, and product-specific queries
- **Multi-LLM Support**: OpenAI GPT, Google Gemini, Groq, and Ollama integration
- **Context Awareness**: Maintains conversation context across interactions

### **🗄️ Advanced Vector Database**
- **6,000+ Products**: Electronics products with specifications and reviews
- **Weaviate Integration**: Production-grade vector database with intelligent fallbacks
- **Hybrid Search**: Combines semantic and keyword search for optimal results

### **📊 Production Monitoring**
- **LangSmith Tracing**: Complete observability of the AI pipeline
- **Performance Analytics**: Response time, accuracy, and satisfaction tracking
- **Health Monitoring**: Automated system health checks and alerting

### **🧪 Quality Assurance**
- **Automated Testing**: Comprehensive test suite with synthetic data generation
- **Performance Benchmarks**: System validation against production targets
- **100% Test Coverage**: All core functionality thoroughly tested

---

## 🚀 Quick Start

### **1. Prerequisites**

- Python 3.12 or higher
- 4GB+ RAM recommended
- API keys for your preferred LLM provider

### **2. Installation**

```bash
# Clone the repository
git clone <your-repository-url>
cd agentic-amazon-product-assistant

# Install dependencies
pip install -e .
```

### **3. Configuration**

```bash
# Set up environment variables
cp .env.example .env

# Configure your API keys (choose your preferred provider)
echo "OPENAI_API_KEY=your_openai_key" >> .env

# Optional: Enable monitoring
echo "LANGSMITH_API_KEY=your_langsmith_key" >> .env
echo "LANGSMITH_PROJECT=amazon-electronics-assistant" >> .env
```

### **4. Validation**

```bash
# Run production test suite
python scripts/production_test_suite.py

# Expected: 6/6 tests passing (100% success rate)
```

### **5. Launch**

```bash
# Start the assistant
streamlit run src/chatbot_ui/enhanced_streamlit_app.py

# Access at: http://localhost:8501
```

---

## 🐳 Docker Deployment

### **Quick Docker Setup**

```bash
# Build production image
docker build -f Dockerfile.production -t amazon-assistant .

# Run with environment variables
docker run -d \
  --name amazon-assistant \
  -p 8501:8501 \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  amazon-assistant
```

### **Docker Compose**

```bash
# Deploy with monitoring
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## 💬 Example Interactions

### **Product Recommendations**
```
User: "What are the best wireless headphones under $200?"

```
