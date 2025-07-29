#!/bin/bash

# Docker entrypoint script for LangGraph AI Product Assistant
# Initializes services and starts both FastAPI and Streamlit

set -e

echo "🚀 Starting LangGraph AI Product Assistant..."

# Wait for PostgreSQL service to be ready
echo "⏳ Waiting for PostgreSQL service..."
timeout=60
while ! pg_isready -h postgres -p 5432 -U postgres > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ PostgreSQL service not ready after 60 seconds"
        echo "🔄 Continuing without persistent state..."
        break
    fi
done

if pg_isready -h postgres -p 5432 -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL service is ready"
    
    # Initialize database schema if needed
    echo "🔍 Initializing database schema..."
    python3 -c "
import os
import sys
sys.path.append('src')
try:
    from langgraph_integration.state.database import DatabaseManager
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    db_manager = DatabaseManager()
    db_manager.initialize()
    print('✅ Database schema initialized successfully')
        
except Exception as e:
    print(f'⚠️  Database initialization failed: {e}')
    print('🔄 Continuing without persistent state...')
    import traceback
    traceback.print_exc()
"
else
    echo "⚠️  PostgreSQL service unavailable, using in-memory state"
fi

# Wait for Weaviate service to be ready
echo "⏳ Waiting for Weaviate service..."
timeout=60
while ! curl -s http://weaviate:8080/v1/meta > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ Weaviate service not ready after 60 seconds"
        echo "🔄 Continuing with embedded fallback..."
        break
    fi
done

if curl -s http://weaviate:8080/v1/meta > /dev/null 2>&1; then
    echo "✅ Weaviate service is ready"
    
    # Initialize Weaviate with data if needed
    echo "🔍 Checking Weaviate initialization..."
    
    # Check if collection already exists and initialize if needed
    python3 -c "
import os
import sys
sys.path.append('src')
try:
    from rag.vector_db_weaviate_simple import ElectronicsVectorDBSimple, setup_vector_database_simple
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Try to connect and check for existing collection
    db = ElectronicsVectorDBSimple()
    stats = db.get_collection_stats()
    
    if stats.get('total_documents', 0) == 0:
        print('📦 Initializing vector database with documents...')
        jsonl_path = 'data/processed/electronics_rag_documents.jsonl'
        if os.path.exists(jsonl_path):
            setup_vector_database_simple(jsonl_path)
            print('✅ Vector database initialized successfully')
        else:
            print('⚠️  JSONL data file not found, continuing with empty database')
    else:
        print(f'✅ Found existing collection with {stats[\"total_documents\"]} documents')
        
except Exception as e:
    print(f'⚠️  Database initialization failed: {e}')
    print('🔄 Continuing without pre-populated database...')
    import traceback
    traceback.print_exc()
"
    echo "✅ Weaviate initialization complete"
else
    echo "⚠️  Weaviate service unavailable, using embedded storage"
fi

# Function to start FastAPI server
start_fastapi() {
    echo "🌐 Starting FastAPI server..."
    cd /app
    python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
    FASTAPI_PID=$!
    echo "✅ FastAPI server started (PID: $FASTAPI_PID)"
}

# Function to start Streamlit app
start_streamlit() {
    echo "🎯 Starting Streamlit application..."
    cd /app
    streamlit run src/chatbot_ui/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 &
    STREAMLIT_PID=$!
    echo "✅ Streamlit app started (PID: $STREAMLIT_PID)"
}

# Start both services
start_fastapi
start_streamlit

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID