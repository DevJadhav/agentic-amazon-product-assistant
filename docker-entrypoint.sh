#!/bin/bash

# Docker entrypoint script for AI Product Assistant
# Initializes Weaviate and starts Streamlit app

set -e

echo "🚀 Starting AI Product Assistant..."

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

# Start Streamlit app
echo "🎯 Starting Streamlit application..."
exec streamlit run src/chatbot_ui/streamlit_app.py --server.address=0.0.0.0 "$@"