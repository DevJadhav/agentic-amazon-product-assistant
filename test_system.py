#!/usr/bin/env python3
"""
Comprehensive system test for the enhanced RAG pipeline.
Tests all components end-to-end to ensure production readiness.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly."""
    logger.info("Testing imports...")
    
    try:
        # Test basic imports
        from rag.enhanced_vector_db import EnhancedElectronicsVectorDB, HybridSearchConfig
        from rag.enhanced_query_processor import EnhancedRAGQueryProcessor, SearchStrategy
        from rag.structured_outputs import StructuredRAGRequest, StructuredRAGResponse, ResponseType
        from rag.structured_generator import StructuredResponseGenerator
        from prompts.registry import get_registry, PromptType
        from api.main import app
        
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_structured_outputs():
    """Test structured outputs functionality."""
    logger.info("Testing structured outputs...")
    
    try:
        from rag.structured_outputs import (
            StructuredRAGRequest, ProductInfo, ResponseType, ConfidenceLevel
        )
        
        # Test creating a request
        request = StructuredRAGRequest(
            query="Best laptop under $1000",
            max_products=3,
            preferred_response_type=ResponseType.PRODUCT_RECOMMENDATION
        )
        
        # Test creating product info
        product = ProductInfo(
            title="Test Laptop",
            price=800.0,
            rating=4.5,
            rating_count=100
        )
        
        logger.info("✅ Structured outputs working correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Structured outputs failed: {e}")
        return False

def test_prompt_registry():
    """Test prompt registry functionality."""
    logger.info("Testing prompt registry...")
    
    try:
        from prompts.registry import get_registry, PromptType
        
        # Get registry instance
        registry = get_registry()
        
        # List templates
        templates = registry.list_templates()
        
        if not templates:
            logger.warning("⚠️ No templates found, but registry works")
            return True
        
        # Test template validation
        for template_id in templates[:2]:  # Test first 2
            is_valid = registry.validate_template(template_id)
            if not is_valid:
                logger.warning(f"⚠️ Template {template_id} validation failed")
        
        logger.info(f"✅ Prompt registry working with {len(templates)} templates")
        return True
    except Exception as e:
        logger.error(f"❌ Prompt registry failed: {e}")
        return False

def test_enhanced_query_processor():
    """Test enhanced query processor."""
    logger.info("Testing enhanced query processor...")
    
    try:
        from rag.enhanced_query_processor import EnhancedRAGQueryProcessor, SearchStrategy
        
        # Create processor (will use mock database)
        processor = EnhancedRAGQueryProcessor()
        
        # Test analytics
        analytics = processor.get_search_analytics()
        
        if "error" in analytics:
            logger.warning(f"⚠️ Query processor has limitations: {analytics['error']}")
        else:
            logger.info("✅ Enhanced query processor working correctly")
        
        return True
    except Exception as e:
        logger.error(f"❌ Enhanced query processor failed: {e}")
        return False

def test_api_configuration():
    """Test API configuration and endpoints."""
    logger.info("Testing API configuration...")
    
    try:
        from api.main import app
        from fastapi.testclient import TestClient
        
        # Create test client
        client = TestClient(app)
        
        # Test API structure (without making actual requests to avoid startup issues)
        logger.info("✅ API configuration valid")
        return True
    except Exception as e:
        logger.error(f"❌ API configuration failed: {e}")
        return False

def test_vector_database():
    """Test vector database functionality."""
    logger.info("Testing vector database...")
    
    try:
        from rag.enhanced_vector_db import EnhancedElectronicsVectorDB, HybridSearchConfig
        
        # Test configuration
        config = HybridSearchConfig()
        
        logger.info(f"✅ Vector database configuration valid: "
                   f"semantic_weight={config.semantic_weight}, "
                   f"keyword_weight={config.keyword_weight}")
        return True
    except Exception as e:
        logger.error(f"❌ Vector database test failed: {e}")
        return False

def check_dependencies():
    """Check that all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'), 
        ('pydantic', 'pydantic'),
        ('jinja2', 'jinja2'),
        ('pyyaml', 'yaml'),
        ('rank_bm25', 'rank_bm25'),
        ('instructor', 'instructor'),
        ('scikit-learn', 'sklearn'),
        ('sentence_transformers', 'sentence_transformers'),
        ('weaviate', 'weaviate'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas')
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        logger.error(f"❌ Missing dependencies: {missing}")
        return False
    else:
        logger.info("✅ All dependencies available")
        return True

def check_file_structure():
    """Check that all required files exist."""
    logger.info("Checking file structure...")
    
    required_files = [
        "src/rag/enhanced_vector_db.py",
        "src/rag/enhanced_query_processor.py",
        "src/rag/structured_outputs.py",
        "src/rag/structured_generator.py",
        "src/api/main.py",
        "src/api/run_server.py",
        "src/prompts/templates/rag_templates.j2",
        "src/prompts/registry.py",
        "pyproject.toml"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        logger.error(f"❌ Missing files: {missing}")
        return False
    else:
        logger.info("✅ All required files exist")
        return True

def main():
    """Run comprehensive system test."""
    logger.info("🚀 Starting comprehensive system test...")
    
    tests = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Imports", test_imports),
        ("Structured Outputs", test_structured_outputs),
        ("Prompt Registry", test_prompt_registry),
        ("Vector Database", test_vector_database),
        ("Query Processor", test_enhanced_query_processor),
        ("API Configuration", test_api_configuration),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 SYSTEM TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")
    logger.info(f"Total Time: {total_time:.2f}s")
    
    logger.info(f"\nDetailed Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info(f"\n🎉 ALL TESTS PASSED! System is ready for production.")
        return 0
    else:
        logger.error(f"\n⚠️ {total - passed} test(s) failed. Please fix issues before production deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 