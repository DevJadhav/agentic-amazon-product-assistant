"""
Test script for LangGraph tools.
Simple tests to verify tool functionality.
"""

import logging
from typing import Dict, Any

from .vector_search_tool import VectorSearchTool
from .product_analysis_tool import ProductAnalysisTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_search_tool():
    """Test vector search tool functionality."""
    
    print("Testing Vector Search Tool...")
    
    tool = VectorSearchTool()
    
    # Test tool info
    info = tool.get_tool_info()
    print(f"Tool Info: {info}")
    
    # Test connection
    connection_test = tool.test_connection()
    print(f"Connection Test: {connection_test}")
    
    # Test search
    try:
        result = tool._run(
            query="wireless headphones",
            search_type="hybrid",
            max_products=3,
            max_reviews=2
        )
        print(f"Search Result: {result}")
        
        if "error" not in result:
            print("✅ Vector search tool working correctly")
        else:
            print(f"❌ Vector search tool error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Vector search tool exception: {e}")


def test_product_analysis_tool():
    """Test product analysis tool functionality."""
    
    print("\nTesting Product Analysis Tool...")
    
    tool = ProductAnalysisTool()
    
    # Create sample product data
    sample_products = [
        {
            "id": "1",
            "content": "Wireless Bluetooth headphones with noise-canceling technology",
            "metadata": {
                "title": "Sony WH-1000XM4 Headphones",
                "price": "299.99",
                "average_rating": "4.5",
                "rating_number": 1250,
                "store": "Amazon"
            }
        },
        {
            "id": "2", 
            "content": "Portable wireless speaker with waterproof design",
            "metadata": {
                "title": "JBL Flip 5 Speaker",
                "price": "89.99",
                "average_rating": "4.2",
                "rating_number": 850,
                "store": "Amazon"
            }
        }
    ]
    
    # Test comparison analysis
    try:
        result = tool._run(
            products=sample_products,
            analysis_type="comparison",
            include_summary=True
        )
        
        print(f"Comparison Analysis: {result}")
        
        if "error" not in result:
            print("✅ Product analysis tool working correctly")
        else:
            print(f"❌ Product analysis tool error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Product analysis tool exception: {e}")
    
    # Test pricing analysis
    try:
        result = tool._run(
            products=sample_products,
            analysis_type="pricing",
            include_summary=True
        )
        
        print(f"Pricing Analysis: {result}")
        
    except Exception as e:
        print(f"❌ Pricing analysis exception: {e}")


def run_all_tests():
    """Run all tool tests."""
    
    print("🧪 Running LangGraph Tools Tests\n")
    
    test_vector_search_tool()
    test_product_analysis_tool()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    run_all_tests()