"""
Test script for LangGraph agent workflows.
Simple tests to verify agent functionality.
"""

import asyncio
import logging
from typing import Dict, Any

from .core.agent_builder import AgentGraphBuilder
from .core.state_schemas import create_initial_state
from .core.utils import generate_session_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ambient_agent():
    """Test the ambient style agent workflow."""
    
    print("🤖 Testing Ambient Style Agent...")
    
    try:
        # Create agent builder
        builder = AgentGraphBuilder()
        
        # Build ambient agent graph
        agent_graph = builder.create_ambient_agent_graph()
        
        # Create test state
        session_id = generate_session_id()
        test_query = "What are the best wireless headphones under $200?"
        
        initial_state = create_initial_state(
            session_id=session_id,
            query=test_query,
            max_products=3,
            max_reviews=2
        )
        
        print(f"Session ID: {session_id}")
        print(f"Test Query: {test_query}")
        print(f"Initial State: {initial_state['current_step']}")
        
        # Execute agent workflow
        print("\n🔄 Executing agent workflow...")
        
        result = await agent_graph.ainvoke(initial_state)
        
        print(f"\n✅ Agent workflow completed!")
        print(f"Final Step: {result.get('current_step', 'unknown')}")
        print(f"Workflow Status: {result.get('workflow_status', 'unknown')}")
        print(f"Products Found: {len(result.get('selected_products', []))}")
        print(f"Reviews Found: {len(result.get('review_summaries', []))}")
        
        # Show final response
        final_response = result.get('final_response', 'No response generated')
        print(f"\n📝 Final Response:")
        print(final_response)
        
        # Show any errors
        if result.get('error_state'):
            print(f"\n⚠️ Errors encountered: {result['error_state']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        logger.error(f"Agent test failed: {e}")
        return False


async def test_product_search_agent():
    """Test the product search specialized agent."""
    
    print("\n🔍 Testing Product Search Agent...")
    
    try:
        # Create agent builder
        builder = AgentGraphBuilder()
        
        # Build product search graph
        search_graph = builder.build_product_search_graph()
        
        # Create test state
        session_id = generate_session_id()
        test_query = "gaming laptop under $1000"
        
        initial_state = create_initial_state(
            session_id=session_id,
            query=test_query,
            max_products=5,
            max_reviews=3
        )
        
        print(f"Test Query: {test_query}")
        
        # Execute search workflow
        result = await search_graph.ainvoke(initial_state)
        
        print(f"✅ Product search completed!")
        print(f"Final Step: {result.get('current_step', 'unknown')}")
        print(f"Query Intent: {result.get('query_intent', 'unknown')}")
        print(f"Entities: {result.get('extracted_entities', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Product search test failed: {e}")
        logger.error(f"Product search test failed: {e}")
        return False


async def test_comparison_workflow():
    """Test the comparison workflow."""
    
    print("\n⚖️ Testing Comparison Workflow...")
    
    try:
        # Create agent builder
        builder = AgentGraphBuilder()
        
        # Build comparison graph
        comparison_graph = builder.build_comparison_graph()
        
        # Create test state
        session_id = generate_session_id()
        test_query = "compare iPhone 14 vs Samsung Galaxy S23"
        
        initial_state = create_initial_state(
            session_id=session_id,
            query=test_query,
            max_products=4,
            max_reviews=2
        )
        
        print(f"Test Query: {test_query}")
        
        # Execute comparison workflow
        result = await comparison_graph.ainvoke(initial_state)
        
        print(f"✅ Comparison workflow completed!")
        print(f"Final Step: {result.get('current_step', 'unknown')}")
        print(f"Final Response: {result.get('final_response', 'No response')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        logger.error(f"Comparison test failed: {e}")
        return False


def test_agent_builder_info():
    """Test agent builder information methods."""
    
    print("\n📋 Testing Agent Builder Info...")
    
    try:
        builder = AgentGraphBuilder()
        
        # Get available graphs
        available_graphs = builder.get_available_graphs()
        print(f"Available Graphs: {list(available_graphs.keys())}")
        
        for graph_name, description in available_graphs.items():
            print(f"  - {graph_name}: {description}")
        
        print("✅ Agent builder info test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent builder info test failed: {e}")
        return False


async def run_all_tests():
    """Run all agent tests."""
    
    print("🧪 Running LangGraph Agent Tests\n")
    
    results = []
    
    # Test agent builder info (synchronous)
    results.append(test_agent_builder_info())
    
    # Test async workflows
    results.append(await test_ambient_agent())
    results.append(await test_product_search_agent())
    results.append(await test_comparison_workflow())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All agent tests passed!")
    else:
        print("⚠️ Some tests failed - check logs for details")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_tests())