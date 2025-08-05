"""
End-to-end integration tests for complete user journeys.
Tests complete workflows from intent classification to cart operations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from ..core.router.master_graph import MasterAgentGraph
from ..core.router.router_node import RouterNode
from ..core.router.intent_classifier import IntentClassifier, IntentResult
from ..core.router.clarification_handler import ClarificationHandler
from ..core.shopping_cart_agent import ShoppingCartAgent
from ..state.shopping_cart_manager import ShoppingCartManager
from ..tools.shopping_cart_tools import AddToCartTool, RemoveFromCartTool, ListCartTool
from ..core.state_schemas import create_initial_state, AgentState
from ..core.agent_builder import AgentGraphBuilder


class TestEndToEndCartWorkflows:
    """Test complete end-to-end cart workflows."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager for E2E tests."""
        manager = Mock(spec=ShoppingCartManager)
        
        # Mock cart operations with realistic responses
        manager.add_item.return_value = {
            "success": True,
            "message": "Added Gaming Laptop to cart",
            "item": {
                "id": "cart_item_1",
                "product_id": "LAPTOP_GAMING_001",
                "product_title": "Gaming Laptop",
                "quantity": 1,
                "product_price": 1299.99,
                "subtotal": 1299.99,
                "added_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            },
            "action": "added"
        }
        
        manager.get_cart_contents.return_value = [
            {
                "id": "cart_item_1",
                "product_id": "LAPTOP_GAMING_001",
                "product_title": "Gaming Laptop",
                "quantity": 1,
                "product_price": 1299.99,
                "subtotal": 1299.99,
                "product_image_url": "https://example.com/laptop.jpg",
                "product_metadata": {"category": "electronics", "brand": "TechBrand"},
                "added_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            }
        ]
        
        manager.get_cart_summary.return_value = {
            "session_id": "e2e_test_session",
            "total_items": 1,
            "total_value": 1299.99,
            "unique_products": 1,
            "is_empty": False,
            "last_updated": "2024-01-01T12:00:00"
        }
        
        return manager
    
    @pytest.fixture
    def mock_agent_builder(self):
        """Create mock agent builder for E2E tests."""
        builder = Mock(spec=AgentGraphBuilder)
        
        # Mock QA agent graph
        mock_qa_graph = Mock()
        mock_qa_result = {
            "final_response": "Here are the best gaming laptops based on your requirements...",
            "workflow_status": "completed",
            "session_id": "e2e_test_session",
            "conversation_turn": 1
        }
        mock_qa_graph.ainvoke = AsyncMock(return_value=mock_qa_result)
        builder.create_ambient_agent_graph.return_value = mock_qa_graph
        
        return builder
    
    @pytest.fixture
    def master_graph(self, mock_agent_builder, mock_cart_manager):
        """Create master graph for E2E testing."""
        config = {
            "router": {"confidence_threshold": 0.7},
            "classifier": {"confidence_threshold": 0.7},
            "clarification": {"max_clarification_attempts": 3}
        }
        
        # Create master graph with mocked dependencies
        graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        
        # Replace cart agent's cart manager with mock
        graph.shopping_cart_agent.cart_manager = mock_cart_manager
        if hasattr(graph.shopping_cart_agent, 'tools'):
            graph.shopping_cart_agent.tools = [
                AddToCartTool(cart_manager=mock_cart_manager),
                RemoveFromCartTool(cart_manager=mock_cart_manager),
                ListCartTool(cart_manager=mock_cart_manager)
            ]
        
        return graph
    
    @pytest.mark.asyncio
    async def test_complete_add_to_cart_journey(self, master_graph, mock_cart_manager):
        """Test complete user journey: intent classification -> cart agent -> add item."""
        
        # Create initial state with clear cart intent
        state = create_initial_state(
            session_id="e2e_test_session",
            query="I want to add this gaming laptop to my cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_GAMING_001",
                "title": "Gaming Laptop",
                "price": 1299.99,
                "image_url": "https://example.com/laptop.jpg",
                "metadata": {"category": "electronics", "brand": "TechBrand"}
            }
        )
        
        # Process through complete workflow
        result = await master_graph.process_query(state)
        
        # Verify end-to-end results
        assert result["workflow_status"] == "completed"
        assert result["final_response"] is not None
        assert "Gaming Laptop" in result["final_response"]
        assert "cart" in result["final_response"].lower()
        
        # Verify routing worked correctly
        assert "response_metadata" in result
        metadata = result["response_metadata"]
        assert metadata["agent_used"] == "shopping_cart_agent"
        assert metadata["routing_successful"] is True
        
        # Verify cart operations were called
        mock_cart_manager.add_item.assert_called_once()
        mock_cart_manager.get_cart_contents.assert_called()
        mock_cart_manager.get_cart_summary.assert_called()
        
        # Verify cart state in response
        assert result["cart_updated"] is True
        assert result["cart_item_count"] == 1
        assert result["cart_total"] == 1299.99
    
    @pytest.mark.asyncio
    async def test_complete_qa_to_cart_journey(self, master_graph, mock_cart_manager, mock_agent_builder):
        """Test journey: QA question -> product info -> add to cart."""
        
        # Step 1: Ask QA question
        qa_state = create_initial_state(
            session_id="e2e_test_session",
            query="What are the best gaming laptops under $1500?"
        )
        
        qa_result = await master_graph.process_query(qa_state)
        
        # Verify QA response
        assert qa_result["workflow_status"] == "completed"
        assert qa_result["response_metadata"]["agent_used"] == "qa_agent"
        assert "gaming laptops" in qa_result["final_response"]
        
        # Step 2: Follow up with cart addition
        cart_state = create_initial_state(
            session_id="e2e_test_session",
            query="Add the first laptop to my cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_GAMING_001",
                "title": "Gaming Laptop",
                "price": 1299.99
            },
            conversation_turn=2
        )
        
        cart_result = await master_graph.process_query(cart_state)
        
        # Verify cart addition
        assert cart_result["workflow_status"] == "completed"
        assert cart_result["response_metadata"]["agent_used"] == "shopping_cart_agent"
        assert cart_result["cart_updated"] is True
        
        # Verify both agents were used in sequence
        mock_agent_builder.create_ambient_agent_graph.assert_called()
        mock_cart_manager.add_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_complete_cart_management_journey(self, master_graph, mock_cart_manager):
        """Test complete cart management: add -> list -> remove."""
        
        session_id = "e2e_cart_management"
        
        # Step 1: Add item to cart
        add_state = create_initial_state(
            session_id=session_id,
            query="add gaming laptop to cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_GAMING_001",
                "title": "Gaming Laptop",
                "price": 1299.99
            }
        )
        
        add_result = await master_graph.process_query(add_state)
        assert add_result["cart_updated"] is True
        assert add_result["response_metadata"]["agent_used"] == "shopping_cart_agent"
        
        # Step 2: List cart contents
        mock_cart_manager.reset_mock()  # Reset call counts
        
        list_state = create_initial_state(
            session_id=session_id,
            query="show me my cart",
            conversation_turn=2
        )
        
        list_result = await master_graph.process_query(list_state)
        assert list_result["workflow_status"] == "completed"
        assert list_result["response_metadata"]["agent_used"] == "shopping_cart_agent"
        assert "Gaming Laptop" in list_result["final_response"]
        
        # Step 3: Remove item from cart
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Removed Gaming Laptop from cart",
            "item": {"product_id": "LAPTOP_GAMING_001", "quantity": 0},
            "action": "removed",
            "removed_completely": True
        }
        
        mock_cart_manager.get_cart_contents.return_value = []
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 0,
            "total_value": 0.0,
            "unique_products": 0,
            "is_empty": True
        }
        
        remove_state = create_initial_state(
            session_id=session_id,
            query="remove gaming laptop from cart",
            conversation_turn=3
        )
        
        remove_result = await master_graph.process_query(remove_state)
        assert remove_result["cart_updated"] is True
        assert remove_result["cart_item_count"] == 0
        assert "removed" in remove_result["final_response"].lower()
        
        # Verify all cart operations were called
        mock_cart_manager.add_item.assert_called()
        mock_cart_manager.get_cart_contents.assert_called()
        mock_cart_manager.remove_item.assert_called()
    
    @pytest.mark.asyncio
    async def test_clarification_to_resolution_journey(self, master_graph, mock_cart_manager):
        """Test journey: unclear intent -> clarification -> resolution."""
        
        session_id = "e2e_clarification"
        
        # Step 1: Submit unclear query
        unclear_state = create_initial_state(
            session_id=session_id,
            query="I want this"
        )
        
        unclear_result = await master_graph.process_query(unclear_state)
        
        # Should trigger clarification
        assert unclear_result["workflow_status"] == "completed"
        assert "response_metadata" in unclear_result
        metadata = unclear_result["response_metadata"]
        assert metadata.get("clarification_requested") is True
        assert "clarify" in unclear_result["final_response"].lower() or "what" in unclear_result["final_response"].lower()
        
        # Step 2: Provide clarification
        clarification_state = create_initial_state(
            session_id=session_id,
            query="I want to add it to my cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_GAMING_001",
                "title": "Gaming Laptop",
                "price": 1299.99
            },
            conversation_turn=2
        )
        
        resolved_result = await master_graph.process_query(clarification_state)
        
        # Should resolve to cart operation
        assert resolved_result["workflow_status"] == "completed"
        assert resolved_result["response_metadata"]["agent_used"] == "shopping_cart_agent"
        assert resolved_result["cart_updated"] is True
        
        # Verify cart manager was called after clarification
        mock_cart_manager.add_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_recovery_journey(self, master_graph, mock_cart_manager):
        """Test error recovery in end-to-end workflow."""
        
        # Mock cart manager to fail initially, then succeed
        mock_cart_manager.add_item.side_effect = [
            {"success": False, "error": "Database temporarily unavailable"},
            {
                "success": True,
                "message": "Added Gaming Laptop to cart",
                "item": {"product_id": "LAPTOP_GAMING_001", "quantity": 1},
                "action": "added"
            }
        ]
        
        session_id = "e2e_error_recovery"
        
        # First attempt - should handle error gracefully
        error_state = create_initial_state(
            session_id=session_id,
            query="add gaming laptop to cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_GAMING_001",
                "title": "Gaming Laptop",
                "price": 1299.99
            }
        )
        
        error_result = await master_graph.process_query(error_state)
        
        # Should complete with error message
        assert error_result["workflow_status"] == "completed"
        assert error_result["cart_updated"] is False
        assert "error" in error_result["final_response"].lower() or "trouble" in error_result["final_response"].lower()
        
        # Second attempt - should succeed
        retry_state = create_initial_state(
            session_id=session_id,
            query="try adding gaming laptop to cart again",
            selected_product_for_cart={
                "product_id": "LAPTOP_GAMING_001",
                "title": "Gaming Laptop",
                "price": 1299.99
            },
            conversation_turn=2
        )
        
        success_result = await master_graph.process_query(retry_state)
        
        # Should succeed on retry
        assert success_result["workflow_status"] == "completed"
        assert success_result["cart_updated"] is True
        assert "added" in success_result["final_response"].lower()
        
        # Verify both attempts called cart manager
        assert mock_cart_manager.add_item.call_count == 2


class TestDualToolArchitectureIntegration:
    """Test integration of dual tool architecture (MCP vs function calling)."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        manager = Mock(spec=ShoppingCartManager)
        manager.add_item.return_value = {
            "success": True,
            "message": "Added item to cart",
            "item": {"product_id": "TEST_PRODUCT", "quantity": 1},
            "action": "added"
        }
        return manager
    
    @pytest.fixture
    def mock_agent_builder(self):
        """Create mock agent builder."""
        builder = Mock(spec=AgentGraphBuilder)
        
        # Mock QA agent that uses MCP tools
        mock_qa_graph = Mock()
        mock_qa_result = {
            "final_response": "Product information retrieved via MCP tools",
            "workflow_status": "completed",
            "tool_calls": [
                {"tool_name": "vector_search_mcp", "tool_type": "mcp"},
                {"tool_name": "product_analysis_mcp", "tool_type": "mcp"}
            ]
        }
        mock_qa_graph.ainvoke = AsyncMock(return_value=mock_qa_result)
        builder.create_ambient_agent_graph.return_value = mock_qa_graph
        
        return builder
    
    @pytest.mark.asyncio
    async def test_mcp_vs_function_calling_integration(self, mock_agent_builder, mock_cart_manager):
        """Test that QA agent uses MCP tools while cart agent uses function calling."""
        
        config = {"router": {"confidence_threshold": 0.7}}
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager
        
        session_id = "dual_tool_test"
        
        # Test QA agent with MCP tools
        qa_state = create_initial_state(
            session_id=session_id,
            query="what are the specifications of this laptop?"
        )
        
        qa_result = await master_graph.process_query(qa_state)
        
        # Verify QA agent was used
        assert qa_result["response_metadata"]["agent_used"] == "qa_agent"
        assert "MCP tools" in qa_result["final_response"]
        
        # Verify MCP tools were called
        mock_agent_builder.create_ambient_agent_graph.assert_called()
        
        # Test cart agent with function calling
        cart_state = create_initial_state(
            session_id=session_id,
            query="add this laptop to my cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_001",
                "title": "Test Laptop",
                "price": 999.99
            },
            conversation_turn=2
        )
        
        cart_result = await master_graph.process_query(cart_state)
        
        # Verify cart agent was used with function calling
        assert cart_result["response_metadata"]["agent_used"] == "shopping_cart_agent"
        assert cart_result["cart_updated"] is True
        
        # Verify function calling tools were used (not MCP)
        mock_cart_manager.add_item.assert_called_once()
        
        # Verify both tool types coexist
        assert qa_result["workflow_status"] == "completed"
        assert cart_result["workflow_status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_tool_isolation_and_independence(self, mock_agent_builder, mock_cart_manager):
        """Test that MCP and function calling tools operate independently."""
        
        config = {"router": {"confidence_threshold": 0.7}}
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager
        
        session_id = "tool_isolation_test"
        
        # Simulate concurrent requests using different tool types
        qa_state = create_initial_state(
            session_id=session_id + "_qa",
            query="analyze this product's reviews"
        )
        
        cart_state = create_initial_state(
            session_id=session_id + "_cart",
            query="add product to cart",
            selected_product_for_cart={
                "product_id": "PRODUCT_001",
                "title": "Test Product"
            }
        )
        
        # Process both concurrently
        qa_task = master_graph.process_query(qa_state)
        cart_task = master_graph.process_query(cart_state)
        
        qa_result, cart_result = await asyncio.gather(qa_task, cart_task)
        
        # Both should complete successfully
        assert qa_result["workflow_status"] == "completed"
        assert cart_result["workflow_status"] == "completed"
        
        # Verify correct tool types were used
        assert qa_result["response_metadata"]["agent_used"] == "qa_agent"
        assert cart_result["response_metadata"]["agent_used"] == "shopping_cart_agent"
        
        # Verify no cross-contamination
        mock_agent_builder.create_ambient_agent_graph.assert_called()
        mock_cart_manager.add_item.assert_called_once()


class TestDatabasePersistenceIntegration:
    """Test database persistence and session isolation."""
    
    @pytest.fixture
    def mock_cart_manager_with_persistence(self):
        """Create mock cart manager that simulates persistence."""
        manager = Mock(spec=ShoppingCartManager)
        
        # Simulate persistent storage
        self.cart_storage = {}
        
        def mock_add_item(session_id, product_id, product_title, quantity=1, **kwargs):
            if session_id not in self.cart_storage:
                self.cart_storage[session_id] = {}
            
            if product_id in self.cart_storage[session_id]:
                # Update existing item
                self.cart_storage[session_id][product_id]["quantity"] += quantity
                action = "updated"
            else:
                # Add new item
                self.cart_storage[session_id][product_id] = {
                    "product_id": product_id,
                    "product_title": product_title,
                    "quantity": quantity,
                    "product_price": kwargs.get("price", 0.0)
                }
                action = "added"
            
            return {
                "success": True,
                "message": f"{action.title()} {product_title}",
                "item": self.cart_storage[session_id][product_id],
                "action": action
            }
        
        def mock_get_cart_contents(session_id):
            return list(self.cart_storage.get(session_id, {}).values())
        
        def mock_get_cart_summary(session_id):
            items = self.cart_storage.get(session_id, {})
            total_items = sum(item["quantity"] for item in items.values())
            total_value = sum(item["quantity"] * item["product_price"] for item in items.values())
            
            return {
                "session_id": session_id,
                "total_items": total_items,
                "total_value": total_value,
                "unique_products": len(items),
                "is_empty": len(items) == 0
            }
        
        manager.add_item.side_effect = mock_add_item
        manager.get_cart_contents.side_effect = mock_get_cart_contents
        manager.get_cart_summary.side_effect = mock_get_cart_summary
        
        return manager
    
    @pytest.mark.asyncio
    async def test_session_isolation(self, mock_cart_manager_with_persistence):
        """Test that cart data is properly isolated between sessions."""
        
        config = {"router": {"confidence_threshold": 0.7}}
        mock_agent_builder = Mock()
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager_with_persistence
        
        # Session 1: Add laptop
        session1_state = create_initial_state(
            session_id="session_1",
            query="add laptop to cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_001",
                "title": "Laptop",
                "price": 1000.0
            }
        )
        
        session1_result = await master_graph.process_query(session1_state)
        assert session1_result["cart_updated"] is True
        assert session1_result["cart_item_count"] == 1
        
        # Session 2: Add phone
        session2_state = create_initial_state(
            session_id="session_2",
            query="add phone to cart",
            selected_product_for_cart={
                "product_id": "PHONE_001",
                "title": "Phone",
                "price": 800.0
            }
        )
        
        session2_result = await master_graph.process_query(session2_state)
        assert session2_result["cart_updated"] is True
        assert session2_result["cart_item_count"] == 1
        
        # Verify session isolation - each session should only see its own items
        session1_list_state = create_initial_state(
            session_id="session_1",
            query="show my cart",
            conversation_turn=2
        )
        
        session1_list_result = await master_graph.process_query(session1_list_state)
        assert "Laptop" in session1_list_result["final_response"]
        assert "Phone" not in session1_list_result["final_response"]
        
        session2_list_state = create_initial_state(
            session_id="session_2",
            query="show my cart",
            conversation_turn=2
        )
        
        session2_list_result = await master_graph.process_query(session2_list_state)
        assert "Phone" in session2_list_result["final_response"]
        assert "Laptop" not in session2_list_result["final_response"]
    
    @pytest.mark.asyncio
    async def test_cart_persistence_across_requests(self, mock_cart_manager_with_persistence):
        """Test that cart data persists across multiple requests in same session."""
        
        config = {"router": {"confidence_threshold": 0.7}}
        mock_agent_builder = Mock()
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager_with_persistence
        
        session_id = "persistence_test_session"
        
        # Request 1: Add first item
        add1_state = create_initial_state(
            session_id=session_id,
            query="add laptop to cart",
            selected_product_for_cart={
                "product_id": "LAPTOP_001",
                "title": "Gaming Laptop",
                "price": 1200.0
            }
        )
        
        add1_result = await master_graph.process_query(add1_state)
        assert add1_result["cart_item_count"] == 1
        assert add1_result["cart_total"] == 1200.0
        
        # Request 2: Add second item
        add2_state = create_initial_state(
            session_id=session_id,
            query="add mouse to cart",
            selected_product_for_cart={
                "product_id": "MOUSE_001",
                "title": "Gaming Mouse",
                "price": 80.0
            },
            conversation_turn=2
        )
        
        add2_result = await master_graph.process_query(add2_state)
        assert add2_result["cart_item_count"] == 2
        assert add2_result["cart_total"] == 1280.0
        
        # Request 3: List cart - should show both items
        list_state = create_initial_state(
            session_id=session_id,
            query="show my cart",
            conversation_turn=3
        )
        
        list_result = await master_graph.process_query(list_state)
        assert "Gaming Laptop" in list_result["final_response"]
        assert "Gaming Mouse" in list_result["final_response"]
        assert list_result["cart_item_count"] == 2
        assert list_result["cart_total"] == 1280.0


class TestPerformanceBenchmarks:
    """Performance benchmarks comparing system with and without routing overhead."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for performance testing."""
        mock_cart_manager = Mock(spec=ShoppingCartManager)
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added item",
            "item": {"product_id": "PERF_TEST", "quantity": 1},
            "action": "added"
        }
        mock_cart_manager.get_cart_contents.return_value = []
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 0,
            "total_value": 0.0,
            "is_empty": True
        }
        
        mock_agent_builder = Mock()
        mock_qa_graph = Mock()
        mock_qa_graph.ainvoke = AsyncMock(return_value={
            "final_response": "QA response",
            "workflow_status": "completed"
        })
        mock_agent_builder.create_ambient_agent_graph.return_value = mock_qa_graph
        
        return mock_cart_manager, mock_agent_builder
    
    @pytest.mark.asyncio
    async def test_routing_performance_overhead(self, mock_components):
        """Test performance overhead of routing system."""
        mock_cart_manager, mock_agent_builder = mock_components
        
        config = {"router": {"confidence_threshold": 0.7}}
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager
        
        # Test queries
        test_queries = [
            "add laptop to cart",
            "what are the features of this phone?",
            "remove item from cart",
            "show me product reviews",
            "list my cart contents"
        ]
        
        import time
        
        # Measure routing performance
        start_time = time.time()
        
        for i, query in enumerate(test_queries * 10):  # 50 total queries
            state = create_initial_state(
                session_id=f"perf_session_{i}",
                query=query,
                selected_product_for_cart={
                    "product_id": f"PRODUCT_{i}",
                    "title": f"Product {i}"
                } if "add" in query else None
            )
            
            result = await master_graph.process_query(state)
            assert result["workflow_status"] == "completed"
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_query = total_time / 50
        
        # Performance assertions
        assert total_time < 10.0, f"Total time too high: {total_time} seconds"
        assert avg_time_per_query < 0.2, f"Average time per query too high: {avg_time_per_query} seconds"
        
        # Log performance metrics
        print(f"\nPerformance Metrics:")
        print(f"Total time for 50 queries: {total_time:.2f} seconds")
        print(f"Average time per query: {avg_time_per_query:.3f} seconds")
        print(f"Queries per second: {50/total_time:.1f}")
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, mock_components):
        """Test performance under concurrent load."""
        mock_cart_manager, mock_agent_builder = mock_components
        
        config = {"router": {"confidence_threshold": 0.7}}
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager
        
        # Create concurrent requests
        concurrent_states = [
            create_initial_state(
                session_id=f"concurrent_session_{i}",
                query=f"add product {i} to cart",
                selected_product_for_cart={
                    "product_id": f"CONCURRENT_PRODUCT_{i}",
                    "title": f"Concurrent Product {i}"
                }
            )
            for i in range(20)
        ]
        
        import time
        start_time = time.time()
        
        # Process all requests concurrently
        tasks = [master_graph.process_query(state) for state in concurrent_states]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify all requests completed successfully
        assert len(results) == 20
        for result in results:
            assert result["workflow_status"] == "completed"
        
        # Performance assertions for concurrent processing
        assert concurrent_time < 5.0, f"Concurrent processing too slow: {concurrent_time} seconds"
        
        print(f"\nConcurrent Performance Metrics:")
        print(f"Time for 20 concurrent requests: {concurrent_time:.2f} seconds")
        print(f"Concurrent throughput: {20/concurrent_time:.1f} requests/second")


class TestErrorScenariosAndRecovery:
    """Test error scenarios and recovery mechanisms across all components."""
    
    @pytest.fixture
    def failing_cart_manager(self):
        """Create cart manager that fails in various ways."""
        manager = Mock(spec=ShoppingCartManager)
        
        # Configure different failure modes
        manager.add_item.side_effect = [
            Exception("Database connection failed"),  # First call fails
            {"success": False, "error": "Validation failed"},  # Second call returns error
            {  # Third call succeeds
                "success": True,
                "message": "Added item after retry",
                "item": {"product_id": "RETRY_PRODUCT", "quantity": 1},
                "action": "added"
            }
        ]
        
        return manager
    
    @pytest.fixture
    def failing_agent_builder(self):
        """Create agent builder that fails in various ways."""
        builder = Mock()
        
        # First call fails, second succeeds
        failing_graph = Mock()
        failing_graph.ainvoke = AsyncMock(side_effect=Exception("QA agent failed"))
        
        working_graph = Mock()
        working_graph.ainvoke = AsyncMock(return_value={
            "final_response": "QA response after recovery",
            "workflow_status": "completed"
        })
        
        builder.create_ambient_agent_graph.side_effect = [failing_graph, working_graph]
        
        return builder
    
    @pytest.mark.asyncio
    async def test_cart_error_recovery(self, failing_cart_manager):
        """Test recovery from cart operation errors."""
        
        config = {"router": {"confidence_threshold": 0.7}}
        mock_agent_builder = Mock()
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = failing_cart_manager
        
        session_id = "error_recovery_test"
        
        # First attempt - should handle exception
        state1 = create_initial_state(
            session_id=session_id,
            query="add laptop to cart",
            selected_product_for_cart={
                "product_id": "ERROR_PRODUCT_1",
                "title": "Error Product 1"
            }
        )
        
        result1 = await master_graph.process_query(state1)
        assert result1["workflow_status"] == "completed"
        assert result1["cart_updated"] is False
        assert "error" in result1["final_response"].lower() or "trouble" in result1["final_response"].lower()
        
        # Second attempt - should handle error response
        state2 = create_initial_state(
            session_id=session_id,
            query="add phone to cart",
            selected_product_for_cart={
                "product_id": "ERROR_PRODUCT_2",
                "title": "Error Product 2"
            },
            conversation_turn=2
        )
        
        result2 = await master_graph.process_query(state2)
        assert result2["workflow_status"] == "completed"
        assert result2["cart_updated"] is False
        assert "validation failed" in result2["final_response"].lower() or "error" in result2["final_response"].lower()
        
        # Third attempt - should succeed
        state3 = create_initial_state(
            session_id=session_id,
            query="add tablet to cart",
            selected_product_for_cart={
                "product_id": "RETRY_PRODUCT",
                "title": "Retry Product"
            },
            conversation_turn=3
        )
        
        result3 = await master_graph.process_query(state3)
        assert result3["workflow_status"] == "completed"
        assert result3["cart_updated"] is True
        assert "added" in result3["final_response"].lower()
        
        # Verify all attempts were made
        assert failing_cart_manager.add_item.call_count == 3
    
    @pytest.mark.asyncio
    async def test_qa_agent_error_recovery(self, failing_agent_builder):
        """Test recovery from QA agent errors."""
        
        config = {"router": {"confidence_threshold": 0.7}}
        mock_cart_manager = Mock()
        master_graph = MasterAgentGraph(config, agent_builder=failing_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager
        
        session_id = "qa_error_recovery"
        
        # First QA request - should fail and recover
        qa_state1 = create_initial_state(
            session_id=session_id,
            query="what are the features of this laptop?"
        )
        
        qa_result1 = await master_graph.process_query(qa_state1)
        assert qa_result1["workflow_status"] == "completed"
        assert "error" in qa_result1["final_response"].lower() or "trouble" in qa_result1["final_response"].lower()
        
        # Second QA request - should succeed
        qa_state2 = create_initial_state(
            session_id=session_id,
            query="tell me about this phone",
            conversation_turn=2
        )
        
        qa_result2 = await master_graph.process_query(qa_state2)
        assert qa_result2["workflow_status"] == "completed"
        assert "QA response after recovery" in qa_result2["final_response"]
        
        # Verify both attempts were made
        assert failing_agent_builder.create_ambient_agent_graph.call_count == 2
    
    @pytest.mark.asyncio
    async def test_routing_error_fallback(self):
        """Test fallback behavior when routing fails."""
        
        config = {"router": {"confidence_threshold": 0.7}}
        mock_agent_builder = Mock()
        mock_cart_manager = Mock()
        
        master_graph = MasterAgentGraph(config, agent_builder=mock_agent_builder)
        master_graph.shopping_cart_agent.cart_manager = mock_cart_manager
        
        # Mock router to fail
        with patch.object(master_graph.router_node, 'route_message', side_effect=Exception("Router failed")):
            
            # Mock QA agent as fallback
            mock_qa_graph = Mock()
            mock_qa_graph.ainvoke = AsyncMock(return_value={
                "final_response": "Fallback QA response",
                "workflow_status": "completed"
            })
            mock_agent_builder.create_ambient_agent_graph.return_value = mock_qa_graph
            
            state = create_initial_state(
                session_id="routing_error_test",
                query="test query that causes routing error"
            )
            
            result = await master_graph.process_query(state)
            
            # Should fallback to QA agent
            assert result["workflow_status"] == "completed"
            assert result["response_metadata"]["agent_used"] == "qa_agent"
            assert "Fallback QA response" in result["final_response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])