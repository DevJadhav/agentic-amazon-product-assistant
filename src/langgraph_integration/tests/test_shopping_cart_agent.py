"""
Unit tests for Shopping Cart Agent workflow nodes.
Tests each workflow node independently and the complete agent workflow.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from ..core.shopping_cart_agent import ShoppingCartAgent
from ..core.state_schemas import AgentState, create_initial_state
from ..state.shopping_cart_manager import ShoppingCartManager


class TestShoppingCartAgent:
    """Test suite for Shopping Cart Agent."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager for testing."""
        manager = Mock(spec=ShoppingCartManager)
        
        # Mock cart operations
        manager.add_item.return_value = {
            "success": True,
            "message": "Added item to cart",
            "item": {
                "product_id": "test_product",
                "product_title": "Test Product",
                "quantity": 1,
                "product_price": 29.99
            },
            "action": "added"
        }
        
        manager.remove_item.return_value = {
            "success": True,
            "message": "Removed item from cart",
            "item": {
                "product_id": "test_product",
                "product_title": "Test Product",
                "quantity": 0
            },
            "action": "removed",
            "removed_completely": True
        }
        
        manager.get_cart_contents.return_value = [
            {
                "product_id": "test_product",
                "product_title": "Test Product",
                "quantity": 1,
                "product_price": 29.99,
                "product_image_url": "http://example.com/image.jpg",
                "product_metadata": {},
                "added_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "subtotal": 29.99
            }
        ]
        
        manager.get_cart_summary.return_value = {
            "total_items": 1,
            "total_value": 29.99,
            "unique_products": 1,
            "is_empty": False
        }
        
        return manager
    
    @pytest.fixture
    def cart_agent(self, mock_cart_manager):
        """Create Shopping Cart Agent for testing."""
        config = {
            "max_tool_calls": 5,
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini"
        }
        return ShoppingCartAgent(config, mock_cart_manager)
    
    @pytest.fixture
    def sample_state(self):
        """Create sample agent state for testing."""
        return create_initial_state(
            session_id="test_session",
            query="add this product to my cart",
            selected_product_for_cart={
                "product_id": "test_product",
                "title": "Test Product",
                "price": 29.99,
                "image_url": "http://example.com/image.jpg",
                "metadata": {"category": "electronics"}
            }
        )
    
    def test_agent_initialization(self, mock_cart_manager):
        """Test Shopping Cart Agent initialization."""
        config = {"max_tool_calls": 3}
        agent = ShoppingCartAgent(config, mock_cart_manager)
        
        assert agent.cart_manager == mock_cart_manager
        assert len(agent.tools) == 4  # add, remove, list, clear
        assert agent.max_tool_calls == 3
        assert agent.tool_integration is not None
        
        # Check available tools through integration
        available_tools = agent.tool_integration.get_available_tools()
        assert "add_to_cart" in available_tools
        assert "remove_from_cart" in available_tools
        assert "list_cart" in available_tools
        assert "clear_cart" in available_tools
    
    def test_create_graph(self, cart_agent):
        """Test workflow graph creation."""
        graph = cart_agent.create_graph()
        
        # Check that all nodes are present
        nodes = list(graph.nodes.keys())
        expected_nodes = [
            "analyze_cart_request",
            "execute_cart_operation", 
            "generate_cart_response",
            "update_cart_state"
        ]
        
        for node in expected_nodes:
            assert node in nodes
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_add_operation(self, cart_agent, sample_state):
        """Test cart request analysis for add operation."""
        
        # Test add operation analysis
        state = sample_state.copy()
        state["current_query"] = "add this product to my cart"
        
        result = await cart_agent._analyze_cart_request(state)
        
        assert result["current_step"] == "analyze_cart_request"
        assert result["cart_operation"] == "add"
        assert result["cart_operation_success"] is True
        assert "product_id" in result["cart_operation_params"]
        assert result["cart_operation_params"]["product_id"] == "test_product"
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_remove_operation(self, cart_agent, sample_state):
        """Test cart request analysis for remove operation."""
        
        state = sample_state.copy()
        state["current_query"] = "remove this item from my cart"
        
        result = await cart_agent._analyze_cart_request(state)
        
        assert result["cart_operation"] == "remove"
        assert result["cart_operation_success"] is True
        assert "product_id" in result["cart_operation_params"]
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_list_operation(self, cart_agent, sample_state):
        """Test cart request analysis for list operation."""
        
        state = sample_state.copy()
        state["current_query"] = "show me what's in my cart"
        
        result = await cart_agent._analyze_cart_request(state)
        
        assert result["cart_operation"] == "list"
        assert result["cart_operation_success"] is True
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_clear_operation(self, cart_agent, sample_state):
        """Test cart request analysis for clear operation."""
        
        state = sample_state.copy()
        state["current_query"] = "clear my cart"
        
        result = await cart_agent._analyze_cart_request(state)
        
        assert result["cart_operation"] == "clear"
        assert result["cart_operation_success"] is True
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_quantity_extraction(self, cart_agent, sample_state):
        """Test quantity extraction from user queries."""
        
        state = sample_state.copy()
        state["current_query"] = "add 3 items to my cart"
        
        result = await cart_agent._analyze_cart_request(state)
        
        assert result["cart_operation"] == "add"
        assert result["cart_operation_params"].get("quantity") == 3
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_missing_product(self, cart_agent, sample_state):
        """Test cart request analysis with missing product information."""
        
        state = sample_state.copy()
        state["current_query"] = "add to cart"
        state["selected_product_for_cart"] = None
        
        result = await cart_agent._analyze_cart_request(state)
        
        assert result["cart_operation"] == "add"
        assert result["cart_operation_success"] is False
        assert "missing product information" in result["cart_operation_message"]
    
    @pytest.mark.asyncio
    async def test_execute_cart_operation_add_success(self, cart_agent, sample_state):
        """Test successful add operation execution."""
        
        state = sample_state.copy()
        state["cart_operation"] = "add"
        state["cart_operation_success"] = True
        state["cart_operation_params"] = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 1,
            "price": 29.99
        }
        
        with patch.object(cart_agent, '_call_cart_tool_integrated', new_callable=AsyncMock) as mock_call:
            from ..core.tool_integration import ToolCallResult
            mock_call.return_value = ToolCallResult(
                tool_name="add_to_cart",
                success=True,
                result={
                    "success": True,
                    "message": "Added to cart",
                    "cart_updated": True,
                    "item": {"product_id": "test_product", "quantity": 1}
                },
                execution_time=0.5
            )
            
            result = await cart_agent._execute_cart_operation(state)
            
            assert result["current_step"] == "execute_cart_operation"
            assert result["cart_operation_success"] is True
            assert result["cart_updated"] is True
            assert len(result["tool_calls"]) == 1
            
            mock_call.assert_called_once_with("add", state["cart_operation_params"])
    
    @pytest.mark.asyncio
    async def test_execute_cart_operation_failed_analysis(self, cart_agent, sample_state):
        """Test operation execution when analysis failed."""
        
        state = sample_state.copy()
        state["cart_operation"] = "add"
        state["cart_operation_success"] = False
        state["cart_operation_message"] = "Analysis failed"
        
        result = await cart_agent._execute_cart_operation(state)
        
        assert result["cart_operation_result"]["success"] is False
        assert "analysis failure" in result["cart_operation_result"]["error"]
    
    @pytest.mark.asyncio
    async def test_execute_cart_operation_tool_error(self, cart_agent, sample_state):
        """Test operation execution with tool error."""
        
        state = sample_state.copy()
        state["cart_operation"] = "add"
        state["cart_operation_success"] = True
        state["cart_operation_params"] = {"product_id": "test_product"}
        
        with patch.object(cart_agent, '_call_cart_tool_integrated', new_callable=AsyncMock) as mock_call:
            from ..core.tool_integration import ToolCallResult
            mock_call.return_value = ToolCallResult(
                tool_name="add_to_cart",
                success=True,
                result={
                    "success": False,
                    "error": "Tool execution failed",
                    "message": "Failed to add item"
                },
                execution_time=0.2
            )
            
            result = await cart_agent._execute_cart_operation(state)
            
            assert result["cart_operation_success"] is False
            assert result["cart_operation_result"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_generate_cart_response_add_success(self, cart_agent, sample_state):
        """Test response generation for successful add operation."""
        
        state = sample_state.copy()
        state["cart_operation"] = "add"
        state["cart_operation_result"] = {
            "success": True,
            "message": "Added to cart",
            "item": {
                "product_title": "Test Product",
                "quantity": 1
            },
            "action": "added"
        }
        state["cart_updated"] = True
        
        with patch.object(cart_agent, '_get_current_cart_status', new_callable=AsyncMock) as mock_status:
            mock_status.return_value = "Your cart now has 1 item."
            
            result = await cart_agent._generate_cart_response(state)
            
            assert result["current_step"] == "generate_cart_response"
            assert "Test Product" in result["final_response"]
            assert "added" in result["final_response"]
            assert "Your cart now has 1 item" in result["final_response"]
    
    @pytest.mark.asyncio
    async def test_generate_cart_response_list_empty(self, cart_agent, sample_state):
        """Test response generation for empty cart list."""
        
        state = sample_state.copy()
        state["cart_operation"] = "list"
        state["cart_operation_result"] = {
            "success": True,
            "is_empty": True,
            "cart_items": [],
            "message": "Your cart is empty"
        }
        
        result = await cart_agent._generate_cart_response(state)
        
        assert "empty" in result["final_response"].lower()
    
    @pytest.mark.asyncio
    async def test_generate_cart_response_list_with_items(self, cart_agent, sample_state):
        """Test response generation for cart list with items."""
        
        state = sample_state.copy()
        state["cart_operation"] = "list"
        state["cart_operation_result"] = {
            "success": True,
            "is_empty": False,
            "cart_items": [
                {
                    "product_title": "Test Product",
                    "quantity": 2,
                    "price_per_unit": 29.99,
                    "subtotal": 59.98
                }
            ],
            "cart_summary": {
                "total_value": 59.98
            }
        }
        
        result = await cart_agent._generate_cart_response(state)
        
        response = result["final_response"]
        assert "Test Product" in response
        assert "2 x" in response
        assert "$59.98" in response
        assert "Total:" in response
    
    @pytest.mark.asyncio
    async def test_generate_cart_response_error(self, cart_agent, sample_state):
        """Test response generation for operation error."""
        
        state = sample_state.copy()
        state["cart_operation"] = "add"
        state["cart_operation_result"] = {
            "success": False,
            "error": "Product not found",
            "message": "Failed to add item"
        }
        
        result = await cart_agent._generate_cart_response(state)
        
        assert "couldn't complete" in result["final_response"].lower()
        assert "Failed to add item" in result["final_response"]
    
    @pytest.mark.asyncio
    async def test_update_cart_state(self, cart_agent, sample_state, mock_cart_manager):
        """Test cart state update node."""
        
        state = sample_state.copy()
        
        result = await cart_agent._update_cart_state(state)
        
        assert result["current_step"] == "update_cart_state"
        assert result["workflow_status"] == "completed"
        assert "current_cart_contents" in result
        assert "cart_item_count" in result
        assert "cart_total" in result
        
        # Verify cart manager was called
        mock_cart_manager.get_cart_contents.assert_called_once_with("test_session")
        mock_cart_manager.get_cart_summary.assert_called_once_with("test_session")
    
    def test_classify_cart_operation(self, cart_agent):
        """Test cart operation classification from queries."""
        
        test_cases = [
            ("add this to my cart", "add"),
            ("put this item in cart", "add"),
            ("remove this from cart", "remove"),
            ("delete this item", "remove"),
            ("show my cart", "list"),
            ("what's in my cart", "list"),
            ("clear my cart", "clear"),
            ("empty my cart", "clear"),
            ("unknown query", "list")  # default
        ]
        
        for query, expected_operation in test_cases:
            result = cart_agent._classify_cart_operation(query)
            assert result == expected_operation, f"Query '{query}' should classify as '{expected_operation}'"
    
    def test_extract_quantity_from_query(self, cart_agent):
        """Test quantity extraction from user queries."""
        
        test_cases = [
            ("add 3 items", 3),
            ("remove 2 pieces", 2),
            ("put 5 units in cart", 5),
            ("add 1 of this", 1),
            ("no quantity here", None),
            ("add zero items", None)  # Invalid quantity
        ]
        
        for query, expected_quantity in test_cases:
            result = cart_agent._extract_quantity_from_query(query)
            assert result == expected_quantity, f"Query '{query}' should extract quantity {expected_quantity}"
    
    def test_validate_cart_operation_add_valid(self, cart_agent):
        """Test validation of valid add operation."""
        
        params = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 2
        }
        
        result = cart_agent._validate_cart_operation("add", params)
        
        assert result["valid"] is True
    
    def test_validate_cart_operation_add_missing_product(self, cart_agent):
        """Test validation of add operation with missing product."""
        
        params = {"quantity": 1}
        
        result = cart_agent._validate_cart_operation("add", params)
        
        assert result["valid"] is False
        assert "missing product information" in result["message"]
    
    def test_validate_cart_operation_add_invalid_quantity(self, cart_agent):
        """Test validation of add operation with invalid quantity."""
        
        params = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 0
        }
        
        result = cart_agent._validate_cart_operation("add", params)
        
        assert result["valid"] is False
        assert "Invalid quantity" in result["message"]
    
    def test_validate_cart_operation_remove_missing_product(self, cart_agent):
        """Test validation of remove operation with missing product."""
        
        params = {}
        
        result = cart_agent._validate_cart_operation("remove", params)
        
        assert result["valid"] is False
        assert "missing product information" in result["message"]
    
    def test_validate_cart_operation_list_always_valid(self, cart_agent):
        """Test that list operation is always valid."""
        
        result = cart_agent._validate_cart_operation("list", {})
        
        assert result["valid"] is True
    
    def test_validate_cart_operation_clear_always_valid(self, cart_agent):
        """Test that clear operation is always valid."""
        
        result = cart_agent._validate_cart_operation("clear", {})
        
        assert result["valid"] is True
    
    @pytest.mark.asyncio
    async def test_call_cart_tool_integrated_add(self, cart_agent):
        """Test calling add cart tool through integration."""
        
        cart_agent.session_id = "test_session"
        
        params = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 1,
            "price": 29.99
        }
        
        result = await cart_agent._call_cart_tool_integrated("add", params)
        
        assert result.success is True
        assert result.tool_name == "add_to_cart"
        assert result.result["success"] is True
    
    @pytest.mark.asyncio
    async def test_call_cart_tool_integrated_unknown_operation(self, cart_agent):
        """Test calling cart tool with unknown operation through integration."""
        
        cart_agent.session_id = "test_session"
        
        result = await cart_agent._call_cart_tool_integrated("unknown", {})
        
        assert result.success is False
        assert "No tool available" in result.error
    
    @pytest.mark.asyncio
    async def test_call_cart_tool_integrated_list_operation(self, cart_agent):
        """Test calling list cart tool through integration."""
        
        cart_agent.session_id = "test_session"
        
        result = await cart_agent._call_cart_tool_integrated("list", {})
        
        assert result.success is True
        assert result.tool_name == "list_cart"
        assert result.result["success"] is True
    
    def test_create_operation_response_add_success(self, cart_agent):
        """Test creating response for successful add operation."""
        
        result = {
            "success": True,
            "item": {
                "product_title": "Test Product",
                "quantity": 2
            },
            "action": "added"
        }
        
        response = cart_agent._create_operation_response("add", result)
        
        assert "added" in response.lower()
        assert "Test Product" in response
        assert "2 x" in response
    
    def test_create_operation_response_add_updated(self, cart_agent):
        """Test creating response for add operation that updated quantity."""
        
        result = {
            "success": True,
            "item": {
                "product_title": "Test Product",
                "quantity": 3
            },
            "action": "updated"
        }
        
        response = cart_agent._create_operation_response("add", result)
        
        assert "updated" in response.lower()
        assert "Test Product" in response
        assert "3" in response
    
    def test_create_operation_response_remove_complete(self, cart_agent):
        """Test creating response for complete item removal."""
        
        result = {
            "success": True,
            "item": {
                "product_title": "Test Product"
            },
            "removed_completely": True
        }
        
        response = cart_agent._create_operation_response("remove", result)
        
        assert "removed" in response.lower()
        assert "Test Product" in response
    
    def test_create_operation_response_error(self, cart_agent):
        """Test creating response for operation error."""
        
        result = {
            "success": False,
            "error": "Product not found",
            "message": "Item not found"
        }
        
        response = cart_agent._create_operation_response("add", result)
        
        assert "couldn't complete" in response.lower()
        assert "Item not found" in response
    
    @pytest.mark.asyncio
    async def test_get_current_cart_status_empty(self, cart_agent, mock_cart_manager):
        """Test getting cart status when cart is empty."""
        
        mock_cart_manager.get_cart_summary.return_value = {
            "is_empty": True,
            "total_items": 0,
            "total_value": 0.0
        }
        
        cart_agent.session_id = "test_session"
        status = await cart_agent._get_current_cart_status()
        
        assert "empty" in status.lower()
    
    @pytest.mark.asyncio
    async def test_get_current_cart_status_with_items(self, cart_agent, mock_cart_manager):
        """Test getting cart status with items."""
        
        mock_cart_manager.get_cart_summary.return_value = {
            "is_empty": False,
            "total_items": 2,
            "total_value": 59.98
        }
        
        cart_agent.session_id = "test_session"
        status = await cart_agent._get_current_cart_status()
        
        assert "2 items" in status
        assert "$59.98" in status
    
    def test_get_agent_info(self, cart_agent):
        """Test getting agent information."""
        
        info = cart_agent.get_agent_info()
        
        assert info["agent_type"] == "ShoppingCartAgent"
        assert "tools_available" in info
        assert len(info["tools_available"]) == 4
        assert "add_to_cart" in info["tools_available"]
        assert info["tool_count"] == 4
        assert info["cart_manager_connected"] is True
        assert "supported_operations" in info


class TestShoppingCartAgentIntegration:
    """Integration tests for complete Shopping Cart Agent workflow."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager for integration testing."""
        manager = Mock(spec=ShoppingCartManager)
        
        # Mock successful operations
        manager.add_item.return_value = {
            "success": True,
            "message": "Added Test Product to cart",
            "item": {
                "product_id": "test_product",
                "product_title": "Test Product",
                "quantity": 1,
                "product_price": 29.99,
                "subtotal": 29.99
            },
            "action": "added"
        }
        
        manager.get_cart_contents.return_value = [
            {
                "product_id": "test_product",
                "product_title": "Test Product",
                "quantity": 1,
                "product_price": 29.99,
                "product_image_url": "http://example.com/image.jpg",
                "product_metadata": {},
                "added_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "subtotal": 29.99
            }
        ]
        
        manager.get_cart_summary.return_value = {
            "total_items": 1,
            "total_value": 29.99,
            "unique_products": 1,
            "is_empty": False
        }
        
        return manager
    
    @pytest.fixture
    def cart_agent(self, mock_cart_manager):
        """Create Shopping Cart Agent for integration testing."""
        config = {"max_tool_calls": 5}
        return ShoppingCartAgent(config, mock_cart_manager)
    
    @pytest.mark.asyncio
    async def test_complete_add_workflow(self, cart_agent, mock_cart_manager):
        """Test complete add to cart workflow."""
        
        # Create initial state
        state = create_initial_state(
            session_id="test_session",
            query="add this product to my cart",
            selected_product_for_cart={
                "product_id": "test_product",
                "title": "Test Product",
                "price": 29.99
            }
        )
        
        # Process through the agent
        result = await cart_agent.process_query(state)
        
        # Verify final state
        assert result["workflow_status"] == "completed"
        assert result["final_response"] is not None
        assert "Test Product" in result["final_response"]
        assert result["cart_updated"] is True
        assert result["cart_item_count"] == 1
        assert result["cart_total"] == 29.99
        
        # Verify cart manager was called
        mock_cart_manager.add_item.assert_called_once()
        mock_cart_manager.get_cart_contents.assert_called()
        mock_cart_manager.get_cart_summary.assert_called()
    
    @pytest.mark.asyncio
    async def test_complete_list_workflow(self, cart_agent, mock_cart_manager):
        """Test complete list cart workflow."""
        
        state = create_initial_state(
            session_id="test_session",
            query="show me my cart"
        )
        
        result = await cart_agent.process_query(state)
        
        assert result["workflow_status"] == "completed"
        assert result["final_response"] is not None
        assert "Test Product" in result["final_response"]
        
        # Verify cart manager was called for listing
        mock_cart_manager.get_cart_contents.assert_called()
        mock_cart_manager.get_cart_summary.assert_called()
    
    @pytest.mark.asyncio
    async def test_workflow_with_error_handling(self, cart_agent, mock_cart_manager):
        """Test workflow error handling."""
        
        # Mock cart manager to raise exception
        mock_cart_manager.add_item.side_effect = Exception("Database error")
        
        state = create_initial_state(
            session_id="test_session",
            query="add product to cart",
            selected_product_for_cart={
                "product_id": "test_product",
                "title": "Test Product"
            }
        )
        
        result = await cart_agent.process_query(state)
        
        # Should complete with error state
        assert result["workflow_status"] in ["completed", "error"]
        assert "error" in result["final_response"].lower()


if __name__ == "__main__":
    pytest.main([__file__])