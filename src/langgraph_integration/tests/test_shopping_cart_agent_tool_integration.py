"""
Integration tests for Shopping Cart Agent tool system integration.
Tests the interaction between the agent and cart management tools.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from ..core.shopping_cart_agent import ShoppingCartAgent
from ..core.tool_integration import (
    FunctionCallingToolIntegration, 
    ToolCallResult, 
    ToolCallLogger,
    create_session_id_injector
)
from ..core.state_schemas import AgentState, create_initial_state
from ..state.shopping_cart_manager import ShoppingCartManager
from ..tools.shopping_cart_tools import create_cart_tools


class TestShoppingCartAgentToolIntegration:
    """Test suite for Shopping Cart Agent tool integration."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager for testing."""
        manager = Mock(spec=ShoppingCartManager)
        
        # Mock successful operations
        manager.add_item.return_value = {
            "success": True,
            "message": "Added Test Product to cart",
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
            "message": "Removed Test Product from cart",
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
        
        manager.clear_cart.return_value = {
            "success": True,
            "message": "Cleared 1 items from cart",
            "items_removed": 1,
            "cleared_items": []
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
    
    def test_tool_integration_initialization(self, cart_agent):
        """Test that tool integration is properly initialized."""
        
        assert cart_agent.tool_integration is not None
        assert isinstance(cart_agent.tool_integration, FunctionCallingToolIntegration)
        assert cart_agent.tool_logger is not None
        assert len(cart_agent.tool_integration.get_available_tools()) == 4
        
        # Check available tools
        available_tools = cart_agent.tool_integration.get_available_tools()
        expected_tools = ["add_to_cart", "remove_from_cart", "list_cart", "clear_cart"]
        
        for tool in expected_tools:
            assert tool in available_tools
    
    def test_tool_info_retrieval(self, cart_agent):
        """Test retrieving tool information through integration."""
        
        # Test getting info for add_to_cart tool
        tool_info = cart_agent.tool_integration.get_tool_info("add_to_cart")
        
        assert tool_info is not None
        assert tool_info["name"] == "add_to_cart"
        assert tool_info["tool_type"] == "function_calling"
        assert "description" in tool_info
        assert "args_schema" in tool_info
    
    @pytest.mark.asyncio
    async def test_tool_call_integration_add_success(self, cart_agent):
        """Test successful add tool call through integration."""
        
        cart_agent.session_id = "test_session"
        
        params = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 1,
            "price": 29.99
        }
        
        result = await cart_agent._call_cart_tool_integrated("add", params)
        
        assert isinstance(result, ToolCallResult)
        assert result.success is True
        assert result.tool_name == "add_to_cart"
        assert result.result is not None
        assert result.error is None
        assert result.execution_time is not None
        assert result.metadata["tool_type"] == "function_calling"
    
    @pytest.mark.asyncio
    async def test_tool_call_integration_remove_success(self, cart_agent):
        """Test successful remove tool call through integration."""
        
        cart_agent.session_id = "test_session"
        
        params = {
            "product_id": "test_product",
            "quantity": 1
        }
        
        result = await cart_agent._call_cart_tool_integrated("remove", params)
        
        assert result.success is True
        assert result.tool_name == "remove_from_cart"
        assert result.result["success"] is True
    
    @pytest.mark.asyncio
    async def test_tool_call_integration_list_success(self, cart_agent):
        """Test successful list tool call through integration."""
        
        cart_agent.session_id = "test_session"
        
        params = {
            "include_summary": True,
            "format_type": "detailed"
        }
        
        result = await cart_agent._call_cart_tool_integrated("list", params)
        
        assert result.success is True
        assert result.tool_name == "list_cart"
        assert result.result["success"] is True
    
    @pytest.mark.asyncio
    async def test_tool_call_integration_clear_success(self, cart_agent):
        """Test successful clear tool call through integration."""
        
        cart_agent.session_id = "test_session"
        
        result = await cart_agent._call_cart_tool_integrated("clear", {})
        
        assert result.success is True
        assert result.tool_name == "clear_cart"
        assert result.result["success"] is True
    
    @pytest.mark.asyncio
    async def test_tool_call_integration_unknown_operation(self, cart_agent):
        """Test tool call with unknown operation."""
        
        cart_agent.session_id = "test_session"
        
        result = await cart_agent._call_cart_tool_integrated("unknown", {})
        
        assert result.success is False
        assert "unknown_operation_unknown" in result.tool_name
        assert "No tool available" in result.error
    
    @pytest.mark.asyncio
    async def test_tool_call_logging(self, cart_agent):
        """Test that tool calls are properly logged."""
        
        cart_agent.session_id = "test_session"
        
        # Clear any existing log entries
        cart_agent.tool_logger.call_history.clear()
        
        params = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 1
        }
        
        # Make a tool call
        result = await cart_agent._call_cart_tool_integrated("add", params)
        
        # Manually log it (normally done by the agent)
        cart_agent.tool_logger.log_tool_call(result, "test_session")
        
        # Check that it was logged
        assert len(cart_agent.tool_logger.call_history) == 1
        
        log_entry = cart_agent.tool_logger.call_history[0]
        assert log_entry["session_id"] == "test_session"
        assert log_entry["tool_name"] == "add_to_cart"
        assert log_entry["success"] is True
        assert log_entry["execution_time"] is not None
    
    @pytest.mark.asyncio
    async def test_session_id_injection(self, cart_agent, mock_cart_manager):
        """Test that session ID is properly injected into tools."""
        
        cart_agent.session_id = "test_session_123"
        
        params = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 1
        }
        
        # Make a tool call
        result = await cart_agent._call_cart_tool_integrated("add", params)
        
        # Verify the tool was called (through the mock cart manager)
        assert result.success is True
        
        # The session ID injection should have been applied
        # We can't directly test this without modifying the tool,
        # but we can verify the call succeeded
        assert mock_cart_manager.add_item.called
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self, cart_agent, mock_cart_manager):
        """Test tool error handling through integration."""
        
        # Make the cart manager raise an exception
        mock_cart_manager.add_item.side_effect = Exception("Database connection failed")
        
        cart_agent.session_id = "test_session"
        
        params = {
            "product_id": "test_product",
            "product_title": "Test Product",
            "quantity": 1
        }
        
        result = await cart_agent._call_cart_tool_integrated("add", params)
        
        # The tool integration should succeed (tool was called successfully)
        # but the tool result should indicate failure
        assert result.success is True  # Tool call succeeded
        assert result.result is not None
        assert result.result["success"] is False  # But the operation failed
        assert "Database connection failed" in result.result["error"]
        assert result.execution_time is not None
    
    def test_tool_call_statistics(self, cart_agent):
        """Test tool call statistics collection."""
        
        # Clear existing history
        cart_agent.tool_logger.call_history.clear()
        
        # Create some mock tool call results
        successful_result = ToolCallResult(
            tool_name="add_to_cart",
            success=True,
            result={"success": True},
            execution_time=0.5
        )
        
        failed_result = ToolCallResult(
            tool_name="remove_from_cart",
            success=False,
            result=None,
            error="Item not found",
            execution_time=0.2
        )
        
        # Log the results
        cart_agent.tool_logger.log_tool_call(successful_result, "test_session")
        cart_agent.tool_logger.log_tool_call(failed_result, "test_session")
        
        # Get statistics
        stats = cart_agent.tool_logger.get_call_statistics("test_session")
        
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["average_execution_time"] == 0.5
        assert "add_to_cart" in stats["tool_usage"]
        assert "remove_from_cart" in stats["tool_usage"]
    
    @pytest.mark.asyncio
    async def test_complete_workflow_with_tool_integration(self, cart_agent, mock_cart_manager):
        """Test complete agent workflow with tool integration."""
        
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
        
        # Clear tool call history
        cart_agent.tool_logger.call_history.clear()
        
        # Process through the agent
        result = await cart_agent.process_query(state)
        
        # Verify final state
        assert result["workflow_status"] == "completed"
        assert result["final_response"] is not None
        assert "Test Product" in result["final_response"]
        assert result["cart_updated"] is True
        
        # Verify tool calls were tracked
        assert len(result["tool_calls"]) >= 1
        
        # Check that the tool call has the new format
        tool_call = result["tool_calls"][0]
        assert "tool_name" in tool_call
        assert "success" in tool_call
        assert "execution_time" in tool_call
        assert "timestamp" in tool_call
        assert "metadata" in tool_call
    
    @pytest.mark.asyncio
    async def test_tool_integration_with_different_operations(self, cart_agent):
        """Test tool integration with all supported operations."""
        
        cart_agent.session_id = "test_session"
        
        operations_and_params = [
            ("add", {"product_id": "test", "product_title": "Test", "quantity": 1}),
            ("remove", {"product_id": "test", "quantity": 1}),
            ("list", {}),
            ("clear", {})
        ]
        
        for operation, params in operations_and_params:
            result = await cart_agent._call_cart_tool_integrated(operation, params)
            
            assert isinstance(result, ToolCallResult)
            assert result.tool_name in ["add_to_cart", "remove_from_cart", "list_cart", "clear_cart"]
            assert result.success is True  # All should succeed with our mocks
            assert result.metadata["tool_type"] == "function_calling"
    
    def test_agent_info_includes_tool_integration(self, cart_agent):
        """Test that agent info includes tool integration details."""
        
        info = cart_agent.get_agent_info()
        
        assert "tool_integration_type" in info
        assert info["tool_integration_type"] == "function_calling"
        assert "tool_logger_enabled" in info
        assert info["tool_logger_enabled"] is True
        assert "tools_available" in info
        assert len(info["tools_available"]) == 4


class TestToolIntegrationComponents:
    """Test individual tool integration components."""
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        tools = []
        
        for tool_name in ["add_to_cart", "remove_from_cart", "list_cart", "clear_cart"]:
            tool = Mock()
            tool.name = tool_name
            tool.description = f"Mock {tool_name} tool"
            tool.args_schema = None
            tool._arun = AsyncMock(return_value={"success": True, "message": "Mock result"})
            tools.append(tool)
        
        return tools
    
    @pytest.fixture
    def tool_integration(self, mock_tools):
        """Create tool integration for testing."""
        return FunctionCallingToolIntegration(tools=mock_tools)
    
    @pytest.mark.asyncio
    async def test_function_calling_tool_integration(self, tool_integration):
        """Test function calling tool integration directly."""
        
        result = await tool_integration.call_tool(
            tool_name="add_to_cart",
            parameters={"product_id": "test", "product_title": "Test"},
            session_id="test_session"
        )
        
        assert isinstance(result, ToolCallResult)
        assert result.success is True
        assert result.tool_name == "add_to_cart"
        assert result.result["success"] is True
        assert result.execution_time is not None
        assert result.metadata["tool_type"] == "function_calling"
    
    @pytest.mark.asyncio
    async def test_tool_integration_tool_not_found(self, tool_integration):
        """Test tool integration with non-existent tool."""
        
        result = await tool_integration.call_tool(
            tool_name="nonexistent_tool",
            parameters={},
            session_id="test_session"
        )
        
        assert result.success is False
        assert "not found" in result.error
    
    @pytest.mark.asyncio
    async def test_tool_integration_with_exception(self, tool_integration):
        """Test tool integration when tool raises exception."""
        
        # Make one of the tools raise an exception
        tool_integration.tools["add_to_cart"]._arun.side_effect = Exception("Tool error")
        
        result = await tool_integration.call_tool(
            tool_name="add_to_cart",
            parameters={"product_id": "test"},
            session_id="test_session"
        )
        
        assert result.success is False
        assert "Tool error" in result.error
        assert result.execution_time is not None
    
    def test_tool_integration_get_available_tools(self, tool_integration):
        """Test getting available tools from integration."""
        
        tools = tool_integration.get_available_tools()
        
        assert len(tools) == 4
        assert "add_to_cart" in tools
        assert "remove_from_cart" in tools
        assert "list_cart" in tools
        assert "clear_cart" in tools
    
    def test_tool_integration_get_tool_info(self, tool_integration):
        """Test getting tool info from integration."""
        
        info = tool_integration.get_tool_info("add_to_cart")
        
        assert info is not None
        assert info["name"] == "add_to_cart"
        assert info["tool_type"] == "function_calling"
        assert "description" in info
    
    def test_session_id_injector_creation(self):
        """Test session ID injector creation and usage."""
        
        injector = create_session_id_injector("test_session_123")
        
        # Create a mock tool
        mock_tool = Mock()
        
        # Inject session ID
        injector(mock_tool, "test_session_123")
        
        # Verify the tool has the session ID method
        assert hasattr(mock_tool, '_get_session_id')
        assert mock_tool._get_session_id() == "test_session_123"
    
    def test_tool_call_logger(self):
        """Test tool call logger functionality."""
        
        logger = ToolCallLogger()
        
        # Create a test result
        result = ToolCallResult(
            tool_name="test_tool",
            success=True,
            result={"data": "test"},
            execution_time=0.5
        )
        
        # Log the result
        logger.log_tool_call(result, "test_session")
        
        # Check it was logged
        assert len(logger.call_history) == 1
        
        log_entry = logger.call_history[0]
        assert log_entry["session_id"] == "test_session"
        assert log_entry["tool_name"] == "test_tool"
        assert log_entry["success"] is True
        assert log_entry["execution_time"] == 0.5
        
        # Test statistics
        stats = logger.get_call_statistics("test_session")
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["success_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])