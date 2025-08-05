"""
Tests for API integration with shopping cart functionality.
Tests enhanced response models and cart-specific endpoints.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from typing import Dict, Any

from ..api.langgraph_handler import LangGraphAPIHandler
from ..api.models import LangGraphQueryRequest, EnhancedQueryResponse
from ..state.shopping_cart_manager import ShoppingCartManager


class TestAPICartIntegration:
    """Test API integration with shopping cart functionality."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock shopping cart manager."""
        cart_manager = Mock(spec=ShoppingCartManager)
        
        # Mock cart contents
        cart_manager.get_cart_contents.return_value = [
            {
                "id": "1",
                "product_id": "test_product_1",
                "product_title": "Test Product 1",
                "product_price": 29.99,
                "quantity": 2,
                "subtotal": 59.98
            }
        ]
        
        # Mock cart summary
        cart_manager.get_cart_summary.return_value = {
            "session_id": "test_session",
            "total_items": 2,
            "total_value": 59.98,
            "unique_products": 1,
            "is_empty": False
        }
        
        return cart_manager
    
    @pytest.fixture
    def mock_master_graph(self):
        """Create mock master graph."""
        master_graph = Mock()
        master_graph.process_query = AsyncMock()
        
        # Mock successful routing response
        master_graph.process_query.return_value = {
            "session_id": "test_session",
            "current_query": "add product to cart",
            "final_response": "Added product to your cart successfully",
            "conversation_turn": 1,
            "workflow_status": "completed",
            "current_step": "response_finalization_and_formatting",
            "routing_decision": "route_to_cart_agent",
            "target_agent": "shopping_cart_agent",
            "intent_confidence": 0.95,
            "cart_operation": "add",
            "cart_updated": True,
            "cart_operation_result": {
                "success": True,
                "message": "Added Test Product to cart"
            },
            "response_metadata": {
                "agent_used": "cart_agent",
                "routing_successful": True
            },
            "routing_metadata": {
                "router_execution": {
                    "executed_at": datetime.now(timezone.utc).isoformat(),
                    "router_node_successful": True
                }
            }
        }
        
        return master_graph
    
    @pytest.fixture
    def api_handler(self, mock_cart_manager, mock_master_graph):
        """Create API handler with mocked dependencies."""
        with patch('src.langgraph_integration.api.langgraph_handler.VectorSearchTool') as mock_vector_tool, \
             patch('src.langgraph_integration.api.langgraph_handler.ProductAnalysisTool') as mock_product_tool:
            
            # Mock tool initialization
            mock_vector_tool.return_value = Mock()
            mock_product_tool.return_value = Mock()
            
            handler = LangGraphAPIHandler()
            handler.cart_manager = mock_cart_manager
            handler.master_graph = mock_master_graph
            return handler
    
    @pytest.mark.asyncio
    async def test_enhanced_query_processing_with_cart(self, api_handler):
        """Test enhanced query processing includes cart data."""
        
        request = LangGraphQueryRequest(
            query="add wireless headphones to my cart",
            session_id="test_session",
            agent_type="master_routing"
        )
        
        response = await api_handler.process_query_with_enhanced_routing(request)
        
        # Verify response type and structure
        assert isinstance(response, EnhancedQueryResponse)
        assert response.query == "add wireless headphones to my cart"
        assert response.session_id == "test_session"
        assert response.agent_used == "cart_agent"
        assert response.routing_decision == "route_to_cart_agent"
        
        # Verify cart data is included
        assert response.cart_updated is True
        assert response.cart_item_count == 2
        assert response.cart_total == 59.98
        assert response.cart_data is not None
        assert len(response.cart_data) == 1
        
        # Verify tools called
        assert "cart_add" in response.tools_called
        
        # Verify routing information
        assert response.intent_confidence == 0.95
        assert "routing_metadata" in response.metadata
    
    @pytest.mark.asyncio
    async def test_enhanced_query_processing_qa_agent(self, api_handler, mock_master_graph):
        """Test enhanced query processing routes to QA agent."""
        
        # Mock QA agent response
        mock_master_graph.process_query.return_value = {
            "session_id": "test_session",
            "current_query": "what are the best wireless headphones?",
            "final_response": "Here are the top wireless headphones...",
            "conversation_turn": 1,
            "workflow_status": "completed",
            "routing_decision": "route_to_qa_agent",
            "target_agent": "product_qa_agent",
            "intent_confidence": 0.88,
            "cart_updated": False,
            "selected_products": [{"id": "1"}, {"id": "2"}],
            "response_metadata": {
                "agent_used": "qa_agent",
                "routing_successful": True
            }
        }
        
        request = LangGraphQueryRequest(
            query="what are the best wireless headphones?",
            session_id="test_session",
            agent_type="master_routing"
        )
        
        response = await api_handler.process_query_with_enhanced_routing(request)
        
        # Verify QA agent routing
        assert response.agent_used == "qa_agent"
        assert response.routing_decision == "route_to_qa_agent"
        assert response.cart_updated is False
        assert response.products_found == 2
        
        # Cart data should still be included but not updated
        assert response.cart_data is not None
        assert response.cart_item_count == 2  # From existing cart
    
    def test_get_cart_contents_endpoint(self, api_handler):
        """Test cart contents API endpoint."""
        
        result = api_handler.get_cart_contents("test_session")
        
        assert result["success"] is True
        assert result["session_id"] == "test_session"
        assert len(result["cart_contents"]) == 1
        assert result["cart_summary"]["total_items"] == 2
        assert result["cart_summary"]["total_value"] == 59.98
    
    def test_add_to_cart_endpoint(self, api_handler, mock_cart_manager):
        """Test add to cart API endpoint."""
        
        # Mock successful add operation
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added Test Product to cart",
            "item": {
                "id": "2",
                "product_id": "test_product_2",
                "product_title": "Test Product 2",
                "quantity": 1,
                "product_price": 19.99
            },
            "action": "added"
        }
        
        result = api_handler.add_to_cart(
            session_id="test_session",
            product_id="test_product_2",
            product_title="Test Product 2",
            quantity=1,
            price=19.99
        )
        
        assert result["success"] is True
        assert result["message"] == "Added Test Product to cart"
        assert result["item"]["product_id"] == "test_product_2"
        assert "cart_summary" in result
        
        # Verify cart manager was called correctly
        mock_cart_manager.add_item.assert_called_once_with(
            session_id="test_session",
            product_id="test_product_2",
            product_title="Test Product 2",
            quantity=1,
            price=19.99,
            image_url=None,
            metadata=None
        )
    
    def test_remove_from_cart_endpoint(self, api_handler, mock_cart_manager):
        """Test remove from cart API endpoint."""
        
        # Mock successful remove operation
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Removed Test Product from cart",
            "item": {
                "id": "1",
                "product_id": "test_product_1",
                "product_title": "Test Product 1"
            },
            "action": "removed",
            "removed_completely": True
        }
        
        result = api_handler.remove_from_cart(
            session_id="test_session",
            product_id="test_product_1"
        )
        
        assert result["success"] is True
        assert result["message"] == "Removed Test Product from cart"
        assert result["removed_completely"] is True
        assert "cart_summary" in result
        
        # Verify cart manager was called correctly
        mock_cart_manager.remove_item.assert_called_once_with(
            session_id="test_session",
            product_id="test_product_1",
            quantity=None
        )
    
    def test_clear_cart_endpoint(self, api_handler, mock_cart_manager):
        """Test clear cart API endpoint."""
        
        # Mock successful clear operation
        mock_cart_manager.clear_cart.return_value = {
            "success": True,
            "message": "Cleared 2 items from cart",
            "items_removed": 2,
            "cleared_items": [
                {"product_id": "test_product_1", "quantity": 2}
            ]
        }
        
        result = api_handler.clear_cart("test_session")
        
        assert result["success"] is True
        assert result["message"] == "Cleared 2 items from cart"
        assert result["items_removed"] == 2
        assert "cart_summary" in result
        
        # Verify cart manager was called correctly
        mock_cart_manager.clear_cart.assert_called_once_with("test_session")
    
    def test_agent_capabilities_includes_cart_features(self, api_handler):
        """Test agent capabilities includes cart functionality."""
        
        capabilities = api_handler.get_agent_capabilities()
        
        assert "shopping_cart" in capabilities.available_agents
        assert "master_routing" in capabilities.available_agents
        assert "shopping_cart" in capabilities.tools_available
        
        # Verify cart-related features
        assert capabilities.features["shopping_cart"] is True
        assert capabilities.features["intelligent_routing"] is True
        assert capabilities.features["intent_classification"] is True
        assert capabilities.features["dual_tool_support"] is True
    
    def test_extract_routing_information(self, api_handler):
        """Test routing information extraction from state."""
        
        state = {
            "routing_decision": "route_to_cart_agent",
            "target_agent": "shopping_cart_agent",
            "intent_confidence": 0.92,
            "response_metadata": {
                "agent_used": "cart_agent",
                "clarification_requested": False
            },
            "routing_metadata": {
                "router_execution": {
                    "executed_at": "2024-01-01T12:00:00Z"
                }
            }
        }
        
        routing_info = api_handler._extract_routing_information(state)
        
        assert routing_info["routing_decision"] == "route_to_cart_agent"
        assert routing_info["agent_used"] == "cart_agent"
        assert routing_info["intent_confidence"] == 0.92
        assert routing_info["target_agent"] == "shopping_cart_agent"
        assert routing_info["clarification_requested"] is False
    
    def test_extract_tools_called(self, api_handler):
        """Test tools called extraction from state."""
        
        state = {
            "cart_operation": "add",
            "search_results": {"products": []},
            "product_analysis_results": {"analysis": "test"},
            "intermediate_steps": [
                {"tool": "vector_search", "result": "success"},
                {"tool": "custom_tool", "result": "success"}
            ]
        }
        
        tools_called = api_handler._extract_tools_called(state)
        
        assert "cart_add" in tools_called
        assert "vector_search" in tools_called
        assert "product_analysis" in tools_called
        assert "custom_tool" in tools_called
    
    @pytest.mark.asyncio
    async def test_error_handling_includes_cart_data(self, api_handler, mock_master_graph):
        """Test error responses still include cart data."""
        
        # Mock master graph failure
        mock_master_graph.process_query.side_effect = Exception("Test error")
        
        request = LangGraphQueryRequest(
            query="test query",
            session_id="test_session",
            agent_type="master_routing"
        )
        
        response = await api_handler.process_query_with_enhanced_routing(request)
        
        # Verify error response structure
        assert response.error_state is not None
        assert "error processing your request" in response.response
        assert response.routing_decision == "error"
        assert response.agent_used == "error"
        
        # Verify cart data is still included
        assert response.cart_data is not None
        assert response.cart_item_count == 2
        assert response.cart_total == 59.98
    
    def test_cart_data_retrieval_error_handling(self, api_handler, mock_cart_manager):
        """Test cart data retrieval handles errors gracefully."""
        
        # Mock cart manager failure
        mock_cart_manager.get_cart_contents.side_effect = Exception("Database error")
        mock_cart_manager.get_cart_summary.side_effect = Exception("Database error")
        
        cart_data = api_handler._get_cart_data_for_response("test_session")
        
        # Verify graceful error handling
        assert cart_data["contents"] == []
        assert cart_data["summary"]["total_items"] == 0
        assert cart_data["summary"]["total_value"] == 0.0
        assert cart_data["summary"]["is_empty"] is True


class TestEnhancedResponseModel:
    """Test enhanced response model serialization and validation."""
    
    def test_enhanced_response_model_creation(self):
        """Test creating enhanced response model with all fields."""
        
        response = EnhancedQueryResponse(
            query="test query",
            response="test response",
            session_id="test_session",
            conversation_turn=1,
            agent_workflow="master_routing",
            routing_decision="route_to_cart_agent",
            agent_used="cart_agent",
            intent_confidence=0.95,
            cart_data=[{"product_id": "test"}],
            cart_updated=True,
            cart_item_count=1,
            cart_total=29.99,
            tools_called=["cart_add"],
            context={"test": "context"},
            metadata={"test": "metadata"},
            processing_time=1.5,
            workflow_steps=["step1", "step2"],
            products_found=0,
            reviews_found=0
        )
        
        # Verify all fields are set correctly
        assert response.query == "test query"
        assert response.routing_decision == "route_to_cart_agent"
        assert response.agent_used == "cart_agent"
        assert response.intent_confidence == 0.95
        assert response.cart_updated is True
        assert response.cart_item_count == 1
        assert response.cart_total == 29.99
        assert response.tools_called == ["cart_add"]
    
    def test_enhanced_response_model_defaults(self):
        """Test enhanced response model with default values."""
        
        response = EnhancedQueryResponse(
            query="test query",
            response="test response",
            session_id="test_session",
            conversation_turn=1,
            agent_workflow="master_routing",
            agent_used="qa_agent",
            context={},
            metadata={},
            processing_time=1.0,
            workflow_steps=["step1"]
        )
        
        # Verify default values
        assert response.routing_decision is None
        assert response.intent_confidence is None
        assert response.cart_data is None
        assert response.cart_updated is False
        assert response.cart_item_count == 0
        assert response.cart_total is None
        assert response.tools_called == []
        assert response.products_found == 0
        assert response.reviews_found == 0
        assert response.error_state is None
    
    def test_enhanced_response_model_serialization(self):
        """Test enhanced response model can be serialized to dict."""
        
        response = EnhancedQueryResponse(
            query="test query",
            response="test response",
            session_id="test_session",
            conversation_turn=1,
            agent_workflow="master_routing",
            agent_used="cart_agent",
            cart_updated=True,
            cart_item_count=2,
            cart_total=59.98,
            context={},
            metadata={},
            processing_time=1.0,
            workflow_steps=["step1"]
        )
        
        # Test serialization
        response_dict = response.model_dump()
        
        assert response_dict["query"] == "test query"
        assert response_dict["agent_used"] == "cart_agent"
        assert response_dict["cart_updated"] is True
        assert response_dict["cart_item_count"] == 2
        assert response_dict["cart_total"] == 59.98