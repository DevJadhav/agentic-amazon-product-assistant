"""
Simple integration test for API handler with cart functionality.
Tests basic functionality without complex dependencies.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from ..api.langgraph_handler import LangGraphAPIHandler
from ..api.models import EnhancedQueryResponse


class TestAPIHandlerIntegration:
    """Test API handler integration with minimal dependencies."""
    
    def test_enhanced_response_model_validation(self):
        """Test that enhanced response model validates correctly."""
        
        response = EnhancedQueryResponse(
            query="test query",
            response="test response", 
            session_id="test_session",
            conversation_turn=1,
            agent_workflow="master_routing",
            agent_used="cart_agent",
            cart_data=[{"product_id": "test_product", "quantity": 1}],
            cart_updated=True,
            cart_item_count=1,
            cart_total=29.99,
            tools_called=["cart_add"],
            context={},
            metadata={},
            processing_time=1.0,
            workflow_steps=["step1"]
        )
        
        # Verify response structure
        assert response.query == "test query"
        assert response.agent_used == "cart_agent"
        assert response.cart_updated is True
        assert response.cart_item_count == 1
        assert response.cart_total == 29.99
        assert len(response.cart_data) == 1
        assert response.cart_data[0]["product_id"] == "test_product"
        assert "cart_add" in response.tools_called
    
    @patch('src.langgraph_integration.api.langgraph_handler.VectorSearchTool')
    @patch('src.langgraph_integration.api.langgraph_handler.ProductAnalysisTool')
    @patch('src.langgraph_integration.api.langgraph_handler.get_global_cart_manager')
    def test_api_handler_initialization(self, mock_cart_manager, mock_product_tool, mock_vector_tool):
        """Test API handler initializes with cart functionality."""
        
        # Mock dependencies
        mock_vector_tool.return_value = Mock()
        mock_product_tool.return_value = Mock()
        mock_cart_manager.return_value = Mock()
        
        # Initialize handler
        handler = LangGraphAPIHandler()
        
        # Verify initialization
        assert handler.cart_manager is not None
        assert handler.master_graph is None  # Lazy-loaded
        assert hasattr(handler, 'vector_search_tool')
        assert hasattr(handler, 'product_analysis_tool')
    
    @patch('src.langgraph_integration.api.langgraph_handler.VectorSearchTool')
    @patch('src.langgraph_integration.api.langgraph_handler.ProductAnalysisTool')
    def test_cart_data_retrieval_helper(self, mock_product_tool, mock_vector_tool):
        """Test cart data retrieval helper method."""
        
        # Mock dependencies
        mock_vector_tool.return_value = Mock()
        mock_product_tool.return_value = Mock()
        
        # Mock cart manager
        mock_cart_manager = Mock()
        mock_cart_manager.get_cart_contents.return_value = [
            {"product_id": "test", "quantity": 1, "price": 29.99}
        ]
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 1,
            "total_value": 29.99,
            "is_empty": False
        }
        
        # Initialize handler
        handler = LangGraphAPIHandler()
        handler.cart_manager = mock_cart_manager
        
        # Test cart data retrieval
        cart_data = handler._get_cart_data_for_response("test_session")
        
        assert "contents" in cart_data
        assert "summary" in cart_data
        assert len(cart_data["contents"]) == 1
        assert cart_data["summary"]["total_items"] == 1
        assert cart_data["summary"]["total_value"] == 29.99
    
    @patch('src.langgraph_integration.api.langgraph_handler.VectorSearchTool')
    @patch('src.langgraph_integration.api.langgraph_handler.ProductAnalysisTool')
    def test_routing_information_extraction(self, mock_product_tool, mock_vector_tool):
        """Test routing information extraction from state."""
        
        # Mock dependencies
        mock_vector_tool.return_value = Mock()
        mock_product_tool.return_value = Mock()
        
        # Initialize handler
        handler = LangGraphAPIHandler()
        
        # Test state with routing information
        state = {
            "routing_decision": "route_to_cart_agent",
            "target_agent": "shopping_cart_agent",
            "intent_confidence": 0.95,
            "response_metadata": {
                "agent_used": "cart_agent",
                "clarification_requested": False
            }
        }
        
        routing_info = handler._extract_routing_information(state)
        
        assert routing_info["routing_decision"] == "route_to_cart_agent"
        assert routing_info["agent_used"] == "cart_agent"
        assert routing_info["intent_confidence"] == 0.95
        assert routing_info["target_agent"] == "shopping_cart_agent"
        assert routing_info["clarification_requested"] is False
    
    @patch('src.langgraph_integration.api.langgraph_handler.VectorSearchTool')
    @patch('src.langgraph_integration.api.langgraph_handler.ProductAnalysisTool')
    def test_tools_called_extraction(self, mock_product_tool, mock_vector_tool):
        """Test tools called extraction from state."""
        
        # Mock dependencies
        mock_vector_tool.return_value = Mock()
        mock_product_tool.return_value = Mock()
        
        # Initialize handler
        handler = LangGraphAPIHandler()
        
        # Test state with various tool calls
        state = {
            "cart_operation": "add",
            "search_results": {"products": []},
            "product_analysis_results": {"analysis": "test"},
            "intermediate_steps": [
                {"tool": "vector_search", "result": "success"}
            ]
        }
        
        tools_called = handler._extract_tools_called(state)
        
        assert "cart_add" in tools_called
        assert "vector_search" in tools_called
        assert "product_analysis" in tools_called
    
    def test_enhanced_response_serialization(self):
        """Test enhanced response can be serialized properly."""
        
        response = EnhancedQueryResponse(
            query="test query",
            response="test response",
            session_id="test_session", 
            conversation_turn=1,
            agent_workflow="master_routing",
            agent_used="cart_agent",
            routing_decision="route_to_cart_agent",
            intent_confidence=0.95,
            cart_data=[{"product_id": "test"}],
            cart_updated=True,
            cart_item_count=1,
            cart_total=29.99,
            tools_called=["cart_add"],
            context={"test": "context"},
            metadata={"test": "metadata"},
            processing_time=1.5,
            workflow_steps=["step1", "step2"]
        )
        
        # Test serialization to dict
        response_dict = response.model_dump()
        
        # Verify all fields are present
        assert response_dict["query"] == "test query"
        assert response_dict["agent_used"] == "cart_agent"
        assert response_dict["routing_decision"] == "route_to_cart_agent"
        assert response_dict["intent_confidence"] == 0.95
        assert response_dict["cart_updated"] is True
        assert response_dict["cart_item_count"] == 1
        assert response_dict["cart_total"] == 29.99
        assert response_dict["tools_called"] == ["cart_add"]
        
        # Test JSON serialization
        import json
        json_str = json.dumps(response_dict)
        assert json_str is not None
        
        # Test deserialization
        parsed_dict = json.loads(json_str)
        assert parsed_dict["query"] == "test query"
        assert parsed_dict["cart_updated"] is True