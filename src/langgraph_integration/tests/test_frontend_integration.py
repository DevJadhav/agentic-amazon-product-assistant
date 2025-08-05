"""
Integration tests for frontend integration with real-time cart updates.
Tests the complete flow from API to frontend cart display.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from ..api.langgraph_handler import LangGraphAPIHandler
from ..api.models import EnhancedQueryResponse
from ..state.shopping_cart_manager import ShoppingCartManager


class TestFrontendCartIntegration:
    """Test frontend integration with cart functionality."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager for frontend tests."""
        manager = Mock(spec=ShoppingCartManager)
        
        # Mock cart data for frontend display
        manager.get_cart_contents.return_value = [
            {
                "id": "cart_item_1",
                "product_id": "LAPTOP_001",
                "product_title": "Gaming Laptop",
                "quantity": 1,
                "product_price": 1299.99,
                "subtotal": 1299.99,
                "product_image_url": "https://example.com/laptop.jpg",
                "product_metadata": {
                    "category": "electronics",
                    "brand": "TechBrand",
                    "rating": 4.5
                },
                "added_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            },
            {
                "id": "cart_item_2",
                "product_id": "MOUSE_001",
                "product_title": "Gaming Mouse",
                "quantity": 2,
                "product_price": 79.99,
                "subtotal": 159.98,
                "product_image_url": "https://example.com/mouse.jpg",
                "product_metadata": {
                    "category": "accessories",
                    "brand": "TechBrand"
                },
                "added_at": "2024-01-01T13:00:00",
                "updated_at": "2024-01-01T13:00:00"
            }
        ]
        
        manager.get_cart_summary.return_value = {
            "session_id": "frontend_test_session",
            "total_items": 3,  # 1 + 2
            "total_value": 1459.97,  # 1299.99 + 159.98
            "unique_products": 2,
            "is_empty": False,
            "last_updated": "2024-01-01T13:00:00"
        }
        
        return manager
    
    @pytest.fixture
    def api_handler(self, mock_cart_manager):
        """Create API handler with mock cart manager."""
        handler = LangGraphAPIHandler()
        
        # Mock the master graph
        mock_master_graph = Mock()
        mock_result = {
            "final_response": "Added Gaming Laptop to your cart",
            "workflow_status": "completed",
            "session_id": "frontend_test_session",
            "conversation_turn": 1,
            "cart_updated": True,
            "cart_item_count": 3,
            "cart_total": 1459.97,
            "current_cart_contents": mock_cart_manager.get_cart_contents(),
            "response_metadata": {
                "agent_used": "cart_agent",
                "routing_decision": "cart",
                "routing_successful": True,
                "tools_called": ["add_to_cart"]
            }
        }
        mock_master_graph.process_query.return_value = mock_result
        
        # Replace the handler's master graph
        handler.master_graph = mock_master_graph
        
        return handler
    
    def test_enhanced_query_response_model(self, mock_cart_manager):
        """Test enhanced query response model with cart data."""
        
        # Create enhanced response with cart data
        response = EnhancedQueryResponse(
            query="add gaming laptop to cart",
            response="Added Gaming Laptop to your cart",
            session_id="frontend_test_session",
            conversation_turn=1,
            cart_data={
                "items": mock_cart_manager.get_cart_contents(),
                "summary": mock_cart_manager.get_cart_summary()
            },
            cart_updated=True,
            cart_item_count=3,
            cart_total=1459.97,
            routing_decision="cart",
            agent_used="cart_agent",
            tools_called=["add_to_cart"]
        )
        
        # Verify response structure
        assert response.query == "add gaming laptop to cart"
        assert response.cart_updated is True
        assert response.cart_item_count == 3
        assert response.cart_total == 1459.97
        assert response.routing_decision == "cart"
        assert response.agent_used == "cart_agent"
        assert "add_to_cart" in response.tools_called
        
        # Verify cart data structure
        assert "items" in response.cart_data
        assert "summary" in response.cart_data
        assert len(response.cart_data["items"]) == 2
        assert response.cart_data["summary"]["total_items"] == 3
        assert response.cart_data["summary"]["total_value"] == 1459.97
    
    @pytest.mark.asyncio
    async def test_api_cart_data_inclusion(self, api_handler, mock_cart_manager):
        """Test that API responses include cart data for frontend."""
        
        # Mock request data
        request_data = {
            "query": "add gaming laptop to cart",
            "session_id": "frontend_test_session",
            "selected_product": {
                "product_id": "LAPTOP_001",
                "title": "Gaming Laptop",
                "price": 1299.99,
                "image_url": "https://example.com/laptop.jpg"
            }
        }
        
        # Process query through API handler
        response = await api_handler.process_query(request_data)
        
        # Verify response includes cart data
        assert isinstance(response, EnhancedQueryResponse)
        assert response.cart_updated is True
        assert response.cart_item_count == 3
        assert response.cart_total == 1459.97
        
        # Verify cart data is properly formatted for frontend
        assert "items" in response.cart_data
        assert "summary" in response.cart_data
        
        cart_items = response.cart_data["items"]
        assert len(cart_items) == 2
        
        # Verify first item structure
        laptop_item = next(item for item in cart_items if item["product_id"] == "LAPTOP_001")
        assert laptop_item["product_title"] == "Gaming Laptop"
        assert laptop_item["quantity"] == 1
        assert laptop_item["subtotal"] == 1299.99
        assert laptop_item["product_image_url"] == "https://example.com/laptop.jpg"
        assert "category" in laptop_item["product_metadata"]
        
        # Verify summary data
        cart_summary = response.cart_data["summary"]
        assert cart_summary["total_items"] == 3
        assert cart_summary["total_value"] == 1459.97
        assert cart_summary["unique_products"] == 2
        assert cart_summary["is_empty"] is False
    
    @pytest.mark.asyncio
    async def test_real_time_cart_updates(self, api_handler, mock_cart_manager):
        """Test real-time cart updates through API."""
        
        session_id = "realtime_test_session"
        
        # Initial cart state (empty)
        mock_cart_manager.get_cart_contents.return_value = []
        mock_cart_manager.get_cart_summary.return_value = {
            "session_id": session_id,
            "total_items": 0,
            "total_value": 0.0,
            "unique_products": 0,
            "is_empty": True
        }
        
        # Update master graph mock for empty cart
        api_handler.master_graph.process_query.return_value = {
            "final_response": "Your cart is empty",
            "workflow_status": "completed",
            "session_id": session_id,
            "conversation_turn": 1,
            "cart_updated": False,
            "cart_item_count": 0,
            "cart_total": 0.0,
            "current_cart_contents": [],
            "response_metadata": {
                "agent_used": "cart_agent",
                "routing_decision": "cart"
            }
        }
        
        # Request 1: List empty cart
        list_request = {
            "query": "show my cart",
            "session_id": session_id
        }
        
        list_response = await api_handler.process_query(list_request)
        
        assert list_response.cart_item_count == 0
        assert list_response.cart_total == 0.0
        assert list_response.cart_data["summary"]["is_empty"] is True
        
        # Update mock for cart with item
        mock_cart_manager.get_cart_contents.return_value = [
            {
                "id": "new_item",
                "product_id": "PHONE_001",
                "product_title": "Smartphone",
                "quantity": 1,
                "product_price": 699.99,
                "subtotal": 699.99,
                "product_image_url": "https://example.com/phone.jpg",
                "product_metadata": {"category": "electronics"},
                "added_at": "2024-01-01T14:00:00",
                "updated_at": "2024-01-01T14:00:00"
            }
        ]
        
        mock_cart_manager.get_cart_summary.return_value = {
            "session_id": session_id,
            "total_items": 1,
            "total_value": 699.99,
            "unique_products": 1,
            "is_empty": False,
            "last_updated": "2024-01-01T14:00:00"
        }
        
        # Update master graph mock for cart with item
        api_handler.master_graph.process_query.return_value = {
            "final_response": "Added Smartphone to your cart",
            "workflow_status": "completed",
            "session_id": session_id,
            "conversation_turn": 2,
            "cart_updated": True,
            "cart_item_count": 1,
            "cart_total": 699.99,
            "current_cart_contents": mock_cart_manager.get_cart_contents(),
            "response_metadata": {
                "agent_used": "cart_agent",
                "routing_decision": "cart",
                "tools_called": ["add_to_cart"]
            }
        }
        
        # Request 2: Add item to cart
        add_request = {
            "query": "add smartphone to cart",
            "session_id": session_id,
            "selected_product": {
                "product_id": "PHONE_001",
                "title": "Smartphone",
                "price": 699.99
            }
        }
        
        add_response = await api_handler.process_query(add_request)
        
        # Verify real-time update
        assert add_response.cart_updated is True
        assert add_response.cart_item_count == 1
        assert add_response.cart_total == 699.99
        
        # Verify cart data reflects the update
        cart_items = add_response.cart_data["items"]
        assert len(cart_items) == 1
        assert cart_items[0]["product_title"] == "Smartphone"
        assert cart_items[0]["subtotal"] == 699.99
        
        cart_summary = add_response.cart_data["summary"]
        assert cart_summary["is_empty"] is False
        assert cart_summary["total_items"] == 1
        assert cart_summary["total_value"] == 699.99
    
    def test_cart_data_serialization(self, mock_cart_manager):
        """Test cart data serialization for frontend consumption."""
        
        # Create response with complex cart data
        response = EnhancedQueryResponse(
            query="test query",
            response="test response",
            session_id="serialization_test",
            conversation_turn=1,
            cart_data={
                "items": mock_cart_manager.get_cart_contents(),
                "summary": mock_cart_manager.get_cart_summary()
            },
            cart_updated=True,
            cart_item_count=3,
            cart_total=1459.97,
            routing_decision="cart",
            agent_used="cart_agent",
            tools_called=["add_to_cart", "list_cart"]
        )
        
        # Serialize to JSON (as would happen in API response)
        json_data = response.model_dump()
        
        # Verify JSON structure
        assert "cart_data" in json_data
        assert "items" in json_data["cart_data"]
        assert "summary" in json_data["cart_data"]
        
        # Verify items are properly serialized
        items = json_data["cart_data"]["items"]
        assert len(items) == 2
        
        laptop_item = next(item for item in items if item["product_id"] == "LAPTOP_001")
        assert laptop_item["product_title"] == "Gaming Laptop"
        assert laptop_item["quantity"] == 1
        assert laptop_item["subtotal"] == 1299.99
        assert laptop_item["product_metadata"]["category"] == "electronics"
        
        # Verify summary is properly serialized
        summary = json_data["cart_data"]["summary"]
        assert summary["total_items"] == 3
        assert summary["total_value"] == 1459.97
        assert summary["is_empty"] is False
        
        # Verify other fields
        assert json_data["cart_updated"] is True
        assert json_data["cart_item_count"] == 3
        assert json_data["cart_total"] == 1459.97
        assert json_data["tools_called"] == ["add_to_cart", "list_cart"]
    
    @pytest.mark.asyncio
    async def test_error_handling_in_frontend_integration(self, api_handler, mock_cart_manager):
        """Test error handling in frontend integration."""
        
        # Mock cart manager to fail
        mock_cart_manager.get_cart_contents.side_effect = Exception("Database error")
        mock_cart_manager.get_cart_summary.side_effect = Exception("Database error")
        
        # Update master graph to handle error
        api_handler.master_graph.process_query.return_value = {
            "final_response": "I'm having trouble accessing your cart right now. Please try again.",
            "workflow_status": "completed",
            "session_id": "error_test_session",
            "conversation_turn": 1,
            "cart_updated": False,
            "cart_item_count": 0,
            "cart_total": 0.0,
            "current_cart_contents": [],
            "error_state": "Cart data unavailable",
            "response_metadata": {
                "agent_used": "cart_agent",
                "routing_decision": "cart",
                "agent_error": True
            }
        }
        
        request_data = {
            "query": "show my cart",
            "session_id": "error_test_session"
        }
        
        response = await api_handler.process_query(request_data)
        
        # Verify error is handled gracefully
        assert response.cart_updated is False
        assert response.cart_item_count == 0
        assert response.cart_total == 0.0
        
        # Verify cart data is empty/safe
        assert response.cart_data is not None
        assert "items" in response.cart_data
        assert "summary" in response.cart_data
        assert response.cart_data["items"] == []
        
        # Verify error message is user-friendly
        assert "trouble" in response.response.lower() or "error" in response.response.lower()
    
    def test_cart_display_formatting(self, mock_cart_manager):
        """Test cart data formatting for display purposes."""
        
        cart_items = mock_cart_manager.get_cart_contents()
        cart_summary = mock_cart_manager.get_cart_summary()
        
        # Test formatting for frontend display
        formatted_items = []
        for item in cart_items:
            formatted_item = {
                "id": item["id"],
                "title": item["product_title"],
                "quantity": item["quantity"],
                "price": f"${item['product_price']:.2f}",
                "subtotal": f"${item['subtotal']:.2f}",
                "image": item["product_image_url"],
                "brand": item["product_metadata"].get("brand", "Unknown"),
                "category": item["product_metadata"].get("category", "Other"),
                "added_date": item["added_at"][:10]  # Just the date part
            }
            formatted_items.append(formatted_item)
        
        # Verify formatting
        assert len(formatted_items) == 2
        
        laptop_item = next(item for item in formatted_items if "Laptop" in item["title"])
        assert laptop_item["price"] == "$1299.99"
        assert laptop_item["subtotal"] == "$1299.99"
        assert laptop_item["brand"] == "TechBrand"
        assert laptop_item["category"] == "electronics"
        assert laptop_item["added_date"] == "2024-01-01"
        
        mouse_item = next(item for item in formatted_items if "Mouse" in item["title"])
        assert mouse_item["price"] == "$79.99"
        assert mouse_item["subtotal"] == "$159.98"
        assert mouse_item["quantity"] == 2
        
        # Test summary formatting
        formatted_summary = {
            "total_items": cart_summary["total_items"],
            "total_value": f"${cart_summary['total_value']:.2f}",
            "unique_products": cart_summary["unique_products"],
            "is_empty": cart_summary["is_empty"],
            "last_updated": cart_summary["last_updated"]
        }
        
        assert formatted_summary["total_items"] == 3
        assert formatted_summary["total_value"] == "$1459.97"
        assert formatted_summary["unique_products"] == 2
        assert formatted_summary["is_empty"] is False


class TestStreamlitCartIntegration:
    """Test Streamlit frontend cart integration."""
    
    def test_cart_state_manager_simulation(self):
        """Test cart state management simulation for Streamlit."""
        
        # Simulate Streamlit session state
        mock_session_state = {}
        
        class MockCartStateManager:
            def __init__(self):
                self.session_key = "shopping_cart_data"
            
            def update_cart_display(self, cart_data):
                mock_session_state[self.session_key] = cart_data
            
            def get_cart_data(self):
                return mock_session_state.get(self.session_key, {})
            
            def render_cart_sidebar(self):
                cart_data = self.get_cart_data()
                if not cart_data or cart_data.get("summary", {}).get("is_empty", True):
                    return "Your cart is empty"
                
                items = cart_data.get("items", [])
                summary = cart_data.get("summary", {})
                
                display_text = f"Cart ({summary.get('total_items', 0)} items)\n"
                display_text += f"Total: ${summary.get('total_value', 0.0):.2f}\n\n"
                
                for item in items:
                    display_text += f"• {item['product_title']} (x{item['quantity']}) - ${item['subtotal']:.2f}\n"
                
                return display_text
        
        cart_manager = MockCartStateManager()
        
        # Test empty cart
        empty_display = cart_manager.render_cart_sidebar()
        assert "empty" in empty_display.lower()
        
        # Test cart with items
        cart_data = {
            "items": [
                {
                    "product_title": "Gaming Laptop",
                    "quantity": 1,
                    "subtotal": 1299.99
                },
                {
                    "product_title": "Gaming Mouse",
                    "quantity": 2,
                    "subtotal": 159.98
                }
            ],
            "summary": {
                "total_items": 3,
                "total_value": 1459.97,
                "is_empty": False
            }
        }
        
        cart_manager.update_cart_display(cart_data)
        cart_display = cart_manager.render_cart_sidebar()
        
        # Verify cart display
        assert "Cart (3 items)" in cart_display
        assert "Total: $1459.97" in cart_display
        assert "Gaming Laptop (x1) - $1299.99" in cart_display
        assert "Gaming Mouse (x2) - $159.98" in cart_display
        
        # Test cart data retrieval
        retrieved_data = cart_manager.get_cart_data()
        assert retrieved_data == cart_data
        assert len(retrieved_data["items"]) == 2
        assert retrieved_data["summary"]["total_items"] == 3
    
    def test_cart_tab_switching_simulation(self):
        """Test cart tab switching simulation."""
        
        # Simulate tab state
        mock_tab_state = {"active_tab": "suggestions"}
        
        def switch_to_cart_tab():
            mock_tab_state["active_tab"] = "cart"
            return "Cart tab activated"
        
        def switch_to_suggestions_tab():
            mock_tab_state["active_tab"] = "suggestions"
            return "Suggestions tab activated"
        
        def get_active_tab_content():
            if mock_tab_state["active_tab"] == "cart":
                return {
                    "type": "cart",
                    "content": "Cart contents displayed here",
                    "actions": ["clear_cart", "checkout"]
                }
            else:
                return {
                    "type": "suggestions",
                    "content": "Product suggestions displayed here",
                    "actions": ["view_product", "add_to_cart"]
                }
        
        # Test initial state
        initial_content = get_active_tab_content()
        assert initial_content["type"] == "suggestions"
        
        # Test switching to cart tab
        switch_result = switch_to_cart_tab()
        assert "Cart tab activated" in switch_result
        
        cart_content = get_active_tab_content()
        assert cart_content["type"] == "cart"
        assert "clear_cart" in cart_content["actions"]
        assert "checkout" in cart_content["actions"]
        
        # Test switching back to suggestions
        switch_to_suggestions_tab()
        suggestions_content = get_active_tab_content()
        assert suggestions_content["type"] == "suggestions"
        assert "add_to_cart" in suggestions_content["actions"]
    
    def test_responsive_cart_display_simulation(self):
        """Test responsive cart display for different screen sizes."""
        
        def format_cart_for_screen_size(cart_data, screen_size="desktop"):
            items = cart_data.get("items", [])
            summary = cart_data.get("summary", {})
            
            if screen_size == "mobile":
                # Compact mobile format
                display = {
                    "format": "compact",
                    "header": f"{summary.get('total_items', 0)} items • ${summary.get('total_value', 0.0):.2f}",
                    "items": [
                        {
                            "title": item["product_title"][:20] + "..." if len(item["product_title"]) > 20 else item["product_title"],
                            "quantity": item["quantity"],
                            "price": f"${item['subtotal']:.2f}"
                        }
                        for item in items
                    ]
                }
            elif screen_size == "tablet":
                # Medium tablet format
                display = {
                    "format": "medium",
                    "header": f"Cart: {summary.get('total_items', 0)} items",
                    "total": f"Total: ${summary.get('total_value', 0.0):.2f}",
                    "items": [
                        {
                            "title": item["product_title"],
                            "quantity": f"Qty: {item['quantity']}",
                            "price": f"${item['subtotal']:.2f}",
                            "image": item.get("product_image_url")
                        }
                        for item in items
                    ]
                }
            else:
                # Full desktop format
                display = {
                    "format": "full",
                    "header": f"Shopping Cart ({summary.get('total_items', 0)} items)",
                    "total": f"Total: ${summary.get('total_value', 0.0):.2f}",
                    "items": [
                        {
                            "title": item["product_title"],
                            "quantity": item["quantity"],
                            "unit_price": f"${item['product_price']:.2f}",
                            "subtotal": f"${item['subtotal']:.2f}",
                            "image": item.get("product_image_url"),
                            "metadata": item.get("product_metadata", {})
                        }
                        for item in items
                    ]
                }
            
            return display
        
        # Test data
        cart_data = {
            "items": [
                {
                    "product_title": "High-Performance Gaming Laptop with RGB Keyboard",
                    "quantity": 1,
                    "product_price": 1299.99,
                    "subtotal": 1299.99,
                    "product_image_url": "https://example.com/laptop.jpg",
                    "product_metadata": {"brand": "TechBrand", "category": "electronics"}
                }
            ],
            "summary": {
                "total_items": 1,
                "total_value": 1299.99
            }
        }
        
        # Test mobile format
        mobile_display = format_cart_for_screen_size(cart_data, "mobile")
        assert mobile_display["format"] == "compact"
        assert "1 items • $1299.99" in mobile_display["header"]
        assert len(mobile_display["items"][0]["title"]) <= 23  # Truncated title
        
        # Test tablet format
        tablet_display = format_cart_for_screen_size(cart_data, "tablet")
        assert tablet_display["format"] == "medium"
        assert "Cart: 1 items" in tablet_display["header"]
        assert "Total: $1299.99" in tablet_display["total"]
        assert "image" in tablet_display["items"][0]
        
        # Test desktop format
        desktop_display = format_cart_for_screen_size(cart_data, "desktop")
        assert desktop_display["format"] == "full"
        assert "Shopping Cart (1 items)" in desktop_display["header"]
        assert "unit_price" in desktop_display["items"][0]
        assert "metadata" in desktop_display["items"][0]
        assert desktop_display["items"][0]["metadata"]["brand"] == "TechBrand"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])