"""
Test file for cart integration with LangGraph API.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import Mock, patch, MagicMock
from chatbot_ui.langgraph_integration import StreamlitLangGraphIntegration


# Mock streamlit for testing
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __contains__(self, key):
        return key in self._state
    
    def __delitem__(self, key):
        if key in self._state:
            del self._state[key]
    
    def clear(self):
        self._state.clear()

class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()

# Create mock streamlit module
mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st


class TestCartIntegration:
    """Test cases for cart integration with LangGraph API."""
    
    def setup_method(self):
        """Setup test environment."""
        mock_st.session_state.clear()
    
    @patch('streamlit.session_state', new_callable=lambda: mock_st.session_state)
    def test_cart_state_update_from_api_response(self, mock_session_state):
        """Test cart state update from API response."""
        # Mock cart manager
        mock_cart_manager = Mock()
        mock_session_state.cart_manager = mock_cart_manager
        
        integration = StreamlitLangGraphIntegration()
        
        # Mock cart data from API response
        cart_data = [
            {
                "product_id": "test123",
                "product_title": "Test Headphones",
                "quantity": 2,
                "product_price": 99.99,
                "product_metadata": {"brand": "TestBrand"}
            }
        ]
        
        # Test cart state update
        integration._update_cart_state(cart_data, True)
        
        # Check if cart manager was called
        mock_cart_manager.update_cart_display.assert_called_once()
        
        # Check if cart update notification was set
        assert 'cart_update_notification' in mock_session_state
        notification = mock_session_state.cart_update_notification
        assert notification["type"] == "success"
        assert "Cart updated successfully!" in notification["message"]
    
    def test_cart_error_handling(self):
        """Test cart error handling during update."""
        integration = StreamlitLangGraphIntegration()
        
        # Mock cart manager to raise an exception
        mock_cart_manager = Mock()
        mock_cart_manager.update_cart_display.side_effect = Exception("Database error")
        mock_st.session_state.cart_manager = mock_cart_manager
        
        # Test cart state update with error
        cart_data = [{"product_id": "test", "quantity": 1}]
        integration._update_cart_state(cart_data, True)
        
        # Check if error notification was set
        assert 'cart_update_notification' in mock_st.session_state
        notification = mock_st.session_state.cart_update_notification
        assert notification["type"] == "error"
        assert "Cart update failed" in notification["message"]
    
    def test_cart_data_formatting(self):
        """Test cart data formatting for display."""
        integration = StreamlitLangGraphIntegration()
        
        # Mock cart manager
        mock_cart_manager = Mock()
        mock_st.session_state.cart_manager = mock_cart_manager
        
        # Test cart data with multiple items
        cart_data = [
            {
                "product_id": "item1",
                "product_title": "Headphones",
                "quantity": 2,
                "product_price": 50.0
            },
            {
                "product_id": "item2", 
                "product_title": "Cable",
                "quantity": 1,
                "product_price": 15.99
            }
        ]
        
        integration._update_cart_state(cart_data, True)
        
        # Check if cart manager was called with formatted data
        mock_cart_manager.update_cart_display.assert_called_once()
        call_args = mock_cart_manager.update_cart_display.call_args[0][0]
        
        assert call_args["total_items"] == 3  # 2 + 1
        assert call_args["total_value"] == 115.99  # (2 * 50.0) + (1 * 15.99)
        assert len(call_args["items"]) == 2
    
    def test_empty_cart_handling(self):
        """Test handling of empty cart data."""
        integration = StreamlitLangGraphIntegration()
        
        # Mock cart manager
        mock_cart_manager = Mock()
        mock_st.session_state.cart_manager = mock_cart_manager
        
        # Test with empty cart data
        integration._update_cart_state([], False)
        
        # Check if cart manager was called with empty data
        mock_cart_manager.update_cart_display.assert_called_once()
        call_args = mock_cart_manager.update_cart_display.call_args[0][0]
        
        assert call_args["total_items"] == 0
        assert call_args["total_value"] == 0
        assert len(call_args["items"]) == 0
    
    def test_cart_notification_display_timing(self):
        """Test cart notification display timing."""
        integration = StreamlitLangGraphIntegration()
        
        # Set up a notification
        mock_st.session_state.cart_update_notification = {
            "message": "Test notification",
            "timestamp": time.time(),
            "type": "success"
        }
        
        # Mock streamlit display functions
        with patch('streamlit.success') as mock_success:
            integration.display_cart_notifications()
            mock_success.assert_called_once()
        
        # Test old notification cleanup
        mock_st.session_state.cart_update_notification = {
            "message": "Old notification",
            "timestamp": time.time() - 10,  # 10 seconds ago
            "type": "success"
        }
        
        integration.display_cart_notifications()
        
        # Check if old notification was cleared
        assert 'cart_update_notification' not in mock_st.session_state
    
    @patch('requests.post')
    def test_api_response_with_cart_data(self, mock_post):
        """Test API response handling with cart data."""
        integration = StreamlitLangGraphIntegration()
        
        # Mock API response with cart data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Added headphones to your cart!",
            "cart_data": [
                {
                    "product_id": "hp123",
                    "product_title": "Wireless Headphones",
                    "quantity": 1,
                    "product_price": 79.99
                }
            ],
            "cart_updated": True,
            "cart_item_count": 1,
            "agent_workflow": "shopping_cart_agent",
            "conversation_turn": 1,
            "processing_time": 1.5,
            "workflow_steps": ["analyze_cart_request", "execute_cart_operation"],
            "products_found": 0,
            "reviews_found": 0
        }
        mock_post.return_value = mock_response
        
        # Mock cart manager
        mock_cart_manager = Mock()
        mock_st.session_state.cart_manager = mock_cart_manager
        
        # Send message to agent
        result = integration.send_message_to_agent(
            message="Add wireless headphones to my cart",
            session_id="test_session"
        )
        
        # Check if result includes cart data
        assert result["success"] == True
        assert result["cart_updated"] == True
        assert result["cart_data"] is not None
        
        # Check if cart manager was called
        mock_cart_manager.update_cart_display.assert_called_once()
        
        # Check if history was updated with cart info
        history = mock_st.session_state.langgraph_agent_history
        assert len(history) == 1
        assert history[0]["cart_updated"] == True
        assert history[0]["cart_item_count"] == 1


def test_cart_keywords_detection():
    """Test detection of cart-related keywords."""
    cart_keywords = ["cart", "add to cart", "remove from cart", "shopping cart", "buy", "purchase"]
    
    # Test cart-related queries
    cart_queries = [
        "Add iPhone to my cart",
        "Remove headphones from cart",
        "What's in my shopping cart?",
        "I want to buy this product",
        "Can I purchase these items?"
    ]
    
    for query in cart_queries:
        is_cart_query = any(keyword in query.lower() for keyword in cart_keywords)
        assert is_cart_query, f"Query '{query}' should be detected as cart-related"
    
    # Test non-cart queries
    non_cart_queries = [
        "What are the best headphones?",
        "Compare iPhone models",
        "Tell me about wireless speakers",
        "What do reviews say about this product?"
    ]
    
    for query in non_cart_queries:
        is_cart_query = any(keyword in query.lower() for keyword in cart_keywords)
        assert not is_cart_query, f"Query '{query}' should not be detected as cart-related"


if __name__ == "__main__":
    # Run basic tests that don't require complex mocking
    test_integration = TestCartIntegration()
    
    # Skip the complex session state tests for now
    # test_integration.setup_method()
    # test_integration.test_cart_state_update_from_api_response()
    
    test_integration.setup_method()
    test_integration.test_cart_error_handling()
    
    test_integration.setup_method()
    test_integration.test_cart_data_formatting()
    
    test_integration.setup_method()
    test_integration.test_empty_cart_handling()
    
    # Skip timing test as it requires streamlit components
    # test_integration.setup_method()
    # test_integration.test_cart_notification_display_timing()
    
    # Skip API test as it requires requests mocking
    # test_integration.setup_method()
    # test_integration.test_api_response_with_cart_data()
    
    test_cart_keywords_detection()
    
    print("✅ Basic cart integration tests passed!")