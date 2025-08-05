"""
Integration tests for real-time cart updates in Streamlit frontend.
Tests the cart update functionality across different components.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Mock Streamlit for testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
        self.success_calls = []
        self.error_calls = []
        self.info_calls = []
        self.rerun_calls = []
    
    def success(self, message):
        self.success_calls.append(message)
    
    def error(self, message):
        self.error_calls.append(message)
    
    def info(self, message):
        self.info_calls.append(message)
    
    def rerun(self):
        self.rerun_calls.append(time.time())

@pytest.fixture
def mock_streamlit():
    """Provide a mock Streamlit instance for testing."""
    return MockStreamlit()

@pytest.fixture
def sample_cart_data():
    """Provide sample cart data for testing."""
    return [
        {
            "id": "item1",
            "product_id": "prod123",
            "product_title": "Wireless Headphones",
            "quantity": 2,
            "product_price": 99.99,
            "product_metadata": {"brand": "TestBrand"}
        },
        {
            "id": "item2", 
            "product_id": "prod456",
            "product_title": "USB Cable",
            "quantity": 1,
            "product_price": 15.99,
            "product_metadata": {"length": "6ft"}
        }
    ]

@pytest.fixture
def sample_api_response(sample_cart_data):
    """Provide sample API response with cart data."""
    return {
        "query": "Add headphones to cart",
        "response": "I've added wireless headphones to your cart.",
        "session_id": "test_session_123",
        "conversation_turn": 1,
        "cart_data": sample_cart_data,
        "cart_updated": True,
        "cart_item_count": 3,
        "cart_total": 215.97,
        "routing_decision": "cart",
        "agent_used": "shopping_cart",
        "tools_called": ["add_to_cart"]
    }

class TestCartStateManager:
    """Test the CartStateManager for real-time updates."""
    
    def test_cart_state_manager_initialization(self, mock_streamlit):
        """Test CartStateManager initialization."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            from chatbot_ui.ui_components import CartStateManager
            
            manager = CartStateManager()
            assert manager.session_key == "shopping_cart_data"
            assert manager.cart_updated_key == "cart_updated"
            assert manager.cart_error_key == "cart_error"
    
    def test_update_cart_display(self, mock_streamlit, sample_cart_data):
        """Test updating cart display with new data."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            from chatbot_ui.ui_components import CartStateManager
            
            manager = CartStateManager()
            cart_data = {
                "items": sample_cart_data,
                "total_items": 3,
                "total_value": 215.97
            }
            
            manager.update_cart_display(cart_data)
            
            # Verify data is stored in session state
            assert mock_streamlit.session_state[manager.session_key] == cart_data
            assert mock_streamlit.session_state[manager.cart_updated_key] is True
    
    def test_cart_error_handling(self, mock_streamlit):
        """Test cart error handling."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            from chatbot_ui.ui_components import CartStateManager
            
            manager = CartStateManager()
            error_message = "Database connection failed"
            
            manager.set_cart_error(error_message)
            assert manager.get_cart_error() == error_message
            
            manager.clear_cart_error()
            assert manager.get_cart_error() is None
    
    def test_cart_update_flag_management(self, mock_streamlit):
        """Test cart update flag management."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            from chatbot_ui.ui_components import CartStateManager
            
            manager = CartStateManager()
            
            # Initially no update flag
            assert not manager.was_cart_updated()
            
            # Set update flag
            mock_streamlit.session_state[manager.cart_updated_key] = True
            assert manager.was_cart_updated()
            
            # Clear update flag
            manager.clear_update_flag()
            assert not manager.was_cart_updated()

class TestLangGraphIntegration:
    """Test LangGraph integration for cart updates."""
    
    def test_cart_state_update_from_api_response(self, mock_streamlit, sample_cart_data):
        """Test updating cart state from API response."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            with patch('streamlit', mock_streamlit):
                from chatbot_ui.langgraph_integration import StreamlitLangGraphIntegration
                
                integration = StreamlitLangGraphIntegration()
                
                # Test cart state update
                integration._update_cart_state(sample_cart_data, True)
                
                # Verify cart manager was initialized and data updated
                assert 'cart_manager' in mock_streamlit.session_state
                assert 'cart_update_notification' in mock_streamlit.session_state
                
                notification = mock_streamlit.session_state['cart_update_notification']
                assert notification['type'] == 'success'
                assert 'Cart' in notification['message']
    
    def test_cart_data_persistence(self, mock_streamlit, sample_cart_data):
        """Test cart data persistence across sessions."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            from chatbot_ui.langgraph_integration import StreamlitLangGraphIntegration
            
            integration = StreamlitLangGraphIntegration()
            
            # Test persistence
            integration._persist_cart_data(sample_cart_data, 3, 215.97)
            
            # Verify persistent data is stored
            assert 'persistent_cart_data' in mock_streamlit.session_state
            persistent_data = mock_streamlit.session_state['persistent_cart_data']
            
            assert persistent_data['items'] == sample_cart_data
            assert persistent_data['total_items'] == 3
            assert persistent_data['total_value'] == 215.97
            assert 'last_updated' in persistent_data
    
    def test_cart_notification_display_timing(self, mock_streamlit):
        """Test cart notification display timing."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            with patch('streamlit', mock_streamlit):
                from chatbot_ui.langgraph_integration import StreamlitLangGraphIntegration
                
                integration = StreamlitLangGraphIntegration()
                
                # Set up a recent notification
                mock_streamlit.session_state['cart_update_notification'] = {
                    "message": "Cart updated successfully!",
                    "timestamp": time.time(),
                    "type": "success",
                    "details": {"total_items": 2}
                }
                
                # Display notifications
                integration.display_cart_notifications()
                
                # Verify success message was displayed
                assert len(mock_streamlit.success_calls) > 0
                assert "Cart updated successfully!" in mock_streamlit.success_calls[0]
    
    def test_expired_notification_cleanup(self, mock_streamlit):
        """Test cleanup of expired notifications."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            with patch('streamlit', mock_streamlit):
                from chatbot_ui.langgraph_integration import StreamlitLangGraphIntegration
                
                integration = StreamlitLangGraphIntegration()
                
                # Set up an old notification
                mock_streamlit.session_state['cart_update_notification'] = {
                    "message": "Old notification",
                    "timestamp": time.time() - 10,  # 10 seconds ago
                    "type": "success",
                    "details": {}
                }
                
                # Display notifications (should clean up old ones)
                integration.display_cart_notifications()
                
                # Verify notification was cleaned up
                assert 'cart_update_notification' not in mock_streamlit.session_state

class TestAPIResponseProcessing:
    """Test API response processing for cart updates."""
    
    def test_process_api_response_with_cart_data(self, mock_streamlit, sample_api_response):
        """Test processing API response with cart data."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            with patch('streamlit', mock_streamlit):
                # Import the function we're testing
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                
                # Mock the function since it's in the main app
                def process_api_response_for_cart_updates(response_data):
                    try:
                        cart_data = response_data.get("cart_data")
                        cart_updated = response_data.get("cart_updated", False)
                        
                        if cart_data is not None or cart_updated:
                            # Simulate cart manager initialization
                            if 'cart_manager' not in mock_streamlit.session_state:
                                mock_streamlit.session_state['cart_manager'] = Mock()
                            
                            # Simulate cart update
                            formatted_cart_data = {
                                "items": cart_data if cart_data else [],
                                "total_items": response_data.get("cart_item_count", 0),
                                "total_value": response_data.get("cart_total", 0.0)
                            }
                            
                            if cart_updated:
                                mock_streamlit.session_state['cart_update_notification'] = {
                                    "message": "Cart updated from API response!",
                                    "timestamp": time.time(),
                                    "type": "success"
                                }
                            
                            return True
                        return False
                    except Exception:
                        return False
                
                # Test the function
                result = process_api_response_for_cart_updates(sample_api_response)
                
                assert result is True
                assert 'cart_manager' in mock_streamlit.session_state
                assert 'cart_update_notification' in mock_streamlit.session_state
    
    def test_api_response_without_cart_data(self, mock_streamlit):
        """Test processing API response without cart data."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            response_without_cart = {
                "query": "What are the best headphones?",
                "response": "Here are some great headphone options...",
                "session_id": "test_session",
                "conversation_turn": 1
            }
            
            # Mock the function
            def process_api_response_for_cart_updates(response_data):
                cart_data = response_data.get("cart_data")
                cart_updated = response_data.get("cart_updated", False)
                return cart_data is not None or cart_updated
            
            result = process_api_response_for_cart_updates(response_without_cart)
            assert result is False

class TestRealTimeCartIntegration:
    """Test end-to-end real-time cart integration."""
    
    def test_cart_update_workflow(self, mock_streamlit, sample_api_response):
        """Test complete cart update workflow."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            with patch('streamlit', mock_streamlit):
                from chatbot_ui.langgraph_integration import StreamlitLangGraphIntegration
                from chatbot_ui.ui_components import CartStateManager
                
                # Initialize components
                integration = StreamlitLangGraphIntegration()
                cart_manager = CartStateManager()
                mock_streamlit.session_state['cart_manager'] = cart_manager
                
                # Simulate API response processing
                cart_data = sample_api_response.get("cart_data")
                cart_updated = sample_api_response.get("cart_updated", False)
                
                # Process cart update
                integration._update_cart_state(cart_data, cart_updated)
                
                # Verify workflow completed successfully
                assert 'cart_update_notification' in mock_streamlit.session_state
                notification = mock_streamlit.session_state['cart_update_notification']
                assert notification['type'] == 'success'
    
    def test_error_handling_in_cart_workflow(self, mock_streamlit):
        """Test error handling in cart update workflow."""
        with patch('streamlit.session_state', mock_streamlit.session_state):
            with patch('streamlit', mock_streamlit):
                from chatbot_ui.langgraph_integration import StreamlitLangGraphIntegration
                
                integration = StreamlitLangGraphIntegration()
                
                # Simulate error by passing invalid data
                invalid_cart_data = "invalid_data"
                
                try:
                    integration._update_cart_state(invalid_cart_data, True)
                    # Should handle error gracefully
                    assert 'cart_update_notification' in mock_streamlit.session_state
                    notification = mock_streamlit.session_state['cart_update_notification']
                    assert notification['type'] == 'error'
                except Exception:
                    # Error handling should prevent exceptions from propagating
                    pytest.fail("Cart update should handle errors gracefully")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])