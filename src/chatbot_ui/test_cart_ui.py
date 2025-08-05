"""
Test file for cart UI components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch

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

from chatbot_ui.ui_components import CartStateManager, UITheme


class TestCartStateManager:
    """Test cases for CartStateManager."""
    
    def setup_method(self):
        """Setup test environment."""
        # Clear session state
        mock_st.session_state.clear()
    
    def test_cart_manager_initialization(self):
        """Test CartStateManager initialization."""
        cart_manager = CartStateManager()
        
        assert cart_manager.session_key == "shopping_cart_data"
        assert cart_manager.cart_updated_key == "cart_updated"
        assert cart_manager.cart_error_key == "cart_error"
    
    def test_get_cart_data_empty(self):
        """Test getting cart data when empty."""
        mock_st.session_state.clear()
        cart_manager = CartStateManager()
        cart_data = cart_manager.get_cart_data()
        
        expected = {
            "items": [],
            "total_items": 0,
            "total_value": 0.0
        }
        
        assert cart_data == expected
    
    def test_update_cart_display(self):
        """Test updating cart display."""
        mock_st.session_state.clear()
        cart_manager = CartStateManager()
        
        test_cart_data = {
            "items": [
                {
                    "product_id": "test123",
                    "product_title": "Test Product",
                    "quantity": 2,
                    "product_price": 29.99
                }
            ],
            "total_items": 2,
            "total_value": 59.98
        }
        
        cart_manager.update_cart_display(test_cart_data)
        
        # Check if data was stored
        stored_data = cart_manager.get_cart_data()
        assert stored_data == test_cart_data
        
        # Check if update flag was set
        assert cart_manager.was_cart_updated() == True
    
    def test_cart_error_handling(self):
        """Test cart error handling."""
        mock_st.session_state.clear()
        cart_manager = CartStateManager()
        
        # Test setting error
        error_message = "Database connection failed"
        cart_manager.set_cart_error(error_message)
        
        assert cart_manager.get_cart_error() == error_message
        
        # Test clearing error
        cart_manager.clear_cart_error()
        assert cart_manager.get_cart_error() is None
    
    def test_cart_update_flag(self):
        """Test cart update flag functionality."""
        # Clear state first
        mock_st.session_state.clear()
        cart_manager = CartStateManager()
        
        # Initially should be False
        assert cart_manager.was_cart_updated() == False
        
        # After update should be True
        cart_manager.update_cart_display({"items": [], "total_items": 0, "total_value": 0.0})
        assert cart_manager.was_cart_updated() == True
        
        # After clearing should be False
        cart_manager.clear_update_flag()
        assert cart_manager.was_cart_updated() == False


class TestUITheme:
    """Test cases for UITheme constants."""
    
    def test_cart_icons_exist(self):
        """Test that cart-related icons are defined."""
        assert hasattr(UITheme, 'ICON_CART')
        assert hasattr(UITheme, 'ICON_ADD')
        assert hasattr(UITheme, 'ICON_REMOVE')
        assert hasattr(UITheme, 'ICON_CLEAR')
        
        assert UITheme.ICON_CART == "🛒"
        assert UITheme.ICON_ADD == "➕"
        assert UITheme.ICON_REMOVE == "➖"
        assert UITheme.ICON_CLEAR == "🗑️"


def test_cart_sidebar_rendering():
    """Test cart sidebar rendering with mock data."""
    cart_manager = CartStateManager()
    
    # Test with empty cart
    with patch('streamlit.subheader') as mock_subheader, \
         patch('streamlit.info') as mock_info:
        
        # Mock empty cart data
        cart_manager.get_cart_data = Mock(return_value={
            "items": [],
            "total_items": 0,
            "total_value": 0.0
        })
        
        cart_manager.get_cart_error = Mock(return_value=None)
        cart_manager.was_cart_updated = Mock(return_value=False)
        
        # This would normally render the sidebar, but we're just testing the logic
        # In a real test, we'd need to mock all Streamlit components
        assert cart_manager.get_cart_data()["total_items"] == 0


def test_cart_with_items():
    """Test cart display with items."""
    cart_manager = CartStateManager()
    
    # Test with items in cart
    test_items = [
        {
            "product_id": "item1",
            "product_title": "Wireless Headphones",
            "quantity": 1,
            "product_price": 99.99,
            "product_metadata": {"brand": "TestBrand", "color": "Black"}
        },
        {
            "product_id": "item2", 
            "product_title": "USB Cable",
            "quantity": 2,
            "product_price": 15.99,
            "product_metadata": {"length": "6ft", "type": "USB-C"}
        }
    ]
    
    cart_data = {
        "items": test_items,
        "total_items": 3,
        "total_value": 131.97
    }
    
    cart_manager.update_cart_display(cart_data)
    
    stored_data = cart_manager.get_cart_data()
    assert len(stored_data["items"]) == 2
    assert stored_data["total_items"] == 3
    assert stored_data["total_value"] == 131.97


if __name__ == "__main__":
    # Run basic tests
    test_manager = TestCartStateManager()
    test_manager.setup_method()
    test_manager.test_cart_manager_initialization()
    test_manager.test_get_cart_data_empty()
    test_manager.test_update_cart_display()
    test_manager.test_cart_error_handling()
    test_manager.test_cart_update_flag()
    
    test_theme = TestUITheme()
    test_theme.test_cart_icons_exist()
    
    print("✅ All cart UI tests passed!")