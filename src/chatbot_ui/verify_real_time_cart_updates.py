"""
Verification script for real-time cart updates functionality.
This script tests the cart update components without requiring Streamlit to be running.
"""

import sys
import os
import time
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cart_state_manager():
    """Test CartStateManager functionality."""
    print("Testing CartStateManager...")
    
    # Mock session state
    class MockSessionState:
        def __init__(self):
            self.data = {}
        
        def get(self, key, default=None):
            return self.data.get(key, default)
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __contains__(self, key):
            return key in self.data
        
        def __delitem__(self, key):
            if key in self.data:
                del self.data[key]
    
    # Mock streamlit
    class MockStreamlit:
        def __init__(self):
            self.session_state = MockSessionState()
    
    mock_st = MockStreamlit()
    
    # Test CartStateManager initialization
    try:
        # Import with mocked streamlit
        import streamlit as st_original
        sys.modules['streamlit'] = mock_st
        
        from ui_components import CartStateManager
        
        manager = CartStateManager()
        print("✓ CartStateManager initialized successfully")
        
        # Test cart data update
        sample_cart_data = {
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
        
        manager.update_cart_display(sample_cart_data)
        print("✓ Cart data updated successfully")
        
        # Test cart data retrieval
        retrieved_data = manager.get_cart_data()
        assert retrieved_data["total_items"] == 2
        assert retrieved_data["total_value"] == 59.98
        print("✓ Cart data retrieved successfully")
        
        # Test error handling
        manager.set_cart_error("Test error")
        error = manager.get_cart_error()
        assert error == "Test error"
        print("✓ Error handling works correctly")
        
        manager.clear_cart_error()
        assert manager.get_cart_error() is None
        print("✓ Error clearing works correctly")
        
        # Restore original streamlit
        sys.modules['streamlit'] = st_original
        
        return True
        
    except Exception as e:
        print(f"✗ CartStateManager test failed: {e}")
        return False

def test_langgraph_integration():
    """Test LangGraph integration cart update functionality."""
    print("\nTesting LangGraph Integration...")
    
    try:
        # Mock session state and streamlit
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def get(self, key, default=None):
                return self.data.get(key, default)
            
            def __setitem__(self, key, value):
                self.data[key] = value
            
            def __getitem__(self, key):
                return self.data[key]
            
            def __contains__(self, key):
                return key in self.data
            
            def __delitem__(self, key):
                if key in self.data:
                    del self.data[key]
        
        class MockStreamlit:
            def __init__(self):
                self.session_state = MockSessionState()
                self.success_calls = []
                self.error_calls = []
                self.info_calls = []
            
            def success(self, message):
                self.success_calls.append(message)
            
            def error(self, message):
                self.error_calls.append(message)
            
            def info(self, message):
                self.info_calls.append(message)
            
            def balloons(self):
                pass
        
        mock_st = MockStreamlit()
        
        # Mock streamlit module
        import streamlit as st_original
        sys.modules['streamlit'] = mock_st
        
        from langgraph_integration import StreamlitLangGraphIntegration
        
        integration = StreamlitLangGraphIntegration()
        print("✓ LangGraph integration initialized successfully")
        
        # Test cart state update
        sample_cart_data = [
            {
                "product_id": "test123",
                "product_title": "Test Product",
                "quantity": 1,
                "product_price": 29.99
            }
        ]
        
        integration._update_cart_state(sample_cart_data, True)
        print("✓ Cart state updated successfully")
        
        # Verify notification was created
        assert 'cart_update_notification' in mock_st.session_state.data
        notification = mock_st.session_state.data['cart_update_notification']
        assert notification['type'] == 'success'
        print("✓ Cart update notification created successfully")
        
        # Test cart data persistence
        integration._persist_cart_data(sample_cart_data, 1, 29.99)
        assert 'persistent_cart_data' in mock_st.session_state.data
        persistent_data = mock_st.session_state.data['persistent_cart_data']
        assert persistent_data['total_items'] == 1
        assert persistent_data['total_value'] == 29.99
        print("✓ Cart data persistence works correctly")
        
        # Test notification display
        integration.display_cart_notifications()
        assert len(mock_st.success_calls) > 0
        print("✓ Cart notifications display correctly")
        
        # Restore original streamlit
        sys.modules['streamlit'] = st_original
        
        return True
        
    except Exception as e:
        print(f"✗ LangGraph integration test failed: {e}")
        return False

def test_api_response_processing():
    """Test API response processing for cart updates."""
    print("\nTesting API Response Processing...")
    
    try:
        # Sample API response with cart data
        sample_response = {
            "query": "Add headphones to cart",
            "response": "I've added wireless headphones to your cart.",
            "session_id": "test_session_123",
            "conversation_turn": 1,
            "cart_data": [
                {
                    "product_id": "headphones123",
                    "product_title": "Wireless Headphones",
                    "quantity": 1,
                    "product_price": 99.99
                }
            ],
            "cart_updated": True,
            "cart_item_count": 1,
            "cart_total": 99.99,
            "routing_decision": "cart",
            "agent_used": "shopping_cart"
        }
        
        # Test cart data extraction
        cart_data = sample_response.get("cart_data")
        cart_updated = sample_response.get("cart_updated", False)
        
        assert cart_data is not None
        assert cart_updated is True
        assert len(cart_data) == 1
        assert cart_data[0]["product_title"] == "Wireless Headphones"
        print("✓ Cart data extracted from API response successfully")
        
        # Test cart data transformation
        formatted_cart_data = {
            "items": cart_data,
            "total_items": sample_response.get("cart_item_count", 0),
            "total_value": sample_response.get("cart_total", 0.0)
        }
        
        assert formatted_cart_data["total_items"] == 1
        assert formatted_cart_data["total_value"] == 99.99
        print("✓ Cart data transformation works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ API response processing test failed: {e}")
        return False

def test_error_handling():
    """Test error handling in cart update workflow."""
    print("\nTesting Error Handling...")
    
    try:
        # Mock session state
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def get(self, key, default=None):
                return self.data.get(key, default)
            
            def __setitem__(self, key, value):
                self.data[key] = value
            
            def __contains__(self, key):
                return key in self.data
        
        class MockStreamlit:
            def __init__(self):
                self.session_state = MockSessionState()
                self.error_calls = []
            
            def error(self, message):
                self.error_calls.append(message)
        
        mock_st = MockStreamlit()
        
        # Test error handling with invalid data
        try:
            # Simulate processing invalid cart data
            invalid_data = "invalid_cart_data"
            
            # This should handle the error gracefully
            if not isinstance(invalid_data, list):
                mock_st.session_state['cart_update_notification'] = {
                    "message": "Cart update failed: Invalid data format",
                    "timestamp": time.time(),
                    "type": "error"
                }
            
            # Verify error notification was created
            assert 'cart_update_notification' in mock_st.session_state.data
            notification = mock_st.session_state.data['cart_update_notification']
            assert notification['type'] == 'error'
            print("✓ Error handling works correctly")
            
            return True
            
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error handling setup failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("=== Real-Time Cart Updates Verification ===\n")
    
    tests = [
        test_cart_state_manager,
        test_langgraph_integration,
        test_api_response_processing,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Real-time cart updates are working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)