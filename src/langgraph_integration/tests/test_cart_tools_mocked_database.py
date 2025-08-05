"""
Unit tests for cart tool functionality with mocked database operations.
Tests all cart tools with various edge cases and error conditions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from decimal import Decimal

from ..tools.shopping_cart_tools import (
    AddToCartTool, RemoveFromCartTool, ListCartTool, ClearCartTool,
    create_cart_tools, get_cart_tool_by_name
)
from ..state.shopping_cart_manager import ShoppingCartManager


class TestAddToCartToolMockedDatabase:
    """Test AddToCartTool with mocked database operations."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager with database operations."""
        manager = Mock(spec=ShoppingCartManager)
        return manager
    
    @pytest.fixture
    def add_tool(self, mock_cart_manager):
        """Create AddToCartTool with mock manager."""
        return AddToCartTool(cart_manager=mock_cart_manager)
    
    def test_add_new_item_success(self, add_tool, mock_cart_manager):
        """Test successfully adding new item to cart."""
        # Mock successful add operation
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added 2 x Gaming Laptop to cart",
            "item": {
                "id": "item_123",
                "product_id": "LAPTOP_001",
                "product_title": "Gaming Laptop",
                "quantity": 2,
                "product_price": 1299.99,
                "subtotal": 2599.98,
                "added_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            },
            "action": "added"
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="LAPTOP_001",
                product_title="Gaming Laptop",
                quantity=2,
                price=1299.99,
                image_url="https://example.com/laptop.jpg",
                metadata={"category": "electronics", "brand": "TechBrand"}
            )
        
        # Verify result
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["action"] == "added"
        assert result["item"]["product_id"] == "LAPTOP_001"
        assert result["item"]["quantity"] == 2
        assert result["item"]["subtotal"] == 2599.98
        assert "Added 2 x Gaming Laptop to cart" in result["message"]
        
        # Verify cart manager was called correctly
        mock_cart_manager.add_item.assert_called_once_with(
            session_id="test_session",
            product_id="LAPTOP_001",
            product_title="Gaming Laptop",
            quantity=2,
            price=1299.99,
            image_url="https://example.com/laptop.jpg",
            metadata={"category": "electronics", "brand": "TechBrand"}
        )
    
    def test_add_existing_item_quantity_update(self, add_tool, mock_cart_manager):
        """Test adding to existing item (quantity update)."""
        # Mock quantity update operation
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Updated quantity to 5",
            "item": {
                "id": "item_123",
                "product_id": "PHONE_001",
                "product_title": "Smartphone",
                "quantity": 5,  # Updated from 3 to 5
                "product_price": 699.99,
                "subtotal": 3499.95,
                "added_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T13:00:00"
            },
            "action": "updated"
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="PHONE_001",
                product_title="Smartphone",
                quantity=2,  # Adding 2 more to existing 3
                price=699.99
            )
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["action"] == "updated"
        assert result["item"]["quantity"] == 5
        assert "Updated quantity to 5" in result["message"]
    
    def test_add_item_database_error(self, add_tool, mock_cart_manager):
        """Test add item with database error."""
        # Mock database error
        mock_cart_manager.add_item.return_value = {
            "success": False,
            "error": "Database connection timeout",
            "item": None
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="ERROR_PRODUCT",
                product_title="Error Product",
                quantity=1
            )
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert "Database connection timeout" in result["error"]
        assert result["item"] is None
    
    def test_add_item_validation_errors(self, add_tool):
        """Test add item input validation."""
        test_cases = [
            # (product_id, product_title, quantity, price, expected_error)
            ("", "Valid Title", 1, None, "Product ID is required"),
            ("VALID_ID", "", 1, None, "Product title is required"),
            ("VALID_ID", "Valid Title", 0, None, "Quantity must be greater than 0"),
            ("VALID_ID", "Valid Title", -1, None, "Quantity must be greater than 0"),
            ("VALID_ID", "Valid Title", 101, None, "Quantity cannot exceed 100"),
            ("VALID_ID", "Valid Title", 1, -10.0, "Price cannot be negative"),
        ]
        
        for product_id, product_title, quantity, price, expected_error in test_cases:
            result = add_tool._run(
                product_id=product_id,
                product_title=product_title,
                quantity=quantity,
                price=price
            )
            
            assert result["success"] is False
            assert expected_error in result["error"]
            assert result["cart_updated"] is False
    
    def test_add_item_with_complex_metadata(self, add_tool, mock_cart_manager):
        """Test adding item with complex metadata."""
        complex_metadata = {
            "category": "electronics",
            "subcategory": "laptops",
            "brand": "TechBrand",
            "model": "Pro-X1",
            "specifications": {
                "cpu": "Intel i7",
                "ram": "16GB",
                "storage": "1TB SSD"
            },
            "features": ["backlit keyboard", "fingerprint reader", "USB-C"],
            "warranty": "2 years",
            "rating": 4.5,
            "review_count": 1250
        }
        
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added laptop with detailed specs",
            "item": {
                "product_id": "COMPLEX_LAPTOP",
                "product_title": "Professional Laptop",
                "quantity": 1,
                "product_metadata": complex_metadata
            },
            "action": "added"
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="COMPLEX_LAPTOP",
                product_title="Professional Laptop",
                quantity=1,
                metadata=complex_metadata
            )
        
        assert result["success"] is True
        mock_cart_manager.add_item.assert_called_once()
        call_args = mock_cart_manager.add_item.call_args
        assert call_args[1]["metadata"] == complex_metadata
    
    def test_add_item_exception_handling(self, add_tool, mock_cart_manager):
        """Test exception handling in add item."""
        # Mock cart manager to raise exception
        mock_cart_manager.add_item.side_effect = Exception("Unexpected database error")
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="EXCEPTION_TEST",
                product_title="Exception Test Product",
                quantity=1
            )
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert "Tool error" in result["error"]
        assert "Unexpected database error" in result["error"]


class TestRemoveFromCartToolMockedDatabase:
    """Test RemoveFromCartTool with mocked database operations."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def remove_tool(self, mock_cart_manager):
        """Create RemoveFromCartTool with mock manager."""
        return RemoveFromCartTool(cart_manager=mock_cart_manager)
    
    def test_remove_item_completely(self, remove_tool, mock_cart_manager):
        """Test completely removing item from cart."""
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Removed Gaming Mouse from cart",
            "item": {
                "id": "item_456",
                "product_id": "MOUSE_001",
                "product_title": "Gaming Mouse",
                "quantity": 0,  # Completely removed
                "product_price": 79.99
            },
            "action": "removed",
            "removed_completely": True
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(product_id="MOUSE_001")
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["action"] == "removed"
        assert result["removed_completely"] is True
        assert "Removed Gaming Mouse from cart" in result["message"]
        
        # Verify cart manager was called correctly
        mock_cart_manager.remove_item.assert_called_once_with(
            session_id="test_session",
            product_id="MOUSE_001",
            quantity=None  # Remove all
        )
    
    def test_remove_item_partial_quantity(self, remove_tool, mock_cart_manager):
        """Test removing partial quantity from cart."""
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Updated quantity to 2",
            "item": {
                "id": "item_789",
                "product_id": "CABLE_001",
                "product_title": "USB Cable",
                "quantity": 2,  # Reduced from 5 to 2
                "product_price": 15.99,
                "subtotal": 31.98
            },
            "action": "updated",
            "removed_completely": False
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(
                product_id="CABLE_001",
                quantity=3  # Remove 3 out of 5
            )
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["action"] == "updated"
        assert result["removed_completely"] is False
        assert result["item"]["quantity"] == 2
        
        mock_cart_manager.remove_item.assert_called_once_with(
            session_id="test_session",
            product_id="CABLE_001",
            quantity=3
        )
    
    def test_remove_nonexistent_item(self, remove_tool, mock_cart_manager):
        """Test removing item that doesn't exist in cart."""
        mock_cart_manager.remove_item.return_value = {
            "success": False,
            "error": "Item not found in cart",
            "item": None,
            "removed_completely": False
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(product_id="NONEXISTENT_ITEM")
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert result["removed_completely"] is False
        assert "Item not found in cart" in result["error"]
    
    def test_remove_item_quantity_exceeds_available(self, remove_tool, mock_cart_manager):
        """Test removing more quantity than available."""
        # When quantity exceeds available, should remove completely
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Removed all 2 items (requested 5)",
            "item": {
                "product_id": "LIMITED_ITEM",
                "product_title": "Limited Item",
                "quantity": 0
            },
            "action": "removed",
            "removed_completely": True
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(
                product_id="LIMITED_ITEM",
                quantity=5  # More than available
            )
        
        assert result["success"] is True
        assert result["removed_completely"] is True
        assert "Removed all 2 items" in result["message"]
    
    def test_remove_item_database_error(self, remove_tool, mock_cart_manager):
        """Test remove item with database error."""
        mock_cart_manager.remove_item.return_value = {
            "success": False,
            "error": "Database lock timeout",
            "item": None,
            "removed_completely": False
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(product_id="DB_ERROR_ITEM")
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert "Database lock timeout" in result["error"]


class TestListCartToolMockedDatabase:
    """Test ListCartTool with mocked database operations."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def list_tool(self, mock_cart_manager):
        """Create ListCartTool with mock manager."""
        return ListCartTool(cart_manager=mock_cart_manager)
    
    def test_list_cart_with_multiple_items(self, list_tool, mock_cart_manager):
        """Test listing cart with multiple items."""
        mock_items = [
            {
                "id": "item_1",
                "product_id": "LAPTOP_001",
                "product_title": "Gaming Laptop",
                "quantity": 1,
                "product_price": 1299.99,
                "subtotal": 1299.99,
                "product_image_url": "https://example.com/laptop.jpg",
                "product_metadata": {"category": "electronics", "brand": "TechBrand"},
                "added_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            },
            {
                "id": "item_2",
                "product_id": "MOUSE_001",
                "product_title": "Gaming Mouse",
                "quantity": 2,
                "product_price": 79.99,
                "subtotal": 159.98,
                "product_image_url": "https://example.com/mouse.jpg",
                "product_metadata": {"category": "accessories"},
                "added_at": "2024-01-01T13:00:00",
                "updated_at": "2024-01-01T13:00:00"
            },
            {
                "id": "item_3",
                "product_id": "KEYBOARD_001",
                "product_title": "Mechanical Keyboard",
                "quantity": 1,
                "product_price": 149.99,
                "subtotal": 149.99,
                "product_image_url": None,  # No image
                "product_metadata": {},
                "added_at": "2024-01-01T14:00:00",
                "updated_at": "2024-01-01T14:00:00"
            }
        ]
        
        mock_summary = {
            "session_id": "test_session",
            "total_items": 4,  # 1 + 2 + 1
            "total_value": 1609.96,  # Sum of subtotals
            "unique_products": 3,
            "is_empty": False,
            "last_updated": "2024-01-01T14:00:00"
        }
        
        mock_cart_manager.get_cart_contents.return_value = mock_items
        mock_cart_manager.get_cart_summary.return_value = mock_summary
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            result = list_tool._run(
                include_summary=True,
                format_type="detailed"
            )
        
        # Verify result structure
        assert result["success"] is True
        assert result["item_count"] == 3  # Unique products
        assert result["is_empty"] is False
        assert result["total_items"] == 4  # Total quantity
        assert result["total_value"] == 1609.96
        assert len(result["cart_items"]) == 3
        
        # Verify item details
        laptop_item = next(item for item in result["cart_items"] if item["product_id"] == "LAPTOP_001")
        assert laptop_item["product_title"] == "Gaming Laptop"
        assert laptop_item["quantity"] == 1
        assert laptop_item["subtotal"] == 1299.99
        assert "metadata" in laptop_item  # Detailed format includes metadata
        
        mouse_item = next(item for item in result["cart_items"] if item["product_id"] == "MOUSE_001")
        assert mouse_item["quantity"] == 2
        assert mouse_item["subtotal"] == 159.98
    
    def test_list_empty_cart(self, list_tool, mock_cart_manager):
        """Test listing empty cart."""
        mock_cart_manager.get_cart_contents.return_value = []
        mock_cart_manager.get_cart_summary.return_value = {
            "session_id": "test_session",
            "total_items": 0,
            "total_value": 0.0,
            "unique_products": 0,
            "is_empty": True,
            "last_updated": None
        }
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            result = list_tool._run()
        
        assert result["success"] is True
        assert result["item_count"] == 0
        assert result["is_empty"] is True
        assert result["total_items"] == 0
        assert result["total_value"] == 0.0
        assert result["cart_items"] == []
        assert result["message"] == "Your cart is empty"
    
    def test_list_cart_format_types(self, list_tool, mock_cart_manager):
        """Test different cart listing format types."""
        mock_item = {
            "id": "item_1",
            "product_id": "TEST_PRODUCT",
            "product_title": "Test Product",
            "quantity": 2,
            "product_price": 50.0,
            "subtotal": 100.0,
            "product_image_url": "https://example.com/test.jpg",
            "product_metadata": {"category": "test", "brand": "TestBrand"},
            "added_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00"
        }
        
        mock_cart_manager.get_cart_contents.return_value = [mock_item]
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 2,
            "total_value": 100.0,
            "unique_products": 1,
            "is_empty": False
        }
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            # Test minimal format
            result_minimal = list_tool._run(format_type="minimal")
            minimal_item = result_minimal["cart_items"][0]
            assert set(minimal_item.keys()) == {"product_title", "quantity"}
            
            # Test summary format
            result_summary = list_tool._run(format_type="summary")
            summary_item = result_summary["cart_items"][0]
            expected_summary_keys = {"product_id", "product_title", "quantity", "subtotal"}
            assert set(summary_item.keys()) == expected_summary_keys
            
            # Test detailed format
            result_detailed = list_tool._run(format_type="detailed")
            detailed_item = result_detailed["cart_items"][0]
            assert "metadata" in detailed_item
            assert "added_at" in detailed_item
            assert "updated_at" in detailed_item
            assert "image_url" in detailed_item
    
    def test_list_cart_database_error(self, list_tool, mock_cart_manager):
        """Test list cart with database error."""
        mock_cart_manager.get_cart_contents.side_effect = Exception("Database connection failed")
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            result = list_tool._run()
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert "Tool error" in result["error"]
        assert "Database connection failed" in result["error"]
    
    def test_list_cart_with_corrupted_data(self, list_tool, mock_cart_manager):
        """Test list cart with corrupted item data."""
        # Mock corrupted item data
        corrupted_items = [
            {
                "id": "item_1",
                "product_id": "VALID_ITEM",
                "product_title": "Valid Item",
                "quantity": 1,
                "product_price": 50.0,
                "subtotal": 50.0
            },
            {
                # Missing required fields
                "id": "item_2",
                "product_id": "CORRUPTED_ITEM"
                # Missing product_title, quantity, etc.
            },
            {
                "id": "item_3",
                "product_id": "INVALID_PRICE_ITEM",
                "product_title": "Invalid Price Item",
                "quantity": 1,
                "product_price": "not_a_number",  # Invalid price
                "subtotal": None
            }
        ]
        
        mock_cart_manager.get_cart_contents.return_value = corrupted_items
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 3,
            "total_value": 50.0,
            "unique_products": 3,
            "is_empty": False
        }
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            result = list_tool._run()
        
        # Should handle corrupted data gracefully
        assert result["success"] is True
        # Should include valid items and handle corrupted ones
        assert len(result["cart_items"]) >= 1  # At least the valid item


class TestClearCartToolMockedDatabase:
    """Test ClearCartTool with mocked database operations."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def clear_tool(self, mock_cart_manager):
        """Create ClearCartTool with mock manager."""
        return ClearCartTool(cart_manager=mock_cart_manager)
    
    def test_clear_cart_with_items(self, clear_tool, mock_cart_manager):
        """Test clearing cart with multiple items."""
        mock_cleared_items = [
            {"product_id": "LAPTOP_001", "product_title": "Gaming Laptop", "quantity": 1},
            {"product_id": "MOUSE_001", "product_title": "Gaming Mouse", "quantity": 2},
            {"product_id": "KEYBOARD_001", "product_title": "Mechanical Keyboard", "quantity": 1}
        ]
        
        mock_cart_manager.clear_cart.return_value = {
            "success": True,
            "message": "Cleared 4 items from cart",
            "items_removed": 4,  # Total quantity removed
            "cleared_items": mock_cleared_items
        }
        
        with patch.object(clear_tool, '_get_session_id', return_value="test_session"):
            result = clear_tool._run()
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["items_removed"] == 4
        assert len(result["cleared_items"]) == 3  # 3 unique products
        assert "Cleared 4 items from cart" in result["message"]
        
        # Verify specific items were cleared
        laptop_item = next(item for item in result["cleared_items"] if item["product_id"] == "LAPTOP_001")
        assert laptop_item["product_title"] == "Gaming Laptop"
        assert laptop_item["quantity"] == 1
        
        mouse_item = next(item for item in result["cleared_items"] if item["product_id"] == "MOUSE_001")
        assert mouse_item["quantity"] == 2
        
        mock_cart_manager.clear_cart.assert_called_once_with("test_session")
    
    def test_clear_empty_cart(self, clear_tool, mock_cart_manager):
        """Test clearing already empty cart."""
        mock_cart_manager.clear_cart.return_value = {
            "success": True,
            "message": "Cleared 0 items from cart",
            "items_removed": 0,
            "cleared_items": []
        }
        
        with patch.object(clear_tool, '_get_session_id', return_value="test_session"):
            result = clear_tool._run()
        
        assert result["success"] is True
        assert result["cart_updated"] is True  # Still considered an update
        assert result["items_removed"] == 0
        assert result["cleared_items"] == []
        assert "Cleared 0 items from cart" in result["message"]
    
    def test_clear_cart_database_error(self, clear_tool, mock_cart_manager):
        """Test clear cart with database error."""
        mock_cart_manager.clear_cart.return_value = {
            "success": False,
            "error": "Database transaction failed",
            "items_removed": 0,
            "cleared_items": []
        }
        
        with patch.object(clear_tool, '_get_session_id', return_value="test_session"):
            result = clear_tool._run()
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert result["items_removed"] == 0
        assert "Database transaction failed" in result["error"]
    
    def test_clear_cart_partial_failure(self, clear_tool, mock_cart_manager):
        """Test clear cart with partial failure."""
        # Some items cleared, but operation had issues
        mock_cart_manager.clear_cart.return_value = {
            "success": False,
            "error": "Some items could not be removed due to constraints",
            "items_removed": 2,  # Partial success
            "cleared_items": [
                {"product_id": "ITEM_1", "product_title": "Item 1", "quantity": 1},
                {"product_id": "ITEM_2", "product_title": "Item 2", "quantity": 1}
            ]
        }
        
        with patch.object(clear_tool, '_get_session_id', return_value="test_session"):
            result = clear_tool._run()
        
        assert result["success"] is False
        assert result["cart_updated"] is True  # Some items were removed
        assert result["items_removed"] == 2
        assert len(result["cleared_items"]) == 2
        assert "Some items could not be removed" in result["error"]


class TestCartToolsIntegrationMocked:
    """Test integration between different cart tools with mocked database."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager for integration tests."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def cart_tools(self, mock_cart_manager):
        """Create all cart tools with shared mock manager."""
        return {
            "add": AddToCartTool(cart_manager=mock_cart_manager),
            "remove": RemoveFromCartTool(cart_manager=mock_cart_manager),
            "list": ListCartTool(cart_manager=mock_cart_manager),
            "clear": ClearCartTool(cart_manager=mock_cart_manager)
        }
    
    def test_add_list_remove_workflow(self, cart_tools, mock_cart_manager):
        """Test workflow: add item -> list cart -> remove item."""
        session_id = "workflow_session"
        
        # Step 1: Add item
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added item to cart",
            "item": {"product_id": "WORKFLOW_ITEM", "quantity": 1},
            "action": "added"
        }
        
        with patch.object(cart_tools["add"], '_get_session_id', return_value=session_id):
            add_result = cart_tools["add"]._run(
                product_id="WORKFLOW_ITEM",
                product_title="Workflow Test Item",
                quantity=1
            )
        
        assert add_result["success"] is True
        assert add_result["cart_updated"] is True
        
        # Step 2: List cart
        mock_cart_manager.get_cart_contents.return_value = [
            {
                "product_id": "WORKFLOW_ITEM",
                "product_title": "Workflow Test Item",
                "quantity": 1,
                "product_price": 25.0,
                "subtotal": 25.0
            }
        ]
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 1,
            "total_value": 25.0,
            "unique_products": 1,
            "is_empty": False
        }
        
        with patch.object(cart_tools["list"], '_get_session_id', return_value=session_id):
            list_result = cart_tools["list"]._run()
        
        assert list_result["success"] is True
        assert list_result["item_count"] == 1
        assert list_result["cart_items"][0]["product_id"] == "WORKFLOW_ITEM"
        
        # Step 3: Remove item
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Removed item from cart",
            "item": {"product_id": "WORKFLOW_ITEM", "quantity": 0},
            "action": "removed",
            "removed_completely": True
        }
        
        with patch.object(cart_tools["remove"], '_get_session_id', return_value=session_id):
            remove_result = cart_tools["remove"]._run(product_id="WORKFLOW_ITEM")
        
        assert remove_result["success"] is True
        assert remove_result["removed_completely"] is True
        
        # Verify all tools used the same session
        mock_cart_manager.add_item.assert_called_with(
            session_id=session_id,
            product_id="WORKFLOW_ITEM",
            product_title="Workflow Test Item",
            quantity=1,
            price=None,
            image_url=None,
            metadata={}
        )
        mock_cart_manager.get_cart_contents.assert_called_with(session_id)
        mock_cart_manager.remove_item.assert_called_with(
            session_id=session_id,
            product_id="WORKFLOW_ITEM",
            quantity=None
        )
    
    def test_bulk_operations_workflow(self, cart_tools, mock_cart_manager):
        """Test bulk operations: add multiple items -> list -> clear all."""
        session_id = "bulk_session"
        
        # Add multiple items
        items_to_add = [
            ("BULK_ITEM_1", "Bulk Item 1", 2),
            ("BULK_ITEM_2", "Bulk Item 2", 1),
            ("BULK_ITEM_3", "Bulk Item 3", 3)
        ]
        
        for i, (product_id, title, quantity) in enumerate(items_to_add):
            mock_cart_manager.add_item.return_value = {
                "success": True,
                "message": f"Added {title}",
                "item": {"product_id": product_id, "quantity": quantity},
                "action": "added"
            }
            
            with patch.object(cart_tools["add"], '_get_session_id', return_value=session_id):
                result = cart_tools["add"]._run(
                    product_id=product_id,
                    product_title=title,
                    quantity=quantity
                )
            
            assert result["success"] is True
        
        # List all items
        mock_cart_items = [
            {
                "product_id": product_id,
                "product_title": title,
                "quantity": quantity,
                "product_price": 10.0 * (i + 1),
                "subtotal": 10.0 * (i + 1) * quantity
            }
            for i, (product_id, title, quantity) in enumerate(items_to_add)
        ]
        
        mock_cart_manager.get_cart_contents.return_value = mock_cart_items
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 6,  # 2 + 1 + 3
            "total_value": 140.0,  # 20 + 20 + 90
            "unique_products": 3,
            "is_empty": False
        }
        
        with patch.object(cart_tools["list"], '_get_session_id', return_value=session_id):
            list_result = cart_tools["list"]._run()
        
        assert list_result["success"] is True
        assert list_result["item_count"] == 3
        assert list_result["total_items"] == 6
        assert list_result["total_value"] == 140.0
        
        # Clear all items
        mock_cart_manager.clear_cart.return_value = {
            "success": True,
            "message": "Cleared 6 items from cart",
            "items_removed": 6,
            "cleared_items": [
                {"product_id": pid, "product_title": title, "quantity": qty}
                for pid, title, qty in items_to_add
            ]
        }
        
        with patch.object(cart_tools["clear"], '_get_session_id', return_value=session_id):
            clear_result = cart_tools["clear"]._run()
        
        assert clear_result["success"] is True
        assert clear_result["items_removed"] == 6
        assert len(clear_result["cleared_items"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])