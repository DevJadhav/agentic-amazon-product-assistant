"""
Integration tests for shopping cart tools.
Tests tool interactions with database manager and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from decimal import Decimal

from ..tools.shopping_cart_tools import (
    AddToCartTool, RemoveFromCartTool, ListCartTool, ClearCartTool,
    create_cart_tools, get_cart_tool_by_name, get_cart_tools_info
)
from ..state.shopping_cart_manager import ShoppingCartManager


class TestAddToCartTool:
    """Test suite for AddToCartTool."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def add_tool(self, mock_cart_manager):
        """Create AddToCartTool with mock manager."""
        return AddToCartTool(cart_manager=mock_cart_manager)
    
    def test_successful_add_new_item(self, add_tool, mock_cart_manager):
        """Test successfully adding new item to cart."""
        # Mock successful add operation
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added 2 x Test Product to cart",
            "item": {
                "product_id": "PROD123",
                "product_title": "Test Product",
                "quantity": 2,
                "product_price": 29.99,
                "subtotal": 59.98
            },
            "action": "added"
        }
        
        # Mock session ID
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="PROD123",
                product_title="Test Product",
                quantity=2,
                price=29.99
            )
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["action"] == "added"
        assert result["item"]["product_id"] == "PROD123"
        assert result["item"]["quantity"] == 2
        assert "Added 2 x Test Product to cart" in result["message"]
        
        # Verify cart manager was called correctly
        mock_cart_manager.add_item.assert_called_once_with(
            session_id="test_session",
            product_id="PROD123",
            product_title="Test Product",
            quantity=2,
            price=29.99,
            image_url=None,
            metadata={}
        )
    
    def test_successful_add_with_metadata(self, add_tool, mock_cart_manager):
        """Test adding item with metadata."""
        metadata = {"category": "electronics", "brand": "TestBrand"}
        
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added 1 x Test Product to cart",
            "item": {
                "product_id": "PROD123",
                "product_title": "Test Product",
                "quantity": 1,
                "product_metadata": metadata
            },
            "action": "added"
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="PROD123",
                product_title="Test Product",
                quantity=1,
                metadata=metadata
            )
        
        assert result["success"] is True
        mock_cart_manager.add_item.assert_called_once_with(
            session_id="test_session",
            product_id="PROD123",
            product_title="Test Product",
            quantity=1,
            price=None,
            image_url=None,
            metadata=metadata
        )
    
    def test_add_item_validation_errors(self, add_tool):
        """Test validation errors for add item."""
        
        # Test empty product ID
        result = add_tool._run(
            product_id="",
            product_title="Test Product",
            quantity=1
        )
        assert result["success"] is False
        assert "Product ID is required" in result["error"]
        
        # Test empty product title
        result = add_tool._run(
            product_id="PROD123",
            product_title="",
            quantity=1
        )
        assert result["success"] is False
        assert "Product title is required" in result["error"]
        
        # Test invalid quantity
        result = add_tool._run(
            product_id="PROD123",
            product_title="Test Product",
            quantity=0
        )
        assert result["success"] is False
        assert "Quantity must be greater than 0" in result["error"]
        
        # Test quantity too high
        result = add_tool._run(
            product_id="PROD123",
            product_title="Test Product",
            quantity=101
        )
        assert result["success"] is False
        assert "Quantity cannot exceed 100 items" in result["error"]
        
        # Test negative price
        result = add_tool._run(
            product_id="PROD123",
            product_title="Test Product",
            quantity=1,
            price=-10.0
        )
        assert result["success"] is False
        assert "Price cannot be negative" in result["error"]
    
    def test_add_item_database_error(self, add_tool, mock_cart_manager):
        """Test handling database errors."""
        mock_cart_manager.add_item.return_value = {
            "success": False,
            "error": "Database connection failed"
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="PROD123",
                product_title="Test Product",
                quantity=1
            )
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert "Database connection failed" in result["error"]
    
    def test_add_item_exception_handling(self, add_tool, mock_cart_manager):
        """Test exception handling in add item."""
        mock_cart_manager.add_item.side_effect = Exception("Unexpected error")
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            result = add_tool._run(
                product_id="PROD123",
                product_title="Test Product",
                quantity=1
            )
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert "Tool error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_async_add_item(self, add_tool, mock_cart_manager):
        """Test async add item operation."""
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added 1 x Test Product to cart",
            "item": {"product_id": "PROD123"},
            "action": "added"
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            # Test that async version calls sync version
            result = await add_tool._arun(
                product_id="PROD123",
                product_title="Test Product",
                quantity=1
            )
        
        assert result["success"] is True


class TestRemoveFromCartTool:
    """Test suite for RemoveFromCartTool."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def remove_tool(self, mock_cart_manager):
        """Create RemoveFromCartTool with mock manager."""
        return RemoveFromCartTool(cart_manager=mock_cart_manager)
    
    def test_successful_remove_complete(self, remove_tool, mock_cart_manager):
        """Test successfully removing item completely."""
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Removed Test Product from cart",
            "item": {
                "product_id": "PROD123",
                "product_title": "Test Product",
                "quantity": 2
            },
            "action": "removed",
            "removed_completely": True
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(product_id="PROD123")
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["removed_completely"] is True
        assert result["action"] == "removed"
        
        mock_cart_manager.remove_item.assert_called_once_with(
            session_id="test_session",
            product_id="PROD123",
            quantity=None
        )
    
    def test_successful_remove_partial(self, remove_tool, mock_cart_manager):
        """Test successfully removing partial quantity."""
        mock_cart_manager.remove_item.return_value = {
            "success": True,
            "message": "Updated quantity to 1",
            "item": {
                "product_id": "PROD123",
                "product_title": "Test Product",
                "quantity": 1
            },
            "action": "updated",
            "removed_completely": False
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(product_id="PROD123", quantity=2)
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["removed_completely"] is False
        assert result["action"] == "updated"
        
        mock_cart_manager.remove_item.assert_called_once_with(
            session_id="test_session",
            product_id="PROD123",
            quantity=2
        )
    
    def test_remove_item_not_found(self, remove_tool, mock_cart_manager):
        """Test removing non-existent item."""
        mock_cart_manager.remove_item.return_value = {
            "success": False,
            "error": "Item not found in cart",
            "item": None,
            "removed_completely": False
        }
        
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            result = remove_tool._run(product_id="NONEXISTENT")
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert result["removed_completely"] is False
        assert "Item not found in cart" in result["error"]
    
    def test_remove_validation_errors(self, remove_tool):
        """Test validation errors for remove item."""
        
        # Test empty product ID
        result = remove_tool._run(product_id="")
        assert result["success"] is False
        assert "Product ID is required" in result["error"]
        
        # Test invalid quantity
        result = remove_tool._run(product_id="PROD123", quantity=0)
        assert result["success"] is False
        assert "Quantity must be greater than 0" in result["error"]


class TestListCartTool:
    """Test suite for ListCartTool."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def list_tool(self, mock_cart_manager):
        """Create ListCartTool with mock manager."""
        return ListCartTool(cart_manager=mock_cart_manager)
    
    def test_list_cart_with_items(self, list_tool, mock_cart_manager):
        """Test listing cart with items."""
        mock_items = [
            {
                "product_id": "PROD1",
                "product_title": "Product 1",
                "quantity": 2,
                "product_price": 10.00,
                "subtotal": 20.00,
                "product_image_url": "image1.jpg",
                "product_metadata": {"category": "electronics"},
                "added_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            },
            {
                "product_id": "PROD2",
                "product_title": "Product 2",
                "quantity": 1,
                "product_price": 25.00,
                "subtotal": 25.00,
                "product_image_url": None,
                "product_metadata": {},
                "added_at": "2024-01-02T12:00:00",
                "updated_at": "2024-01-02T12:00:00"
            }
        ]
        
        mock_summary = {
            "total_items": 3,
            "total_value": 45.00,
            "unique_products": 2,
            "is_empty": False
        }
        
        mock_cart_manager.get_cart_contents.return_value = mock_items
        mock_cart_manager.get_cart_summary.return_value = mock_summary
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            result = list_tool._run(include_summary=True, format_type="detailed")
        
        assert result["success"] is True
        assert result["item_count"] == 2
        assert result["is_empty"] is False
        assert result["total_items"] == 3
        assert result["total_value"] == 45.00
        assert len(result["cart_items"]) == 2
        
        # Check detailed formatting
        first_item = result["cart_items"][0]
        assert first_item["product_id"] == "PROD1"
        assert first_item["quantity"] == 2
        assert first_item["subtotal"] == 20.00
        assert "metadata" in first_item
    
    def test_list_empty_cart(self, list_tool, mock_cart_manager):
        """Test listing empty cart."""
        mock_cart_manager.get_cart_contents.return_value = []
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 0,
            "total_value": 0.0,
            "unique_products": 0,
            "is_empty": True
        }
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            result = list_tool._run()
        
        assert result["success"] is True
        assert result["item_count"] == 0
        assert result["is_empty"] is True
        assert result["message"] == "Your cart is empty"
    
    def test_list_cart_format_types(self, list_tool, mock_cart_manager):
        """Test different format types."""
        mock_items = [{
            "product_id": "PROD1",
            "product_title": "Product 1",
            "quantity": 2,
            "product_price": 10.00,
            "subtotal": 20.00,
            "product_image_url": "image1.jpg",
            "product_metadata": {"category": "electronics"},
            "added_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00"
        }]
        
        mock_cart_manager.get_cart_contents.return_value = mock_items
        mock_cart_manager.get_cart_summary.return_value = {"total_items": 2}
        
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            # Test minimal format
            result = list_tool._run(format_type="minimal")
            minimal_item = result["cart_items"][0]
            assert set(minimal_item.keys()) == {"product_title", "quantity"}
            
            # Test summary format
            result = list_tool._run(format_type="summary")
            summary_item = result["cart_items"][0]
            assert set(summary_item.keys()) == {"product_id", "product_title", "quantity", "subtotal"}
            
            # Test detailed format
            result = list_tool._run(format_type="detailed")
            detailed_item = result["cart_items"][0]
            assert "metadata" in detailed_item
            assert "added_at" in detailed_item


class TestClearCartTool:
    """Test suite for ClearCartTool."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def clear_tool(self, mock_cart_manager):
        """Create ClearCartTool with mock manager."""
        return ClearCartTool(cart_manager=mock_cart_manager)
    
    def test_successful_clear_cart(self, clear_tool, mock_cart_manager):
        """Test successfully clearing cart."""
        mock_cart_manager.clear_cart.return_value = {
            "success": True,
            "message": "Cleared 3 items from cart",
            "items_removed": 3,
            "cleared_items": [
                {"product_id": "PROD1", "quantity": 2},
                {"product_id": "PROD2", "quantity": 1}
            ]
        }
        
        with patch.object(clear_tool, '_get_session_id', return_value="test_session"):
            result = clear_tool._run()
        
        assert result["success"] is True
        assert result["cart_updated"] is True
        assert result["items_removed"] == 3
        assert len(result["cleared_items"]) == 2
        assert "Cleared 3 items from cart" in result["message"]
        
        mock_cart_manager.clear_cart.assert_called_once_with("test_session")
    
    def test_clear_empty_cart(self, clear_tool, mock_cart_manager):
        """Test clearing empty cart."""
        mock_cart_manager.clear_cart.return_value = {
            "success": True,
            "message": "Cleared 0 items from cart",
            "items_removed": 0,
            "cleared_items": []
        }
        
        with patch.object(clear_tool, '_get_session_id', return_value="test_session"):
            result = clear_tool._run()
        
        assert result["success"] is True
        assert result["items_removed"] == 0
        assert result["cleared_items"] == []
    
    def test_clear_cart_error(self, clear_tool, mock_cart_manager):
        """Test clear cart with error."""
        mock_cart_manager.clear_cart.return_value = {
            "success": False,
            "error": "Database error",
            "items_removed": 0
        }
        
        with patch.object(clear_tool, '_get_session_id', return_value="test_session"):
            result = clear_tool._run()
        
        assert result["success"] is False
        assert result["cart_updated"] is False
        assert "Database error" in result["error"]


class TestCartToolUtilities:
    """Test utility functions for cart tools."""
    
    def test_create_cart_tools(self):
        """Test creating all cart tools."""
        with patch('src.langgraph_integration.tools.shopping_cart_tools.get_global_cart_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            tools = create_cart_tools()
            
            assert len(tools) == 4
            assert any(tool.name == "add_to_cart" for tool in tools)
            assert any(tool.name == "remove_from_cart" for tool in tools)
            assert any(tool.name == "list_cart" for tool in tools)
            assert any(tool.name == "clear_cart" for tool in tools)
    
    def test_create_cart_tools_with_manager(self):
        """Test creating cart tools with specific manager."""
        mock_manager = Mock(spec=ShoppingCartManager)
        
        tools = create_cart_tools(mock_manager)
        
        assert len(tools) == 4
        for tool in tools:
            assert tool.cart_manager == mock_manager
    
    def test_get_cart_tool_by_name(self):
        """Test getting specific cart tool by name."""
        with patch('src.langgraph_integration.tools.shopping_cart_tools.get_global_cart_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            # Test valid tool names
            add_tool = get_cart_tool_by_name("add_to_cart")
            assert add_tool is not None
            assert add_tool.name == "add_to_cart"
            
            remove_tool = get_cart_tool_by_name("remove_from_cart")
            assert remove_tool is not None
            assert remove_tool.name == "remove_from_cart"
            
            list_tool = get_cart_tool_by_name("list_cart")
            assert list_tool is not None
            assert list_tool.name == "list_cart"
            
            clear_tool = get_cart_tool_by_name("clear_cart")
            assert clear_tool is not None
            assert clear_tool.name == "clear_cart"
            
            # Test invalid tool name
            invalid_tool = get_cart_tool_by_name("invalid_tool")
            assert invalid_tool is None
    
    def test_get_cart_tools_info(self):
        """Test getting cart tools information."""
        info = get_cart_tools_info()
        
        assert info["tool_count"] == 4
        assert info["supports_async"] is True
        assert info["requires_session"] is True
        assert len(info["available_tools"]) == 4
        
        # Check tool information structure
        for tool_info in info["available_tools"]:
            assert "name" in tool_info
            assert "description" in tool_info
            assert "required_params" in tool_info
            assert "optional_params" in tool_info


class TestCartToolsIntegration:
    """Integration tests for cart tools with real cart manager."""
    
    @pytest.fixture
    def integration_cart_manager(self):
        """Create cart manager for integration tests."""
        mock_db = Mock()
        return ShoppingCartManager(mock_db)
    
    def test_add_and_list_integration(self, integration_cart_manager):
        """Test integration between add and list tools."""
        # Mock database responses
        integration_cart_manager.db_manager.execute_query.return_value = []  # No existing item
        integration_cart_manager.db_manager.execute_update.return_value = 1
        
        # Mock connection for insert
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": "new-item-id"}
        
        cursor_context = Mock()
        cursor_context.__enter__ = Mock(return_value=mock_cursor)
        cursor_context.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = cursor_context
        
        connection_context = Mock()
        connection_context.__enter__ = Mock(return_value=mock_connection)
        connection_context.__exit__ = Mock(return_value=None)
        integration_cart_manager.db_manager.get_connection.return_value = connection_context
        
        # Mock getting inserted item
        inserted_item = {
            "id": "new-item-id",
            "product_id": "PROD123",
            "product_title": "Test Product",
            "product_price": Decimal("29.99"),
            "product_image_url": None,
            "quantity": 1,
            "product_metadata": {},
            "added_at": datetime.now(),
            "updated_at": datetime.now()
        }
        integration_cart_manager.db_manager.execute_query.side_effect = [[], [inserted_item], [inserted_item]]
        
        # Create tools
        add_tool = AddToCartTool(cart_manager=integration_cart_manager)
        list_tool = ListCartTool(cart_manager=integration_cart_manager)
        
        # Add item
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            add_result = add_tool._run(
                product_id="PROD123",
                product_title="Test Product",
                quantity=1,
                price=29.99
            )
        
        assert add_result["success"] is True
        
        # List cart
        with patch.object(list_tool, '_get_session_id', return_value="test_session"):
            list_result = list_tool._run()
        
        assert list_result["success"] is True
        assert list_result["item_count"] == 1
    
    def test_add_remove_integration(self, integration_cart_manager):
        """Test integration between add and remove tools."""
        # Mock existing item for removal
        existing_item = {
            "id": "item-id",
            "product_id": "PROD123",
            "product_title": "Test Product",
            "product_price": Decimal("29.99"),
            "product_image_url": None,
            "quantity": 2,
            "product_metadata": {},
            "added_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        integration_cart_manager.db_manager.execute_query.return_value = [existing_item]
        integration_cart_manager.db_manager.execute_update.return_value = 1
        
        # Create tools
        remove_tool = RemoveFromCartTool(cart_manager=integration_cart_manager)
        
        # Remove item
        with patch.object(remove_tool, '_get_session_id', return_value="test_session"):
            remove_result = remove_tool._run(product_id="PROD123")
        
        assert remove_result["success"] is True
        assert remove_result["removed_completely"] is True