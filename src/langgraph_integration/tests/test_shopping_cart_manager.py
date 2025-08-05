"""
Unit tests for ShoppingCartManager class.
Tests CRUD operations, session isolation, and error handling.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from decimal import Decimal

from ..state.shopping_cart_manager import ShoppingCartManager, create_shopping_cart_manager, get_global_cart_manager
from ..state.database import DatabaseManager


class TestShoppingCartManager:
    """Test suite for ShoppingCartManager."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        mock_db = Mock(spec=DatabaseManager)
        return mock_db
    
    @pytest.fixture
    def cart_manager(self, mock_db_manager):
        """Create cart manager with mock database."""
        return ShoppingCartManager(mock_db_manager)
    
    @pytest.fixture
    def sample_cart_item(self):
        """Sample cart item data."""
        return {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "product_id": "PROD123",
            "product_title": "Test Product",
            "product_price": Decimal("29.99"),
            "product_image_url": "https://example.com/image.jpg",
            "quantity": 2,
            "product_metadata": {"category": "electronics"},
            "added_at": datetime(2024, 1, 1, 12, 0, 0),
            "updated_at": datetime(2024, 1, 1, 12, 0, 0)
        }
    
    def test_init_with_default_db_manager(self):
        """Test initialization with default database manager."""
        with patch('src.langgraph_integration.state.shopping_cart_manager.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            manager = ShoppingCartManager()
            
            assert manager.db_manager == mock_db
            mock_get_db.assert_called_once()
    
    def test_init_with_custom_db_manager(self, mock_db_manager):
        """Test initialization with custom database manager."""
        manager = ShoppingCartManager(mock_db_manager)
        
        assert manager.db_manager == mock_db_manager
    
    def test_add_item_new_product(self, cart_manager, mock_db_manager):
        """Test adding new product to cart."""
        # Mock no existing item
        mock_db_manager.execute_query.return_value = []
        
        # Mock successful insertion with proper context manager
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": "new-item-id"}
        
        # Create proper context manager mocks
        cursor_context = Mock()
        cursor_context.__enter__ = Mock(return_value=mock_cursor)
        cursor_context.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = cursor_context
        
        connection_context = Mock()
        connection_context.__enter__ = Mock(return_value=mock_connection)
        connection_context.__exit__ = Mock(return_value=None)
        mock_db_manager.get_connection.return_value = connection_context
        
        # Mock getting the inserted item
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
        mock_db_manager.execute_query.side_effect = [[], [inserted_item]]
        
        result = cart_manager.add_item(
            session_id="session123",
            product_id="PROD123",
            product_title="Test Product",
            quantity=1,
            price=29.99
        )
        
        assert result["success"] is True
        assert result["action"] == "added"
        assert result["item"]["product_id"] == "PROD123"
        assert result["item"]["quantity"] == 1
        assert "Added 1 x Test Product to cart" in result["message"]
    
    def test_add_item_existing_product(self, cart_manager, mock_db_manager, sample_cart_item):
        """Test adding to existing product in cart (quantity update)."""
        # Mock existing item
        mock_db_manager.execute_query.return_value = [sample_cart_item]
        
        # Mock successful quantity update
        mock_db_manager.execute_update.return_value = 1
        
        # Mock getting updated item
        updated_item = sample_cart_item.copy()
        updated_item["quantity"] = 4  # 2 + 2
        mock_db_manager.execute_query.side_effect = [[sample_cart_item], [updated_item]]
        
        result = cart_manager.add_item(
            session_id="session123",
            product_id="PROD123",
            product_title="Test Product",
            quantity=2
        )
        
        assert result["success"] is True
        assert result["action"] == "updated"
        assert result["item"]["quantity"] == 4
    
    def test_add_item_invalid_quantity(self, cart_manager):
        """Test adding item with invalid quantity."""
        result = cart_manager.add_item(
            session_id="session123",
            product_id="PROD123",
            product_title="Test Product",
            quantity=0
        )
        
        assert result["success"] is False
        assert "Quantity must be greater than 0" in result["error"]
        assert result["item"] is None
    
    def test_add_item_database_error(self, cart_manager, mock_db_manager):
        """Test add item with database error."""
        mock_db_manager.execute_query.side_effect = Exception("Database connection failed")
        
        result = cart_manager.add_item(
            session_id="session123",
            product_id="PROD123",
            product_title="Test Product",
            quantity=1
        )
        
        assert result["success"] is False
        assert "Database error" in result["error"]
        assert result["item"] is None
    
    def test_remove_item_complete_removal(self, cart_manager, mock_db_manager, sample_cart_item):
        """Test complete removal of item from cart."""
        # Mock existing item
        mock_db_manager.execute_query.return_value = [sample_cart_item]
        
        # Mock successful deletion
        mock_db_manager.execute_update.return_value = 1
        
        result = cart_manager.remove_item(
            session_id="session123",
            product_id="PROD123"
        )
        
        assert result["success"] is True
        assert result["action"] == "removed"
        assert result["removed_completely"] is True
        assert "Removed Test Product from cart" in result["message"]
    
    def test_remove_item_partial_removal(self, cart_manager, mock_db_manager, sample_cart_item):
        """Test partial removal of item from cart."""
        # Mock existing item with quantity 3
        existing_item = sample_cart_item.copy()
        existing_item["quantity"] = 3
        mock_db_manager.execute_query.return_value = [existing_item]
        
        # Mock successful quantity update
        mock_db_manager.execute_update.return_value = 1
        
        # Mock getting updated item
        updated_item = existing_item.copy()
        updated_item["quantity"] = 1  # 3 - 2
        mock_db_manager.execute_query.side_effect = [[existing_item], [updated_item]]
        
        result = cart_manager.remove_item(
            session_id="session123",
            product_id="PROD123",
            quantity=2
        )
        
        assert result["success"] is True
        assert result["action"] == "updated"
        assert result["item"]["quantity"] == 1
        assert "removed_completely" not in result or result["removed_completely"] is False
    
    def test_remove_item_not_found(self, cart_manager, mock_db_manager):
        """Test removing non-existent item."""
        # Mock no existing item
        mock_db_manager.execute_query.return_value = []
        
        result = cart_manager.remove_item(
            session_id="session123",
            product_id="NONEXISTENT"
        )
        
        assert result["success"] is False
        assert "Item not found in cart" in result["error"]
        assert result["item"] is None
        assert result["removed_completely"] is False
    
    def test_remove_item_quantity_exceeds_current(self, cart_manager, mock_db_manager, sample_cart_item):
        """Test removing more quantity than available (should remove completely)."""
        # Mock existing item with quantity 2
        mock_db_manager.execute_query.return_value = [sample_cart_item]
        
        # Mock successful deletion
        mock_db_manager.execute_update.return_value = 1
        
        result = cart_manager.remove_item(
            session_id="session123",
            product_id="PROD123",
            quantity=5  # More than available
        )
        
        assert result["success"] is True
        assert result["action"] == "removed"
        assert result["removed_completely"] is True
    
    def test_get_cart_contents_with_items(self, cart_manager, mock_db_manager):
        """Test getting cart contents with multiple items."""
        mock_items = [
            {
                "id": "item1",
                "product_id": "PROD1",
                "product_title": "Product 1",
                "product_price": Decimal("10.00"),
                "product_image_url": "image1.jpg",
                "quantity": 2,
                "product_metadata": {"category": "books"},
                "added_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 1)
            },
            {
                "id": "item2",
                "product_id": "PROD2",
                "product_title": "Product 2",
                "product_price": Decimal("25.50"),
                "product_image_url": None,
                "quantity": 1,
                "product_metadata": {},
                "added_at": datetime(2024, 1, 2),
                "updated_at": datetime(2024, 1, 2)
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_items
        
        result = cart_manager.get_cart_contents("session123")
        
        assert len(result) == 2
        assert result[0]["product_id"] == "PROD1"
        assert result[0]["quantity"] == 2
        assert result[0]["subtotal"] == 20.0  # 10.00 * 2
        assert result[1]["product_id"] == "PROD2"
        assert result[1]["quantity"] == 1
        assert result[1]["subtotal"] == 25.5  # 25.50 * 1
    
    def test_get_cart_contents_empty_cart(self, cart_manager, mock_db_manager):
        """Test getting contents of empty cart."""
        mock_db_manager.execute_query.return_value = []
        
        result = cart_manager.get_cart_contents("session123")
        
        assert result == []
    
    def test_get_cart_contents_database_error(self, cart_manager, mock_db_manager):
        """Test get cart contents with database error."""
        mock_db_manager.execute_query.side_effect = Exception("Database error")
        
        result = cart_manager.get_cart_contents("session123")
        
        assert result == []
    
    def test_get_cart_summary_with_data(self, cart_manager, mock_db_manager):
        """Test getting cart summary with data."""
        mock_summary = {
            "session_id": "session123",
            "total_items": 5,
            "total_value": Decimal("75.50"),
            "item_count": 3,
            "last_updated": datetime(2024, 1, 1, 12, 0, 0)
        }
        
        mock_db_manager.execute_query.return_value = [mock_summary]
        
        result = cart_manager.get_cart_summary("session123")
        
        assert result["session_id"] == "session123"
        assert result["total_items"] == 5
        assert result["total_value"] == 75.50
        assert result["unique_products"] == 3
        assert result["is_empty"] is False
        assert result["last_updated"] == "2024-01-01T12:00:00"
    
    def test_get_cart_summary_empty_cart(self, cart_manager, mock_db_manager):
        """Test getting summary of empty cart."""
        mock_db_manager.execute_query.return_value = []
        
        result = cart_manager.get_cart_summary("session123")
        
        assert result["session_id"] == "session123"
        assert result["total_items"] == 0
        assert result["total_value"] == 0.0
        assert result["unique_products"] == 0
        assert result["is_empty"] is True
        assert result["last_updated"] is None
    
    def test_get_cart_summary_database_error(self, cart_manager, mock_db_manager):
        """Test get cart summary with database error."""
        mock_db_manager.execute_query.side_effect = Exception("Database error")
        
        result = cart_manager.get_cart_summary("session123")
        
        assert result["session_id"] == "session123"
        assert result["total_items"] == 0
        assert result["total_value"] == 0.0
        assert result["is_empty"] is True
        assert "error" in result
    
    def test_clear_cart_with_items(self, cart_manager, mock_db_manager):
        """Test clearing cart with existing items."""
        # Mock current cart contents
        current_items = [
            {"product_id": "PROD1", "quantity": 2},
            {"product_id": "PROD2", "quantity": 1}
        ]
        
        # Mock get_cart_contents call
        with patch.object(cart_manager, 'get_cart_contents', return_value=current_items):
            # Mock successful deletion
            mock_db_manager.execute_update.return_value = 2
            
            result = cart_manager.clear_cart("session123")
            
            assert result["success"] is True
            assert result["items_removed"] == 2
            assert "Cleared 2 items from cart" in result["message"]
            assert result["cleared_items"] == current_items
    
    def test_clear_cart_empty_cart(self, cart_manager, mock_db_manager):
        """Test clearing empty cart."""
        # Mock empty cart
        with patch.object(cart_manager, 'get_cart_contents', return_value=[]):
            mock_db_manager.execute_update.return_value = 0
            
            result = cart_manager.clear_cart("session123")
            
            assert result["success"] is True
            assert result["items_removed"] == 0
            assert "Cleared 0 items from cart" in result["message"]
    
    def test_clear_cart_database_error(self, cart_manager, mock_db_manager):
        """Test clear cart with database error."""
        with patch.object(cart_manager, 'get_cart_contents', return_value=[]):
            mock_db_manager.execute_update.side_effect = Exception("Database error")
            
            result = cart_manager.clear_cart("session123")
            
            assert result["success"] is False
            assert "Database error" in result["error"]
            assert result["items_removed"] == 0
    
    def test_update_item_metadata_success(self, cart_manager, mock_db_manager, sample_cart_item):
        """Test successful metadata update."""
        new_metadata = {"category": "updated", "tags": ["new", "improved"]}
        
        # Mock existing item
        mock_db_manager.execute_query.return_value = [sample_cart_item]
        
        # Mock successful update
        mock_db_manager.execute_update.return_value = 1
        
        # Mock getting updated item
        updated_item = sample_cart_item.copy()
        updated_item["product_metadata"] = new_metadata
        mock_db_manager.execute_query.side_effect = [[sample_cart_item], [updated_item]]
        
        result = cart_manager.update_item_metadata(
            session_id="session123",
            product_id="PROD123",
            metadata=new_metadata
        )
        
        assert result["success"] is True
        assert result["item"]["product_metadata"] == new_metadata
        assert "Item metadata updated successfully" in result["message"]
    
    def test_update_item_metadata_not_found(self, cart_manager, mock_db_manager):
        """Test updating metadata for non-existent item."""
        mock_db_manager.execute_query.return_value = []
        
        result = cart_manager.update_item_metadata(
            session_id="session123",
            product_id="NONEXISTENT",
            metadata={"test": "data"}
        )
        
        assert result["success"] is False
        assert "Item not found in cart" in result["error"]
        assert result["item"] is None
    
    def test_get_cart_item_found(self, cart_manager, mock_db_manager, sample_cart_item):
        """Test getting specific cart item that exists."""
        mock_db_manager.execute_query.return_value = [sample_cart_item]
        
        result = cart_manager.get_cart_item("session123", "PROD123")
        
        assert result is not None
        assert result["product_id"] == "PROD123"
        assert result["product_title"] == "Test Product"
        assert result["quantity"] == 2
    
    def test_get_cart_item_not_found(self, cart_manager, mock_db_manager):
        """Test getting specific cart item that doesn't exist."""
        mock_db_manager.execute_query.return_value = []
        
        result = cart_manager.get_cart_item("session123", "NONEXISTENT")
        
        assert result is None
    
    def test_get_cart_item_database_error(self, cart_manager, mock_db_manager):
        """Test get cart item with database error."""
        mock_db_manager.execute_query.side_effect = Exception("Database error")
        
        result = cart_manager.get_cart_item("session123", "PROD123")
        
        assert result is None
    
    def test_format_cart_item_with_price(self, cart_manager, sample_cart_item):
        """Test formatting cart item with price."""
        result = cart_manager._format_cart_item(sample_cart_item)
        
        assert result["id"] == str(sample_cart_item["id"])
        assert result["product_id"] == "PROD123"
        assert result["product_title"] == "Test Product"
        assert result["product_price"] == 29.99
        assert result["quantity"] == 2
        assert result["subtotal"] == 59.98  # 29.99 * 2
        assert result["added_at"] == "2024-01-01T12:00:00"
        assert result["updated_at"] == "2024-01-01T12:00:00"
    
    def test_format_cart_item_without_price(self, cart_manager, sample_cart_item):
        """Test formatting cart item without price."""
        sample_cart_item["product_price"] = None
        
        result = cart_manager._format_cart_item(sample_cart_item)
        
        assert result["product_price"] is None
        assert result["subtotal"] is None
    
    def test_format_cart_item_none(self, cart_manager):
        """Test formatting None cart item."""
        result = cart_manager._format_cart_item(None)
        
        assert result is None


class TestShoppingCartManagerUtilities:
    """Test utility functions for shopping cart manager."""
    
    def test_create_shopping_cart_manager_with_db(self):
        """Test creating cart manager with specific database manager."""
        mock_db = Mock(spec=DatabaseManager)
        
        manager = create_shopping_cart_manager(mock_db)
        
        assert isinstance(manager, ShoppingCartManager)
        assert manager.db_manager == mock_db
    
    def test_create_shopping_cart_manager_without_db(self):
        """Test creating cart manager without specific database manager."""
        with patch('src.langgraph_integration.state.shopping_cart_manager.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            manager = create_shopping_cart_manager()
            
            assert isinstance(manager, ShoppingCartManager)
            assert manager.db_manager == mock_db
    
    def test_get_global_cart_manager(self):
        """Test getting global cart manager instance."""
        with patch('src.langgraph_integration.state.shopping_cart_manager.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            manager = get_global_cart_manager()
            
            assert isinstance(manager, ShoppingCartManager)
            assert manager.db_manager == mock_db


class TestShoppingCartManagerIntegration:
    """Integration tests for shopping cart manager with real database operations."""
    
    @pytest.fixture
    def integration_db_manager(self):
        """Create database manager for integration tests."""
        # This would use a test database in real integration tests
        mock_db = Mock(spec=DatabaseManager)
        return mock_db
    
    def test_session_isolation(self, integration_db_manager):
        """Test that cart operations are properly isolated by session."""
        cart_manager = ShoppingCartManager(integration_db_manager)
        
        # Mock different sessions having different cart contents with all required fields
        session1_items = [{
            "id": "item1",
            "product_id": "PROD1", 
            "product_title": "Product 1",
            "product_price": Decimal("10.00"),
            "product_image_url": None,
            "quantity": 1,
            "product_metadata": {},
            "added_at": datetime.now(),
            "updated_at": datetime.now()
        }]
        session2_items = [{
            "id": "item2",
            "product_id": "PROD2", 
            "product_title": "Product 2",
            "product_price": Decimal("20.00"),
            "product_image_url": None,
            "quantity": 2,
            "product_metadata": {},
            "added_at": datetime.now(),
            "updated_at": datetime.now()
        }]
        
        def mock_query_side_effect(query, params):
            session_id = params[0] if params else None
            if session_id == "session1":
                return session1_items
            elif session_id == "session2":
                return session2_items
            return []
        
        integration_db_manager.execute_query.side_effect = mock_query_side_effect
        
        # Test session isolation
        session1_cart = cart_manager.get_cart_contents("session1")
        session2_cart = cart_manager.get_cart_contents("session2")
        
        assert len(session1_cart) == 1
        assert len(session2_cart) == 1
        assert session1_cart[0]["product_id"] == "PROD1"
        assert session2_cart[0]["product_id"] == "PROD2"
    
    def test_concurrent_operations_safety(self, integration_db_manager):
        """Test that concurrent cart operations are handled safely."""
        cart_manager = ShoppingCartManager(integration_db_manager)
        
        # Mock successful operations
        integration_db_manager.execute_query.return_value = []
        integration_db_manager.execute_update.return_value = 1
        
        # Mock connection context manager properly
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": "new-item-id"}
        
        # Create proper context manager mocks
        cursor_context = Mock()
        cursor_context.__enter__ = Mock(return_value=mock_cursor)
        cursor_context.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = cursor_context
        
        connection_context = Mock()
        connection_context.__enter__ = Mock(return_value=mock_connection)
        connection_context.__exit__ = Mock(return_value=None)
        integration_db_manager.get_connection.return_value = connection_context
        
        # Mock getting inserted items
        inserted_item1 = {
            "id": "item1",
            "product_id": "PROD1",
            "product_title": "Product 1",
            "product_price": Decimal("10.00"),
            "product_image_url": None,
            "quantity": 1,
            "product_metadata": {},
            "added_at": datetime.now(),
            "updated_at": datetime.now()
        }
        inserted_item2 = {
            "id": "item2",
            "product_id": "PROD2",
            "product_title": "Product 2",
            "product_price": Decimal("20.00"),
            "product_image_url": None,
            "quantity": 1,
            "product_metadata": {},
            "added_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Set up side effects for multiple calls
        integration_db_manager.execute_query.side_effect = [[], [inserted_item1], [], [inserted_item2]]
        
        # Simulate concurrent add operations
        result1 = cart_manager.add_item("session1", "PROD1", "Product 1", 1)
        result2 = cart_manager.add_item("session1", "PROD2", "Product 2", 1)
        
        # Both operations should succeed
        assert result1["success"] is True
        assert result2["success"] is True
        
        # Verify database operations were called
        assert integration_db_manager.execute_query.call_count >= 2