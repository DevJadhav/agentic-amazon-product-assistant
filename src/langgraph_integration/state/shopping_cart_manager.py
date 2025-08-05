"""
Shopping Cart Database Manager for LangGraph agent workflows.
Handles CRUD operations for shopping cart functionality with session isolation.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from .database import DatabaseManager, get_database_manager
from .connection_pool import get_optimized_pool
from .cart_error_handler import CartErrorHandler, CartErrorType
from ..monitoring.performance_monitor import get_performance_monitor, performance_track
from ..monitoring.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class ShoppingCartManager:
    """Manages shopping cart database operations with session isolation."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, 
                 error_handler: Optional[CartErrorHandler] = None):
        """Initialize shopping cart manager."""
        self.db_manager = db_manager or get_database_manager()
        self.optimized_pool = get_optimized_pool()
        self.error_handler = error_handler or CartErrorHandler()
        self.perf_monitor = get_performance_monitor()
        self.metrics_collector = get_metrics_collector()
        self.logger = logging.getLogger(__name__)
    
    @performance_track("cart_add_item")
    def add_item(
        self, 
        session_id: str, 
        product_id: str, 
        product_title: str, 
        quantity: int = 1,
        price: Optional[float] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add or update item in cart with duplicate handling and quantity updates.
        
        Args:
            session_id: User session identifier
            product_id: Unique product identifier
            product_title: Product display name
            quantity: Number of items to add (default: 1)
            price: Product price per unit
            image_url: Product image URL
            metadata: Additional product metadata
            
        Returns:
            Dict containing operation result and cart item details
        """
        try:
            # Record metrics
            self.metrics_collector.increment_counter("cart_add_item_requests")
            
            # Validate quantity
            if quantity <= 0:
                return self.error_handler.handle_quantity_validation_error(
                    ValueError(f"Invalid quantity: {quantity}"), quantity, session_id
                )
            
            # Validate product data
            if not product_id or not product_title:
                return self.error_handler.handle_product_validation_error(
                    ValueError("Missing required product information"),
                    {"product_id": product_id, "product_title": product_title},
                    session_id
                )
            
            # Check if item already exists in cart
            existing_item = self._get_cart_item(session_id, product_id)
            
            if existing_item:
                # Update existing item quantity
                new_quantity = existing_item["quantity"] + quantity
                return self._update_item_quantity(session_id, product_id, new_quantity)
            else:
                # Add new item to cart
                return self._insert_new_item(
                    session_id, product_id, product_title, quantity, 
                    price, image_url, metadata
                )
                
        except Exception as e:
            self.logger.error(f"Failed to add item to cart: {e}")
            return self.error_handler.handle_cart_operation_error(
                e, "add_item", session_id, {
                    "product_id": product_id,
                    "product_title": product_title,
                    "quantity": quantity
                }
            )
    
    @performance_track("cart_remove_item")
    def remove_item(
        self, 
        session_id: str, 
        product_id: str,
        quantity: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Remove item from cart with partial and complete removal support.
        
        Args:
            session_id: User session identifier
            product_id: Product to remove
            quantity: Number of items to remove (None = remove all)
            
        Returns:
            Dict containing operation result and updated item details
        """
        try:
            # Validate quantity if provided
            if quantity is not None and quantity <= 0:
                return self.error_handler.handle_quantity_validation_error(
                    ValueError(f"Invalid removal quantity: {quantity}"), quantity, session_id
                )
            
            # Get existing item
            existing_item = self._get_cart_item(session_id, product_id)
            
            if not existing_item:
                return self.error_handler.handle_cart_operation_error(
                    ValueError("Item not found in cart"), "remove_item", session_id,
                    {"product_id": product_id, "quantity": quantity}
                )
            
            current_quantity = existing_item["quantity"]
            
            # Determine removal behavior
            if quantity is None or quantity >= current_quantity:
                # Remove item completely
                return self._delete_cart_item(session_id, product_id, existing_item)
            else:
                # Partial removal - update quantity
                new_quantity = current_quantity - quantity
                return self._update_item_quantity(session_id, product_id, new_quantity)
                
        except Exception as e:
            self.logger.error(f"Failed to remove item from cart: {e}")
            return self.error_handler.handle_cart_operation_error(
                e, "remove_item", session_id, {
                    "product_id": product_id,
                    "quantity": quantity
                }
            )
    
    @performance_track("cart_get_contents")
    def get_cart_contents(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all items in cart for session with proper isolation.
        
        Args:
            session_id: User session identifier
            
        Returns:
            List of cart items with details
        """
        try:
            query = """
            SELECT 
                id,
                product_id,
                product_title,
                product_price,
                product_image_url,
                quantity,
                product_metadata,
                added_at,
                updated_at
            FROM shopping_cart
            WHERE session_id = %s
            ORDER BY added_at ASC
            """
            
            results = self.db_manager.execute_query(query, (session_id,))
            
            # Convert to proper format
            cart_items = []
            for item in results:
                cart_item = {
                    "id": str(item["id"]),
                    "product_id": item["product_id"],
                    "product_title": item["product_title"],
                    "product_price": float(item["product_price"]) if item["product_price"] else None,
                    "product_image_url": item["product_image_url"],
                    "quantity": item["quantity"],
                    "product_metadata": item["product_metadata"] or {},
                    "added_at": item["added_at"].isoformat() if item["added_at"] else None,
                    "updated_at": item["updated_at"].isoformat() if item["updated_at"] else None,
                    "subtotal": (
                        float(item["product_price"]) * item["quantity"] 
                        if item["product_price"] else None
                    )
                }
                cart_items.append(cart_item)
            
            return cart_items
            
        except Exception as e:
            self.logger.error(f"Failed to get cart contents: {e}")
            error_response = self.error_handler.handle_database_error(
                e, "get_cart_contents", session_id
            )
            # Return empty list for cart contents, but log the error
            return []
    
    def get_cart_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get cart summary with totals and item counts.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Dict containing cart summary information
        """
        try:
            # Use the database function for optimized summary
            query = "SELECT * FROM get_cart_summary(%s)"
            
            results = self.db_manager.execute_query(query, (session_id,))
            
            if results:
                summary_data = results[0]
                return {
                    "session_id": session_id,
                    "total_items": summary_data["total_items"] or 0,
                    "total_value": float(summary_data["total_value"]) if summary_data["total_value"] else 0.0,
                    "unique_products": summary_data["item_count"] or 0,
                    "last_updated": (
                        summary_data["last_updated"].isoformat() 
                        if summary_data["last_updated"] else None
                    ),
                    "is_empty": (summary_data["total_items"] or 0) == 0
                }
            else:
                # No cart session exists yet
                return {
                    "session_id": session_id,
                    "total_items": 0,
                    "total_value": 0.0,
                    "unique_products": 0,
                    "last_updated": None,
                    "is_empty": True
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get cart summary: {e}")
            error_response = self.error_handler.handle_database_error(
                e, "get_cart_summary", session_id
            )
            return {
                "session_id": session_id,
                "total_items": 0,
                "total_value": 0.0,
                "unique_products": 0,
                "last_updated": None,
                "is_empty": True,
                "error": str(e),
                "error_details": error_response
            }
    
    @performance_track("cart_clear")
    def clear_cart(self, session_id: str) -> Dict[str, Any]:
        """
        Clear all items from cart for session.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Dict containing operation result
        """
        try:
            # Get current cart contents for response
            current_items = self.get_cart_contents(session_id)
            
            # Delete all items for session
            query = "DELETE FROM shopping_cart WHERE session_id = %s"
            deleted_count = self.db_manager.execute_update(query, (session_id,))
            
            return {
                "success": True,
                "message": f"Cleared {deleted_count} items from cart",
                "items_removed": len(current_items),
                "cleared_items": current_items
            }
            
        except Exception as e:
            self.logger.error(f"Failed to clear cart: {e}")
            return self.error_handler.handle_cart_operation_error(
                e, "clear_cart", session_id
            )
    
    def update_item_metadata(
        self, 
        session_id: str, 
        product_id: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update metadata for a cart item.
        
        Args:
            session_id: User session identifier
            product_id: Product to update
            metadata: New metadata to set
            
        Returns:
            Dict containing operation result
        """
        try:
            # Check if item exists
            existing_item = self._get_cart_item(session_id, product_id)
            
            if not existing_item:
                return {
                    "success": False,
                    "error": "Item not found in cart",
                    "item": None
                }
            
            # Update metadata
            query = """
            UPDATE shopping_cart 
            SET product_metadata = %s, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s AND product_id = %s
            """
            
            updated_count = self.db_manager.execute_update(
                query, 
                (json.dumps(metadata), session_id, product_id)
            )
            
            if updated_count > 0:
                # Get updated item
                updated_item = self._get_cart_item(session_id, product_id)
                return {
                    "success": True,
                    "message": "Item metadata updated successfully",
                    "item": self._format_cart_item(updated_item)
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to update item metadata",
                    "item": None
                }
                
        except Exception as e:
            self.logger.error(f"Failed to update item metadata: {e}")
            return {
                "success": False,
                "error": f"Database error: {str(e)}",
                "item": None
            }
    
    def get_cart_item(self, session_id: str, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific cart item details.
        
        Args:
            session_id: User session identifier
            product_id: Product to retrieve
            
        Returns:
            Cart item details or None if not found
        """
        try:
            item = self._get_cart_item(session_id, product_id)
            return self._format_cart_item(item) if item else None
            
        except Exception as e:
            self.logger.error(f"Failed to get cart item: {e}")
            self.error_handler.handle_database_error(
                e, "get_cart_item", session_id, {"product_id": product_id}
            )
            return None
    
    def is_cart_available(self) -> bool:
        """Check if cart functionality is currently available."""
        return self.error_handler.is_cart_functionality_available()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get cart error handling statistics."""
        return self.error_handler.get_error_statistics()
    
    def create_graceful_degradation_response(self, operation: str, session_id: str) -> Dict[str, Any]:
        """Create graceful degradation response when cart is unavailable."""
        return self.error_handler.create_graceful_degradation_response(
            Exception("Cart service unavailable"), operation, session_id
        )
    
    # Private helper methods
    
    def _get_cart_item(self, session_id: str, product_id: str) -> Optional[Dict[str, Any]]:
        """Get raw cart item from database."""
        query = """
        SELECT 
            id, product_id, product_title, product_price, product_image_url,
            quantity, product_metadata, added_at, updated_at
        FROM shopping_cart
        WHERE session_id = %s AND product_id = %s
        """
        
        results = self.db_manager.execute_query(query, (session_id, product_id))
        return results[0] if results else None
    
    def _insert_new_item(
        self, 
        session_id: str, 
        product_id: str, 
        product_title: str, 
        quantity: int,
        price: Optional[float],
        image_url: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Insert new item into cart."""
        
        query = """
        INSERT INTO shopping_cart 
        (session_id, product_id, product_title, product_price, product_image_url, 
         quantity, product_metadata, added_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        RETURNING id
        """
        
        metadata_json = json.dumps(metadata or {})
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    query, 
                    (session_id, product_id, product_title, price, image_url, 
                     quantity, metadata_json)
                )
                result = cursor.fetchone()
                conn.commit()
                
                item_id = str(result["id"])
        
        # Get the inserted item for response
        new_item = self._get_cart_item(session_id, product_id)
        
        return {
            "success": True,
            "message": f"Added {quantity} x {product_title} to cart",
            "item": self._format_cart_item(new_item),
            "action": "added"
        }
    
    def _update_item_quantity(
        self, 
        session_id: str, 
        product_id: str, 
        new_quantity: int
    ) -> Dict[str, Any]:
        """Update item quantity in cart."""
        
        if new_quantity <= 0:
            # Remove item if quantity becomes zero or negative
            existing_item = self._get_cart_item(session_id, product_id)
            return self._delete_cart_item(session_id, product_id, existing_item)
        
        query = """
        UPDATE shopping_cart 
        SET quantity = %s, updated_at = CURRENT_TIMESTAMP
        WHERE session_id = %s AND product_id = %s
        """
        
        updated_count = self.db_manager.execute_update(
            query, 
            (new_quantity, session_id, product_id)
        )
        
        if updated_count > 0:
            # Get updated item
            updated_item = self._get_cart_item(session_id, product_id)
            return {
                "success": True,
                "message": f"Updated quantity to {new_quantity}",
                "item": self._format_cart_item(updated_item),
                "action": "updated"
            }
        else:
            return {
                "success": False,
                "error": "Failed to update item quantity",
                "item": None
            }
    
    def _delete_cart_item(
        self, 
        session_id: str, 
        product_id: str, 
        existing_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delete item from cart completely."""
        
        query = "DELETE FROM shopping_cart WHERE session_id = %s AND product_id = %s"
        
        deleted_count = self.db_manager.execute_update(query, (session_id, product_id))
        
        if deleted_count > 0:
            return {
                "success": True,
                "message": f"Removed {existing_item['product_title']} from cart",
                "item": self._format_cart_item(existing_item),
                "action": "removed",
                "removed_completely": True
            }
        else:
            return {
                "success": False,
                "error": "Failed to remove item from cart",
                "item": None,
                "removed_completely": False
            }
    
    def _format_cart_item(self, item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Format cart item for consistent response structure."""
        
        if not item:
            return None
        
        return {
            "id": str(item["id"]),
            "product_id": item["product_id"],
            "product_title": item["product_title"],
            "product_price": float(item["product_price"]) if item["product_price"] else None,
            "product_image_url": item["product_image_url"],
            "quantity": item["quantity"],
            "product_metadata": item["product_metadata"] or {},
            "added_at": item["added_at"].isoformat() if item["added_at"] else None,
            "updated_at": item["updated_at"].isoformat() if item["updated_at"] else None,
            "subtotal": (
                float(item["product_price"]) * item["quantity"] 
                if item["product_price"] else None
            )
        }


# Utility functions for cart operations

def create_shopping_cart_manager(db_manager: Optional[DatabaseManager] = None) -> ShoppingCartManager:
    """Create a new shopping cart manager instance."""
    return ShoppingCartManager(db_manager)


def get_global_cart_manager() -> ShoppingCartManager:
    """Get global shopping cart manager instance."""
    # Use global database manager
    return ShoppingCartManager(get_database_manager())