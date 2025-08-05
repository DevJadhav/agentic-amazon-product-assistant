"""
Shopping cart function calling tools for LangGraph agent workflows.
Provides cart management capabilities with product validation and error handling.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..state.shopping_cart_manager import ShoppingCartManager, get_global_cart_manager

logger = logging.getLogger(__name__)


class AddToCartInput(BaseModel):
    """Input schema for add to cart tool."""
    
    product_id: str = Field(description="Unique identifier for the product")
    product_title: str = Field(description="Display name of the product")
    quantity: int = Field(
        default=1,
        description="Number of items to add to cart",
        ge=1,
        le=100
    )
    price: Optional[float] = Field(
        default=None,
        description="Price per unit of the product",
        ge=0
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL of the product image"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional product metadata (category, brand, etc.)"
    )


class RemoveFromCartInput(BaseModel):
    """Input schema for remove from cart tool."""
    
    product_id: str = Field(description="Unique identifier for the product to remove")
    quantity: Optional[int] = Field(
        default=None,
        description="Number of items to remove (None = remove all)",
        ge=1,
        le=100
    )


class ListCartInput(BaseModel):
    """Input schema for list cart tool."""
    
    include_summary: bool = Field(
        default=True,
        description="Whether to include cart summary with totals"
    )
    format_type: str = Field(
        default="detailed",
        description="Format type: 'detailed', 'summary', or 'minimal'"
    )


class AddToCartTool(BaseTool):
    """Tool for adding products to shopping cart."""
    
    name: str = "add_to_cart"
    description: str = """
    Add a product to the user's shopping cart with quantity and validation.
    
    This tool handles:
    - Adding new products to cart
    - Updating quantities for existing products (combines quantities)
    - Product validation and error handling
    - Price and metadata storage
    
    Use this tool when a user wants to add items to their cart for later purchase.
    """
    
    args_schema: Type[AddToCartInput] = AddToCartInput
    
    def __init__(self, cart_manager: Optional[ShoppingCartManager] = None, **kwargs):
        """Initialize add to cart tool."""
        super().__init__(**kwargs)
        self._cart_manager = cart_manager or get_global_cart_manager()
        self._logger = logging.getLogger(__name__)
    
    @property
    def cart_manager(self) -> ShoppingCartManager:
        """Get cart manager instance."""
        return self._cart_manager
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger
    
    def _run(
        self,
        product_id: str,
        product_title: str,
        quantity: int = 1,
        price: Optional[float] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute add to cart operation synchronously."""
        
        try:
            # Get session ID from context (this would be injected by the agent)
            session_id = self._get_session_id()
            
            # Validate inputs
            validation_result = self._validate_product_data(
                product_id, product_title, quantity, price
            )
            
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "item": None,
                    "cart_updated": False,
                    "message": f"Failed to add {product_title} to cart: {validation_result['error']}"
                }
            
            # Add item to cart
            result = self.cart_manager.add_item(
                session_id=session_id,
                product_id=product_id,
                product_title=product_title,
                quantity=quantity,
                price=price,
                image_url=image_url,
                metadata=metadata or {}
            )
            
            # Format response
            if result["success"]:
                return {
                    "success": True,
                    "message": result["message"],
                    "item": result["item"],
                    "action": result["action"],
                    "cart_updated": True,
                    "tool": "add_to_cart",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "item": None,
                    "cart_updated": False,
                    "message": f"Failed to add {product_title} to cart: {result['error']}"
                }
                
        except Exception as e:
            self.logger.error(f"Add to cart operation failed: {e}")
            return {
                "success": False,
                "error": f"Tool error: {str(e)}",
                "item": None,
                "cart_updated": False,
                "message": f"Failed to add {product_title} to cart due to system error"
            }
    
    async def _arun(
        self,
        product_id: str,
        product_title: str,
        quantity: int = 1,
        price: Optional[float] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute add to cart operation asynchronously."""
        
        # For now, use synchronous implementation
        return self._run(product_id, product_title, quantity, price, image_url, metadata)
    
    def _validate_product_data(
        self,
        product_id: str,
        product_title: str,
        quantity: int,
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Validate product data before adding to cart."""
        
        if not product_id or not product_id.strip():
            return {"valid": False, "error": "Product ID is required"}
        
        if not product_title or not product_title.strip():
            return {"valid": False, "error": "Product title is required"}
        
        if quantity <= 0:
            return {"valid": False, "error": "Quantity must be greater than 0"}
        
        if quantity > 100:
            return {"valid": False, "error": "Quantity cannot exceed 100 items"}
        
        if price is not None and price < 0:
            return {"valid": False, "error": "Price cannot be negative"}
        
        return {"valid": True, "error": None}
    
    def _get_session_id(self) -> str:
        """Get session ID from context or generate default."""
        
        # In a real implementation, this would be injected by the agent
        # For now, return a default session ID
        return "default_session"


class RemoveFromCartTool(BaseTool):
    """Tool for removing products from shopping cart."""
    
    name: str = "remove_from_cart"
    description: str = """
    Remove a product from the user's shopping cart with flexible quantity options.
    
    This tool handles:
    - Complete removal of products from cart
    - Partial removal by specifying quantity
    - Error handling for non-existent items
    - Confirmation of removal operations
    
    Use this tool when a user wants to remove items from their cart.
    """
    
    args_schema: Type[RemoveFromCartInput] = RemoveFromCartInput
    
    def __init__(self, cart_manager: Optional[ShoppingCartManager] = None, **kwargs):
        """Initialize remove from cart tool."""
        super().__init__(**kwargs)
        self._cart_manager = cart_manager or get_global_cart_manager()
        self._logger = logging.getLogger(__name__)
    
    @property
    def cart_manager(self) -> ShoppingCartManager:
        """Get cart manager instance."""
        return self._cart_manager
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger
    
    def _run(
        self,
        product_id: str,
        quantity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute remove from cart operation synchronously."""
        
        try:
            # Get session ID from context
            session_id = self._get_session_id()
            
            # Validate inputs
            if not product_id or not product_id.strip():
                return {
                    "success": False,
                    "error": "Product ID is required",
                    "item": None,
                    "cart_updated": False,
                    "removed_completely": False,
                    "message": "Failed to remove item: Product ID is required"
                }
            
            if quantity is not None and quantity <= 0:
                return {
                    "success": False,
                    "error": "Quantity must be greater than 0 if specified",
                    "item": None,
                    "cart_updated": False,
                    "removed_completely": False,
                    "message": "Failed to remove item: Invalid quantity"
                }
            
            # Remove item from cart
            result = self.cart_manager.remove_item(
                session_id=session_id,
                product_id=product_id,
                quantity=quantity
            )
            
            # Format response
            if result["success"]:
                return {
                    "success": True,
                    "message": result["message"],
                    "item": result["item"],
                    "action": result["action"],
                    "removed_completely": result.get("removed_completely", False),
                    "cart_updated": True,
                    "tool": "remove_from_cart",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "item": None,
                    "cart_updated": False,
                    "removed_completely": False,
                    "message": f"Failed to remove item: {result['error']}"
                }
                
        except Exception as e:
            self.logger.error(f"Remove from cart operation failed: {e}")
            return {
                "success": False,
                "error": f"Tool error: {str(e)}",
                "item": None,
                "cart_updated": False,
                "removed_completely": False,
                "message": "Failed to remove item due to system error"
            }
    
    async def _arun(
        self,
        product_id: str,
        quantity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute remove from cart operation asynchronously."""
        
        # For now, use synchronous implementation
        return self._run(product_id, quantity)
    
    def _get_session_id(self) -> str:
        """Get session ID from context or generate default."""
        
        # In a real implementation, this would be injected by the agent
        return "default_session"


class ListCartTool(BaseTool):
    """Tool for listing shopping cart contents."""
    
    name: str = "list_cart"
    description: str = """
    List all items in the user's shopping cart with formatting options.
    
    This tool provides:
    - Complete cart contents with item details
    - Cart summary with totals and counts
    - Flexible formatting options (detailed, summary, minimal)
    - Empty cart handling
    
    Use this tool when a user wants to view their current cart contents.
    """
    
    args_schema: Type[ListCartInput] = ListCartInput
    
    def __init__(self, cart_manager: Optional[ShoppingCartManager] = None, **kwargs):
        """Initialize list cart tool."""
        super().__init__(**kwargs)
        self._cart_manager = cart_manager or get_global_cart_manager()
        self._logger = logging.getLogger(__name__)
    
    @property
    def cart_manager(self) -> ShoppingCartManager:
        """Get cart manager instance."""
        return self._cart_manager
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger
    
    def _run(
        self,
        include_summary: bool = True,
        format_type: str = "detailed"
    ) -> Dict[str, Any]:
        """Execute list cart operation synchronously."""
        
        try:
            # Get session ID from context
            session_id = self._get_session_id()
            
            # Get cart contents
            cart_items = self.cart_manager.get_cart_contents(session_id)
            
            # Get cart summary if requested
            cart_summary = None
            if include_summary:
                cart_summary = self.cart_manager.get_cart_summary(session_id)
            
            # Format response based on format type
            if format_type == "minimal":
                formatted_items = self._format_minimal(cart_items)
            elif format_type == "summary":
                formatted_items = self._format_summary(cart_items)
            else:  # detailed
                formatted_items = self._format_detailed(cart_items)
            
            # Create response
            response = {
                "success": True,
                "cart_items": formatted_items,
                "item_count": len(cart_items),
                "is_empty": len(cart_items) == 0,
                "tool": "list_cart",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if cart_summary:
                response["cart_summary"] = cart_summary
                response["total_items"] = cart_summary.get("total_items", 0)
                response["total_value"] = cart_summary.get("total_value", 0.0)
                response["unique_products"] = cart_summary.get("unique_products", 0)
            
            # Add appropriate message
            if len(cart_items) == 0:
                response["message"] = "Your cart is empty"
            else:
                item_word = "item" if len(cart_items) == 1 else "items"
                response["message"] = f"Your cart contains {len(cart_items)} {item_word}"
                
                if cart_summary and cart_summary.get("total_value", 0) > 0:
                    response["message"] += f" with a total value of ${cart_summary['total_value']:.2f}"
            
            return response
            
        except Exception as e:
            self.logger.error(f"List cart operation failed: {e}")
            return {
                "success": False,
                "error": f"Tool error: {str(e)}",
                "cart_items": [],
                "item_count": 0,
                "is_empty": True,
                "message": "Failed to retrieve cart contents due to system error"
            }
    
    async def _arun(
        self,
        include_summary: bool = True,
        format_type: str = "detailed"
    ) -> Dict[str, Any]:
        """Execute list cart operation asynchronously."""
        
        # For now, use synchronous implementation
        return self._run(include_summary, format_type)
    
    def _format_detailed(self, cart_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format cart items with full details."""
        
        formatted = []
        
        for item in cart_items:
            formatted_item = {
                "product_id": item["product_id"],
                "product_title": item["product_title"],
                "quantity": item["quantity"],
                "price_per_unit": item["product_price"],
                "subtotal": item["subtotal"],
                "image_url": item["product_image_url"],
                "metadata": item["product_metadata"],
                "added_at": item["added_at"],
                "updated_at": item["updated_at"]
            }
            formatted.append(formatted_item)
        
        return formatted
    
    def _format_summary(self, cart_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format cart items with summary information."""
        
        formatted = []
        
        for item in cart_items:
            formatted_item = {
                "product_id": item["product_id"],
                "product_title": item["product_title"],
                "quantity": item["quantity"],
                "subtotal": item["subtotal"]
            }
            formatted.append(formatted_item)
        
        return formatted
    
    def _format_minimal(self, cart_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format cart items with minimal information."""
        
        formatted = []
        
        for item in cart_items:
            formatted_item = {
                "product_title": item["product_title"],
                "quantity": item["quantity"]
            }
            formatted.append(formatted_item)
        
        return formatted
    
    def _get_session_id(self) -> str:
        """Get session ID from context or generate default."""
        
        # In a real implementation, this would be injected by the agent
        return "default_session"


class ClearCartTool(BaseTool):
    """Tool for clearing all items from shopping cart."""
    
    name: str = "clear_cart"
    description: str = """
    Clear all items from the user's shopping cart.
    
    This tool provides:
    - Complete cart clearing functionality
    - Confirmation of cleared items
    - Safety confirmation before clearing
    
    Use this tool when a user wants to empty their entire cart.
    """
    
    def __init__(self, cart_manager: Optional[ShoppingCartManager] = None, **kwargs):
        """Initialize clear cart tool."""
        super().__init__(**kwargs)
        self._cart_manager = cart_manager or get_global_cart_manager()
        self._logger = logging.getLogger(__name__)
    
    @property
    def cart_manager(self) -> ShoppingCartManager:
        """Get cart manager instance."""
        return self._cart_manager
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger
    
    def _run(self) -> Dict[str, Any]:
        """Execute clear cart operation synchronously."""
        
        try:
            # Get session ID from context
            session_id = self._get_session_id()
            
            # Clear cart
            result = self.cart_manager.clear_cart(session_id)
            
            # Format response
            if result["success"]:
                return {
                    "success": True,
                    "message": result["message"],
                    "items_removed": result["items_removed"],
                    "cleared_items": result["cleared_items"],
                    "cart_updated": True,
                    "tool": "clear_cart",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "items_removed": 0,
                    "cart_updated": False,
                    "message": f"Failed to clear cart: {result['error']}"
                }
                
        except Exception as e:
            self.logger.error(f"Clear cart operation failed: {e}")
            return {
                "success": False,
                "error": f"Tool error: {str(e)}",
                "items_removed": 0,
                "cart_updated": False,
                "message": "Failed to clear cart due to system error"
            }
    
    async def _arun(self) -> Dict[str, Any]:
        """Execute clear cart operation asynchronously."""
        
        # For now, use synchronous implementation
        return self._run()
    
    def _get_session_id(self) -> str:
        """Get session ID from context or generate default."""
        
        # In a real implementation, this would be injected by the agent
        return "default_session"


# Utility functions for creating cart tools

def create_cart_tools(cart_manager: Optional[ShoppingCartManager] = None) -> List[BaseTool]:
    """Create all shopping cart tools with shared cart manager."""
    
    manager = cart_manager or get_global_cart_manager()
    
    return [
        AddToCartTool(cart_manager=manager),
        RemoveFromCartTool(cart_manager=manager),
        ListCartTool(cart_manager=manager),
        ClearCartTool(cart_manager=manager)
    ]


def get_cart_tool_by_name(tool_name: str, cart_manager: Optional[ShoppingCartManager] = None) -> Optional[BaseTool]:
    """Get specific cart tool by name."""
    
    manager = cart_manager or get_global_cart_manager()
    
    tool_map = {
        "add_to_cart": AddToCartTool(cart_manager=manager),
        "remove_from_cart": RemoveFromCartTool(cart_manager=manager),
        "list_cart": ListCartTool(cart_manager=manager),
        "clear_cart": ClearCartTool(cart_manager=manager)
    }
    
    return tool_map.get(tool_name)


def get_cart_tools_info() -> Dict[str, Any]:
    """Get information about available cart tools."""
    
    return {
        "available_tools": [
            {
                "name": "add_to_cart",
                "description": "Add products to shopping cart",
                "required_params": ["product_id", "product_title"],
                "optional_params": ["quantity", "price", "image_url", "metadata"]
            },
            {
                "name": "remove_from_cart", 
                "description": "Remove products from shopping cart",
                "required_params": ["product_id"],
                "optional_params": ["quantity"]
            },
            {
                "name": "list_cart",
                "description": "List all items in shopping cart",
                "required_params": [],
                "optional_params": ["include_summary", "format_type"]
            },
            {
                "name": "clear_cart",
                "description": "Clear all items from shopping cart",
                "required_params": [],
                "optional_params": []
            }
        ],
        "tool_count": 4,
        "supports_async": True,
        "requires_session": True
    }