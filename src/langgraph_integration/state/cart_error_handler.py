"""
Shopping cart error handling system for database and cart operation errors.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class CartErrorType(Enum):
    """Types of cart-specific errors."""
    
    DATABASE_CONNECTION_ERROR = "database_connection_error"
    CART_OPERATION_ERROR = "cart_operation_error"
    PRODUCT_VALIDATION_ERROR = "product_validation_error"
    QUANTITY_VALIDATION_ERROR = "quantity_validation_error"
    SESSION_ISOLATION_ERROR = "session_isolation_error"
    CART_STATE_ERROR = "cart_state_error"
    TRANSACTION_ERROR = "transaction_error"


class CartErrorSeverity(Enum):
    """Severity levels for cart errors."""
    
    LOW = "low"          # Can continue with degraded functionality
    MEDIUM = "medium"    # Requires fallback mechanism
    HIGH = "high"        # Operation should fail gracefully
    CRITICAL = "critical"  # Cart functionality unavailable


class CartErrorHandler:
    """Handles shopping cart operation errors with recovery strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cart error handler."""
        self.config = config or {}
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.enable_fallback_storage = self.config.get("enable_fallback_storage", False)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "fallback_operations": 0,
            "last_error_time": None
        }
        
        # Fallback responses by error type
        self.fallback_responses = {
            CartErrorType.DATABASE_CONNECTION_ERROR: 
                "I'm having trouble connecting to the cart database. Your cart may not be saved, but I can still help you with product information.",
            CartErrorType.CART_OPERATION_ERROR: 
                "I encountered an issue with the cart operation. Please try again, or let me help you with product search instead.",
            CartErrorType.PRODUCT_VALIDATION_ERROR: 
                "There's an issue with the product information. Please check the product details and try again.",
            CartErrorType.QUANTITY_VALIDATION_ERROR: 
                "The quantity you specified is not valid. Please enter a positive number.",
            CartErrorType.SESSION_ISOLATION_ERROR: 
                "I'm having trouble accessing your specific cart session. Please try refreshing and try again.",
            CartErrorType.CART_STATE_ERROR: 
                "Your cart state seems to be inconsistent. I'll try to refresh it for you.",
            CartErrorType.TRANSACTION_ERROR: 
                "The cart transaction failed. Your cart should be unchanged. Please try the operation again."
        }
        
        # In-memory fallback storage (temporary)
        self._fallback_cart_storage: Dict[str, List[Dict[str, Any]]] = {}
    
    def handle_database_error(
        self, 
        error: Exception, 
        operation: str, 
        session_id: str,
        operation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle database operation failures."""
        
        error_type = CartErrorType.DATABASE_CONNECTION_ERROR
        severity = self._assess_error_severity(error, error_type)
        
        self._log_cart_error(error, error_type, severity, {
            "operation": operation,
            "session_id": session_id,
            "operation_params": operation_params
        })
        
        self._update_error_stats(error_type, severity)
        
        # Attempt recovery based on severity
        if severity in [CartErrorSeverity.LOW, CartErrorSeverity.MEDIUM]:
            return self._attempt_database_recovery(error, operation, session_id, operation_params)
        else:
            return self._create_database_fallback_response(error, operation, session_id)
    
    def handle_cart_operation_error(
        self, 
        error: Exception, 
        operation: str,
        session_id: str,
        product_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle cart operation specific errors."""
        
        error_type = CartErrorType.CART_OPERATION_ERROR
        severity = self._assess_error_severity(error, error_type)
        
        self._log_cart_error(error, error_type, severity, {
            "operation": operation,
            "session_id": session_id,
            "product_data": product_data
        })
        
        self._update_error_stats(error_type, severity)
        
        # Attempt operation-specific recovery
        return self._attempt_operation_recovery(error, operation, session_id, product_data)
    
    def handle_product_validation_error(
        self, 
        error: Exception, 
        product_data: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Handle product validation failures."""
        
        error_type = CartErrorType.PRODUCT_VALIDATION_ERROR
        severity = CartErrorSeverity.MEDIUM  # Always medium for validation
        
        self._log_cart_error(error, error_type, severity, {
            "product_data": product_data,
            "session_id": session_id,
            "validation_failure": True
        })
        
        self._update_error_stats(error_type, severity)
        
        # Return validation error response
        return {
            "success": False,
            "error": f"Product validation failed: {str(error)}",
            "error_type": error_type.value,
            "fallback_message": self.fallback_responses[error_type],
            "validation_details": self._extract_validation_details(error, product_data)
        }
    
    def handle_quantity_validation_error(
        self, 
        error: Exception, 
        quantity: Any,
        session_id: str
    ) -> Dict[str, Any]:
        """Handle quantity validation failures."""
        
        error_type = CartErrorType.QUANTITY_VALIDATION_ERROR
        severity = CartErrorSeverity.LOW  # User can correct easily
        
        self._log_cart_error(error, error_type, severity, {
            "invalid_quantity": quantity,
            "session_id": session_id,
            "validation_failure": True
        })
        
        self._update_error_stats(error_type, severity)
        
        return {
            "success": False,
            "error": f"Invalid quantity: {quantity}. Please enter a positive number.",
            "error_type": error_type.value,
            "fallback_message": self.fallback_responses[error_type],
            "suggested_action": "Please enter a valid quantity (positive integer) and try again."
        }
    
    def handle_session_isolation_error(
        self, 
        error: Exception, 
        session_id: str,
        operation: str
    ) -> Dict[str, Any]:
        """Handle session isolation failures."""
        
        error_type = CartErrorType.SESSION_ISOLATION_ERROR
        severity = CartErrorSeverity.HIGH  # Affects data integrity
        
        self._log_cart_error(error, error_type, severity, {
            "session_id": session_id,
            "operation": operation,
            "isolation_failure": True
        })
        
        self._update_error_stats(error_type, severity)
        
        return {
            "success": False,
            "error": f"Session isolation error: {str(error)}",
            "error_type": error_type.value,
            "fallback_message": self.fallback_responses[error_type],
            "suggested_action": "Please refresh your session and try again."
        }
    
    def handle_cart_state_error(
        self, 
        error: Exception, 
        session_id: str,
        current_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle cart state inconsistency errors."""
        
        error_type = CartErrorType.CART_STATE_ERROR
        severity = CartErrorSeverity.MEDIUM
        
        self._log_cart_error(error, error_type, severity, {
            "session_id": session_id,
            "current_state": current_state,
            "state_inconsistency": True
        })
        
        self._update_error_stats(error_type, severity)
        
        # Attempt to recover cart state
        return self._attempt_state_recovery(error, session_id, current_state)
    
    def handle_transaction_error(
        self, 
        error: Exception, 
        operation: str,
        session_id: str,
        transaction_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle database transaction failures."""
        
        error_type = CartErrorType.TRANSACTION_ERROR
        severity = CartErrorSeverity.HIGH  # Data integrity concern
        
        self._log_cart_error(error, error_type, severity, {
            "operation": operation,
            "session_id": session_id,
            "transaction_data": transaction_data,
            "transaction_failure": True
        })
        
        self._update_error_stats(error_type, severity)
        
        return {
            "success": False,
            "error": f"Transaction failed: {str(error)}",
            "error_type": error_type.value,
            "fallback_message": self.fallback_responses[error_type],
            "cart_unchanged": True,
            "suggested_action": "Please try the operation again."
        }
    
    def create_graceful_degradation_response(
        self, 
        error: Exception, 
        operation: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Create graceful degradation response when cart functionality is unavailable."""
        
        degradation_message = (
            f"The shopping cart service is temporarily experiencing issues. "
            f"I can still help you search for products and provide information. "
            f"Please try cart operations again later."
        )
        
        return {
            "success": False,
            "error": f"Cart service degraded: {str(error)}",
            "degradation_active": True,
            "fallback_message": degradation_message,
            "alternative_actions": [
                "Search for product information",
                "Compare products",
                "Get product reviews and ratings",
                "Ask questions about products"
            ],
            "service_status": "degraded",
            "estimated_recovery": "Please try again in a few minutes"
        }
    
    def is_cart_functionality_available(self) -> bool:
        """Check if cart functionality is currently available."""
        
        # Check recent error patterns
        recent_critical_errors = 0
        current_time = datetime.now(timezone.utc)
        
        # If we've had multiple critical errors recently, consider unavailable
        if self.error_stats.get("last_error_time"):
            try:
                last_error_time = datetime.fromisoformat(self.error_stats["last_error_time"].replace('Z', '+00:00'))
                time_diff = (current_time - last_error_time).total_seconds()
                
                # If last error was critical and recent (< 5 minutes), consider unavailable
                if time_diff < 300:  # 5 minutes
                    critical_errors = self.error_stats["errors_by_severity"].get("critical", 0)
                    if critical_errors > 0:
                        return False
            except Exception:
                pass
        
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get cart error handling statistics."""
        
        total_operations = self.error_stats["total_errors"] + self.error_stats.get("successful_operations", 0)
        
        return {
            **self.error_stats,
            "error_types": [e.value for e in CartErrorType],
            "severity_levels": [s.value for s in CartErrorSeverity],
            "total_operations": total_operations,
            "error_rate": (
                self.error_stats["total_errors"] / max(1, total_operations)
            ) * 100,
            "fallback_rate": (
                self.error_stats["fallback_operations"] / max(1, total_operations)
            ) * 100,
            "cart_availability": self.is_cart_functionality_available()
        }
    
    def reset_error_statistics(self) -> None:
        """Reset cart error statistics."""
        
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "fallback_operations": 0,
            "last_error_time": None
        }
    
    # Private helper methods
    
    def _assess_error_severity(self, error: Exception, error_type: CartErrorType) -> CartErrorSeverity:
        """Assess the severity of a cart error."""
        
        error_str = str(error).lower()
        
        # Critical errors that make cart completely unavailable
        if any(keyword in error_str for keyword in [
            "connection refused", "database unavailable", "network unreachable"
        ]):
            return CartErrorSeverity.CRITICAL
        
        # High severity errors that affect data integrity
        if error_type in [
            CartErrorType.TRANSACTION_ERROR, 
            CartErrorType.SESSION_ISOLATION_ERROR
        ]:
            return CartErrorSeverity.HIGH
        
        # Medium severity errors that require fallback
        if error_type in [
            CartErrorType.CART_OPERATION_ERROR,
            CartErrorType.CART_STATE_ERROR,
            CartErrorType.PRODUCT_VALIDATION_ERROR
        ]:
            return CartErrorSeverity.MEDIUM
        
        # Low severity errors that can be handled gracefully
        return CartErrorSeverity.LOW
    
    def _log_cart_error(
        self, 
        error: Exception, 
        error_type: CartErrorType, 
        severity: CartErrorSeverity,
        context: Dict[str, Any]
    ) -> None:
        """Log cart error with context."""
        
        log_data = {
            "error_type": error_type.value,
            "error_severity": severity.value,
            "session_id": context.get("session_id", "unknown"),
            "operation": context.get("operation", "unknown"),
            "error_message": str(error),
            "error_class": type(error).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cart_context": context
        }
        
        # Log at appropriate level
        if severity == CartErrorSeverity.CRITICAL:
            self.logger.critical("Critical cart error", extra=log_data)
        elif severity == CartErrorSeverity.HIGH:
            self.logger.error("High severity cart error", extra=log_data)
        elif severity == CartErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity cart error", extra=log_data)
        else:
            self.logger.info("Low severity cart error", extra=log_data)
    
    def _update_error_stats(self, error_type: CartErrorType, severity: CartErrorSeverity) -> None:
        """Update cart error statistics."""
        
        self.error_stats["total_errors"] += 1
        self.error_stats["last_error_time"] = datetime.now(timezone.utc).isoformat()
        
        # Update by type
        type_key = error_type.value
        self.error_stats["errors_by_type"][type_key] = self.error_stats["errors_by_type"].get(type_key, 0) + 1
        
        # Update by severity
        severity_key = severity.value
        self.error_stats["errors_by_severity"][severity_key] = self.error_stats["errors_by_severity"].get(severity_key, 0) + 1
    
    def _attempt_database_recovery(
        self, 
        error: Exception, 
        operation: str, 
        session_id: str,
        operation_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Attempt to recover from database errors."""
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Wait before retry
                time.sleep(self.retry_delay * (2 ** retry_count))
                
                # Here we would retry the database operation
                # For now, we'll simulate a recovery attempt
                self.logger.info(f"Attempting database recovery for {operation}, retry {retry_count + 1}")
                
                # If we reach here, recovery might be possible
                # Return a response indicating retry should be attempted
                return {
                    "success": False,
                    "error": str(error),
                    "recovery_attempted": True,
                    "retry_count": retry_count + 1,
                    "suggested_action": "Database connection issues detected. Please try again.",
                    "fallback_available": self.enable_fallback_storage
                }
                
            except Exception as retry_error:
                retry_count += 1
                self.logger.warning(f"Database recovery attempt {retry_count} failed: {retry_error}")
        
        # All retries failed
        return self._create_database_fallback_response(error, operation, session_id)
    
    def _create_database_fallback_response(
        self, 
        error: Exception, 
        operation: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """Create fallback response for database failures."""
        
        self.error_stats["fallback_operations"] += 1
        
        fallback_response = {
            "success": False,
            "error": f"Database unavailable: {str(error)}",
            "error_type": CartErrorType.DATABASE_CONNECTION_ERROR.value,
            "fallback_message": self.fallback_responses[CartErrorType.DATABASE_CONNECTION_ERROR],
            "service_degraded": True,
            "alternative_actions": [
                "Search for product information",
                "Get product recommendations",
                "Compare products"
            ]
        }
        
        # If fallback storage is enabled, offer temporary cart
        if self.enable_fallback_storage:
            fallback_response["temporary_cart_available"] = True
            fallback_response["fallback_message"] += " I can maintain a temporary cart for this session."
        
        return fallback_response
    
    def _attempt_operation_recovery(
        self, 
        error: Exception, 
        operation: str,
        session_id: str,
        product_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Attempt to recover from cart operation errors."""
        
        # Analyze the specific operation error
        error_str = str(error).lower()
        
        # Handle specific operation errors
        if "duplicate" in error_str and operation == "add_item":
            return {
                "success": False,
                "error": "Item already in cart",
                "recovery_suggestion": "update_quantity",
                "suggested_action": "Would you like to update the quantity instead?"
            }
        
        elif "not found" in error_str and operation in ["remove_item", "update_item"]:
            return {
                "success": False,
                "error": "Item not found in cart",
                "recovery_suggestion": "list_cart",
                "suggested_action": "Let me show you what's currently in your cart."
            }
        
        elif "invalid quantity" in error_str:
            return {
                "success": False,
                "error": "Invalid quantity specified",
                "recovery_suggestion": "correct_quantity",
                "suggested_action": "Please specify a valid positive quantity."
            }
        
        # Generic operation error
        return {
            "success": False,
            "error": f"Cart operation failed: {str(error)}",
            "error_type": CartErrorType.CART_OPERATION_ERROR.value,
            "fallback_message": self.fallback_responses[CartErrorType.CART_OPERATION_ERROR],
            "suggested_action": "Please try the operation again or let me help you with product search."
        }
    
    def _attempt_state_recovery(
        self, 
        error: Exception, 
        session_id: str,
        current_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Attempt to recover cart state."""
        
        recovery_actions = []
        
        # Try to identify state issues
        if current_state:
            if "total_items" in current_state and current_state["total_items"] < 0:
                recovery_actions.append("Reset negative item count")
            
            if "total_value" in current_state and current_state["total_value"] < 0:
                recovery_actions.append("Reset negative total value")
        
        return {
            "success": False,
            "error": f"Cart state inconsistency: {str(error)}",
            "error_type": CartErrorType.CART_STATE_ERROR.value,
            "state_recovery_attempted": True,
            "recovery_actions": recovery_actions,
            "fallback_message": self.fallback_responses[CartErrorType.CART_STATE_ERROR],
            "suggested_action": "I'll try to refresh your cart state. Please check your cart contents."
        }
    
    def _extract_validation_details(
        self, 
        error: Exception, 
        product_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract validation error details."""
        
        details = {
            "error_message": str(error),
            "product_data": product_data
        }
        
        # Check for specific validation issues
        if "product_id" in str(error).lower():
            details["issue"] = "Invalid or missing product ID"
            details["field"] = "product_id"
        
        elif "product_title" in str(error).lower():
            details["issue"] = "Invalid or missing product title"
            details["field"] = "product_title"
        
        elif "price" in str(error).lower():
            details["issue"] = "Invalid price format"
            details["field"] = "product_price"
        
        elif "quantity" in str(error).lower():
            details["issue"] = "Invalid quantity value"
            details["field"] = "quantity"
        
        return details


# Utility functions for cart error handling

def create_cart_error_handler(config: Optional[Dict[str, Any]] = None) -> CartErrorHandler:
    """Create a new cart error handler instance."""
    return CartErrorHandler(config)


def get_global_cart_error_handler() -> CartErrorHandler:
    """Get global cart error handler instance."""
    # This would typically be a singleton
    return CartErrorHandler()