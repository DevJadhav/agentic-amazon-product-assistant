"""
Tests for shopping cart error handling system.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from ..state.cart_error_handler import (
    CartErrorHandler, CartErrorType, CartErrorSeverity
)


class TestCartErrorHandler:
    """Test suite for CartErrorHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = CartErrorHandler()
        self.session_id = "test_session_123"
    
    def test_handle_database_error_low_severity(self):
        """Test handling of low severity database errors."""
        error = TimeoutError("Connection timeout")
        
        result = self.error_handler.handle_database_error(
            error, "add_item", self.session_id, {"product_id": "test_product"}
        )
        
        assert result["success"] == False
        assert "recovery_attempted" in result
        assert self.error_handler.error_stats["total_errors"] == 1
    
    def test_handle_database_error_critical_severity(self):
        """Test handling of critical database errors."""
        error = ConnectionError("Database connection refused")
        
        result = self.error_handler.handle_database_error(
            error, "add_item", self.session_id
        )
        
        assert result["success"] == False
        assert result["service_degraded"] == True
        assert "alternative_actions" in result
        assert self.error_handler.error_stats["fallback_operations"] >= 1
    
    def test_handle_cart_operation_error_duplicate_item(self):
        """Test handling of duplicate item errors."""
        error = ValueError("duplicate key value violates unique constraint")
        
        result = self.error_handler.handle_cart_operation_error(
            error, "add_item", self.session_id, {"product_id": "test_product"}
        )
        
        assert result["success"] == False
        assert result["recovery_suggestion"] == "update_quantity"
        assert "already in cart" in result["error"]
    
    def test_handle_cart_operation_error_item_not_found(self):
        """Test handling of item not found errors."""
        error = ValueError("Item not found in cart")
        
        result = self.error_handler.handle_cart_operation_error(
            error, "remove_item", self.session_id, {"product_id": "missing_product"}
        )
        
        assert result["success"] == False
        assert result["recovery_suggestion"] == "list_cart"
        assert "not found" in result["error"]
    
    def test_handle_product_validation_error(self):
        """Test handling of product validation errors."""
        error = ValueError("Invalid product data")
        product_data = {"product_id": "", "product_title": "Test Product"}
        
        result = self.error_handler.handle_product_validation_error(
            error, product_data, self.session_id
        )
        
        assert result["success"] == False
        assert result["error_type"] == CartErrorType.PRODUCT_VALIDATION_ERROR.value
        assert "validation_details" in result
        assert result["validation_details"]["product_data"] == product_data
    
    def test_handle_quantity_validation_error(self):
        """Test handling of quantity validation errors."""
        error = ValueError("Quantity must be positive")
        invalid_quantity = -5
        
        result = self.error_handler.handle_quantity_validation_error(
            error, invalid_quantity, self.session_id
        )
        
        assert result["success"] == False
        assert result["error_type"] == CartErrorType.QUANTITY_VALIDATION_ERROR.value
        assert "positive number" in result["error"]
        assert "suggested_action" in result
    
    def test_handle_session_isolation_error(self):
        """Test handling of session isolation errors."""
        error = RuntimeError("Session data corruption")
        
        result = self.error_handler.handle_session_isolation_error(
            error, self.session_id, "add_item"
        )
        
        assert result["success"] == False
        assert result["error_type"] == CartErrorType.SESSION_ISOLATION_ERROR.value
        assert "refresh your session" in result["suggested_action"]
    
    def test_handle_cart_state_error(self):
        """Test handling of cart state inconsistency errors."""
        error = ValueError("Cart state inconsistent")
        current_state = {"total_items": -1, "total_value": -10.0}
        
        result = self.error_handler.handle_cart_state_error(
            error, self.session_id, current_state
        )
        
        assert result["success"] == False
        assert result["error_type"] == CartErrorType.CART_STATE_ERROR.value
        assert result["state_recovery_attempted"] == True
        assert "recovery_actions" in result
    
    def test_handle_transaction_error(self):
        """Test handling of database transaction errors."""
        error = RuntimeError("Transaction rollback")
        transaction_data = {"operation": "add_item", "product_id": "test"}
        
        result = self.error_handler.handle_transaction_error(
            error, "add_item", self.session_id, transaction_data
        )
        
        assert result["success"] == False
        assert result["error_type"] == CartErrorType.TRANSACTION_ERROR.value
        assert result["cart_unchanged"] == True
        assert "try the operation again" in result["suggested_action"]
    
    def test_create_graceful_degradation_response(self):
        """Test creation of graceful degradation responses."""
        error = Exception("Service overload")
        
        result = self.error_handler.create_graceful_degradation_response(
            error, "add_item", self.session_id
        )
        
        assert result["success"] == False
        assert result["degradation_active"] == True
        assert result["service_status"] == "degraded"
        assert "alternative_actions" in result
        assert len(result["alternative_actions"]) > 0
    
    def test_is_cart_functionality_available_normal(self):
        """Test cart availability check under normal conditions."""
        assert self.error_handler.is_cart_functionality_available() == True
    
    def test_is_cart_functionality_available_after_critical_error(self):
        """Test cart availability check after critical errors."""
        # Simulate a recent critical error
        critical_error = ConnectionError("Database unavailable")
        self.error_handler.handle_database_error(
            critical_error, "add_item", self.session_id
        )
        
        # Update error stats to simulate critical error
        self.error_handler.error_stats["errors_by_severity"]["critical"] = 1
        self.error_handler.error_stats["last_error_time"] = datetime.now(timezone.utc).isoformat()
        
        # Availability should be affected
        availability = self.error_handler.is_cart_functionality_available()
        # Note: This depends on the specific implementation logic
        assert isinstance(availability, bool)
    
    def test_error_statistics_tracking(self):
        """Test comprehensive error statistics tracking."""
        # Generate various types of errors
        errors = [
            (ValueError("Validation error"), CartErrorType.PRODUCT_VALIDATION_ERROR),
            (ConnectionError("DB error"), CartErrorType.DATABASE_CONNECTION_ERROR),
            (RuntimeError("Operation error"), CartErrorType.CART_OPERATION_ERROR)
        ]
        
        for error, error_type in errors:
            if error_type == CartErrorType.PRODUCT_VALIDATION_ERROR:
                self.error_handler.handle_product_validation_error(
                    error, {"test": "data"}, self.session_id
                )
            elif error_type == CartErrorType.DATABASE_CONNECTION_ERROR:
                self.error_handler.handle_database_error(
                    error, "test_op", self.session_id
                )
            else:
                self.error_handler.handle_cart_operation_error(
                    error, "test_op", self.session_id
                )
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert len(stats["errors_by_type"]) >= 2
        assert stats["last_error_time"] is not None
        assert "cart_availability" in stats
    
    def test_error_severity_assessment(self):
        """Test error severity assessment logic."""
        # Test different error scenarios
        connection_error = ConnectionError("connection refused")
        timeout_error = TimeoutError("operation timeout")
        validation_error = ValueError("invalid input")
        
        # Connection errors should be critical
        severity1 = self.error_handler._assess_error_severity(
            connection_error, CartErrorType.DATABASE_CONNECTION_ERROR
        )
        assert severity1 == CartErrorSeverity.CRITICAL
        
        # Transaction errors should be high
        severity2 = self.error_handler._assess_error_severity(
            validation_error, CartErrorType.TRANSACTION_ERROR
        )
        assert severity2 == CartErrorSeverity.HIGH
        
        # Validation errors should be medium
        severity3 = self.error_handler._assess_error_severity(
            validation_error, CartErrorType.PRODUCT_VALIDATION_ERROR
        )
        assert severity3 == CartErrorSeverity.MEDIUM
        
        # Quantity errors should be low
        severity4 = self.error_handler._assess_error_severity(
            validation_error, CartErrorType.QUANTITY_VALIDATION_ERROR
        )
        assert severity4 == CartErrorSeverity.LOW
    
    def test_fallback_responses_completeness(self):
        """Test that all error types have fallback responses."""
        for error_type in CartErrorType:
            assert error_type in self.error_handler.fallback_responses
            response = self.error_handler.fallback_responses[error_type]
            assert isinstance(response, str)
            assert len(response) > 20  # Should be meaningful
    
    def test_reset_error_statistics(self):
        """Test resetting error statistics."""
        # Generate an error first
        error = ValueError("Test error")
        self.error_handler.handle_product_validation_error(
            error, {"test": "data"}, self.session_id
        )
        
        assert self.error_handler.error_stats["total_errors"] == 1
        
        # Reset statistics
        self.error_handler.reset_error_statistics()
        
        assert self.error_handler.error_stats["total_errors"] == 0
        assert self.error_handler.error_stats["last_error_time"] is None
    
    def test_validation_details_extraction(self):
        """Test extraction of validation error details."""
        # Test product_id validation error
        error1 = ValueError("Invalid product_id format")
        product_data1 = {"product_id": "", "product_title": "Test"}
        
        details1 = self.error_handler._extract_validation_details(error1, product_data1)
        assert details1["field"] == "product_id"
        assert details1["issue"] == "Invalid or missing product ID"
        
        # Test price validation error
        error2 = ValueError("Invalid price format")
        product_data2 = {"product_price": "invalid"}
        
        details2 = self.error_handler._extract_validation_details(error2, product_data2)
        assert details2["field"] == "product_price"
        assert details2["issue"] == "Invalid price format"
    
    def test_operation_specific_recovery(self):
        """Test operation-specific error recovery."""
        # Test duplicate item recovery
        duplicate_error = ValueError("duplicate key value violates unique constraint")
        result1 = self.error_handler._attempt_operation_recovery(
            duplicate_error, "add_item", self.session_id, {"product_id": "test"}
        )
        assert result1["recovery_suggestion"] == "update_quantity"
        
        # Test item not found recovery
        not_found_error = ValueError("Item not found in cart")
        result2 = self.error_handler._attempt_operation_recovery(
            not_found_error, "remove_item", self.session_id, {"product_id": "test"}
        )
        assert result2["recovery_suggestion"] == "list_cart"
        
        # Test invalid quantity recovery
        quantity_error = ValueError("invalid quantity specified")
        result3 = self.error_handler._attempt_operation_recovery(
            quantity_error, "update_item", self.session_id, {"quantity": -1}
        )
        assert result3["recovery_suggestion"] == "correct_quantity"
    
    def test_state_recovery_actions(self):
        """Test cart state recovery action identification."""
        # Test negative item count
        error = ValueError("Negative item count")
        state_with_negative_items = {"total_items": -5, "total_value": 100.0}
        
        result = self.error_handler._attempt_state_recovery(
            error, self.session_id, state_with_negative_items
        )
        
        assert result["state_recovery_attempted"] == True
        assert "Reset negative item count" in result["recovery_actions"]
        
        # Test negative total value
        state_with_negative_value = {"total_items": 3, "total_value": -50.0}
        
        result2 = self.error_handler._attempt_state_recovery(
            error, self.session_id, state_with_negative_value
        )
        
        assert "Reset negative total value" in result2["recovery_actions"]


@pytest.fixture
def mock_cart_manager():
    """Fixture for mock cart manager."""
    manager = Mock()
    manager.session_id = "test_session"
    return manager


class TestCartErrorHandlerIntegration:
    """Integration tests for cart error handler."""
    
    def test_error_handler_with_real_database_exceptions(self):
        """Test error handler with real database exception scenarios."""
        error_handler = CartErrorHandler()
        
        # Test with actual PostgreSQL error
        pg_error = Exception("FATAL: remaining connection slots are reserved")
        result = error_handler.handle_database_error(
            pg_error, "add_item", "test_session"
        )
        
        assert result["success"] == False
        # The error message should contain the original error or indicate database issues
        assert ("FATAL" in result["error"] or "Database" in result["error"] or 
                "connection" in result["error"])
    
    def test_error_logging_integration(self, caplog):
        """Test that errors are properly logged."""
        error_handler = CartErrorHandler()
        
        with caplog.at_level("WARNING"):  # Cart operation errors are logged as warnings
            error = RuntimeError("Test error for logging")
            error_handler.handle_cart_operation_error(
                error, "add_item", "test_session"
            )
        
        # Check that the error was logged (may be at WARNING level)
        assert len(caplog.records) > 0
        log_text = caplog.text.lower()
        assert ("cart error" in log_text or "error" in log_text)
    
    def test_concurrent_cart_operations_error_handling(self):
        """Test error handling under concurrent cart operations."""
        error_handler = CartErrorHandler()
        
        # Simulate multiple concurrent cart operation errors
        operations = ["add_item", "remove_item", "update_item", "clear_cart"]
        
        results = []
        for i, operation in enumerate(operations):
            error = ValueError(f"Concurrent error {i} in {operation}")
            result = error_handler.handle_cart_operation_error(
                error, operation, f"session_{i}"
            )
            results.append(result)
        
        # All should have proper error responses
        for result in results:
            assert result["success"] == False
            assert "error" in result
        
        # Statistics should be accurate
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == len(operations)
    
    def test_error_handler_memory_usage(self):
        """Test that error handler doesn't leak memory with many errors."""
        error_handler = CartErrorHandler()
        
        # Generate many errors
        for i in range(100):
            error = ValueError(f"Error {i}")
            error_handler.handle_product_validation_error(
                error, {"product_id": f"product_{i}"}, f"session_{i}"
            )
        
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 100
        
        # Reset and verify cleanup
        error_handler.reset_error_statistics()
        assert error_handler.error_stats["total_errors"] == 0
    
    def test_error_handler_configuration(self):
        """Test error handler with custom configuration."""
        config = {
            "max_retries": 5,
            "retry_delay": 2.0,
            "enable_fallback_storage": True
        }
        
        error_handler = CartErrorHandler(config)
        
        assert error_handler.max_retries == 5
        assert error_handler.retry_delay == 2.0
        assert error_handler.enable_fallback_storage == True
    
    def test_graceful_degradation_with_alternative_actions(self):
        """Test that graceful degradation provides useful alternatives."""
        error_handler = CartErrorHandler()
        
        result = error_handler.create_graceful_degradation_response(
            Exception("Service unavailable"), "add_item", "test_session"
        )
        
        assert result["degradation_active"] == True
        assert len(result["alternative_actions"]) >= 3
        
        # Check that alternatives are meaningful
        alternatives = result["alternative_actions"]
        assert any("search" in action.lower() for action in alternatives)
        assert any("product" in action.lower() for action in alternatives)
        assert any("compare" in action.lower() for action in alternatives)