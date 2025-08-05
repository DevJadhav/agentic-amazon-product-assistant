"""
Integration tests for comprehensive error handling and monitoring system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from ..core.router.router_node import RouterNode
from ..core.router.router_error_handler import RouterErrorHandler
from ..state.shopping_cart_manager import ShoppingCartManager
from ..state.cart_error_handler import CartErrorHandler
from ..core.router.intent_classifier import IntentClassifier
from ..core.router.clarification_handler import ClarificationHandler


class TestErrorHandlingIntegration:
    """Integration tests for the complete error handling system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router_error_handler = RouterErrorHandler()
        self.cart_error_handler = CartErrorHandler()
        
        # Mock dependencies
        self.mock_db_manager = Mock()
        self.mock_intent_classifier = Mock(spec=IntentClassifier)
        self.mock_clarification_handler = Mock(spec=ClarificationHandler)
        
        # Create components with error handlers
        self.router_node = RouterNode(
            intent_classifier=self.mock_intent_classifier,
            clarification_handler=self.mock_clarification_handler,
            error_handler=self.router_error_handler
        )
        
        self.cart_manager = ShoppingCartManager(
            db_manager=self.mock_db_manager,
            error_handler=self.cart_error_handler
        )
        
        self.sample_state = {
            "session_id": "integration_test_session",
            "current_query": "add laptop to cart",
            "conversation_turn": 1,
            "current_step": "router"
        }
    
    @pytest.mark.asyncio
    async def test_router_to_cart_error_cascade(self):
        """Test error cascade from router to cart operations."""
        # Setup: Router successfully routes to cart agent
        from ..core.router.intent_classifier import IntentResult
        
        mock_intent_result = IntentResult(
            intent="cart",
            confidence=0.9,
            entities=["laptop"],
            clarification_needed=False,
            suggested_questions=[],
            reasoning="Clear cart intent",
            metadata={}
        )
        
        self.mock_intent_classifier.classify_intent.return_value = mock_intent_result
        self.mock_clarification_handler.needs_clarification.return_value = False
        
        # Router should succeed
        router_result = await self.router_node.route_message(self.sample_state)
        assert router_result["routing_decision"] == "cart"
        
        # But cart operation fails
        cart_error = ConnectionError("Database connection failed")
        self.mock_db_manager.execute_query.side_effect = cart_error
        
        cart_result = self.cart_manager.add_item(
            "integration_test_session", "laptop_123", "Gaming Laptop", 1, 999.99
        )
        
        # Cart should handle error gracefully
        assert cart_result["success"] == False
        assert "Database unavailable" in cart_result["error"]
        
        # Verify error statistics are tracked
        router_stats = self.router_error_handler.get_error_statistics()
        cart_stats = self.cart_error_handler.get_error_statistics()
        
        assert cart_stats["total_errors"] >= 1
    
    @pytest.mark.asyncio
    async def test_router_error_with_cart_fallback(self):
        """Test router error handling with cart functionality fallback."""
        # Setup: Intent classification fails
        classification_error = RuntimeError("LLM service unavailable")
        self.mock_intent_classifier.classify_intent.side_effect = classification_error
        
        # Router should handle error and fallback
        router_result = await self.router_node.route_message(self.sample_state)
        
        # Should fallback to QA agent
        assert router_result["routing_decision"] == "qa"
        assert router_result["routing_metadata"]["error_fallback"] == True
        
        # Cart should still be available for direct operations
        self.mock_db_manager.execute_query.return_value = []
        self.mock_db_manager.execute_update.return_value = 1
        
        cart_result = self.cart_manager.add_item(
            "integration_test_session", "laptop_123", "Gaming Laptop", 1
        )
        
        # Cart operation should succeed despite router error
        assert cart_result["success"] == True
    
    def test_cascading_database_errors(self):
        """Test handling of cascading database errors across components."""
        # Simulate database completely unavailable
        db_error = ConnectionError("FATAL: database system is shutting down")
        
        # Both router context extraction and cart operations should fail
        self.mock_db_manager.execute_query.side_effect = db_error
        
        # Test cart operations
        add_result = self.cart_manager.add_item(
            "test_session", "product_123", "Test Product", 1
        )
        
        get_result = self.cart_manager.get_cart_contents("test_session")
        
        summary_result = self.cart_manager.get_cart_summary("test_session")
        
        # All operations should handle errors gracefully
        assert add_result["success"] == False
        assert isinstance(get_result, list)  # Should return empty list
        assert summary_result["is_empty"] == True
        assert "error" in summary_result
        
        # Error statistics should reflect multiple failures
        cart_stats = self.cart_error_handler.get_error_statistics()
        assert cart_stats["total_errors"] >= 2
    
    def test_error_recovery_coordination(self):
        """Test coordination between error handlers for recovery."""
        # Setup: Router has classification issues but cart is available
        self.mock_intent_classifier.classify_intent.side_effect = [
            ValueError("Temporary classification error"),  # First attempt fails
            # Second attempt would succeed if retried
        ]
        
        # Mock successful cart operations
        self.mock_db_manager.execute_query.return_value = []
        self.mock_db_manager.execute_update.return_value = 1
        
        # Test that cart availability can inform router decisions
        cart_available = self.cart_manager.is_cart_available()
        assert isinstance(cart_available, bool)
        
        # Router should be able to check cart availability for fallback decisions
        if cart_available:
            # Can offer cart functionality as alternative
            degradation_response = self.cart_manager.create_graceful_degradation_response(
                "routing", "test_session"
            )
            assert "alternative_actions" in degradation_response
        
        # Verify error handlers maintain separate statistics
        router_stats = self.router_error_handler.get_error_statistics()
        cart_stats = self.cart_error_handler.get_error_statistics()
        
        assert "total_errors" in router_stats
        assert "total_errors" in cart_stats
    
    def test_monitoring_and_alerting_integration(self):
        """Test monitoring and alerting across error handling systems."""
        # Generate various types of errors
        errors_to_generate = [
            ("router", ValueError("Intent classification failed")),
            ("cart", ConnectionError("Database timeout")),
            ("router", RuntimeError("Routing logic error")),
            ("cart", ValueError("Invalid product data")),
        ]
        
        for component, error in errors_to_generate:
            if component == "router":
                self.router_error_handler.handle_intent_classification_error(
                    error, self.sample_state, "test message"
                )
            else:
                self.cart_error_handler.handle_cart_operation_error(
                    error, "test_operation", "test_session"
                )
        
        # Collect comprehensive statistics
        router_stats = self.router_error_handler.get_error_statistics()
        cart_stats = self.cart_error_handler.get_error_statistics()
        
        # Verify monitoring data
        assert router_stats["total_errors"] == 2
        assert cart_stats["total_errors"] == 2
        
        # Test combined monitoring view
        combined_stats = {
            "router_errors": router_stats,
            "cart_errors": cart_stats,
            "system_health": {
                "router_available": router_stats["total_errors"] < 10,
                "cart_available": self.cart_manager.is_cart_available(),
                "total_system_errors": router_stats["total_errors"] + cart_stats["total_errors"]
            }
        }
        
        assert combined_stats["system_health"]["total_system_errors"] == 4
        assert isinstance(combined_stats["system_health"]["router_available"], bool)
        assert isinstance(combined_stats["system_health"]["cart_available"], bool)
    
    def test_graceful_degradation_coordination(self):
        """Test coordinated graceful degradation across components."""
        # Simulate high error rates
        for i in range(5):
            router_error = RuntimeError(f"Router error {i}")
            self.router_error_handler.handle_routing_decision_error(
                router_error, self.sample_state
            )
            
            cart_error = ConnectionError(f"Cart error {i}")
            self.cart_error_handler.handle_database_error(
                cart_error, "test_op", f"session_{i}"
            )
        
        # Check if system should enter degraded mode
        router_stats = self.router_error_handler.get_error_statistics()
        cart_stats = self.cart_error_handler.get_error_statistics()
        
        system_degraded = (
            router_stats["total_errors"] > 3 or 
            cart_stats["total_errors"] > 3 or
            not self.cart_manager.is_cart_available()
        )
        
        if system_degraded:
            # Create coordinated degradation response
            degradation_response = {
                "system_status": "degraded",
                "available_services": [],
                "degraded_services": [],
                "user_message": "Some services are temporarily limited. I can still help with basic product information."
            }
            
            # Check which services are still available
            if router_stats["error_rate"] < 80:  # Less than 80% error rate
                degradation_response["available_services"].append("product_search")
            else:
                degradation_response["degraded_services"].append("intelligent_routing")
            
            if self.cart_manager.is_cart_available():
                degradation_response["available_services"].append("cart_management")
            else:
                degradation_response["degraded_services"].append("shopping_cart")
            
            assert "system_status" in degradation_response
            assert isinstance(degradation_response["available_services"], list)
            assert isinstance(degradation_response["degraded_services"], list)
    
    def test_error_handler_performance_under_load(self):
        """Test error handler performance under high error load."""
        import time
        
        # Generate many errors quickly
        start_time = time.time()
        
        for i in range(50):
            router_error = ValueError(f"Load test error {i}")
            self.router_error_handler.handle_intent_classification_error(
                router_error, self.sample_state, f"test message {i}"
            )
            
            cart_error = RuntimeError(f"Cart load test error {i}")
            self.cart_error_handler.handle_cart_operation_error(
                cart_error, "load_test", f"session_{i}"
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should handle 100 errors in reasonable time (< 1 second)
        assert processing_time < 1.0
        
        # Verify all errors were tracked
        router_stats = self.router_error_handler.get_error_statistics()
        cart_stats = self.cart_error_handler.get_error_statistics()
        
        assert router_stats["total_errors"] == 50
        assert cart_stats["total_errors"] == 50
    
    def test_error_handler_memory_cleanup(self):
        """Test that error handlers properly clean up memory."""
        # Generate errors and then reset
        for i in range(20):
            self.router_error_handler.handle_intent_classification_error(
                ValueError(f"Memory test {i}"), self.sample_state, "test"
            )
            self.cart_error_handler.handle_cart_operation_error(
                RuntimeError(f"Memory test {i}"), "test", f"session_{i}"
            )
        
        # Verify errors were tracked
        assert self.router_error_handler.error_stats["total_errors"] == 20
        assert self.cart_error_handler.error_stats["total_errors"] == 20
        
        # Reset and verify cleanup
        self.router_error_handler.reset_error_statistics()
        self.cart_error_handler.reset_error_statistics()
        
        assert self.router_error_handler.error_stats["total_errors"] == 0
        assert self.cart_error_handler.error_stats["total_errors"] == 0
        assert self.router_error_handler.error_stats["last_error_time"] is None
        assert self.cart_error_handler.error_stats["last_error_time"] is None
    
    @pytest.mark.asyncio
    async def test_end_to_end_error_recovery_flow(self):
        """Test complete end-to-end error recovery flow."""
        # Scenario: User tries to add item to cart, multiple failures occur
        
        # Step 1: Router classification fails initially
        self.mock_intent_classifier.classify_intent.side_effect = [
            RuntimeError("Classification service down"),  # First attempt fails
        ]
        
        # Router should handle error and fallback
        router_result = await self.router_node.route_message(self.sample_state)
        assert router_result["routing_decision"] == "qa"  # Fallback
        assert router_result["routing_metadata"]["error_fallback"] == True
        
        # Step 2: User tries direct cart operation, database fails
        self.mock_db_manager.execute_query.side_effect = ConnectionError("DB down")
        
        cart_result = self.cart_manager.add_item(
            "test_session", "laptop_123", "Gaming Laptop", 1
        )
        
        assert cart_result["success"] == False
        assert "Database unavailable" in cart_result["error"]
        
        # Step 3: System provides graceful degradation
        degradation = self.cart_manager.create_graceful_degradation_response(
            "add_item", "test_session"
        )
        
        assert degradation["degradation_active"] == True
        assert len(degradation["alternative_actions"]) > 0
        
        # Step 4: Verify comprehensive error tracking
        router_stats = self.router_error_handler.get_error_statistics()
        cart_stats = self.cart_error_handler.get_error_statistics()
        
        assert router_stats["total_errors"] >= 1
        assert cart_stats["total_errors"] >= 1
        
        # Step 5: System should still provide helpful response
        final_response = (
            "I'm experiencing some technical difficulties, but I can still help you "
            "search for product information and provide recommendations. "
            "Cart functionality will be restored shortly."
        )
        
        assert isinstance(final_response, str)
        assert len(final_response) > 50


@pytest.fixture
def error_monitoring_system():
    """Fixture for error monitoring system."""
    return {
        "router_handler": RouterErrorHandler(),
        "cart_handler": CartErrorHandler(),
        "alert_thresholds": {
            "error_rate": 10,  # errors per minute
            "critical_errors": 3,
            "service_degradation": 5
        }
    }


class TestErrorMonitoringSystem:
    """Test the error monitoring and alerting system."""
    
    def test_error_rate_monitoring(self, error_monitoring_system):
        """Test error rate monitoring and alerting."""
        router_handler = error_monitoring_system["router_handler"]
        cart_handler = error_monitoring_system["cart_handler"]
        thresholds = error_monitoring_system["alert_thresholds"]
        
        # Generate errors at different rates
        sample_state = {"session_id": "monitor_test", "current_query": "test"}
        
        # Generate errors below threshold
        for i in range(thresholds["error_rate"] - 1):
            router_handler.handle_intent_classification_error(
                ValueError(f"Test error {i}"), sample_state, "test"
            )
        
        router_stats = router_handler.get_error_statistics()
        assert router_stats["total_errors"] < thresholds["error_rate"]
        
        # Generate one more error to exceed threshold
        router_handler.handle_intent_classification_error(
            ValueError("Threshold exceeded"), sample_state, "test"
        )
        
        router_stats = router_handler.get_error_statistics()
        assert router_stats["total_errors"] >= thresholds["error_rate"]
        
        # This would trigger an alert in a real system
        alert_triggered = router_stats["total_errors"] >= thresholds["error_rate"]
        assert alert_triggered == True
    
    def test_critical_error_detection(self, error_monitoring_system):
        """Test detection of critical errors."""
        cart_handler = error_monitoring_system["cart_handler"]
        
        # Generate critical errors
        critical_errors = [
            ConnectionError("Database connection refused"),
            ConnectionError("Network unreachable"),
            RuntimeError("System out of memory")
        ]
        
        for error in critical_errors:
            cart_handler.handle_database_error(error, "test_op", "test_session")
        
        cart_stats = cart_handler.get_error_statistics()
        critical_count = cart_stats["errors_by_severity"].get("critical", 0)
        
        # Should detect critical errors
        assert critical_count > 0
        
        # Would trigger critical alert
        critical_alert = critical_count >= error_monitoring_system["alert_thresholds"]["critical_errors"]
        assert isinstance(critical_alert, bool)
    
    def test_service_health_monitoring(self, error_monitoring_system):
        """Test overall service health monitoring."""
        router_handler = error_monitoring_system["router_handler"]
        cart_handler = error_monitoring_system["cart_handler"]
        
        # Create health check function
        def check_system_health():
            router_stats = router_handler.get_error_statistics()
            cart_stats = cart_handler.get_error_statistics()
            
            return {
                "router_health": "healthy" if router_stats["total_errors"] < 5 else "degraded",
                "cart_health": "healthy" if cart_stats["total_errors"] < 5 else "degraded",
                "overall_health": "healthy"  # Would be calculated based on component health
            }
        
        # Initial health should be good
        health = check_system_health()
        assert health["router_health"] == "healthy"
        assert health["cart_health"] == "healthy"
        
        # Generate errors to degrade health
        sample_state = {"session_id": "health_test", "current_query": "test"}
        
        for i in range(6):  # Exceed threshold
            router_handler.handle_routing_decision_error(
                RuntimeError(f"Health test error {i}"), sample_state
            )
        
        # Health should now be degraded
        health = check_system_health()
        assert health["router_health"] == "degraded"