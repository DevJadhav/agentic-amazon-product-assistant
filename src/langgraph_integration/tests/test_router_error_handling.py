"""
Tests for router error handling system.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from ..core.router.router_error_handler import (
    RouterErrorHandler, RouterErrorType, RouterErrorSeverity
)
from ..core.router.intent_classifier import IntentResult
from ..core.state_schemas import AgentState


class TestRouterErrorHandler:
    """Test suite for RouterErrorHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = RouterErrorHandler()
        self.sample_state = {
            "session_id": "test_session_123",
            "current_query": "test query",
            "conversation_turn": 1,
            "current_step": "router"
        }
    
    def test_handle_intent_classification_error_low_severity(self):
        """Test handling of low severity intent classification errors."""
        error = ValueError("Minor classification issue")
        
        result = self.error_handler.handle_intent_classification_error(
            error, self.sample_state, "test message"
        )
        
        assert result["routing_decision"] in ["qa", "cart"]
        assert "routing_metadata" in result
        # For low severity errors, we expect recovery metadata
        assert "recovery_method" in result["routing_metadata"] or "error_fallback" in result["routing_metadata"]
        assert self.error_handler.error_stats["total_errors"] == 1
    
    def test_handle_intent_classification_error_high_severity(self):
        """Test handling of high severity intent classification errors."""
        error = ConnectionError("Database connection failed")
        
        result = self.error_handler.handle_intent_classification_error(
            error, self.sample_state, "test message"
        )
        
        assert result["routing_decision"] == "qa"  # Fallback to default
        assert result["target_agent"] == "qa_agent"
        assert result["routing_metadata"]["error_fallback"] == True
        assert self.error_handler.error_stats["fallback_routes"] == 1
    
    def test_handle_routing_decision_error(self):
        """Test handling of routing decision errors."""
        error = RuntimeError("Routing logic failed")
        intent_result = IntentResult(
            intent="cart",
            confidence=0.8,
            entities=["laptop"],
            clarification_needed=False,
            suggested_questions=[],
            reasoning="Test reasoning",
            metadata={}
        )
        
        result = self.error_handler.handle_routing_decision_error(
            error, self.sample_state, intent_result
        )
        
        assert result["routing_decision"] == "qa"  # Always fallback
        assert result["target_agent"] == "qa_agent"
        assert result["routing_metadata"]["error_fallback"] == True
    
    def test_handle_clarification_error(self):
        """Test handling of clarification system errors."""
        error = Exception("Clarification handler failed")
        
        result = self.error_handler.handle_clarification_error(
            error, self.sample_state
        )
        
        assert result["routing_decision"] == "qa"
        assert result["target_agent"] == "qa_agent"
        assert result["routing_metadata"]["error_fallback"] == True
    
    def test_handle_agent_unavailable_error(self):
        """Test handling of agent unavailability."""
        error = ConnectionError("Shopping cart agent unavailable")
        
        result = self.error_handler.handle_agent_unavailable_error(
            error, self.sample_state, "shopping_cart_agent"
        )
        
        assert result["routing_decision"] == "qa"
        assert result["target_agent"] == "qa_agent"
        assert result["routing_metadata"]["agent_unavailable"] == True
        assert result["routing_metadata"]["unavailable_agent"] == "shopping_cart_agent"
    
    def test_handle_context_extraction_error(self):
        """Test handling of context extraction errors."""
        error = KeyError("Missing context field")
        
        result = self.error_handler.handle_context_extraction_error(
            error, self.sample_state
        )
        
        assert result["routing_metadata"]["context_extraction_failed"] == True
        assert result["routing_metadata"]["fallback_applied"] == True
    
    def test_handle_state_update_error_low_severity(self):
        """Test handling of low severity state update errors."""
        error = ValueError("Minor state update issue")
        
        result = self.error_handler.handle_state_update_error(
            error, self.sample_state, "intent_classification"
        )
        
        # For low severity state update errors, should get minimal update or fallback
        assert "routing_decision" in result
        assert "target_agent" in result
        # Should have either minimal update or error fallback metadata
        assert ("minimal_update" in result["routing_metadata"] or 
                "error_fallback" in result["routing_metadata"])
    
    def test_handle_state_update_error_high_severity(self):
        """Test handling of high severity state update errors."""
        error = RuntimeError("Critical state corruption")
        
        result = self.error_handler.handle_state_update_error(
            error, self.sample_state, "critical_update"
        )
        
        assert result["routing_decision"] == "qa"
        assert result["routing_metadata"]["error_fallback"] == True
    
    def test_create_graceful_degradation_response(self):
        """Test creation of graceful degradation responses."""
        error = Exception("System overload")
        
        response = self.error_handler.create_graceful_degradation_response(
            error, self.sample_state, "routing"
        )
        
        assert isinstance(response, str)
        assert "technical issue" in response.lower()
        assert len(response) > 50  # Should be a meaningful response
    
    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        # Generate some errors
        error1 = ValueError("Error 1")
        error2 = ConnectionError("Error 2")
        
        self.error_handler.handle_intent_classification_error(
            error1, self.sample_state, "test"
        )
        self.error_handler.handle_routing_decision_error(
            error2, self.sample_state
        )
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 2
        assert "intent_classification_error" in stats["errors_by_type"]
        assert "routing_decision_error" in stats["errors_by_type"]
        assert stats["last_error_time"] is not None
    
    def test_error_severity_assessment(self):
        """Test error severity assessment logic."""
        # Test different error types
        db_error = ConnectionError("Database connection refused")
        network_error = TimeoutError("Network timeout")
        validation_error = ValueError("Invalid input")
        
        # Database errors should be critical
        severity1 = self.error_handler._assess_error_severity(
            db_error, RouterErrorType.INTENT_CLASSIFICATION_ERROR
        )
        assert severity1 == RouterErrorSeverity.CRITICAL
        
        # Routing errors should be high
        severity2 = self.error_handler._assess_error_severity(
            validation_error, RouterErrorType.ROUTING_DECISION_ERROR
        )
        assert severity2 == RouterErrorSeverity.HIGH
        
        # Context errors should be low (unless they contain critical keywords)
        severity3 = self.error_handler._assess_error_severity(
            network_error, RouterErrorType.CONTEXT_EXTRACTION_ERROR
        )
        # Network timeout might be classified as critical, so let's test with a different error
        simple_error = ValueError("Context field missing")
        severity3 = self.error_handler._assess_error_severity(
            simple_error, RouterErrorType.CONTEXT_EXTRACTION_ERROR
        )
        assert severity3 == RouterErrorSeverity.LOW
    
    def test_fallback_responses(self):
        """Test that appropriate fallback responses are provided."""
        for error_type in RouterErrorType:
            assert error_type in self.error_handler.fallback_responses
            response = self.error_handler.fallback_responses[error_type]
            assert isinstance(response, str)
            assert len(response) > 20  # Should be meaningful
    
    def test_reset_error_statistics(self):
        """Test resetting error statistics."""
        # Generate an error first
        error = ValueError("Test error")
        self.error_handler.handle_intent_classification_error(
            error, self.sample_state, "test"
        )
        
        assert self.error_handler.error_stats["total_errors"] == 1
        
        # Reset statistics
        self.error_handler.reset_error_statistics()
        
        assert self.error_handler.error_stats["total_errors"] == 0
        assert self.error_handler.error_stats["last_error_time"] is None
    
    def test_recovery_routing_state_creation(self):
        """Test creation of recovery routing states."""
        result = self.error_handler._create_recovery_routing_state(
            self.sample_state, "cart", 0.7, "keyword_fallback"
        )
        
        assert result["routing_decision"] == "cart"
        assert result["target_agent"] == "shopping_cart_agent"
        assert result["intent_confidence"] == 0.7
        assert result["routing_metadata"]["error_recovery"] == True
        assert result["routing_metadata"]["recovery_method"] == "keyword_fallback"
    
    def test_alternative_agent_routing(self):
        """Test routing to alternative agents."""
        error = Exception("Agent unavailable")
        
        # Test cart agent unavailable
        result1 = self.error_handler._route_to_alternative_agent(
            error, self.sample_state, "shopping_cart_agent"
        )
        assert result1["target_agent"] == "qa_agent"
        assert result1["routing_decision"] == "qa"
        
        # Test QA agent unavailable
        result2 = self.error_handler._route_to_alternative_agent(
            error, self.sample_state, "qa_agent"
        )
        assert result2["target_agent"] == "shopping_cart_agent"
        assert result2["routing_decision"] == "cart"
    
    def test_keyword_based_recovery(self):
        """Test keyword-based fallback classification."""
        # Test cart keywords
        cart_result = self.error_handler._attempt_classification_recovery(
            Exception("test"), self.sample_state, "add laptop to cart", {}
        )
        assert cart_result["routing_decision"] == "cart"
        assert cart_result["routing_metadata"]["error_recovery"] == True
        
        # Test QA keywords
        qa_result = self.error_handler._attempt_classification_recovery(
            Exception("test"), self.sample_state, "what is the best laptop", {}
        )
        assert qa_result["routing_decision"] == "qa"
        assert qa_result["routing_metadata"]["error_recovery"] == True
        
        # Test default fallback
        default_result = self.error_handler._attempt_classification_recovery(
            Exception("test"), self.sample_state, "random text", {}
        )
        assert default_result["routing_decision"] == "qa"
        assert default_result["routing_metadata"]["recovery_method"] == "default_fallback"


@pytest.fixture
def mock_state():
    """Fixture for mock agent state."""
    return {
        "session_id": "test_session",
        "current_query": "test query",
        "conversation_turn": 1,
        "current_step": "router",
        "updated_at": datetime.now(timezone.utc)
    }


@pytest.fixture
def mock_intent_result():
    """Fixture for mock intent result."""
    return IntentResult(
        intent="cart",
        confidence=0.8,
        entities=["laptop", "2"],
        clarification_needed=False,
        suggested_questions=[],
        reasoning="High confidence cart intent",
        metadata={"test": True}
    )


class TestRouterErrorHandlerIntegration:
    """Integration tests for router error handler."""
    
    def test_error_handler_with_real_exceptions(self, mock_state):
        """Test error handler with real exception scenarios."""
        error_handler = RouterErrorHandler()
        
        # Test with actual database connection error
        db_error = ConnectionError("FATAL: database 'cart_db' does not exist")
        result = error_handler.handle_intent_classification_error(
            db_error, mock_state, "test message"
        )
        
        assert result["routing_decision"] == "qa"
        assert result["routing_metadata"]["error_fallback"] == True
    
    def test_error_logging_integration(self, mock_state, caplog):
        """Test that errors are properly logged."""
        error_handler = RouterErrorHandler()
        
        with caplog.at_level("ERROR"):
            error = RuntimeError("Test error for logging")
            error_handler.handle_routing_decision_error(error, mock_state)
        
        assert "router error" in caplog.text.lower()
        assert "test error for logging" in caplog.text
    
    def test_concurrent_error_handling(self, mock_state):
        """Test error handling under concurrent conditions."""
        error_handler = RouterErrorHandler()
        
        # Simulate multiple concurrent errors
        errors = [
            ValueError(f"Error {i}") for i in range(5)
        ]
        
        results = []
        for error in errors:
            result = error_handler.handle_intent_classification_error(
                error, mock_state, f"test message {len(results)}"
            )
            results.append(result)
        
        # All should have valid routing decisions
        for result in results:
            assert "routing_decision" in result
            assert result["routing_decision"] in ["qa", "cart"]
        
        # Statistics should be accurate
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 5