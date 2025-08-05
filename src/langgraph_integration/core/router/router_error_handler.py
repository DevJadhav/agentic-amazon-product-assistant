"""
Router-specific error handling system for intent classification and routing failures.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum

from ..state_schemas import AgentState
from .intent_classifier import IntentResult

logger = logging.getLogger(__name__)


class RouterErrorType(Enum):
    """Types of router-specific errors."""
    
    INTENT_CLASSIFICATION_ERROR = "intent_classification_error"
    ROUTING_DECISION_ERROR = "routing_decision_error"
    CLARIFICATION_ERROR = "clarification_error"
    AGENT_UNAVAILABLE_ERROR = "agent_unavailable_error"
    CONTEXT_EXTRACTION_ERROR = "context_extraction_error"
    STATE_UPDATE_ERROR = "state_update_error"


class RouterErrorSeverity(Enum):
    """Severity levels for router errors."""
    
    LOW = "low"          # Can continue with fallback routing
    MEDIUM = "medium"    # Requires fallback to default agent
    HIGH = "high"        # Should terminate with error message
    CRITICAL = "critical"  # System-wide routing issue


class RouterErrorHandler:
    """Handles errors in router node operations with recovery strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize router error handler."""
        self.config = config or {}
        self.max_retries = self.config.get("max_retries", 2)
        self.retry_delay = self.config.get("retry_delay", 0.5)
        self.default_agent = self.config.get("default_agent", "qa_agent")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "fallback_routes": 0,
            "last_error_time": None
        }
        
        # Fallback responses by error type
        self.fallback_responses = {
            RouterErrorType.INTENT_CLASSIFICATION_ERROR: 
                "I'm having trouble understanding your request. Let me help you search for products.",
            RouterErrorType.ROUTING_DECISION_ERROR: 
                "I encountered an issue with routing. I'll help you with product information instead.",
            RouterErrorType.CLARIFICATION_ERROR: 
                "I'm having trouble with clarification. Let me try to help you directly.",
            RouterErrorType.AGENT_UNAVAILABLE_ERROR: 
                "The requested service is temporarily unavailable. I'll help you with general product search.",
            RouterErrorType.CONTEXT_EXTRACTION_ERROR: 
                "I couldn't process the conversation context. Starting fresh with your current request.",
            RouterErrorType.STATE_UPDATE_ERROR: 
                "I encountered a technical issue but can still help you. What are you looking for?"
        }
    
    def handle_intent_classification_error(
        self, 
        error: Exception, 
        state: AgentState, 
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Handle intent classification failures."""
        
        error_type = RouterErrorType.INTENT_CLASSIFICATION_ERROR
        severity = self._assess_error_severity(error, error_type)
        
        self._log_router_error(error, error_type, severity, state, {
            "message": message,
            "context": context,
            "operation": "intent_classification"
        })
        
        self._update_error_stats(error_type, severity)
        
        # Attempt recovery based on severity
        if severity == RouterErrorSeverity.LOW:
            return self._attempt_classification_recovery(error, state, message, context)
        elif severity == RouterErrorSeverity.MEDIUM:
            return self._attempt_classification_recovery(error, state, message, context)
        else:
            return self._create_fallback_routing_state(error, error_type, state)
    
    def handle_routing_decision_error(
        self, 
        error: Exception, 
        state: AgentState,
        intent_result: Optional[IntentResult] = None
    ) -> AgentState:
        """Handle routing decision failures."""
        
        error_type = RouterErrorType.ROUTING_DECISION_ERROR
        severity = self._assess_error_severity(error, error_type)
        
        self._log_router_error(error, error_type, severity, state, {
            "intent_result": intent_result.intent if intent_result else None,
            "confidence": intent_result.confidence if intent_result else None,
            "operation": "routing_decision"
        })
        
        self._update_error_stats(error_type, severity)
        
        # Always fallback to default agent for routing errors
        return self._create_fallback_routing_state(error, error_type, state)
    
    def handle_clarification_error(
        self, 
        error: Exception, 
        state: AgentState,
        clarification_context: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Handle clarification system failures."""
        
        error_type = RouterErrorType.CLARIFICATION_ERROR
        severity = self._assess_error_severity(error, error_type)
        
        self._log_router_error(error, error_type, severity, state, {
            "clarification_context": clarification_context,
            "operation": "clarification_handling"
        })
        
        self._update_error_stats(error_type, severity)
        
        # For clarification errors, route to default agent
        return self._create_fallback_routing_state(error, error_type, state)
    
    def handle_agent_unavailable_error(
        self, 
        error: Exception, 
        state: AgentState,
        target_agent: str
    ) -> AgentState:
        """Handle cases where target agent is unavailable."""
        
        error_type = RouterErrorType.AGENT_UNAVAILABLE_ERROR
        severity = RouterErrorSeverity.MEDIUM  # Always medium for agent unavailability
        
        self._log_router_error(error, error_type, severity, state, {
            "target_agent": target_agent,
            "operation": "agent_routing"
        })
        
        self._update_error_stats(error_type, severity)
        
        # Route to alternative agent
        return self._route_to_alternative_agent(error, state, target_agent)
    
    def handle_context_extraction_error(
        self, 
        error: Exception, 
        state: AgentState
    ) -> AgentState:
        """Handle context extraction failures."""
        
        error_type = RouterErrorType.CONTEXT_EXTRACTION_ERROR
        severity = RouterErrorSeverity.LOW  # Can continue without context
        
        self._log_router_error(error, error_type, severity, state, {
            "operation": "context_extraction"
        })
        
        self._update_error_stats(error_type, severity)
        
        # Continue with empty context
        updated_state = state.copy()
        updated_state["routing_metadata"] = {
            "context_extraction_failed": True,
            "error_message": str(error),
            "fallback_applied": True
        }
        
        return updated_state
    
    def handle_state_update_error(
        self, 
        error: Exception, 
        state: AgentState,
        update_operation: str
    ) -> AgentState:
        """Handle state update failures."""
        
        error_type = RouterErrorType.STATE_UPDATE_ERROR
        severity = self._assess_error_severity(error, error_type)
        
        self._log_router_error(error, error_type, severity, state, {
            "update_operation": update_operation,
            "operation": "state_update"
        })
        
        self._update_error_stats(error_type, severity)
        
        if severity == RouterErrorSeverity.LOW:
            # Continue with minimal state updates
            return self._create_minimal_state_update(state, error)
        else:
            return self._create_fallback_routing_state(error, error_type, state)
    
    def create_graceful_degradation_response(
        self, 
        error: Exception, 
        state: AgentState,
        operation: str
    ) -> str:
        """Create graceful degradation response for router failures."""
        
        session_id = state.get("session_id", "unknown")
        current_query = state.get("current_query", "")
        
        # Base response
        base_response = "I encountered a technical issue with request routing, but I can still help you. "
        
        # Add context-specific guidance
        if "cart" in current_query.lower() or "add" in current_query.lower():
            base_response += "If you're looking to manage your shopping cart, please try rephrasing your request with clear cart actions like 'add to cart' or 'show my cart'."
        elif any(word in current_query.lower() for word in ["what", "how", "compare", "review"]):
            base_response += "I'll help you search for product information. What would you like to know?"
        else:
            base_response += "What can I help you find today?"
        
        self.logger.info(f"Created graceful degradation response for session {session_id}")
        
        return base_response
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get router error handling statistics."""
        
        total_routes = self.error_stats["total_errors"] + self.error_stats.get("successful_routes", 0)
        
        return {
            **self.error_stats,
            "error_types": [e.value for e in RouterErrorType],
            "severity_levels": [s.value for s in RouterErrorSeverity],
            "total_routes": total_routes,
            "error_rate": (
                self.error_stats["total_errors"] / max(1, total_routes)
            ) * 100,
            "fallback_rate": (
                self.error_stats["fallback_routes"] / max(1, total_routes)
            ) * 100
        }
    
    def reset_error_statistics(self) -> None:
        """Reset router error statistics."""
        
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "fallback_routes": 0,
            "last_error_time": None
        }
    
    # Private helper methods
    
    def _assess_error_severity(self, error: Exception, error_type: RouterErrorType) -> RouterErrorSeverity:
        """Assess the severity of a router error."""
        
        error_str = str(error).lower()
        
        # Critical errors that affect system-wide routing
        if any(keyword in error_str for keyword in ["database", "connection refused", "network unreachable"]):
            return RouterErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in [RouterErrorType.STATE_UPDATE_ERROR, RouterErrorType.ROUTING_DECISION_ERROR]:
            return RouterErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in [RouterErrorType.INTENT_CLASSIFICATION_ERROR, RouterErrorType.CLARIFICATION_ERROR]:
            return RouterErrorSeverity.MEDIUM
        
        # Low severity errors (including context extraction)
        if error_type == RouterErrorType.CONTEXT_EXTRACTION_ERROR:
            return RouterErrorSeverity.LOW
        
        return RouterErrorSeverity.LOW
    
    def _log_router_error(
        self, 
        error: Exception, 
        error_type: RouterErrorType, 
        severity: RouterErrorSeverity,
        state: AgentState,
        context: Dict[str, Any]
    ) -> None:
        """Log router error with context."""
        
        log_data = {
            "error_type": error_type.value,
            "error_severity": severity.value,
            "session_id": state.get("session_id", "unknown"),
            "current_step": state.get("current_step", "router"),
            "conversation_turn": state.get("conversation_turn", 0),
            "error_message": str(error),
            "error_class": type(error).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "router_context": context
        }
        
        # Log at appropriate level
        if severity == RouterErrorSeverity.CRITICAL:
            self.logger.critical("Critical router error", extra=log_data)
        elif severity == RouterErrorSeverity.HIGH:
            self.logger.error("High severity router error", extra=log_data)
        elif severity == RouterErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity router error", extra=log_data)
        else:
            self.logger.info("Low severity router error", extra=log_data)
    
    def _update_error_stats(self, error_type: RouterErrorType, severity: RouterErrorSeverity) -> None:
        """Update router error statistics."""
        
        self.error_stats["total_errors"] += 1
        self.error_stats["last_error_time"] = datetime.now(timezone.utc).isoformat()
        
        # Update by type
        type_key = error_type.value
        self.error_stats["errors_by_type"][type_key] = self.error_stats["errors_by_type"].get(type_key, 0) + 1
        
        # Update by severity
        severity_key = severity.value
        self.error_stats["errors_by_severity"][severity_key] = self.error_stats["errors_by_severity"].get(severity_key, 0) + 1
    
    def _attempt_classification_recovery(
        self, 
        error: Exception, 
        state: AgentState, 
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> AgentState:
        """Attempt to recover from intent classification error."""
        
        # Simple keyword-based fallback classification
        message_lower = message.lower()
        
        # Check for obvious cart keywords
        cart_keywords = ["cart", "add", "buy", "purchase", "remove", "basket"]
        if any(keyword in message_lower for keyword in cart_keywords):
            return self._create_recovery_routing_state(state, "cart", 0.6, "keyword_fallback")
        
        # Check for obvious QA keywords
        qa_keywords = ["what", "how", "compare", "review", "tell me", "explain"]
        if any(keyword in message_lower for keyword in qa_keywords):
            return self._create_recovery_routing_state(state, "qa", 0.6, "keyword_fallback")
        
        # Default to QA agent
        return self._create_recovery_routing_state(state, "qa", 0.5, "default_fallback")
    
    def _create_fallback_routing_state(
        self, 
        error: Exception, 
        error_type: RouterErrorType, 
        state: AgentState
    ) -> AgentState:
        """Create fallback routing state for errors."""
        
        updated_state = state.copy()
        
        # Route to default agent
        updated_state["routing_decision"] = "qa"  # Default to QA
        updated_state["target_agent"] = self.default_agent
        updated_state["user_intent"] = "qa"
        updated_state["intent_confidence"] = 0.5
        
        # Add error metadata
        updated_state["routing_metadata"] = {
            "error_fallback": True,
            "error_type": error_type.value,
            "error_message": str(error),
            "fallback_agent": self.default_agent,
            "fallback_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Track fallback statistics
        self.error_stats["fallback_routes"] += 1
        
        self.logger.info(f"Created fallback routing to {self.default_agent} for session {state.get('session_id')}")
        
        return updated_state
    
    def _route_to_alternative_agent(
        self, 
        error: Exception, 
        state: AgentState, 
        unavailable_agent: str
    ) -> AgentState:
        """Route to alternative agent when target is unavailable."""
        
        # Determine alternative agent
        if unavailable_agent == "shopping_cart_agent":
            alternative_agent = "qa_agent"
            alternative_intent = "qa"
            message = "The shopping cart service is temporarily unavailable. I'll help you with product information instead."
        elif unavailable_agent == "qa_agent":
            alternative_agent = "shopping_cart_agent"
            alternative_intent = "cart"
            message = "The product search service is temporarily unavailable. I can help you manage your shopping cart instead."
        else:
            alternative_agent = self.default_agent
            alternative_intent = "qa"
            message = "The requested service is unavailable. I'll help you with general product search."
        
        updated_state = state.copy()
        updated_state["routing_decision"] = alternative_intent
        updated_state["target_agent"] = alternative_agent
        updated_state["user_intent"] = alternative_intent
        updated_state["intent_confidence"] = 0.7
        
        # Add routing metadata
        updated_state["routing_metadata"] = {
            "agent_unavailable": True,
            "unavailable_agent": unavailable_agent,
            "alternative_agent": alternative_agent,
            "fallback_message": message,
            "fallback_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Routed to alternative agent {alternative_agent} due to {unavailable_agent} unavailability")
        
        return updated_state
    
    def _create_recovery_routing_state(
        self, 
        state: AgentState, 
        intent: str, 
        confidence: float,
        recovery_method: str
    ) -> AgentState:
        """Create routing state for successful error recovery."""
        
        updated_state = state.copy()
        
        # Set routing decision
        updated_state["routing_decision"] = intent
        updated_state["target_agent"] = "shopping_cart_agent" if intent == "cart" else "qa_agent"
        updated_state["user_intent"] = intent
        updated_state["intent_confidence"] = confidence
        
        # Add recovery metadata
        updated_state["routing_metadata"] = {
            "error_recovery": True,
            "recovery_method": recovery_method,
            "recovery_confidence": confidence,
            "recovery_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Successfully recovered routing with {recovery_method}: {intent}")
        
        return updated_state
    
    def _create_minimal_state_update(self, state: AgentState, error: Exception) -> AgentState:
        """Create minimal state update when full update fails."""
        
        updated_state = state.copy()
        
        # Add minimal routing metadata
        updated_state["routing_metadata"] = {
            "minimal_update": True,
            "state_update_error": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Ensure basic routing fields exist
        if "routing_decision" not in updated_state:
            updated_state["routing_decision"] = "qa"
        if "target_agent" not in updated_state:
            updated_state["target_agent"] = self.default_agent
        
        return updated_state