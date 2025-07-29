"""
Error handling and recovery system for LangGraph agent workflows.
Provides graceful degradation and fallback mechanisms.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

from .state_schemas import AgentState, update_state_step
from .utils import create_error_response

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in agent workflows."""
    
    TOOL_ERROR = "tool_error"
    LLM_ERROR = "llm_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"          # Can continue with degraded functionality
    MEDIUM = "medium"    # Requires fallback mechanism
    HIGH = "high"        # Workflow should be terminated
    CRITICAL = "critical"  # System-wide issue


class AgentErrorHandler:
    """Handles errors in LangGraph agent workflows with recovery strategies."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize error handler."""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "last_error_time": None
        }
        
        # Fallback responses by error type
        self.fallback_responses = {
            ErrorType.TOOL_ERROR: "I'm having trouble accessing the product database right now. Let me try to help you with general information.",
            ErrorType.LLM_ERROR: "I'm experiencing some technical difficulties with my language processing. Please try rephrasing your question.",
            ErrorType.DATABASE_ERROR: "The product database is temporarily unavailable. I can provide general assistance based on my knowledge.",
            ErrorType.NETWORK_ERROR: "I'm having connectivity issues. Please check your connection and try again.",
            ErrorType.VALIDATION_ERROR: "There seems to be an issue with your request format. Please try rephrasing your question.",
            ErrorType.TIMEOUT_ERROR: "Your request is taking longer than expected. Please try a simpler query or try again later.",
            ErrorType.UNKNOWN_ERROR: "I encountered an unexpected issue. Please try again or contact support if the problem persists."
        }
    
    def handle_error(
        self, 
        error: Exception, 
        state: AgentState, 
        error_context: Dict[str, Any] = None
    ) -> AgentState:
        """Handle an error and attempt recovery."""
        
        # Classify error
        error_type = self._classify_error(error)
        error_severity = self._assess_severity(error, error_type)
        
        # Log error
        self._log_error(error, error_type, error_severity, state, error_context)
        
        # Update statistics
        self._update_error_stats(error_type, error_severity)
        
        # Attempt recovery based on error type and severity
        if error_severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            return self._attempt_recovery(error, error_type, state, error_context)
        else:
            return self._create_terminal_error_state(error, error_type, state)
    
    def handle_tool_error(self, error: Exception, state: AgentState, tool_name: str = None) -> AgentState:
        """Handle tool-specific errors with targeted recovery."""
        
        self.logger.error(f"Tool error in {tool_name}: {error}")
        
        # Try alternative approaches based on the tool
        if tool_name == "vector_search":
            return self._handle_vector_search_error(error, state)
        elif tool_name == "product_analysis":
            return self._handle_analysis_error(error, state)
        else:
            return self._create_fallback_state(error, ErrorType.TOOL_ERROR, state)
    
    def handle_llm_error(self, error: Exception, state: AgentState) -> AgentState:
        """Handle LLM provider errors with provider switching."""
        
        self.logger.error(f"LLM error: {error}")
        
        # Try switching to backup provider
        current_provider = state.get("llm_provider", "openai")
        backup_providers = self._get_backup_providers(current_provider)
        
        for backup_provider in backup_providers:
            try:
                # Update state with backup provider
                updated_state = update_state_step(
                    state,
                    "llm_fallback",
                    llm_provider=backup_provider,
                    retry_count=state.get("retry_count", 0) + 1
                )
                
                self.logger.info(f"Switched to backup LLM provider: {backup_provider}")
                return updated_state
                
            except Exception as backup_error:
                self.logger.warning(f"Backup provider {backup_provider} also failed: {backup_error}")
                continue
        
        # All providers failed, create fallback state
        return self._create_fallback_state(error, ErrorType.LLM_ERROR, state)
    
    def handle_state_persistence_error(self, error: Exception, session_id: str) -> None:
        """Handle state persistence errors."""
        
        self.logger.error(f"State persistence error for session {session_id}: {error}")
        
        # Try to save to backup storage or continue without persistence
        try:
            # Could implement backup storage here
            self.logger.warning(f"Continuing without state persistence for session {session_id}")
        except Exception as backup_error:
            self.logger.error(f"Backup persistence also failed: {backup_error}")
    
    def create_fallback_response(self, query: str, error: Exception, context: Dict[str, Any] = None) -> str:
        """Create a fallback response when all recovery attempts fail."""
        
        error_type = self._classify_error(error)
        base_response = self.fallback_responses.get(error_type, self.fallback_responses[ErrorType.UNKNOWN_ERROR])
        
        # Customize response based on query context
        if context and context.get("query_intent"):
            intent = context["query_intent"]
            
            if intent == "comparison":
                base_response += " I can still help you understand the differences between products based on general knowledge."
            elif intent == "recommendation":
                base_response += " I can provide general recommendations based on common preferences."
            elif intent == "reviews":
                base_response += " While I can't access current reviews, I can share general insights about product categories."
        
        return base_response
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        
        return {
            **self.error_stats,
            "error_types": list(ErrorType),
            "severity_levels": list(ErrorSeverity),
            "fallback_responses_available": len(self.fallback_responses)
        }
    
    def reset_error_statistics(self) -> None:
        """Reset error statistics."""
        
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "last_error_time": None
        }
    
    # Private helper methods
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error by type."""
        
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ["connection", "network", "timeout", "unreachable"]):
            return ErrorType.NETWORK_ERROR
        
        # Database-related errors
        if any(keyword in error_str for keyword in ["database", "postgres", "sql", "connection pool"]):
            return ErrorType.DATABASE_ERROR
        
        # LLM-related errors
        if any(keyword in error_str for keyword in ["openai", "groq", "api key", "rate limit", "model"]):
            return ErrorType.LLM_ERROR
        
        # Tool-related errors
        if any(keyword in error_str for keyword in ["tool", "vector", "search", "weaviate"]):
            return ErrorType.TOOL_ERROR
        
        # Validation errors
        if any(keyword in error_type_name for keyword in ["validation", "value", "type"]):
            return ErrorType.VALIDATION_ERROR
        
        # Timeout errors
        if any(keyword in error_str for keyword in ["timeout", "deadline", "expired"]):
            return ErrorType.TIMEOUT_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def _assess_severity(self, error: Exception, error_type: ErrorType) -> ErrorSeverity:
        """Assess error severity."""
        
        # Critical errors that affect system-wide functionality
        if error_type == ErrorType.DATABASE_ERROR:
            return ErrorSeverity.HIGH
        
        # High severity errors that should terminate workflow
        if error_type in [ErrorType.VALIDATION_ERROR, ErrorType.UNKNOWN_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors that require fallback
        if error_type in [ErrorType.LLM_ERROR, ErrorType.TOOL_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors that can be handled gracefully
        if error_type in [ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT_ERROR]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _log_error(
        self, 
        error: Exception, 
        error_type: ErrorType, 
        error_severity: ErrorSeverity, 
        state: AgentState,
        error_context: Dict[str, Any] = None
    ) -> None:
        """Log error with context."""
        
        log_data = {
            "error_type": error_type.value,
            "error_severity": error_severity.value,
            "session_id": state.get("session_id", "unknown"),
            "current_step": state.get("current_step", "unknown"),
            "conversation_turn": state.get("conversation_turn", 0),
            "error_message": str(error),
            "error_class": type(error).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error_context:
            log_data["context"] = error_context
        
        # Log at appropriate level based on severity
        if error_severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical agent error", extra=log_data)
        elif error_severity == ErrorSeverity.HIGH:
            self.logger.error("High severity agent error", extra=log_data)
        elif error_severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity agent error", extra=log_data)
        else:
            self.logger.info("Low severity agent error", extra=log_data)
    
    def _update_error_stats(self, error_type: ErrorType, error_severity: ErrorSeverity) -> None:
        """Update error statistics."""
        
        self.error_stats["total_errors"] += 1
        self.error_stats["last_error_time"] = datetime.utcnow().isoformat()
        
        # Update by type
        type_key = error_type.value
        self.error_stats["errors_by_type"][type_key] = self.error_stats["errors_by_type"].get(type_key, 0) + 1
        
        # Update by severity
        severity_key = error_severity.value
        self.error_stats["errors_by_severity"][severity_key] = self.error_stats["errors_by_severity"].get(severity_key, 0) + 1
    
    def _attempt_recovery(
        self, 
        error: Exception, 
        error_type: ErrorType, 
        state: AgentState,
        error_context: Dict[str, Any] = None
    ) -> AgentState:
        """Attempt to recover from error."""
        
        retry_count = state.get("retry_count", 0)
        
        if retry_count < self.max_retries:
            # Implement retry with exponential backoff
            delay = self.retry_delay * (2 ** retry_count)
            time.sleep(delay)
            
            # Update state for retry
            updated_state = update_state_step(
                state,
                "error_recovery",
                retry_count=retry_count + 1,
                error_state=f"Retrying after {error_type.value}: {str(error)}"
            )
            
            self.logger.info(f"Attempting recovery (retry {retry_count + 1}/{self.max_retries})")
            return updated_state
        else:
            # Max retries reached, create fallback state
            return self._create_fallback_state(error, error_type, state)
    
    def _create_fallback_state(self, error: Exception, error_type: ErrorType, state: AgentState) -> AgentState:
        """Create fallback state when recovery fails."""
        
        fallback_response = self.create_fallback_response(
            state.get("current_query", ""),
            error,
            {"query_intent": state.get("query_intent")}
        )
        
        return update_state_step(
            state,
            "error_fallback",
            final_response=fallback_response,
            workflow_status="completed",
            error_state=f"{error_type.value}: {str(error)}"
        )
    
    def _create_terminal_error_state(self, error: Exception, error_type: ErrorType, state: AgentState) -> AgentState:
        """Create terminal error state for critical errors."""
        
        error_response = f"I'm sorry, but I encountered a critical error and cannot process your request: {str(error)}"
        
        return update_state_step(
            state,
            "terminal_error",
            final_response=error_response,
            workflow_status="error",
            error_state=f"CRITICAL {error_type.value}: {str(error)}"
        )
    
    def _handle_vector_search_error(self, error: Exception, state: AgentState) -> AgentState:
        """Handle vector search specific errors."""
        
        # Try using cached results or simplified search
        fallback_response = "I'm having trouble accessing the product database. Let me provide some general recommendations based on your query."
        
        return update_state_step(
            state,
            "vector_search_fallback",
            selected_products=[],
            review_summaries=[],
            final_response=fallback_response,
            error_state=f"Vector search failed: {str(error)}"
        )
    
    def _handle_analysis_error(self, error: Exception, state: AgentState) -> AgentState:
        """Handle product analysis specific errors."""
        
        # Continue without analysis
        products = state.get("selected_products", [])
        
        if products:
            fallback_response = f"I found {len(products)} products for your query, but I'm having trouble with detailed analysis right now. Here are the basic results."
        else:
            fallback_response = "I'm having trouble with product analysis right now. Please try a simpler query."
        
        return update_state_step(
            state,
            "analysis_fallback",
            final_response=fallback_response,
            error_state=f"Product analysis failed: {str(error)}"
        )
    
    def _get_backup_providers(self, current_provider: str) -> List[str]:
        """Get backup LLM providers."""
        
        all_providers = ["openai", "groq", "google", "ollama"]
        
        # Remove current provider and return others
        backup_providers = [p for p in all_providers if p != current_provider]
        
        # Prioritize based on reliability
        priority_order = ["openai", "groq", "google", "ollama"]
        
        return sorted(backup_providers, key=lambda x: priority_order.index(x) if x in priority_order else 999)