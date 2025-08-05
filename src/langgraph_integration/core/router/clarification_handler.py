"""
Clarification handling system for ambiguous user intents.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from .intent_classifier import IntentResult

logger = logging.getLogger(__name__)


@dataclass
class ClarificationAttempt:
    """Record of a clarification attempt."""
    
    timestamp: datetime
    original_message: str
    intent_result: IntentResult
    questions_asked: List[str]
    user_response: Optional[str] = None
    resolved: bool = False


class ClarificationHandler:
    """Handles clarification requests for ambiguous user intents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize clarification handler with configuration."""
        self.config = config or {}
        self.max_clarification_attempts = self.config.get("max_clarification_attempts", 3)
        self.clarification_timeout = self.config.get("clarification_timeout", 300)  # 5 minutes
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track clarification attempts per session
        self._clarification_history: Dict[str, List[ClarificationAttempt]] = {}
    
    def needs_clarification(self, intent_result: IntentResult, session_id: str) -> bool:
        """
        Determine if clarification is needed based on intent result and history.
        
        Args:
            intent_result: Result from intent classification
            session_id: Session identifier for tracking attempts
            
        Returns:
            True if clarification is needed, False otherwise
        """
        # Check if intent result indicates clarification is needed
        if not intent_result.clarification_needed:
            return False
        
        # Check if we've exceeded max attempts for this session
        attempts = self._get_clarification_attempts(session_id)
        unresolved_attempts = [attempt for attempt in attempts if not attempt.resolved]
        
        if len(unresolved_attempts) >= self.max_clarification_attempts:
            self.logger.warning(f"Max clarification attempts ({self.max_clarification_attempts}) reached for session {session_id}")
            return False
        
        # Check if recent attempts are still unresolved
        recent_unresolved = [
            attempt for attempt in unresolved_attempts 
            if self._is_recent_attempt(attempt)
        ]
        
        if recent_unresolved:
            self.logger.info(f"Recent unresolved clarification exists for session {session_id}")
            return False
        
        return True
    
    def create_clarification_request(self, intent_result: IntentResult, session_id: str, 
                                   original_message: str) -> Dict[str, Any]:
        """
        Create a clarification request for the user.
        
        Args:
            intent_result: Result from intent classification
            session_id: Session identifier
            original_message: Original user message that needs clarification
            
        Returns:
            Dictionary containing clarification request details
        """
        # Record this clarification attempt
        attempt = ClarificationAttempt(
            timestamp=datetime.now(timezone.utc),
            original_message=original_message,
            intent_result=intent_result,
            questions_asked=intent_result.suggested_questions.copy()
        )
        
        self._add_clarification_attempt(session_id, attempt)
        
        # Generate clarification message
        clarification_message = self._generate_clarification_message(intent_result, original_message)
        
        # Create response structure
        clarification_request = {
            "type": "clarification_request",
            "message": clarification_message,
            "questions": intent_result.suggested_questions,
            "original_message": original_message,
            "attempt_number": len(self._get_clarification_attempts(session_id)),
            "max_attempts": self.max_clarification_attempts,
            "session_id": session_id,
            "timestamp": attempt.timestamp.isoformat(),
            "metadata": {
                "intent_confidence": intent_result.confidence,
                "entities_found": intent_result.entities,
                "reasoning": intent_result.reasoning
            }
        }
        
        self.logger.info(f"Created clarification request for session {session_id}, attempt {clarification_request['attempt_number']}")
        
        return clarification_request
    
    def process_clarification_response(self, response: str, session_id: str) -> Optional[IntentResult]:
        """
        Process user's response to clarification request.
        
        Args:
            response: User's response to clarification
            session_id: Session identifier
            
        Returns:
            IntentResult if clarification was successful, None otherwise
        """
        attempts = self._get_clarification_attempts(session_id)
        if not attempts:
            self.logger.warning(f"No clarification attempts found for session {session_id}")
            return None
        
        # Get the most recent unresolved attempt
        latest_attempt = None
        for attempt in reversed(attempts):
            if not attempt.resolved and self._is_recent_attempt(attempt):
                latest_attempt = attempt
                break
        
        if not latest_attempt:
            self.logger.warning(f"No recent unresolved clarification attempt found for session {session_id}")
            return None
        
        # Process the response
        resolved_intent = self._resolve_clarification(response, latest_attempt)
        
        if resolved_intent:
            # Mark attempt as resolved
            latest_attempt.user_response = response
            latest_attempt.resolved = True
            
            self.logger.info(f"Clarification resolved for session {session_id}: {resolved_intent.intent}")
            return resolved_intent
        else:
            self.logger.info(f"Clarification not resolved for session {session_id}")
            return None
    
    def create_fallback_response(self, session_id: str) -> Dict[str, Any]:
        """
        Create fallback response when clarification attempts are exhausted.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary containing fallback response
        """
        attempts = self._get_clarification_attempts(session_id)
        attempt_count = len(attempts)
        
        fallback_messages = [
            "I'm having trouble understanding what you'd like me to help you with. Let me route you to our general product search to get started.",
            "Since I'm not sure about your specific needs, I'll help you search for products. You can always ask me to add items to your cart later.",
            "I'll default to helping you search for product information. If you need cart management, just let me know!"
        ]
        
        # Choose message based on attempt count
        message_index = min(attempt_count - 1, len(fallback_messages) - 1)
        fallback_message = fallback_messages[message_index]
        
        fallback_response = {
            "type": "fallback_response",
            "message": fallback_message,
            "default_intent": "qa",  # Default to QA agent
            "attempt_count": attempt_count,
            "max_attempts_reached": attempt_count >= self.max_clarification_attempts,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Created fallback response for session {session_id} after {attempt_count} attempts")
        
        return fallback_response
    
    def get_clarification_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get clarification history for a session."""
        attempts = self._get_clarification_attempts(session_id)
        
        return [
            {
                "timestamp": attempt.timestamp.isoformat(),
                "original_message": attempt.original_message,
                "questions_asked": attempt.questions_asked,
                "user_response": attempt.user_response,
                "resolved": attempt.resolved,
                "intent_confidence": attempt.intent_result.confidence,
                "entities": attempt.intent_result.entities
            }
            for attempt in attempts
        ]
    
    def clear_session_history(self, session_id: str) -> None:
        """Clear clarification history for a session."""
        if session_id in self._clarification_history:
            del self._clarification_history[session_id]
            self.logger.info(f"Cleared clarification history for session {session_id}")
    
    def _get_clarification_attempts(self, session_id: str) -> List[ClarificationAttempt]:
        """Get clarification attempts for a session."""
        return self._clarification_history.get(session_id, [])
    
    def _add_clarification_attempt(self, session_id: str, attempt: ClarificationAttempt) -> None:
        """Add a clarification attempt to session history."""
        if session_id not in self._clarification_history:
            self._clarification_history[session_id] = []
        
        self._clarification_history[session_id].append(attempt)
        
        # Clean up old attempts to prevent memory leaks
        self._cleanup_old_attempts(session_id)
    
    def _is_recent_attempt(self, attempt: ClarificationAttempt) -> bool:
        """Check if an attempt is recent (within timeout)."""
        time_diff = (datetime.now(timezone.utc) - attempt.timestamp).total_seconds()
        return time_diff <= self.clarification_timeout
    
    def _cleanup_old_attempts(self, session_id: str) -> None:
        """Clean up old clarification attempts for a session."""
        attempts = self._clarification_history.get(session_id, [])
        
        # Keep only recent attempts or resolved attempts from the last hour
        cutoff_time = datetime.now(timezone.utc)
        recent_attempts = []
        
        for attempt in attempts:
            time_diff = (cutoff_time - attempt.timestamp).total_seconds()
            
            # Keep if recent or if resolved within last hour
            if time_diff <= self.clarification_timeout or (attempt.resolved and time_diff <= 3600):
                recent_attempts.append(attempt)
        
        self._clarification_history[session_id] = recent_attempts
    
    def _generate_clarification_message(self, intent_result: IntentResult, original_message: str) -> str:
        """Generate a clarification message based on intent result."""
        
        base_message = "I'm not quite sure how to help you with that. "
        
        # Add context based on what was found
        if intent_result.entities:
            entities_text = ", ".join(intent_result.entities[:3])
            base_message += f"I noticed you mentioned {entities_text}. "
        
        # Add the main clarification
        if intent_result.suggested_questions:
            base_message += intent_result.suggested_questions[0]
        else:
            base_message += "Could you please clarify what you'd like me to help you with?"
        
        return base_message
    
    def _resolve_clarification(self, response: str, attempt: ClarificationAttempt) -> Optional[IntentResult]:
        """
        Attempt to resolve clarification based on user response.
        
        Args:
            response: User's clarification response
            attempt: Original clarification attempt
            
        Returns:
            IntentResult if resolved, None otherwise
        """
        response_lower = response.lower().strip()
        
        # Check for explicit intent indicators
        cart_indicators = ['cart', 'basket', 'add', 'buy', 'purchase', 'shopping', 'order']
        qa_indicators = ['information', 'info', 'details', 'search', 'find', 'learn', 'compare', 'review']
        
        cart_matches = sum(1 for indicator in cart_indicators if indicator in response_lower)
        qa_matches = sum(1 for indicator in qa_indicators if indicator in response_lower)
        
        # Determine intent based on matches
        if cart_matches > qa_matches and cart_matches > 0:
            return IntentResult(
                intent="cart",
                confidence=0.8,
                entities=attempt.intent_result.entities,
                clarification_needed=False,
                suggested_questions=[],
                reasoning=f"User clarified cart intent with response: '{response}'",
                metadata={
                    "resolved_from_clarification": True,
                    "original_confidence": attempt.intent_result.confidence,
                    "clarification_response": response
                }
            )
        elif qa_matches > cart_matches and qa_matches > 0:
            return IntentResult(
                intent="qa",
                confidence=0.8,
                entities=attempt.intent_result.entities,
                clarification_needed=False,
                suggested_questions=[],
                reasoning=f"User clarified QA intent with response: '{response}'",
                metadata={
                    "resolved_from_clarification": True,
                    "original_confidence": attempt.intent_result.confidence,
                    "clarification_response": response
                }
            )
        
        # Check for yes/no responses to specific questions
        if any(word in response_lower for word in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay']):
            # If the first suggested question was about cart, assume cart intent
            first_question = attempt.questions_asked[0] if attempt.questions_asked else ""
            if any(word in first_question.lower() for word in cart_indicators):
                return IntentResult(
                    intent="cart",
                    confidence=0.7,
                    entities=attempt.intent_result.entities,
                    clarification_needed=False,
                    suggested_questions=[],
                    reasoning=f"User confirmed cart intent with positive response to: '{first_question}'",
                    metadata={
                        "resolved_from_clarification": True,
                        "original_confidence": attempt.intent_result.confidence,
                        "clarification_response": response
                    }
                )
        
        # If we can't resolve, return None
        return None