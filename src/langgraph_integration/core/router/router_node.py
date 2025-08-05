"""
Router node implementation for intelligent agent routing.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ..state_schemas import AgentState
from .intent_classifier import IntentClassifier, IntentResult
from .clarification_handler import ClarificationHandler
from .router_error_handler import RouterErrorHandler

logger = logging.getLogger(__name__)


class RouterNode:
    """Main router node for agent selection and clarification handling."""
    
    def __init__(self, intent_classifier: Optional[IntentClassifier] = None,
                 clarification_handler: Optional[ClarificationHandler] = None,
                 error_handler: Optional[RouterErrorHandler] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize router node with components.
        
        Args:
            intent_classifier: Intent classification component
            clarification_handler: Clarification handling component
            error_handler: Router error handling component
            config: Router configuration
        """
        self.config = config or {}
        self.intent_classifier = intent_classifier or IntentClassifier(self.config.get("classifier", {}))
        self.clarification_handler = clarification_handler or ClarificationHandler(self.config.get("clarification", {}))
        self.error_handler = error_handler or RouterErrorHandler(self.config.get("error_handler", {}))
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Routing statistics for monitoring
        self._routing_stats = {
            "total_routes": 0,
            "qa_routes": 0,
            "cart_routes": 0,
            "clarifications": 0,
            "fallbacks": 0
        }
    
    async def route_message(self, state: AgentState) -> AgentState:
        """
        Route message to appropriate agent or request clarification.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with routing decision
        """
        session_id = state["session_id"]
        current_query = state["current_query"]
        
        self.logger.info(f"Routing message for session {session_id}: '{current_query[:50]}...'")
        
        try:
            # Extract context for better classification
            try:
                context = self._extract_context(state)
            except Exception as context_error:
                self.logger.warning(f"Context extraction failed for session {session_id}: {context_error}")
                return self.error_handler.handle_context_extraction_error(context_error, state)
            
            # Classify intent
            try:
                intent_result = self.intent_classifier.classify_intent(current_query, context)
            except Exception as classification_error:
                self.logger.error(f"Intent classification failed for session {session_id}: {classification_error}")
                return self.error_handler.handle_intent_classification_error(
                    classification_error, state, current_query, context
                )
            
            # Update routing statistics
            self._routing_stats["total_routes"] += 1
            
            # Update state with classification results
            try:
                updated_state = self._update_state_with_intent(state, intent_result)
            except Exception as state_error:
                self.logger.error(f"State update failed for session {session_id}: {state_error}")
                return self.error_handler.handle_state_update_error(state_error, state, "intent_classification")
            
            # Determine routing action
            try:
                if self.clarification_handler.needs_clarification(intent_result, session_id):
                    return await self._handle_clarification(updated_state, intent_result)
                else:
                    return await self._route_to_agent(updated_state, intent_result)
            except Exception as routing_error:
                self.logger.error(f"Routing decision failed for session {session_id}: {routing_error}")
                return self.error_handler.handle_routing_decision_error(routing_error, state, intent_result)
                
        except Exception as e:
            self.logger.error(f"Unexpected router error for session {session_id}: {e}")
            return self.error_handler.handle_routing_decision_error(e, state)
    
    async def process_clarification_response(self, state: AgentState) -> AgentState:
        """
        Process user's response to a clarification request.
        
        Args:
            state: Current agent state with clarification response
            
        Returns:
            Updated agent state with resolved routing or further clarification
        """
        session_id = state["session_id"]
        current_query = state["current_query"]
        
        self.logger.info(f"Processing clarification response for session {session_id}")
        
        try:
            # Attempt to resolve clarification
            try:
                resolved_intent = self.clarification_handler.process_clarification_response(
                    current_query, session_id
                )
            except Exception as clarification_error:
                self.logger.error(f"Clarification processing failed for session {session_id}: {clarification_error}")
                return self.error_handler.handle_clarification_error(clarification_error, state)
            
            if resolved_intent:
                # Clarification resolved, route to agent
                try:
                    updated_state = self._update_state_with_intent(state, resolved_intent)
                    return await self._route_to_agent(updated_state, resolved_intent)
                except Exception as routing_error:
                    return self.error_handler.handle_routing_decision_error(routing_error, state, resolved_intent)
            else:
                # Clarification not resolved, try again or fallback
                try:
                    context = self._extract_context(state)
                    intent_result = self.intent_classifier.classify_intent(current_query, context)
                    
                    if self.clarification_handler.needs_clarification(intent_result, session_id):
                        return await self._handle_clarification(state, intent_result)
                    else:
                        return await self._handle_fallback(state)
                except Exception as fallback_error:
                    return self.error_handler.handle_clarification_error(fallback_error, state)
                    
        except Exception as e:
            self.logger.error(f"Unexpected clarification processing error for session {session_id}: {e}")
            return self.error_handler.handle_clarification_error(e, state)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring."""
        return {
            **self._routing_stats,
            "qa_percentage": (self._routing_stats["qa_routes"] / max(1, self._routing_stats["total_routes"])) * 100,
            "cart_percentage": (self._routing_stats["cart_routes"] / max(1, self._routing_stats["total_routes"])) * 100,
            "clarification_rate": (self._routing_stats["clarifications"] / max(1, self._routing_stats["total_routes"])) * 100,
            "fallback_rate": (self._routing_stats["fallbacks"] / max(1, self._routing_stats["total_routes"])) * 100
        }
    
    def reset_routing_stats(self) -> None:
        """Reset routing statistics."""
        self._routing_stats = {
            "total_routes": 0,
            "qa_routes": 0,
            "cart_routes": 0,
            "clarifications": 0,
            "fallbacks": 0
        }
    
    def _extract_context(self, state: AgentState) -> Dict[str, Any]:
        """Extract conversation context for better intent classification."""
        context = {}
        
        # Check for recent cart activity
        tool_calls = state.get("tool_calls", [])
        recent_cart_tools = [
            call for call in tool_calls[-5:]  # Last 5 tool calls
            if call.get("tool_name", "").startswith(("add_to_cart", "remove_from_cart", "list_cart"))
        ]
        context["recent_cart_activity"] = len(recent_cart_tools) > 0
        
        # Check for recent QA activity
        recent_qa_tools = [
            call for call in tool_calls[-5:]
            if call.get("tool_name", "").startswith(("vector_search", "product_analysis"))
        ]
        context["recent_qa_activity"] = len(recent_qa_tools) > 0
        
        # Extract entities from previous queries
        previous_entities = []
        for step in state.get("intermediate_steps", [])[-3:]:  # Last 3 steps
            if "entities" in step:
                previous_entities.extend(step["entities"])
        context["previous_entities"] = list(set(previous_entities))
        
        # Conversation turn context
        context["conversation_turn"] = state.get("conversation_turn", 1)
        context["has_search_results"] = bool(state.get("search_results"))
        context["has_selected_products"] = bool(state.get("selected_products"))
        
        return context
    
    def _update_state_with_intent(self, state: AgentState, intent_result: IntentResult) -> AgentState:
        """Update agent state with intent classification results."""
        updated_state = state.copy()
        
        # Add router-specific state fields
        updated_state["user_intent"] = intent_result.intent
        updated_state["intent_confidence"] = intent_result.confidence
        updated_state["clarification_needed"] = intent_result.clarification_needed
        updated_state["suggested_questions"] = intent_result.suggested_questions
        updated_state["extracted_entities"] = intent_result.entities
        
        # Add routing metadata
        updated_state["routing_metadata"] = {
            "classification_reasoning": intent_result.reasoning,
            "classification_timestamp": datetime.now(timezone.utc).isoformat(),
            "classifier_metadata": intent_result.metadata
        }
        
        # Update general state fields
        updated_state["current_step"] = "router"
        updated_state["updated_at"] = datetime.now(timezone.utc)
        
        return updated_state
    
    async def _route_to_agent(self, state: AgentState, intent_result: IntentResult) -> AgentState:
        """Route to the appropriate agent based on intent."""
        updated_state = state.copy()
        
        if intent_result.intent == "cart":
            updated_state["routing_decision"] = "cart"
            updated_state["target_agent"] = "shopping_cart_agent"
            self._routing_stats["cart_routes"] += 1
            self.logger.info(f"Routing to Shopping Cart Agent for session {state['session_id']}")
            
        elif intent_result.intent == "qa":
            updated_state["routing_decision"] = "qa"
            updated_state["target_agent"] = "qa_agent"
            self._routing_stats["qa_routes"] += 1
            self.logger.info(f"Routing to QA Agent for session {state['session_id']}")
            
        else:
            # Fallback to QA agent for unclear intents
            updated_state["routing_decision"] = "qa"
            updated_state["target_agent"] = "qa_agent"
            self._routing_stats["fallbacks"] += 1
            self.logger.info(f"Fallback routing to QA Agent for session {state['session_id']}")
        
        # Add routing timestamp
        updated_state["routing_metadata"]["routing_timestamp"] = datetime.now(timezone.utc).isoformat()
        updated_state["routing_metadata"]["confidence_threshold"] = self.confidence_threshold
        
        return updated_state
    
    async def _handle_clarification(self, state: AgentState, intent_result: IntentResult) -> AgentState:
        """Handle clarification request."""
        session_id = state["session_id"]
        current_query = state["current_query"]
        
        # Create clarification request
        clarification_request = self.clarification_handler.create_clarification_request(
            intent_result, session_id, current_query
        )
        
        # Update state for clarification
        updated_state = state.copy()
        updated_state["routing_decision"] = "clarification"
        updated_state["final_response"] = clarification_request["message"]
        updated_state["workflow_status"] = "completed"  # End workflow after clarification
        updated_state["response_metadata"] = {
            "response_type": "clarification_request",
            "clarification_data": clarification_request
        }
        
        # Track clarification statistics
        self._routing_stats["clarifications"] += 1
        
        self.logger.info(f"Requesting clarification for session {session_id}")
        
        return updated_state
    
    async def _handle_fallback(self, state: AgentState) -> AgentState:
        """Handle fallback when clarification attempts are exhausted."""
        session_id = state["session_id"]
        
        # Create fallback response
        fallback_response = self.clarification_handler.create_fallback_response(session_id)
        
        # Route to default agent (QA)
        updated_state = state.copy()
        updated_state["routing_decision"] = "qa"
        updated_state["target_agent"] = "qa_agent"
        updated_state["routing_metadata"]["fallback_applied"] = True
        updated_state["routing_metadata"]["fallback_reason"] = "max_clarification_attempts_reached"
        
        # Track fallback statistics
        self._routing_stats["fallbacks"] += 1
        
        self.logger.info(f"Applying fallback routing for session {session_id}")
        
        return updated_state
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get combined routing and error statistics."""
        routing_stats = self.get_routing_stats()
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            "routing_stats": routing_stats,
            "error_stats": error_stats,
            "combined_metrics": {
                "total_requests": routing_stats["total_routes"] + error_stats["total_errors"],
                "success_rate": (
                    routing_stats["total_routes"] / 
                    max(1, routing_stats["total_routes"] + error_stats["total_errors"])
                ) * 100,
                "error_recovery_rate": error_stats.get("recovery_success_rate", 0.0)
            }
        }