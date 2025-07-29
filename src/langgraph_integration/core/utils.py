"""
Utility functions for LangGraph agent workflows.
Provides helper functions for graph construction and state management.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .state_schemas import AgentState, ConversationState, validate_state

logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """Generate a unique session ID for conversations."""
    return f"session_{uuid.uuid4().hex[:16]}"


def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return f"trace_{uuid.uuid4().hex[:12]}"


def create_agent_config(
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    max_products: int = 5,
    max_reviews: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 500,
    **kwargs
) -> Dict[str, Any]:
    """Create configuration for agent workflows."""
    
    config = {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "max_products": max_products,
        "max_reviews": max_reviews,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_tracing": kwargs.get("enable_tracing", True),
        "enable_memory": kwargs.get("enable_memory", True),
        "memory_window": kwargs.get("memory_window", 10),
        "retry_attempts": kwargs.get("retry_attempts", 3),
        "timeout_seconds": kwargs.get("timeout_seconds", 30)
    }
    
    # Add any additional configuration
    config.update(kwargs)
    
    return config


def create_system_message(content: str) -> SystemMessage:
    """Create a system message for agent workflows."""
    return SystemMessage(content=content)


def create_human_message(content: str) -> HumanMessage:
    """Create a human message for agent workflows."""
    return HumanMessage(content=content)


def create_ai_message(content: str) -> AIMessage:
    """Create an AI message for agent workflows."""
    return AIMessage(content=content)


def extract_message_content(message: BaseMessage) -> str:
    """Extract content from a message object."""
    if hasattr(message, 'content'):
        return str(message.content)
    return str(message)


def format_conversation_history(messages: List[BaseMessage], limit: int = 10) -> str:
    """Format conversation history for context."""
    
    if not messages:
        return "No previous conversation history."
    
    # Take the last 'limit' messages
    recent_messages = messages[-limit:] if len(messages) > limit else messages
    
    formatted_history = []
    for msg in recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
        content = extract_message_content(msg)
        formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history)


def calculate_conversation_age(created_at: datetime) -> timedelta:
    """Calculate the age of a conversation."""
    return datetime.utcnow() - created_at


def should_summarize_conversation(
    messages: List[BaseMessage], 
    max_length: int = 20,
    max_age_hours: int = 24
) -> bool:
    """Determine if a conversation should be summarized."""
    
    if len(messages) > max_length:
        return True
    
    # Check if we have creation time in the first message
    if messages and hasattr(messages[0], 'additional_kwargs'):
        created_at = messages[0].additional_kwargs.get('created_at')
        if created_at:
            age = datetime.utcnow() - created_at
            if age.total_seconds() > max_age_hours * 3600:
                return True
    
    return False


def extract_entities_from_query(query: str) -> List[str]:
    """Extract potential product entities from user query."""
    
    # Simple entity extraction - can be enhanced with NLP models
    product_keywords = [
        "iphone", "samsung", "galaxy", "macbook", "laptop", "tablet",
        "headphones", "earbuds", "airpods", "speaker", "router", "cable",
        "charger", "keyboard", "mouse", "monitor", "tv", "kindle",
        "echo", "alexa", "fire", "stick"
    ]
    
    query_lower = query.lower()
    entities = []
    
    for keyword in product_keywords:
        if keyword in query_lower:
            entities.append(keyword)
    
    return entities


def classify_query_intent(query: str) -> str:
    """Classify the intent of a user query."""
    
    query_lower = query.lower()
    
    # Intent patterns
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        return "comparison"
    elif any(word in query_lower for word in ["recommend", "suggest", "best", "good"]):
        return "recommendation"
    elif any(word in query_lower for word in ["review", "opinion", "feedback", "experience"]):
        return "reviews"
    elif any(word in query_lower for word in ["problem", "issue", "complaint", "wrong"]):
        return "complaints"
    elif any(word in query_lower for word in ["price", "cost", "cheap", "expensive", "budget"]):
        return "pricing"
    elif any(word in query_lower for word in ["feature", "spec", "specification", "detail"]):
        return "features"
    else:
        return "general"


def create_error_response(error: Exception, session_id: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    
    return {
        "error": True,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "fallback_response": "I apologize, but I encountered an error processing your request. Please try again."
    }


def sanitize_state_for_storage(state: AgentState) -> Dict[str, Any]:
    """Sanitize agent state for database storage."""
    
    sanitized = {}
    
    for key, value in state.items():
        if key == "messages":
            # Convert messages to serializable format
            sanitized[key] = [
                {
                    "type": type(msg).__name__,
                    "content": extract_message_content(msg),
                    "timestamp": datetime.utcnow().isoformat()
                }
                for msg in value
            ]
        elif isinstance(value, datetime):
            sanitized[key] = value.isoformat()
        elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    
    return sanitized


def restore_state_from_storage(stored_data: Dict[str, Any]) -> AgentState:
    """Restore agent state from database storage."""
    
    restored = stored_data.copy()
    
    # Restore datetime fields
    for field in ["created_at", "updated_at"]:
        if field in restored and isinstance(restored[field], str):
            restored[field] = datetime.fromisoformat(restored[field])
    
    # Restore messages
    if "messages" in restored and isinstance(restored["messages"], list):
        messages = []
        for msg_data in restored["messages"]:
            if isinstance(msg_data, dict):
                msg_type = msg_data.get("type", "HumanMessage")
                content = msg_data.get("content", "")
                
                if msg_type == "HumanMessage":
                    messages.append(HumanMessage(content=content))
                elif msg_type == "AIMessage":
                    messages.append(AIMessage(content=content))
                elif msg_type == "SystemMessage":
                    messages.append(SystemMessage(content=content))
        
        restored["messages"] = messages
    
    return restored


def log_agent_step(session_id: str, step_name: str, metadata: Dict[str, Any] = None):
    """Log agent workflow step for debugging and monitoring."""
    
    log_data = {
        "session_id": session_id,
        "step": step_name,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if metadata:
        log_data["metadata"] = metadata
    
    logger.info(f"Agent step: {step_name}", extra=log_data)