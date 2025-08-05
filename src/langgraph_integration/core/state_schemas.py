"""
State schemas for LangGraph agent workflows.
Defines the structure of conversation and agent states.
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from datetime import datetime, timezone


class ConversationState(TypedDict):
    """Shared state across all agent workflows."""
    
    # Core conversation data
    messages: List[BaseMessage]
    session_id: str
    conversation_turn: int
    
    # Query processing
    current_query: str
    query_type: str
    extracted_entities: List[str]
    
    # Search and retrieval
    search_results: Dict[str, Any]
    selected_products: List[Dict[str, Any]]
    review_summaries: List[Dict[str, Any]]
    
    # Response generation
    context_for_llm: str
    final_response: Optional[str]


class AgentState(TypedDict):
    """Complete agent state for LangGraph workflows."""
    
    # Core conversation data
    messages: List[BaseMessage]
    session_id: str
    conversation_turn: int
    created_at: datetime
    updated_at: datetime
    
    # Query processing
    current_query: str
    query_type: str
    extracted_entities: List[str]
    query_intent: Optional[str]
    
    # Router state extensions
    user_intent: Optional[str]  # 'qa', 'cart', 'unclear'
    intent_confidence: float
    clarification_needed: bool
    suggested_questions: List[str]
    routing_decision: Optional[str]  # 'qa', 'cart', 'clarification'
    target_agent: Optional[str]
    routing_metadata: Dict[str, Any]
    clarification_attempts: int
    clarification_history: List[str]
    
    # Shopping cart state extensions
    cart_operation: Optional[str]  # 'add', 'remove', 'list', 'clear'
    cart_operation_params: Dict[str, Any]
    cart_operation_result: Optional[Dict[str, Any]]
    current_cart_contents: List[Dict[str, Any]]
    cart_item_count: int
    cart_total: Optional[float]
    cart_updated: bool
    selected_product_for_cart: Optional[Dict[str, Any]]
    cart_operation_success: bool
    cart_operation_message: str
    
    # Search and retrieval
    search_results: Dict[str, Any]
    selected_products: List[Dict[str, Any]]
    review_summaries: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]
    
    # Agent workflow
    current_step: str
    tool_calls: List[Dict[str, Any]]
    intermediate_steps: List[Dict[str, Any]]
    workflow_status: str
    
    # Response generation
    context_for_llm: str
    final_response: Optional[str]
    response_metadata: Dict[str, Any]
    
    # Performance and monitoring
    performance_metrics: Dict[str, Any]
    error_state: Optional[str]
    retry_count: int
    
    # Configuration
    max_products: int
    max_reviews: int
    llm_provider: str
    llm_model: str


class ToolCallState(TypedDict):
    """State for individual tool calls."""
    
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Optional[Dict[str, Any]]
    execution_time: Optional[float]
    error: Optional[str]
    timestamp: datetime


class WorkflowCheckpoint(TypedDict):
    """Checkpoint data for workflow persistence."""
    
    checkpoint_id: str
    session_id: str
    state_data: AgentState
    node_name: str
    timestamp: datetime
    metadata: Dict[str, Any]


def create_initial_state(session_id: str, query: str, **kwargs) -> AgentState:
    """Create initial agent state for a new conversation."""
    
    now = datetime.now(timezone.utc)
    
    # Create base state
    base_state = AgentState(
        # Core conversation data
        messages=[],
        session_id=session_id,
        conversation_turn=1,
        created_at=now,
        updated_at=now,
        
        # Query processing
        current_query=query,
        query_type="unknown",
        extracted_entities=kwargs.get("extracted_entities", []),
        query_intent=None,
        
        # Router state extensions
        user_intent=kwargs.get("user_intent", None),
        intent_confidence=kwargs.get("intent_confidence", 0.0),
        clarification_needed=kwargs.get("clarification_needed", False),
        suggested_questions=kwargs.get("suggested_questions", []),
        routing_decision=kwargs.get("routing_decision", None),
        target_agent=kwargs.get("target_agent", None),
        routing_metadata=kwargs.get("routing_metadata", {}),
        clarification_attempts=kwargs.get("clarification_attempts", 0),
        clarification_history=kwargs.get("clarification_history", []),
        
        # Shopping cart state extensions
        cart_operation=kwargs.get("cart_operation", None),
        cart_operation_params=kwargs.get("cart_operation_params", {}),
        cart_operation_result=kwargs.get("cart_operation_result", None),
        current_cart_contents=kwargs.get("current_cart_contents", []),
        cart_item_count=kwargs.get("cart_item_count", 0),
        cart_total=kwargs.get("cart_total", None),
        cart_updated=kwargs.get("cart_updated", False),
        selected_product_for_cart=kwargs.get("selected_product_for_cart", None),
        cart_operation_success=kwargs.get("cart_operation_success", False),
        cart_operation_message=kwargs.get("cart_operation_message", ""),
        
        # Search and retrieval
        search_results=kwargs.get("search_results", {}),
        selected_products=kwargs.get("selected_products", []),
        review_summaries=kwargs.get("review_summaries", []),
        search_metadata=kwargs.get("search_metadata", {}),
        
        # Agent workflow
        current_step=kwargs.get("current_step", "start"),
        tool_calls=kwargs.get("tool_calls", []),
        intermediate_steps=kwargs.get("intermediate_steps", []),
        workflow_status=kwargs.get("workflow_status", "running"),
        
        # Response generation
        context_for_llm=kwargs.get("context_for_llm", ""),
        final_response=kwargs.get("final_response", None),
        response_metadata=kwargs.get("response_metadata", {}),
        
        # Performance and monitoring
        performance_metrics=kwargs.get("performance_metrics", {}),
        error_state=kwargs.get("error_state", None),
        retry_count=kwargs.get("retry_count", 0),
        
        # Configuration
        max_products=kwargs.get("max_products", 5),
        max_reviews=kwargs.get("max_reviews", 3),
        llm_provider=kwargs.get("llm_provider", "openai"),
        llm_model=kwargs.get("llm_model", "gpt-4o-mini")
    )
    
    return base_state


def update_state_step(state: AgentState, step_name: str, **updates) -> AgentState:
    """Update agent state for a new workflow step."""
    
    updated_state = state.copy()
    updated_state["current_step"] = step_name
    updated_state["updated_at"] = datetime.now(timezone.utc)
    
    # Apply any additional updates
    for key, value in updates.items():
        if key in updated_state:
            updated_state[key] = value
    
    return updated_state


def validate_state(state: AgentState) -> bool:
    """Validate agent state structure and required fields."""
    
    required_fields = [
        "session_id", "conversation_turn", "current_query",
        "messages", "current_step", "workflow_status"
    ]
    
    for field in required_fields:
        if field not in state:
            return False
    
    # Validate data types
    if not isinstance(state["messages"], list):
        return False
    
    if not isinstance(state["session_id"], str):
        return False
    
    if not isinstance(state["conversation_turn"], int):
        return False
    
    if not isinstance(state["current_query"], str):
        return False
    
    if not isinstance(state["current_step"], str):
        return False
    
    if not isinstance(state["workflow_status"], str):
        return False
    
    # Validate workflow status values
    valid_statuses = ["running", "completed", "error", "paused"]
    if state["workflow_status"] not in valid_statuses:
        return False
    
    # Validate conversation turn is positive
    if state["conversation_turn"] < 1:
        return False
    
    return True


def validate_conversation_state(state: ConversationState) -> bool:
    """Validate conversation state structure."""
    
    required_fields = [
        "messages", "session_id", "conversation_turn",
        "current_query", "query_type"
    ]
    
    for field in required_fields:
        if field not in state:
            return False
    
    # Validate data types
    if not isinstance(state["messages"], list):
        return False
    
    if not isinstance(state["session_id"], str):
        return False
    
    if not isinstance(state["conversation_turn"], int):
        return False
    
    return True


def merge_states(base_state: AgentState, updates: Dict[str, Any]) -> AgentState:
    """Merge updates into base state safely."""
    
    merged_state = base_state.copy()
    
    for key, value in updates.items():
        if key in merged_state:
            merged_state[key] = value
    
    # Update timestamp
    merged_state["updated_at"] = datetime.now(timezone.utc)
    
    return merged_state


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """Get a summary of the agent state."""
    
    return {
        "session_id": state.get("session_id", "unknown"),
        "conversation_turn": state.get("conversation_turn", 0),
        "current_step": state.get("current_step", "unknown"),
        "workflow_status": state.get("workflow_status", "unknown"),
        "query_type": state.get("query_type", "unknown"),
        "message_count": len(state.get("messages", [])),
        "has_search_results": bool(state.get("search_results")),
        "has_final_response": bool(state.get("final_response")),
        "error_state": state.get("error_state"),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at")
    }