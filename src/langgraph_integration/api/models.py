"""
Pydantic models for LangGraph API integration.
Defines request/response schemas for agent workflows.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class LangGraphQueryRequest(BaseModel):
    """Request model for LangGraph agent queries."""
    
    query: str = Field(..., description="User query", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    max_products: int = Field(5, description="Maximum products to return", ge=1, le=20)
    max_reviews: int = Field(3, description="Maximum reviews to analyze", ge=0, le=10)
    llm_provider: str = Field("openai", description="LLM provider to use")
    llm_model: str = Field("gpt-4o-mini", description="LLM model to use")
    temperature: float = Field(0.7, description="LLM temperature", ge=0.0, le=2.0)
    enable_memory: bool = Field(True, description="Enable conversation memory")
    agent_type: str = Field("ambient", description="Type of agent workflow to use")


class LangGraphQueryResponse(BaseModel):
    """Response model for LangGraph agent queries."""
    
    query: str
    response: str
    session_id: str
    conversation_turn: int
    agent_workflow: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    workflow_steps: List[str]
    products_found: int
    reviews_found: int
    error_state: Optional[str] = None


class AgentStatusResponse(BaseModel):
    """Response model for agent status information."""
    
    session_id: str
    current_step: str
    workflow_status: str
    conversation_turn: int
    message_count: int
    last_activity: datetime
    performance_metrics: Dict[str, Any]
    error_state: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    
    session_id: str
    messages: List[Dict[str, Any]]
    total_turns: int
    conversation_age_hours: float
    summary: Optional[str] = None


class AgentCapabilitiesResponse(BaseModel):
    """Response model for agent capabilities."""
    
    available_agents: List[str]
    supported_providers: List[str]
    database_status: str
    tools_available: List[str]
    features: Dict[str, bool]