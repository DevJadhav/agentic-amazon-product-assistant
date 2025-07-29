"""
LangGraph API handler for FastAPI integration.
Handles LangGraph agent workflow execution and state management.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.agent_builder import AgentGraphBuilder
from ..core.state_schemas import create_initial_state, get_state_summary
from ..core.utils import generate_session_id, create_agent_config
from ..state.state_manager import LangGraphStateManager
from ..tools.vector_search_tool import VectorSearchTool
from ..tools.product_analysis_tool import ProductAnalysisTool

from .models import (
    LangGraphQueryRequest, 
    LangGraphQueryResponse, 
    AgentStatusResponse,
    ConversationHistoryResponse,
    AgentCapabilitiesResponse
)

logger = logging.getLogger(__name__)


class LangGraphAPIHandler:
    """Handles LangGraph integration with existing FastAPI endpoints."""
    
    def __init__(self):
        """Initialize LangGraph API handler."""
        self.agent_builder = AgentGraphBuilder()
        self.state_manager = LangGraphStateManager()
        self.logger = logging.getLogger(__name__)
        
        # Cache compiled graphs for performance
        self._graph_cache = {}
        
        # Initialize tools for capability checking
        self.vector_search_tool = VectorSearchTool()
        self.product_analysis_tool = ProductAnalysisTool()
    
    async def process_query_with_agent(
        self, 
        request: LangGraphQueryRequest
    ) -> LangGraphQueryResponse:
        """Process query using LangGraph agent workflow."""
        
        start_time = time.time()
        
        try:
            # Generate or use provided session ID
            session_id = request.session_id or generate_session_id()
            
            # Create agent configuration
            agent_config = create_agent_config(
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
                max_products=request.max_products,
                max_reviews=request.max_reviews,
                temperature=request.temperature,
                enable_memory=request.enable_memory
            )
            
            # Get or create agent graph
            agent_graph = self._get_agent_graph(request.agent_type, agent_config)
            
            # Load existing state or create new one
            existing_state = None
            if request.enable_memory:
                existing_state = self.state_manager.load_state(session_id)
            
            if existing_state:
                # Update existing state with new query
                state = existing_state.copy()
                state["current_query"] = request.query
                state["conversation_turn"] += 1
                state["updated_at"] = datetime.utcnow()
                state["workflow_status"] = "running"
                state["current_step"] = "start"
                state["error_state"] = None
            else:
                # Create initial state
                state = create_initial_state(
                    session_id=session_id,
                    query=request.query,
                    max_products=request.max_products,
                    max_reviews=request.max_reviews,
                    llm_provider=request.llm_provider,
                    llm_model=request.llm_model
                )
            
            # Execute agent workflow
            self.logger.info(f"Executing {request.agent_type} agent for session {session_id}")
            
            result_state = await agent_graph.ainvoke(state)
            
            # Save state if memory is enabled
            if request.enable_memory:
                self.state_manager.save_state(session_id, result_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract workflow steps from intermediate steps
            workflow_steps = self._extract_workflow_steps(result_state)
            
            # Build response
            response = LangGraphQueryResponse(
                query=request.query,
                response=result_state.get("final_response", "No response generated"),
                session_id=session_id,
                conversation_turn=result_state.get("conversation_turn", 1),
                agent_workflow=request.agent_type,
                context={
                    "query_intent": result_state.get("query_intent"),
                    "extracted_entities": result_state.get("extracted_entities", []),
                    "search_results": result_state.get("search_results", {}),
                    "search_metadata": result_state.get("search_metadata", {})
                },
                metadata={
                    "workflow_status": result_state.get("workflow_status", "unknown"),
                    "current_step": result_state.get("current_step", "unknown"),
                    "llm_provider": request.llm_provider,
                    "llm_model": request.llm_model,
                    "enable_memory": request.enable_memory
                },
                processing_time=processing_time,
                workflow_steps=workflow_steps,
                products_found=len(result_state.get("selected_products", [])),
                reviews_found=len(result_state.get("review_summaries", [])),
                error_state=result_state.get("error_state")
            )
            
            self.logger.info(f"Agent workflow completed in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Agent workflow failed: {e}")
            
            # Return error response
            return LangGraphQueryResponse(
                query=request.query,
                response=f"I apologize, but I encountered an error processing your request: {str(e)}",
                session_id=request.session_id or "error",
                conversation_turn=1,
                agent_workflow=request.agent_type,
                context={},
                metadata={"error": str(e)},
                processing_time=processing_time,
                workflow_steps=["error"],
                products_found=0,
                reviews_found=0,
                error_state=str(e)
            )
    
    def get_conversation_history(self, session_id: str) -> ConversationHistoryResponse:
        """Get conversation history for a session."""
        
        try:
            # Load conversation state
            state = self.state_manager.load_state(session_id)
            
            if not state:
                return ConversationHistoryResponse(
                    session_id=session_id,
                    messages=[],
                    total_turns=0,
                    conversation_age_hours=0.0
                )
            
            # Format messages
            messages = []
            for msg in state.get("messages", []):
                messages.append({
                    "type": type(msg).__name__.replace("Message", "").lower(),
                    "content": str(msg.content) if hasattr(msg, 'content') else str(msg),
                    "timestamp": datetime.utcnow().isoformat()  # Placeholder
                })
            
            # Calculate conversation age
            created_at = state.get("created_at", datetime.utcnow())
            age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
            
            # Get conversation summary if available
            summary = self.state_manager.memory_manager.summarize_long_conversations(session_id)
            
            return ConversationHistoryResponse(
                session_id=session_id,
                messages=messages,
                total_turns=state.get("conversation_turn", 0),
                conversation_age_hours=age_hours,
                summary=summary if summary else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return ConversationHistoryResponse(
                session_id=session_id,
                messages=[],
                total_turns=0,
                conversation_age_hours=0.0
            )
    
    def get_agent_status(self, session_id: str) -> AgentStatusResponse:
        """Get current agent status for a session."""
        
        try:
            state = self.state_manager.load_state(session_id)
            
            if not state:
                return AgentStatusResponse(
                    session_id=session_id,
                    current_step="not_found",
                    workflow_status="not_found",
                    conversation_turn=0,
                    message_count=0,
                    last_activity=datetime.utcnow(),
                    performance_metrics={}
                )
            
            return AgentStatusResponse(
                session_id=session_id,
                current_step=state.get("current_step", "unknown"),
                workflow_status=state.get("workflow_status", "unknown"),
                conversation_turn=state.get("conversation_turn", 0),
                message_count=len(state.get("messages", [])),
                last_activity=state.get("updated_at", datetime.utcnow()),
                performance_metrics=state.get("performance_metrics", {}),
                error_state=state.get("error_state")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get agent status: {e}")
            return AgentStatusResponse(
                session_id=session_id,
                current_step="error",
                workflow_status="error",
                conversation_turn=0,
                message_count=0,
                last_activity=datetime.utcnow(),
                performance_metrics={},
                error_state=str(e)
            )
    
    def get_agent_capabilities(self) -> AgentCapabilitiesResponse:
        """Get information about agent capabilities."""
        
        try:
            # Get available agent types
            available_agents = list(self.agent_builder.get_available_graphs().keys())
            
            # Check tool availability
            vector_tool_info = self.vector_search_tool.test_connection()
            vector_available = vector_tool_info.get("status") == "success"
            
            # Check database status
            try:
                db_health = self.state_manager.state_store.db_manager.get_database_stats()
                db_status = "healthy" if db_health else "unavailable"
            except:
                db_status = "unavailable"
            
            return AgentCapabilitiesResponse(
                available_agents=available_agents,
                supported_providers=["openai", "groq", "google", "ollama"],
                database_status=db_status,
                tools_available=["vector_search", "product_analysis"],
                features={
                    "conversation_memory": True,
                    "multi_turn_conversations": True,
                    "product_search": vector_available,
                    "product_analysis": True,
                    "comparison": True,
                    "recommendations": True,
                    "review_analysis": vector_available,
                    "persistent_state": db_status == "healthy"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get agent capabilities: {e}")
            return AgentCapabilitiesResponse(
                available_agents=["ambient"],
                supported_providers=["openai"],
                database_status="error",
                tools_available=[],
                features={"error": True}
            )
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        
        try:
            return self.state_manager.memory_manager.clear_conversation_memory(session_id)
        except Exception as e:
            self.logger.error(f"Failed to clear conversation: {e}")
            return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about all sessions."""
        
        try:
            return self.state_manager.get_state_statistics()
        except Exception as e:
            self.logger.error(f"Failed to get session statistics: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _get_agent_graph(self, agent_type: str, config: Dict[str, Any]):
        """Get or create agent graph with caching."""
        
        cache_key = f"{agent_type}_{hash(str(sorted(config.items())))}"
        
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        
        # Create new graph based on type
        if agent_type == "ambient":
            graph = self.agent_builder.create_ambient_agent_graph()
        elif agent_type == "product_search":
            graph = self.agent_builder.build_product_search_graph()
        elif agent_type == "review_analysis":
            graph = self.agent_builder.build_review_analysis_graph()
        elif agent_type == "comparison":
            graph = self.agent_builder.build_comparison_graph()
        elif agent_type == "recommendation":
            graph = self.agent_builder.build_recommendation_graph()
        else:
            # Default to ambient agent
            graph = self.agent_builder.create_ambient_agent_graph()
        
        # Cache the graph
        self._graph_cache[cache_key] = graph
        
        return graph
    
    def _extract_workflow_steps(self, state: Dict[str, Any]) -> List[str]:
        """Extract workflow steps from agent state."""
        
        steps = []
        
        # Add intermediate steps if available
        intermediate_steps = state.get("intermediate_steps", [])
        for step in intermediate_steps:
            if isinstance(step, dict) and "step" in step:
                steps.append(step["step"])
        
        # Add current step
        current_step = state.get("current_step")
        if current_step and current_step not in steps:
            steps.append(current_step)
        
        # If no steps found, create basic workflow
        if not steps:
            workflow_status = state.get("workflow_status", "unknown")
            if workflow_status == "completed":
                steps = ["analyze_query", "search", "generate_response"]
            else:
                steps = [current_step or "unknown"]
        
        return steps