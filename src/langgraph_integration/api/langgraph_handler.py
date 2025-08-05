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
from ..state.shopping_cart_manager import ShoppingCartManager, get_global_cart_manager
from ..core.router.master_graph import MasterAgentGraph

from .models import (
    LangGraphQueryRequest, 
    LangGraphQueryResponse,
    EnhancedQueryResponse,
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
        
        # Initialize shopping cart manager
        self.cart_manager = get_global_cart_manager()
        
        # Initialize master graph for routing
        self.master_graph = None  # Lazy-loaded to avoid circular dependencies
    
    async def process_query_with_enhanced_routing(
        self, 
        request: LangGraphQueryRequest
    ) -> EnhancedQueryResponse:
        """Process query using enhanced routing with cart functionality."""
        
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
            
            # Initialize master graph if needed
            if self.master_graph is None:
                self.master_graph = MasterAgentGraph(agent_config, self.agent_builder)
            
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
            
            # Execute master graph with routing
            self.logger.info(f"Executing master graph with routing for session {session_id}")
            
            result_state = await self.master_graph.process_query(state)
            
            # Save state if memory is enabled
            if request.enable_memory:
                self.state_manager.save_state(session_id, result_state)
            
            # Get cart data for session
            cart_data = self._get_cart_data_for_response(session_id)
            
            # Extract routing and tool information
            routing_info = self._extract_routing_information(result_state)
            tools_called = self._extract_tools_called(result_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract workflow steps from intermediate steps
            workflow_steps = self._extract_workflow_steps(result_state)
            
            # Build enhanced response
            response = EnhancedQueryResponse(
                query=request.query,
                response=result_state.get("final_response", "No response generated"),
                session_id=session_id,
                conversation_turn=result_state.get("conversation_turn", 1),
                agent_workflow=request.agent_type,
                routing_decision=routing_info.get("routing_decision"),
                agent_used=routing_info.get("agent_used", "unknown"),
                intent_confidence=routing_info.get("intent_confidence"),
                cart_data=cart_data.get("contents"),
                cart_updated=result_state.get("cart_updated", False),
                cart_item_count=cart_data.get("summary", {}).get("total_items", 0),
                cart_total=cart_data.get("summary", {}).get("total_value"),
                tools_called=tools_called,
                context={
                    "query_intent": result_state.get("query_intent"),
                    "extracted_entities": result_state.get("extracted_entities", []),
                    "search_results": result_state.get("search_results", {}),
                    "search_metadata": result_state.get("search_metadata", {}),
                    "cart_operation": result_state.get("cart_operation"),
                    "cart_operation_result": result_state.get("cart_operation_result")
                },
                metadata={
                    "workflow_status": result_state.get("workflow_status", "unknown"),
                    "current_step": result_state.get("current_step", "unknown"),
                    "llm_provider": request.llm_provider,
                    "llm_model": request.llm_model,
                    "enable_memory": request.enable_memory,
                    "routing_metadata": result_state.get("routing_metadata", {}),
                    "response_metadata": result_state.get("response_metadata", {})
                },
                processing_time=processing_time,
                workflow_steps=workflow_steps,
                products_found=len(result_state.get("selected_products", [])),
                reviews_found=len(result_state.get("review_summaries", [])),
                error_state=result_state.get("error_state")
            )
            
            # Log cart operation details for monitoring
            if result_state.get("cart_updated"):
                cart_operation = result_state.get("cart_operation", "unknown")
                self.logger.info(
                    f"Cart operation performed: {cart_operation} for session {session_id}, "
                    f"new item count: {cart_data.get('summary', {}).get('total_items', 0)}"
                )
            
            # Log routing decision for monitoring
            routing_decision = routing_info.get("routing_decision", "unknown")
            agent_used = routing_info.get("agent_used", "unknown")
            self.logger.info(
                f"Enhanced routing completed: {routing_decision} -> {agent_used} "
                f"for session {session_id} in {processing_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Enhanced routing workflow failed: {e}")
            
            # Get cart data even for error responses
            cart_data = self._get_cart_data_for_response(request.session_id or "error")
            
            # Return error response
            return EnhancedQueryResponse(
                query=request.query,
                response=f"I apologize, but I encountered an error processing your request: {str(e)}",
                session_id=request.session_id or "error",
                conversation_turn=1,
                agent_workflow=request.agent_type,
                routing_decision="error",
                agent_used="error",
                intent_confidence=0.0,
                cart_data=cart_data.get("contents"),
                cart_updated=False,
                cart_item_count=cart_data.get("summary", {}).get("total_items", 0),
                cart_total=cart_data.get("summary", {}).get("total_value"),
                tools_called=[],
                context={},
                metadata={"error": str(e)},
                processing_time=processing_time,
                workflow_steps=["error"],
                products_found=0,
                reviews_found=0,
                error_state=str(e)
            )

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
            
            # Check cart functionality
            cart_available = True
            try:
                cart_summary = self.cart_manager.get_cart_summary("test_session")
                cart_available = True
            except:
                cart_available = False
            
            return AgentCapabilitiesResponse(
                available_agents=available_agents + ["shopping_cart", "master_routing"],
                supported_providers=["openai", "groq", "google", "ollama"],
                database_status=db_status,
                tools_available=["vector_search", "product_analysis", "shopping_cart"],
                features={
                    "conversation_memory": True,
                    "multi_turn_conversations": True,
                    "product_search": vector_available,
                    "product_analysis": True,
                    "comparison": True,
                    "recommendations": True,
                    "review_analysis": vector_available,
                    "persistent_state": db_status == "healthy",
                    "shopping_cart": cart_available,
                    "intelligent_routing": True,
                    "intent_classification": True,
                    "dual_tool_support": True
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
    
    def get_cart_contents(self, session_id: str) -> Dict[str, Any]:
        """Get shopping cart contents for a session."""
        try:
            cart_contents = self.cart_manager.get_cart_contents(session_id)
            cart_summary = self.cart_manager.get_cart_summary(session_id)
            
            return {
                "session_id": session_id,
                "cart_contents": cart_contents,
                "cart_summary": cart_summary,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Failed to get cart contents: {e}")
            return {
                "session_id": session_id,
                "cart_contents": [],
                "cart_summary": {"total_items": 0, "total_value": 0.0, "is_empty": True},
                "success": False,
                "error": str(e)
            }
    
    def add_to_cart(
        self, 
        session_id: str, 
        product_id: str, 
        product_title: str,
        quantity: int = 1,
        price: Optional[float] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add item to shopping cart."""
        try:
            result = self.cart_manager.add_item(
                session_id=session_id,
                product_id=product_id,
                product_title=product_title,
                quantity=quantity,
                price=price,
                image_url=image_url,
                metadata=metadata
            )
            
            # Add cart summary to response
            if result["success"]:
                cart_summary = self.cart_manager.get_cart_summary(session_id)
                result["cart_summary"] = cart_summary
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to add item to cart: {e}")
            return {
                "success": False,
                "error": str(e),
                "item": None
            }
    
    def remove_from_cart(
        self, 
        session_id: str, 
        product_id: str,
        quantity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Remove item from shopping cart."""
        try:
            result = self.cart_manager.remove_item(
                session_id=session_id,
                product_id=product_id,
                quantity=quantity
            )
            
            # Add cart summary to response
            if result["success"]:
                cart_summary = self.cart_manager.get_cart_summary(session_id)
                result["cart_summary"] = cart_summary
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to remove item from cart: {e}")
            return {
                "success": False,
                "error": str(e),
                "item": None,
                "removed_completely": False
            }
    
    def clear_cart(self, session_id: str) -> Dict[str, Any]:
        """Clear all items from shopping cart."""
        try:
            result = self.cart_manager.clear_cart(session_id)
            
            # Add updated cart summary
            if result["success"]:
                cart_summary = self.cart_manager.get_cart_summary(session_id)
                result["cart_summary"] = cart_summary
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to clear cart: {e}")
            return {
                "success": False,
                "error": str(e),
                "items_removed": 0
            }
    
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
    
    def _get_cart_data_for_response(self, session_id: str) -> Dict[str, Any]:
        """Get cart data for API response."""
        try:
            cart_contents = self.cart_manager.get_cart_contents(session_id)
            cart_summary = self.cart_manager.get_cart_summary(session_id)
            
            return {
                "contents": cart_contents,
                "summary": cart_summary
            }
        except Exception as e:
            self.logger.error(f"Failed to get cart data for response: {e}")
            return {
                "contents": [],
                "summary": {
                    "total_items": 0,
                    "total_value": 0.0,
                    "unique_products": 0,
                    "is_empty": True
                }
            }
    
    def _extract_routing_information(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract routing information from agent state."""
        routing_metadata = state.get("routing_metadata", {})
        response_metadata = state.get("response_metadata", {})
        
        return {
            "routing_decision": state.get("routing_decision"),
            "agent_used": response_metadata.get("agent_used", "unknown"),
            "intent_confidence": state.get("intent_confidence", 0.0),
            "target_agent": state.get("target_agent"),
            "clarification_requested": response_metadata.get("clarification_requested", False)
        }
    
    def _extract_tools_called(self, state: Dict[str, Any]) -> List[str]:
        """Extract list of tools called during workflow execution."""
        tools_called = []
        
        # Check for cart operations
        if state.get("cart_operation"):
            tools_called.append(f"cart_{state['cart_operation']}")
        
        # Check for search operations
        if state.get("search_results"):
            tools_called.append("vector_search")
        
        # Check for product analysis
        if state.get("product_analysis_results"):
            tools_called.append("product_analysis")
        
        # Check intermediate steps for tool calls
        intermediate_steps = state.get("intermediate_steps", [])
        for step in intermediate_steps:
            if isinstance(step, dict) and "tool" in step:
                tool_name = step["tool"]
                if tool_name not in tools_called:
                    tools_called.append(tool_name)
        
        return tools_called