"""
Master graph implementation that orchestrates routing and agent execution.
Provides improved organization and naming conventions for LangGraph workflows.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph

from ..state_schemas import AgentState, update_state_step
from ..base_agent import BaseAgent
# AgentGraphBuilder will be injected to avoid circular imports
from .router_node import RouterNode
from .intent_classifier import IntentClassifier
from .clarification_handler import ClarificationHandler

logger = logging.getLogger(__name__)


class ShoppingCartAgent(BaseAgent):
    """Agent specialized for shopping cart management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Shopping Cart Agent."""
        super().__init__(config)
        self.cart_manager = None  # Will be injected when available
        
    def create_graph(self) -> StateGraph:
        """Create shopping cart management workflow."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_cart_request", self._analyze_cart_request)
        workflow.add_node("execute_cart_operation", self._execute_cart_operation)
        workflow.add_node("generate_cart_response", self._generate_cart_response)
        workflow.add_node("update_cart_state", self._update_cart_state)
        
        # Define edges
        workflow.add_edge(START, "analyze_cart_request")
        workflow.add_edge("analyze_cart_request", "execute_cart_operation")
        workflow.add_edge("execute_cart_operation", "generate_cart_response")
        workflow.add_edge("generate_cart_response", "update_cart_state")
        workflow.add_edge("update_cart_state", END)
        
        return workflow
    
    async def _analyze_cart_request(self, state: AgentState) -> AgentState:
        """Analyze cart operation request."""
        query = state["current_query"].lower()
        
        # Determine cart operation type
        if any(word in query for word in ["add", "put", "place"]):
            operation = "add"
        elif any(word in query for word in ["remove", "delete", "take out"]):
            operation = "remove"
        elif any(word in query for word in ["show", "view", "list", "display"]):
            operation = "list"
        elif any(word in query for word in ["clear", "empty"]):
            operation = "clear"
        else:
            operation = "list"  # Default to showing cart
        
        # Extract potential product information
        entities = state.get("extracted_entities", [])
        
        updated_state = update_state_step(
            state,
            "analyze_cart_request",
            cart_operation=operation,
            cart_operation_params={
                "entities": entities,
                "original_query": state["current_query"]
            }
        )
        
        self.logger.info(f"Analyzed cart request: operation={operation}, entities={entities}")
        
        return updated_state
    
    async def _execute_cart_operation(self, state: AgentState) -> AgentState:
        """Execute the requested cart operation using tools."""
        operation = state.get("cart_operation", "list")
        params = state.get("cart_operation_params", {})
        
        # For now, simulate cart operations since tools aren't fully integrated yet
        # This will be replaced with actual tool calls in later tasks
        
        if operation == "add":
            result = {
                "success": True,
                "message": "Item would be added to cart",
                "operation": "add",
                "details": params
            }
        elif operation == "remove":
            result = {
                "success": True,
                "message": "Item would be removed from cart",
                "operation": "remove",
                "details": params
            }
        elif operation == "list":
            result = {
                "success": True,
                "message": "Cart contents would be displayed",
                "operation": "list",
                "cart_contents": []  # Empty for now
            }
        elif operation == "clear":
            result = {
                "success": True,
                "message": "Cart would be cleared",
                "operation": "clear"
            }
        else:
            result = {
                "success": False,
                "message": f"Unknown cart operation: {operation}",
                "operation": operation
            }
        
        updated_state = update_state_step(
            state,
            "execute_cart_operation",
            cart_operation_result=result,
            cart_operation_success=result["success"],
            cart_updated=result["success"] and operation in ["add", "remove", "clear"]
        )
        
        self.logger.info(f"Executed cart operation: {operation}, success={result['success']}")
        
        return updated_state
    
    async def _generate_cart_response(self, state: AgentState) -> AgentState:
        """Generate response based on cart operation results."""
        result = state.get("cart_operation_result", {}) or {}
        operation = result.get("operation", "unknown")
        success = result.get("success", False)
        
        if success:
            if operation == "add":
                response = f"I would add the requested item to your cart. {result.get('message', '')}"
            elif operation == "remove":
                response = f"I would remove the requested item from your cart. {result.get('message', '')}"
            elif operation == "list":
                cart_contents = result.get("cart_contents", [])
                if cart_contents:
                    response = f"Here are the items in your cart: {', '.join(cart_contents)}"
                else:
                    response = "Your cart is currently empty."
            elif operation == "clear":
                response = "I would clear all items from your cart."
            else:
                response = f"Cart operation '{operation}' completed successfully."
        else:
            response = f"I couldn't complete the cart operation: {result.get('message', 'Unknown error')}"
        
        updated_state = update_state_step(
            state,
            "generate_cart_response",
            final_response=response,
            cart_operation_message=response
        )
        
        return updated_state
    
    async def _update_cart_state(self, state: AgentState) -> AgentState:
        """Update conversation state with cart information."""
        # Update cart-related state fields
        updated_state = update_state_step(
            state,
            "update_cart_state",
            workflow_status="completed"
        )
        
        # Add cart metadata to response
        if "response_metadata" not in updated_state:
            updated_state["response_metadata"] = {}
        
        updated_state["response_metadata"]["cart_operation_performed"] = True
        updated_state["response_metadata"]["cart_operation_type"] = state.get("cart_operation", "unknown")
        
        return updated_state


class MasterAgentGraph:
    """
    Master graph that orchestrates routing and agent execution.
    
    This class provides the main orchestration layer for the multi-agent system,
    implementing intelligent routing between specialized agents based on user intent.
    Features improved organization, clear naming conventions, and comprehensive
    error handling with fallback mechanisms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, agent_builder=None):
        """
        Initialize master agent graph with configuration and dependencies.
        
        Args:
            config: Configuration dictionary with router, classifier, and agent settings
            agent_builder: AgentGraphBuilder instance for creating specialized agents
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize routing components with descriptive names
        self.intent_router = RouterNode(
            intent_classifier=IntentClassifier(self.config.get("classifier", {})),
            clarification_handler=ClarificationHandler(self.config.get("clarification", {})),
            config=self.config.get("router", {})
        )
        
        # Initialize specialized agents with clear naming
        self.agent_builder = agent_builder  # Injected to avoid circular imports
        self.qa_agent_graph = None  # Lazy-loaded QA agent
        self.shopping_cart_agent = ShoppingCartAgent(self.config.get("cart_agent", {}))
        
        # Graph compilation and metadata
        self._compiled_master_graph: Optional[CompiledStateGraph] = None
        self._graph_metadata = {
            "created_at": datetime.now(timezone.utc),
            "version": "1.0.0",
            "agent_count": 2,
            "routing_enabled": True
        }
    
    def create_master_graph(self) -> StateGraph:
        """
        Create the master routing graph with improved organization and naming.
        
        This method constructs the main workflow graph that orchestrates:
        1. Intent classification and routing
        2. Specialized agent execution (QA or Shopping Cart)
        3. Clarification handling for ambiguous queries
        4. Response finalization with consistent formatting
        
        Returns:
            StateGraph: The master routing workflow graph
        """
        
        # Create workflow with descriptive name
        master_workflow = StateGraph(AgentState)
        
        # Add nodes with clear, descriptive names
        master_workflow.add_node("intent_classification_and_routing", self._execute_intent_router)
        master_workflow.add_node("product_qa_agent_execution", self._execute_product_qa_agent)
        master_workflow.add_node("shopping_cart_agent_execution", self._execute_shopping_cart_agent)
        master_workflow.add_node("clarification_request_handling", self._handle_clarification_request)
        master_workflow.add_node("response_finalization_and_formatting", self._finalize_and_format_response)
        
        # Define routing edges with descriptive routing logic
        master_workflow.add_edge(START, "intent_classification_and_routing")
        
        # Conditional routing based on intent classification results
        master_workflow.add_conditional_edges(
            "intent_classification_and_routing",
            self._determine_agent_routing_decision,
            {
                "route_to_qa_agent": "product_qa_agent_execution",
                "route_to_cart_agent": "shopping_cart_agent_execution", 
                "request_clarification": "clarification_request_handling"
            }
        )
        
        # Agent completion edges leading to response finalization
        master_workflow.add_edge("product_qa_agent_execution", "response_finalization_and_formatting")
        master_workflow.add_edge("shopping_cart_agent_execution", "response_finalization_and_formatting")
        
        # Clarification exit path that terminates graph execution
        master_workflow.add_edge("clarification_request_handling", END)
        
        # Final response completion
        master_workflow.add_edge("response_finalization_and_formatting", END)
        
        return master_workflow
    
    def compile_graph(self) -> CompiledStateGraph:
        """
        Compile the master graph for execution.
        
        Returns:
            CompiledStateGraph: The compiled master routing graph ready for execution
        """
        if self._compiled_master_graph is None:
            master_graph = self.create_master_graph()
            self._compiled_master_graph = master_graph.compile()
            
            # Update metadata
            self._graph_metadata["compiled_at"] = datetime.now(timezone.utc)
            self._graph_metadata["compilation_successful"] = True
            
            self.logger.info("Master routing graph compiled successfully")
        
        return self._compiled_master_graph
    
    async def process_query(self, state: AgentState) -> AgentState:
        """
        Process a query through the master routing graph.
        
        This is the main entry point for query processing, handling:
        - Graph compilation and execution
        - Error handling with graceful fallbacks
        - Performance monitoring and logging
        
        Args:
            state: Initial agent state with user query
            
        Returns:
            AgentState: Final state with response and metadata
        """
        session_id = state.get("session_id", "unknown")
        query = state.get("current_query", "")
        
        try:
            self.logger.info(f"Starting master graph processing for session {session_id}")
            
            # Compile and execute the master graph
            compiled_master_graph = self.compile_graph()
            result = await compiled_master_graph.ainvoke(state)
            
            # Add processing metadata
            if "response_metadata" not in result:
                result["response_metadata"] = {}
            
            result["response_metadata"]["master_graph_processing"] = {
                "processing_successful": True,
                "processing_completed_at": datetime.now(timezone.utc).isoformat(),
                "graph_version": self._graph_metadata.get("version", "unknown")
            }
            
            self.logger.info(f"Master graph processing completed successfully for session {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Master graph processing failed for session {session_id}: {e}")
            
            # Create comprehensive error state with fallback response
            error_state = state.copy()
            error_state["error_state"] = f"Master graph processing error: {str(e)}"
            error_state["workflow_status"] = "error"
            error_state["final_response"] = (
                "I apologize, but I encountered an error processing your request. "
                "Please try rephrasing your question or try again in a moment."
            )
            error_state["updated_at"] = datetime.now(timezone.utc)
            
            # Add error metadata
            error_state["response_metadata"] = {
                "master_graph_processing": {
                    "processing_successful": False,
                    "error_message": str(e),
                    "error_occurred_at": datetime.now(timezone.utc).isoformat(),
                    "fallback_response_provided": True
                }
            }
            
            return error_state
    
    async def _execute_intent_router(self, state: AgentState) -> AgentState:
        """
        Execute intent classification and routing logic.
        
        This node analyzes the user's query to determine intent and route
        to the appropriate specialized agent or request clarification.
        
        Args:
            state: Current agent state with user query
            
        Returns:
            AgentState: Updated state with routing decision and metadata
        """
        session_id = state.get("session_id", "unknown")
        query = state.get("current_query", "")
        
        self.logger.info(f"Executing intent classification for session {session_id}: '{query[:50]}...'")
        
        try:
            # Execute routing through the intent router
            routed_state = await self.intent_router.route_message(state)
            
            # Add routing execution metadata
            if "routing_metadata" not in routed_state:
                routed_state["routing_metadata"] = {}
            
            routed_state["routing_metadata"]["router_execution"] = {
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "router_node_successful": True,
                "query_processed": query
            }
            
            self.logger.info(
                f"Intent routing completed for session {session_id}, "
                f"decision: {routed_state.get('routing_decision', 'unknown')}"
            )
            
            return routed_state
            
        except Exception as e:
            self.logger.error(f"Intent routing failed for session {session_id}: {e}")
            
            # Create fallback routing decision
            fallback_state = state.copy()
            fallback_state["routing_decision"] = "route_to_qa_agent"  # Fallback to QA
            fallback_state["target_agent"] = "product_qa_agent"
            fallback_state["error_state"] = f"Router execution error: {str(e)}"
            fallback_state["routing_metadata"] = {
                "router_execution": {
                    "executed_at": datetime.now(timezone.utc).isoformat(),
                    "router_node_successful": False,
                    "error_message": str(e),
                    "fallback_applied": True,
                    "fallback_target": "product_qa_agent"
                }
            }
            
            return fallback_state
    
    async def _execute_product_qa_agent(self, state: AgentState) -> AgentState:
        """
        Execute Product QA Agent workflow for product search and analysis.
        
        This node handles product-related queries including:
        - Product search and recommendations
        - Feature comparisons and analysis
        - Review summaries and insights
        - General product information requests
        
        Args:
            state: Current agent state with routing decision
            
        Returns:
            AgentState: Updated state with QA agent results and metadata
        """
        session_id = state.get("session_id", "unknown")
        query = state.get("current_query", "")
        
        self.logger.info(f"Executing Product QA Agent for session {session_id}")
        
        try:
            # Lazy-load QA agent graph to avoid circular dependencies
            if self.qa_agent_graph is None:
                if self.agent_builder is None:
                    raise ValueError("Agent builder not available for QA agent creation")
                
                self.logger.info("Creating Product QA Agent graph")
                self.qa_agent_graph = self.agent_builder.create_ambient_agent_graph()
            
            # Execute QA agent workflow
            qa_result = await self.qa_agent_graph.ainvoke(state)
            
            # Enhance result with agent execution metadata
            if "response_metadata" not in qa_result:
                qa_result["response_metadata"] = {}
            
            qa_result["response_metadata"]["specialized_agent_execution"] = {
                "agent_type": "product_qa_agent",
                "agent_name": "Product QA Agent",
                "execution_successful": True,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "routing_decision_honored": True,
                "query_type": "product_information"
            }
            
            # Legacy compatibility fields
            qa_result["response_metadata"]["agent_used"] = "qa_agent"
            qa_result["response_metadata"]["routing_successful"] = True
            
            self.logger.info(f"Product QA Agent execution completed successfully for session {session_id}")
            return qa_result
            
        except Exception as e:
            self.logger.error(f"Product QA Agent execution failed for session {session_id}: {e}")
            
            # Create comprehensive error state with fallback response
            error_state = state.copy()
            error_state["error_state"] = f"Product QA Agent execution error: {str(e)}"
            error_state["final_response"] = (
                "I apologize, but I encountered an issue searching for product information. "
                "Please try rephrasing your question or being more specific about what you're looking for."
            )
            error_state["workflow_status"] = "completed"
            
            # Add error metadata
            if "response_metadata" not in error_state:
                error_state["response_metadata"] = {}
            
            error_state["response_metadata"]["specialized_agent_execution"] = {
                "agent_type": "product_qa_agent",
                "agent_name": "Product QA Agent",
                "execution_successful": False,
                "error_message": str(e),
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "fallback_response_provided": True
            }
            
            # Legacy compatibility fields
            error_state["response_metadata"]["agent_used"] = "qa_agent"
            error_state["response_metadata"]["agent_error"] = True
            error_state["response_metadata"]["routing_successful"] = False
            
            return error_state
    
    async def _execute_shopping_cart_agent(self, state: AgentState) -> AgentState:
        """
        Execute Shopping Cart Agent workflow for cart management operations.
        
        This node handles shopping cart-related queries including:
        - Adding items to cart
        - Removing items from cart
        - Viewing cart contents
        - Clearing cart
        - Cart status and summary information
        
        Args:
            state: Current agent state with routing decision
            
        Returns:
            AgentState: Updated state with cart agent results and metadata
        """
        session_id = state.get("session_id", "unknown")
        query = state.get("current_query", "")
        
        self.logger.info(f"Executing Shopping Cart Agent for session {session_id}")
        
        try:
            # Execute shopping cart agent workflow
            cart_result = await self.shopping_cart_agent.process_query(state)
            
            # Enhance result with agent execution metadata
            if "response_metadata" not in cart_result:
                cart_result["response_metadata"] = {}
            
            cart_result["response_metadata"]["specialized_agent_execution"] = {
                "agent_type": "shopping_cart_agent",
                "agent_name": "Shopping Cart Agent",
                "execution_successful": True,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "routing_decision_honored": True,
                "query_type": "cart_management",
                "cart_operation_performed": cart_result.get("cart_updated", False)
            }
            
            # Legacy compatibility fields
            cart_result["response_metadata"]["agent_used"] = "cart_agent"
            cart_result["response_metadata"]["routing_successful"] = True
            
            self.logger.info(f"Shopping Cart Agent execution completed successfully for session {session_id}")
            return cart_result
            
        except Exception as e:
            self.logger.error(f"Shopping Cart Agent execution failed for session {session_id}: {e}")
            
            # Create comprehensive error state with fallback response
            error_state = state.copy()
            error_state["error_state"] = f"Shopping Cart Agent execution error: {str(e)}"
            error_state["final_response"] = (
                "I apologize, but I encountered an issue with cart management. "
                "Please try your cart operation again or contact support if the problem persists."
            )
            error_state["workflow_status"] = "completed"
            
            # Add error metadata
            if "response_metadata" not in error_state:
                error_state["response_metadata"] = {}
            
            error_state["response_metadata"]["specialized_agent_execution"] = {
                "agent_type": "shopping_cart_agent",
                "agent_name": "Shopping Cart Agent",
                "execution_successful": False,
                "error_message": str(e),
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "fallback_response_provided": True
            }
            
            # Legacy compatibility fields
            error_state["response_metadata"]["agent_used"] = "cart_agent"
            error_state["response_metadata"]["agent_error"] = True
            error_state["response_metadata"]["routing_successful"] = False
            
            return error_state
    
    async def _handle_clarification_request(self, state: AgentState) -> AgentState:
        """
        Handle clarification requests for ambiguous or unclear queries.
        
        This node processes cases where the user's intent cannot be determined
        with sufficient confidence, providing clarifying questions and terminating
        the workflow to await user response.
        
        Args:
            state: Current agent state with clarification response
            
        Returns:
            AgentState: Final state with clarification message and termination
        """
        session_id = state.get("session_id", "unknown")
        query = state.get("current_query", "")
        
        self.logger.info(f"Handling clarification request for session {session_id}")
        
        # Clarification response should already be set by the router
        # Ensure workflow is properly completed and terminated
        updated_state = update_state_step(
            state,
            "clarification_request_handling",
            workflow_status="completed"
        )
        
        # Add comprehensive clarification metadata
        if "response_metadata" not in updated_state:
            updated_state["response_metadata"] = {}
        
        updated_state["response_metadata"]["clarification_handling"] = {
            "clarification_requested": True,
            "workflow_terminated": True,
            "termination_reason": "user_clarification_required",
            "handled_at": datetime.now(timezone.utc).isoformat(),
            "original_query": query,
            "suggested_questions": updated_state.get("suggested_questions", []),
            "clarification_attempts": updated_state.get("clarification_attempts", 0)
        }
        
        # Legacy compatibility fields
        updated_state["response_metadata"]["clarification_requested"] = True
        updated_state["response_metadata"]["workflow_terminated"] = True
        
        self.logger.info(
            f"Clarification request handled for session {session_id}, "
            f"workflow terminated for user response"
        )
        
        return updated_state
    
    async def _finalize_and_format_response(self, state: AgentState) -> AgentState:
        """
        Finalize response with consistent formatting and comprehensive metadata.
        
        This node provides the final processing step for successful agent executions,
        ensuring consistent response formatting, complete metadata, and proper
        workflow completion status.
        
        Args:
            state: Current agent state after specialized agent execution
            
        Returns:
            AgentState: Final state with formatted response and complete metadata
        """
        session_id = state.get("session_id", "unknown")
        agent_used = state.get("response_metadata", {}).get("agent_used", "unknown")
        
        self.logger.info(f"Finalizing response for session {session_id}, agent: {agent_used}")
        
        # Ensure response metadata structure exists
        if "response_metadata" not in state:
            state["response_metadata"] = {}
        
        # Add comprehensive routing and processing metadata
        state["response_metadata"]["master_graph_finalization"] = {
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "routing_decision": state.get("routing_decision", "unknown"),
            "intent_confidence": state.get("intent_confidence", 0.0),
            "target_agent": state.get("target_agent", "unknown"),
            "processing_successful": True,
            "response_formatted": True
        }
        
        # Legacy compatibility fields
        state["response_metadata"]["routing_decision"] = state.get("routing_decision", "unknown")
        state["response_metadata"]["intent_confidence"] = state.get("intent_confidence", 0.0)
        state["response_metadata"]["routing_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add workflow completion metadata
        state["response_metadata"]["workflow_completion"] = {
            "completed_successfully": True,
            "completion_timestamp": datetime.now(timezone.utc).isoformat(),
            "final_step": "response_finalization_and_formatting",
            "total_processing_nodes": self._count_processing_nodes(state)
        }
        
        # Ensure workflow is marked as completed
        updated_state = update_state_step(
            state,
            "response_finalization_and_formatting",
            workflow_status="completed"
        )
        
        self.logger.info(
            f"Response finalization completed for session {session_id}, "
            f"agent: {agent_used}, workflow: completed"
        )
        
        return updated_state
    
    def _count_processing_nodes(self, state: AgentState) -> int:
        """
        Count the number of processing nodes executed in the workflow.
        
        Args:
            state: Current agent state
            
        Returns:
            int: Number of processing nodes executed
        """
        # Count based on intermediate steps or routing path
        intermediate_steps = state.get("intermediate_steps", [])
        if intermediate_steps:
            return len(intermediate_steps)
        
        # Fallback: estimate based on routing decision
        routing_decision = state.get("routing_decision", "")
        if routing_decision == "request_clarification":
            return 2  # Router + Clarification
        elif routing_decision in ["route_to_qa_agent", "route_to_cart_agent"]:
            return 3  # Router + Agent + Finalization
        else:
            return 1  # At least router
    
    def _determine_agent_routing_decision(self, state: AgentState) -> str:
        """
        Determine agent routing decision from state with improved validation.
        
        This method maps the router's intent classification results to specific
        agent execution paths, providing clear routing logic and comprehensive
        validation with fallback handling.
        
        Args:
            state: Current agent state with routing decision
            
        Returns:
            str: Routing decision for conditional edges
        """
        routing_decision = state.get("routing_decision", "clarification")
        intent_confidence = state.get("intent_confidence", 0.0)
        session_id = state.get("session_id", "unknown")
        
        # Map router decisions to descriptive edge names
        routing_map = {
            "qa": "route_to_qa_agent",
            "cart": "route_to_cart_agent",
            "clarification": "request_clarification"
        }
        
        # Validate and map routing decision
        if routing_decision in routing_map:
            mapped_decision = routing_map[routing_decision]
            
            self.logger.info(
                f"Routing decision for session {session_id}: {routing_decision} -> {mapped_decision} "
                f"(confidence: {intent_confidence:.2f})"
            )
            
            return mapped_decision
        
        # Handle invalid routing decisions with fallback
        self.logger.warning(
            f"Invalid routing decision '{routing_decision}' for session {session_id}, "
            f"defaulting to clarification request"
        )
        
        return "request_clarification"
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing statistics from the intent router.
        
        Returns:
            Dict[str, Any]: Detailed routing statistics and performance metrics
        """
        base_stats = self.intent_router.get_routing_stats()
        
        # Enhance with master graph metadata
        enhanced_stats = {
            **base_stats,
            "master_graph_metadata": {
                "graph_version": self._graph_metadata.get("version", "unknown"),
                "created_at": self._graph_metadata.get("created_at"),
                "compiled": self._compiled_master_graph is not None,
                "compilation_timestamp": self._graph_metadata.get("compiled_at")
            },
            "agent_availability": {
                "product_qa_agent": self.agent_builder is not None,
                "shopping_cart_agent": True,
                "total_available_agents": 2
            }
        }
        
        return enhanced_stats
    
    def reset_routing_statistics(self) -> None:
        """Reset routing statistics and performance metrics."""
        self.intent_router.reset_routing_stats()
        self.logger.info("Master graph routing statistics reset")
    
    def get_master_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the master graph structure and status.
        
        Returns:
            Dict[str, Any]: Complete master graph information and metadata
        """
        return {
            "graph_metadata": {
                "graph_type": "master_routing_graph",
                "graph_name": "Master Agent Routing Graph",
                "version": self._graph_metadata.get("version", "1.0.0"),
                "description": "Orchestrates intelligent routing between specialized agents"
            },
            "compilation_status": {
                "compiled": self._compiled_master_graph is not None,
                "compilation_successful": self._graph_metadata.get("compilation_successful", False),
                "compiled_at": self._graph_metadata.get("compiled_at")
            },
            "available_agents": {
                "product_qa_agent": {
                    "name": "Product QA Agent",
                    "description": "Handles product search, analysis, and recommendations",
                    "available": self.agent_builder is not None,
                    "lazy_loaded": self.qa_agent_graph is None
                },
                "shopping_cart_agent": {
                    "name": "Shopping Cart Agent", 
                    "description": "Manages shopping cart operations and state",
                    "available": True,
                    "initialized": True
                }
            },
            "routing_configuration": {
                "intent_classification_enabled": True,
                "clarification_handling_enabled": True,
                "confidence_threshold": self.config.get("router", {}).get("confidence_threshold", 0.7),
                "max_clarification_attempts": self.config.get("clarification", {}).get("max_clarification_attempts", 3)
            },
            "workflow_nodes": {
                "total_nodes": 5,
                "node_names": [
                    "intent_classification_and_routing",
                    "product_qa_agent_execution", 
                    "shopping_cart_agent_execution",
                    "clarification_request_handling",
                    "response_finalization_and_formatting"
                ],
                "routing_edges": [
                    "route_to_qa_agent",
                    "route_to_cart_agent", 
                    "request_clarification"
                ]
            },
            "performance_metrics": self.get_routing_statistics(),
            "configuration": self.config
        }
    
    def get_agent_hierarchy_documentation(self) -> Dict[str, Any]:
        """
        Get documentation of the agent hierarchy and relationships.
        
        Returns:
            Dict[str, Any]: Agent hierarchy documentation and relationship mapping
        """
        return {
            "hierarchy_structure": {
                "master_graph": {
                    "level": 0,
                    "role": "orchestration",
                    "description": "Top-level routing and orchestration",
                    "manages": ["intent_router", "specialized_agents"]
                },
                "intent_router": {
                    "level": 1,
                    "role": "routing",
                    "description": "Intent classification and routing decisions",
                    "components": ["intent_classifier", "clarification_handler"]
                },
                "specialized_agents": {
                    "level": 1,
                    "role": "execution",
                    "description": "Domain-specific query processing",
                    "agents": ["product_qa_agent", "shopping_cart_agent"]
                }
            },
            "agent_relationships": {
                "product_qa_agent": {
                    "handles": ["product_search", "product_analysis", "recommendations", "comparisons"],
                    "tools": ["vector_search_mcp", "product_analysis_mcp"],
                    "tool_type": "mcp_tools",
                    "fallback_for": ["unclear_product_queries"]
                },
                "shopping_cart_agent": {
                    "handles": ["cart_add", "cart_remove", "cart_list", "cart_clear"],
                    "tools": ["add_to_cart", "remove_from_cart", "list_cart", "clear_cart"],
                    "tool_type": "function_calling",
                    "state_management": "persistent_database"
                }
            },
            "routing_logic": {
                "intent_classification": {
                    "method": "keyword_and_context_analysis",
                    "confidence_threshold": self.config.get("router", {}).get("confidence_threshold", 0.7),
                    "fallback_strategy": "clarification_request"
                },
                "clarification_handling": {
                    "triggers": ["low_confidence", "ambiguous_intent", "multiple_intents"],
                    "max_attempts": self.config.get("clarification", {}).get("max_clarification_attempts", 3),
                    "fallback_agent": "product_qa_agent"
                }
            },
            "workflow_patterns": {
                "successful_routing": [
                    "intent_classification_and_routing",
                    "specialized_agent_execution", 
                    "response_finalization_and_formatting"
                ],
                "clarification_required": [
                    "intent_classification_and_routing",
                    "clarification_request_handling"
                ],
                "error_fallback": [
                    "intent_classification_and_routing",
                    "product_qa_agent_execution",
                    "response_finalization_and_formatting"
                ]
            }
        }
    
    # Legacy compatibility methods
    def get_routing_stats(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.get_routing_statistics()
    
    def reset_routing_stats(self) -> None:
        """Legacy method for backward compatibility."""
        return self.reset_routing_statistics()
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.get_master_graph_info()