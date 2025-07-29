"""
Base agent class for LangGraph workflows.
Provides common functionality for all agent implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph

from .state_schemas import AgentState, validate_state
from .utils import log_agent_step, create_error_response

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all LangGraph agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base agent with configuration."""
        self.config = config
        self.graph: Optional[CompiledStateGraph] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph. Must be implemented by subclasses."""
        pass
    
    def compile_graph(self) -> CompiledStateGraph:
        """Compile the agent's workflow graph."""
        if self.graph is None:
            graph = self.create_graph()
            self.graph = graph.compile()
        return self.graph
    
    async def process_query(self, state: AgentState) -> AgentState:
        """Process a query using the agent's workflow."""
        
        if not validate_state(state):
            raise ValueError("Invalid agent state provided")
        
        try:
            log_agent_step(
                state["session_id"], 
                "agent_start",
                {"agent_type": self.__class__.__name__}
            )
            
            # Compile graph if not already done
            compiled_graph = self.compile_graph()
            
            # Execute the workflow
            result = await compiled_graph.ainvoke(state)
            
            log_agent_step(
                state["session_id"],
                "agent_complete",
                {"final_step": result.get("current_step", "unknown")}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent processing failed: {e}")
            
            # Update state with error information
            error_state = state.copy()
            error_state["error_state"] = str(e)
            error_state["workflow_status"] = "error"
            error_state["updated_at"] = datetime.utcnow()
            
            return error_state
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_type": self.__class__.__name__,
            "config": self.config,
            "graph_compiled": self.graph is not None
        }


class ProductSearchAgent(BaseAgent):
    """Agent specialized for product search and information retrieval."""
    
    def create_graph(self) -> StateGraph:
        """Create product search workflow graph."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("search_products", self._search_products_node)
        workflow.add_node("format_results", self._format_results_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Define edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "search_products")
        workflow.add_edge("search_products", "format_results")
        workflow.add_edge("format_results", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow
    
    async def _analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze the user query for product search."""
        
        from .utils import extract_entities_from_query, classify_query_intent
        
        query = state["current_query"]
        
        # Extract entities and classify intent
        entities = extract_entities_from_query(query)
        intent = classify_query_intent(query)
        
        # Update state
        updated_state = state.copy()
        updated_state["extracted_entities"] = entities
        updated_state["query_intent"] = intent
        updated_state["current_step"] = "analyze_query"
        updated_state["updated_at"] = datetime.utcnow()
        
        log_agent_step(
            state["session_id"],
            "query_analyzed",
            {"entities": entities, "intent": intent}
        )
        
        return updated_state
    
    async def _search_products_node(self, state: AgentState) -> AgentState:
        """Search for products based on query analysis."""
        
        # This will be implemented when we create the vector search tool
        # For now, return state with placeholder results
        
        updated_state = state.copy()
        updated_state["search_results"] = {"placeholder": "search_results"}
        updated_state["current_step"] = "search_products"
        updated_state["updated_at"] = datetime.utcnow()
        
        log_agent_step(
            state["session_id"],
            "products_searched",
            {"query": state["current_query"]}
        )
        
        return updated_state
    
    async def _format_results_node(self, state: AgentState) -> AgentState:
        """Format search results for response generation."""
        
        updated_state = state.copy()
        updated_state["context_for_llm"] = "Formatted search results will go here"
        updated_state["current_step"] = "format_results"
        updated_state["updated_at"] = datetime.utcnow()
        
        return updated_state
    
    async def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final response based on search results."""
        
        updated_state = state.copy()
        updated_state["final_response"] = "Generated response will go here"
        updated_state["workflow_status"] = "completed"
        updated_state["current_step"] = "generate_response"
        updated_state["updated_at"] = datetime.utcnow()
        
        return updated_state


class ReviewAnalysisAgent(BaseAgent):
    """Agent specialized for review analysis and sentiment processing."""
    
    def create_graph(self) -> StateGraph:
        """Create review analysis workflow graph."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("search_reviews", self._search_reviews_node)
        workflow.add_node("analyze_sentiment", self._analyze_sentiment_node)
        workflow.add_node("generate_summary", self._generate_summary_node)
        
        # Define edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "search_reviews")
        workflow.add_edge("search_reviews", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        return workflow
    
    async def _analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze query for review-specific processing."""
        
        updated_state = state.copy()
        updated_state["current_step"] = "analyze_query"
        updated_state["updated_at"] = datetime.utcnow()
        
        return updated_state
    
    async def _search_reviews_node(self, state: AgentState) -> AgentState:
        """Search for relevant reviews."""
        
        updated_state = state.copy()
        updated_state["current_step"] = "search_reviews"
        updated_state["updated_at"] = datetime.utcnow()
        
        return updated_state
    
    async def _analyze_sentiment_node(self, state: AgentState) -> AgentState:
        """Analyze sentiment of retrieved reviews."""
        
        updated_state = state.copy()
        updated_state["current_step"] = "analyze_sentiment"
        updated_state["updated_at"] = datetime.utcnow()
        
        return updated_state
    
    async def _generate_summary_node(self, state: AgentState) -> AgentState:
        """Generate summary of review analysis."""
        
        updated_state = state.copy()
        updated_state["final_response"] = "Review analysis summary will go here"
        updated_state["workflow_status"] = "completed"
        updated_state["current_step"] = "generate_summary"
        updated_state["updated_at"] = datetime.utcnow()
        
        return updated_state