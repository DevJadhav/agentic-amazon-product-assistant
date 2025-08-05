"""
Integration tests for router node with existing agent infrastructure.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from ..core.router.master_graph import MasterAgentGraph, ShoppingCartAgent
from ..core.router.router_node import RouterNode
from ..core.router.intent_classifier import IntentClassifier, IntentResult
from ..core.router.clarification_handler import ClarificationHandler
from ..core.agent_builder import AgentGraphBuilder
from ..core.state_schemas import create_initial_state, AgentState


class TestShoppingCartAgent:
    """Test cases for Shopping Cart Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"test": True}
        self.agent = ShoppingCartAgent(self.config)
        self.session_id = "test_cart_session"
    
    def test_cart_agent_initialization(self):
        """Test cart agent initialization."""
        assert self.agent.config == self.config
        assert self.agent.cart_manager is None
        assert hasattr(self.agent, 'logger')
    
    def test_create_graph(self):
        """Test cart agent graph creation."""
        graph = self.agent.create_graph()
        
        # Verify graph structure
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        
        # Check that required nodes exist
        expected_nodes = [
            "analyze_cart_request",
            "execute_cart_operation", 
            "generate_cart_response",
            "update_cart_state"
        ]
        
        for node in expected_nodes:
            assert node in graph.nodes
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_add(self):
        """Test cart request analysis for add operations."""
        state = create_initial_state(
            self.session_id,
            "add wireless headphones to cart",
            extracted_entities=["wireless headphones"]
        )
        
        result = await self.agent._analyze_cart_request(state)
        
        assert result["current_step"] == "analyze_cart_request"
        assert result["cart_operation"] == "add"
        assert "entities" in result["cart_operation_params"]
        assert "wireless headphones" in result["cart_operation_params"]["entities"]
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_remove(self):
        """Test cart request analysis for remove operations."""
        state = create_initial_state(
            self.session_id,
            "remove laptop from cart"
        )
        
        result = await self.agent._analyze_cart_request(state)
        
        assert result["cart_operation"] == "remove"
        assert result["cart_operation_params"]["original_query"] == "remove laptop from cart"
    
    @pytest.mark.asyncio
    async def test_analyze_cart_request_list(self):
        """Test cart request analysis for list operations."""
        state = create_initial_state(
            self.session_id,
            "show me my cart"
        )
        
        result = await self.agent._analyze_cart_request(state)
        
        assert result["cart_operation"] == "list"
    
    @pytest.mark.asyncio
    async def test_execute_cart_operation_add(self):
        """Test cart operation execution for add."""
        state = create_initial_state(
            self.session_id,
            "test query",
            cart_operation="add",
            cart_operation_params={"entities": ["test product"]}
        )
        
        result = await self.agent._execute_cart_operation(state)
        
        assert result["current_step"] == "execute_cart_operation"
        assert result["cart_operation_success"] is True
        assert result["cart_updated"] is True
        assert result["cart_operation_result"]["operation"] == "add"
    
    @pytest.mark.asyncio
    async def test_execute_cart_operation_list(self):
        """Test cart operation execution for list."""
        state = create_initial_state(
            self.session_id,
            "test query",
            cart_operation="list"
        )
        
        result = await self.agent._execute_cart_operation(state)
        
        assert result["cart_operation_success"] is True
        assert result["cart_updated"] is False  # List doesn't modify cart
        assert "cart_contents" in result["cart_operation_result"]
    
    @pytest.mark.asyncio
    async def test_generate_cart_response_success(self):
        """Test cart response generation for successful operations."""
        state = create_initial_state(
            self.session_id,
            "test query",
            cart_operation_result={
                "success": True,
                "operation": "add",
                "message": "Item added successfully"
            }
        )
        
        result = await self.agent._generate_cart_response(state)
        
        assert result["current_step"] == "generate_cart_response"
        assert "final_response" in result
        assert "add" in result["final_response"].lower()
        assert result["cart_operation_message"] == result["final_response"]
    
    @pytest.mark.asyncio
    async def test_generate_cart_response_empty_cart(self):
        """Test cart response generation for empty cart."""
        state = create_initial_state(
            self.session_id,
            "test query",
            cart_operation_result={
                "success": True,
                "operation": "list",
                "cart_contents": []
            }
        )
        
        result = await self.agent._generate_cart_response(state)
        
        assert "empty" in result["final_response"].lower()
    
    @pytest.mark.asyncio
    async def test_update_cart_state(self):
        """Test cart state update."""
        state = create_initial_state(
            self.session_id,
            "test query",
            cart_operation="add"
        )
        
        result = await self.agent._update_cart_state(state)
        
        assert result["current_step"] == "update_cart_state"
        assert result["workflow_status"] == "completed"
        assert "response_metadata" in result
        assert result["response_metadata"]["cart_operation_performed"] is True
        assert result["response_metadata"]["cart_operation_type"] == "add"


class TestMasterAgentGraph:
    """Test cases for Master Agent Graph."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.7},
            "classifier": {"test": True},
            "clarification": {"max_clarification_attempts": 3}
        }
        # Create a mock agent builder to avoid circular imports
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
        self.session_id = "test_master_session"
    
    def test_master_graph_initialization(self):
        """Test master graph initialization."""
        assert self.master_graph.config == self.config
        assert isinstance(self.master_graph.router_node, RouterNode)
        assert self.master_graph.agent_builder is self.mock_agent_builder
        assert isinstance(self.master_graph.cart_agent, ShoppingCartAgent)
        assert self.master_graph._compiled_graph is None
    
    def test_create_master_graph(self):
        """Test master graph creation."""
        graph = self.master_graph.create_master_graph()
        
        # Verify graph structure
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        
        # Check that required nodes exist
        expected_nodes = [
            "router",
            "qa_agent",
            "cart_agent",
            "clarification",
            "finalize_response"
        ]
        
        for node in expected_nodes:
            assert node in graph.nodes
    
    def test_compile_graph(self):
        """Test master graph compilation."""
        compiled_graph = self.master_graph.compile_graph()
        
        assert compiled_graph is not None
        assert self.master_graph._compiled_graph is not None
        assert hasattr(compiled_graph, 'ainvoke')
        
        # Second call should return cached version
        compiled_graph2 = self.master_graph.compile_graph()
        assert compiled_graph is compiled_graph2
    
    def test_route_decision_valid(self):
        """Test route decision with valid routing decisions."""
        # Test QA routing
        state = create_initial_state(self.session_id, "test", routing_decision="qa")
        decision = self.master_graph._route_decision(state)
        assert decision == "qa"
        
        # Test cart routing
        state = create_initial_state(self.session_id, "test", routing_decision="cart")
        decision = self.master_graph._route_decision(state)
        assert decision == "cart"
        
        # Test clarification routing
        state = create_initial_state(self.session_id, "test", routing_decision="clarification")
        decision = self.master_graph._route_decision(state)
        assert decision == "clarification"
    
    def test_route_decision_invalid(self):
        """Test route decision with invalid routing decisions."""
        state = create_initial_state(self.session_id, "test", routing_decision="invalid")
        decision = self.master_graph._route_decision(state)
        assert decision == "clarification"  # Should default to clarification
        
        # Test missing routing decision
        state = create_initial_state(self.session_id, "test")
        decision = self.master_graph._route_decision(state)
        assert decision == "clarification"
    
    @pytest.mark.asyncio
    async def test_router_node_execution(self):
        """Test router node execution."""
        state = create_initial_state(
            self.session_id,
            "add headphones to cart"
        )
        
        # Mock the router node to return a cart routing decision
        with patch.object(self.master_graph.router_node, 'route_message') as mock_route:
            mock_route.return_value = state.copy()
            mock_route.return_value["routing_decision"] = "cart"
            mock_route.return_value["user_intent"] = "cart"
            mock_route.return_value["intent_confidence"] = 0.9
            
            result = await self.master_graph._router_node(state)
            
            assert result["routing_decision"] == "cart"
            assert result["user_intent"] == "cart"
            assert result["intent_confidence"] == 0.9
            mock_route.assert_called_once_with(state)
    
    @pytest.mark.asyncio
    async def test_execute_cart_agent(self):
        """Test cart agent execution."""
        state = create_initial_state(
            self.session_id,
            "add laptop to cart",
            routing_decision="cart"
        )
        
        result = await self.master_graph._execute_cart_agent(state)
        
        assert result["workflow_status"] == "completed"
        assert "response_metadata" in result
        assert result["response_metadata"]["agent_used"] == "cart_agent"
        assert result["response_metadata"]["routing_successful"] is True
        assert "final_response" in result
    
    @pytest.mark.asyncio
    async def test_execute_qa_agent_error_handling(self):
        """Test QA agent execution with error handling."""
        state = create_initial_state(
            self.session_id,
            "what are good laptops",
            routing_decision="qa"
        )
        
        # Mock the QA agent to raise an exception
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        self.mock_agent_builder.create_ambient_agent_graph.return_value = mock_graph
        
        result = await self.master_graph._execute_qa_agent(state)
        
        assert "error_state" in result
        assert "QA agent error" in result["error_state"]
        assert "final_response" in result
        assert "response_metadata" in result
        assert result["response_metadata"]["agent_used"] == "qa_agent"
        assert result["response_metadata"]["agent_error"] is True
    
    @pytest.mark.asyncio
    async def test_execute_cart_agent_error_handling(self):
        """Test cart agent execution with error handling."""
        state = create_initial_state(
            self.session_id,
            "add item to cart",
            routing_decision="cart"
        )
        
        # Mock the cart agent to raise an exception
        with patch.object(self.master_graph.cart_agent, 'process_query') as mock_process:
            mock_process.side_effect = Exception("Cart error")
            
            result = await self.master_graph._execute_cart_agent(state)
            
            assert "error_state" in result
            assert "Cart agent error" in result["error_state"]
            assert "final_response" in result
            assert "response_metadata" in result
            assert result["response_metadata"]["agent_used"] == "cart_agent"
            assert result["response_metadata"]["agent_error"] is True
    
    @pytest.mark.asyncio
    async def test_handle_clarification(self):
        """Test clarification handling."""
        state = create_initial_state(
            self.session_id,
            "unclear query",
            routing_decision="clarification",
            final_response="Could you clarify what you're looking for?"
        )
        
        result = await self.master_graph._handle_clarification(state)
        
        assert result["current_step"] == "clarification"
        assert result["workflow_status"] == "completed"
        assert "response_metadata" in result
        assert result["response_metadata"]["clarification_requested"] is True
        assert result["response_metadata"]["workflow_terminated"] is True
    
    @pytest.mark.asyncio
    async def test_finalize_response(self):
        """Test response finalization."""
        state = create_initial_state(
            self.session_id,
            "test query",
            routing_decision="qa",
            intent_confidence=0.8,
            final_response="Test response"
        )
        
        result = await self.master_graph._finalize_response(state)
        
        assert result["current_step"] == "finalize_response"
        assert result["workflow_status"] == "completed"
        assert "response_metadata" in result
        assert result["response_metadata"]["routing_decision"] == "qa"
        assert result["response_metadata"]["intent_confidence"] == 0.8
        assert "routing_timestamp" in result["response_metadata"]
    
    @pytest.mark.asyncio
    async def test_process_query_success(self):
        """Test successful query processing through master graph."""
        state = create_initial_state(
            self.session_id,
            "show my cart"
        )
        
        # Mock the compiled graph
        mock_graph = Mock()
        mock_result = state.copy()
        mock_result["workflow_status"] = "completed"
        mock_result["final_response"] = "Your cart is empty"
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)
        
        with patch.object(self.master_graph, 'compile_graph', return_value=mock_graph):
            result = await self.master_graph.process_query(state)
            
            assert result["workflow_status"] == "completed"
            assert result["final_response"] == "Your cart is empty"
            mock_graph.ainvoke.assert_called_once_with(state)
    
    @pytest.mark.asyncio
    async def test_process_query_error(self):
        """Test query processing with error handling."""
        state = create_initial_state(
            self.session_id,
            "test query"
        )
        
        # Mock the compiled graph to raise an exception
        with patch.object(self.master_graph, 'compile_graph', side_effect=Exception("Graph error")):
            result = await self.master_graph.process_query(state)
            
            assert "error_state" in result
            assert "Master graph error" in result["error_state"]
            assert result["workflow_status"] == "error"
            assert "final_response" in result
    
    def test_get_routing_stats(self):
        """Test routing statistics retrieval."""
        stats = self.master_graph.get_routing_stats()
        
        assert isinstance(stats, dict)
        assert "total_routes" in stats
        assert "qa_routes" in stats
        assert "cart_routes" in stats
        assert "clarifications" in stats
    
    def test_reset_routing_stats(self):
        """Test routing statistics reset."""
        # Get initial stats
        initial_stats = self.master_graph.get_routing_stats()
        
        # Reset stats
        self.master_graph.reset_routing_stats()
        
        # Verify reset
        reset_stats = self.master_graph.get_routing_stats()
        assert reset_stats["total_routes"] == 0
        assert reset_stats["qa_routes"] == 0
        assert reset_stats["cart_routes"] == 0
        assert reset_stats["clarifications"] == 0
    
    def test_get_graph_info(self):
        """Test graph information retrieval."""
        info = self.master_graph.get_graph_info()
        
        assert info["graph_type"] == "master_routing_graph"
        assert info["compiled"] is False  # Not compiled yet
        assert "qa_agent" in info["available_agents"]
        assert "cart_agent" in info["available_agents"]
        assert "routing_stats" in info
        assert info["config"] == self.config


class TestAgentBuilderIntegration:
    """Test integration of master graph with agent builder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"test": True}
        self.agent_builder = AgentGraphBuilder(self.config)
    
    def test_create_master_routing_graph(self):
        """Test master routing graph creation through agent builder."""
        graph = self.agent_builder.create_master_routing_graph()
        
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
    
    def test_create_product_qa_agent_graph(self):
        """Test Product QA Agent graph creation through agent builder."""
        graph = self.agent_builder.create_product_qa_agent_graph()
        
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
    
    def test_create_shopping_cart_agent_graph(self):
        """Test Shopping Cart Agent graph creation through agent builder."""
        graph = self.agent_builder.create_shopping_cart_agent_graph()
        
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
    
    def test_get_available_graphs_includes_master(self):
        """Test that available graphs includes master routing graph with improved naming."""
        graphs = self.agent_builder.get_available_graphs()
        
        assert "master_routing_graph" in graphs
        assert "routing" in graphs["master_routing_graph"].lower()
        assert "product_qa_agent" in graphs
        assert "shopping_cart_agent" in graphs
        assert "ambient_agent" in graphs  # Legacy graphs should still be there
        assert "product_search_agent" in graphs
    
    def test_get_agent_hierarchy_mapping(self):
        """Test agent hierarchy mapping retrieval."""
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        
        assert "orchestration_layer" in hierarchy
        assert "specialized_agents" in hierarchy
        assert "legacy_agents" in hierarchy
        assert "routing_patterns" in hierarchy
        
        # Check orchestration layer
        orchestration = hierarchy["orchestration_layer"]
        assert "master_routing_graph" in orchestration
        
        # Check specialized agents
        specialized = hierarchy["specialized_agents"]
        assert "product_qa_agent" in specialized
        assert "shopping_cart_agent" in specialized
        
        # Check legacy agents
        legacy = hierarchy["legacy_agents"]
        assert "ambient_agent" in legacy
        assert legacy["ambient_agent"]["status"] == "legacy"
    
    def test_get_graph_naming_conventions(self):
        """Test graph naming conventions documentation."""
        conventions = self.agent_builder.get_graph_naming_conventions()
        
        assert "node_naming_conventions" in conventions
        assert "edge_naming_conventions" in conventions
        assert "agent_naming_conventions" in conventions
        assert "method_naming_conventions" in conventions
        
        # Check node naming conventions
        node_conventions = conventions["node_naming_conventions"]
        assert "pattern" in node_conventions
        assert "examples" in node_conventions
        assert "guidelines" in node_conventions
        
        # Check examples are present
        examples = node_conventions["examples"]
        assert "intent_classification_and_routing" in examples
        assert "product_qa_agent_execution" in examples


class TestEndToEndRouterIntegration:
    """End-to-end integration tests for router with agents."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.7},
            "classifier": {"confidence_threshold": 0.7},
            "clarification": {"max_clarification_attempts": 3}
        }
        # Create a mock agent builder for end-to-end tests
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
        self.session_id = "e2e_test_session"
    
    @pytest.mark.asyncio
    async def test_cart_intent_end_to_end(self):
        """Test end-to-end cart intent processing."""
        state = create_initial_state(
            self.session_id,
            "add wireless headphones to my cart"
        )
        
        # Mock the router to return cart intent
        with patch.object(self.master_graph.router_node, 'route_message') as mock_route:
            routed_state = state.copy()
            routed_state["routing_decision"] = "cart"
            routed_state["user_intent"] = "cart"
            routed_state["intent_confidence"] = 0.9
            mock_route.return_value = routed_state
            
            result = await self.master_graph.process_query(state)
            
            # Verify cart processing occurred
            assert result["workflow_status"] == "completed"
            assert "final_response" in result
            assert "response_metadata" in result
            assert result["response_metadata"]["agent_used"] == "cart_agent"
    
    @pytest.mark.asyncio
    async def test_qa_intent_end_to_end(self):
        """Test end-to-end QA intent processing."""
        state = create_initial_state(
            self.session_id,
            "what are the best wireless headphones?"
        )
        
        # Mock the router to return QA intent
        with patch.object(self.master_graph.router_node, 'route_message') as mock_route:
            routed_state = state.copy()
            routed_state["routing_decision"] = "qa"
            routed_state["user_intent"] = "qa"
            routed_state["intent_confidence"] = 0.8
            mock_route.return_value = routed_state
            
            # Mock the QA agent to avoid external dependencies
            mock_qa_result = state.copy()
            mock_qa_result["final_response"] = "Here are the best wireless headphones..."
            mock_qa_result["workflow_status"] = "completed"
            
            mock_graph = Mock()
            mock_graph.ainvoke = AsyncMock(return_value=mock_qa_result)
            self.mock_agent_builder.create_ambient_agent_graph.return_value = mock_graph
            
            result = await self.master_graph.process_query(state)
            
            # Verify QA processing occurred
            assert result["workflow_status"] == "completed"
            assert "final_response" in result
            assert "response_metadata" in result
            assert result["response_metadata"]["agent_used"] == "qa_agent"
    
    @pytest.mark.asyncio
    async def test_clarification_end_to_end(self):
        """Test end-to-end clarification processing."""
        state = create_initial_state(
            self.session_id,
            "help me"  # Ambiguous query
        )
        
        # Mock the router to return clarification
        with patch.object(self.master_graph.router_node, 'route_message') as mock_route:
            clarification_state = state.copy()
            clarification_state["routing_decision"] = "clarification"
            clarification_state["final_response"] = "Could you please clarify what you need help with?"
            mock_route.return_value = clarification_state
            
            result = await self.master_graph.process_query(state)
            
            # Verify clarification processing occurred
            assert result["workflow_status"] == "completed"
            assert "final_response" in result
            assert "response_metadata" in result
            assert result["response_metadata"]["clarification_requested"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])