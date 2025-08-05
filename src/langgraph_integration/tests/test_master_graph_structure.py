"""
Unit tests for master graph construction and edge routing.
Tests the improved organization and naming conventions.
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


class TestMasterGraphStructure:
    """Test cases for master graph structure and organization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.7},
            "classifier": {"test": True},
            "clarification": {"max_clarification_attempts": 3},
            "cart_agent": {"test": True}
        }
        # Create a mock agent builder to avoid circular imports
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
        self.session_id = "test_master_graph_session"
    
    def test_master_graph_initialization_with_improved_naming(self):
        """Test master graph initialization with improved component naming."""
        assert self.master_graph.config == self.config
        assert isinstance(self.master_graph.intent_router, RouterNode)
        assert self.master_graph.agent_builder is self.mock_agent_builder
        assert isinstance(self.master_graph.shopping_cart_agent, ShoppingCartAgent)
        assert self.master_graph._compiled_master_graph is None
        
        # Check metadata structure
        assert "created_at" in self.master_graph._graph_metadata
        assert self.master_graph._graph_metadata["version"] == "1.0.0"
        assert self.master_graph._graph_metadata["agent_count"] == 2
        assert self.master_graph._graph_metadata["routing_enabled"] is True
    
    def test_create_master_graph_with_descriptive_names(self):
        """Test master graph creation with descriptive node and edge names."""
        graph = self.master_graph.create_master_graph()
        
        # Verify graph structure
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        
        # Check that nodes have descriptive names
        expected_nodes = [
            "intent_classification_and_routing",
            "product_qa_agent_execution",
            "shopping_cart_agent_execution",
            "clarification_request_handling",
            "response_finalization_and_formatting"
        ]
        
        for node in expected_nodes:
            assert node in graph.nodes, f"Node '{node}' not found in graph"
        
        # Verify node count
        assert len(graph.nodes) == 5
    
    def test_compile_graph_with_metadata_tracking(self):
        """Test master graph compilation with metadata tracking."""
        compiled_graph = self.master_graph.compile_graph()
        
        assert compiled_graph is not None
        assert self.master_graph._compiled_master_graph is not None
        assert hasattr(compiled_graph, 'ainvoke')
        
        # Check metadata updates
        assert "compiled_at" in self.master_graph._graph_metadata
        assert self.master_graph._graph_metadata["compilation_successful"] is True
        
        # Second call should return cached version
        compiled_graph2 = self.master_graph.compile_graph()
        assert compiled_graph is compiled_graph2
    
    def test_determine_agent_routing_decision_with_descriptive_edges(self):
        """Test routing decision mapping to descriptive edge names."""
        # Test QA routing
        state = create_initial_state(self.session_id, "test", routing_decision="qa")
        decision = self.master_graph._determine_agent_routing_decision(state)
        assert decision == "route_to_qa_agent"
        
        # Test cart routing
        state = create_initial_state(self.session_id, "test", routing_decision="cart")
        decision = self.master_graph._determine_agent_routing_decision(state)
        assert decision == "route_to_cart_agent"
        
        # Test clarification routing
        state = create_initial_state(self.session_id, "test", routing_decision="clarification")
        decision = self.master_graph._determine_agent_routing_decision(state)
        assert decision == "request_clarification"
    
    def test_routing_decision_validation_and_fallback(self):
        """Test routing decision validation with fallback handling."""
        # Test invalid routing decision
        state = create_initial_state(self.session_id, "test", routing_decision="invalid")
        decision = self.master_graph._determine_agent_routing_decision(state)
        assert decision == "request_clarification"  # Should default to clarification
        
        # Test missing routing decision
        state = create_initial_state(self.session_id, "test")
        decision = self.master_graph._determine_agent_routing_decision(state)
        assert decision == "request_clarification"
    
    def test_count_processing_nodes(self):
        """Test processing node counting functionality."""
        # Test with intermediate steps
        state = create_initial_state(
            self.session_id, 
            "test",
            intermediate_steps=[{"step": 1}, {"step": 2}, {"step": 3}]
        )
        count = self.master_graph._count_processing_nodes(state)
        assert count == 3
        
        # Test with routing decision - clarification
        state = create_initial_state(self.session_id, "test", routing_decision="request_clarification")
        count = self.master_graph._count_processing_nodes(state)
        assert count == 2
        
        # Test with routing decision - agent execution
        state = create_initial_state(self.session_id, "test", routing_decision="route_to_qa_agent")
        count = self.master_graph._count_processing_nodes(state)
        assert count == 3
        
        # Test fallback
        state = create_initial_state(self.session_id, "test")
        count = self.master_graph._count_processing_nodes(state)
        assert count == 1


class TestMasterGraphNodeExecution:
    """Test cases for master graph node execution with improved naming."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.7},
            "classifier": {"test": True},
            "clarification": {"max_clarification_attempts": 3}
        }
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
        self.session_id = "test_node_execution_session"
    
    @pytest.mark.asyncio
    async def test_execute_intent_router_with_metadata(self):
        """Test intent router execution with comprehensive metadata."""
        state = create_initial_state(
            self.session_id,
            "add headphones to cart"
        )
        
        # Mock the intent router to return a cart routing decision
        with patch.object(self.master_graph.intent_router, 'route_message') as mock_route:
            routed_state = state.copy()
            routed_state["routing_decision"] = "cart"
            routed_state["user_intent"] = "cart"
            routed_state["intent_confidence"] = 0.9
            mock_route.return_value = routed_state
            
            result = await self.master_graph._execute_intent_router(state)
            
            assert result["routing_decision"] == "cart"
            assert result["user_intent"] == "cart"
            assert result["intent_confidence"] == 0.9
            
            # Check metadata
            assert "routing_metadata" in result
            assert "router_execution" in result["routing_metadata"]
            assert result["routing_metadata"]["router_execution"]["router_node_successful"] is True
            assert "executed_at" in result["routing_metadata"]["router_execution"]
            
            mock_route.assert_called_once_with(state)
    
    @pytest.mark.asyncio
    async def test_execute_intent_router_error_handling(self):
        """Test intent router execution with error handling and fallback."""
        state = create_initial_state(
            self.session_id,
            "test query"
        )
        
        # Mock the intent router to raise an exception
        with patch.object(self.master_graph.intent_router, 'route_message') as mock_route:
            mock_route.side_effect = Exception("Router error")
            
            result = await self.master_graph._execute_intent_router(state)
            
            # Should fallback to QA agent
            assert result["routing_decision"] == "route_to_qa_agent"
            assert result["target_agent"] == "product_qa_agent"
            assert "error_state" in result
            assert "Router execution error" in result["error_state"]
            
            # Check error metadata
            assert "routing_metadata" in result
            assert "router_execution" in result["routing_metadata"]
            assert result["routing_metadata"]["router_execution"]["router_node_successful"] is False
            assert result["routing_metadata"]["router_execution"]["fallback_applied"] is True
    
    @pytest.mark.asyncio
    async def test_execute_product_qa_agent_with_enhanced_metadata(self):
        """Test Product QA agent execution with enhanced metadata."""
        state = create_initial_state(
            self.session_id,
            "what are good laptops",
            routing_decision="qa"
        )
        
        # Mock the QA agent
        mock_qa_result = state.copy()
        mock_qa_result["final_response"] = "Here are the best laptops..."
        mock_qa_result["workflow_status"] = "completed"
        
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_qa_result)
        self.mock_agent_builder.create_ambient_agent_graph.return_value = mock_graph
        
        result = await self.master_graph._execute_product_qa_agent(state)
        
        assert result["workflow_status"] == "completed"
        assert result["final_response"] == "Here are the best laptops..."
        
        # Check enhanced metadata
        assert "response_metadata" in result
        assert "specialized_agent_execution" in result["response_metadata"]
        
        agent_metadata = result["response_metadata"]["specialized_agent_execution"]
        assert agent_metadata["agent_type"] == "product_qa_agent"
        assert agent_metadata["agent_name"] == "Product QA Agent"
        assert agent_metadata["execution_successful"] is True
        assert agent_metadata["routing_decision_honored"] is True
        assert agent_metadata["query_type"] == "product_information"
        
        # Check legacy compatibility
        assert result["response_metadata"]["agent_used"] == "qa_agent"
        assert result["response_metadata"]["routing_successful"] is True
    
    @pytest.mark.asyncio
    async def test_execute_product_qa_agent_error_handling(self):
        """Test Product QA agent execution with comprehensive error handling."""
        state = create_initial_state(
            self.session_id,
            "what are good laptops",
            routing_decision="qa"
        )
        
        # Mock the QA agent to raise an exception
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("QA agent error"))
        self.mock_agent_builder.create_ambient_agent_graph.return_value = mock_graph
        
        result = await self.master_graph._execute_product_qa_agent(state)
        
        assert "error_state" in result
        assert "Product QA Agent execution error" in result["error_state"]
        assert result["workflow_status"] == "completed"
        assert "final_response" in result
        
        # Check error metadata
        assert "response_metadata" in result
        assert "specialized_agent_execution" in result["response_metadata"]
        
        agent_metadata = result["response_metadata"]["specialized_agent_execution"]
        assert agent_metadata["agent_type"] == "product_qa_agent"
        assert agent_metadata["execution_successful"] is False
        assert agent_metadata["fallback_response_provided"] is True
        
        # Check legacy compatibility
        assert result["response_metadata"]["agent_used"] == "qa_agent"
        assert result["response_metadata"]["agent_error"] is True
        assert result["response_metadata"]["routing_successful"] is False
    
    @pytest.mark.asyncio
    async def test_execute_shopping_cart_agent_with_enhanced_metadata(self):
        """Test Shopping Cart agent execution with enhanced metadata."""
        state = create_initial_state(
            self.session_id,
            "add laptop to cart",
            routing_decision="cart"
        )
        
        # Mock the shopping cart agent
        with patch.object(self.master_graph.shopping_cart_agent, 'process_query') as mock_process:
            cart_result = state.copy()
            cart_result["final_response"] = "Added laptop to cart"
            cart_result["workflow_status"] = "completed"
            cart_result["cart_updated"] = True
            mock_process.return_value = cart_result
            
            result = await self.master_graph._execute_shopping_cart_agent(state)
            
            assert result["workflow_status"] == "completed"
            assert result["final_response"] == "Added laptop to cart"
            
            # Check enhanced metadata
            assert "response_metadata" in result
            assert "specialized_agent_execution" in result["response_metadata"]
            
            agent_metadata = result["response_metadata"]["specialized_agent_execution"]
            assert agent_metadata["agent_type"] == "shopping_cart_agent"
            assert agent_metadata["agent_name"] == "Shopping Cart Agent"
            assert agent_metadata["execution_successful"] is True
            assert agent_metadata["routing_decision_honored"] is True
            assert agent_metadata["query_type"] == "cart_management"
            assert agent_metadata["cart_operation_performed"] is True
            
            # Check legacy compatibility
            assert result["response_metadata"]["agent_used"] == "cart_agent"
            assert result["response_metadata"]["routing_successful"] is True
    
    @pytest.mark.asyncio
    async def test_execute_shopping_cart_agent_error_handling(self):
        """Test Shopping Cart agent execution with comprehensive error handling."""
        state = create_initial_state(
            self.session_id,
            "add item to cart",
            routing_decision="cart"
        )
        
        # Mock the cart agent to raise an exception
        with patch.object(self.master_graph.shopping_cart_agent, 'process_query') as mock_process:
            mock_process.side_effect = Exception("Cart error")
            
            result = await self.master_graph._execute_shopping_cart_agent(state)
            
            assert "error_state" in result
            assert "Shopping Cart Agent execution error" in result["error_state"]
            assert result["workflow_status"] == "completed"
            assert "final_response" in result
            
            # Check error metadata
            assert "response_metadata" in result
            assert "specialized_agent_execution" in result["response_metadata"]
            
            agent_metadata = result["response_metadata"]["specialized_agent_execution"]
            assert agent_metadata["agent_type"] == "shopping_cart_agent"
            assert agent_metadata["execution_successful"] is False
            assert agent_metadata["fallback_response_provided"] is True
            
            # Check legacy compatibility
            assert result["response_metadata"]["agent_used"] == "cart_agent"
            assert result["response_metadata"]["agent_error"] is True
            assert result["response_metadata"]["routing_successful"] is False
    
    @pytest.mark.asyncio
    async def test_handle_clarification_request_with_comprehensive_metadata(self):
        """Test clarification request handling with comprehensive metadata."""
        state = create_initial_state(
            self.session_id,
            "unclear query",
            routing_decision="clarification",
            final_response="Could you clarify what you're looking for?",
            suggested_questions=["Are you looking for products?", "Do you need cart help?"],
            clarification_attempts=1
        )
        
        result = await self.master_graph._handle_clarification_request(state)
        
        assert result["current_step"] == "clarification_request_handling"
        assert result["workflow_status"] == "completed"
        
        # Check comprehensive metadata
        assert "response_metadata" in result
        assert "clarification_handling" in result["response_metadata"]
        
        clarification_metadata = result["response_metadata"]["clarification_handling"]
        assert clarification_metadata["clarification_requested"] is True
        assert clarification_metadata["workflow_terminated"] is True
        assert clarification_metadata["termination_reason"] == "user_clarification_required"
        assert clarification_metadata["original_query"] == "unclear query"
        assert len(clarification_metadata["suggested_questions"]) == 2
        assert clarification_metadata["clarification_attempts"] == 1
        
        # Check legacy compatibility
        assert result["response_metadata"]["clarification_requested"] is True
        assert result["response_metadata"]["workflow_terminated"] is True
    
    @pytest.mark.asyncio
    async def test_finalize_and_format_response_with_comprehensive_metadata(self):
        """Test response finalization with comprehensive metadata."""
        state = create_initial_state(
            self.session_id,
            "test query",
            routing_decision="qa",
            intent_confidence=0.8,
            target_agent="product_qa_agent",
            final_response="Test response",
            response_metadata={"agent_used": "qa_agent"}
        )
        
        result = await self.master_graph._finalize_and_format_response(state)
        
        assert result["current_step"] == "response_finalization_and_formatting"
        assert result["workflow_status"] == "completed"
        
        # Check comprehensive metadata
        assert "response_metadata" in result
        assert "master_graph_finalization" in result["response_metadata"]
        
        finalization_metadata = result["response_metadata"]["master_graph_finalization"]
        assert finalization_metadata["routing_decision"] == "qa"
        assert finalization_metadata["intent_confidence"] == 0.8
        assert finalization_metadata["target_agent"] == "product_qa_agent"
        assert finalization_metadata["processing_successful"] is True
        assert finalization_metadata["response_formatted"] is True
        
        # Check workflow completion metadata
        assert "workflow_completion" in result["response_metadata"]
        completion_metadata = result["response_metadata"]["workflow_completion"]
        assert completion_metadata["completed_successfully"] is True
        assert completion_metadata["final_step"] == "response_finalization_and_formatting"
        assert completion_metadata["total_processing_nodes"] >= 1
        
        # Check legacy compatibility
        assert result["response_metadata"]["routing_decision"] == "qa"
        assert result["response_metadata"]["intent_confidence"] == 0.8


class TestMasterGraphInformationMethods:
    """Test cases for master graph information and documentation methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.8},
            "classifier": {"test": True},
            "clarification": {"max_clarification_attempts": 5}
        }
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
    
    def test_get_routing_statistics_with_enhanced_metadata(self):
        """Test routing statistics retrieval with enhanced metadata."""
        # Mock the intent router stats
        mock_stats = {
            "total_routes": 100,
            "qa_routes": 60,
            "cart_routes": 30,
            "clarifications": 10,
            "fallbacks": 5
        }
        
        with patch.object(self.master_graph.intent_router, 'get_routing_stats', return_value=mock_stats):
            stats = self.master_graph.get_routing_statistics()
            
            # Check base stats are included
            assert stats["total_routes"] == 100
            assert stats["qa_routes"] == 60
            assert stats["cart_routes"] == 30
            
            # Check enhanced metadata
            assert "master_graph_metadata" in stats
            graph_metadata = stats["master_graph_metadata"]
            assert graph_metadata["graph_version"] == "1.0.0"
            assert "created_at" in graph_metadata
            assert graph_metadata["compiled"] is False  # Not compiled yet
            
            # Check agent availability
            assert "agent_availability" in stats
            agent_availability = stats["agent_availability"]
            assert agent_availability["product_qa_agent"] is True  # Mock agent builder available
            assert agent_availability["shopping_cart_agent"] is True
            assert agent_availability["total_available_agents"] == 2
    
    def test_get_master_graph_info_comprehensive(self):
        """Test comprehensive master graph information retrieval."""
        info = self.master_graph.get_master_graph_info()
        
        # Check graph metadata
        assert "graph_metadata" in info
        graph_metadata = info["graph_metadata"]
        assert graph_metadata["graph_type"] == "master_routing_graph"
        assert graph_metadata["graph_name"] == "Master Agent Routing Graph"
        assert graph_metadata["version"] == "1.0.0"
        assert "description" in graph_metadata
        
        # Check compilation status
        assert "compilation_status" in info
        compilation_status = info["compilation_status"]
        assert compilation_status["compiled"] is False
        assert compilation_status["compilation_successful"] is False
        
        # Check available agents
        assert "available_agents" in info
        agents = info["available_agents"]
        
        assert "product_qa_agent" in agents
        qa_agent = agents["product_qa_agent"]
        assert qa_agent["name"] == "Product QA Agent"
        assert "description" in qa_agent
        assert qa_agent["available"] is True  # Mock agent builder available
        assert qa_agent["lazy_loaded"] is True  # Not loaded yet
        
        assert "shopping_cart_agent" in agents
        cart_agent = agents["shopping_cart_agent"]
        assert cart_agent["name"] == "Shopping Cart Agent"
        assert "description" in cart_agent
        assert cart_agent["available"] is True
        assert cart_agent["initialized"] is True
        
        # Check routing configuration
        assert "routing_configuration" in info
        routing_config = info["routing_configuration"]
        assert routing_config["intent_classification_enabled"] is True
        assert routing_config["clarification_handling_enabled"] is True
        assert routing_config["confidence_threshold"] == 0.8
        assert routing_config["max_clarification_attempts"] == 5
        
        # Check workflow nodes
        assert "workflow_nodes" in info
        workflow_nodes = info["workflow_nodes"]
        assert workflow_nodes["total_nodes"] == 5
        assert len(workflow_nodes["node_names"]) == 5
        assert len(workflow_nodes["routing_edges"]) == 3
        
        # Check performance metrics and configuration
        assert "performance_metrics" in info
        assert "configuration" in info
        assert info["configuration"] == self.config
    
    def test_get_agent_hierarchy_documentation(self):
        """Test agent hierarchy documentation retrieval."""
        hierarchy = self.master_graph.get_agent_hierarchy_documentation()
        
        # Check hierarchy structure
        assert "hierarchy_structure" in hierarchy
        structure = hierarchy["hierarchy_structure"]
        
        assert "master_graph" in structure
        master_level = structure["master_graph"]
        assert master_level["level"] == 0
        assert master_level["role"] == "orchestration"
        assert "manages" in master_level
        
        assert "intent_router" in structure
        router_level = structure["intent_router"]
        assert router_level["level"] == 1
        assert router_level["role"] == "routing"
        assert "components" in router_level
        
        assert "specialized_agents" in structure
        agents_level = structure["specialized_agents"]
        assert agents_level["level"] == 1
        assert agents_level["role"] == "execution"
        assert "agents" in agents_level
        
        # Check agent relationships
        assert "agent_relationships" in hierarchy
        relationships = hierarchy["agent_relationships"]
        
        assert "product_qa_agent" in relationships
        qa_relationship = relationships["product_qa_agent"]
        assert "handles" in qa_relationship
        assert "tools" in qa_relationship
        assert qa_relationship["tool_type"] == "mcp_tools"
        
        assert "shopping_cart_agent" in relationships
        cart_relationship = relationships["shopping_cart_agent"]
        assert "handles" in cart_relationship
        assert "tools" in cart_relationship
        assert cart_relationship["tool_type"] == "function_calling"
        assert cart_relationship["state_management"] == "persistent_database"
        
        # Check routing logic
        assert "routing_logic" in hierarchy
        routing_logic = hierarchy["routing_logic"]
        
        assert "intent_classification" in routing_logic
        classification = routing_logic["intent_classification"]
        assert classification["confidence_threshold"] == 0.8
        assert classification["fallback_strategy"] == "clarification_request"
        
        assert "clarification_handling" in routing_logic
        clarification = routing_logic["clarification_handling"]
        assert clarification["max_attempts"] == 5
        assert clarification["fallback_agent"] == "product_qa_agent"
        
        # Check workflow patterns
        assert "workflow_patterns" in hierarchy
        patterns = hierarchy["workflow_patterns"]
        
        assert "successful_routing" in patterns
        assert len(patterns["successful_routing"]) == 3
        
        assert "clarification_required" in patterns
        assert len(patterns["clarification_required"]) == 2
        
        assert "error_fallback" in patterns
        assert len(patterns["error_fallback"]) == 3
    
    def test_legacy_compatibility_methods(self):
        """Test legacy compatibility methods."""
        # Mock the intent router stats for legacy methods
        mock_stats = {"total_routes": 50}
        
        with patch.object(self.master_graph.intent_router, 'get_routing_stats', return_value=mock_stats):
            # Test legacy get_routing_stats
            legacy_stats = self.master_graph.get_routing_stats()
            enhanced_stats = self.master_graph.get_routing_statistics()
            assert legacy_stats == enhanced_stats
            
            # Test legacy reset_routing_stats
            with patch.object(self.master_graph.intent_router, 'reset_routing_stats') as mock_reset:
                self.master_graph.reset_routing_stats()
                mock_reset.assert_called_once()
            
            # Test legacy get_graph_info
            legacy_info = self.master_graph.get_graph_info()
            enhanced_info = self.master_graph.get_master_graph_info()
            assert legacy_info == enhanced_info


class TestMasterGraphEndToEndStructure:
    """End-to-end tests for master graph structure and organization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.7},
            "classifier": {"confidence_threshold": 0.7},
            "clarification": {"max_clarification_attempts": 3}
        }
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
        self.session_id = "e2e_structure_test_session"
    
    @pytest.mark.asyncio
    async def test_process_query_with_comprehensive_metadata(self):
        """Test end-to-end query processing with comprehensive metadata tracking."""
        state = create_initial_state(
            self.session_id,
            "show my cart"
        )
        
        # Mock the compiled graph
        mock_graph = Mock()
        mock_result = state.copy()
        mock_result["workflow_status"] = "completed"
        mock_result["final_response"] = "Your cart is empty"
        mock_result["response_metadata"] = {"agent_used": "cart_agent"}
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)
        
        with patch.object(self.master_graph, 'compile_graph', return_value=mock_graph):
            result = await self.master_graph.process_query(state)
            
            assert result["workflow_status"] == "completed"
            assert result["final_response"] == "Your cart is empty"
            
            # Check comprehensive processing metadata
            assert "response_metadata" in result
            assert "master_graph_processing" in result["response_metadata"]
            
            processing_metadata = result["response_metadata"]["master_graph_processing"]
            assert processing_metadata["processing_successful"] is True
            assert "processing_completed_at" in processing_metadata
            assert processing_metadata["graph_version"] == "1.0.0"
            
            mock_graph.ainvoke.assert_called_once_with(state)
    
    @pytest.mark.asyncio
    async def test_process_query_error_handling_with_comprehensive_metadata(self):
        """Test query processing error handling with comprehensive metadata."""
        state = create_initial_state(
            self.session_id,
            "test query"
        )
        
        # Mock the compiled graph to raise an exception
        with patch.object(self.master_graph, 'compile_graph', side_effect=Exception("Graph compilation error")):
            result = await self.master_graph.process_query(state)
            
            assert "error_state" in result
            assert "Master graph processing error" in result["error_state"]
            assert result["workflow_status"] == "error"
            assert "final_response" in result
            
            # Check comprehensive error metadata
            assert "response_metadata" in result
            assert "master_graph_processing" in result["response_metadata"]
            
            processing_metadata = result["response_metadata"]["master_graph_processing"]
            assert processing_metadata["processing_successful"] is False
            assert "error_message" in processing_metadata
            assert "error_occurred_at" in processing_metadata
            assert processing_metadata["fallback_response_provided"] is True
    
    def test_graph_structure_consistency(self):
        """Test that graph structure is consistent and well-organized."""
        # Create and compile the graph
        graph = self.master_graph.create_master_graph()
        compiled_graph = self.master_graph.compile_graph()
        
        # Get comprehensive information
        graph_info = self.master_graph.get_master_graph_info()
        hierarchy_docs = self.master_graph.get_agent_hierarchy_documentation()
        
        # Verify consistency between actual graph and documentation
        actual_nodes = list(graph.nodes.keys())
        documented_nodes = graph_info["workflow_nodes"]["node_names"]
        
        assert len(actual_nodes) == len(documented_nodes)
        for node in documented_nodes:
            assert node in actual_nodes, f"Documented node '{node}' not found in actual graph"
        
        # Verify agent relationships are consistent
        available_agents = graph_info["available_agents"]
        documented_agents = hierarchy_docs["agent_relationships"]
        
        assert "product_qa_agent" in available_agents
        assert "product_qa_agent" in documented_agents
        assert "shopping_cart_agent" in available_agents
        assert "shopping_cart_agent" in documented_agents
        
        # Verify routing configuration consistency
        routing_config = graph_info["routing_configuration"]
        routing_logic = hierarchy_docs["routing_logic"]
        
        assert routing_config["confidence_threshold"] == routing_logic["intent_classification"]["confidence_threshold"]
        assert routing_config["max_clarification_attempts"] == routing_logic["clarification_handling"]["max_attempts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])