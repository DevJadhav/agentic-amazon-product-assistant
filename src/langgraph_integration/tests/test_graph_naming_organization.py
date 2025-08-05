"""
Tests for graph naming conventions and organization improvements.
Verifies consistent naming across all agents and clear documentation.
"""

import pytest
from unittest.mock import Mock, patch

from ..core.agent_builder import AgentGraphBuilder
from ..core.router.master_graph import MasterAgentGraph
from ..core.state_schemas import create_initial_state


class TestGraphNamingConventions:
    """Test cases for graph naming conventions and consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.7},
            "classifier": {"test": True},
            "clarification": {"max_clarification_attempts": 3}
        }
        self.agent_builder = AgentGraphBuilder(self.config)
    
    def test_available_graphs_naming_consistency(self):
        """Test that available graphs follow consistent naming conventions."""
        graphs = self.agent_builder.get_available_graphs()
        
        # Check that all graph names follow conventions
        expected_patterns = {
            "master_routing_graph": "orchestration",
            "product_qa_agent": "specialized_agent",
            "shopping_cart_agent": "specialized_agent",
            "ambient_agent": "legacy",
            "product_search_agent": "legacy",
            "review_analysis_agent": "legacy",
            "product_comparison_agent": "legacy",
            "product_recommendation_agent": "legacy"
        }
        
        for graph_name in expected_patterns:
            assert graph_name in graphs, f"Expected graph '{graph_name}' not found"
            
            # Check naming pattern consistency
            if expected_patterns[graph_name] == "specialized_agent":
                assert graph_name.endswith("_agent"), f"Specialized agent '{graph_name}' should end with '_agent'"
            elif expected_patterns[graph_name] == "orchestration":
                assert "graph" in graph_name, f"Orchestration graph '{graph_name}' should contain 'graph'"
    
    def test_graph_descriptions_are_descriptive(self):
        """Test that graph descriptions are clear and descriptive."""
        graphs = self.agent_builder.get_available_graphs()
        
        for graph_name, description in graphs.items():
            # Check description is not empty
            assert description.strip(), f"Graph '{graph_name}' has empty description"
            
            # Check description is reasonably long (more than just a few words)
            assert len(description.split()) >= 3, f"Graph '{graph_name}' description too short: '{description}'"
            
            # Check description contains key information
            if "agent" in graph_name:
                assert any(word in description.lower() for word in ["agent", "handles", "manages", "specialized"]), \
                    f"Agent description should mention its role: '{description}'"
            
            if "graph" in graph_name:
                assert any(word in description.lower() for word in ["graph", "routing", "orchestration"]), \
                    f"Graph description should mention its orchestration role: '{description}'"
    
    def test_node_naming_conventions_documentation(self):
        """Test node naming conventions documentation."""
        conventions = self.agent_builder.get_graph_naming_conventions()
        
        node_conventions = conventions["node_naming_conventions"]
        
        # Check required fields
        assert "pattern" in node_conventions
        assert "examples" in node_conventions
        assert "guidelines" in node_conventions
        
        # Check pattern is descriptive
        pattern = node_conventions["pattern"]
        assert len(pattern) > 5, "Pattern should be descriptive"
        assert "_" in pattern, "Pattern should show underscore usage"
        
        # Check examples follow the pattern
        examples = node_conventions["examples"]
        assert len(examples) >= 3, "Should have multiple examples"
        
        for example in examples:
            assert "_" in example, f"Example '{example}' should use underscores"
            assert example.islower(), f"Example '{example}' should be lowercase"
            assert len(example.split("_")) >= 2, f"Example '{example}' should have multiple words"
        
        # Check guidelines are comprehensive
        guidelines = node_conventions["guidelines"]
        assert len(guidelines) >= 3, "Should have multiple guidelines"
        
        guideline_text = " ".join(guidelines).lower()
        assert "descriptive" in guideline_text, "Guidelines should mention being descriptive"
        assert "consistent" in guideline_text, "Guidelines should mention consistency"
    
    def test_edge_naming_conventions_documentation(self):
        """Test edge naming conventions documentation."""
        conventions = self.agent_builder.get_graph_naming_conventions()
        
        edge_conventions = conventions["edge_naming_conventions"]
        
        # Check required fields
        assert "pattern" in edge_conventions
        assert "examples" in edge_conventions
        assert "guidelines" in edge_conventions
        
        # Check examples are routing-related
        examples = edge_conventions["examples"]
        assert len(examples) >= 2, "Should have multiple edge examples"
        
        for example in examples:
            assert "_" in example, f"Edge example '{example}' should use underscores"
            assert any(word in example for word in ["route", "request", "to"]), \
                f"Edge example '{example}' should indicate routing action"
    
    def test_agent_naming_conventions_documentation(self):
        """Test agent naming conventions documentation."""
        conventions = self.agent_builder.get_graph_naming_conventions()
        
        agent_conventions = conventions["agent_naming_conventions"]
        
        # Check required fields
        assert "pattern" in agent_conventions
        assert "examples" in agent_conventions
        assert "guidelines" in agent_conventions
        
        # Check examples follow agent naming patterns
        examples = agent_conventions["examples"]
        
        for example in examples:
            if "agent" in example:
                assert example.endswith("_agent"), f"Agent example '{example}' should end with '_agent'"
            if "graph" in example:
                assert "graph" in example, f"Graph example '{example}' should contain 'graph'"
    
    def test_method_naming_conventions_documentation(self):
        """Test method naming conventions documentation."""
        conventions = self.agent_builder.get_graph_naming_conventions()
        
        method_conventions = conventions["method_naming_conventions"]
        
        # Check required fields
        assert "pattern" in method_conventions
        assert "examples" in method_conventions
        assert "guidelines" in method_conventions
        
        # Check examples follow method naming patterns
        examples = method_conventions["examples"]
        
        for example in examples:
            assert example.startswith("_"), f"Method example '{example}' should start with underscore"
            assert "_" in example[1:], f"Method example '{example}' should have multiple words"
            assert any(verb in example for verb in ["execute", "handle", "process", "finalize"]), \
                f"Method example '{example}' should contain action verb"


class TestAgentHierarchyDocumentation:
    """Test cases for agent hierarchy documentation and relationships."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "router": {"confidence_threshold": 0.8},
            "clarification": {"max_clarification_attempts": 5}
        }
        self.agent_builder = AgentGraphBuilder(self.config)
    
    def test_agent_hierarchy_structure(self):
        """Test agent hierarchy structure documentation."""
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        
        # Check main sections
        assert "orchestration_layer" in hierarchy
        assert "specialized_agents" in hierarchy
        assert "legacy_agents" in hierarchy
        assert "routing_patterns" in hierarchy
        
        # Check orchestration layer
        orchestration = hierarchy["orchestration_layer"]
        assert "master_routing_graph" in orchestration
        
        master_info = orchestration["master_routing_graph"]
        assert master_info["role"] == "orchestration"
        assert "description" in master_info
        assert "manages" in master_info
        assert "components" in master_info
        
        # Check it manages the right agents
        managed_agents = master_info["manages"]
        assert "product_qa_agent" in managed_agents
        assert "shopping_cart_agent" in managed_agents
    
    def test_specialized_agents_documentation(self):
        """Test specialized agents documentation."""
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        
        specialized = hierarchy["specialized_agents"]
        
        # Check required agents
        assert "product_qa_agent" in specialized
        assert "shopping_cart_agent" in specialized
        
        # Check product QA agent
        qa_agent = specialized["product_qa_agent"]
        assert qa_agent["role"] == "product_information"
        assert "description" in qa_agent
        assert "tools" in qa_agent
        assert qa_agent["tool_type"] == "mcp_tools"
        assert "workflows" in qa_agent
        assert "fallback_for" in qa_agent
        
        # Check shopping cart agent
        cart_agent = specialized["shopping_cart_agent"]
        assert cart_agent["role"] == "cart_management"
        assert "description" in cart_agent
        assert "tools" in cart_agent
        assert cart_agent["tool_type"] == "function_calling"
        assert cart_agent["state_management"] == "persistent_database"
    
    def test_legacy_agents_documentation(self):
        """Test legacy agents documentation."""
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        
        legacy = hierarchy["legacy_agents"]
        
        # Check legacy agents are documented
        expected_legacy = [
            "ambient_agent",
            "product_search_agent", 
            "review_analysis_agent",
            "product_comparison_agent",
            "product_recommendation_agent"
        ]
        
        for agent_name in expected_legacy:
            assert agent_name in legacy, f"Legacy agent '{agent_name}' not documented"
            
            agent_info = legacy[agent_name]
            assert agent_info["status"] == "legacy"
            assert "description" in agent_info
            assert "role" in agent_info
            
            # Check replacement or integration info
            assert "replacement" in agent_info or "integrated_into" in agent_info, \
                f"Legacy agent '{agent_name}' should have replacement/integration info"
    
    def test_routing_patterns_documentation(self):
        """Test routing patterns documentation."""
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        
        routing_patterns = hierarchy["routing_patterns"]
        
        # Check intent-based routing
        assert "intent_based_routing" in routing_patterns
        intent_routing = routing_patterns["intent_based_routing"]
        assert "description" in intent_routing
        assert "confidence_threshold" in intent_routing
        assert intent_routing["confidence_threshold"] == 0.8  # From config
        assert intent_routing["fallback_strategy"] == "clarification_request"
        
        # Check clarification handling
        assert "clarification_handling" in routing_patterns
        clarification = routing_patterns["clarification_handling"]
        assert "description" in clarification
        assert "max_attempts" in clarification
        assert clarification["max_attempts"] == 5  # From config
        assert clarification["fallback_agent"] == "product_qa_agent"
    
    def test_hierarchy_consistency_with_available_graphs(self):
        """Test that hierarchy documentation is consistent with available graphs."""
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        available_graphs = self.agent_builder.get_available_graphs()
        
        # Check orchestration layer consistency
        orchestration = hierarchy["orchestration_layer"]
        for graph_name in orchestration:
            assert graph_name in available_graphs, \
                f"Orchestration graph '{graph_name}' not in available graphs"
        
        # Check specialized agents consistency
        specialized = hierarchy["specialized_agents"]
        for agent_name in specialized:
            assert agent_name in available_graphs, \
                f"Specialized agent '{agent_name}' not in available graphs"
        
        # Check legacy agents consistency
        legacy = hierarchy["legacy_agents"]
        for agent_name in legacy:
            assert agent_name in available_graphs, \
                f"Legacy agent '{agent_name}' not in available graphs"


class TestMasterGraphNamingConsistency:
    """Test cases for master graph naming consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"test": True}
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
    
    def test_master_graph_node_names_follow_conventions(self):
        """Test that master graph node names follow naming conventions."""
        graph = self.master_graph.create_master_graph()
        
        expected_nodes = [
            "intent_classification_and_routing",
            "product_qa_agent_execution",
            "shopping_cart_agent_execution",
            "clarification_request_handling",
            "response_finalization_and_formatting"
        ]
        
        actual_nodes = list(graph.nodes.keys())
        
        # Check all expected nodes exist
        for node in expected_nodes:
            assert node in actual_nodes, f"Expected node '{node}' not found in graph"
        
        # Check node naming conventions
        for node in actual_nodes:
            # Should use underscores
            assert "_" in node, f"Node '{node}' should use underscores"
            
            # Should be lowercase
            assert node.islower(), f"Node '{node}' should be lowercase"
            
            # Should be descriptive (multiple words)
            assert len(node.split("_")) >= 2, f"Node '{node}' should have multiple words"
            
            # Should indicate action and purpose
            words = node.split("_")
            action_words = ["classification", "execution", "handling", "finalization"]
            assert any(action in node for action in action_words), \
                f"Node '{node}' should indicate its action/purpose"
    
    def test_master_graph_method_names_follow_conventions(self):
        """Test that master graph method names follow conventions."""
        # Get all method names that are node handlers
        node_methods = [
            "_execute_intent_router",
            "_execute_product_qa_agent", 
            "_execute_shopping_cart_agent",
            "_handle_clarification_request",
            "_finalize_and_format_response"
        ]
        
        for method_name in node_methods:
            # Should start with underscore (private)
            assert method_name.startswith("_"), f"Method '{method_name}' should be private"
            
            # Should use underscores
            assert "_" in method_name[1:], f"Method '{method_name}' should use underscores"
            
            # Should be lowercase
            assert method_name.islower(), f"Method '{method_name}' should be lowercase"
            
            # Should contain action verb
            action_verbs = ["execute", "handle", "finalize", "process"]
            assert any(verb in method_name for verb in action_verbs), \
                f"Method '{method_name}' should contain action verb"
            
            # Should indicate what it operates on
            if "execute" in method_name:
                assert any(target in method_name for target in ["router", "agent"]), \
                    f"Execute method '{method_name}' should indicate target"
    
    def test_routing_decision_names_follow_conventions(self):
        """Test that routing decision names follow conventions."""
        # Test the routing decision mapping
        state = create_initial_state("test_session", "test query")
        
        # Test valid routing decisions
        valid_decisions = ["qa", "cart", "clarification"]
        expected_mappings = {
            "qa": "route_to_qa_agent",
            "cart": "route_to_cart_agent", 
            "clarification": "request_clarification"
        }
        
        for decision, expected_mapping in expected_mappings.items():
            state["routing_decision"] = decision
            mapped_decision = self.master_graph._determine_agent_routing_decision(state)
            
            assert mapped_decision == expected_mapping, \
                f"Decision '{decision}' should map to '{expected_mapping}', got '{mapped_decision}'"
            
            # Check mapping follows conventions
            if "route_to" in mapped_decision:
                assert mapped_decision.startswith("route_to_"), \
                    f"Route mapping '{mapped_decision}' should start with 'route_to_'"
                assert mapped_decision.endswith("_agent"), \
                    f"Route mapping '{mapped_decision}' should end with '_agent'"
            elif "request" in mapped_decision:
                assert "clarification" in mapped_decision, \
                    f"Request mapping '{mapped_decision}' should mention clarification"
    
    def test_master_graph_info_methods_follow_conventions(self):
        """Test that master graph info methods follow naming conventions."""
        # Test new method names
        new_methods = [
            "get_routing_statistics",
            "reset_routing_statistics", 
            "get_master_graph_info",
            "get_agent_hierarchy_documentation"
        ]
        
        for method_name in new_methods:
            # Should exist
            assert hasattr(self.master_graph, method_name), \
                f"Method '{method_name}' should exist"
            
            # Should be callable
            method = getattr(self.master_graph, method_name)
            assert callable(method), f"'{method_name}' should be callable"
            
            # Should follow naming conventions
            assert method_name.islower() or "_" in method_name, \
                f"Method '{method_name}' should use lowercase or underscores"
            
            if method_name.startswith("get_"):
                assert not method_name.startswith("_"), \
                    f"Public getter '{method_name}' should not start with underscore"
        
        # Test legacy compatibility methods exist
        legacy_methods = ["get_routing_stats", "reset_routing_stats", "get_graph_info"]
        
        for method_name in legacy_methods:
            assert hasattr(self.master_graph, method_name), \
                f"Legacy method '{method_name}' should exist for compatibility"


class TestNamingConsistencyAcrossSystem:
    """Test cases for naming consistency across the entire system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"test": True}
        self.agent_builder = AgentGraphBuilder(self.config)
        self.mock_agent_builder = Mock()
        self.master_graph = MasterAgentGraph(self.config, agent_builder=self.mock_agent_builder)
    
    def test_agent_names_consistency_between_builder_and_master_graph(self):
        """Test agent name consistency between agent builder and master graph."""
        # Get available graphs from builder
        available_graphs = self.agent_builder.get_available_graphs()
        
        # Get hierarchy from builder
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        
        # Get master graph info
        master_info = self.master_graph.get_master_graph_info()
        
        # Check specialized agents consistency
        builder_specialized = hierarchy["specialized_agents"]
        master_agents = master_info["available_agents"]
        
        for agent_name in builder_specialized:
            assert agent_name in master_agents, \
                f"Agent '{agent_name}' in builder hierarchy not found in master graph"
            assert agent_name in available_graphs, \
                f"Agent '{agent_name}' in hierarchy not found in available graphs"
    
    def test_routing_configuration_consistency(self):
        """Test routing configuration consistency across components."""
        # Get configuration from different sources
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        master_info = self.master_graph.get_master_graph_info()
        
        # Check confidence threshold consistency
        hierarchy_threshold = hierarchy["routing_patterns"]["intent_based_routing"]["confidence_threshold"]
        master_threshold = master_info["routing_configuration"]["confidence_threshold"]
        
        assert hierarchy_threshold == master_threshold, \
            "Confidence threshold should be consistent across components"
        
        # Check clarification attempts consistency
        hierarchy_attempts = hierarchy["routing_patterns"]["clarification_handling"]["max_attempts"]
        master_attempts = master_info["routing_configuration"]["max_clarification_attempts"]
        
        assert hierarchy_attempts == master_attempts, \
            "Max clarification attempts should be consistent across components"
    
    def test_workflow_node_names_consistency(self):
        """Test workflow node names consistency in documentation."""
        # Get master graph info
        master_info = self.master_graph.get_master_graph_info()
        
        # Get actual graph nodes
        actual_graph = self.master_graph.create_master_graph()
        actual_nodes = list(actual_graph.nodes.keys())
        
        # Get documented nodes
        documented_nodes = master_info["workflow_nodes"]["node_names"]
        
        # Check consistency
        assert len(actual_nodes) == len(documented_nodes), \
            "Number of actual nodes should match documented nodes"
        
        for node in documented_nodes:
            assert node in actual_nodes, \
                f"Documented node '{node}' not found in actual graph"
        
        for node in actual_nodes:
            assert node in documented_nodes, \
                f"Actual node '{node}' not found in documentation"
    
    def test_agent_tool_type_consistency(self):
        """Test agent tool type consistency in documentation."""
        hierarchy = self.agent_builder.get_agent_hierarchy_mapping()
        specialized_agents = hierarchy["specialized_agents"]
        
        # Check product QA agent uses MCP tools
        qa_agent = specialized_agents["product_qa_agent"]
        assert qa_agent["tool_type"] == "mcp_tools"
        assert all("mcp" in tool for tool in qa_agent["tools"])
        
        # Check shopping cart agent uses function calling
        cart_agent = specialized_agents["shopping_cart_agent"]
        assert cart_agent["tool_type"] == "function_calling"
        assert all("mcp" not in tool for tool in cart_agent["tools"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])