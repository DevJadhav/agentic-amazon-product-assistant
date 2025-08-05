# Agent Hierarchy Documentation

This document provides comprehensive documentation of the agent hierarchy, relationships, and naming conventions used in the LangGraph integration system.

## Overview

The system implements a hierarchical multi-agent architecture with intelligent routing capabilities. The architecture consists of three main layers:

1. **Orchestration Layer**: Master routing graph that coordinates agent selection
2. **Specialized Agents**: Domain-specific agents for different query types
3. **Legacy Agents**: Previous agent implementations maintained for compatibility

## Agent Hierarchy Structure

### Orchestration Layer

#### Master Routing Graph
- **Role**: Orchestration and routing
- **Description**: Top-level routing and agent coordination
- **Components**: 
  - Intent Router (intent classification and routing decisions)
  - Clarification Handler (handles ambiguous queries)
- **Manages**: Product QA Agent, Shopping Cart Agent
- **Routing Logic**: Intent classification based with confidence thresholds

### Specialized Agents

#### Product QA Agent
- **Role**: Product information and analysis
- **Description**: Handles product search, analysis, and recommendations
- **Tools**: Vector Search MCP, Product Analysis MCP
- **Tool Type**: MCP Tools
- **Workflows**: Search, Analysis, Comparison, Recommendation
- **Fallback For**: Unclear product queries, general queries

#### Shopping Cart Agent
- **Role**: Cart management operations
- **Description**: Manages shopping cart operations and state
- **Tools**: Add to Cart, Remove from Cart, List Cart, Clear Cart
- **Tool Type**: Function Calling
- **Workflows**: Add, Remove, List, Clear
- **State Management**: Persistent database storage

### Legacy Agents

#### Ambient Agent
- **Status**: Legacy
- **Role**: General purpose
- **Description**: Legacy ambient agent, now integrated as Product QA Agent
- **Replacement**: Product QA Agent

#### Specialized Workflow Agents
- **Product Search Agent**: Integrated into Product QA Agent
- **Review Analysis Agent**: Integrated into Product QA Agent
- **Product Comparison Agent**: Integrated into Product QA Agent
- **Product Recommendation Agent**: Integrated into Product QA Agent

## Routing Patterns

### Intent-Based Routing
- **Method**: Keyword and context analysis
- **Confidence Threshold**: Configurable (default: 0.7)
- **Fallback Strategy**: Clarification request

### Clarification Handling
- **Triggers**: Low confidence, ambiguous intent, multiple intents
- **Max Attempts**: Configurable (default: 3)
- **Fallback Agent**: Product QA Agent

## Workflow Patterns

### Successful Routing
1. Intent Classification and Routing
2. Specialized Agent Execution
3. Response Finalization and Formatting

### Clarification Required
1. Intent Classification and Routing
2. Clarification Request Handling

### Error Fallback
1. Intent Classification and Routing
2. Product QA Agent Execution (fallback)
3. Response Finalization and Formatting

## Naming Conventions

### Node Naming Conventions
- **Pattern**: `action_description_and_purpose`
- **Examples**:
  - `intent_classification_and_routing`
  - `product_qa_agent_execution`
  - `shopping_cart_agent_execution`
  - `clarification_request_handling`
  - `response_finalization_and_formatting`
- **Guidelines**:
  - Use descriptive names that clearly indicate the node's purpose
  - Include both the action and the domain/context
  - Use underscores to separate words
  - Avoid abbreviations unless widely understood
  - Be consistent across similar node types

### Edge Naming Conventions
- **Pattern**: `action_to_target`
- **Examples**:
  - `route_to_qa_agent`
  - `route_to_cart_agent`
  - `request_clarification`
- **Guidelines**:
  - Use verb phrases that describe the routing action
  - Include the target destination
  - Be consistent with conditional edge naming
  - Make routing logic clear from the name

### Agent Naming Conventions
- **Pattern**: `domain_agent_type`
- **Examples**:
  - `product_qa_agent`
  - `shopping_cart_agent`
  - `master_routing_graph`
- **Guidelines**:
  - Include the domain or specialization
  - Use 'agent' suffix for execution agents
  - Use 'graph' suffix for orchestration graphs
  - Avoid generic names like 'main' or 'default'

### Method Naming Conventions
- **Pattern**: `action_domain_purpose`
- **Examples**:
  - `_execute_intent_router`
  - `_execute_product_qa_agent`
  - `_handle_clarification_request`
  - `_finalize_and_format_response`
- **Guidelines**:
  - Use descriptive method names that indicate purpose
  - Prefix private methods with underscore
  - Include the domain or component being acted upon
  - Use consistent verb patterns (execute, handle, process, etc.)

## Agent Relationships

### Tool Architecture
- **Product QA Agent**: Uses MCP (Model Context Protocol) tools for external integrations
- **Shopping Cart Agent**: Uses function calling tools for direct database operations

### State Management
- **Product QA Agent**: Stateless operation with conversation context
- **Shopping Cart Agent**: Persistent state management with database storage

### Error Handling
- **Fallback Strategy**: Route to Product QA Agent for unhandled cases
- **Error Recovery**: Graceful degradation with informative error messages
- **Retry Logic**: Configurable retry attempts for transient failures

## Configuration

### Router Configuration
```python
{
    "router": {
        "confidence_threshold": 0.7
    },
    "classifier": {
        "confidence_threshold": 0.7
    },
    "clarification": {
        "max_clarification_attempts": 3
    }
}
```

### Agent Configuration
```python
{
    "cart_agent": {
        "max_tool_calls": 5
    },
    "qa_agent": {
        "max_products": 5,
        "max_reviews": 3
    }
}
```

## Migration Guide

### From Legacy Agents
1. **Ambient Agent** → **Product QA Agent**: Direct replacement with enhanced capabilities
2. **Specialized Workflow Agents** → **Product QA Agent**: Integrated workflows within single agent
3. **Direct Agent Calls** → **Master Routing Graph**: Use routing for intelligent agent selection

### Configuration Updates
- Update agent names in configuration files
- Adjust routing thresholds based on performance requirements
- Configure clarification handling parameters

## Performance Considerations

### Routing Overhead
- Intent classification adds minimal latency (~10-50ms)
- Caching available for repeated similar queries
- Fallback mechanisms ensure reliability

### Agent Efficiency
- Lazy loading of specialized agents reduces memory usage
- Connection pooling for database operations
- Optimized tool calling patterns

## Monitoring and Debugging

### Available Metrics
- Routing statistics (success rates, agent distribution)
- Agent performance metrics (execution time, error rates)
- Tool usage statistics (call frequency, success rates)

### Debugging Tools
- Comprehensive logging with structured metadata
- State inspection utilities
- Routing decision tracing

## Future Extensibility

### Adding New Agents
1. Implement agent following naming conventions
2. Add to agent hierarchy documentation
3. Update routing logic if needed
4. Add appropriate tests

### Extending Routing Logic
1. Update intent classifier with new patterns
2. Add routing decisions to master graph
3. Update documentation and tests
4. Consider backward compatibility

## API Reference

### AgentGraphBuilder Methods
- `create_master_routing_graph()`: Create master orchestration graph
- `create_product_qa_agent_graph()`: Create Product QA Agent
- `create_shopping_cart_agent_graph()`: Create Shopping Cart Agent
- `get_available_graphs()`: List all available agent graphs
- `get_agent_hierarchy_mapping()`: Get complete hierarchy documentation
- `get_graph_naming_conventions()`: Get naming conventions documentation

### MasterAgentGraph Methods
- `process_query(state)`: Main entry point for query processing
- `get_routing_statistics()`: Get routing performance metrics
- `get_master_graph_info()`: Get comprehensive graph information
- `get_agent_hierarchy_documentation()`: Get hierarchy documentation

## Best Practices

### Development
1. Follow naming conventions consistently
2. Add comprehensive tests for new components
3. Update documentation when making changes
4. Consider backward compatibility

### Deployment
1. Monitor routing statistics for optimization opportunities
2. Adjust confidence thresholds based on production data
3. Implement proper error handling and logging
4. Use configuration management for environment-specific settings

### Maintenance
1. Regularly review and update legacy agent integrations
2. Monitor performance metrics and optimize as needed
3. Keep documentation synchronized with code changes
4. Plan for future extensibility requirements