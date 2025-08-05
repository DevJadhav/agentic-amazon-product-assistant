# Developer Guide: Extending the Dual Tool Architecture

## Overview

The Dual Tool Architecture enables different agents to use different tool interfaces based on their specific needs and integration requirements. This system supports both MCP (Model Context Protocol) tools for external integrations and regular function calling tools for direct system operations.

## Architecture Principles

### Tool Interface Separation

The system implements two distinct tool interfaces:

1. **MCP Tools**: Used by the QA Agent for external data sources and services
2. **Function Calling Tools**: Used by the Shopping Cart Agent for direct database operations

### Design Benefits

- **Flexibility**: Each agent can use the most appropriate tool interface
- **Performance**: Direct function calls for internal operations, MCP for external services
- **Maintainability**: Clear separation of concerns between tool types
- **Extensibility**: Easy to add new agents with different tool requirements

## Current Implementation

### QA Agent with MCP Tools

The QA Agent uses MCP tools for external integrations:

```python
class ProductQAAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mcp_client = MCPClient()
        self.tools = self._initialize_mcp_tools()
    
    def _initialize_mcp_tools(self):
        return [
            VectorSearchMCPTool(self.mcp_client),
            ProductAnalysisMCPTool(self.mcp_client)
        ]
    
    async def _execute_tools(self, state: AgentState) -> AgentState:
        # Use MCP protocol for tool execution
        tool_results = await self.mcp_client.call_tools(
            tools=self.selected_tools,
            parameters=self.tool_parameters
        )
        return self._process_mcp_results(tool_results, state)
```

### Shopping Cart Agent with Function Calling

The Shopping Cart Agent uses direct function calling:

```python
class ShoppingCartAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], cart_manager: ShoppingCartManager):
        super().__init__(config)
        self.cart_manager = cart_manager
        self.tools = self._initialize_function_tools()
    
    def _initialize_function_tools(self):
        return [
            AddToCartTool(cart_manager=self.cart_manager),
            RemoveFromCartTool(cart_manager=self.cart_manager),
            ListCartTool(cart_manager=self.cart_manager)
        ]
    
    async def _execute_tools(self, state: AgentState) -> AgentState:
        # Use direct function calling
        tool_results = []
        for tool in self.selected_tools:
            result = await tool.execute(self.tool_parameters)
            tool_results.append(result)
        return self._process_function_results(tool_results, state)
```

## Extending the Architecture

### Adding a New Agent with MCP Tools

#### Step 1: Define the Agent Class

```python
from src.langgraph_integration.core.base_agent import BaseAgent
from src.langgraph_integration.tools.mcp_client import MCPClient

class CustomMCPAgent(BaseAgent):
    """Custom agent using MCP tools for external integrations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mcp_client = MCPClient()
        self.tools = self._initialize_mcp_tools()
    
    def _initialize_mcp_tools(self):
        """Initialize MCP tools for this agent."""
        return [
            CustomExternalTool(self.mcp_client),
            AnotherExternalTool(self.mcp_client)
        ]
    
    def create_graph(self) -> StateGraph:
        """Create the agent workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("execute_mcp_tools", self._execute_mcp_tools)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define edges
        workflow.add_edge(START, "analyze_request")
        workflow.add_edge("analyze_request", "execute_mcp_tools")
        workflow.add_edge("execute_mcp_tools", "process_results")
        workflow.add_edge("process_results", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow
    
    async def _execute_mcp_tools(self, state: AgentState) -> AgentState:
        """Execute MCP tools and process results."""
        try:
            # Prepare MCP tool calls
            tool_calls = self._prepare_mcp_calls(state)
            
            # Execute via MCP client
            results = await self.mcp_client.call_tools(tool_calls)
            
            # Process and store results
            state["tool_results"] = results
            state["tools_executed"] = [call["tool"] for call in tool_calls]
            
        except Exception as e:
            state["tool_error"] = str(e)
            state["tool_results"] = []
        
        return state
```

#### Step 2: Create MCP Tool Implementations

```python
from src.langgraph_integration.tools.base_mcp_tool import BaseMCPTool

class CustomExternalTool(BaseMCPTool):
    """Custom MCP tool for external service integration."""
    
    name = "custom_external_tool"
    description = "Integrates with external custom service"
    
    def __init__(self, mcp_client: MCPClient):
        super().__init__(mcp_client)
        self.service_endpoint = "https://api.external-service.com"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MCP tool call."""
        try:
            # Prepare MCP request
            mcp_request = {
                "tool": self.name,
                "parameters": parameters,
                "endpoint": self.service_endpoint
            }
            
            # Execute via MCP client
            result = await self.mcp_client.call_tool(mcp_request)
            
            return {
                "success": True,
                "data": result,
                "tool": self.name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }
```

#### Step 3: Register with Router

```python
# In router configuration
class RouterNode:
    def __init__(self):
        self.agents = {
            "qa": ProductQAAgent,
            "cart": ShoppingCartAgent,
            "custom": CustomMCPAgent  # Add new agent
        }
    
    def _classify_intent(self, message: str) -> str:
        # Add classification logic for new agent
        if self._is_custom_intent(message):
            return "custom"
        # ... existing logic
```

### Adding a New Agent with Function Calling Tools

#### Step 1: Define the Agent Class

```python
from src.langgraph_integration.core.base_agent import BaseAgent
from src.langgraph_integration.tools.base_function_tool import BaseFunctionTool

class CustomFunctionAgent(BaseAgent):
    """Custom agent using function calling tools for direct operations."""
    
    def __init__(self, config: Dict[str, Any], service_manager: ServiceManager):
        super().__init__(config)
        self.service_manager = service_manager
        self.tools = self._initialize_function_tools()
    
    def _initialize_function_tools(self):
        """Initialize function calling tools for this agent."""
        return [
            CustomOperationTool(service_manager=self.service_manager),
            AnotherOperationTool(service_manager=self.service_manager)
        ]
    
    async def _execute_function_tools(self, state: AgentState) -> AgentState:
        """Execute function calling tools directly."""
        try:
            results = []
            
            for tool in self.selected_tools:
                # Direct function call
                result = await tool.execute(self.tool_parameters)
                results.append(result)
            
            state["tool_results"] = results
            state["tools_executed"] = [tool.name for tool in self.selected_tools]
            
        except Exception as e:
            state["tool_error"] = str(e)
            state["tool_results"] = []
        
        return state
```

#### Step 2: Create Function Calling Tool Implementations

```python
from src.langgraph_integration.tools.base_function_tool import BaseFunctionTool

class CustomOperationTool(BaseFunctionTool):
    """Custom function calling tool for direct operations."""
    
    name = "custom_operation_tool"
    description = "Performs custom operations directly"
    
    def __init__(self, service_manager: ServiceManager):
        super().__init__()
        self.service_manager = service_manager
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the function call directly."""
        try:
            # Direct service call
            result = await self.service_manager.perform_operation(
                operation=parameters.get("operation"),
                data=parameters.get("data")
            )
            
            return {
                "success": True,
                "result": result,
                "tool": self.name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }
```

## Tool Interface Patterns

### MCP Tool Pattern

Use MCP tools when:
- Integrating with external services
- Need protocol-level abstraction
- Require standardized communication
- Working with third-party APIs

```python
class ExternalServiceMCPTool(BaseMCPTool):
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # MCP protocol handling
        mcp_request = self._build_mcp_request(parameters)
        response = await self.mcp_client.send_request(mcp_request)
        return self._process_mcp_response(response)
```

### Function Calling Pattern

Use function calling tools when:
- Direct system operations needed
- Performance is critical
- Working with internal services
- Need fine-grained control

```python
class DirectOperationTool(BaseFunctionTool):
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Direct function call
        result = await self.service.direct_operation(parameters)
        return self._format_result(result)
```

## Best Practices

### Tool Selection Guidelines

#### Choose MCP Tools When:
- **External Integration**: Connecting to third-party services
- **Protocol Standardization**: Need consistent communication patterns
- **Service Abstraction**: Want to abstract service-specific details
- **Future Flexibility**: May need to swap service implementations

#### Choose Function Calling When:
- **Performance Critical**: Need minimal overhead
- **Direct Control**: Require fine-grained operation control
- **Internal Services**: Working with internal system components
- **Simple Operations**: Straightforward function calls suffice

### Implementation Guidelines

#### 1. Tool Interface Consistency

```python
# Consistent interface across tool types
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        pass
```

#### 2. Error Handling Standardization

```python
# Standardized error handling
def handle_tool_error(self, error: Exception, tool_name: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": str(error),
        "error_type": type(error).__name__,
        "tool": tool_name,
        "timestamp": datetime.utcnow().isoformat()
    }
```

#### 3. Result Format Consistency

```python
# Consistent result format
class ToolResult:
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    tool: str
    execution_time: float
    metadata: Dict[str, Any]
```

## Advanced Patterns

### Hybrid Tool Agents

Some agents may need both tool types:

```python
class HybridAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mcp_tools = self._initialize_mcp_tools()
        self.function_tools = self._initialize_function_tools()
    
    async def _execute_tools(self, state: AgentState) -> AgentState:
        # Determine tool type based on operation
        if self._needs_external_data(state):
            return await self._execute_mcp_tools(state)
        else:
            return await self._execute_function_tools(state)
```

### Tool Composition

Combine multiple tools for complex operations:

```python
class CompositeOperation:
    def __init__(self, mcp_tools: List[BaseMCPTool], 
                 function_tools: List[BaseFunctionTool]):
        self.mcp_tools = mcp_tools
        self.function_tools = function_tools
    
    async def execute_composite(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Execute MCP tools first
        external_data = await self._execute_mcp_tools(parameters)
        
        # Use results in function tools
        internal_result = await self._execute_function_tools({
            **parameters,
            "external_data": external_data
        })
        
        return self._combine_results(external_data, internal_result)
```

### Tool Caching

Implement caching for expensive operations:

```python
class CachedTool(BaseTool):
    def __init__(self, cache_ttl: int = 3600):
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = self._generate_cache_key(parameters)
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]["result"]
        
        result = await self._execute_uncached(parameters)
        self._cache_result(cache_key, result)
        
        return result
```

## Testing Strategies

### Unit Testing Tools

```python
import pytest
from unittest.mock import AsyncMock, Mock

class TestCustomMCPTool:
    @pytest.fixture
    def mcp_client_mock(self):
        return AsyncMock()
    
    @pytest.fixture
    def tool(self, mcp_client_mock):
        return CustomExternalTool(mcp_client_mock)
    
    async def test_successful_execution(self, tool, mcp_client_mock):
        # Setup
        mcp_client_mock.call_tool.return_value = {"data": "test_result"}
        parameters = {"param1": "value1"}
        
        # Execute
        result = await tool.execute(parameters)
        
        # Assert
        assert result["success"] is True
        assert result["data"]["data"] == "test_result"
        mcp_client_mock.call_tool.assert_called_once()
```

### Integration Testing

```python
class TestDualToolIntegration:
    async def test_mcp_and_function_tool_coordination(self):
        # Test that both tool types work together
        qa_agent = ProductQAAgent(config)
        cart_agent = ShoppingCartAgent(config, cart_manager)
        
        # Execute QA operation (MCP)
        qa_result = await qa_agent.process_query("find laptops")
        
        # Execute cart operation (Function calling)
        cart_result = await cart_agent.process_query("add laptop to cart")
        
        # Verify both operations succeeded
        assert qa_result["success"]
        assert cart_result["success"]
```

## Performance Considerations

### Tool Execution Optimization

#### MCP Tool Optimization
- **Connection Pooling**: Reuse MCP connections
- **Batch Requests**: Combine multiple MCP calls
- **Async Execution**: Use async/await for concurrent calls

```python
class OptimizedMCPAgent:
    async def _execute_mcp_tools_batch(self, tool_calls: List[Dict]) -> List[Dict]:
        # Execute multiple MCP calls concurrently
        tasks = [
            self.mcp_client.call_tool(call) 
            for call in tool_calls
        ]
        return await asyncio.gather(*tasks)
```

#### Function Tool Optimization
- **Direct Calls**: Minimize abstraction overhead
- **Connection Reuse**: Share database connections
- **Caching**: Cache frequently accessed data

```python
class OptimizedFunctionAgent:
    async def _execute_function_tools_optimized(self, tools: List[BaseFunctionTool]) -> List[Dict]:
        # Execute with shared resources
        async with self.resource_pool.acquire() as resource:
            results = []
            for tool in tools:
                tool.set_resource(resource)
                result = await tool.execute_optimized()
                results.append(result)
            return results
```

## Monitoring and Observability

### Tool Execution Metrics

```python
class ToolMetrics:
    def __init__(self):
        self.execution_times = {}
        self.success_rates = {}
        self.error_counts = {}
    
    def record_execution(self, tool_name: str, execution_time: float, success: bool):
        # Record metrics for monitoring
        if tool_name not in self.execution_times:
            self.execution_times[tool_name] = []
        
        self.execution_times[tool_name].append(execution_time)
        
        if success:
            self.success_rates[tool_name] = self.success_rates.get(tool_name, 0) + 1
        else:
            self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1
```

### Logging Standards

```python
import logging
from typing import Dict, Any

class ToolLogger:
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def log_tool_execution(self, tool_type: str, tool_name: str, 
                          parameters: Dict[str, Any], result: Dict[str, Any]):
        self.logger.info(
            "Tool execution",
            extra={
                "tool_type": tool_type,  # "mcp" or "function"
                "tool_name": tool_name,
                "parameters": parameters,
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0)
            }
        )
```

## Migration Guide

### From Single Tool Type to Dual Architecture

#### Step 1: Identify Tool Requirements
- Analyze current tools and their usage patterns
- Determine which tools should use MCP vs function calling
- Plan migration strategy

#### Step 2: Implement Base Classes
- Create base classes for both tool types
- Ensure consistent interfaces
- Add migration utilities

#### Step 3: Migrate Tools Gradually
- Start with new tools using appropriate interfaces
- Migrate existing tools one at a time
- Maintain backward compatibility during transition

#### Step 4: Update Agent Configurations
- Modify agent initialization to support both tool types
- Update routing logic if needed
- Test thoroughly before deployment

## Troubleshooting

### Common Issues

#### MCP Connection Problems
```python
# Debug MCP connectivity
async def debug_mcp_connection(mcp_client: MCPClient):
    try:
        health_check = await mcp_client.health_check()
        print(f"MCP Health: {health_check}")
    except Exception as e:
        print(f"MCP Connection Error: {e}")
```

#### Function Tool Errors
```python
# Debug function tool execution
async def debug_function_tool(tool: BaseFunctionTool, parameters: Dict):
    try:
        result = await tool.execute(parameters)
        print(f"Tool Result: {result}")
    except Exception as e:
        print(f"Function Tool Error: {e}")
        import traceback
        traceback.print_exc()
```

### Performance Issues

#### Slow MCP Calls
- Check network connectivity
- Verify MCP server performance
- Consider caching strategies
- Implement timeout handling

#### Function Tool Bottlenecks
- Profile function execution
- Check database connection pools
- Optimize query patterns
- Consider async execution

## Future Enhancements

### Planned Features

1. **Tool Auto-Discovery**: Automatically detect and register new tools
2. **Dynamic Tool Selection**: Choose tool type based on runtime conditions
3. **Tool Composition Framework**: Easier composition of complex operations
4. **Performance Analytics**: Advanced performance monitoring and optimization
5. **Tool Marketplace**: Registry of available tools and their capabilities

### Extension Points

The architecture provides several extension points:

- **Custom Tool Types**: Add new tool interface types beyond MCP and function calling
- **Tool Middleware**: Add middleware for cross-cutting concerns (logging, caching, etc.)
- **Agent Templates**: Create templates for common agent patterns
- **Tool Orchestration**: Advanced tool coordination and workflow management

## Conclusion

The Dual Tool Architecture provides a flexible and extensible foundation for building agents with different tool requirements. By supporting both MCP and function calling tools, the system can optimize for different use cases while maintaining consistency and ease of development.

Key benefits:
- **Flexibility**: Choose the right tool interface for each use case
- **Performance**: Optimize tool execution based on requirements
- **Maintainability**: Clear separation of concerns and consistent patterns
- **Extensibility**: Easy to add new agents and tool types

Follow the patterns and guidelines in this document to successfully extend the architecture for your specific needs.