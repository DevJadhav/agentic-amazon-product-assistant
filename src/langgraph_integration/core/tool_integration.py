"""
Tool integration system for LangGraph agents.
Provides unified interface for function calling tools and MCP tools.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool

from .utils import log_agent_step

logger = logging.getLogger(__name__)


class ToolCallResult:
    """Result of a tool call with metadata."""
    
    def __init__(
        self,
        tool_name: str,
        success: bool,
        result: Any,
        error: Optional[str] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ToolIntegrationInterface(ABC):
    """Abstract interface for tool integration systems."""
    
    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: str
    ) -> ToolCallResult:
        """Call a tool with parameters and return result."""
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        pass
    
    @abstractmethod
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        pass


class FunctionCallingToolIntegration(ToolIntegrationInterface):
    """Integration for LangChain function calling tools."""
    
    def __init__(self, tools: List[BaseTool], session_id_injector: Optional[callable] = None):
        """
        Initialize function calling tool integration.
        
        Args:
            tools: List of LangChain BaseTool instances
            session_id_injector: Function to inject session ID into tools
        """
        self.tools = {tool.name: tool for tool in tools}
        self.session_id_injector = session_id_injector
        self.logger = logging.getLogger(__name__)
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: str
    ) -> ToolCallResult:
        """Call a function calling tool."""
        
        start_time = datetime.utcnow()
        
        try:
            # Get the tool
            tool = self.tools.get(tool_name)
            if not tool:
                return ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Tool '{tool_name}' not found"
                )
            
            # Inject session ID if injector is provided
            if self.session_id_injector:
                self.session_id_injector(tool, session_id)
            
            # Log tool call start
            log_agent_step(
                session_id,
                "tool_call_start",
                {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "tool_type": "function_calling"
                }
            )
            
            # Call the tool
            if hasattr(tool, '_arun'):
                result = await tool._arun(**parameters)
            else:
                result = tool._run(**parameters)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log successful tool call
            log_agent_step(
                session_id,
                "tool_call_success",
                {
                    "tool_name": tool_name,
                    "execution_time": execution_time,
                    "result_type": type(result).__name__
                }
            )
            
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"tool_type": "function_calling"}
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            self.logger.error(f"Function calling tool '{tool_name}' failed: {error_msg}")
            
            # Log failed tool call
            log_agent_step(
                session_id,
                "tool_call_error",
                {
                    "tool_name": tool_name,
                    "error": error_msg,
                    "execution_time": execution_time
                }
            )
            
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                metadata={"tool_type": "function_calling"}
            )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available function calling tools."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a function calling tool."""
        
        tool = self.tools.get(tool_name)
        if not tool:
            return None
        
        return {
            "name": tool.name,
            "description": tool.description,
            "tool_type": "function_calling",
            "args_schema": tool.args_schema.__name__ if tool.args_schema else None,
            "return_direct": getattr(tool, 'return_direct', False)
        }


class MCPToolIntegration(ToolIntegrationInterface):
    """Integration for MCP (Model Context Protocol) tools."""
    
    def __init__(self, mcp_client: Any):
        """
        Initialize MCP tool integration.
        
        Args:
            mcp_client: MCP client instance
        """
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(__name__)
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: str
    ) -> ToolCallResult:
        """Call an MCP tool."""
        
        start_time = datetime.utcnow()
        
        try:
            # Log tool call start
            log_agent_step(
                session_id,
                "tool_call_start",
                {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "tool_type": "mcp"
                }
            )
            
            # Call MCP tool
            result = await self.mcp_client.call_tool(tool_name, parameters)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log successful tool call
            log_agent_step(
                session_id,
                "tool_call_success",
                {
                    "tool_name": tool_name,
                    "execution_time": execution_time,
                    "result_type": type(result).__name__
                }
            )
            
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"tool_type": "mcp"}
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            self.logger.error(f"MCP tool '{tool_name}' failed: {error_msg}")
            
            # Log failed tool call
            log_agent_step(
                session_id,
                "tool_call_error",
                {
                    "tool_name": tool_name,
                    "error": error_msg,
                    "execution_time": execution_time
                }
            )
            
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                metadata={"tool_type": "mcp"}
            )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        # This would query the MCP client for available tools
        # Implementation depends on MCP client interface
        return []
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an MCP tool."""
        # This would query the MCP client for tool information
        # Implementation depends on MCP client interface
        return None


class HybridToolIntegration:
    """Hybrid tool integration supporting both function calling and MCP tools."""
    
    def __init__(
        self,
        function_calling_integration: Optional[FunctionCallingToolIntegration] = None,
        mcp_integration: Optional[MCPToolIntegration] = None
    ):
        """
        Initialize hybrid tool integration.
        
        Args:
            function_calling_integration: Function calling tool integration
            mcp_integration: MCP tool integration
        """
        self.function_calling = function_calling_integration
        self.mcp = mcp_integration
        self.logger = logging.getLogger(__name__)
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: str,
        tool_type: Optional[str] = None
    ) -> ToolCallResult:
        """
        Call a tool using the appropriate integration.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            session_id: Session identifier
            tool_type: Preferred tool type ('function_calling' or 'mcp')
        """
        
        # Try function calling tools first if no type specified or if specified
        if (tool_type is None or tool_type == "function_calling") and self.function_calling:
            if tool_name in self.function_calling.get_available_tools():
                return await self.function_calling.call_tool(tool_name, parameters, session_id)
        
        # Try MCP tools if function calling didn't work or if specified
        if (tool_type is None or tool_type == "mcp") and self.mcp:
            if tool_name in self.mcp.get_available_tools():
                return await self.mcp.call_tool(tool_name, parameters, session_id)
        
        # Tool not found in any integration
        return ToolCallResult(
            tool_name=tool_name,
            success=False,
            result=None,
            error=f"Tool '{tool_name}' not found in any integration"
        )
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get all available tools by type."""
        
        tools = {}
        
        if self.function_calling:
            tools["function_calling"] = self.function_calling.get_available_tools()
        
        if self.mcp:
            tools["mcp"] = self.mcp.get_available_tools()
        
        return tools
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tool from any integration."""
        
        # Try function calling first
        if self.function_calling:
            info = self.function_calling.get_tool_info(tool_name)
            if info:
                return info
        
        # Try MCP
        if self.mcp:
            info = self.mcp.get_tool_info(tool_name)
            if info:
                return info
        
        return None


class ToolCallLogger:
    """Logger for tool calls with monitoring and analytics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.call_history: List[Dict[str, Any]] = []
    
    def log_tool_call(self, result: ToolCallResult, session_id: str):
        """Log a tool call result."""
        
        log_entry = {
            "session_id": session_id,
            "timestamp": result.timestamp.isoformat(),
            "tool_name": result.tool_name,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error,
            "metadata": result.metadata
        }
        
        self.call_history.append(log_entry)
        
        # Log to standard logger
        if result.success:
            self.logger.info(
                f"Tool call successful: {result.tool_name} "
                f"({result.execution_time:.3f}s)"
            )
        else:
            self.logger.error(
                f"Tool call failed: {result.tool_name} - {result.error}"
            )
    
    def get_call_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get tool call statistics."""
        
        # Filter by session if specified
        calls = self.call_history
        if session_id:
            calls = [call for call in calls if call["session_id"] == session_id]
        
        if not calls:
            return {"total_calls": 0}
        
        successful_calls = [call for call in calls if call["success"]]
        failed_calls = [call for call in calls if not call["success"]]
        
        # Calculate execution time statistics
        execution_times = [
            call["execution_time"] for call in successful_calls 
            if call["execution_time"] is not None
        ]
        
        stats = {
            "total_calls": len(calls),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "success_rate": len(successful_calls) / len(calls) if calls else 0,
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "tool_usage": {}
        }
        
        # Tool usage statistics
        for call in calls:
            tool_name = call["tool_name"]
            if tool_name not in stats["tool_usage"]:
                stats["tool_usage"][tool_name] = {"calls": 0, "successes": 0}
            
            stats["tool_usage"][tool_name]["calls"] += 1
            if call["success"]:
                stats["tool_usage"][tool_name]["successes"] += 1
        
        return stats


# Global tool call logger instance
_global_tool_logger = ToolCallLogger()


def get_global_tool_logger() -> ToolCallLogger:
    """Get the global tool call logger instance."""
    return _global_tool_logger


def create_session_id_injector(session_id: str) -> callable:
    """Create a session ID injector function for tools."""
    
    def inject_session_id(tool: BaseTool, target_session_id: str):
        """Inject session ID into a tool."""
        
        def get_session_id():
            return target_session_id
        
        # Override the tool's session ID method
        tool._get_session_id = get_session_id
    
    return inject_session_id