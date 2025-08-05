"""
Shopping Cart Agent for LangGraph workflows.
Specialized agent for shopping cart management operations using function calling tools.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from .base_agent import BaseAgent
from .state_schemas import AgentState, update_state_step
from .utils import log_agent_step, create_error_response
from .tool_integration import (
    FunctionCallingToolIntegration, 
    ToolCallResult, 
    get_global_tool_logger,
    create_session_id_injector
)
from ..tools.shopping_cart_tools import create_cart_tools, get_cart_tools_info
from ..state.shopping_cart_manager import ShoppingCartManager, get_global_cart_manager

logger = logging.getLogger(__name__)


class ShoppingCartAgent(BaseAgent):
    """Agent specialized for shopping cart management operations."""
    
    def __init__(self, config: Dict[str, Any], cart_manager: Optional[ShoppingCartManager] = None):
        """Initialize Shopping Cart Agent with configuration and cart manager."""
        super().__init__(config)
        self.cart_manager = cart_manager or get_global_cart_manager()
        self.tools = create_cart_tools(self.cart_manager)
        self.logger = logging.getLogger(__name__)
        
        # Create tool integration system
        self.tool_integration = FunctionCallingToolIntegration(
            tools=self.tools,
            session_id_injector=None  # Will be set during processing
        )
        
        # Tool call logger
        self.tool_logger = get_global_tool_logger()
        
        # Agent configuration
        self.max_tool_calls = config.get("max_tool_calls", 5)
        self.session_id = None  # Will be set from state during processing
    
    def create_graph(self) -> StateGraph:
        """Create shopping cart management workflow graph."""
        
        workflow = StateGraph(AgentState)
        
        # Add workflow nodes
        workflow.add_node("analyze_cart_request", self._analyze_cart_request)
        workflow.add_node("execute_cart_operation", self._execute_cart_operation)
        workflow.add_node("generate_cart_response", self._generate_cart_response)
        workflow.add_node("update_cart_state", self._update_cart_state)
        
        # Define workflow edges
        workflow.add_edge(START, "analyze_cart_request")
        workflow.add_edge("analyze_cart_request", "execute_cart_operation")
        workflow.add_edge("execute_cart_operation", "generate_cart_response")
        workflow.add_edge("generate_cart_response", "update_cart_state")
        workflow.add_edge("update_cart_state", END)
        
        return workflow
    
    async def _analyze_cart_request(self, state: AgentState) -> AgentState:
        """
        Analyze cart operation request for operation type detection.
        
        This node examines the user's query to determine:
        - What cart operation they want to perform (add, remove, list, clear)
        - Extract relevant parameters (product info, quantities, etc.)
        - Validate the request and prepare for execution
        """
        
        try:
            # Set session ID for this processing
            self.session_id = state["session_id"]
            
            log_agent_step(
                state["session_id"],
                "cart_request_analysis_start",
                {"query": state["current_query"]}
            )
            
            # Analyze the user query to determine cart operation
            query = state["current_query"].lower().strip()
            
            # Determine cart operation type
            cart_operation = self._classify_cart_operation(query)
            
            # Extract operation parameters based on operation type
            operation_params = self._extract_operation_parameters(query, cart_operation, state)
            
            # Validate the operation and parameters
            validation_result = self._validate_cart_operation(cart_operation, operation_params)
            
            # Update state with analysis results
            updated_state = update_state_step(
                state,
                "analyze_cart_request",
                cart_operation=cart_operation,
                cart_operation_params=operation_params,
                cart_operation_success=validation_result["valid"],
                cart_operation_message=validation_result.get("message", "")
            )
            
            log_agent_step(
                state["session_id"],
                "cart_request_analyzed",
                {
                    "operation": cart_operation,
                    "params": operation_params,
                    "valid": validation_result["valid"]
                }
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Cart request analysis failed: {e}")
            
            error_state = update_state_step(
                state,
                "analyze_cart_request",
                cart_operation="error",
                cart_operation_success=False,
                cart_operation_message=f"Failed to analyze cart request: {str(e)}",
                error_state=str(e)
            )
            
            return error_state
    
    async def _execute_cart_operation(self, state: AgentState) -> AgentState:
        """
        Execute cart operation using function calling tools.
        
        This node:
        - Calls the appropriate cart tool based on the analyzed operation
        - Handles tool execution and error scenarios
        - Processes tool results for response generation
        """
        
        try:
            log_agent_step(
                state["session_id"],
                "cart_operation_execution_start",
                {"operation": state.get("cart_operation")}
            )
            
            cart_operation = state.get("cart_operation")
            operation_params = state.get("cart_operation_params", {})
            
            # Skip execution if analysis failed
            if not state.get("cart_operation_success", False):
                return update_state_step(
                    state,
                    "execute_cart_operation",
                    cart_operation_result={
                        "success": False,
                        "error": "Cannot execute operation due to analysis failure",
                        "message": state.get("cart_operation_message", "Unknown error")
                    }
                )
            
            # Execute the appropriate tool using tool integration
            tool_call_result = await self._call_cart_tool_integrated(cart_operation, operation_params)
            
            # Log the tool call
            self.tool_logger.log_tool_call(tool_call_result, state["session_id"])
            
            # Convert tool call result to legacy format for compatibility
            tool_result = tool_call_result.result if tool_call_result.success else {
                "success": False,
                "error": tool_call_result.error,
                "message": f"Tool call failed: {tool_call_result.error}"
            }
            
            # Update state with tool execution results
            # Operation success depends on both tool call success AND operation result
            operation_success = tool_call_result.success and tool_result.get("success", False)
            
            updated_state = update_state_step(
                state,
                "execute_cart_operation",
                cart_operation_result=tool_result,
                cart_operation_success=operation_success,
                cart_updated=tool_result.get("cart_updated", False) if operation_success else False
            )
            
            # Add tool call to tracking using new format
            updated_state["tool_calls"] = state.get("tool_calls", []) + [tool_call_result.to_dict()]
            
            log_agent_step(
                state["session_id"],
                "cart_operation_executed",
                {
                    "operation": cart_operation,
                    "success": tool_result.get("success", False),
                    "result": tool_result
                }
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Cart operation execution failed: {e}")
            
            error_result = {
                "success": False,
                "error": f"Tool execution error: {str(e)}",
                "message": "Failed to execute cart operation due to system error"
            }
            
            error_state = update_state_step(
                state,
                "execute_cart_operation",
                cart_operation_result=error_result,
                cart_operation_success=False,
                error_state=str(e)
            )
            
            return error_state
    
    async def _generate_cart_response(self, state: AgentState) -> AgentState:
        """
        Generate response based on cart operation results with operation confirmations.
        
        This node:
        - Creates user-friendly responses based on tool results
        - Includes operation confirmations and cart status
        - Handles both success and error scenarios
        """
        
        try:
            log_agent_step(
                state["session_id"],
                "cart_response_generation_start",
                {"operation": state.get("cart_operation")}
            )
            
            cart_operation = state.get("cart_operation")
            operation_result = state.get("cart_operation_result", {})
            
            # Generate response based on operation type and result
            response = self._create_operation_response(cart_operation, operation_result)
            
            # Add cart status information if operation was successful
            if operation_result.get("success") and state.get("cart_updated"):
                cart_status = await self._get_current_cart_status()
                response += f"\n\n{cart_status}"
            
            # Update state with generated response
            updated_state = update_state_step(
                state,
                "generate_cart_response",
                final_response=response,
                context_for_llm=f"Cart operation: {cart_operation}, Result: {operation_result}",
                response_metadata={
                    "operation_type": cart_operation,
                    "operation_success": operation_result.get("success", False),
                    "cart_updated": state.get("cart_updated", False)
                }
            )
            
            log_agent_step(
                state["session_id"],
                "cart_response_generated",
                {"response_length": len(response)}
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Cart response generation failed: {e}")
            
            error_response = "I encountered an error while processing your cart request. Please try again."
            
            error_state = update_state_step(
                state,
                "generate_cart_response",
                final_response=error_response,
                error_state=str(e)
            )
            
            return error_state
    
    async def _update_cart_state(self, state: AgentState) -> AgentState:
        """
        Update cart state for conversation context.
        
        This node:
        - Updates the conversation state with current cart information
        - Refreshes cart contents and summary data
        - Prepares state for potential follow-up interactions
        """
        
        try:
            log_agent_step(
                state["session_id"],
                "cart_state_update_start",
                {}
            )
            
            # Get current cart contents and summary
            current_cart_contents = self.cart_manager.get_cart_contents(state["session_id"])
            cart_summary = self.cart_manager.get_cart_summary(state["session_id"])
            
            # Update state with current cart information
            updated_state = update_state_step(
                state,
                "update_cart_state",
                current_cart_contents=current_cart_contents,
                cart_item_count=len(current_cart_contents),
                cart_total=cart_summary.get("total_value", 0.0),
                workflow_status="completed"
            )
            
            # Add cart summary to response metadata
            updated_state["response_metadata"]["cart_summary"] = cart_summary
            
            log_agent_step(
                state["session_id"],
                "cart_state_updated",
                {
                    "item_count": len(current_cart_contents),
                    "total_value": cart_summary.get("total_value", 0.0)
                }
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Cart state update failed: {e}")
            
            # Don't fail the entire workflow for state update errors
            error_state = update_state_step(
                state,
                "update_cart_state",
                workflow_status="completed",
                error_state=f"State update error: {str(e)}"
            )
            
            return error_state
    
    # Helper methods for cart operation processing
    
    def _classify_cart_operation(self, query: str) -> str:
        """Classify the type of cart operation from user query."""
        
        # Simple keyword-based classification
        # In a production system, this could use more sophisticated NLP
        
        if any(word in query for word in ["add", "put", "place", "include"]):
            return "add"
        elif any(word in query for word in ["remove", "delete", "take out", "drop"]):
            return "remove"
        elif any(word in query for word in ["show", "list", "view", "see", "display", "what's in"]):
            return "list"
        elif any(word in query for word in ["clear", "empty", "remove all", "delete all"]):
            return "clear"
        else:
            # Default to list if unclear
            return "list"
    
    def _extract_operation_parameters(
        self, 
        query: str, 
        operation: str, 
        state: AgentState
    ) -> Dict[str, Any]:
        """Extract parameters for cart operation from query and state."""
        
        params = {}
        
        if operation == "add":
            # For add operations, try to extract product information
            # This is a simplified implementation - in production, you'd use
            # more sophisticated entity extraction
            
            # Check if there's a selected product in the state
            selected_product = state.get("selected_product_for_cart")
            if selected_product:
                params.update({
                    "product_id": selected_product.get("product_id", "unknown"),
                    "product_title": selected_product.get("title", "Unknown Product"),
                    "price": selected_product.get("price"),
                    "image_url": selected_product.get("image_url"),
                    "metadata": selected_product.get("metadata", {})
                })
            
            # Try to extract quantity from query
            quantity = self._extract_quantity_from_query(query)
            if quantity:
                params["quantity"] = quantity
            
        elif operation == "remove":
            # For remove operations, extract product ID and optional quantity
            selected_product = state.get("selected_product_for_cart")
            if selected_product:
                params["product_id"] = selected_product.get("product_id", "unknown")
            
            quantity = self._extract_quantity_from_query(query)
            if quantity:
                params["quantity"] = quantity
        
        elif operation in ["list", "clear"]:
            # These operations don't need additional parameters
            pass
        
        return params
    
    def _extract_quantity_from_query(self, query: str) -> Optional[int]:
        """Extract quantity from user query using simple pattern matching."""
        
        import re
        
        # Look for patterns like "2 items", "add 3", "remove 1", etc.
        quantity_patterns = [
            r'\b(\d+)\s*(?:items?|pieces?|units?)\b',
            r'\b(?:add|remove|put|take)\s*(\d+)\b',
            r'\b(\d+)\s*(?:of|x)\b'
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, query.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _validate_cart_operation(
        self, 
        operation: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate cart operation and parameters."""
        
        if operation == "add":
            if not params.get("product_id") or not params.get("product_title"):
                return {
                    "valid": False,
                    "message": "Cannot add item to cart: missing product information. Please select a product first."
                }
            
            quantity = params.get("quantity", 1)
            if quantity <= 0 or quantity > 100:
                return {
                    "valid": False,
                    "message": "Invalid quantity. Please specify a quantity between 1 and 100."
                }
        
        elif operation == "remove":
            if not params.get("product_id"):
                return {
                    "valid": False,
                    "message": "Cannot remove item: missing product information. Please specify which item to remove."
                }
        
        elif operation in ["list", "clear"]:
            # These operations are always valid
            pass
        
        else:
            return {
                "valid": False,
                "message": f"Unknown cart operation: {operation}"
            }
        
        return {"valid": True, "message": "Operation is valid"}
    
    async def _call_cart_tool_integrated(
        self, 
        operation: str, 
        params: Dict[str, Any]
    ) -> ToolCallResult:
        """Call the appropriate cart tool using the tool integration system."""
        
        # Map operations to tool names
        tool_name_map = {
            "add": "add_to_cart",
            "remove": "remove_from_cart",
            "list": "list_cart",
            "clear": "clear_cart"
        }
        
        tool_name = tool_name_map.get(operation)
        if not tool_name:
            return ToolCallResult(
                tool_name=f"unknown_operation_{operation}",
                success=False,
                result=None,
                error=f"No tool available for operation: {operation}"
            )
        
        # Set up session ID injector for this call
        if self.session_id:
            injector = create_session_id_injector(self.session_id)
            self.tool_integration.session_id_injector = injector
        
        # Prepare parameters based on operation type
        if operation == "list":
            # List operation has specific parameter handling
            tool_params = {
                "include_summary": params.get("include_summary", True),
                "format_type": params.get("format_type", "detailed")
            }
        elif operation == "clear":
            # Clear operation has no parameters
            tool_params = {}
        else:
            # Add and remove operations use params directly
            tool_params = params
        
        # Call the tool through integration system
        return await self.tool_integration.call_tool(
            tool_name=tool_name,
            parameters=tool_params,
            session_id=self.session_id or "unknown_session"
        )
    

    
    def _create_operation_response(
        self, 
        operation: str, 
        result: Dict[str, Any]
    ) -> str:
        """Create user-friendly response based on operation result."""
        
        if not result.get("success", False):
            error_message = result.get("message", result.get("error", "Unknown error"))
            return f"I couldn't complete that cart operation. {error_message}"
        
        # Success responses based on operation type
        if operation == "add":
            item = result.get("item", {})
            product_title = item.get("product_title", "item")
            quantity = item.get("quantity", 1)
            
            if result.get("action") == "updated":
                return f"I've updated the quantity of {product_title} in your cart to {quantity}."
            else:
                return f"I've added {quantity} x {product_title} to your cart."
        
        elif operation == "remove":
            item = result.get("item", {})
            product_title = item.get("product_title", "item")
            
            if result.get("removed_completely", False):
                return f"I've removed {product_title} from your cart."
            else:
                quantity = item.get("quantity", 0)
                return f"I've updated {product_title} in your cart. New quantity: {quantity}."
        
        elif operation == "list":
            if result.get("is_empty", True):
                return "Your cart is empty."
            else:
                items = result.get("cart_items", [])
                item_count = len(items)
                
                response = f"Your cart contains {item_count} item{'s' if item_count != 1 else ''}:\n\n"
                
                for item in items:
                    title = item.get("product_title", "Unknown")
                    quantity = item.get("quantity", 1)
                    price = item.get("price_per_unit")
                    
                    line = f"• {quantity} x {title}"
                    if price:
                        subtotal = item.get("subtotal", price * quantity)
                        line += f" - ${subtotal:.2f}"
                    
                    response += line + "\n"
                
                # Add total if available
                cart_summary = result.get("cart_summary", {})
                total_value = cart_summary.get("total_value", 0)
                if total_value > 0:
                    response += f"\nTotal: ${total_value:.2f}"
                
                return response
        
        elif operation == "clear":
            items_removed = result.get("items_removed", 0)
            if items_removed == 0:
                return "Your cart was already empty."
            else:
                return f"I've cleared your cart. Removed {items_removed} item{'s' if items_removed != 1 else ''}."
        
        # Fallback response
        return result.get("message", "Cart operation completed successfully.")
    
    async def _get_current_cart_status(self) -> str:
        """Get current cart status for inclusion in responses."""
        
        try:
            cart_summary = self.cart_manager.get_cart_summary(self.session_id)
            
            if cart_summary.get("is_empty", True):
                return "Your cart is now empty."
            else:
                total_items = cart_summary.get("total_items", 0)
                total_value = cart_summary.get("total_value", 0)
                
                status = f"Your cart now has {total_items} item{'s' if total_items != 1 else ''}"
                if total_value > 0:
                    status += f" with a total value of ${total_value:.2f}"
                status += "."
                
                return status
        
        except Exception as e:
            self.logger.error(f"Failed to get cart status: {e}")
            return ""
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this Shopping Cart Agent."""
        
        base_info = super().get_agent_info()
        
        cart_info = {
            "tools_available": self.tool_integration.get_available_tools(),
            "tool_count": len(self.tools),
            "cart_manager_connected": self.cart_manager is not None,
            "max_tool_calls": self.max_tool_calls,
            "supported_operations": ["add", "remove", "list", "clear"],
            "tool_integration_type": "function_calling",
            "tool_logger_enabled": self.tool_logger is not None
        }
        
        base_info.update(cart_info)
        return base_info