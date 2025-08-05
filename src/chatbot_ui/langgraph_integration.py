"""
Streamlit integration for LangGraph agent workflows.
Provides session management and agent interaction capabilities.
"""

import streamlit as st
import requests
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configuration for API endpoints
API_BASE_URL = "http://localhost:8000"  # Adjust as needed


class StreamlitLangGraphIntegration:
    """Manages LangGraph integration with Streamlit frontend."""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        """Initialize Streamlit LangGraph integration."""
        self.api_base_url = api_base_url
        self._initialize_cart_manager()
    
    def initialize_session(self) -> str:
        """Initialize or get existing session ID."""
        
        if 'langgraph_session_id' not in st.session_state:
            # Generate new session ID
            session_id = f"streamlit_{uuid.uuid4().hex[:16]}"
            st.session_state.langgraph_session_id = session_id
            st.session_state.langgraph_conversation_turn = 0
            st.session_state.langgraph_agent_history = []
            
            # Initialize session metadata
            st.session_state.langgraph_session_metadata = {
                "created_at": datetime.now().isoformat(),
                "total_queries": 0,
                "agent_type": "ambient",
                "enable_memory": True
            }
        
        return st.session_state.langgraph_session_id
    
    def _initialize_cart_manager(self):
        """Initialize cart state manager."""
        if 'cart_manager' not in st.session_state:
            try:
                from chatbot_ui.ui_components import CartStateManager
                st.session_state.cart_manager = CartStateManager()
            except (ImportError, AttributeError):
                # Fallback if ui_components not available or streamlit not properly initialized
                st.session_state.cart_manager = None
    
    def _update_cart_state(self, cart_data: List[Dict[str, Any]], cart_updated: bool):
        """Update cart state from API response."""
        try:
            # Ensure cart manager is available
            if 'cart_manager' not in st.session_state or st.session_state.cart_manager is None:
                self._initialize_cart_manager()
            
            if st.session_state.cart_manager is not None:
                # Transform cart data to expected format
                formatted_cart_data = {
                    "items": cart_data if cart_data else [],
                    "total_items": sum(item.get("quantity", 0) for item in (cart_data or [])),
                    "total_value": sum(
                        item.get("quantity", 0) * item.get("product_price", 0) 
                        for item in (cart_data or [])
                        if item.get("product_price") is not None
                    )
                }
                
                # Update cart display
                st.session_state.cart_manager.update_cart_display(formatted_cart_data)
                
                # Store cart update notification with enhanced details
                if cart_updated:
                    # Determine the type of cart operation for better messaging
                    operation_type = "updated"
                    if formatted_cart_data["total_items"] == 0:
                        operation_type = "cleared"
                    elif len(cart_data or []) == 1 and cart_data[0].get("quantity", 0) > 0:
                        operation_type = "item added"
                    
                    st.session_state.cart_update_notification = {
                        "message": f"Cart {operation_type} successfully!",
                        "timestamp": time.time(),
                        "type": "success",
                        "details": {
                            "total_items": formatted_cart_data["total_items"],
                            "total_value": formatted_cart_data["total_value"],
                            "operation": operation_type
                        }
                    }
                    
                    # Force UI refresh by updating a counter
                    if 'cart_update_counter' not in st.session_state:
                        st.session_state.cart_update_counter = 0
                    st.session_state.cart_update_counter += 1
            else:
                # Cart manager not available, just store notification
                if cart_updated:
                    st.session_state.cart_update_notification = {
                        "message": "Cart updated (display unavailable)",
                        "timestamp": time.time(),
                        "type": "info",
                        "details": {}
                    }
        except Exception as e:
            # Handle cart update errors gracefully
            if hasattr(st.session_state, 'cart_manager') and st.session_state.cart_manager is not None:
                try:
                    st.session_state.cart_manager.set_cart_error(f"Failed to update cart: {str(e)}")
                except:
                    pass
            
            # Store error notification
            st.session_state.cart_update_notification = {
                "message": f"Cart update failed: {str(e)}",
                "timestamp": time.time(),
                "type": "error",
                "details": {"error": str(e)}
            }
    
    def send_message_to_agent(
        self, 
        message: str, 
        session_id: str,
        agent_type: str = "ambient",
        max_products: int = 5,
        max_reviews: int = 3,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        enable_memory: bool = True
    ) -> Dict[str, Any]:
        """Send message to LangGraph agent and get response."""
        
        try:
            # Prepare request payload
            payload = {
                "query": message,
                "session_id": session_id,
                "max_products": max_products,
                "max_reviews": max_reviews,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "temperature": temperature,
                "enable_memory": enable_memory,
                "agent_type": agent_type
            }
            
            # Send request to agent endpoint
            response = requests.post(
                f"{self.api_base_url}/agent/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update session state
                st.session_state.langgraph_conversation_turn = result.get("conversation_turn", 0)
                st.session_state.langgraph_session_metadata["total_queries"] += 1
                
                # Process cart data if present
                cart_data = result.get("cart_data")
                cart_updated = result.get("cart_updated", False)
                
                if cart_data is not None or cart_updated:
                    # Update cart state in session
                    self._update_cart_state(cart_data, cart_updated)
                    
                    # Store cart data for persistence
                    self._persist_cart_data(cart_data, result.get("cart_item_count", 0), result.get("cart_total"))
                
                # Add to history
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": message,
                    "response": result.get("response", ""),
                    "agent_workflow": result.get("agent_workflow", "unknown"),
                    "processing_time": result.get("processing_time", 0),
                    "products_found": result.get("products_found", 0),
                    "reviews_found": result.get("reviews_found", 0),
                    "workflow_steps": result.get("workflow_steps", []),
                    "error_state": result.get("error_state"),
                    "cart_updated": cart_updated,
                    "cart_item_count": result.get("cart_item_count", 0),
                    "routing_decision": result.get("routing_decision"),
                    "agent_used": result.get("agent_used", "unknown"),
                    "tools_called": result.get("tools_called", [])
                }
                
                if 'langgraph_agent_history' not in st.session_state:
                    st.session_state.langgraph_agent_history = []
                
                st.session_state.langgraph_agent_history.append(history_entry)
                
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "data": result,
                    "cart_updated": cart_updated,
                    "cart_data": cart_data
                }
            else:
                return {
                    "success": False,
                    "error": f"API request failed with status {response.status_code}",
                    "response": "I apologize, but I'm having trouble connecting to the agent service."
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "response": "The request took too long to process. Please try again."
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error",
                "response": "I'm having trouble connecting to the agent service. Please check if the API is running."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"An unexpected error occurred: {str(e)}"
            }
    
    def get_conversation_state(self, session_id: str) -> Dict[str, Any]:
        """Get current conversation state from the agent."""
        
        try:
            response = requests.get(
                f"{self.api_base_url}/agent/status/{session_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get conversation state: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history from the agent."""
        
        try:
            response = requests.get(
                f"{self.api_base_url}/agent/history/{session_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get conversation history: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history."""
        
        try:
            response = requests.delete(
                f"{self.api_base_url}/agent/conversation/{session_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                # Clear local session state
                if 'langgraph_agent_history' in st.session_state:
                    st.session_state.langgraph_agent_history = []
                st.session_state.langgraph_conversation_turn = 0
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Failed to clear conversation: {e}")
            return False
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities and system status."""
        
        try:
            response = requests.get(
                f"{self.api_base_url}/agent/capabilities",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get capabilities: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def display_agent_thinking(self, workflow_steps: List[str]) -> None:
        """Display agent thinking process and workflow steps."""
        
        if not workflow_steps:
            return
        
        with st.expander("🤖 Agent Workflow Steps", expanded=False):
            for i, step in enumerate(workflow_steps, 1):
                # Format step names for better readability
                formatted_step = step.replace("_", " ").title()
                st.write(f"{i}. {formatted_step}")
    
    def display_session_info(self, session_id: str) -> None:
        """Display session information and statistics."""
        
        with st.sidebar:
            st.subheader("🔗 Session Info")
            
            # Session ID (truncated for display)
            st.text(f"Session: {session_id[-8:]}")
            
            # Conversation statistics
            turn = st.session_state.get('langgraph_conversation_turn', 0)
            total_queries = st.session_state.get('langgraph_session_metadata', {}).get('total_queries', 0)
            
            st.metric("Conversation Turn", turn)
            st.metric("Total Queries", total_queries)
            
            # Agent status
            agent_status = self.get_conversation_state(session_id)
            if "error" not in agent_status:
                st.write(f"**Status:** {agent_status.get('workflow_status', 'Unknown')}")
                st.write(f"**Current Step:** {agent_status.get('current_step', 'Unknown')}")
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation"):
                if self.clear_conversation(session_id):
                    st.success("Conversation cleared!")
                    st.rerun()
                else:
                    st.error("Failed to clear conversation")
    
    def display_agent_capabilities(self) -> None:
        """Display agent capabilities and system status."""
        
        capabilities = self.get_agent_capabilities()
        
        if "error" in capabilities:
            st.error(f"Failed to get agent capabilities: {capabilities['error']}")
            return
        
        with st.expander("🛠️ Agent Capabilities", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Agents:**")
                for agent in capabilities.get("available_agents", []):
                    st.write(f"• {agent.title()}")
                
                st.write("**Supported Providers:**")
                for provider in capabilities.get("supported_providers", []):
                    st.write(f"• {provider.upper()}")
            
            with col2:
                st.write("**System Status:**")
                db_status = capabilities.get("database_status", "unknown")
                st.write(f"• Database: {db_status}")
                
                tools = capabilities.get("tools_available", [])
                st.write(f"• Tools: {len(tools)} available")
                
                features = capabilities.get("features", {})
                enabled_features = [k for k, v in features.items() if v]
                st.write(f"• Features: {len(enabled_features)} enabled")
    
    def display_cart_notifications(self):
        """Display cart update notifications with enhanced details."""
        if 'cart_update_notification' in st.session_state:
            notification = st.session_state.cart_update_notification
            
            # Only show recent notifications (within last 8 seconds for better UX)
            if time.time() - notification["timestamp"] < 8:
                details = notification.get("details", {})
                
                if notification["type"] == "success":
                    # Enhanced success message with cart details
                    message = f"🛒 {notification['message']}"
                    if details.get("total_items", 0) > 0:
                        message += f" ({details['total_items']} items"
                        if details.get("total_value", 0) > 0:
                            message += f", ${details['total_value']:.2f}"
                        message += ")"
                    
                    st.success(message)
                    
                    # Show additional visual feedback for specific operations
                    if details.get("operation") == "item added":
                        st.balloons()  # Celebratory animation for adding items
                        
                elif notification["type"] == "error":
                    st.error(f"🛒 {notification['message']}")
                    
                    # Show error details if available
                    if details.get("error"):
                        with st.expander("Error Details", expanded=False):
                            st.code(details["error"])
                else:
                    st.info(f"🛒 {notification['message']}")
            else:
                # Clear old notifications
                del st.session_state.cart_update_notification
    
    def _persist_cart_data(self, cart_data: List[Dict[str, Any]], item_count: int, cart_total: Optional[float]):
        """Persist cart data in session state for cross-tab access."""
        try:
            # Store cart data that can be accessed from any tab
            st.session_state.persistent_cart_data = {
                "items": cart_data if cart_data else [],
                "total_items": item_count,
                "total_value": cart_total or 0.0,
                "last_updated": time.time(),
                "session_id": st.session_state.get('langgraph_session_id', 'unknown')
            }
            
            # Update cart history for analytics
            if 'cart_history' not in st.session_state:
                st.session_state.cart_history = []
            
            # Add to cart history (keep last 10 operations)
            st.session_state.cart_history.append({
                "timestamp": time.time(),
                "item_count": item_count,
                "total_value": cart_total or 0.0,
                "operation": "update"
            })
            
            # Keep only recent history
            if len(st.session_state.cart_history) > 10:
                st.session_state.cart_history = st.session_state.cart_history[-10:]
                
        except Exception as e:
            # Log error but don't fail the main operation
            if hasattr(st.session_state, 'cart_manager') and st.session_state.cart_manager is not None:
                try:
                    st.session_state.cart_manager.set_cart_error(f"Failed to persist cart data: {str(e)}")
                except:
                    pass

    def get_persistent_cart_data(self) -> Dict[str, Any]:
        """Get persistent cart data from session state."""
        return st.session_state.get('persistent_cart_data', {
            "items": [],
            "total_items": 0,
            "total_value": 0.0,
            "last_updated": 0,
            "session_id": "unknown"
        })

    def display_conversation_history(self) -> None:
        """Display conversation history in sidebar."""
        
        history = st.session_state.get('langgraph_agent_history', [])
        
        if not history:
            return
        
        with st.sidebar:
            st.subheader("📜 Recent Conversations")
            
            # Show last few conversations
            for i, entry in enumerate(reversed(history[-5:]), 1):
                with st.expander(f"Query {len(history) - i + 1}", expanded=False):
                    st.write(f"**Q:** {entry['query'][:50]}...")
                    st.write(f"**A:** {entry['response'][:100]}...")
                    st.write(f"**Agent:** {entry['agent_workflow']}")
                    st.write(f"**Time:** {entry['processing_time']:.2f}s")
                    
                    if entry['products_found'] > 0:
                        st.write(f"**Products:** {entry['products_found']}")
                    
                    if entry['reviews_found'] > 0:
                        st.write(f"**Reviews:** {entry['reviews_found']}")
                    
                    # Show cart information if available
                    if entry.get('cart_updated'):
                        st.write(f"**Cart:** Updated ({entry.get('cart_item_count', 0)} items)")


def create_langgraph_integration(api_base_url: str = API_BASE_URL) -> StreamlitLangGraphIntegration:
    """Create a new LangGraph integration instance."""
    return StreamlitLangGraphIntegration(api_base_url)