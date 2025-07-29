"""
Conversation memory manager for LangGraph agent workflows.
Manages conversation context and memory across turns.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from ..core.state_schemas import AgentState
from ..core.utils import (
    format_conversation_history, 
    should_summarize_conversation,
    extract_message_content
)
from .postgres_store import PostgreSQLStateStore

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """Manages conversation context and memory across turns."""
    
    def __init__(self, state_store: Optional[PostgreSQLStateStore] = None):
        """Initialize conversation memory manager."""
        self.state_store = state_store or PostgreSQLStateStore()
        self.logger = logging.getLogger(__name__)
        
        # Memory configuration
        self.max_memory_length = 20  # Maximum messages to keep in active memory
        self.summary_threshold = 15  # When to start summarizing
        self.context_window = 10     # Messages to include in context
    
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Add a message to conversation memory."""
        
        try:
            # Load current state
            state = self.state_store.load_conversation_state(session_id)
            
            if state is None:
                # Create new conversation state
                from ..core.state_schemas import create_initial_state
                state = create_initial_state(session_id, extract_message_content(message))
            
            # Add message to state
            messages = state.get("messages", [])
            messages.append(message)
            
            # Check if we need to summarize
            if should_summarize_conversation(messages, self.summary_threshold):
                messages = self._summarize_conversation(messages)
            
            # Update state
            state["messages"] = messages
            state["updated_at"] = datetime.utcnow()
            
            # Save updated state
            self.state_store.save_conversation_state(session_id, state)
            
            self.logger.info(f"Added message to conversation {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add message to memory: {e}")
            raise
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[BaseMessage]:
        """Get conversation history as BaseMessage objects."""
        
        try:
            state = self.state_store.load_conversation_state(session_id)
            
            if state is None:
                return []
            
            messages = state.get("messages", [])
            
            # Return last 'limit' messages
            return messages[-limit:] if len(messages) > limit else messages
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def get_conversation_context(self, session_id: str, current_query: str) -> str:
        """Get formatted conversation context for LLM."""
        
        try:
            messages = self.get_conversation_history(session_id, self.context_window)
            
            if not messages:
                return f"Current query: {current_query}"
            
            # Format conversation history
            history = format_conversation_history(messages, self.context_window)
            
            return f"Conversation History:\n{history}\n\nCurrent Query: {current_query}"
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return f"Current query: {current_query}"
    
    def extract_relevant_context(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """Extract relevant context from conversation history."""
        
        try:
            messages = self.get_conversation_history(session_id, self.max_memory_length)
            
            context = {
                "session_id": session_id,
                "current_query": current_query,
                "message_count": len(messages),
                "conversation_summary": "",
                "recent_topics": [],
                "user_preferences": {},
                "previous_products": [],
                "query_patterns": []
            }
            
            if not messages:
                return context
            
            # Extract topics and patterns from recent messages
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            
            topics = set()
            products = set()
            
            for message in recent_messages:
                content = extract_message_content(message).lower()
                
                # Extract product mentions
                product_keywords = [
                    "iphone", "samsung", "laptop", "tablet", "headphones",
                    "speaker", "router", "cable", "charger", "keyboard", "mouse"
                ]
                
                for keyword in product_keywords:
                    if keyword in content:
                        products.add(keyword)
                
                # Extract topic keywords
                if any(word in content for word in ["compare", "comparison", "vs"]):
                    topics.add("comparison")
                elif any(word in content for word in ["recommend", "suggest", "best"]):
                    topics.add("recommendation")
                elif any(word in content for word in ["review", "opinion", "feedback"]):
                    topics.add("reviews")
                elif any(word in content for word in ["price", "cost", "budget"]):
                    topics.add("pricing")
            
            context["recent_topics"] = list(topics)
            context["previous_products"] = list(products)
            
            # Generate conversation summary
            if len(messages) > 3:
                context["conversation_summary"] = self._generate_conversation_summary(messages)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to extract relevant context: {e}")
            return {
                "session_id": session_id,
                "current_query": current_query,
                "error": str(e)
            }
    
    def summarize_long_conversations(self, session_id: str) -> str:
        """Summarize long conversations to manage memory."""
        
        try:
            messages = self.get_conversation_history(session_id, self.max_memory_length)
            
            if len(messages) <= self.summary_threshold:
                return ""
            
            return self._generate_conversation_summary(messages)
            
        except Exception as e:
            self.logger.error(f"Failed to summarize conversation: {e}")
            return ""
    
    def clear_conversation_memory(self, session_id: str) -> bool:
        """Clear conversation memory for a session."""
        
        try:
            # Load current state
            state = self.state_store.load_conversation_state(session_id)
            
            if state is None:
                return True
            
            # Clear messages but keep other state
            state["messages"] = []
            state["conversation_turn"] = 1
            state["updated_at"] = datetime.utcnow()
            
            # Save cleared state
            self.state_store.save_conversation_state(session_id, state)
            
            self.logger.info(f"Cleared conversation memory for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear conversation memory: {e}")
            return False
    
    def get_memory_stats(self, session_id: str) -> Dict[str, Any]:
        """Get memory statistics for a conversation."""
        
        try:
            messages = self.get_conversation_history(session_id, self.max_memory_length)
            session_stats = self.state_store.get_session_stats(session_id)
            
            stats = {
                "session_id": session_id,
                "active_messages": len(messages),
                "memory_usage": "normal",
                "needs_summarization": should_summarize_conversation(messages, self.summary_threshold),
                "last_activity": datetime.utcnow().isoformat()
            }
            
            # Add session stats if available
            if session_stats:
                stats.update(session_stats)
            
            # Determine memory usage level
            if len(messages) > self.summary_threshold:
                stats["memory_usage"] = "high"
            elif len(messages) > self.context_window:
                stats["memory_usage"] = "medium"
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"session_id": session_id, "error": str(e)}
    
    def optimize_memory(self, session_id: str) -> Dict[str, Any]:
        """Optimize memory usage for a conversation."""
        
        try:
            state = self.state_store.load_conversation_state(session_id)
            
            if state is None:
                return {"status": "no_conversation"}
            
            messages = state.get("messages", [])
            original_count = len(messages)
            
            if len(messages) <= self.summary_threshold:
                return {"status": "no_optimization_needed", "message_count": original_count}
            
            # Summarize and optimize
            optimized_messages = self._summarize_conversation(messages)
            
            # Update state
            state["messages"] = optimized_messages
            state["updated_at"] = datetime.utcnow()
            
            # Save optimized state
            self.state_store.save_conversation_state(session_id, state)
            
            result = {
                "status": "optimized",
                "original_count": original_count,
                "optimized_count": len(optimized_messages),
                "reduction": original_count - len(optimized_messages)
            }
            
            self.logger.info(f"Optimized memory for session {session_id}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize memory: {e}")
            return {"status": "error", "error": str(e)}
    
    # Private helper methods
    
    def _summarize_conversation(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Summarize conversation to reduce memory usage."""
        
        if len(messages) <= self.summary_threshold:
            return messages
        
        # Keep the first few and last few messages
        keep_start = 2
        keep_end = self.context_window
        
        if len(messages) <= keep_start + keep_end:
            return messages
        
        # Create summary of middle messages
        middle_messages = messages[keep_start:-keep_end]
        summary = self._generate_conversation_summary(middle_messages)
        
        # Create summarized conversation
        summarized = messages[:keep_start]
        summarized.append(SystemMessage(content=f"[Conversation Summary: {summary}]"))
        summarized.extend(messages[-keep_end:])
        
        return summarized
    
    def _generate_conversation_summary(self, messages: List[BaseMessage]) -> str:
        """Generate a summary of conversation messages."""
        
        if not messages:
            return "No conversation history"
        
        # Simple summary generation - can be enhanced with LLM
        topics = set()
        products = set()
        user_queries = []
        
        for message in messages:
            content = extract_message_content(message).lower()
            
            if isinstance(message, HumanMessage):
                user_queries.append(content[:100])  # First 100 chars
            
            # Extract topics and products
            if "compare" in content or "vs" in content:
                topics.add("product comparison")
            elif "recommend" in content or "suggest" in content:
                topics.add("recommendations")
            elif "review" in content:
                topics.add("reviews")
            elif "price" in content or "cost" in content:
                topics.add("pricing")
            
            # Extract product mentions
            product_keywords = [
                "iphone", "samsung", "laptop", "tablet", "headphones",
                "speaker", "router", "cable", "charger"
            ]
            
            for keyword in product_keywords:
                if keyword in content:
                    products.add(keyword)
        
        # Build summary
        summary_parts = []
        
        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(topics)}")
        
        if products:
            summary_parts.append(f"Products mentioned: {', '.join(products)}")
        
        if user_queries:
            summary_parts.append(f"Recent queries: {len(user_queries)} questions asked")
        
        return "; ".join(summary_parts) if summary_parts else "General conversation about electronics products"