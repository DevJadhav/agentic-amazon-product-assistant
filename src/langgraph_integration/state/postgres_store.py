"""
PostgreSQL state store for LangGraph agent workflows.
Handles persistent storage and retrieval of conversation states.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import UUID

from langchain_core.messages import BaseMessage

from ..core.state_schemas import AgentState, WorkflowCheckpoint, validate_state
from ..core.utils import sanitize_state_for_storage, restore_state_from_storage
from .database import DatabaseManager, get_database_manager

logger = logging.getLogger(__name__)


class PostgreSQLStateStore:
    """Persistent state storage using PostgreSQL."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize PostgreSQL state store."""
        self.db_manager = db_manager or get_database_manager()
        self.logger = logging.getLogger(__name__)
    
    def save_conversation_state(self, session_id: str, state: AgentState) -> None:
        """Save conversation state to database."""
        
        try:
            # Validate state before saving
            if not validate_state(state):
                raise ValueError("Invalid agent state provided")
            
            # Ensure conversation exists
            conversation_id = self._ensure_conversation_exists(session_id, state)
            
            # Sanitize state for storage
            sanitized_state = sanitize_state_for_storage(state)
            
            # Save agent state
            self._save_agent_state(conversation_id, sanitized_state)
            
            # Save messages
            self._save_conversation_messages(conversation_id, state)
            
            # Update conversation timestamp
            self._update_conversation_timestamp(session_id)
            
            self.logger.info(f"Saved conversation state for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation state: {e}")
            raise
    
    def load_conversation_state(self, session_id: str) -> Optional[AgentState]:
        """Load conversation state from database."""
        
        try:
            # Get conversation
            conversation = self._get_conversation(session_id)
            if not conversation:
                return None
            
            conversation_id = conversation["id"]
            
            # Get latest agent state
            agent_state = self._get_latest_agent_state(conversation_id)
            if not agent_state:
                return None
            
            # Get conversation messages
            messages = self._get_conversation_messages(conversation_id)
            
            # Restore state from storage format
            restored_state = restore_state_from_storage(agent_state["state_data"])
            
            # Add messages to state
            restored_state["messages"] = messages
            
            self.logger.info(f"Loaded conversation state for session {session_id}")
            
            return restored_state
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation state: {e}")
            return None
    
    def update_conversation_turn(self, session_id: str, turn_data: Dict[str, Any]) -> None:
        """Update conversation with new turn data."""
        
        try:
            # Get conversation
            conversation = self._get_conversation(session_id)
            if not conversation:
                raise ValueError(f"Conversation not found for session {session_id}")
            
            conversation_id = conversation["id"]
            
            # Add new message if provided
            if "message" in turn_data:
                self._add_conversation_message(
                    conversation_id,
                    turn_data["turn_number"],
                    turn_data["message_type"],
                    turn_data["message"],
                    turn_data.get("metadata", {})
                )
            
            # Update conversation timestamp
            self._update_conversation_timestamp(session_id)
            
            self.logger.info(f"Updated conversation turn for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update conversation turn: {e}")
            raise
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Clean up old conversation sessions."""
        
        try:
            deleted_count = self.db_manager.cleanup_old_conversations(max_age_days)
            self.logger.info(f"Cleaned up {deleted_count} old sessions")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation message history."""
        
        try:
            conversation = self._get_conversation(session_id)
            if not conversation:
                return []
            
            conversation_id = conversation["id"]
            
            query = """
            SELECT turn_number, message_type, content, metadata, created_at
            FROM conversation_messages
            WHERE conversation_id = %s
            ORDER BY turn_number DESC, created_at DESC
            LIMIT %s
            """
            
            results = self.db_manager.execute_query(query, (conversation_id, limit))
            
            # Reverse to get chronological order
            return list(reversed(results))
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save workflow checkpoint."""
        
        try:
            # Ensure conversation exists
            conversation_id = self._ensure_conversation_exists(
                checkpoint["session_id"], 
                checkpoint["state_data"]
            )
            
            # Save checkpoint
            query = """
            INSERT INTO agent_states (conversation_id, state_data, checkpoint_id, created_at)
            VALUES (%s, %s, %s, %s)
            """
            
            sanitized_state = sanitize_state_for_storage(checkpoint["state_data"])
            
            self.db_manager.execute_update(
                query,
                (
                    conversation_id,
                    json.dumps(sanitized_state),
                    checkpoint["checkpoint_id"],
                    checkpoint["timestamp"]
                )
            )
            
            self.logger.info(f"Saved checkpoint {checkpoint['checkpoint_id']}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load workflow checkpoint."""
        
        try:
            query = """
            SELECT c.session_id, a.state_data, a.checkpoint_id, a.created_at
            FROM agent_states a
            JOIN conversations c ON a.conversation_id = c.id
            WHERE a.checkpoint_id = %s
            ORDER BY a.created_at DESC
            LIMIT 1
            """
            
            results = self.db_manager.execute_query(query, (checkpoint_id,))
            
            if not results:
                return None
            
            result = results[0]
            
            return WorkflowCheckpoint(
                checkpoint_id=result["checkpoint_id"],
                session_id=result["session_id"],
                state_data=restore_state_from_storage(result["state_data"]),
                node_name="unknown",  # Not stored in current schema
                timestamp=result["created_at"],
                metadata={}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation session."""
        
        try:
            conversation = self._get_conversation(session_id)
            if not conversation:
                return {}
            
            conversation_id = conversation["id"]
            
            query = """
            SELECT 
                COUNT(DISTINCT turn_number) as total_turns,
                COUNT(*) as total_messages,
                MIN(created_at) as first_message,
                MAX(created_at) as last_message
            FROM conversation_messages
            WHERE conversation_id = %s
            """
            
            results = self.db_manager.execute_query(query, (conversation_id,))
            
            if results:
                stats = results[0]
                stats["session_id"] = session_id
                stats["conversation_age"] = (
                    datetime.utcnow() - conversation["created_at"]
                ).total_seconds() / 3600  # hours
                
                return stats
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get session stats: {e}")
            return {}
    
    # Private helper methods
    
    def _ensure_conversation_exists(self, session_id: str, state: AgentState) -> str:
        """Ensure conversation record exists and return conversation ID."""
        
        # Check if conversation exists
        conversation = self._get_conversation(session_id)
        
        if conversation:
            return conversation["id"]
        
        # Create new conversation
        query = """
        INSERT INTO conversations (session_id, created_at, updated_at, metadata)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """
        
        metadata = {
            "llm_provider": state.get("llm_provider", "unknown"),
            "llm_model": state.get("llm_model", "unknown"),
            "initial_query": state.get("current_query", "")
        }
        
        now = datetime.utcnow()
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (session_id, now, now, json.dumps(metadata)))
                result = cursor.fetchone()
                conn.commit()
                
                return str(result["id"])
    
    def _get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation record by session ID."""
        
        query = """
        SELECT id, session_id, created_at, updated_at, user_id, metadata
        FROM conversations
        WHERE session_id = %s
        """
        
        results = self.db_manager.execute_query(query, (session_id,))
        return results[0] if results else None
    
    def _save_agent_state(self, conversation_id: str, state_data: Dict[str, Any]) -> None:
        """Save agent state data."""
        
        query = """
        INSERT INTO agent_states (conversation_id, state_data, created_at)
        VALUES (%s, %s, %s)
        """
        
        self.db_manager.execute_update(
            query,
            (conversation_id, json.dumps(state_data), datetime.utcnow())
        )
    
    def _get_latest_agent_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get latest agent state for conversation."""
        
        query = """
        SELECT state_data, created_at
        FROM agent_states
        WHERE conversation_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        results = self.db_manager.execute_query(query, (conversation_id,))
        return results[0] if results else None
    
    def _save_conversation_messages(self, conversation_id: str, state: AgentState) -> None:
        """Save conversation messages from state."""
        
        messages = state.get("messages", [])
        turn_number = state.get("conversation_turn", 1)
        
        for i, message in enumerate(messages):
            # Skip if message already exists
            if self._message_exists(conversation_id, turn_number, i):
                continue
            
            message_type = type(message).__name__.replace("Message", "").lower()
            content = str(message.content) if hasattr(message, 'content') else str(message)
            
            self._add_conversation_message(
                conversation_id,
                turn_number,
                message_type,
                content,
                {"message_index": i}
            )
    
    def _add_conversation_message(
        self, 
        conversation_id: str, 
        turn_number: int, 
        message_type: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """Add a single conversation message."""
        
        query = """
        INSERT INTO conversation_messages 
        (conversation_id, turn_number, message_type, content, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        self.db_manager.execute_update(
            query,
            (
                conversation_id,
                turn_number,
                message_type,
                content,
                json.dumps(metadata),
                datetime.utcnow()
            )
        )
    
    def _message_exists(self, conversation_id: str, turn_number: int, message_index: int) -> bool:
        """Check if a message already exists."""
        
        query = """
        SELECT 1 FROM conversation_messages
        WHERE conversation_id = %s 
        AND turn_number = %s 
        AND metadata->>'message_index' = %s
        LIMIT 1
        """
        
        results = self.db_manager.execute_query(
            query, 
            (conversation_id, turn_number, str(message_index))
        )
        
        return len(results) > 0
    
    def _get_conversation_messages(self, conversation_id: str) -> List[BaseMessage]:
        """Get conversation messages as BaseMessage objects."""
        
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        query = """
        SELECT message_type, content, metadata
        FROM conversation_messages
        WHERE conversation_id = %s
        ORDER BY turn_number ASC, created_at ASC
        """
        
        results = self.db_manager.execute_query(query, (conversation_id,))
        
        messages = []
        for row in results:
            message_type = row["message_type"]
            content = row["content"]
            
            if message_type == "human":
                messages.append(HumanMessage(content=content))
            elif message_type == "ai":
                messages.append(AIMessage(content=content))
            elif message_type == "system":
                messages.append(SystemMessage(content=content))
        
        return messages
    
    def _update_conversation_timestamp(self, session_id: str) -> None:
        """Update conversation's last updated timestamp."""
        
        query = """
        UPDATE conversations
        SET updated_at = %s
        WHERE session_id = %s
        """
        
        self.db_manager.execute_update(query, (datetime.utcnow(), session_id))