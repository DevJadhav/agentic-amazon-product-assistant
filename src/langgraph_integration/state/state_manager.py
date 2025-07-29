"""
State manager for LangGraph agent workflows.
Integrates LangGraph state with PostgreSQL persistence and checkpointing.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.base import CheckpointMetadata, CheckpointTuple

from ..core.state_schemas import AgentState, WorkflowCheckpoint, validate_state
from ..core.utils import sanitize_state_for_storage, restore_state_from_storage
from .postgres_store import PostgreSQLStateStore
from .memory_manager import ConversationMemoryManager

logger = logging.getLogger(__name__)


class LangGraphStateManager:
    """Manages LangGraph state with PostgreSQL persistence."""
    
    def __init__(self, state_store: Optional[PostgreSQLStateStore] = None):
        """Initialize state manager."""
        self.state_store = state_store or PostgreSQLStateStore()
        self.memory_manager = ConversationMemoryManager(self.state_store)
        self.logger = logging.getLogger(__name__)
    
    def save_state(self, session_id: str, state: AgentState) -> bool:
        """Save agent state with validation."""
        
        try:
            # Validate state before saving
            if not validate_state(state):
                self.logger.error(f"Invalid state for session {session_id}")
                return False
            
            # Save to persistent storage
            self.state_store.save_conversation_state(session_id, state)
            
            self.logger.info(f"Saved state for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state for session {session_id}: {e}")
            return False
    
    def load_state(self, session_id: str) -> Optional[AgentState]:
        """Load agent state from storage."""
        
        try:
            state = self.state_store.load_conversation_state(session_id)
            
            if state and validate_state(state):
                self.logger.info(f"Loaded state for session {session_id}")
                return state
            else:
                self.logger.warning(f"No valid state found for session {session_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load state for session {session_id}: {e}")
            return None
    
    def create_checkpoint(self, session_id: str, state: AgentState, node_name: str) -> str:
        """Create a checkpoint for the current state."""
        
        try:
            checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:12]}"
            
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=checkpoint_id,
                session_id=session_id,
                state_data=state,
                node_name=node_name,
                timestamp=datetime.utcnow(),
                metadata={"created_by": "state_manager"}
            )
            
            self.state_store.save_checkpoint(checkpoint)
            
            self.logger.info(f"Created checkpoint {checkpoint_id} for session {session_id}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return ""
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[AgentState]:
        """Restore state from a checkpoint."""
        
        try:
            checkpoint = self.state_store.load_checkpoint(checkpoint_id)
            
            if checkpoint:
                self.logger.info(f"Restored checkpoint {checkpoint_id}")
                return checkpoint["state_data"]
            else:
                self.logger.warning(f"Checkpoint {checkpoint_id} not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return None
    
    def update_state_step(self, session_id: str, step_name: str, **updates) -> bool:
        """Update state for a new workflow step."""
        
        try:
            # Load current state
            state = self.load_state(session_id)
            
            if not state:
                self.logger.error(f"No state found for session {session_id}")
                return False
            
            # Update state
            from ..core.state_schemas import update_state_step
            updated_state = update_state_step(state, step_name, **updates)
            
            # Save updated state
            return self.save_state(session_id, updated_state)
            
        except Exception as e:
            self.logger.error(f"Failed to update state step: {e}")
            return False
    
    def get_conversation_context(self, session_id: str, current_query: str) -> str:
        """Get conversation context for LLM."""
        
        return self.memory_manager.get_conversation_context(session_id, current_query)
    
    def add_message_to_conversation(self, session_id: str, message) -> bool:
        """Add a message to the conversation."""
        
        try:
            self.memory_manager.add_message(session_id, message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add message: {e}")
            return False
    
    def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """Clean up old conversation states."""
        
        return self.state_store.cleanup_old_sessions(max_age_days)
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored states."""
        
        try:
            return self.state_store.db_manager.get_database_stats()
        except Exception as e:
            self.logger.error(f"Failed to get state statistics: {e}")
            return {}


class PostgreSQLCheckpointSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver using PostgreSQL."""
    
    def __init__(self, state_store: Optional[PostgreSQLStateStore] = None):
        """Initialize PostgreSQL checkpoint saver."""
        super().__init__()
        self.state_store = state_store or PostgreSQLStateStore()
        self.logger = logging.getLogger(__name__)
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata
    ) -> None:
        """Save a checkpoint to PostgreSQL."""
        
        try:
            # Extract session information from config
            session_id = config.get("configurable", {}).get("session_id", "default")
            
            # Create checkpoint data
            checkpoint_data = WorkflowCheckpoint(
                checkpoint_id=str(uuid.uuid4()),
                session_id=session_id,
                state_data=checkpoint,  # This will be the actual checkpoint data
                node_name=metadata.get("step", "unknown"),
                timestamp=datetime.utcnow(),
                metadata=dict(metadata)
            )
            
            # Save checkpoint
            self.state_store.save_checkpoint(checkpoint_data)
            
            self.logger.info(f"Saved LangGraph checkpoint for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save LangGraph checkpoint: {e}")
            raise
    
    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get the latest checkpoint tuple for a configuration."""
        
        try:
            session_id = config.get("configurable", {}).get("session_id", "default")
            
            # Load latest state
            state = self.state_store.load_conversation_state(session_id)
            
            if not state:
                return None
            
            # Convert to checkpoint format
            checkpoint = Checkpoint(
                v=1,
                ts=state.get("updated_at", datetime.utcnow()).isoformat(),
                id=str(uuid.uuid4()),
                channel_values=state,
                channel_versions={},
                versions_seen={}
            )
            
            metadata = CheckpointMetadata(
                source="database",
                step=state.get("current_step", "unknown"),
                writes={}
            )
            
            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint tuple: {e}")
            return None
    
    def list(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None,
        before: Optional[str] = None
    ) -> List[CheckpointTuple]:
        """List checkpoints for a configuration."""
        
        try:
            session_id = config.get("configurable", {}).get("session_id", "default")
            
            # Get conversation history
            history = self.state_store.get_conversation_history(session_id, limit or 10)
            
            checkpoints = []
            
            for i, msg_data in enumerate(history):
                checkpoint = Checkpoint(
                    v=1,
                    ts=msg_data.get("created_at", datetime.utcnow()).isoformat(),
                    id=str(uuid.uuid4()),
                    channel_values={"message": msg_data},
                    channel_versions={},
                    versions_seen={}
                )
                
                metadata = CheckpointMetadata(
                    source="database",
                    step=f"turn_{msg_data.get('turn_number', i)}",
                    writes={}
                )
                
                checkpoints.append(CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                    metadata=metadata
                ))
            
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []


def create_state_manager(state_store: Optional[PostgreSQLStateStore] = None) -> LangGraphStateManager:
    """Create a new state manager instance."""
    return LangGraphStateManager(state_store)


def create_checkpoint_saver(state_store: Optional[PostgreSQLStateStore] = None) -> PostgreSQLCheckpointSaver:
    """Create a new checkpoint saver instance."""
    return PostgreSQLCheckpointSaver(state_store)