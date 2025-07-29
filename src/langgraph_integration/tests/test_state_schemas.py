"""
Unit tests for state schemas and validation.
"""

import pytest
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

from ..core.state_schemas import (
    create_initial_state,
    update_state_step,
    validate_state,
    validate_conversation_state,
    merge_states,
    get_state_summary
)


class TestStateSchemas:
    """Test state schema functions."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        session_id = "test_session"
        query = "test query"
        
        state = create_initial_state(session_id, query)
        
        assert state["session_id"] == session_id
        assert state["current_query"] == query
        assert state["conversation_turn"] == 1
        assert state["workflow_status"] == "running"
        assert state["current_step"] == "start"
        assert isinstance(state["messages"], list)
        assert len(state["messages"]) == 0
        assert isinstance(state["created_at"], datetime)
        assert isinstance(state["updated_at"], datetime)
    
    def test_create_initial_state_with_kwargs(self):
        """Test initial state creation with additional parameters."""
        session_id = "test_session"
        query = "test query"
        
        state = create_initial_state(
            session_id, 
            query,
            max_products=10,
            max_reviews=5,
            llm_provider="groq",
            llm_model="llama-3.1-70b-versatile"
        )
        
        assert state["max_products"] == 10
        assert state["max_reviews"] == 5
        assert state["llm_provider"] == "groq"
        assert state["llm_model"] == "llama-3.1-70b-versatile"
    
    def test_update_state_step(self):
        """Test state step updates."""
        initial_state = create_initial_state("test_session", "test query")
        
        updated_state = update_state_step(
            initial_state,
            "search_products",
            selected_products=[{"id": "1", "title": "Test Product"}],
            search_results={"total": 1}
        )
        
        assert updated_state["current_step"] == "search_products"
        assert len(updated_state["selected_products"]) == 1
        assert updated_state["search_results"]["total"] == 1
        assert updated_state["updated_at"] > initial_state["updated_at"]
    
    def test_validate_state_valid(self):
        """Test state validation with valid state."""
        state = create_initial_state("test_session", "test query")
        
        assert validate_state(state) is True
    
    def test_validate_state_missing_fields(self):
        """Test state validation with missing required fields."""
        state = create_initial_state("test_session", "test query")
        
        # Remove required field
        del state["session_id"]
        
        assert validate_state(state) is False
    
    def test_validate_state_invalid_types(self):
        """Test state validation with invalid data types."""
        state = create_initial_state("test_session", "test query")
        
        # Invalid type for messages
        state["messages"] = "not a list"
        
        assert validate_state(state) is False
    
    def test_validate_state_invalid_workflow_status(self):
        """Test state validation with invalid workflow status."""
        state = create_initial_state("test_session", "test query")
        
        state["workflow_status"] = "invalid_status"
        
        assert validate_state(state) is False
    
    def test_validate_state_invalid_conversation_turn(self):
        """Test state validation with invalid conversation turn."""
        state = create_initial_state("test_session", "test query")
        
        state["conversation_turn"] = 0  # Should be >= 1
        
        assert validate_state(state) is False
    
    def test_merge_states(self):
        """Test state merging."""
        base_state = create_initial_state("test_session", "test query")
        
        updates = {
            "current_step": "new_step",
            "selected_products": [{"id": "1"}],
            "final_response": "test response"
        }
        
        merged_state = merge_states(base_state, updates)
        
        assert merged_state["current_step"] == "new_step"
        assert len(merged_state["selected_products"]) == 1
        assert merged_state["final_response"] == "test response"
        assert merged_state["updated_at"] > base_state["updated_at"]
    
    def test_get_state_summary(self):
        """Test state summary generation."""
        state = create_initial_state("test_session", "test query")
        state["messages"] = [HumanMessage(content="Hello"), AIMessage(content="Hi")]
        state["search_results"] = {"products": []}
        state["final_response"] = "Test response"
        
        summary = get_state_summary(state)
        
        assert summary["session_id"] == "test_session"
        assert summary["conversation_turn"] == 1
        assert summary["current_step"] == "start"
        assert summary["workflow_status"] == "running"
        assert summary["message_count"] == 2
        assert summary["has_search_results"] is True
        assert summary["has_final_response"] is True
        assert summary["error_state"] is None


if __name__ == "__main__":
    pytest.main([__file__])