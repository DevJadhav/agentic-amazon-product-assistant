"""
Unit tests for intent classification system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ..core.router.intent_classifier import IntentClassifier, IntentResult
from ..core.router.clarification_handler import ClarificationHandler, ClarificationAttempt
from ..core.router.router_node import RouterNode
from ..core.state_schemas import create_initial_state, AgentState


class TestIntentClassifier:
    """Test cases for IntentClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier()
    
    def test_cart_intent_classification(self):
        """Test classification of cart-related intents."""
        test_cases = [
            ("add this laptop to my cart", "cart"),
            ("put the iPhone in my basket", "cart"),
            ("remove the headphones from cart", "cart"),
            ("show me my cart contents", "cart"),
            ("I want to buy 2 tablets", "cart"),
            ("add 3 of these to cart", "cart"),
            ("delete this item from my cart", "cart"),
            ("clear my shopping cart", "cart"),
        ]
        
        for message, expected_intent in test_cases:
            result = self.classifier.classify_intent(message)
            assert result.intent == expected_intent, f"Failed for message: '{message}'"
            assert result.confidence > 0.5, f"Low confidence for cart intent: {result.confidence}"
    
    def test_qa_intent_classification(self):
        """Test classification of QA-related intents."""
        test_cases = [
            ("what are the features of this laptop?", "qa"),
            ("how does this phone compare to others?", "qa"),
            ("tell me about the reviews for this product", "qa"),
            ("which tablet is best for gaming?", "qa"),
            ("what's the price of this camera?", "qa"),
            ("explain the specifications", "qa"),
            ("recommend a good smartphone", "qa"),
            ("what are the differences between these models?", "qa"),
        ]
        
        for message, expected_intent in test_cases:
            result = self.classifier.classify_intent(message)
            assert result.intent == expected_intent, f"Failed for message: '{message}'"
            assert result.confidence > 0.5, f"Low confidence for QA intent: {result.confidence}"
    
    def test_unclear_intent_classification(self):
        """Test classification of unclear intents."""
        test_cases = [
            ("hello", "unclear"),
            ("", "unclear"),
            ("   ", "unclear"),
            ("hmm", "unclear"),
            ("maybe", "unclear"),
            ("I don't know", "unclear"),
        ]
        
        for message, expected_intent in test_cases:
            result = self.classifier.classify_intent(message)
            assert result.intent == expected_intent, f"Failed for message: '{message}'"
            assert result.clarification_needed is True
    
    def test_entity_extraction(self):
        """Test entity extraction from messages."""
        test_cases = [
            ("add 2 laptops to cart", ["2 laptops"]),
            ("I want to buy 3 iPhone 15", ["3 iPhone", "iPhone 15"]),
            ("put one tablet in my cart", ["one tablet"]),
            ("add the MacBook Pro to cart", ["MacBook Pro"]),
            ("remove 5 items from cart", ["5 items"]),
        ]
        
        for message, expected_entities in test_cases:
            result = self.classifier.classify_intent(message)
            # Check that at least some expected entities are found
            found_entities = [entity for entity in expected_entities if any(expected in entity for expected in result.entities)]
            assert len(found_entities) > 0, f"No expected entities found in: {result.entities} for message: '{message}'"
    
    def test_confidence_scoring(self):
        """Test confidence scoring accuracy."""
        # High confidence cases
        high_confidence_cases = [
            "add this laptop to my cart",
            "what are the features of this phone?",
            "remove item from cart",
            "what are the specifications of this tablet?"  # Changed to clearer QA intent
        ]
        
        for message in high_confidence_cases:
            result = self.classifier.classify_intent(message)
            assert result.confidence > 0.7, f"Expected high confidence for: '{message}', got {result.confidence}"
        
        # Low confidence cases
        low_confidence_cases = [
            "maybe",
            "I'm not sure",
            "hmm",
            "hello"
        ]
        
        for message in low_confidence_cases:
            result = self.classifier.classify_intent(message)
            assert result.confidence < 0.5, f"Expected low confidence for: '{message}', got {result.confidence}"
    
    def test_context_influence(self):
        """Test how context influences classification."""
        message = "add it"
        
        # Without context - should be unclear
        result_no_context = self.classifier.classify_intent(message)
        assert result_no_context.intent == "unclear"
        
        # With cart context - should lean towards cart
        cart_context = {"recent_cart_activity": True}
        result_cart_context = self.classifier.classify_intent(message, cart_context)
        # Should have higher cart score due to context
        assert result_cart_context.metadata["cart_score"] > result_no_context.metadata["cart_score"]
    
    def test_ambiguous_messages(self):
        """Test handling of ambiguous messages."""
        ambiguous_cases = [
            "laptop",
            "this one",
            "yes",
            "no",
            "okay"
        ]
        
        for message in ambiguous_cases:
            result = self.classifier.classify_intent(message)
            # Should either be unclear or have low confidence
            assert result.intent == "unclear" or result.confidence < 0.7, f"Message '{message}' should be ambiguous"
    
    def test_mixed_intent_messages(self):
        """Test messages with mixed intent signals."""
        mixed_cases = [
            ("I want to know about this laptop and add it to cart", "cart"),  # Action wins
            ("what's the price? I might buy it", "qa"),  # Question wins
            ("compare these phones and add the best one", "qa"),  # QA wins due to "compare"
        ]
        
        for message, expected_intent in mixed_cases:
            result = self.classifier.classify_intent(message)
            assert result.intent == expected_intent, f"Failed for mixed message: '{message}', got {result.intent}"


class TestClarificationHandler:
    """Test cases for ClarificationHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ClarificationHandler({"max_clarification_attempts": 3})
        self.session_id = "test_session_123"
    
    def test_needs_clarification_basic(self):
        """Test basic clarification need detection."""
        # Create unclear intent result
        unclear_result = IntentResult(
            intent="unclear",
            confidence=0.3,
            entities=[],
            clarification_needed=True,
            suggested_questions=["What would you like to do?"],
            reasoning="Low confidence",
            metadata={}
        )
        
        # Should need clarification
        assert self.handler.needs_clarification(unclear_result, self.session_id) is True
        
        # Clear intent should not need clarification
        clear_result = IntentResult(
            intent="cart",
            confidence=0.8,
            entities=["laptop"],
            clarification_needed=False,
            suggested_questions=[],
            reasoning="High confidence",
            metadata={}
        )
        
        assert self.handler.needs_clarification(clear_result, self.session_id) is False
    
    def test_max_clarification_attempts(self):
        """Test maximum clarification attempts limit."""
        unclear_result = IntentResult(
            intent="unclear",
            confidence=0.3,
            entities=[],
            clarification_needed=True,
            suggested_questions=["What would you like to do?"],
            reasoning="Low confidence",
            metadata={}
        )
        
        # Create 3 clarification attempts
        for i in range(3):
            self.handler.create_clarification_request(unclear_result, self.session_id, f"message {i}")
        
        # 4th attempt should be denied
        assert self.handler.needs_clarification(unclear_result, self.session_id) is False
    
    def test_create_clarification_request(self):
        """Test clarification request creation."""
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.4,
            entities=["laptop"],
            clarification_needed=True,
            suggested_questions=["Are you looking for information or want to add to cart?"],
            reasoning="Ambiguous intent",
            metadata={}
        )
        
        request = self.handler.create_clarification_request(
            intent_result, self.session_id, "laptop"
        )
        
        assert request["type"] == "clarification_request"
        assert "laptop" in request["message"] or len(request["questions"]) > 0
        assert request["session_id"] == self.session_id
        assert request["attempt_number"] == 1
        assert request["original_message"] == "laptop"
    
    def test_process_clarification_response(self):
        """Test processing of clarification responses."""
        # First create a clarification request
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.4,
            entities=["laptop"],
            clarification_needed=True,
            suggested_questions=["Are you looking for information or want to add to cart?"],
            reasoning="Ambiguous intent",
            metadata={}
        )
        
        self.handler.create_clarification_request(intent_result, self.session_id, "laptop")
        
        # Test cart clarification
        cart_response = self.handler.process_clarification_response("I want to add it to cart", self.session_id)
        assert cart_response is not None
        assert cart_response.intent == "cart"
        assert cart_response.confidence > 0.5
        
        # Create another clarification for QA test
        self.handler.create_clarification_request(intent_result, self.session_id, "phone")
        
        # Test QA clarification
        qa_response = self.handler.process_clarification_response("I want information about it", self.session_id)
        assert qa_response is not None
        assert qa_response.intent == "qa"
        assert qa_response.confidence > 0.5
    
    def test_fallback_response(self):
        """Test fallback response creation."""
        # Create max attempts
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.3,
            entities=[],
            clarification_needed=True,
            suggested_questions=["What would you like to do?"],
            reasoning="Low confidence",
            metadata={}
        )
        
        for i in range(3):
            self.handler.create_clarification_request(intent_result, self.session_id, f"message {i}")
        
        fallback = self.handler.create_fallback_response(self.session_id)
        
        assert fallback["type"] == "fallback_response"
        assert fallback["default_intent"] == "qa"
        assert fallback["max_attempts_reached"] is True
        assert fallback["attempt_count"] == 3
    
    def test_clarification_history(self):
        """Test clarification history tracking."""
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.4,
            entities=["laptop"],
            clarification_needed=True,
            suggested_questions=["What would you like to do?"],
            reasoning="Ambiguous intent",
            metadata={}
        )
        
        # Create a few clarification requests
        self.handler.create_clarification_request(intent_result, self.session_id, "message 1")
        self.handler.create_clarification_request(intent_result, self.session_id, "message 2")
        
        history = self.handler.get_clarification_history(self.session_id)
        
        assert len(history) == 2
        assert history[0]["original_message"] == "message 1"
        assert history[1]["original_message"] == "message 2"
        assert all(not attempt["resolved"] for attempt in history)
    
    def test_session_cleanup(self):
        """Test session history cleanup."""
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.4,
            entities=[],
            clarification_needed=True,
            suggested_questions=["What would you like to do?"],
            reasoning="Ambiguous intent",
            metadata={}
        )
        
        self.handler.create_clarification_request(intent_result, self.session_id, "test message")
        
        # Verify history exists
        history = self.handler.get_clarification_history(self.session_id)
        assert len(history) == 1
        
        # Clear session
        self.handler.clear_session_history(self.session_id)
        
        # Verify history is cleared
        history = self.handler.get_clarification_history(self.session_id)
        assert len(history) == 0


class TestRouterNode:
    """Test cases for RouterNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = RouterNode()
        self.session_id = "test_session_456"
    
    @pytest.mark.asyncio
    async def test_route_clear_cart_intent(self):
        """Test routing of clear cart intent."""
        state = create_initial_state(self.session_id, "add laptop to cart")
        
        result_state = await self.router.route_message(state)
        
        assert result_state["routing_decision"] == "cart"
        assert result_state["target_agent"] == "shopping_cart_agent"
        assert result_state["user_intent"] == "cart"
        assert result_state["intent_confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_route_clear_qa_intent(self):
        """Test routing of clear QA intent."""
        state = create_initial_state(self.session_id, "what are the features of this laptop?")
        
        result_state = await self.router.route_message(state)
        
        assert result_state["routing_decision"] == "qa"
        assert result_state["target_agent"] == "qa_agent"
        assert result_state["user_intent"] == "qa"
        assert result_state["intent_confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_route_unclear_intent_clarification(self):
        """Test routing of unclear intent to clarification."""
        state = create_initial_state(self.session_id, "maybe")
        
        result_state = await self.router.route_message(state)
        
        assert result_state["routing_decision"] == "clarification"
        assert result_state["workflow_status"] == "completed"
        assert result_state["final_response"] is not None
        assert len(result_state["suggested_questions"]) > 0
    
    @pytest.mark.asyncio
    async def test_context_extraction(self):
        """Test context extraction from state."""
        state = create_initial_state(self.session_id, "add it")
        
        # Add some tool calls to simulate context
        state["tool_calls"] = [
            {"tool_name": "add_to_cart", "tool_input": {"product": "laptop"}},
            {"tool_name": "vector_search", "tool_input": {"query": "phones"}}
        ]
        
        # Extract context (this is tested indirectly through routing)
        result_state = await self.router.route_message(state)
        
        # Should have routing metadata
        assert "routing_metadata" in result_state
        assert "classification_timestamp" in result_state["routing_metadata"]
    
    @pytest.mark.asyncio
    async def test_routing_statistics(self):
        """Test routing statistics tracking."""
        initial_stats = self.router.get_routing_stats()
        assert initial_stats["total_routes"] == 0
        
        # Route a cart message
        cart_state = create_initial_state(self.session_id, "add laptop to cart")
        await self.router.route_message(cart_state)
        
        # Route a QA message
        qa_state = create_initial_state(self.session_id + "_2", "what is this laptop?")
        await self.router.route_message(qa_state)
        
        # Route an unclear message
        unclear_state = create_initial_state(self.session_id + "_3", "maybe")
        await self.router.route_message(unclear_state)
        
        stats = self.router.get_routing_stats()
        assert stats["total_routes"] == 3
        assert stats["cart_routes"] == 1
        assert stats["qa_routes"] == 1
        assert stats["clarifications"] == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test router error handling."""
        # Create a state that might cause issues
        state = create_initial_state(self.session_id, "test message")
        
        # Mock the intent classifier to raise an exception
        with patch.object(self.router.intent_classifier, 'classify_intent', side_effect=Exception("Test error")):
            result_state = await self.router.route_message(state)
            
            # Should fallback to QA agent
            assert result_state["routing_decision"] == "qa"
            assert result_state["target_agent"] == "qa_agent"
            assert result_state["error_state"] is not None
            assert "Test error" in result_state["error_state"]
    
    @pytest.mark.asyncio
    async def test_process_clarification_response(self):
        """Test processing clarification responses."""
        # First create a clarification
        unclear_state = create_initial_state(self.session_id, "maybe")
        await self.router.route_message(unclear_state)
        
        # Now process a clarification response
        clarification_state = create_initial_state(self.session_id, "I want to add something to cart")
        result_state = await self.router.process_clarification_response(clarification_state)
        
        # Should resolve to cart intent
        assert result_state["routing_decision"] == "cart"
        assert result_state["target_agent"] == "shopping_cart_agent"
    
    def test_reset_routing_stats(self):
        """Test resetting routing statistics."""
        # Generate some stats
        self.router._routing_stats["total_routes"] = 10
        self.router._routing_stats["cart_routes"] = 5
        
        # Reset
        self.router.reset_routing_stats()
        
        stats = self.router.get_routing_stats()
        assert stats["total_routes"] == 0
        assert stats["cart_routes"] == 0


class TestIntegration:
    """Integration tests for the complete intent classification system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier()
        self.handler = ClarificationHandler()
        self.router = RouterNode(self.classifier, self.handler)
        self.session_id = "integration_test_session"
    
    @pytest.mark.asyncio
    async def test_complete_cart_workflow(self):
        """Test complete workflow for cart intent."""
        # Clear cart intent
        state = create_initial_state(self.session_id, "add MacBook Pro to my cart")
        result_state = await self.router.route_message(state)
        
        assert result_state["routing_decision"] == "cart"
        assert result_state["user_intent"] == "cart"
        assert "MacBook Pro" in result_state["extracted_entities"] or "MacBook" in str(result_state["extracted_entities"])
        assert result_state["intent_confidence"] > 0.7
    
    @pytest.mark.asyncio
    async def test_complete_qa_workflow(self):
        """Test complete workflow for QA intent."""
        # Clear QA intent
        state = create_initial_state(self.session_id, "what are the best features of the iPhone 15?")
        result_state = await self.router.route_message(state)
        
        assert result_state["routing_decision"] == "qa"
        assert result_state["user_intent"] == "qa"
        assert result_state["intent_confidence"] > 0.7
    
    @pytest.mark.asyncio
    async def test_complete_clarification_workflow(self):
        """Test complete workflow for clarification."""
        # Unclear intent
        state = create_initial_state(self.session_id, "hmm")
        result_state = await self.router.route_message(state)
        
        assert result_state["routing_decision"] == "clarification"
        assert result_state["clarification_needed"] is True
        assert len(result_state["suggested_questions"]) > 0
        assert result_state["workflow_status"] == "completed"
        
        # Follow up with clarification response
        clarification_state = create_initial_state(self.session_id, "I want product information")
        resolved_state = await self.router.process_clarification_response(clarification_state)
        
        assert resolved_state["routing_decision"] == "qa"
        assert resolved_state["user_intent"] == "qa"
    
    @pytest.mark.asyncio
    async def test_fallback_after_max_attempts(self):
        """Test fallback behavior after maximum clarification attempts."""
        # Create multiple unclear intents
        for i in range(3):
            state = create_initial_state(self.session_id, f"unclear message {i}")
            await self.router.route_message(state)
        
        # Next unclear message should trigger fallback
        state = create_initial_state(self.session_id, "still unclear")
        result_state = await self.router.route_message(state)
        
        # Should fallback to QA agent
        assert result_state["routing_decision"] == "qa"
        assert result_state["target_agent"] == "qa_agent"
        
        # Check routing stats
        stats = self.router.get_routing_stats()
        assert stats["fallbacks"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])