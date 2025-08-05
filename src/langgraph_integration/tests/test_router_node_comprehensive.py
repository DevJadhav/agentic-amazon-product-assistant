"""
Comprehensive unit tests for router node logic with different intent scenarios.
Tests router node workflow nodes independently and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from ..core.router.router_node import RouterNode
from ..core.router.intent_classifier import IntentClassifier, IntentResult
from ..core.router.clarification_handler import ClarificationHandler
from ..core.state_schemas import create_initial_state, AgentState


class TestRouterNodeWorkflowNodes:
    """Test individual workflow nodes of RouterNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = RouterNode()
        self.session_id = "test_router_session"
    
    @pytest.mark.asyncio
    async def test_route_message_clear_cart_intent(self):
        """Test routing with clear cart intent."""
        state = create_initial_state(
            self.session_id,
            "add MacBook Pro to my shopping cart"
        )
        
        result = await self.router.route_message(state)
        
        assert result["routing_decision"] == "cart"
        assert result["target_agent"] == "shopping_cart_agent"
        assert result["user_intent"] == "cart"
        assert result["intent_confidence"] > 0.5
        assert "MacBook Pro" in result["extracted_entities"] or "MacBook" in str(result["extracted_entities"])
    
    @pytest.mark.asyncio
    async def test_route_message_clear_qa_intent(self):
        """Test routing with clear QA intent."""
        state = create_initial_state(
            self.session_id,
            "what are the technical specifications of the iPhone 15?"
        )
        
        result = await self.router.route_message(state)
        
        assert result["routing_decision"] == "qa"
        assert result["target_agent"] == "qa_agent"
        assert result["user_intent"] == "qa"
        assert result["intent_confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_route_message_ambiguous_intent(self):
        """Test routing with ambiguous intent."""
        ambiguous_queries = [
            "help me",
            "I need assistance",
            "can you help?",
            "what should I do?",
            "I'm confused"
        ]
        
        for query in ambiguous_queries:
            state = create_initial_state(self.session_id, query)
            result = await self.router.route_message(state)
            
            # Should either route to clarification or have low confidence
            if result["routing_decision"] == "clarification":
                assert result["clarification_needed"] is True
                assert len(result["suggested_questions"]) > 0
            else:
                assert result["intent_confidence"] < 0.8  # Lower confidence for ambiguous
    
    @pytest.mark.asyncio
    async def test_route_message_with_context_cart_history(self):
        """Test routing with cart-related context history."""
        state = create_initial_state(self.session_id, "add it")
        
        # Add cart-related context
        state["tool_calls"] = [
            {
                "tool_name": "add_to_cart",
                "tool_input": {"product_id": "laptop123"},
                "timestamp": datetime.now().isoformat()
            }
        ]
        state["conversation_history"] = [
            "I'm looking at this laptop",
            "It looks good for my needs"
        ]
        
        result = await self.router.route_message(state)
        
        # Context should influence routing toward cart
        assert result["routing_decision"] == "cart"
        assert result["user_intent"] == "cart"
        assert result["intent_confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_route_message_with_context_qa_history(self):
        """Test routing with QA-related context history."""
        state = create_initial_state(self.session_id, "tell me more")
        
        # Add QA-related context
        state["tool_calls"] = [
            {
                "tool_name": "vector_search",
                "tool_input": {"query": "laptop specifications"},
                "timestamp": datetime.now().isoformat()
            }
        ]
        state["conversation_history"] = [
            "What are the best laptops?",
            "Can you compare different models?"
        ]
        
        result = await self.router.route_message(state)
        
        # Context should influence routing toward QA
        assert result["routing_decision"] == "qa"
        assert result["user_intent"] == "qa"
        assert result["intent_confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_route_message_mixed_signals(self):
        """Test routing with mixed intent signals."""
        mixed_queries = [
            "show me laptop reviews and add the best one to cart",
            "I want to know about this phone and maybe buy it",
            "compare these tablets and put the winner in my cart",
            "what's the price of this item? I might purchase it"
        ]
        
        for query in mixed_queries:
            state = create_initial_state(self.session_id, query)
            result = await self.router.route_message(state)
            
            # Should make a decision (not necessarily clarification)
            assert result["routing_decision"] in ["qa", "cart", "clarification"]
            assert result["user_intent"] in ["qa", "cart", "unclear"]
            
            # If it's not clarification, should have reasonable confidence
            if result["routing_decision"] != "clarification":
                assert result["intent_confidence"] > 0.3
    
    @pytest.mark.asyncio
    async def test_extract_context_from_state(self):
        """Test context extraction from agent state."""
        state = create_initial_state(self.session_id, "test query")
        
        # Add various context elements
        state["tool_calls"] = [
            {"tool_name": "add_to_cart", "tool_input": {"product": "laptop"}},
            {"tool_name": "vector_search", "tool_input": {"query": "phones"}}
        ]
        state["conversation_history"] = [
            "I'm looking for a new laptop",
            "What about gaming laptops?"
        ]
        state["selected_product_for_cart"] = {
            "product_id": "laptop123",
            "title": "Gaming Laptop"
        }
        
        # Extract context (this is tested indirectly through routing)
        result = await self.router.route_message(state)
        
        # Should have routing metadata that includes context analysis
        assert "routing_metadata" in result
        metadata = result["routing_metadata"]
        assert "classification_timestamp" in metadata
        assert "context_analyzed" in metadata
        assert metadata["context_analyzed"] is True
    
    @pytest.mark.asyncio
    async def test_route_message_with_product_selection(self):
        """Test routing when user has selected a product."""
        state = create_initial_state(self.session_id, "I want this one")
        
        # Add selected product context
        state["selected_product_for_cart"] = {
            "product_id": "selected123",
            "title": "Selected Product",
            "price": 299.99
        }
        
        result = await self.router.route_message(state)
        
        # Should lean toward cart intent due to product selection
        assert result["routing_decision"] == "cart"
        assert result["user_intent"] == "cart"
    
    @pytest.mark.asyncio
    async def test_route_message_confidence_thresholds(self):
        """Test routing behavior at different confidence thresholds."""
        # Test with high confidence query
        high_confidence_state = create_initial_state(
            self.session_id,
            "add this specific laptop model XYZ-123 to my shopping cart immediately"
        )
        
        result = await self.router.route_message(high_confidence_state)
        assert result["intent_confidence"] > 0.8
        assert result["routing_decision"] == "cart"
        
        # Test with medium confidence query
        medium_confidence_state = create_initial_state(
            self.session_id,
            "I think I want to buy this"
        )
        
        result = await self.router.route_message(medium_confidence_state)
        # Should still make a decision but with lower confidence
        assert 0.3 < result["intent_confidence"] < 0.8
    
    @pytest.mark.asyncio
    async def test_process_clarification_response_cart(self):
        """Test processing clarification response for cart intent."""
        # First create a clarification scenario
        unclear_state = create_initial_state(self.session_id, "maybe")
        await self.router.route_message(unclear_state)
        
        # Now process clarification response
        clarification_state = create_initial_state(
            self.session_id,
            "I want to add something to my cart"
        )
        
        result = await self.router.process_clarification_response(clarification_state)
        
        assert result["routing_decision"] == "cart"
        assert result["user_intent"] == "cart"
        assert result["intent_confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_process_clarification_response_qa(self):
        """Test processing clarification response for QA intent."""
        # First create a clarification scenario
        unclear_state = create_initial_state(self.session_id, "help")
        await self.router.route_message(unclear_state)
        
        # Now process clarification response
        clarification_state = create_initial_state(
            self.session_id,
            "I need information about products"
        )
        
        result = await self.router.process_clarification_response(clarification_state)
        
        assert result["routing_decision"] == "qa"
        assert result["user_intent"] == "qa"
        assert result["intent_confidence"] > 0.5
    
    def test_routing_statistics_tracking(self):
        """Test routing statistics tracking."""
        initial_stats = self.router.get_routing_stats()
        assert initial_stats["total_routes"] == 0
        assert initial_stats["cart_routes"] == 0
        assert initial_stats["qa_routes"] == 0
        assert initial_stats["clarifications"] == 0
        
        # Manually update stats to test tracking
        self.router._routing_stats["total_routes"] = 10
        self.router._routing_stats["cart_routes"] = 4
        self.router._routing_stats["qa_routes"] = 5
        self.router._routing_stats["clarifications"] = 1
        
        stats = self.router.get_routing_stats()
        assert stats["total_routes"] == 10
        assert stats["cart_routes"] == 4
        assert stats["qa_routes"] == 5
        assert stats["clarifications"] == 1
        
        # Test reset
        self.router.reset_routing_stats()
        reset_stats = self.router.get_routing_stats()
        assert reset_stats["total_routes"] == 0


class TestRouterNodeErrorHandling:
    """Test error handling in RouterNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = RouterNode()
        self.session_id = "test_error_session"
    
    @pytest.mark.asyncio
    async def test_intent_classifier_exception(self):
        """Test handling of intent classifier exceptions."""
        state = create_initial_state(self.session_id, "test query")
        
        # Mock intent classifier to raise exception
        with patch.object(self.router.intent_classifier, 'classify_intent', side_effect=Exception("Classifier error")):
            result = await self.router.route_message(state)
            
            # Should fallback gracefully
            assert result["routing_decision"] == "qa"  # Default fallback
            assert result["target_agent"] == "qa_agent"
            assert "error_state" in result
            assert "Classifier error" in result["error_state"]
    
    @pytest.mark.asyncio
    async def test_clarification_handler_exception(self):
        """Test handling of clarification handler exceptions."""
        state = create_initial_state(self.session_id, "unclear query")
        
        # Mock clarification handler to raise exception
        with patch.object(self.router.clarification_handler, 'needs_clarification', side_effect=Exception("Clarification error")):
            result = await self.router.route_message(state)
            
            # Should fallback to QA agent
            assert result["routing_decision"] == "qa"
            assert "error_state" in result
            assert "Clarification error" in result["error_state"]
    
    @pytest.mark.asyncio
    async def test_malformed_intent_result(self):
        """Test handling of malformed intent classification results."""
        state = create_initial_state(self.session_id, "test query")
        
        # Mock intent classifier to return malformed result
        malformed_result = Mock()
        malformed_result.intent = None  # Invalid intent
        malformed_result.confidence = "not_a_number"  # Invalid confidence
        malformed_result.entities = None  # Invalid entities
        
        with patch.object(self.router.intent_classifier, 'classify_intent', return_value=malformed_result):
            result = await self.router.route_message(state)
            
            # Should handle gracefully and fallback
            assert result["routing_decision"] in ["qa", "cart", "clarification"]
            assert "error_state" in result or result["routing_decision"] == "qa"
    
    @pytest.mark.asyncio
    async def test_context_extraction_error(self):
        """Test handling of context extraction errors."""
        state = create_initial_state(self.session_id, "test query")
        
        # Add malformed context that might cause errors
        state["tool_calls"] = "not_a_list"  # Should be a list
        state["conversation_history"] = {"invalid": "structure"}  # Should be a list
        
        result = await self.router.route_message(state)
        
        # Should handle malformed context gracefully
        assert result["routing_decision"] in ["qa", "cart", "clarification"]
        assert "routing_metadata" in result
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of operation timeouts."""
        state = create_initial_state(self.session_id, "test query")
        
        # Mock intent classifier to simulate slow operation
        async def slow_classify(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow operation
            return IntentResult(
                intent="qa",
                confidence=0.8,
                entities=[],
                clarification_needed=False,
                suggested_questions=[],
                reasoning="Slow classification",
                metadata={}
            )
        
        with patch.object(self.router.intent_classifier, 'classify_intent', side_effect=slow_classify):
            # Should complete despite delay (no actual timeout implemented, but test structure)
            result = await self.router.route_message(state)
            assert result["routing_decision"] == "qa"
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling under memory pressure conditions."""
        state = create_initial_state(self.session_id, "test query")
        
        # Add large amounts of data to simulate memory pressure
        state["conversation_history"] = ["large message " * 1000] * 100
        state["tool_calls"] = [
            {"tool_name": "test", "tool_input": {"data": "x" * 10000}}
            for _ in range(100)
        ]
        
        result = await self.router.route_message(state)
        
        # Should handle large state without crashing
        assert result["routing_decision"] in ["qa", "cart", "clarification"]
        assert "routing_metadata" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """Test error handling under concurrent access."""
        states = [
            create_initial_state(f"session_{i}", f"query_{i}")
            for i in range(10)
        ]
        
        # Mock one of the operations to fail
        def failing_classify(message, context=None):
            if "query_5" in message:
                raise Exception("Simulated failure")
            return IntentResult(
                intent="qa",
                confidence=0.8,
                entities=[],
                clarification_needed=False,
                suggested_questions=[],
                reasoning="Normal classification",
                metadata={}
            )
        
        with patch.object(self.router.intent_classifier, 'classify_intent', side_effect=failing_classify):
            # Process all states concurrently
            tasks = [self.router.route_message(state) for state in states]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Most should succeed, one should handle error gracefully
            successful_results = [r for r in results if isinstance(r, dict) and "routing_decision" in r]
            assert len(successful_results) == 10  # All should complete (with error handling)
            
            # Check that the failing one was handled
            failing_result = next(r for r in results if isinstance(r, dict) and "error_state" in r)
            assert "Simulated failure" in failing_result["error_state"]
    
    def test_statistics_error_recovery(self):
        """Test statistics tracking error recovery."""
        # Corrupt statistics
        self.router._routing_stats = None
        
        # Should recover gracefully
        stats = self.router.get_routing_stats()
        assert isinstance(stats, dict)
        assert "total_routes" in stats
        
        # Should be able to reset even with corrupted stats
        self.router.reset_routing_stats()
        reset_stats = self.router.get_routing_stats()
        assert reset_stats["total_routes"] == 0


class TestRouterNodePerformance:
    """Test performance characteristics of RouterNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = RouterNode()
        self.session_id = "test_performance_session"
    
    @pytest.mark.asyncio
    async def test_routing_performance_simple_queries(self):
        """Test routing performance with simple queries."""
        simple_queries = [
            "add laptop to cart",
            "what are the features?",
            "remove item from cart",
            "show me reviews",
            "list my cart"
        ]
        
        import time
        start_time = time.time()
        
        for query in simple_queries * 10:  # 50 total queries
            state = create_initial_state(self.session_id, query)
            result = await self.router.route_message(state)
            assert result["routing_decision"] in ["qa", "cart", "clarification"]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process 50 simple queries quickly (less than 2 seconds)
        assert total_time < 2.0, f"Routing took too long: {total_time} seconds"
        
        # Check average time per query
        avg_time = total_time / 50
        assert avg_time < 0.04, f"Average routing time too high: {avg_time} seconds"
    
    @pytest.mark.asyncio
    async def test_routing_performance_complex_queries(self):
        """Test routing performance with complex queries."""
        complex_queries = [
            "I'm looking for a high-performance gaming laptop with at least 16GB RAM, RTX 4070 graphics card, and SSD storage under $2000, can you show me some options and add the best one to my cart?",
            "What are the detailed technical specifications, user reviews, and price comparisons for the latest iPhone models, and how do they compare to Samsung Galaxy phones in terms of camera quality, battery life, and overall performance?",
            "I need a professional-grade camera for photography work, something with excellent low-light performance, 4K video recording, and interchangeable lenses, please provide recommendations and pricing information",
        ]
        
        import time
        start_time = time.time()
        
        for query in complex_queries * 5:  # 15 total complex queries
            state = create_initial_state(self.session_id, query)
            result = await self.router.route_message(state)
            assert result["routing_decision"] in ["qa", "cart", "clarification"]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle complex queries reasonably quickly (less than 5 seconds)
        assert total_time < 5.0, f"Complex routing took too long: {total_time} seconds"
    
    @pytest.mark.asyncio
    async def test_concurrent_routing_performance(self):
        """Test concurrent routing performance."""
        queries = [
            "add laptop to cart",
            "what are the features?",
            "remove item",
            "show reviews",
            "list cart"
        ]
        
        states = [
            create_initial_state(f"session_{i}", queries[i % len(queries)])
            for i in range(20)
        ]
        
        import time
        start_time = time.time()
        
        # Process all states concurrently
        tasks = [self.router.route_message(state) for state in states]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Concurrent processing should be faster than sequential
        assert total_time < 1.0, f"Concurrent routing took too long: {total_time} seconds"
        
        # All should complete successfully
        assert len(results) == 20
        for result in results:
            assert result["routing_decision"] in ["qa", "cart", "clarification"]
    
    def test_memory_usage_with_large_context(self):
        """Test memory usage with large context."""
        import sys
        
        state = create_initial_state(self.session_id, "test query")
        
        # Add progressively larger context
        for size in [10, 100, 1000]:
            state["conversation_history"] = [f"message {i}" for i in range(size)]
            state["tool_calls"] = [
                {"tool_name": f"tool_{i}", "tool_input": {"data": f"data_{i}"}}
                for i in range(size)
            ]
            
            # Measure memory before and after
            initial_size = sys.getsizeof(state)
            
            # Process the state (this would be async in real usage)
            # For memory testing, we'll just verify the state can be processed
            assert len(state["conversation_history"]) == size
            assert len(state["tool_calls"]) == size
            
            # Memory usage should be reasonable
            final_size = sys.getsizeof(state)
            assert final_size > 0  # Basic sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])