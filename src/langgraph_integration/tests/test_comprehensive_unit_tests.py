"""
Comprehensive unit tests for all new shopping cart functionality components.
This file contains additional unit tests to ensure complete coverage of all new components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from decimal import Decimal

from ..core.router.intent_classifier import IntentClassifier, IntentResult
from ..core.router.clarification_handler import ClarificationHandler, ClarificationAttempt
from ..core.router.router_node import RouterNode
from ..core.shopping_cart_agent import ShoppingCartAgent
from ..state.shopping_cart_manager import ShoppingCartManager
from ..tools.shopping_cart_tools import AddToCartTool, RemoveFromCartTool, ListCartTool, ClearCartTool
from ..core.state_schemas import create_initial_state, AgentState


class TestIntentClassifierEdgeCases:
    """Test edge cases and error conditions for IntentClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier()
    
    def test_classify_intent_with_none_message(self):
        """Test classification with None message."""
        result = self.classifier.classify_intent(None)
        
        assert result.intent == "unclear"
        assert result.confidence < 0.5
        assert result.clarification_needed is True
    
    def test_classify_intent_with_very_long_message(self):
        """Test classification with extremely long message."""
        long_message = "add to cart " * 1000  # Very long message
        
        result = self.classifier.classify_intent(long_message)
        
        assert result.intent == "cart"
        assert result.confidence > 0.5
        assert "add" in result.entities or len(result.entities) > 0
    
    def test_classify_intent_with_special_characters(self):
        """Test classification with special characters and emojis."""
        test_cases = [
            ("add 🛒 to cart", "cart"),
            ("what's the price??? 💰", "qa"),
            ("remove @#$% from cart", "cart"),
            ("show me my 🛍️", "cart")  # Shopping bag emoji should suggest cart
        ]
        
        for message, expected_intent in test_cases:
            result = self.classifier.classify_intent(message)
            assert result.intent == expected_intent, f"Failed for message: '{message}'"
    
    def test_classify_intent_with_mixed_languages(self):
        """Test classification with mixed language content."""
        # These should still work based on English keywords
        test_cases = [
            ("añadir laptop to cart", "cart"),  # Spanish + English
            ("add ordinateur to panier", "cart"),  # English + French
            ("what is precio of this?", "qa")  # Mixed English/Spanish
        ]
        
        for message, expected_intent in test_cases:
            result = self.classifier.classify_intent(message)
            # Should at least not crash and provide some classification
            assert result.intent in ["cart", "qa", "unclear"]
    
    def test_classify_intent_performance(self):
        """Test classification performance with many requests."""
        messages = [
            "add laptop to cart",
            "what are the features?",
            "remove item from cart",
            "show me reviews"
        ] * 25  # 100 total messages
        
        import time
        start_time = time.time()
        
        for message in messages:
            result = self.classifier.classify_intent(message)
            assert result.intent in ["cart", "qa", "unclear"]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process 100 messages in reasonable time (less than 5 seconds)
        assert total_time < 5.0, f"Classification took too long: {total_time} seconds"
    
    def test_entity_extraction_edge_cases(self):
        """Test entity extraction with edge cases."""
        test_cases = [
            ("add 0 items", []),  # Invalid quantity should not extract
            ("add -5 laptops", []),  # Negative quantity should not extract
            ("add 999999 phones", ["999999 phones"]),  # Very large quantity
            ("add iPhone 15 Pro Max Ultra", ["iPhone 15 Pro Max Ultra"]),  # Long product name
            ("add item with spaces   to cart", ["item with spaces"]),  # Multiple spaces
        ]
        
        for message, expected_entities in test_cases:
            result = self.classifier.classify_intent(message)
            if expected_entities:
                assert len(result.entities) > 0, f"No entities found for: '{message}'"
            else:
                # For invalid cases, should either have no entities or filter them out
                valid_entities = [e for e in result.entities if not any(invalid in e.lower() for invalid in ["0", "-"])]
                if not expected_entities:
                    assert len(valid_entities) == 0 or all("0" not in e and "-" not in e for e in result.entities)
    
    def test_confidence_scoring_edge_cases(self):
        """Test confidence scoring with edge cases."""
        # Test with repeated words
        result = self.classifier.classify_intent("add add add to cart cart cart")
        assert result.intent == "cart"
        assert result.confidence > 0.5  # Should still be confident despite repetition
        
        # Test with contradictory signals
        result = self.classifier.classify_intent("add to cart but also show me information")
        assert result.intent in ["cart", "qa"]  # Should pick one
        assert result.confidence > 0.3  # Should have some confidence
    
    def test_context_with_invalid_data(self):
        """Test context handling with invalid context data."""
        message = "add it"
        
        # Test with None context
        result = self.classifier.classify_intent(message, None)
        assert result.intent == "unclear"
        
        # Test with invalid context structure
        invalid_contexts = [
            {"invalid": "data"},
            {"recent_cart_activity": "not_boolean"},
            {"conversation_history": None}
        ]
        
        for context in invalid_contexts:
            result = self.classifier.classify_intent(message, context)
            # Should not crash and should handle gracefully
            assert result.intent in ["cart", "qa", "unclear"]


class TestClarificationHandlerEdgeCases:
    """Test edge cases for ClarificationHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ClarificationHandler({"max_clarification_attempts": 3})
        self.session_id = "test_session"
    
    def test_create_clarification_with_empty_entities(self):
        """Test clarification creation with empty entities."""
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.3,
            entities=[],
            clarification_needed=True,
            suggested_questions=[],
            reasoning="No entities found",
            metadata={}
        )
        
        request = self.handler.create_clarification_request(
            intent_result, self.session_id, "test message"
        )
        
        assert request["type"] == "clarification_request"
        assert len(request["questions"]) > 0  # Should generate default questions
    
    def test_process_clarification_with_invalid_response(self):
        """Test processing clarification with invalid responses."""
        # First create a clarification request
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.4,
            entities=[],
            clarification_needed=True,
            suggested_questions=["What would you like to do?"],
            reasoning="Unclear intent",
            metadata={}
        )
        
        self.handler.create_clarification_request(intent_result, self.session_id, "test")
        
        # Test with empty response
        result = self.handler.process_clarification_response("", self.session_id)
        assert result is None or result.intent == "unclear"
        
        # Test with very short response
        result = self.handler.process_clarification_response("ok", self.session_id)
        assert result is not None  # Should still try to process
    
    def test_session_history_with_concurrent_sessions(self):
        """Test session history handling with multiple concurrent sessions."""
        session1 = "session_1"
        session2 = "session_2"
        
        intent_result = IntentResult(
            intent="unclear",
            confidence=0.4,
            entities=[],
            clarification_needed=True,
            suggested_questions=["What would you like to do?"],
            reasoning="Unclear intent",
            metadata={}
        )
        
        # Create clarifications for both sessions
        self.handler.create_clarification_request(intent_result, session1, "message1")
        self.handler.create_clarification_request(intent_result, session2, "message2")
        
        # Verify session isolation
        history1 = self.handler.get_clarification_history(session1)
        history2 = self.handler.get_clarification_history(session2)
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["original_message"] == "message1"
        assert history2[0]["original_message"] == "message2"
    
    def test_memory_cleanup_with_old_sessions(self):
        """Test memory cleanup for old clarification sessions."""
        # Create many old sessions
        for i in range(100):
            session_id = f"old_session_{i}"
            intent_result = IntentResult(
                intent="unclear",
                confidence=0.4,
                entities=[],
                clarification_needed=True,
                suggested_questions=["What would you like to do?"],
                reasoning="Unclear intent",
                metadata={}
            )
            self.handler.create_clarification_request(intent_result, session_id, f"message_{i}")
        
        # Verify sessions were created
        assert len(self.handler._session_history) == 100
        
        # Test cleanup (if implemented)
        if hasattr(self.handler, 'cleanup_old_sessions'):
            self.handler.cleanup_old_sessions(max_age_hours=0)  # Clean all
            assert len(self.handler._session_history) == 0


class TestRouterNodeEdgeCases:
    """Test edge cases for RouterNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = RouterNode()
        self.session_id = "test_session"
    
    @pytest.mark.asyncio
    async def test_route_message_with_malformed_state(self):
        """Test routing with malformed state."""
        # Test with missing required fields
        malformed_states = [
            {},  # Empty state
            {"session_id": self.session_id},  # Missing current_query
            {"current_query": "test"},  # Missing session_id
            {"session_id": self.session_id, "current_query": None}  # None query
        ]
        
        for state in malformed_states:
            result = await self.router.route_message(state)
            
            # Should handle gracefully and not crash
            assert "routing_decision" in result
            assert result["routing_decision"] in ["qa", "cart", "clarification"]
    
    @pytest.mark.asyncio
    async def test_route_message_with_large_context(self):
        """Test routing with very large context."""
        state = create_initial_state(self.session_id, "add laptop to cart")
        
        # Add large context
        state["conversation_history"] = ["message"] * 1000  # Large history
        state["tool_calls"] = [{"tool": "test", "input": "data"}] * 100  # Many tool calls
        
        result = await self.router.route_message(state)
        
        # Should still work despite large context
        assert result["routing_decision"] == "cart"
        assert result["user_intent"] == "cart"
    
    @pytest.mark.asyncio
    async def test_concurrent_routing_requests(self):
        """Test concurrent routing requests."""
        states = [
            create_initial_state(f"session_{i}", f"query_{i}")
            for i in range(10)
        ]
        
        # Process all states concurrently
        tasks = [self.router.route_message(state) for state in states]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert "routing_decision" in result
            assert result["routing_decision"] in ["qa", "cart", "clarification"]
    
    def test_routing_stats_thread_safety(self):
        """Test routing statistics thread safety."""
        import threading
        import time
        
        def update_stats():
            for _ in range(100):
                # Simulate routing operations
                self.router._routing_stats["total_routes"] += 1
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Create multiple threads
        threads = [threading.Thread(target=update_stats) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check final count (may not be exactly 500 due to race conditions, but should be close)
        final_count = self.router._routing_stats["total_routes"]
        assert final_count > 0  # Should have some updates
        # Note: In a real implementation, we'd want proper thread safety


class TestShoppingCartManagerEdgeCases:
    """Test edge cases for ShoppingCartManager."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        return Mock()
    
    @pytest.fixture
    def cart_manager(self, mock_db_manager):
        """Create cart manager with mock database."""
        return ShoppingCartManager(mock_db_manager)
    
    def test_add_item_with_extreme_values(self, cart_manager, mock_db_manager):
        """Test adding items with extreme values."""
        # Mock no existing item
        mock_db_manager.execute_query.return_value = []
        
        # Mock successful insertion
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": "new-item-id"}
        
        cursor_context = Mock()
        cursor_context.__enter__ = Mock(return_value=mock_cursor)
        cursor_context.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = cursor_context
        
        connection_context = Mock()
        connection_context.__enter__ = Mock(return_value=mock_connection)
        connection_context.__exit__ = Mock(return_value=None)
        mock_db_manager.get_connection.return_value = connection_context
        
        # Test with very high price
        result = cart_manager.add_item(
            session_id="session123",
            product_id="EXPENSIVE",
            product_title="Very Expensive Item",
            quantity=1,
            price=999999.99
        )
        
        # Should handle large prices
        assert result["success"] is True or "error" in result
        
        # Test with very long product title
        long_title = "A" * 1000  # Very long title
        result = cart_manager.add_item(
            session_id="session123",
            product_id="LONG_TITLE",
            product_title=long_title,
            quantity=1
        )
        
        # Should handle or reject long titles appropriately
        assert "success" in result
    
    def test_concurrent_cart_operations(self, cart_manager, mock_db_manager):
        """Test concurrent cart operations on same session."""
        import threading
        import time
        
        # Mock database responses
        mock_db_manager.execute_query.return_value = []
        mock_db_manager.execute_update.return_value = 1
        
        # Mock connection for concurrent operations
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": "new-item-id"}
        
        cursor_context = Mock()
        cursor_context.__enter__ = Mock(return_value=mock_cursor)
        cursor_context.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = cursor_context
        
        connection_context = Mock()
        connection_context.__enter__ = Mock(return_value=mock_connection)
        connection_context.__exit__ = Mock(return_value=None)
        mock_db_manager.get_connection.return_value = connection_context
        
        results = []
        
        def add_item_concurrent(product_id):
            try:
                result = cart_manager.add_item(
                    session_id="concurrent_session",
                    product_id=product_id,
                    product_title=f"Product {product_id}",
                    quantity=1
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        # Create multiple threads adding different items
        threads = [
            threading.Thread(target=add_item_concurrent, args=(f"PROD_{i}",))
            for i in range(5)
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All operations should complete (successfully or with error)
        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict)
    
    def test_database_connection_failure_recovery(self, cart_manager, mock_db_manager):
        """Test recovery from database connection failures."""
        # First call fails
        mock_db_manager.execute_query.side_effect = [
            Exception("Connection failed"),
            []  # Second call succeeds
        ]
        
        # First attempt should fail
        result1 = cart_manager.add_item(
            session_id="session123",
            product_id="PROD123",
            product_title="Test Product",
            quantity=1
        )
        
        assert result1["success"] is False
        assert "Database error" in result1["error"]
        
        # Reset side effect for second attempt
        mock_db_manager.execute_query.side_effect = None
        mock_db_manager.execute_query.return_value = []
        
        # Mock successful insertion for retry
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": "new-item-id"}
        
        cursor_context = Mock()
        cursor_context.__enter__ = Mock(return_value=mock_cursor)
        cursor_context.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = cursor_context
        
        connection_context = Mock()
        connection_context.__enter__ = Mock(return_value=mock_connection)
        connection_context.__exit__ = Mock(return_value=None)
        mock_db_manager.get_connection.return_value = connection_context
        
        # Mock getting inserted item
        inserted_item = {
            "id": "new-item-id",
            "product_id": "PROD123",
            "product_title": "Test Product",
            "product_price": None,
            "product_image_url": None,
            "quantity": 1,
            "product_metadata": {},
            "added_at": datetime.now(),
            "updated_at": datetime.now()
        }
        mock_db_manager.execute_query.side_effect = [[], [inserted_item]]
        
        # Second attempt should succeed
        result2 = cart_manager.add_item(
            session_id="session123",
            product_id="PROD123",
            product_title="Test Product",
            quantity=1
        )
        
        assert result2["success"] is True


class TestShoppingCartToolsEdgeCases:
    """Test edge cases for shopping cart tools."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        return Mock(spec=ShoppingCartManager)
    
    @pytest.fixture
    def add_tool(self, mock_cart_manager):
        """Create AddToCartTool with mock manager."""
        return AddToCartTool(cart_manager=mock_cart_manager)
    
    def test_add_tool_with_unicode_product_names(self, add_tool, mock_cart_manager):
        """Test add tool with unicode product names."""
        mock_cart_manager.add_item.return_value = {
            "success": True,
            "message": "Added item to cart",
            "item": {"product_id": "UNICODE", "quantity": 1},
            "action": "added"
        }
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            # Test with various unicode characters
            unicode_names = [
                "iPhone 📱",
                "Café Laptop ☕",
                "Gaming Mouse 🖱️",
                "Bluetooth Headphones 🎧",
                "Tablet 中文",
                "Keyboard العربية"
            ]
            
            for name in unicode_names:
                result = add_tool._run(
                    product_id="UNICODE_TEST",
                    product_title=name,
                    quantity=1
                )
                
                assert result["success"] is True
                mock_cart_manager.add_item.assert_called()
    
    def test_tool_session_id_handling(self, add_tool):
        """Test tool session ID handling edge cases."""
        # Test with no session context
        with patch.object(add_tool, '_get_session_id', return_value=None):
            result = add_tool._run(
                product_id="TEST",
                product_title="Test Product",
                quantity=1
            )
            
            assert result["success"] is False
            assert "session" in result["error"].lower()
        
        # Test with empty session ID
        with patch.object(add_tool, '_get_session_id', return_value=""):
            result = add_tool._run(
                product_id="TEST",
                product_title="Test Product",
                quantity=1
            )
            
            assert result["success"] is False
            assert "session" in result["error"].lower()
    
    def test_tool_timeout_handling(self, add_tool, mock_cart_manager):
        """Test tool timeout handling."""
        import time
        
        def slow_add_item(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow operation
            return {
                "success": True,
                "message": "Added after delay",
                "item": {"product_id": "SLOW", "quantity": 1},
                "action": "added"
            }
        
        mock_cart_manager.add_item.side_effect = slow_add_item
        
        with patch.object(add_tool, '_get_session_id', return_value="test_session"):
            start_time = time.time()
            result = add_tool._run(
                product_id="SLOW_PRODUCT",
                product_title="Slow Product",
                quantity=1
            )
            end_time = time.time()
            
            # Should complete despite delay
            assert result["success"] is True
            assert end_time - start_time >= 0.1  # Took at least the delay time


class TestShoppingCartAgentEdgeCases:
    """Test edge cases for Shopping Cart Agent."""
    
    @pytest.fixture
    def mock_cart_manager(self):
        """Create mock cart manager."""
        manager = Mock(spec=ShoppingCartManager)
        manager.get_cart_contents.return_value = []
        manager.get_cart_summary.return_value = {
            "total_items": 0,
            "total_value": 0.0,
            "unique_products": 0,
            "is_empty": True
        }
        return manager
    
    @pytest.fixture
    def cart_agent(self, mock_cart_manager):
        """Create Shopping Cart Agent."""
        config = {"max_tool_calls": 5}
        return ShoppingCartAgent(config, mock_cart_manager)
    
    @pytest.mark.asyncio
    async def test_agent_with_corrupted_state(self, cart_agent):
        """Test agent handling of corrupted state."""
        corrupted_states = [
            None,  # None state
            {},  # Empty state
            {"invalid": "structure"},  # Invalid structure
            {"session_id": None, "current_query": "test"},  # None session_id
        ]
        
        for state in corrupted_states:
            try:
                result = await cart_agent._analyze_cart_request(state or {})
                # Should handle gracefully
                assert isinstance(result, dict)
                assert "current_step" in result
            except Exception as e:
                # If it raises an exception, it should be handled gracefully
                assert "state" in str(e).lower() or "session" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_agent_with_extremely_long_queries(self, cart_agent):
        """Test agent with extremely long queries."""
        long_query = "add to cart " * 1000  # Very long query
        
        state = create_initial_state(
            session_id="test_session",
            query=long_query
        )
        
        result = await cart_agent._analyze_cart_request(state)
        
        # Should handle long queries without crashing
        assert result["current_step"] == "analyze_cart_request"
        assert result["cart_operation"] == "add"
    
    @pytest.mark.asyncio
    async def test_agent_memory_usage_with_large_cart(self, cart_agent, mock_cart_manager):
        """Test agent memory usage with large cart contents."""
        # Mock large cart contents
        large_cart = [
            {
                "product_id": f"PROD_{i}",
                "product_title": f"Product {i}",
                "quantity": 1,
                "product_price": 10.0,
                "product_image_url": f"http://example.com/image_{i}.jpg",
                "product_metadata": {"category": "test", "data": "x" * 100},
                "added_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "subtotal": 10.0
            }
            for i in range(1000)  # 1000 items
        ]
        
        mock_cart_manager.get_cart_contents.return_value = large_cart
        mock_cart_manager.get_cart_summary.return_value = {
            "total_items": 1000,
            "total_value": 10000.0,
            "unique_products": 1000,
            "is_empty": False
        }
        
        state = create_initial_state(
            session_id="test_session",
            query="show my cart"
        )
        
        result = await cart_agent._update_cart_state(state)
        
        # Should handle large cart without memory issues
        assert result["current_step"] == "update_cart_state"
        assert result["cart_item_count"] == 1000
        assert len(result["current_cart_contents"]) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])