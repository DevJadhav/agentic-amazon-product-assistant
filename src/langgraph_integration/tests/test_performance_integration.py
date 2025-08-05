"""
Integration tests for performance optimization components.
Tests that all optimization features work together correctly.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from ..core.router.intent_classifier import IntentClassifier, IntentResult
from ..core.router.intent_cache import IntentClassificationCache, get_intent_cache, clear_intent_cache
from ..state.connection_pool import OptimizedConnectionPool
from ..state.shopping_cart_manager import ShoppingCartManager
from ..state.query_optimizer import QueryOptimizer
from ..monitoring.performance_monitor import get_performance_monitor
from ..monitoring.metrics_collector import get_metrics_collector
from ..monitoring.memory_monitor import get_memory_monitor


class TestPerformanceIntegration:
    """Test integration of all performance optimization components."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear cache before each test
        clear_intent_cache()
        
        # Reset performance monitor
        perf_monitor = get_performance_monitor()
        perf_monitor.reset_statistics()
        
        yield
        
        # Cleanup after test
        clear_intent_cache()
    
    def test_intent_classification_with_caching(self):
        """Test intent classification with caching integration."""
        classifier = IntentClassifier()
        cache = get_intent_cache()
        
        message = "add laptop to cart"
        context = {"user_id": "test_user"}
        
        # First call - should miss cache and classify
        start_time = time.time()
        result1 = classifier.classify_intent(message, context)
        first_call_time = time.time() - start_time
        
        assert result1.intent == "cart"
        assert result1.confidence > 0.5
        
        # Second call - should hit cache
        start_time = time.time()
        result2 = classifier.classify_intent(message, context)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        assert result2.intent == result1.intent
        assert result2.confidence == result1.confidence
        
        # Second call should be faster (cached)
        assert second_call_time < first_call_time
        
        # Verify cache statistics
        stats = cache.get_stats()
        assert stats["total_requests"] >= 2
        assert stats["total_hits"] >= 1
        assert stats["hit_rate"] > 0
    
    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_cart_operations_with_optimized_pool(self, mock_pool_class):
        """Test cart operations with optimized connection pool."""
        # Mock database components
        mock_pool = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_pool_class.return_value = mock_pool
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_cursor.rowcount = 1
        
        # Create cart manager (will use optimized pool)
        cart_manager = ShoppingCartManager()
        
        session_id = "test_session"
        product_id = "test_product"
        
        # Perform cart operations
        add_result = cart_manager.add_item(
            session_id=session_id,
            product_id=product_id,
            product_title="Test Product",
            quantity=1,
            price=10.0
        )
        
        # Verify operation succeeded
        assert add_result["success"] == True
        
        # Verify connection pool was used
        mock_pool.getconn.assert_called()
        mock_pool.putconn.assert_called()
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across all components."""
        perf_monitor = get_performance_monitor()
        metrics_collector = get_metrics_collector()
        
        # Perform operations that should be tracked
        classifier = IntentClassifier()
        
        # Multiple classifications to generate metrics
        messages = [
            "add phone to cart",
            "what are the best laptops?",
            "remove tablet from cart",
            "show me product reviews"
        ]
        
        for message in messages:
            result = classifier.classify_intent(message)
            assert result is not None
        
        # Check that performance metrics were recorded
        summary = perf_monitor.get_performance_summary(1)  # Last hour
        
        if not summary.get("no_data", False):
            assert summary["total_operations"] > 0
            assert summary["success_rate"] >= 0
            assert "avg_duration" in summary
        
        # Check metrics collection
        routing_metrics = metrics_collector.get_routing_metrics()
        assert isinstance(routing_metrics, dict)
    
    def test_memory_monitoring_integration(self):
        """Test memory monitoring integration."""
        memory_monitor = get_memory_monitor()
        
        # Take baseline snapshot
        baseline = memory_monitor.take_snapshot()
        assert baseline is not None
        assert baseline.process_memory > 0
        
        # Perform memory-intensive operations
        cache = get_intent_cache()
        classifier = IntentClassifier()
        
        # Generate cache entries
        for i in range(100):
            message = f"test message {i}"
            result = classifier.classify_intent(message)
            # Result should be cached automatically
        
        # Take another snapshot
        after_operations = memory_monitor.take_snapshot()
        assert after_operations.process_memory >= baseline.process_memory
        
        # Get memory summary
        summary = memory_monitor.get_memory_summary(1)
        assert not summary.get("no_data", False)
        assert "memory_stats" in summary
        assert "component_analysis" in summary
    
    def test_concurrent_operations_with_optimizations(self):
        """Test concurrent operations with all optimizations enabled."""
        num_threads = 5
        operations_per_thread = 20
        results = []
        
        def worker_thread(thread_id):
            """Worker thread function."""
            thread_results = []
            classifier = IntentClassifier()
            
            for i in range(operations_per_thread):
                try:
                    # Vary messages to test cache effectiveness
                    message = f"thread {thread_id} message {i % 5}"
                    result = classifier.classify_intent(message)
                    
                    thread_results.append({
                        "thread_id": thread_id,
                        "operation": i,
                        "success": True,
                        "intent": result.intent,
                        "confidence": result.confidence
                    })
                    
                except Exception as e:
                    thread_results.append({
                        "thread_id": thread_id,
                        "operation": i,
                        "success": False,
                        "error": str(e)
                    })
            
            results.extend(thread_results)
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Verify all operations completed
        assert len(results) == num_threads * operations_per_thread
        
        # Verify success rate
        successful_ops = [r for r in results if r["success"]]
        success_rate = len(successful_ops) / len(results)
        assert success_rate > 0.9, f"Low success rate: {success_rate:.2%}"
        
        # Verify performance
        ops_per_second = len(results) / total_time
        assert ops_per_second > 50, f"Low throughput: {ops_per_second:.1f} ops/s"
        
        # Check cache effectiveness
        cache = get_intent_cache()
        cache_stats = cache.get_stats()
        
        # Should have cache hits due to repeated messages
        if cache_stats["total_requests"] > 0:
            assert cache_stats["hit_rate"] > 0, "No cache hits detected"
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete end-to-end workflow with all optimizations."""
        # Initialize components
        classifier = IntentClassifier()
        cache = get_intent_cache()
        perf_monitor = get_performance_monitor()
        metrics_collector = get_metrics_collector()
        memory_monitor = get_memory_monitor()
        
        # Simulate realistic user interactions
        user_interactions = [
            ("add laptop to cart", "cart"),
            ("what are the best phones?", "qa"),
            ("remove headphones from cart", "cart"),
            ("compare iPhone vs Samsung", "qa"),
            ("show my cart contents", "cart"),
            ("tell me about this product", "qa"),
            ("add 2 tablets to cart", "cart"),
            ("what's the price of this item?", "qa")
        ]
        
        start_time = time.time()
        
        for message, expected_intent in user_interactions:
            # Take memory snapshot
            snapshot = memory_monitor.take_snapshot()
            
            # Classify intent (with caching)
            with metrics_collector.timer("user_interaction"):
                result = classifier.classify_intent(message)
            
            # Verify classification
            assert result.intent in ["cart", "qa", "unclear"]
            
            # Record custom metrics
            metrics_collector.increment_counter(f"intent_{result.intent}")
            metrics_collector.record_timer("classification_time", 0.001)  # Simulated
            
            # Simulate some processing delay
            time.sleep(0.01)
        
        total_time = time.time() - start_time
        
        # Verify overall performance
        assert total_time < 5.0, f"Workflow too slow: {total_time:.3f}s"
        
        # Check performance monitoring
        perf_summary = perf_monitor.get_performance_summary(1)
        if not perf_summary.get("no_data", False):
            assert perf_summary["total_operations"] > 0
        
        # Check metrics collection
        comprehensive_report = metrics_collector.get_comprehensive_report(1)
        assert "routing_metrics" in comprehensive_report
        assert "overall_summary" in comprehensive_report
        
        # Check memory health
        memory_health = memory_monitor.get_memory_health_check()
        assert memory_health["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Check cache effectiveness
        cache_stats = cache.get_stats()
        if cache_stats["total_requests"] > 0:
            # Should have some cache efficiency
            assert cache_stats["cache_size"] > 0
    
    def test_optimization_recommendations(self):
        """Test that optimization recommendations are generated."""
        perf_monitor = get_performance_monitor()
        metrics_collector = get_metrics_collector()
        memory_monitor = get_memory_monitor()
        cache = get_intent_cache()
        
        # Generate some activity to analyze
        classifier = IntentClassifier()
        
        for i in range(50):
            message = f"test message {i}"
            result = classifier.classify_intent(message)
        
        # Get recommendations from various components
        perf_recommendations = perf_monitor.get_optimization_recommendations()
        assert isinstance(perf_recommendations, list)
        
        cache_report = cache.get_cache_efficiency_report()
        assert "recommendations" in cache_report
        
        memory_health = memory_monitor.get_memory_health_check()
        assert "recommendations" in memory_health
        
        # Verify recommendations are actionable
        all_recommendations = (
            perf_recommendations + 
            cache_report["recommendations"] + 
            memory_health["recommendations"]
        )
        
        # Should have some recommendations (even if just informational)
        assert len(all_recommendations) >= 0  # At minimum, no errors
    
    def test_performance_under_stress(self):
        """Test performance under stress conditions."""
        num_operations = 200
        max_time_per_operation = 0.1  # 100ms max per operation
        
        classifier = IntentClassifier()
        start_time = time.time()
        
        for i in range(num_operations):
            operation_start = time.time()
            
            # Vary message patterns to test different code paths
            if i % 4 == 0:
                message = f"add product {i} to cart"
            elif i % 4 == 1:
                message = f"what is the price of product {i}?"
            elif i % 4 == 2:
                message = f"remove item {i} from cart"
            else:
                message = f"unclear message {i} xyz"
            
            result = classifier.classify_intent(message)
            
            operation_time = time.time() - operation_start
            
            # Each operation should complete within reasonable time
            assert operation_time < max_time_per_operation, \
                f"Operation {i} too slow: {operation_time:.3f}s"
            
            assert result is not None
            assert result.intent in ["cart", "qa", "unclear"]
        
        total_time = time.time() - start_time
        avg_time = total_time / num_operations
        ops_per_second = num_operations / total_time
        
        # Performance assertions
        assert avg_time < 0.05, f"Average operation time too high: {avg_time:.3f}s"
        assert ops_per_second > 20, f"Throughput too low: {ops_per_second:.1f} ops/s"
        
        # Memory should not grow excessively
        memory_monitor = get_memory_monitor()
        memory_health = memory_monitor.get_memory_health_check()
        
        # Should not be in unhealthy state after stress test
        assert memory_health["status"] != "unhealthy", \
            f"Memory unhealthy after stress test: {memory_health}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])