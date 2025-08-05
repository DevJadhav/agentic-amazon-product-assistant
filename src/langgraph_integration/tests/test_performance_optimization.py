"""
Performance tests and benchmarks for optimization components.
Tests intent caching, connection pooling, metrics collection, and memory monitoring.
"""

import pytest
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from ..core.router.intent_cache import IntentClassificationCache, get_intent_cache
from ..core.router.intent_classifier import IntentResult, IntentClassifier
from ..state.connection_pool import OptimizedConnectionPool
from ..monitoring.metrics_collector import MetricsCollector, MetricType
from ..monitoring.memory_monitor import MemoryMonitor
from ..state.query_optimizer import QueryOptimizer


class TestIntentClassificationCache:
    """Test intent classification caching performance."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        return IntentClassificationCache(max_size=100, default_ttl=60)
    
    @pytest.fixture
    def sample_intent_result(self):
        """Create sample intent result."""
        return IntentResult(
            intent="cart",
            confidence=0.8,
            entities=["laptop", "2"],
            clarification_needed=False,
            suggested_questions=[],
            reasoning="Cart keywords detected",
            metadata={"test": True}
        )
    
    def test_cache_basic_operations(self, cache, sample_intent_result):
        """Test basic cache operations."""
        message = "add 2 laptops to cart"
        
        # Test cache miss
        result = cache.get(message)
        assert result is None
        
        # Test cache set
        cache.set(message, sample_intent_result)
        
        # Test cache hit
        result = cache.get(message)
        assert result is not None
        assert result.intent == "cart"
        assert result.confidence == 0.8
    
    def test_cache_with_context(self, cache, sample_intent_result):
        """Test cache with context."""
        message = "add laptop"
        context = {"user_id": "123", "session": "abc"}
        
        # Set with context
        cache.set(message, sample_intent_result, context)
        
        # Get with same context - should hit
        result = cache.get(message, context)
        assert result is not None
        
        # Get with different context - should miss
        different_context = {"user_id": "456", "session": "def"}
        result = cache.get(message, different_context)
        assert result is None
    
    def test_cache_ttl_expiration(self, cache, sample_intent_result):
        """Test cache TTL expiration."""
        message = "test message"
        
        # Set with short TTL
        cache.set(message, sample_intent_result, ttl=0.1)
        
        # Should hit immediately
        result = cache.get(message)
        assert result is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should miss after expiration
        result = cache.get(message)
        assert result is None
    
    def test_cache_lru_eviction(self, sample_intent_result):
        """Test LRU eviction."""
        cache = IntentClassificationCache(max_size=3, default_ttl=60)
        
        # Fill cache to capacity
        for i in range(3):
            cache.set(f"message_{i}", sample_intent_result)
        
        # Access first message to make it recently used
        cache.get("message_0")
        
        # Add new message - should evict message_1 (least recently used)
        cache.set("message_3", sample_intent_result)
        
        # Check eviction
        assert cache.get("message_0") is not None  # Recently used
        assert cache.get("message_1") is None      # Evicted
        assert cache.get("message_2") is not None  # Still there
        assert cache.get("message_3") is not None  # Newly added
    
    def test_cache_performance_benchmark(self, cache, sample_intent_result):
        """Benchmark cache performance."""
        num_operations = 1000
        messages = [f"test message {i}" for i in range(num_operations)]
        
        # Benchmark cache set operations
        start_time = time.time()
        for message in messages:
            cache.set(message, sample_intent_result)
        set_time = time.time() - start_time
        
        # Benchmark cache get operations (hits)
        start_time = time.time()
        for message in messages:
            result = cache.get(message)
            assert result is not None
        get_time = time.time() - start_time
        
        # Performance assertions
        assert set_time < 1.0, f"Cache set operations too slow: {set_time:.3f}s"
        assert get_time < 0.5, f"Cache get operations too slow: {get_time:.3f}s"
        
        # Calculate operations per second
        set_ops_per_sec = num_operations / set_time
        get_ops_per_sec = num_operations / get_time
        
        assert set_ops_per_sec > 1000, f"Set OPS too low: {set_ops_per_sec:.0f}"
        assert get_ops_per_sec > 2000, f"Get OPS too low: {get_ops_per_sec:.0f}"
    
    def test_cache_concurrent_access(self, cache, sample_intent_result):
        """Test concurrent cache access."""
        num_threads = 10
        operations_per_thread = 100
        results = []
        
        def cache_worker(thread_id):
            thread_results = []
            for i in range(operations_per_thread):
                message = f"thread_{thread_id}_message_{i}"
                
                # Set operation
                cache.set(message, sample_intent_result)
                
                # Get operation
                result = cache.get(message)
                thread_results.append(result is not None)
            
            results.extend(thread_results)
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Verify all operations succeeded
        assert all(results), "Some cache operations failed under concurrency"
        assert len(results) == num_threads * operations_per_thread
        
        # Performance check
        total_ops = len(results) * 2  # Set + Get
        ops_per_sec = total_ops / total_time
        assert ops_per_sec > 1000, f"Concurrent OPS too low: {ops_per_sec:.0f}"
    
    def test_cache_statistics(self, cache, sample_intent_result):
        """Test cache statistics collection."""
        # Perform operations
        cache.set("message1", sample_intent_result)
        cache.set("message2", sample_intent_result)
        
        cache.get("message1")  # Hit
        cache.get("message1")  # Hit
        cache.get("message3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1
        assert stats["total_requests"] == 3
        assert stats["hit_rate"] == 2/3
        assert stats["cache_size"] == 2


class TestOptimizedConnectionPool:
    """Test optimized database connection pool."""
    
    @pytest.fixture
    def mock_db_config(self):
        """Mock database configuration."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "min_connections": 2,
            "max_connections": 10
        }
    
    @pytest.fixture
    def pool(self, mock_db_config):
        """Create connection pool for testing."""
        with patch('psycopg2.pool.ThreadedConnectionPool'):
            return OptimizedConnectionPool(**mock_db_config)
    
    def test_pool_initialization(self, mock_db_config):
        """Test connection pool initialization."""
        with patch('psycopg2.pool.ThreadedConnectionPool') as mock_pool:
            pool = OptimizedConnectionPool(**mock_db_config)
            
            mock_pool.assert_called_once()
            assert pool.min_connections == 2
            assert pool.max_connections == 10
    
    def test_connection_acquisition_performance(self, pool):
        """Test connection acquisition performance."""
        num_acquisitions = 100
        
        # Mock connection
        mock_conn = Mock()
        pool.pool.getconn.return_value = mock_conn
        
        start_time = time.time()
        
        for _ in range(num_acquisitions):
            with pool.get_connection() as conn:
                assert conn is not None
        
        total_time = time.time() - start_time
        avg_time = total_time / num_acquisitions
        
        # Should be very fast
        assert avg_time < 0.001, f"Connection acquisition too slow: {avg_time:.6f}s"
    
    def test_concurrent_connection_usage(self, pool):
        """Test concurrent connection usage."""
        num_threads = 20
        operations_per_thread = 10
        results = []
        
        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [{"result": "test"}]
        
        pool.pool.getconn.return_value = mock_conn
        
        def db_worker():
            thread_results = []
            for _ in range(operations_per_thread):
                try:
                    result = pool.execute_query("SELECT 1")
                    thread_results.append(len(result) > 0)
                except Exception as e:
                    thread_results.append(False)
            
            results.extend(thread_results)
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=db_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Verify all operations succeeded
        assert all(results), "Some database operations failed under concurrency"
        
        # Performance check
        total_ops = len(results)
        ops_per_sec = total_ops / total_time
        assert ops_per_sec > 100, f"Concurrent DB OPS too low: {ops_per_sec:.0f}"
    
    def test_pool_optimization(self, pool):
        """Test pool auto-optimization."""
        # Simulate high usage
        pool._pool_stats["pool_exhaustions"] = 5
        
        # Mock current stats
        with patch.object(pool, 'get_pool_stats') as mock_stats:
            mock_stats.return_value = {
                "current_state": {"active_connections": 8},
                "performance": {"avg_query_time": 2.0},
                "errors": {"pool_exhaustions": 5}
            }
            
            # Run optimization
            result = pool.optimize_pool()
            
            assert "optimizations_applied" in result
            assert len(result["optimizations_applied"]) > 0
    
    def test_pool_health_check(self, pool):
        """Test pool health check."""
        # Mock successful connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {"health_check": 1}
        
        pool.pool.getconn.return_value = mock_conn
        
        health = pool.health_check()
        
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "response_time" in health
        assert "stats" in health


class TestMetricsCollector:
    """Test metrics collection performance."""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector for testing."""
        return MetricsCollector(max_points_per_metric=1000, retention_hours=1)
    
    def test_metric_recording_performance(self, collector):
        """Test metric recording performance."""
        num_metrics = 10000
        
        start_time = time.time()
        
        for i in range(num_metrics):
            collector.record_metric(f"test_metric_{i % 100}", float(i))
        
        total_time = time.time() - start_time
        
        # Should be very fast
        assert total_time < 1.0, f"Metric recording too slow: {total_time:.3f}s"
        
        ops_per_sec = num_metrics / total_time
        assert ops_per_sec > 10000, f"Metric recording OPS too low: {ops_per_sec:.0f}"
    
    def test_concurrent_metric_recording(self, collector):
        """Test concurrent metric recording."""
        num_threads = 10
        metrics_per_thread = 1000
        
        def metric_worker(thread_id):
            for i in range(metrics_per_thread):
                collector.record_metric(f"thread_{thread_id}_metric", float(i))
        
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=metric_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        total_metrics = num_threads * metrics_per_thread
        
        ops_per_sec = total_metrics / total_time
        assert ops_per_sec > 5000, f"Concurrent metric OPS too low: {ops_per_sec:.0f}"
    
    def test_metric_aggregation_performance(self, collector):
        """Test metric aggregation performance."""
        # Record many metrics
        for i in range(1000):
            collector.record_metric("test_metric", float(i))
        
        # Test summary generation
        start_time = time.time()
        summary = collector.get_metric_summary("test_metric")
        summary_time = time.time() - start_time
        
        assert summary is not None
        assert summary_time < 0.1, f"Metric summary too slow: {summary_time:.3f}s"
        
        # Test comprehensive report
        start_time = time.time()
        report = collector.get_comprehensive_report()
        report_time = time.time() - start_time
        
        assert report is not None
        assert report_time < 1.0, f"Comprehensive report too slow: {report_time:.3f}s"
    
    def test_timer_context_manager(self, collector):
        """Test timer context manager performance."""
        num_timings = 1000
        
        start_time = time.time()
        
        for i in range(num_timings):
            with collector.timer(f"test_operation_{i % 10}"):
                time.sleep(0.001)  # Simulate work
        
        total_time = time.time() - start_time
        
        # Should complete reasonably quickly despite sleep
        assert total_time < 5.0, f"Timer operations too slow: {total_time:.3f}s"
        
        # Check that metrics were recorded
        summary = collector.get_metric_summary("test_operation_0")
        assert summary is not None
        assert summary.count > 0


class TestMemoryMonitor:
    """Test memory monitoring performance."""
    
    @pytest.fixture
    def monitor(self):
        """Create memory monitor for testing."""
        return MemoryMonitor(monitoring_interval=0.1, snapshot_retention=100)
    
    def test_snapshot_performance(self, monitor):
        """Test memory snapshot performance."""
        num_snapshots = 100
        
        start_time = time.time()
        
        for _ in range(num_snapshots):
            snapshot = monitor.take_snapshot()
            assert snapshot is not None
        
        total_time = time.time() - start_time
        avg_time = total_time / num_snapshots
        
        # Should be fast
        assert avg_time < 0.01, f"Memory snapshot too slow: {avg_time:.6f}s"
    
    def test_component_tracking(self, monitor):
        """Test component memory tracking."""
        # Register test component
        test_memory = [1024]  # Mutable for closure
        
        def track_test_component():
            return test_memory[0]
        
        monitor.register_component_tracker("test_component", track_test_component)
        
        # Take snapshot
        snapshot = monitor.take_snapshot()
        
        assert "test_component" in snapshot.component_memory
        assert snapshot.component_memory["test_component"] == 1024
        
        # Change memory and take another snapshot
        test_memory[0] = 2048
        snapshot2 = monitor.take_snapshot()
        
        assert snapshot2.component_memory["test_component"] == 2048
    
    def test_leak_detection_performance(self, monitor):
        """Test memory leak detection performance."""
        # Register component with growing memory
        memory_size = [1024]
        
        def growing_component():
            memory_size[0] += 100  # Simulate memory growth
            return memory_size[0]
        
        monitor.register_component_tracker("growing_component", growing_component)
        
        # Take multiple snapshots to establish pattern
        for _ in range(10):
            monitor.take_snapshot()
            time.sleep(0.01)
        
        # Test leak detection
        start_time = time.time()
        leaks = monitor.detect_memory_leaks()
        detection_time = time.time() - start_time
        
        # Should be fast
        assert detection_time < 0.1, f"Leak detection too slow: {detection_time:.3f}s"
        
        # Should detect the growing component
        assert len(leaks) > 0
        assert any(leak.component == "growing_component" for leak in leaks)
    
    def test_memory_optimization(self, monitor):
        """Test memory optimization performance."""
        start_time = time.time()
        result = monitor.optimize_memory()
        optimization_time = time.time() - start_time
        
        # Should complete quickly
        assert optimization_time < 1.0, f"Memory optimization too slow: {optimization_time:.3f}s"
        
        assert "optimizations_performed" in result
        assert "memory_after_optimization" in result


class TestQueryOptimizer:
    """Test query optimization performance."""
    
    @pytest.fixture
    def optimizer(self):
        """Create query optimizer for testing."""
        with patch('src.langgraph_integration.state.database.get_database_manager'):
            return QueryOptimizer()
    
    def test_query_optimization_performance(self, optimizer):
        """Test query optimization performance."""
        test_queries = [
            "SELECT * FROM shopping_cart WHERE session_id = %s",
            "SELECT COUNT(*) FROM shopping_cart WHERE session_id = %s",
            "INSERT INTO shopping_cart (session_id, product_id) VALUES (%s, %s)",
            "UPDATE shopping_cart SET quantity = %s WHERE session_id = %s AND product_id = %s",
            "DELETE FROM shopping_cart WHERE session_id = %s AND product_id = %s"
        ]
        
        start_time = time.time()
        
        for query in test_queries:
            optimized_query, info = optimizer.optimize_query(query)
            assert optimized_query is not None
            assert info is not None
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_queries)
        
        # Should be fast
        assert avg_time < 0.01, f"Query optimization too slow: {avg_time:.6f}s"
    
    def test_cart_query_analysis(self, optimizer):
        """Test cart query analysis performance."""
        # Mock database manager
        optimizer.db_manager.execute_query = Mock(return_value=[
            {"QUERY PLAN": [{"Total Cost": 100.0, "Actual Total Time": 50.0}]}
        ])
        
        start_time = time.time()
        analysis = optimizer.analyze_cart_queries()
        analysis_time = time.time() - start_time
        
        # Should complete quickly
        assert analysis_time < 1.0, f"Cart query analysis too slow: {analysis_time:.3f}s"
        
        assert len(analysis) > 0
        for query_name, query_analysis in analysis.items():
            assert "query" in query_analysis
            assert "structure_analysis" in query_analysis
    
    def test_index_recommendations(self, optimizer):
        """Test index recommendation performance."""
        # Mock database queries
        optimizer.db_manager.execute_query = Mock(return_value=[
            {"indexname": "test_index", "indexdef": "CREATE INDEX test_index ON table (column)"}
        ])
        
        start_time = time.time()
        recommendations = optimizer.recommend_indexes()
        recommendation_time = time.time() - start_time
        
        # Should be fast
        assert recommendation_time < 0.5, f"Index recommendations too slow: {recommendation_time:.3f}s"
        
        assert isinstance(recommendations, list)


class TestIntegrationPerformance:
    """Test integrated performance of all optimization components."""
    
    def test_end_to_end_performance(self):
        """Test end-to-end performance with all optimizations enabled."""
        # Initialize all components
        cache = IntentClassificationCache(max_size=1000)
        collector = MetricsCollector(max_points_per_metric=1000)
        monitor = MemoryMonitor(monitoring_interval=1.0)
        
        # Mock intent classifier
        classifier = Mock()
        classifier.classify_intent.return_value = IntentResult(
            intent="cart",
            confidence=0.8,
            entities=["laptop"],
            clarification_needed=False,
            suggested_questions=[],
            reasoning="Test",
            metadata={}
        )
        
        # Simulate realistic workload
        num_requests = 100
        messages = [f"add laptop {i} to cart" for i in range(num_requests)]
        
        start_time = time.time()
        
        for i, message in enumerate(messages):
            # Check cache first
            cached_result = cache.get(message)
            
            if cached_result is None:
                # Classify intent
                with collector.timer("intent_classification"):
                    result = classifier.classify_intent(message)
                
                # Cache result
                cache.set(message, result)
            else:
                result = cached_result
            
            # Record metrics
            collector.increment_counter("requests_processed")
            collector.record_timer("request_processing_time", 0.001)
            
            # Simulate some processing
            time.sleep(0.001)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 5.0, f"End-to-end processing too slow: {total_time:.3f}s"
        
        requests_per_sec = num_requests / total_time
        assert requests_per_sec > 20, f"Request throughput too low: {requests_per_sec:.1f} req/s"
        
        # Verify cache effectiveness
        cache_stats = cache.get_stats()
        assert cache_stats["hit_rate"] > 0.0  # Should have some cache hits
        
        # Verify metrics collection
        summary = collector.get_metric_summary("requests_processed")
        assert summary is not None
        assert summary.sum == num_requests
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        monitor = MemoryMonitor(monitoring_interval=0.1)
        
        # Take baseline snapshot
        baseline = monitor.take_snapshot()
        
        # Simulate sustained load
        cache = IntentClassificationCache(max_size=10000)
        collector = MetricsCollector(max_points_per_metric=10000)
        
        # Generate load
        for i in range(1000):
            # Cache operations
            result = IntentResult(
                intent="test",
                confidence=0.8,
                entities=[],
                clarification_needed=False,
                suggested_questions=[],
                reasoning="test",
                metadata={}
            )
            cache.set(f"message_{i}", result)
            
            # Metrics operations
            collector.record_metric(f"metric_{i % 10}", float(i))
        
        # Take final snapshot
        final = monitor.take_snapshot()
        
        # Memory growth should be reasonable
        memory_growth = final.process_memory - baseline.process_memory
        growth_mb = memory_growth / (1024 * 1024)
        
        # Should not grow excessively (allow up to 50MB growth)
        assert growth_mb < 50, f"Excessive memory growth: {growth_mb:.1f}MB"
        
        # Memory usage should be reasonable
        assert final.memory_usage_percent < 90, f"High memory usage: {final.memory_usage_percent:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])