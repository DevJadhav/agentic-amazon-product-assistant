#!/usr/bin/env python3
"""
Performance optimization demonstration script.
Shows all optimization components working together with benchmarks.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langgraph_integration.core.router.intent_classifier import IntentClassifier
from langgraph_integration.core.router.intent_cache import get_intent_cache, clear_intent_cache
from langgraph_integration.state.connection_pool import get_optimized_pool
from langgraph_integration.state.shopping_cart_manager import ShoppingCartManager
from langgraph_integration.state.query_optimizer import get_query_optimizer
from langgraph_integration.monitoring.performance_monitor import get_performance_monitor
from langgraph_integration.monitoring.metrics_collector import get_metrics_collector
from langgraph_integration.monitoring.memory_monitor import get_memory_monitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceOptimizationDemo:
    """Demonstrates all performance optimization features."""
    
    def __init__(self):
        """Initialize demo components."""
        self.intent_classifier = IntentClassifier()
        self.intent_cache = get_intent_cache()
        self.connection_pool = get_optimized_pool()
        self.cart_manager = ShoppingCartManager()
        self.query_optimizer = get_query_optimizer()
        self.perf_monitor = get_performance_monitor()
        self.metrics_collector = get_metrics_collector()
        self.memory_monitor = get_memory_monitor()
        
        logger.info("Performance optimization demo initialized")
    
    def run_intent_classification_benchmark(self, num_requests: int = 1000) -> Dict[str, Any]:
        """Benchmark intent classification with and without caching."""
        logger.info(f"Running intent classification benchmark with {num_requests} requests")
        
        # Test messages
        test_messages = [
            "add laptop to cart",
            "remove phone from cart",
            "show me my cart",
            "what are the best headphones?",
            "compare iPhone vs Samsung",
            "add 2 tablets to cart",
            "clear my cart",
            "tell me about this product",
            "buy 3 cameras",
            "what's in my shopping cart?"
        ]
        
        # Clear cache for fair comparison
        clear_intent_cache()
        
        # Benchmark without cache (cold start)
        start_time = time.time()
        for i in range(num_requests):
            message = test_messages[i % len(test_messages)]
            result = self.intent_classifier.classify_intent(message)
        cold_time = time.time() - start_time
        
        # Benchmark with cache (warm cache)
        start_time = time.time()
        for i in range(num_requests):
            message = test_messages[i % len(test_messages)]
            result = self.intent_classifier.classify_intent(message)
        warm_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = self.intent_cache.get_stats()
        
        return {
            "num_requests": num_requests,
            "cold_time": cold_time,
            "warm_time": warm_time,
            "cold_rps": num_requests / cold_time,
            "warm_rps": num_requests / warm_time,
            "speedup": cold_time / warm_time,
            "cache_stats": cache_stats
        }
    
    def run_database_benchmark(self, num_operations: int = 500) -> Dict[str, Any]:
        """Benchmark database operations with connection pooling."""
        logger.info(f"Running database benchmark with {num_operations} operations")
        
        session_id = "benchmark_session"
        
        # Clear any existing cart data
        try:
            self.cart_manager.clear_cart(session_id)
        except:
            pass  # Ignore if cart doesn't exist
        
        # Benchmark cart operations
        start_time = time.time()
        
        # Add items
        for i in range(num_operations // 4):
            self.cart_manager.add_item(
                session_id=session_id,
                product_id=f"product_{i}",
                product_title=f"Test Product {i}",
                quantity=1,
                price=float(10 + i)
            )
        
        # Get cart contents
        for i in range(num_operations // 4):
            contents = self.cart_manager.get_cart_contents(session_id)
        
        # Update quantities
        for i in range(num_operations // 4):
            self.cart_manager.add_item(
                session_id=session_id,
                product_id=f"product_{i % 10}",
                product_title=f"Test Product {i % 10}",
                quantity=2
            )
        
        # Remove items
        for i in range(num_operations // 4):
            try:
                self.cart_manager.remove_item(
                    session_id=session_id,
                    product_id=f"product_{i % 10}"
                )
            except:
                pass  # Ignore if item doesn't exist
        
        total_time = time.time() - start_time
        
        # Get connection pool stats
        pool_stats = self.connection_pool.get_pool_stats()
        
        # Clean up
        try:
            self.cart_manager.clear_cart(session_id)
        except:
            pass
        
        return {
            "num_operations": num_operations,
            "total_time": total_time,
            "operations_per_second": num_operations / total_time,
            "avg_operation_time": total_time / num_operations,
            "pool_stats": pool_stats
        }
    
    def run_query_optimization_benchmark(self) -> Dict[str, Any]:
        """Benchmark query optimization."""
        logger.info("Running query optimization benchmark")
        
        # Test queries
        test_queries = [
            "SELECT * FROM shopping_cart WHERE session_id = %s",
            "SELECT COUNT(*), SUM(quantity * product_price) FROM shopping_cart WHERE session_id = %s",
            "INSERT INTO shopping_cart (session_id, product_id, product_title, quantity) VALUES (%s, %s, %s, %s)",
            "UPDATE shopping_cart SET quantity = %s WHERE session_id = %s AND product_id = %s",
            "DELETE FROM shopping_cart WHERE session_id = %s AND product_id = %s"
        ]
        
        optimization_results = []
        
        for query in test_queries:
            start_time = time.time()
            optimized_query, optimization_info = self.query_optimizer.optimize_query(query)
            optimization_time = time.time() - start_time
            
            optimization_results.append({
                "original_query": query[:50] + "...",
                "optimization_time": optimization_time,
                "optimizations_applied": optimization_info.get("optimizations_applied", []),
                "estimated_improvement": optimization_info.get("estimated_improvement", 0.0)
            })
        
        # Get index recommendations
        start_time = time.time()
        index_recommendations = self.query_optimizer.recommend_indexes()
        recommendation_time = time.time() - start_time
        
        return {
            "query_optimizations": optimization_results,
            "index_recommendations": len(index_recommendations),
            "recommendation_time": recommendation_time,
            "performance_report": self.query_optimizer.get_performance_report()
        }
    
    def run_memory_monitoring_benchmark(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Benchmark memory monitoring."""
        logger.info(f"Running memory monitoring benchmark for {duration_seconds} seconds")
        
        # Take baseline snapshot
        baseline = self.memory_monitor.take_snapshot()
        
        # Simulate memory-intensive operations
        data_structures = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Create some data structures
            data_structures.append([i for i in range(1000)])
            
            # Simulate cart operations
            if len(data_structures) % 10 == 0:
                session_id = f"memory_test_{len(data_structures)}"
                try:
                    self.cart_manager.add_item(
                        session_id=session_id,
                        product_id="memory_test_product",
                        product_title="Memory Test Product",
                        quantity=1
                    )
                except:
                    pass
            
            # Clean up periodically
            if len(data_structures) > 100:
                data_structures = data_structures[-50:]
            
            time.sleep(0.1)
        
        # Take final snapshot
        final = self.memory_monitor.take_snapshot()
        
        # Get memory summary
        memory_summary = self.memory_monitor.get_memory_summary(hours=1)
        
        # Detect leaks
        leaks = self.memory_monitor.detect_memory_leaks()
        
        return {
            "duration": duration_seconds,
            "baseline_memory": baseline.process_memory,
            "final_memory": final.process_memory,
            "memory_growth": final.process_memory - baseline.process_memory,
            "memory_summary": memory_summary,
            "detected_leaks": len(leaks),
            "leak_details": [leak.to_dict() for leak in leaks]
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all optimization features."""
        logger.info("Starting comprehensive performance benchmark")
        
        results = {}
        
        # Intent classification benchmark
        logger.info("1/4 - Intent Classification Benchmark")
        results["intent_classification"] = self.run_intent_classification_benchmark(500)
        
        # Database benchmark
        logger.info("2/4 - Database Operations Benchmark")
        results["database_operations"] = self.run_database_benchmark(200)
        
        # Query optimization benchmark
        logger.info("3/4 - Query Optimization Benchmark")
        results["query_optimization"] = self.run_query_optimization_benchmark()
        
        # Memory monitoring benchmark
        logger.info("4/4 - Memory Monitoring Benchmark")
        results["memory_monitoring"] = self.run_memory_monitoring_benchmark(15)
        
        # Get overall performance summary
        perf_summary = self.perf_monitor.get_performance_summary(1)
        metrics_report = self.metrics_collector.get_comprehensive_report(1)
        memory_health = self.memory_monitor.get_memory_health_check()
        
        results["overall_performance"] = {
            "performance_summary": perf_summary,
            "metrics_report": metrics_report,
            "memory_health": memory_health
        }
        
        return results
    
    def print_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Print formatted benchmark results."""
        print("\n" + "="*80)
        print("PERFORMANCE OPTIMIZATION BENCHMARK RESULTS")
        print("="*80)
        
        # Intent Classification Results
        intent_results = results["intent_classification"]
        print(f"\n📊 INTENT CLASSIFICATION BENCHMARK")
        print(f"   Requests: {intent_results['num_requests']}")
        print(f"   Cold Start: {intent_results['cold_time']:.3f}s ({intent_results['cold_rps']:.0f} req/s)")
        print(f"   With Cache: {intent_results['warm_time']:.3f}s ({intent_results['warm_rps']:.0f} req/s)")
        print(f"   Speedup: {intent_results['speedup']:.1f}x")
        print(f"   Cache Hit Rate: {intent_results['cache_stats']['hit_rate']:.1%}")
        
        # Database Results
        db_results = results["database_operations"]
        print(f"\n🗄️  DATABASE OPERATIONS BENCHMARK")
        print(f"   Operations: {db_results['num_operations']}")
        print(f"   Total Time: {db_results['total_time']:.3f}s")
        print(f"   Throughput: {db_results['operations_per_second']:.0f} ops/s")
        print(f"   Avg Operation: {db_results['avg_operation_time']*1000:.1f}ms")
        
        # Query Optimization Results
        query_results = results["query_optimization"]
        print(f"\n🔍 QUERY OPTIMIZATION BENCHMARK")
        print(f"   Queries Analyzed: {len(query_results['query_optimizations'])}")
        print(f"   Index Recommendations: {query_results['index_recommendations']}")
        print(f"   Recommendation Time: {query_results['recommendation_time']*1000:.1f}ms")
        
        # Memory Monitoring Results
        memory_results = results["memory_monitoring"]
        print(f"\n🧠 MEMORY MONITORING BENCHMARK")
        print(f"   Duration: {memory_results['duration']}s")
        print(f"   Memory Growth: {memory_results['memory_growth']/1024/1024:.1f}MB")
        print(f"   Detected Leaks: {memory_results['detected_leaks']}")
        
        # Overall Performance
        overall = results["overall_performance"]
        perf_summary = overall["performance_summary"]
        memory_health = overall["memory_health"]
        
        print(f"\n📈 OVERALL PERFORMANCE SUMMARY")
        if not perf_summary.get("no_data", False):
            print(f"   Total Operations: {perf_summary.get('total_operations', 0)}")
            print(f"   Success Rate: {perf_summary.get('success_rate', 0):.1%}")
            print(f"   Avg Response Time: {perf_summary.get('avg_duration', 0)*1000:.1f}ms")
        
        print(f"   Memory Status: {memory_health['status'].upper()}")
        print(f"   Memory Usage: {memory_health['current_memory_usage']:.1f}%")
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)


def main():
    """Main demo function."""
    try:
        demo = PerformanceOptimizationDemo()
        
        print("🚀 Starting Performance Optimization Demo")
        print("This will benchmark all optimization components...")
        
        # Run comprehensive benchmark
        results = demo.run_comprehensive_benchmark()
        
        # Print results
        demo.print_benchmark_results(results)
        
        # Optionally save results to file
        import json
        with open("performance_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: performance_benchmark_results.json")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())