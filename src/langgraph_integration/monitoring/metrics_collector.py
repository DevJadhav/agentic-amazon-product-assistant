"""
Comprehensive performance metrics collection for routing and cart operations.
Collects, aggregates, and analyzes performance data across all system components.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import statistics
from enum import Enum

from .performance_monitor import get_performance_monitor, performance_track

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "type": self.metric_type.value
        }


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float
    std_dev: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
            "std_dev": self.std_dev
        }


class MetricsCollector:
    """Comprehensive metrics collection and analysis system."""
    
    def __init__(self, 
                 max_points_per_metric: int = 10000,
                 retention_hours: int = 24,
                 aggregation_interval: int = 60):
        """
        Initialize metrics collector.
        
        Args:
            max_points_per_metric: Maximum data points to keep per metric
            retention_hours: How long to retain metric data
            aggregation_interval: Interval for metric aggregation in seconds
        """
        self.max_points_per_metric = max_points_per_metric
        self.retention_hours = retention_hours
        self.aggregation_interval = aggregation_interval
        
        # Metric storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._metric_types: Dict[str, MetricType] = {}
        self._lock = threading.RLock()
        
        # Aggregated metrics
        self._aggregated_metrics: Dict[str, List[MetricSummary]] = defaultdict(list)
        self._last_aggregation = time.time()
        
        # Performance monitor integration
        self.perf_monitor = get_performance_monitor()
        
        # Metric collection callbacks
        self._collectors: List[Callable[[], Dict[str, float]]] = []
        
        # Start background aggregation
        self._aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._aggregation_thread.start()
        
        # Register built-in collectors
        self._register_builtin_collectors()
        
        logger.info(f"Metrics collector initialized with {max_points_per_metric} points per metric")
    
    def record_metric(self, name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     metric_type: MetricType = MetricType.GAUGE) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            metric_type: Type of metric
        """
        with self._lock:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self._metrics[name].append(point)
            self._metric_types[name] = metric_type
            
            logger.debug(f"Recorded metric: {name}={value} (type: {metric_type.value})")
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, tags, MetricType.COUNTER)
    
    def set_gauge(self, name: str, value: float, 
                  tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value, tags, MetricType.GAUGE)
    
    def record_timer(self, name: str, duration: float, 
                    tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric (duration in seconds)."""
        self.record_metric(name, duration, tags, MetricType.TIMER)
    
    def record_histogram(self, name: str, value: float, 
                        tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self.record_metric(name, value, tags, MetricType.HISTOGRAM)
    
    @performance_track("metrics_timer_context")
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        class TimerContext:
            def __init__(self, collector, metric_name, metric_tags):
                self.collector = collector
                self.name = metric_name
                self.tags = metric_tags
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    duration = time.time() - self.start_time
                    self.collector.record_timer(self.name, duration, self.tags)
        
        return TimerContext(self, name, tags)
    
    def get_metric_summary(self, name: str, hours: int = 1) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            hours: Number of hours to include in summary
            
        Returns:
            MetricSummary or None if metric not found
        """
        with self._lock:
            if name not in self._metrics:
                return None
            
            cutoff_time = time.time() - (hours * 3600)
            points = [p for p in self._metrics[name] if p.timestamp >= cutoff_time]
            
            if not points:
                return None
            
            values = [p.value for p in points]
            
            return MetricSummary(
                name=name,
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                p95=self._percentile(values, 0.95),
                p99=self._percentile(values, 0.99),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0.0
            )
    
    def get_all_metrics_summary(self, hours: int = 1) -> Dict[str, MetricSummary]:
        """Get summary for all metrics."""
        summaries = {}
        
        with self._lock:
            for name in self._metrics.keys():
                summary = self.get_metric_summary(name, hours)
                if summary:
                    summaries[name] = summary
        
        return summaries
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing-specific performance metrics."""
        routing_metrics = {}
        
        # Intent classification metrics
        intent_summary = self.get_metric_summary("intent_classification_time")
        if intent_summary:
            routing_metrics["intent_classification"] = intent_summary.to_dict()
        
        # Router decision metrics
        router_summary = self.get_metric_summary("router_decision_time")
        if router_summary:
            routing_metrics["router_decision"] = router_summary.to_dict()
        
        # Clarification metrics
        clarification_summary = self.get_metric_summary("clarification_requests")
        if clarification_summary:
            routing_metrics["clarification_requests"] = clarification_summary.to_dict()
        
        # Cache metrics
        cache_hit_rate = self._calculate_cache_hit_rate()
        if cache_hit_rate is not None:
            routing_metrics["cache_hit_rate"] = cache_hit_rate
        
        return routing_metrics
    
    def get_cart_metrics(self) -> Dict[str, Any]:
        """Get cart operation performance metrics."""
        cart_metrics = {}
        
        # Cart operation metrics
        for operation in ["add_to_cart", "remove_from_cart", "list_cart"]:
            summary = self.get_metric_summary(f"cart_{operation}_time")
            if summary:
                cart_metrics[f"{operation}_performance"] = summary.to_dict()
        
        # Database operation metrics
        db_summary = self.get_metric_summary("cart_db_operation_time")
        if db_summary:
            cart_metrics["database_performance"] = db_summary.to_dict()
        
        # Cart state metrics
        cart_size_summary = self.get_metric_summary("cart_item_count")
        if cart_size_summary:
            cart_metrics["cart_size_stats"] = cart_size_summary.to_dict()
        
        return cart_metrics
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        db_metrics = {}
        
        # Connection pool metrics
        pool_summary = self.get_metric_summary("db_connection_pool_size")
        if pool_summary:
            db_metrics["connection_pool"] = pool_summary.to_dict()
        
        # Query performance
        query_summary = self.get_metric_summary("db_query_time")
        if query_summary:
            db_metrics["query_performance"] = query_summary.to_dict()
        
        # Connection acquisition time
        conn_summary = self.get_metric_summary("db_connection_acquisition_time")
        if conn_summary:
            db_metrics["connection_acquisition"] = conn_summary.to_dict()
        
        return db_metrics
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory usage metrics."""
        memory_metrics = {}
        
        # Cart state memory
        cart_memory_summary = self.get_metric_summary("cart_state_memory_usage")
        if cart_memory_summary:
            memory_metrics["cart_state"] = cart_memory_summary.to_dict()
        
        # Cache memory
        cache_memory_summary = self.get_metric_summary("cache_memory_usage")
        if cache_memory_summary:
            memory_metrics["cache"] = cache_memory_summary.to_dict()
        
        # Overall memory
        total_memory_summary = self.get_metric_summary("total_memory_usage")
        if total_memory_summary:
            memory_metrics["total"] = total_memory_summary.to_dict()
        
        return memory_metrics
    
    def get_comprehensive_report(self, hours: int = 1) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "report_period_hours": hours,
            "generated_at": datetime.utcnow().isoformat(),
            "routing_metrics": self.get_routing_metrics(),
            "cart_metrics": self.get_cart_metrics(),
            "database_metrics": self.get_database_metrics(),
            "memory_metrics": self.get_memory_metrics(),
            "overall_summary": self._get_overall_summary(hours),
            "performance_alerts": self._generate_performance_alerts(),
            "optimization_recommendations": self._generate_optimization_recommendations()
        }
    
    def register_collector(self, collector_func: Callable[[], Dict[str, float]]) -> None:
        """Register a custom metric collector function."""
        self._collectors.append(collector_func)
        logger.info(f"Registered custom metric collector: {collector_func.__name__}")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format.lower() == "json":
            return self._export_json()
        elif format.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_old_metrics(self) -> int:
        """Clean up old metric data points."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        cleaned_count = 0
        
        with self._lock:
            for name, points in self._metrics.items():
                original_size = len(points)
                
                # Filter out old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old metric points")
        
        return cleaned_count
    
    # Private methods
    
    def _register_builtin_collectors(self) -> None:
        """Register built-in metric collectors."""
        
        def collect_system_metrics() -> Dict[str, float]:
            """Collect system-level metrics."""
            import psutil
            
            return {
                "system_cpu_percent": psutil.cpu_percent(),
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_disk_percent": psutil.disk_usage('/').percent
            }
        
        def collect_performance_monitor_metrics() -> Dict[str, float]:
            """Collect metrics from performance monitor."""
            summary = self.perf_monitor.get_performance_summary(1)  # Last hour
            
            metrics = {}
            if not summary.get("no_data", False):
                metrics["total_operations"] = summary.get("total_operations", 0)
                metrics["success_rate"] = summary.get("success_rate", 0.0)
                metrics["avg_duration"] = summary.get("avg_duration", 0.0)
            
            return metrics
        
        # Register collectors (with error handling)
        try:
            self.register_collector(collect_system_metrics)
        except ImportError:
            logger.warning("psutil not available, skipping system metrics collection")
        
        self.register_collector(collect_performance_monitor_metrics)
    
    def _aggregation_loop(self) -> None:
        """Background thread for metric aggregation."""
        while True:
            try:
                time.sleep(self.aggregation_interval)
                self._aggregate_metrics()
                self._collect_custom_metrics()
                self.cleanup_old_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
    
    def _aggregate_metrics(self) -> None:
        """Aggregate metrics for efficient querying."""
        current_time = time.time()
        
        with self._lock:
            for name in self._metrics.keys():
                summary = self.get_metric_summary(name, hours=1)
                if summary:
                    self._aggregated_metrics[name].append(summary)
                    
                    # Keep only recent aggregations
                    cutoff_time = current_time - (self.retention_hours * 3600)
                    self._aggregated_metrics[name] = [
                        s for s in self._aggregated_metrics[name] 
                        if s.count > 0  # Keep valid summaries
                    ][-100:]  # Keep last 100 aggregations
        
        self._last_aggregation = current_time
    
    def _collect_custom_metrics(self) -> None:
        """Collect metrics from registered collectors."""
        for collector in self._collectors:
            try:
                metrics = collector()
                for name, value in metrics.items():
                    self.set_gauge(name, value, {"source": "collector"})
                    
            except Exception as e:
                logger.error(f"Error in custom metric collector {collector.__name__}: {e}")
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]
    
    def _calculate_cache_hit_rate(self) -> Optional[float]:
        """Calculate cache hit rate from metrics."""
        hits_summary = self.get_metric_summary("cache_hits")
        misses_summary = self.get_metric_summary("cache_misses")
        
        if hits_summary and misses_summary:
            total_requests = hits_summary.sum + misses_summary.sum
            if total_requests > 0:
                return hits_summary.sum / total_requests
        
        return None
    
    def _get_overall_summary(self, hours: int) -> Dict[str, Any]:
        """Get overall system performance summary."""
        all_summaries = self.get_all_metrics_summary(hours)
        
        if not all_summaries:
            return {"no_data": True}
        
        # Calculate overall statistics
        total_operations = sum(s.count for s in all_summaries.values())
        avg_response_time = statistics.mean([s.mean for s in all_summaries.values() if s.mean > 0])
        
        return {
            "total_metrics": len(all_summaries),
            "total_operations": total_operations,
            "avg_response_time": avg_response_time,
            "slowest_operation": max(all_summaries.values(), key=lambda s: s.mean).name if all_summaries else None
        }
    
    def _generate_performance_alerts(self) -> List[str]:
        """Generate performance alerts based on metrics."""
        alerts = []
        
        # Check for slow operations
        all_summaries = self.get_all_metrics_summary(1)
        for name, summary in all_summaries.items():
            if "time" in name.lower() and summary.mean > 5.0:  # 5 second threshold
                alerts.append(f"Slow operation detected: {name} avg={summary.mean:.2f}s")
        
        # Check cache hit rate
        cache_hit_rate = self._calculate_cache_hit_rate()
        if cache_hit_rate is not None and cache_hit_rate < 0.5:
            alerts.append(f"Low cache hit rate: {cache_hit_rate:.1%}")
        
        # Check memory usage
        memory_metrics = self.get_memory_metrics()
        for metric_name, metric_data in memory_metrics.items():
            if metric_data.get("mean", 0) > 0.8:  # 80% threshold
                alerts.append(f"High memory usage in {metric_name}: {metric_data['mean']:.1%}")
        
        return alerts
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze routing performance
        routing_metrics = self.get_routing_metrics()
        if routing_metrics.get("intent_classification", {}).get("mean", 0) > 1.0:
            recommendations.append("Consider optimizing intent classification - average time > 1s")
        
        # Analyze cart performance
        cart_metrics = self.get_cart_metrics()
        for operation, data in cart_metrics.items():
            if "performance" in operation and data.get("mean", 0) > 2.0:
                recommendations.append(f"Optimize {operation} - average time > 2s")
        
        # Analyze database performance
        db_metrics = self.get_database_metrics()
        if db_metrics.get("query_performance", {}).get("mean", 0) > 1.0:
            recommendations.append("Consider database query optimization - average time > 1s")
        
        return recommendations
    
    def _export_json(self) -> str:
        """Export metrics in JSON format."""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {}
        }
        
        with self._lock:
            for name, points in self._metrics.items():
                export_data["metrics"][name] = [p.to_dict() for p in points]
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for name, points in self._metrics.items():
                if not points:
                    continue
                
                latest_point = points[-1]
                metric_name = name.replace(".", "_").replace("-", "_")
                
                # Add help and type comments
                lines.append(f"# HELP {metric_name} {name}")
                lines.append(f"# TYPE {metric_name} {latest_point.metric_type.value}")
                
                # Add metric value
                tags_str = ""
                if latest_point.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in latest_point.tags.items()]
                    tags_str = "{" + ",".join(tag_pairs) + "}"
                
                lines.append(f"{metric_name}{tags_str} {latest_point.value} {int(latest_point.timestamp * 1000)}")
        
        return "\n".join(lines)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Convenience function to record a metric."""
    collector = get_metrics_collector()
    collector.record_metric(name, value, tags)