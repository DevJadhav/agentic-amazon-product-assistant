"""
Performance monitoring and optimization for LangGraph agents.
Tracks performance metrics and provides optimization recommendations.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for agent operations."""
    
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark operation as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class PerformanceMonitor:
    """Monitors and optimizes agent performance."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor."""
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_calls": 0,
            "total_duration": 0.0,
            "success_count": 0,
            "error_count": 0,
            "avg_duration": 0.0,
            "min_duration": float('inf'),
            "max_duration": 0.0,
            "recent_durations": deque(maxlen=100)
        })
        
        # Performance thresholds
        self.thresholds = {
            "slow_operation": 5.0,  # seconds
            "very_slow_operation": 10.0,  # seconds
            "high_error_rate": 0.1,  # 10%
            "memory_warning": 0.8,  # 80% of available memory
        }
        
        # Cache for optimization
        self._cache = {}
        self._cache_ttl = {}
        self._cache_max_size = 1000
        self._cache_default_ttl = 300  # 5 minutes
    
    def start_operation(self, operation_name: str, metadata: Dict[str, Any] = None) -> PerformanceMetrics:
        """Start tracking an operation."""
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        return metrics
    
    def finish_operation(self, metrics: PerformanceMetrics, success: bool = True, error_message: Optional[str] = None):
        """Finish tracking an operation."""
        
        metrics.finish(success, error_message)
        
        # Update statistics
        self._update_operation_stats(metrics)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Log slow operations
        if metrics.duration and metrics.duration > self.thresholds["slow_operation"]:
            logger.warning(f"Slow operation detected: {metrics.operation_name} took {metrics.duration:.2f}s")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.start_time >= cutoff_time]
        
        if not recent_metrics:
            return {"no_data": True, "period_hours": hours}
        
        # Calculate overall statistics
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        failed_operations = total_operations - successful_operations
        
        durations = [m.duration for m in recent_metrics if m.duration is not None]
        
        summary = {
            "period_hours": hours,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
            "error_rate": failed_operations / total_operations if total_operations > 0 else 0,
        }
        
        if durations:
            summary.update({
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations)
            })
        
        # Operation breakdown
        operation_breakdown = defaultdict(lambda: {"count": 0, "avg_duration": 0, "success_rate": 0})
        
        for metrics in recent_metrics:
            op_name = metrics.operation_name
            operation_breakdown[op_name]["count"] += 1
            
            if metrics.duration:
                current_avg = operation_breakdown[op_name]["avg_duration"]
                current_count = operation_breakdown[op_name]["count"]
                operation_breakdown[op_name]["avg_duration"] = (
                    (current_avg * (current_count - 1) + metrics.duration) / current_count
                )
            
            if metrics.success:
                operation_breakdown[op_name]["success_rate"] = (
                    operation_breakdown[op_name].get("success_count", 0) + 1
                ) / operation_breakdown[op_name]["count"]
                operation_breakdown[op_name]["success_count"] = operation_breakdown[op_name].get("success_count", 0) + 1
        
        summary["operations"] = dict(operation_breakdown)
        
        # Performance alerts
        alerts = []
        
        if summary["error_rate"] > self.thresholds["high_error_rate"]:
            alerts.append(f"High error rate: {summary['error_rate']:.1%}")
        
        if durations and summary["avg_duration"] > self.thresholds["slow_operation"]:
            alerts.append(f"Slow average response time: {summary['avg_duration']:.2f}s")
        
        summary["alerts"] = alerts
        
        return summary
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific operation."""
        
        if operation_name not in self.operation_stats:
            return {"operation_name": operation_name, "no_data": True}
        
        stats = self.operation_stats[operation_name].copy()
        stats["operation_name"] = operation_name
        
        # Calculate percentiles from recent durations
        recent_durations = list(stats["recent_durations"])
        if recent_durations:
            recent_durations.sort()
            n = len(recent_durations)
            
            stats["p50_duration"] = recent_durations[n // 2]
            stats["p90_duration"] = recent_durations[int(n * 0.9)]
            stats["p95_duration"] = recent_durations[int(n * 0.95)]
        
        return stats
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        
        recommendations = []
        
        # Check for slow operations
        for op_name, stats in self.operation_stats.items():
            if stats["avg_duration"] > self.thresholds["slow_operation"]:
                recommendations.append(
                    f"Optimize '{op_name}' operation - average duration: {stats['avg_duration']:.2f}s"
                )
            
            if stats["error_count"] > 0:
                error_rate = stats["error_count"] / stats["total_calls"]
                if error_rate > self.thresholds["high_error_rate"]:
                    recommendations.append(
                        f"Investigate errors in '{op_name}' operation - error rate: {error_rate:.1%}"
                    )
        
        # Check cache performance
        cache_hit_rate = self._calculate_cache_hit_rate()
        if cache_hit_rate < 0.5:  # Less than 50% hit rate
            recommendations.append(
                f"Consider optimizing caching strategy - current hit rate: {cache_hit_rate:.1%}"
            )
        
        # Check for memory usage patterns
        if len(self.metrics_history) >= self.max_history * 0.9:
            recommendations.append(
                "Consider increasing metrics history size or implementing data archiving"
            )
        
        return recommendations
    
    def cache_get(self, key: str) -> Any:
        """Get value from performance cache."""
        
        if key not in self._cache:
            return None
        
        # Check TTL
        if key in self._cache_ttl and time.time() > self._cache_ttl[key]:
            del self._cache[key]
            del self._cache_ttl[key]
            return None
        
        return self._cache[key]
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in performance cache."""
        
        # Implement simple LRU eviction
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entries
            oldest_keys = list(self._cache.keys())[:self._cache_max_size // 4]
            for old_key in oldest_keys:
                self._cache.pop(old_key, None)
                self._cache_ttl.pop(old_key, None)
        
        self._cache[key] = value
        
        if ttl is not None:
            self._cache_ttl[key] = time.time() + ttl
        else:
            self._cache_ttl[key] = time.time() + self._cache_default_ttl
    
    def clear_cache(self) -> None:
        """Clear performance cache."""
        self._cache.clear()
        self._cache_ttl.clear()
    
    def performance_decorator(self, operation_name: str = None):
        """Decorator to automatically track function performance."""
        
        def decorator(func):
            nonlocal operation_name
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    metrics = self.start_operation(operation_name)
                    try:
                        result = await func(*args, **kwargs)
                        self.finish_operation(metrics, success=True)
                        return result
                    except Exception as e:
                        self.finish_operation(metrics, success=False, error_message=str(e))
                        raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    metrics = self.start_operation(operation_name)
                    try:
                        result = func(*args, **kwargs)
                        self.finish_operation(metrics, success=True)
                        return result
                    except Exception as e:
                        self.finish_operation(metrics, success=False, error_message=str(e))
                        raise
                
                return sync_wrapper
        
        return decorator
    
    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.metrics_history.clear()
        self.operation_stats.clear()
        self.clear_cache()
    
    # Private helper methods
    
    def _update_operation_stats(self, metrics: PerformanceMetrics) -> None:
        """Update operation statistics."""
        
        op_name = metrics.operation_name
        stats = self.operation_stats[op_name]
        
        stats["total_calls"] += 1
        
        if metrics.success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1
        
        if metrics.duration is not None:
            stats["total_duration"] += metrics.duration
            stats["avg_duration"] = stats["total_duration"] / stats["total_calls"]
            stats["min_duration"] = min(stats["min_duration"], metrics.duration)
            stats["max_duration"] = max(stats["max_duration"], metrics.duration)
            stats["recent_durations"].append(metrics.duration)
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        
        # This is a simplified calculation
        # In a real implementation, you'd track cache hits/misses
        
        if not self._cache:
            return 0.0
        
        # Estimate based on cache size vs theoretical maximum
        return min(len(self._cache) / self._cache_max_size, 1.0)


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


def performance_track(operation_name: str = None):
    """Decorator to track function performance."""
    monitor = get_performance_monitor()
    return monitor.performance_decorator(operation_name)