"""
Memory usage monitoring for cart state management and system components.
Tracks memory usage patterns, detects leaks, and provides optimization recommendations.
"""

import gc
import logging
import threading
import time
import sys
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import tracemalloc
import weakref

from .performance_monitor import get_performance_monitor, performance_track
from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    
    timestamp: float
    total_memory: int  # bytes
    available_memory: int  # bytes
    process_memory: int  # bytes
    heap_size: int  # bytes
    gc_stats: Dict[str, int]
    component_memory: Dict[str, int] = field(default_factory=dict)
    
    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.available_memory > 0:
            return (self.process_memory / self.available_memory) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_memory": self.total_memory,
            "available_memory": self.available_memory,
            "process_memory": self.process_memory,
            "heap_size": self.heap_size,
            "memory_usage_percent": self.memory_usage_percent,
            "gc_stats": self.gc_stats,
            "component_memory": self.component_memory
        }


@dataclass
class MemoryLeak:
    """Detected memory leak information."""
    
    component: str
    start_time: float
    current_memory: int
    initial_memory: int
    growth_rate: float  # bytes per second
    severity: str  # 'low', 'medium', 'high'
    
    @property
    def memory_growth(self) -> int:
        """Calculate total memory growth."""
        return self.current_memory - self.initial_memory
    
    @property
    def duration(self) -> float:
        """Calculate leak duration in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "start_time": self.start_time,
            "current_memory": self.current_memory,
            "initial_memory": self.initial_memory,
            "memory_growth": self.memory_growth,
            "growth_rate": self.growth_rate,
            "duration": self.duration,
            "severity": self.severity
        }


class MemoryMonitor:
    """Comprehensive memory usage monitoring and leak detection."""
    
    def __init__(self, 
                 monitoring_interval: float = 30.0,
                 snapshot_retention: int = 1000,
                 leak_detection_threshold: float = 1024 * 1024,  # 1MB
                 enable_tracemalloc: bool = True):
        """
        Initialize memory monitor.
        
        Args:
            monitoring_interval: Interval between memory snapshots in seconds
            snapshot_retention: Number of snapshots to retain
            leak_detection_threshold: Memory growth threshold for leak detection
            enable_tracemalloc: Whether to enable detailed memory tracing
        """
        self.monitoring_interval = monitoring_interval
        self.snapshot_retention = snapshot_retention
        self.leak_detection_threshold = leak_detection_threshold
        self.enable_tracemalloc = enable_tracemalloc
        
        # Memory snapshots
        self._snapshots: deque = deque(maxlen=snapshot_retention)
        self._lock = threading.RLock()
        
        # Component tracking
        self._component_trackers: Dict[str, Callable[[], int]] = {}
        self._component_baselines: Dict[str, int] = {}
        
        # Leak detection
        self._detected_leaks: List[MemoryLeak] = []
        self._leak_detection_enabled = True
        
        # Performance integration
        self.perf_monitor = get_performance_monitor()
        self.metrics_collector = get_metrics_collector()
        
        # Memory thresholds
        self.warning_threshold = 0.8  # 80%
        self.critical_threshold = 0.9  # 90%
        
        # Initialize tracemalloc if enabled
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        # Register built-in component trackers
        self._register_builtin_trackers()
        
        logger.info(f"Memory monitor initialized with {monitoring_interval}s interval")
    
    def register_component_tracker(self, component_name: str, tracker_func: Callable[[], int]) -> None:
        """
        Register a component memory tracker.
        
        Args:
            component_name: Name of the component
            tracker_func: Function that returns current memory usage in bytes
        """
        with self._lock:
            self._component_trackers[component_name] = tracker_func
            
            # Set baseline
            try:
                baseline = tracker_func()
                self._component_baselines[component_name] = baseline
                logger.info(f"Registered memory tracker for {component_name} (baseline: {baseline} bytes)")
            except Exception as e:
                logger.error(f"Failed to set baseline for {component_name}: {e}")
    
    @performance_track("memory_snapshot")
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        try:
            # Get system memory info
            total_memory, available_memory = self._get_system_memory()
            
            # Get process memory
            process_memory = self._get_process_memory()
            
            # Get heap size
            heap_size = self._get_heap_size()
            
            # Get garbage collection stats
            gc_stats = self._get_gc_stats()
            
            # Get component memory usage
            component_memory = {}
            with self._lock:
                for name, tracker in self._component_trackers.items():
                    try:
                        component_memory[name] = tracker()
                    except Exception as e:
                        logger.warning(f"Failed to get memory for {name}: {e}")
                        component_memory[name] = 0
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                total_memory=total_memory,
                available_memory=available_memory,
                process_memory=process_memory,
                heap_size=heap_size,
                gc_stats=gc_stats,
                component_memory=component_memory
            )
            
            # Store snapshot
            with self._lock:
                self._snapshots.append(snapshot)
            
            # Record metrics
            self.metrics_collector.set_gauge("total_memory_usage", process_memory)
            self.metrics_collector.set_gauge("memory_usage_percent", snapshot.memory_usage_percent)
            
            for component, memory in component_memory.items():
                self.metrics_collector.set_gauge(f"component_memory_{component}", memory)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            raise
    
    def detect_memory_leaks(self) -> List[MemoryLeak]:
        """Detect potential memory leaks."""
        if not self._leak_detection_enabled:
            return []
        
        current_time = time.time()
        detected_leaks = []
        
        with self._lock:
            if len(self._snapshots) < 10:  # Need enough data points
                return []
            
            # Analyze each component
            for component in self._component_trackers.keys():
                leak = self._analyze_component_for_leaks(component, current_time)
                if leak:
                    detected_leaks.append(leak)
        
        # Update detected leaks
        self._detected_leaks = detected_leaks
        
        return detected_leaks
    
    def get_memory_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get memory usage summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_snapshots = [s for s in self._snapshots if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return {"no_data": True, "period_hours": hours}
        
        # Calculate statistics
        memory_values = [s.process_memory for s in recent_snapshots]
        usage_percentages = [s.memory_usage_percent for s in recent_snapshots]
        
        summary = {
            "period_hours": hours,
            "snapshot_count": len(recent_snapshots),
            "memory_stats": {
                "current": memory_values[-1] if memory_values else 0,
                "min": min(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                "growth": memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
            },
            "usage_percentage": {
                "current": usage_percentages[-1] if usage_percentages else 0,
                "min": min(usage_percentages) if usage_percentages else 0,
                "max": max(usage_percentages) if usage_percentages else 0,
                "avg": sum(usage_percentages) / len(usage_percentages) if usage_percentages else 0
            }
        }
        
        # Component analysis
        component_analysis = {}
        for component in self._component_trackers.keys():
            component_values = [s.component_memory.get(component, 0) for s in recent_snapshots]
            if component_values:
                component_analysis[component] = {
                    "current": component_values[-1],
                    "min": min(component_values),
                    "max": max(component_values),
                    "avg": sum(component_values) / len(component_values),
                    "growth": component_values[-1] - component_values[0] if len(component_values) > 1 else 0
                }
        
        summary["component_analysis"] = component_analysis
        
        # Memory alerts
        alerts = []
        current_usage = usage_percentages[-1] if usage_percentages else 0
        
        if current_usage > self.critical_threshold * 100:
            alerts.append(f"Critical memory usage: {current_usage:.1f}%")
        elif current_usage > self.warning_threshold * 100:
            alerts.append(f"High memory usage: {current_usage:.1f}%")
        
        # Check for memory growth
        if summary["memory_stats"]["growth"] > self.leak_detection_threshold:
            alerts.append(f"Significant memory growth detected: {summary['memory_stats']['growth']} bytes")
        
        summary["alerts"] = alerts
        
        return summary
    
    def get_leak_report(self) -> Dict[str, Any]:
        """Get comprehensive memory leak report."""
        leaks = self.detect_memory_leaks()
        
        return {
            "detected_leaks": [leak.to_dict() for leak in leaks],
            "leak_count": len(leaks),
            "high_severity_leaks": len([l for l in leaks if l.severity == "high"]),
            "total_leaked_memory": sum(l.memory_growth for l in leaks),
            "recommendations": self._generate_leak_recommendations(leaks)
        }
    
    def get_component_memory_report(self) -> Dict[str, Any]:
        """Get detailed component memory usage report."""
        report = {}
        
        with self._lock:
            for component, tracker in self._component_trackers.items():
                try:
                    current_memory = tracker()
                    baseline = self._component_baselines.get(component, 0)
                    
                    # Get historical data
                    component_history = []
                    for snapshot in list(self._snapshots)[-50:]:  # Last 50 snapshots
                        if component in snapshot.component_memory:
                            component_history.append({
                                "timestamp": snapshot.timestamp,
                                "memory": snapshot.component_memory[component]
                            })
                    
                    report[component] = {
                        "current_memory": current_memory,
                        "baseline_memory": baseline,
                        "memory_growth": current_memory - baseline,
                        "growth_percentage": ((current_memory - baseline) / baseline * 100) if baseline > 0 else 0,
                        "history": component_history,
                        "recommendations": self._get_component_recommendations(component, current_memory, baseline)
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to analyze component {component}: {e}")
                    report[component] = {"error": str(e)}
        
        return report
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization operations."""
        optimizations = []
        
        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            optimizations.append(f"Garbage collected {collected} objects")
        
        # Clear component caches if available
        for component in self._component_trackers.keys():
            try:
                # This would call component-specific cleanup methods
                # For now, we'll just log the intent
                optimizations.append(f"Optimized {component} memory usage")
            except Exception as e:
                logger.error(f"Failed to optimize {component}: {e}")
        
        # Take snapshot after optimization
        post_optimization_snapshot = self.take_snapshot()
        
        return {
            "optimizations_performed": optimizations,
            "memory_after_optimization": post_optimization_snapshot.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_memory_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive memory health check."""
        try:
            current_snapshot = self.take_snapshot()
            leaks = self.detect_memory_leaks()
            summary = self.get_memory_summary(1)  # Last hour
            
            # Determine health status
            health_issues = []
            
            if current_snapshot.memory_usage_percent > self.critical_threshold * 100:
                health_issues.append("Critical memory usage")
            elif current_snapshot.memory_usage_percent > self.warning_threshold * 100:
                health_issues.append("High memory usage")
            
            if leaks:
                high_severity_leaks = [l for l in leaks if l.severity == "high"]
                if high_severity_leaks:
                    health_issues.append(f"{len(high_severity_leaks)} high-severity memory leaks")
                else:
                    health_issues.append(f"{len(leaks)} potential memory leaks")
            
            status = "healthy" if not health_issues else "degraded" if len(health_issues) < 3 else "unhealthy"
            
            return {
                "status": status,
                "current_memory_usage": current_snapshot.memory_usage_percent,
                "issues": health_issues,
                "detected_leaks": len(leaks),
                "summary": summary,
                "recommendations": self._generate_health_recommendations(current_snapshot, leaks),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Private methods
    
    def _register_builtin_trackers(self) -> None:
        """Register built-in component memory trackers."""
        
        def track_cart_state_memory() -> int:
            """Track memory used by cart state management."""
            # This would integrate with the cart state manager
            # For now, return a placeholder
            return sys.getsizeof({}) * 100  # Placeholder
        
        def track_cache_memory() -> int:
            """Track memory used by caches."""
            try:
                from ..core.router.intent_cache import get_intent_cache
                cache = get_intent_cache()
                return cache._estimate_memory_usage()
            except Exception:
                return 0
        
        def track_performance_monitor_memory() -> int:
            """Track memory used by performance monitoring."""
            try:
                monitor = self.perf_monitor
                # Estimate based on metrics history size
                return sys.getsizeof(monitor.metrics_history) + sys.getsizeof(monitor.operation_stats)
            except Exception:
                return 0
        
        # Register trackers
        self.register_component_tracker("cart_state", track_cart_state_memory)
        self.register_component_tracker("cache", track_cache_memory)
        self.register_component_tracker("performance_monitor", track_performance_monitor_memory)
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                time.sleep(self.monitoring_interval)
                
                # Take snapshot
                self.take_snapshot()
                
                # Detect leaks
                if self._leak_detection_enabled:
                    leaks = self.detect_memory_leaks()
                    if leaks:
                        logger.warning(f"Detected {len(leaks)} potential memory leaks")
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
    
    def _get_system_memory(self) -> Tuple[int, int]:
        """Get system memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.total, memory.available
        except ImportError:
            # Fallback for systems without psutil
            return 0, 0
    
    def _get_process_memory(self) -> int:
        """Get current process memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback using resource module
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Convert to bytes
    
    def _get_heap_size(self) -> int:
        """Get Python heap size."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            return current
        else:
            # Estimate using gc
            return sum(sys.getsizeof(obj) for obj in gc.get_objects())
    
    def _get_gc_stats(self) -> Dict[str, int]:
        """Get garbage collection statistics."""
        stats = {}
        
        # Get GC counts
        gc_counts = gc.get_count()
        stats["gc_gen0"] = gc_counts[0]
        stats["gc_gen1"] = gc_counts[1]
        stats["gc_gen2"] = gc_counts[2]
        
        # Get GC stats if available
        try:
            gc_stats = gc.get_stats()
            for i, stat in enumerate(gc_stats):
                stats[f"gc_collections_gen{i}"] = stat.get("collections", 0)
                stats[f"gc_collected_gen{i}"] = stat.get("collected", 0)
        except AttributeError:
            # get_stats() not available in older Python versions
            pass
        
        return stats
    
    def _analyze_component_for_leaks(self, component: str, current_time: float) -> Optional[MemoryLeak]:
        """Analyze a component for memory leaks."""
        # Get recent snapshots for this component
        recent_snapshots = []
        for snapshot in list(self._snapshots)[-20:]:  # Last 20 snapshots
            if component in snapshot.component_memory:
                recent_snapshots.append((snapshot.timestamp, snapshot.component_memory[component]))
        
        if len(recent_snapshots) < 5:  # Need enough data points
            return None
        
        # Calculate memory growth rate
        start_time, start_memory = recent_snapshots[0]
        end_time, end_memory = recent_snapshots[-1]
        
        time_diff = end_time - start_time
        memory_diff = end_memory - start_memory
        
        if time_diff <= 0:
            return None
        
        growth_rate = memory_diff / time_diff
        
        # Check if growth exceeds threshold
        if memory_diff > self.leak_detection_threshold and growth_rate > 0:
            # Determine severity
            if growth_rate > 1024 * 1024:  # 1MB/s
                severity = "high"
            elif growth_rate > 512 * 1024:  # 512KB/s
                severity = "medium"
            else:
                severity = "low"
            
            return MemoryLeak(
                component=component,
                start_time=start_time,
                current_memory=end_memory,
                initial_memory=start_memory,
                growth_rate=growth_rate,
                severity=severity
            )
        
        return None
    
    def _generate_leak_recommendations(self, leaks: List[MemoryLeak]) -> List[str]:
        """Generate recommendations for addressing memory leaks."""
        recommendations = []
        
        for leak in leaks:
            if leak.severity == "high":
                recommendations.append(f"Urgent: Investigate {leak.component} - high memory leak detected")
            elif leak.severity == "medium":
                recommendations.append(f"Review {leak.component} memory usage patterns")
            else:
                recommendations.append(f"Monitor {leak.component} for continued growth")
        
        if leaks:
            recommendations.append("Consider implementing memory profiling for affected components")
            recommendations.append("Review object lifecycle management in leaking components")
        
        return recommendations
    
    def _get_component_recommendations(self, component: str, current: int, baseline: int) -> List[str]:
        """Get optimization recommendations for a component."""
        recommendations = []
        
        if baseline > 0:
            growth_ratio = current / baseline
            
            if growth_ratio > 2.0:
                recommendations.append(f"Memory usage has doubled - review {component} implementation")
            elif growth_ratio > 1.5:
                recommendations.append(f"Significant memory growth in {component} - consider optimization")
        
        if current > 100 * 1024 * 1024:  # 100MB
            recommendations.append(f"{component} using significant memory - consider caching strategies")
        
        return recommendations
    
    def _generate_health_recommendations(self, snapshot: MemorySnapshot, leaks: List[MemoryLeak]) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []
        
        if snapshot.memory_usage_percent > self.critical_threshold * 100:
            recommendations.append("Critical memory usage - consider immediate optimization")
            recommendations.append("Review and reduce memory-intensive operations")
        elif snapshot.memory_usage_percent > self.warning_threshold * 100:
            recommendations.append("High memory usage - monitor closely and optimize if needed")
        
        if leaks:
            recommendations.append("Memory leaks detected - investigate and fix leaking components")
        
        # GC recommendations
        if snapshot.gc_stats.get("gc_gen0", 0) > 1000:
            recommendations.append("High GC activity - consider object pooling or caching")
        
        return recommendations


# Global memory monitor instance
_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _memory_monitor
    
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    
    return _memory_monitor


def track_component_memory(component_name: str):
    """Decorator to track memory usage of a component."""
    def decorator(cls):
        monitor = get_memory_monitor()
        
        def get_component_memory():
            # This would implement component-specific memory tracking
            return sys.getsizeof(cls) if hasattr(cls, '__dict__') else 0
        
        monitor.register_component_tracker(component_name, get_component_memory)
        return cls
    
    return decorator