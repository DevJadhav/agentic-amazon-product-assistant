"""
Optimized database connection pooling for cart operations and state management.
Implements intelligent connection management with monitoring and optimization.
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass
from queue import Queue, Empty
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
import psycopg2.extensions

from ..monitoring.performance_monitor import get_performance_monitor, performance_track

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for a database connection."""
    
    connection_id: str
    created_at: float
    last_used: float
    total_queries: int
    total_time: float
    active_time: float
    error_count: int
    is_active: bool
    
    @property
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used if not self.is_active else 0.0
    
    @property
    def avg_query_time(self) -> float:
        """Get average query time."""
        return self.total_time / self.total_queries if self.total_queries > 0 else 0.0


class OptimizedConnectionPool:
    """Optimized database connection pool with monitoring and auto-tuning."""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 user: str = None,
                 password: str = None,
                 min_connections: int = None,
                 max_connections: int = None,
                 connection_timeout: float = 30.0,
                 idle_timeout: float = 300.0,  # 5 minutes
                 max_connection_age: float = 3600.0,  # 1 hour
                 enable_monitoring: bool = True):
        """
        Initialize optimized connection pool.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            connection_timeout: Timeout for getting connection from pool
            idle_timeout: Timeout for idle connections
            max_connection_age: Maximum age for connections before refresh
            enable_monitoring: Whether to enable connection monitoring
        """
        # Database configuration
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "langgraph_assistant")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Pool configuration
        self.min_connections = min_connections or int(os.getenv("POSTGRES_MIN_CONN", "2"))
        self.max_connections = max_connections or int(os.getenv("POSTGRES_MAX_CONN", "20"))
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.max_connection_age = max_connection_age
        self.enable_monitoring = enable_monitoring
        
        # Connection pool
        self.pool: Optional[ThreadedConnectionPool] = None
        self._pool_lock = threading.RLock()
        
        # Connection monitoring
        self._connection_stats: Dict[str, ConnectionStats] = {}
        self._stats_lock = threading.RLock()
        
        # Pool statistics
        self._pool_stats = {
            "total_connections_created": 0,
            "total_connections_closed": 0,
            "total_queries_executed": 0,
            "total_query_time": 0.0,
            "connection_errors": 0,
            "pool_exhaustions": 0,
            "last_optimization": time.time()
        }
        
        # Performance monitor
        self.perf_monitor = get_performance_monitor()
        
        # Auto-optimization settings
        self.auto_optimize = True
        self.optimization_interval = 300  # 5 minutes
        
        # Initialize pool
        self._initialize_pool()
        
        # Start monitoring thread
        if self.enable_monitoring:
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        try:
            with self._pool_lock:
                if self.pool:
                    self.pool.closeall()
                
                self.pool = ThreadedConnectionPool(
                    self.min_connections,
                    self.max_connections,
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    cursor_factory=RealDictCursor,
                    # Connection-level optimizations
                    connect_timeout=10,
                    application_name="langgraph_assistant_optimized"
                )
                
                logger.info(f"Initialized connection pool: {self.min_connections}-{self.max_connections} connections")
                
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    @performance_track("db_connection_get")
    def get_connection(self):
        """Get optimized database connection from pool."""
        connection = None
        connection_id = None
        start_time = time.time()
        
        try:
            with self._pool_lock:
                if not self.pool:
                    raise RuntimeError("Connection pool not initialized")
                
                try:
                    connection = self.pool.getconn()
                    connection_id = self._get_connection_id(connection)
                    
                    # Update connection stats
                    if self.enable_monitoring:
                        self._update_connection_stats(connection_id, start_time, is_active=True)
                    
                except Exception as e:
                    self._pool_stats["pool_exhaustions"] += 1
                    logger.warning(f"Failed to get connection from pool: {e}")
                    raise
            
            # Configure connection for optimal performance
            self._optimize_connection(connection)
            
            yield connection
            
        except Exception as e:
            self._pool_stats["connection_errors"] += 1
            logger.error(f"Connection error: {e}")
            raise
            
        finally:
            if connection and self.pool:
                try:
                    # Update stats before returning connection
                    if self.enable_monitoring and connection_id:
                        self._update_connection_stats(connection_id, start_time, is_active=False)
                    
                    with self._pool_lock:
                        self.pool.putconn(connection)
                        
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
    
    @performance_track("db_query_execute")
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute optimized SELECT query."""
        start_time = time.time()
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query, params)
                    results = [dict(row) for row in cursor.fetchall()]
                    
                    # Update query stats
                    query_time = time.time() - start_time
                    self._pool_stats["total_queries_executed"] += 1
                    self._pool_stats["total_query_time"] += query_time
                    
                    return results
                    
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    raise
    
    @performance_track("db_update_execute")
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute optimized INSERT/UPDATE/DELETE query."""
        start_time = time.time()
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query, params)
                    conn.commit()
                    affected_rows = cursor.rowcount
                    
                    # Update query stats
                    query_time = time.time() - start_time
                    self._pool_stats["total_queries_executed"] += 1
                    self._pool_stats["total_query_time"] += query_time
                    
                    return affected_rows
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Update execution failed: {e}")
                    raise
    
    def execute_batch(self, queries: List[Tuple[str, tuple]]) -> List[Any]:
        """Execute batch of queries in a single transaction."""
        start_time = time.time()
        results = []
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    for query, params in queries:
                        cursor.execute(query, params)
                        
                        # Collect results for SELECT queries
                        if query.strip().upper().startswith('SELECT'):
                            results.append([dict(row) for row in cursor.fetchall()])
                        else:
                            results.append(cursor.rowcount)
                    
                    conn.commit()
                    
                    # Update stats
                    query_time = time.time() - start_time
                    self._pool_stats["total_queries_executed"] += len(queries)
                    self._pool_stats["total_query_time"] += query_time
                    
                    return results
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Batch execution failed: {e}")
                    raise
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._stats_lock:
            active_connections = 0
            idle_connections = 0
            total_connection_age = 0.0
            total_idle_time = 0.0
            
            for stats in self._connection_stats.values():
                if stats.is_active:
                    active_connections += 1
                else:
                    idle_connections += 1
                
                total_connection_age += stats.age
                total_idle_time += stats.idle_time
            
            total_connections = len(self._connection_stats)
            avg_connection_age = total_connection_age / total_connections if total_connections > 0 else 0.0
            avg_idle_time = total_idle_time / idle_connections if idle_connections > 0 else 0.0
            
            avg_query_time = (
                self._pool_stats["total_query_time"] / self._pool_stats["total_queries_executed"]
                if self._pool_stats["total_queries_executed"] > 0 else 0.0
            )
            
            return {
                "pool_config": {
                    "min_connections": self.min_connections,
                    "max_connections": self.max_connections,
                    "connection_timeout": self.connection_timeout,
                    "idle_timeout": self.idle_timeout,
                    "max_connection_age": self.max_connection_age
                },
                "current_state": {
                    "total_connections": total_connections,
                    "active_connections": active_connections,
                    "idle_connections": idle_connections,
                    "avg_connection_age": avg_connection_age,
                    "avg_idle_time": avg_idle_time
                },
                "performance": {
                    "total_queries": self._pool_stats["total_queries_executed"],
                    "avg_query_time": avg_query_time,
                    "total_query_time": self._pool_stats["total_query_time"],
                    "queries_per_second": self._calculate_qps()
                },
                "errors": {
                    "connection_errors": self._pool_stats["connection_errors"],
                    "pool_exhaustions": self._pool_stats["pool_exhaustions"]
                },
                "optimization": {
                    "last_optimization": datetime.fromtimestamp(
                        self._pool_stats["last_optimization"]
                    ).isoformat(),
                    "auto_optimize_enabled": self.auto_optimize
                }
            }
    
    def optimize_pool(self) -> Dict[str, Any]:
        """Optimize pool configuration based on usage patterns."""
        stats = self.get_pool_stats()
        optimizations = []
        
        # Analyze connection usage
        active_ratio = stats["current_state"]["active_connections"] / self.max_connections
        avg_query_time = stats["performance"]["avg_query_time"]
        pool_exhaustions = stats["errors"]["pool_exhaustions"]
        
        # Optimize pool size
        if pool_exhaustions > 0 and active_ratio > 0.8:
            new_max = min(self.max_connections + 5, 50)  # Cap at 50
            optimizations.append(f"Increase max_connections from {self.max_connections} to {new_max}")
            self.max_connections = new_max
        
        elif active_ratio < 0.3 and self.max_connections > self.min_connections + 5:
            new_max = max(self.max_connections - 2, self.min_connections + 2)
            optimizations.append(f"Decrease max_connections from {self.max_connections} to {new_max}")
            self.max_connections = new_max
        
        # Optimize timeouts based on query performance
        if avg_query_time > 5.0:  # Slow queries
            if self.connection_timeout < 60:
                self.connection_timeout = min(self.connection_timeout * 1.5, 60)
                optimizations.append(f"Increase connection_timeout to {self.connection_timeout}")
        
        # Clean up old connections
        cleaned_connections = self._cleanup_old_connections()
        if cleaned_connections > 0:
            optimizations.append(f"Cleaned up {cleaned_connections} old connections")
        
        # Reinitialize pool if significant changes were made
        if any("connections" in opt for opt in optimizations):
            self._initialize_pool()
            optimizations.append("Reinitialized connection pool with new settings")
        
        self._pool_stats["last_optimization"] = time.time()
        
        logger.info(f"Pool optimization completed: {optimizations}")
        
        return {
            "optimizations_applied": optimizations,
            "new_stats": self.get_pool_stats()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive pool health check."""
        try:
            start_time = time.time()
            
            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 as health_check")
                    result = cursor.fetchone()
            
            response_time = time.time() - start_time
            
            stats = self.get_pool_stats()
            
            # Determine health status
            health_issues = []
            
            if response_time > 5.0:
                health_issues.append(f"Slow response time: {response_time:.2f}s")
            
            if stats["errors"]["pool_exhaustions"] > 10:
                health_issues.append("High pool exhaustion rate")
            
            if stats["current_state"]["active_connections"] / self.max_connections > 0.9:
                health_issues.append("Pool near capacity")
            
            status = "healthy" if not health_issues else "degraded"
            
            return {
                "status": status,
                "response_time": response_time,
                "issues": health_issues,
                "stats": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pool health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def close(self) -> None:
        """Close connection pool and cleanup resources."""
        with self._pool_lock:
            if self.pool:
                self.pool.closeall()
                self.pool = None
        
        with self._stats_lock:
            self._connection_stats.clear()
        
        logger.info("Connection pool closed")
    
    # Private methods
    
    def _get_connection_id(self, connection) -> str:
        """Get unique identifier for connection."""
        return f"conn_{id(connection)}"
    
    def _update_connection_stats(self, connection_id: str, start_time: float, is_active: bool) -> None:
        """Update connection statistics."""
        with self._stats_lock:
            if connection_id not in self._connection_stats:
                self._connection_stats[connection_id] = ConnectionStats(
                    connection_id=connection_id,
                    created_at=start_time,
                    last_used=start_time,
                    total_queries=0,
                    total_time=0.0,
                    active_time=0.0,
                    error_count=0,
                    is_active=is_active
                )
            else:
                stats = self._connection_stats[connection_id]
                stats.last_used = time.time()
                stats.is_active = is_active
                
                if not is_active:  # Connection returned to pool
                    query_time = time.time() - start_time
                    stats.total_queries += 1
                    stats.total_time += query_time
                    stats.active_time += query_time
    
    def _optimize_connection(self, connection) -> None:
        """Apply connection-level optimizations."""
        try:
            # Set optimal isolation level for read operations
            connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED)
            
            # Enable autocommit for single queries (disabled for transactions)
            # connection.autocommit = True  # Commented out to maintain transaction control
            
        except Exception as e:
            logger.warning(f"Failed to optimize connection: {e}")
    
    def _cleanup_old_connections(self) -> int:
        """Clean up old and idle connections."""
        cleaned_count = 0
        current_time = time.time()
        
        with self._stats_lock:
            expired_connections = []
            
            for conn_id, stats in self._connection_stats.items():
                # Remove stats for very old connections
                if stats.age > self.max_connection_age * 2:
                    expired_connections.append(conn_id)
                # Remove stats for long-idle connections
                elif stats.idle_time > self.idle_timeout * 2:
                    expired_connections.append(conn_id)
            
            for conn_id in expired_connections:
                del self._connection_stats[conn_id]
                cleaned_count += 1
        
        return cleaned_count
    
    def _calculate_qps(self) -> float:
        """Calculate queries per second."""
        # Simple calculation based on recent activity
        # In a real implementation, this would use a sliding window
        total_queries = self._pool_stats["total_queries_executed"]
        if total_queries == 0:
            return 0.0
        
        # Estimate based on last hour of activity
        return total_queries / 3600.0  # Simplified calculation
    
    def _monitoring_loop(self) -> None:
        """Background monitoring and optimization loop."""
        while True:
            try:
                time.sleep(self.optimization_interval)
                
                if self.auto_optimize:
                    self.optimize_pool()
                
                # Cleanup old connection stats
                self._cleanup_old_connections()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")


# Global optimized pool instance
_optimized_pool: Optional[OptimizedConnectionPool] = None


def get_optimized_pool() -> OptimizedConnectionPool:
    """Get global optimized connection pool instance."""
    global _optimized_pool
    
    if _optimized_pool is None:
        _optimized_pool = OptimizedConnectionPool()
    
    return _optimized_pool


def create_optimized_pool(**kwargs) -> OptimizedConnectionPool:
    """Create a new optimized connection pool instance."""
    return OptimizedConnectionPool(**kwargs)