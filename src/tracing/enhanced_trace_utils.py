"""
Optimized Langsmith Tracing with Session-Based Initialization and Zero-Redundancy Design
Production-ready AI pipeline monitoring with efficient session management and comprehensive analytics.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import uuid
import json
import asyncio
from collections import defaultdict, deque
import os

from langsmith import traceable, Client as LangsmithClient
from langsmith.schemas import Run, Example

logger = logging.getLogger(__name__)

@dataclass
class TraceSession:
    """Optimized trace session with efficient state management."""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    active: bool = True
    
    # Performance optimization
    batch_traces: List[Dict] = field(default_factory=list)
    batch_size: int = 10
    auto_flush_interval: int = 300  # 5 minutes
    
    # Session analytics
    query_patterns: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0

@dataclass
class TraceMetrics:
    """Comprehensive trace metrics for analytics."""
    total_traces: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    unique_sessions: int = 0
    active_sessions: int = 0
    
    # Performance metrics
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Business metrics
    user_satisfaction: float = 0.0
    conversion_rate: float = 0.0
    retention_rate: float = 0.0

class EnhancedTracingManager:
    """Enhanced tracing manager with zero-redundancy design and session optimization."""
    
    def __init__(self, 
                 project_name: str = "amazon-electronics-assistant",
                 enable_langsmith: bool = True,
                 batch_size: int = 10,
                 flush_interval: int = 300):
        """Initialize the enhanced tracing manager."""
        
        self.project_name = project_name
        self.enable_langsmith = enable_langsmith
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Session management
        self.sessions: Dict[str, TraceSession] = {}
        self.session_lock = threading.Lock()
        
        # Performance optimization
        self.trace_buffer: deque = deque(maxlen=1000)
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        
        # Analytics
        self.global_metrics = TraceMetrics()
        self.hourly_metrics: Dict[str, TraceMetrics] = defaultdict(TraceMetrics)
        
        # Initialize LangSmith client with session-based optimization
        self.langsmith_client = self._initialize_langsmith_client()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info(f"Enhanced Tracing Manager initialized for project: {project_name}")
    
    def _initialize_langsmith_client(self) -> Optional[LangsmithClient]:
        """Initialize LangSmith client with optimized configuration."""
        if not self.enable_langsmith:
            return None
        
        api_key = os.getenv('LANGSMITH_API_KEY')
        if not api_key:
            logger.warning("LANGSMITH_API_KEY not found. Tracing will be local only.")
            return None
        
        try:
            client = LangsmithClient(
                api_key=api_key,
                api_url=os.getenv('LANGSMITH_API_URL', 'https://api.smith.langchain.com')
            )
            
            # Verify connection
            try:
                client.read_project(project_name=self.project_name)
                logger.info(f"Connected to existing LangSmith project: {self.project_name}")
            except:
                # Create project if it doesn't exist
                client.create_project(
                    project_name=self.project_name,
                    description="Enhanced Amazon Electronics Assistant with production monitoring"
                )
                logger.info(f"Created new LangSmith project: {self.project_name}")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            return None
    
    def _start_background_tasks(self):
        """Start background tasks for session management and batch processing."""
        
        def background_worker():
            """Background worker for periodic tasks."""
            while True:
                try:
                    # Flush batched traces
                    self._flush_batch_traces()
                    
                    # Clean up inactive sessions
                    self._cleanup_inactive_sessions()
                    
                    # Update metrics cache
                    self._update_metrics_cache()
                    
                    time.sleep(self.flush_interval)
                    
                except Exception as e:
                    logger.error(f"Background worker error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        # Start background thread
        background_thread = threading.Thread(target=background_worker, daemon=True)
        background_thread.start()
    
    def create_session(self, 
                      user_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create optimized trace session with zero-redundancy design."""
        
        session_id = str(uuid.uuid4())
        
        session = TraceSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            metadata=metadata or {},
            batch_size=self.batch_size,
            auto_flush_interval=self.flush_interval
        )
        
        with self.session_lock:
            self.sessions[session_id] = session
        
        # Update global metrics
        self.global_metrics.unique_sessions += 1
        self.global_metrics.active_sessions += 1
        
        logger.info(f"Created optimized trace session: {session_id}")
        return session_id
    
    @contextmanager
    def trace_context(self, 
                     session_id: str,
                     operation_name: str,
                     **kwargs):
        """Optimized trace context manager with session-based batching."""
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        error = None
        result = None
        
        try:
            yield trace_id
            
        except Exception as e:
            error = e
            raise
            
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Create optimized trace record
            trace_record = {
                'trace_id': trace_id,
                'session_id': session_id,
                'operation_name': operation_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'success': error is None,
                'error': str(error) if error else None,
                'metadata': kwargs,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to session batch
            self._add_to_session_batch(session_id, trace_record)
            
            # Update session analytics
            self._update_session_analytics(session_id, trace_record)
    
    def _add_to_session_batch(self, session_id: str, trace_record: Dict[str, Any]):
        """Add trace to session batch with zero-redundancy optimization."""
        
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found for trace batching")
                return
            
            # Add to batch
            session.batch_traces.append(trace_record)
            session.trace_count += 1
            session.last_activity = datetime.now()
            
            # Auto-flush if batch is full
            if len(session.batch_traces) >= session.batch_size:
                self._flush_session_batch(session_id)
    
    def _flush_session_batch(self, session_id: str):
        """Flush session batch to LangSmith with optimized batching."""
        
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session or not session.batch_traces:
                return
            
            # Extract traces for flushing
            traces_to_flush = session.batch_traces.copy()
            session.batch_traces.clear()
        
        # Send to LangSmith if available
        if self.langsmith_client:
            try:
                self._send_traces_to_langsmith(traces_to_flush)
            except Exception as e:
                logger.error(f"Failed to send traces to LangSmith: {e}")
        
        # Add to local buffer for analytics
        self.trace_buffer.extend(traces_to_flush)
        
        logger.debug(f"Flushed {len(traces_to_flush)} traces for session {session_id}")
    
    def _flush_batch_traces(self):
        """Flush all pending batched traces."""
        
        with self.session_lock:
            session_ids = list(self.sessions.keys())
        
        for session_id in session_ids:
            self._flush_session_batch(session_id)
    
    def _send_traces_to_langsmith(self, traces: List[Dict[str, Any]]):
        """Send traces to LangSmith with optimized batch processing."""
        
        if not self.langsmith_client:
            return
        
        # Convert traces to LangSmith format
        langsmith_runs = []
        
        for trace in traces:
            run = {
                'name': trace['operation_name'],
                'run_type': 'chain',  # or 'llm', 'tool', etc.
                'start_time': datetime.fromtimestamp(trace['start_time']),
                'end_time': datetime.fromtimestamp(trace['end_time']),
                'extra': {
                    'session_id': trace['session_id'],
                    'trace_id': trace['trace_id'],
                    'metadata': trace.get('metadata', {})
                },
                'error': trace.get('error'),
                'tags': [f"session:{trace['session_id']}"]
            }
            
            if trace.get('success'):
                run['status'] = 'success'
            else:
                run['status'] = 'error'
            
            langsmith_runs.append(run)
        
        # Batch send to LangSmith
        try:
            # Use batch API if available
            for run in langsmith_runs:
                self.langsmith_client.create_run(**run)
            
            logger.debug(f"Successfully sent {len(langsmith_runs)} traces to LangSmith")
            
        except Exception as e:
            logger.error(f"Failed to send batch traces to LangSmith: {e}")
    
    def _update_session_analytics(self, session_id: str, trace_record: Dict[str, Any]):
        """Update session analytics with zero-redundancy design."""
        
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            # Update session metrics
            duration = trace_record['duration']
            session.response_times.append(duration)
            
            if trace_record['success']:
                session.success_count += 1
            else:
                session.error_count += 1
            
            # Update query patterns
            operation = trace_record['operation_name']
            session.query_patterns[operation] += 1
        
        # Update global metrics
        self._update_global_metrics(trace_record)
    
    def _update_global_metrics(self, trace_record: Dict[str, Any]):
        """Update global metrics efficiently."""
        
        duration = trace_record['duration']
        
        # Update global counters
        self.global_metrics.total_traces += 1
        
        if trace_record['success']:
            self.global_metrics.success_rate = (
                (self.global_metrics.success_rate * (self.global_metrics.total_traces - 1) + 1) / 
                self.global_metrics.total_traces
            )
        else:
            self.global_metrics.error_rate = (
                (self.global_metrics.error_rate * (self.global_metrics.total_traces - 1) + 1) / 
                self.global_metrics.total_traces
            )
        
        # Update average response time
        self.global_metrics.avg_response_time = (
            (self.global_metrics.avg_response_time * (self.global_metrics.total_traces - 1) + duration) / 
            self.global_metrics.total_traces
        )
        
        # Update hourly metrics
        hour_key = datetime.now().strftime('%Y-%m-%d-%H')
        hourly_metric = self.hourly_metrics[hour_key]
        hourly_metric.total_traces += 1
        hourly_metric.avg_response_time = (
            (hourly_metric.avg_response_time * (hourly_metric.total_traces - 1) + duration) / 
            hourly_metric.total_traces
        )
    
    def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions to prevent memory leaks."""
        
        cutoff_time = datetime.now() - timedelta(hours=24)  # 24 hour session timeout
        
        with self.session_lock:
            inactive_sessions = [
                session_id for session_id, session in self.sessions.items()
                if session.last_activity < cutoff_time
            ]
            
            for session_id in inactive_sessions:
                # Flush any remaining traces
                self._flush_session_batch(session_id)
                
                # Remove session
                session = self.sessions.pop(session_id)
                session.active = False
                
                # Update global metrics
                self.global_metrics.active_sessions -= 1
        
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
    
    def _update_metrics_cache(self):
        """Update metrics cache for fast retrieval."""
        
        with self.cache_lock:
            # Calculate percentiles from recent traces
            recent_times = [trace['duration'] for trace in list(self.trace_buffer)[-100:]]
            
            if recent_times:
                recent_times.sort()
                n = len(recent_times)
                
                self.global_metrics.p50_response_time = recent_times[int(n * 0.5)]
                self.global_metrics.p95_response_time = recent_times[int(n * 0.95)]
                self.global_metrics.p99_response_time = recent_times[int(n * 0.99)]
            
            # Update cache timestamp
            self.metrics_cache['last_updated'] = datetime.now().isoformat()
            self.metrics_cache['global_metrics'] = self.global_metrics
    
    @traceable
    def trace_rag_query(self, 
                       session_id: str,
                       query: str,
                       response: str,
                       context: Dict[str, Any],
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Trace RAG query with optimized session management."""
        
        with self.trace_context(
            session_id=session_id,
            operation_name='rag_query',
            query=query,
            response_length=len(response),
            context_size=len(str(context)),
            metadata=metadata or {}
        ) as trace_id:
            
            # Additional RAG-specific analytics
            self._analyze_rag_performance(session_id, query, response, context)
            
            return trace_id
    
    def _analyze_rag_performance(self, 
                                session_id: str,
                                query: str, 
                                response: str,
                                context: Dict[str, Any]):
        """Analyze RAG-specific performance metrics."""
        
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            # Query classification
            query_lower = query.lower()
            if any(word in query_lower for word in ['compare', 'vs', 'versus']):
                session.query_patterns['comparison'] += 1
            elif any(word in query_lower for word in ['best', 'recommend', 'suggest']):
                session.query_patterns['recommendation'] += 1
            elif any(word in query_lower for word in ['review', 'opinion', 'feedback']):
                session.query_patterns['review_analysis'] += 1
            else:
                session.query_patterns['general'] += 1
            
            # Context utilization
            if context and 'documents' in context:
                doc_count = len(context['documents'])
                session.metadata['avg_context_docs'] = session.metadata.get('avg_context_docs', 0) * 0.9 + doc_count * 0.1
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session analytics."""
        
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            # Calculate session metrics
            total_traces = session.success_count + session.error_count
            success_rate = session.success_count / total_traces if total_traces > 0 else 0
            
            avg_response_time = (
                sum(session.response_times) / len(session.response_times) 
                if session.response_times else 0
            )
            
            return {
                'session_id': session_id,
                'user_id': session.user_id,
                'start_time': session.start_time.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'total_traces': total_traces,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'query_patterns': dict(session.query_patterns),
                'metadata': session.metadata,
                'active': session.active,
                'session_duration': (session.last_activity - session.start_time).total_seconds()
            }
    
    def get_global_analytics(self) -> Dict[str, Any]:
        """Get comprehensive global analytics."""
        
        return {
            'global_metrics': {
                'total_traces': self.global_metrics.total_traces,
                'success_rate': self.global_metrics.success_rate,
                'error_rate': self.global_metrics.error_rate,
                'avg_response_time': self.global_metrics.avg_response_time,
                'p50_response_time': self.global_metrics.p50_response_time,
                'p95_response_time': self.global_metrics.p95_response_time,
                'p99_response_time': self.global_metrics.p99_response_time,
                'unique_sessions': self.global_metrics.unique_sessions,
                'active_sessions': self.global_metrics.active_sessions
            },
            'system_health': {
                'langsmith_connected': self.langsmith_client is not None,
                'buffer_size': len(self.trace_buffer),
                'session_count': len(self.sessions),
                'batch_processing_enabled': True
            },
            'performance_optimization': {
                'zero_redundancy_design': True,
                'session_based_batching': True,
                'auto_cleanup_enabled': True,
                'metrics_caching_enabled': True
            }
        }
    
    def get_hourly_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get hourly performance trends."""
        
        current_time = datetime.now()
        trends = {}
        
        for i in range(hours):
            hour_time = current_time - timedelta(hours=i)
            hour_key = hour_time.strftime('%Y-%m-%d-%H')
            
            metric = self.hourly_metrics.get(hour_key, TraceMetrics())
            trends[hour_key] = {
                'total_traces': metric.total_traces,
                'avg_response_time': metric.avg_response_time,
                'success_rate': metric.success_rate
            }
        
        return trends
    
    def export_analytics(self, output_path: str):
        """Export comprehensive analytics to file."""
        
        analytics_data = {
            'export_time': datetime.now().isoformat(),
            'global_analytics': self.get_global_analytics(),
            'hourly_trends': self.get_hourly_trends(),
            'session_summaries': [
                self.get_session_analytics(session_id) 
                for session_id in list(self.sessions.keys())[:10]  # Limit for performance
            ],
            'recent_traces': [
                trace for trace in list(self.trace_buffer)[-50:]  # Last 50 traces
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        logger.info(f"Analytics exported to {output_path}")
    
    def close_session(self, session_id: str):
        """Close session and flush remaining traces."""
        
        # Flush any remaining traces
        self._flush_session_batch(session_id)
        
        with self.session_lock:
            session = self.sessions.get(session_id)
            if session:
                session.active = False
                self.global_metrics.active_sessions -= 1
        
        logger.info(f"Closed session: {session_id}")
    
    def shutdown(self):
        """Shutdown tracing manager gracefully."""
        
        logger.info("Shutting down Enhanced Tracing Manager...")
        
        # Flush all remaining traces
        self._flush_batch_traces()
        
        # Close all sessions
        with self.session_lock:
            for session_id in list(self.sessions.keys()):
                self.close_session(session_id)
        
        logger.info("Enhanced Tracing Manager shutdown complete")


# Global instance for easy access
_global_tracer: Optional[EnhancedTracingManager] = None

def get_global_tracer() -> EnhancedTracingManager:
    """Get or create global tracer instance."""
    global _global_tracer
    
    if _global_tracer is None:
        _global_tracer = EnhancedTracingManager()
    
    return _global_tracer

def initialize_tracing(project_name: str = "amazon-electronics-assistant") -> EnhancedTracingManager:
    """Initialize enhanced tracing system."""
    global _global_tracer
    
    _global_tracer = EnhancedTracingManager(project_name=project_name)
    return _global_tracer