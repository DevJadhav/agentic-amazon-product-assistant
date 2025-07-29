"""
Health check system for LangGraph agent infrastructure.
Monitors system components and provides health status.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from ..state.database import get_database_manager, check_database_health
from ..tools.vector_search_tool import VectorSearchTool
from ..tools.product_analysis_tool import ProductAnalysisTool

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Health status for individual components."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    
    name: str
    status: ComponentStatus
    message: str
    response_time: Optional[float] = None
    last_check: Optional[datetime] = None
    error_count: int = 0
    metadata: Dict[str, Any] = None


class SystemHealthStatus:
    """Overall system health status."""
    
    def __init__(self):
        self.overall_status = ComponentStatus.UNKNOWN
        self.components: Dict[str, ComponentHealth] = {}
        self.last_check = datetime.utcnow()
        self.check_duration = 0.0
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.overall_status == ComponentStatus.HEALTHY
    
    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components."""
        return [
            name for name, health in self.components.items()
            if health.status == ComponentStatus.UNHEALTHY
        ]
    
    def get_degraded_components(self) -> List[str]:
        """Get list of degraded components."""
        return [
            name for name, health in self.components.items()
            if health.status == ComponentStatus.DEGRADED
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "overall_status": self.overall_status.value,
            "last_check": self.last_check.isoformat(),
            "check_duration": self.check_duration,
            "components": {
                name: {
                    "status": health.status.value,
                    "message": health.message,
                    "response_time": health.response_time,
                    "last_check": health.last_check.isoformat() if health.last_check else None,
                    "error_count": health.error_count,
                    "metadata": health.metadata or {}
                }
                for name, health in self.components.items()
            },
            "summary": {
                "total_components": len(self.components),
                "healthy_components": len([h for h in self.components.values() if h.status == ComponentStatus.HEALTHY]),
                "degraded_components": len(self.get_degraded_components()),
                "unhealthy_components": len(self.get_unhealthy_components())
            }
        }


class HealthChecker:
    """Comprehensive health checker for LangGraph agent system."""
    
    def __init__(self):
        """Initialize health checker."""
        self.logger = logging.getLogger(__name__)
        self.last_full_check: Optional[datetime] = None
        self.check_history: List[SystemHealthStatus] = []
        self.max_history = 100
        
        # Component checkers
        self.component_checkers = {
            "database": self._check_database_health,
            "vector_search": self._check_vector_search_health,
            "product_analysis": self._check_product_analysis_health,
            "agent_builder": self._check_agent_builder_health,
            "state_manager": self._check_state_manager_health
        }
    
    def check_system_health(self, quick_check: bool = False) -> SystemHealthStatus:
        """Perform comprehensive system health check."""
        
        start_time = time.time()
        health_status = SystemHealthStatus()
        
        try:
            # Check each component
            for component_name, checker_func in self.component_checkers.items():
                try:
                    if quick_check and component_name in ["product_analysis", "agent_builder"]:
                        # Skip slower checks in quick mode
                        continue
                    
                    component_health = checker_func()
                    health_status.components[component_name] = component_health
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {component_name}: {e}")
                    health_status.components[component_name] = ComponentHealth(
                        name=component_name,
                        status=ComponentStatus.UNHEALTHY,
                        message=f"Health check failed: {str(e)}",
                        last_check=datetime.utcnow(),
                        error_count=1
                    )
            
            # Determine overall status
            health_status.overall_status = self._determine_overall_status(health_status.components)
            health_status.check_duration = time.time() - start_time
            health_status.last_check = datetime.utcnow()
            
            # Store in history
            self._store_health_check(health_status)
            
            self.logger.info(f"System health check completed: {health_status.overall_status.value}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            
            # Return error status
            health_status.overall_status = ComponentStatus.UNHEALTHY
            health_status.check_duration = time.time() - start_time
            health_status.last_check = datetime.utcnow()
            
            return health_status
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health check history for the specified time period."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            {
                "timestamp": status.last_check.isoformat(),
                "overall_status": status.overall_status.value,
                "check_duration": status.check_duration,
                "component_count": len(status.components),
                "unhealthy_count": len(status.get_unhealthy_components())
            }
            for status in self.check_history
            if status.last_check >= cutoff_time
        ]
    
    def get_component_trends(self, component_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get health trends for a specific component."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        component_history = []
        for status in self.check_history:
            if status.last_check >= cutoff_time and component_name in status.components:
                component_health = status.components[component_name]
                component_history.append({
                    "timestamp": status.last_check.isoformat(),
                    "status": component_health.status.value,
                    "response_time": component_health.response_time,
                    "error_count": component_health.error_count
                })
        
        # Calculate trends
        if component_history:
            status_counts = {}
            response_times = []
            
            for entry in component_history:
                status = entry["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                
                if entry["response_time"]:
                    response_times.append(entry["response_time"])
            
            return {
                "component": component_name,
                "period_hours": hours,
                "total_checks": len(component_history),
                "status_distribution": status_counts,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else None,
                "max_response_time": max(response_times) if response_times else None,
                "min_response_time": min(response_times) if response_times else None,
                "history": component_history
            }
        
        return {"component": component_name, "no_data": True}
    
    # Component-specific health checkers
    
    def _check_database_health(self) -> ComponentHealth:
        """Check database health."""
        
        start_time = time.time()
        
        try:
            db_health = check_database_health()
            response_time = time.time() - start_time
            
            if db_health.get("connected", False):
                stats = db_health.get("stats", {})
                
                return ComponentHealth(
                    name="database",
                    status=ComponentStatus.HEALTHY,
                    message="Database is connected and operational",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    metadata={
                        "total_conversations": stats.get("total_conversations", 0),
                        "total_messages": stats.get("total_messages", 0),
                        "active_conversations_24h": stats.get("active_conversations_24h", 0)
                    }
                )
            else:
                return ComponentHealth(
                    name="database",
                    status=ComponentStatus.UNHEALTHY,
                    message=f"Database connection failed: {db_health.get('error', 'Unknown error')}",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    error_count=1
                )
                
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=ComponentStatus.UNHEALTHY,
                message=f"Database health check failed: {str(e)}",
                response_time=time.time() - start_time,
                last_check=datetime.utcnow(),
                error_count=1
            )
    
    def _check_vector_search_health(self) -> ComponentHealth:
        """Check vector search tool health."""
        
        start_time = time.time()
        
        try:
            vector_tool = VectorSearchTool()
            test_result = vector_tool.test_connection()
            response_time = time.time() - start_time
            
            if test_result.get("status") == "success":
                return ComponentHealth(
                    name="vector_search",
                    status=ComponentStatus.HEALTHY,
                    message="Vector search is operational",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    metadata={
                        "database_available": test_result.get("database_available", False),
                        "enhanced_processor": test_result.get("enhanced_processor", False)
                    }
                )
            else:
                return ComponentHealth(
                    name="vector_search",
                    status=ComponentStatus.DEGRADED,
                    message=f"Vector search issues: {test_result.get('message', 'Unknown issue')}",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    error_count=1
                )
                
        except Exception as e:
            return ComponentHealth(
                name="vector_search",
                status=ComponentStatus.UNHEALTHY,
                message=f"Vector search health check failed: {str(e)}",
                response_time=time.time() - start_time,
                last_check=datetime.utcnow(),
                error_count=1
            )
    
    def _check_product_analysis_health(self) -> ComponentHealth:
        """Check product analysis tool health."""
        
        start_time = time.time()
        
        try:
            analysis_tool = ProductAnalysisTool()
            
            # Test with sample data
            sample_products = [
                {
                    "id": "test1",
                    "content": "Test product 1",
                    "metadata": {"title": "Test Product 1", "price": "99.99", "average_rating": "4.5"}
                },
                {
                    "id": "test2", 
                    "content": "Test product 2",
                    "metadata": {"title": "Test Product 2", "price": "149.99", "average_rating": "4.2"}
                }
            ]
            
            result = analysis_tool._run(
                products=sample_products,
                analysis_type="comparison",
                include_summary=True
            )
            
            response_time = time.time() - start_time
            
            if "error" not in result:
                return ComponentHealth(
                    name="product_analysis",
                    status=ComponentStatus.HEALTHY,
                    message="Product analysis is operational",
                    response_time=response_time,
                    last_check=datetime.utcnow()
                )
            else:
                return ComponentHealth(
                    name="product_analysis",
                    status=ComponentStatus.DEGRADED,
                    message=f"Product analysis issues: {result.get('error', 'Unknown error')}",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    error_count=1
                )
                
        except Exception as e:
            return ComponentHealth(
                name="product_analysis",
                status=ComponentStatus.UNHEALTHY,
                message=f"Product analysis health check failed: {str(e)}",
                response_time=time.time() - start_time,
                last_check=datetime.utcnow(),
                error_count=1
            )
    
    def _check_agent_builder_health(self) -> ComponentHealth:
        """Check agent builder health."""
        
        start_time = time.time()
        
        try:
            from ..core.agent_builder import AgentGraphBuilder
            
            builder = AgentGraphBuilder()
            available_graphs = builder.get_available_graphs()
            
            response_time = time.time() - start_time
            
            if available_graphs:
                return ComponentHealth(
                    name="agent_builder",
                    status=ComponentStatus.HEALTHY,
                    message="Agent builder is operational",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    metadata={
                        "available_graphs": len(available_graphs),
                        "graph_types": list(available_graphs.keys())
                    }
                )
            else:
                return ComponentHealth(
                    name="agent_builder",
                    status=ComponentStatus.DEGRADED,
                    message="Agent builder has no available graphs",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    error_count=1
                )
                
        except Exception as e:
            return ComponentHealth(
                name="agent_builder",
                status=ComponentStatus.UNHEALTHY,
                message=f"Agent builder health check failed: {str(e)}",
                response_time=time.time() - start_time,
                last_check=datetime.utcnow(),
                error_count=1
            )
    
    def _check_state_manager_health(self) -> ComponentHealth:
        """Check state manager health."""
        
        start_time = time.time()
        
        try:
            from ..state.state_manager import LangGraphStateManager
            
            state_manager = LangGraphStateManager()
            stats = state_manager.get_state_statistics()
            
            response_time = time.time() - start_time
            
            if stats and "error" not in stats:
                return ComponentHealth(
                    name="state_manager",
                    status=ComponentStatus.HEALTHY,
                    message="State manager is operational",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    metadata=stats
                )
            else:
                return ComponentHealth(
                    name="state_manager",
                    status=ComponentStatus.DEGRADED,
                    message="State manager has limited functionality",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    error_count=1
                )
                
        except Exception as e:
            return ComponentHealth(
                name="state_manager",
                status=ComponentStatus.UNHEALTHY,
                message=f"State manager health check failed: {str(e)}",
                response_time=time.time() - start_time,
                last_check=datetime.utcnow(),
                error_count=1
            )
    
    # Helper methods
    
    def _determine_overall_status(self, components: Dict[str, ComponentHealth]) -> ComponentStatus:
        """Determine overall system status based on component health."""
        
        if not components:
            return ComponentStatus.UNKNOWN
        
        statuses = [health.status for health in components.values()]
        
        # If any component is unhealthy, system is unhealthy
        if ComponentStatus.UNHEALTHY in statuses:
            return ComponentStatus.UNHEALTHY
        
        # If any component is degraded, system is degraded
        if ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        
        # If all components are healthy, system is healthy
        if all(status == ComponentStatus.HEALTHY for status in statuses):
            return ComponentStatus.HEALTHY
        
        return ComponentStatus.UNKNOWN
    
    def _store_health_check(self, health_status: SystemHealthStatus) -> None:
        """Store health check result in history."""
        
        self.check_history.append(health_status)
        
        # Limit history size
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]
        
        self.last_full_check = health_status.last_check