"""
Performance Tracker for the AI-Powered Amazon Product Assistant.
Handles performance monitoring, metrics collection, and analytics.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import streamlit as st
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    total_time_ms: float
    rag_time_ms: float
    llm_time_ms: float
    llm_provider: str
    llm_model: str
    use_rag: bool
    message_length: int
    business_metrics: Dict[str, float]
    
    @property
    def rag_percentage(self) -> float:
        """Get RAG time as percentage of total."""
        if self.total_time_ms > 0 and self.rag_time_ms > 0:
            return (self.rag_time_ms / self.total_time_ms) * 100
        return 0.0
    
    @property
    def llm_percentage(self) -> float:
        """Get LLM time as percentage of total."""
        if self.total_time_ms > 0:
            return (self.llm_time_ms / self.total_time_ms) * 100
        return 100.0


class PerformanceTracker:
    """Tracks and analyzes system performance metrics."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self._ensure_session_state()
    
    def _ensure_session_state(self):
        """Ensure session state is properly initialized."""
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
        
        if 'provider_model_stats' not in st.session_state:
            st.session_state.provider_model_stats = {}
    
    def track_performance(self, total_time_ms: float, rag_time_ms: float, 
                         llm_time_ms: float, llm_provider: str, 
                         llm_model: str, use_rag: bool, 
                         message_length: int, 
                         business_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Track a performance measurement."""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            total_time_ms=total_time_ms,
            rag_time_ms=rag_time_ms,
            llm_time_ms=llm_time_ms,
            llm_provider=llm_provider,
            llm_model=llm_model,
            use_rag=use_rag,
            message_length=message_length,
            business_metrics=business_metrics or {}
        )
        
        # Store in session state
        st.session_state.last_performance = asdict(metrics)
        st.session_state.performance_history.append(asdict(metrics))
        
        # Keep history size manageable (last 100 entries)
        if len(st.session_state.performance_history) > 100:
            st.session_state.performance_history = st.session_state.performance_history[-100:]
    
    def get_latest_performance(self) -> Optional[Dict[str, Any]]:
        """Get the latest performance metrics."""
        return st.session_state.get('last_performance')
    
    def get_performance_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get performance history."""
        history = st.session_state.get('performance_history', [])
        return history[-limit:] if history else []
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get provider-specific statistics."""
        return st.session_state.get('provider_model_stats', {})
    
    def analyze_provider_performance(self) -> List[Dict[str, Any]]:
        """Analyze and compare provider performance."""
        provider_stats = self.get_provider_stats()
        comparison_data = []
        
        for key, stats in provider_stats.items():
            if stats["total_queries"] == 0:
                continue
            
            # Calculate averages
            avg_total_time = stats["total_time_ms"] / stats["total_queries"]
            avg_llm_time = stats["total_llm_time_ms"] / stats["total_queries"]
            avg_rag_time = (stats["total_rag_time_ms"] / stats["rag_queries"] 
                          if stats["rag_queries"] > 0 else 0)
            
            comparison_data.append({
                "provider": stats["provider"],
                "model": stats["model"],
                "total_queries": stats["total_queries"],
                "avg_total_time_ms": round(avg_total_time, 1),
                "avg_llm_time_ms": round(avg_llm_time, 1),
                "avg_rag_time_ms": round(avg_rag_time, 1) if avg_rag_time > 0 else None,
                "min_llm_time_ms": round(stats["min_llm_time_ms"], 1),
                "max_llm_time_ms": round(stats["max_llm_time_ms"], 1),
                "rag_queries": stats["rag_queries"],
                "non_rag_queries": stats["non_rag_queries"]
            })
        
        # Sort by average LLM time (fastest first)
        comparison_data.sort(key=lambda x: x["avg_llm_time_ms"])
        
        return comparison_data
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and trends."""
        provider_stats = self.analyze_provider_performance()
        
        if not provider_stats:
            return {"status": "no_data", "message": "No performance data available yet"}
        
        insights = {
            "status": "success",
            "fastest_provider": None,
            "slowest_provider": None,
            "speed_difference": None,
            "recommendations": []
        }
        
        if len(provider_stats) > 0:
            fastest = provider_stats[0]
            slowest = provider_stats[-1] if len(provider_stats) > 1 else None
            
            insights["fastest_provider"] = {
                "name": fastest["provider"],
                "model": fastest["model"],
                "avg_time_ms": fastest["avg_llm_time_ms"]
            }
            
            if slowest:
                insights["slowest_provider"] = {
                    "name": slowest["provider"],
                    "model": slowest["model"],
                    "avg_time_ms": slowest["avg_llm_time_ms"]
                }
                
                insights["speed_difference"] = slowest["avg_llm_time_ms"] - fastest["avg_llm_time_ms"]
                
                # Generate recommendations
                if insights["speed_difference"] > 1000:
                    insights["recommendations"].append(
                        f"Consider using {fastest['provider']} for better performance"
                    )
                
                if fastest["avg_llm_time_ms"] < 500:
                    insights["recommendations"].append(
                        f"{fastest['provider']} is delivering excellent sub-500ms responses"
                    )
        
        return insights
    
    def get_current_provider_trend(self, provider: str, model: str, 
                                  window_size: int = 10) -> Dict[str, Any]:
        """Get performance trend for current provider/model."""
        key = f"{provider}::{model}"
        stats = st.session_state.get('provider_model_stats', {}).get(key)
        
        if not stats or not stats.get("recent_performances"):
            return {"status": "no_data"}
        
        recent_perfs = stats["recent_performances"][-window_size:]
        
        if len(recent_perfs) < 3:
            return {"status": "insufficient_data", "data_points": len(recent_perfs)}
        
        # Calculate trend
        recent_times = [p["llm_time_ms"] for p in recent_perfs]
        
        # Simple moving average comparison
        first_half = recent_times[:len(recent_times)//2]
        second_half = recent_times[len(recent_times)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        trend_change = avg_second - avg_first
        
        if abs(trend_change) < 50:  # Less than 50ms change
            trend = "stable"
        elif trend_change > 0:
            trend = "slower"
        else:
            trend = "faster"
        
        return {
            "status": "success",
            "trend": trend,
            "avg_recent": round(avg_second, 1),
            "avg_previous": round(avg_first, 1),
            "change_ms": round(trend_change, 1),
            "data_points": len(recent_times)
        }
    
    def export_performance_data(self) -> str:
        """Export performance data as JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        export_data = {
            "export_timestamp": timestamp,
            "provider_model_stats": self.get_provider_stats(),
            "performance_history": self.get_performance_history(50),
            "insights": self.get_performance_insights()
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def get_rag_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest RAG performance metrics."""
        last_rag_result = st.session_state.get('last_rag_result')
        
        if not last_rag_result:
            return None
        
        context = last_rag_result.get("context", {})
        perf_metrics = context.get("performance_metrics", {})
        
        if not perf_metrics:
            return None
        
        # Extract and format metrics
        formatted_metrics = {
            "processing_time_ms": last_rag_result.get("processing_time_ms", 0),
            "num_products": context.get("num_products", 0),
            "num_reviews": context.get("num_reviews", 0),
            "query_type": context.get("query_type", "Unknown")
        }
        
        # Add detailed search metrics if available
        for search_type, metrics in perf_metrics.items():
            if isinstance(metrics, dict) and "embedding_metrics" in metrics:
                formatted_metrics[f"{search_type}_metrics"] = {
                    "embedding_time_ms": metrics["embedding_metrics"].get("embedding_time_ms", 0),
                    "search_time_ms": metrics["search_metrics"].get("search_time_ms", 0),
                    "relevance_score": metrics["quality_metrics"].get("relevance_score", 0)
                }
        
        return formatted_metrics
    
    def calculate_session_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the current session."""
        history = self.get_performance_history()
        
        if not history:
            return {
                "total_queries": 0,
                "avg_response_time_ms": 0,
                "rag_usage_rate": 0,
                "avg_business_satisfaction": 0
            }
        
        total_queries = len(history)
        total_time = sum(h["total_time_ms"] for h in history)
        rag_queries = sum(1 for h in history if h.get("use_rag", False))
        
        # Calculate business metrics averages
        satisfaction_scores = []
        for h in history:
            business_metrics = h.get("business_metrics", {})
            if "user_satisfaction_prediction" in business_metrics:
                satisfaction_scores.append(business_metrics["user_satisfaction_prediction"])
        
        avg_satisfaction = (sum(satisfaction_scores) / len(satisfaction_scores) 
                          if satisfaction_scores else 0)
        
        return {
            "total_queries": total_queries,
            "avg_response_time_ms": round(total_time / total_queries, 1),
            "rag_usage_rate": round(rag_queries / total_queries * 100, 1),
            "avg_business_satisfaction": round(avg_satisfaction, 2)
        }


# Global performance tracker instance
performance_tracker = PerformanceTracker() 