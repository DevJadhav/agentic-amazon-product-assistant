"""
Enhanced tracing utilities for LangSmith instrumentation.
Provides context propagation, performance monitoring, and business metrics.
"""

import time
import uuid
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from collections import defaultdict
from langsmith import traceable


class QueryIntent(Enum):
    """Classification of user query intents."""
    PRODUCT_INFO = "product_info"
    PRODUCT_REVIEW = "product_review"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    COMPLAINT = "complaint"
    USE_CASE = "use_case"
    GENERAL = "general"


class UserType(Enum):
    """Classification of user types based on behavior."""
    RESEARCHER = "researcher"  # Detailed, analytical queries
    BUYER = "buyer"  # Price-focused, comparison queries
    CASUAL = "casual"  # General, exploratory queries
    TROUBLESHOOTER = "troubleshooter"  # Problem-solving queries


@dataclass
class TraceContext:
    """Context object for trace propagation."""
    trace_id: str
    session_id: str
    conversation_turn: int
    user_type: Optional[UserType] = None
    query_intent: Optional[QueryIntent] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since context creation."""
        return time.time() - self.start_time
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value


class TraceContextManager:
    """Manages trace context propagation across operations."""
    
    def __init__(self):
        self._contexts: Dict[str, TraceContext] = {}
        self._current_context_id: Optional[str] = None
    
    def create_context(self, session_id: Optional[str] = None, conversation_turn: int = 0) -> TraceContext:
        """Create a new trace context."""
        context = TraceContext(
            trace_id=str(uuid.uuid4()),
            session_id=session_id or str(uuid.uuid4()),
            conversation_turn=conversation_turn
        )
        self._contexts[context.trace_id] = context
        self._current_context_id = context.trace_id
        return context
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get the current trace context."""
        if self._current_context_id:
            return self._contexts.get(self._current_context_id)
        return None
    
    def update_context(self, **kwargs) -> None:
        """Update the current trace context."""
        context = self.get_current_context()
        if context:
            for key, value in kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)
                else:
                    context.add_metadata(key, value)
    
    def cleanup_old_contexts(self, max_age_seconds: float = 3600) -> int:
        """Clean up old contexts to prevent memory leaks."""
        current_time = time.time()
        to_remove = []
        
        for trace_id, context in self._contexts.items():
            if current_time - context.start_time > max_age_seconds:
                to_remove.append(trace_id)
        
        for trace_id in to_remove:
            del self._contexts[trace_id]
            
        return len(to_remove)


class BusinessMetricsAnalyzer:
    """Analyzes business-level metrics from user interactions."""
    
    # Class-level constants for better performance
    PRODUCT_KEYWORDS: FrozenSet[str] = frozenset({
        'phone', 'iphone', 'android', 'smartphone', 'mobile',
        'laptop', 'computer', 'pc', 'macbook', 'tablet', 'ipad',
        'headphones', 'earbuds', 'speakers', 'audio', 'bluetooth',
        'charger', 'cable', 'usb', 'lightning', 'power',
        'router', 'wifi', 'internet', 'network', 'ethernet',
        'camera', 'video', 'streaming', 'tv', 'monitor'
    })
    
    BRAND_NAMES: FrozenSet[str] = frozenset({
        'apple', 'samsung', 'sony', 'lg', 'hp', 'dell', 'lenovo', 'asus',
        'microsoft', 'google', 'amazon', 'bose', 'jbl', 'logitech'
    })
    
    ACTIONABLE_PATTERNS: Tuple[str, ...] = (
        r'should', r'recommend', r'suggest', r'try', r'consider',
        r'look for', r'check', r'compare', r'avoid', r'choose'
    )
    
    def __init__(self):
        # Compile regex patterns once for better performance
        self.intent_patterns = {
            QueryIntent.PRODUCT_INFO: [
                re.compile(r'what is', re.IGNORECASE),
                re.compile(r'tell me about', re.IGNORECASE),
                re.compile(r'features of', re.IGNORECASE),
                re.compile(r'specs|specifications', re.IGNORECASE)
            ],
            QueryIntent.PRODUCT_REVIEW: [
                re.compile(r'reviews?', re.IGNORECASE),
                re.compile(r'what do people say', re.IGNORECASE),
                re.compile(r'opinions?|feedback|experience', re.IGNORECASE)
            ],
            QueryIntent.COMPARISON: [
                re.compile(r'compare|vs|versus|difference|better|best', re.IGNORECASE)
            ],
            QueryIntent.RECOMMENDATION: [
                re.compile(r'recommend|suggest|best|should i|good for', re.IGNORECASE)
            ],
            QueryIntent.COMPLAINT: [
                re.compile(r'problem|issue|complaint|broken|doesn\'t work|bad', re.IGNORECASE)
            ],
            QueryIntent.USE_CASE: [
                re.compile(r'good for|use for|suitable|work with|compatible', re.IGNORECASE)
            ]
        }
        
        # Compile actionable patterns
        self.actionable_pattern = re.compile('|'.join(self.ACTIONABLE_PATTERNS), re.IGNORECASE)
    
    @traceable
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a user query using optimized pattern matching."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return QueryIntent.GENERAL
    
    @lru_cache(maxsize=128)
    @traceable
    def calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1) with caching."""
        words = query.split()
        word_count = len(words)
        
        # Optimized calculations
        factors = {
            'word_count': min(word_count / 20, 1.0),
            'question_words': len(re.findall(r'\b(?:what|how|why|when|where|which|who)\b', query.lower())) * 0.1,
            'technical_terms': sum(1 for word in words if word.lower() in self.PRODUCT_KEYWORDS) * 0.05,
            'punctuation': query.count('?') * 0.05 + query.count('!') * 0.05
        }
        
        return min(sum(factors.values()), 1.0)
    
    @lru_cache(maxsize=128)
    @traceable
    def measure_specificity(self, query: str) -> float:
        """Measure query specificity (0-1) with caching."""
        query_lower = query.lower()
        words = set(query_lower.split())
        
        specificity_indicators = {
            'product_mentions': len(words & self.PRODUCT_KEYWORDS) * 0.2,
            'numbers': len(re.findall(r'\d+', query)) * 0.1,
            'brand_names': len(words & self.BRAND_NAMES) * 0.2,
            'model_indicators': len(re.findall(r'model|version|generation|series', query_lower)) * 0.1
        }
        
        return min(sum(specificity_indicators.values()), 1.0)
    
    @traceable
    def extract_product_focus(self, query: str) -> List[str]:
        """Extract product categories mentioned in query using set operations."""
        query_words = set(query.lower().split())
        return list(query_words & self.PRODUCT_KEYWORDS)
    
    @staticmethod
    @traceable
    def categorize_response_length(response: str) -> str:
        """Categorize response length with optimized thresholds."""
        word_count = len(response.split())
        
        if word_count < 50:
            return "short"
        elif word_count < 150:
            return "medium"
        elif word_count < 300:
            return "long"
        else:
            return "very_long"
    
    @traceable
    def measure_response_specificity(self, response: str, query: str) -> float:
        """Measure how specifically the response addresses the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
            
        overlap = len(query_words & response_words)
        return overlap / len(query_words)
    
    @traceable
    def count_product_mentions(self, response: str) -> int:
        """Count product mentions in response using set operations."""
        response_words = set(response.lower().split())
        return len(response_words & self.PRODUCT_KEYWORDS)
    
    @traceable
    def detect_actionable_content(self, response: str) -> bool:
        """Detect if response contains actionable advice."""
        return bool(self.actionable_pattern.search(response))
    
    @traceable
    def predict_follow_up(self, query: str, response: str) -> float:
        """Predict likelihood of follow-up question (0-1)."""
        # Use cached complexity calculation
        complexity = self.calculate_complexity(query)
        response_length = self.categorize_response_length(response)
        
        factors = {
            'question_complexity': complexity * 0.3,
            'response_length': 0.5 if response_length == "short" else 0.2,
            'comparison_mentioned': 0.3 if 'compare' in response.lower() else 0.0,
            'multiple_options': 0.4 if response.lower().count('option') + response.lower().count('choice') > 1 else 0.0
        }
        
        return min(sum(factors.values()), 1.0)
    
    @traceable
    def estimate_conversion_potential(self, query: str, context: Dict[str, Any]) -> float:
        """Estimate potential for user to make a purchase (0-1)."""
        intent = self.classify_intent(query)
        
        conversion_weights = {
            QueryIntent.RECOMMENDATION: 0.8,
            QueryIntent.COMPARISON: 0.7,
            QueryIntent.PRODUCT_INFO: 0.5,
            QueryIntent.USE_CASE: 0.6,
            QueryIntent.PRODUCT_REVIEW: 0.4,
            QueryIntent.COMPLAINT: 0.2,
            QueryIntent.GENERAL: 0.1
        }
        
        base_score = conversion_weights.get(intent, 0.1)
        
        # Adjust based on context
        product_boost = 1.2 if context.get('num_products', 0) > 0 else 1.0
        review_boost = 1.1 if context.get('num_reviews', 0) > 0 else 1.0
        
        return min(base_score * product_boost * review_boost, 1.0)
    
    @traceable
    def predict_satisfaction(self, query: str, response: str) -> float:
        """Predict user satisfaction with response (0-1)."""
        response_length = self.categorize_response_length(response)
        
        factors = {
            'specificity_match': self.measure_response_specificity(response, query) * 0.3,
            'response_length_appropriate': 0.3 if response_length in ["medium", "long"] else 0.1,
            'actionable_content': 0.2 if self.detect_actionable_content(response) else 0.0,
            'product_mentions': min(self.count_product_mentions(response) * 0.05, 0.2)
        }
        
        return min(sum(factors.values()), 1.0)
    
    @traceable
    def classify_user_type(self, query_history: List[str]) -> UserType:
        """Classify user type based on query history."""
        if not query_history:
            return UserType.CASUAL
        
        # Calculate metrics efficiently
        complexities = [self.calculate_complexity(q) for q in query_history]
        avg_complexity = sum(complexities) / len(complexities)
        
        intents = [self.classify_intent(q) for q in query_history]
        comparison_ratio = intents.count(QueryIntent.COMPARISON) / len(intents)
        
        detailed_ratio = sum(1 for q in query_history if len(q.split()) > 10) / len(query_history)
        
        # Classification logic
        if avg_complexity > 0.7:
            return UserType.RESEARCHER
        elif comparison_ratio > 0.3:
            return UserType.BUYER
        elif detailed_ratio > 0.4:
            return UserType.TROUBLESHOOTER
        else:
            return UserType.CASUAL


class VectorPerformanceMonitor:
    """Monitors vector database performance metrics."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache: Dict[int, float] = {}  # query_hash -> timestamp
        self.cache_size = cache_size
        self.metrics = defaultdict(int)
        self.metrics['cache_hits'] = 0
        self.metrics['cache_misses'] = 0
    
    @traceable
    def track_embedding_performance(self, query: str, embedding_time: float, embedding_dims: int) -> Dict[str, Any]:
        """Track embedding generation performance."""
        word_count = len(query.split())
        char_count = len(query)
        
        return {
            "query_length": char_count,
            "word_count": word_count,
            "embedding_time_ms": round(embedding_time * 1000, 2),
            "embedding_dimensions": embedding_dims,
            "tokens_per_second": word_count / embedding_time if embedding_time > 0 else 0,
            "characters_per_ms": char_count / (embedding_time * 1000) if embedding_time > 0 else 0
        }
    
    @traceable
    def track_search_performance(self, search_time: float, results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Track vector search performance."""
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        
        # Calculate diversity metrics efficiently
        categories = [m.get('category', 'unknown') for m in metadatas]
        unique_categories = len(set(categories))
        
        # Use numpy for statistical calculations if available
        if distances:
            distance_stats = {
                "avg_similarity_distance": float(np.mean(distances)),
                "min_similarity_distance": float(np.min(distances)),
                "max_similarity_distance": float(np.max(distances)),
                "std_similarity_distance": float(np.std(distances))
            }
        else:
            distance_stats = {
                "avg_similarity_distance": 0.0,
                "min_similarity_distance": 0.0,
                "max_similarity_distance": 0.0,
                "std_similarity_distance": 0.0
            }
        
        return {
            "search_time_ms": round(search_time * 1000, 2),
            "results_count": len(distances),
            **distance_stats,
            "result_diversity_score": unique_categories / len(categories) if categories else 0,
            "cache_hit": self._check_cache_hit(query),
            "cache_hit_rate": self.get_cache_hit_rate()
        }
    
    def _check_cache_hit(self, query: str) -> bool:
        """Check if query is in cache with LRU eviction."""
        query_hash = hash(query)
        current_time = time.time()
        
        if query_hash in self.cache:
            self.metrics['cache_hits'] += 1
            self.cache[query_hash] = current_time  # Update timestamp
            return True
        else:
            self.metrics['cache_misses'] += 1
            self.cache[query_hash] = current_time
            
            # LRU eviction if cache is full
            if len(self.cache) > self.cache_size:
                oldest_key = min(self.cache, key=self.cache.get)
                del self.cache[oldest_key]
                
            return False
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        return self.metrics['cache_hits'] / total if total > 0 else 0.0
    
    @traceable
    def analyze_search_quality(self, query: str, results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of search results."""
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        
        if not metadatas:
            return {
                "relevance_score": 0.0,
                "query_coverage": 0.0,
                "result_completeness": 0.0,
                "context_enrichment": {
                    "products_found": 0,
                    "reviews_found": 0,
                    "total_context_items": 0
                }
            }
        
        # Optimize relevance checking
        query_terms = set(query.lower().split())
        relevant_results = 0
        covered_terms = set()
        
        for metadata in metadatas:
            # Combine title and description for efficiency
            content = f"{metadata.get('title', '')} {metadata.get('description', '')}".lower()
            content_words = set(content.split())
            
            # Check relevance
            if query_terms & content_words:
                relevant_results += 1
                covered_terms.update(query_terms & content_words)
        
        return {
            "relevance_score": relevant_results / len(metadatas),
            "query_coverage": len(covered_terms) / len(query_terms) if query_terms else 0,
            "result_completeness": min(len(metadatas) / 5, 1.0),  # Assuming 5 is optimal
            "context_enrichment": {
                "products_found": context.get('num_products', 0),
                "reviews_found": context.get('num_reviews', 0),
                "total_context_items": context.get('num_products', 0) + context.get('num_reviews', 0)
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        return {
            "cache_hit_rate": self.get_cache_hit_rate(),
            "cache_size": len(self.cache),
            "total_queries": self.metrics['cache_hits'] + self.metrics['cache_misses'],
            **self.metrics
        }


# Global instances with configuration
trace_manager = TraceContextManager()
business_analyzer = BusinessMetricsAnalyzer()
performance_monitor = VectorPerformanceMonitor(cache_size=1000)


@traceable
def create_enhanced_trace_context(session_id: Optional[str] = None, conversation_turn: int = 0) -> TraceContext:
    """Create enhanced trace context with global manager."""
    return trace_manager.create_context(session_id, conversation_turn)


@traceable
def get_current_trace_context() -> Optional[TraceContext]:
    """Get current trace context."""
    return trace_manager.get_current_context()


@traceable
def update_trace_context(**kwargs) -> None:
    """Update current trace context."""
    trace_manager.update_context(**kwargs)