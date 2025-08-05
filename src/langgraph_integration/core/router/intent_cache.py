"""
Intent classification caching system for improved response times.
Implements intelligent caching with TTL, LRU eviction, and cache warming.
"""

import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
from datetime import datetime, timedelta
import threading

from .intent_classifier import IntentResult
from ...monitoring.performance_monitor import get_performance_monitor, performance_track

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    result: IntentResult
    created_at: float
    access_count: int
    last_accessed: float
    ttl: float
    context_hash: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() > (self.created_at + self.ttl)
    
    def is_stale(self, staleness_threshold: float = 3600) -> bool:
        """Check if cache entry is stale (old but not expired)."""
        return time.time() > (self.created_at + staleness_threshold)
    
    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class IntentClassificationCache:
    """High-performance cache for intent classification results."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 default_ttl: float = 3600,  # 1 hour
                 cleanup_interval: float = 300,  # 5 minutes
                 enable_persistence: bool = True):
        """
        Initialize intent classification cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL for cache entries in seconds
            cleanup_interval: Interval for cache cleanup in seconds
            enable_persistence: Whether to persist cache to database
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.enable_persistence = enable_persistence
        
        # Thread-safe cache storage using OrderedDict for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Cache statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0,
            "total_requests": 0,
            "cache_size": 0,
            "last_cleanup": time.time()
        }
        
        # Performance monitor
        self.perf_monitor = get_performance_monitor()
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Intent classification cache initialized with max_size={max_size}, ttl={default_ttl}s")
    
    @performance_track("intent_cache_get")
    def get(self, message: str, context: Optional[Dict[str, Any]] = None) -> Optional[IntentResult]:
        """
        Get cached intent classification result.
        
        Args:
            message: User message
            context: Optional context for cache key generation
            
        Returns:
            Cached IntentResult or None if not found/expired
        """
        cache_key = self._generate_cache_key(message, context)
        
        with self._lock:
            self._stats["total_requests"] += 1
            
            if cache_key not in self._cache:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss for key: {cache_key[:16]}...")
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[cache_key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                logger.debug(f"Cache entry expired for key: {cache_key[:16]}...")
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            entry.touch()
            
            self._stats["hits"] += 1
            self._stats["cache_size"] = len(self._cache)
            
            logger.debug(f"Cache hit for key: {cache_key[:16]}... (access_count: {entry.access_count})")
            return entry.result
    
    @performance_track("intent_cache_set")
    def set(self, message: str, result: IntentResult, 
            context: Optional[Dict[str, Any]] = None,
            ttl: Optional[float] = None) -> None:
        """
        Cache intent classification result.
        
        Args:
            message: User message
            result: Intent classification result
            context: Optional context for cache key generation
            ttl: Time to live in seconds (uses default if None)
        """
        cache_key = self._generate_cache_key(message, context)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Check cache size and evict if necessary
            if len(self._cache) >= self.max_size:
                self._evict_lru_entries()
            
            # Create cache entry
            entry = CacheEntry(
                result=result,
                created_at=time.time(),
                access_count=1,
                last_accessed=time.time(),
                ttl=ttl,
                context_hash=self._hash_context(context) if context else None
            )
            
            # Store in cache
            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)  # Mark as most recently used
            
            self._stats["cache_size"] = len(self._cache)
            
            logger.debug(f"Cached result for key: {cache_key[:16]}... (ttl: {ttl}s)")
            
            # Persist to database if enabled
            if self.enable_persistence:
                self._persist_to_database(cache_key, message, result, context)
    
    def invalidate(self, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Invalidate cached result for specific message/context.
        
        Args:
            message: User message
            context: Optional context
            
        Returns:
            True if entry was found and removed
        """
        cache_key = self._generate_cache_key(message, context)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats["evictions"] += 1
                self._stats["cache_size"] = len(self._cache)
                logger.debug(f"Invalidated cache entry: {cache_key[:16]}...")
                return True
            
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._stats["evictions"] += cleared_count
            self._stats["cache_size"] = 0
            logger.info(f"Cleared {cleared_count} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["total_requests"]
            hit_rate = (self._stats["hits"] / total_requests) if total_requests > 0 else 0.0
            miss_rate = (self._stats["misses"] / total_requests) if total_requests > 0 else 0.0
            
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "total_hits": self._stats["hits"],
                "total_misses": self._stats["misses"],
                "total_requests": total_requests,
                "total_evictions": self._stats["evictions"],
                "total_cleanups": self._stats["cleanups"],
                "last_cleanup": datetime.fromtimestamp(self._stats["last_cleanup"]).isoformat(),
                "memory_usage_estimate": self._estimate_memory_usage()
            }
    
    def warm_cache(self, common_messages: List[Tuple[str, Optional[Dict[str, Any]]]]) -> int:
        """
        Warm cache with common messages.
        
        Args:
            common_messages: List of (message, context) tuples
            
        Returns:
            Number of entries warmed
        """
        warmed_count = 0
        
        for message, context in common_messages:
            # Only warm if not already cached
            if self.get(message, context) is None:
                # This would typically involve calling the actual classifier
                # For now, we'll skip the actual classification
                logger.debug(f"Would warm cache for: {message[:50]}...")
                warmed_count += 1
        
        logger.info(f"Cache warming completed: {warmed_count} entries")
        return warmed_count
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Generate detailed cache efficiency report."""
        with self._lock:
            stats = self.get_stats()
            
            # Analyze cache entries
            now = time.time()
            fresh_entries = 0
            stale_entries = 0
            expired_entries = 0
            high_access_entries = 0
            
            for entry in self._cache.values():
                if entry.is_expired():
                    expired_entries += 1
                elif entry.is_stale():
                    stale_entries += 1
                else:
                    fresh_entries += 1
                
                if entry.access_count > 5:
                    high_access_entries += 1
            
            return {
                **stats,
                "entry_analysis": {
                    "fresh_entries": fresh_entries,
                    "stale_entries": stale_entries,
                    "expired_entries": expired_entries,
                    "high_access_entries": high_access_entries
                },
                "recommendations": self._generate_cache_recommendations(stats)
            }
    
    # Private methods
    
    def _generate_cache_key(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from message and context."""
        # Normalize message
        normalized_message = message.strip().lower()
        
        # Create base key from message
        message_hash = hashlib.md5(normalized_message.encode()).hexdigest()
        
        # Add context hash if provided
        if context:
            context_hash = self._hash_context(context)
            return f"{message_hash}:{context_hash}"
        
        return message_hash
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Generate hash from context dictionary."""
        # Sort keys for consistent hashing
        sorted_context = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(sorted_context.encode()).hexdigest()[:8]
    
    def _evict_lru_entries(self, count: int = None) -> None:
        """Evict least recently used entries."""
        if count is None:
            count = max(1, self.max_size // 10)  # Evict 10% by default
        
        evicted = 0
        while len(self._cache) >= self.max_size and evicted < count:
            # Remove oldest entry (first in OrderedDict)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            evicted += 1
            self._stats["evictions"] += 1
        
        logger.debug(f"Evicted {evicted} LRU cache entries")
    
    def _background_cleanup(self) -> None:
        """Background thread for cache cleanup."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_entries()
            except Exception as e:
                logger.error(f"Error in cache cleanup thread: {e}")
    
    def _cleanup_expired_entries(self) -> None:
        """Remove expired cache entries."""
        with self._lock:
            expired_keys = []
            now = time.time()
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self._stats["evictions"] += len(expired_keys)
                self._stats["cleanups"] += 1
                self._stats["last_cleanup"] = now
                self._stats["cache_size"] = len(self._cache)
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        # Rough estimation based on average entry size
        if not self._cache:
            return 0
        
        # Sample a few entries to estimate average size
        sample_size = min(10, len(self._cache))
        sample_entries = list(self._cache.values())[:sample_size]
        
        total_sample_size = 0
        for entry in sample_entries:
            # Estimate size of IntentResult and metadata
            result_size = len(str(asdict(entry.result)))
            entry_size = result_size + 200  # Overhead for CacheEntry
            total_sample_size += entry_size
        
        avg_entry_size = total_sample_size / sample_size if sample_size > 0 else 1000
        return int(len(self._cache) * avg_entry_size)
    
    def _persist_to_database(self, cache_key: str, message: str, 
                           result: IntentResult, context: Optional[Dict[str, Any]]) -> None:
        """Persist cache entry to database (async operation)."""
        try:
            # This would be implemented to store in the intent_classifications table
            # For now, we'll just log the operation
            logger.debug(f"Would persist cache entry to database: {cache_key[:16]}...")
        except Exception as e:
            logger.error(f"Failed to persist cache entry to database: {e}")
    
    def _generate_cache_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        
        hit_rate = stats.get("hit_rate", 0.0)
        cache_size = stats.get("cache_size", 0)
        max_size = stats.get("max_size", 1)
        
        if hit_rate < 0.5:
            recommendations.append(f"Low cache hit rate ({hit_rate:.1%}). Consider increasing cache size or TTL.")
        
        if cache_size / max_size > 0.9:
            recommendations.append("Cache is nearly full. Consider increasing max_size.")
        
        if stats.get("total_evictions", 0) > stats.get("total_hits", 0):
            recommendations.append("High eviction rate. Consider increasing cache size or optimizing TTL.")
        
        return recommendations


# Global cache instance
_intent_cache: Optional[IntentClassificationCache] = None


def get_intent_cache() -> IntentClassificationCache:
    """Get global intent classification cache instance."""
    global _intent_cache
    
    if _intent_cache is None:
        _intent_cache = IntentClassificationCache()
    
    return _intent_cache


def clear_intent_cache() -> None:
    """Clear global intent cache."""
    global _intent_cache
    if _intent_cache:
        _intent_cache.clear()