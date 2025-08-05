"""
Query optimization and indexing for cart database operations.
Implements intelligent query optimization, index management, and performance analysis.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from .database import get_database_manager
from ..monitoring.performance_monitor import get_performance_monitor, performance_track
from ..monitoring.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Database query execution plan analysis."""
    
    query: str
    estimated_cost: float
    actual_time: Optional[float]
    rows_examined: int
    rows_returned: int
    index_usage: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "estimated_cost": self.estimated_cost,
            "actual_time": self.actual_time,
            "rows_examined": self.rows_examined,
            "rows_returned": self.rows_returned,
            "index_usage": self.index_usage,
            "recommendations": self.recommendations
        }


@dataclass
class IndexRecommendation:
    """Index creation recommendation."""
    
    table_name: str
    columns: List[str]
    index_type: str  # 'btree', 'hash', 'gin', etc.
    estimated_benefit: float
    query_patterns: List[str]
    creation_sql: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_name": self.table_name,
            "columns": self.columns,
            "index_type": self.index_type,
            "estimated_benefit": self.estimated_benefit,
            "query_patterns": self.query_patterns,
            "creation_sql": self.creation_sql
        }


class QueryOptimizer:
    """Intelligent query optimization and index management system."""
    
    def __init__(self):
        """Initialize query optimizer."""
        self.db_manager = get_database_manager()
        self.perf_monitor = get_performance_monitor()
        self.metrics_collector = get_metrics_collector()
        
        # Query pattern analysis
        self._query_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._slow_queries: List[Dict[str, Any]] = []
        
        # Index management
        self._existing_indexes: Dict[str, List[str]] = {}
        self._index_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Optimization cache
        self._optimization_cache: Dict[str, str] = {}
        
        # Performance thresholds
        self.slow_query_threshold = 1.0  # seconds
        self.high_cost_threshold = 1000.0
        
        logger.info("Query optimizer initialized")
    
    @performance_track("query_optimization")
    def optimize_query(self, query: str, params: Optional[tuple] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize a SQL query for better performance.
        
        Args:
            query: SQL query to optimize
            params: Query parameters
            
        Returns:
            Tuple of (optimized_query, optimization_info)
        """
        # Check cache first
        query_hash = self._hash_query(query)
        if query_hash in self._optimization_cache:
            cached_query = self._optimization_cache[query_hash]
            return cached_query, {"source": "cache", "original_query": query}
        
        # Analyze query
        analysis = self._analyze_query(query, params)
        
        # Apply optimizations
        optimized_query = self._apply_optimizations(query, analysis)
        
        # Cache result
        self._optimization_cache[query_hash] = optimized_query
        
        optimization_info = {
            "original_query": query,
            "optimizations_applied": analysis.get("optimizations", []),
            "estimated_improvement": analysis.get("estimated_improvement", 0.0),
            "analysis": analysis
        }
        
        return optimized_query, optimization_info
    
    @performance_track("query_execution_analysis")
    def execute_with_analysis(self, query: str, params: Optional[tuple] = None) -> Tuple[List[Dict[str, Any]], QueryPlan]:
        """
        Execute query with performance analysis.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Tuple of (results, query_plan)
        """
        start_time = time.time()
        
        # Get execution plan
        plan = self._get_execution_plan(query, params)
        
        # Execute query
        results = self.db_manager.execute_query(query, params)
        
        # Update plan with actual execution time
        execution_time = time.time() - start_time
        plan.actual_time = execution_time
        
        # Record metrics
        self.metrics_collector.record_timer("optimized_query_time", execution_time)
        
        # Track slow queries
        if execution_time > self.slow_query_threshold:
            self._track_slow_query(query, params, execution_time, plan)
        
        # Update query patterns
        self._update_query_patterns(query, execution_time, plan)
        
        return results, plan
    
    def analyze_cart_queries(self) -> Dict[str, Any]:
        """Analyze cart-specific query performance."""
        cart_queries = [
            # Shopping cart queries
            "SELECT * FROM shopping_cart WHERE session_id = %s",
            "INSERT INTO shopping_cart (session_id, product_id, product_title, quantity, product_price) VALUES (%s, %s, %s, %s, %s)",
            "UPDATE shopping_cart SET quantity = %s WHERE session_id = %s AND product_id = %s",
            "DELETE FROM shopping_cart WHERE session_id = %s AND product_id = %s",
            "SELECT COUNT(*), SUM(quantity * product_price) FROM shopping_cart WHERE session_id = %s",
            
            # Cart session queries
            "SELECT * FROM cart_sessions WHERE session_id = %s",
            "UPDATE cart_sessions SET total_items = %s, total_value = %s WHERE session_id = %s",
            
            # Intent classification queries
            "SELECT * FROM intent_classifications WHERE message_text = %s AND context_hash = %s",
            "INSERT INTO intent_classifications (message_text, classified_intent, confidence_score, context_hash) VALUES (%s, %s, %s, %s)"
        ]
        
        analysis_results = {}
        
        for query in cart_queries:
            query_name = self._get_query_name(query)
            
            # Analyze query structure
            structure_analysis = self._analyze_query_structure(query)
            
            # Get execution plan (with dummy parameters)
            dummy_params = self._generate_dummy_params(query)
            plan = self._get_execution_plan(query, dummy_params)
            
            analysis_results[query_name] = {
                "query": query,
                "structure_analysis": structure_analysis,
                "execution_plan": plan.to_dict(),
                "optimization_recommendations": self._get_query_recommendations(query, plan)
            }
        
        return analysis_results
    
    def recommend_indexes(self) -> List[IndexRecommendation]:
        """Generate index recommendations based on query patterns."""
        recommendations = []
        
        # Analyze shopping cart table
        cart_recommendations = self._analyze_table_indexes("shopping_cart")
        recommendations.extend(cart_recommendations)
        
        # Analyze cart sessions table
        session_recommendations = self._analyze_table_indexes("cart_sessions")
        recommendations.extend(session_recommendations)
        
        # Analyze intent classifications table
        intent_recommendations = self._analyze_table_indexes("intent_classifications")
        recommendations.extend(intent_recommendations)
        
        # Analyze conversation tables
        conv_recommendations = self._analyze_table_indexes("conversations")
        recommendations.extend(conv_recommendations)
        
        msg_recommendations = self._analyze_table_indexes("conversation_messages")
        recommendations.extend(msg_recommendations)
        
        return recommendations
    
    def create_recommended_indexes(self, recommendations: List[IndexRecommendation]) -> Dict[str, bool]:
        """Create recommended indexes."""
        results = {}
        
        for rec in recommendations:
            try:
                # Check if index already exists
                if self._index_exists(rec.table_name, rec.columns):
                    results[f"{rec.table_name}_{'-'.join(rec.columns)}"] = True
                    continue
                
                # Create index
                self.db_manager.execute_update(rec.creation_sql)
                results[f"{rec.table_name}_{'-'.join(rec.columns)}"] = True
                
                logger.info(f"Created index on {rec.table_name}({', '.join(rec.columns)})")
                
            except Exception as e:
                logger.error(f"Failed to create index on {rec.table_name}: {e}")
                results[f"{rec.table_name}_{'-'.join(rec.columns)}"] = False
        
        # Refresh index cache
        self._refresh_index_cache()
        
        return results
    
    def get_index_usage_stats(self) -> Dict[str, Any]:
        """Get index usage statistics."""
        try:
            # Query PostgreSQL statistics
            stats_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_tup_read,
                idx_tup_fetch,
                idx_scan
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
            """
            
            results = self.db_manager.execute_query(stats_query)
            
            # Organize by table
            table_stats = {}
            for row in results:
                table_name = row["tablename"]
                if table_name not in table_stats:
                    table_stats[table_name] = []
                
                table_stats[table_name].append({
                    "index_name": row["indexname"],
                    "tuples_read": row["idx_tup_read"],
                    "tuples_fetched": row["idx_tup_fetch"],
                    "scans": row["idx_scan"]
                })
            
            return table_stats
            
        except Exception as e:
            logger.error(f"Failed to get index usage stats: {e}")
            return {}
    
    def optimize_cart_schema(self) -> Dict[str, Any]:
        """Optimize shopping cart database schema."""
        optimizations = []
        
        # Check current indexes
        current_indexes = self._get_table_indexes("shopping_cart")
        
        # Essential indexes for cart operations
        essential_indexes = [
            ("shopping_cart", ["session_id"], "btree"),
            ("shopping_cart", ["session_id", "product_id"], "btree"),
            ("shopping_cart", ["updated_at"], "btree"),
            ("cart_sessions", ["session_id"], "btree"),
            ("cart_sessions", ["last_updated"], "btree"),
            ("intent_classifications", ["message_text"], "hash"),
            ("intent_classifications", ["context_hash"], "btree"),
            ("intent_classifications", ["created_at"], "btree")
        ]
        
        # Create missing indexes
        for table, columns, index_type in essential_indexes:
            if not self._index_exists(table, columns):
                index_name = f"idx_{table}_{'_'.join(columns)}"
                create_sql = f"CREATE INDEX {index_name} ON {table} USING {index_type} ({', '.join(columns)})"
                
                try:
                    self.db_manager.execute_update(create_sql)
                    optimizations.append(f"Created index: {index_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to create index {index_name}: {e}")
        
        # Analyze table statistics
        self._analyze_table_statistics()
        
        return {
            "optimizations_applied": optimizations,
            "index_recommendations": self.recommend_indexes(),
            "query_analysis": self.analyze_cart_queries()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive query performance report."""
        return {
            "slow_queries": self._slow_queries[-10:],  # Last 10 slow queries
            "query_patterns": self._get_query_pattern_summary(),
            "index_usage": self.get_index_usage_stats(),
            "optimization_cache_stats": {
                "cached_queries": len(self._optimization_cache),
                "cache_hit_rate": self._calculate_cache_hit_rate()
            },
            "recommendations": {
                "indexes": [rec.to_dict() for rec in self.recommend_indexes()],
                "query_optimizations": self._get_optimization_recommendations()
            }
        }
    
    # Private methods
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching."""
        import hashlib
        normalized_query = re.sub(r'\s+', ' ', query.strip().lower())
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def _analyze_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """Analyze query for optimization opportunities."""
        analysis = {
            "query_type": self._get_query_type(query),
            "tables_involved": self._extract_tables(query),
            "where_conditions": self._extract_where_conditions(query),
            "joins": self._extract_joins(query),
            "optimizations": [],
            "estimated_improvement": 0.0
        }
        
        # Check for common optimization opportunities
        if "SELECT *" in query.upper():
            analysis["optimizations"].append("Replace SELECT * with specific columns")
            analysis["estimated_improvement"] += 0.1
        
        if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
            analysis["optimizations"].append("Consider adding LIMIT clause")
            analysis["estimated_improvement"] += 0.05
        
        # Check for missing indexes
        for table in analysis["tables_involved"]:
            if not self._has_adequate_indexes(table, analysis["where_conditions"]):
                analysis["optimizations"].append(f"Consider adding index on {table}")
                analysis["estimated_improvement"] += 0.2
        
        return analysis
    
    def _apply_optimizations(self, query: str, analysis: Dict[str, Any]) -> str:
        """Apply query optimizations."""
        optimized_query = query
        
        # Apply specific optimizations based on analysis
        for optimization in analysis["optimizations"]:
            if "SELECT *" in optimization:
                # This would require more sophisticated parsing
                # For now, we'll leave the query as-is
                pass
            elif "LIMIT" in optimization:
                if "ORDER BY" in optimized_query.upper() and "LIMIT" not in optimized_query.upper():
                    optimized_query += " LIMIT 1000"  # Default limit
        
        return optimized_query
    
    def _get_execution_plan(self, query: str, params: Optional[tuple] = None) -> QueryPlan:
        """Get query execution plan."""
        try:
            # Use EXPLAIN ANALYZE for detailed plan
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            
            results = self.db_manager.execute_query(explain_query, params)
            
            if results:
                plan_data = results[0]["QUERY PLAN"][0]
                
                return QueryPlan(
                    query=query,
                    estimated_cost=plan_data.get("Total Cost", 0.0),
                    actual_time=plan_data.get("Actual Total Time", 0.0) / 1000.0,  # Convert to seconds
                    rows_examined=plan_data.get("Actual Rows", 0),
                    rows_returned=plan_data.get("Actual Rows", 0),
                    index_usage=self._extract_index_usage(plan_data),
                    recommendations=[]
                )
            
        except Exception as e:
            logger.warning(f"Failed to get execution plan: {e}")
        
        # Return basic plan if EXPLAIN fails
        return QueryPlan(
            query=query,
            estimated_cost=0.0,
            actual_time=None,
            rows_examined=0,
            rows_returned=0,
            index_usage=[],
            recommendations=[]
        )
    
    def _track_slow_query(self, query: str, params: Optional[tuple], 
                         execution_time: float, plan: QueryPlan) -> None:
        """Track slow query for analysis."""
        slow_query_info = {
            "query": query,
            "params": str(params) if params else None,
            "execution_time": execution_time,
            "timestamp": datetime.utcnow().isoformat(),
            "plan": plan.to_dict()
        }
        
        self._slow_queries.append(slow_query_info)
        
        # Keep only recent slow queries
        if len(self._slow_queries) > 100:
            self._slow_queries = self._slow_queries[-50:]
        
        logger.warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}...")
    
    def _update_query_patterns(self, query: str, execution_time: float, plan: QueryPlan) -> None:
        """Update query pattern statistics."""
        pattern = self._get_query_pattern(query)
        
        if pattern not in self._query_patterns:
            self._query_patterns[pattern] = []
        
        self._query_patterns[pattern].append({
            "execution_time": execution_time,
            "timestamp": time.time(),
            "cost": plan.estimated_cost
        })
        
        # Keep only recent patterns
        cutoff_time = time.time() - 3600  # Last hour
        self._query_patterns[pattern] = [
            p for p in self._query_patterns[pattern] 
            if p["timestamp"] > cutoff_time
        ]
    
    def _analyze_table_indexes(self, table_name: str) -> List[IndexRecommendation]:
        """Analyze and recommend indexes for a specific table."""
        recommendations = []
        
        # Get current indexes
        current_indexes = self._get_table_indexes(table_name)
        
        # Table-specific recommendations
        if table_name == "shopping_cart":
            # Session-based queries
            if not self._has_index(current_indexes, ["session_id"]):
                recommendations.append(IndexRecommendation(
                    table_name=table_name,
                    columns=["session_id"],
                    index_type="btree",
                    estimated_benefit=0.8,
                    query_patterns=["SELECT * FROM shopping_cart WHERE session_id = ?"],
                    creation_sql=f"CREATE INDEX idx_{table_name}_session_id ON {table_name} (session_id)"
                ))
            
            # Composite index for cart operations
            if not self._has_index(current_indexes, ["session_id", "product_id"]):
                recommendations.append(IndexRecommendation(
                    table_name=table_name,
                    columns=["session_id", "product_id"],
                    index_type="btree",
                    estimated_benefit=0.9,
                    query_patterns=["UPDATE/DELETE WHERE session_id = ? AND product_id = ?"],
                    creation_sql=f"CREATE UNIQUE INDEX idx_{table_name}_session_product ON {table_name} (session_id, product_id)"
                ))
        
        elif table_name == "intent_classifications":
            # Message text hash index
            if not self._has_index(current_indexes, ["message_text"]):
                recommendations.append(IndexRecommendation(
                    table_name=table_name,
                    columns=["message_text"],
                    index_type="hash",
                    estimated_benefit=0.7,
                    query_patterns=["SELECT * WHERE message_text = ?"],
                    creation_sql=f"CREATE INDEX idx_{table_name}_message_text ON {table_name} USING hash (message_text)"
                ))
        
        return recommendations
    
    def _get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get existing indexes for a table."""
        try:
            query = """
            SELECT 
                indexname,
                indexdef
            FROM pg_indexes 
            WHERE tablename = %s
            """
            
            return self.db_manager.execute_query(query, (table_name,))
            
        except Exception as e:
            logger.error(f"Failed to get indexes for {table_name}: {e}")
            return []
    
    def _index_exists(self, table_name: str, columns: List[str]) -> bool:
        """Check if an index exists on specified columns."""
        indexes = self._get_table_indexes(table_name)
        
        for index in indexes:
            index_def = index["indexdef"].lower()
            if all(col.lower() in index_def for col in columns):
                return True
        
        return False
    
    def _has_index(self, indexes: List[Dict[str, Any]], columns: List[str]) -> bool:
        """Check if indexes list contains index on specified columns."""
        for index in indexes:
            index_def = index["indexdef"].lower()
            if all(col.lower() in index_def for col in columns):
                return True
        return False
    
    def _refresh_index_cache(self) -> None:
        """Refresh cached index information."""
        self._existing_indexes.clear()
        
        # Get all table indexes
        tables = ["shopping_cart", "cart_sessions", "intent_classifications", 
                 "conversations", "conversation_messages"]
        
        for table in tables:
            self._existing_indexes[table] = self._get_table_indexes(table)
    
    def _get_query_type(self, query: str) -> str:
        """Determine query type (SELECT, INSERT, UPDATE, DELETE)."""
        query_upper = query.strip().upper()
        if query_upper.startswith("SELECT"):
            return "SELECT"
        elif query_upper.startswith("INSERT"):
            return "INSERT"
        elif query_upper.startswith("UPDATE"):
            return "UPDATE"
        elif query_upper.startswith("DELETE"):
            return "DELETE"
        else:
            return "OTHER"
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query."""
        # Simple regex-based extraction
        # In production, would use a proper SQL parser
        tables = []
        
        # Look for FROM and JOIN clauses
        from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))
        
        join_matches = re.findall(r'JOIN\s+(\w+)', query, re.IGNORECASE)
        tables.extend(join_matches)
        
        return list(set(tables))
    
    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions from query."""
        conditions = []
        
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            # Simple condition extraction
            condition_parts = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
            conditions.extend([c.strip() for c in condition_parts])
        
        return conditions
    
    def _extract_joins(self, query: str) -> List[str]:
        """Extract JOIN information from query."""
        joins = re.findall(r'((?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+\w+\s+ON\s+[^)]+)', query, re.IGNORECASE)
        return [j.strip() for j in joins]
    
    def _has_adequate_indexes(self, table: str, conditions: List[str]) -> bool:
        """Check if table has adequate indexes for conditions."""
        # Simplified check - in production would be more sophisticated
        if table not in self._existing_indexes:
            return False
        
        # Check if any condition columns have indexes
        for condition in conditions:
            # Extract column name from condition (simplified)
            column_match = re.search(r'(\w+)\s*[=<>]', condition)
            if column_match:
                column = column_match.group(1)
                if any(column.lower() in idx["indexdef"].lower() 
                      for idx in self._existing_indexes[table]):
                    return True
        
        return False
    
    def _get_query_name(self, query: str) -> str:
        """Generate a descriptive name for a query."""
        query_type = self._get_query_type(query)
        tables = self._extract_tables(query)
        table_str = "_".join(tables) if tables else "unknown"
        return f"{query_type.lower()}_{table_str}"
    
    def _get_query_pattern(self, query: str) -> str:
        """Get query pattern for grouping similar queries."""
        # Normalize query by removing specific values
        pattern = re.sub(r"'[^']*'", "'?'", query)
        pattern = re.sub(r'\b\d+\b', '?', pattern)
        pattern = re.sub(r'%s', '?', pattern)
        return pattern.strip()
    
    def _analyze_query_structure(self, query: str) -> Dict[str, Any]:
        """Analyze query structure for optimization opportunities."""
        return {
            "query_type": self._get_query_type(query),
            "tables": self._extract_tables(query),
            "has_where": "WHERE" in query.upper(),
            "has_order_by": "ORDER BY" in query.upper(),
            "has_limit": "LIMIT" in query.upper(),
            "has_joins": "JOIN" in query.upper(),
            "estimated_complexity": self._estimate_query_complexity(query)
        }
    
    def _estimate_query_complexity(self, query: str) -> str:
        """Estimate query complexity."""
        complexity_score = 0
        
        if "JOIN" in query.upper():
            complexity_score += 2
        if "ORDER BY" in query.upper():
            complexity_score += 1
        if "GROUP BY" in query.upper():
            complexity_score += 2
        if "HAVING" in query.upper():
            complexity_score += 1
        
        if complexity_score == 0:
            return "simple"
        elif complexity_score <= 2:
            return "moderate"
        else:
            return "complex"
    
    def _generate_dummy_params(self, query: str) -> Optional[tuple]:
        """Generate dummy parameters for query analysis."""
        param_count = query.count('%s')
        if param_count == 0:
            return None
        
        # Generate appropriate dummy values
        dummy_values = []
        for _ in range(param_count):
            dummy_values.append("dummy_value")
        
        return tuple(dummy_values)
    
    def _get_query_recommendations(self, query: str, plan: QueryPlan) -> List[str]:
        """Get optimization recommendations for a query."""
        recommendations = []
        
        if plan.estimated_cost > self.high_cost_threshold:
            recommendations.append("High cost query - consider optimization")
        
        if not plan.index_usage:
            recommendations.append("No indexes used - consider adding indexes")
        
        if "SELECT *" in query.upper():
            recommendations.append("Avoid SELECT * - specify needed columns")
        
        return recommendations
    
    def _extract_index_usage(self, plan_data: Dict[str, Any]) -> List[str]:
        """Extract index usage from execution plan."""
        # Simplified extraction - would be more sophisticated in production
        indexes = []
        
        def extract_from_node(node):
            if isinstance(node, dict):
                if "Index Name" in node:
                    indexes.append(node["Index Name"])
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        extract_from_node(value)
            elif isinstance(node, list):
                for item in node:
                    extract_from_node(item)
        
        extract_from_node(plan_data)
        return indexes
    
    def _analyze_table_statistics(self) -> None:
        """Analyze table statistics for optimization."""
        try:
            stats_query = """
            SELECT 
                schemaname,
                tablename,
                n_tup_ins,
                n_tup_upd,
                n_tup_del,
                n_live_tup,
                n_dead_tup,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            """
            
            results = self.db_manager.execute_query(stats_query)
            
            for row in results:
                table_name = row["tablename"]
                
                # Check if table needs maintenance
                dead_ratio = row["n_dead_tup"] / max(row["n_live_tup"], 1)
                if dead_ratio > 0.1:  # More than 10% dead tuples
                    logger.info(f"Table {table_name} may need VACUUM (dead ratio: {dead_ratio:.2f})")
                
        except Exception as e:
            logger.error(f"Failed to analyze table statistics: {e}")
    
    def _get_query_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of query patterns."""
        summary = {}
        
        for pattern, executions in self._query_patterns.items():
            if executions:
                times = [e["execution_time"] for e in executions]
                summary[pattern] = {
                    "count": len(executions),
                    "avg_time": sum(times) / len(times),
                    "max_time": max(times),
                    "min_time": min(times)
                }
        
        return summary
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate optimization cache hit rate."""
        # This would be tracked in a real implementation
        return 0.75  # Placeholder
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get general optimization recommendations."""
        recommendations = []
        
        # Analyze slow queries
        if len(self._slow_queries) > 10:
            recommendations.append("High number of slow queries detected - review query patterns")
        
        # Analyze cache performance
        if self._calculate_cache_hit_rate() < 0.5:
            recommendations.append("Low optimization cache hit rate - consider tuning")
        
        return recommendations


# Global query optimizer instance
_query_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer
    
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    
    return _query_optimizer