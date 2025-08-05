# Troubleshooting Guide: Shopping Cart and Routing Issues

## Overview

This guide provides comprehensive troubleshooting procedures for common issues encountered with the shopping cart agent system, including router node problems, cart operations failures, database issues, and performance problems.

## Quick Diagnostic Commands

### System Health Check

```python
# Run comprehensive system health check
from src.langgraph_integration.state.migrations import check_migration_status
from src.langgraph_integration.state.database import get_database_manager
from src.langgraph_integration.state.shopping_cart_manager import ShoppingCartManager

# Check database connection
db_manager = get_database_manager()
print(f"Database connection: {'OK' if db_manager.test_connection() else 'FAILED'}")

# Check migration status
migration_status = check_migration_status()
print(f"Migrations: {migration_status['successful_migrations']}/{migration_status['total_migrations']} successful")

# Test cart operations
cart_manager = ShoppingCartManager(db_manager)
test_result = cart_manager.add_item("health_test", "test_product", "Test Product", 1)
print(f"Cart operations: {'OK' if test_result.get('success') else 'FAILED'}")
```

### Log Analysis Commands

```bash
# Check recent errors in logs
tail -n 100 logs/application.log | grep -i error

# Check router-specific issues
grep -i "router\|intent\|classification" logs/application.log | tail -20

# Check cart-specific issues
grep -i "cart\|shopping" logs/application.log | tail -20

# Check database connection issues
grep -i "database\|connection\|postgres" logs/application.log | tail -20
```

## Router Node Issues

### Issue: Intent Classification Failures

#### Symptoms
- Users receive "I'm not sure what you're asking" responses frequently
- Router always defaults to clarification mode
- High error rates in router logs

#### Diagnostic Steps

```python
# Test intent classification directly
from src.langgraph_integration.core.router.intent_classifier import IntentClassifier

classifier = IntentClassifier()

# Test with known patterns
test_messages = [
    "add laptop to cart",
    "find gaming laptops",
    "show my cart",
    "compare these products"
]

for message in test_messages:
    result = classifier.classify_intent(message, {})
    print(f"Message: '{message}' -> Intent: {result.intent}, Confidence: {result.confidence}")
```

#### Common Causes and Solutions

**Cause 1: Low Confidence Threshold**
```python
# Check current threshold
from src.langgraph_integration.core.router.router_node import RouterNode

router = RouterNode()
print(f"Current confidence threshold: {router.confidence_threshold}")

# Adjust threshold if too high
router.confidence_threshold = 0.6  # Lower from default 0.7
```

**Cause 2: Missing or Corrupted Intent Patterns**
```python
# Verify intent patterns are loaded
classifier = IntentClassifier()
print(f"Cart patterns: {classifier.cart_patterns}")
print(f"QA patterns: {classifier.qa_patterns}")

# Reload patterns if needed
classifier._load_patterns()
```

**Cause 3: Context Extraction Errors**
```python
# Test context extraction
from src.langgraph_integration.core.router.router_node import RouterNode

router = RouterNode()
test_state = {
    "messages": [{"content": "I want to buy a laptop"}],
    "current_query": "add it to cart"
}

context = router._extract_context(test_state)
print(f"Extracted context: {context}")
```

#### Solutions

1. **Adjust Confidence Threshold**
```python
# In configuration
router_config = {
    "confidence_threshold": 0.6,  # Lower threshold
    "enable_keyword_fallback": True
}
```

2. **Update Intent Patterns**
```python
# Add more patterns to intent classifier
additional_cart_patterns = [
    "purchase", "buy", "get", "want", "need"
]
classifier.cart_patterns.extend(additional_cart_patterns)
```

3. **Enable Debug Logging**
```python
import logging
logging.getLogger('src.langgraph_integration.core.router').setLevel(logging.DEBUG)
```

### Issue: Routing Loops or Infinite Clarification

#### Symptoms
- System keeps asking for clarification repeatedly
- Users stuck in clarification loop
- High clarification attempt counts

#### Diagnostic Steps

```python
# Check clarification attempt tracking
from src.langgraph_integration.core.router.clarification_handler import ClarificationHandler

handler = ClarificationHandler()

# Check session clarification history
test_session = "problematic_session_id"
attempts = handler.get_clarification_attempts(test_session)
print(f"Clarification attempts for session {test_session}: {attempts}")

# Check max attempts setting
print(f"Max clarification attempts: {handler.max_clarification_attempts}")
```

#### Solutions

1. **Reduce Max Clarification Attempts**
```python
# Configure lower max attempts
clarification_config = {
    "max_clarification_attempts": 2,  # Reduce from default 3
    "fallback_to_qa_after_max": True
}
```

2. **Implement Fallback Strategy**
```python
# Ensure fallback to QA agent after max attempts
def handle_max_clarification_reached(state):
    state["routing_decision"] = "qa"
    state["clarification_exhausted"] = True
    return state
```

3. **Clear Clarification History**
```python
# Reset clarification attempts for problematic sessions
handler.clear_clarification_history("session_id")
```

### Issue: Agent Routing Errors

#### Symptoms
- Requests routed to wrong agent
- Agent not found errors
- Routing decision inconsistencies

#### Diagnostic Steps

```python
# Test routing decisions
from src.langgraph_integration.core.router.master_graph import MasterAgentGraph

master_graph = MasterAgentGraph()

# Test routing for different message types
test_cases = [
    ("add laptop to cart", "cart"),
    ("find gaming laptops", "qa"),
    ("what's in my cart", "cart"),
    ("compare these products", "qa")
]

for message, expected_route in test_cases:
    state = {"current_query": message, "messages": []}
    result = master_graph._route_decision(state)
    print(f"Message: '{message}' -> Expected: {expected_route}, Actual: {result}")
```

#### Solutions

1. **Verify Agent Registration**
```python
# Check available agents
master_graph = MasterAgentGraph()
print(f"Available agents: {list(master_graph.agents.keys())}")

# Ensure all required agents are registered
required_agents = ["qa", "cart"]
missing_agents = [agent for agent in required_agents if agent not in master_graph.agents]
if missing_agents:
    print(f"Missing agents: {missing_agents}")
```

2. **Update Routing Logic**
```python
# Fix routing decision logic
def _route_decision(self, state):
    routing_decision = state.get("routing_decision", "clarification")
    
    # Validate routing decision
    valid_routes = ["qa", "cart", "clarification"]
    if routing_decision not in valid_routes:
        logging.warning(f"Invalid routing decision: {routing_decision}, defaulting to qa")
        return "qa"
    
    return routing_decision
```

## Shopping Cart Issues

### Issue: Cart Operations Failing

#### Symptoms
- "Cart operation failed" error messages
- Items not being added to cart
- Cart contents not displaying

#### Diagnostic Steps

```python
# Test cart manager directly
from src.langgraph_integration.state.shopping_cart_manager import ShoppingCartManager
from src.langgraph_integration.state.database import get_database_manager

cart_manager = ShoppingCartManager(get_database_manager())

# Test basic operations
test_session = "debug_session"

# Test add item
add_result = cart_manager.add_item(test_session, "test_product", "Test Product", 1)
print(f"Add item result: {add_result}")

# Test get cart
cart_contents = cart_manager.get_cart_contents(test_session)
print(f"Cart contents: {cart_contents}")

# Test remove item
remove_result = cart_manager.remove_item(test_session, "test_product")
print(f"Remove item result: {remove_result}")
```

#### Common Causes and Solutions

**Cause 1: Database Connection Issues**
```python
# Test database connection
db_manager = get_database_manager()
try:
    connection = db_manager.get_connection()
    print("Database connection: OK")
    connection.close()
except Exception as e:
    print(f"Database connection failed: {e}")
    
    # Check connection parameters
    print(f"Database config: {db_manager.config}")
```

**Cause 2: Missing Database Schema**
```python
# Verify cart tables exist
from src.langgraph_integration.state.migrations import validate_shopping_cart_schema

validation = validate_shopping_cart_schema()
print(f"Schema validation: {validation}")

# Run migrations if needed
if not all(validation.values()):
    from src.langgraph_integration.state.migrations import run_migrations
    migration_results = run_migrations()
    print(f"Migration results: {migration_results}")
```

**Cause 3: Invalid Product Data**
```python
# Test with valid product data
valid_product_data = {
    "session_id": "test_session",
    "product_id": "valid_product_123",
    "product_title": "Valid Test Product",
    "quantity": 1,
    "price": 99.99
}

result = cart_manager.add_item(**valid_product_data)
print(f"Valid data test: {result}")
```

#### Solutions

1. **Fix Database Connection**
```python
# Update database configuration
import os
os.environ['DATABASE_URL'] = 'postgresql://user:password@localhost:5432/dbname'

# Restart database connection pool
db_manager.close_all_connections()
db_manager = get_database_manager()
```

2. **Run Schema Migrations**
```bash
# Run migration script
python -m src.langgraph_integration.state.migrations
```

3. **Validate Input Data**
```python
# Add input validation
def validate_cart_input(product_id, product_title, quantity):
    errors = []
    
    if not product_id or len(product_id.strip()) == 0:
        errors.append("Product ID is required")
    
    if not product_title or len(product_title.strip()) == 0:
        errors.append("Product title is required")
    
    if quantity <= 0:
        errors.append("Quantity must be positive")
    
    return errors
```

### Issue: Cart Data Inconsistencies

#### Symptoms
- Cart totals don't match item counts
- Duplicate items in cart
- Cart summary out of sync

#### Diagnostic Steps

```python
# Check cart data consistency
def check_cart_consistency(session_id):
    cart_manager = ShoppingCartManager(get_database_manager())
    
    # Get cart items
    items = cart_manager.get_cart_contents(session_id)
    
    # Get cart summary
    summary = cart_manager.get_cart_summary(session_id)
    
    # Calculate expected totals
    expected_items = len(items)
    expected_quantity = sum(item.get('quantity', 0) for item in items)
    expected_value = sum(item.get('quantity', 0) * item.get('product_price', 0) for item in items if item.get('product_price'))
    
    print(f"Items count - Expected: {expected_items}, Actual: {summary.get('total_items', 0)}")
    print(f"Total quantity - Expected: {expected_quantity}, Actual: {summary.get('total_quantity', 0)}")
    print(f"Total value - Expected: {expected_value:.2f}, Actual: {summary.get('total_value', 0):.2f}")
    
    return {
        "items_match": expected_items == summary.get('total_items', 0),
        "quantity_match": expected_quantity == summary.get('total_quantity', 0),
        "value_match": abs(expected_value - summary.get('total_value', 0)) < 0.01
    }

# Test consistency
consistency = check_cart_consistency("test_session")
print(f"Consistency check: {consistency}")
```

#### Solutions

1. **Rebuild Cart Summary**
```sql
-- Manually rebuild cart session summaries
INSERT INTO cart_sessions (session_id, total_items, total_value, last_updated)
SELECT 
    session_id,
    COUNT(*) as total_items,
    COALESCE(SUM(quantity * COALESCE(product_price, 0)), 0) as total_value,
    CURRENT_TIMESTAMP
FROM shopping_cart 
GROUP BY session_id
ON CONFLICT (session_id) 
DO UPDATE SET
    total_items = EXCLUDED.total_items,
    total_value = EXCLUDED.total_value,
    last_updated = EXCLUDED.last_updated;
```

2. **Fix Duplicate Items**
```python
# Remove duplicate items (keep most recent)
def remove_duplicates(session_id):
    db_manager = get_database_manager()
    
    remove_duplicates_sql = """
    DELETE FROM shopping_cart 
    WHERE id NOT IN (
        SELECT DISTINCT ON (session_id, product_id) id
        FROM shopping_cart 
        WHERE session_id = %s
        ORDER BY session_id, product_id, updated_at DESC
    ) AND session_id = %s
    """
    
    db_manager.execute_update(remove_duplicates_sql, (session_id, session_id))
```

3. **Refresh Triggers**
```sql
-- Drop and recreate triggers if they're not working
DROP TRIGGER IF EXISTS update_cart_session_summary_trigger ON shopping_cart;

CREATE TRIGGER update_cart_session_summary_trigger
    AFTER INSERT OR UPDATE OR DELETE ON shopping_cart
    FOR EACH ROW
    EXECUTE FUNCTION update_cart_session_summary();
```

### Issue: Cart Performance Problems

#### Symptoms
- Slow cart operations
- Timeouts on cart queries
- High database load

#### Diagnostic Steps

```python
import time

# Measure cart operation performance
def measure_cart_performance(session_id, num_operations=10):
    cart_manager = ShoppingCartManager(get_database_manager())
    
    # Measure add operations
    start_time = time.time()
    for i in range(num_operations):
        cart_manager.add_item(session_id, f"perf_test_{i}", f"Performance Test {i}", 1)
    add_time = time.time() - start_time
    
    # Measure get operations
    start_time = time.time()
    for i in range(num_operations):
        cart_manager.get_cart_contents(session_id)
    get_time = time.time() - start_time
    
    # Measure remove operations
    start_time = time.time()
    for i in range(num_operations):
        cart_manager.remove_item(session_id, f"perf_test_{i}")
    remove_time = time.time() - start_time
    
    print(f"Add operations: {add_time:.3f}s ({add_time/num_operations*1000:.1f}ms per operation)")
    print(f"Get operations: {get_time:.3f}s ({get_time/num_operations*1000:.1f}ms per operation)")
    print(f"Remove operations: {remove_time:.3f}s ({remove_time/num_operations*1000:.1f}ms per operation)")

measure_cart_performance("perf_test_session")
```

#### Solutions

1. **Optimize Database Queries**
```sql
-- Add missing indexes
CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_updated ON shopping_cart(session_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_product_session ON shopping_cart(product_id, session_id);

-- Update table statistics
ANALYZE shopping_cart;
ANALYZE cart_sessions;
```

2. **Implement Connection Pooling**
```python
# Configure connection pooling
from src.langgraph_integration.state.connection_pool import ConnectionPool

# Initialize connection pool
pool = ConnectionPool(
    min_connections=5,
    max_connections=20,
    database_url=os.environ['DATABASE_URL']
)

# Use pooled connections
cart_manager = ShoppingCartManager(pool)
```

3. **Add Caching**
```python
# Implement cart caching
from functools import lru_cache
import time

class CachedCartManager:
    def __init__(self, cart_manager):
        self.cart_manager = cart_manager
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_cart_contents(self, session_id):
        cache_key = f"cart_{session_id}"
        now = time.time()
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return cached_data
        
        # Fetch from database
        data = self.cart_manager.get_cart_contents(session_id)
        self.cache[cache_key] = (data, now)
        return data
    
    def invalidate_cache(self, session_id):
        cache_key = f"cart_{session_id}"
        if cache_key in self.cache:
            del self.cache[cache_key]
```

## Database Issues

### Issue: Connection Pool Exhaustion

#### Symptoms
- "Connection pool exhausted" errors
- Timeouts on database operations
- High number of idle connections

#### Diagnostic Steps

```python
# Check connection pool status
from src.langgraph_integration.state.database import get_database_manager

db_manager = get_database_manager()
pool_stats = db_manager.get_pool_stats()
print(f"Pool stats: {pool_stats}")

# Check active connections
active_connections_sql = """
SELECT count(*) as active_connections
FROM pg_stat_activity 
WHERE state = 'active' AND datname = current_database()
"""

result = db_manager.execute_query(active_connections_sql)
print(f"Active connections: {result[0]['active_connections']}")
```

#### Solutions

1. **Increase Pool Size**
```python
# Configure larger connection pool
pool_config = {
    "min_connections": 10,
    "max_connections": 50,
    "connection_timeout": 30
}
```

2. **Fix Connection Leaks**
```python
# Ensure connections are properly closed
def safe_cart_operation(operation_func, *args, **kwargs):
    connection = None
    try:
        connection = db_manager.get_connection()
        return operation_func(connection, *args, **kwargs)
    finally:
        if connection:
            connection.close()
```

3. **Monitor Connection Usage**
```python
# Add connection monitoring
import logging

class MonitoredDatabaseManager:
    def __init__(self, base_manager):
        self.base_manager = base_manager
        self.connection_count = 0
    
    def get_connection(self):
        self.connection_count += 1
        logging.info(f"Connection acquired, total: {self.connection_count}")
        return self.base_manager.get_connection()
    
    def close_connection(self, connection):
        self.connection_count -= 1
        logging.info(f"Connection closed, total: {self.connection_count}")
        return self.base_manager.close_connection(connection)
```

### Issue: Database Lock Contention

#### Symptoms
- Deadlock errors
- Long-running transactions
- Cart operations timing out

#### Diagnostic Steps

```sql
-- Check for locks
SELECT 
    pg_stat_activity.pid,
    pg_stat_activity.query,
    pg_locks.mode,
    pg_locks.granted
FROM pg_stat_activity
JOIN pg_locks ON pg_stat_activity.pid = pg_locks.pid
WHERE pg_stat_activity.datname = current_database()
AND pg_locks.relation = 'shopping_cart'::regclass;

-- Check for long-running transactions
SELECT 
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
AND state = 'active';
```

#### Solutions

1. **Optimize Transaction Scope**
```python
# Keep transactions short
def add_item_optimized(self, session_id, product_id, product_title, quantity):
    try:
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Single atomic operation
                cursor.execute("""
                    INSERT INTO shopping_cart (session_id, product_id, product_title, quantity)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (session_id, product_id)
                    DO UPDATE SET quantity = shopping_cart.quantity + EXCLUDED.quantity
                """, (session_id, product_id, product_title, quantity))
                conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

2. **Use Advisory Locks**
```python
# Implement advisory locking for cart operations
def with_cart_lock(session_id, operation_func):
    lock_id = hash(session_id) % (2**31)  # Convert to 32-bit integer
    
    try:
        # Acquire advisory lock
        lock_sql = "SELECT pg_advisory_lock(%s)"
        db_manager.execute_query(lock_sql, (lock_id,))
        
        # Perform operation
        return operation_func()
        
    finally:
        # Release advisory lock
        unlock_sql = "SELECT pg_advisory_unlock(%s)"
        db_manager.execute_query(unlock_sql, (lock_id,))
```

## Frontend Integration Issues

### Issue: Cart Display Not Updating

#### Symptoms
- Cart sidebar shows stale data
- Cart changes not reflected in UI
- Frontend errors related to cart display

#### Diagnostic Steps

```python
# Test API response includes cart data
import requests

response = requests.post('/api/query', json={
    "query": "add laptop to cart",
    "session_id": "test_session",
    "include_cart_data": True
})

print(f"Response includes cart_data: {'cart_data' in response.json()}")
print(f"Cart updated flag: {response.json().get('cart_updated', False)}")
```

#### Solutions

1. **Verify API Response Format**
```python
# Ensure API returns cart data
class EnhancedQueryResponse:
    def __init__(self, query_response, cart_data=None):
        self.query = query_response.query
        self.response = query_response.response
        self.cart_data = cart_data
        self.cart_updated = cart_data is not None
        self.cart_item_count = len(cart_data.get('items', [])) if cart_data else 0
```

2. **Fix Frontend State Management**
```python
# In Streamlit app
def update_cart_display(api_response):
    if 'cart_data' in api_response and api_response['cart_updated']:
        st.session_state['cart_data'] = api_response['cart_data']
        st.rerun()  # Force UI refresh
```

3. **Add Error Handling**
```python
# Handle cart display errors gracefully
def safe_render_cart():
    try:
        cart_data = st.session_state.get('cart_data', {})
        render_cart_sidebar(cart_data)
    except Exception as e:
        st.error(f"Error displaying cart: {e}")
        st.session_state['cart_data'] = {}  # Reset to empty cart
```

## Performance Issues

### Issue: Slow Response Times

#### Symptoms
- Long delays in agent responses
- Timeouts on user requests
- High CPU or memory usage

#### Diagnostic Steps

```python
# Profile system performance
import cProfile
import time

def profile_query_processing():
    profiler = cProfile.Profile()
    
    # Profile a typical query
    profiler.enable()
    
    # Simulate typical operations
    start_time = time.time()
    
    # Router processing
    from src.langgraph_integration.core.router.router_node import RouterNode
    router = RouterNode()
    
    # Cart operations
    from src.langgraph_integration.state.shopping_cart_manager import ShoppingCartManager
    cart_manager = ShoppingCartManager(get_database_manager())
    
    # Measure operations
    test_session = "perf_test"
    cart_manager.add_item(test_session, "test_product", "Test Product", 1)
    cart_manager.get_cart_contents(test_session)
    
    end_time = time.time()
    profiler.disable()
    
    print(f"Total time: {end_time - start_time:.3f}s")
    profiler.print_stats(sort='cumulative')

profile_query_processing()
```

#### Solutions

1. **Optimize Database Queries**
```sql
-- Add query-specific indexes
CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_product_updated 
ON shopping_cart(session_id, product_id, updated_at);

-- Use EXPLAIN to analyze slow queries
EXPLAIN ANALYZE SELECT * FROM shopping_cart WHERE session_id = 'test_session';
```

2. **Implement Caching**
```python
# Add Redis caching
import redis

class CachedOperations:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 300  # 5 minutes
    
    def get_cached_cart(self, session_id):
        cache_key = f"cart:{session_id}"
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        return None
    
    def cache_cart(self, session_id, cart_data):
        cache_key = f"cart:{session_id}"
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(cart_data))
```

3. **Optimize Agent Processing**
```python
# Use async processing where possible
import asyncio

async def process_query_async(query, session_id):
    # Parallel processing of independent operations
    tasks = [
        classify_intent_async(query),
        get_cart_summary_async(session_id),
        get_conversation_context_async(session_id)
    ]
    
    intent_result, cart_summary, context = await asyncio.gather(*tasks)
    
    # Continue with routing logic
    return route_to_agent(intent_result, cart_summary, context)
```

## Monitoring and Alerting

### Setting Up Monitoring

```python
# Health check endpoint
def health_check():
    checks = {
        "database": check_database_health(),
        "cart_operations": check_cart_operations_health(),
        "router": check_router_health(),
        "cache": check_cache_health()
    }
    
    overall_health = all(checks.values())
    
    return {
        "status": "healthy" if overall_health else "unhealthy",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }

# Performance metrics collection
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "cart_operations": {"count": 0, "total_time": 0},
            "router_decisions": {"count": 0, "total_time": 0},
            "database_queries": {"count": 0, "total_time": 0}
        }
    
    def record_operation(self, operation_type, duration):
        if operation_type in self.metrics:
            self.metrics[operation_type]["count"] += 1
            self.metrics[operation_type]["total_time"] += duration
    
    def get_average_times(self):
        averages = {}
        for op_type, data in self.metrics.items():
            if data["count"] > 0:
                averages[op_type] = data["total_time"] / data["count"]
            else:
                averages[op_type] = 0
        return averages
```

### Alerting Rules

```python
# Define alerting thresholds
ALERT_THRESHOLDS = {
    "cart_operation_failure_rate": 0.05,  # 5%
    "router_classification_failure_rate": 0.10,  # 10%
    "database_connection_failure_rate": 0.01,  # 1%
    "average_response_time": 2.0,  # 2 seconds
    "cart_operation_time": 1.0  # 1 second
}

def check_alerts():
    metrics = get_current_metrics()
    alerts = []
    
    for metric, threshold in ALERT_THRESHOLDS.items():
        current_value = metrics.get(metric, 0)
        if current_value > threshold:
            alerts.append({
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
                "severity": "high" if current_value > threshold * 2 else "medium"
            })
    
    return alerts
```

## Emergency Procedures

### System Recovery

```bash
#!/bin/bash
# emergency_recovery.sh

echo "Starting emergency recovery procedures..."

# 1. Check system status
echo "Checking system health..."
python -c "
from src.langgraph_integration.state.database import get_database_manager
db = get_database_manager()
print('Database:', 'OK' if db.test_connection() else 'FAILED')
"

# 2. Restart services if needed
echo "Restarting application services..."
systemctl restart shopping-cart-agent
systemctl restart nginx

# 3. Clear problematic cache
echo "Clearing cache..."
redis-cli FLUSHDB

# 4. Run health checks
echo "Running post-recovery health checks..."
curl -f http://localhost:8000/health || echo "Health check failed"

echo "Emergency recovery completed"
```

### Data Recovery

```python
# Recover corrupted cart data
def recover_cart_data():
    db_manager = get_database_manager()
    
    # Find sessions with inconsistent data
    inconsistent_sessions_sql = """
    SELECT sc.session_id, 
           COUNT(*) as actual_items,
