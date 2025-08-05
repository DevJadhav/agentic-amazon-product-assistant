# Shopping Cart Database Schema

This document describes the database schema for the shopping cart functionality in the LangGraph integration.

## Overview

The shopping cart schema extends the existing conversation state database with three new tables:
- `shopping_cart`: Stores individual cart items for each session
- `cart_sessions`: Maintains cart summaries and totals
- `intent_classifications`: Caches intent classification results for performance

## Tables

### shopping_cart

Stores individual items in user shopping carts.

```sql
CREATE TABLE shopping_cart (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    product_title VARCHAR(500) NOT NULL,
    product_price DECIMAL(10,2),
    product_image_url VARCHAR(1000),
    quantity INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    product_metadata JSONB DEFAULT '{}'::jsonb,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, product_id)
);
```

**Key Features:**
- Unique constraint on `(session_id, product_id)` prevents duplicate items
- Quantity must be positive (enforced by CHECK constraint)
- Automatic timestamps for tracking when items were added/updated
- JSONB metadata field for storing additional product information

### cart_sessions

Maintains summary information for each cart session.

```sql
CREATE TABLE cart_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    total_items INTEGER DEFAULT 0 CHECK (total_items >= 0),
    total_value DECIMAL(10,2) DEFAULT 0.00 CHECK (total_value >= 0),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

**Key Features:**
- Automatically updated via triggers when cart items change
- Stores aggregated totals for quick access
- Non-negative constraints on totals
- Metadata field for session-specific information

### intent_classifications

Caches intent classification results for performance optimization.

```sql
CREATE TABLE intent_classifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_text TEXT NOT NULL,
    classified_intent VARCHAR(100) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    context_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

**Key Features:**
- Confidence score constrained between 0 and 1
- Context hash for cache invalidation
- Automatic cleanup of old entries

## Indexes

### shopping_cart Indexes
- `idx_shopping_cart_session_id`: Fast lookups by session
- `idx_shopping_cart_product_id`: Fast lookups by product
- `idx_shopping_cart_updated_at`: Time-based queries
- `idx_shopping_cart_added_at`: Creation time queries
- `idx_shopping_cart_session_product`: Composite index for unique constraint

### cart_sessions Indexes
- `idx_cart_sessions_last_updated`: Time-based queries
- `idx_cart_sessions_total_items`: Queries by item count

### intent_classifications Indexes
- `idx_intent_classifications_message_hash`: Fast message lookups
- `idx_intent_classifications_context_hash`: Context-based lookups
- `idx_intent_classifications_created_at`: Time-based cleanup
- `idx_intent_classifications_intent`: Intent-based queries

## Triggers and Functions

### Automatic Timestamp Updates

```sql
CREATE TRIGGER update_shopping_cart_updated_at_trigger
    BEFORE UPDATE ON shopping_cart 
    FOR EACH ROW 
    EXECUTE FUNCTION update_shopping_cart_updated_at();
```

Updates the `updated_at` timestamp whenever a cart item is modified.

### Cart Session Summary Updates

```sql
CREATE TRIGGER update_cart_session_summary_trigger
    AFTER INSERT OR UPDATE OR DELETE ON shopping_cart
    FOR EACH ROW
    EXECUTE FUNCTION update_cart_session_summary();
```

Automatically maintains the `cart_sessions` table with current totals whenever cart items change.

### Utility Functions

#### cleanup_old_intent_classifications(max_age_days)
Removes old intent classification cache entries to prevent unbounded growth.

#### get_cart_summary(session_id)
Returns comprehensive cart summary including item counts and totals.

## Migration System

### Migration Tracking

The system uses a `schema_migrations` table to track applied migrations:

```sql
CREATE TABLE schema_migrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    version VARCHAR(50) NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

### Running Migrations

Use the migration script to apply the shopping cart schema:

```bash
python src/langgraph_integration/scripts/run_shopping_cart_migration.py
```

Or use the Python API:

```python
from langgraph_integration.state.migrations import run_migrations

# Run all migrations
results = run_migrations()

# Check if successful
if all(results.values()):
    print("All migrations completed successfully")
```

### Validation

Validate that the schema was created correctly:

```python
from langgraph_integration.state.migrations import validate_shopping_cart_schema

validation_results = validate_shopping_cart_schema()
print(validation_results)
```

## Usage Examples

### Adding Items to Cart

```python
from langgraph_integration.state.database import get_database_manager

db_manager = get_database_manager()

# Add item to cart
add_item_sql = """
INSERT INTO shopping_cart (session_id, product_id, product_title, quantity, product_price)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (session_id, product_id) 
DO UPDATE SET 
    quantity = shopping_cart.quantity + EXCLUDED.quantity,
    updated_at = CURRENT_TIMESTAMP
"""

db_manager.execute_update(add_item_sql, (
    "user_session_123",
    "prod_456", 
    "Wireless Headphones",
    1,
    99.99
))
```

### Getting Cart Contents

```python
# Get all items in a cart
get_cart_sql = """
SELECT product_id, product_title, quantity, product_price, 
       (quantity * COALESCE(product_price, 0)) as line_total
FROM shopping_cart 
WHERE session_id = %s
ORDER BY added_at DESC
"""

cart_items = db_manager.execute_query(get_cart_sql, ("user_session_123",))
```

### Getting Cart Summary

```python
# Get cart summary using the utility function
get_summary_sql = "SELECT * FROM get_cart_summary(%s)"
summary = db_manager.execute_query(get_summary_sql, ("user_session_123",))
```

## Performance Considerations

1. **Indexing**: All frequently queried columns are indexed
2. **Triggers**: Cart session updates are handled automatically via triggers
3. **Caching**: Intent classifications are cached to reduce computation
4. **Cleanup**: Old cache entries are cleaned up automatically
5. **Connection Pooling**: Database connections are pooled for efficiency

## Security Considerations

1. **Session Isolation**: Cart data is isolated by session_id
2. **Input Validation**: All inputs are validated via constraints
3. **SQL Injection Prevention**: Use parameterized queries
4. **Data Integrity**: Foreign key constraints and checks ensure data consistency

## Monitoring and Maintenance

### Regular Maintenance Tasks

1. **Cache Cleanup**: Run `cleanup_old_intent_classifications()` periodically
2. **Index Maintenance**: Monitor index usage and performance
3. **Session Cleanup**: Remove old cart sessions as needed
4. **Migration Status**: Check migration status regularly

### Monitoring Queries

```sql
-- Check cart activity
SELECT COUNT(*) as active_carts, 
       SUM(total_items) as total_items,
       AVG(total_value) as avg_cart_value
FROM cart_sessions 
WHERE last_updated > NOW() - INTERVAL '24 hours';

-- Check cache hit rates
SELECT classified_intent, COUNT(*) as cache_entries
FROM intent_classifications 
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY classified_intent;

-- Check migration status
SELECT migration_name, version, applied_at, success
FROM schema_migrations
ORDER BY applied_at DESC;
```

## Troubleshooting

### Common Issues

1. **Migration Failures**: Check database permissions and connectivity
2. **Constraint Violations**: Ensure data meets constraint requirements
3. **Performance Issues**: Check index usage and query plans
4. **Trigger Issues**: Verify trigger functions are created correctly

### Diagnostic Queries

```sql
-- Check table existence
SELECT table_name 
FROM information_schema.tables 
WHERE table_name IN ('shopping_cart', 'cart_sessions', 'intent_classifications');

-- Check indexes
SELECT indexname, tablename 
FROM pg_indexes 
WHERE tablename IN ('shopping_cart', 'cart_sessions', 'intent_classifications');

-- Check triggers
SELECT trigger_name, event_object_table 
FROM information_schema.triggers 
WHERE event_object_table IN ('shopping_cart', 'cart_sessions');
```