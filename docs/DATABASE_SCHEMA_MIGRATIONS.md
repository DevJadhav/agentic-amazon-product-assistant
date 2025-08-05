# Database Schema Changes and Migration Procedures

## Overview

This document describes the database schema changes introduced for the shopping cart agent system, including new tables, indexes, functions, and triggers. It also provides comprehensive migration procedures for deploying these changes to existing systems.

## Schema Changes Summary

### New Tables

The shopping cart functionality introduces three new tables to the existing PostgreSQL database:

1. **shopping_cart**: Stores individual cart items for each session
2. **cart_sessions**: Maintains cart summary information for quick access
3. **intent_classifications**: Caches intent classification results for performance

### New Database Objects

- **Functions**: 4 new PostgreSQL functions for cart operations and maintenance
- **Triggers**: 2 triggers for automatic data maintenance
- **Indexes**: 12 new indexes for query optimization
- **Constraints**: Check constraints for data integrity

## Detailed Schema Changes

### 1. Shopping Cart Table

The primary table for storing cart items with full product information.

```sql
CREATE TABLE IF NOT EXISTS shopping_cart (
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
- **UUID Primary Key**: Ensures unique identification across distributed systems
- **Session Isolation**: Each session maintains separate cart data
- **Product Metadata**: JSONB field for flexible product information storage
- **Quantity Constraints**: Ensures positive quantities only
- **Unique Constraint**: Prevents duplicate products per session
- **Timestamps**: Automatic tracking of creation and modification times

**Indexes:**
```sql
CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_id ON shopping_cart(session_id);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_product_id ON shopping_cart(product_id);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_updated_at ON shopping_cart(updated_at);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_added_at ON shopping_cart(added_at);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_product ON shopping_cart(session_id, product_id);
```

### 2. Cart Sessions Table

Summary table for quick access to cart totals and statistics.

```sql
CREATE TABLE IF NOT EXISTS cart_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    total_items INTEGER DEFAULT 0 CHECK (total_items >= 0),
    total_value DECIMAL(10,2) DEFAULT 0.00 CHECK (total_value >= 0),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

**Key Features:**
- **Session-based Primary Key**: One record per session
- **Aggregate Data**: Pre-calculated totals for performance
- **Check Constraints**: Ensures non-negative values
- **Metadata Support**: Extensible session information storage

**Indexes:**
```sql
CREATE INDEX IF NOT EXISTS idx_cart_sessions_last_updated ON cart_sessions(last_updated);
CREATE INDEX IF NOT EXISTS idx_cart_sessions_total_items ON cart_sessions(total_items);
```

### 3. Intent Classifications Table

Caching table for router performance optimization.

```sql
CREATE TABLE IF NOT EXISTS intent_classifications (
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
- **Classification Caching**: Stores intent classification results
- **Confidence Scoring**: Tracks classification confidence levels
- **Context Awareness**: Includes context hash for cache invalidation
- **TTL Support**: Created timestamp for cache expiration

**Indexes:**
```sql
CREATE INDEX IF NOT EXISTS idx_intent_classifications_message_hash ON intent_classifications(md5(message_text));
CREATE INDEX IF NOT EXISTS idx_intent_classifications_context_hash ON intent_classifications(context_hash);
CREATE INDEX IF NOT EXISTS idx_intent_classifications_created_at ON intent_classifications(created_at);
CREATE INDEX IF NOT EXISTS idx_intent_classifications_intent ON intent_classifications(classified_intent);
```

## Database Functions

### 1. Update Timestamp Function

Automatically updates the `updated_at` timestamp when cart items are modified.

```sql
CREATE OR REPLACE FUNCTION update_shopping_cart_updated_at()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$ language 'plpgsql';
```

### 2. Cart Session Summary Function

Maintains cart session summaries automatically when cart items change.

```sql
CREATE OR REPLACE FUNCTION update_cart_session_summary()
RETURNS TRIGGER AS $
BEGIN
    -- Handles INSERT, UPDATE, and DELETE operations
    -- Updates cart_sessions table with current totals
    -- Maintains data consistency across operations
END;
$ language 'plpgsql';
```

### 3. Cleanup Function

Removes old intent classification cache entries for maintenance.

```sql
CREATE OR REPLACE FUNCTION cleanup_old_intent_classifications(max_age_days INTEGER DEFAULT 7)
RETURNS INTEGER AS $
BEGIN
    DELETE FROM intent_classifications 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * max_age_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$ language 'plpgsql';
```

### 4. Cart Summary Query Function

Provides optimized cart summary queries.

```sql
CREATE OR REPLACE FUNCTION get_cart_summary(p_session_id VARCHAR(255))
RETURNS TABLE(
    session_id VARCHAR(255),
    total_items INTEGER,
    total_value DECIMAL(10,2),
    item_count INTEGER,
    last_updated TIMESTAMP
) AS $
BEGIN
    -- Returns comprehensive cart summary for a session
END;
$ language 'plpgsql';
```

## Database Triggers

### 1. Timestamp Update Trigger

```sql
CREATE TRIGGER update_shopping_cart_updated_at_trigger
    BEFORE UPDATE ON shopping_cart 
    FOR EACH ROW 
    EXECUTE FUNCTION update_shopping_cart_updated_at();
```

### 2. Session Summary Update Trigger

```sql
CREATE TRIGGER update_cart_session_summary_trigger
    AFTER INSERT OR UPDATE OR DELETE ON shopping_cart
    FOR EACH ROW
    EXECUTE FUNCTION update_cart_session_summary();
```

## Migration Procedures

### Migration Management System

The system includes a comprehensive migration management framework:

```python
class MigrationManager:
    """Manages database schema migrations with versioning and rollback support."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or get_database_manager()
        self._ensure_migrations_table()
    
    def apply_migration(self, migration_name: str, migration_sql: str, version: str = "1.0.0") -> bool:
        """Apply a single migration with error handling and logging."""
        pass
    
    def run_all_migrations(self) -> Dict[str, bool]:
        """Run all available migrations in sequence."""
        pass
```

### Migration Tracking

All migrations are tracked in a dedicated table:

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    version VARCHAR(50) NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

## Deployment Procedures

### Pre-Deployment Checklist

1. **Database Backup**: Create full database backup before migration
2. **Environment Validation**: Verify target environment configuration
3. **Permission Check**: Ensure database user has required privileges
4. **Dependency Verification**: Confirm PostgreSQL version compatibility
5. **Rollback Plan**: Prepare rollback procedures if needed

### Step-by-Step Migration Process

#### Step 1: Environment Preparation

```bash
# 1. Create database backup
pg_dump -h localhost -U username -d database_name > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Verify database connection
python -c "from src.langgraph_integration.state.database import get_database_manager; print('Connection OK' if get_database_manager().test_connection() else 'Connection Failed')"

# 3. Check current schema version
python -c "from src.langgraph_integration.state.migrations import check_migration_status; print(check_migration_status())"
```

#### Step 2: Run Migrations

```python
# Using Python migration manager
from src.langgraph_integration.state.migrations import run_migrations

# Run all migrations
results = run_migrations()
print(f"Migration results: {results}")

# Check migration status
from src.langgraph_integration.state.migrations import check_migration_status
status = check_migration_status()
print(f"Migration status: {status}")
```

#### Step 3: Validation

```python
# Validate schema changes
from src.langgraph_integration.state.migrations import validate_shopping_cart_schema

validation_results = validate_shopping_cart_schema()
print(f"Schema validation: {validation_results}")

# Test cart functionality
from src.langgraph_integration.state.shopping_cart_manager import ShoppingCartManager
from src.langgraph_integration.state.database import get_database_manager

cart_manager = ShoppingCartManager(get_database_manager())
test_result = cart_manager.add_item("test_session", "test_product", "Test Product", 1)
print(f"Cart test: {test_result}")
```

### Production Deployment Script

```bash
#!/bin/bash
# production_migration.sh

set -e  # Exit on any error

echo "Starting shopping cart schema migration..."

# Configuration
DB_HOST=${DB_HOST:-localhost}
DB_NAME=${DB_NAME:-production_db}
DB_USER=${DB_USER:-app_user}
BACKUP_DIR=${BACKUP_DIR:-/backups}

# Create backup
echo "Creating database backup..."
BACKUP_FILE="${BACKUP_DIR}/pre_migration_backup_$(date +%Y%m%d_%H%M%S).sql"
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_FILE
echo "Backup created: $BACKUP_FILE"

# Run migrations
echo "Running database migrations..."
python -m src.langgraph_integration.state.migrations

# Validate results
echo "Validating migration results..."
python -c "
from src.langgraph_integration.state.migrations import check_migration_status, validate_shopping_cart_schema
import sys

status = check_migration_status()
validation = validate_shopping_cart_schema()

print(f'Migration Status: {status}')
print(f'Schema Validation: {validation}')

# Check for failures
if status['failed_migrations'] > 0:
    print('ERROR: Some migrations failed!')
    sys.exit(1)

if not all(validation.values()):
    print('ERROR: Schema validation failed!')
    sys.exit(1)

print('All migrations completed successfully!')
"

echo "Migration completed successfully!"
```

### Rollback Procedures

#### Automatic Rollback (if implemented)

```python
from src.langgraph_integration.state.migrations import MigrationManager

migration_manager = MigrationManager()
success = migration_manager.rollback_migration("add_shopping_cart_tables")
```

#### Manual Rollback

```sql
-- Manual rollback SQL (use with caution)
BEGIN;

-- Drop triggers first
DROP TRIGGER IF EXISTS update_cart_session_summary_trigger ON shopping_cart;
DROP TRIGGER IF EXISTS update_shopping_cart_updated_at_trigger ON shopping_cart;

-- Drop functions
DROP FUNCTION IF EXISTS update_cart_session_summary();
DROP FUNCTION IF EXISTS update_shopping_cart_updated_at();
DROP FUNCTION IF EXISTS cleanup_old_intent_classifications(INTEGER);
DROP FUNCTION IF EXISTS get_cart_summary(VARCHAR(255));

-- Drop tables (in reverse dependency order)
DROP TABLE IF EXISTS intent_classifications;
DROP TABLE IF EXISTS cart_sessions;
DROP TABLE IF EXISTS shopping_cart;

-- Remove migration record
DELETE FROM schema_migrations WHERE migration_name = 'add_shopping_cart_tables';

COMMIT;
```

#### Database Restore (if needed)

```bash
# Restore from backup
pg_restore -h localhost -U username -d database_name backup_file.sql

# Or using psql
psql -h localhost -U username -d database_name < backup_file.sql
```

## Performance Considerations

### Index Strategy

The schema includes comprehensive indexing for optimal performance:

1. **Primary Access Patterns**: Session-based queries are optimized with session_id indexes
2. **Product Lookups**: Product-based queries use product_id indexes
3. **Temporal Queries**: Time-based queries use timestamp indexes
4. **Composite Indexes**: Multi-column indexes for complex queries

### Query Optimization

```sql
-- Optimized cart retrieval query
SELECT sc.*, cs.total_items, cs.total_value
FROM shopping_cart sc
JOIN cart_sessions cs ON sc.session_id = cs.session_id
WHERE sc.session_id = $1
ORDER BY sc.added_at DESC;

-- Optimized cart summary query
SELECT * FROM get_cart_summary($1);
```

### Maintenance Procedures

#### Regular Maintenance Tasks

```sql
-- Clean up old intent classifications (run daily)
SELECT cleanup_old_intent_classifications(7);

-- Update table statistics (run weekly)
ANALYZE shopping_cart;
ANALYZE cart_sessions;
ANALYZE intent_classifications;

-- Reindex if needed (run monthly)
REINDEX TABLE shopping_cart;
REINDEX TABLE cart_sessions;
```

#### Monitoring Queries

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE tablename IN ('shopping_cart', 'cart_sessions', 'intent_classifications');

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE tablename IN ('shopping_cart', 'cart_sessions', 'intent_classifications');
```

## Security Considerations

### Data Protection

1. **Session Isolation**: Each session's cart data is completely isolated
2. **Input Validation**: All inputs are validated and sanitized
3. **SQL Injection Prevention**: All queries use parameterized statements
4. **Access Control**: Database permissions restrict access to cart tables

### Privacy Compliance

1. **No Personal Data**: Cart tables store no personally identifiable information
2. **Session-based Identification**: Uses session IDs instead of user identifiers
3. **Data Retention**: Automatic cleanup of old data
4. **Audit Trail**: Migration tracking provides change history

## Troubleshooting

### Common Migration Issues

#### Issue: Migration Fails with Permission Error

```bash
# Solution: Grant required permissions
GRANT CREATE ON DATABASE database_name TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT CREATE ON SCHEMA public TO app_user;
```

#### Issue: UUID Extension Not Available

```sql
-- Solution: Install UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Or use gen_random_uuid() if available (PostgreSQL 13+)
```

#### Issue: JSONB Not Supported

```sql
-- Solution: Upgrade PostgreSQL or use JSON
-- Replace JSONB with JSON in schema if needed
```

### Validation Failures

#### Check Table Creation

```sql
SELECT table_name, table_type 
FROM information_schema.tables 
WHERE table_name IN ('shopping_cart', 'cart_sessions', 'intent_classifications');
```

#### Check Index Creation

```sql
SELECT indexname, tablename 
FROM pg_indexes 
WHERE tablename IN ('shopping_cart', 'cart_sessions', 'intent_classifications');
```

#### Check Function Creation

```sql
SELECT routine_name, routine_type 
FROM information_schema.routines 
WHERE routine_name IN (
    'update_shopping_cart_updated_at',
    'update_cart_session_summary',
    'cleanup_old_intent_classifications',
    'get_cart_summary'
);
```

## Testing Procedures

### Pre-Migration Testing

```python
# Test database connection
from src.langgraph_integration.state.database import get_database_manager

db_manager = get_database_manager()
assert db_manager.test_connection(), "Database connection failed"

# Test existing functionality
# Run existing tests to ensure no regression
```

### Post-Migration Testing

```python
# Test cart functionality
from src.langgraph_integration.state.shopping_cart_manager import ShoppingCartManager

cart_manager = ShoppingCartManager(db_manager)

# Test add item
result = cart_manager.add_item("test_session", "test_product", "Test Product", 1)
assert result["success"], f"Add item failed: {result}"

# Test get cart
cart_contents = cart_manager.get_cart_contents("test_session")
assert len(cart_contents) == 1, "Cart should contain one item"

# Test remove item
result = cart_manager.remove_item("test_session", "test_product")
assert result["success"], f"Remove item failed: {result}"

# Test empty cart
cart_contents = cart_manager.get_cart_contents("test_session")
assert len(cart_contents) == 0, "Cart should be empty"
```

### Performance Testing

```python
import time
from concurrent.futures import ThreadPoolExecutor

def test_concurrent_cart_operations():
    """Test cart operations under concurrent load."""
    
    def add_items(session_id, num_items):
        cart_manager = ShoppingCartManager(get_database_manager())
        for i in range(num_items):
            cart_manager.add_item(session_id, f"product_{i}", f"Product {i}", 1)
    
    # Test with multiple sessions
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for session in range(10):
            future = executor.submit(add_items, f"session_{session}", 10)
            futures.append(future)
        
        # Wait for completion
        for future in futures:
            future.result()
    
    print("Concurrent operations test completed successfully")
```

## Monitoring and Maintenance

### Health Checks

```python
def check_cart_system_health():
    """Comprehensive health check for cart system."""
    
    health_status = {
        "database_connection": False,
        "cart_tables_exist": False,
        "cart_operations_working": False,
        "performance_acceptable": False
    }
    
    try:
        # Test database connection
        db_manager = get_database_manager()
        health_status["database_connection"] = db_manager.test_connection()
        
        # Test table existence
        validation = validate_shopping_cart_schema()
        health_status["cart_tables_exist"] = all(validation.values())
        
        # Test cart operations
        cart_manager = ShoppingCartManager(db_manager)
        test_session = f"health_check_{int(time.time())}"
        
        # Add and remove test item
        add_result = cart_manager.add_item(test_session, "health_test", "Health Test", 1)
        remove_result = cart_manager.remove_item(test_session, "health_test")
        
        health_status["cart_operations_working"] = (
            add_result.get("success", False) and 
            remove_result.get("success", False)
        )
        
        # Test performance (should complete within reasonable time)
        start_time = time.time()
        cart_manager.get_cart_contents(test_session)
        query_time = time.time() - start_time
        
        health_status["performance_acceptable"] = query_time < 0.1  # 100ms threshold
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
    
    return health_status
```

### Automated Maintenance

```python
def run_maintenance_tasks():
    """Run automated maintenance tasks."""
    
    db_manager = get_database_manager()
    
    # Clean up old intent classifications
    cleanup_query = "SELECT cleanup_old_intent_classifications(7)"
    result = db_manager.execute_query(cleanup_query)
    logger.info(f"Cleaned up {result[0]['cleanup_old_intent_classifications']} old classifications")
    
    # Update table statistics
    analyze_queries = [
        "ANALYZE shopping_cart",
        "ANALYZE cart_sessions", 
        "ANALYZE intent_classifications"
    ]
    
    for query in analyze_queries:
        db_manager.execute_update(query)
    
    logger.info("Maintenance tasks completed successfully")
```

## Conclusion

The shopping cart database schema provides a robust foundation for cart functionality with comprehensive migration procedures. The schema is designed for:

- **Performance**: Optimized indexes and summary tables
- **Reliability**: Comprehensive error handling and validation
- **Maintainability**: Clear migration procedures and monitoring
- **Scalability**: Efficient queries and data structures
- **Security**: Session isolation and input validation

Follow the procedures in this document to safely deploy the schema changes to your environment. Always test migrations in a staging environment before applying to production, and maintain regular backups for rollback capability.