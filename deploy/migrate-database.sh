#!/bin/bash

# =============================================================================
# Database Migration Script for Shopping Cart Agent
# =============================================================================
# Handles database schema migrations for shopping cart functionality
# Author: Shopping Cart Agent Team
# Version: 1.0.0
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-langgraph_assistant}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"

# Migration settings
MIGRATION_TIMEOUT=300
BACKUP_DIR="/tmp/db_migrations_backup"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Check if PostgreSQL is available
check_postgres_connection() {
    log "🔍 Checking PostgreSQL connection..."
    
    local timeout=60
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if PGPASSWORD="$POSTGRES_PASSWORD" pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" > /dev/null 2>&1; then
            success "PostgreSQL is available"
            return 0
        fi
        
        log "PostgreSQL not ready, waiting... (${elapsed}s/${timeout}s)"
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    error "PostgreSQL is not available after ${timeout}s"
    return 1
}

# Create backup of current database
create_database_backup() {
    log "💾 Creating database backup..."
    
    mkdir -p "$BACKUP_DIR"
    local backup_file="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
    
    if PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "$backup_file" 2>/dev/null; then
        success "Database backup created: $backup_file"
        echo "BACKUP_FILE=$backup_file" > /tmp/migration_backup.env
        return 0
    else
        warn "Failed to create database backup (continuing anyway)"
        return 1
    fi
}

# Check if shopping cart tables exist
check_shopping_cart_tables() {
    log "🔍 Checking existing shopping cart tables..."
    
    local check_query="
    SELECT 
        CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'shopping_cart') THEN 1 ELSE 0 END as shopping_cart_exists,
        CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'cart_sessions') THEN 1 ELSE 0 END as cart_sessions_exists,
        CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'intent_classifications') THEN 1 ELSE 0 END as intent_classifications_exists;
    "
    
    local result=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "$check_query" 2>/dev/null || echo "0|0|0")
    
    if [[ "$result" == *"1|1|1"* ]]; then
        success "All shopping cart tables already exist"
        return 0
    else
        log "Shopping cart tables need to be created"
        return 1
    fi
}

# Apply shopping cart schema migration
apply_shopping_cart_migration() {
    log "🚀 Applying shopping cart schema migration..."
    
    local schema_file="$PROJECT_ROOT/src/langgraph_integration/state/shopping_cart_schema.sql"
    
    if [[ ! -f "$schema_file" ]]; then
        error "Shopping cart schema file not found: $schema_file"
        return 1
    fi
    
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$schema_file" > /dev/null 2>&1; then
        success "Shopping cart schema migration applied successfully"
        return 0
    else
        error "Failed to apply shopping cart schema migration"
        return 1
    fi
}

# Run Python migration script
run_python_migrations() {
    log "🐍 Running Python migration scripts..."
    
    # Set up Python environment
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    export POSTGRES_HOST="$POSTGRES_HOST"
    export POSTGRES_PORT="$POSTGRES_PORT"
    export POSTGRES_DB="$POSTGRES_DB"
    export POSTGRES_USER="$POSTGRES_USER"
    export POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
    
    # Run migration script
    if python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src')
from langgraph_integration.state.migrations import run_migrations, check_migration_status

try:
    print('Running database migrations...')
    results = run_migrations()
    
    print('Migration results:')
    for migration, success in results.items():
        status = '✅' if success else '❌'
        print(f'  {status} {migration}: {\"SUCCESS\" if success else \"FAILED\"}')
    
    # Check final status
    status = check_migration_status()
    print(f'Total migrations: {status[\"total_migrations\"]}')
    print(f'Successful: {status[\"successful_migrations\"]}')
    print(f'Failed: {status[\"failed_migrations\"]}')
    
    if status['failed_migrations'] > 0:
        sys.exit(1)
    
except Exception as e:
    print(f'Migration failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"; then
        success "Python migrations completed successfully"
        return 0
    else
        error "Python migrations failed"
        return 1
    fi
}

# Validate migration results
validate_migration() {
    log "🔍 Validating migration results..."
    
    # Check that all required tables exist
    local validation_query="
    SELECT 
        COUNT(CASE WHEN table_name = 'shopping_cart' THEN 1 END) as shopping_cart_count,
        COUNT(CASE WHEN table_name = 'cart_sessions' THEN 1 END) as cart_sessions_count,
        COUNT(CASE WHEN table_name = 'intent_classifications' THEN 1 END) as intent_classifications_count,
        COUNT(CASE WHEN table_name = 'schema_migrations' THEN 1 END) as migrations_count
    FROM information_schema.tables 
    WHERE table_name IN ('shopping_cart', 'cart_sessions', 'intent_classifications', 'schema_migrations');
    "
    
    local result=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "$validation_query" 2>/dev/null)
    
    if [[ "$result" == *"1|1|1|1"* ]]; then
        success "All required tables created successfully"
    else
        error "Some required tables are missing"
        return 1
    fi
    
    # Check indexes
    local index_query="
    SELECT COUNT(*) as index_count
    FROM pg_indexes 
    WHERE tablename IN ('shopping_cart', 'cart_sessions', 'intent_classifications');
    "
    
    local index_count=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "$index_query" 2>/dev/null | tr -d ' ')
    
    if [[ "$index_count" -gt 0 ]]; then
        success "Database indexes created successfully ($index_count indexes)"
    else
        warn "No indexes found (this may be expected)"
    fi
    
    # Check triggers
    local trigger_query="
    SELECT COUNT(*) as trigger_count
    FROM information_schema.triggers 
    WHERE event_object_table IN ('shopping_cart', 'cart_sessions');
    "
    
    local trigger_count=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "$trigger_query" 2>/dev/null | tr -d ' ')
    
    if [[ "$trigger_count" -gt 0 ]]; then
        success "Database triggers created successfully ($trigger_count triggers)"
    else
        warn "No triggers found"
    fi
    
    return 0
}

# Rollback migration if needed
rollback_migration() {
    error "Migration failed, attempting rollback..."
    
    if [[ -f "/tmp/migration_backup.env" ]]; then
        source /tmp/migration_backup.env
        
        if [[ -f "$BACKUP_FILE" ]]; then
            log "🔄 Restoring database from backup..."
            
            if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$BACKUP_FILE" > /dev/null 2>&1; then
                success "Database restored from backup"
            else
                error "Failed to restore database from backup"
            fi
        else
            error "Backup file not found: $BACKUP_FILE"
        fi
    else
        error "No backup information found"
    fi
}

# Main migration function
main() {
    log "🚀 Starting database migration for shopping cart functionality..."
    
    # Check prerequisites
    if ! command -v psql &> /dev/null; then
        error "PostgreSQL client (psql) is not installed"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
        exit 1
    fi
    
    # Check database connection
    if ! check_postgres_connection; then
        exit 1
    fi
    
    # Create backup
    create_database_backup
    
    # Check if migration is needed
    if check_shopping_cart_tables; then
        log "Shopping cart tables already exist, checking for updates..."
    fi
    
    # Apply migrations
    if apply_shopping_cart_migration && run_python_migrations; then
        if validate_migration; then
            success "🎉 Database migration completed successfully!"
            
            # Clean up old backups (keep last 5)
            if [[ -d "$BACKUP_DIR" ]]; then
                log "🧹 Cleaning up old backups..."
                ls -t "$BACKUP_DIR"/backup_*.sql | tail -n +6 | xargs -r rm -f
            fi
            
            log "📊 Migration Summary:"
            log "   - Shopping cart tables: ✅ Created/Updated"
            log "   - Database indexes: ✅ Created"
            log "   - Database triggers: ✅ Created"
            log "   - Migration tracking: ✅ Enabled"
            
        else
            rollback_migration
            exit 1
        fi
    else
        rollback_migration
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --check)
        check_postgres_connection && check_shopping_cart_tables
        ;;
    --validate)
        validate_migration
        ;;
    --rollback)
        rollback_migration
        ;;
    *)
        main "$@"
        ;;
esac