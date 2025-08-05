"""
Database migration utilities for shopping cart schema and other schema changes.
Handles versioned migrations and schema updates.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .database import DatabaseManager, get_database_manager

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize migration manager."""
        self.db_manager = db_manager or get_database_manager()
        self.migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        self._ensure_migrations_table()
    
    def _ensure_migrations_table(self):
        """Create migrations tracking table if it doesn't exist."""
        
        create_migrations_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            version VARCHAR(50) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN DEFAULT true,
            error_message TEXT,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at ON schema_migrations(applied_at);
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_migrations_table_sql)
                    conn.commit()
            
            logger.info("Migrations table created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise
    
    def get_applied_migrations(self) -> List[Dict[str, any]]:
        """Get list of applied migrations."""
        
        query = """
        SELECT migration_name, version, applied_at, success, error_message
        FROM schema_migrations
        ORDER BY applied_at ASC
        """
        
        return self.db_manager.execute_query(query)
    
    def is_migration_applied(self, migration_name: str) -> bool:
        """Check if a migration has been applied."""
        
        query = """
        SELECT COUNT(*) as count
        FROM schema_migrations
        WHERE migration_name = %s AND success = true
        """
        
        result = self.db_manager.execute_query(query, (migration_name,))
        return result[0]['count'] > 0 if result else False
    
    def apply_migration(self, migration_name: str, migration_sql: str, version: str = "1.0.0") -> bool:
        """Apply a single migration."""
        
        if self.is_migration_applied(migration_name):
            logger.info(f"Migration {migration_name} already applied, skipping")
            return True
        
        logger.info(f"Applying migration: {migration_name}")
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Execute the migration SQL
                    cursor.execute(migration_sql)
                    
                    # Record the migration
                    record_migration_sql = """
                    INSERT INTO schema_migrations (migration_name, version, success)
                    VALUES (%s, %s, %s)
                    """
                    cursor.execute(record_migration_sql, (migration_name, version, True))
                    
                    conn.commit()
            
            logger.info(f"Migration {migration_name} applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply migration {migration_name}: {e}")
            
            # Record the failed migration
            try:
                record_failure_sql = """
                INSERT INTO schema_migrations (migration_name, version, success, error_message)
                VALUES (%s, %s, %s, %s)
                """
                self.db_manager.execute_update(
                    record_failure_sql, 
                    (migration_name, version, False, str(e))
                )
            except Exception as record_error:
                logger.error(f"Failed to record migration failure: {record_error}")
            
            return False
    
    def apply_shopping_cart_migration(self) -> bool:
        """Apply the shopping cart schema migration."""
        
        migration_name = "add_shopping_cart_tables"
        version = "1.0.0"
        
        # Read the shopping cart schema SQL
        script_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(script_dir, 'shopping_cart_schema.sql')
        
        try:
            with open(schema_path, 'r') as f:
                migration_sql = f.read()
        except FileNotFoundError:
            logger.error(f"Shopping cart schema file not found: {schema_path}")
            return False
        
        return self.apply_migration(migration_name, migration_sql, version)
    
    def run_all_migrations(self) -> Dict[str, bool]:
        """Run all available migrations."""
        
        results = {}
        
        # Apply shopping cart migration
        results['shopping_cart'] = self.apply_shopping_cart_migration()
        
        # Add other migrations here as needed
        
        return results
    
    def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a migration (if rollback SQL is available)."""
        
        # For now, this is a placeholder
        # In a full implementation, you'd store rollback SQL and execute it
        logger.warning(f"Rollback not implemented for migration: {migration_name}")
        return False
    
    def get_migration_status(self) -> Dict[str, any]:
        """Get overall migration status."""
        
        applied_migrations = self.get_applied_migrations()
        
        return {
            "total_migrations": len(applied_migrations),
            "successful_migrations": len([m for m in applied_migrations if m['success']]),
            "failed_migrations": len([m for m in applied_migrations if not m['success']]),
            "last_migration": applied_migrations[-1] if applied_migrations else None,
            "migrations": applied_migrations
        }


def run_migrations() -> Dict[str, bool]:
    """Run all database migrations."""
    
    migration_manager = MigrationManager()
    return migration_manager.run_all_migrations()


def check_migration_status() -> Dict[str, any]:
    """Check the status of all migrations."""
    
    migration_manager = MigrationManager()
    return migration_manager.get_migration_status()


def apply_shopping_cart_schema() -> bool:
    """Apply shopping cart schema migration."""
    
    migration_manager = MigrationManager()
    return migration_manager.apply_shopping_cart_migration()


# Utility functions for specific schema operations

def create_shopping_cart_indexes() -> bool:
    """Create additional indexes for shopping cart performance."""
    
    db_manager = get_database_manager()
    
    additional_indexes_sql = """
    -- Additional performance indexes for shopping cart
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_price_range ON shopping_cart(product_price) WHERE product_price IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_quantity_high ON shopping_cart(quantity) WHERE quantity > 1;
    CREATE INDEX IF NOT EXISTS idx_cart_sessions_value_range ON cart_sessions(total_value) WHERE total_value > 0;
    
    -- Composite indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_updated ON shopping_cart(session_id, updated_at);
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_product_session ON shopping_cart(product_id, session_id);
    """
    
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(additional_indexes_sql)
                conn.commit()
        
        logger.info("Additional shopping cart indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create additional indexes: {e}")
        return False


def validate_shopping_cart_schema() -> Dict[str, bool]:
    """Validate that shopping cart schema is properly created."""
    
    db_manager = get_database_manager()
    
    validation_queries = {
        "shopping_cart_table": """
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = 'shopping_cart'
        """,
        "cart_sessions_table": """
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = 'cart_sessions'
        """,
        "intent_classifications_table": """
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = 'intent_classifications'
        """,
        "shopping_cart_indexes": """
            SELECT COUNT(*) as count 
            FROM pg_indexes 
            WHERE tablename = 'shopping_cart'
        """,
        "cart_triggers": """
            SELECT COUNT(*) as count 
            FROM information_schema.triggers 
            WHERE event_object_table = 'shopping_cart'
        """
    }
    
    results = {}
    
    for check_name, query in validation_queries.items():
        try:
            result = db_manager.execute_query(query)
            results[check_name] = result[0]['count'] > 0 if result else False
        except Exception as e:
            logger.error(f"Validation check {check_name} failed: {e}")
            results[check_name] = False
    
    return results