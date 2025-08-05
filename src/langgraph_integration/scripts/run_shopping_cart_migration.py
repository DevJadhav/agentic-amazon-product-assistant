#!/usr/bin/env python3
"""
Script to run shopping cart database migrations.
This script can be used to apply the shopping cart schema to an existing database.
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from langgraph_integration.state.migrations import (
    run_migrations,
    check_migration_status,
    validate_shopping_cart_schema,
    apply_shopping_cart_schema
)
from langgraph_integration.state.database import check_database_health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run shopping cart migrations."""
    
    print("Shopping Cart Database Migration Tool")
    print("=" * 40)
    
    # Check database health first
    print("\n1. Checking database health...")
    health_status = check_database_health()
    
    if health_status['status'] == 'healthy':
        print("✅ Database is healthy and accessible")
        print(f"   - Total conversations: {health_status.get('stats', {}).get('total_conversations', 'N/A')}")
        print(f"   - Total messages: {health_status.get('stats', {}).get('total_messages', 'N/A')}")
    else:
        print("❌ Database health check failed:")
        print(f"   Error: {health_status.get('error', 'Unknown error')}")
        print("\nPlease ensure PostgreSQL is running and accessible.")
        return 1
    
    # Check current migration status
    print("\n2. Checking migration status...")
    try:
        migration_status = check_migration_status()
        print(f"✅ Migration system initialized")
        print(f"   - Total migrations: {migration_status['total_migrations']}")
        print(f"   - Successful: {migration_status['successful_migrations']}")
        print(f"   - Failed: {migration_status['failed_migrations']}")
        
        if migration_status['last_migration']:
            last = migration_status['last_migration']
            print(f"   - Last migration: {last['migration_name']} (v{last['version']})")
    
    except Exception as e:
        print(f"⚠️  Migration system not yet initialized: {e}")
    
    # Run migrations
    print("\n3. Running shopping cart migrations...")
    try:
        migration_results = run_migrations()
        
        print("Migration results:")
        for migration_name, success in migration_results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   - {migration_name}: {status}")
        
        if all(migration_results.values()):
            print("\n🎉 All migrations completed successfully!")
        else:
            print("\n⚠️  Some migrations failed. Check logs for details.")
            return 1
    
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger.exception("Migration error")
        return 1
    
    # Validate schema
    print("\n4. Validating shopping cart schema...")
    try:
        validation_results = validate_shopping_cart_schema()
        
        print("Schema validation results:")
        for check_name, passed in validation_results.items():
            status = "✅ Pass" if passed else "❌ Fail"
            print(f"   - {check_name}: {status}")
        
        if all(validation_results.values()):
            print("\n🎉 Schema validation passed!")
        else:
            print("\n⚠️  Schema validation failed. Some components may not be properly created.")
            return 1
    
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        logger.exception("Validation error")
        return 1
    
    print("\n" + "=" * 40)
    print("Shopping cart database setup completed successfully!")
    print("\nYou can now:")
    print("- Use the shopping cart functionality in the agent")
    print("- Add items to user shopping carts")
    print("- Track cart sessions and totals")
    print("- Use intent classification caching")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())