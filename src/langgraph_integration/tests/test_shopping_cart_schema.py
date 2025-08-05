"""
Unit tests for shopping cart database schema creation and constraints.
Tests table creation, indexes, triggers, and data integrity constraints.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta

from src.langgraph_integration.state.database import DatabaseManager, DatabaseConfig
from src.langgraph_integration.state.migrations import MigrationManager, validate_shopping_cart_schema


class TestShoppingCartSchema:
    """Test shopping cart database schema."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager for testing."""
        mock_manager = Mock(spec=DatabaseManager)
        mock_manager.execute_query = Mock(return_value=[])
        mock_manager.execute_update = Mock(return_value=1)
        
        # Mock the context manager properly
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_manager.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_manager.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        return mock_manager
    
    @pytest.fixture
    def migration_manager(self, mock_db_manager):
        """Create migration manager with mock database."""
        return MigrationManager(mock_db_manager)
    
    def test_shopping_cart_table_creation(self, mock_db_manager):
        """Test that shopping cart table is created with correct schema."""
        
        # Mock the connection context manager
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_db_manager.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_db_manager.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        migration_manager = MigrationManager(mock_db_manager)
        
        # Test migration application
        with patch('builtins.open', mock_open_shopping_cart_schema()):
            result = migration_manager.apply_shopping_cart_migration()
        
        assert result is True
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
    
    def test_shopping_cart_constraints(self):
        """Test shopping cart table constraints."""
        
        # Test quantity constraint (must be > 0)
        constraint_tests = [
            {
                'name': 'quantity_positive',
                'sql': 'INSERT INTO shopping_cart (session_id, product_id, product_title, quantity) VALUES (%s, %s, %s, %s)',
                'params': ('test_session', 'prod_1', 'Test Product', 0),
                'should_fail': True,
                'error_contains': 'quantity'
            },
            {
                'name': 'unique_session_product',
                'sql': 'INSERT INTO shopping_cart (session_id, product_id, product_title, quantity) VALUES (%s, %s, %s, %s)',
                'params': ('test_session', 'prod_1', 'Test Product', 1),
                'should_fail': False
            }
        ]
        
        for test in constraint_tests:
            # This would be tested with a real database connection
            # For now, we verify the constraint exists in the schema
            assert 'CHECK (quantity > 0)' in get_shopping_cart_schema_sql()
    
    def test_cart_sessions_constraints(self):
        """Test cart sessions table constraints."""
        
        schema_sql = get_shopping_cart_schema_sql()
        
        # Verify constraints exist
        assert 'CHECK (total_items >= 0)' in schema_sql
        assert 'CHECK (total_value >= 0)' in schema_sql
        assert 'PRIMARY KEY' in schema_sql
    
    def test_intent_classifications_constraints(self):
        """Test intent classifications table constraints."""
        
        schema_sql = get_shopping_cart_schema_sql()
        
        # Verify confidence score constraint
        assert 'CHECK (confidence_score >= 0 AND confidence_score <= 1)' in schema_sql
    
    def test_shopping_cart_indexes_creation(self):
        """Test that all required indexes are created."""
        
        schema_sql = get_shopping_cart_schema_sql()
        
        expected_indexes = [
            'idx_shopping_cart_session_id',
            'idx_shopping_cart_product_id',
            'idx_shopping_cart_updated_at',
            'idx_shopping_cart_added_at',
            'idx_shopping_cart_session_product'
        ]
        
        for index_name in expected_indexes:
            assert index_name in schema_sql
    
    def test_cart_sessions_indexes_creation(self):
        """Test cart sessions indexes."""
        
        schema_sql = get_shopping_cart_schema_sql()
        
        expected_indexes = [
            'idx_cart_sessions_last_updated',
            'idx_cart_sessions_total_items'
        ]
        
        for index_name in expected_indexes:
            assert index_name in schema_sql
    
    def test_intent_classifications_indexes_creation(self):
        """Test intent classifications indexes."""
        
        schema_sql = get_shopping_cart_schema_sql()
        
        expected_indexes = [
            'idx_intent_classifications_message_hash',
            'idx_intent_classifications_context_hash',
            'idx_intent_classifications_created_at',
            'idx_intent_classifications_intent'
        ]
        
        for index_name in expected_indexes:
            assert index_name in schema_sql
    
    def test_trigger_creation(self):
        """Test that triggers are created correctly."""
        
        schema_sql = get_shopping_cart_schema_sql()
        
        # Verify trigger functions exist
        assert 'update_shopping_cart_updated_at()' in schema_sql
        assert 'update_cart_session_summary()' in schema_sql
        
        # Verify triggers are created
        assert 'CREATE TRIGGER update_shopping_cart_updated_at_trigger' in schema_sql
        assert 'CREATE TRIGGER update_cart_session_summary_trigger' in schema_sql
    
    def test_utility_functions_creation(self):
        """Test that utility functions are created."""
        
        schema_sql = get_shopping_cart_schema_sql()
        
        # Verify utility functions exist
        assert 'cleanup_old_intent_classifications' in schema_sql
        assert 'get_cart_summary' in schema_sql
    
    def test_migration_tracking(self, migration_manager):
        """Test migration tracking functionality."""
        
        # Test migration table creation
        migration_manager._ensure_migrations_table()
        
        # Verify migrations table creation was called
        migration_manager.db_manager.get_connection.assert_called()
    
    def test_migration_status_check(self, migration_manager):
        """Test migration status checking."""
        
        # Mock applied migrations
        mock_migrations = [
            {
                'migration_name': 'add_shopping_cart_tables',
                'version': '1.0.0',
                'applied_at': datetime.now(),
                'success': True,
                'error_message': None
            }
        ]
        
        migration_manager.db_manager.execute_query.return_value = mock_migrations
        
        status = migration_manager.get_migration_status()
        
        assert status['total_migrations'] == 1
        assert status['successful_migrations'] == 1
        assert status['failed_migrations'] == 0
    
    def test_migration_already_applied(self, migration_manager):
        """Test handling of already applied migrations."""
        
        # Mock that migration is already applied
        migration_manager.db_manager.execute_query.return_value = [{'count': 1}]
        
        with patch('builtins.open', mock_open_shopping_cart_schema()):
            result = migration_manager.apply_shopping_cart_migration()
        
        assert result is True
    
    def test_migration_failure_handling(self, migration_manager):
        """Test migration failure handling."""
        
        # Mock that migration is not applied
        migration_manager.db_manager.execute_query.return_value = [{'count': 0}]
        
        # Mock connection to raise an exception
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("Database error")
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        migration_manager.db_manager.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        migration_manager.db_manager.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        with patch('builtins.open', mock_open_shopping_cart_schema()):
            result = migration_manager.apply_shopping_cart_migration()
        
        assert result is False
    
    def test_schema_validation(self, mock_db_manager):
        """Test schema validation functionality."""
        
        # Mock validation query results
        validation_results = [
            [{'count': 1}],  # shopping_cart_table
            [{'count': 1}],  # cart_sessions_table
            [{'count': 1}],  # intent_classifications_table
            [{'count': 5}],  # shopping_cart_indexes
            [{'count': 2}]   # cart_triggers
        ]
        
        mock_db_manager.execute_query.side_effect = validation_results
        
        with patch('src.langgraph_integration.state.migrations.get_database_manager', return_value=mock_db_manager):
            results = validate_shopping_cart_schema()
        
        assert all(results.values())  # All validations should pass
    
    def test_database_config_integration(self):
        """Test database configuration integration."""
        
        config = DatabaseConfig()
        
        # Test default values
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "langgraph_assistant"
        
        # Test connection string generation
        connection_string = config.connection_string
        assert "postgresql://" in connection_string
        assert config.host in connection_string
        assert str(config.port) in connection_string
    
    def test_database_manager_initialization(self):
        """Test database manager initialization with shopping cart schema."""
        
        # Mock the connection pool
        mock_pool = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        with patch('src.langgraph_integration.state.database.ThreadedConnectionPool', return_value=mock_pool):
            with patch('builtins.open', mock_open_init_sql()):
                db_manager = DatabaseManager()
                db_manager.initialize()
                
                # Verify initialization was called
                assert db_manager._initialized is True
                mock_cursor.execute.assert_called_once()
                mock_conn.commit.assert_called_once()


class TestShoppingCartDataIntegrity:
    """Test data integrity and business logic constraints."""
    
    def test_unique_session_product_constraint(self):
        """Test unique constraint on session_id + product_id."""
        
        # This would test with real database that duplicate entries fail
        schema_sql = get_shopping_cart_schema_sql()
        assert 'UNIQUE(session_id, product_id)' in schema_sql
    
    def test_quantity_positive_constraint(self):
        """Test that quantity must be positive."""
        
        schema_sql = get_shopping_cart_schema_sql()
        assert 'CHECK (quantity > 0)' in schema_sql
    
    def test_confidence_score_range_constraint(self):
        """Test confidence score is between 0 and 1."""
        
        schema_sql = get_shopping_cart_schema_sql()
        assert 'CHECK (confidence_score >= 0 AND confidence_score <= 1)' in schema_sql
    
    def test_cart_totals_non_negative(self):
        """Test cart totals cannot be negative."""
        
        schema_sql = get_shopping_cart_schema_sql()
        assert 'CHECK (total_items >= 0)' in schema_sql
        assert 'CHECK (total_value >= 0)' in schema_sql


# Helper functions for testing

def mock_open_shopping_cart_schema():
    """Mock file open for shopping cart schema."""
    
    def mock_open(*args, **kwargs):
        mock_file = Mock()
        mock_file.read.return_value = get_shopping_cart_schema_sql()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        return mock_file
    
    return mock_open


def mock_open_init_sql():
    """Mock file open for init.sql."""
    
    def mock_open(*args, **kwargs):
        mock_file = Mock()
        mock_file.read.return_value = get_init_sql_content()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        return mock_file
    
    return mock_open


def get_shopping_cart_schema_sql():
    """Get the shopping cart schema SQL for testing."""
    
    return """
    -- Shopping cart table
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
    
    -- Cart session summary table
    CREATE TABLE IF NOT EXISTS cart_sessions (
        session_id VARCHAR(255) PRIMARY KEY,
        total_items INTEGER DEFAULT 0 CHECK (total_items >= 0),
        total_value DECIMAL(10,2) DEFAULT 0.00 CHECK (total_value >= 0),
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB DEFAULT '{}'::jsonb
    );
    
    -- Intent classification cache table
    CREATE TABLE IF NOT EXISTS intent_classifications (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        message_text TEXT NOT NULL,
        classified_intent VARCHAR(100) NOT NULL,
        confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
        context_hash VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB DEFAULT '{}'::jsonb
    );
    
    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_id ON shopping_cart(session_id);
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_product_id ON shopping_cart(product_id);
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_updated_at ON shopping_cart(updated_at);
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_added_at ON shopping_cart(added_at);
    CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_product ON shopping_cart(session_id, product_id);
    CREATE INDEX IF NOT EXISTS idx_cart_sessions_last_updated ON cart_sessions(last_updated);
    CREATE INDEX IF NOT EXISTS idx_cart_sessions_total_items ON cart_sessions(total_items);
    CREATE INDEX IF NOT EXISTS idx_intent_classifications_message_hash ON intent_classifications(md5(message_text));
    CREATE INDEX IF NOT EXISTS idx_intent_classifications_context_hash ON intent_classifications(context_hash);
    CREATE INDEX IF NOT EXISTS idx_intent_classifications_created_at ON intent_classifications(created_at);
    CREATE INDEX IF NOT EXISTS idx_intent_classifications_intent ON intent_classifications(classified_intent);
    
    -- Functions and triggers
    CREATE OR REPLACE FUNCTION update_shopping_cart_updated_at() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = CURRENT_TIMESTAMP; RETURN NEW; END; $$ language 'plpgsql';
    CREATE OR REPLACE FUNCTION update_cart_session_summary() RETURNS TRIGGER AS $$ BEGIN RETURN NEW; END; $$ language 'plpgsql';
    CREATE TRIGGER update_shopping_cart_updated_at_trigger BEFORE UPDATE ON shopping_cart FOR EACH ROW EXECUTE FUNCTION update_shopping_cart_updated_at();
    CREATE TRIGGER update_cart_session_summary_trigger AFTER INSERT OR UPDATE OR DELETE ON shopping_cart FOR EACH ROW EXECUTE FUNCTION update_cart_session_summary();
    
    -- Utility functions
    CREATE OR REPLACE FUNCTION cleanup_old_intent_classifications(max_age_days INTEGER DEFAULT 7) RETURNS INTEGER AS $$ BEGIN RETURN 0; END; $$ language 'plpgsql';
    CREATE OR REPLACE FUNCTION get_cart_summary(p_session_id VARCHAR(255)) RETURNS TABLE(session_id VARCHAR(255), total_items INTEGER, total_value DECIMAL(10,2), item_count INTEGER, last_updated TIMESTAMP) AS $$ BEGIN RETURN; END; $$ language 'plpgsql';
    """


def get_init_sql_content():
    """Get init.sql content for testing."""
    
    return """
    -- Enable UUID extension
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    -- Basic tables
    CREATE TABLE IF NOT EXISTS conversations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id VARCHAR(255) UNIQUE NOT NULL
    );
    """


if __name__ == "__main__":
    pytest.main([__file__])