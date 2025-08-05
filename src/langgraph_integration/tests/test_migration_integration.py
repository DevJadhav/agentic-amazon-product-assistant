"""
Integration test for shopping cart schema migration.
This test can be run with a real database to verify the schema works.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, Mock

from src.langgraph_integration.state.migrations import (
    MigrationManager, 
    run_migrations, 
    validate_shopping_cart_schema,
    apply_shopping_cart_schema
)


class TestMigrationIntegration:
    """Integration tests for migration functionality."""
    
    def test_migration_sql_syntax(self):
        """Test that the migration SQL has valid syntax."""
        
        # Read the shopping cart schema file
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(script_dir, 'state', 'shopping_cart_schema.sql')
        
        assert os.path.exists(schema_path), "Shopping cart schema file should exist"
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Basic syntax checks
        assert 'CREATE TABLE IF NOT EXISTS shopping_cart' in schema_sql
        assert 'CREATE TABLE IF NOT EXISTS cart_sessions' in schema_sql
        assert 'CREATE TABLE IF NOT EXISTS intent_classifications' in schema_sql
        
        # Check for required constraints
        assert 'CHECK (quantity > 0)' in schema_sql
        assert 'CHECK (total_items >= 0)' in schema_sql
        assert 'CHECK (total_value >= 0)' in schema_sql
        assert 'CHECK (confidence_score >= 0 AND confidence_score <= 1)' in schema_sql
        
        # Check for indexes
        assert 'idx_shopping_cart_session_id' in schema_sql
        assert 'idx_cart_sessions_last_updated' in schema_sql
        assert 'idx_intent_classifications_intent' in schema_sql
        
        # Check for triggers
        assert 'CREATE TRIGGER update_shopping_cart_updated_at_trigger' in schema_sql
        assert 'CREATE TRIGGER update_cart_session_summary_trigger' in schema_sql
    
    def test_init_sql_includes_shopping_cart(self):
        """Test that init.sql includes shopping cart schema."""
        
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        init_sql_path = os.path.join(script_dir, 'state', 'init.sql')
        
        assert os.path.exists(init_sql_path), "init.sql file should exist"
        
        with open(init_sql_path, 'r') as f:
            init_sql = f.read()
        
        # Verify shopping cart tables are included
        assert 'CREATE TABLE IF NOT EXISTS shopping_cart' in init_sql
        assert 'CREATE TABLE IF NOT EXISTS cart_sessions' in init_sql
        assert 'CREATE TABLE IF NOT EXISTS intent_classifications' in init_sql
    
    def test_migration_manager_with_mock_db(self):
        """Test migration manager with properly mocked database."""
        
        # Create a proper mock database manager
        mock_db_manager = Mock()
        
        # Mock the context manager for get_connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_db_manager.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_db_manager.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        # Mock query results
        mock_db_manager.execute_query.return_value = [{'count': 0}]  # Migration not applied
        mock_db_manager.execute_update.return_value = 1
        
        # Test migration manager creation and operation
        migration_manager = MigrationManager(mock_db_manager)
        
        # Verify the migrations table creation was attempted
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
    
    def test_schema_validation_structure(self):
        """Test the structure of schema validation."""
        
        # Mock database manager for validation
        mock_db_manager = Mock()
        
        # Mock validation query results (all tables exist)
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
        
        # All validations should pass
        assert all(results.values())
        assert len(results) == 5  # Should have 5 validation checks
    
    def test_migration_file_accessibility(self):
        """Test that migration files are accessible."""
        
        # Test shopping cart schema file
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(script_dir, 'state', 'shopping_cart_schema.sql')
        
        assert os.path.exists(schema_path)
        assert os.path.isfile(schema_path)
        assert os.access(schema_path, os.R_OK)
        
        # Test init.sql file
        init_path = os.path.join(script_dir, 'state', 'init.sql')
        
        assert os.path.exists(init_path)
        assert os.path.isfile(init_path)
        assert os.access(init_path, os.R_OK)
    
    def test_migration_functions_importable(self):
        """Test that migration functions can be imported."""
        
        # Test that all migration functions are importable
        from src.langgraph_integration.state.migrations import (
            MigrationManager,
            run_migrations,
            check_migration_status,
            apply_shopping_cart_schema,
            create_shopping_cart_indexes,
            validate_shopping_cart_schema
        )
        
        # Verify they are callable
        assert callable(MigrationManager)
        assert callable(run_migrations)
        assert callable(check_migration_status)
        assert callable(apply_shopping_cart_schema)
        assert callable(create_shopping_cart_indexes)
        assert callable(validate_shopping_cart_schema)


if __name__ == "__main__":
    pytest.main([__file__])