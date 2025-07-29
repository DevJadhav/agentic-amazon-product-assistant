"""
Database connection and management utilities for PostgreSQL state storage.
Handles connection pooling, migrations, and database operations.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta

import asyncpg
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration settings."""
    
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "langgraph_assistant")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.min_connections = int(os.getenv("POSTGRES_MIN_CONN", "1"))
        self.max_connections = int(os.getenv("POSTGRES_MAX_CONN", "20"))
        
    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Get async PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager."""
        self.config = config or DatabaseConfig()
        self.pool: Optional[ThreadedConnectionPool] = None
        self.async_pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize database connections and create tables."""
        try:
            # Create synchronous connection pool
            self.pool = ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                cursor_factory=RealDictCursor
            )
            
            # Create tables if they don't exist
            self._create_tables()
            
            self._initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def initialize_async(self):
        """Initialize async database connections."""
        try:
            # Create async connection pool
            self.async_pool = await asyncpg.create_pool(
                self.config.async_connection_string,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections
            )
            
            logger.info("Async database pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async database pool: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        
        create_tables_sql = """
        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id VARCHAR(255),
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- Conversation messages table
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
            turn_number INTEGER NOT NULL,
            message_type VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Agent states table
        CREATE TABLE IF NOT EXISTS agent_states (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
            state_data JSONB NOT NULL,
            checkpoint_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON conversation_messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_messages_turn_number ON conversation_messages(conversation_id, turn_number);
        CREATE INDEX IF NOT EXISTS idx_agent_states_conversation_id ON agent_states(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_agent_states_checkpoint_id ON agent_states(checkpoint_id);
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_tables_sql)
                conn.commit()
        
        logger.info("Database tables created successfully")
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async database connection from pool."""
        if not self.async_pool:
            await self.initialize_async()
        
        async with self.async_pool.acquire() as connection:
            yield connection
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool."""
        if not self._initialized:
            self.initialize()
        
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
    
    async def execute_async_query(self, query: str, *params) -> List[Dict[str, Any]]:
        """Execute async SELECT query and return results."""
        
        async with self.get_async_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def execute_async_update(self, query: str, *params) -> str:
        """Execute async INSERT/UPDATE/DELETE query and return status."""
        
        async with self.get_async_connection() as conn:
            return await conn.execute(query, *params)
    
    def cleanup_old_conversations(self, max_age_days: int = 30) -> int:
        """Clean up old conversations and related data."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        cleanup_query = """
        DELETE FROM conversations 
        WHERE updated_at < %s
        """
        
        deleted_count = self.execute_update(cleanup_query, (cutoff_date,))
        
        logger.info(f"Cleaned up {deleted_count} old conversations")
        return deleted_count
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM conversations) as total_conversations,
            (SELECT COUNT(*) FROM conversation_messages) as total_messages,
            (SELECT COUNT(*) FROM agent_states) as total_states,
            (SELECT COUNT(*) FROM conversations WHERE updated_at > NOW() - INTERVAL '24 hours') as active_conversations_24h,
            (SELECT COUNT(*) FROM conversations WHERE updated_at > NOW() - INTERVAL '7 days') as active_conversations_7d
        """
        
        results = self.execute_query(stats_query)
        return results[0] if results else {}
    
    def close(self):
        """Close database connections."""
        
        if self.pool:
            self.pool.closeall()
            self.pool = None
        
        if self.async_pool:
            asyncio.create_task(self.async_pool.close())
            self.async_pool = None
        
        self._initialized = False
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
    
    return _db_manager


def create_database_connection() -> DatabaseManager:
    """Create a new database manager instance."""
    return DatabaseManager()


# Migration utilities

def run_migrations():
    """Run database migrations."""
    
    db_manager = get_database_manager()
    
    # For now, just ensure tables are created
    # In the future, this could handle schema updates
    db_manager.initialize()
    
    logger.info("Database migrations completed")


def check_database_health() -> Dict[str, Any]:
    """Check database health and connectivity."""
    
    try:
        db_manager = get_database_manager()
        
        # Test connection
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        
        # Get stats
        stats = db_manager.get_database_stats()
        
        return {
            "status": "healthy",
            "connected": True,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }