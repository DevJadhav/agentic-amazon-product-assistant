"""
Vector search tool for LangGraph agent workflows.
Wraps existing vector database operations behind a tool interface.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Type
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Import existing vector database components
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

from rag.enhanced_vector_db import EnhancedElectronicsVectorDB, setup_enhanced_vector_database
from rag.enhanced_query_processor import EnhancedRAGQueryProcessor, SearchStrategy

logger = logging.getLogger(__name__)


class VectorSearchInput(BaseModel):
    """Input schema for vector search tool."""
    
    query: str = Field(description="Search query for products and reviews")
    search_type: str = Field(
        default="hybrid",
        description="Type of search: 'semantic', 'keyword', 'hybrid', or 'adaptive'"
    )
    max_products: int = Field(
        default=5,
        description="Maximum number of products to return",
        ge=1,
        le=20
    )
    max_reviews: int = Field(
        default=3,
        description="Maximum number of reviews to return",
        ge=0,
        le=10
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Filter by document type: 'product' or 'review_summary'"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include search metadata in results"
    )


class VectorSearchTool(BaseTool):
    """Tool interface for vector database operations."""
    
    name: str = "vector_search"
    description: str = """
    Search for Amazon Electronics products and reviews using vector similarity.
    
    This tool can perform different types of searches:
    - semantic: Uses vector embeddings for semantic similarity
    - keyword: Uses BM25 and TF-IDF for keyword matching
    - hybrid: Combines semantic and keyword search with re-ranking
    - adaptive: Automatically selects the best search strategy
    
    Use this tool when you need to find relevant products or reviews based on user queries.
    """
    
    args_schema: Type[VectorSearchInput] = VectorSearchInput
    
    # Declare vector_db as a Pydantic field
    vector_db: Optional[EnhancedElectronicsVectorDB] = Field(default=None, exclude=True)
    
    def __init__(self, vector_db: Optional[EnhancedElectronicsVectorDB] = None, **kwargs):
        """Initialize vector search tool."""
        # Set vector_db before calling super().__init__()
        if vector_db is not None:
            kwargs['vector_db'] = vector_db
        
        super().__init__(**kwargs)
        
        self.query_processor: Optional[EnhancedRAGQueryProcessor] = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize components if not provided
        if not self.vector_db:
            self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initialize vector database and query processor."""
        try:
            # Find JSONL data file
            jsonl_path = self._find_jsonl_path()
            
            if jsonl_path:
                self.vector_db = setup_enhanced_vector_database(
                    jsonl_path,
                    enable_keyword_search=True,
                    enable_reranking=True
                )
                
                self.query_processor = EnhancedRAGQueryProcessor(
                    vector_db=self.vector_db
                )
                
                self.logger.info("Vector search tool initialized with enhanced database")
            else:
                self.logger.warning("JSONL file not found, using mock database")
                self._initialize_mock_db()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            self._initialize_mock_db()
    
    def _initialize_mock_db(self):
        """Initialize mock database for testing."""
        try:
            from rag.mock_vector_db import MockElectronicsVectorDB
            self.vector_db = MockElectronicsVectorDB()
            self.logger.info("Vector search tool initialized with mock database")
        except ImportError:
            self.logger.error("Mock database not available")
            self.vector_db = None
    
    def _find_jsonl_path(self) -> Optional[str]:
        """Find JSONL data file."""
        possible_paths = [
            "data/processed/electronics_rag_documents.jsonl",
            "../data/processed/electronics_rag_documents.jsonl",
            "../../data/processed/electronics_rag_documents.jsonl",
            Path(__file__).parent.parent.parent.parent / "data/processed/electronics_rag_documents.jsonl"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(path)
        
        return None
    
    def _run(
        self,
        query: str,
        search_type: str = "hybrid",
        max_products: int = 5,
        max_reviews: int = 3,
        doc_type: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Execute vector search synchronously."""
        
        try:
            if not self.vector_db:
                return {
                    "error": "Vector database not available",
                    "results": [],
                    "metadata": {"tool": "vector_search", "status": "error"}
                }
            
            # Use enhanced query processor if available
            if self.query_processor:
                return self._search_with_processor(
                    query, search_type, max_products, max_reviews, include_metadata
                )
            else:
                return self._search_direct(
                    query, search_type, max_products, max_reviews, doc_type, include_metadata
                )
                
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return {
                "error": str(e),
                "results": [],
                "metadata": {"tool": "vector_search", "status": "error"}
            }
    
    async def _arun(
        self,
        query: str,
        search_type: str = "hybrid",
        max_products: int = 5,
        max_reviews: int = 3,
        doc_type: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Execute vector search asynchronously."""
        
        # For now, use synchronous implementation
        # Can be enhanced with async vector database operations
        return self._run(query, search_type, max_products, max_reviews, doc_type, include_metadata)
    
    def _search_with_processor(
        self,
        query: str,
        search_type: str,
        max_products: int,
        max_reviews: int,
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Search using enhanced query processor."""
        
        try:
            # Map search type to strategy
            strategy_map = {
                "semantic": SearchStrategy.SEMANTIC_ONLY,
                "keyword": SearchStrategy.KEYWORD_ONLY,
                "hybrid": SearchStrategy.HYBRID,
                "adaptive": SearchStrategy.ADAPTIVE
            }
            
            search_strategy = strategy_map.get(search_type, SearchStrategy.HYBRID)
            
            # Process query
            result = self.query_processor.process_query_enhanced(
                query=query,
                max_products=max_products,
                max_reviews=max_reviews,
                search_strategy=search_strategy
            )
            
            if not result.get("success", True):
                return {
                    "error": result.get("error", "Query processing failed"),
                    "results": [],
                    "metadata": {"tool": "vector_search", "status": "error"}
                }
            
            # Format results
            context = result.get("context", {})
            
            formatted_results = {
                "products": context.get("products", []),
                "reviews": context.get("reviews", []),
                "total_results": context.get("total_results", 0),
                "query_type": context.get("query_type", "unknown")
            }
            
            metadata = {
                "tool": "vector_search",
                "status": "success",
                "search_strategy": context.get("search_strategy", search_type),
                "processing_time_ms": result.get("performance", {}).get("processing_time_ms", 0),
                "reranking_applied": context.get("reranking_applied", False)
            }
            
            if include_metadata:
                formatted_results["metadata"] = metadata
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Enhanced search failed: {e}")
            raise
    
    def _search_direct(
        self,
        query: str,
        search_type: str,
        max_products: int,
        max_reviews: int,
        doc_type: Optional[str],
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Search using direct vector database operations."""
        
        try:
            results = {"products": [], "reviews": [], "total_results": 0}
            
            if search_type == "semantic":
                search_results = self.vector_db.semantic_search(
                    query=query,
                    doc_type=doc_type,
                    n_results=max_products + max_reviews
                )
            elif search_type == "keyword":
                search_results = self.vector_db.keyword_search(
                    query=query,
                    n_results=max_products + max_reviews
                )
            elif search_type in ["hybrid", "adaptive"]:
                from rag.enhanced_vector_db import HybridSearchConfig
                config = HybridSearchConfig(
                    max_results=max_products + max_reviews,
                    enable_reranking=True
                )
                search_results = self.vector_db.hybrid_search_enhanced(
                    query=query,
                    config=config,
                    doc_type=doc_type
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
            # Process search results
            if search_results:
                products = []
                reviews = []
                
                for result in search_results:
                    result_data = {
                        "id": result.id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "score": result.score,
                        "rank": result.rank
                    }
                    
                    doc_type_from_result = result.metadata.get("doc_type", "unknown")
                    
                    if doc_type_from_result == "product" and len(products) < max_products:
                        products.append(result_data)
                    elif doc_type_from_result == "review_summary" and len(reviews) < max_reviews:
                        reviews.append(result_data)
                
                results["products"] = products
                results["reviews"] = reviews
                results["total_results"] = len(products) + len(reviews)
            
            if include_metadata:
                results["metadata"] = {
                    "tool": "vector_search",
                    "status": "success",
                    "search_type": search_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Direct search failed: {e}")
            raise
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about the tool's capabilities."""
        
        return {
            "name": self.name,
            "description": self.description,
            "search_types": ["semantic", "keyword", "hybrid", "adaptive"],
            "max_products": 20,
            "max_reviews": 10,
            "database_available": self.vector_db is not None,
            "enhanced_processor": self.query_processor is not None,
            "supports_async": True
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test vector database connection and functionality."""
        
        try:
            if not self.vector_db:
                return {
                    "status": "error",
                    "message": "Vector database not available"
                }
            
            # Test with a simple query
            test_result = self._run(
                query="test query",
                search_type="semantic",
                max_products=1,
                max_reviews=1
            )
            
            if "error" in test_result:
                return {
                    "status": "error",
                    "message": test_result["error"]
                }
            
            return {
                "status": "success",
                "message": "Vector search tool is working correctly",
                "test_results": test_result.get("total_results", 0)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection test failed: {e}"
            }