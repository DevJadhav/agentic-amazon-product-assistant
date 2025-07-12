"""
Enhanced RAG Query Processor with hybrid retrieval and structured outputs.
Extends the base query processor with advanced search capabilities.
"""

import json
import logging
import re
import uuid
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from langsmith import traceable

# Import base query processor
from .query_processor import RAGQueryProcessor, QueryType, QueryContext

# Import enhanced vector database
from .enhanced_vector_db import EnhancedElectronicsVectorDB, HybridSearchConfig, SearchResult
from .structured_outputs import StructuredRAGResponse, ResponseType, StructuredRAGRequest
from .structured_generator import StructuredResponseGenerator

# Import prompt registry
try:
    from prompts.registry import get_registry, PromptType
    _has_prompt_registry = True
except ImportError:
    _has_prompt_registry = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQueryContext:
    """Enhanced query context with additional metadata."""
    query: str
    query_type: QueryType
    products: List[Dict[str, Any]] = field(default_factory=list)
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    has_results: bool = False
    total_results: int = 0
    search_strategy: str = "hybrid"
    reranking_applied: bool = False
    processing_time: float = 0.0


class SearchStrategy(Enum):
    """Search strategy enumeration."""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class EnhancedRAGQueryProcessor(RAGQueryProcessor):
    """Enhanced RAG query processor with hybrid retrieval capabilities."""
    
    def __init__(self, vector_db: Optional[EnhancedElectronicsVectorDB] = None,
                 default_search_strategy: SearchStrategy = SearchStrategy.HYBRID,
                 structured_generator: Optional[StructuredResponseGenerator] = None):
        """Initialize enhanced query processor."""
        # Don't call parent __init__ to avoid double initialization
        self.vector_db = vector_db
        self.default_search_strategy = default_search_strategy
        self.structured_generator = structured_generator
        
        # Initialize query patterns from parent
        self._query_patterns = self._compile_query_patterns()
        
        # Initialize enhanced vector database if not provided
        if not self.vector_db:
            self._initialize_enhanced_vector_db()
    
    def _initialize_enhanced_vector_db(self) -> None:
        """Initialize enhanced vector database with hybrid retrieval."""
        logger.info("Initializing enhanced vector database with hybrid retrieval...")
        
        # Find JSONL file
        jsonl_path = self._find_jsonl_path()
        if not jsonl_path:
            logger.warning("JSONL file not found - using fallback mock database")
            self._initialize_fallback_vector_db()
            return
        
        try:
            from .enhanced_vector_db import setup_enhanced_vector_database
            
            self.vector_db = setup_enhanced_vector_database(
                jsonl_path,
                enable_keyword_search=True,
                enable_reranking=True
            )
            logger.info("Enhanced vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced vector database: {e}")
            logger.info("Falling back to basic vector database")
            super()._initialize_vector_db()
    
    def _find_jsonl_path(self) -> Optional[str]:
        """Find JSONL file path."""
        # Try various possible paths
        possible_paths = [
            "data/processed/electronics_rag_documents.jsonl",
            "data/processed/electronics_top1000_products.jsonl",
            "src/chatbot_ui/data/processed/electronics_rag_documents.jsonl",
            "src/chatbot_ui/data/processed/electronics_top1000_products.jsonl"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def _initialize_fallback_vector_db(self) -> None:
        """Initialize fallback vector database."""
        try:
            from .mock_vector_db import MockElectronicsVectorDB
            self.vector_db = MockElectronicsVectorDB()
            logger.info("Fallback mock vector database initialized")
        except ImportError:
            logger.warning("Fallback vector database not available")
            self.vector_db = None
    
    @traceable
    def determine_search_strategy(self, query: str, query_type: QueryType) -> SearchStrategy:
        """Determine optimal search strategy based on query characteristics."""
        
        # Rule-based strategy selection
        if query_type == QueryType.PRODUCT_COMPARISON:
            return SearchStrategy.HYBRID  # Need both semantic and keyword matching
        elif query_type == QueryType.PRODUCT_RECOMMENDATION:
            return SearchStrategy.SEMANTIC_ONLY  # Semantic similarity more important
        elif query_type in [QueryType.PRODUCT_INFO, QueryType.PRODUCT_REVIEWS]:
            return SearchStrategy.HYBRID  # Benefit from both approaches
        elif "specific" in query.lower() or "exact" in query.lower():
            return SearchStrategy.KEYWORD_ONLY  # Exact matches preferred
        else:
            return self.default_search_strategy
    
    @traceable
    def build_enhanced_context(self, query: str, max_products: int = 5, 
                             max_reviews: int = 3, 
                             search_strategy: Optional[SearchStrategy] = None,
                             trace_id: Optional[str] = None) -> EnhancedQueryContext:
        """Build enhanced context using hybrid retrieval."""
        start_time = time.time()
        
        # Input validation
        if not query or not query.strip():
            return EnhancedQueryContext(
                query="", 
                query_type=QueryType.GENERAL_SEARCH,
                metadata={"error": "Empty query"}
            )
        
        # Check vector database
        if not self.vector_db:
            return EnhancedQueryContext(
                query=query,
                query_type=QueryType.GENERAL_SEARCH,
                metadata={"error": "Database not available"}
            )
        
        # Analyze query
        query_type, extracted_terms = self.analyze_query(query, trace_id)
        
        # Determine search strategy
        if search_strategy is None:
            search_strategy = self.determine_search_strategy(query, query_type)
        
        # Configure hybrid search
        config = HybridSearchConfig(
            max_results=max_products + max_reviews,
            enable_reranking=True,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        # Perform search based on strategy
        search_results = []
        
        if search_strategy == SearchStrategy.SEMANTIC_ONLY:
            search_results = self.vector_db.semantic_search(
                query, n_results=config.max_results
            )
        elif search_strategy == SearchStrategy.KEYWORD_ONLY:
            search_results = self.vector_db.keyword_search(
                query, n_results=config.max_results
            )
        elif search_strategy == SearchStrategy.HYBRID:
            search_results = self.vector_db.hybrid_search_enhanced(
                query, config=config
            )
        elif search_strategy == SearchStrategy.ADAPTIVE:
            # Try hybrid first, fall back to semantic if needed
            search_results = self.vector_db.hybrid_search_enhanced(
                query, config=config
            )
            if not search_results:
                search_results = self.vector_db.semantic_search(
                    query, n_results=config.max_results
                )
        
        # Separate products and reviews
        products = []
        reviews = []
        
        for result in search_results:
            if result.metadata.get("doc_type") == "product" and len(products) < max_products:
                products.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score,
                    "search_type": result.search_type
                })
            elif result.metadata.get("doc_type") == "review_summary" and len(reviews) < max_reviews:
                reviews.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score,
                    "search_type": result.search_type
                })
        
        processing_time = time.time() - start_time
        
        # Build enhanced context
        context = EnhancedQueryContext(
            query=query,
            query_type=query_type,
            products=products,
            reviews=reviews,
            search_results=search_results,
            metadata={
                "extracted_terms": extracted_terms,
                "search_strategy": search_strategy.value,
                "config": config.__dict__,
                "trace_id": trace_id
            },
            has_results=bool(products or reviews),
            total_results=len(search_results),
            search_strategy=search_strategy.value,
            reranking_applied=config.enable_reranking,
            processing_time=processing_time
        )
        
        logger.info(f"Enhanced context built: {len(products)} products, {len(reviews)} reviews, "
                   f"strategy: {search_strategy.value}, time: {processing_time:.3f}s")
        
        return context
    
    @traceable
    def process_query_enhanced(self, query: str, max_products: int = 5, 
                             max_reviews: int = 3,
                             search_strategy: Optional[SearchStrategy] = None) -> Dict[str, Any]:
        """Process query with enhanced hybrid retrieval."""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        logger.info(f"Processing enhanced query [trace_id={trace_id}]: {query}")
        
        try:
            # Build enhanced context
            context = self.build_enhanced_context(
                query, max_products, max_reviews, search_strategy, trace_id
            )
            
            # Generate enhanced prompt
            prompt = self.generate_enhanced_rag_prompt(context)
            
            processing_time = time.time() - start_time
            
            # Create comprehensive result
            result = {
                "query": query,
                "context": {
                    "query_type": context.query_type.value,
                    "products": context.products,
                    "reviews": context.reviews,
                    "search_results": [
                        {
                            "id": r.id,
                            "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                            "score": r.score,
                            "rank": r.rank,
                            "search_type": r.search_type
                        }
                        for r in context.search_results[:10]  # Limit for output size
                    ],
                    "metadata": context.metadata,
                    "has_results": context.has_results,
                    "total_results": context.total_results,
                    "search_strategy": context.search_strategy,
                    "reranking_applied": context.reranking_applied
                },
                "prompt": prompt,
                "performance": {
                    "processing_time": processing_time,
                    "context_processing_time": context.processing_time,
                    "trace_id": trace_id,
                    "search_strategy": context.search_strategy
                },
                "success": True
            }
            
            logger.info(f"Enhanced query processed successfully in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "trace_id": trace_id
            }
    
    def generate_enhanced_rag_prompt(self, context: EnhancedQueryContext) -> str:
        """Generate enhanced RAG prompt with hybrid search context using template registry."""
        
        # Try to use prompt registry if available
        if _has_prompt_registry:
            try:
                prompt_registry = get_registry()
                
                # Map query type to prompt type
                prompt_type_mapping = {
                    QueryType.PRODUCT_RECOMMENDATION: PromptType.PRODUCT_RECOMMENDATION,
                    QueryType.PRODUCT_COMPARISON: PromptType.PRODUCT_COMPARISON,
                    QueryType.PRODUCT_INFO: PromptType.PRODUCT_INFO,
                    QueryType.PRODUCT_REVIEWS: PromptType.REVIEW_SUMMARY,
                    QueryType.PRODUCT_COMPLAINTS: PromptType.TROUBLESHOOTING,
                    QueryType.GENERAL_SEARCH: PromptType.GENERAL_QUERY
                }
                
                prompt_type = prompt_type_mapping.get(context.query_type, PromptType.GENERAL_QUERY)
                
                # Prepare search context for template
                search_context = {
                    "query_type": context.query_type.value,
                    "search_strategy": context.search_strategy,
                    "total_results": context.total_results,
                    "reranking_applied": context.reranking_applied
                }
                
                # Render prompt using template
                rendered_prompt = prompt_registry.render_rag_prompt(
                    prompt_type=prompt_type,
                    query=context.query,
                    products=context.products,
                    reviews=context.reviews,
                    search_context=search_context
                )
                
                return rendered_prompt
                
            except Exception as e:
                logger.warning(f"Failed to use prompt registry, falling back to hardcoded prompt: {e}")
        
        # Fallback to hardcoded prompt if registry is not available
        return self._generate_fallback_prompt(context)
    
    def _generate_fallback_prompt(self, context: EnhancedQueryContext) -> str:
        """Generate fallback prompt when prompt registry is not available."""
        
        # Enhanced prompt template
        prompt_template = """You are an expert AI assistant specializing in Amazon Electronics products. 
Use the following context to provide a comprehensive, accurate, and helpful response.

SEARCH CONTEXT:
- Query Type: {query_type}
- Search Strategy: {search_strategy}
- Results Found: {total_results}
- Reranking Applied: {reranking_applied}

PRODUCT INFORMATION:
{product_context}

REVIEW INSIGHTS:
{review_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Provide a direct, helpful response to the user's query
2. Use specific product details and review insights when available
3. Mention relevant product names, prices, and key features
4. If comparing products, create a clear comparison
5. Base recommendations on both product specifications and user reviews
6. Be concise but comprehensive
7. If no relevant information is found, acknowledge this and suggest alternative approaches

RESPONSE:"""
        
        # Build product context
        product_context = ""
        if context.products:
            product_context = "Products Found:\n"
            for i, product in enumerate(context.products[:5], 1):
                metadata = product.get("metadata", {})
                product_context += f"{i}. {metadata.get('title', 'Unknown Product')}\n"
                product_context += f"   - Price: ${metadata.get('price', 'N/A')}\n"
                product_context += f"   - Rating: {metadata.get('average_rating', 'N/A')}/5 ({metadata.get('rating_number', 0)} reviews)\n"
                product_context += f"   - Search Score: {product.get('score', 0):.3f} ({product.get('search_type', 'unknown')})\n"
                product_context += f"   - Details: {product.get('content', '')[:200]}...\n\n"
        else:
            product_context = "No specific products found in the database."
        
        # Build review context
        review_context = ""
        if context.reviews:
            review_context = "Review Insights:\n"
            for i, review in enumerate(context.reviews[:3], 1):
                metadata = review.get("metadata", {})
                review_context += f"{i}. Product: {metadata.get('title', 'Unknown')}\n"
                review_context += f"   - Review Summary: {review.get('content', '')[:300]}...\n"
                review_context += f"   - Search Score: {review.get('score', 0):.3f} ({review.get('search_type', 'unknown')})\n\n"
        else:
            review_context = "No review insights available."
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            query_type=context.query_type.value,
            search_strategy=context.search_strategy,
            total_results=context.total_results,
            reranking_applied=context.reranking_applied,
            product_context=product_context,
            review_context=review_context,
            query=context.query
        )
        
        return formatted_prompt
    
    @traceable
    def process_query_structured(self, 
                                request: StructuredRAGRequest) -> StructuredRAGResponse:
        """Process query with structured outputs."""
        
        if not self.structured_generator:
            raise ValueError("Structured generator not configured")
        
        # Build enhanced context
        context = self.build_enhanced_context(
            request.query,
            max_products=request.max_products,
            max_reviews=request.max_reviews,
            search_strategy=SearchStrategy(request.search_strategy) if request.search_strategy else None
        )
        
        # Convert enhanced context to format expected by structured generator
        context_dict = {
            "query": request.query,
            "context": {
                "products": context.products,
                "reviews": context.reviews,
                "search_results": [
                    {
                        "id": r.id,
                        "content": r.content,
                        "score": r.score,
                        "search_type": r.search_type
                    }
                    for r in context.search_results
                ],
                "metadata": context.metadata,
                "has_results": context.has_results,
                "total_results": context.total_results
            }
        }
        
        # Generate structured response
        structured_response = self.structured_generator.generate_rag_response(
            query=request.query,
            context=context_dict,
            response_type=request.preferred_response_type
        )
        
        # Add search metadata
        structured_response.search_metadata = {
            "search_strategy": context.search_strategy,
            "reranking_applied": context.reranking_applied,
            "total_results": context.total_results,
            "processing_time": context.processing_time
        }
        
        return structured_response
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get analytics about search performance."""
        if not isinstance(self.vector_db, EnhancedElectronicsVectorDB):
            return {"error": "Enhanced vector database not available"}
        
        stats = self.vector_db.get_collection_stats()
        
        return {
            "database_stats": stats,
            "search_capabilities": {
                "semantic_search": True,
                "keyword_search": self.vector_db.enable_keyword_search,
                "hybrid_search": True,
                "reranking": self.vector_db.enable_reranking,
                "structured_outputs": bool(self.structured_generator)
            },
            "indexes": {
                "products": bool(self.vector_db.indexes.get("products", {}).get("bm25")),
                "reviews": bool(self.vector_db.indexes.get("reviews", {}).get("bm25")),
                "combined": bool(self.vector_db.indexes.get("combined", {}).get("bm25"))
            }
        }


def create_enhanced_rag_processor(vector_db: Optional[EnhancedElectronicsVectorDB] = None,
                                 structured_generator: Optional[StructuredResponseGenerator] = None) -> EnhancedRAGQueryProcessor:
    """Create enhanced RAG processor with hybrid retrieval capabilities."""
    return EnhancedRAGQueryProcessor(vector_db=vector_db, structured_generator=structured_generator) 