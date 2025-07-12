"""
RAG Query Processor for Amazon Electronics Assistant.
Handles query analysis, document retrieval, and context generation for LLM responses.
"""

import json
import logging
import re
import uuid
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Pattern, FrozenSet, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import os
from langsmith import traceable

# Import enhanced tracing utilities
try:
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.append(str(parent_dir))
    
    from tracing.trace_utils import (
        TraceContext, get_current_trace_context, create_enhanced_trace_context,
        update_trace_context, business_analyzer, BusinessMetricsAnalyzer
    )
    _has_tracing = True
except ImportError:
    # Fallback for cases where tracing utils are not available
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from tracing.trace_utils import TraceContext
    else:
        TraceContext = Any
    
    get_current_trace_context = lambda: None
    create_enhanced_trace_context = lambda **kwargs: None
    update_trace_context = lambda **kwargs: None
    business_analyzer = None
    BusinessMetricsAnalyzer = None
    _has_tracing = False

# Import configuration
try:
    from chatbot_ui.core.config import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Enumeration of query types."""
    PRODUCT_INFO = "product_info"
    PRODUCT_REVIEWS = "product_reviews"
    PRODUCT_COMPLAINTS = "product_complaints"
    PRODUCT_COMPARISON = "product_comparison"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    USE_CASE = "use_case"
    GENERAL_SEARCH = "general_search"


@dataclass
class QueryContext:
    """Structured context for RAG queries."""
    query: str
    query_type: QueryType
    products: List[Dict[str, Any]] = field(default_factory=list)
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_results(self) -> int:
        """Get total number of results."""
        return len(self.products) + len(self.reviews)
    
    @property
    def has_results(self) -> bool:
        """Check if context has any results."""
        return bool(self.products or self.reviews)


class QueryPattern(NamedTuple):
    """Named tuple for query patterns."""
    type: QueryType
    pattern: Pattern[str]
    extractor: Optional[callable] = None





class RAGQueryProcessor:
    """Processes user queries using RAG with Amazon Electronics data."""
    
    # Class-level constants for better performance
    PRODUCT_INDICATORS: FrozenSet[Pattern[str]] = frozenset({
        re.compile(r"iphone\s*\d*", re.IGNORECASE),
        re.compile(r"samsung\s*galaxy", re.IGNORECASE),
        re.compile(r"macbook", re.IGNORECASE),
        re.compile(r"fire\s*tv", re.IGNORECASE),
        re.compile(r"kindle", re.IGNORECASE),
        re.compile(r"echo\s*dot", re.IGNORECASE),
        re.compile(r"airpods", re.IGNORECASE),
        re.compile(r"cat\s*\d*\s*cable", re.IGNORECASE),
        re.compile(r"ethernet\s*cable", re.IGNORECASE),
        re.compile(r"usb\s*cable", re.IGNORECASE),
        re.compile(r"charger", re.IGNORECASE),
        re.compile(r"laptop", re.IGNORECASE),
        re.compile(r"tablet", re.IGNORECASE),
        re.compile(r"headphones", re.IGNORECASE),
        re.compile(r"speaker", re.IGNORECASE),
        re.compile(r"router", re.IGNORECASE),
        re.compile(r"backpack", re.IGNORECASE)
    })
    
    BUDGET_TERMS: FrozenSet[str] = frozenset({
        "budget", "cheap", "affordable", "inexpensive", "under", "less than", "below"
    })
    
    def __init__(self, vector_db: Optional[Any] = None):
        """Initialize the RAG query processor."""
        self.vector_db = vector_db
        
        # Initialize query patterns once
        self._query_patterns = self._compile_query_patterns()
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize vector database if not provided
        if not self.vector_db:
            self._initialize_vector_db()
    
    def _initialize_vector_db(self) -> None:
        """Initialize vector database with robust error handling and fallback."""
        logger.info("Initializing Weaviate vector database...")
        
        # Find JSONL file first
        jsonl_path = self._find_jsonl_path()
        if not jsonl_path:
            logger.warning("JSONL file not found - using fallback mock database")
            self._initialize_fallback_vector_db()
            return
        
        # Try to initialize simplified Weaviate database first
        try:
            logger.info(f"Attempting to initialize simplified Weaviate database with path: {jsonl_path}")
            
            # Use simplified Weaviate implementation (no FAISS)
            from .vector_db_weaviate_simple import setup_vector_database_simple
            
            self.vector_db = setup_vector_database_simple(jsonl_path)
            logger.info("Simplified Weaviate vector database initialized successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize simplified Weaviate database: {error_msg}")
            logger.info("Using fallback mock database for continued operation")
            self._initialize_fallback_vector_db()
                
    def _initialize_fallback_vector_db(self) -> None:
        """Initialize a fallback vector database that doesn't rely on FAISS."""
        try:
            # Create a mock vector database for testing
            from .mock_vector_db import MockElectronicsVectorDB
            self.vector_db = MockElectronicsVectorDB()
            logger.info("Fallback mock vector database initialized successfully")
        except ImportError:
            logger.warning("Fallback vector database not available - system will have limited functionality")
            self.vector_db = None
    
    @staticmethod
    def _find_jsonl_path() -> Optional[str]:
        """Find the JSONL data file in various possible locations."""
        possible_paths = [
            "data/processed/electronics_rag_documents.jsonl",
            "../data/processed/electronics_rag_documents.jsonl",
            "../../data/processed/electronics_rag_documents.jsonl",
            Path(__file__).parent.parent.parent / "data/processed/electronics_rag_documents.jsonl"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(path)
        
        return None
    
    def _compile_query_patterns(self) -> List[QueryPattern]:
        """Compile all query patterns once for better performance."""
        patterns = [
            # Product info patterns
            QueryPattern(
                QueryType.PRODUCT_INFO,
                re.compile(r"what.*about\s+(.+)|tell me about\s+(.+)|describe\s+(.+)|information.*about\s+(.+)|details.*about\s+(.+)", re.IGNORECASE)
            ),
            
            # Product reviews patterns
            QueryPattern(
                QueryType.PRODUCT_REVIEWS,
                re.compile(r"reviews?\s+(?:for|of|about)\s+(.+)|what.*people.*say.*about\s+(.+)|feedback.*(?:for|about)\s+(.+)|opinions?.*(?:on|about)\s+(.+)|experiences?.*with\s+(.+)", re.IGNORECASE)
            ),
            
            # Product complaints patterns
            QueryPattern(
                QueryType.PRODUCT_COMPLAINTS,
                re.compile(r"(?:problems?|issues?|complaints?).*(?:with|about)\s+(.+)|what.*wrong.*with\s+(.+)|negative.*(?:reviews?|feedback).*(?:for|about)\s+(.+)|main.*complaints?.*about\s+(.+)|criticism.*(?:of|about)\s+(.+)", re.IGNORECASE)
            ),
            
            # Product comparison patterns
            QueryPattern(
                QueryType.PRODUCT_COMPARISON,
                re.compile(r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)|difference.*between\s+(.+?)\s+and\s+(.+)|(.+?)\s+vs\s+(.+)|which.*better.*(.+?)\s+or\s+(.+)|(.+?)\s+versus\s+(.+)", re.IGNORECASE)
            ),
            
            # Product recommendation patterns
            QueryPattern(
                QueryType.PRODUCT_RECOMMENDATION,
                re.compile(r"recommend.*(.+)|suggest.*(.+)|best\s+(.+)|alternative.*(?:to|for)\s+(.+)|similar.*(?:to|like)\s+(.+)", re.IGNORECASE)
            ),
            
            # Use case patterns
            QueryPattern(
                QueryType.USE_CASE,
                re.compile(r"is\s+(.+?)\s+good\s+for\s+(.+)|can\s+(.+?)\s+be\s+used\s+for\s+(.+)|suitable.*(.+?).*for\s+(.+)|(.+?)\s+for\s+(.+)|best.*(.+?).*for\s+(.+)", re.IGNORECASE)
            )
        ]
        
        return patterns
    
    @traceable
    def analyze_query(self, query: str, trace_id: Optional[str] = None) -> Tuple[QueryType, List[str]]:
        """
        Analyze query to determine type and extract key terms.
        
        Time Complexity: O(n) where n is number of patterns
        Space Complexity: O(m) where m is length of extracted terms
        """
        # Get or create trace context
        trace_context = get_current_trace_context()
        if not trace_context and trace_id:
            trace_context = create_enhanced_trace_context()
        
        # Enhanced analysis using business analyzer if available
        if business_analyzer:
            self._perform_business_analysis(query, trace_context)
        
        # Check each pattern
        for pattern in self._query_patterns:
            match = pattern.pattern.search(query)
            if match:
                # Extract non-None groups
                extracted_terms = [g for g in match.groups() if g is not None]
                return pattern.type, extracted_terms
        
        # Default to general search
        return QueryType.GENERAL_SEARCH, [query.strip()]
    
    def _perform_business_analysis(self, query: str, trace_context: Optional[TraceContext]) -> None:
        """Perform business analysis on the query."""
        if not business_analyzer or not trace_context:
            return
        
        query_intent = business_analyzer.classify_intent(query)
        complexity = business_analyzer.calculate_complexity(query)
        specificity = business_analyzer.measure_specificity(query)
        product_focus = business_analyzer.extract_product_focus(query)
        
        # Update trace context with business metrics
        update_trace_context(
            query_intent=query_intent,
            complexity_score=complexity,
            specificity_score=specificity,
            product_focus=product_focus
        )
    
    @lru_cache(maxsize=128)
    @traceable
    def extract_product_names(self, query: str) -> List[str]:
        """
        Extract potential product names from query using compiled patterns.
        
        Time Complexity: O(p) where p is number of product patterns
        Space Complexity: O(k) where k is number of matches
        """
        products = set()  # Use set to avoid duplicates
        
        for pattern in self.PRODUCT_INDICATORS:
            matches = pattern.findall(query)
            products.update(matches)
        
        return list(products)
    
    @traceable
    def build_context(self, query: str, max_products: int = 5, max_reviews: int = 3, 
                     trace_id: Optional[str] = None) -> QueryContext:
        """Build context for RAG query with optimized search strategies."""
        # Input validation
        if not query or not query.strip():
            return QueryContext(query="", query_type=QueryType.GENERAL_SEARCH, 
                              metadata={"error": "Empty query"})
        
        # Get or propagate trace context
        trace_context = get_current_trace_context()
        if not trace_context and trace_id:
            trace_context = create_enhanced_trace_context()
        
        # Check vector database
        if not self.vector_db:
            logger.error("Vector database not initialized")
            return QueryContext(
                query=query, 
                query_type=QueryType.GENERAL_SEARCH,
                metadata={
                    "error": "Database not available",
                    "trace_id": trace_context.trace_id if trace_context else None,
                    "fallback_response": True
                }
            )
        
        # Analyze query
        query_type, extracted_terms = self.analyze_query(query, trace_id)
        logger.info(f"Query type: {query_type.value}, Terms: {extracted_terms}")
        
        # Build metadata
        metadata = self._build_metadata(query_type, extracted_terms, trace_context, trace_id)
        
        # Create context
        context = QueryContext(query=query, query_type=query_type, metadata=metadata)
        
        # Add database status to metadata
        if self.vector_db:
            context.metadata["database_status"] = "available"
            context.metadata["is_mock"] = getattr(self.vector_db, 'is_mock', False)
        else:
            context.metadata["database_status"] = "unavailable"
            context.metadata["is_mock"] = False
        
        # Execute search strategy based on query type
        search_strategy = self._get_search_strategy(query_type)
        search_strategy(context, query, extracted_terms, max_products, max_reviews)
        
        return context
    
    def _build_metadata(self, query_type: QueryType, extracted_terms: List[str], 
                       trace_context: Optional[TraceContext], trace_id: Optional[str]) -> Dict[str, Any]:
        """Build metadata for the query context."""
        return {
            "query_type": query_type.value,
            "extracted_terms": extracted_terms,
            "search_strategy": "semantic",
            "trace_context": {
                "trace_id": trace_context.trace_id if trace_context else trace_id,
                "session_id": trace_context.session_id if trace_context else None,
                "conversation_turn": trace_context.conversation_turn if trace_context else 0
            }
        }
    
    def _get_search_strategy(self, query_type: QueryType) -> callable:
        """Get the appropriate search strategy for the query type."""
        strategies = {
            QueryType.PRODUCT_COMPARISON: self._search_comparison,
            QueryType.PRODUCT_COMPLAINTS: self._search_complaints,
            QueryType.PRODUCT_RECOMMENDATION: self._search_recommendation,
            QueryType.USE_CASE: self._search_use_case,
            QueryType.PRODUCT_INFO: self._search_general,
            QueryType.PRODUCT_REVIEWS: self._search_general,
            QueryType.GENERAL_SEARCH: self._search_general
        }
        return strategies.get(query_type, self._search_general)
    
    def _search_comparison(self, context: QueryContext, query: str, extracted_terms: List[str], 
                          max_products: int, max_reviews: int) -> None:
        """Search strategy for product comparisons."""
        if len(extracted_terms) >= 2:
            # Search for each product separately
            for term in extracted_terms[:2]:
                self._add_product_results(context, term.strip(), n_results=2)
            
            # Get comparative reviews
            combined_query = " ".join(extracted_terms)
            self._add_review_results(context, combined_query, n_results=max_reviews)
    
    def _search_complaints(self, context: QueryContext, query: str, extracted_terms: List[str], 
                          max_products: int, max_reviews: int) -> None:
        """Search strategy for product complaints."""
        if extracted_terms:
            # Focus on negative sentiment
            negative_query = f"{extracted_terms[0]} problems issues complaints negative"
            self._add_review_results(context, negative_query, n_results=max_reviews * 2)
            
            # Also get product info for context
            self._add_product_results(context, extracted_terms[0], n_results=max_products // 2)
    
    def _search_recommendation(self, context: QueryContext, query: str, extracted_terms: List[str], 
                              max_products: int, max_reviews: int) -> None:
        """Search strategy for product recommendations."""
        if extracted_terms:
            # Check if it's a budget query
            is_budget_query = any(term in query.lower() for term in self.BUDGET_TERMS)
            
            # Search with appropriate filters
            price_range = (0, 100) if is_budget_query else None
            self._add_product_results(context, extracted_terms[0], n_results=max_products, 
                                    price_range=price_range)
            
            # Update metadata
            context.metadata["search_strategy"] = "recommendation"
            context.metadata["budget_query"] = is_budget_query
    
    def _search_use_case(self, context: QueryContext, query: str, extracted_terms: List[str], 
                        max_products: int, max_reviews: int) -> None:
        """Search strategy for use case queries."""
        if len(extracted_terms) >= 2:
            # Search for products suitable for use case
            use_case_query = f"{extracted_terms[0]} {extracted_terms[1]} suitable for"
            self._add_product_results(context, use_case_query, n_results=max_products)
            
            # Get reviews mentioning the use case
            self._add_review_results(context, extracted_terms[1], n_results=max_reviews)
    
    def _search_general(self, context: QueryContext, query: str, extracted_terms: List[str], 
                       max_products: int, max_reviews: int) -> None:
        """General search strategy using hybrid search."""
        try:
            hybrid_results = self.vector_db.hybrid_search(
                query, 
                n_results=max_products + max_reviews,
                include_products=True,
                include_reviews=True
            )
            
            if "error" not in hybrid_results:
                self._process_hybrid_results(context, hybrid_results, max_products, max_reviews)
        except Exception as e:
            logger.error(f"Error in general search: {e}")
            context.metadata["error"] = str(e)
    
    def _add_product_results(self, context: QueryContext, query: str, n_results: int, 
                           price_range: Optional[Tuple[float, float]] = None) -> None:
        """Add product search results to context."""
        try:
            product_results = self.vector_db.search_products(
                query, n_results=n_results, price_range=price_range
            )
            if "error" not in product_results:
                context.products.extend(self._format_search_results(product_results))
        except Exception as e:
            logger.error(f"Error searching products: {e}")
    
    def _add_review_results(self, context: QueryContext, query: str, n_results: int) -> None:
        """Add review search results to context."""
        try:
            review_results = self.vector_db.search_reviews(query, n_results=n_results)
            if "error" not in review_results:
                context.reviews.extend(self._format_search_results(review_results))
        except Exception as e:
            logger.error(f"Error searching reviews: {e}")
    
    def _process_hybrid_results(self, context: QueryContext, hybrid_results: Dict[str, Any], 
                               max_products: int, max_reviews: int) -> None:
        """Process hybrid search results and add to context."""
        formatted_results = self._format_search_results(hybrid_results)
        
        for result in formatted_results:
            if result["type"] == "product" and len(context.products) < max_products:
                context.products.append(result)
            elif result["type"] == "review_summary" and len(context.reviews) < max_reviews:
                context.reviews.append(result)
    
    @traceable
    def _format_search_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format search results into a consistent structure.
        
        Time Complexity: O(n) where n is number of results
        Space Complexity: O(n) for formatted results
        """
        if "results" not in search_results or "error" in search_results:
            return []
        
        formatted = []
        results = search_results["results"]
        
        # Safely extract results
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else []
        
        # Process each result
        for i in range(len(ids)):
            try:
                formatted.append({
                    "id": ids[i],
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                    "type": metadatas[i].get("type", "unknown") if i < len(metadatas) else "unknown"
                })
            except (IndexError, KeyError) as e:
                logger.warning(f"Error formatting result {i}: {e}")
                continue
        
        return formatted
    
    @traceable
    def generate_rag_prompt(self, context: QueryContext) -> str:
        """Generate enhanced prompt with RAG context using template approach."""
        if not context.has_results:
            return self._generate_no_results_prompt(context)
        
        # Build prompt sections
        sections = [
            self._generate_header_section(),
            self._generate_query_section(context),
            self._generate_products_section(context),
            self._generate_reviews_section(context),
            self._generate_instructions_section(context)
        ]
        
        # Filter out empty sections and join
        return "\n".join(filter(None, sections))
    
    def _generate_no_results_prompt(self, context: QueryContext) -> str:
        """Generate prompt when no results are found."""
        return (
            f"I couldn't find specific product information for your query: '{context.query}'. "
            f"Please provide a general response based on your knowledge about electronics products."
        )
    
    def _generate_header_section(self) -> str:
        """Generate header section of the prompt."""
        return (
            "You are an AI assistant helping users with Amazon Electronics products. "
            "Answer the user's question using the provided product information and customer reviews."
        )
    
    def _generate_query_section(self, context: QueryContext) -> str:
        """Generate query information section."""
        return f"\nUser Question: {context.query}\nQuery Type: {context.query_type.value}"
    
    def _generate_products_section(self, context: QueryContext) -> str:
        """Generate products section of the prompt."""
        if not context.products:
            return ""
        
        sections = ["\n=== PRODUCT INFORMATION ==="]
        
        for i, product in enumerate(context.products, 1):
            metadata = product.get("metadata", {})
            content = product.get("content", "")[:500]
            
            sections.append(f"""
Product {i}:
Title: {metadata.get('title', 'N/A')}
Price: ${metadata.get('price', 'N/A')}
Rating: {metadata.get('average_rating', 'N/A')}/5 ({metadata.get('rating_number', 'N/A')} ratings)
Store: {metadata.get('store', 'N/A')}
Content: {content}...
""")
        
        return "\n".join(sections)
    
    def _generate_reviews_section(self, context: QueryContext) -> str:
        """Generate reviews section of the prompt."""
        if not context.reviews:
            return ""
        
        sections = ["\n=== CUSTOMER REVIEWS SUMMARY ==="]
        
        for i, review in enumerate(context.reviews, 1):
            metadata = review.get("metadata", {})
            content = review.get("content", "")[:400]
            
            sections.append(f"""
Review Summary {i}:
Product: {metadata.get('product_title', 'N/A')}
Total Reviews: {metadata.get('total_reviews', 'N/A')}
Positive: {metadata.get('positive_reviews', 'N/A')}, Negative: {metadata.get('negative_reviews', 'N/A')}
Summary: {content}...
""")
        
        return "\n".join(sections)
    
    def _generate_instructions_section(self, context: QueryContext) -> str:
        """Generate instructions based on query type."""
        instructions = {
            QueryType.PRODUCT_COMPARISON: (
                "Compare the products based on features, price, ratings, and customer feedback. "
                "Highlight key differences and similarities."
            ),
            QueryType.PRODUCT_COMPLAINTS: (
                "Focus on the negative aspects and common complaints mentioned in the reviews. "
                "Be balanced and mention both positives and negatives."
            ),
            QueryType.PRODUCT_RECOMMENDATION: (
                "Recommend products based on the user's needs. "
                "Consider price, ratings, features, and customer satisfaction."
            ),
            QueryType.USE_CASE: (
                "Evaluate whether the products are suitable for the specific use case mentioned. "
                "Use product features and customer reviews."
            )
        }
        
        instruction = instructions.get(context.query_type, 
                                     "Provide a comprehensive answer using the product information and customer reviews.")
        
        return f"\n=== INSTRUCTIONS ===\n{instruction}\n\nPlease provide a helpful, accurate, and well-structured response based on the information above."
    
    @traceable
    def process_query(self, query: str, max_products: int = 5, max_reviews: int = 3) -> Dict[str, Any]:
        """Process query with enhanced tracing and context building."""
        
        # Start time for performance measurement
        start_time = time.time()
        
        # Generate unique trace ID
        trace_id = str(uuid.uuid4())
        
        logger.info(f"Processing query [trace_id={trace_id}]: {query}")
        
        try:
            # Build context with tracing
            context = self.build_context(query, max_products, max_reviews, trace_id)
            
            # Generate RAG prompt
            prompt = self.generate_rag_prompt(context)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create comprehensive result with performance metrics
            result = {
                "query": query,
                "context": {
                    "query_type": context.query_type.value,
                    "products": context.products,
                    "reviews": context.reviews,
                    "metadata": context.metadata,
                    "has_results": context.has_results,
                    "total_results": context.total_results
                },
                "prompt": prompt,
                "performance": {
                    "processing_time": processing_time,
                    "trace_id": trace_id,
                    "async_execution": False
                }
            }
            
            logger.info(f"Query processed successfully in {processing_time:.3f}s [trace_id={trace_id}]")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query processing failed after {processing_time:.3f}s [trace_id={trace_id}]: {e}")
            
            # Return error result
            return {
                "query": query,
                "error": str(e),
                "performance": {
                    "processing_time": processing_time,
                    "trace_id": trace_id,
                    "async_execution": False
                }
            }
    
    # ========== ASYNC METHODS FOR PERFORMANCE OPTIMIZATION ==========
    
    @traceable
    async def build_context_async(self, query: str, max_products: int = 5, max_reviews: int = 3, 
                                trace_id: Optional[str] = None) -> QueryContext:
        """Build context asynchronously for RAG query processing."""
        
        # Create trace context if available
        trace_context = None
        if _has_tracing:
            trace_context = create_enhanced_trace_context(
                trace_id=trace_id or str(uuid.uuid4()),
                operation="build_context_async",
                query=query,
                max_products=max_products,
                max_reviews=max_reviews
            )
        
        # Analyze query asynchronously
        query_type, extracted_terms = await self._analyze_query_async(query, trace_id)
        
        # Build initial context
        context = QueryContext(
            query=query,
            query_type=query_type,
            products=[],
            reviews=[],
            metadata={}
        )
        
        # Perform business analysis if available
        if _has_tracing and trace_context:
            await self._perform_business_analysis_async(query, trace_context)
        
        # Use appropriate search strategy
        search_strategy = self._get_search_strategy(query_type)
        
        # Execute search strategy asynchronously
        await self._execute_search_strategy_async(
            search_strategy, context, query, extracted_terms, max_products, max_reviews
        )
        
        # Build metadata
        context.metadata = self._build_metadata(query_type, extracted_terms, trace_context, trace_id)
        
        logger.info(f"Built context for query_type={query_type.value}, "
                   f"products={len(context.products)}, reviews={len(context.reviews)}")
        
        return context
    
    async def _analyze_query_async(self, query: str, trace_id: Optional[str] = None) -> Tuple[QueryType, List[str]]:
        """Analyze query asynchronously to determine type and extract terms."""
        
        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.analyze_query,
            query,
            trace_id
        )
    
    async def _perform_business_analysis_async(self, query: str, trace_context: Optional[TraceContext]) -> None:
        """Perform business analysis asynchronously."""
        if not (_has_tracing and business_analyzer):
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._perform_business_analysis,
            query,
            trace_context
        )
    
    async def _execute_search_strategy_async(self, search_strategy: callable, context: QueryContext, 
                                           query: str, extracted_terms: List[str], 
                                           max_products: int, max_reviews: int) -> None:
        """Execute search strategy asynchronously."""
        
        # Run search strategy in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            search_strategy,
            context,
            query,
            extracted_terms,
            max_products,
            max_reviews
        )
    
    async def _add_product_results_async(self, context: QueryContext, query: str, n_results: int, 
                                       price_range: Optional[Tuple[float, float]] = None) -> None:
        """Add product results asynchronously."""
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._add_product_results,
            context,
            query,
            n_results,
            price_range
        )
    
    async def _add_review_results_async(self, context: QueryContext, query: str, n_results: int) -> None:
        """Add review results asynchronously."""
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._add_review_results,
            context,
            query,
            n_results
        )
    
    @traceable
    async def process_query_async(self, query: str, max_products: int = 5, max_reviews: int = 3) -> Dict[str, Any]:
        """Process query asynchronously with enhanced tracing and context building."""
        
        # Start time for performance measurement
        start_time = asyncio.get_event_loop().time()
        
        # Generate unique trace ID
        trace_id = str(uuid.uuid4())
        
        logger.info(f"Processing query asynchronously [trace_id={trace_id}]: {query}")
        
        try:
            # Build context asynchronously
            context = await self.build_context_async(query, max_products, max_reviews, trace_id)
            
            # Generate RAG prompt
            prompt = self.generate_rag_prompt(context)
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create comprehensive result with performance metrics
            result = {
                "query": query,
                "context": {
                    "query_type": context.query_type.value,
                    "products": context.products,
                    "reviews": context.reviews,
                    "metadata": context.metadata,
                    "has_results": context.has_results,
                    "total_results": context.total_results
                },
                "prompt": prompt,
                "performance": {
                    "processing_time": processing_time,
                    "trace_id": trace_id,
                    "async_execution": True
                }
            }
            
            logger.info(f"Query processed successfully in {processing_time:.3f}s [trace_id={trace_id}]")
            
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Query processing failed after {processing_time:.3f}s [trace_id={trace_id}]: {e}")
            
            # Return error result
            return {
                "query": query,
                "error": str(e),
                "performance": {
                    "processing_time": processing_time,
                    "trace_id": trace_id,
                    "async_execution": True
                }
            }
    
    def __del__(self):
        """Clean up thread pool on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


@lru_cache(maxsize=1)
@traceable
def create_rag_processor(jsonl_path: Optional[str] = None) -> RAGQueryProcessor:
    """Create and initialize RAG query processor with caching."""
    return RAGQueryProcessor()


if __name__ == "__main__":
    # Test the RAG processor
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = create_rag_processor()
    
    # Test queries
    test_queries = [
        "What do people say about iPhone charger cables?",
        "Is the Fire TV good for streaming?",
        "Compare ethernet cables and USB cables",
        "What are the main complaints about laptop backpacks?",
        "Recommend a budget-friendly alternative to expensive tablets",
        ""  # Test empty query
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing: {query}")
        print('='*60)
        
        result = processor.process_query(query)
        
        if result["success"]:
            print(f"Query Type: {result['metadata']['query_type']}")
            print(f"Products Found: {result['metadata']['num_products']}")
            print(f"Reviews Found: {result['metadata']['num_reviews']}")
            print(f"Has Results: {result['metadata']['has_results']}")
            
            if result['metadata']['has_results']:
                print(f"\nEnhanced Prompt Preview:")
                print(result["enhanced_prompt"][:300] + "...")
        else:
            print(f"Error: {result['error']}")
        
        print(f"\nProcessing completed in {processor.__class__.__name__}")