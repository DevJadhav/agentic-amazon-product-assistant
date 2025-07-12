"""
Enhanced Vector Database Implementation with GTE-large embeddings and advanced metadata filtering.
Provides production-grade semantic search with hybrid queries and intelligent filtering.
"""

import json
import logging
import os
import time
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum

from langsmith import traceable

logger = logging.getLogger(__name__)

class EmbeddingModel(Enum):
    """Supported embedding models with their configurations."""
    GTE_LARGE = "thenlper/gte-large"  # 1024 dimensions, high performance
    GTE_BASE = "thenlper/gte-base"    # 768 dimensions, balanced
    MINI_LM = "all-MiniLM-L6-v2"     # 384 dimensions, fast
    BGE_LARGE = "BAAI/bge-large-en-v1.5"  # 1024 dimensions, high quality

@dataclass
class SearchFilter:
    """Advanced search filter configuration."""
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    rating_min: Optional[float] = None
    rating_max: Optional[float] = None
    review_count_min: Optional[int] = None
    categories: Optional[List[str]] = None
    store: Optional[str] = None
    doc_types: Optional[List[str]] = None
    features_include: Optional[List[str]] = None
    features_exclude: Optional[List[str]] = None

@dataclass
class SearchConfig:
    """Search configuration with advanced options."""
    n_results: int = 10
    include_metadata: bool = True
    include_distances: bool = True
    rerank_results: bool = True
    hybrid_alpha: float = 0.7  # Weight for semantic vs keyword search
    diversity_threshold: float = 0.8  # Minimum diversity for results
    enable_caching: bool = True
    cache_ttl: int = 300  # Cache TTL in seconds

class EnhancedVectorDB:
    """Enhanced vector database with GTE-large embeddings and advanced filtering."""
    
    def __init__(self, 
                 data_directory: str = "data/weaviate_db",
                 embedding_model: EmbeddingModel = EmbeddingModel.GTE_LARGE,
                 enable_async: bool = True):
        """Initialize the enhanced vector database."""
        self.data_directory = data_directory
        self.collection_name = "EnhancedElectronics"
        self.embedding_model_enum = embedding_model
        self.embedding_model = embedding_model.value
        self.enable_async = enable_async
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model}")
        self.sentence_transformer = SentenceTransformer(self.embedding_model)
        self.embedding_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
        # Thread pool for async operations
        if enable_async:
            self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Caching system
        self.search_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize Weaviate client
        self.client = self._initialize_client()
        
        # Create enhanced collection
        self.create_enhanced_collection(delete_existing=False)
        
        # Performance metrics
        self.metrics = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0,
            'last_reset': datetime.now()
        }
    
    def _initialize_client(self) -> weaviate.Client:
        """Initialize Weaviate client with enhanced configuration."""
        weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
        weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
        weaviate_grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        
        try:
            if weaviate_host != "localhost":
                # Production environment
                logger.info(f"Connecting to Weaviate service at {weaviate_host}:{weaviate_port}")
                client = weaviate.connect_to_local(
                    host=weaviate_host,
                    port=weaviate_port,
                    grpc_port=weaviate_grpc_port,
                    headers={"X-OpenAI-Api-Key": "dummy"}
                )
            else:
                # Local development environment
                logger.info("Using embedded Weaviate for local development")
                client = weaviate.connect_to_embedded(
                    port=8079,
                    grpc_port=50050,
                    headers={"X-OpenAI-Api-Key": "dummy"},
                    environment_variables={
                        'QUERY_DEFAULTS_LIMIT': '25',
                        'DEFAULT_VECTORIZER_MODULE': 'none',
                        'ENABLE_MODULES': 'text2vec-transformers',
                        'TRANSFORMERS_INFERENCE_API': 'http://t2v-transformers:8080'
                    }
                )
            
            logger.info("Successfully connected to Weaviate")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise
    
    def create_enhanced_collection(self, delete_existing: bool = False) -> bool:
        """Create enhanced collection with comprehensive properties."""
        try:
            if self.client.collections.exists(self.collection_name):
                if delete_existing:
                    self.client.collections.delete(self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                else:
                    self.collection = self.client.collections.get(self.collection_name)
                    logger.info(f"Retrieved existing collection: {self.collection_name}")
                    return True
            
            # Enhanced properties with comprehensive metadata
            properties = [
                # Core content
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The main searchable content of the document"
                ),
                Property(
                    name="doc_type",
                    data_type=DataType.TEXT,
                    description="Type of document (product, review_summary, enhanced_product)"
                ),
                
                # Product identification
                Property(
                    name="parent_asin",
                    data_type=DataType.TEXT,
                    description="Parent ASIN of the product"
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Product title"
                ),
                Property(
                    name="description",
                    data_type=DataType.TEXT,
                    description="Product description"
                ),
                Property(
                    name="features",
                    data_type=DataType.TEXT,
                    description="Product features and specifications"
                ),
                
                # Pricing and ratings
                Property(
                    name="price",
                    data_type=DataType.NUMBER,
                    description="Current price of the product"
                ),
                Property(
                    name="average_rating",
                    data_type=DataType.NUMBER,
                    description="Average customer rating (1-5 scale)"
                ),
                Property(
                    name="rating_number",
                    data_type=DataType.INT,
                    description="Total number of ratings"
                ),
                Property(
                    name="review_count",
                    data_type=DataType.INT,
                    description="Total number of reviews"
                ),
                
                # Business metadata
                Property(
                    name="store",
                    data_type=DataType.TEXT,
                    description="Store or brand name"
                ),
                Property(
                    name="categories",
                    data_type=DataType.TEXT,
                    description="Product categories as JSON array"
                ),
                Property(
                    name="primary_category",
                    data_type=DataType.TEXT,
                    description="Primary category for filtering"
                ),
                Property(
                    name="subcategory",
                    data_type=DataType.TEXT,
                    description="Secondary category for filtering"
                ),
                
                # Enhanced analytics (from advanced processing)
                Property(
                    name="sentiment_score",
                    data_type=DataType.NUMBER,
                    description="Overall sentiment score from reviews"
                ),
                Property(
                    name="popularity_score",
                    data_type=DataType.NUMBER,
                    description="Popularity score based on reviews and ratings"
                ),
                Property(
                    name="price_competitiveness",
                    data_type=DataType.TEXT,
                    description="Price tier (budget, mid-range, premium, luxury)"
                ),
                Property(
                    name="review_velocity",
                    data_type=DataType.NUMBER,
                    description="Average reviews per month"
                ),
                Property(
                    name="rating_consistency",
                    data_type=DataType.NUMBER,
                    description="Rating consistency score (0-1)"
                ),
                
                # Temporal data
                Property(
                    name="first_review_date",
                    data_type=DataType.TEXT,
                    description="Date of first review (ISO format)"
                ),
                Property(
                    name="latest_review_date",
                    data_type=DataType.TEXT,
                    description="Date of latest review (ISO format)"
                ),
                Property(
                    name="peak_review_month",
                    data_type=DataType.TEXT,
                    description="Month with highest review volume"
                ),
                
                # Search optimization
                Property(
                    name="search_keywords",
                    data_type=DataType.TEXT,
                    description="Optimized search keywords for the product"
                ),
                Property(
                    name="feature_tags",
                    data_type=DataType.TEXT,
                    description="Extracted feature tags for filtering"
                ),
                
                # System metadata
                Property(
                    name="embedding_model",
                    data_type=DataType.TEXT,
                    description="Model used for generating embeddings"
                ),
                Property(
                    name="last_updated",
                    data_type=DataType.TEXT,
                    description="Last update timestamp"
                ),
                Property(
                    name="data_quality_score",
                    data_type=DataType.NUMBER,
                    description="Data quality score (0-1)"
                )
            ]
            
            # Create collection with enhanced configuration
            self.collection = self.client.collections.create(
                name=self.collection_name,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=512,  # Higher for better quality
                    max_connections=64,   # Higher for better recall
                    ef=-1,               # Dynamic ef
                    skip=False,
                    vector_cache_max_objects=100000,
                    flat_search_cutoff=40000,
                    cleanup_interval_seconds=300
                ),
                # Enable inverted index for faster filtering
                inverted_index_config=Configure.inverted_index(
                    bm25_b=0.75,
                    bm25_k1=1.2,
                    cleanup_interval_seconds=60,
                    stopwords_preset="en"
                )
            )
            
            logger.info(f"Created enhanced collection: {self.collection_name}")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create enhanced collection: {e}")
            return False
    
    def _generate_enhanced_embedding(self, text: str) -> np.ndarray:
        """Generate high-quality embedding using GTE-large or specified model."""
        try:
            # Preprocessing for better embeddings
            text = text.strip()
            if not text:
                return np.zeros(self.embedding_dimension, dtype=np.float32)
            
            # Truncate if too long (model-specific limits)
            max_length = 512 if "gte" in self.embedding_model.lower() else 256
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
            
            # Generate embedding with normalization
            embedding = self.sentence_transformer.encode(
                text,
                normalize_embeddings=True,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)[:100]}...")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def _build_search_filters(self, search_filter: SearchFilter) -> Dict[str, Any]:
        """Build comprehensive Weaviate filters from SearchFilter object."""
        conditions = []
        
        # Document type filter
        if search_filter.doc_types:
            if len(search_filter.doc_types) == 1:
                conditions.append({
                    "path": ["doc_type"],
                    "operator": "Equal",
                    "valueText": search_filter.doc_types[0]
                })
            else:
                conditions.append({
                    "path": ["doc_type"],
                    "operator": "ContainsAny",
                    "valueText": search_filter.doc_types
                })
        
        # Price range filter
        if search_filter.price_min is not None:
            conditions.append({
                "path": ["price"],
                "operator": "GreaterThanEqual",
                "valueNumber": search_filter.price_min
            })
        
        if search_filter.price_max is not None:
            conditions.append({
                "path": ["price"],
                "operator": "LessThanEqual",
                "valueNumber": search_filter.price_max
            })
        
        # Rating filter
        if search_filter.rating_min is not None:
            conditions.append({
                "path": ["average_rating"],
                "operator": "GreaterThanEqual",
                "valueNumber": search_filter.rating_min
            })
        
        if search_filter.rating_max is not None:
            conditions.append({
                "path": ["average_rating"],
                "operator": "LessThanEqual",
                "valueNumber": search_filter.rating_max
            })
        
        # Review count filter
        if search_filter.review_count_min is not None:
            conditions.append({
                "path": ["review_count"],
                "operator": "GreaterThanEqual",
                "valueInt": search_filter.review_count_min
            })
        
        # Category filter
        if search_filter.categories:
            for category in search_filter.categories:
                conditions.append({
                    "path": ["categories"],
                    "operator": "Like",
                    "valueText": f"*{category}*"
                })
        
        # Store filter
        if search_filter.store:
            conditions.append({
                "path": ["store"],
                "operator": "Equal",
                "valueText": search_filter.store
            })
        
        # Feature inclusion filter
        if search_filter.features_include:
            for feature in search_filter.features_include:
                conditions.append({
                    "path": ["features"],
                    "operator": "Like",
                    "valueText": f"*{feature}*"
                })
        
        # Feature exclusion filter
        if search_filter.features_exclude:
            for feature in search_filter.features_exclude:
                conditions.append({
                    "path": ["features"],
                    "operator": "NotEqual",
                    "valueText": f"*{feature}*"
                })
        
        # Combine conditions with AND operator
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {
                "operator": "And",
                "operands": conditions
            }
    
    def _get_cache_key(self, query: str, search_filter: SearchFilter, config: SearchConfig) -> str:
        """Generate cache key for search results."""
        import hashlib
        
        key_data = {
            'query': query,
            'filter': search_filter.__dict__ if search_filter else {},
            'config': {
                'n_results': config.n_results,
                'hybrid_alpha': config.hybrid_alpha,
                'diversity_threshold': config.diversity_threshold
            }
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if results are cached and still valid."""
        if not hasattr(self, 'search_cache'):
            return None
        
        with self.cache_lock:
            if cache_key in self.search_cache:
                cached_result, timestamp = self.search_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=300):  # 5 min TTL
                    self.metrics['cache_hits'] += 1
                    return cached_result
                else:
                    # Remove expired cache entry
                    del self.search_cache[cache_key]
        
        return None
    
    def _update_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Update cache with new results."""
        with self.cache_lock:
            # Limit cache size
            if len(self.search_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(self.search_cache.keys())[:100]
                for key in oldest_keys:
                    del self.search_cache[key]
            
            self.search_cache[cache_key] = (result, datetime.now())
    
    def _diversify_results(self, results: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """Remove similar results to increase diversity."""
        if len(results) <= 1:
            return results
        
        diversified = [results[0]]  # Always include the top result
        
        for candidate in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            candidate_embedding = self._generate_enhanced_embedding(
                candidate.get('content', '')
            )
            
            for selected in diversified:
                selected_embedding = self._generate_enhanced_embedding(
                    selected.get('content', '')
                )
                
                # Calculate cosine similarity
                similarity = np.dot(candidate_embedding, selected_embedding) / (
                    np.linalg.norm(candidate_embedding) * np.linalg.norm(selected_embedding)
                )
                
                if similarity > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diversified.append(candidate)
        
        return diversified
    
    @traceable
    def enhanced_search(self, 
                       query: str, 
                       search_filter: Optional[SearchFilter] = None,
                       config: Optional[SearchConfig] = None) -> Dict[str, Any]:
        """Perform enhanced semantic search with advanced filtering and caching."""
        start_time = time.time()
        
        if config is None:
            config = SearchConfig()
        
        if search_filter is None:
            search_filter = SearchFilter()
        
        self.metrics['total_searches'] += 1
        
        try:
            # Check cache if enabled
            cache_key = None
            if config.enable_caching:
                cache_key = self._get_cache_key(query, search_filter, config)
                cached_result = self._check_cache(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cached_result
            
            # Generate query embedding
            query_embedding = self._generate_enhanced_embedding(query)
            
            # Build filters
            where_filter = self._build_search_filters(search_filter)
            
            # Perform vector search
            search_params = {
                "near_vector": query_embedding.tolist(),
                "limit": min(config.n_results * 2, 100),  # Get extra for diversity filtering
                "return_metadata": ["distance", "score"] if config.include_distances else None
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            response = self.collection.query.near_vector(**search_params)
            
            # Process results
            results = []
            for obj in response.objects:
                result = {
                    'id': str(obj.uuid),
                    'content': obj.properties.get('content', ''),
                    'metadata': obj.properties if config.include_metadata else {},
                }
                
                if config.include_distances and hasattr(obj, 'metadata'):
                    result['distance'] = getattr(obj.metadata, 'distance', None)
                    result['score'] = getattr(obj.metadata, 'score', None)
                
                results.append(result)
            
            # Apply diversity filtering if enabled
            if config.diversity_threshold < 1.0 and len(results) > 1:
                results = self._diversify_results(results, config.diversity_threshold)
            
            # Limit to requested number of results
            results = results[:config.n_results]
            
            # Prepare response
            search_result = {
                "results": {
                    "ids": [[r['id'] for r in results]],
                    "documents": [[r['content'] for r in results]],
                    "metadatas": [[r['metadata'] for r in results]],
                    "distances": [[r.get('distance', 0.0) for r in results]] if config.include_distances else None
                },
                "query": query,
                "n_results": len(results),
                "search_time": time.time() - start_time,
                "embedding_model": self.embedding_model,
                "filters_applied": search_filter.__dict__,
                "from_cache": False
            }
            
            # Update cache
            if config.enable_caching and cache_key:
                self._update_cache(cache_key, search_result)
            
            # Update metrics
            self.metrics['avg_search_time'] = (
                (self.metrics['avg_search_time'] * (self.metrics['total_searches'] - 1) + 
                 search_result['search_time']) / self.metrics['total_searches']
            )
            
            logger.debug(f"Enhanced search completed in {search_result['search_time']:.3f}s")
            return search_result
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "search_time": time.time() - start_time
            }
    
    def hybrid_search_with_keywords(self, 
                                   query: str,
                                   keywords: List[str],
                                   search_filter: Optional[SearchFilter] = None,
                                   config: Optional[SearchConfig] = None) -> Dict[str, Any]:
        """Perform hybrid search combining semantic similarity and keyword matching."""
        if config is None:
            config = SearchConfig()
        
        try:
            # Perform semantic search
            semantic_results = self.enhanced_search(query, search_filter, config)
            
            # Perform keyword-based search
            keyword_query = " ".join(keywords)
            keyword_filter = search_filter or SearchFilter()
            
            # Build keyword search using BM25
            where_filter = self._build_search_filters(keyword_filter)
            
            keyword_response = self.collection.query.bm25(
                query=keyword_query,
                limit=config.n_results,
                where=where_filter
            )
            
            # Process keyword results
            keyword_results = []
            for obj in keyword_response.objects:
                result = {
                    'id': str(obj.uuid),
                    'content': obj.properties.get('content', ''),
                    'metadata': obj.properties,
                    'score': getattr(obj.metadata, 'score', 0.0) if hasattr(obj, 'metadata') else 0.0
                }
                keyword_results.append(result)
            
            # Combine results using weighted scoring
            combined_results = {}
            alpha = config.hybrid_alpha
            
            # Add semantic results
            if 'results' in semantic_results and semantic_results['results']['ids']:
                for i, (id_, content, metadata, distance) in enumerate(zip(
                    semantic_results['results']['ids'][0],
                    semantic_results['results']['documents'][0],
                    semantic_results['results']['metadatas'][0],
                    semantic_results['results']['distances'][0] if semantic_results['results']['distances'] else [0.0] * len(semantic_results['results']['ids'][0])
                )):
                    semantic_score = 1.0 - distance  # Convert distance to similarity
                    combined_results[id_] = {
                        'id': id_,
                        'content': content,
                        'metadata': metadata,
                        'semantic_score': semantic_score,
                        'keyword_score': 0.0,
                        'hybrid_score': alpha * semantic_score
                    }
            
            # Add keyword results
            for i, result in enumerate(keyword_results):
                id_ = result['id']
                keyword_score = result['score']
                
                if id_ in combined_results:
                    combined_results[id_]['keyword_score'] = keyword_score
                    combined_results[id_]['hybrid_score'] = (
                        alpha * combined_results[id_]['semantic_score'] + 
                        (1 - alpha) * keyword_score
                    )
                else:
                    combined_results[id_] = {
                        'id': id_,
                        'content': result['content'],
                        'metadata': result['metadata'],
                        'semantic_score': 0.0,
                        'keyword_score': keyword_score,
                        'hybrid_score': (1 - alpha) * keyword_score
                    }
            
            # Sort by hybrid score and limit results
            final_results = sorted(
                combined_results.values(),
                key=lambda x: x['hybrid_score'],
                reverse=True
            )[:config.n_results]
            
            return {
                "results": {
                    "ids": [[r['id'] for r in final_results]],
                    "documents": [[r['content'] for r in final_results]],
                    "metadatas": [[r['metadata'] for r in final_results]],
                    "hybrid_scores": [[r['hybrid_score'] for r in final_results]],
                    "semantic_scores": [[r['semantic_score'] for r in final_results]],
                    "keyword_scores": [[r['keyword_score'] for r in final_results]]
                },
                "query": query,
                "keywords": keywords,
                "n_results": len(final_results),
                "hybrid_alpha": alpha,
                "embedding_model": self.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"error": str(e), "query": query, "keywords": keywords}
    
    def get_recommendations(self, 
                           product_id: str,
                           recommendation_type: str = "similar",
                           n_results: int = 5) -> Dict[str, Any]:
        """Get product recommendations based on a given product."""
        try:
            # Get the source product
            source_response = self.collection.query.fetch_object_by_id(product_id)
            
            if not source_response:
                return {"error": "Product not found", "product_id": product_id}
            
            source_product = source_response.properties
            
            if recommendation_type == "similar":
                # Find similar products by content similarity
                content = source_product.get('content', '')
                search_filter = SearchFilter(doc_types=['product'])
                
                results = self.enhanced_search(
                    content,
                    search_filter=search_filter,
                    config=SearchConfig(n_results=n_results + 1)  # +1 to exclude self
                )
                
                # Filter out the source product
                if 'results' in results and results['results']['ids']:
                    filtered_ids = []
                    filtered_docs = []
                    filtered_meta = []
                    
                    for i, id_ in enumerate(results['results']['ids'][0]):
                        if id_ != product_id:
                            filtered_ids.append(id_)
                            filtered_docs.append(results['results']['documents'][0][i])
                            filtered_meta.append(results['results']['metadatas'][0][i])
                    
                    results['results']['ids'] = [filtered_ids[:n_results]]
                    results['results']['documents'] = [filtered_docs[:n_results]]
                    results['results']['metadatas'] = [filtered_meta[:n_results]]
            
            elif recommendation_type == "complementary":
                # Find complementary products in same category
                categories = source_product.get('categories', '')
                price = source_product.get('price', 0)
                
                # Look for products in same category but different price range
                search_filter = SearchFilter(
                    doc_types=['product'],
                    categories=[categories] if categories else None,
                    price_min=price * 0.5 if price else None,
                    price_max=price * 1.5 if price else None
                )
                
                category_search = source_product.get('primary_category', 'electronics')
                results = self.enhanced_search(
                    category_search,
                    search_filter=search_filter,
                    config=SearchConfig(n_results=n_results)
                )
            
            elif recommendation_type == "upgrade":
                # Find higher-rated or more expensive products in same category
                rating = source_product.get('average_rating', 0)
                price = source_product.get('price', 0)
                
                search_filter = SearchFilter(
                    doc_types=['product'],
                    rating_min=max(rating, 4.0),
                    price_min=price * 1.2 if price else None
                )
                
                results = self.enhanced_search(
                    source_product.get('title', ''),
                    search_filter=search_filter,
                    config=SearchConfig(n_results=n_results)
                )
            
            else:
                return {"error": f"Unknown recommendation type: {recommendation_type}"}
            
            results['recommendation_type'] = recommendation_type
            results['source_product_id'] = product_id
            return results
            
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return {"error": str(e), "product_id": product_id}
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced vector database."""
        try:
            # Get basic collection stats
            basic_stats = self.get_collection_stats()
            
            # Get performance metrics
            performance_stats = {
                'total_searches': self.metrics['total_searches'],
                'cache_hits': self.metrics['cache_hits'],
                'cache_hit_rate': self.metrics['cache_hits'] / max(self.metrics['total_searches'], 1),
                'average_search_time': self.metrics['avg_search_time'],
                'last_reset': self.metrics['last_reset'].isoformat()
            }
            
            # Get embedding model info
            model_stats = {
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.embedding_dimension,
                'model_enum': self.embedding_model_enum.value
            }
            
            # Combine all stats
            enhanced_stats = {
                **basic_stats,
                'performance': performance_stats,
                'model': model_stats,
                'cache_size': len(self.search_cache),
                'async_enabled': self.enable_async
            }
            
            return enhanced_stats
            
        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return {"error": str(e), "async_enabled": self.enable_async}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get basic collection statistics."""
        try:
            # Get total count
            total_response = self.collection.aggregate.over_all(total_count=True)
            total_count = total_response.total_count
            
            # Get document type breakdown
            doc_types = {}
            for doc_type in ['product', 'review_summary', 'enhanced_product']:
                try:
                    type_response = self.collection.aggregate.over_all(
                        total_count=True,
                        where={
                            "path": ["doc_type"],
                            "operator": "Equal",
                            "valueText": doc_type
                        }
                    )
                    doc_types[doc_type] = type_response.total_count
                except Exception:
                    doc_types[doc_type] = 0
            
            return {
                "total_documents": total_count,
                "document_types": doc_types,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "is_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def close(self) -> None:
        """Close connections and clean up resources."""
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=True)
            if hasattr(self, 'client'):
                self.client.close()
            logger.info("Enhanced vector database connections closed")
        except Exception as e:
            logger.error(f"Error closing enhanced vector database: {e}")
    
    def __del__(self) -> None:
        """Cleanup on object deletion."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup
    
    # ========== ASYNC METHODS FOR PERFORMANCE OPTIMIZATION ==========
    
    @traceable
    async def enhanced_search_async(self, 
                                  query: str, 
                                  search_filter: Optional[SearchFilter] = None,
                                  config: Optional[SearchConfig] = None) -> Dict[str, Any]:
        """Perform enhanced semantic search asynchronously."""
        
        if not self.enable_async:
            # Fallback to synchronous if async is disabled
            return self.enhanced_search(query, search_filter, config)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.enhanced_search,
            query,
            search_filter,
            config
        )
    
    async def similarity_search_async(self, 
                                    query: str, 
                                    k: int = 5,
                                    search_filter: Optional[SearchFilter] = None) -> List[Dict[str, Any]]:
        """Perform similarity search asynchronously."""
        
        if not self.enable_async:
            return self.similarity_search(query, k, search_filter)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.similarity_search,
            query,
            k,
            search_filter
        )
    
    async def hybrid_search_async(self, 
                                query: str, 
                                search_filter: Optional[SearchFilter] = None,
                                config: Optional[SearchConfig] = None) -> Dict[str, Any]:
        """Perform hybrid search asynchronously."""
        
        if not self.enable_async:
            return self.hybrid_search(query, search_filter, config)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.hybrid_search,
            query,
            search_filter,
            config
        )
    
    async def get_collection_stats_async(self) -> Dict[str, Any]:
        """Get collection statistics asynchronously."""
        
        if not self.enable_async:
            return self.get_collection_stats()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_collection_stats
        )
    
    async def get_enhanced_stats_async(self) -> Dict[str, Any]:
        """Get enhanced statistics asynchronously."""
        
        if not self.enable_async:
            return self.get_enhanced_stats()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_enhanced_stats
        )
    
    async def _generate_enhanced_embedding_async(self, text: str) -> np.ndarray:
        """Generate enhanced embedding asynchronously."""
        
        if not self.enable_async:
            return self._generate_enhanced_embedding(text)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_enhanced_embedding,
            text
        )
    
    async def _store_enhanced_batch_async(self, batch_data: List[Dict]) -> bool:
        """Store enhanced batch asynchronously."""
        
        if not self.enable_async:
            return self._store_enhanced_batch(batch_data)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._store_enhanced_batch,
            batch_data
        )


@traceable
def setup_enhanced_vector_database(jsonl_path: str, 
                                  embedding_model: EmbeddingModel = EmbeddingModel.GTE_LARGE,
                                  data_directory: str = "data/weaviate_db") -> EnhancedVectorDB:
    """Set up and populate the enhanced vector database."""
    logger.info(f"Setting up enhanced vector database with {embedding_model.value}")
    
    # Initialize enhanced database
    db = EnhancedVectorDB(
        data_directory=data_directory,
        embedding_model=embedding_model,
        enable_async=True
    )
    
    # Load and process documents
    try:
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    
                    # Enhance document with additional metadata
                    enhanced_doc = _enhance_document_metadata(doc)
                    documents.append(enhanced_doc)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {jsonl_path}")
        
        # Process documents in optimized batches
        batch_size = 50  # Smaller batches for GTE-large
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            success = db._store_enhanced_batch(batch)
            
            if success:
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            else:
                logger.error(f"Failed to process batch {i//batch_size + 1}")
        
        # Log final statistics
        stats = db.get_enhanced_stats()
        logger.info(f"Enhanced database setup complete: {stats}")
        
        return db
        
    except Exception as e:
        logger.error(f"Failed to setup enhanced vector database: {e}")
        raise


def _enhance_document_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance document with additional metadata for better search and filtering."""
    enhanced_doc = doc.copy()
    
    # Extract primary category
    categories = doc.get('categories', [])
    if isinstance(categories, list) and len(categories) > 0:
        enhanced_doc['primary_category'] = categories[0]
        enhanced_doc['subcategory'] = categories[1] if len(categories) > 1 else categories[0]
    elif isinstance(categories, str):
        try:
            categories_list = json.loads(categories)
            enhanced_doc['primary_category'] = categories_list[0] if categories_list else 'electronics'
            enhanced_doc['subcategory'] = categories_list[1] if len(categories_list) > 1 else categories_list[0]
        except:
            enhanced_doc['primary_category'] = 'electronics'
            enhanced_doc['subcategory'] = 'electronics'
    
    # Calculate popularity score
    review_count = doc.get('review_count', 0)
    avg_rating = doc.get('average_rating', 0)
    enhanced_doc['popularity_score'] = (review_count * 0.7 + avg_rating * 100 * 0.3) if review_count > 0 else 0
    
    # Determine price competitiveness
    price = doc.get('price')
    if price:
        if price < 25:
            enhanced_doc['price_competitiveness'] = 'budget'
        elif price < 100:
            enhanced_doc['price_competitiveness'] = 'mid-range'
        elif price < 500:
            enhanced_doc['price_competitiveness'] = 'premium'
        else:
            enhanced_doc['price_competitiveness'] = 'luxury'
    else:
        enhanced_doc['price_competitiveness'] = 'unknown'
    
    # Extract search keywords
    title = doc.get('title', '')
    features = doc.get('features', '')
    description = doc.get('description', '')
    
    # Simple keyword extraction
    import re
    text = f"{title} {features} {description}".lower()
    keywords = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    enhanced_doc['search_keywords'] = ' '.join(set(keywords)[:20])  # Top 20 unique keywords
    
    # Add system metadata
    enhanced_doc['embedding_model'] = EmbeddingModel.GTE_LARGE.value
    enhanced_doc['last_updated'] = datetime.now().isoformat()
    enhanced_doc['data_quality_score'] = _calculate_data_quality_score(doc)
    
    return enhanced_doc


def _calculate_data_quality_score(doc: Dict[str, Any]) -> float:
    """Calculate a data quality score for the document."""
    score = 0.0
    max_score = 10.0
    
    # Check presence of key fields
    if doc.get('title'):
        score += 2.0
    if doc.get('content'):
        score += 2.0
    if doc.get('price') is not None:
        score += 1.0
    if doc.get('average_rating') is not None:
        score += 1.0
    if doc.get('review_count', 0) > 0:
        score += 1.0
    if doc.get('categories'):
        score += 1.0
    if doc.get('features'):
        score += 1.0
    if doc.get('description'):
        score += 1.0
    
    return min(score / max_score, 1.0)


# Add method to EnhancedVectorDB class
def _store_enhanced_batch(self, batch_data: List[Dict]) -> bool:
    """Store batch of enhanced documents with metadata."""
    try:
        objects_to_insert = []
        for doc in batch_data:
            # Generate embedding for content
            content = doc.get("content", "")
            if not content:
                # Create content from available fields
                title = doc.get('title', '')
                description = doc.get('description', '')
                features = doc.get('features', '')
                content = f"{title} {description} {features}".strip()
            
            embedding = self._generate_enhanced_embedding(content)
            
            # Prepare properties (exclude system fields)
            properties = {k: v for k, v in doc.items() 
                         if k not in ['id', 'vector'] and v is not None}
            
            # Ensure required fields have default values
            properties.setdefault('doc_type', 'product')
            properties.setdefault('content', content)
            
            # Create DataObject
            data_object = DataObject(
                properties=properties,
                vector=embedding.tolist()
            )
            objects_to_insert.append(data_object)
        
        # Batch insert to Weaviate
        response = self.collection.data.insert_many(objects_to_insert)
        
        if hasattr(response, 'failed_objects') and response.failed_objects:
            logger.warning(f"Failed to insert {len(response.failed_objects)} objects")
            for failed_obj in response.failed_objects[:3]:
                logger.warning(f"Failed object error: {failed_obj.message}")
            return len(response.failed_objects) == 0
        
        logger.debug(f"Successfully stored {len(objects_to_insert)} enhanced documents")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store enhanced batch: {e}")
        return False

# Monkey patch the method to the class
EnhancedVectorDB._store_enhanced_batch = _store_enhanced_batch