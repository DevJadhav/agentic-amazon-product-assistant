"""
Enhanced Vector database implementation with hybrid retrieval and re-ranking.
Extends the base functionality with BM25 keyword search, re-ranking, and multiple indexes.
"""

import json
import logging
import os
import time
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject

from langsmith import traceable

# Import base vector database
from .vector_db_weaviate_simple import ElectronicsVectorDBSimple

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int
    search_type: str  # 'semantic', 'keyword', 'hybrid'


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    max_results: int = 20
    rerank_top_k: int = 10
    enable_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class EnhancedElectronicsVectorDB(ElectronicsVectorDBSimple):
    """Enhanced vector database with hybrid retrieval and re-ranking capabilities."""
    
    def __init__(self, data_directory: str = "data/weaviate_db", 
                 enable_keyword_search: bool = True,
                 enable_reranking: bool = True):
        """Initialize enhanced vector database."""
        super().__init__(data_directory)
        
        # Configuration
        self.enable_keyword_search = enable_keyword_search
        self.enable_reranking = enable_reranking
        
        # Initialize keyword search components
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.document_corpus = []
        self.document_metadata = []
        
        # Initialize re-ranking model
        self.rerank_model = None
        if enable_reranking:
            self._initialize_rerank_model()
        
        # Multiple indexes for different search strategies
        self.indexes = {
            "products": {"bm25": None, "tfidf": None, "docs": [], "metadata": []},
            "reviews": {"bm25": None, "tfidf": None, "docs": [], "metadata": []},
            "combined": {"bm25": None, "tfidf": None, "docs": [], "metadata": []}
        }
        
        # Build keyword search indexes
        if enable_keyword_search:
            self._build_keyword_indexes()
    
    def _initialize_rerank_model(self):
        """Initialize cross-encoder model for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            self.rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
            logger.info("Re-ranking model initialized successfully")
        except ImportError:
            logger.warning("CrossEncoder not available, disabling re-ranking")
            self.enable_reranking = False
        except Exception as e:
            logger.error(f"Failed to initialize re-ranking model: {e}")
            self.enable_reranking = False
    
    def _build_keyword_indexes(self):
        """Build BM25 and TF-IDF indexes for keyword search."""
        try:
            # Get all documents from Weaviate
            documents = self._get_all_documents()
            
            if not documents:
                logger.warning("No documents found for building keyword indexes")
                return
            
            # Separate documents by type
            product_docs = []
            review_docs = []
            all_docs = []
            
            product_metadata = []
            review_metadata = []
            all_metadata = []
            
            for doc in documents:
                content = doc.get("content", "")
                doc_type = doc.get("doc_type", "")
                
                all_docs.append(content)
                all_metadata.append(doc)
                
                if doc_type == "product":
                    product_docs.append(content)
                    product_metadata.append(doc)
                elif doc_type == "review_summary":
                    review_docs.append(content)
                    review_metadata.append(doc)
            
            # Build indexes for each document type
            self._build_index_for_type("products", product_docs, product_metadata)
            self._build_index_for_type("reviews", review_docs, review_metadata)
            self._build_index_for_type("combined", all_docs, all_metadata)
            
            logger.info(f"Built keyword indexes: {len(product_docs)} products, {len(review_docs)} reviews, {len(all_docs)} total")
            
        except Exception as e:
            logger.error(f"Failed to build keyword indexes: {e}")
            self.enable_keyword_search = False
    
    def _build_index_for_type(self, index_type: str, documents: List[str], metadata: List[Dict]):
        """Build BM25 and TF-IDF indexes for a specific document type."""
        if not documents:
            return
        
        try:
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in documents]
            
            # Build BM25 index
            bm25_index = BM25Okapi(tokenized_docs)
            
            # Build TF-IDF index
            tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            
            # Store indexes
            self.indexes[index_type] = {
                "bm25": bm25_index,
                "tfidf": tfidf_vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "docs": documents,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to build {index_type} index: {e}")
    
    def _get_all_documents(self) -> List[Dict]:
        """Retrieve all documents from Weaviate for indexing."""
        try:
            # Query all documents
            response = self.collection.query.fetch_objects(limit=10000)
            
            documents = []
            for obj in response.objects:
                doc = obj.properties.copy()
                doc["id"] = str(obj.uuid)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    @traceable
    def keyword_search(self, query: str, index_type: str = "combined", 
                      n_results: int = 10, method: str = "bm25") -> List[SearchResult]:
        """Perform keyword search using BM25 or TF-IDF."""
        if not self.enable_keyword_search or index_type not in self.indexes:
            return []
        
        index_data = self.indexes[index_type]
        if not index_data.get("bm25") or not index_data.get("docs"):
            return []
        
        try:
            if method == "bm25":
                return self._bm25_search(query, index_data, n_results)
            elif method == "tfidf":
                return self._tfidf_search(query, index_data, n_results)
            else:
                logger.warning(f"Unknown search method: {method}")
                return []
                
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, index_data: Dict, n_results: int) -> List[SearchResult]:
        """Perform BM25 search."""
        tokenized_query = query.lower().split()
        scores = index_data["bm25"].get_scores(tokenized_query)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include results with positive scores
                results.append(SearchResult(
                    id=index_data["metadata"][idx].get("id", f"doc_{idx}"),
                    content=index_data["docs"][idx],
                    metadata=index_data["metadata"][idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    search_type="keyword_bm25"
                ))
        
        return results
    
    def _tfidf_search(self, query: str, index_data: Dict, n_results: int) -> List[SearchResult]:
        """Perform TF-IDF search."""
        query_vector = index_data["tfidf"].transform([query])
        similarities = cosine_similarity(query_vector, index_data["tfidf_matrix"]).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0:  # Only include results with positive similarity
                results.append(SearchResult(
                    id=index_data["metadata"][idx].get("id", f"doc_{idx}"),
                    content=index_data["docs"][idx],
                    metadata=index_data["metadata"][idx],
                    score=float(similarities[idx]),
                    rank=rank + 1,
                    search_type="keyword_tfidf"
                ))
        
        return results
    
    @traceable
    def semantic_search(self, query: str, doc_type: Optional[str] = None, 
                       n_results: int = 10) -> List[SearchResult]:
        """Perform semantic search using vector embeddings."""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build where conditions
            where_conditions = {}
            if doc_type:
                where_conditions["doc_type"] = {"$eq": doc_type}
            
            # Perform search
            response = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=n_results,
                where=where_conditions if where_conditions else None
            )
            
            # Convert to SearchResult objects
            results = []
            for rank, obj in enumerate(response.objects):
                results.append(SearchResult(
                    id=str(obj.uuid),
                    content=obj.properties.get("content", ""),
                    metadata=obj.properties,
                    score=1.0 - getattr(obj, 'metadata', {}).get('distance', 0.0),
                    rank=rank + 1,
                    search_type="semantic"
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    @traceable
    def hybrid_search_enhanced(self, query: str, config: HybridSearchConfig = None, 
                             doc_type: Optional[str] = None) -> List[SearchResult]:
        """Perform enhanced hybrid search combining semantic and keyword search."""
        if config is None:
            config = HybridSearchConfig()
        
        # Determine index type based on doc_type
        if doc_type == "product":
            index_type = "products"
        elif doc_type == "review_summary":
            index_type = "reviews"
        else:
            index_type = "combined"
        
        # Perform semantic search
        semantic_results = self.semantic_search(
            query, doc_type=doc_type, n_results=config.max_results
        )
        
        # Perform keyword search
        keyword_results = []
        if self.enable_keyword_search:
            bm25_results = self.keyword_search(
                query, index_type=index_type, n_results=config.max_results, method="bm25"
            )
            tfidf_results = self.keyword_search(
                query, index_type=index_type, n_results=config.max_results, method="tfidf"
            )
            keyword_results = bm25_results + tfidf_results
        
        # Combine and normalize scores
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, config
        )
        
        # Re-rank if enabled
        if config.enable_reranking and self.enable_reranking:
            combined_results = self._rerank_results(query, combined_results, config.rerank_top_k)
        
        # Return top results
        return combined_results[:config.max_results]
    
    def _combine_search_results(self, semantic_results: List[SearchResult], 
                              keyword_results: List[SearchResult], 
                              config: HybridSearchConfig) -> List[SearchResult]:
        """Combine semantic and keyword search results with weighted scoring."""
        # Normalize scores
        semantic_scores = {r.id: r.score for r in semantic_results}
        keyword_scores = {}
        
        # Combine keyword results (take best score per document)
        for result in keyword_results:
            if result.id not in keyword_scores or result.score > keyword_scores[result.id]:
                keyword_scores[result.id] = result.score
        
        # Normalize scores to 0-1 range
        if semantic_scores:
            max_semantic = max(semantic_scores.values())
            min_semantic = min(semantic_scores.values())
            if max_semantic > min_semantic:
                semantic_scores = {k: (v - min_semantic) / (max_semantic - min_semantic) 
                                 for k, v in semantic_scores.items()}
        
        if keyword_scores:
            max_keyword = max(keyword_scores.values())
            min_keyword = min(keyword_scores.values())
            if max_keyword > min_keyword:
                keyword_scores = {k: (v - min_keyword) / (max_keyword - min_keyword) 
                                for k, v in keyword_scores.items()}
        
        # Combine results
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []
        
        # Create lookup for result objects
        result_lookup = {}
        for result in semantic_results + keyword_results:
            result_lookup[result.id] = result
        
        for doc_id in all_doc_ids:
            semantic_score = semantic_scores.get(doc_id, 0.0)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            
            # Weighted combination
            combined_score = (config.semantic_weight * semantic_score + 
                            config.keyword_weight * keyword_score)
            
            # Use the result object (prefer semantic if available)
            if doc_id in result_lookup:
                result = result_lookup[doc_id]
                result.score = combined_score
                result.search_type = "hybrid"
                combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    @traceable
    def _rerank_results(self, query: str, results: List[SearchResult], 
                       top_k: int) -> List[SearchResult]:
        """Re-rank results using cross-encoder model."""
        if not self.rerank_model or not results:
            return results
        
        try:
            # Prepare query-document pairs for re-ranking
            pairs = [(query, result.content) for result in results[:top_k]]
            
            # Get re-ranking scores
            rerank_scores = self.rerank_model.predict(pairs)
            
            # Update results with re-ranking scores
            for i, score in enumerate(rerank_scores):
                if i < len(results):
                    results[i].score = float(score)
                    results[i].search_type = "hybrid_reranked"
            
            # Sort by re-ranking score
            results[:top_k] = sorted(results[:top_k], key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results[:top_k]):
                result.rank = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results
    
    def save_indexes(self, filepath: str):
        """Save keyword search indexes to disk."""
        try:
            index_data = {
                "indexes": self.indexes,
                "enable_keyword_search": self.enable_keyword_search,
                "enable_reranking": self.enable_reranking
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"Indexes saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save indexes: {e}")
    
    def load_indexes(self, filepath: str):
        """Load keyword search indexes from disk."""
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.indexes = index_data.get("indexes", {})
            self.enable_keyword_search = index_data.get("enable_keyword_search", True)
            self.enable_reranking = index_data.get("enable_reranking", True)
            
            logger.info(f"Indexes loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load indexes: {e}")


@traceable
def setup_enhanced_vector_database(jsonl_path: str, 
                                  data_directory: str = "data/weaviate_db",
                                  enable_keyword_search: bool = True,
                                  enable_reranking: bool = True) -> EnhancedElectronicsVectorDB:
    """Set up enhanced vector database with hybrid retrieval capabilities."""
    
    logger.info("Setting up enhanced vector database with hybrid retrieval...")
    
    # Create enhanced vector database
    vector_db = EnhancedElectronicsVectorDB(
        data_directory=data_directory,
        enable_keyword_search=enable_keyword_search,
        enable_reranking=enable_reranking
    )
    
    # Load data if JSONL file exists
    if os.path.exists(jsonl_path):
        logger.info(f"Loading data from {jsonl_path}")
        # The base class handles data loading
        # We just need to rebuild indexes after loading
        if enable_keyword_search:
            vector_db._build_keyword_indexes()
    else:
        logger.warning(f"JSONL file not found: {jsonl_path}")
    
    logger.info("Enhanced vector database setup complete")
    return vector_db 