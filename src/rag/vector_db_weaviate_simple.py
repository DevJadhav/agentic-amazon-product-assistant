"""
Simplified Vector database implementation for Amazon Electronics RAG system.
Handles document ingestion, embedding, and retrieval using Weaviate only (no FAISS).
"""

import json
import logging
import os
import time
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from langsmith import traceable

logger = logging.getLogger(__name__)


class ElectronicsVectorDBSimple:
    """Simplified vector database for Amazon Electronics using only Weaviate."""
    
    def __init__(self, data_directory: str = "data/weaviate_db"):
        """Initialize the vector database with Weaviate backend only."""
        self.data_directory = data_directory
        self.collection_name = "Electronics"
        
        # Initialize embedding model
        self.embedding_model = self._get_default_embedding_model()
        self.sentence_transformer = SentenceTransformer(self.embedding_model)
        self.embedding_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
        
        # Initialize Weaviate client
        self.client = self._initialize_client()
        
        # Create collection
        self.create_collection(delete_existing=False)
    
    def _get_default_embedding_model(self) -> str:
        """Get default embedding model name - using all-MiniLM-L6-v2 for stability."""
        return "all-MiniLM-L6-v2"  # Smaller, more stable model
    
    def _initialize_client(self) -> weaviate.Client:
        """Initialize Weaviate client with Docker and local support."""
        # Check if we're in a Docker environment (Weaviate service available)
        weaviate_host = os.getenv("WEAVIATE_HOST")
        weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
        weaviate_grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        
        try:
            if weaviate_host:
                # Docker environment - connect to Weaviate service
                logger.info(f"Connecting to Weaviate service at {weaviate_host}:{weaviate_port}")
                client = weaviate.connect_to_local(
                    host=weaviate_host,
                    port=weaviate_port,
                    grpc_port=weaviate_grpc_port,
                    headers={"X-OpenAI-Api-Key": "dummy"}  # Dummy key since we're using custom embeddings
                )
                logger.info("Successfully connected to Weaviate service")
            else:
                # Local environment - use embedded Weaviate
                logger.info("Using embedded Weaviate for local development")
                client = weaviate.connect_to_embedded(
                    port=8079,  # Use different port to avoid conflicts
                    grpc_port=50050,  # Use different port to avoid conflicts
                    headers={"X-OpenAI-Api-Key": "dummy"}  # Dummy key since we're using custom embeddings
                )
                logger.info("Successfully connected to embedded Weaviate")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise
    
    def create_collection(self, delete_existing: bool = False) -> bool:
        """Create or get the collection for storing documents."""
        try:
            # Check if collection exists
            if self.client.collections.exists(self.collection_name):
                if delete_existing:
                    self.client.collections.delete(self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                else:
                    self.collection = self.client.collections.get(self.collection_name)
                    logger.info(f"Retrieved existing collection: {self.collection_name}")
                    return True
            
            # Define properties for the collection
            properties = [
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The main content of the document"
                ),
                Property(
                    name="doc_type",
                    data_type=DataType.TEXT,
                    description="Type of document (product or review_summary)"
                ),
                Property(
                    name="parent_asin",
                    data_type=DataType.TEXT,
                    description="Parent ASIN of the product"
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Title of the product"
                ),
                Property(
                    name="price",
                    data_type=DataType.NUMBER,
                    description="Price of the product"
                ),
                Property(
                    name="average_rating",
                    data_type=DataType.NUMBER,
                    description="Average rating of the product"
                ),
                Property(
                    name="rating_number",
                    data_type=DataType.INT,
                    description="Number of ratings"
                ),
                Property(
                    name="review_count",
                    data_type=DataType.INT,
                    description="Number of reviews"
                ),
                Property(
                    name="store",
                    data_type=DataType.TEXT,
                    description="Store name"
                ),
                Property(
                    name="categories",
                    data_type=DataType.TEXT,
                    description="Product categories as JSON string"
                ),
                Property(
                    name="features",
                    data_type=DataType.TEXT,
                    description="Product features"
                ),
            ]
            
            # Create new collection
            self.collection = self.client.collections.create(
                name=self.collection_name,
                properties=properties,
                # Configure for custom embeddings
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            return False
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using SentenceTransformer."""
        try:
            embedding = self.sentence_transformer.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def _store_batch(self, batch_data: List[Dict]) -> bool:
        """Store batch of documents with embeddings using proper DataObject structure."""
        try:
            objects_to_insert = []
            for doc in batch_data:
                # Generate embedding for content
                embedding = self._generate_embedding(doc.get("content", ""))
                
                # Create proper properties dict without forbidden fields
                properties = {k: v for k, v in doc.items() if k not in ['id', 'vector']}
                
                # Create DataObject with proper structure
                data_object = DataObject(
                    properties=properties,
                    vector=embedding.tolist()
                )
                objects_to_insert.append(data_object)
            
            # Batch insert to Weaviate
            response = self.collection.data.insert_many(objects_to_insert)
            if hasattr(response, 'failed_objects') and response.failed_objects:
                logger.warning(f"Failed to insert {len(response.failed_objects)} objects")
                for failed_obj in response.failed_objects[:3]:  # Log first 3 failures
                    logger.warning(f"Failed object error: {failed_obj.message}")
                return False
            
            logger.info(f"Successfully stored {len(objects_to_insert)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            return False
    
    def search_products(self, query: str, n_results: int = 5, 
                       price_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Search for products using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build where conditions
            where_conditions = {"doc_type": {"$eq": "product"}}
            if price_range:
                where_conditions["price"] = {
                    "$gte": price_range[0], 
                    "$lte": price_range[1]
                }
            
            # Perform search
            response = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=n_results,
                where=where_conditions
            )
            
            # Format results
            ids = [obj.uuid for obj in response.objects]
            documents = [obj.properties.get("content", "") for obj in response.objects]
            metadatas = [obj.properties for obj in response.objects]
            distances = [getattr(obj, 'metadata', {}).get('distance', 0.0) for obj in response.objects]
            
            return {
                "results": {
                    "ids": [ids],
                    "documents": [documents],
                    "metadatas": [metadatas],
                    "distances": [distances]
                },
                "query": query,
                "n_results": len(ids)
            }
            
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return {"error": str(e), "query": query}
    
    def search_reviews(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Search for review summaries using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build where conditions
            where_conditions = {"doc_type": {"$eq": "review_summary"}}
            
            # Perform search
            response = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=n_results,
                where=where_conditions
            )
            
            # Format results
            ids = [obj.uuid for obj in response.objects]
            documents = [obj.properties.get("content", "") for obj in response.objects]
            metadatas = [obj.properties for obj in response.objects]
            distances = [getattr(obj, 'metadata', {}).get('distance', 0.0) for obj in response.objects]
            
            return {
                "results": {
                    "ids": [ids],
                    "documents": [documents],
                    "metadatas": [metadatas],
                    "distances": [distances]
                },
                "query": query,
                "n_results": len(ids)
            }
            
        except Exception as e:
            logger.error(f"Review search failed: {e}")
            return {"error": str(e), "query": query}
    
    def hybrid_search(self, query: str, n_results: int = 8, 
                     include_products: bool = True, 
                     include_reviews: bool = True) -> Dict[str, Any]:
        """Perform hybrid search combining products and reviews."""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build where conditions for hybrid search
            doc_types = []
            if include_products:
                doc_types.append("product")
            if include_reviews:
                doc_types.append("review_summary")
            
            where_conditions = {"doc_type": {"$in": doc_types}}
            
            # Perform search
            response = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=n_results,
                where=where_conditions
            )
            
            # Format results
            ids = [obj.uuid for obj in response.objects]
            documents = [obj.properties.get("content", "") for obj in response.objects]
            metadatas = [obj.properties for obj in response.objects]
            distances = [getattr(obj, 'metadata', {}).get('distance', 0.0) for obj in response.objects]
            
            return {
                "results": {
                    "ids": [ids],
                    "documents": [documents],
                    "metadatas": [metadatas],
                    "distances": [distances]
                },
                "query": query,
                "n_results": len(ids)
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"error": str(e), "query": query}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics with proper API usage."""
        try:
            # Get total count using proper v4 API
            total_response = self.collection.aggregate.over_all(total_count=True)
            total_count = total_response.total_count
            
            # Get counts by document type using separate queries  
            try:
                product_response = self.collection.aggregate.over_all(
                    total_count=True,
                    where={
                        "path": ["doc_type"],
                        "operator": "Equal", 
                        "valueText": "product"
                    }
                )
                product_count = product_response.total_count
            except Exception as e:
                logger.warning(f"Failed to get product count: {e}")
                product_count = 0
            
            try:
                review_response = self.collection.aggregate.over_all(
                    total_count=True,
                    where={
                        "path": ["doc_type"],
                        "operator": "Equal",
                        "valueText": "review_summary"
                    }
                )
                review_count = review_response.total_count
            except Exception as e:
                logger.warning(f"Failed to get review count: {e}")
                review_count = 0
            
            return {
                "total_documents": total_count,
                "document_types": {
                    "product": product_count,
                    "review_summary": review_count
                },
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "is_mock": False
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_documents": 0,
                "document_types": {"product": 0, "review_summary": 0},
                "collection_name": self.collection_name,
                "error": str(e),
                "is_mock": False
            }
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def close(self) -> None:
        """Close the database connection."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info("Closed Weaviate connection")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        self.close()


@traceable
def setup_vector_database_simple(jsonl_path: str, data_directory: str = "data/weaviate_db") -> ElectronicsVectorDBSimple:
    """Set up and populate the simplified vector database with documents from JSONL file."""
    logger.info(f"Setting up simplified vector database from {jsonl_path}")
    
    # Initialize database
    db = ElectronicsVectorDBSimple(data_directory)
    
    # Load and process documents
    try:
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {jsonl_path}")
        
        # Process documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            success = db._store_batch(batch)
            if success:
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            else:
                logger.error(f"Failed to process batch {i//batch_size + 1}")
        
        # Log final statistics
        stats = db.get_collection_stats()
        logger.info(f"Database setup complete: {stats}")
        
        return db
        
    except Exception as e:
        logger.error(f"Failed to setup vector database: {e}")
        raise 