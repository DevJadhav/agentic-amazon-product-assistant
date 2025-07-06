"""
Mock Vector Database for Fallback Operations
Provides basic functionality when vector database fails to load.
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class MockElectronicsVectorDB:
    """Mock vector database that provides fallback functionality."""
    
    def __init__(self):
        """Initialize the mock vector database."""
        logger.info("Initializing MockElectronicsVectorDB")
        self.products = []
        self.reviews = []
        self.collection_name = "Electronics"
        self.is_mock = True
        
        # Load some sample data if available
        self._load_sample_data()
    
    def _load_sample_data(self) -> None:
        """Load sample data for mock responses."""
        # Sample product data
        self.products = [
            {
                "id": "mock_product_1",
                "title": "iPhone Lightning Cable",
                "description": "High-quality lightning cable for iPhone charging",
                "price": 19.99,
                "rating": 4.5,
                "category": "Electronics",
                "type": "product"
            },
            {
                "id": "mock_product_2", 
                "title": "Samsung Galaxy USB-C Cable",
                "description": "Fast charging USB-C cable for Samsung devices",
                "price": 15.99,
                "rating": 4.3,
                "category": "Electronics",
                "type": "product"
            },
            {
                "id": "mock_product_3",
                "title": "Wireless Bluetooth Headphones",
                "description": "Noise-canceling wireless headphones with great sound quality",
                "price": 89.99,
                "rating": 4.7,
                "category": "Electronics",
                "type": "product"
            }
        ]
        
        # Sample review data
        self.reviews = [
            {
                "id": "mock_review_1",
                "product_id": "mock_product_1",
                "summary": "Great cable, works perfectly with my iPhone. Fast charging and durable.",
                "rating": 5,
                "sentiment": "positive",
                "type": "review_summary"
            },
            {
                "id": "mock_review_2",
                "product_id": "mock_product_2",
                "summary": "Good quality cable, charges quickly. No issues after 6 months of use.",
                "rating": 4,
                "sentiment": "positive", 
                "type": "review_summary"
            },
            {
                "id": "mock_review_3",
                "product_id": "mock_product_3",
                "summary": "Excellent sound quality and noise cancellation. Battery lasts all day.",
                "rating": 5,
                "sentiment": "positive",
                "type": "review_summary"
            }
        ]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "total_documents": len(self.products) + len(self.reviews),
            "document_types": {
                "product": len(self.products),
                "review_summary": len(self.reviews)
            },
            "collection_name": self.collection_name,
            "is_mock": True
        }
    
    def search_products(self, query: str, n_results: int = 5, 
                       price_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Search for products using simple keyword matching."""
        logger.info(f"Mock search for products with query: '{query}'")
        
        # Simple keyword matching
        query_lower = query.lower()
        matched_products = []
        
        for product in self.products:
            # Check if any query terms match product fields
            if (query_lower in product["title"].lower() or 
                query_lower in product["description"].lower() or
                any(term in product["title"].lower() for term in query_lower.split())):
                
                # Apply price filtering if specified
                if price_range:
                    min_price, max_price = price_range
                    if not (min_price <= product["price"] <= max_price):
                        continue
                
                matched_products.append(product)
        
        # If no matches, return all products (limited by n_results)
        if not matched_products:
            matched_products = self.products[:n_results]
        
        # Limit results
        matched_products = matched_products[:n_results]
        
        return {
            "results": {
                "ids": [[p["id"] for p in matched_products]],
                "documents": [[p["title"] + " - " + p["description"] for p in matched_products]],
                "metadatas": [[p for p in matched_products]],
                "distances": [[random.uniform(0.1, 0.9) for _ in matched_products]]
            },
            "query": query,
            "n_results": len(matched_products),
            "is_mock": True
        }
    
    def search_reviews(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Search for reviews using simple keyword matching."""
        logger.info(f"Mock search for reviews with query: '{query}'")
        
        # Simple keyword matching
        query_lower = query.lower()
        matched_reviews = []
        
        for review in self.reviews:
            # Check if any query terms match review fields
            if (query_lower in review["summary"].lower() or
                any(term in review["summary"].lower() for term in query_lower.split())):
                matched_reviews.append(review)
        
        # If no matches, return all reviews (limited by n_results)
        if not matched_reviews:
            matched_reviews = self.reviews[:n_results]
        
        # Limit results
        matched_reviews = matched_reviews[:n_results]
        
        return {
            "results": {
                "ids": [[r["id"] for r in matched_reviews]],
                "documents": [[r["summary"] for r in matched_reviews]],
                "metadatas": [[r for r in matched_reviews]],
                "distances": [[random.uniform(0.1, 0.9) for _ in matched_reviews]]
            },
            "query": query,
            "n_results": len(matched_reviews),
            "is_mock": True
        }
    
    def hybrid_search(self, query: str, n_results: int = 8, 
                     include_products: bool = True, 
                     include_reviews: bool = True) -> Dict[str, Any]:
        """Perform hybrid search combining products and reviews."""
        logger.info(f"Mock hybrid search with query: '{query}'")
        
        results = []
        
        # Search products if requested
        if include_products:
            product_results = self.search_products(query, n_results // 2)
            if product_results.get("results"):
                for i, doc in enumerate(product_results["results"]["documents"][0]):
                    results.append({
                        "id": product_results["results"]["ids"][0][i],
                        "document": doc,
                        "metadata": product_results["results"]["metadatas"][0][i],
                        "distance": product_results["results"]["distances"][0][i]
                    })
        
        # Search reviews if requested
        if include_reviews:
            review_results = self.search_reviews(query, n_results // 2)
            if review_results.get("results"):
                for i, doc in enumerate(review_results["results"]["documents"][0]):
                    results.append({
                        "id": review_results["results"]["ids"][0][i],
                        "document": doc,
                        "metadata": review_results["results"]["metadatas"][0][i],
                        "distance": review_results["results"]["distances"][0][i]
                    })
        
        # Limit total results
        results = results[:n_results]
        
        return {
            "results": {
                "ids": [[r["id"] for r in results]],
                "documents": [[r["document"] for r in results]],
                "metadatas": [[r["metadata"] for r in results]],
                "distances": [[r["distance"] for r in results]]
            },
            "query": query,
            "n_results": len(results),
            "is_mock": True
        }
    
    def delete_collection(self) -> bool:
        """Delete the collection (mock implementation)."""
        logger.info("Mock delete collection called")
        return True
    
    def close(self) -> None:
        """Close the database connection (mock implementation)."""
        logger.info("Closing mock vector database")
    
    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        self.close() 