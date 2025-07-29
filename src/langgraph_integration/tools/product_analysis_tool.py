"""
Product analysis tool for LangGraph agent workflows.
Provides product feature comparison and analysis capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProductAnalysisInput(BaseModel):
    """Input schema for product analysis tool."""
    
    products: List[Dict[str, Any]] = Field(
        description="List of product data to analyze"
    )
    analysis_type: str = Field(
        default="comparison",
        description="Type of analysis: 'comparison', 'features', 'pricing', 'ratings'"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus on (e.g., ['price', 'rating', 'features'])"
    )
    include_summary: bool = Field(
        default=True,
        description="Whether to include analysis summary"
    )


class ProductAnalysisTool(BaseTool):
    """Tool for analyzing product features and specifications."""
    
    name: str = "product_analysis"
    description: str = """
    Analyze and compare Amazon Electronics products based on their features, specifications, 
    pricing, and ratings.
    
    This tool can perform different types of analysis:
    - comparison: Compare multiple products side-by-side
    - features: Analyze product features and specifications
    - pricing: Analyze pricing patterns and value propositions
    - ratings: Analyze customer ratings and satisfaction
    
    Use this tool when you need to provide detailed product analysis or comparisons.
    """
    
    args_schema: Type[ProductAnalysisInput] = ProductAnalysisInput
    
    def __init__(self, **kwargs):
        """Initialize product analysis tool."""
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
    
    def _run(
        self,
        products: List[Dict[str, Any]],
        analysis_type: str = "comparison",
        focus_areas: Optional[List[str]] = None,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Execute product analysis synchronously."""
        
        try:
            if not products:
                return {
                    "error": "No products provided for analysis",
                    "analysis": {},
                    "metadata": {"tool": "product_analysis", "status": "error"}
                }
            
            # Route to appropriate analysis method
            if analysis_type == "comparison":
                return self._compare_products(products, focus_areas, include_summary)
            elif analysis_type == "features":
                return self._analyze_features(products, focus_areas, include_summary)
            elif analysis_type == "pricing":
                return self._analyze_pricing(products, include_summary)
            elif analysis_type == "ratings":
                return self._analyze_ratings(products, include_summary)
            else:
                return {
                    "error": f"Unknown analysis type: {analysis_type}",
                    "analysis": {},
                    "metadata": {"tool": "product_analysis", "status": "error"}
                }
                
        except Exception as e:
            self.logger.error(f"Product analysis failed: {e}")
            return {
                "error": str(e),
                "analysis": {},
                "metadata": {"tool": "product_analysis", "status": "error"}
            }
    
    async def _arun(
        self,
        products: List[Dict[str, Any]],
        analysis_type: str = "comparison",
        focus_areas: Optional[List[str]] = None,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Execute product analysis asynchronously."""
        
        # For now, use synchronous implementation
        return self._run(products, analysis_type, focus_areas, include_summary)
    
    def _compare_products(
        self,
        products: List[Dict[str, Any]],
        focus_areas: Optional[List[str]],
        include_summary: bool
    ) -> Dict[str, Any]:
        """Compare multiple products side-by-side."""
        
        if len(products) < 2:
            return {
                "error": "At least 2 products required for comparison",
                "analysis": {},
                "metadata": {"tool": "product_analysis", "status": "error"}
            }
        
        comparison = {
            "product_count": len(products),
            "products": [],
            "comparison_matrix": {},
            "recommendations": []
        }
        
        # Extract product information
        for i, product in enumerate(products):
            metadata = product.get("metadata", {})
            
            product_info = {
                "index": i,
                "title": metadata.get("title", "Unknown Product"),
                "price": self._parse_price(metadata.get("price")),
                "rating": self._parse_rating(metadata.get("average_rating")),
                "rating_count": metadata.get("rating_number", 0),
                "store": metadata.get("store", "Unknown"),
                "features": self._extract_features(product.get("content", ""))
            }
            
            comparison["products"].append(product_info)
        
        # Create comparison matrix
        comparison["comparison_matrix"] = self._create_comparison_matrix(
            comparison["products"], focus_areas
        )
        
        # Generate recommendations
        if include_summary:
            comparison["recommendations"] = self._generate_recommendations(
                comparison["products"]
            )
            comparison["summary"] = self._generate_comparison_summary(comparison)
        
        return {
            "analysis": comparison,
            "metadata": {
                "tool": "product_analysis",
                "status": "success",
                "analysis_type": "comparison",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _analyze_features(
        self,
        products: List[Dict[str, Any]],
        focus_areas: Optional[List[str]],
        include_summary: bool
    ) -> Dict[str, Any]:
        """Analyze product features and specifications."""
        
        analysis = {
            "product_count": len(products),
            "feature_analysis": [],
            "common_features": [],
            "unique_features": {}
        }
        
        all_features = set()
        product_features = {}
        
        # Extract features from each product
        for i, product in enumerate(products):
            metadata = product.get("metadata", {})
            content = product.get("content", "")
            
            features = self._extract_features(content)
            product_features[i] = features
            all_features.update(features.keys())
            
            analysis["feature_analysis"].append({
                "product_index": i,
                "title": metadata.get("title", "Unknown Product"),
                "features": features,
                "feature_count": len(features)
            })
        
        # Find common and unique features
        if len(products) > 1:
            common_features = set(product_features[0].keys())
            for features in product_features.values():
                common_features &= set(features.keys())
            
            analysis["common_features"] = list(common_features)
            
            # Find unique features for each product
            for i, features in product_features.items():
                unique = set(features.keys()) - common_features
                if unique:
                    product_title = analysis["feature_analysis"][i]["title"]
                    analysis["unique_features"][product_title] = list(unique)
        
        if include_summary:
            analysis["summary"] = self._generate_feature_summary(analysis)
        
        return {
            "analysis": analysis,
            "metadata": {
                "tool": "product_analysis",
                "status": "success",
                "analysis_type": "features",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _analyze_pricing(
        self,
        products: List[Dict[str, Any]],
        include_summary: bool
    ) -> Dict[str, Any]:
        """Analyze pricing patterns and value propositions."""
        
        analysis = {
            "product_count": len(products),
            "pricing_data": [],
            "price_range": {},
            "value_analysis": []
        }
        
        prices = []
        
        # Extract pricing information
        for i, product in enumerate(products):
            metadata = product.get("metadata", {})
            
            price = self._parse_price(metadata.get("price"))
            rating = self._parse_rating(metadata.get("average_rating"))
            
            pricing_info = {
                "product_index": i,
                "title": metadata.get("title", "Unknown Product"),
                "price": price,
                "rating": rating,
                "value_score": self._calculate_value_score(price, rating)
            }
            
            analysis["pricing_data"].append(pricing_info)
            
            if price is not None:
                prices.append(price)
        
        # Calculate price statistics
        if prices:
            analysis["price_range"] = {
                "min": min(prices),
                "max": max(prices),
                "average": sum(prices) / len(prices),
                "median": sorted(prices)[len(prices) // 2]
            }
            
            # Categorize products by price
            avg_price = analysis["price_range"]["average"]
            
            for item in analysis["pricing_data"]:
                if item["price"] is not None:
                    if item["price"] < avg_price * 0.8:
                        item["price_category"] = "budget"
                    elif item["price"] > avg_price * 1.2:
                        item["price_category"] = "premium"
                    else:
                        item["price_category"] = "mid-range"
        
        # Generate value analysis
        analysis["value_analysis"] = sorted(
            analysis["pricing_data"],
            key=lambda x: x.get("value_score", 0),
            reverse=True
        )
        
        if include_summary:
            analysis["summary"] = self._generate_pricing_summary(analysis)
        
        return {
            "analysis": analysis,
            "metadata": {
                "tool": "product_analysis",
                "status": "success",
                "analysis_type": "pricing",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _analyze_ratings(
        self,
        products: List[Dict[str, Any]],
        include_summary: bool
    ) -> Dict[str, Any]:
        """Analyze customer ratings and satisfaction."""
        
        analysis = {
            "product_count": len(products),
            "rating_data": [],
            "rating_statistics": {},
            "satisfaction_levels": {}
        }
        
        ratings = []
        rating_counts = []
        
        # Extract rating information
        for i, product in enumerate(products):
            metadata = product.get("metadata", {})
            
            rating = self._parse_rating(metadata.get("average_rating"))
            rating_count = metadata.get("rating_number", 0)
            
            rating_info = {
                "product_index": i,
                "title": metadata.get("title", "Unknown Product"),
                "rating": rating,
                "rating_count": rating_count,
                "satisfaction_level": self._categorize_satisfaction(rating)
            }
            
            analysis["rating_data"].append(rating_info)
            
            if rating is not None:
                ratings.append(rating)
                rating_counts.append(rating_count)
        
        # Calculate rating statistics
        if ratings:
            analysis["rating_statistics"] = {
                "average_rating": sum(ratings) / len(ratings),
                "highest_rating": max(ratings),
                "lowest_rating": min(ratings),
                "total_reviews": sum(rating_counts),
                "products_above_4": len([r for r in ratings if r >= 4.0]),
                "products_below_3": len([r for r in ratings if r < 3.0])
            }
            
            # Satisfaction level distribution
            satisfaction_counts = {}
            for item in analysis["rating_data"]:
                level = item["satisfaction_level"]
                satisfaction_counts[level] = satisfaction_counts.get(level, 0) + 1
            
            analysis["satisfaction_levels"] = satisfaction_counts
        
        if include_summary:
            analysis["summary"] = self._generate_rating_summary(analysis)
        
        return {
            "analysis": analysis,
            "metadata": {
                "tool": "product_analysis",
                "status": "success",
                "analysis_type": "ratings",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    # Helper methods
    
    def _parse_price(self, price_str: Any) -> Optional[float]:
        """Parse price string to float."""
        if price_str is None:
            return None
        
        try:
            # Remove currency symbols and convert to float
            price_clean = str(price_str).replace("$", "").replace(",", "").strip()
            return float(price_clean)
        except (ValueError, AttributeError):
            return None
    
    def _parse_rating(self, rating_str: Any) -> Optional[float]:
        """Parse rating string to float."""
        if rating_str is None:
            return None
        
        try:
            return float(rating_str)
        except (ValueError, TypeError):
            return None
    
    def _extract_features(self, content: str) -> Dict[str, str]:
        """Extract features from product content."""
        
        features = {}
        
        # Simple feature extraction - can be enhanced with NLP
        feature_keywords = {
            "wireless": "connectivity",
            "bluetooth": "connectivity", 
            "wifi": "connectivity",
            "usb": "connectivity",
            "battery": "power",
            "rechargeable": "power",
            "waterproof": "durability",
            "portable": "design",
            "lightweight": "design",
            "compact": "design",
            "noise-canceling": "audio",
            "stereo": "audio",
            "hd": "display",
            "4k": "display",
            "led": "display"
        }
        
        content_lower = content.lower()
        
        for keyword, category in feature_keywords.items():
            if keyword in content_lower:
                features[keyword] = category
        
        return features
    
    def _create_comparison_matrix(
        self,
        products: List[Dict[str, Any]],
        focus_areas: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Create comparison matrix for products."""
        
        matrix = {}
        
        # Default comparison areas
        areas = focus_areas or ["price", "rating", "features"]
        
        for area in areas:
            matrix[area] = {}
            
            if area == "price":
                prices = [p.get("price") for p in products]
                matrix[area]["values"] = prices
                matrix[area]["best"] = min([p for p in prices if p is not None], default=None)
                matrix[area]["worst"] = max([p for p in prices if p is not None], default=None)
                
            elif area == "rating":
                ratings = [p.get("rating") for p in products]
                matrix[area]["values"] = ratings
                matrix[area]["best"] = max([r for r in ratings if r is not None], default=None)
                matrix[area]["worst"] = min([r for r in ratings if r is not None], default=None)
                
            elif area == "features":
                feature_counts = [len(p.get("features", {})) for p in products]
                matrix[area]["values"] = feature_counts
                matrix[area]["best"] = max(feature_counts)
                matrix[area]["worst"] = min(feature_counts)
        
        return matrix
    
    def _calculate_value_score(self, price: Optional[float], rating: Optional[float]) -> float:
        """Calculate value score based on price and rating."""
        
        if price is None or rating is None:
            return 0.0
        
        if price == 0:
            return rating * 2  # Free products get bonus
        
        # Value = rating / (price / 100) - higher rating and lower price = better value
        return rating / (price / 100)
    
    def _categorize_satisfaction(self, rating: Optional[float]) -> str:
        """Categorize satisfaction level based on rating."""
        
        if rating is None:
            return "unknown"
        elif rating >= 4.5:
            return "excellent"
        elif rating >= 4.0:
            return "very_good"
        elif rating >= 3.5:
            return "good"
        elif rating >= 3.0:
            return "fair"
        else:
            return "poor"
    
    def _generate_recommendations(self, products: List[Dict[str, Any]]) -> List[str]:
        """Generate product recommendations based on analysis."""
        
        recommendations = []
        
        # Find best value product
        best_value = max(products, key=lambda p: self._calculate_value_score(p.get("price"), p.get("rating")))
        recommendations.append(f"Best value: {best_value['title']}")
        
        # Find highest rated product
        highest_rated = max(products, key=lambda p: p.get("rating", 0))
        if highest_rated != best_value:
            recommendations.append(f"Highest rated: {highest_rated['title']}")
        
        # Find budget option
        budget_products = [p for p in products if p.get("price") is not None]
        if budget_products:
            cheapest = min(budget_products, key=lambda p: p["price"])
            if cheapest not in [best_value, highest_rated]:
                recommendations.append(f"Budget option: {cheapest['title']}")
        
        return recommendations
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> str:
        """Generate summary for product comparison."""
        
        product_count = comparison["product_count"]
        recommendations = comparison.get("recommendations", [])
        
        summary = f"Compared {product_count} products. "
        
        if recommendations:
            summary += "Key recommendations: " + "; ".join(recommendations)
        
        return summary
    
    def _generate_feature_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate summary for feature analysis."""
        
        product_count = analysis["product_count"]
        common_features = len(analysis.get("common_features", []))
        unique_features = len(analysis.get("unique_features", {}))
        
        return f"Analyzed features of {product_count} products. Found {common_features} common features and {unique_features} products with unique features."
    
    def _generate_pricing_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate summary for pricing analysis."""
        
        price_range = analysis.get("price_range", {})
        
        if not price_range:
            return "No pricing information available for analysis."
        
        min_price = price_range.get("min", 0)
        max_price = price_range.get("max", 0)
        avg_price = price_range.get("average", 0)
        
        return f"Price range: ${min_price:.2f} - ${max_price:.2f}, average: ${avg_price:.2f}"
    
    def _generate_rating_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate summary for rating analysis."""
        
        stats = analysis.get("rating_statistics", {})
        
        if not stats:
            return "No rating information available for analysis."
        
        avg_rating = stats.get("average_rating", 0)
        total_reviews = stats.get("total_reviews", 0)
        above_4 = stats.get("products_above_4", 0)
        
        return f"Average rating: {avg_rating:.1f}/5.0 based on {total_reviews} total reviews. {above_4} products rated 4+ stars."