"""
Unit tests for LangGraph tools.
"""

import pytest
from unittest.mock import Mock, patch

from ..tools.vector_search_tool import VectorSearchTool, VectorSearchInput
from ..tools.product_analysis_tool import ProductAnalysisTool, ProductAnalysisInput


class TestVectorSearchTool:
    """Test vector search tool."""
    
    def test_tool_initialization(self):
        """Test tool initialization."""
        tool = VectorSearchTool()
        
        assert tool.name == "vector_search"
        assert "search" in tool.description.lower()
        assert tool.args_schema == VectorSearchInput
    
    def test_vector_search_input_validation(self):
        """Test input validation."""
        # Valid input
        valid_input = VectorSearchInput(
            query="test query",
            search_type="hybrid",
            max_products=5,
            max_reviews=3
        )
        
        assert valid_input.query == "test query"
        assert valid_input.search_type == "hybrid"
        assert valid_input.max_products == 5
        assert valid_input.max_reviews == 3
    
    def test_vector_search_input_defaults(self):
        """Test input defaults."""
        input_data = VectorSearchInput(query="test query")
        
        assert input_data.search_type == "hybrid"
        assert input_data.max_products == 5
        assert input_data.max_reviews == 3
        assert input_data.doc_type is None
        assert input_data.include_metadata is True
    
    def test_vector_search_input_validation_errors(self):
        """Test input validation errors."""
        # Invalid max_products (too high)
        with pytest.raises(ValueError):
            VectorSearchInput(query="test", max_products=25)
        
        # Invalid max_products (too low)
        with pytest.raises(ValueError):
            VectorSearchInput(query="test", max_products=0)
        
        # Invalid max_reviews (too high)
        with pytest.raises(ValueError):
            VectorSearchInput(query="test", max_reviews=15)
    
    @patch('langgraph_integration.tools.vector_search_tool.setup_enhanced_vector_database')
    def test_tool_with_mock_database(self, mock_setup):
        """Test tool with mocked database."""
        # Mock the database setup
        mock_db = Mock()
        mock_setup.return_value = mock_db
        
        tool = VectorSearchTool()
        
        # Test get_tool_info
        info = tool.get_tool_info()
        
        assert "name" in info
        assert "search_types" in info
        assert isinstance(info["search_types"], list)
    
    def test_tool_without_database(self):
        """Test tool behavior without database."""
        tool = VectorSearchTool()
        
        # Should handle missing database gracefully
        result = tool._run(query="test query")
        
        assert "error" in result or "results" in result
    
    def test_get_tool_info(self):
        """Test tool info retrieval."""
        tool = VectorSearchTool()
        info = tool.get_tool_info()
        
        assert info["name"] == "vector_search"
        assert "search_types" in info
        assert "max_products" in info
        assert "max_reviews" in info
        assert "supports_async" in info


class TestProductAnalysisTool:
    """Test product analysis tool."""
    
    def test_tool_initialization(self):
        """Test tool initialization."""
        tool = ProductAnalysisTool()
        
        assert tool.name == "product_analysis"
        assert "analyze" in tool.description.lower()
        assert tool.args_schema == ProductAnalysisInput
    
    def test_product_analysis_input_validation(self):
        """Test input validation."""
        products = [
            {"id": "1", "metadata": {"title": "Product 1", "price": "99.99"}}
        ]
        
        valid_input = ProductAnalysisInput(
            products=products,
            analysis_type="comparison",
            focus_areas=["price", "rating"],
            include_summary=True
        )
        
        assert len(valid_input.products) == 1
        assert valid_input.analysis_type == "comparison"
        assert valid_input.focus_areas == ["price", "rating"]
        assert valid_input.include_summary is True
    
    def test_product_analysis_input_defaults(self):
        """Test input defaults."""
        products = [{"id": "1"}]
        input_data = ProductAnalysisInput(products=products)
        
        assert input_data.analysis_type == "comparison"
        assert input_data.focus_areas is None
        assert input_data.include_summary is True
    
    def test_comparison_analysis(self):
        """Test product comparison analysis."""
        tool = ProductAnalysisTool()
        
        products = [
            {
                "id": "1",
                "content": "Wireless headphones with noise canceling",
                "metadata": {
                    "title": "Sony WH-1000XM4",
                    "price": "299.99",
                    "average_rating": "4.5",
                    "rating_number": 1250
                }
            },
            {
                "id": "2",
                "content": "Bluetooth headphones with long battery life",
                "metadata": {
                    "title": "Bose QuietComfort 35",
                    "price": "249.99",
                    "average_rating": "4.3",
                    "rating_number": 980
                }
            }
        ]
        
        result = tool._run(
            products=products,
            analysis_type="comparison",
            include_summary=True
        )
        
        assert "analysis" in result
        assert "metadata" in result
        assert result["metadata"]["status"] == "success"
        
        analysis = result["analysis"]
        assert analysis["product_count"] == 2
        assert "products" in analysis
        assert "comparison_matrix" in analysis
        assert "recommendations" in analysis
    
    def test_pricing_analysis(self):
        """Test pricing analysis."""
        tool = ProductAnalysisTool()
        
        products = [
            {
                "id": "1",
                "metadata": {"title": "Product 1", "price": "99.99", "average_rating": "4.5"}
            },
            {
                "id": "2", 
                "metadata": {"title": "Product 2", "price": "149.99", "average_rating": "4.2"}
            }
        ]
        
        result = tool._run(
            products=products,
            analysis_type="pricing",
            include_summary=True
        )
        
        assert "analysis" in result
        analysis = result["analysis"]
        assert "pricing_data" in analysis
        assert "price_range" in analysis
        assert "value_analysis" in analysis
        
        # Check price range calculation
        price_range = analysis["price_range"]
        assert price_range["min"] == 99.99
        assert price_range["max"] == 149.99
    
    def test_features_analysis(self):
        """Test features analysis."""
        tool = ProductAnalysisTool()
        
        products = [
            {
                "id": "1",
                "content": "Wireless bluetooth headphones with noise-canceling",
                "metadata": {"title": "Product 1"}
            },
            {
                "id": "2",
                "content": "Portable wireless speaker with waterproof design",
                "metadata": {"title": "Product 2"}
            }
        ]
        
        result = tool._run(
            products=products,
            analysis_type="features",
            include_summary=True
        )
        
        assert "analysis" in result
        analysis = result["analysis"]
        assert "feature_analysis" in analysis
        assert "common_features" in analysis
        assert "unique_features" in analysis
    
    def test_ratings_analysis(self):
        """Test ratings analysis."""
        tool = ProductAnalysisTool()
        
        products = [
            {
                "id": "1",
                "metadata": {
                    "title": "Product 1",
                    "average_rating": "4.5",
                    "rating_number": 1000
                }
            },
            {
                "id": "2",
                "metadata": {
                    "title": "Product 2", 
                    "average_rating": "4.2",
                    "rating_number": 500
                }
            }
        ]
        
        result = tool._run(
            products=products,
            analysis_type="ratings",
            include_summary=True
        )
        
        assert "analysis" in result
        analysis = result["analysis"]
        assert "rating_data" in analysis
        assert "rating_statistics" in analysis
        assert "satisfaction_levels" in analysis
        
        # Check statistics
        stats = analysis["rating_statistics"]
        assert stats["total_reviews"] == 1500
        assert stats["products_above_4"] == 2
    
    def test_empty_products_error(self):
        """Test error handling for empty products list."""
        tool = ProductAnalysisTool()
        
        result = tool._run(products=[], analysis_type="comparison")
        
        assert "error" in result
        assert result["metadata"]["status"] == "error"
    
    def test_invalid_analysis_type(self):
        """Test error handling for invalid analysis type."""
        tool = ProductAnalysisTool()
        
        products = [{"id": "1", "metadata": {"title": "Test"}}]
        result = tool._run(products=products, analysis_type="invalid_type")
        
        assert "error" in result
        assert "Unknown analysis type" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__])