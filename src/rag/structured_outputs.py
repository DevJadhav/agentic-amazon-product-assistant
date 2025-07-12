"""
Structured output models for RAG pipeline using Pydantic.
Defines schemas for structured responses from the AI assistant.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence level for AI responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResponseType(str, Enum):
    """Type of response from the AI assistant."""
    PRODUCT_RECOMMENDATION = "product_recommendation"
    PRODUCT_COMPARISON = "product_comparison"
    PRODUCT_INFO = "product_info"
    REVIEW_SUMMARY = "review_summary"
    TROUBLESHOOTING = "troubleshooting"
    GENERAL_QUERY = "general_query"


class ProductInfo(BaseModel):
    """Structured product information."""
    title: str = Field(..., description="Product title")
    asin: Optional[str] = Field(None, description="Amazon ASIN")
    price: Optional[float] = Field(None, description="Product price", ge=0)
    currency: str = Field("USD", description="Price currency")
    rating: Optional[float] = Field(None, description="Average rating", ge=0, le=5)
    rating_count: Optional[int] = Field(None, description="Number of ratings", ge=0)
    availability: Optional[str] = Field(None, description="Product availability")
    brand: Optional[str] = Field(None, description="Product brand")
    category: Optional[str] = Field(None, description="Product category")
    key_features: List[str] = Field(default_factory=list, description="Key product features")
    pros: List[str] = Field(default_factory=list, description="Product advantages")
    cons: List[str] = Field(default_factory=list, description="Product disadvantages")
    search_score: Optional[float] = Field(None, description="Search relevance score")
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v is not None and v < 0:
            raise ValueError('Price cannot be negative')
        return v
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if v is not None and (v < 0 or v > 5):
            raise ValueError('Rating must be between 0 and 5')
        return v


class ComparisonCriteria(BaseModel):
    """Criteria for product comparison."""
    name: str = Field(..., description="Criteria name")
    description: str = Field(..., description="Criteria description")
    weight: float = Field(1.0, description="Criteria weight in comparison", ge=0, le=1)


class ProductComparison(BaseModel):
    """Structured product comparison."""
    products: List[ProductInfo] = Field(..., description="Products being compared", min_length=2)
    criteria: List[ComparisonCriteria] = Field(..., description="Comparison criteria")
    winner: Optional[str] = Field(None, description="Overall winner product title")
    summary: str = Field(..., description="Comparison summary")
    recommendation: str = Field(..., description="Final recommendation")


class ReviewInsight(BaseModel):
    """Structured review insight."""
    aspect: str = Field(..., description="Review aspect (e.g., 'battery life', 'build quality')")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Sentiment of the aspect")
    frequency: int = Field(..., description="How often this aspect is mentioned", ge=1)
    example_quotes: List[str] = Field(default_factory=list, description="Example quotes from reviews")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.MEDIUM, description="Confidence in this insight")


class ReviewSummary(BaseModel):
    """Structured review summary."""
    product_title: str = Field(..., description="Product title")
    total_reviews: int = Field(..., description="Total number of reviews analyzed", ge=0)
    average_rating: float = Field(..., description="Average rating", ge=0, le=5)
    positive_insights: List[ReviewInsight] = Field(default_factory=list, description="Positive review insights")
    negative_insights: List[ReviewInsight] = Field(default_factory=list, description="Negative review insights")
    overall_sentiment: Literal["positive", "negative", "mixed"] = Field(..., description="Overall sentiment")
    key_takeaways: List[str] = Field(default_factory=list, description="Key takeaways from reviews")


class TroubleshootingStep(BaseModel):
    """Structured troubleshooting step."""
    step_number: int = Field(..., description="Step number", ge=1)
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Detailed step description")
    expected_outcome: str = Field(..., description="Expected outcome after this step")
    difficulty: Literal["easy", "medium", "hard"] = Field("medium", description="Difficulty level")


class TroubleshootingGuide(BaseModel):
    """Structured troubleshooting guide."""
    problem: str = Field(..., description="Problem description")
    product_context: Optional[str] = Field(None, description="Product context if applicable")
    steps: List[TroubleshootingStep] = Field(..., description="Troubleshooting steps", min_length=1)
    additional_resources: List[str] = Field(default_factory=list, description="Additional help resources")
    estimated_time: Optional[str] = Field(None, description="Estimated time to complete")


class Recommendation(BaseModel):
    """Structured recommendation."""
    product: ProductInfo = Field(..., description="Recommended product")
    reasoning: str = Field(..., description="Reasoning for recommendation")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    use_case_fit: str = Field(..., description="How well it fits the use case")
    alternatives: List[ProductInfo] = Field(default_factory=list, description="Alternative products")


class StructuredRAGResponse(BaseModel):
    """Main structured response from RAG system."""
    response_type: ResponseType = Field(..., description="Type of response")
    query: str = Field(..., description="Original user query")
    summary: str = Field(..., description="Brief response summary")
    detailed_response: str = Field(..., description="Detailed response text")
    confidence: ConfidenceLevel = Field(..., description="Overall confidence level")
    
    # Type-specific structured data
    product_recommendations: List[Recommendation] = Field(default_factory=list, description="Product recommendations")
    product_comparison: Optional[ProductComparison] = Field(None, description="Product comparison")
    product_info: Optional[ProductInfo] = Field(None, description="Single product information")
    review_summary: Optional[ReviewSummary] = Field(None, description="Review summary")
    troubleshooting_guide: Optional[TroubleshootingGuide] = Field(None, description="Troubleshooting guide")
    
    # Metadata
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    sources_used: List[str] = Field(default_factory=list, description="Sources used for response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    @model_validator(mode='after')
    def validate_response_consistency(self):
        """Validate that response content matches response type."""
        response_type = self.response_type
        
        if response_type == ResponseType.PRODUCT_RECOMMENDATION:
            if not self.product_recommendations:
                raise ValueError('Product recommendations required for PRODUCT_RECOMMENDATION type')
        elif response_type == ResponseType.PRODUCT_COMPARISON:
            if not self.product_comparison:
                raise ValueError('Product comparison required for PRODUCT_COMPARISON type')
        elif response_type == ResponseType.PRODUCT_INFO:
            if not self.product_info:
                raise ValueError('Product info required for PRODUCT_INFO type')
        elif response_type == ResponseType.REVIEW_SUMMARY:
            if not self.review_summary:
                raise ValueError('Review summary required for REVIEW_SUMMARY type')
        elif response_type == ResponseType.TROUBLESHOOTING:
            if not self.troubleshooting_guide:
                raise ValueError('Troubleshooting guide required for TROUBLESHOOTING type')
        
        return self


class RAGSystemMetrics(BaseModel):
    """Metrics for RAG system performance."""
    query_processing_time: float = Field(..., description="Time to process query", ge=0)
    retrieval_time: float = Field(..., description="Time for document retrieval", ge=0)
    generation_time: float = Field(..., description="Time for response generation", ge=0)
    total_documents_searched: int = Field(..., description="Total documents searched", ge=0)
    relevant_documents_found: int = Field(..., description="Relevant documents found", ge=0)
    search_strategy_used: str = Field(..., description="Search strategy used")
    reranking_applied: bool = Field(False, description="Whether reranking was applied")
    
    @property
    def total_time(self) -> float:
        """Calculate total processing time."""
        return self.query_processing_time + self.retrieval_time + self.generation_time


class StructuredRAGRequest(BaseModel):
    """Structured request to RAG system."""
    query: str = Field(..., description="User query", min_length=1)
    max_products: int = Field(5, description="Maximum products to return", ge=1, le=20)
    max_reviews: int = Field(3, description="Maximum reviews to analyze", ge=0, le=10)
    preferred_response_type: Optional[ResponseType] = Field(None, description="Preferred response type")
    search_strategy: Optional[str] = Field(None, description="Preferred search strategy")
    include_alternatives: bool = Field(True, description="Include alternative products")
    confidence_threshold: float = Field(0.5, description="Minimum confidence threshold", ge=0, le=1)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


# Export all models
__all__ = [
    'ConfidenceLevel',
    'ResponseType', 
    'ProductInfo',
    'ComparisonCriteria',
    'ProductComparison',
    'ReviewInsight',
    'ReviewSummary',
    'TroubleshootingStep',
    'TroubleshootingGuide',
    'Recommendation',
    'StructuredRAGResponse',
    'RAGSystemMetrics',
    'StructuredRAGRequest'
] 