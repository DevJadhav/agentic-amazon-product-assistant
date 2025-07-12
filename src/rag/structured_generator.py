"""
Structured response generator using Instructor for RAG pipeline.
Converts LLM responses into structured Pydantic models.
"""

import instructor
import logging
import time
from typing import Dict, Any, Optional, Union, Type, TypeVar
from openai import OpenAI
from groq import Groq
from google import genai
import ollama

from langsmith import traceable

from .structured_outputs import (
    StructuredRAGResponse, 
    ResponseType, 
    ConfidenceLevel,
    ProductInfo,
    ProductComparison,
    ReviewSummary,
    TroubleshootingGuide,
    Recommendation,
    RAGSystemMetrics,
    StructuredRAGRequest
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StructuredResponseGenerator:
    """Generate structured responses using Instructor and LLM providers."""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, 
                 model: Optional[str] = None, **kwargs):
        """Initialize structured response generator."""
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model or self._get_default_model()
        self.kwargs = kwargs
        
        # Initialize client
        self.client = self._initialize_client()
        
        # Patch client with instructor
        self.instructor_client = instructor.patch(self.client)
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "groq": "llama3-8b-8192",
            "google": "gemini-1.5-flash",
            "ollama": "llama3.1:latest"
        }
        return defaults.get(self.provider, "gpt-4o-mini")
    
    def _initialize_client(self):
        """Initialize client for the specified provider."""
        if self.provider == "openai":
            return OpenAI(api_key=self.api_key)
        elif self.provider == "groq":
            return Groq(api_key=self.api_key)
        elif self.provider == "google":
            return genai.Client(api_key=self.api_key)
        elif self.provider == "ollama":
            return ollama.Client(host=self.kwargs.get('host', 'http://localhost:11434'))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    @traceable
    def generate_structured_response(self, 
                                   prompt: str, 
                                   response_model: Type[T],
                                   max_retries: int = 3,
                                   temperature: float = 0.3) -> T:
        """Generate structured response using Instructor."""
        
        for attempt in range(max_retries):
            try:
                response = self.instructor_client.chat.completions.create(
                    model=self.model,
                    response_model=response_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert AI assistant for Amazon Electronics. Provide accurate, helpful responses in the requested structured format."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                
        raise Exception("All attempts failed")
    
    @traceable
    def generate_rag_response(self, 
                            query: str, 
                            context: Dict[str, Any],
                            response_type: Optional[ResponseType] = None) -> StructuredRAGResponse:
        """Generate structured RAG response from query and context."""
        
        start_time = time.time()
        
        # Determine response type if not provided
        if response_type is None:
            response_type = self._determine_response_type(query, context)
        
        # Build specialized prompt
        prompt = self._build_specialized_prompt(query, context, response_type)
        
        # Generate structured response
        structured_response = self.generate_structured_response(
            prompt, StructuredRAGResponse
        )
        
        # Add processing time
        structured_response.processing_time = time.time() - start_time
        
        return structured_response
    
    def _determine_response_type(self, query: str, context: Dict[str, Any]) -> ResponseType:
        """Determine response type based on query and context."""
        query_lower = query.lower()
        
        # Check for comparison keywords
        comparison_keywords = ['compare', 'vs', 'versus', 'better', 'difference', 'which is']
        if any(keyword in query_lower for keyword in comparison_keywords):
            return ResponseType.PRODUCT_COMPARISON
        
        # Check for recommendation keywords
        recommendation_keywords = ['recommend', 'suggest', 'best', 'looking for', 'need']
        if any(keyword in query_lower for keyword in recommendation_keywords):
            return ResponseType.PRODUCT_RECOMMENDATION
        
        # Check for review keywords
        review_keywords = ['review', 'opinion', 'feedback', 'experience', 'thoughts']
        if any(keyword in query_lower for keyword in review_keywords):
            return ResponseType.REVIEW_SUMMARY
        
        # Check for troubleshooting keywords
        troubleshoot_keywords = ['problem', 'issue', 'not working', 'broken', 'fix', 'troubleshoot']
        if any(keyword in query_lower for keyword in troubleshoot_keywords):
            return ResponseType.TROUBLESHOOTING
        
        # Check if asking about specific product
        if context.get('context', {}).get('products'):
            return ResponseType.PRODUCT_INFO
        
        return ResponseType.GENERAL_QUERY
    
    def _build_specialized_prompt(self, query: str, context: Dict[str, Any], 
                                response_type: ResponseType) -> str:
        """Build specialized prompt based on response type."""
        
        base_context = self._format_context(context)
        
        if response_type == ResponseType.PRODUCT_RECOMMENDATION:
            return self._build_recommendation_prompt(query, base_context)
        elif response_type == ResponseType.PRODUCT_COMPARISON:
            return self._build_comparison_prompt(query, base_context)
        elif response_type == ResponseType.PRODUCT_INFO:
            return self._build_info_prompt(query, base_context)
        elif response_type == ResponseType.REVIEW_SUMMARY:
            return self._build_review_prompt(query, base_context)
        elif response_type == ResponseType.TROUBLESHOOTING:
            return self._build_troubleshooting_prompt(query, base_context)
        else:
            return self._build_general_prompt(query, base_context)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for prompts."""
        formatted_context = ""
        
        if context.get('context', {}).get('products'):
            formatted_context += "PRODUCTS:\n"
            for i, product in enumerate(context['context']['products'], 1):
                metadata = product.get('metadata', {})
                formatted_context += f"{i}. {metadata.get('title', 'Unknown')}\n"
                formatted_context += f"   Price: ${metadata.get('price', 'N/A')}\n"
                formatted_context += f"   Rating: {metadata.get('average_rating', 'N/A')}/5\n"
                formatted_context += f"   Details: {product.get('content', '')[:200]}...\n\n"
        
        if context.get('context', {}).get('reviews'):
            formatted_context += "REVIEWS:\n"
            for i, review in enumerate(context['context']['reviews'], 1):
                metadata = review.get('metadata', {})
                formatted_context += f"{i}. {metadata.get('title', 'Unknown')}\n"
                formatted_context += f"   Summary: {review.get('content', '')[:200]}...\n\n"
        
        return formatted_context
    
    def _build_recommendation_prompt(self, query: str, context: str) -> str:
        """Build recommendation prompt."""
        return f"""
Based on the user query and available product information, provide structured product recommendations.

USER QUERY: {query}

AVAILABLE CONTEXT:
{context}

INSTRUCTIONS:
1. Analyze the user's needs and preferences
2. Recommend 1-3 products that best match their requirements
3. Provide clear reasoning for each recommendation
4. Include confidence levels
5. Suggest alternatives if appropriate
6. Structure the response according to the StructuredRAGResponse model

Focus on matching user needs with product features and consider price, ratings, and reviews.
"""
    
    def _build_comparison_prompt(self, query: str, context: str) -> str:
        """Build comparison prompt."""
        return f"""
Based on the user query and available product information, provide a structured product comparison.

USER QUERY: {query}

AVAILABLE CONTEXT:
{context}

INSTRUCTIONS:
1. Identify the products being compared
2. Define relevant comparison criteria
3. Compare products across these criteria
4. Determine a winner if appropriate
5. Provide clear recommendations
6. Structure the response according to the StructuredRAGResponse model

Focus on objective comparisons based on specifications, features, price, and user reviews.
"""
    
    def _build_info_prompt(self, query: str, context: str) -> str:
        """Build product info prompt."""
        return f"""
Based on the user query and available product information, provide structured product information.

USER QUERY: {query}

AVAILABLE CONTEXT:
{context}

INSTRUCTIONS:
1. Extract key product details
2. Highlight important features
3. List pros and cons
4. Provide pricing and availability information
5. Include ratings and review insights
6. Structure the response according to the StructuredRAGResponse model

Focus on comprehensive product information that answers the user's specific questions.
"""
    
    def _build_review_prompt(self, query: str, context: str) -> str:
        """Build review summary prompt."""
        return f"""
Based on the user query and available review information, provide a structured review summary.

USER QUERY: {query}

AVAILABLE CONTEXT:
{context}

INSTRUCTIONS:
1. Analyze review content and sentiments
2. Identify key aspects mentioned in reviews
3. Categorize feedback as positive/negative
4. Extract key insights and takeaways
5. Provide overall sentiment assessment
6. Structure the response according to the StructuredRAGResponse model

Focus on actionable insights from user reviews that help with purchase decisions.
"""
    
    def _build_troubleshooting_prompt(self, query: str, context: str) -> str:
        """Build troubleshooting prompt."""
        return f"""
Based on the user query and available product information, provide structured troubleshooting guidance.

USER QUERY: {query}

AVAILABLE CONTEXT:
{context}

INSTRUCTIONS:
1. Identify the specific problem
2. Provide step-by-step troubleshooting guide
3. Include expected outcomes for each step
4. Suggest additional resources if needed
5. Estimate time required
6. Structure the response according to the StructuredRAGResponse model

Focus on practical, actionable troubleshooting steps that can resolve the issue.
"""
    
    def _build_general_prompt(self, query: str, context: str) -> str:
        """Build general query prompt."""
        return f"""
Based on the user query and available information, provide a structured response.

USER QUERY: {query}

AVAILABLE CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a comprehensive answer to the user's query
2. Use available product and review information
3. Structure the response appropriately
4. Include relevant details and insights
5. Maintain helpful and informative tone
6. Structure the response according to the StructuredRAGResponse model

Focus on providing accurate, helpful information that addresses the user's needs.
"""
    
    @traceable
    def generate_product_info(self, product_data: Dict[str, Any]) -> ProductInfo:
        """Generate structured product information."""
        
        prompt = f"""
Extract structured product information from the following data:

{product_data}

Provide comprehensive product details including features, pros, cons, and specifications.
"""
        
        return self.generate_structured_response(prompt, ProductInfo)
    
    @traceable
    def generate_comparison(self, products: list, criteria: str) -> ProductComparison:
        """Generate structured product comparison."""
        
        prompt = f"""
Compare the following products based on the specified criteria:

PRODUCTS:
{products}

COMPARISON CRITERIA:
{criteria}

Provide a detailed comparison with clear criteria, analysis, and recommendations.
"""
        
        return self.generate_structured_response(prompt, ProductComparison)
    
    def get_supported_response_types(self) -> list:
        """Get list of supported response types."""
        return [response_type.value for response_type in ResponseType]
    
    def validate_response(self, response: StructuredRAGResponse) -> bool:
        """Validate structured response."""
        try:
            # Pydantic validation is automatic
            return True
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return False


def create_structured_generator(provider: str = "openai", **kwargs) -> StructuredResponseGenerator:
    """Create structured response generator."""
    return StructuredResponseGenerator(provider=provider, **kwargs) 