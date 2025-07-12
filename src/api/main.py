"""
FastAPI server for RAG-powered Amazon Electronics Assistant.
Provides REST API endpoints for querying the assistant.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import time
import uuid
from pathlib import Path
import os
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import RAG components
from rag.enhanced_query_processor import EnhancedRAGQueryProcessor, SearchStrategy
from rag.enhanced_vector_db import EnhancedElectronicsVectorDB, setup_enhanced_vector_database
from rag.structured_outputs import StructuredRAGRequest, StructuredRAGResponse, ResponseType
from rag.structured_generator import StructuredResponseGenerator

# Import configuration
try:
    from chatbot_ui.core.config import config
except ImportError:
    config = None

# API models
class QueryRequest(BaseModel):
    """Basic query request."""
    query: str = Field(..., description="User query", min_length=1)
    max_products: int = Field(5, description="Maximum products to return", ge=1, le=20)
    max_reviews: int = Field(3, description="Maximum reviews to analyze", ge=0, le=10)
    search_strategy: Optional[str] = Field(None, description="Search strategy (semantic_only, keyword_only, hybrid, adaptive)")


class QueryResponse(BaseModel):
    """Basic query response."""
    query: str
    response: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    database_status: str
    search_capabilities: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    timestamp: str
    trace_id: Optional[str] = None


# Global variables for RAG components
rag_processor: Optional[EnhancedRAGQueryProcessor] = None
structured_generator: Optional[StructuredResponseGenerator] = None
vector_db: Optional[EnhancedElectronicsVectorDB] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global rag_processor, structured_generator, vector_db
    
    logger.info("Starting Amazon Electronics Assistant API...")
    
    try:
        # Initialize vector database
        jsonl_path = find_jsonl_path()
        if jsonl_path:
            vector_db = setup_enhanced_vector_database(
                jsonl_path,
                enable_keyword_search=True,
                enable_reranking=True
            )
            logger.info("Vector database initialized successfully")
        else:
            logger.warning("JSONL file not found, using mock database")
            vector_db = None
        
        # Initialize structured generator
        provider = "openai"  # Default provider
        api_key = None
        
        if config:
            api_key = config.OPENAI_API_KEY.get_secret_value()
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            structured_generator = StructuredResponseGenerator(
                provider=provider,
                api_key=api_key
            )
            logger.info("Structured generator initialized successfully")
        else:
            logger.warning("API key not found, structured responses disabled")
            structured_generator = None
        
        # Initialize RAG processor
        rag_processor = EnhancedRAGQueryProcessor(
            vector_db=vector_db,
            structured_generator=structured_generator
        )
        
        logger.info("RAG processor initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Amazon Electronics Assistant API...")
    
    if vector_db:
        try:
            vector_db.close()
            logger.info("Vector database closed")
        except Exception as e:
            logger.error(f"Error closing vector database: {e}")


# FastAPI app
app = FastAPI(
    title="Amazon Electronics Assistant API",
    description="REST API for RAG-powered Amazon Electronics product assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def find_jsonl_path() -> Optional[str]:
    """Find JSONL file path."""
    possible_paths = [
        "data/processed/electronics_rag_documents.jsonl",
        "data/processed/electronics_top1000_products.jsonl",
        "../data/processed/electronics_rag_documents.jsonl",
        "../data/processed/electronics_top1000_products.jsonl"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return None


def get_rag_processor() -> EnhancedRAGQueryProcessor:
    """Dependency to get RAG processor."""
    if not rag_processor:
        raise HTTPException(status_code=503, detail="RAG processor not available")
    return rag_processor


def get_structured_generator() -> StructuredResponseGenerator:
    """Dependency to get structured generator."""
    if not structured_generator:
        raise HTTPException(status_code=503, detail="Structured generator not available")
    return structured_generator


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        database_status = "available" if vector_db else "unavailable"
        
        search_capabilities = {}
        if rag_processor:
            search_capabilities = rag_processor.get_search_analytics().get("search_capabilities", {})
        
        return HealthResponse(
            status="healthy",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            version="1.0.0",
            database_status=database_status,
            search_capabilities=search_capabilities
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/query", response_model=QueryResponse)
async def query_basic(
    request: QueryRequest,
    processor: EnhancedRAGQueryProcessor = Depends(get_rag_processor)
):
    """Basic query endpoint."""
    try:
        start_time = time.time()
        
        # Determine search strategy
        search_strategy = None
        if request.search_strategy:
            try:
                search_strategy = SearchStrategy(request.search_strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid search strategy: {request.search_strategy}")
        
        # Process query
        result = processor.process_query_enhanced(
            query=request.query,
            max_products=request.max_products,
            max_reviews=request.max_reviews,
            search_strategy=search_strategy
        )
        
        if not result.get("success", True):
            raise HTTPException(status_code=500, detail=result.get("error", "Query processing failed"))
        
        # Generate response text from context
        response_text = generate_response_text(request.query, result["context"])
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            context=result["context"],
            metadata={
                "search_strategy": result["context"].get("search_strategy", "unknown"),
                "reranking_applied": result["context"].get("reranking_applied", False),
                "total_results": result["context"].get("total_results", 0),
                "trace_id": result.get("performance", {}).get("trace_id")
            },
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/structured", response_model=StructuredRAGResponse)
async def query_structured(
    request: StructuredRAGRequest,
    processor: EnhancedRAGQueryProcessor = Depends(get_rag_processor)
):
    """Structured query endpoint."""
    try:
        if not processor.structured_generator:
            raise HTTPException(status_code=503, detail="Structured responses not available")
        
        # Process query with structured output
        result = processor.process_query_structured(request)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/analytics")
async def get_search_analytics(
    processor: EnhancedRAGQueryProcessor = Depends(get_rag_processor)
):
    """Get search analytics and capabilities."""
    try:
        analytics = processor.get_search_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/response-types")
async def get_response_types():
    """Get supported response types."""
    try:
        return {
            "response_types": [rt.value for rt in ResponseType],
            "search_strategies": [ss.value for ss in SearchStrategy],
            "description": "Supported response types and search strategies"
        }
    except Exception as e:
        logger.error(f"Response types retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics."""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        stats = vector_db.get_collection_stats()
        return {
            "database_stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_response_text(query: str, context: Dict[str, Any]) -> str:
    """Generate response text from context (fallback for basic queries)."""
    
    products = context.get("products", [])
    reviews = context.get("reviews", [])
    
    if not products and not reviews:
        return "I couldn't find any specific products or reviews matching your query. Please try a different search term or be more specific about what you're looking for."
    
    response_parts = []
    
    if products:
        response_parts.append(f"I found {len(products)} relevant products:")
        for i, product in enumerate(products[:3], 1):
            metadata = product.get("metadata", {})
            title = metadata.get("title", "Unknown Product")
            price = metadata.get("price", "N/A")
            rating = metadata.get("average_rating", "N/A")
            
            response_parts.append(f"{i}. {title}")
            response_parts.append(f"   Price: ${price}, Rating: {rating}/5")
    
    if reviews:
        response_parts.append(f"\nBased on {len(reviews)} review summaries:")
        for review in reviews[:2]:
            content = review.get("content", "")
            if content:
                response_parts.append(f"• {content[:150]}...")
    
    return "\n".join(response_parts)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            trace_id=str(uuid.uuid4())
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 