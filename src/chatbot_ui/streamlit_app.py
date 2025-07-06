import streamlit as st
from langsmith import traceable
from openai import OpenAI
from groq import Groq
from google import genai
import ollama
import sys
import time
import logging
from pathlib import Path
from typing import List

# Add the parent directory to sys.path to import from rag module
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from chatbot_ui.core.config import config
from rag.query_processor import create_rag_processor

# Configure logger
logger = logging.getLogger(__name__)

# Import enhanced tracing utilities
try:
    from tracing.trace_utils import (
        create_enhanced_trace_context, get_current_trace_context,
        update_trace_context, business_analyzer
    )
    from tracing.business_intelligence import (
        track_business_interaction, get_business_session_summary,
        business_tracker
    )
except ImportError:
    # Fallback for cases where tracing utils are not available
    create_enhanced_trace_context = lambda **kwargs: None
    get_current_trace_context = lambda: None
    update_trace_context = lambda **kwargs: None
    business_analyzer = None
    track_business_interaction = lambda *args, **kwargs: {}
    get_business_session_summary = lambda *args: {}
    business_tracker = None

def get_llm_client():
    """Get or create LLM client based on current provider."""
    provider = st.session_state.get('provider', 'OpenAI')
    if provider == "OpenAI":
        return OpenAI(api_key=config.OPENAI_API_KEY.get_secret_value())
    elif provider == "Groq":
        return Groq(api_key=config.GROQ_API_KEY.get_secret_value())
    elif provider == "Ollama":
        return ollama.Client(host=config.OLLAMA_BASE_URL)
    else:
        return genai.Client(api_key=config.GOOGLE_API_KEY.get_secret_value())

def get_query_suggestions(partial_query: str, rag_processor) -> List[str]:
    """Generate query suggestions based on partial input and product database."""
    if not rag_processor or not partial_query or len(partial_query) < 3:
        return []
    
    # Common product terms that might trigger suggestions
    product_terms = {
        "iphone": ["iPhone charger cables", "iPhone accessories", "iPhone cases"],
        "cable": ["USB cables", "Ethernet cables", "charging cables", "Lightning cables"],
        "headphone": ["wireless headphones", "noise-canceling headphones", "gaming headphones"],
        "tablet": ["budget tablets", "iPad alternatives", "Android tablets"],
        "laptop": ["laptop backpacks", "laptop accessories", "budget laptops"],
        "router": ["wireless routers", "gaming routers", "mesh routers"],
        "charger": ["phone chargers", "wireless chargers", "fast chargers"],
        "speaker": ["Bluetooth speakers", "smart speakers", "portable speakers"],
        "keyboard": ["mechanical keyboards", "gaming keyboards", "wireless keyboards"],
        "mouse": ["gaming mice", "wireless mice", "ergonomic mice"]
    }
    
    suggestions = []
    partial_lower = partial_query.lower()
    
    for term, product_suggestions in product_terms.items():
        if term in partial_lower:
            for suggestion in product_suggestions:
                if suggestion.lower() not in partial_lower:
                    suggestions.append(f"What do people say about {suggestion}?")
                    suggestions.append(f"Compare {suggestion} with alternatives")
                    if len(suggestions) >= 6:  # Limit suggestions
                        break
            break
    
    return suggestions[:6]  # Maximum 6 suggestions

def display_enhanced_response(response: str, rag_context=None):
    """Display response with enhanced formatting and context cards."""
    
    # Display the main response
    st.markdown(response)
    
    # If RAG context is available, show additional context
    if rag_context and hasattr(st.session_state, 'last_rag_result'):
        rag_result = st.session_state.last_rag_result
        context = rag_result["context"]
        
        if context["num_products"] > 0 or context["num_reviews"] > 0:
            with st.expander("🔍 Retrieved Context", expanded=False):
                
                # Create tabs for products and reviews
                if context["num_products"] > 0 and context["num_reviews"] > 0:
                    prod_tab, review_tab = st.tabs(["📦 Products", "⭐ Reviews"])
                elif context["num_products"] > 0:
                    prod_tab = st.container()
                    review_tab = None
                elif context["num_reviews"] > 0:
                    review_tab = st.container()
                    prod_tab = None
                
                # Display product cards
                if context["num_products"] > 0 and prod_tab is not None:
                    with prod_tab:
                        st.write(f"**Found {context['num_products']} relevant products:**")
                        
                        # Create a simple product card layout
                        for i in range(min(context["num_products"], 3)):  # Show max 3 products
                            with st.container():
                                st.write(f"**Product {i+1}**")
                                # Note: In a real implementation, you'd extract this from the RAG context
                                # For now, we'll show a placeholder since the actual product data 
                                # would need to be passed through the context
                                st.caption("📊 Product information retrieved from database")
                                st.divider()
                
                # Display review summaries
                if context["num_reviews"] > 0 and review_tab is not None:
                    with review_tab:
                        st.write(f"**Found {context['num_reviews']} relevant review summaries:**")
                        
                        # Create review summary cards
                        for i in range(min(context["num_reviews"], 3)):  # Show max 3 reviews
                            with st.container():
                                st.write(f"**Review Summary {i+1}**")
                                st.caption("💬 Customer feedback summary from database")
                                st.divider()
                
                # Show query analysis
                st.subheader("🎯 Query Analysis")
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.write(f"**Query Type:** {context['query_type']}")
                    st.write(f"**Processing Time:** {rag_result['processing_time_ms']}ms")
                
                with analysis_col2:
                    if context.get('extracted_terms'):
                        st.write(f"**Extracted Terms:** {', '.join(context['extracted_terms'])}")
                    else:
                        st.write("**Extracted Terms:** None detected")

def initialize_rag_processor():
    """Initialize RAG processor with robust error handling."""
    try:
        processor = create_rag_processor()
        
        # Check if we have a vector database
        has_vector_db = processor.vector_db is not None if processor else False
        is_mock = getattr(processor.vector_db, 'is_mock', False) if has_vector_db else False
        
        return {
            "processor": processor,
            "status": "success",
            "has_vector_db": has_vector_db,
            "is_mock": is_mock,
            "message": "Mock database in use - limited functionality" if is_mock else "Full RAG system active"
        }
    except Exception as e:
        logger.error(f"Failed to initialize RAG processor: {e}")
        return {
            "processor": None,
            "status": "error", 
            "error": str(e),
            "has_vector_db": False,
            "is_mock": False,
            "message": "RAG system unavailable"
        }

# Initialize RAG processor with caching and single trace
@st.cache_resource
@traceable
def get_rag_processor():
    """Cached RAG processor initialization with tracing."""
    result = initialize_rag_processor()
    return result

def initialize_langsmith_tracing():
    """Initialize LangSmith tracing (cached wrapper handles tracing)."""
    if config.LANGSMITH_API_KEY:
        try:
            import os
            # Set LangSmith environment variables
            os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY.get_secret_value()
            os.environ["LANGSMITH_TRACING"] = "true"
            return {
                "status": "success",
                "message": "🔍 LangSmith tracing enabled"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ LangSmith initialization failed: {str(e)}",
                "fallback": "Continuing without tracing..."
            }
    else:
        return {
            "status": "disabled",
            "message": "ℹ️ LangSmith tracing disabled (no LANGSMITH_API_KEY)"
        }

@traceable
def apply_rag_enhancement(query: str, rag_processor, max_products: int, max_reviews: int, session_id: str = None, conversation_turn: int = 0):
    """Apply RAG enhancement to user query with comprehensive tracing."""
    start_time = time.time()
    
    # Create enhanced trace context
    trace_context = create_enhanced_trace_context(
        session_id=session_id or str(st.session_state.get('session_id', 'default')),
        conversation_turn=conversation_turn
    )
    
    try:
        # Build context with enhanced tracing
        context = rag_processor.build_context(
            query,
            max_products=max_products,
            max_reviews=max_reviews,
            trace_id=trace_context.trace_id if trace_context else None
        )
        
        rag_applied = bool(context.products or context.reviews)
        enhanced_prompt = None
        
        if rag_applied:
            enhanced_prompt = rag_processor.generate_rag_prompt(context)
        
        processing_time = time.time() - start_time
        
        # Enhanced result with comprehensive context
        result = {
            "status": "success",
            "rag_applied": rag_applied,
            "context": {
                "query_type": context.query_type,
                "num_products": len(context.products),
                "num_reviews": len(context.reviews),
                "extracted_terms": context.metadata.get("extracted_terms", []),
                "performance_metrics": context.metadata.get("performance_metrics", {})
            },
            "enhanced_prompt": enhanced_prompt,
            "processing_time_ms": round(processing_time * 1000, 2),
            "original_query": query,
            "trace_metadata": {
                "trace_id": trace_context.trace_id if trace_context else None,
                "session_id": trace_context.session_id if trace_context else session_id,
                "conversation_turn": conversation_turn
            }
        }
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "processing_time_ms": round(processing_time * 1000, 2),
            "original_query": query,
            "fallback": "Using original query",
            "trace_metadata": {
                "trace_id": trace_context.trace_id if trace_context else None,
                "session_id": trace_context.session_id if trace_context else session_id,
                "conversation_turn": conversation_turn
            }
        }

@traceable
def call_llm_provider(provider: str, model_name: str, messages: list, temperature: float, 
                     max_tokens: int, top_p: float, top_k: int, client):
    """Call specific LLM provider with comprehensive tracing."""
    start_time = time.time()
    
    # Prepare request metadata
    request_metadata = {
        "provider": provider,
        "model": model_name,
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k if provider == "Google" else None
        },
        "message_count": len(messages),
        "total_chars": sum(len(msg.get("content", "")) for msg in messages)
    }
    
    try:
        if provider == "Google":
            # Format messages for Google GenAI - filter out empty content and convert roles
            google_messages = []
            for message in messages:
                if message.get("content") and message["content"].strip():  # Only include non-empty messages
                    # Convert OpenAI role format to Google format
                    google_role = "user" if message["role"] == "user" else "model"
                    google_messages.append({
                        "role": google_role,
                        "parts": [{"text": message["content"]}]
                    })
            
            response = client.models.generate_content(
                model=model_name,
                contents=google_messages,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k
                }
            ).text
        elif provider == "Ollama":
            # Ollama API format
            response = client.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )["message"]["content"]
        else:
            # OpenAI and Groq support top_p but not top_k
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            response = completion.choices[0].message.content
        
        response_time = time.time() - start_time
        
        return {
            "status": "success",
            "response": response,
            "response_length": len(response),
            "response_time_ms": round(response_time * 1000, 2),
            "request_metadata": request_metadata
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "response_time_ms": round(response_time * 1000, 2),
            "request_metadata": request_metadata
        }

@traceable
def run_llm(client, messages):
    """Enhanced run_llm function with comprehensive business intelligence tracing."""
    overall_start_time = time.time()
    
    # Get configuration from session state
    provider = st.session_state.get('provider', 'OpenAI')
    model_name = st.session_state.get('model_name', 'gpt-4o-mini')
    temperature = st.session_state.get('temperature', 0.7)
    max_tokens = st.session_state.get('max_tokens', 500)
    top_p = st.session_state.get('top_p', 1.0)
    top_k = st.session_state.get('top_k', 40)
    use_rag = st.session_state.get('use_rag', False)
    
    # Session management for enhanced tracing
    session_id = st.session_state.get('session_id', str(time.time()))
    if 'session_id' not in st.session_state:
        st.session_state.session_id = session_id
    
    conversation_turn = len([m for m in messages if m['role'] == 'user'])
    
    # Initialize response structure
    trace_data = {
        "provider": provider,
        "model": model_name,
        "use_rag": use_rag,
        "message_count": len(messages),
        "session_id": session_id,
        "conversation_turn": conversation_turn
    }
    
    # Prepare messages for LLM
    llm_messages = messages.copy()
    rag_result = None
    business_intelligence = None
    
    # Apply RAG if enabled and user message exists
    if use_rag and st.session_state.rag_processor and len(messages) > 0:
        latest_message = messages[-1]
        if latest_message["role"] == "user":
            # Get RAG configuration
            max_products = st.session_state.get('max_products', 5)
            max_reviews = st.session_state.get('max_reviews', 3)
            
            # Apply RAG enhancement with enhanced tracing
            rag_result = apply_rag_enhancement(
                latest_message["content"],
                st.session_state.rag_processor,
                max_products,
                max_reviews,
                session_id,
                conversation_turn
            )
            
            trace_data["rag_result"] = rag_result
            
            # Update message if RAG enhancement was successful
            if rag_result["status"] == "success" and rag_result["rag_applied"]:
                llm_messages[-1] = {
                    "role": "user",
                    "content": rag_result["enhanced_prompt"]
                }
                
                # Store RAG context info for monitoring tab
                st.session_state.last_rag_result = rag_result
            
            elif rag_result["status"] == "error":
                # Store RAG error for monitoring tab
                st.session_state.last_rag_error = rag_result["error"]
    
    # Call LLM provider with tracing
    llm_result = call_llm_provider(
        provider, model_name, llm_messages, 
        temperature, max_tokens, top_p, top_k, client
    )
    
    total_time = time.time() - overall_start_time
    
    # Business Intelligence Tracking
    if llm_result["status"] == "success" and len(messages) > 0:
        latest_user_message = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        response = llm_result["response"]
        context = rag_result.get("context", {}) if rag_result else {}
        
        # Track business intelligence
        try:
            business_intelligence = track_business_interaction(
                query=latest_user_message,
                response=response,
                context=context,
                session_id=session_id,
                conversation_turn=conversation_turn
            )
            
            # Store business intelligence for monitoring
            st.session_state.last_business_intelligence = business_intelligence
            
        except Exception as e:
            st.session_state.last_business_intelligence = {"error": str(e)}
    
    # Combine all trace data
    final_trace = {
        **trace_data,
        "llm_result": llm_result,
        "business_intelligence": business_intelligence,
        "total_time_ms": round(total_time * 1000, 2),
        "success": llm_result["status"] == "success"
    }
    
    # Store enhanced performance metrics for monitoring tab
    if llm_result["status"] == "success":
        current_performance = {
            "timestamp": time.time(),
            "total_time_ms": final_trace['total_time_ms'],
            "rag_time_ms": rag_result['processing_time_ms'] if rag_result and rag_result["status"] == "success" else 0,
            "llm_time_ms": llm_result['response_time_ms'],
            "llm_provider": provider,
            "llm_model": model_name,
            "use_rag": use_rag,
            "message_length": len(llm_messages[-1]["content"]) if llm_messages else 0,
            "business_metrics": business_intelligence.get("business_metrics", {}) if business_intelligence else {}
        }
        
        # Store latest performance
        st.session_state.last_performance = current_performance
        
        # Initialize historical performance tracking if not exists
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
        
        # Add to historical tracking
        st.session_state.performance_history.append(current_performance)
        
        # Initialize provider/model statistics if not exists
        if 'provider_model_stats' not in st.session_state:
            st.session_state.provider_model_stats = {}
        
        # Track provider/model specific statistics
        provider_model_key = f"{provider}::{model_name}"
        if provider_model_key not in st.session_state.provider_model_stats:
            st.session_state.provider_model_stats[provider_model_key] = {
                "provider": provider,
                "model": model_name,
                "total_queries": 0,
                "total_time_ms": 0,
                "total_rag_time_ms": 0,
                "total_llm_time_ms": 0,
                "rag_queries": 0,
                "non_rag_queries": 0,
                "min_llm_time_ms": float('inf'),
                "max_llm_time_ms": 0,
                "recent_performances": []
            }
        
        # Update provider/model statistics
        stats = st.session_state.provider_model_stats[provider_model_key]
        stats["total_queries"] += 1
        stats["total_time_ms"] += current_performance["total_time_ms"]
        stats["total_rag_time_ms"] += current_performance["rag_time_ms"]
        stats["total_llm_time_ms"] += current_performance["llm_time_ms"]
        
        if use_rag:
            stats["rag_queries"] += 1
        else:
            stats["non_rag_queries"] += 1
            
        stats["min_llm_time_ms"] = min(stats["min_llm_time_ms"], current_performance["llm_time_ms"])
        stats["max_llm_time_ms"] = max(stats["max_llm_time_ms"], current_performance["llm_time_ms"])
        
        # Keep recent performances (last 10)
        stats["recent_performances"].append(current_performance)
        if len(stats["recent_performances"]) > 10:
            stats["recent_performances"] = stats["recent_performances"][-10:]
    
    # Return response or handle error
    if llm_result["status"] == "success":
        return llm_result["response"]
    else:
        st.error(f"LLM Error ({llm_result['error_type']}): {llm_result['error']}")
        return "I apologize, but I encountered an error processing your request. Please try again."

# Initialize LangSmith tracing with caching and single trace
@st.cache_resource  
@traceable
def get_langsmith_initialization():
    """Cached LangSmith initialization with tracing."""
    return initialize_langsmith_tracing()

# Initialize once per session using session state
if "initialization_complete" not in st.session_state:
    # Initialize LangSmith tracing
    langsmith_result = get_langsmith_initialization()
    st.session_state.langsmith_result = langsmith_result
    
    # Initialize RAG processor
    rag_init_result = get_rag_processor()
    st.session_state.rag_init_result = rag_init_result
    st.session_state.rag_processor = rag_init_result["processor"]
    
    st.session_state.initialization_complete = True
else:
    # Use cached results
    langsmith_result = st.session_state.langsmith_result
    rag_init_result = st.session_state.rag_init_result
    rag_processor = st.session_state.rag_processor

# Configure page styling for better UX and visual hierarchy
st.markdown("""
<style>
/* Enhanced Visual Hierarchy */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Title - Large and prominent */
h1 {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #2986cc !important;
    margin-bottom: 0.5rem !important;
    text-align: center;
}

/* Tab Headers - Medium size */
.stTabs [data-baseweb="tab-list"] button {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
}

/* Section Headers (h2) - Prominent but smaller than title */
h2 {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #2986cc !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
    border-bottom: 2px solid #4CAF50;
    padding-bottom: 0.5rem;
}

/* Subsection Headers (h3) - Clear hierarchy */
h3 {
    font-size: 1.3rem !important;
    font-weight: 500 !important;
    color: #2986cc !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.8rem !important;
}

/* Regular text - Optimized readability */
.stMarkdown p, .stMarkdown li, .stMarkdown div {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: #E2E8F0 !important;
}

/* Metric labels - Clear and consistent */
.metric-container label {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #CBD5E0 !important;
}

/* Metric values - Prominent display */
.metric-container [data-testid="metric-container"] > div:first-child {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
}

/* Table headers - Professional appearance */
.stDataFrame thead th {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    background-color: #2D3748 !important;
    color: #F7FAFC !important;
    border-bottom: 2px solid #4A5568 !important;
    padding: 0.75rem !important;
}

/* Table cells - Improved readability */
.stDataFrame tbody td {
    font-size: 0.85rem !important;
    padding: 0.5rem !important;
    color: #E2E8F0 !important;
    background-color: #1A202C !important;
}

/* Info boxes and alerts - Consistent sizing */
.stAlert, .stInfo, .stSuccess, .stWarning, .stError {
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
}

/* Chat messages - Optimized for conversation with proper contrast */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* User messages - Dark background with light text */
.stChatMessage[data-testid="user-message"] {
    background-color: #2E3B4E !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
    border: 1px solid #4A5568 !important;
}

.stChatMessage[data-testid="user-message"] .stMarkdown {
    color: #F7FAFC !important;
    font-weight: 500 !important;
}

/* Assistant messages - Dark background with light text and accent */
.stChatMessage[data-testid="assistant-message"] {
    background-color: #1A202C !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
    border-left: 4px solid #4CAF50 !important;
    border: 1px solid #2D3748 !important;
}

.stChatMessage[data-testid="assistant-message"] .stMarkdown {
    color: #E2E8F0 !important;
    font-weight: 400 !important;
}

/* General chat message content - Light text for dark backgrounds */
.stChatMessage .stMarkdown p,
.stChatMessage .stMarkdown div,
.stChatMessage .stMarkdown span,
.stChatMessage div[data-testid="stMarkdownContainer"] {
    color: #E2E8F0 !important;
    background-color: transparent !important;
}

/* Chat input styling */
.stChatInput input {
    font-size: 0.95rem !important;
    color: #F7FAFC !important;
    background-color: #2D3748 !important;
    border: 1px solid #4A5568 !important;
}

/* Message avatar styling */
.stChatMessage img {
    border-radius: 50% !important;
    border: 2px solid #E0E0E0 !important;
}

/* Force light text in all chat message elements */
div[data-testid="chatAvatarIcon-user"] ~ div,
div[data-testid="chatAvatarIcon-assistant"] ~ div {
    color: #E2E8F0 !important;
}

/* Override any inherited styling in chat messages with light text */
.stChatMessage * {
    color: #E2E8F0 !important;
}

/* User message specific light text */
.stChatMessage[data-testid="user-message"] * {
    color: #F7FAFC !important;
}

/* Specific targeting for markdown content in chat - light text */
.stChatMessage .element-container .stMarkdown,
.stChatMessage .element-container .stMarkdown *,
.stChatMessage [data-testid="stMarkdownContainer"],
.stChatMessage [data-testid="stMarkdownContainer"] * {
    color: #E2E8F0 !important;
    text-shadow: none !important;
}

/* User message markdown - even lighter */
.stChatMessage[data-testid="user-message"] .element-container .stMarkdown,
.stChatMessage[data-testid="user-message"] .element-container .stMarkdown *,
.stChatMessage[data-testid="user-message"] [data-testid="stMarkdownContainer"],
.stChatMessage[data-testid="user-message"] [data-testid="stMarkdownContainer"] * {
    color: #F7FAFC !important;
}

/* Ensure code blocks in chat are readable on dark background */
.stChatMessage code,
.stChatMessage pre {
    background-color: #4A5568 !important;
    color: #F7FAFC !important;
    border: 1px solid #718096 !important;
    padding: 0.25rem 0.5rem !important;
    border-radius: 4px !important;
}

/* Captions - Subtle but readable */
.stCaption {
    font-size: 0.8rem !important;
    color: #A0AEC0 !important;
    line-height: 1.4 !important;
}

/* Button text - Clear and accessible */
.stButton button {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Sidebar text - Compact but readable */
.stSidebar .stMarkdown {
    font-size: 0.9rem !important;
}

/* Code blocks - Monospace optimization */
code {
    font-size: 0.85rem !important;
    background-color: #2D3748 !important;
    color: #F7FAFC !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
    border: 1px solid #4A5568 !important;
}

/* Expandable sections - Clear hierarchy */
.streamlit-expanderHeader {
    font-size: 1rem !important;
    font-weight: 500 !important;
}

/* Performance metrics - Enhanced visibility */
.performance-metric {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* Provider icons - Consistent sizing */
.provider-icon {
    font-size: 1.2rem !important;
}

/* Enhanced data table styling */
.stDataFrame {
    border-radius: 8px !important;
    border: 1px solid #E0E0E0 !important;
    overflow: hidden !important;
}

/* System status indicators */
.status-indicator {
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Performance cards */
.performance-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    padding: 1.5rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    margin-bottom: 1rem !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.2rem !important; }
    .metric-container [data-testid="metric-container"] > div:first-child {
        font-size: 1.5rem !important;
    }
}

@media (max-width: 480px) {
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        font-size: 0.9rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Set up page title and description
st.title("🛍️ Amazon Electronics Assistant")
if st.session_state.get('rag_processor'):
    st.caption("Ask me about electronics products, reviews, comparisons, and recommendations!")
else:
    st.caption("General AI assistant (RAG not available)")

# Create main tab interface
tab_query, tab_config, tab_monitoring = st.tabs(["💬 Query", "🔧 Configuration", "📊 Monitoring"])

with tab_config:
    st.header("System Configuration")
    
    # System Status Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 System Status")
        
        # LangSmith Status
        if langsmith_result["status"] == "success":
            st.success(langsmith_result["message"])
        elif langsmith_result["status"] == "error":
            st.error(langsmith_result["message"])
            st.info(langsmith_result["fallback"])
        else:
            st.info(langsmith_result["message"])
        
        # RAG Status
        if rag_init_result["status"] == "success" and rag_init_result["has_vector_db"]:
            st.success("🔍 RAG system initialized")
        elif rag_init_result["status"] == "success":
            st.warning("⚠️ RAG processor created but vector DB unavailable")
        else:
            st.error(f"❌ RAG initialization failed: {rag_init_result.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("🤖 Model Selection")
        
        # Provider and model selection
        provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google", "Ollama"])
        
        if provider == "OpenAI":
            model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
        elif provider == "Groq":
            model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
        elif provider == "Ollama":
            model_name = st.selectbox("Model", ["gemma3n:e4b", "llama3.2:latest"], help="Make sure the model is available in your Ollama installation")
        else:
            model_name = st.selectbox("Model", ["gemini-2.0-flash"])

        # Save provider and model to session state
        st.session_state.provider = provider
        st.session_state.model_name = model_name
        
        # Parameter support info
        if provider == "Google":
            st.success("✅ All parameters supported")
        elif provider == "Ollama":
            st.success("✅ Temperature supported (local model)")
        else:
            st.warning("⚠️ Top-k not supported by OpenAI/Groq")
    
    st.divider()
    
    # Model Parameters Section
    st.subheader("⚙️ Model Parameters")
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get('temperature', 0.7),
            step=0.1,
            help="Controls randomness: 0 = focused, 2 = very creative"
        )
        st.session_state.temperature = temperature
        
        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=st.session_state.get('max_tokens', 500),
            step=50,
            help="Maximum length of the response"
        )
        st.session_state.max_tokens = max_tokens
    
    with param_col2:
        # Top-p slider
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('top_p', 1.0),
            step=0.05,
            help="Controls diversity: lower values = more focused, higher values = more diverse"
        )
        st.session_state.top_p = top_p
        
        # Top-k slider
        top_k = st.slider(
            "Top-k",
            min_value=1,
            max_value=100,
            value=st.session_state.get('top_k', 40),
            step=1,
            help="Limits vocabulary to top k tokens (Google only)"
        )
        st.session_state.top_k = top_k
    
    # Display current configuration
    st.info(f"🎛️ Current Config: Temp={temperature} | Tokens={max_tokens} | Top-p={top_p} | Top-k={top_k}")
    
    st.divider()
    
    # RAG Configuration Section
    st.subheader("🔍 RAG Configuration")
    
    rag_col1, rag_col2 = st.columns(2)
    
    with rag_col1:
        # RAG toggle
        use_rag = st.checkbox(
            "Enable RAG (Product Search)",
            value=st.session_state.get('use_rag', True),
            help="Use product database for enhanced responses about electronics"
        )
        st.session_state.use_rag = use_rag
        
        if use_rag and st.session_state.get('rag_processor'):
            st.success("🛍️ RAG will search electronics database for relevant context")
        elif use_rag and not st.session_state.get('rag_processor'):
            st.error("RAG system not available - check data files")
        else:
            st.info("🤖 Standard chatbot mode")
    
    with rag_col2:
        if use_rag and st.session_state.get('rag_processor'):
            # RAG configuration options
            max_products = st.slider(
                "Max Products",
                min_value=1,
                max_value=10,
                value=st.session_state.get('max_products', 5),
                help="Maximum number of products to retrieve"
            )
            st.session_state.max_products = max_products
            
            max_reviews = st.slider(
                "Max Reviews",
                min_value=1,
                max_value=8,
                value=st.session_state.get('max_reviews', 3),
                help="Maximum number of review summaries to retrieve"
            )
            st.session_state.max_reviews = max_reviews

with tab_query:
    st.header("Chat Interface")
    
    # Enhanced example queries with categories
    if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
        with st.expander("💡 Example Queries by Category", expanded=True):
            
            # Organize examples by category
            example_categories = {
                "🔍 Product Information": [
                    "What are the key features of iPhone charging cables?",
                    "Tell me about Fire TV Stick performance and capabilities"
                ],
                "⭐ Reviews & Feedback": [
                    "What do people say about iPhone charger cables?",
                    "What are customer experiences with Bluetooth earbuds?"
                ],
                "⚖️ Product Comparisons": [
                    "Compare Fire TV and regular tablets",
                    "Compare Ethernet cables vs USB cables for data transfer"
                ],
                "❗ Common Complaints": [
                    "What are the main complaints about laptop backpacks?",
                    "What problems do people have with wireless routers?"
                ],
                "💰 Budget Recommendations": [
                    "Recommend a budget-friendly tablet under $100",
                    "Suggest affordable alternatives to expensive noise-canceling headphones"
                ],
                "🎯 Use Case Evaluation": [
                    "Is the Ethernet cable good for gaming?",
                    "Can a smartwatch be used for fitness tracking effectively?"
                ]
            }
            
            for category, queries in example_categories.items():
                st.write(f"**{category}**")
                cols = st.columns(len(queries))
                for i, query in enumerate(queries):
                    with cols[i]:
                        if st.button(query, key=f"example_{category}_{i}", help="Click to use this query"):
                            st.session_state.prefilled_query = query
                            st.rerun()
                st.write("")  # Add some spacing
    
    # Query History
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if st.session_state.query_history:
        with st.expander("🕐 Recent Queries"):
            selected_history = st.selectbox(
                "Select a previous query:",
                options=[""] + st.session_state.query_history[-10:],  # Last 10 queries
                index=0,
                key="history_selector"
            )
            if selected_history:
                st.session_state.prefilled_query = selected_history
                st.rerun()
    
    # Quick Filters (when RAG is enabled)
    if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
        st.subheader("🎛️ Quick Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            query_type_filter = st.selectbox(
                "Query Type",
                ["Any", "Product Info", "Reviews", "Comparison", "Complaints", "Recommendations", "Use Case"],
                key="query_type_filter"
            )
        
        with filter_col2:
            category_filter = st.selectbox(
                "Category",
                ["Any", "Cables", "Audio", "Tablets", "Networking", "Gaming", "Accessories"],
                key="category_filter"
            )
        
        with filter_col3:
            price_filter = st.selectbox(
                "Price Range",
                ["Any", "Under $50", "$50-$100", "$100-$200", "Over $200"],
                key="price_filter"
            )
        
        # Apply filters button
        if st.button("🔍 Apply Filters to Next Query", help="Filters will be applied to your next question"):
            filter_context = []
            if query_type_filter != "Any":
                filter_context.append(f"Focus on {query_type_filter.lower()}")
            if category_filter != "Any":
                filter_context.append(f"in {category_filter.lower()} category")
            if price_filter != "Any":
                filter_context.append(f"with price range {price_filter.lower()}")
            
            if filter_context:
                st.session_state.active_filters = " ".join(filter_context)
                st.success(f"Filters applied: {st.session_state.active_filters}")
    
    st.divider()
    
    # Chat History Display
    if "messages" not in st.session_state:
        welcome_msg = "Hello! I'm your Amazon Electronics Assistant. "
        if st.session_state.get('rag_processor'):
            welcome_msg += "Ask me about electronics products, reviews, comparisons, and recommendations from our database of 1,000 products and 20,000 reviews!"
        else:
            welcome_msg += "How can I assist you today?"
        
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Use enhanced display for assistant messages when RAG is enabled
            if (message["role"] == "assistant" and 
                i > 0 and  # Not the welcome message
                st.session_state.get('use_rag', False)):
                display_enhanced_response(message["content"], rag_context=False)
            else:
                st.markdown(message["content"])

    # Initialize query input state
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # Handle prefilled query from example buttons
    if "prefilled_query" in st.session_state:
        st.session_state.query_input = st.session_state.prefilled_query
        del st.session_state.prefilled_query  # Clear after using

    # Build dynamic placeholder text
    placeholder_text = "Type your question here..."
    if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
        placeholder_text = "Ask about electronics products, reviews, comparisons, or use the examples above"
    
    # Add filter context to placeholder if active
    if hasattr(st.session_state, 'active_filters'):
        placeholder_text = f"{placeholder_text} (Filters: {st.session_state.active_filters})"
    
    # Query input (outside form to allow suggestions)
    query_input = st.text_input(
        "Your Question:",
        value=st.session_state.query_input,
        placeholder=placeholder_text,
        key="query_text_input",
        help="Ask about products, reviews, comparisons, or use the examples above"
    )
    
    # Show query suggestions if user is typing (outside form)
    if query_input and len(query_input) >= 3 and st.session_state.get('rag_processor'):
        suggestions = get_query_suggestions(query_input, st.session_state.rag_processor)
        if suggestions:
            st.write("💡 **Suggestions:**")
            suggestion_cols = st.columns(min(len(suggestions), 3))
            for i, suggestion in enumerate(suggestions):
                with suggestion_cols[i % 3]:
                    if st.button(suggestion, key=f"suggestion_{i}", help="Click to use this suggestion"):
                        st.session_state.query_input = suggestion
                        st.rerun()
    
    # Form for submit and clear buttons only
    with st.form(key="query_form", clear_on_submit=False):
        submit_col, clear_col = st.columns([3, 1])
        with submit_col:
            submit_button = st.form_submit_button("🚀 Send", use_container_width=True)
        with clear_col:
            clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
        
    if clear_button:
        st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
        st.session_state.query_input = ""  # Clear query input
        if hasattr(st.session_state, 'active_filters'):
            del st.session_state.active_filters
        st.rerun()
    
    if submit_button and query_input.strip():
        # Add active filters to query if they exist
        final_query = query_input
        if hasattr(st.session_state, 'active_filters'):
            final_query = f"{query_input} ({st.session_state.active_filters})"
            del st.session_state.active_filters  # Clear after use
        
        # Add to query history
        if query_input not in st.session_state.query_history:
            st.session_state.query_history.append(query_input)
        
        # Clear the session state input
        st.session_state.query_input = ""
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..." if not st.session_state.get('use_rag', False) else "Searching products and reviews..."):
                client = get_llm_client()
                output = run_llm(client, st.session_state.messages)
                
                # Use enhanced display if RAG was used
                if st.session_state.get('use_rag', False):
                    display_enhanced_response(output, rag_context=True)
                else:
                    st.write(output)
        
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.rerun()

with tab_monitoring:
    st.header("Performance Monitoring")
    
    # Performance Metrics Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Session Statistics")
        
        # Calculate session stats
        all_messages = st.session_state.get('messages', [])
        user_messages = len([m for m in all_messages if m['role'] == 'user'])
        assistant_messages = len([m for m in all_messages if m['role'] == 'assistant'])
        total_conversation_messages = user_messages + assistant_messages
        
        # Display conversation metrics
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("User Queries", user_messages)
            st.metric("Assistant Responses", assistant_messages)
        with col1b:
            st.metric("Total Conversation", total_conversation_messages)
            # Show conversation balance with smaller font
            if user_messages > 0:
                balance_ratio = assistant_messages / user_messages
                # More accurate balance logic accounting for welcome messages and normal chat flow
                if balance_ratio < 0.8:  # More user messages than assistant (assistant behind)
                    balance_status = "🟡 Pending"
                elif balance_ratio <= 2.0:  # Normal range (1:1 to 1:2 ratio, accounts for welcome messages)
                    balance_status = "🟢 Balanced"
                else:  # Significantly more assistant messages (unusual, >2:1 ratio)
                    balance_status = "🔴 Unbalanced"
                
                # Use custom styled display instead of st.metric for smaller text
                st.markdown("""
                <div style="background-color: #2D3748; padding: 0.75rem; border-radius: 6px; border: 1px solid #4A5568;">
                    <div style="font-size: 0.9rem; color: #CBD5E0; font-weight: 500; margin-bottom: 0.25rem;">Ratio (Messages)</div>
                    <div style="font-size: 1rem; color: #E2E8F0; font-weight: 500;">{}</div>
                </div>
                """.format(balance_status), unsafe_allow_html=True)
        st.metric("Query History", len(st.session_state.get('query_history', [])))
        
        # System Configuration Status
        st.subheader("⚙️ Current Configuration")
        current_provider = st.session_state.get('provider', 'Not Set')
        current_model = st.session_state.get('model_name', 'Not Set')
        rag_enabled = st.session_state.get('use_rag', False)
        
        st.write(f"**Provider:** {current_provider}")
        st.write(f"**Model:** {current_model}")
        st.write(f"**RAG Status:** {'🟢 Enabled' if rag_enabled else '🔴 Disabled'}")
        st.write(f"**Temperature:** {st.session_state.get('temperature', 0.7)}")
        st.write(f"**Max Tokens:** {st.session_state.get('max_tokens', 500)}")
    
    with col2:
        st.subheader("🔍 Recent Activity")
        
        # Show recent queries if any
        recent_queries = st.session_state.get('query_history', [])[-5:]  # Last 5 queries
        if recent_queries:
            for i, query in enumerate(reversed(recent_queries), 1):
                st.write(f"{i}. {query[:50]}{'...' if len(query) > 50 else ''}")
        else:
            st.write("No queries yet")
        
        # Enhanced RAG Performance (if available)
        if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
            st.subheader("🛍️ Enhanced RAG Performance")
            st.write(f"**Max Products:** {st.session_state.get('max_products', 5)}")
            st.write(f"**Max Reviews:** {st.session_state.get('max_reviews', 3)}")
            st.write("**Database:** 1,000 products, 20,000 reviews")
            
            # Display last RAG result if available
            if hasattr(st.session_state, 'last_rag_result'):
                rag_res = st.session_state.last_rag_result
                context = rag_res["context"]
                st.success(f"Last Query: Found {context['num_products']} products, {context['num_reviews']} reviews")
                st.caption(f"Query type: {context['query_type']}")
                st.caption(f"Processing time: {rag_res['processing_time_ms']}ms")
                
                # Display performance metrics if available
                perf_metrics = context.get('performance_metrics', {})
                if perf_metrics:
                    with st.expander("🔍 Vector Database Performance", expanded=False):
                        for search_type, metrics in perf_metrics.items():
                            if isinstance(metrics, dict) and 'embedding_metrics' in metrics:
                                st.write(f"**{search_type.title()} Search:**")
                                emb_metrics = metrics['embedding_metrics']
                                search_metrics = metrics['search_metrics']
                                quality_metrics = metrics['quality_metrics']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Embedding Time", f"{emb_metrics.get('embedding_time_ms', 0):.1f}ms")
                                with col2:
                                    st.metric("Search Time", f"{search_metrics.get('search_time_ms', 0):.1f}ms")
                                with col3:
                                    st.metric("Relevance Score", f"{quality_metrics.get('relevance_score', 0):.2f}")
            
            # Display RAG error if any
            if hasattr(st.session_state, 'last_rag_error'):
                st.error(f"Last RAG Error: {st.session_state.last_rag_error}")
        
        # Business Intelligence Metrics
        if hasattr(st.session_state, 'last_business_intelligence'):
            business_intel = st.session_state.last_business_intelligence
            
            if "error" not in business_intel:
                st.subheader("📊 Business Intelligence")
                
                # User Journey Insights
                user_journey = business_intel.get('user_journey', {})
                if user_journey:
                    col1, col2 = st.columns(2)
                    with col1:
                        user_type = user_journey.get('user_type', 'Unknown')
                        user_type_str = user_type.value if hasattr(user_type, 'value') else str(user_type)
                        st.metric("User Type", user_type_str.title())
                        
                        journey_stage = user_journey.get('journey_stage', 'Unknown')
                        journey_stage_str = journey_stage.value if hasattr(journey_stage, 'value') else str(journey_stage)
                        st.metric("Journey Stage", journey_stage_str.title())
                    with col2:
                        st.metric("Queries in Session", user_journey.get('queries_count', 0))
                        avg_satisfaction = sum(user_journey.get('satisfaction_scores', [0])) / max(len(user_journey.get('satisfaction_scores', [1])), 1)
                        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")
                
                # Business Metrics
                business_metrics = business_intel.get('business_metrics', {})
                if business_metrics:
                    with st.expander("📈 Detailed Business Metrics", expanded=False):
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Conversion Potential", f"{business_metrics.get('conversion_potential', 0):.2f}")
                            st.metric("Response Quality", f"{business_metrics.get('response_quality_score', 0):.2f}")
                        with metric_col2:
                            st.metric("Recommendation Effectiveness", f"{business_metrics.get('recommendation_effectiveness', 0):.2f}")
                            st.metric("Query Success Rate", f"{business_metrics.get('query_success_rate', 0):.2f}")
                
                # Feature Usage
                feature_usage = business_intel.get('feature_usage', {})
                if feature_usage:
                    with st.expander("🎯 Feature Usage Analytics", expanded=False):
                        for feature, count in feature_usage.items():
                            st.write(f"**{feature.replace('_', ' ').title()}:** {count}")
    
    # Enhanced Real-time Performance Metrics
    if hasattr(st.session_state, 'last_performance'):
        st.divider()
        st.subheader("⚡ Latest Query Performance")
        
        perf = st.session_state.last_performance
        
        # Display LLM provider information
        llm_provider = perf.get('llm_provider', st.session_state.get('provider', 'Unknown'))
        llm_model = perf.get('llm_model', st.session_state.get('model_name', 'Unknown'))
        
        # Provider-specific emoji mapping
        provider_emoji = {
            'OpenAI': '🔥',
            'Groq': '⚡',
            'Google': '🧠',
            'Ollama': '🏠'
        }
        
        provider_icon = provider_emoji.get(llm_provider, '🤖')
        st.markdown(f"""
        <div style="background-color: #2D3748; padding: 1rem; border-radius: 8px; border-left: 4px solid #90CAF9; margin-bottom: 1rem;">
            <span style="font-size: 1.2rem;">{provider_icon}</span> 
            <strong style="font-size: 1.1rem; color: #90CAF9;">{llm_provider}</strong> 
            <span style="color: #CBD5E0; font-size: 0.95rem;">| Model: {llm_model}</span>
        </div>
        """, unsafe_allow_html=True)
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("Total Time", f"{perf['total_time_ms']}ms")
        with perf_col2:
            st.metric("RAG Time", f"{perf['rag_time_ms']}ms")
        with perf_col3:
            st.metric(f"{provider_icon} LLM Time", f"{perf['llm_time_ms']}ms")
        
        # Performance breakdown chart
        if perf['rag_time_ms'] > 0:
            rag_percentage = (perf['rag_time_ms'] / perf['total_time_ms']) * 100
            llm_percentage = (perf['llm_time_ms'] / perf['total_time_ms']) * 100
            st.caption(f"Breakdown: RAG {rag_percentage:.1f}% | {provider_icon} {llm_provider} {llm_percentage:.1f}%")
        else:
            st.caption(f"100% {provider_icon} {llm_provider} processing")
        
        # Provider-specific performance insights
        if llm_provider == "Groq":
            if perf['llm_time_ms'] > 1000:
                st.warning("⚡ Groq response slower than expected (usually <500ms)")
            else:
                st.success("⚡ Groq delivering high-speed responses")
        elif llm_provider == "OpenAI":
            if perf['llm_time_ms'] > 5000:
                st.warning("🔥 OpenAI response time above average")
            else:
                st.info("🔥 OpenAI performing within normal range")
        elif llm_provider == "Ollama":
            if perf['llm_time_ms'] > 10000:
                st.warning("🏠 Local Ollama processing may benefit from optimization")
            else:
                st.success("🏠 Local Ollama performing well")
        
        # Business performance metrics
        business_metrics = perf.get('business_metrics', {})
        if business_metrics:
            st.subheader("📊 Business Performance Metrics")
            biz_col1, biz_col2, biz_col3, biz_col4 = st.columns(4)
            
            with biz_col1:
                st.metric("User Satisfaction", f"{business_metrics.get('user_satisfaction_prediction', 0):.2f}")
            with biz_col2:
                st.metric("Conversion Potential", f"{business_metrics.get('conversion_potential', 0):.2f}")
            with biz_col3:
                st.metric("Response Quality", f"{business_metrics.get('response_quality_score', 0):.2f}")
            with biz_col4:
                st.metric("Success Rate", f"{business_metrics.get('query_success_rate', 0):.2f}")
    
    st.divider()
    
    # Provider/Model Performance Comparison
    if hasattr(st.session_state, 'provider_model_stats') and st.session_state.provider_model_stats:
        st.subheader("🏆 Provider & Model Performance Comparison")
        
        provider_emoji = {
            'OpenAI': '🔥',
            'Groq': '⚡', 
            'Google': '🧠',
            'Ollama': '🏠'
        }
        
        # Create comparison table
        comparison_data = []
        for provider_model_key, stats in st.session_state.provider_model_stats.items():
            provider = stats["provider"]
            model = stats["model"]
            emoji = provider_emoji.get(provider, '🤖')
            
            # Calculate averages
            avg_total_time = stats["total_time_ms"] / stats["total_queries"]
            avg_llm_time = stats["total_llm_time_ms"] / stats["total_queries"]
            avg_rag_time = stats["total_rag_time_ms"] / stats["rag_queries"] if stats["rag_queries"] > 0 else 0
            
            comparison_data.append({
                "Provider": f"{emoji} {provider}",
                "Model": model,
                "Total Queries": stats["total_queries"],
                "Avg Total Time (ms)": round(avg_total_time, 1),
                "Avg LLM Time (ms)": round(avg_llm_time, 1),
                "Avg RAG Time (ms)": round(avg_rag_time, 1) if avg_rag_time > 0 else "N/A",
                "Min LLM Time (ms)": round(stats["min_llm_time_ms"], 1),
                "Max LLM Time (ms)": round(stats["max_llm_time_ms"], 1),
                "RAG Queries": stats["rag_queries"],
                "Non-RAG Queries": stats["non_rag_queries"]
            })
        
        # Sort by average LLM time (fastest first)
        comparison_data.sort(key=lambda x: x["Avg LLM Time (ms)"])
        
        # Display comparison table
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance insights
        if len(comparison_data) > 1:
            fastest = comparison_data[0]
            slowest = comparison_data[-1]
            
            st.subheader("📊 Performance Insights")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.markdown(f"""
                <div style="background-color: #1A202C; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #4CAF50;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🏃‍♂️</div>
                    <div style="font-weight: 600; color: #81C784; font-size: 1rem;">Fastest</div>
                    <div style="font-size: 0.9rem; color: #E2E8F0;">{fastest['Provider']} {fastest['Model']}</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #81C784; margin-top: 0.5rem;">{fastest['Avg LLM Time (ms)']}ms</div>
                </div>
                """, unsafe_allow_html=True)
                
            with insight_col2:
                st.markdown(f"""
                <div style="background-color: #1A202C; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #FFB74D;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🐌</div>
                    <div style="font-weight: 600; color: #FFB74D; font-size: 1rem;">Slowest</div>
                    <div style="font-size: 0.9rem; color: #E2E8F0;">{slowest['Provider']} {slowest['Model']}</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #FFB74D; margin-top: 0.5rem;">{slowest['Avg LLM Time (ms)']}ms</div>
                </div>
                """, unsafe_allow_html=True)
                
            with insight_col3:
                speed_diff = slowest['Avg LLM Time (ms)'] - fastest['Avg LLM Time (ms)']
                st.markdown(f"""
                <div style="background-color: #1A202C; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #CE93D8;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
                    <div style="font-weight: 600; color: #CE93D8; font-size: 1rem;">Speed Gap</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #CE93D8; margin-top: 0.5rem;">{speed_diff:.1f}ms</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recent performance trends for current provider/model
        current_provider = st.session_state.get('provider', 'Unknown')
        current_model = st.session_state.get('model_name', 'Unknown')
        current_key = f"{current_provider}::{current_model}"
        
        if current_key in st.session_state.provider_model_stats:
            current_stats = st.session_state.provider_model_stats[current_key]
            recent_perfs = current_stats["recent_performances"]
            
            if len(recent_perfs) >= 3:
                st.subheader(f"📈 Recent Trend: {provider_emoji.get(current_provider, '🤖')} {current_provider} {current_model}")
                
                # Create trend data
                recent_times = [p["llm_time_ms"] for p in recent_perfs[-10:]]
                recent_timestamps = [p["timestamp"] for p in recent_perfs[-10:]]
                
                # Simple trend analysis
                if len(recent_times) >= 3:
                    avg_recent_3 = sum(recent_times[-3:]) / 3
                    avg_older_3 = sum(recent_times[-6:-3]) / 3 if len(recent_times) >= 6 else avg_recent_3
                    
                    trend_col1, trend_col2, trend_col3 = st.columns(3)
                    
                    with trend_col1:
                        st.metric("Last 3 Avg", f"{avg_recent_3:.1f}ms")
                    with trend_col2:
                        st.metric("Previous 3 Avg", f"{avg_older_3:.1f}ms")
                    with trend_col3:
                        trend_change = avg_recent_3 - avg_older_3
                        trend_status = "🔺 Slower" if trend_change > 50 else "🔻 Faster" if trend_change < -50 else "➡️ Stable"
                        st.metric("Trend", trend_status)
        
        # Export functionality
        if st.button("📥 Export Performance Data"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            export_data = {
                "export_timestamp": timestamp,
                "provider_model_stats": st.session_state.provider_model_stats,
                "performance_history": st.session_state.performance_history[-50:]  # Last 50 records
            }
            
            import json
            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="💾 Download Performance Report",
                data=json_data,
                file_name=f"llm_performance_report_{timestamp}.json",
                mime="application/json"
            )
            st.success("Performance data ready for download!")
    
    st.divider()
    
    # LangSmith Tracing Integration
    if langsmith_result["status"] == "success":
        st.subheader("📈 LangSmith Tracing Dashboard")
        st.success("LangSmith tracing is active! Visit your [LangSmith Dashboard](https://smith.langchain.com) to view detailed traces.")
        st.info("Navigate to your project to see:")
        st.write("- RAG query processing metrics")
        st.write("- LLM provider performance")
        st.write("- Response timing analysis")
        st.write("- Error tracking and debugging")
    else:
        st.subheader("📈 LangSmith Tracing")
        st.warning("LangSmith tracing is not active. Add LANGSMITH_API_KEY to enable detailed monitoring.")
    
    # Session Analytics
    if hasattr(st.session_state, 'session_id') and business_tracker:
        try:
            session_summary = get_business_session_summary(st.session_state.session_id)
            
            if "error" not in session_summary:
                st.subheader("👤 Session Analytics")
                
                session_overview = session_summary.get('session_overview', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**User Type:** {session_overview.get('user_type', 'Unknown').title()}")
                    st.write(f"**Journey Stage:** {session_overview.get('journey_stage', 'Unknown').title()}")
                    st.write(f"**Total Queries:** {session_overview.get('total_queries', 0)}")
                
                with col2:
                    st.write(f"**Session Duration:** {session_overview.get('session_duration', 0):.1f}s")
                    st.write(f"**Avg Satisfaction:** {session_overview.get('avg_satisfaction', 0):.2f}")
                
                # Business recommendations
                recommendations = session_summary.get('recommendations', [])
                if recommendations:
                    with st.expander("💡 Business Recommendations", expanded=False):
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
        
        except Exception as e:
            st.warning(f"Session analytics unavailable: {str(e)}")
    
    # System Health Check
    st.subheader("🏥 System Health")
    health_checks = []
    
    # Check API configurations
    if config.OPENAI_API_KEY:
        health_checks.append("✅ OpenAI API configured")
    else:
        health_checks.append("❌ OpenAI API not configured")
    
    if config.GROQ_API_KEY:
        health_checks.append("✅ Groq API configured")
    else:
        health_checks.append("❌ Groq API not configured")
    
    if config.GOOGLE_API_KEY:
        health_checks.append("✅ Google API configured")
    else:
        health_checks.append("❌ Google API not configured")
    
    # Check RAG system
    if rag_init_result["status"] == "success" and rag_init_result["has_vector_db"]:
        health_checks.append("✅ RAG system operational")
    else:
        health_checks.append("❌ RAG system unavailable")
    
    for check in health_checks:
        st.write(check)

