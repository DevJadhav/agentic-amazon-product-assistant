"""
UI Components for the AI-Powered Amazon Product Assistant.
Provides reusable UI components and helper functions for Streamlit app.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import time


class UITheme:
    """UI theme and styling constants."""
    
    # Provider emojis
    PROVIDER_EMOJIS = {
        'OpenAI': '🔥',
        'Groq': '⚡',
        'Google': '🧠',
        'Ollama': '🏠'
    }
    
    # Status indicators
    STATUS_SUCCESS = "🟢"
    STATUS_WARNING = "🟡"
    STATUS_ERROR = "🔴"
    
    # Feature icons
    ICON_QUERY = "💬"
    ICON_SEARCH = "🔍"
    ICON_PRODUCT = "📦"
    ICON_REVIEW = "⭐"
    ICON_METRICS = "📊"
    ICON_PERFORMANCE = "⚡"
    ICON_BUSINESS = "📈"
    
    # Colors (for custom HTML)
    COLOR_PRIMARY = "#90CAF9"
    COLOR_SUCCESS = "#81C784"
    COLOR_WARNING = "#FFB74D"
    COLOR_ERROR = "#E57373"
    COLOR_BACKGROUND = "#2D3748"
    COLOR_TEXT = "#E2E8F0"


class QuerySuggestionManager:
    """Manages query suggestions based on product database."""
    
    # Product term mappings for suggestions
    PRODUCT_TERMS = {
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
    
    @classmethod
    def get_suggestions(cls, partial_query: str, max_suggestions: int = 6) -> List[str]:
        """Generate query suggestions based on partial input."""
        if not partial_query or len(partial_query) < 3:
            return []
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        for term, product_suggestions in cls.PRODUCT_TERMS.items():
            if term in partial_lower:
                for suggestion in product_suggestions:
                    if suggestion.lower() not in partial_lower:
                        suggestions.append(f"What do people say about {suggestion}?")
                        suggestions.append(f"Compare {suggestion} with alternatives")
                        if len(suggestions) >= max_suggestions:
                            return suggestions[:max_suggestions]
        
        return suggestions[:max_suggestions]


class ResponseFormatter:
    """Formats and displays enhanced responses."""
    
    @staticmethod
    def display_enhanced_response(response: str, rag_context: Optional[Dict[str, Any]] = None):
        """Display response with enhanced formatting and context cards."""
        # Display the main response
        st.markdown(response)
        
        # If RAG context is available, show additional context
        if rag_context and hasattr(st.session_state, 'last_rag_result'):
            rag_result = st.session_state.last_rag_result
            context = rag_result.get("context", {})
            
            if context.get("num_products", 0) > 0 or context.get("num_reviews", 0) > 0:
                with st.expander(f"{UITheme.ICON_SEARCH} Retrieved Context", expanded=False):
                    ResponseFormatter._display_context_tabs(context)
                    ResponseFormatter._display_query_analysis(context, rag_result)
    
    @staticmethod
    def _display_context_tabs(context: Dict[str, Any]):
        """Display product and review tabs."""
        num_products = context.get("num_products", 0)
        num_reviews = context.get("num_reviews", 0)
        
        if num_products > 0 and num_reviews > 0:
            prod_tab, review_tab = st.tabs([f"{UITheme.ICON_PRODUCT} Products", f"{UITheme.ICON_REVIEW} Reviews"])
        elif num_products > 0:
            prod_tab = st.container()
            review_tab = None
        elif num_reviews > 0:
            review_tab = st.container()
            prod_tab = None
        else:
            return
        
        # Display product cards
        if num_products > 0 and prod_tab is not None:
            with prod_tab:
                st.write(f"**Found {num_products} relevant products:**")
                for i in range(min(num_products, 3)):
                    with st.container():
                        st.write(f"**Product {i+1}**")
                        st.caption(f"{UITheme.ICON_METRICS} Product information retrieved from database")
                        st.divider()
        
        # Display review summaries
        if num_reviews > 0 and review_tab is not None:
            with review_tab:
                st.write(f"**Found {num_reviews} relevant review summaries:**")
                for i in range(min(num_reviews, 3)):
                    with st.container():
                        st.write(f"**Review Summary {i+1}**")
                        st.caption(f"{UITheme.ICON_QUERY} Customer feedback summary from database")
                        st.divider()
    
    @staticmethod
    def _display_query_analysis(context: Dict[str, Any], rag_result: Dict[str, Any]):
        """Display query analysis metrics."""
        st.subheader(f"{UITheme.ICON_SEARCH} Query Analysis")
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.write(f"**Query Type:** {context.get('query_type', 'Unknown')}")
            st.write(f"**Processing Time:** {rag_result.get('processing_time_ms', 0)}ms")
        
        with analysis_col2:
            extracted_terms = context.get('extracted_terms', [])
            if extracted_terms:
                st.write(f"**Extracted Terms:** {', '.join(extracted_terms)}")
            else:
                st.write("**Extracted Terms:** None detected")


class ExampleQueryProvider:
    """Provides example queries for different categories."""
    
    EXAMPLE_QUERIES = {
        "product_info": [
            "What features do the latest Fire TV Sticks have?",
            "Tell me about MacBook Air specifications",
            "What's included with Echo Dot 5th generation?"
        ],
        "reviews": [
            "What do people say about iPhone charging cables?",
            "Show me reviews for budget wireless headphones",
            "Are customers happy with Cat 6 ethernet cables?"
        ],
        "comparison": [
            "Compare iPad vs Samsung Galaxy Tab",
            "Difference between USB-C and Lightning cables",
            "Fire TV Stick vs Roku streaming devices"
        ],
        "recommendations": [
            "Recommend a budget laptop for students",
            "Best wireless router for gaming",
            "Suggest alternatives to expensive tablets"
        ],
        "troubleshooting": [
            "Common problems with laptop backpacks",
            "Issues with wireless mouse connectivity",
            "Fire TV remote not working solutions"
        ]
    }
    
    @classmethod
    def get_random_examples(cls, category: Optional[str] = None, count: int = 3) -> List[str]:
        """Get random example queries from a category or all categories."""
        import random
        
        if category and category in cls.EXAMPLE_QUERIES:
            examples = cls.EXAMPLE_QUERIES[category]
        else:
            # Mix from all categories
            all_examples = []
            for queries in cls.EXAMPLE_QUERIES.values():
                all_examples.extend(queries)
            examples = all_examples
        
        return random.sample(examples, min(count, len(examples)))


class MetricsDisplay:
    """Handles display of various metrics and performance indicators."""
    
    @staticmethod
    def display_metric_card(title: str, value: Any, icon: str = "", 
                          color: str = UITheme.COLOR_PRIMARY):
        """Display a custom metric card."""
        st.markdown(f"""
        <div style="background-color: {UITheme.COLOR_BACKGROUND}; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    border-left: 4px solid {color}; 
                    margin-bottom: 1rem;">
            <span style="font-size: 1.2rem;">{icon}</span> 
            <strong style="font-size: 1.1rem; color: {color};">{title}</strong> 
            <div style="font-size: 1.5rem; font-weight: 600; color: {UITheme.COLOR_TEXT}; margin-top: 0.5rem;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_performance_metrics(performance_data: Dict[str, Any]):
        """Display performance metrics in a structured format."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Time", f"{performance_data.get('total_time_ms', 0)}ms")
        with col2:
            st.metric("RAG Time", f"{performance_data.get('rag_time_ms', 0)}ms")
        with col3:
            provider = performance_data.get('llm_provider', 'LLM')
            icon = UITheme.PROVIDER_EMOJIS.get(provider, '🤖')
            st.metric(f"{icon} LLM Time", f"{performance_data.get('llm_time_ms', 0)}ms")
    
    @staticmethod
    def display_business_metrics(business_metrics: Dict[str, Any]):
        """Display business intelligence metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("User Satisfaction", 
                     f"{business_metrics.get('user_satisfaction_prediction', 0):.2f}")
        with col2:
            st.metric("Conversion Potential", 
                     f"{business_metrics.get('conversion_potential', 0):.2f}")
        with col3:
            st.metric("Response Quality", 
                     f"{business_metrics.get('response_quality_score', 0):.2f}")
        with col4:
            st.metric("Success Rate", 
                     f"{business_metrics.get('query_success_rate', 0):.2f}")


class SessionManager:
    """Manages session state and analytics."""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables."""
        defaults = {
            'messages': [],
            'query_history': [],
            'session_id': str(time.time()),
            'provider': 'OpenAI',
            'model_name': 'gpt-4o-mini',
            'temperature': 0.7,
            'max_tokens': 500,
            'top_p': 1.0,
            'top_k': 40,
            'use_rag': False,
            'max_products': 5,
            'max_reviews': 3,
            'performance_history': [],
            'provider_model_stats': {}
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def get_session_stats() -> Dict[str, Any]:
        """Get current session statistics."""
        messages = st.session_state.get('messages', [])
        user_messages = len([m for m in messages if m['role'] == 'user'])
        assistant_messages = len([m for m in messages if m['role'] == 'assistant'])
        
        return {
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'total_messages': user_messages + assistant_messages,
            'query_history_count': len(st.session_state.get('query_history', [])),
            'session_id': st.session_state.get('session_id', 'Unknown')
        }
    
    @staticmethod
    def add_to_history(query: str):
        """Add query to history if not already present."""
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)


class SystemHealthChecker:
    """Checks and displays system health status."""
    
    @staticmethod
    def get_health_checks(config: Any, rag_status: Dict[str, Any], 
                         langsmith_status: Dict[str, Any]) -> List[str]:
        """Get system health check results."""
        checks = []
        
        # API configurations
        if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
            checks.append(f"{UITheme.STATUS_SUCCESS} OpenAI API configured")
        else:
            checks.append(f"{UITheme.STATUS_ERROR} OpenAI API not configured")
        
        if hasattr(config, 'GROQ_API_KEY') and config.GROQ_API_KEY:
            checks.append(f"{UITheme.STATUS_SUCCESS} Groq API configured")
        else:
            checks.append(f"{UITheme.STATUS_ERROR} Groq API not configured")
        
        if hasattr(config, 'GOOGLE_API_KEY') and config.GOOGLE_API_KEY:
            checks.append(f"{UITheme.STATUS_SUCCESS} Google API configured")
        else:
            checks.append(f"{UITheme.STATUS_ERROR} Google API not configured")
        
        # RAG system
        if rag_status.get("status") == "success" and rag_status.get("has_vector_db"):
            checks.append(f"{UITheme.STATUS_SUCCESS} RAG system operational")
        else:
            checks.append(f"{UITheme.STATUS_ERROR} RAG system unavailable")
        
        # LangSmith tracing
        if langsmith_status.get("status") == "success":
            checks.append(f"{UITheme.STATUS_SUCCESS} LangSmith tracing active")
        else:
            checks.append(f"{UITheme.STATUS_WARNING} LangSmith tracing inactive")
        
        return checks 