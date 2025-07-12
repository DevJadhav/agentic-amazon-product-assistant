"""
Enhanced Streamlit Application for Amazon Product Assistant
Professional tab-based interface with smart query suggestions, real-time monitoring, and enhanced response visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import core modules
try:
    from chatbot_ui.core.config import config
    from rag.query_processor import create_rag_processor
    from rag.enhanced_vector_db import EnhancedVectorDB, SearchFilter, SearchConfig, EmbeddingModel
    from visualization.interactive_dashboards import InteractiveDashboard
    from tracing.trace_utils import create_enhanced_trace_context, get_current_trace_context
    from tracing.business_intelligence import track_business_interaction, business_tracker
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Import LLM clients
from openai import OpenAI
from groq import Groq
from google import genai
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSessionState:
    """Enhanced session state management with persistence and analytics."""
    
    @staticmethod
    def initialize():
        """Initialize enhanced session state variables."""
        defaults = {
            # Core state
            'session_id': str(uuid.uuid4()),
            'conversation_history': [],
            'last_response_time': None,
            'total_queries': 0,
            'successful_queries': 0,
            'provider': 'OpenAI',
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 1000,
            
            # Enhanced features
            'query_suggestions_enabled': True,
            'auto_suggestions': [],
            'user_preferences': {},
            'search_filters': SearchFilter(),
            'search_config': SearchConfig(),
            'analytics_enabled': True,
            'real_time_monitoring': True,
            
            # Performance tracking
            'response_times': [],
            'user_satisfaction': [],
            'query_types': [],
            'error_count': 0,
            'cache_hits': 0,
            
            # UI state
            'current_tab': 'Assistant',
            'sidebar_expanded': True,
            'theme': 'light',
            'show_advanced_options': False,
            'show_debug_info': False,
            
            # RAG state
            'vector_db': None,
            'rag_processor': None,
            'last_rag_result': None,
            'search_history': [],
            
            # Dashboard state
            'dashboard': None,
            'dashboard_data_loaded': False,
            'analytics_data': {},
            
            # Monitoring
            'system_health': {'status': 'healthy', 'last_check': datetime.now()},
            'performance_metrics': {},
            'user_journey': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

class EnhancedUI:
    """Enhanced UI components with professional styling and real-time features."""
    
    def __init__(self):
        self.session_state = EnhancedSessionState()
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize enhanced UI components."""
        # Initialize session state
        self.session_state.initialize()
        
        # Initialize vector database if not already done
        if st.session_state.vector_db is None:
            self.initialize_vector_db()
        
        # Initialize dashboard if not already done
        if st.session_state.dashboard is None:
            self.initialize_dashboard()
    
    def initialize_vector_db(self):
        """Initialize enhanced vector database."""
        try:
            with st.spinner("Initializing enhanced vector database..."):
                st.session_state.vector_db = EnhancedVectorDB(
                    embedding_model=EmbeddingModel.GTE_LARGE,
                    enable_async=True
                )
                st.session_state.rag_processor = create_rag_processor()
            st.success("Vector database initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize vector database: {e}")
            logger.error(f"Vector DB initialization error: {e}")
    
    def initialize_dashboard(self):
        """Initialize analytics dashboard."""
        try:
            st.session_state.dashboard = InteractiveDashboard()
            data_dir = Path("../data/processed")
            if data_dir.exists():
                st.session_state.dashboard_data_loaded = st.session_state.dashboard.load_data(data_dir)
        except Exception as e:
            logger.error(f"Dashboard initialization error: {e}")
    
    def render_header(self):
        """Render enhanced header with branding and status indicators."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("""
            <div style='display: flex; align-items: center;'>
                <h1 style='margin: 0; color: #1f77b4;'>🛒 Amazon Electronics Assistant</h1>
                <span style='margin-left: 10px; padding: 2px 8px; background: #28a745; color: white; border-radius: 12px; font-size: 12px;'>ENHANCED</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # System status indicator
            health_status = st.session_state.system_health['status']
            status_color = '#28a745' if health_status == 'healthy' else '#dc3545'
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='color: {status_color}; font-weight: bold;'>●</div>
                <small>System {health_status.title()}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Performance metrics
            avg_response_time = sum(st.session_state.response_times[-10:]) / len(st.session_state.response_times) if st.session_state.response_times else 0
            st.metric("Avg Response", f"{avg_response_time:.2f}s", delta=None)
    
    def render_sidebar(self):
        """Render enhanced sidebar with comprehensive configuration options."""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # Provider selection with enhanced options
            providers = {
                'OpenAI': {'models': ['gpt-4', 'gpt-3.5-turbo'], 'icon': '🤖'},
                'Groq': {'models': ['mixtral-8x7b-32768', 'llama2-70b-4096'], 'icon': '⚡'},
                'Google': {'models': ['gemini-pro', 'gemini-pro-vision'], 'icon': '🔍'},
                'Ollama': {'models': ['llama2', 'mistral'], 'icon': '🦙'}
            }
            
            selected_provider = st.selectbox(
                "AI Provider",
                options=list(providers.keys()),
                index=list(providers.keys()).index(st.session_state.provider),
                format_func=lambda x: f"{providers[x]['icon']} {x}"
            )
            st.session_state.provider = selected_provider
            
            # Model selection
            available_models = providers[selected_provider]['models']
            st.session_state.model = st.selectbox(
                "Model",
                options=available_models,
                index=0 if st.session_state.model not in available_models else available_models.index(st.session_state.model)
            )
            
            # Enhanced model parameters
            st.markdown("#### Model Parameters")
            st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)
            st.session_state.max_tokens = st.slider("Max Tokens", 100, 4000, st.session_state.max_tokens, 100)
            
            # Search configuration
            st.markdown("#### Search Configuration")
            st.session_state.search_config.n_results = st.slider("Results Count", 1, 20, st.session_state.search_config.n_results)
            st.session_state.search_config.hybrid_alpha = st.slider("Semantic Weight", 0.0, 1.0, st.session_state.search_config.hybrid_alpha, 0.1)
            st.session_state.search_config.diversity_threshold = st.slider("Diversity", 0.0, 1.0, st.session_state.search_config.diversity_threshold, 0.1)
            
            # Advanced filters
            with st.expander("🔍 Advanced Filters"):
                price_range = st.slider("Price Range ($)", 0, 1000, (0, 1000))
                st.session_state.search_filters.price_min = price_range[0] if price_range[0] > 0 else None
                st.session_state.search_filters.price_max = price_range[1] if price_range[1] < 1000 else None
                
                rating_range = st.slider("Rating Range", 1.0, 5.0, (1.0, 5.0), 0.1)
                st.session_state.search_filters.rating_min = rating_range[0] if rating_range[0] > 1.0 else None
                st.session_state.search_filters.rating_max = rating_range[1] if rating_range[1] < 5.0 else None
                
                min_reviews = st.number_input("Minimum Reviews", 0, 10000, 0)
                st.session_state.search_filters.review_count_min = min_reviews if min_reviews > 0 else None
                
                categories = st.multiselect("Categories", 
                    options=['Electronics', 'Computers', 'Cell Phones', 'Accessories', 'Audio', 'Video'],
                    default=[]
                )
                st.session_state.search_filters.categories = categories if categories else None
            
            # UI preferences
            st.markdown("#### Preferences")
            st.session_state.query_suggestions_enabled = st.checkbox("Smart Suggestions", st.session_state.query_suggestions_enabled)
            st.session_state.analytics_enabled = st.checkbox("Analytics", st.session_state.analytics_enabled)
            st.session_state.show_debug_info = st.checkbox("Debug Info", st.session_state.show_debug_info)
            
            # Performance monitoring
            if st.session_state.real_time_monitoring:
                st.markdown("#### 📊 Live Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Queries", st.session_state.total_queries)
                    st.metric("Success Rate", f"{(st.session_state.successful_queries/max(st.session_state.total_queries, 1)*100):.1f}%")
                with col2:
                    st.metric("Cache Hits", st.session_state.cache_hits)
                    st.metric("Errors", st.session_state.error_count)
            
            # Session actions
            st.markdown("#### Actions")
            if st.button("🔄 Reset Session"):
                self.reset_session()
            
            if st.button("📊 Export Analytics"):
                self.export_analytics()
    
    def render_main_interface(self):
        """Render the main tabbed interface."""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 Assistant", "📊 Analytics", "🔍 Search", "📈 Dashboard", "⚙️ Admin"
        ])
        
        with tab1:
            self.render_assistant_tab()
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_search_tab()
        
        with tab4:
            self.render_dashboard_tab()
        
        with tab5:
            self.render_admin_tab()
    
    def render_assistant_tab(self):
        """Render the main assistant interface with enhanced features."""
        st.markdown("### 💬 AI Assistant")
        
        # Enhanced query input with smart suggestions
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_area(
                "Ask about electronics products:",
                placeholder="e.g., 'What are the best wireless headphones under $200?'",
                height=100,
                key="query_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            submit_button = st.button("🚀 Send", type="primary", use_container_width=True)
            
            if st.button("🎲 Random", use_container_width=True):
                query = self.get_random_query()
                st.session_state.query_input = query
                st.rerun()
        
        # Smart suggestions
        if st.session_state.query_suggestions_enabled and query:
            suggestions = self.generate_smart_suggestions(query)
            if suggestions:
                st.markdown("#### 💡 Smart Suggestions")
                cols = st.columns(min(len(suggestions), 3))
                for i, suggestion in enumerate(suggestions[:3]):
                    with cols[i]:
                        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                            st.session_state.query_input = suggestion
                            st.rerun()
        
        # Example queries
        with st.expander("📝 Example Queries"):
            examples = [
                "Compare iPhone charger cables under $30",
                "What do people complain about with wireless earbuds?",
                "Best gaming keyboards for programming",
                "Tablet recommendations for students",
                "Bluetooth speakers with best battery life"
            ]
            
            cols = st.columns(2)
            for i, example in enumerate(examples):
                with cols[i % 2]:
                    if st.button(example, key=f"example_{i}"):
                        st.session_state.query_input = example
                        st.rerun()
        
        # Process query
        if submit_button and query:
            self.process_query(query)
        
        # Display conversation history
        self.render_conversation_history()
    
    def render_analytics_tab(self):
        """Render advanced analytics and insights."""
        st.markdown("### 📊 Performance Analytics")
        
        if not st.session_state.analytics_enabled:
            st.info("Analytics disabled. Enable in sidebar to view performance data.")
            return
        
        # Real-time metrics dashboard
        self.render_realtime_metrics()
        
        # User journey analysis
        st.markdown("#### 🛤️ User Journey")
        if st.session_state.user_journey:
            journey_df = pd.DataFrame(st.session_state.user_journey)
            
            # Journey timeline
            fig = px.timeline(
                journey_df,
                x_start="timestamp",
                x_end="timestamp",
                y="action_type",
                color="success",
                title="User Journey Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user journey data available yet.")
        
        # Query analytics
        st.markdown("#### 🔍 Query Analytics")
        if st.session_state.conversation_history:
            queries = [msg for msg in st.session_state.conversation_history if msg['role'] == 'user']
            
            if queries:
                # Query length distribution
                query_lengths = [len(q['content'].split()) for q in queries]
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(x=query_lengths, title="Query Length Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Query types
                    query_types = self.classify_queries([q['content'] for q in queries])
                    fig = px.pie(values=list(query_types.values()), names=list(query_types.keys()), 
                               title="Query Types")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Performance trends
        st.markdown("#### ⚡ Performance Trends")
        if st.session_state.response_times:
            response_df = pd.DataFrame({
                'Query': range(1, len(st.session_state.response_times) + 1),
                'Response Time': st.session_state.response_times
            })
            
            fig = px.line(response_df, x='Query', y='Response Time', 
                         title="Response Time Trend")
            fig.add_hline(y=response_df['Response Time'].mean(), 
                         annotation_text="Average", line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_search_tab(self):
        """Render advanced search interface."""
        st.markdown("### 🔍 Advanced Search")
        
        # Search interface
        search_query = st.text_input("Search products and reviews:", 
                                   placeholder="Enter your search query...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_type = st.selectbox("Search Type", 
                                     ["Hybrid", "Semantic Only", "Keyword Only"])
        with col2:
            sort_by = st.selectbox("Sort By", 
                                 ["Relevance", "Rating", "Price", "Reviews"])
        with col3:
            results_per_page = st.selectbox("Results", [10, 20, 50])
        
        if st.button("🔍 Search") and search_query:
            self.perform_advanced_search(search_query, search_type, sort_by, results_per_page)
        
        # Display search results
        if 'search_results' in st.session_state and st.session_state.search_results:
            self.render_search_results(st.session_state.search_results)
    
    def render_dashboard_tab(self):
        """Render the comprehensive analytics dashboard."""
        st.markdown("### 📈 Comprehensive Dashboard")
        
        if not st.session_state.dashboard_data_loaded:
            st.warning("Dashboard data not loaded. Loading data from processed files...")
            
            if st.button("Load Dashboard Data"):
                data_dir = Path("../data/processed")
                if st.session_state.dashboard.load_data(data_dir):
                    st.session_state.dashboard_data_loaded = True
                    st.success("Dashboard data loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load dashboard data.")
            return
        
        # Dashboard selection
        dashboard_type = st.selectbox(
            "Select Dashboard",
            ["Overview", "Temporal Trends", "Category Insights", "Rating Patterns"]
        )
        
        # Render selected dashboard
        if dashboard_type == "Overview":
            figures = st.session_state.dashboard.create_comprehensive_overview()
        elif dashboard_type == "Temporal Trends":
            figures = st.session_state.dashboard.create_temporal_trends_dashboard()
        elif dashboard_type == "Category Insights":
            figures = st.session_state.dashboard.create_category_insights_dashboard()
        else:
            figures = st.session_state.dashboard.create_rating_patterns_dashboard()
        
        # Display figures
        for title, fig in figures.items():
            st.plotly_chart(fig, use_container_width=True)
    
    def render_admin_tab(self):
        """Render admin interface with system management."""
        st.markdown("### ⚙️ System Administration")
        
        # System health
        st.markdown("#### 🏥 System Health")
        health_col1, health_col2, health_col3 = st.columns(3)
        
        with health_col1:
            st.metric("Vector DB Status", "Healthy" if st.session_state.vector_db else "Offline")
        with health_col2:
            st.metric("Cache Size", len(getattr(st.session_state.vector_db, 'search_cache', {})))
        with health_col3:
            st.metric("Memory Usage", "Normal")  # Could implement actual memory monitoring
        
        # Database management
        st.markdown("#### 🗄️ Database Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Get Database Stats"):
                if st.session_state.vector_db:
                    stats = st.session_state.vector_db.get_enhanced_stats()
                    st.json(stats)
        
        with col2:
            if st.button("🔄 Refresh Cache"):
                if st.session_state.vector_db:
                    st.session_state.vector_db.search_cache.clear()
                    st.success("Cache cleared successfully!")
        
        # Configuration export/import
        st.markdown("#### 💾 Configuration")
        if st.button("📤 Export Configuration"):
            config_data = {
                'provider': st.session_state.provider,
                'model': st.session_state.model,
                'temperature': st.session_state.temperature,
                'max_tokens': st.session_state.max_tokens,
                'search_filters': st.session_state.search_filters.__dict__,
                'search_config': st.session_state.search_config.__dict__
            }
            st.download_button(
                label="Download Config",
                data=json.dumps(config_data, indent=2),
                file_name="assistant_config.json",
                mime="application/json"
            )
        
        # Logs viewer
        st.markdown("#### 📜 System Logs")
        if st.checkbox("Show Debug Logs"):
            log_container = st.container()
            with log_container:
                st.text_area("Recent Logs", value=self.get_recent_logs(), height=200)
    
    def render_conversation_history(self):
        """Render enhanced conversation history with rich formatting."""
        if not st.session_state.conversation_history:
            st.info("No conversation history yet. Start by asking a question!")
            return
        
        st.markdown("#### 💬 Conversation History")
        
        for i, message in enumerate(reversed(st.session_state.conversation_history[-10:])):  # Show last 10
            with st.container():
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style='background: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        <strong>👤 You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        <strong>🤖 Assistant:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show additional context if available
                    if 'metadata' in message:
                        with st.expander("View Context", expanded=False):
                            st.json(message['metadata'])
    
    def render_realtime_metrics(self):
        """Render real-time performance metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        success_rate = (st.session_state.successful_queries / max(st.session_state.total_queries, 1)) * 100
        avg_response_time = sum(st.session_state.response_times[-10:]) / len(st.session_state.response_times) if st.session_state.response_times else 0
        cache_hit_rate = (st.session_state.cache_hits / max(st.session_state.total_queries, 1)) * 100
        
        with col1:
            st.metric("Success Rate", f"{success_rate:.1f}%", 
                     delta=f"{success_rate-90:.1f}%" if success_rate < 90 else None)
        
        with col2:
            st.metric("Avg Response", f"{avg_response_time:.2f}s",
                     delta=f"{avg_response_time-2:.2f}s" if avg_response_time > 2 else None)
        
        with col3:
            st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")
        
        with col4:
            st.metric("Total Queries", st.session_state.total_queries)
        
        # Response time chart
        if st.session_state.response_times:
            recent_times = st.session_state.response_times[-20:]  # Last 20 queries
            fig = go.Figure(data=go.Scatter(
                y=recent_times,
                mode='lines+markers',
                name='Response Time',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Response Time Trend (Last 20 Queries)",
                yaxis_title="Time (seconds)",
                xaxis_title="Query Number",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def process_query(self, query: str):
        """Process user query with enhanced tracking and error handling."""
        start_time = time.time()
        st.session_state.total_queries += 1
        
        try:
            # Add user message to history
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now().isoformat()
            })
            
            # Track user journey
            st.session_state.user_journey.append({
                'timestamp': datetime.now(),
                'action_type': 'query',
                'content': query[:50] + '...' if len(query) > 50 else query,
                'success': None  # Will be updated after processing
            })
            
            # Process with RAG
            with st.spinner("🔍 Searching products and generating response..."):
                if st.session_state.rag_processor:
                    # Enhanced search with new vector DB
                    if st.session_state.vector_db:
                        search_results = st.session_state.vector_db.enhanced_search(
                            query=query,
                            search_filter=st.session_state.search_filters,
                            config=st.session_state.search_config
                        )
                        st.session_state.last_rag_result = search_results
                    
                    # Generate response using RAG processor
                    response = st.session_state.rag_processor.process_query(
                        query,
                        conversation_history=st.session_state.conversation_history[-5:],  # Last 5 exchanges
                        additional_context=st.session_state.last_rag_result
                    )
                else:
                    response = "RAG processor not available. Please check configuration."
            
            # Calculate response time
            response_time = time.time() - start_time
            st.session_state.response_times.append(response_time)
            st.session_state.last_response_time = response_time
            st.session_state.successful_queries += 1
            
            # Add assistant response to history
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat(),
                'response_time': response_time,
                'metadata': st.session_state.last_rag_result
            })
            
            # Update user journey
            st.session_state.user_journey[-1]['success'] = True
            st.session_state.user_journey[-1]['response_time'] = response_time
            
            # Display response with enhanced formatting
            self.display_enhanced_response(response)
            
            # Business intelligence tracking
            if st.session_state.analytics_enabled:
                self.track_business_metrics(query, response, response_time)
            
        except Exception as e:
            st.session_state.error_count += 1
            st.session_state.user_journey[-1]['success'] = False
            st.session_state.user_journey[-1]['error'] = str(e)
            
            st.error(f"Error processing query: {e}")
            logger.error(f"Query processing error: {e}")
    
    def display_enhanced_response(self, response: str):
        """Display response with enhanced formatting and context."""
        # Main response
        st.markdown("### 🤖 Response")
        st.markdown(response)
        
        # Context cards if RAG data available
        if st.session_state.last_rag_result and 'results' in st.session_state.last_rag_result:
            results = st.session_state.last_rag_result['results']
            
            if results['documents'] and results['documents'][0]:
                st.markdown("#### 📚 Source Context")
                
                # Show top 3 sources in expandable cards
                for i, (doc, meta) in enumerate(zip(results['documents'][0][:3], results['metadatas'][0][:3])):
                    with st.expander(f"Source {i+1}: {meta.get('title', 'Unknown Product')[:50]}...", expanded=i==0):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Content:** {doc[:300]}...")
                            
                        with col2:
                            if meta.get('price'):
                                st.metric("Price", f"${meta['price']}")
                            if meta.get('average_rating'):
                                st.metric("Rating", f"{meta['average_rating']}/5")
                            if meta.get('review_count'):
                                st.metric("Reviews", meta['review_count'])
        
        # Response quality feedback
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.conversation_history)}"):
                self.record_feedback(True)
        
        with col2:
            if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.conversation_history)}"):
                self.record_feedback(False)
        
        with col3:
            if st.session_state.last_response_time:
                st.info(f"⚡ Responded in {st.session_state.last_response_time:.2f}s")
    
    def generate_smart_suggestions(self, partial_query: str) -> List[str]:
        """Generate smart query suggestions based on input and context."""
        suggestions = []
        
        # Keyword-based suggestions
        keywords = {
            'phone': ['iPhone accessories', 'Android phones', 'phone cases'],
            'laptop': ['laptop stands', 'laptop bags', 'gaming laptops'],
            'headphones': ['wireless headphones', 'noise-canceling headphones'],
            'charger': ['wireless chargers', 'fast chargers', 'car chargers'],
            'speaker': ['Bluetooth speakers', 'smart speakers', 'portable speakers'],
            'camera': ['security cameras', 'webcams', 'action cameras'],
            'tablet': ['iPad alternatives', 'drawing tablets', 'tablet accessories'],
            'watch': ['smartwatches', 'fitness trackers', 'watch bands']
        }
        
        query_lower = partial_query.lower()
        for keyword, related in keywords.items():
            if keyword in query_lower:
                for item in related:
                    suggestions.append(f"Compare {item} under $100")
                    suggestions.append(f"What are the best {item}?")
                    suggestions.append(f"User reviews for {item}")
                break
        
        # Add generic suggestions if no specific matches
        if not suggestions:
            suggestions = [
                "Show me trending electronics",
                "Compare wireless earbuds",
                "Best gaming accessories",
                "Budget-friendly tech gadgets"
            ]
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def get_random_query(self) -> str:
        """Get a random sample query for testing."""
        import random
        
        queries = [
            "What are the best wireless headphones under $200?",
            "Compare iPhone charger cables",
            "Show me gaming keyboards with RGB lighting",
            "What do people say about budget tablets?",
            "Best Bluetooth speakers for outdoor use",
            "Laptop accessories for remote work",
            "Smartwatch recommendations for fitness",
            "User complaints about wireless earbuds",
            "Compare mechanical keyboards for programming",
            "Best value smartphone accessories"
        ]
        
        return random.choice(queries)
    
    def classify_queries(self, queries: List[str]) -> Dict[str, int]:
        """Classify queries into types for analytics."""
        types = {
            'Product Search': 0,
            'Comparison': 0,
            'Reviews': 0,
            'Recommendations': 0,
            'Price Query': 0,
            'Other': 0
        }
        
        for query in queries:
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
                types['Comparison'] += 1
            elif any(word in query_lower for word in ['review', 'opinion', 'complaint', 'problem']):
                types['Reviews'] += 1
            elif any(word in query_lower for word in ['best', 'recommend', 'suggest', 'should i']):
                types['Recommendations'] += 1
            elif any(word in query_lower for word in ['price', 'cost', 'cheap', 'expensive', '$']):
                types['Price Query'] += 1
            elif any(word in query_lower for word in ['show', 'find', 'search', 'looking for']):
                types['Product Search'] += 1
            else:
                types['Other'] += 1
        
        return types
    
    def record_feedback(self, helpful: bool):
        """Record user feedback for analytics."""
        st.session_state.user_satisfaction.append(helpful)
        
        if helpful:
            st.success("Thank you for your feedback!")
        else:
            feedback = st.text_input("How can we improve this response?", key=f"feedback_{time.time()}")
            if feedback:
                st.success("Thank you for your detailed feedback!")
    
    def track_business_metrics(self, query: str, response: str, response_time: float):
        """Track business intelligence metrics."""
        try:
            if business_tracker:
                business_tracker.track_interaction(
                    query=query,
                    response=response,
                    response_time=response_time,
                    user_id=st.session_state.session_id,
                    session_data=st.session_state.to_dict() if hasattr(st.session_state, 'to_dict') else {}
                )
        except Exception as e:
            logger.error(f"Business tracking error: {e}")
    
    def perform_advanced_search(self, query: str, search_type: str, sort_by: str, results_count: int):
        """Perform advanced search with specified parameters."""
        try:
            if st.session_state.vector_db:
                config = SearchConfig(n_results=results_count)
                
                if search_type == "Semantic Only":
                    config.hybrid_alpha = 1.0
                elif search_type == "Keyword Only":
                    config.hybrid_alpha = 0.0
                else:  # Hybrid
                    config.hybrid_alpha = 0.7
                
                search_results = st.session_state.vector_db.enhanced_search(
                    query=query,
                    search_filter=st.session_state.search_filters,
                    config=config
                )
                
                st.session_state.search_results = search_results
                st.session_state.search_history.append({
                    'query': query,
                    'type': search_type,
                    'timestamp': datetime.now(),
                    'results_count': search_results.get('n_results', 0)
                })
                
        except Exception as e:
            st.error(f"Search failed: {e}")
    
    def render_search_results(self, results: Dict[str, Any]):
        """Render formatted search results."""
        if 'results' not in results or not results['results']['documents']:
            st.info("No results found.")
            return
        
        st.markdown(f"#### Found {results['n_results']} results")
        
        for i, (doc, meta) in enumerate(zip(results['results']['documents'][0], results['results']['metadatas'][0])):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{meta.get('title', 'Unknown Product')}**")
                    st.markdown(f"{doc[:200]}...")
                    
                    # Tags
                    tags = []
                    if meta.get('doc_type'):
                        tags.append(meta['doc_type'])
                    if meta.get('primary_category'):
                        tags.append(meta['primary_category'])
                    
                    if tags:
                        tag_str = " | ".join([f"`{tag}`" for tag in tags])
                        st.markdown(tag_str)
                
                with col2:
                    if meta.get('price'):
                        st.metric("Price", f"${meta['price']}")
                    if meta.get('average_rating'):
                        st.metric("Rating", f"{meta['average_rating']}/5")
                
                st.markdown("---")
    
    def reset_session(self):
        """Reset session state and clear data."""
        keys_to_preserve = ['vector_db', 'rag_processor', 'dashboard']
        preserved_data = {key: st.session_state.get(key) for key in keys_to_preserve}
        
        for key in list(st.session_state.keys()):
            if key not in keys_to_preserve:
                del st.session_state[key]
        
        # Restore preserved data
        for key, value in preserved_data.items():
            if value is not None:
                st.session_state[key] = value
        
        # Reinitialize
        self.session_state.initialize()
        st.success("Session reset successfully!")
        st.rerun()
    
    def export_analytics(self):
        """Export analytics data."""
        analytics_data = {
            'session_id': st.session_state.session_id,
            'total_queries': st.session_state.total_queries,
            'successful_queries': st.session_state.successful_queries,
            'error_count': st.session_state.error_count,
            'response_times': st.session_state.response_times,
            'user_satisfaction': st.session_state.user_satisfaction,
            'conversation_history': st.session_state.conversation_history,
            'user_journey': st.session_state.user_journey,
            'search_history': st.session_state.search_history
        }
        
        st.download_button(
            label="📊 Download Analytics",
            data=json.dumps(analytics_data, indent=2, default=str),
            file_name=f"analytics_{st.session_state.session_id[:8]}.json",
            mime="application/json"
        )
    
    def get_recent_logs(self) -> str:
        """Get recent log messages for display."""
        # This would typically read from actual log files
        return f"""
Recent System Activity:
{datetime.now().strftime('%H:%M:%S')} - Session initialized
{datetime.now().strftime('%H:%M:%S')} - Vector database connected
{datetime.now().strftime('%H:%M:%S')} - RAG processor ready
{datetime.now().strftime('%H:%M:%S')} - Analytics enabled
Total queries processed: {st.session_state.total_queries}
Cache hit rate: {st.session_state.cache_hits / max(st.session_state.total_queries, 1) * 100:.1f}%
        """


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Amazon Electronics Assistant",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and render enhanced UI
    ui = EnhancedUI()
    
    # Render header
    ui.render_header()
    
    # Render sidebar
    ui.render_sidebar()
    
    # Render main interface
    ui.render_main_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Enhanced Amazon Electronics Assistant** | "
        f"Session: {st.session_state.session_id[:8]} | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()