"""
Agent interface for Streamlit LangGraph integration.
Provides the UI components for agent-based conversations.
"""

import streamlit as st
from typing import Dict, Any
from .langgraph_integration import create_langgraph_integration


def render_agent_interface():
    """Render the LangGraph agent interface."""
    
    st.header("🤖 AI Agent Assistant")
    st.write("Experience advanced conversational AI with persistent memory and specialized workflows.")
    
    # Initialize LangGraph integration
    if 'langgraph_integration' not in st.session_state:
        st.session_state.langgraph_integration = create_langgraph_integration()
    
    langgraph = st.session_state.langgraph_integration
    
    # Initialize session
    session_id = langgraph.initialize_session()
    
    # Display agent capabilities
    langgraph.display_agent_capabilities()
    
    # Agent configuration sidebar
    with st.sidebar:
        st.subheader("🔧 Agent Configuration")
        
        # Agent type selection
        agent_type = st.selectbox(
            "Agent Type",
            ["ambient", "product_search", "review_analysis", "comparison", "recommendation"],
            index=0,
            help="Choose the type of agent workflow"
        )
        
        # LLM provider selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "groq", "google", "ollama"],
            index=0
        )
        
        # Model selection based on provider
        if llm_provider == "openai":
            llm_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        elif llm_provider == "groq":
            llm_model = st.selectbox("Model", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"])
        elif llm_provider == "google":
            llm_model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        else:  # ollama
            llm_model = st.selectbox("Model", ["llama3.1", "llama3.2", "mistral"])
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_products = st.slider("Max Products", 1, 20, 5)
            max_reviews = st.slider("Max Reviews", 0, 10, 3)
            enable_memory = st.checkbox("Enable Memory", value=True)
        
        # Session management
        langgraph.display_session_info(session_id)
        langgraph.display_conversation_history()
    
    # Main chat interface
    st.subheader("💬 Conversation")
    
    # Initialize chat history
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    
    # Display chat messages
    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display agent workflow steps if available
            if message["role"] == "assistant" and "workflow_steps" in message:
                langgraph.display_agent_thinking(message["workflow_steps"])
            
            # Display processing info
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                with st.expander("📊 Processing Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
                    
                    with col2:
                        st.metric("Products Found", metadata.get('products_found', 0))
                    
                    with col3:
                        st.metric("Reviews Found", metadata.get('reviews_found', 0))
                    
                    if metadata.get('error_state'):
                        st.error(f"Error: {metadata['error_state']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me about electronics products..."):
        # Add user message to chat history
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                response_data = langgraph.send_message_to_agent(
                    message=prompt,
                    session_id=session_id,
                    agent_type=agent_type,
                    max_products=max_products,
                    max_reviews=max_reviews,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    temperature=temperature,
                    enable_memory=enable_memory
                )
            
            if response_data["success"]:
                response = response_data["response"]
                agent_data = response_data["data"]
                
                # Display response
                st.markdown(response)
                
                # Display workflow steps
                workflow_steps = agent_data.get("workflow_steps", [])
                if workflow_steps:
                    langgraph.display_agent_thinking(workflow_steps)
                
                # Display processing details
                with st.expander("📊 Processing Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Processing Time", f"{agent_data.get('processing_time', 0):.2f}s")
                    
                    with col2:
                        st.metric("Products Found", agent_data.get('products_found', 0))
                    
                    with col3:
                        st.metric("Reviews Found", agent_data.get('reviews_found', 0))
                    
                    # Show context information
                    context = agent_data.get("context", {})
                    if context:
                        st.write("**Query Analysis:**")
                        st.write(f"- Intent: {context.get('query_intent', 'Unknown')}")
                        entities = context.get('extracted_entities', [])
                        if entities:
                            st.write(f"- Entities: {', '.join(entities)}")
                    
                    # Show any errors
                    if agent_data.get('error_state'):
                        st.error(f"Error: {agent_data['error_state']}")
                
                # Add assistant message to chat history
                st.session_state.agent_messages.append({
                    "role": "assistant", 
                    "content": response,
                    "workflow_steps": workflow_steps,
                    "metadata": {
                        "processing_time": agent_data.get('processing_time', 0),
                        "products_found": agent_data.get('products_found', 0),
                        "reviews_found": agent_data.get('reviews_found', 0),
                        "agent_workflow": agent_data.get('agent_workflow', 'unknown'),
                        "error_state": agent_data.get('error_state')
                    }
                })
            else:
                # Display error
                error_response = response_data["response"]
                st.error(error_response)
                
                # Add error message to chat history
                st.session_state.agent_messages.append({
                    "role": "assistant", 
                    "content": error_response,
                    "metadata": {"error": response_data.get("error", "Unknown error")}
                })
    
    # Quick action buttons
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎧 Best Headphones"):
            st.session_state.agent_messages.append({
                "role": "user", 
                "content": "What are the best wireless headphones under $200?"
            })
            st.rerun()
    
    with col2:
        if st.button("💻 Gaming Laptops"):
            st.session_state.agent_messages.append({
                "role": "user", 
                "content": "Compare gaming laptops under $1000"
            })
            st.rerun()
    
    with col3:
        if st.button("📱 Phone Accessories"):
            st.session_state.agent_messages.append({
                "role": "user", 
                "content": "What phone accessories do people recommend?"
            })
            st.rerun()
    
    with col4:
        if st.button("🔌 Charging Cables"):
            st.session_state.agent_messages.append({
                "role": "user", 
                "content": "What are the most reliable USB-C cables?"
            })
            st.rerun()
    
    # Agent status and statistics
    with st.expander("📈 Agent Statistics", expanded=False):
        # Get agent status
        agent_status = langgraph.get_conversation_state(session_id)
        
        if "error" not in agent_status:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Conversation Turn", agent_status.get("conversation_turn", 0))
            
            with col2:
                st.metric("Message Count", agent_status.get("message_count", 0))
            
            with col3:
                st.metric("Current Step", agent_status.get("current_step", "Unknown"))
            
            # Last activity
            last_activity = agent_status.get("last_activity")
            if last_activity:
                st.write(f"**Last Activity:** {last_activity}")
            
            # Performance metrics
            perf_metrics = agent_status.get("performance_metrics", {})
            if perf_metrics:
                st.write("**Performance Metrics:**")
                st.json(perf_metrics)
        else:
            st.error(f"Failed to get agent status: {agent_status['error']}")


def render_agent_comparison():
    """Render comparison between traditional RAG and LangGraph agents."""
    
    st.header("⚖️ RAG vs Agent Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Traditional RAG")
        st.write("""
        **Characteristics:**
        - Direct query → search → response
        - Stateless interactions
        - Single-step processing
        - Limited context awareness
        - Fast but simple
        
        **Best for:**
        - Quick product lookups
        - Simple Q&A
        - Stateless applications
        """)
    
    with col2:
        st.subheader("🤖 LangGraph Agents")
        st.write("""
        **Characteristics:**
        - Multi-step reasoning workflows
        - Persistent conversation memory
        - Tool-based interactions
        - Context-aware responses
        - Sophisticated but slower
        
        **Best for:**
        - Complex product comparisons
        - Multi-turn conversations
        - Personalized recommendations
        - Detailed analysis
        """)
    
    st.subheader("🎯 When to Use Each Approach")
    
    comparison_data = {
        "Aspect": [
            "Response Time",
            "Memory",
            "Complexity",
            "Accuracy",
            "Personalization",
            "Resource Usage"
        ],
        "Traditional RAG": [
            "Fast (< 2s)",
            "None",
            "Simple",
            "Good",
            "Limited",
            "Low"
        ],
        "LangGraph Agent": [
            "Moderate (2-5s)",
            "Persistent",
            "Advanced",
            "Excellent",
            "High",
            "Higher"
        ]
    }
    
    st.table(comparison_data)