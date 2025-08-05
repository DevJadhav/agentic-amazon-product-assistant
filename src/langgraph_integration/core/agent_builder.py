"""
Agent graph builder for creating LangGraph workflows.
Provides factory methods for different types of agent workflows.
"""

import logging
from typing import Dict, Any, Optional, Type, List
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph

from .state_schemas import AgentState, create_initial_state, update_state_step
from .base_agent import BaseAgent, ProductSearchAgent, ReviewAnalysisAgent
from .router.master_graph import MasterAgentGraph
from .utils import (
    create_agent_config, 
    extract_entities_from_query, 
    classify_query_intent,
    log_agent_step
)

logger = logging.getLogger(__name__)


class AgentGraphBuilder:
    """Builds and configures LangGraph workflows for different query types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent graph builder."""
        self.config = config or create_agent_config()
        self.logger = logging.getLogger(__name__)
    
    def build_product_search_graph(self) -> CompiledStateGraph:
        """Build a product search agent workflow."""
        
        agent = ProductSearchAgent(self.config)
        return agent.compile_graph()
    
    def build_review_analysis_graph(self) -> CompiledStateGraph:
        """Build a review analysis agent workflow."""
        
        agent = ReviewAnalysisAgent(self.config)
        return agent.compile_graph()
    
    def build_comparison_graph(self) -> CompiledStateGraph:
        """Build a product comparison agent workflow."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for comparison workflow
        workflow.add_node("analyze_comparison_query", self._analyze_comparison_query)
        workflow.add_node("search_comparison_products", self._search_comparison_products)
        workflow.add_node("compare_features", self._compare_features)
        workflow.add_node("generate_comparison", self._generate_comparison)
        
        # Define edges
        workflow.add_edge(START, "analyze_comparison_query")
        workflow.add_edge("analyze_comparison_query", "search_comparison_products")
        workflow.add_edge("search_comparison_products", "compare_features")
        workflow.add_edge("compare_features", "generate_comparison")
        workflow.add_edge("generate_comparison", END)
        
        return workflow.compile()
    
    def build_recommendation_graph(self) -> CompiledStateGraph:
        """Build a product recommendation agent workflow."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for recommendation workflow
        workflow.add_node("analyze_preferences", self._analyze_preferences)
        workflow.add_node("search_candidates", self._search_candidates)
        workflow.add_node("rank_products", self._rank_products)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        
        # Define edges
        workflow.add_edge(START, "analyze_preferences")
        workflow.add_edge("analyze_preferences", "search_candidates")
        workflow.add_edge("search_candidates", "rank_products")
        workflow.add_edge("rank_products", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        
        return workflow.compile()
    
    def create_ambient_agent_graph(self) -> CompiledStateGraph:
        """Create the main ambient style agent workflow."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("route_to_specialist", self._route_to_specialist_node)
        workflow.add_node("search_products", self._search_products_node)
        workflow.add_node("analyze_reviews", self._analyze_reviews_node)
        workflow.add_node("compare_products", self._compare_products_node)
        workflow.add_node("generate_recommendations", self._generate_recommendations_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("update_memory", self._update_memory_node)
        
        # Define edges and routing logic
        workflow.add_edge(START, "analyze_query")
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_query,
            {
                "products": "search_products",
                "reviews": "analyze_reviews", 
                "comparison": "compare_products",
                "recommendation": "generate_recommendations",
                "general": "search_products"
            }
        )
        
        # All paths lead to response generation
        workflow.add_edge("search_products", "generate_response")
        workflow.add_edge("analyze_reviews", "generate_response")
        workflow.add_edge("compare_products", "generate_response")
        workflow.add_edge("generate_recommendations", "generate_response")
        
        # Final steps
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow.compile()
    
    # Node implementations for ambient agent
    
    async def _analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze the user query and extract relevant information."""
        
        query = state["current_query"]
        
        # Extract entities and classify intent
        entities = extract_entities_from_query(query)
        intent = classify_query_intent(query)
        
        # Update state
        updated_state = update_state_step(
            state, 
            "analyze_query",
            extracted_entities=entities,
            query_intent=intent
        )
        
        log_agent_step(
            state["session_id"],
            "query_analyzed",
            {"entities": entities, "intent": intent, "query": query}
        )
        
        return updated_state
    
    def _route_query(self, state: AgentState) -> str:
        """Route query to appropriate specialist based on intent."""
        
        intent = state.get("query_intent", "general")
        
        # Route based on intent
        if intent in ["comparison", "compare"]:
            return "comparison"
        elif intent in ["reviews", "feedback", "complaints"]:
            return "reviews"
        elif intent in ["recommendation", "suggest"]:
            return "recommendation"
        elif intent in ["features", "pricing", "general"]:
            return "products"
        else:
            return "general"
    
    async def _route_to_specialist_node(self, state: AgentState) -> AgentState:
        """Route to specialist agent based on query type."""
        
        # This node is used for conditional routing
        # The actual routing is handled by _route_query
        
        updated_state = update_state_step(state, "route_to_specialist")
        
        log_agent_step(
            state["session_id"],
            "routed_to_specialist",
            {"intent": state.get("query_intent", "unknown")}
        )
        
        return updated_state
    
    async def _search_products_node(self, state: AgentState) -> AgentState:
        """Search for products using vector search tool."""
        
        try:
            # Import and use vector search tool
            from ..tools.vector_search_tool import VectorSearchTool
            
            search_tool = VectorSearchTool()
            
            # Perform search
            search_result = await search_tool._arun(
                query=state["current_query"],
                search_type="hybrid",
                max_products=state.get("max_products", 5),
                max_reviews=state.get("max_reviews", 3)
            )
            
            # Extract products and reviews
            products = search_result.get("products", [])
            reviews = search_result.get("reviews", [])
            
            updated_state = update_state_step(
                state,
                "search_products",
                search_results=search_result,
                selected_products=products,
                review_summaries=reviews
            )
            
            log_agent_step(
                state["session_id"],
                "products_searched",
                {
                    "query": state["current_query"],
                    "products_found": len(products),
                    "reviews_found": len(reviews)
                }
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            
            # Return state with error information
            updated_state = update_state_step(
                state,
                "search_products",
                search_results={"error": str(e)},
                selected_products=[],
                error_state=f"Product search failed: {e}"
            )
            
            return updated_state
    
    async def _analyze_reviews_node(self, state: AgentState) -> AgentState:
        """Analyze reviews for the query."""
        
        try:
            # Import and use vector search tool for reviews
            from ..tools.vector_search_tool import VectorSearchTool
            
            search_tool = VectorSearchTool()
            
            # Search specifically for reviews
            search_result = await search_tool._arun(
                query=state["current_query"],
                search_type="hybrid",
                max_products=0,  # Focus on reviews only
                max_reviews=state.get("max_reviews", 5),
                doc_type="review_summary"
            )
            
            reviews = search_result.get("reviews", [])
            
            updated_state = update_state_step(
                state,
                "analyze_reviews",
                review_summaries=reviews,
                search_results=search_result
            )
            
            log_agent_step(
                state["session_id"],
                "reviews_analyzed",
                {
                    "query": state["current_query"],
                    "reviews_found": len(reviews)
                }
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Review analysis failed: {e}")
            
            updated_state = update_state_step(
                state,
                "analyze_reviews",
                review_summaries=[],
                error_state=f"Review analysis failed: {e}"
            )
            
            return updated_state
    
    async def _compare_products_node(self, state: AgentState) -> AgentState:
        """Compare products based on features and reviews."""
        
        try:
            # First search for products to compare
            from ..tools.vector_search_tool import VectorSearchTool
            from ..tools.product_analysis_tool import ProductAnalysisTool
            
            search_tool = VectorSearchTool()
            analysis_tool = ProductAnalysisTool()
            
            # Search for products
            search_result = await search_tool._arun(
                query=state["current_query"],
                search_type="hybrid",
                max_products=state.get("max_products", 5),
                max_reviews=2
            )
            
            products = search_result.get("products", [])
            
            # Analyze products for comparison if we have multiple products
            comparison_result = {}
            if len(products) >= 2:
                comparison_result = analysis_tool._run(
                    products=products,
                    analysis_type="comparison",
                    include_summary=True
                )
            
            updated_state = update_state_step(
                state,
                "compare_products",
                search_results=search_result,
                selected_products=products,
                search_metadata={
                    "comparison_type": "features",
                    "comparison_analysis": comparison_result
                }
            )
            
            log_agent_step(
                state["session_id"],
                "products_compared",
                {
                    "entities": state.get("extracted_entities", []),
                    "products_found": len(products),
                    "comparison_performed": len(products) >= 2
                }
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Product comparison failed: {e}")
            
            updated_state = update_state_step(
                state,
                "compare_products",
                search_metadata={"comparison_type": "features"},
                error_state=f"Product comparison failed: {e}"
            )
            
            return updated_state
    
    async def _generate_recommendations_node(self, state: AgentState) -> AgentState:
        """Generate product recommendations."""
        
        try:
            # Search for products and analyze for recommendations
            from ..tools.vector_search_tool import VectorSearchTool
            from ..tools.product_analysis_tool import ProductAnalysisTool
            
            search_tool = VectorSearchTool()
            analysis_tool = ProductAnalysisTool()
            
            # Search for products
            search_result = await search_tool._arun(
                query=state["current_query"],
                search_type="hybrid",
                max_products=state.get("max_products", 8),  # Get more for better recommendations
                max_reviews=state.get("max_reviews", 3)
            )
            
            products = search_result.get("products", [])
            
            # Analyze products for recommendations
            recommendation_analysis = {}
            if products:
                # Use pricing analysis to find best value products
                recommendation_analysis = analysis_tool._run(
                    products=products,
                    analysis_type="pricing",
                    include_summary=True
                )
            
            # Select top recommended products (limit to max_products)
            max_products = state.get("max_products", 5)
            recommended_products = products[:max_products]
            
            updated_state = update_state_step(
                state,
                "generate_recommendations",
                search_results=search_result,
                selected_products=recommended_products,
                search_metadata={
                    "recommendation_type": "value_based",
                    "recommendation_analysis": recommendation_analysis,
                    "total_candidates": len(products)
                }
            )
            
            log_agent_step(
                state["session_id"],
                "recommendations_generated",
                {
                    "query": state["current_query"],
                    "candidates_found": len(products),
                    "recommendations_selected": len(recommended_products)
                }
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            
            updated_state = update_state_step(
                state,
                "generate_recommendations",
                selected_products=[],
                search_metadata={"recommendation_type": "general"},
                error_state=f"Recommendation generation failed: {e}"
            )
            
            return updated_state
    
    async def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final response based on collected information."""
        
        try:
            # Build comprehensive context for response generation
            context_for_llm = self._build_llm_context(state)
            
            # Generate response based on query intent and available data
            response = self._generate_contextual_response(state, context_for_llm)
            
            updated_state = update_state_step(
                state,
                "generate_response",
                context_for_llm=context_for_llm,
                final_response=response,
                workflow_status="completed"
            )
            
            log_agent_step(
                state["session_id"],
                "response_generated",
                {
                    "context_length": len(context_for_llm),
                    "response_length": len(response),
                    "query_intent": state.get("query_intent", "unknown")
                }
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            
            # Fallback response
            fallback_response = f"I apologize, but I encountered an issue processing your query about '{state['current_query']}'. Please try rephrasing your question."
            
            updated_state = update_state_step(
                state,
                "generate_response",
                context_for_llm="Error occurred during processing",
                final_response=fallback_response,
                workflow_status="completed",
                error_state=f"Response generation failed: {e}"
            )
            
            return updated_state
    
    def _build_llm_context(self, state: AgentState) -> str:
        """Build comprehensive context for LLM response generation."""
        
        context_parts = []
        
        # Add query information
        query = state.get("current_query", "")
        query_intent = state.get("query_intent", "general")
        entities = state.get("extracted_entities", [])
        
        context_parts.append(f"User Query: {query}")
        context_parts.append(f"Query Intent: {query_intent}")
        
        if entities:
            context_parts.append(f"Extracted Entities: {', '.join(entities)}")
        
        # Add product information
        products = state.get("selected_products", [])
        if products:
            context_parts.append(f"\n=== PRODUCT INFORMATION ({len(products)} products) ===")
            
            for i, product in enumerate(products[:5], 1):  # Limit to top 5
                metadata = product.get("metadata", {})
                context_parts.append(f"""
Product {i}:
- Title: {metadata.get('title', 'Unknown Product')}
- Price: ${metadata.get('price', 'N/A')}
- Rating: {metadata.get('average_rating', 'N/A')}/5 ({metadata.get('rating_number', 'N/A')} reviews)
- Store: {metadata.get('store', 'N/A')}
- Content: {product.get('content', '')[:200]}...
""")
        
        # Add review information
        reviews = state.get("review_summaries", [])
        if reviews:
            context_parts.append(f"\n=== REVIEW SUMMARIES ({len(reviews)} reviews) ===")
            
            for i, review in enumerate(reviews[:3], 1):  # Limit to top 3
                metadata = review.get("metadata", {})
                context_parts.append(f"""
Review Summary {i}:
- Product: {metadata.get('product_title', 'N/A')}
- Summary: {review.get('content', '')[:300]}...
""")
        
        # Add analysis results
        search_metadata = state.get("search_metadata", {})
        if search_metadata:
            if "comparison_analysis" in search_metadata:
                analysis = search_metadata["comparison_analysis"]
                if analysis and "analysis" in analysis:
                    context_parts.append(f"\n=== COMPARISON ANALYSIS ===")
                    comparison_data = analysis["analysis"]
                    if "recommendations" in comparison_data:
                        context_parts.append(f"Recommendations: {'; '.join(comparison_data['recommendations'])}")
            
            if "recommendation_analysis" in search_metadata:
                analysis = search_metadata["recommendation_analysis"]
                if analysis and "analysis" in analysis:
                    context_parts.append(f"\n=== RECOMMENDATION ANALYSIS ===")
                    rec_data = analysis["analysis"]
                    if "summary" in rec_data:
                        context_parts.append(f"Analysis Summary: {rec_data['summary']}")
        
        return "\n".join(context_parts)
    
    def _generate_contextual_response(self, state: AgentState, context: str) -> str:
        """Generate contextual response based on query intent and available data."""
        
        query = state.get("current_query", "")
        query_intent = state.get("query_intent", "general")
        products = state.get("selected_products", [])
        reviews = state.get("review_summaries", [])
        search_metadata = state.get("search_metadata", {})
        
        # Handle different query intents
        if query_intent == "comparison" and len(products) >= 2:
            return self._generate_comparison_response(query, products, search_metadata)
        elif query_intent in ["recommendation", "suggest"] and products:
            return self._generate_recommendation_response(query, products, search_metadata)
        elif query_intent in ["reviews", "feedback"] and reviews:
            return self._generate_review_response(query, reviews, products)
        elif query_intent == "pricing" and products:
            return self._generate_pricing_response(query, products, search_metadata)
        elif products or reviews:
            return self._generate_general_response(query, products, reviews)
        else:
            return self._generate_no_results_response(query)
    
    def _generate_comparison_response(self, query: str, products: List[Dict], metadata: Dict) -> str:
        """Generate response for product comparison queries."""
        
        if len(products) < 2:
            return f"I found {len(products)} product for your comparison query '{query}', but I need at least 2 products to make a meaningful comparison. Please try a more specific search."
        
        response_parts = [f"I found {len(products)} products to compare for your query '{query}':"]
        
        # List products with key details
        for i, product in enumerate(products[:3], 1):
            metadata_info = product.get("metadata", {})
            title = metadata_info.get("title", "Unknown Product")
            price = metadata_info.get("price", "N/A")
            rating = metadata_info.get("average_rating", "N/A")
            
            response_parts.append(f"{i}. {title} - ${price}, {rating}/5 stars")
        
        # Add comparison insights if available
        comparison_analysis = metadata.get("comparison_analysis", {})
        if comparison_analysis and "analysis" in comparison_analysis:
            recommendations = comparison_analysis["analysis"].get("recommendations", [])
            if recommendations:
                response_parts.append(f"\nKey recommendations: {'; '.join(recommendations)}")
        
        response_parts.append("\nWould you like me to focus on any specific aspects like price, features, or customer reviews?")
        
        return "\n".join(response_parts)
    
    def _generate_recommendation_response(self, query: str, products: List[Dict], metadata: Dict) -> str:
        """Generate response for recommendation queries."""
        
        response_parts = [f"Based on your query '{query}', here are my top recommendations:"]
        
        # Show top recommended products
        for i, product in enumerate(products[:3], 1):
            metadata_info = product.get("metadata", {})
            title = metadata_info.get("title", "Unknown Product")
            price = metadata_info.get("price", "N/A")
            rating = metadata_info.get("average_rating", "N/A")
            rating_count = metadata_info.get("rating_number", "N/A")
            
            response_parts.append(f"{i}. {title}")
            response_parts.append(f"   Price: ${price} | Rating: {rating}/5 ({rating_count} reviews)")
        
        # Add recommendation reasoning if available
        rec_analysis = metadata.get("recommendation_analysis", {})
        if rec_analysis and "analysis" in rec_analysis:
            summary = rec_analysis["analysis"].get("summary", "")
            if summary:
                response_parts.append(f"\nAnalysis: {summary}")
        
        total_candidates = metadata.get("total_candidates", len(products))
        if total_candidates > len(products):
            response_parts.append(f"\nI analyzed {total_candidates} products to bring you these top recommendations.")
        
        return "\n".join(response_parts)
    
    def _generate_review_response(self, query: str, reviews: List[Dict], products: List[Dict]) -> str:
        """Generate response for review-focused queries."""
        
        response_parts = [f"Here's what customers are saying about '{query}':"]
        
        # Summarize reviews
        for i, review in enumerate(reviews[:3], 1):
            content = review.get("content", "")
            metadata_info = review.get("metadata", {})
            product_title = metadata_info.get("product_title", "Product")
            
            response_parts.append(f"{i}. {product_title}:")
            response_parts.append(f"   {content[:200]}...")
        
        # Add product context if available
        if products:
            response_parts.append(f"\nI also found {len(products)} related products if you'd like specific recommendations.")
        
        return "\n".join(response_parts)
    
    def _generate_pricing_response(self, query: str, products: List[Dict], metadata: Dict) -> str:
        """Generate response for pricing-focused queries."""
        
        response_parts = [f"Here's the pricing information for '{query}':"]
        
        # Show products with pricing focus
        prices = []
        for i, product in enumerate(products[:5], 1):
            metadata_info = product.get("metadata", {})
            title = metadata_info.get("title", "Unknown Product")
            price = metadata_info.get("price", "N/A")
            rating = metadata_info.get("average_rating", "N/A")
            
            response_parts.append(f"{i}. {title} - ${price} ({rating}/5 stars)")
            
            try:
                prices.append(float(str(price).replace("$", "").replace(",", "")))
            except (ValueError, AttributeError):
                pass
        
        # Add pricing analysis
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            
            response_parts.append(f"\nPrice range: ${min_price:.2f} - ${max_price:.2f}")
            response_parts.append(f"Average price: ${avg_price:.2f}")
        
        return "\n".join(response_parts)
    
    def _generate_general_response(self, query: str, products: List[Dict], reviews: List[Dict]) -> str:
        """Generate general response when products or reviews are found."""
        
        response_parts = [f"I found relevant information for your query '{query}':"]
        
        if products:
            response_parts.append(f"\n📦 Products ({len(products)} found):")
            for i, product in enumerate(products[:3], 1):
                metadata_info = product.get("metadata", {})
                title = metadata_info.get("title", "Unknown Product")
                price = metadata_info.get("price", "N/A")
                rating = metadata_info.get("average_rating", "N/A")
                
                response_parts.append(f"{i}. {title} - ${price}, {rating}/5 stars")
        
        if reviews:
            response_parts.append(f"\n⭐ Customer Reviews ({len(reviews)} summaries):")
            for i, review in enumerate(reviews[:2], 1):
                content = review.get("content", "")
                response_parts.append(f"{i}. {content[:150]}...")
        
        response_parts.append("\nWould you like more details about any specific product or aspect?")
        
        return "\n".join(response_parts)
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generate response when no relevant results are found."""
        
        return f"I couldn't find specific products or reviews matching '{query}'. This might be because:\n\n" \
               f"• The product isn't in our electronics database\n" \
               f"• The search terms might be too specific or contain typos\n" \
               f"• Try using more general terms (e.g., 'wireless headphones' instead of specific model numbers)\n\n" \
               f"Would you like to try a different search or ask about a general product category?"
    
    async def _update_memory_node(self, state: AgentState) -> AgentState:
        """Update conversation memory with the interaction."""
        
        # Add messages to conversation history
        messages = state["messages"].copy()
        
        # Add user message if not already present
        if not any(isinstance(msg, HumanMessage) and msg.content == state["current_query"] for msg in messages):
            messages.append(HumanMessage(content=state["current_query"]))
        
        # Add assistant response
        if state.get("final_response"):
            messages.append(AIMessage(content=state["final_response"]))
        
        updated_state = update_state_step(
            state,
            "update_memory",
            messages=messages,
            conversation_turn=state["conversation_turn"] + 1
        )
        
        log_agent_step(
            state["session_id"],
            "memory_updated",
            {"message_count": len(messages)}
        )
        
        return updated_state
    
    # Comparison workflow nodes
    
    async def _analyze_comparison_query(self, state: AgentState) -> AgentState:
        """Analyze query for product comparison."""
        
        updated_state = update_state_step(state, "analyze_comparison_query")
        return updated_state
    
    async def _search_comparison_products(self, state: AgentState) -> AgentState:
        """Search for products to compare."""
        
        updated_state = update_state_step(state, "search_comparison_products")
        return updated_state
    
    async def _compare_features(self, state: AgentState) -> AgentState:
        """Compare features of selected products."""
        
        updated_state = update_state_step(state, "compare_features")
        return updated_state
    
    async def _generate_comparison(self, state: AgentState) -> AgentState:
        """Generate comparison response."""
        
        updated_state = update_state_step(
            state, 
            "generate_comparison",
            final_response="Comparison response will be generated here",
            workflow_status="completed"
        )
        return updated_state
    
    # Recommendation workflow nodes
    
    async def _analyze_preferences(self, state: AgentState) -> AgentState:
        """Analyze user preferences for recommendations."""
        
        updated_state = update_state_step(state, "analyze_preferences")
        return updated_state
    
    async def _search_candidates(self, state: AgentState) -> AgentState:
        """Search for candidate products for recommendation."""
        
        updated_state = update_state_step(state, "search_candidates")
        return updated_state
    
    async def _rank_products(self, state: AgentState) -> AgentState:
        """Rank products based on user preferences."""
        
        updated_state = update_state_step(state, "rank_products")
        return updated_state
    
    async def _generate_recommendations(self, state: AgentState) -> AgentState:
        """Generate final recommendations."""
        
        updated_state = update_state_step(
            state,
            "generate_recommendations", 
            final_response="Recommendations will be generated here",
            workflow_status="completed"
        )
        return updated_state
    
    def create_master_routing_graph(self) -> CompiledStateGraph:
        """
        Create the master routing graph with intelligent agent selection.
        
        This method creates the main orchestration graph that provides:
        - Intent classification and routing
        - Specialized agent execution (Product QA and Shopping Cart)
        - Clarification handling for ambiguous queries
        - Response finalization with consistent formatting
        
        Returns:
            CompiledStateGraph: Compiled master routing graph ready for execution
        """
        
        master_agent_graph = MasterAgentGraph(self.config, agent_builder=self)
        return master_agent_graph.compile_graph()
    
    def create_product_qa_agent_graph(self) -> CompiledStateGraph:
        """
        Create the Product QA Agent graph for product-related queries.
        
        This is an alias for the ambient agent graph with improved naming
        to reflect its role in the master routing system.
        
        Returns:
            CompiledStateGraph: Compiled Product QA Agent graph
        """
        return self.create_ambient_agent_graph()
    
    def create_shopping_cart_agent_graph(self) -> CompiledStateGraph:
        """
        Create the Shopping Cart Agent graph for cart management operations.
        
        Note: This method creates a standalone cart agent graph. In the master
        routing system, the cart agent is integrated directly into the master graph.
        
        Returns:
            CompiledStateGraph: Compiled Shopping Cart Agent graph
        """
        from .router.master_graph import ShoppingCartAgent
        
        cart_agent = ShoppingCartAgent(self.config.get("cart_agent", {}))
        return cart_agent.compile_graph()
    
    def get_available_graphs(self) -> Dict[str, str]:
        """
        Get list of available agent graphs with improved naming and organization.
        
        Returns:
            Dict[str, str]: Dictionary mapping graph names to descriptions
        """
        
        return {
            # Master orchestration graph
            "master_routing_graph": "Master routing graph with intelligent agent selection and orchestration",
            
            # Specialized agent graphs
            "product_qa_agent": "Product QA Agent for search, analysis, and recommendations",
            "shopping_cart_agent": "Shopping Cart Agent for cart management operations",
            
            # Legacy/specialized workflow graphs
            "ambient_agent": "Main ambient style agent for general product queries",
            "product_search_agent": "Specialized agent for product search workflows",
            "review_analysis_agent": "Specialized agent for review analysis workflows", 
            "product_comparison_agent": "Specialized agent for product comparison workflows",
            "product_recommendation_agent": "Specialized agent for product recommendation workflows"
        }
    
    def get_agent_hierarchy_mapping(self) -> Dict[str, Any]:
        """
        Get the agent hierarchy and relationship mapping.
        
        Returns:
            Dict[str, Any]: Complete agent hierarchy documentation
        """
        
        return {
            "orchestration_layer": {
                "master_routing_graph": {
                    "role": "orchestration",
                    "description": "Top-level routing and agent coordination",
                    "manages": ["product_qa_agent", "shopping_cart_agent"],
                    "components": ["intent_router", "clarification_handler"],
                    "routing_logic": "intent_classification_based"
                }
            },
            "specialized_agents": {
                "product_qa_agent": {
                    "role": "product_information",
                    "description": "Handles product search, analysis, and recommendations",
                    "tools": ["vector_search_mcp", "product_analysis_mcp"],
                    "tool_type": "mcp_tools",
                    "workflows": ["search", "analysis", "comparison", "recommendation"],
                    "fallback_for": ["unclear_product_queries", "general_queries"]
                },
                "shopping_cart_agent": {
                    "role": "cart_management",
                    "description": "Manages shopping cart operations and state",
                    "tools": ["add_to_cart", "remove_from_cart", "list_cart", "clear_cart"],
                    "tool_type": "function_calling",
                    "workflows": ["add", "remove", "list", "clear"],
                    "state_management": "persistent_database"
                }
            },
            "legacy_agents": {
                "ambient_agent": {
                    "role": "general_purpose",
                    "description": "Legacy ambient agent, now integrated as Product QA Agent",
                    "status": "legacy",
                    "replacement": "product_qa_agent"
                },
                "product_search_agent": {
                    "role": "specialized_workflow",
                    "description": "Specialized product search workflow",
                    "status": "legacy",
                    "integrated_into": "product_qa_agent"
                },
                "review_analysis_agent": {
                    "role": "specialized_workflow", 
                    "description": "Specialized review analysis workflow",
                    "status": "legacy",
                    "integrated_into": "product_qa_agent"
                },
                "product_comparison_agent": {
                    "role": "specialized_workflow",
                    "description": "Specialized product comparison workflow", 
                    "status": "legacy",
                    "integrated_into": "product_qa_agent"
                },
                "product_recommendation_agent": {
                    "role": "specialized_workflow",
                    "description": "Specialized product recommendation workflow",
                    "status": "legacy", 
                    "integrated_into": "product_qa_agent"
                }
            },
            "routing_patterns": {
                "intent_based_routing": {
                    "description": "Routes based on classified user intent",
                    "confidence_threshold": self.config.get("router", {}).get("confidence_threshold", 0.7),
                    "fallback_strategy": "clarification_request"
                },
                "clarification_handling": {
                    "description": "Handles ambiguous or unclear user queries",
                    "max_attempts": self.config.get("clarification", {}).get("max_clarification_attempts", 3),
                    "fallback_agent": "product_qa_agent"
                }
            }
        }
    
    def get_graph_naming_conventions(self) -> Dict[str, Any]:
        """
        Get the graph naming conventions and standards.
        
        Returns:
            Dict[str, Any]: Naming conventions documentation
        """
        
        return {
            "node_naming_conventions": {
                "pattern": "action_description_and_purpose",
                "examples": [
                    "intent_classification_and_routing",
                    "product_qa_agent_execution", 
                    "shopping_cart_agent_execution",
                    "clarification_request_handling",
                    "response_finalization_and_formatting"
                ],
                "guidelines": [
                    "Use descriptive names that clearly indicate the node's purpose",
                    "Include both the action and the domain/context",
                    "Use underscores to separate words",
                    "Avoid abbreviations unless widely understood",
                    "Be consistent across similar node types"
                ]
            },
            "edge_naming_conventions": {
                "pattern": "action_to_target",
                "examples": [
                    "route_to_qa_agent",
                    "route_to_cart_agent",
                    "request_clarification"
                ],
                "guidelines": [
                    "Use verb phrases that describe the routing action",
                    "Include the target destination",
                    "Be consistent with conditional edge naming",
                    "Make routing logic clear from the name"
                ]
            },
            "agent_naming_conventions": {
                "pattern": "domain_agent_type",
                "examples": [
                    "product_qa_agent",
                    "shopping_cart_agent",
                    "master_routing_graph"
                ],
                "guidelines": [
                    "Include the domain or specialization",
                    "Use 'agent' suffix for execution agents",
                    "Use 'graph' suffix for orchestration graphs",
                    "Avoid generic names like 'main' or 'default'"
                ]
            },
            "method_naming_conventions": {
                "pattern": "action_domain_purpose",
                "examples": [
                    "_execute_intent_router",
                    "_execute_product_qa_agent",
                    "_handle_clarification_request",
                    "_finalize_and_format_response"
                ],
                "guidelines": [
                    "Use descriptive method names that indicate purpose",
                    "Prefix private methods with underscore",
                    "Include the domain or component being acted upon",
                    "Use consistent verb patterns (execute, handle, process, etc.)"
                ]
            }
        }