"""
Business Intelligence and User Experience Tracing Module.
Provides comprehensive business-level analytics and user journey tracking.
"""

import time
import json
from typing import Dict, Any, List, Optional, Set, FrozenSet, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from functools import lru_cache
from collections import defaultdict

from langsmith import traceable

from .trace_utils import (
    business_analyzer, get_current_trace_context, 
    UserType, QueryIntent
)


class UserJourneyStage(Enum):
    """Stages in the user journey."""
    EXPLORATION = "exploration"  # Initial browsing, general queries
    RESEARCH = "research"  # Detailed product investigation
    COMPARISON = "comparison"  # Comparing options
    DECISION = "decision"  # Ready to make choice
    ACTION = "action"  # Taking action (purchase-intent)
    SUPPORT = "support"  # Post-purchase or troubleshooting


@dataclass
class UserJourney:
    """Tracks user journey and engagement patterns."""
    session_id: str
    user_type: Optional[UserType] = None
    journey_stage: UserJourneyStage = UserJourneyStage.EXPLORATION
    queries_count: int = 0
    session_start_time: float = field(default_factory=time.time)
    pain_points: List[str] = field(default_factory=list)
    conversion_indicators: List[str] = field(default_factory=list)
    satisfaction_scores: List[float] = field(default_factory=list)
    
    @property
    def session_duration(self) -> float:
        """Calculate current session duration."""
        return time.time() - self.session_start_time
    
    @property
    def average_satisfaction(self) -> float:
        """Calculate average satisfaction score."""
        return sum(self.satisfaction_scores) / len(self.satisfaction_scores) if self.satisfaction_scores else 0.5
    
    def add_pain_point(self, pain_point: str) -> None:
        """Add a pain point if not already present."""
        if pain_point not in self.pain_points:
            self.pain_points.append(pain_point)
    
    def add_conversion_indicator(self, indicator: str) -> None:
        """Add a conversion indicator if not already present."""
        if indicator not in self.conversion_indicators:
            self.conversion_indicators.append(indicator)


@dataclass
class BusinessMetrics:
    """Business-level metrics for user interactions."""
    user_satisfaction_prediction: float
    conversion_potential: float
    recommendation_effectiveness: float
    feature_usage_score: float
    query_success_rate: float
    response_quality_score: float


class IntentAnalyzer:
    """Analyzes business intent from queries."""
    
    # Class-level constants for performance
    PURCHASE_INDICATORS: FrozenSet[str] = frozenset({
        "buy", "purchase", "order", "price", "cost", "cheap", "budget",
        "best", "recommend", "should i", "worth it", "deal", "discount"
    })
    
    RESEARCH_INDICATORS: FrozenSet[str] = frozenset({
        "review", "opinion", "experience", "feedback", "pros", "cons",
        "comparison", "vs", "versus", "compare", "difference", "rating"
    })
    
    SUPPORT_INDICATORS: FrozenSet[str] = frozenset({
        "problem", "issue", "help", "trouble", "broken", "fix",
        "setup", "install", "configure", "error", "not working"
    })
    
    URGENCY_INDICATORS: FrozenSet[str] = frozenset({
        "urgent", "asap", "quickly", "now", "immediate", "fast", "rush"
    })
    
    @staticmethod
    @lru_cache(maxsize=256)
    def calculate_intent_score(query: str, indicators: FrozenSet[str]) -> float:
        """Calculate intent score based on keyword presence."""
        query_words = set(query.lower().split())
        matches = len(query_words & indicators)
        return min(matches / len(indicators), 1.0)
    
    @classmethod
    def analyze_business_intent(cls, query: str) -> Dict[str, float]:
        """Analyze various business intents from query."""
        return {
            "purchase_intent": cls.calculate_intent_score(query, cls.PURCHASE_INDICATORS),
            "research_intent": cls.calculate_intent_score(query, cls.RESEARCH_INDICATORS),
            "support_intent": cls.calculate_intent_score(query, cls.SUPPORT_INDICATORS),
            "urgency_score": cls.calculate_intent_score(query, cls.URGENCY_INDICATORS)
        }


class ResponseQualityAnalyzer:
    """Analyzes response quality metrics."""
    
    # Class-level constants
    VALUE_INDICATORS: FrozenSet[str] = frozenset({
        "save", "deal", "discount", "value", "worth", "benefit", 
        "affordable", "economical", "bargain"
    })
    
    TRUST_INDICATORS: FrozenSet[str] = frozenset({
        "review", "rating", "customer", "feedback", "verified", 
        "tested", "certified", "authentic", "genuine"
    })
    
    CONCERN_INDICATORS: FrozenSet[str] = frozenset({
        "however", "but", "although", "consider", "note",
        "warning", "caution", "limitation"
    })
    
    @classmethod
    def analyze_quality_metrics(cls, response: str) -> Dict[str, Any]:
        """Analyze various quality metrics from response."""
        response_lower = response.lower()
        response_words = set(response_lower.split())
        
        # Calculate scores using set operations
        value_score = len(response_words & cls.VALUE_INDICATORS) / len(cls.VALUE_INDICATORS)
        trust_score = len(response_words & cls.TRUST_INDICATORS) / len(cls.TRUST_INDICATORS)
        
        # Completeness indicators
        completeness_factors = {
            "provides_specs": any(word in response_lower for word in ["specification", "feature", "dimension"]),
            "mentions_price": any(word in response_lower for word in ["price", "cost", "$"]),
            "includes_comparison": any(word in response_lower for word in ["compare", "versus", "alternative"]),
            "addresses_concerns": len(response_words & cls.CONCERN_INDICATORS) > 0
        }
        
        completeness_score = sum(completeness_factors.values()) / len(completeness_factors)
        
        # Professional tone check
        unprofessional_words = {"um", "uh", "like", "basically", "stuff", "thing"}
        professional_tone = 1.0 if not (response_words & unprofessional_words) else 0.5
        
        return {
            "value_proposition_score": value_score,
            "trust_building_score": trust_score,
            "completeness_score": completeness_score,
            "professional_tone": professional_tone,
            "completeness_factors": completeness_factors
        }
    
    @staticmethod
    def analyze_response_structure(response: str) -> Dict[str, int]:
        """Analyze structural characteristics of response."""
        return {
            "character_count": len(response),
            "word_count": len(response.split()),
            "sentence_count": len([s for s in response.split('.') if s.strip()]),
            "paragraph_count": len([p for p in response.split('\n\n') if p.strip()]),
            "bullet_points": response.count('•') + response.count('*') + response.count('-'),
            "has_structure": bool('\n' in response or '•' in response or '*' in response)
        }


class EngagementCalculator:
    """Calculates user engagement metrics."""
    
    @staticmethod
    def calculate_session_metrics(journey: UserJourney, conversation_turn: int) -> Dict[str, Any]:
        """Calculate session-based engagement metrics."""
        session_duration = journey.session_duration
        avg_time_per_query = session_duration / journey.queries_count if journey.queries_count > 0 else 0
        
        return {
            "session_duration": session_duration,
            "queries_per_session": journey.queries_count,
            "avg_time_per_query": avg_time_per_query,
            "conversation_depth": conversation_turn,
            "query_progression_score": min(conversation_turn * 0.1, 1.0),
            "engagement_indicators": {
                "repeat_user": journey.queries_count > 1,
                "deep_session": conversation_turn > 3,
                "active_session": session_duration < 1800,  # 30 minutes
                "research_mode": journey.journey_stage in [UserJourneyStage.RESEARCH, UserJourneyStage.COMPARISON]
            }
        }
    
    @staticmethod
    def calculate_conversion_metrics(query_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate conversion-related metrics."""
        business_scores = query_analysis.get("business_intent_scores", {})
        
        # Base conversion potential from intent
        conversion_base = (
            business_scores.get("purchase_intent", 0) * 0.5 +
            business_scores.get("research_intent", 0) * 0.3 +
            (1 - business_scores.get("support_intent", 0)) * 0.2
        )
        
        # Boost based on context usage
        context_boost = 1.0
        if context.get('num_products', 0) > 0:
            context_boost *= 1.2
        if context.get('num_reviews', 0) > 0:
            context_boost *= 1.1
        
        conversion_potential = min(conversion_base * context_boost, 1.0)
        
        # Urgency factor
        urgency_factor = query_analysis.get("business_intent_scores", {}).get("urgency_score", 0)
        
        return {
            "conversion_potential": conversion_potential,
            "urgency_factor": urgency_factor,
            "purchase_readiness": conversion_potential * (1 + urgency_factor * 0.5)
        }


class BusinessIntelligenceTracker:
    """Tracks business intelligence metrics and user experience."""
    
    def __init__(self):
        self.user_journeys: Dict[str, UserJourney] = {}
        self.session_metrics: Dict[str, Dict] = {}
        self.feature_usage = defaultdict(int)
        
        # Initialize analyzers
        self.intent_analyzer = IntentAnalyzer()
        self.quality_analyzer = ResponseQualityAnalyzer()
        self.engagement_calculator = EngagementCalculator()
    
    @traceable
    def track_user_interaction(
        self, 
        query: str, 
        response: str, 
        context: Dict[str, Any],
        session_id: str,
        conversation_turn: int
    ) -> Dict[str, Any]:
        """Track comprehensive user interaction with business intelligence."""
        
        # Get or create user journey
        journey = self._get_or_create_journey(session_id)
        journey.queries_count += 1
        
        # Perform analyses
        query_analysis = self._analyze_query(query)
        response_analysis = self._analyze_response(response, query)
        engagement_metrics = self._calculate_engagement(journey, conversation_turn, query_analysis, context)
        
        # Update journey based on analysis
        self._update_journey(journey, query_analysis, response_analysis)
        
        # Calculate business metrics
        business_metrics = self._calculate_business_metrics(
            query_analysis, response_analysis, engagement_metrics, journey
        )
        
        # Track feature usage
        self._track_features(query, context)
        
        # Update session metrics
        self._update_session_metrics(session_id, journey)
        
        return {
            "query_analysis": query_analysis,
            "response_analysis": response_analysis,
            "engagement_metrics": engagement_metrics,
            "business_metrics": asdict(business_metrics),
            "user_journey": {
                "session_id": journey.session_id,
                "user_type": journey.user_type.value if journey.user_type else None,
                "journey_stage": journey.journey_stage.value,
                "queries_count": journey.queries_count,
                "session_duration": journey.session_duration,
                "average_satisfaction": journey.average_satisfaction,
                "pain_points": journey.pain_points,
                "conversion_indicators": journey.conversion_indicators
            },
            "feature_usage": dict(self.feature_usage)
        }
    
    def _get_or_create_journey(self, session_id: str) -> UserJourney:
        """Get existing journey or create new one."""
        if session_id not in self.user_journeys:
            self.user_journeys[session_id] = UserJourney(session_id=session_id)
        return self.user_journeys[session_id]
    
    @traceable
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query from business perspective."""
        # Use business analyzer if available
        if business_analyzer:
            intent = business_analyzer.classify_intent(query)
            complexity = business_analyzer.calculate_complexity(query)
            specificity = business_analyzer.measure_specificity(query)
            product_focus = business_analyzer.extract_product_focus(query)
        else:
            intent = QueryIntent.GENERAL
            complexity = 0.5
            specificity = 0.5
            product_focus = []
        
        # Business intent analysis
        business_intent_scores = self.intent_analyzer.analyze_business_intent(query)
        
        # Query characteristics
        query_words = query.split()
        
        return {
            "intent_category": intent.value if hasattr(intent, 'value') else str(intent),
            "complexity_score": complexity,
            "specificity_level": specificity,
            "product_focus": product_focus,
            "business_intent_scores": business_intent_scores,
            "query_characteristics": {
                "word_count": len(query_words),
                "question_type": "question" if "?" in query else "statement",
                "has_urgency": business_intent_scores["urgency_score"] > 0
            }
        }
    
    @traceable
    def _analyze_response(self, response: str, query: str) -> Dict[str, Any]:
        """Analyze response quality from business perspective."""
        # Basic analysis
        if business_analyzer:
            length_category = business_analyzer.categorize_response_length(response)
            specificity_match = business_analyzer.measure_response_specificity(response, query)
            product_mentions = business_analyzer.count_product_mentions(response)
            actionable_content = business_analyzer.detect_actionable_content(response)
        else:
            length_category = "medium"
            specificity_match = 0.5
            product_mentions = 0
            actionable_content = False
        
        # Quality metrics
        quality_metrics = self.quality_analyzer.analyze_quality_metrics(response)
        
        # Response structure
        structure = self.quality_analyzer.analyze_response_structure(response)
        
        return {
            "length_category": length_category,
            "specificity_match": specificity_match,
            "product_mentions": product_mentions,
            "actionable_advice": actionable_content,
            "business_quality_metrics": quality_metrics,
            "response_characteristics": structure
        }
    
    @traceable
    def _calculate_engagement(self, journey: UserJourney, conversation_turn: int, 
                            query_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate user engagement metrics."""
        # Session metrics
        session_metrics = self.engagement_calculator.calculate_session_metrics(journey, conversation_turn)
        
        # Conversion metrics
        conversion_metrics = self.engagement_calculator.calculate_conversion_metrics(query_analysis, context)
        
        # Predicted follow-up likelihood
        follow_up_likelihood = 0.0
        if business_analyzer:
            # Note: This is simplified - in reality would need actual query/response
            follow_up_likelihood = min(conversation_turn * 0.15, 0.8)
        
        return {
            **session_metrics,
            **conversion_metrics,
            "follow_up_likelihood": follow_up_likelihood,
            "context_utilization": bool(context.get('num_products', 0) or context.get('num_reviews', 0))
        }
    
    @traceable
    def _update_journey(self, journey: UserJourney, query_analysis: Dict[str, Any], 
                       response_analysis: Dict[str, Any]) -> None:
        """Update user journey based on interaction patterns."""
        business_scores = query_analysis.get("business_intent_scores", {})
        
        # Update user type if not set
        if not journey.user_type:
            journey.user_type = self._determine_user_type(business_scores)
        
        # Update journey stage
        journey.journey_stage = self._determine_journey_stage(
            journey, business_scores, query_analysis.get("intent_category", "")
        )
        
        # Update satisfaction
        if business_analyzer:
            # Simplified satisfaction calculation
            satisfaction = (
                response_analysis.get("specificity_match", 0.5) * 0.4 +
                response_analysis.get("business_quality_metrics", {}).get("completeness_score", 0.5) * 0.3 +
                (0.8 if response_analysis.get("actionable_advice") else 0.4) * 0.3
            )
            journey.satisfaction_scores.append(satisfaction)
        
        # Detect pain points
        self._detect_pain_points(journey, query_analysis, response_analysis)
        
        # Detect conversion indicators
        self._detect_conversion_indicators(journey, query_analysis, business_scores)
    
    def _determine_user_type(self, business_scores: Dict[str, float]) -> UserType:
        """Determine user type based on intent scores."""
        if business_scores.get("research_intent", 0) > 0.3:
            return UserType.RESEARCHER
        elif business_scores.get("purchase_intent", 0) > 0.3:
            return UserType.BUYER
        elif business_scores.get("support_intent", 0) > 0.3:
            return UserType.TROUBLESHOOTER
        else:
            return UserType.CASUAL
    
    def _determine_journey_stage(self, journey: UserJourney, business_scores: Dict[str, float], 
                                intent_category: str) -> UserJourneyStage:
        """Determine current journey stage."""
        if business_scores.get("support_intent", 0) > 0.3:
            return UserJourneyStage.SUPPORT
        elif business_scores.get("purchase_intent", 0) > 0.4:
            return UserJourneyStage.ACTION
        elif "comparison" in intent_category.lower():
            return UserJourneyStage.COMPARISON
        elif journey.queries_count > 3:
            return UserJourneyStage.DECISION
        elif journey.queries_count > 1:
            return UserJourneyStage.RESEARCH
        else:
            return UserJourneyStage.EXPLORATION
    
    def _detect_pain_points(self, journey: UserJourney, query_analysis: Dict[str, Any], 
                           response_analysis: Dict[str, Any]) -> None:
        """Detect and record pain points."""
        if response_analysis.get("specificity_match", 1.0) < 0.3:
            journey.add_pain_point("low_response_relevance")
        
        if (query_analysis.get("complexity_score", 0) > 0.7 and 
            response_analysis.get("length_category") == "short"):
            journey.add_pain_point("insufficient_detail_for_complex_query")
        
        if not response_analysis.get("actionable_advice"):
            journey.add_pain_point("lack_of_actionable_guidance")
    
    def _detect_conversion_indicators(self, journey: UserJourney, query_analysis: Dict[str, Any], 
                                    business_scores: Dict[str, float]) -> None:
        """Detect and record conversion indicators."""
        if business_scores.get("purchase_intent", 0) > 0.5:
            journey.add_conversion_indicator("high_purchase_intent")
        
        if query_analysis.get("business_intent_scores", {}).get("urgency_score", 0) > 0:
            journey.add_conversion_indicator("urgency_expressed")
        
        if journey.journey_stage == UserJourneyStage.ACTION:
            journey.add_conversion_indicator("action_stage_reached")
    
    @traceable
    def _calculate_business_metrics(self, query_analysis: Dict[str, Any], 
                                  response_analysis: Dict[str, Any],
                                  engagement_metrics: Dict[str, Any], 
                                  journey: UserJourney) -> BusinessMetrics:
        """Calculate comprehensive business metrics."""
        # User satisfaction
        satisfaction = journey.average_satisfaction
        
        # Conversion potential
        conversion = engagement_metrics.get("conversion_potential", 0.0)
        
        # Recommendation effectiveness
        quality_metrics = response_analysis.get("business_quality_metrics", {})
        recommendation_effectiveness = (
            quality_metrics.get("completeness_score", 0.0) * 0.4 +
            response_analysis.get("specificity_match", 0.0) * 0.3 +
            (1.0 if response_analysis.get("actionable_advice", False) else 0.0) * 0.3
        )
        
        # Feature usage score
        total_usage = sum(self.feature_usage.values())
        feature_usage_score = min(total_usage / 100, 1.0) if total_usage > 0 else 0.0
        
        # Query success rate (based on satisfaction)
        query_success_rate = satisfaction
        
        # Response quality score
        response_quality = (
            quality_metrics.get("completeness_score", 0.0) * 0.3 +
            quality_metrics.get("trust_building_score", 0.0) * 0.3 +
            quality_metrics.get("value_proposition_score", 0.0) * 0.2 +
            quality_metrics.get("professional_tone", 0.0) * 0.2
        )
        
        return BusinessMetrics(
            user_satisfaction_prediction=satisfaction,
            conversion_potential=conversion,
            recommendation_effectiveness=recommendation_effectiveness,
            feature_usage_score=feature_usage_score,
            query_success_rate=query_success_rate,
            response_quality_score=response_quality
        )
    
    @traceable
    def _track_features(self, query: str, context: Dict[str, Any]) -> None:
        """Track usage of specific features."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Feature detection patterns
        feature_patterns = {
            "rag_usage": lambda: context.get('num_products', 0) > 0 or context.get('num_reviews', 0) > 0,
            "filter_usage": lambda: bool(query_words & {"under", "below", "cheap", "budget", "expensive", "over"}),
            "comparison_queries": lambda: bool(query_words & {"compare", "vs", "versus", "difference", "better"}),
            "recommendation_requests": lambda: bool(query_words & {"recommend", "suggest", "best", "should", "which"}),
            "review_inquiries": lambda: bool(query_words & {"review", "opinion", "feedback", "experience", "rating"})
        }
        
        # Update feature usage
        for feature, detector in feature_patterns.items():
            if detector():
                self.feature_usage[feature] += 1
    
    def _update_session_metrics(self, session_id: str, journey: UserJourney) -> None:
        """Update session-level metrics."""
        self.session_metrics[session_id] = {
            "last_interaction": time.time(),
            "total_queries": journey.queries_count,
            "user_type": journey.user_type.value if journey.user_type else "unknown",
            "journey_stage": journey.journey_stage.value,
            "avg_satisfaction": journey.average_satisfaction,
            "session_duration": journey.session_duration
        }
    
    @traceable
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary for business analysis."""
        if session_id not in self.user_journeys:
            return {"error": "Session not found"}
        
        journey = self.user_journeys[session_id]
        session_metrics = self.session_metrics.get(session_id, {})
        
        # Generate insights
        insights = self._generate_session_insights(journey)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(journey)
        
        return {
            "session_overview": {
                "session_id": session_id,
                "user_type": journey.user_type.value if journey.user_type else "unknown",
                "journey_stage": journey.journey_stage.value,
                "total_queries": journey.queries_count,
                "session_duration": journey.session_duration,
                "avg_satisfaction": journey.average_satisfaction
            },
            "pain_points": journey.pain_points,
            "conversion_indicators": journey.conversion_indicators,
            "business_insights": insights,
            "recommendations": recommendations
        }
    
    def _generate_session_insights(self, journey: UserJourney) -> Dict[str, Any]:
        """Generate insights from user journey."""
        return {
            "user_engagement_level": (
                "high" if journey.queries_count > 5 else 
                "medium" if journey.queries_count > 2 else 
                "low"
            ),
            "conversion_likelihood": (
                "high" if journey.journey_stage == UserJourneyStage.ACTION else
                "medium" if journey.journey_stage in [UserJourneyStage.DECISION, UserJourneyStage.COMPARISON] else
                "low"
            ),
            "support_needs": len([p for p in journey.pain_points if "insufficient" in p or "low" in p]),
            "satisfaction_trend": (
                "improving" if len(journey.satisfaction_scores) > 1 and 
                journey.satisfaction_scores[-1] > journey.satisfaction_scores[0] else
                "stable"
            )
        }
    
    def _generate_recommendations(self, journey: UserJourney) -> List[str]:
        """Generate business recommendations based on user journey."""
        recommendations = []
        
        # User type specific recommendations
        user_recommendations = {
            UserType.RESEARCHER: "Provide more detailed technical specifications and comparisons",
            UserType.BUYER: "Emphasize value propositions and price comparisons",
            UserType.TROUBLESHOOTER: "Offer proactive support resources and troubleshooting guides",
            UserType.CASUAL: "Provide clear, simple explanations with visual aids"
        }
        
        if journey.user_type and journey.user_type in user_recommendations:
            recommendations.append(user_recommendations[journey.user_type])
        
        # Pain point specific recommendations
        if "low_response_relevance" in journey.pain_points:
            recommendations.append("Improve RAG context retrieval and response relevance")
        
        if "lack_of_actionable_guidance" in journey.pain_points:
            recommendations.append("Include more specific action items and next steps")
        
        # Journey stage recommendations
        if journey.queries_count > 5 and journey.journey_stage == UserJourneyStage.EXPLORATION:
            recommendations.append("Guide user toward more specific product recommendations")
        
        # Satisfaction-based recommendations
        if journey.average_satisfaction < 0.6:
            recommendations.append("Review response quality and provide more actionable insights")
        
        return recommendations
    
    def cleanup_old_sessions(self, max_age_seconds: float = 3600) -> int:
        """Clean up old sessions to prevent memory leaks."""
        current_time = time.time()
        to_remove = []
        
        for session_id, metrics in self.session_metrics.items():
            if current_time - metrics.get("last_interaction", 0) > max_age_seconds:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            self.user_journeys.pop(session_id, None)
            self.session_metrics.pop(session_id, None)
        
        return len(to_remove)


# Global business intelligence tracker
business_tracker = BusinessIntelligenceTracker()


@traceable
def track_business_interaction(
    query: str, 
    response: str, 
    context: Dict[str, Any],
    session_id: Optional[str] = None,
    conversation_turn: int = 0
) -> Dict[str, Any]:
    """Track business-level user interaction with comprehensive analytics."""
    return business_tracker.track_user_interaction(
        query, response, context, session_id or str(time.time()), conversation_turn
    )


@traceable
def get_business_session_summary(session_id: str) -> Dict[str, Any]:
    """Get business intelligence summary for a session."""
    return business_tracker.get_session_summary(session_id)