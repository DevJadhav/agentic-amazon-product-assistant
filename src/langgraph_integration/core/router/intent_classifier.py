"""
Intent classification system for routing user queries to appropriate agents.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from ...monitoring.performance_monitor import get_performance_monitor, performance_track

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Result of intent classification with confidence and metadata."""
    
    intent: str  # 'qa', 'cart', 'unclear'
    confidence: float  # 0.0 to 1.0
    entities: List[str]  # Extracted entities from the message
    clarification_needed: bool
    suggested_questions: List[str]
    reasoning: str  # Explanation of classification decision
    metadata: Dict[str, Any]  # Additional classification metadata


class IntentClassifier:
    """Classifies user intent for routing decisions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize intent classifier with configuration."""
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Intent patterns for rule-based classification
        self._cart_patterns = [
            # Direct cart actions
            r'\b(add|put|place)\b.*\b(cart|basket)\b',
            r'\b(remove|delete|take out)\b.*\b(from|cart|basket)\b',
            r'\b(show|view|list|display)\b.*\b(cart|basket)\b',
            r'\b(clear|empty)\b.*\b(cart|basket)\b',
            r'\b(cart|basket)\b.*\b(contents|items)\b',
            
            # Shopping actions
            r'\bi want to buy\b',
            r'\bi\'ll take\b',
            r'\badd.*to.*cart\b',
            r'\bput.*in.*cart\b',
            r'\bremove.*from.*cart\b',
            r'\bdelete.*from.*cart\b',
            
            # Quantity expressions
            r'\b\d+\s*(of|x)\s*\w+',
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+\w+',
            
            # Purchase intent
            r'\bbuy\s+\d+\b',
            r'\bpurchase\s+\d+\b',
            r'\border\s+\d+\b',
        ]
        
        self._qa_patterns = [
            # Information seeking
            r'\b(what|how|why|when|where|which)\b',
            r'\b(tell me|explain|describe)\b',
            r'\b(compare|comparison|vs|versus)\b',
            r'\b(review|rating|opinion)\b',
            r'\b(feature|specification|spec)\b',
            r'\b(price|cost|expensive|cheap)\b',
            r'\b(recommend|suggestion|best)\b',
            r'\b(difference|similar|like)\b',
            
            # Product information
            r'\b(product|item)\s+(info|information|details)\b',
            r'\b(brand|manufacturer|make)\b',
            r'\b(model|version|type)\b',
            r'\b(availability|available|in stock)\b',
        ]
        
        # Compile patterns for better performance
        self._compiled_cart_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._cart_patterns]
        self._compiled_qa_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._qa_patterns]
        
        # Entity extraction patterns
        self._quantity_patterns = [
            r'\b(\d+)\s*(of|x)?\s*([a-zA-Z]+)',
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-zA-Z]+)',
            r'\b(\d+)\s+(item|product|piece)s?\b',
        ]
        self._compiled_quantity_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._quantity_patterns]
        
        # Product name patterns (simple heuristics)
        self._product_patterns = [
            r'\b([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z0-9]+)',  # Brand Model
            r'\b([a-zA-Z]+\s+\d+[a-zA-Z]*)',  # Product with numbers
            r'"([^"]+)"',  # Quoted product names
            r"'([^']+)'",  # Single quoted product names
        ]
        self._compiled_product_patterns = [re.compile(pattern) for pattern in self._product_patterns]
    
    @performance_track("intent_classification")
    def classify_intent(self, message: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """
        Classify user intent from message and conversation context.
        
        Args:
            message: User's message to classify
            context: Optional conversation context for better classification
            
        Returns:
            IntentResult with classification details
        """
        if not message or not message.strip():
            return self._create_unclear_result(
                message, 
                "Empty or whitespace-only message",
                ["Could you please tell me what you're looking for?"]
            )
        
        message = message.strip()
        context = context or {}
        
        self.logger.debug(f"Classifying intent for message: '{message[:50]}...'")
        
        # Check cache first
        try:
            from .intent_cache import get_intent_cache
            cache = get_intent_cache()
            cached_result = cache.get(message, context)
            if cached_result:
                self.logger.debug("Using cached intent classification result")
                return cached_result
        except ImportError:
            pass  # Cache not available
        
        # Extract entities first
        entities = self._extract_entities(message)
        
        # Calculate intent scores
        cart_score = self._calculate_cart_score(message, entities, context)
        qa_score = self._calculate_qa_score(message, entities, context)
        
        # Determine intent based on scores
        intent, confidence, reasoning = self._determine_intent(cart_score, qa_score, message)
        
        # Check if clarification is needed
        clarification_needed = confidence < self.confidence_threshold
        suggested_questions = []
        
        if clarification_needed:
            suggested_questions = self._generate_clarification_questions(message, entities, cart_score, qa_score)
        
        result = IntentResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            clarification_needed=clarification_needed,
            suggested_questions=suggested_questions,
            reasoning=reasoning,
            metadata={
                "cart_score": cart_score,
                "qa_score": qa_score,
                "message_length": len(message),
                "entity_count": len(entities),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Cache the result
        try:
            from .intent_cache import get_intent_cache
            cache = get_intent_cache()
            cache.set(message, result, context)
        except ImportError:
            pass  # Cache not available
        
        self.logger.info(f"Intent classified as '{intent}' with confidence {confidence:.2f}")
        return result
    
    def get_confidence_score(self, intent: str, message: str) -> float:
        """Get confidence score for a specific intent classification."""
        result = self.classify_intent(message)
        if result.intent == intent:
            return result.confidence
        return 0.0
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract entities from the message (quantities, product names, etc.)."""
        entities = []
        
        # Extract quantities
        for pattern in self._compiled_quantity_patterns:
            matches = pattern.findall(message)
            for match in matches:
                if isinstance(match, tuple):
                    # Join tuple elements, filtering out empty strings
                    entity = " ".join(filter(None, match))
                else:
                    entity = match
                if entity and entity not in entities:
                    entities.append(entity)
        
        # Extract potential product names
        for pattern in self._compiled_product_patterns:
            matches = pattern.findall(message)
            for match in matches:
                if isinstance(match, tuple):
                    entity = " ".join(filter(None, match))
                else:
                    entity = match
                if entity and len(entity) > 2 and entity not in entities:
                    entities.append(entity)
        
        # Extract common shopping terms
        shopping_terms = ['laptop', 'phone', 'tablet', 'headphones', 'camera', 'watch', 'speaker']
        message_lower = message.lower()
        for term in shopping_terms:
            if term in message_lower and term not in entities:
                entities.append(term)
        
        return entities[:10]  # Limit to prevent too many entities
    
    def _calculate_cart_score(self, message: str, entities: List[str], context: Dict[str, Any]) -> float:
        """Calculate confidence score for cart intent."""
        score = 0.0
        
        # Pattern matching
        pattern_matches = 0
        for pattern in self._compiled_cart_patterns:
            if pattern.search(message):
                pattern_matches += 1
        
        # Base score from pattern matches
        if pattern_matches > 0:
            score += min(0.9, pattern_matches * 0.4)
        
        # Boost for explicit cart keywords
        cart_keywords = ['cart', 'basket', 'buy', 'purchase', 'order', 'add', 'remove']
        message_lower = message.lower()
        keyword_count = sum(1 for keyword in cart_keywords if keyword in message_lower)
        if keyword_count > 0:
            score += min(0.5, keyword_count * 0.2)
        
        # Boost for quantity entities
        quantity_entities = [e for e in entities if any(char.isdigit() for char in e)]
        if quantity_entities:
            score += min(0.4, len(quantity_entities) * 0.2)
        
        # Context boost (if previous interactions were cart-related)
        if context.get('recent_cart_activity', False):
            score += 0.2
        
        # Penalty for question words (less likely to be cart actions)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        question_count = sum(1 for word in question_words if word in message_lower)
        if question_count > 0:
            score -= min(0.3, question_count * 0.1)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_qa_score(self, message: str, entities: List[str], context: Dict[str, Any]) -> float:
        """Calculate confidence score for QA intent."""
        score = 0.0
        
        # Pattern matching
        pattern_matches = 0
        for pattern in self._compiled_qa_patterns:
            if pattern.search(message):
                pattern_matches += 1
        
        # Base score from pattern matches
        if pattern_matches > 0:
            score += min(0.9, pattern_matches * 0.35)
        
        # Boost for question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        message_lower = message.lower()
        question_count = sum(1 for word in question_words if word in message_lower)
        if question_count > 0:
            score += min(0.6, question_count * 0.25)
        
        # Boost for information-seeking keywords
        info_keywords = ['tell me', 'explain', 'describe', 'compare', 'review', 'recommend', 'best', 'difference']
        keyword_count = sum(1 for keyword in info_keywords if keyword in message_lower)
        if keyword_count > 0:
            score += min(0.5, keyword_count * 0.2)
        
        # Boost for question marks
        if '?' in message:
            score += 0.3
        
        # Context boost (if previous interactions were QA-related)
        if context.get('recent_qa_activity', False):
            score += 0.2
        
        # Penalty for action words (less likely to be information seeking)
        action_words = ['add', 'remove', 'buy', 'purchase', 'order', 'delete']
        action_count = sum(1 for word in action_words if word in message_lower)
        if action_count > 0:
            score -= min(0.4, action_count * 0.15)
        
        return min(1.0, max(0.0, score))
    
    def _determine_intent(self, cart_score: float, qa_score: float, message: str) -> Tuple[str, float, str]:
        """Determine the final intent based on scores."""
        
        # If both scores are very low, it's unclear
        if cart_score < 0.3 and qa_score < 0.3:
            return "unclear", max(cart_score, qa_score), f"Both cart ({cart_score:.2f}) and QA ({qa_score:.2f}) scores are low"
        
        # If scores are very close, it's unclear
        score_diff = abs(cart_score - qa_score)
        if score_diff < 0.2 and max(cart_score, qa_score) < 0.8:
            return "unclear", max(cart_score, qa_score), f"Cart and QA scores are too close: {cart_score:.2f} vs {qa_score:.2f}"
        
        # Choose the higher scoring intent
        if cart_score > qa_score:
            return "cart", cart_score, f"Cart score ({cart_score:.2f}) higher than QA score ({qa_score:.2f})"
        else:
            return "qa", qa_score, f"QA score ({qa_score:.2f}) higher than cart score ({cart_score:.2f})"
    
    def _generate_clarification_questions(self, message: str, entities: List[str], 
                                        cart_score: float, qa_score: float) -> List[str]:
        """Generate clarification questions for unclear intents."""
        questions = []
        
        # If both scores are low
        if cart_score < 0.3 and qa_score < 0.3:
            questions.extend([
                "Are you looking for information about a product, or would you like to add something to your cart?",
                "Would you like me to help you find product information or manage your shopping cart?"
            ])
        
        # If scores are close
        elif abs(cart_score - qa_score) < 0.2:
            if entities:
                entity_list = ", ".join(entities[:3])
                questions.extend([
                    f"I see you mentioned {entity_list}. Are you looking for information about these items or would you like to add them to your cart?",
                    f"Would you like me to search for details about {entity_list} or help you with cart management?"
                ])
            else:
                questions.extend([
                    "I'm not sure if you want product information or cart management. Could you clarify?",
                    "Are you looking to learn about products or manage your shopping cart?"
                ])
        
        # Fallback questions
        if not questions:
            questions.extend([
                "Could you please clarify what you'd like me to help you with?",
                "I'm not sure how to help. Are you looking for product information or cart management?"
            ])
        
        return questions[:2]  # Limit to 2 questions
    
    def _create_unclear_result(self, message: str, reasoning: str, questions: List[str]) -> IntentResult:
        """Create an IntentResult for unclear intent."""
        return IntentResult(
            intent="unclear",
            confidence=0.0,
            entities=[],
            clarification_needed=True,
            suggested_questions=questions,
            reasoning=reasoning,
            metadata={
                "cart_score": 0.0,
                "qa_score": 0.0,
                "message_length": len(message) if message else 0,
                "entity_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )