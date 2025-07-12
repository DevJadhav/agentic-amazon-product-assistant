"""
Enhanced Query Processor with Advanced Query Type Detection
Intelligent query processing with automatic query type detection, intent classification, and response optimization.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from langsmith import traceable

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enhanced query types for intelligent processing."""
    PRODUCT_SEARCH = "product_search"
    PRODUCT_COMPARISON = "product_comparison"
    REVIEW_ANALYSIS = "review_analysis"
    RECOMMENDATION = "recommendation"
    PRICE_INQUIRY = "price_inquiry"
    TECHNICAL_SPECS = "technical_specs"
    COMPLAINT_ANALYSIS = "complaint_analysis"
    USE_CASE_INQUIRY = "use_case_inquiry"
    BRAND_COMPARISON = "brand_comparison"
    AVAILABILITY_CHECK = "availability_check"
    GENERAL_QUESTION = "general_question"

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class QueryIntent(Enum):
    """User intent classification."""
    RESEARCH = "research"
    PURCHASE = "purchase"
    TROUBLESHOOT = "troubleshoot"
    COMPARE = "compare"
    LEARN = "learn"

@dataclass
class QueryAnalysis:
    """Comprehensive query analysis results."""
    query_type: QueryType
    complexity: QueryComplexity
    intent: QueryIntent
    confidence: float
    entities: Dict[str, List[str]]
    keywords: List[str]
    price_range: Optional[Tuple[float, float]]
    product_categories: List[str]
    brands: List[str]
    features: List[str]
    sentiment: str
    language_patterns: Dict[str, Any]
    search_strategy: Dict[str, Any]

class EnhancedQueryProcessor:
    """Enhanced query processor with ML-based classification and intent detection."""
    
    def __init__(self):
        """Initialize the enhanced query processor."""
        self.query_patterns = self._load_query_patterns()
        self.product_categories = self._load_product_categories()
        self.brand_names = self._load_brand_names()
        self.feature_keywords = self._load_feature_keywords()
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            # In production, this should be pre-installed
            self.nlp = None
        
        # Initialize ML classifier
        self.query_classifier = self._initialize_classifier()
        
        # Performance tracking
        self.query_stats = defaultdict(int)
        
        logger.info("Enhanced Query Processor initialized")
    
    def _load_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Load regex patterns for query type detection."""
        return {
            QueryType.PRODUCT_SEARCH: [
                r'\b(find|search|show|looking for|need)\b.*\b(product|item|device|gadget)\b',
                r'\bwhat.*\b(best|good|top)\b.*\b(for|in)\b',
                r'\b(where|how) to (buy|find|get)\b',
                r'\b(show me|find me)\b.*\b(products|items)\b'
            ],
            QueryType.PRODUCT_COMPARISON: [
                r'\b(compare|comparison|vs|versus|difference|between)\b',
                r'\b(which is better|what.*difference|how.*different)\b',
                r'\b(pros and cons|advantages|disadvantages)\b',
                r'\b([A-Z][a-z]+) (vs|versus) ([A-Z][a-z]+)\b'
            ],
            QueryType.REVIEW_ANALYSIS: [
                r'\b(reviews?|opinions?|feedback|ratings?)\b',
                r'\bwhat.*people.*say\b',
                r'\b(user experience|customer|buyers?)\b.*\b(think|say|report)\b',
                r'\b(complaints?|issues?|problems?)\b.*\b(with|about)\b'
            ],
            QueryType.RECOMMENDATION: [
                r'\b(recommend|suggest|advice|should i)\b',
                r'\b(best|top|good)\b.*\b(for|under|within)\b',
                r'\bwhat.*\b(buy|get|choose|pick)\b',
                r'\b(help me|need help)\b.*\b(choosing|selecting|finding)\b'
            ],
            QueryType.PRICE_INQUIRY: [
                r'\b(price|cost|expensive|cheap|budget|affordable)\b',
                r'\$\d+|\b\d+\s*(dollars?|bucks?)\b',
                r'\b(under|below|less than|within)\b.*\$',
                r'\b(how much|price range|budget)\b'
            ],
            QueryType.TECHNICAL_SPECS: [
                r'\b(specifications?|specs|features?|technical)\b',
                r'\b(dimensions?|weight|size|capacity)\b',
                r'\b(battery|power|voltage|watts?)\b',
                r'\b(compatibility|compatible|works? with)\b'
            ],
            QueryType.COMPLAINT_ANALYSIS: [
                r'\b(problems?|issues?|complaints?|defects?)\b',
                r'\b(broken|defective|faulty|bad|terrible)\b',
                r'\b(return|refund|warranty|replacement)\b',
                r'\bwhy.*\b(breaking|failing|not working)\b'
            ],
            QueryType.USE_CASE_INQUIRY: [
                r'\b(for|suitable for|good for|use for)\b.*\b(gaming|work|office|home)\b',
                r'\b(portable|travel|outdoor|indoor)\b',
                r'\b(professional|business|personal|family)\b',
                r'\bwhat.*\b(use|purpose|application)\b'
            ],
            QueryType.BRAND_COMPARISON: [
                r'\b(Apple|Samsung|Sony|Dell|HP|Lenovo|ASUS|Microsoft)\b.*\b(vs|versus|compared to)\b',
                r'\b(brand|manufacturer|company)\b.*\b(better|best|comparison)\b',
                r'\bwhich brand\b'
            ],
            QueryType.AVAILABILITY_CHECK: [
                r'\b(available|in stock|out of stock|availability)\b',
                r'\b(where.*buy|find.*store|purchase)\b',
                r'\b(shipping|delivery|when.*available)\b'
            ]
        }
    
    def _load_product_categories(self) -> List[str]:
        """Load product categories for entity extraction."""
        return [
            'laptop', 'computer', 'desktop', 'tablet', 'ipad',
            'phone', 'smartphone', 'iphone', 'android',
            'headphones', 'earbuds', 'speakers', 'audio',
            'monitor', 'display', 'tv', 'television',
            'camera', 'webcam', 'photography',
            'router', 'modem', 'networking', 'wifi',
            'keyboard', 'mouse', 'accessories',
            'charger', 'cable', 'adapter', 'power',
            'gaming', 'console', 'controller',
            'smartwatch', 'fitness', 'tracker',
            'drone', 'rc', 'remote control',
            'home automation', 'smart home', 'alexa', 'google home'
        ]
    
    def _load_brand_names(self) -> List[str]:
        """Load brand names for entity extraction."""
        return [
            'Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'HP', 'Lenovo', 'ASUS',
            'Acer', 'Microsoft', 'Google', 'Amazon', 'Bose', 'JBL', 'Beats',
            'Logitech', 'Razer', 'Corsair', 'SteelSeries', 'HyperX',
            'Canon', 'Nikon', 'GoPro', 'DJI', 'Anker', 'Belkin',
            'Netgear', 'TP-Link', 'Linksys', 'ASUS', 'Xiaomi', 'OnePlus'
        ]
    
    def _load_feature_keywords(self) -> Dict[str, List[str]]:
        """Load feature keywords for extraction."""
        return {
            'connectivity': ['wifi', 'bluetooth', 'usb', 'wireless', 'ethernet', 'nfc'],
            'audio': ['noise canceling', 'bass', 'treble', 'stereo', 'surround', 'microphone'],
            'display': ['4k', '1080p', 'hd', 'oled', 'lcd', 'led', 'touchscreen', 'retina'],
            'performance': ['fast', 'slow', 'responsive', 'lag', 'speed', 'performance'],
            'battery': ['battery life', 'charging', 'power', 'battery', 'charge'],
            'build': ['durable', 'sturdy', 'lightweight', 'portable', 'compact', 'build quality'],
            'gaming': ['gaming', 'fps', 'latency', 'rgb', 'mechanical', 'esports'],
            'design': ['design', 'aesthetic', 'color', 'style', 'appearance', 'look']
        }
    
    def _initialize_classifier(self) -> Optional[Pipeline]:
        """Initialize ML classifier for query type detection."""
        try:
            # Training data for query classification
            training_data = self._get_training_data()
            
            if training_data:
                # Create TF-IDF + Naive Bayes pipeline
                classifier = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        ngram_range=(1, 3),
                        max_features=10000,
                        stop_words='english'
                    )),
                    ('classifier', MultinomialNB(alpha=0.1))
                ])
                
                # Train classifier
                texts, labels = zip(*training_data)
                classifier.fit(texts, labels)
                
                logger.info("Query classifier trained successfully")
                return classifier
            
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
        
        return None
    
    def _get_training_data(self) -> List[Tuple[str, str]]:
        """Get training data for query classification."""
        return [
            # Product Search
            ("find the best wireless headphones", "product_search"),
            ("show me budget laptops", "product_search"),
            ("looking for gaming keyboards", "product_search"),
            
            # Comparison
            ("compare iPhone vs Samsung Galaxy", "product_comparison"),
            ("what's the difference between iPad and tablet", "product_comparison"),
            ("MacBook vs Dell laptop pros and cons", "product_comparison"),
            
            # Reviews
            ("what do people say about AirPods", "review_analysis"),
            ("user reviews for wireless mouse", "review_analysis"),
            ("customer complaints about this phone", "review_analysis"),
            
            # Recommendations
            ("recommend good headphones under $100", "recommendation"),
            ("what should I buy for gaming", "recommendation"),
            ("best laptop for programming", "recommendation"),
            
            # Price
            ("how much does iPhone cost", "price_inquiry"),
            ("cheap alternatives to expensive laptop", "price_inquiry"),
            ("budget phones under $300", "price_inquiry"),
            
            # Technical
            ("specifications of this laptop", "technical_specs"),
            ("battery life of wireless earbuds", "technical_specs"),
            ("compatibility with Mac", "technical_specs"),
            
            # More training examples...
        ]
    
    @traceable
    def analyze_query(self, query: str, context: Optional[Dict] = None) -> QueryAnalysis:
        """Perform comprehensive query analysis."""
        query_lower = query.lower().strip()
        
        # Basic analysis
        query_type = self._detect_query_type(query_lower)
        complexity = self._assess_complexity(query_lower)
        intent = self._classify_intent(query_lower)
        confidence = self._calculate_confidence(query_lower, query_type)
        
        # Entity extraction
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query_lower)
        
        # Specific extractions
        price_range = self._extract_price_range(query_lower)
        product_categories = self._extract_product_categories(query_lower)
        brands = self._extract_brands(query)
        features = self._extract_features(query_lower)
        
        # Sentiment and patterns
        sentiment = self._analyze_sentiment(query_lower)
        language_patterns = self._analyze_language_patterns(query)
        
        # Search strategy
        search_strategy = self._determine_search_strategy(
            query_type, complexity, entities, keywords
        )
        
        # Update statistics
        self.query_stats[query_type.value] += 1
        
        return QueryAnalysis(
            query_type=query_type,
            complexity=complexity,
            intent=intent,
            confidence=confidence,
            entities=entities,
            keywords=keywords,
            price_range=price_range,
            product_categories=product_categories,
            brands=brands,
            features=features,
            sentiment=sentiment,
            language_patterns=language_patterns,
            search_strategy=search_strategy
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type using pattern matching and ML."""
        # Try ML classifier first
        if self.query_classifier:
            try:
                predicted_type = self.query_classifier.predict([query])[0]
                return QueryType(predicted_type)
            except Exception as e:
                logger.debug(f"ML classification failed: {e}")
        
        # Fallback to pattern matching
        type_scores = defaultdict(int)
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    type_scores[query_type] += 1
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return QueryType.GENERAL_QUESTION
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on multiple factors."""
        complexity_score = 0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Multiple entities
        entities = self._extract_entities(query)
        entity_count = sum(len(v) for v in entities.values())
        if entity_count > 5:
            complexity_score += 2
        elif entity_count > 2:
            complexity_score += 1
        
        # Multiple criteria
        criteria_patterns = [
            r'\band\b', r'\bor\b', r'\bbut\b', r'\bwith\b', r'\bunder\b',
            r'\bover\b', r'\bbetween\b', r'\bexcept\b'
        ]
        for pattern in criteria_patterns:
            if re.search(pattern, query):
                complexity_score += 1
        
        # Question complexity
        if re.search(r'\bwhy\b|\bhow\b|\bwhen\b|\bwhere\b', query):
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 5:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify user intent from the query."""
        intent_patterns = {
            QueryIntent.PURCHASE: [
                r'\b(buy|purchase|order|get|need to buy)\b',
                r'\b(where.*buy|how.*buy|best place)\b',
                r'\b(should i buy|worth buying|recommend buying)\b'
            ],
            QueryIntent.RESEARCH: [
                r'\b(compare|research|study|analyze|investigate)\b',
                r'\b(tell me about|learn about|information)\b',
                r'\b(specifications|features|details)\b'
            ],
            QueryIntent.COMPARE: [
                r'\b(vs|versus|compare|comparison|difference)\b',
                r'\b(which is better|what.*different)\b',
                r'\b(pros and cons|advantages)\b'
            ],
            QueryIntent.TROUBLESHOOT: [
                r'\b(problem|issue|fix|repair|broken|not working)\b',
                r'\b(troubleshoot|solve|help with)\b',
                r'\b(why.*not|how to fix)\b'
            ],
            QueryIntent.LEARN: [
                r'\b(how.*work|what is|explain|understand)\b',
                r'\b(tutorial|guide|instructions)\b',
                r'\b(learn|teach|show)\b'
            ]
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        return QueryIntent.RESEARCH  # Default intent
    
    def _calculate_confidence(self, query: str, query_type: QueryType) -> float:
        """Calculate confidence score for query type detection."""
        base_confidence = 0.5
        
        # Pattern matching confidence
        patterns = self.query_patterns.get(query_type, [])
        pattern_matches = sum(1 for pattern in patterns 
                            if re.search(pattern, query, re.IGNORECASE))
        pattern_confidence = min(pattern_matches * 0.2, 0.4)
        
        # ML classifier confidence
        ml_confidence = 0.0
        if self.query_classifier:
            try:
                probabilities = self.query_classifier.predict_proba([query])[0]
                ml_confidence = max(probabilities) * 0.3
            except:
                pass
        
        # Keyword presence confidence
        keyword_confidence = 0.0
        for category, keywords in self.feature_keywords.items():
            if any(keyword in query for keyword in keywords):
                keyword_confidence += 0.05
        
        keyword_confidence = min(keyword_confidence, 0.2)
        
        total_confidence = min(
            base_confidence + pattern_confidence + ml_confidence + keyword_confidence,
            1.0
        )
        
        return round(total_confidence, 2)
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from the query."""
        entities = {
            'products': [],
            'brands': [],
            'features': [],
            'locations': [],
            'money': [],
            'organizations': []
        }
        
        if self.nlp:
            doc = self.nlp(query)
            
            for ent in doc.ents:
                if ent.label_ in ['MONEY']:
                    entities['money'].append(ent.text)
                elif ent.label_ in ['ORG']:
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['locations'].append(ent.text)
                elif ent.label_ in ['PRODUCT']:
                    entities['products'].append(ent.text)
        
        # Custom entity extraction
        entities['brands'].extend(self._extract_brands(query))
        entities['products'].extend(self._extract_product_categories(query))
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w{3,}\b', query.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Add bigrams and trigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 6:  # Meaningful bigrams
                keywords.append(bigram)
        
        return list(set(keywords))[:10]  # Return top 10 unique keywords
    
    def _extract_price_range(self, query: str) -> Optional[Tuple[float, float]]:
        """Extract price range from the query."""
        # Pattern for price ranges
        price_patterns = [
            r'under\s*\$?(\d+(?:\.\d{2})?)',
            r'below\s*\$?(\d+(?:\.\d{2})?)',
            r'less than\s*\$?(\d+(?:\.\d{2})?)',
            r'within\s*\$?(\d+(?:\.\d{2})?)',
            r'budget.*?\$?(\d+(?:\.\d{2})?)',
            r'\$?(\d+(?:\.\d{2})?)\s*to\s*\$?(\d+(?:\.\d{2})?)',
            r'\$?(\d+(?:\.\d{2})?)\s*-\s*\$?(\d+(?:\.\d{2})?)',
            r'between\s*\$?(\d+(?:\.\d{2})?)\s*and\s*\$?(\d+(?:\.\d{2})?)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    # Single price (interpreted as maximum)
                    max_price = float(groups[0])
                    return (0.0, max_price)
                elif len(groups) == 2:
                    # Price range
                    min_price = float(groups[0])
                    max_price = float(groups[1])
                    return (min_price, max_price)
        
        return None
    
    def _extract_product_categories(self, query: str) -> List[str]:
        """Extract product categories from the query."""
        found_categories = []
        
        for category in self.product_categories:
            if re.search(r'\b' + re.escape(category) + r'\b', query, re.IGNORECASE):
                found_categories.append(category)
        
        return found_categories
    
    def _extract_brands(self, query: str) -> List[str]:
        """Extract brand names from the query."""
        found_brands = []
        
        for brand in self.brand_names:
            if re.search(r'\b' + re.escape(brand) + r'\b', query, re.IGNORECASE):
                found_brands.append(brand)
        
        return found_brands
    
    def _extract_features(self, query: str) -> List[str]:
        """Extract product features from the query."""
        found_features = []
        
        for category, features in self.feature_keywords.items():
            for feature in features:
                if re.search(r'\b' + re.escape(feature) + r'\b', query, re.IGNORECASE):
                    found_features.append(feature)
        
        return found_features
    
    def _analyze_sentiment(self, query: str) -> str:
        """Analyze sentiment of the query."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'best', 'love', 'awesome', 'fantastic', 'perfect', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'poor', 'disappointing', 'useless', 'broken']
        
        positive_score = sum(1 for word in positive_words if word in query)
        negative_score = sum(1 for word in negative_words if word in query)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def _analyze_language_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze language patterns in the query."""
        patterns = {
            'has_question_words': bool(re.search(r'\b(what|where|when|why|how|which|who)\b', query, re.IGNORECASE)),
            'has_comparison': bool(re.search(r'\b(vs|versus|compare|better|best|difference)\b', query, re.IGNORECASE)),
            'has_negation': bool(re.search(r'\b(not|no|never|none|without)\b', query, re.IGNORECASE)),
            'has_superlatives': bool(re.search(r'\b(best|worst|most|least|top|bottom)\b', query, re.IGNORECASE)),
            'has_uncertainty': bool(re.search(r'\b(maybe|perhaps|might|could|possibly)\b', query, re.IGNORECASE)),
            'is_urgent': bool(re.search(r'\b(urgent|asap|immediately|quick|fast|now)\b', query, re.IGNORECASE)),
            'sentence_length': len(query.split()),
            'has_technical_terms': len(self._extract_features(query.lower())) > 0
        }
        
        return patterns
    
    def _determine_search_strategy(self, 
                                 query_type: QueryType, 
                                 complexity: QueryComplexity,
                                 entities: Dict[str, List[str]], 
                                 keywords: List[str]) -> Dict[str, Any]:
        """Determine optimal search strategy based on query analysis."""
        strategy = {
            'search_methods': [],
            'result_count': 10,
            'filters': {},
            'ranking_factors': [],
            'response_style': 'detailed'
        }
        
        # Search methods based on query type
        if query_type == QueryType.PRODUCT_SEARCH:
            strategy['search_methods'] = ['semantic_search', 'category_filter']
            strategy['result_count'] = 15
            strategy['ranking_factors'] = ['rating', 'review_count', 'price']
            
        elif query_type == QueryType.PRODUCT_COMPARISON:
            strategy['search_methods'] = ['semantic_search', 'brand_filter']
            strategy['result_count'] = 8
            strategy['ranking_factors'] = ['rating', 'features']
            strategy['response_style'] = 'comparison_table'
            
        elif query_type == QueryType.REVIEW_ANALYSIS:
            strategy['search_methods'] = ['review_search', 'sentiment_analysis']
            strategy['result_count'] = 20
            strategy['ranking_factors'] = ['review_sentiment', 'review_length']
            strategy['response_style'] = 'summary'
            
        elif query_type == QueryType.RECOMMENDATION:
            strategy['search_methods'] = ['semantic_search', 'popularity_boost']
            strategy['result_count'] = 12
            strategy['ranking_factors'] = ['rating', 'popularity', 'price_value']
            
        elif query_type == QueryType.PRICE_INQUIRY:
            strategy['search_methods'] = ['price_filter', 'semantic_search']
            strategy['result_count'] = 10
            strategy['ranking_factors'] = ['price', 'value_rating']
            strategy['response_style'] = 'price_focused'
        
        # Adjust based on complexity
        if complexity == QueryComplexity.COMPLEX:
            strategy['result_count'] = min(strategy['result_count'] * 2, 30)
            strategy['search_methods'].append('hybrid_search')
        
        # Add filters based on entities
        if entities.get('brands'):
            strategy['filters']['brands'] = entities['brands']
        
        if entities.get('products'):
            strategy['filters']['categories'] = entities['products']
        
        # Adjust response style based on patterns
        if len(keywords) > 8:
            strategy['response_style'] = 'comprehensive'
        elif len(keywords) < 3:
            strategy['response_style'] = 'concise'
        
        return strategy
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query processing statistics."""
        total_queries = sum(self.query_stats.values())
        
        return {
            'total_queries_processed': total_queries,
            'query_type_distribution': dict(self.query_stats),
            'most_common_type': max(self.query_stats.items(), key=lambda x: x[1])[0] if self.query_stats else None,
            'classifier_available': self.query_classifier is not None,
            'nlp_model_available': self.nlp is not None
        }
    
    def suggest_query_improvements(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Suggest improvements to make the query more effective."""
        suggestions = []
        
        # Suggest adding price range
        if not analysis.price_range and analysis.query_type in [QueryType.PRODUCT_SEARCH, QueryType.RECOMMENDATION]:
            suggestions.append("Consider adding a price range (e.g., 'under $100') for more targeted results")
        
        # Suggest being more specific
        if analysis.complexity == QueryComplexity.SIMPLE and len(analysis.keywords) < 3:
            suggestions.append("Try being more specific about features or requirements")
        
        # Suggest brand preferences
        if not analysis.brands and analysis.query_type == QueryType.PRODUCT_COMPARISON:
            suggestions.append("Mentioning specific brands can help with comparisons")
        
        # Suggest use case
        if not any(use_case in query.lower() for use_case in ['gaming', 'work', 'home', 'travel', 'professional']):
            suggestions.append("Specifying your use case (gaming, work, etc.) can improve recommendations")
        
        return suggestions[:3]  # Return top 3 suggestions


def create_enhanced_query_processor() -> EnhancedQueryProcessor:
    """Create and initialize the enhanced query processor."""
    return EnhancedQueryProcessor()