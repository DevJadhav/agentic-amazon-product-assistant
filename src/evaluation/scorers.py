"""
Scoring functions for RAG system evaluation using LangSmith.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Set, FrozenSet, Tuple
from dataclasses import dataclass
from functools import lru_cache
from langsmith import traceable

@dataclass
class ScoreResult:
    """Result of a scoring function."""
    score: float
    max_score: float
    details: Dict[str, Any]
    explanation: str
    
    @property
    def normalized_score(self) -> float:
        """Get normalized score between 0 and 1."""
        return self.score / self.max_score if self.max_score > 0 else 0.0

class BaseScorer(ABC):
    """Base class for all scorers with common functionality."""
    
    # Common word sets for efficient matching
    TRANSITION_WORDS: FrozenSet[str] = frozenset({
        'however', 'therefore', 'additionally', 'furthermore', 'while', 
        'although', 'moreover', 'consequently', 'nevertheless', 'thus'
    })
    
    ACTIONABLE_TERMS: FrozenSet[str] = frozenset({
        'recommend', 'suggest', 'consider', 'choose', 'select', 
        'look for', 'avoid', 'try', 'use', 'check'
    })
    
    UNCERTAINTY_INDICATORS: FrozenSet[str] = frozenset({
        'typically', 'usually', 'generally', 'often', 'may', 'might', 
        'can', 'could', 'approximately', 'around', 'about', 'varies',
        'sometimes', 'possibly', 'probably', 'likely'
    })
    
    def __init__(self):
        """Initialize the scorer."""
        self._word_cache = {}
    
    @lru_cache(maxsize=256)
    def _extract_words(self, text: str, min_length: int = 3) -> Set[str]:
        """Extract meaningful words from text."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return {word for word in words if len(word) >= min_length}
    
    @lru_cache(maxsize=256)
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_word_overlap(self, text1: str, text2: str, min_word_length: int = 3) -> float:
        """Calculate word overlap between two texts."""
        words1 = self._extract_words(text1, min_word_length)
        words2 = self._extract_words(text2, min_word_length)
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _count_pattern_matches(self, text: str, patterns: Union[Set[str], FrozenSet[str]]) -> int:
        """Count pattern matches in text."""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        return len(text_words & patterns)
    
    @abstractmethod
    @traceable
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Score the response. Must be implemented by subclasses."""
        pass

class RelevanceScorer(BaseScorer):
    """Scores how relevant the response is to the query."""
    
    @traceable
    def __call__(self, query: str, response: str, expected_topics: List[str]) -> Dict[str, Any]:
        """Score relevance based on topic coverage and query alignment."""
        response_lower = response.lower()
        
        # Check topic coverage efficiently
        topic_details = {}
        topics_covered = 0
        
        for topic in expected_topics:
            topic_lower = topic.lower()
            # Create pattern variations
            topic_patterns = {
                topic_lower,
                topic_lower.replace('_', ' '),
                topic_lower.replace('_', '-')
            }
            
            found = any(pattern in response_lower for pattern in topic_patterns)
            topic_details[topic] = found
            if found:
                topics_covered += 1
        
        topic_coverage_score = topics_covered / len(expected_topics) if expected_topics else 1.0
        
        # Check query term coverage using word sets
        query_words = self._extract_words(query)
        response_words = self._extract_words(response)
        
        query_coverage_score = len(query_words & response_words) / len(query_words) if query_words else 1.0
        
        # Combined relevance score
        relevance_score = (topic_coverage_score * 0.7) + (query_coverage_score * 0.3)
        
        return {
            'relevance_score': relevance_score,
            'topic_coverage': topic_coverage_score,
            'query_coverage': query_coverage_score,
            'topics_covered': topics_covered,
            'total_topics': len(expected_topics),
            'topic_details': topic_details
        }

class AccuracyScorer(BaseScorer):
    """Scores factual accuracy by checking key facts and claims."""
    
    # Fact indicators for efficient matching
    FACT_INDICATORS: FrozenSet[str] = frozenset({
        'hours', 'mbps', 'ghz', 'gb', 'tb', 'inches', 'warranty', 
        'compatible', 'supports', 'includes', 'features', 'battery',
        'speed', 'capacity', 'resolution', 'performance'
    })
    
    @traceable
    def __call__(self, response: str, expected_answer: str, expected_products: List[str]) -> Dict[str, Any]:
        """Score accuracy based on factual claims and product mentions."""
        response_lower = response.lower()
        
        # Extract and compare facts
        expected_facts = self._extract_facts(expected_answer)
        response_facts = self._extract_facts(response)
        
        # Check fact overlap
        fact_matches = 0
        for expected_fact in expected_facts:
            if any(self._calculate_word_overlap(expected_fact, response_fact) > 0.7 
                   for response_fact in response_facts):
                fact_matches += 1
        
        fact_accuracy = fact_matches / len(expected_facts) if expected_facts else 1.0
        
        # Check product mention accuracy
        product_details = {}
        products_mentioned = 0
        
        for product in expected_products:
            product_lower = product.lower().replace('_', ' ')
            mentioned = product_lower in response_lower
            product_details[product] = mentioned
            if mentioned:
                products_mentioned += 1
        
        product_accuracy = products_mentioned / len(expected_products) if expected_products else 1.0
        
        # Combined accuracy score
        accuracy_score = (fact_accuracy * 0.8) + (product_accuracy * 0.2)
        
        return {
            'accuracy_score': accuracy_score,
            'fact_accuracy': fact_accuracy,
            'product_accuracy': product_accuracy,
            'facts_matched': fact_matches,
            'total_facts': len(expected_facts),
            'products_mentioned': products_mentioned,
            'total_products': len(expected_products),
            'product_details': product_details
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text efficiently."""
        sentences = self._extract_sentences(text)
        facts = []
        
        for sentence in sentences:
            if len(sentence) < 10:
                continue
                
            # Look for sentences with numbers or fact indicators
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            if (re.search(r'\d+', sentence) or 
                bool(sentence_words & self.FACT_INDICATORS)):
                facts.append(sentence)
        
        return facts

class CompletenessScorer(BaseScorer):
    """Scores how complete the response is."""
    
    # Query type indicators
    QUERY_TYPE_INDICATORS: Dict[str, FrozenSet[str]] = {
        'product_info': frozenset({'features', 'specifications', 'price', 'performance', 'details', 'description'}),
        'product_reviews': frozenset({'customers', 'reviews', 'ratings', 'feedback', 'experience', 'opinion'}),
        'product_comparison': frozenset({'compare', 'versus', 'difference', 'better', 'advantages', 'disadvantages'}),
        'product_complaints': frozenset({'problems', 'issues', 'complaints', 'negative', 'concerns', 'drawbacks'}),
        'product_recommendation': frozenset({'recommend', 'suggest', 'alternative', 'best', 'consider', 'options'}),
        'use_case': frozenset({'suitable', 'appropriate', 'effective', 'works', 'designed', 'ideal'})
    }
    
    @traceable
    def __call__(self, response: str, expected_answer: str, query_type: str) -> Dict[str, Any]:
        """Score completeness based on response length and content depth."""
        # Length-based completeness
        response_words = response.split()
        expected_words = expected_answer.split()
        
        response_length = len(response_words)
        expected_length = len(expected_words)
        
        length_ratio = min(response_length / expected_length, 1.0) if expected_length > 0 else 1.0
        
        # Content depth based on query type
        depth_score = self._assess_content_depth(response, query_type)
        
        # Structure completeness
        structure_score = self._assess_structure(response)
        
        # Combined completeness score
        completeness_score = (length_ratio * 0.4) + (depth_score * 0.4) + (structure_score * 0.2)
        
        return {
            'completeness_score': completeness_score,
            'length_ratio': length_ratio,
            'depth_score': depth_score,
            'structure_score': structure_score,
            'response_length': response_length,
            'expected_length': expected_length
        }
    
    def _assess_content_depth(self, response: str, query_type: str) -> float:
        """Assess content depth based on query type requirements."""
        # Get indicators for query type
        indicators = self.QUERY_TYPE_INDICATORS.get(
            query_type, 
            self.QUERY_TYPE_INDICATORS['product_info']
        )
        
        # Count indicators present
        indicators_found = self._count_pattern_matches(response, indicators)
        
        return min(indicators_found / len(indicators), 1.0)
    
    def _assess_structure(self, response: str) -> float:
        """Assess response structure and organization."""
        sentences = self._extract_sentences(response)
        paragraphs = response.split('\n\n')
        
        # Good structure metrics
        sentence_score = min(len(sentences) / 5, 1.0)  # Ideal ~5 sentences
        paragraph_score = min(len(paragraphs) / 2, 1.0)  # Ideal ~2 paragraphs
        
        # Check for transitions
        has_transitions = bool(self._count_pattern_matches(response, self.TRANSITION_WORDS))
        transition_score = 1.0 if has_transitions else 0.7
        
        return (sentence_score + paragraph_score + transition_score) / 3

class FactualityScorer(BaseScorer):
    """Scores factual correctness using pattern matching."""
    
    # Contradictory pairs for efficient checking
    CONTRADICTORY_PAIRS: List[Tuple[str, str]] = [
        ('cheap', 'expensive'), ('good', 'bad'), ('fast', 'slow'),
        ('reliable', 'unreliable'), ('works', 'broken'), ('new', 'old'),
        ('easy', 'difficult'), ('simple', 'complex'), ('safe', 'dangerous')
    ]
    
    @traceable
    def __call__(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Score factual correctness by checking for contradictions and false claims."""
        # Check for contradictions
        contradiction_score = self._check_contradictions(response)
        
        # Check factual claims
        factual_claims_score = self._verify_factual_claims(response, expected_answer)
        
        # Check uncertainty handling
        uncertainty_score = self._check_uncertainty_handling(response)
        
        factuality_score = (
            contradiction_score * 0.4 + 
            factual_claims_score * 0.4 + 
            uncertainty_score * 0.2
        )
        
        return {
            'factuality_score': factuality_score,
            'contradiction_score': contradiction_score,
            'factual_claims_score': factual_claims_score,
            'uncertainty_score': uncertainty_score
        }
    
    def _check_contradictions(self, response: str) -> float:
        """Check for internal contradictions in the response."""
        response_lower = response.lower()
        sentences = self._extract_sentences(response_lower)
        contradictions = 0
        
        # Check each contradictory pair
        for word1, word2 in self.CONTRADICTORY_PAIRS:
            if word1 in response_lower and word2 in response_lower:
                # Check if they're in the same sentence
                for sentence in sentences:
                    if word1 in sentence and word2 in sentence:
                        contradictions += 1
                        break
        
        # Lower score for more contradictions
        return max(0, 1.0 - (contradictions * 0.2))
    
    @lru_cache(maxsize=128)
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        numbers = []
        for match in re.findall(r'\d+(?:\.\d+)?', text):
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        return numbers
    
    def _verify_factual_claims(self, response: str, expected_answer: str) -> float:
        """Verify factual claims against expected answer."""
        response_numbers = self._extract_numbers(response)
        expected_numbers = self._extract_numbers(expected_answer)
        
        if not expected_numbers:
            return 1.0  # No specific numbers to verify
        
        if not response_numbers:
            return 0.5  # Response lacks numerical claims when expected
        
        # Check if response numbers are reasonable
        reasonable_numbers = 0
        for resp_num in response_numbers:
            for exp_num in expected_numbers:
                # Within 2x order of magnitude is reasonable
                ratio = resp_num / exp_num if exp_num != 0 else float('inf')
                if 0.1 <= ratio <= 10:
                    reasonable_numbers += 1
                    break
        
        return reasonable_numbers / len(response_numbers)
    
    def _check_uncertainty_handling(self, response: str) -> float:
        """Check if response appropriately handles uncertainty."""
        uncertainty_count = self._count_pattern_matches(response, self.UNCERTAINTY_INDICATORS)
        
        # Appropriate level of uncertainty (not too absolute, not too uncertain)
        # Ideal is 1-3 uncertainty indicators
        if uncertainty_count == 0:
            return 0.5  # Too absolute
        elif uncertainty_count <= 3:
            return 1.0  # Good balance
        else:
            return max(0.5, 1.0 - (uncertainty_count - 3) * 0.1)  # Too uncertain

class ResponseQualityScorer(BaseScorer):
    """Overall response quality scorer."""
    
    @traceable
    def __call__(self, response: str, query: str) -> Dict[str, Any]:
        """Score overall response quality including clarity and helpfulness."""
        # Clarity score
        clarity_score = self._assess_clarity(response)
        
        # Helpfulness score
        helpfulness_score = self._assess_helpfulness(response, query)
        
        # Coherence score
        coherence_score = self._assess_coherence(response)
        
        # Combined quality score
        quality_score = (
            clarity_score * 0.4 + 
            helpfulness_score * 0.4 + 
            coherence_score * 0.2
        )
        
        return {
            'quality_score': quality_score,
            'clarity_score': clarity_score,
            'helpfulness_score': helpfulness_score,
            'coherence_score': coherence_score
        }
    
    def _assess_clarity(self, response: str) -> float:
        """Assess response clarity."""
        sentences = self._extract_sentences(response)
        
        if not sentences:
            return 0.5
        
        # Check average sentence length
        total_words = sum(len(s.split()) for s in sentences)
        avg_sentence_length = total_words / len(sentences)
        
        # Ideal sentence length is 15-25 words
        if 15 <= avg_sentence_length <= 25:
            length_score = 1.0
        else:
            deviation = abs(avg_sentence_length - 20) / 20
            length_score = max(0, 1.0 - deviation)
        
        # Check for transitions (indicates clear structure)
        has_transitions = bool(self._count_pattern_matches(response, self.TRANSITION_WORDS))
        structure_score = 1.0 if has_transitions else 0.7
        
        return (length_score + structure_score) / 2
    
    def _assess_helpfulness(self, response: str, query: str) -> float:
        """Assess how helpful the response is."""
        # Check for actionable information
        actionable_count = self._count_pattern_matches(response, self.ACTIONABLE_TERMS)
        actionable_score = min(actionable_count / 2, 1.0)  # 2 actionable terms is ideal
        
        # Check for specific details
        response_lower = response.lower()
        has_specifics = bool(re.search(r'\d+|specific|model|brand|type', response_lower))
        specifics_score = 1.0 if has_specifics else 0.7
        
        # Check query-response alignment
        word_overlap = self._calculate_word_overlap(query, response, min_word_length=4)
        
        return (actionable_score + specifics_score + word_overlap) / 3
    
    def _assess_coherence(self, response: str) -> float:
        """Assess response coherence and flow."""
        sentences = self._extract_sentences(response)
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by default
        
        # Check for transitions
        transition_count = self._count_pattern_matches(response, self.TRANSITION_WORDS)
        transition_score = min(transition_count / max(len(sentences) - 1, 1), 1.0)
        
        # Check for topic consistency (repeated key terms)
        all_words = self._extract_words(response, min_length=5)
        word_freq = {}
        
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Count words that appear multiple times
        repeated_terms = sum(1 for count in word_freq.values() if count > 1)
        consistency_score = min(repeated_terms / 5, 1.0)  # 5 repeated terms is good
        
        return (transition_score + consistency_score) / 2

class CompositeScorer:
    """Combines multiple scorers for comprehensive evaluation."""
    
    def __init__(self):
        """Initialize all component scorers."""
        self.relevance_scorer = RelevanceScorer()
        self.accuracy_scorer = AccuracyScorer()
        self.completeness_scorer = CompletenessScorer()
        self.factuality_scorer = FactualityScorer()
        self.quality_scorer = ResponseQualityScorer()
    
    @traceable
    def score_all(
        self, 
        query: str, 
        response: str, 
        expected_answer: str,
        expected_topics: List[str],
        expected_products: List[str],
        query_type: str
    ) -> Dict[str, Any]:
        """Run all scorers and combine results."""
        # Run individual scorers
        relevance_result = self.relevance_scorer(query, response, expected_topics)
        accuracy_result = self.accuracy_scorer(response, expected_answer, expected_products)
        completeness_result = self.completeness_scorer(response, expected_answer, query_type)
        factuality_result = self.factuality_scorer(response, expected_answer)
        quality_result = self.quality_scorer(response, query)
        
        # Calculate overall score
        overall_score = (
            relevance_result['relevance_score'] * 0.25 +
            accuracy_result['accuracy_score'] * 0.25 +
            completeness_result['completeness_score'] * 0.20 +
            factuality_result['factuality_score'] * 0.15 +
            quality_result['quality_score'] * 0.15
        )
        
        return {
            'overall_score': overall_score,
            'relevance': relevance_result,
            'accuracy': accuracy_result,
            'completeness': completeness_result,
            'factuality': factuality_result,
            'quality': quality_result
        }