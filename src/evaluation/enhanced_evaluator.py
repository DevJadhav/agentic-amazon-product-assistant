"""
Enhanced RAG Evaluation Framework with 5 Core Metrics and 14 Test Examples
Comprehensive evaluation system with Langsmith integration for production-ready RAG assessment.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict
import statistics

from langsmith import traceable, Client as LangsmithClient
from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMetrics:
    """Enhanced evaluation metrics with detailed scoring."""
    # Core 5 metrics
    relevance: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    factuality: float = 0.0
    coherence: float = 0.0
    
    # Additional metrics
    response_time: float = 0.0
    context_utilization: float = 0.0
    user_satisfaction: float = 0.0
    
    # Overall scores
    overall_score: float = 0.0
    weighted_score: float = 0.0
    
    # Detailed breakdown
    detailed_scores: Dict[str, Any] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class TestExample:
    """Enhanced test example with comprehensive metadata."""
    id: str
    query: str
    expected_response: str
    context_requirements: List[str]
    difficulty_level: str  # easy, medium, hard
    query_type: str
    category: str
    evaluation_criteria: Dict[str, float]
    ground_truth_products: List[str] = field(default_factory=list)
    ground_truth_facts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedScorer:
    """Enhanced scorer with multiple evaluation dimensions."""
    
    def __init__(self):
        """Initialize the enhanced scorer."""
        self.metric_weights = {
            'relevance': 0.25,
            'accuracy': 0.25,
            'completeness': 0.20,
            'factuality': 0.20,
            'coherence': 0.10
        }
        
        # Initialize sub-scorers
        self.scorers = {
            'relevance': RelevanceScorer(),
            'accuracy': AccuracyScorer(), 
            'completeness': CompletenessScorer(),
            'factuality': FactualityScorer(),
            'coherence': CoherenceScorer()
        }
    
    @traceable
    def evaluate_response(self, 
                         query: str,
                         response: str,
                         expected_response: str,
                         context: Dict[str, Any],
                         test_example: TestExample) -> EnhancedMetrics:
        """Evaluate response using all 5 core metrics."""
        
        metrics = EnhancedMetrics()
        detailed_scores = {}
        
        # Core metric evaluation
        for metric_name, scorer in self.scorers.items():
            try:
                score, details = scorer.evaluate(
                    query=query,
                    response=response,
                    expected_response=expected_response,
                    context=context,
                    test_example=test_example
                )
                
                setattr(metrics, metric_name, score)
                detailed_scores[metric_name] = details
                
            except Exception as e:
                logger.error(f"Error evaluating {metric_name}: {e}")
                setattr(metrics, metric_name, 0.0)
                detailed_scores[metric_name] = {'error': str(e)}
        
        # Calculate additional metrics
        metrics.context_utilization = self._evaluate_context_utilization(response, context)
        metrics.user_satisfaction = self._estimate_user_satisfaction(metrics)
        
        # Calculate overall scores
        metrics.overall_score = self._calculate_overall_score(metrics)
        metrics.weighted_score = self._calculate_weighted_score(metrics)
        
        # Store detailed information
        metrics.detailed_scores = detailed_scores
        metrics.error_analysis = self._analyze_errors(metrics, test_example)
        metrics.improvement_suggestions = self._generate_improvement_suggestions(metrics, test_example)
        
        return metrics
    
    def _evaluate_context_utilization(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate how well the response utilizes provided context."""
        if not context or 'documents' not in context:
            return 0.0
        
        documents = context.get('documents', [])
        if not documents:
            return 0.0
        
        # Check for context overlap
        context_text = ' '.join(documents).lower()
        response_text = response.lower()
        
        # Simple overlap calculation
        context_words = set(context_text.split())
        response_words = set(response_text.split())
        
        overlap = len(context_words.intersection(response_words))
        total_context_words = len(context_words)
        
        if total_context_words == 0:
            return 0.0
        
        utilization_score = min(overlap / total_context_words, 1.0)
        return round(utilization_score, 2)
    
    def _estimate_user_satisfaction(self, metrics: EnhancedMetrics) -> float:
        """Estimate user satisfaction based on multiple factors."""
        # Weighted combination of metrics that impact user satisfaction
        satisfaction_factors = {
            'relevance': 0.3,
            'accuracy': 0.25,
            'completeness': 0.2,
            'coherence': 0.15,
            'response_time_factor': 0.1
        }
        
        # Response time factor (penalize slow responses)
        time_factor = max(0, 1.0 - (metrics.response_time - 2.0) / 10.0) if metrics.response_time > 2.0 else 1.0
        
        satisfaction = (
            metrics.relevance * satisfaction_factors['relevance'] +
            metrics.accuracy * satisfaction_factors['accuracy'] +
            metrics.completeness * satisfaction_factors['completeness'] +
            metrics.coherence * satisfaction_factors['coherence'] +
            time_factor * satisfaction_factors['response_time_factor']
        )
        
        return round(satisfaction, 2)
    
    def _calculate_overall_score(self, metrics: EnhancedMetrics) -> float:
        """Calculate overall score as simple average of core metrics."""
        core_metrics = [
            metrics.relevance,
            metrics.accuracy,
            metrics.completeness,
            metrics.factuality,
            metrics.coherence
        ]
        
        return round(sum(core_metrics) / len(core_metrics), 2)
    
    def _calculate_weighted_score(self, metrics: EnhancedMetrics) -> float:
        """Calculate weighted score based on metric importance."""
        weighted_sum = (
            metrics.relevance * self.metric_weights['relevance'] +
            metrics.accuracy * self.metric_weights['accuracy'] +
            metrics.completeness * self.metric_weights['completeness'] +
            metrics.factuality * self.metric_weights['factuality'] +
            metrics.coherence * self.metric_weights['coherence']
        )
        
        return round(weighted_sum, 2)
    
    def _analyze_errors(self, metrics: EnhancedMetrics, test_example: TestExample) -> Dict[str, Any]:
        """Analyze errors and failure patterns."""
        errors = {}
        
        # Identify low-performing metrics
        low_threshold = 0.6
        for metric_name in ['relevance', 'accuracy', 'completeness', 'factuality', 'coherence']:
            score = getattr(metrics, metric_name)
            if score < low_threshold:
                errors[f'low_{metric_name}'] = {
                    'score': score,
                    'threshold': low_threshold,
                    'severity': 'high' if score < 0.4 else 'medium'
                }
        
        # Overall performance issues
        if metrics.overall_score < 0.7:
            errors['overall_performance'] = {
                'score': metrics.overall_score,
                'issue': 'Multiple metrics below acceptable threshold'
            }
        
        # Response time issues
        if metrics.response_time > 5.0:
            errors['response_time'] = {
                'time': metrics.response_time,
                'issue': 'Response time exceeds acceptable limit'
            }
        
        return errors
    
    def _generate_improvement_suggestions(self, metrics: EnhancedMetrics, test_example: TestExample) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        if metrics.relevance < 0.7:
            suggestions.append("Improve semantic search relevance by enhancing embedding quality or search algorithms")
        
        if metrics.accuracy < 0.7:
            suggestions.append("Enhance fact-checking mechanisms and improve source reliability")
        
        if metrics.completeness < 0.7:
            suggestions.append("Expand search results and improve information synthesis")
        
        if metrics.factuality < 0.7:
            suggestions.append("Implement stronger fact verification and cross-referencing")
        
        if metrics.coherence < 0.7:
            suggestions.append("Improve response structure and logical flow")
        
        if metrics.context_utilization < 0.5:
            suggestions.append("Better integrate retrieved context into response generation")
        
        if metrics.response_time > 3.0:
            suggestions.append("Optimize search and generation pipeline for faster responses")
        
        return suggestions

class RelevanceScorer:
    """Scorer for response relevance to the query."""
    
    def evaluate(self, query: str, response: str, expected_response: str, context: Dict, test_example: TestExample) -> Tuple[float, Dict]:
        """Evaluate relevance score."""
        # Simple keyword overlap approach (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        response_words -= stop_words
        
        if not query_words:
            return 0.0, {'error': 'No meaningful query words'}
        
        overlap = len(query_words.intersection(response_words))
        relevance_score = overlap / len(query_words)
        
        # Bonus for addressing query intent
        intent_bonus = 0.0
        if any(word in response.lower() for word in ['recommend', 'suggest', 'best', 'compare']):
            intent_bonus = 0.1
        
        final_score = min(relevance_score + intent_bonus, 1.0)
        
        details = {
            'query_words': list(query_words),
            'overlap_count': overlap,
            'relevance_ratio': relevance_score,
            'intent_bonus': intent_bonus,
            'final_score': final_score
        }
        
        return round(final_score, 2), details

class AccuracyScorer:
    """Scorer for factual accuracy of the response."""
    
    def evaluate(self, query: str, response: str, expected_response: str, context: Dict, test_example: TestExample) -> Tuple[float, Dict]:
        """Evaluate accuracy score."""
        # Check against ground truth facts if available
        ground_truth_facts = test_example.ground_truth_facts
        
        if not ground_truth_facts:
            # Fallback to context-based accuracy
            return self._evaluate_context_accuracy(response, context)
        
        # Check how many ground truth facts are correctly mentioned
        correct_facts = 0
        total_facts = len(ground_truth_facts)
        
        for fact in ground_truth_facts:
            if fact.lower() in response.lower():
                correct_facts += 1
        
        accuracy_score = correct_facts / total_facts if total_facts > 0 else 0.0
        
        details = {
            'ground_truth_facts': ground_truth_facts,
            'correct_facts': correct_facts,
            'total_facts': total_facts,
            'accuracy_ratio': accuracy_score
        }
        
        return round(accuracy_score, 2), details
    
    def _evaluate_context_accuracy(self, response: str, context: Dict) -> Tuple[float, Dict]:
        """Evaluate accuracy based on context consistency."""
        if not context or 'documents' not in context:
            return 0.5, {'note': 'No context available for accuracy check'}
        
        # Simple consistency check - response should not contradict context
        # This is a simplified implementation
        consistency_score = 0.8  # Default assumption of consistency
        
        details = {
            'method': 'context_consistency',
            'score': consistency_score,
            'note': 'Simplified consistency evaluation'
        }
        
        return consistency_score, details

class CompletenessScorer:
    """Scorer for response completeness."""
    
    def evaluate(self, query: str, response: str, expected_response: str, context: Dict, test_example: TestExample) -> Tuple[float, Dict]:
        """Evaluate completeness score."""
        # Check against context requirements
        context_requirements = test_example.context_requirements
        
        addressed_requirements = 0
        for requirement in context_requirements:
            if any(keyword in response.lower() for keyword in requirement.lower().split()):
                addressed_requirements += 1
        
        completeness_score = addressed_requirements / len(context_requirements) if context_requirements else 0.5
        
        # Length factor - very short responses are likely incomplete
        word_count = len(response.split())
        length_factor = min(word_count / 50, 1.0) if word_count < 50 else 1.0
        
        final_score = completeness_score * length_factor
        
        details = {
            'context_requirements': context_requirements,
            'addressed_requirements': addressed_requirements,
            'completeness_ratio': completeness_score,
            'word_count': word_count,
            'length_factor': length_factor,
            'final_score': final_score
        }
        
        return round(final_score, 2), details

class FactualityScorer:
    """Scorer for factual correctness."""
    
    def evaluate(self, query: str, response: str, expected_response: str, context: Dict, test_example: TestExample) -> Tuple[float, Dict]:
        """Evaluate factuality score."""
        # Check for common factual errors
        factual_score = 1.0
        issues = []
        
        # Check for contradictions
        contradiction_patterns = [
            (r'not.*available', r'available'),
            (r'does.*not.*work', r'works'),
            (r'no.*support', r'supports')
        ]
        
        for negative_pattern, positive_pattern in contradiction_patterns:
            import re
            if re.search(negative_pattern, response.lower()) and re.search(positive_pattern, response.lower()):
                factual_score -= 0.2
                issues.append(f"Potential contradiction: {negative_pattern} vs {positive_pattern}")
        
        # Check against known product facts if available
        if test_example.ground_truth_products:
            mentioned_products = []
            for product in test_example.ground_truth_products:
                if product.lower() in response.lower():
                    mentioned_products.append(product)
            
            if mentioned_products:
                factual_score = min(factual_score + 0.1, 1.0)  # Bonus for mentioning relevant products
        
        details = {
            'base_score': 1.0,
            'issues': issues,
            'mentioned_products': mentioned_products if test_example.ground_truth_products else [],
            'final_score': max(factual_score, 0.0)
        }
        
        return round(max(factual_score, 0.0), 2), details

class CoherenceScorer:
    """Scorer for response coherence and readability."""
    
    def evaluate(self, query: str, response: str, expected_response: str, context: Dict, test_example: TestExample) -> Tuple[float, Dict]:
        """Evaluate coherence score."""
        coherence_score = 1.0
        issues = []
        
        # Check for basic structure
        sentences = response.split('.')
        if len(sentences) < 2:
            coherence_score -= 0.2
            issues.append("Response too short or lacks sentence structure")
        
        # Check for repetition
        words = response.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        max_repetition = max(word_freq.values()) if word_freq else 1
        if max_repetition > 3:
            coherence_score -= 0.1
            issues.append(f"Excessive word repetition (max: {max_repetition})")
        
        # Check for logical flow markers
        flow_markers = ['first', 'second', 'also', 'additionally', 'however', 'therefore', 'furthermore']
        has_flow_markers = any(marker in response.lower() for marker in flow_markers)
        if has_flow_markers and len(sentences) > 3:
            coherence_score = min(coherence_score + 0.1, 1.0)
        
        details = {
            'sentence_count': len(sentences),
            'max_word_repetition': max_repetition,
            'has_flow_markers': has_flow_markers,
            'issues': issues,
            'final_score': max(coherence_score, 0.0)
        }
        
        return round(max(coherence_score, 0.0), 2), details

class EnhancedEvaluationFramework:
    """Enhanced evaluation framework with comprehensive testing."""
    
    def __init__(self, langsmith_client: Optional[LangsmithClient] = None):
        """Initialize the enhanced evaluation framework."""
        self.langsmith_client = langsmith_client
        self.scorer = EnhancedScorer()
        self.test_examples = self._create_comprehensive_test_suite()
        self.evaluation_history = []
        
        logger.info(f"Enhanced Evaluation Framework initialized with {len(self.test_examples)} test examples")
    
    def _create_comprehensive_test_suite(self) -> List[TestExample]:
        """Create comprehensive test suite with 14 examples covering all scenarios."""
        
        examples = [
            # Easy Examples (1-5)
            TestExample(
                id="easy_001",
                query="What are the best wireless headphones?",
                expected_response="Based on customer reviews and ratings, some of the best wireless headphones include...",
                context_requirements=["product names", "ratings", "key features"],
                difficulty_level="easy",
                query_type="recommendation",
                category="audio",
                evaluation_criteria={"relevance": 0.8, "completeness": 0.7},
                ground_truth_products=["Sony WH-1000XM4", "Bose QuietComfort", "Apple AirPods"],
                ground_truth_facts=["noise cancellation", "battery life", "sound quality"]
            ),
            
            TestExample(
                id="easy_002", 
                query="Show me budget laptops under $500",
                expected_response="Here are some budget-friendly laptops under $500...",
                context_requirements=["price range", "laptop specifications", "value assessment"],
                difficulty_level="easy",
                query_type="product_search",
                category="computers",
                evaluation_criteria={"relevance": 0.8, "accuracy": 0.8},
                ground_truth_facts=["under $500", "budget-friendly", "basic specifications"]
            ),
            
            TestExample(
                id="easy_003",
                query="iPhone charger cable reviews",
                expected_response="Customer reviews for iPhone charger cables show...",
                context_requirements=["review summaries", "common complaints", "recommendations"],
                difficulty_level="easy",
                query_type="review_analysis",
                category="accessories",
                evaluation_criteria={"relevance": 0.8, "factuality": 0.8}
            ),
            
            TestExample(
                id="easy_004",
                query="Gaming mouse with RGB lighting",
                expected_response="Gaming mice with RGB lighting options include...",
                context_requirements=["gaming features", "RGB capability", "product options"],
                difficulty_level="easy",
                query_type="product_search", 
                category="gaming",
                evaluation_criteria={"relevance": 0.8, "completeness": 0.7}
            ),
            
            TestExample(
                id="easy_005",
                query="Bluetooth speaker for outdoor use",
                expected_response="For outdoor use, these Bluetooth speakers offer...",
                context_requirements=["outdoor features", "durability", "battery life"],
                difficulty_level="easy",
                query_type="use_case_inquiry",
                category="audio",
                evaluation_criteria={"relevance": 0.8, "accuracy": 0.7}
            ),
            
            # Medium Examples (6-10)
            TestExample(
                id="medium_006",
                query="Compare MacBook Air vs Dell XPS 13 for programming",
                expected_response="For programming, here's how MacBook Air compares to Dell XPS 13...",
                context_requirements=["detailed comparison", "programming-specific features", "performance metrics"],
                difficulty_level="medium",
                query_type="product_comparison",
                category="computers",
                evaluation_criteria={"relevance": 0.8, "completeness": 0.8, "accuracy": 0.8},
                ground_truth_products=["MacBook Air", "Dell XPS 13"],
                ground_truth_facts=["programming performance", "development tools", "keyboard quality"]
            ),
            
            TestExample(
                id="medium_007",
                query="What do people complain about with wireless earbuds under $100?",
                expected_response="Common complaints about budget wireless earbuds include...",
                context_requirements=["complaint analysis", "price-specific issues", "frequency of problems"],
                difficulty_level="medium",
                query_type="complaint_analysis",
                category="audio",
                evaluation_criteria={"relevance": 0.8, "factuality": 0.8, "completeness": 0.7}
            ),
            
            TestExample(
                id="medium_008",
                query="Best 4K monitor for both gaming and professional work under $600",
                expected_response="For both gaming and professional work under $600, these 4K monitors...",
                context_requirements=["dual-purpose features", "gaming specifications", "professional features", "price constraint"],
                difficulty_level="medium",
                query_type="recommendation",
                category="displays",
                evaluation_criteria={"relevance": 0.8, "completeness": 0.8, "accuracy": 0.7}
            ),
            
            TestExample(
                id="medium_009",
                query="Smartphone camera comparison: iPhone 14 vs Samsung Galaxy S23 vs Pixel 7",
                expected_response="Comparing smartphone cameras across iPhone 14, Samsung Galaxy S23, and Pixel 7...",
                context_requirements=["multi-way comparison", "camera specifications", "image quality analysis"],
                difficulty_level="medium",
                query_type="brand_comparison",
                category="smartphones",
                evaluation_criteria={"relevance": 0.8, "completeness": 0.8, "coherence": 0.8}
            ),
            
            TestExample(
                id="medium_010",
                query="Mechanical keyboard recommendations for typing and coding with quiet switches",
                expected_response="For typing and coding with quiet operation, these mechanical keyboards...",
                context_requirements=["switch types", "noise levels", "typing experience", "coding suitability"],
                difficulty_level="medium",
                query_type="recommendation",
                category="accessories",
                evaluation_criteria={"relevance": 0.8, "completeness": 0.7, "accuracy": 0.8}
            ),
            
            # Hard Examples (11-14)
            TestExample(
                id="hard_011",
                query="Comprehensive analysis of smart home ecosystems: Amazon Alexa vs Google Home vs Apple HomeKit compatibility, privacy concerns, and long-term value for a tech-savvy family",
                expected_response="A comprehensive analysis of smart home ecosystems reveals...",
                context_requirements=["ecosystem comparison", "compatibility analysis", "privacy evaluation", "family considerations", "long-term assessment"],
                difficulty_level="hard",
                query_type="comparison",
                category="smart_home",
                evaluation_criteria={"relevance": 0.8, "completeness": 0.9, "accuracy": 0.8, "coherence": 0.8}
            ),
            
            TestExample(
                id="hard_012",
                query="Enterprise-grade laptop recommendations for software development teams with specific requirements: 32GB RAM, dedicated GPU, excellent Linux compatibility, enterprise support, and budget considerations between $2000-4000 per unit",
                expected_response="For enterprise software development teams with your specific requirements...",
                context_requirements=["enterprise features", "technical specifications", "Linux compatibility", "support options", "budget analysis", "team considerations"],
                difficulty_level="hard",
                query_type="recommendation",
                category="enterprise",
                evaluation_criteria={"relevance": 0.9, "completeness": 0.9, "accuracy": 0.9, "factuality": 0.8}
            ),
            
            TestExample(
                id="hard_013",
                query="Analyze the reliability and common failure patterns of popular wireless routers over 2-3 years of use, including warranty claims, firmware update frequency, and user satisfaction trends",
                expected_response="Analysis of wireless router reliability patterns shows...",
                context_requirements=["reliability data", "failure patterns", "warranty analysis", "firmware tracking", "satisfaction trends", "temporal analysis"],
                difficulty_level="hard",
                query_type="analysis",
                category="networking",
                evaluation_criteria={"relevance": 0.8, "factuality": 0.9, "completeness": 0.9, "coherence": 0.8}
            ),
            
            TestExample(
                id="hard_014",
                query="Multi-criteria decision analysis for choosing between different electric vehicle charging solutions for home installation, considering cost, charging speed, future compatibility, installation requirements, and utility integration",
                expected_response="A multi-criteria analysis of home EV charging solutions considering your requirements...",
                context_requirements=["cost analysis", "charging specifications", "compatibility assessment", "installation requirements", "utility considerations", "decision framework"],
                difficulty_level="hard",
                query_type="decision_analysis",
                category="automotive",
                evaluation_criteria={"relevance": 0.9, "completeness": 0.9, "accuracy": 0.8, "coherence": 0.9}
            )
        ]
        
        return examples
    
    @traceable
    async def evaluate_rag_system(self, 
                                 rag_system_callable: Callable,
                                 test_subset: Optional[List[str]] = None,
                                 detailed_output: bool = True) -> Dict[str, Any]:
        """Evaluate RAG system with comprehensive metrics."""
        
        # Select test examples
        if test_subset:
            examples_to_test = [ex for ex in self.test_examples if ex.id in test_subset]
        else:
            examples_to_test = self.test_examples
        
        logger.info(f"Evaluating RAG system with {len(examples_to_test)} test examples")
        
        # Run evaluations
        results = []
        start_time = time.time()
        
        for example in examples_to_test:
            try:
                # Generate response
                response_start = time.time()
                response_data = await self._generate_response(rag_system_callable, example.query)
                response_time = time.time() - response_start
                
                # Extract response and context
                response = response_data.get('response', '')
                context = response_data.get('context', {})
                
                # Evaluate response
                metrics = self.scorer.evaluate_response(
                    query=example.query,
                    response=response,
                    expected_response=example.expected_response,
                    context=context,
                    test_example=example
                )
                
                metrics.response_time = response_time
                
                # Store result
                result = {
                    'example_id': example.id,
                    'query': example.query,
                    'response': response,
                    'metrics': metrics,
                    'difficulty': example.difficulty_level,
                    'category': example.category,
                    'query_type': example.query_type
                }
                
                results.append(result)
                
                logger.info(f"Evaluated {example.id}: Overall={metrics.overall_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {example.id}: {e}")
                results.append({
                    'example_id': example.id,
                    'error': str(e),
                    'metrics': EnhancedMetrics()
                })
        
        # Aggregate results
        total_time = time.time() - start_time
        aggregated_results = self._aggregate_results(results, total_time)
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': aggregated_results,
            'test_count': len(examples_to_test)
        })
        
        return aggregated_results
    
    async def _generate_response(self, rag_system_callable: Callable, query: str) -> Dict[str, Any]:
        """Generate response using the RAG system."""
        if asyncio.iscoroutinefunction(rag_system_callable):
            return await rag_system_callable(query)
        else:
            return rag_system_callable(query)
    
    def _aggregate_results(self, results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Aggregate evaluation results into comprehensive summary."""
        
        # Filter successful results
        successful_results = [r for r in results if 'error' not in r]
        failed_count = len(results) - len(successful_results)
        
        if not successful_results:
            return {
                'summary': {'total_failures': failed_count},
                'error': 'All evaluations failed'
            }
        
        # Extract metrics
        all_metrics = [r['metrics'] for r in successful_results]
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'relevance': np.mean([m.relevance for m in all_metrics]),
            'accuracy': np.mean([m.accuracy for m in all_metrics]),
            'completeness': np.mean([m.completeness for m in all_metrics]),
            'factuality': np.mean([m.factuality for m in all_metrics]),
            'coherence': np.mean([m.coherence for m in all_metrics]),
            'overall_score': np.mean([m.overall_score for m in all_metrics]),
            'weighted_score': np.mean([m.weighted_score for m in all_metrics]),
            'avg_response_time': np.mean([m.response_time for m in all_metrics]),
            'user_satisfaction': np.mean([m.user_satisfaction for m in all_metrics])
        }
        
        # Performance by difficulty
        difficulty_performance = {}
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_results = [r for r in successful_results if r['difficulty'] == difficulty]
            if difficulty_results:
                difficulty_metrics = [r['metrics'] for r in difficulty_results]
                difficulty_performance[difficulty] = {
                    'count': len(difficulty_results),
                    'overall_score': np.mean([m.overall_score for m in difficulty_metrics]),
                    'success_rate': len(difficulty_results) / len([r for r in results if r.get('difficulty') == difficulty])
                }
        
        # Performance by category
        category_performance = {}
        categories = set(r['category'] for r in successful_results)
        for category in categories:
            category_results = [r for r in successful_results if r['category'] == category]
            category_metrics = [r['metrics'] for r in category_results]
            category_performance[category] = {
                'count': len(category_results),
                'overall_score': np.mean([m.overall_score for m in category_metrics])
            }
        
        # Identify top and bottom performers
        sorted_results = sorted(successful_results, key=lambda x: x['metrics'].overall_score, reverse=True)
        
        # Generate insights
        insights = self._generate_insights(aggregate_metrics, difficulty_performance, category_performance)
        
        return {
            'summary': {
                'total_tests': len(results),
                'successful_tests': len(successful_results),
                'failed_tests': failed_count,
                'success_rate': len(successful_results) / len(results),
                'total_evaluation_time': total_time,
                'avg_response_time': aggregate_metrics['avg_response_time']
            },
            'aggregate_metrics': aggregate_metrics,
            'performance_by_difficulty': difficulty_performance,
            'performance_by_category': category_performance,
            'top_performers': [
                {'id': r['example_id'], 'score': r['metrics'].overall_score, 'query': r['query'][:50] + '...'}
                for r in sorted_results[:3]
            ],
            'bottom_performers': [
                {'id': r['example_id'], 'score': r['metrics'].overall_score, 'query': r['query'][:50] + '...'}
                for r in sorted_results[-3:]
            ],
            'insights': insights,
            'detailed_results': [
                {
                    'id': r['example_id'],
                    'query': r['query'],
                    'overall_score': r['metrics'].overall_score,
                    'metrics_breakdown': {
                        'relevance': r['metrics'].relevance,
                        'accuracy': r['metrics'].accuracy,
                        'completeness': r['metrics'].completeness,
                        'factuality': r['metrics'].factuality,
                        'coherence': r['metrics'].coherence
                    },
                    'improvement_suggestions': r['metrics'].improvement_suggestions
                }
                for r in successful_results
            ] if len(successful_results) <= 20 else []  # Include detailed results only for smaller test sets
        }
    
    def _generate_insights(self, 
                          aggregate_metrics: Dict, 
                          difficulty_performance: Dict, 
                          category_performance: Dict) -> List[str]:
        """Generate actionable insights from evaluation results."""
        insights = []
        
        # Overall performance insights
        overall_score = aggregate_metrics['overall_score']
        if overall_score >= 0.8:
            insights.append("✅ Excellent overall performance - system is production-ready")
        elif overall_score >= 0.7:
            insights.append("⚠️ Good performance with room for improvement")
        else:
            insights.append("❌ Performance below acceptable threshold - significant improvements needed")
        
        # Metric-specific insights
        weakest_metric = min(
            [(k, v) for k, v in aggregate_metrics.items() 
             if k in ['relevance', 'accuracy', 'completeness', 'factuality', 'coherence']],
            key=lambda x: x[1]
        )
        
        insights.append(f"🎯 Focus area: {weakest_metric[0]} (score: {weakest_metric[1]:.2f})")
        
        # Difficulty-based insights
        if 'hard' in difficulty_performance and 'easy' in difficulty_performance:
            hard_score = difficulty_performance['hard']['overall_score']
            easy_score = difficulty_performance['easy']['overall_score']
            score_drop = easy_score - hard_score
            
            if score_drop > 0.2:
                insights.append(f"📉 Significant performance drop on complex queries ({score_drop:.2f})")
            else:
                insights.append("📈 Consistent performance across difficulty levels")
        
        # Response time insights
        avg_time = aggregate_metrics['avg_response_time']
        if avg_time > 5.0:
            insights.append(f"⏱️ Response time needs optimization ({avg_time:.1f}s average)")
        elif avg_time < 2.0:
            insights.append("⚡ Excellent response times")
        
        # Category performance insights
        if category_performance:
            best_category = max(category_performance.items(), key=lambda x: x[1]['overall_score'])
            worst_category = min(category_performance.items(), key=lambda x: x[1]['overall_score'])
            
            insights.append(f"🏆 Strongest category: {best_category[0]} ({best_category[1]['overall_score']:.2f})")
            insights.append(f"🔧 Needs improvement: {worst_category[0]} ({worst_category[1]['overall_score']:.2f})")
        
        return insights
    
    def export_results(self, output_path: Path) -> None:
        """Export evaluation results to file."""
        if not self.evaluation_history:
            logger.warning("No evaluation results to export")
            return
        
        export_data = {
            'framework_info': {
                'version': '2.0',
                'metrics': ['relevance', 'accuracy', 'completeness', 'factuality', 'coherence'],
                'test_examples_count': len(self.test_examples)
            },
            'evaluation_history': self.evaluation_history,
            'test_examples': [
                {
                    'id': ex.id,
                    'query': ex.query,
                    'difficulty': ex.difficulty_level,
                    'category': ex.category,
                    'query_type': ex.query_type
                }
                for ex in self.test_examples
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results exported to {output_path}")


def create_enhanced_evaluation_framework(langsmith_client: Optional[LangsmithClient] = None) -> EnhancedEvaluationFramework:
    """Create and initialize the enhanced evaluation framework."""
    return EnhancedEvaluationFramework(langsmith_client)