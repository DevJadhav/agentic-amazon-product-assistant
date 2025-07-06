"""
Main evaluator class using LangSmith for RAG system evaluation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from langsmith import traceable

from .dataset import EvaluationExample, create_evaluation_dataset
from .scorers import (
    RelevanceScorer, AccuracyScorer, CompletenessScorer, 
    FactualityScorer, ResponseQualityScorer
)

logger = logging.getLogger(__name__)

@dataclass 
class EvaluationMetrics:
    """Container for evaluation metrics."""
    relevance: float
    accuracy: float
    completeness: float
    factuality: float
    quality: float
    overall: float
    details: Dict[str, Any]

class RAGSystemModel:
    """LangSmith model wrapper for RAG system."""
    
    def __init__(self, rag_processor, llm_client):
        """Initialize with RAG processor and LLM client."""
        self._rag_processor = rag_processor
        self._llm_client = llm_client
    
    @traceable
    def predict(self, query: str) -> Dict[str, Any]:
        """Generate response using RAG system."""
        try:
            # Process query through RAG
            rag_result = self._rag_processor.process_query(query)
            
            if not rag_result["success"]:
                return {
                    "response": f"Error processing query: {rag_result.get('error', 'Unknown error')}",
                    "success": False,
                    "error": rag_result.get('error')
                }
            
            # Generate response using enhanced prompt
            enhanced_prompt = rag_result["enhanced_prompt"]
            
            # Call LLM (simplified - adapt to your LLM client)
            try:
                if hasattr(self._llm_client, 'generate'):
                    response = self._llm_client.generate(enhanced_prompt)
                else:
                    # Fallback for different client interfaces
                    response = str(self._llm_client(enhanced_prompt))
                    
                return {
                    "response": response,
                    "success": True,
                    "rag_context": rag_result["context"],
                    "metadata": rag_result["metadata"]
                }
            except Exception as llm_error:
                logger.error(f"LLM error: {llm_error}")
                return {
                    "response": "Error generating response",
                    "success": False, 
                    "error": str(llm_error)
                }
                
        except Exception as e:
            logger.error(f"RAG system error: {e}")
            return {
                "response": f"System error: {str(e)}",
                "success": False,
                "error": str(e)
            }

class RAGEvaluator:
    """Main evaluator for RAG system using LangSmith."""
    
    def __init__(self, rag_processor, llm_client):
        """Initialize evaluator with RAG processor and LLM client."""
        self.rag_processor = rag_processor
        self.llm_client = llm_client
        self.model = RAGSystemModel(rag_processor, llm_client)
        
        # Initialize scorers
        self.relevance_scorer = RelevanceScorer()
        self.accuracy_scorer = AccuracyScorer()
        self.completeness_scorer = CompletenessScorer()
        self.factuality_scorer = FactualityScorer()
        self.quality_scorer = ResponseQualityScorer()
    
    @traceable
    def comprehensive_scorer(self, query: str, output: Dict[str, Any], expected_answer: str, 
                           expected_products: List[str], expected_topics: List[str], 
                           query_type: str) -> Dict[str, Any]:
        """Comprehensive scoring function that combines all metrics."""
        
        if not output.get("success", False):
            return {
                'overall_score': 0.0,
                'relevance': 0.0,
                'accuracy': 0.0,
                'completeness': 0.0,
                'factuality': 0.0,
                'quality': 0.0,
                'error': output.get('error', 'Unknown error')
            }
        
        response = output.get("response", "")
        
        # Calculate individual scores
        relevance_result = self.relevance_scorer(query, response, expected_topics)
        accuracy_result = self.accuracy_scorer(response, expected_answer, expected_products)
        completeness_result = self.completeness_scorer(response, expected_answer, query_type)
        factuality_result = self.factuality_scorer(response, expected_answer)
        quality_result = self.quality_scorer(response, query)
        
        # Extract scores
        relevance_score = relevance_result['relevance_score']
        accuracy_score = accuracy_result['accuracy_score']
        completeness_score = completeness_result['completeness_score']
        factuality_score = factuality_result['factuality_score']
        quality_score = quality_result['quality_score']
        
        # Calculate overall score (weighted average)
        overall_score = (
            relevance_score * 0.25 +
            accuracy_score * 0.25 +
            completeness_score * 0.2 +
            factuality_score * 0.15 +
            quality_score * 0.15
        )
        
        return {
            'overall_score': overall_score,
            'relevance': relevance_score,
            'accuracy': accuracy_score,
            'completeness': completeness_score,
            'factuality': factuality_score,
            'quality': quality_score,
            'details': {
                'relevance_details': relevance_result,
                'accuracy_details': accuracy_result,
                'completeness_details': completeness_result,
                'factuality_details': factuality_result,
                'quality_details': quality_result,
                'rag_metadata': output.get('metadata', {})
            }
        }
    
    def prepare_evaluation_dataset(self, examples: Optional[List[EvaluationExample]] = None) -> List[Dict[str, Any]]:
        """Prepare dataset for LangSmith evaluation."""
        if examples is None:
            examples = create_evaluation_dataset()
        
        dataset = []
        for example in examples:
            dataset.append({
                'query': example.query,
                'expected_answer': example.expected_answer,
                'expected_products': example.expected_products,
                'expected_topics': example.expected_topics,
                'query_type': example.query_type,
                'difficulty': example.difficulty,
                'metadata': example.metadata
            })
        
        return dataset
    
    def create_scorer_functions(self) -> List[Callable]:
        """Create scorer functions for LangSmith evaluation."""
        
        @traceable
        def relevance_scorer(query: str, output: Dict[str, Any], expected_topics: List[str]) -> Dict[str, Any]:
            if not output.get("success", False):
                return {'relevance': 0.0}
            response = output.get("response", "")
            result = self.relevance_scorer(query, response, expected_topics)
            return {'relevance': result['relevance_score']}
        
        @traceable
        def accuracy_scorer(expected_answer: str, output: Dict[str, Any], expected_products: List[str]) -> Dict[str, Any]:
            if not output.get("success", False):
                return {'accuracy': 0.0}
            response = output.get("response", "")
            result = self.accuracy_scorer(response, expected_answer, expected_products)
            return {'accuracy': result['accuracy_score']}
        
        @traceable
        def completeness_scorer(expected_answer: str, output: Dict[str, Any], query_type: str) -> Dict[str, Any]:
            if not output.get("success", False):
                return {'completeness': 0.0}
            response = output.get("response", "")
            result = self.completeness_scorer(response, expected_answer, query_type)
            return {'completeness': result['completeness_score']}
        
        @traceable
        def factuality_scorer(expected_answer: str, output: Dict[str, Any]) -> Dict[str, Any]:
            if not output.get("success", False):
                return {'factuality': 0.0}
            response = output.get("response", "")
            result = self.factuality_scorer(response, expected_answer)
            return {'factuality': result['factuality_score']}
        
        @traceable
        def quality_scorer(query: str, output: Dict[str, Any]) -> Dict[str, Any]:
            if not output.get("success", False):
                return {'quality': 0.0}
            response = output.get("response", "")
            result = self.quality_scorer(response, query)
            return {'quality': result['quality_score']}
        
        @traceable
        def overall_scorer(query: str, output: Dict[str, Any], expected_answer: str, 
                          expected_products: List[str], expected_topics: List[str], 
                          query_type: str) -> Dict[str, Any]:
            return self.comprehensive_scorer(
                query, output, expected_answer, expected_products, 
                expected_topics, query_type
            )
        
        return [relevance_scorer, accuracy_scorer, completeness_scorer, 
                factuality_scorer, quality_scorer, overall_scorer]
    
    async def run_evaluation(self, examples: Optional[List[EvaluationExample]] = None, 
                           project_name: str = "rag-evaluation") -> Dict[str, Any]:
        """Run complete evaluation using LangSmith."""
        
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(examples)
        
        # Run evaluation on each example
        logger.info(f"Running evaluation on {len(dataset)} examples...")
        results = []
        
        for i, example in enumerate(dataset):
            logger.info(f"Evaluating example {i+1}/{len(dataset)}")
            
            # Generate prediction
            output = self.model.predict(example['query'])
            
            # Calculate comprehensive score
            score_result = self.comprehensive_scorer(
                example['query'], output, example['expected_answer'], 
                example['expected_products'], example['expected_topics'], 
                example['query_type']
            )
            
            result_item = {
                'example': example,
                'output': output,
                'scores': score_result
            }
            results.append(result_item)
        
        logger.info("Evaluation completed!")
        
        # Aggregate results
        return {
            'project_name': project_name,
            'total_examples': len(results),
            'results': results,
            'summary': self._calculate_summary_stats(results)
        }
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results."""
        scores = [r['scores']['overall_score'] for r in results if 'scores' in r]
        
        if not scores:
            return {"error": "No valid scores found"}
        
        return {
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'total_examples': len(scores),
            'scores_above_0.8': sum(1 for s in scores if s > 0.8),
            'scores_below_0.5': sum(1 for s in scores if s < 0.5),
        }
    
    def run_single_evaluation(self, query: str, expected_answer: str, 
                            expected_products: List[str], expected_topics: List[str],
                            query_type: str) -> EvaluationMetrics:
        """Run evaluation on a single query."""
        
        # Generate response
        output = self.model.predict(query)
        
        # Calculate comprehensive score
        score_result = self.comprehensive_scorer(
            query, output, expected_answer, expected_products, 
            expected_topics, query_type
        )
        
        return EvaluationMetrics(
            relevance=score_result['relevance'],
            accuracy=score_result['accuracy'], 
            completeness=score_result['completeness'],
            factuality=score_result['factuality'],
            quality=score_result['quality'],
            overall=score_result['overall_score'],
            details=score_result['details']
        )
    
    def analyze_results(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evaluation results and provide insights."""
        
        # Extract scores from evaluation result
        # This will depend on the exact structure returned by LangSmith
        # Placeholder implementation:
        
        scores = []
        if 'model_output' in evaluation_result:
            for result in evaluation_result['model_output']:
                if isinstance(result, dict) and 'overall_score' in result:
                    scores.append(result['overall_score'])
        
        if not scores:
            return {"error": "No valid scores found in evaluation result"}
        
        analysis = {
            'summary': {
                'total_examples': len(scores),
                'average_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'scores_above_0.8': sum(1 for s in scores if s > 0.8),
                'scores_below_0.5': sum(1 for s in scores if s < 0.5)
            },
            'score_distribution': {
                'excellent (>0.8)': sum(1 for s in scores if s > 0.8),
                'good (0.6-0.8)': sum(1 for s in scores if 0.6 <= s <= 0.8),
                'acceptable (0.4-0.6)': sum(1 for s in scores if 0.4 <= s < 0.6),
                'poor (<0.4)': sum(1 for s in scores if s < 0.4)
            }
        }
        
        return analysis

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would require actual RAG processor and LLM client
    # evaluator = RAGEvaluator(rag_processor, llm_client)
    # result = asyncio.run(evaluator.run_evaluation())
    # analysis = evaluator.analyze_results(result)
    # print(analysis)
    
    print("RAG Evaluator ready - integrate with your RAG processor and LLM client")