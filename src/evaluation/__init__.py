"""
Evaluation module for AI-Powered Amazon Product Assistant.
Provides LangSmith-based evaluation framework for RAG system.
"""

from .evaluator import RAGEvaluator, EvaluationMetrics
from .dataset import create_evaluation_dataset, EvaluationExample
from .scorers import (
    RelevanceScorer,
    AccuracyScorer,
    CompletenessScorer,
    FactualityScorer,
    ResponseQualityScorer
)

__all__ = [
    "RAGEvaluator",
    "EvaluationMetrics", 
    "create_evaluation_dataset",
    "EvaluationExample",
    "RelevanceScorer",
    "AccuracyScorer", 
    "CompletenessScorer",
    "FactualityScorer",
    "ResponseQualityScorer"
]