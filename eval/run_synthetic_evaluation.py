#!/usr/bin/env python3
"""
Run synthetic evaluation of the RAG system with benchmark comparison
"""
import os
import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.dataset import create_evaluation_dataset, load_evaluation_dataset
from src.evaluation.synthetic_data_generator import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    create_synthetic_dataset,
    create_mixed_dataset,
    save_synthetic_dataset
)
from src.rag.query_processor import create_rag_processor
from langsmith import traceable

# Mock LLM client for demonstration (reuse from run_evaluation.py)
class MockLLMClient:
    """Mock LLM client for testing evaluation framework."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response based on prompt content."""
        # Enhanced mock responses for synthetic data testing
        if any(keyword in prompt.lower() for keyword in ["headphones", "earbuds", "audio"]):
            return ("Audio products feature varying sound quality, comfort levels, and battery life. "
                   "Premium models offer superior drivers, noise cancellation, and longer battery life. "
                   "Common issues include connection problems and comfort during extended use.")
        
        elif any(keyword in prompt.lower() for keyword in ["laptop", "keyboard", "computing"]):
            return ("Computing devices vary significantly in performance, portability, and features. "
                   "Key considerations include processor speed, memory, storage type, and build quality. "
                   "Professional users prioritize performance while casual users focus on value.")
        
        elif any(keyword in prompt.lower() for keyword in ["smartphone", "mobile", "charger"]):
            return ("Mobile accessories show wide quality variations. Premium cables offer better "
                   "durability and faster charging speeds. Common problems include connector wear "
                   "and compatibility issues with protective cases.")
        
        elif "comparison" in prompt.lower() or " vs " in prompt.lower():
            return ("Product comparisons reveal distinct advantages for each option. Consider your "
                   "specific use case, budget constraints, and feature priorities when choosing. "
                   "Both products have loyal user bases with different strengths.")
        
        elif any(keyword in prompt.lower() for keyword in ["gaming", "controller", "mouse"]):
            return ("Gaming peripherals prioritize responsiveness, customization, and durability. "
                   "Professional gamers value low latency and precise controls, while casual users "
                   "prefer comfort and versatility. RGB lighting is popular but not essential.")
        
        else:
            return ("Based on available product information and user reviews, quality varies by brand "
                   "and price point. Consider your specific needs, budget, and intended use case. "
                   "Research user feedback for real-world performance insights.")


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('synthetic_evaluation.log')
        ]
    )


@traceable
def run_synthetic_evaluation_pipeline(config: SyntheticDataConfig, 
                                    project_name: str = "synthetic-rag-evaluation") -> dict:
    """Run complete synthetic evaluation pipeline with LangSmith tracking."""
    
    # Initialize LangSmith environment
    if "LANGSMITH_PROJECT" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = project_name
    
    logger = logging.getLogger(__name__)
    logger.info("Starting synthetic evaluation pipeline")
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    synthetic_examples = create_synthetic_dataset(config, num_examples=30)
    
    # Save synthetic dataset for review
    synthetic_path = "data/evaluation/synthetic_rag_dataset.json"
    save_synthetic_dataset(synthetic_examples, synthetic_path)
    logger.info(f"Saved {len(synthetic_examples)} synthetic examples to {synthetic_path}")
    
    # Create mixed dataset (combine original + synthetic)
    logger.info("Creating mixed dataset...")
    original_examples = create_evaluation_dataset()
    mixed_examples = create_mixed_dataset(original_examples, synthetic_ratio=0.5, config=config)
    
    # Save mixed dataset
    mixed_path = "data/evaluation/mixed_rag_dataset.json"
    save_synthetic_dataset(mixed_examples, mixed_path)
    logger.info(f"Created mixed dataset with {len(mixed_examples)} total examples")
    
    # Initialize RAG system
    logger.info("Initializing RAG processor...")
    rag_processor = create_rag_processor()
    
    if not rag_processor.vector_db:
        logger.error("Failed to initialize vector database")
        return {"error": "Vector database initialization failed"}
    
    # Initialize evaluator
    llm_client = MockLLMClient()
    evaluator = RAGEvaluator(rag_processor, llm_client)
    
    # Run evaluations on different datasets
    results = {}
    
    # Evaluate synthetic-only dataset
    logger.info("Evaluating synthetic dataset...")
    synthetic_result = asyncio.run(
        evaluator.run_evaluation(synthetic_examples, f"{project_name}-synthetic")
    )
    results["synthetic"] = evaluator.analyze_results(synthetic_result)
    
    # Evaluate mixed dataset
    logger.info("Evaluating mixed dataset...")
    mixed_result = asyncio.run(
        evaluator.run_evaluation(mixed_examples, f"{project_name}-mixed")
    )
    results["mixed"] = evaluator.analyze_results(mixed_result)
    
    # Evaluate original dataset for comparison
    logger.info("Evaluating original dataset...")
    original_result = asyncio.run(
        evaluator.run_evaluation(original_examples, f"{project_name}-original")
    )
    results["original"] = evaluator.analyze_results(original_result)
    
    logger.info("Synthetic evaluation pipeline completed!")
    return results


def main():
    """Main synthetic evaluation runner."""
    parser = argparse.ArgumentParser(description='Run RAG evaluation with synthetic data')
    parser.add_argument('--synthetic-only', action='store_true',
                       help='Generate and evaluate only synthetic data')
    parser.add_argument('--mixed-dataset', action='store_true',
                       help='Create and evaluate mixed dataset (original + synthetic)')
    parser.add_argument('--num-synthetic', type=int, default=30,
                       help='Number of synthetic examples to generate')
    parser.add_argument('--project-name', default='synthetic-rag-evaluation',
                       help='LangSmith project name')
    parser.add_argument('--difficulty-easy', type=float, default=0.3,
                       help='Proportion of easy examples')
    parser.add_argument('--difficulty-medium', type=float, default=0.5,
                       help='Proportion of medium examples')
    parser.add_argument('--difficulty-hard', type=float, default=0.2,
                       help='Proportion of hard examples')
    parser.add_argument('--save-datasets', action='store_true',
                       help='Save generated datasets to files')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configure synthetic data generation
    config = SyntheticDataConfig(
        num_examples_per_category=3,
        difficulty_distribution={
            "easy": args.difficulty_easy,
            "medium": args.difficulty_medium, 
            "hard": args.difficulty_hard
        }
    )
    
    try:
        if args.synthetic_only:
            # Generate and evaluate synthetic data only
            logger.info("Running synthetic-only evaluation...")
            
            # Initialize LangSmith environment
            if "LANGSMITH_PROJECT" not in os.environ:
                os.environ["LANGSMITH_PROJECT"] = args.project_name
            
            synthetic_examples = create_synthetic_dataset(config, args.num_synthetic)
            
            if args.save_datasets:
                save_synthetic_dataset(synthetic_examples, "data/evaluation/synthetic_only_dataset.json")
            
            # Initialize RAG system and evaluator
            rag_processor = create_rag_processor()
            llm_client = MockLLMClient()
            evaluator = RAGEvaluator(rag_processor, llm_client)
            
            # Run evaluation
            result = asyncio.run(
                evaluator.run_evaluation(synthetic_examples, f"{args.project_name}-synthetic")
            )
            
            # Analyze results
            analysis = evaluator.analyze_results(result)
            logger.info("Synthetic evaluation completed!")
            
            if 'summary' in analysis:
                summary = analysis['summary']
                logger.info(f"Results - Total: {summary['total_examples']}, "
                          f"Avg Score: {summary['average_score']:.3f}")
        
        elif args.mixed_dataset:
            # Create and evaluate mixed dataset
            logger.info("Running mixed dataset evaluation...")
            
            # Initialize LangSmith environment
            if "LANGSMITH_PROJECT" not in os.environ:
                os.environ["LANGSMITH_PROJECT"] = args.project_name
            
            original_examples = create_evaluation_dataset()
            mixed_examples = create_mixed_dataset(
                original_examples, 
                synthetic_ratio=0.5, 
                config=config
            )
            
            if args.save_datasets:
                save_synthetic_dataset(mixed_examples, "data/evaluation/mixed_evaluation_dataset.json")
            
            # Initialize RAG system and evaluator
            rag_processor = create_rag_processor()
            llm_client = MockLLMClient()
            evaluator = RAGEvaluator(rag_processor, llm_client)
            
            # Run evaluation
            result = asyncio.run(
                evaluator.run_evaluation(mixed_examples, f"{args.project_name}-mixed")
            )
            
            # Analyze results
            analysis = evaluator.analyze_results(result)
            logger.info("Mixed dataset evaluation completed!")
            
            if 'summary' in analysis:
                summary = analysis['summary']
                logger.info(f"Results - Total: {summary['total_examples']}, "
                          f"Avg Score: {summary['average_score']:.3f}")
                logger.info(f"Synthetic examples: {sum(1 for ex in mixed_examples if ex.metadata.get('synthetic', False))}")
        
        else:
            # Run complete pipeline comparison
            logger.info("Running complete synthetic evaluation pipeline...")
            results = run_synthetic_evaluation_pipeline(config, args.project_name)
            
            # Display comparison results
            logger.info("Evaluation Results Comparison:")
            for dataset_type, analysis in results.items():
                if 'summary' in analysis:
                    summary = analysis['summary']
                    logger.info(f"{dataset_type.title()} Dataset:")
                    logger.info(f"  Examples: {summary['total_examples']}")
                    logger.info(f"  Avg Score: {summary['average_score']:.3f}")
                    logger.info(f"  Score Range: {summary['min_score']:.3f} - {summary['max_score']:.3f}")
            
            logger.info(f"Detailed results available in LangSmith project: {args.project_name}")
    
    except Exception as e:
        logger.error(f"Synthetic evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()