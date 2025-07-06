#!/usr/bin/env python3
"""
Run comprehensive evaluation of the RAG system with improved synthetic data generation
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
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.dataset import create_evaluation_dataset, load_evaluation_dataset, save_evaluation_dataset
from src.evaluation.synthetic_data_generator import SyntheticDataGenerator
from src.rag.query_processor import create_rag_processor

# Simple LLM client mock for demonstration
class MockLLMClient:
    """Mock LLM client for testing evaluation framework."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response based on prompt content."""
        # Extract key information from prompt
        if "iPhone" in prompt or "lightning" in prompt:
            return ("iPhone charging cables feature Lightning connectors, MFi certification, "
                   "various lengths from 3ft to 10ft, and support fast charging. Quality varies "
                   "by brand with Apple and certified third-party options being most reliable.")
        
        elif "Fire TV" in prompt:
            return ("Fire TV Stick provides 1080p streaming, supports Netflix, Prime Video, "
                   "has Alexa Voice Remote, and compact design. Performance is good for most "
                   "streaming but may have occasional buffering with high-demand content.")
        
        elif "backpack" in prompt:
            return ("Laptop backpack durability varies by brand. Higher-end models feature "
                   "reinforced stitching and water-resistant materials. Common issues include "
                   "strap wear and zipper failures after 6-12 months of use.")
        
        elif "earbuds" in prompt:
            return ("Bluetooth earbuds battery life ranges from 3-8 hours per charge. "
                   "Premium models like AirPods deliver 6-8 hours with 20-30 hours total via case. "
                   "Common complaints include battery degradation after 1-2 years.")
        
        elif "ethernet" in prompt and "USB" in prompt:
            return ("Ethernet cables are designed for network connectivity with speeds up to 10Gbps, "
                   "while USB cables prioritize device connectivity with varying speeds up to 40Gbps "
                   "for USB-C. Ethernet provides more stable network connections.")
        
        elif "tablet" in prompt:
            return ("For students, iPads excel in app ecosystem and battery life, while Samsung "
                   "Galaxy tablets offer better multitasking and file management at lower prices.")
        
        elif "router" in prompt:
            return ("Main wireless router complaints include inconsistent WiFi coverage, "
                   "frequent disconnections, slow speeds versus advertised specs, and overheating issues.")
        
        elif "charger" in prompt:
            return ("Smartphone charger problems include cables breaking near connectors, "
                   "slow charging speeds, overheating, and compatibility issues with cases.")
        
        elif "headphones" in prompt and "budget" in prompt:
            return ("Budget noise-canceling alternatives include TaoTronics SoundSurge 60 ($60) "
                   "and Anker Soundcore Life Q20 ($40) with decent ANC and 20-30 hour battery life.")
        
        elif "laptop" in prompt and "budget" in prompt:
            return ("Affordable laptop alternatives under $500 include refurbished ThinkPads, "
                   "Chromebooks like ASUS Chromebook Flip, and budget Windows laptops like Acer Aspire 5.")
        
        elif "mechanical keyboard" in prompt:
            return ("Yes, mechanical keyboards are excellent for programming with tactile feedback, "
                   "customizable switches, and improved durability. Popular switches include Cherry MX Blue, "
                   "Brown, and Red for different preferences.")
        
        elif "smartwatch" in prompt:
            return ("Yes, smartwatches are effective for fitness tracking with heart rate monitoring, "
                   "step counting, GPS tracking, and sleep analysis. Popular options include Apple Watch, "
                   "Garmin, and Fitbit with varying battery life.")
        
        elif "gaming headset" in prompt:
            return ("Top wireless gaming headsets under $150 include SteelSeries Arctis 7 ($140), "
                   "HyperX Cloud Flight ($120), and Corsair HS70 Pro ($100) with quality microphones "
                   "and 15-24 hour battery life.")
        
        elif "USB-C hub" in prompt:
            return ("USB-C hubs for MacBook Pro provide expanded connectivity but can cause overheating. "
                   "Pros include HDMI, USB-A ports, and power delivery. Cons include heat issues and "
                   "compatibility problems. Premium brands like CalDigit offer better reliability.")
        
        else:
            return ("Based on the available product information and customer reviews, here's what I found. "
                   "The products mentioned generally receive positive ratings with some common concerns "
                   "around durability and performance that vary by brand and price point.")

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )

def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(description='Run RAG system evaluation')
    parser.add_argument('--create-dataset', action='store_true', 
                       help='Create and save evaluation dataset')
    parser.add_argument('--dataset-path', default='data/evaluation/rag_evaluation_dataset.json',
                       help='Path to evaluation dataset')
    parser.add_argument('--project-name', default='rag-evaluation',
                       help='LangSmith project name')
    parser.add_argument('--single-query', type=str,
                       help='Run evaluation on a single query')
    parser.add_argument('--mock-llm', action='store_true', default=True,
                       help='Use mock LLM client for testing')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create dataset if requested
    if args.create_dataset:
        logger.info("Creating evaluation dataset...")
        examples = create_evaluation_dataset()
        save_evaluation_dataset(examples, args.dataset_path)
        logger.info(f"Saved {len(examples)} examples to {args.dataset_path}")
        
        # Show dataset summary
        query_types = {}
        difficulties = {}
        for example in examples:
            query_types[example.query_type] = query_types.get(example.query_type, 0) + 1
            difficulties[example.difficulty] = difficulties.get(example.difficulty, 0) + 1
        
        logger.info(f"Query types: {query_types}")
        logger.info(f"Difficulties: {difficulties}")
        return
    
    # Initialize RAG processor
    logger.info("Initializing RAG processor...")
    rag_processor = create_rag_processor()
    
    if not rag_processor.vector_db:
        logger.error("Failed to initialize vector database. Please check data files.")
        return
    
    # Initialize LLM client
    if args.mock_llm:
        logger.info("Using mock LLM client for evaluation")
        llm_client = MockLLMClient()
    else:
        logger.error("Real LLM client not implemented. Use --mock-llm flag.")
        return
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_processor, llm_client)
    
    # Single query evaluation
    if args.single_query:
        logger.info(f"Running single query evaluation: {args.single_query}")
        
        # Use a default example for testing
        result = evaluator.run_single_evaluation(
            query=args.single_query,
            expected_answer="Expected answer for testing purposes",
            expected_products=["test_product"],
            expected_topics=["test_topic"],
            query_type="general_search"
        )
        
        logger.info("Single query evaluation results:")
        logger.info(f"Overall Score: {result.overall:.3f}")
        logger.info(f"Relevance: {result.relevance:.3f}")
        logger.info(f"Accuracy: {result.accuracy:.3f}")
        logger.info(f"Completeness: {result.completeness:.3f}")
        logger.info(f"Factuality: {result.factuality:.3f}")
        logger.info(f"Quality: {result.quality:.3f}")
        return
    
    # Full evaluation
    try:
        # Load or create dataset
        if Path(args.dataset_path).exists():
            logger.info(f"Loading evaluation dataset from {args.dataset_path}")
            examples = load_evaluation_dataset(args.dataset_path)
        else:
            logger.info("Creating new evaluation dataset")
            examples = create_evaluation_dataset()
        
        logger.info(f"Running evaluation on {len(examples)} examples...")
        
        # Run evaluation
        evaluation_result = asyncio.run(
            evaluator.run_evaluation(examples, args.project_name)
        )
        
        # Analyze results
        analysis = evaluator.analyze_results(evaluation_result)
        
        logger.info("Evaluation completed!")
        logger.info("Results Summary:")
        if 'summary' in analysis:
            summary = analysis['summary']
            logger.info(f"  Total Examples: {summary['total_examples']}")
            logger.info(f"  Average Score: {summary['average_score']:.3f}")
            logger.info(f"  Min Score: {summary['min_score']:.3f}")
            logger.info(f"  Max Score: {summary['max_score']:.3f}")
            logger.info(f"  High Performance (>0.8): {summary['scores_above_0.8']}")
            logger.info(f"  Low Performance (<0.5): {summary['scores_below_0.5']}")
        
        if 'score_distribution' in analysis:
            logger.info("Score Distribution:")
            for category, count in analysis['score_distribution'].items():
                logger.info(f"  {category}: {count}")
        
        logger.info(f"Detailed results available in LangSmith project: {args.project_name}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()