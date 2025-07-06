"""
Synthetic test data generation for RAG system evaluation.

This module provides utilities to generate synthetic evaluation examples using
various approaches including template-based generation, LLM-based augmentation,
and perturbation techniques following best practices for synthetic data generation.
"""

import json
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import combinations
from langsmith import traceable

from .dataset import EvaluationExample


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    num_examples_per_category: int = 5
    difficulty_distribution: Dict[str, float] = None
    query_type_weights: Dict[str, float] = None
    variation_techniques: List[str] = None
    
    def __post_init__(self):
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
        if self.query_type_weights is None:
            self.query_type_weights = {
                "product_info": 0.25,
                "product_reviews": 0.20,
                "product_comparison": 0.15,
                "product_complaints": 0.15,
                "product_recommendation": 0.15,
                "use_case": 0.10
            }
        if self.variation_techniques is None:
            self.variation_techniques = ["rephrase", "specificity", "context", "perspective"]


class SyntheticDataGenerator:
    """Generate synthetic evaluation data for RAG system testing."""
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.config = config or SyntheticDataConfig()
        self.product_categories = self._load_product_categories()
        self.query_templates = self._initialize_query_templates()
        self.variation_patterns = self._initialize_variation_patterns()
    
    def _load_product_categories(self) -> Dict[str, List[str]]:
        """Load product categories and examples from existing data."""
        return {
            "audio": ["headphones", "earbuds", "speakers", "microphones", "soundbars"],
            "computing": ["laptops", "keyboards", "mice", "monitors", "tablets"],
            "mobile": ["smartphones", "chargers", "cables", "cases", "power banks"],
            "networking": ["routers", "ethernet cables", "wifi extenders", "modems"],
            "gaming": ["gaming headsets", "controllers", "gaming keyboards", "gaming mice"],
            "accessories": ["backpacks", "stands", "hubs", "adapters", "mounts"],
            "streaming": ["streaming devices", "media players", "smart tv boxes"],
            "storage": ["hard drives", "SSDs", "USB drives", "memory cards"],
            "wearables": ["smartwatches", "fitness trackers", "smart rings"],
            "home": ["smart home devices", "security cameras", "smart plugs"]
        }
    
    def _initialize_query_templates(self) -> Dict[str, List[str]]:
        """Initialize query templates for different types."""
        return {
            "product_info": [
                "What are the key features of {product}?",
                "Tell me about {product} specifications",
                "How does {product} work?",
                "What makes {product} different from competitors?",
                "Explain the benefits of {product}",
                "What should I know about {product} before buying?"
            ],
            "product_reviews": [
                "What do customers say about {product}?",
                "What are the reviews for {product}?",
                "How satisfied are users with {product}?",
                "What feedback do people give about {product}?",
                "What are common opinions on {product}?",
                "How do users rate {product}?"
            ],
            "product_comparison": [
                "Compare {product1} vs {product2}",
                "What's the difference between {product1} and {product2}?",
                "Which is better: {product1} or {product2}?",
                "{product1} versus {product2} - pros and cons",
                "Help me choose between {product1} and {product2}",
                "What are the trade-offs between {product1} and {product2}?"
            ],
            "product_complaints": [
                "What problems do people have with {product}?",
                "What are common complaints about {product}?",
                "What issues should I be aware of with {product}?",
                "What are the downsides of {product}?",
                "What negative feedback exists for {product}?",
                "What are the main criticisms of {product}?"
            ],
            "product_recommendation": [
                "Recommend a good {product} for {use_case}",
                "What's the best {product} for {use_case}?",
                "Suggest a {budget} {product} for {use_case}",
                "Find me a {product} that's good for {use_case}",
                "What {product} would you recommend for {use_case}?",
                "Help me find the right {product} for {use_case}"
            ],
            "use_case": [
                "Is {product} good for {use_case}?",
                "Can {product} be used for {use_case}?",
                "How well does {product} work for {use_case}?",
                "Is {product} suitable for {use_case}?",
                "Would {product} meet my needs for {use_case}?",
                "Can I use {product} for {use_case} effectively?"
            ]
        }
    
    def _initialize_variation_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for creating query variations."""
        return {
            "rephrase": [
                ("What are", "Can you tell me about"),
                ("How does", "How well does"),
                ("Tell me about", "Explain"),
                ("What's the best", "Which is the top"),
                ("Recommend", "Suggest"),
                ("Compare", "What's the difference between")
            ],
            "specificity": [
                ("good", "high-quality"),
                ("cheap", "budget-friendly"),
                ("expensive", "premium"),
                ("popular", "highly-rated"),
                ("new", "latest"),
                ("old", "older generation")
            ],
            "context": [
                "for students",
                "for professionals", 
                "for gaming",
                "for work",
                "for home use",
                "for travel",
                "for beginners",
                "for experts"
            ],
            "perspective": [
                "according to reviews",
                "based on user feedback", 
                "from customer experiences",
                "in terms of value",
                "considering quality",
                "for long-term use"
            ]
        }
    
    @traceable
    def generate_synthetic_examples(self, num_examples: int = 50) -> List[EvaluationExample]:
        """Generate synthetic evaluation examples using multiple techniques."""
        examples = []
        
        # Calculate examples per query type
        examples_per_type = self._distribute_examples_by_type(num_examples)
        
        for query_type, count in examples_per_type.items():
            type_examples = self._generate_examples_for_type(query_type, count)
            examples.extend(type_examples)
        
        # Shuffle to avoid clustering by type
        random.shuffle(examples)
        return examples
    
    def _distribute_examples_by_type(self, total_examples: int) -> Dict[str, int]:
        """Distribute examples across query types based on weights."""
        distribution = {}
        remaining = total_examples
        
        for query_type, weight in self.config.query_type_weights.items():
            count = int(total_examples * weight)
            distribution[query_type] = count
            remaining -= count
        
        # Distribute remaining examples randomly
        types = list(self.config.query_type_weights.keys())
        for _ in range(remaining):
            random.choice(types)
            distribution[random.choice(types)] += 1
        
        return distribution
    
    @traceable
    def _generate_examples_for_type(self, query_type: str, count: int) -> List[EvaluationExample]:
        """Generate examples for a specific query type."""
        examples = []
        
        for _ in range(count):
            if query_type == "product_comparison":
                example = self._generate_comparison_example()
            else:
                example = self._generate_single_product_example(query_type)
            
            if example:
                examples.append(example)
        
        return examples
    
    def _generate_single_product_example(self, query_type: str) -> Optional[EvaluationExample]:
        """Generate example for single product queries."""
        # Select random category and product
        category = random.choice(list(self.product_categories.keys()))
        product = random.choice(self.product_categories[category])
        
        # Create base query from template
        template = random.choice(self.query_templates[query_type])
        
        if query_type == "product_recommendation":
            use_case = self._generate_use_case()
            budget = random.choice(["budget-friendly", "mid-range", "premium", "affordable"])
            query = template.format(product=product, use_case=use_case, budget=budget)
        elif query_type == "use_case":
            use_case = self._generate_use_case()
            query = template.format(product=product, use_case=use_case)
        else:
            query = template.format(product=product)
        
        # Apply variations
        query = self._apply_query_variations(query)
        
        # Generate expected answer and metadata
        expected_answer = self._generate_expected_answer(query_type, product, category)
        difficulty = self._assign_difficulty()
        
        return EvaluationExample(
            query=query,
            expected_answer=expected_answer,
            query_type=query_type,
            expected_products=[product.replace(" ", "_")],
            expected_topics=self._generate_expected_topics(query_type, category),
            difficulty=difficulty,
            metadata={
                "category": "electronics",
                "subcategory": category,
                "synthetic": True,
                "generation_method": "template_based"
            }
        )
    
    def _generate_comparison_example(self) -> Optional[EvaluationExample]:
        """Generate comparison query example."""
        category = random.choice(list(self.product_categories.keys()))
        products = random.sample(self.product_categories[category], min(2, len(self.product_categories[category])))
        
        if len(products) < 2:
            # Fallback to cross-category comparison
            categories = random.sample(list(self.product_categories.keys()), 2)
            products = [
                random.choice(self.product_categories[categories[0]]),
                random.choice(self.product_categories[categories[1]])
            ]
        
        template = random.choice(self.query_templates["product_comparison"])
        query = template.format(product1=products[0], product2=products[1])
        
        # Apply variations
        query = self._apply_query_variations(query)
        
        expected_answer = self._generate_comparison_answer(products[0], products[1])
        difficulty = "hard"  # Comparisons are inherently more complex
        
        return EvaluationExample(
            query=query,
            expected_answer=expected_answer,
            query_type="product_comparison",
            expected_products=[p.replace(" ", "_") for p in products],
            expected_topics=["comparison", "features", "performance", "value", "use_cases"],
            difficulty=difficulty,
            metadata={
                "category": "electronics",
                "subcategory": category,
                "synthetic": True,
                "generation_method": "comparison_template"
            }
        )
    
    def _apply_query_variations(self, query: str) -> str:
        """Apply various techniques to create query variations."""
        for technique in self.config.variation_techniques:
            if random.random() < 0.3:  # 30% chance to apply each technique
                query = self._apply_variation_technique(query, technique)
        
        return query
    
    def _apply_variation_technique(self, query: str, technique: str) -> str:
        """Apply a specific variation technique to the query."""
        if technique == "rephrase":
            for old, new in self.variation_patterns["rephrase"]:
                if old in query:
                    query = query.replace(old, new, 1)
                    break
        
        elif technique == "specificity":
            for old, new in self.variation_patterns["specificity"]:
                if old in query:
                    query = query.replace(old, new, 1)
                    break
        
        elif technique == "context":
            context = random.choice(self.variation_patterns["context"])
            if "?" in query:
                query = query.replace("?", f" {context}?")
            else:
                query += f" {context}"
        
        elif technique == "perspective":
            perspective = random.choice(self.variation_patterns["perspective"])
            if "?" in query:
                query = query.replace("?", f" {perspective}?")
            else:
                query += f" {perspective}"
        
        return query
    
    def _generate_use_case(self) -> str:
        """Generate realistic use cases."""
        use_cases = [
            "gaming", "work", "study", "travel", "home office", "exercise",
            "streaming", "photography", "video editing", "programming",
            "music production", "content creation", "remote work", "education"
        ]
        return random.choice(use_cases)
    
    def _generate_expected_answer(self, query_type: str, product: str, category: str) -> str:
        """Generate realistic expected answers based on query type."""
        answer_templates = {
            "product_info": f"{product.title()} typically features [key specifications], [performance characteristics], and [build quality aspects]. Popular for {category} applications, offering [main benefits] with [price range] pricing.",
            
            "product_reviews": f"Customer reviews for {product} show [satisfaction level] with praise for [positive aspects] and concerns about [common issues]. Users particularly appreciate [standout features] while noting [areas for improvement].",
            
            "product_complaints": f"Common complaints about {product} include [durability issues], [performance problems], [compatibility concerns], and [value-related criticisms]. Users frequently report [specific issues] affecting [usage scenarios].",
            
            "product_recommendation": f"For your needs, consider [recommended models] which offer [key features] within [budget range]. Top choices include [specific products] based on [selection criteria] and [user requirements].",
            
            "product_comparison": f"Comparing these products: [Product A] excels in [strengths] while [Product B] offers [advantages]. Choose [Product A] for [use case A] or [Product B] for [use case B] based on [decision factors].",
            
            "use_case": f"Yes, {product} can be effective for this use case, offering [relevant features] and [performance characteristics]. Consider [important factors] and expect [realistic outcomes] based on [usage requirements]."
        }
        
        template = answer_templates.get(query_type, f"Information about {product} and related {category} products.")
        return template
    
    def _generate_comparison_answer(self, product1: str, product2: str) -> str:
        """Generate comparison answer template."""
        return f"Comparing {product1} vs {product2}: {product1.title()} offers [specific advantages] while {product2.title()} provides [different benefits]. Choose {product1} for [use case 1] or {product2} for [use case 2] based on [key decision factors]."
    
    def _assign_difficulty(self) -> str:
        """Assign difficulty based on configured distribution."""
        rand = random.random()
        cumulative = 0
        
        for difficulty, probability in self.config.difficulty_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return difficulty
        
        return "medium"  # fallback
    
    def _generate_expected_topics(self, query_type: str, category: str) -> List[str]:
        """Generate expected topics based on query type and category."""
        base_topics = {
            "product_info": ["features", "specifications", "performance", "design"],
            "product_reviews": ["user_experience", "satisfaction", "quality", "value"],
            "product_comparison": ["comparison", "features", "performance", "value"],
            "product_complaints": ["issues", "problems", "limitations", "concerns"],
            "product_recommendation": ["recommendations", "selection", "features", "value"],
            "use_case": ["suitability", "performance", "features", "effectiveness"]
        }
        
        category_topics = {
            "audio": ["sound_quality", "comfort", "battery_life", "connectivity"],
            "computing": ["performance", "portability", "display", "battery_life"],
            "mobile": ["compatibility", "durability", "charging_speed", "design"],
            "networking": ["speed", "range", "reliability", "setup"],
            "gaming": ["performance", "latency", "comfort", "features"]
        }
        
        topics = base_topics.get(query_type, ["general"])
        topics.extend(category_topics.get(category, []))
        
        return topics[:4]  # Limit to 4 topics
    
    @traceable
    def augment_existing_dataset(self, original_examples: List[EvaluationExample], 
                                augmentation_factor: float = 0.5) -> List[EvaluationExample]:
        """Augment existing dataset with synthetic variations."""
        augmented = list(original_examples)  # Copy original examples
        
        num_synthetic = int(len(original_examples) * augmentation_factor)
        
        # Create variations of existing examples
        for _ in range(num_synthetic):
            base_example = random.choice(original_examples)
            synthetic_example = self._create_variation(base_example)
            if synthetic_example:
                augmented.append(synthetic_example)
        
        return augmented
    
    def _create_variation(self, base_example: EvaluationExample) -> Optional[EvaluationExample]:
        """Create a variation of an existing example."""
        # Apply query variations
        new_query = self._apply_query_variations(base_example.query)
        
        # Slightly modify expected answer
        new_answer = self._vary_expected_answer(base_example.expected_answer)
        
        # Create new metadata
        new_metadata = base_example.metadata.copy()
        new_metadata["synthetic"] = True
        new_metadata["generation_method"] = "variation"
        new_metadata["base_example"] = base_example.query[:50] + "..."
        
        return EvaluationExample(
            query=new_query,
            expected_answer=new_answer,
            query_type=base_example.query_type,
            expected_products=base_example.expected_products,
            expected_topics=base_example.expected_topics,
            difficulty=base_example.difficulty,
            metadata=new_metadata
        )
    
    def _vary_expected_answer(self, original_answer: str) -> str:
        """Create a slight variation of the expected answer."""
        # Simple synonym replacement and structure variation
        variations = [
            ("excellent", "outstanding"),
            ("good", "solid"),
            ("problems", "issues"),
            ("features", "capabilities"),
            ("popular", "well-regarded"),
            ("typically", "generally"),
            ("offers", "provides"),
            ("include", "feature")
        ]
        
        varied_answer = original_answer
        for old, new in random.sample(variations, min(2, len(variations))):
            if old in varied_answer.lower():
                varied_answer = re.sub(rf'\b{old}\b', new, varied_answer, count=1, flags=re.IGNORECASE)
        
        return varied_answer


@traceable
def create_synthetic_dataset(config: Optional[SyntheticDataConfig] = None, 
                           num_examples: int = 50) -> List[EvaluationExample]:
    """Create a complete synthetic dataset for evaluation."""
    generator = SyntheticDataGenerator(config)
    return generator.generate_synthetic_examples(num_examples)


@traceable
def create_mixed_dataset(original_examples: List[EvaluationExample],
                        synthetic_ratio: float = 0.5,
                        config: Optional[SyntheticDataConfig] = None) -> List[EvaluationExample]:
    """Create a mixed dataset combining original and synthetic examples."""
    num_synthetic = int(len(original_examples) * synthetic_ratio)
    
    generator = SyntheticDataGenerator(config)
    synthetic_examples = generator.generate_synthetic_examples(num_synthetic)
    
    # Combine and shuffle
    mixed_dataset = list(original_examples) + synthetic_examples
    random.shuffle(mixed_dataset)
    
    return mixed_dataset


def save_synthetic_dataset(examples: List[EvaluationExample], filepath: str) -> None:
    """Save synthetic dataset with metadata about generation."""
    data = {
        "metadata": {
            "total_examples": len(examples),
            "synthetic_count": sum(1 for ex in examples if ex.metadata.get("synthetic", False)),
            "generation_timestamp": "generated_synthetic_data",
            "query_type_distribution": {},
            "difficulty_distribution": {}
        },
        "examples": []
    }
    
    # Calculate distributions
    for example in examples:
        qt = example.query_type
        diff = example.difficulty
        data["metadata"]["query_type_distribution"][qt] = data["metadata"]["query_type_distribution"].get(qt, 0) + 1
        data["metadata"]["difficulty_distribution"][diff] = data["metadata"]["difficulty_distribution"].get(diff, 0) + 1
    
    # Convert examples to dict format
    for example in examples:
        data["examples"].append(asdict(example))
    
    # Save to file
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    config = SyntheticDataConfig(
        num_examples_per_category=3,
        difficulty_distribution={"easy": 0.4, "medium": 0.4, "hard": 0.2}
    )
    
    # Generate synthetic dataset
    synthetic_examples = create_synthetic_dataset(config, num_examples=30)
    
    # Save dataset
    save_synthetic_dataset(
        synthetic_examples, 
        "data/evaluation/synthetic_rag_dataset.json"
    )
    
    print(f"Generated {len(synthetic_examples)} synthetic examples")
    
    # Display summary
    query_types = {}
    difficulties = {}
    
    for example in synthetic_examples:
        qt = example.query_type
        diff = example.difficulty
        query_types[qt] = query_types.get(qt, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"Query types: {query_types}")
    print(f"Difficulties: {difficulties}")