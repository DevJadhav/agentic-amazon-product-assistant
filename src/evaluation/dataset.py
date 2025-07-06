"""
Evaluation dataset creation and management for RAG system.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EvaluationExample:
    """Single evaluation example with query, expected answer, and metadata."""
    query: str
    expected_answer: str
    query_type: str
    expected_products: List[str]  # Product IDs that should be retrieved
    expected_topics: List[str]    # Topics that should be covered
    difficulty: str               # easy, medium, hard
    metadata: Dict[str, Any]

def create_evaluation_dataset() -> List[EvaluationExample]:
    """Create comprehensive evaluation dataset for RAG system."""
    
    examples = [
        # Product Information Queries
        EvaluationExample(
            query="What are the key features of iPhone charging cables?",
            expected_answer="iPhone charging cables typically feature Lightning connectors, various lengths (3ft, 6ft, 10ft), MFi certification for compatibility, durable materials like braided nylon, fast charging support, and data transfer capabilities. Quality varies by brand with Apple and certified third-party options being most reliable.",
            query_type="product_info",
            expected_products=["lightning_cable", "iphone_charger", "apple_cable"],
            expected_topics=["features", "compatibility", "durability", "charging_speed"],
            difficulty="easy",
            metadata={"category": "electronics", "subcategory": "cables"}
        ),
        
        EvaluationExample(
            query="Tell me about Fire TV Stick performance and capabilities",
            expected_answer="Fire TV Stick offers 1080p HD streaming, supports popular apps like Netflix, Prime Video, Disney+, has Alexa Voice Remote, WiFi connectivity, and compact design. Performance varies by model - 4K versions support UHD content, while basic models handle HD well. Generally good for streaming but may have some buffering issues with high-demand content.",
            query_type="product_info", 
            expected_products=["fire_tv_stick", "amazon_fire_tv", "streaming_device"],
            expected_topics=["streaming_quality", "app_support", "performance", "alexa_integration"],
            difficulty="medium",
            metadata={"category": "electronics", "subcategory": "streaming"}
        ),

        # Product Review Queries
        EvaluationExample(
            query="What do customers say about laptop backpack durability?",
            expected_answer="Customer reviews on laptop backpack durability are mixed. Positive feedback highlights reinforced stitching, water-resistant materials, and long-lasting zippers. Common complaints include strap wear after 6-12 months, zipper failures, and fabric fraying. Higher-end brands like SwissGear and Targus generally receive better durability ratings than budget options.",
            query_type="product_reviews",
            expected_products=["laptop_backpack", "computer_bag", "backpack"],
            expected_topics=["durability", "stitching", "zippers", "materials", "brand_comparison"],
            difficulty="medium",
            metadata={"category": "electronics", "subcategory": "accessories"}
        ),

        EvaluationExample(
            query="What are people's experiences with Bluetooth earbuds battery life?",
            expected_answer="Bluetooth earbuds battery life experiences vary significantly. Premium models (AirPods, Sony) typically deliver 6-8 hours per charge with case providing 20-30 hours total. Budget options often provide 3-5 hours. Common complaints include battery degradation after 1-2 years, inconsistent battery levels between earbuds, and charging case issues. Users appreciate quick charging features.",
            query_type="product_reviews",
            expected_products=["bluetooth_earbuds", "wireless_earphones", "airpods"],
            expected_topics=["battery_life", "charging_case", "degradation", "brand_differences"],
            difficulty="medium",
            metadata={"category": "electronics", "subcategory": "audio"}
        ),

        # Product Comparison Queries
        EvaluationExample(
            query="Compare Ethernet cables vs USB cables for data transfer",
            expected_answer="Ethernet cables are designed for network connectivity with speeds up to 10Gbps (Cat6a/Cat7), longer distances (100m), and stable connections. USB cables prioritize device connectivity with varying speeds (USB 2.0: 480Mbps, USB 3.0: 5Gbps, USB-C: up to 40Gbps), shorter distances (5m typical), and power delivery capabilities. For pure data transfer, high-speed USB-C cables can match Ethernet, but Ethernet provides more consistent network performance.",
            query_type="product_comparison",
            expected_products=["ethernet_cable", "usb_cable", "network_cable"],
            expected_topics=["speed", "distance", "reliability", "use_cases", "specifications"],
            difficulty="hard",
            metadata={"category": "electronics", "subcategory": "cables"}
        ),

        EvaluationExample(
            query="Samsung Galaxy tablets vs iPad - which is better for students?",
            expected_answer="For students, iPads excel in app ecosystem, battery life (10-12 hours), stylus support (Apple Pencil), and integration with other Apple devices. Samsung Galaxy tablets offer better multitasking, file management, expandable storage, and often lower prices. iPads are better for creative work and note-taking, while Galaxy tablets provide more flexibility for productivity tasks and document management.",
            query_type="product_comparison", 
            expected_products=["ipad", "samsung_tablet", "galaxy_tab"],
            expected_topics=["student_use", "battery_life", "apps", "productivity", "price", "stylus"],
            difficulty="hard",
            metadata={"category": "electronics", "subcategory": "tablets"}
        ),

        # Complaint Analysis Queries
        EvaluationExample(
            query="What are the main complaints about wireless routers?",
            expected_answer="Main wireless router complaints include inconsistent WiFi coverage, especially in larger homes; frequent disconnections requiring restarts; slow speeds compared to advertised specifications; complex setup processes; overheating issues; poor customer support; and firmware problems. Users also report range limitations, interference from other devices, and difficulty with parental controls.",
            query_type="product_complaints",
            expected_products=["wireless_router", "wifi_router", "router"],
            expected_topics=["coverage", "disconnections", "speed", "setup", "overheating", "range"],
            difficulty="medium",
            metadata={"category": "electronics", "subcategory": "networking"}
        ),

        EvaluationExample(
            query="What problems do people have with smartphone chargers?",
            expected_answer="Common smartphone charger problems include cables breaking near connectors, slow charging speeds, overheating during use, compatibility issues with cases, loose connections that require cable positioning, short lifespan (3-6 months), and non-MFi certified cables causing device warnings. Users also report inconsistent charging speeds and cables that stop working intermittently.",
            query_type="product_complaints",
            expected_products=["phone_charger", "smartphone_cable", "charging_cable"],
            expected_topics=["durability", "charging_speed", "overheating", "compatibility", "lifespan"],
            difficulty="easy",
            metadata={"category": "electronics", "subcategory": "charging"}
        ),

        # Product Recommendation Queries
        EvaluationExample(
            query="Recommend a budget-friendly alternative to expensive noise-canceling headphones",
            expected_answer="Budget-friendly alternatives to expensive noise-canceling headphones include TaoTronics SoundSurge 60 ($60), Anker Soundcore Life Q20 ($40), and Cowin E7 ($50). These offer decent active noise cancellation, 20-30 hour battery life, and comfortable fit. While not matching premium brands like Sony or Bose in sound quality, they provide 70-80% of the performance at 20% of the cost.",
            query_type="product_recommendation",
            expected_products=["budget_headphones", "noise_canceling_headphones", "affordable_headphones"],
            expected_topics=["budget", "alternatives", "value", "features", "price_comparison"],
            difficulty="medium",
            metadata={"category": "electronics", "subcategory": "audio"}
        ),

        EvaluationExample(
            query="Suggest affordable laptop alternatives for college students under $500",
            expected_answer="Affordable laptop alternatives under $500 for college students include refurbished business laptops (Lenovo ThinkPad T series, Dell Latitude), Chromebooks (ASUS Chromebook Flip, Acer Chromebook Spin), and budget Windows laptops (Acer Aspire 5, HP Pavilion). Consider 8GB RAM minimum, SSD storage, and good battery life. Refurbished ThinkPads offer best build quality, while Chromebooks excel for basic tasks and battery life.",
            query_type="product_recommendation",
            expected_products=["budget_laptop", "student_laptop", "cheap_laptop"],
            expected_topics=["budget", "students", "specifications", "refurbished", "chromebook"],
            difficulty="hard",
            metadata={"category": "electronics", "subcategory": "computers"}
        ),

        # Use Case Queries
        EvaluationExample(
            query="Is a mechanical keyboard good for programming?",
            expected_answer="Yes, mechanical keyboards are excellent for programming. They offer tactile feedback that improves typing accuracy, customizable key switches for different feel preferences, durability (50-100 million keystrokes), and often include programmable keys. Popular switches for programming include Cherry MX Blue (tactile/clicky), Brown (tactile/quiet), and Red (linear/smooth). Benefits include reduced finger fatigue during long coding sessions and improved typing speed.",
            query_type="use_case",
            expected_products=["mechanical_keyboard", "programming_keyboard", "gaming_keyboard"],
            expected_topics=["programming", "tactile_feedback", "durability", "switch_types", "ergonomics"],
            difficulty="medium",
            metadata={"category": "electronics", "subcategory": "peripherals"}
        ),

        EvaluationExample(
            query="Can a smartwatch be used for fitness tracking effectively?",
            expected_answer="Yes, smartwatches are very effective for fitness tracking. They monitor heart rate, steps, calories, sleep patterns, and specific workouts. Advanced models offer GPS tracking, water resistance for swimming, ECG monitoring, and SpO2 sensors. Popular options include Apple Watch (comprehensive health features), Garmin (specialized fitness), and Fitbit (focused on health tracking). Battery life varies from 1-7 days depending on features used.",
            query_type="use_case",
            expected_products=["smartwatch", "fitness_tracker", "apple_watch"],
            expected_topics=["fitness_tracking", "health_monitoring", "GPS", "battery_life", "features"],
            difficulty="easy",
            metadata={"category": "electronics", "subcategory": "wearables"}
        ),

        # Complex Multi-faceted Queries
        EvaluationExample(
            query="Best wireless gaming headset under $150 with good microphone for team communication",
            expected_answer="Top wireless gaming headsets under $150 with quality microphones include SteelSeries Arctis 7 ($140), HyperX Cloud Flight ($120), and Corsair HS70 Pro ($100). Key features: 2.4GHz wireless connection, 15-24 hour battery life, detachable/flip-to-mute microphones, surround sound support. Arctis 7 offers best overall quality, Cloud Flight provides excellent comfort, and HS70 Pro gives best value. All feature clear voice communication essential for team gaming.",
            query_type="product_recommendation",
            expected_products=["gaming_headset", "wireless_headset", "gaming_microphone"],
            expected_topics=["wireless", "gaming", "microphone_quality", "budget", "battery_life", "comfort"],
            difficulty="hard",
            metadata={"category": "electronics", "subcategory": "gaming"}
        ),

        EvaluationExample(
            query="What are the pros and cons of USB-C hubs for MacBook Pro users?",
            expected_answer="USB-C hubs for MacBook Pro offer expanded connectivity (HDMI, USB-A, SD cards, Ethernet) and power delivery passthrough. Pros: Restores traditional ports, enables dual monitor setups, maintains charging while connected, compact design. Cons: Can cause overheating, some units have connectivity issues, power delivery limitations, compatibility problems with certain peripherals, and quality varies significantly between brands. Premium brands like CalDigit and OWC offer better reliability.",
            query_type="product_info",
            expected_products=["usb_c_hub", "macbook_hub", "usb_hub"],
            expected_topics=["macbook_compatibility", "connectivity", "overheating", "power_delivery", "reliability"],
            difficulty="hard",
            metadata={"category": "electronics", "subcategory": "accessories"}
        )
    ]
    
    return examples

def save_evaluation_dataset(examples: List[EvaluationExample], filepath: str) -> None:
    """Save evaluation dataset to JSON file."""
    data = []
    for example in examples:
        data.append({
            "query": example.query,
            "expected_answer": example.expected_answer,
            "query_type": example.query_type,
            "expected_products": example.expected_products,
            "expected_topics": example.expected_topics,
            "difficulty": example.difficulty,
            "metadata": example.metadata
        })
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_evaluation_dataset(filepath: str) -> List[EvaluationExample]:
    """Load evaluation dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        examples.append(EvaluationExample(
            query=item["query"],
            expected_answer=item["expected_answer"], 
            query_type=item["query_type"],
            expected_products=item["expected_products"],
            expected_topics=item["expected_topics"],
            difficulty=item["difficulty"],
            metadata=item["metadata"]
        ))
    
    return examples

if __name__ == "__main__":
    # Create and save evaluation dataset
    examples = create_evaluation_dataset()
    save_evaluation_dataset(examples, "data/evaluation/rag_evaluation_dataset.json")
    print(f"Created evaluation dataset with {len(examples)} examples")
    
    # Display summary
    query_types = {}
    difficulties = {}
    
    for example in examples:
        query_types[example.query_type] = query_types.get(example.query_type, 0) + 1
        difficulties[example.difficulty] = difficulties.get(example.difficulty, 0) + 1
    
    print(f"\nQuery types: {query_types}")
    print(f"Difficulties: {difficulties}")