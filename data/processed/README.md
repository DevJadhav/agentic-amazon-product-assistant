# Processed Data Documentation

This directory contains processed and optimized datasets derived from the Amazon Reviews 2023 - Electronics Category. All files are in JSONL (JSON Lines) format, with one JSON object per line.

## Dataset Overview

- **Source**: Amazon Reviews 2023 - Electronics Category
- **Processing Date**: 2025-06-26
- **Total Products**: 1,000 carefully selected electronics products
- **Total Reviews**: 20,000 customer reviews (20 reviews per product)
- **Date Range**: 2003-2023 (20 years of review data)

## File Structure

### 1. `electronics_top1000_products.jsonl`
**Purpose**: Core product metadata and specifications
**Records**: 1,000 product records
**Size**: ~5.4MB

#### Schema:
```json
{
  "_schema_description": {
    "main_category": "Primary product category",
    "title": "Product title",
    "average_rating": "Average customer rating (1-5)",
    "rating_number": "Total number of ratings",
    "features": "Product feature list",
    "description": "Product descriptions",
    "price": "Product price (USD)",
    "images": "Product image URLs with variants",
    "videos": "Product video content",
    "store": "Seller/store name",
    "categories": "Hierarchical category path",
    "details": "Technical specifications",
    "parent_asin": "Amazon Standard Identification Number",
    "bought_together": "Frequently bought together items",
    "review_count": "Number of reviews for this product"
  },
  "main_category": "string",
  "title": "string",
  "average_rating": 4.5,
  "rating_number": 1234,
  "features": ["feature1", "feature2"],
  "description": ["description1", "description2"],
  "price": 29.99,
  "images": [
    {
      "thumb": "url",
      "large": "url", 
      "variant": "string",
      "hi_res": "url"
    }
  ],
  "videos": [
    {
      "title": "string",
      "url": "url",
      "user_id": "string"
    }
  ],
  "store": "string",
  "categories": ["category1", "category2"],
  "details": {
    "Brand": "string",
    "Connector Type": "string"
  },
  "parent_asin": "string",
  "bought_together": null,
  "review_count": 20
}
```

#### Key Features:
- Complete product metadata from Amazon
- Hierarchical category structure (up to 7 levels deep)
- Rich media content (images, videos)
- Technical specifications and details
- Pricing and rating information

---

### 2. `electronics_top1000_products_reviews.jsonl`
**Purpose**: Individual customer reviews for all products
**Records**: 20,000 review records
**Size**: ~8.9MB

#### Schema:
```json
{
  "_schema_description": {
    "asin": "Product ASIN",
    "parent_asin": "Parent product ASIN",
    "rating": "Review rating (1-5 stars)",
    "title": "Review title/headline",
    "text": "Full review text content",
    "timestamp": "Unix timestamp (milliseconds)",
    "verified_purchase": "Amazon verified purchase flag",
    "helpful_vote": "Number of helpful votes"
  },
  "asin": "string",
  "parent_asin": "string",
  "rating": 5,
  "title": "string",
  "text": "string",
  "timestamp": 1640995200000,
  "verified_purchase": true,
  "helpful_vote": 3
}
```

#### Key Features:
- 20 reviews sampled per product for balanced representation
- Verified purchase status for authenticity
- Temporal data spanning 2003-2023
- Community engagement metrics (helpful votes)
- Complete review text for sentiment analysis

---

### 3. `electronics_rag_documents.jsonl`
**Purpose**: RAG-optimized documents for retrieval systems
**Records**: 2,000 records (1,000 products + 1,000 review summaries)
**Size**: ~5.9MB

#### Product Records Schema:
```json
{
  "_schema_description": {
    "id": "Unique document ID",
    "type": "Document type",
    "parent_asin": "Product ASIN",
    "title": "Product title",
    "description": "Product description",
    "features": "Concatenated features",
    "price": "Product price",
    "average_rating": "Average rating",
    "rating_number": "Total ratings",
    "review_count": "Review count",
    "store": "Store name",
    "categories": "Category hierarchy",
    "details": "Product specifications",
    "content": "Searchable content summary"
  },
  "id": "product_B08N5WRWNW",
  "type": "product",
  "parent_asin": "string",
  "title": "string",
  "description": "string",
  "features": "string",
  "price": 29.99,
  "average_rating": 4.5,
  "rating_number": 1234,
  "review_count": 20,
  "store": "string",
  "categories": ["Electronics", "Computers"],
  "details": {},
  "content": "string"
}
```

#### Review Summary Records Schema:
```json
{
  "_schema_description": {
    "id": "Unique document ID",
    "type": "Document type",
    "parent_asin": "Product ASIN",
    "product_title": "Associated product title",
    "total_reviews": "Number of reviews summarized",
    "positive_reviews": "Count of positive reviews",
    "negative_reviews": "Count of negative reviews",
    "content": "Aggregated review insights"
  },
  "id": "reviews_B08N5WRWNW",
  "type": "review_summary",
  "parent_asin": "string",
  "product_title": "string",
  "total_reviews": 20,
  "positive_reviews": 15,
  "negative_reviews": 5,
  "content": "string"
}
```

#### Key Features:
- **Paired Structure**: Each product has exactly one review summary
- **Optimized for RAG**: Condensed content for efficient retrieval
- **Linked Records**: Connected via `parent_asin` field
- **Searchable Content**: Pre-processed text for vector embeddings
- **Sentiment Analysis**: Positive/negative review breakdown

---

### 4. `dataset_summary.json`
**Purpose**: Dataset statistics and metadata
**Records**: 1 summary record
**Size**: ~668B

#### Schema:
```json
{
  "_schema_description": {
    "dataset_info": "Metadata about dataset source and processing",
    "statistics": "Statistical summaries of the processed data"
  },
  "dataset_info": {
    "source": "Amazon Reviews 2023 - Electronics Category",
    "citation": "Hou et al. (2024) arXiv:2403.03952",
    "processing_date": "2025-06-26T00:00:00Z",
    "selection_criteria": {
      "target_products": 1000,
      "min_reviews_per_product": 10,
      "max_reviews_per_product": 20
    }
  },
  "statistics": {
    "total_products": 1000,
    "total_reviews": 20000,
    "products_with_metadata": 1000,
    "products_with_reviews": 1000,
    "review_count_stats": {
      "min": 3453,
      "max": 178239,
      "mean": 8022.5,
      "median": 5332.0
    }
  }
}
```

#### Key Features:
- Complete processing metadata
- Data quality metrics
- Selection criteria documentation
- Statistical summaries

## Data Relationships

```
electronics_top1000_products.jsonl (1,000 products)
         │
         ├── Linked by parent_asin
         │
electronics_top1000_products_reviews.jsonl (20,000 reviews)
         │
         └── Summarized into
         
electronics_rag_documents.jsonl (2,000 RAG docs)
         ├── 1,000 product records
         └── 1,000 review summary records
```

## Usage Examples

### Loading Product Data
```python
import json
import pandas as pd

# Load products
products = []
with open('electronics_top1000_products.jsonl', 'r') as f:
    for line in f:
        products.append(json.loads(line.strip()))

df_products = pd.DataFrame(products)
print(f"Loaded {len(df_products)} products")
```

### Loading Reviews
```python
# Load reviews
reviews = []
with open('electronics_top1000_products_reviews.jsonl', 'r') as f:
    for line in f:
        reviews.append(json.loads(line.strip()))

df_reviews = pd.DataFrame(reviews)
print(f"Loaded {len(df_reviews)} reviews")
```

### Loading RAG Documents
```python
# Load RAG documents
rag_docs = []
with open('electronics_rag_documents.jsonl', 'r') as f:
    for line in f:
        rag_docs.append(json.loads(line.strip()))

# Separate by type
products = [doc for doc in rag_docs if doc['type'] == 'product']
summaries = [doc for doc in rag_docs if doc['type'] == 'review_summary']

print(f"Loaded {len(products)} product docs and {len(summaries)} review summaries")
```

## Data Quality Notes

1. **Completeness**: 100% completeness for ratings, stores, categories; 77.2% for pricing
2. **Consistency**: All products have exactly 20 reviews in the reviews file
3. **Validation**: All ASINs are validated and cross-referenced
4. **Deduplication**: No duplicate products or reviews
5. **Temporal Coverage**: Reviews span 20 years (2003-2023)

## Processing Pipeline

The data was processed through the following steps:

1. **Raw Data Ingestion**: Amazon Reviews 2023 dataset (25GB+)
2. **Product Selection**: Top 1000 products by review volume and quality
3. **Review Sampling**: 20 representative reviews per product
4. **Data Cleaning**: Missing value handling, validation
5. **RAG Optimization**: Content summarization and formatting
6. **Quality Assurance**: Statistical validation and integrity checks

## File Sizes and Performance

| File | Records | Size | Load Time* |
|------|---------|------|------------|
| products.jsonl | 1,000 | 5.4MB | ~0.5s |
| reviews.jsonl | 20,000 | 8.9MB | ~1.2s |
| rag_documents.jsonl | 2,000 | 5.9MB | ~0.8s |
| dataset_summary.json | 1 | 668B | <0.1s |

*Approximate load times on standard hardware

## Citation

If you use this dataset, please cite:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
``` 