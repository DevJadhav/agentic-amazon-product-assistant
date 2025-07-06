# Notebooks Documentation

This directory contains Jupyter notebooks for data processing, analysis, and visualization of the Amazon Electronics dataset. Each notebook serves a specific purpose in the data science pipeline.

## Notebook Overview

### 1. `data_preprocessing.ipynb`
**Purpose**: Data extraction and preprocessing pipeline
**Status**: Complete with processed outputs
**Size**: ~164KB (7,108 lines)

#### Functionality:
- **Raw Data Processing**: Extracts and processes large Amazon Reviews 2023 dataset (25GB+)
- **Product Selection**: Identifies top 1,000 electronics products by review volume
- **Review Sampling**: Samples 20 representative reviews per product (20,000 total)
- **Data Cleaning**: Handles missing values, validates data integrity
- **RAG Optimization**: Creates optimized documents for retrieval-augmented generation

#### Key Operations:
1. **Review Counting**: Scans ~43M reviews to count reviews per product
2. **Product Filtering**: Selects products with ≥10 reviews, takes top 1,000
3. **Metadata Extraction**: Extracts product details from meta_Electronics.jsonl
4. **Review Sampling**: Randomly samples 20 reviews per selected product
5. **RAG Document Creation**: Generates paired product-review summary documents
6. **Quality Validation**: Ensures data integrity and completeness

#### Outputs Generated:
- `electronics_top1000_products.jsonl` (1,000 product records)
- `electronics_top1000_products_reviews.jsonl` (20,000 review records)  
- `electronics_rag_documents.jsonl` (2,000 RAG-optimized records)
- `dataset_summary.json` (processing metadata and statistics)

#### Performance Metrics:
- **Processing Time**: ~2-3 minutes for full dataset
- **Memory Usage**: Optimized for streaming processing
- **Data Reduction**: 25GB → ~20MB (99.9% reduction)
- **Quality**: 100% data integrity validation

---

### 2. `data_visualization.ipynb`
**Purpose**: Interactive data exploration and visualization
**Status**: Template notebook for analysis
**Size**: ~726KB (981 lines)

#### Functionality:
- **Comprehensive EDA**: Exploratory data analysis across multiple dimensions
- **Interactive Visualizations**: Plotly and matplotlib charts for data insights
- **Statistical Analysis**: Distribution analysis, correlation studies
- **Trend Identification**: Temporal patterns and seasonal trends

#### Analysis Categories:

##### **Review Distribution Analysis**
- Review count distribution per product
- Rating distribution (1-5 stars)
- Average product ratings analysis
- Review count vs rating correlation

##### **Price Analysis**
- Price range distribution
- Price vs popularity correlation
- Price vs rating analysis
- Market segmentation by price tiers

##### **Category Analysis**
- Product category distribution
- Hierarchical category exploration
- Category popularity metrics
- Cross-category comparisons

##### **Rating Patterns**
- Rating distribution analysis
- Rating trends over time
- Store performance by ratings
- Verified vs unverified purchase patterns

##### **Temporal Analysis**
- Review trends over 20 years (2003-2023)
- Seasonal patterns and cycles
- Day-of-week analysis
- Monthly activity patterns

##### **Text Analysis**
- Review length distribution
- Title and description characteristics
- Content quality metrics
- Text complexity analysis

##### **Store & Brand Analysis**
- Top performing stores
- Brand distribution
- Store review patterns
- Market share analysis

#### Visualization Types:
- **Statistical Charts**: Histograms, box plots, scatter plots
- **Interactive Plots**: Plotly dashboards with zoom/filter capabilities
- **Time Series**: Temporal trend analysis
- **Categorical**: Bar charts, pie charts, treemaps
- **Correlation**: Heatmaps, correlation matrices

---

### 3. `data_visualization_executed.ipynb`
**Purpose**: Complete executed analysis with all outputs
**Status**: Fully executed with results and insights
**Size**: ~888KB (1,155 lines)

#### Key Differences from `data_visualization.ipynb`:
- **Complete Execution**: All cells executed with outputs visible
- **Generated Visualizations**: All charts rendered and saved
- **Statistical Results**: Complete analysis results and summaries
- **Performance Metrics**: Actual runtime and memory usage data

#### Comprehensive Analysis Results:

##### **Dataset Statistics**:
- **Products**: 1,000 electronics items
- **Reviews**: 20,000 customer reviews  
- **Time Span**: 20 years (2003-2023)
- **Review Range**: 3,453 to 178,239 reviews per product
- **Average Rating**: 4.29/5.0 stars

##### **Key Insights Generated**:
- **Rating Distribution**: 68.2% five-star, 8.1% one-star reviews
- **Price Analysis**: Median price $26.99, range $3.49-$1,175.35
- **Category Depth**: Up to 7 hierarchical levels, 312 unique categories
- **Temporal Patterns**: Tuesday most active day, January most active month
- **Recent Activity**: 37.8% of reviews from 2020 onwards

##### **Generated Visualizations**:
- Review distribution histograms
- Price analysis scatter plots  
- Category hierarchy treemaps
- Temporal trend line charts
- Rating correlation heatmaps
- Store performance bar charts

#### Reference Usage:
This notebook serves as the **definitive analysis reference** cited in:
- Project documentation
- README.md files
- Research presentations
- Data quality reports

---

### 4. `verify_api_keys.ipynb`
**Purpose**: API configuration validation and testing
**Status**: Functional testing notebook
**Size**: ~5.4KB (238 lines)

#### Functionality:
- **Multi-Provider Testing**: Validates OpenAI, Google Gemini, and Groq API keys
- **Connection Verification**: Tests actual API connectivity
- **Response Validation**: Confirms proper API responses
- **Environment Setup**: Guides proper .env configuration

#### API Providers Tested:

##### **OpenAI GPT**
- **Model**: gpt-4o-mini
- **Test**: Simple chat completion
- **Validation**: Response quality and format

##### **Google Gemini**
- **Model**: gemini-2.0-flash
- **Test**: Content generation
- **Validation**: API response structure

##### **Groq Llama**
- **Model**: llama-3.3-70b-versatile  
- **Test**: Chat completion
- **Validation**: Response speed and quality

#### Key Features:
- **Environment Variable Loading**: Secure API key management
- **Interactive Prompts**: Fallback for missing keys
- **Error Handling**: Clear error messages for troubleshooting
- **Success Confirmation**: Visual confirmation of working APIs

#### Usage Scenarios:
- **Initial Setup**: First-time API configuration
- **Troubleshooting**: Debugging connection issues
- **Provider Testing**: Comparing API responses
- **Environment Validation**: CI/CD pipeline testing

## Execution Order

For optimal results, run notebooks in this sequence:

1. **`verify_api_keys.ipynb`** - Validate API configuration
2. **`data_preprocessing.ipynb`** - Process raw data (if needed)
3. **`data_visualization.ipynb`** - Explore and analyze data
4. **`data_visualization_executed.ipynb`** - Reference for completed analysis

## Dependencies

### Required Python Packages:
```python
# Data Processing
pandas>=2.3.0
numpy>=2.3.1
json
pathlib
gzip

# Visualization
matplotlib>=3.10.3
seaborn>=0.13.2
plotly>=6.2.0

# API Integration
openai
groq
google-genai

# Utilities
tqdm>=4.67.1
jupyter>=1.1.1
ipykernel>=6.29.5
```

### Environment Variables:
```bash
# Required for API testing
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key

# Optional for tracing
WANDB_API_KEY=your_wandb_key
```

## File Relationships

```
Raw Data (25GB)
    │
    ├── Electronics.jsonl (reviews)
    ├── meta_Electronics.jsonl (products)
    │
    └── data_preprocessing.ipynb
            │
            ├── Generates processed files
            │
            └── data_visualization.ipynb
                    │
                    ├── Analyzes processed data
                    │
                    └── data_visualization_executed.ipynb
                            │
                            └── Complete analysis results
```

## Performance Notes

### Execution Times (Approximate):
- **verify_api_keys.ipynb**: <30 seconds
- **data_preprocessing.ipynb**: 2-3 minutes (full dataset)
- **data_visualization.ipynb**: 1-2 minutes (depends on visualizations)
- **data_visualization_executed.ipynb**: Reference only (pre-executed)

### Memory Requirements:
- **Minimum**: 4GB RAM for basic operations
- **Recommended**: 8GB RAM for smooth processing
- **Optimal**: 16GB RAM for large dataset operations

### Storage Requirements:
- **Input Data**: ~25GB (raw Amazon dataset)
- **Processed Data**: ~20MB (optimized subset)
- **Notebook Outputs**: ~2MB (visualizations and results)

## Troubleshooting

### Common Issues:

1. **Missing Raw Data**:
   ```
   Error: ../data/Electronics.jsonl not found
   Solution: Download Amazon Reviews 2023 dataset
   ```

2. **API Key Errors**:
   ```
   Error: No API key provided
   Solution: Configure .env file with valid API keys
   ```

3. **Memory Issues**:
   ```
   Error: MemoryError during processing
   Solution: Increase available RAM or use streaming processing
   ```

4. **Package Import Errors**:
   ```
   Error: ModuleNotFoundError
   Solution: Run `uv sync` to install dependencies
   ```

## Output Artifacts

### Generated Files:
- **Processed Datasets**: 4 JSONL/JSON files (~20MB total)
- **Visualizations**: PNG/SVG charts (embedded in notebooks)
- **Analysis Reports**: Statistical summaries and insights
- **Processing Logs**: Execution metadata and performance metrics

### Key Metrics Tracked:
- Data quality scores
- Processing performance
- Analysis completeness
- Visualization coverage
- API response validation

## Academic Citation

These notebooks process data from:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
``` 