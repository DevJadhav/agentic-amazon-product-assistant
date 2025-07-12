"""
Advanced Data Processing Pipeline for Amazon Product Assistant
Provides large-scale data processing with temporal trends, category insights, and advanced analytics.
"""

import json
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import gzip
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    target_products: int = 1000
    min_reviews_per_product: int = 10
    max_reviews_per_product: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    temporal_window_days: int = 365
    category_depth: int = 3
    price_buckets: List[float] = None
    enable_async: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        if self.price_buckets is None:
            self.price_buckets = [0, 25, 50, 100, 200, 500, 1000, float('inf')]

class AdvancedDataProcessor:
    """Advanced data processing pipeline with temporal trends and category insights."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.embedding_model = None
        self.product_metadata = []
        self.reviews_data = []
        self.temporal_trends = {}
        self.category_insights = {}
        self.rating_patterns = {}
        
        # Initialize thread pool for async operations
        if config.enable_async:
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
            self.process_executor = ProcessPoolExecutor(max_workers=config.max_workers)
        else:
            self.executor = None
            self.process_executor = None
    
    def load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
    def process_temporal_trends(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal trends in reviews and ratings."""
        logger.info("Processing temporal trends...")
        
        # Convert timestamps to datetime
        for review in reviews:
            if 'timestamp' in review:
                try:
                    review['datetime'] = datetime.fromtimestamp(review['timestamp'])
                except (ValueError, TypeError):
                    review['datetime'] = None
        
        # Filter reviews with valid timestamps
        valid_reviews = [r for r in reviews if r.get('datetime')]
        
        if not valid_reviews:
            return {}
        
        df = pd.DataFrame(valid_reviews)
        df['month_year'] = df['datetime'].dt.to_period('M')
        df['quarter'] = df['datetime'].dt.to_period('Q')
        df['year'] = df['datetime'].dt.year
        
        trends = {
            'monthly_review_counts': df.groupby('month_year').size().to_dict(),
            'monthly_avg_ratings': df.groupby('month_year')['rating'].mean().to_dict(),
            'quarterly_trends': {
                'review_counts': df.groupby('quarter').size().to_dict(),
                'avg_ratings': df.groupby('quarter')['rating'].mean().to_dict(),
                'rating_distribution': df.groupby(['quarter', 'rating']).size().unstack(fill_value=0).to_dict()
            },
            'yearly_trends': {
                'review_counts': df.groupby('year').size().to_dict(),
                'avg_ratings': df.groupby('year')['rating'].mean().to_dict()
            },
            'seasonal_patterns': self._analyze_seasonal_patterns(df),
            'review_velocity': self._calculate_review_velocity(df),
            'sentiment_trends': self._analyze_sentiment_trends(df)
        }
        
        return trends
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in reviews."""
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['quarter_name'] = df['datetime'].dt.quarter.map({
            1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)', 
            3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'
        })
        
        return {
            'monthly_patterns': df.groupby('month')['rating'].agg(['count', 'mean']).to_dict(),
            'weekly_patterns': df.groupby('day_of_week')['rating'].agg(['count', 'mean']).to_dict(),
            'quarterly_patterns': df.groupby('quarter_name')['rating'].agg(['count', 'mean']).to_dict(),
            'peak_review_months': df.groupby('month').size().nlargest(3).to_dict(),
            'low_review_months': df.groupby('month').size().nsmallest(3).to_dict()
        }
    
    def _calculate_review_velocity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate review velocity metrics."""
        # Group by product and calculate velocity
        velocity_data = {}
        
        for parent_asin in df['parent_asin'].unique():
            product_reviews = df[df['parent_asin'] == parent_asin].sort_values('datetime')
            
            if len(product_reviews) >= 2:
                first_review = product_reviews.iloc[0]['datetime']
                last_review = product_reviews.iloc[-1]['datetime']
                days_span = (last_review - first_review).days
                
                if days_span > 0:
                    velocity_data[parent_asin] = {
                        'reviews_per_day': len(product_reviews) / days_span,
                        'reviews_per_month': (len(product_reviews) / days_span) * 30,
                        'total_reviews': len(product_reviews),
                        'days_span': days_span,
                        'first_review': first_review.isoformat(),
                        'last_review': last_review.isoformat()
                    }
        
        # Calculate aggregate metrics
        if velocity_data:
            velocities = [v['reviews_per_day'] for v in velocity_data.values()]
            return {
                'product_velocities': velocity_data,
                'avg_velocity_per_day': np.mean(velocities),
                'median_velocity_per_day': np.median(velocities),
                'max_velocity_product': max(velocity_data.items(), key=lambda x: x[1]['reviews_per_day']),
                'velocity_distribution': np.histogram(velocities, bins=10)[0].tolist()
            }
        
        return {}
    
    def _analyze_sentiment_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        # Simple sentiment based on ratings
        df['sentiment'] = df['rating'].map({
            1: 'very_negative', 2: 'negative', 3: 'neutral', 
            4: 'positive', 5: 'very_positive'
        })
        
        # Monthly sentiment trends
        monthly_sentiment = df.groupby(['month_year', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculate sentiment scores (1-5 scale)
        monthly_scores = df.groupby('month_year')['rating'].agg(['mean', 'std', 'count'])
        
        return {
            'monthly_sentiment_distribution': monthly_sentiment.to_dict(),
            'monthly_sentiment_scores': monthly_scores.to_dict(),
            'sentiment_volatility': monthly_scores['std'].mean(),
            'positive_sentiment_trend': (monthly_sentiment[['positive', 'very_positive']].sum(axis=1) / 
                                       monthly_sentiment.sum(axis=1)).to_dict(),
            'negative_sentiment_trend': (monthly_sentiment[['negative', 'very_negative']].sum(axis=1) / 
                                       monthly_sentiment.sum(axis=1)).to_dict()
        }
    
    def analyze_category_insights(self, products: List[Dict], reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze comprehensive category insights."""
        logger.info("Analyzing category insights...")
        
        # Create mapping of products to reviews
        product_review_map = defaultdict(list)
        for review in reviews:
            product_review_map[review['parent_asin']].append(review)
        
        category_data = defaultdict(lambda: {
            'products': [],
            'total_reviews': 0,
            'avg_rating': 0,
            'price_range': {'min': float('inf'), 'max': 0},
            'top_features': Counter(),
            'review_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0}
        })
        
        for product in products:
            categories = product.get('categories', [])
            if not categories:
                continue
                
            # Process each category level
            for i, category in enumerate(categories[:self.config.category_depth]):
                cat_key = ' > '.join(categories[:i+1])
                cat_data = category_data[cat_key]
                
                cat_data['products'].append(product['parent_asin'])
                
                # Price analysis
                price = product.get('price')
                if price and isinstance(price, (int, float)):
                    cat_data['price_range']['min'] = min(cat_data['price_range']['min'], price)
                    cat_data['price_range']['max'] = max(cat_data['price_range']['max'], price)
                
                # Features analysis
                features = product.get('features', [])
                if features:
                    for feature in features:
                        # Extract key terms from features
                        words = feature.lower().split()
                        for word in words:
                            if len(word) > 3:  # Filter short words
                                cat_data['top_features'][word] += 1
                
                # Reviews analysis
                product_reviews = product_review_map.get(product['parent_asin'], [])
                cat_data['total_reviews'] += len(product_reviews)
                
                if product_reviews:
                    ratings = [r.get('rating', 0) for r in product_reviews]
                    cat_data['avg_rating'] = np.mean(ratings) if ratings else 0
                    
                    # Sentiment analysis
                    for rating in ratings:
                        if rating >= 4:
                            cat_data['review_sentiment']['positive'] += 1
                        elif rating <= 2:
                            cat_data['review_sentiment']['negative'] += 1
                        else:
                            cat_data['review_sentiment']['neutral'] += 1
        
        # Clean up and finalize category data
        final_category_data = {}
        for cat_key, cat_data in category_data.items():
            if cat_data['price_range']['min'] == float('inf'):
                cat_data['price_range'] = {'min': None, 'max': None}
            
            cat_data['product_count'] = len(set(cat_data['products']))
            cat_data['top_features'] = dict(cat_data['top_features'].most_common(10))
            cat_data['avg_reviews_per_product'] = (cat_data['total_reviews'] / cat_data['product_count'] 
                                                 if cat_data['product_count'] > 0 else 0)
            
            final_category_data[cat_key] = cat_data
        
        return {
            'category_breakdown': final_category_data,
            'category_performance': self._rank_categories(final_category_data),
            'category_trends': self._analyze_category_trends(final_category_data),
            'cross_category_insights': self._analyze_cross_category_patterns(final_category_data)
        }
    
    def _rank_categories(self, category_data: Dict) -> Dict[str, Any]:
        """Rank categories by various metrics."""
        categories = list(category_data.keys())
        
        rankings = {
            'by_product_count': sorted(categories, 
                                     key=lambda x: category_data[x]['product_count'], 
                                     reverse=True),
            'by_total_reviews': sorted(categories, 
                                     key=lambda x: category_data[x]['total_reviews'], 
                                     reverse=True),
            'by_avg_rating': sorted(categories, 
                                  key=lambda x: category_data[x]['avg_rating'], 
                                  reverse=True),
            'by_review_density': sorted(categories, 
                                      key=lambda x: category_data[x]['avg_reviews_per_product'], 
                                      reverse=True)
        }
        
        return rankings
    
    def _analyze_category_trends(self, category_data: Dict) -> Dict[str, Any]:
        """Analyze trends across categories."""
        # Calculate category metrics
        metrics = {}
        for cat, data in category_data.items():
            total_sentiment = sum(data['review_sentiment'].values())
            if total_sentiment > 0:
                positive_ratio = data['review_sentiment']['positive'] / total_sentiment
                negative_ratio = data['review_sentiment']['negative'] / total_sentiment
            else:
                positive_ratio = negative_ratio = 0
            
            metrics[cat] = {
                'positive_sentiment_ratio': positive_ratio,
                'negative_sentiment_ratio': negative_ratio,
                'sentiment_score': positive_ratio - negative_ratio,
                'engagement_score': data['avg_reviews_per_product'],
                'rating_score': data['avg_rating']
            }
        
        return {
            'sentiment_leaders': sorted(metrics.items(), 
                                      key=lambda x: x[1]['sentiment_score'], 
                                      reverse=True)[:10],
            'engagement_leaders': sorted(metrics.items(), 
                                       key=lambda x: x[1]['engagement_score'], 
                                       reverse=True)[:10],
            'quality_leaders': sorted(metrics.items(), 
                                    key=lambda x: x[1]['rating_score'], 
                                    reverse=True)[:10],
            'category_metrics': metrics
        }
    
    def _analyze_cross_category_patterns(self, category_data: Dict) -> Dict[str, Any]:
        """Analyze patterns across categories."""
        # Price distribution analysis
        price_by_category = {}
        for cat, data in category_data.items():
            if data['price_range']['min'] is not None:
                price_by_category[cat] = {
                    'min_price': data['price_range']['min'],
                    'max_price': data['price_range']['max'],
                    'price_range': data['price_range']['max'] - data['price_range']['min']
                }
        
        return {
            'price_patterns': price_by_category,
            'feature_overlap': self._analyze_feature_overlap(category_data),
            'category_correlations': self._calculate_category_correlations(category_data)
        }
    
    def _analyze_feature_overlap(self, category_data: Dict) -> Dict[str, Any]:
        """Analyze feature overlap between categories."""
        all_features = set()
        category_features = {}
        
        for cat, data in category_data.items():
            features = set(data['top_features'].keys())
            category_features[cat] = features
            all_features.update(features)
        
        # Calculate feature overlap matrix
        overlap_matrix = {}
        for cat1 in category_features:
            overlap_matrix[cat1] = {}
            for cat2 in category_features:
                if cat1 != cat2:
                    overlap = len(category_features[cat1].intersection(category_features[cat2]))
                    total = len(category_features[cat1].union(category_features[cat2]))
                    overlap_matrix[cat1][cat2] = overlap / total if total > 0 else 0
        
        return {
            'overlap_matrix': overlap_matrix,
            'most_common_features': Counter({feature: sum(1 for cat_features in category_features.values() 
                                                        if feature in cat_features) 
                                           for feature in all_features}).most_common(20)
        }
    
    def _calculate_category_correlations(self, category_data: Dict) -> Dict[str, Any]:
        """Calculate correlations between category metrics."""
        # Extract metrics for correlation analysis
        metrics_data = []
        categories = []
        
        for cat, data in category_data.items():
            if data['product_count'] > 5:  # Only include categories with sufficient data
                metrics_data.append([
                    data['product_count'],
                    data['total_reviews'],
                    data['avg_rating'],
                    data['avg_reviews_per_product'],
                    sum(data['review_sentiment'].values())
                ])
                categories.append(cat)
        
        if len(metrics_data) < 2:
            return {}
        
        # Calculate correlation matrix
        df_metrics = pd.DataFrame(metrics_data, 
                                columns=['product_count', 'total_reviews', 'avg_rating', 
                                       'avg_reviews_per_product', 'total_sentiment'])
        
        correlation_matrix = df_metrics.corr().to_dict()
        
        return {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': self._find_strong_correlations(correlation_matrix),
            'category_clusters': self._identify_category_clusters(df_metrics, categories)
        }
    
    def _find_strong_correlations(self, corr_matrix: Dict) -> List[Tuple[str, str, float]]:
        """Find strong correlations between metrics."""
        strong_corr = []
        metrics = list(corr_matrix.keys())
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics[i+1:], i+1):
                corr_value = corr_matrix[metric1][metric2]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_corr.append((metric1, metric2, corr_value))
        
        return sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)
    
    def _identify_category_clusters(self, df_metrics: pd.DataFrame, categories: List[str]) -> Dict[str, Any]:
        """Identify clusters of similar categories."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_metrics)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(5, len(categories)), random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Group categories by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[f'cluster_{label}'].append(categories[i])
            
            return dict(clusters)
        
        except ImportError:
            logger.warning("sklearn not available for clustering analysis")
            return {}
    
    def analyze_rating_patterns(self, products: List[Dict], reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze comprehensive rating patterns."""
        logger.info("Analyzing rating patterns...")
        
        # Create product-review mapping
        product_review_map = defaultdict(list)
        for review in reviews:
            product_review_map[review['parent_asin']].append(review)
        
        rating_analysis = {
            'overall_distribution': Counter(),
            'price_vs_rating': [],
            'volume_vs_rating': [],
            'temporal_rating_trends': {},
            'product_rating_patterns': {},
            'rating_consistency': {},
            'outlier_analysis': {}
        }
        
        # Overall rating distribution
        all_ratings = [r.get('rating', 0) for r in reviews]
        rating_analysis['overall_distribution'] = dict(Counter(all_ratings))
        
        # Price vs rating analysis
        for product in products:
            parent_asin = product['parent_asin']
            price = product.get('price')
            product_reviews = product_review_map.get(parent_asin, [])
            
            if product_reviews and price:
                avg_rating = np.mean([r.get('rating', 0) for r in product_reviews])
                rating_analysis['price_vs_rating'].append({
                    'parent_asin': parent_asin,
                    'price': price,
                    'avg_rating': avg_rating,
                    'review_count': len(product_reviews)
                })
        
        # Volume vs rating analysis
        for parent_asin, product_reviews in product_review_map.items():
            if product_reviews:
                ratings = [r.get('rating', 0) for r in product_reviews]
                rating_analysis['volume_vs_rating'].append({
                    'parent_asin': parent_asin,
                    'review_count': len(ratings),
                    'avg_rating': np.mean(ratings),
                    'rating_std': np.std(ratings),
                    'rating_range': max(ratings) - min(ratings) if ratings else 0
                })
        
        # Rating consistency analysis
        rating_analysis['rating_consistency'] = self._analyze_rating_consistency(product_review_map)
        
        # Outlier analysis
        rating_analysis['outlier_analysis'] = self._identify_rating_outliers(rating_analysis['volume_vs_rating'])
        
        return rating_analysis
    
    def _analyze_rating_consistency(self, product_review_map: Dict) -> Dict[str, Any]:
        """Analyze rating consistency across products."""
        consistency_metrics = []
        
        for parent_asin, reviews in product_review_map.items():
            if len(reviews) >= 5:  # Need minimum reviews for consistency analysis
                ratings = [r.get('rating', 0) for r in reviews]
                
                # Calculate various consistency metrics
                std_dev = np.std(ratings)
                coefficient_of_variation = std_dev / np.mean(ratings) if np.mean(ratings) > 0 else 0
                rating_range = max(ratings) - min(ratings)
                
                # Calculate percentage of ratings within 1 point of mean
                mean_rating = np.mean(ratings)
                close_to_mean = sum(1 for r in ratings if abs(r - mean_rating) <= 1) / len(ratings)
                
                consistency_metrics.append({
                    'parent_asin': parent_asin,
                    'review_count': len(reviews),
                    'rating_std': std_dev,
                    'coefficient_of_variation': coefficient_of_variation,
                    'rating_range': rating_range,
                    'close_to_mean_ratio': close_to_mean,
                    'consistency_score': (1 - coefficient_of_variation) * close_to_mean
                })
        
        # Sort by consistency score
        consistency_metrics.sort(key=lambda x: x['consistency_score'], reverse=True)
        
        return {
            'product_consistency': consistency_metrics,
            'most_consistent': consistency_metrics[:10],
            'least_consistent': consistency_metrics[-10:],
            'avg_consistency_score': np.mean([m['consistency_score'] for m in consistency_metrics]),
            'consistency_distribution': np.histogram([m['consistency_score'] for m in consistency_metrics], 
                                                   bins=10)[0].tolist()
        }
    
    def _identify_rating_outliers(self, volume_rating_data: List[Dict]) -> Dict[str, Any]:
        """Identify rating outliers and anomalies."""
        if not volume_rating_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(volume_rating_data)
        
        # Identify outliers using IQR method
        Q1_rating = df['avg_rating'].quantile(0.25)
        Q3_rating = df['avg_rating'].quantile(0.75)
        IQR_rating = Q3_rating - Q1_rating
        
        rating_outliers = df[
            (df['avg_rating'] < (Q1_rating - 1.5 * IQR_rating)) |
            (df['avg_rating'] > (Q3_rating + 1.5 * IQR_rating))
        ]
        
        # Identify products with unusually high variance
        high_variance = df[df['rating_std'] > df['rating_std'].quantile(0.95)]
        
        # Identify products with unusually high/low volume
        Q1_volume = df['review_count'].quantile(0.25)
        Q3_volume = df['review_count'].quantile(0.75)
        IQR_volume = Q3_volume - Q1_volume
        
        volume_outliers = df[
            (df['review_count'] < (Q1_volume - 1.5 * IQR_volume)) |
            (df['review_count'] > (Q3_volume + 1.5 * IQR_volume))
        ]
        
        return {
            'rating_outliers': rating_outliers.to_dict('records'),
            'high_variance_products': high_variance.to_dict('records'),
            'volume_outliers': volume_outliers.to_dict('records'),
            'outlier_summary': {
                'total_products': len(df),
                'rating_outliers_count': len(rating_outliers),
                'high_variance_count': len(high_variance),
                'volume_outliers_count': len(volume_outliers)
            }
        }
    
    def generate_embeddings(self, documents: List[Dict]) -> List[Dict]:
        """Generate embeddings for all documents."""
        if not self.embedding_model:
            self.load_embedding_model()
        
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        # Extract text content for embedding
        texts = []
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                # Fallback: create content from available fields
                title = doc.get('title', '')
                description = doc.get('description', '')
                features = doc.get('features', '')
                content = f"{title} {description} {features}".strip()
            texts.append(content)
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_tensor=False)
            embeddings.extend(batch_embeddings.tolist())
        
        # Add embeddings to documents
        enhanced_documents = []
        for doc, embedding in zip(documents, embeddings):
            enhanced_doc = doc.copy()
            enhanced_doc['embedding'] = embedding
            enhanced_doc['embedding_model'] = self.config.embedding_model
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
    
    async def generate_embeddings_async(self, documents: List[Dict]) -> List[Dict]:
        """Generate embeddings for documents asynchronously."""
        if not self.config.enable_async:
            return self.generate_embeddings(documents)
        
        logger.info(f"Generating embeddings for {len(documents)} documents asynchronously...")
        
        # Load model if not already loaded
        if not self.embedding_model:
            await self.load_embedding_model_async()
        
        # Process documents in batches
        batch_size = 50
        enhanced_documents = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = await self._process_embedding_batch_async(batch)
            enhanced_documents.extend(batch_results)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Processed {i + len(batch)}/{len(documents)} documents")
        
        logger.info(f"Generated embeddings for {len(enhanced_documents)} documents")
        return enhanced_documents
    
    async def load_embedding_model_async(self):
        """Load sentence transformer model asynchronously."""
        if not self.config.enable_async:
            return self.load_embedding_model()
        
        logger.info(f"Loading embedding model asynchronously: {self.config.embedding_model}")
        
        loop = asyncio.get_event_loop()
        self.embedding_model = await loop.run_in_executor(
            self.executor,
            lambda: SentenceTransformer(self.config.embedding_model)
        )
        
        logger.info("Embedding model loaded successfully")
    
    async def _process_embedding_batch_async(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of embeddings asynchronously."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._process_embedding_batch_sync,
            batch
        )
    
    def _process_embedding_batch_sync(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of embeddings synchronously (for thread execution)."""
        enhanced_docs = []
        
        for doc in batch:
            # Generate embedding
            try:
                content = doc.get('content', '')
                if content:
                    embedding = self.embedding_model.encode(content, normalize_embeddings=True)
                    doc['embedding'] = embedding.tolist()
                    doc['embedding_dimension'] = len(embedding)
                else:
                    doc['embedding'] = []
                    doc['embedding_dimension'] = 0
                    
            except Exception as e:
                logger.error(f"Error generating embedding for document: {e}")
                doc['embedding'] = []
                doc['embedding_dimension'] = 0
            
            enhanced_docs.append(doc)
        
        return enhanced_docs
    
    async def process_temporal_trends_async(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal trends asynchronously."""
        if not self.config.enable_async:
            return self.process_temporal_trends(reviews)
        
        logger.info("Processing temporal trends asynchronously...")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.process_temporal_trends,
            reviews
        )
    
    async def analyze_category_insights_async(self, products: List[Dict], reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze category insights asynchronously."""
        if not self.config.enable_async:
            return self.analyze_category_insights(products, reviews)
        
        logger.info("Analyzing category insights asynchronously...")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_executor,  # Use process executor for CPU-intensive task
            self.analyze_category_insights,
            products,
            reviews
        )
    
    async def analyze_rating_patterns_async(self, products: List[Dict], reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze rating patterns asynchronously."""
        if not self.config.enable_async:
            return self.analyze_rating_patterns(products, reviews)
        
        logger.info("Analyzing rating patterns asynchronously...")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_executor,  # Use process executor for CPU-intensive task
            self.analyze_rating_patterns,
            products,
            reviews
        )
    
    def create_comprehensive_dataset(self, 
                                   input_dir: Path, 
                                   output_dir: Path) -> Dict[str, Any]:
        """Create comprehensive dataset with all analyses."""
        logger.info("Creating comprehensive dataset...")
        
        # Load processed data
        products_file = input_dir / "electronics_top1000_products.jsonl"
        reviews_file = input_dir / "electronics_top1000_products_reviews.jsonl"
        
        products = []
        with open(products_file, 'r', encoding='utf-8') as f:
            for line in f:
                products.append(json.loads(line))
        
        reviews = []
        with open(reviews_file, 'r', encoding='utf-8') as f:
            for line in f:
                reviews.append(json.loads(line))
        
        logger.info(f"Loaded {len(products)} products and {len(reviews)} reviews")
        
        # Perform all analyses
        temporal_trends = self.process_temporal_trends(reviews)
        category_insights = self.analyze_category_insights(products, reviews)
        rating_patterns = self.analyze_rating_patterns(products, reviews)
        
        # Create enhanced RAG documents
        rag_documents = self._create_enhanced_rag_documents(products, reviews, 
                                                           temporal_trends, 
                                                           category_insights, 
                                                           rating_patterns)
        
        # Generate embeddings
        enhanced_documents = self.generate_embeddings(rag_documents)
        
        # Save all outputs
        output_dir.mkdir(exist_ok=True)
        
        # Save enhanced RAG documents
        rag_file = output_dir / "enhanced_rag_documents.jsonl"
        with open(rag_file, 'w', encoding='utf-8') as f:
            for doc in enhanced_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # Save analysis results
        analyses = {
            'temporal_trends': temporal_trends,
            'category_insights': category_insights,
            'rating_patterns': rating_patterns,
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__,
                'document_count': len(enhanced_documents),
                'embedding_model': self.config.embedding_model
            }
        }
        
        analyses_file = output_dir / "comprehensive_analysis.json"
        with open(analyses_file, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved {len(enhanced_documents)} enhanced documents to {rag_file}")
        logger.info(f"Saved comprehensive analysis to {analyses_file}")
        
        return {
            'enhanced_documents': enhanced_documents,
            'analyses': analyses,
            'files_created': [str(rag_file), str(analyses_file)]
        }
    
    async def create_comprehensive_dataset_async(self, 
                                               input_dir: Path, 
                                               output_dir: Path) -> Dict[str, Any]:
        """Create comprehensive dataset asynchronously."""
        if not self.config.enable_async:
            return self.create_comprehensive_dataset(input_dir, output_dir)
        
        logger.info("Creating comprehensive dataset asynchronously...")
        
        start_time = asyncio.get_event_loop().time()
        
        # Load data files asynchronously
        products_task = self._load_jsonl_async(input_dir / "electronics_top1000_products.jsonl")
        reviews_task = self._load_jsonl_async(input_dir / "electronics_top1000_products_reviews.jsonl")
        
        products, reviews = await asyncio.gather(products_task, reviews_task)
        
        logger.info(f"Loaded {len(products)} products and {len(reviews)} reviews")
        
        # Process analytics in parallel
        temporal_task = self.process_temporal_trends_async(reviews)
        category_task = self.analyze_category_insights_async(products, reviews)
        rating_task = self.analyze_rating_patterns_async(products, reviews)
        
        temporal_trends, category_insights, rating_patterns = await asyncio.gather(
            temporal_task, category_task, rating_task
        )
        
        # Create enhanced documents
        enhanced_docs = await self._create_enhanced_rag_documents_async(
            products, reviews, temporal_trends, category_insights, rating_patterns
        )
        
        # Generate embeddings
        final_docs = await self.generate_embeddings_async(enhanced_docs)
        
        # Save results
        await self._save_results_async(output_dir, {
            'products': products,
            'reviews': reviews,
            'temporal_trends': temporal_trends,
            'category_insights': category_insights,
            'rating_patterns': rating_patterns,
            'enhanced_documents': final_docs
        })
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        summary = {
            'total_products': len(products),
            'total_reviews': len(reviews),
            'enhanced_documents': len(final_docs),
            'processing_time': processing_time,
            'async_execution': True,
            'temporal_trends': bool(temporal_trends),
            'category_insights': bool(category_insights),
            'rating_patterns': bool(rating_patterns)
        }
        
        logger.info(f"Dataset creation completed in {processing_time:.2f} seconds")
        return summary
    
    def _create_enhanced_rag_documents(self, 
                                     products: List[Dict], 
                                     reviews: List[Dict],
                                     temporal_trends: Dict,
                                     category_insights: Dict,
                                     rating_patterns: Dict) -> List[Dict]:
        """Create enhanced RAG documents with all analysis insights."""
        enhanced_documents = []
        
        # Create product-review mapping
        product_review_map = defaultdict(list)
        for review in reviews:
            product_review_map[review['parent_asin']].append(review)
        
        for product in products:
            parent_asin = product['parent_asin']
            product_reviews = product_review_map.get(parent_asin, [])
            
            # Enhanced product document
            doc = {
                'id': f"product_{parent_asin}",
                'type': 'product',
                'parent_asin': parent_asin,
                'title': product.get('title', ''),
                'description': ' '.join(product.get('description', [])) if product.get('description') else '',
                'features': ' '.join(product.get('features', [])) if product.get('features') else '',
                'price': product.get('price'),
                'average_rating': product.get('average_rating'),
                'rating_number': product.get('rating_number'),
                'review_count': product.get('review_count'),
                'store': product.get('store', ''),
                'categories': product.get('categories', []),
                'details': product.get('details', {}),
                
                # Enhanced analytics
                'analytics': {
                    'review_velocity': self._get_product_velocity(parent_asin, temporal_trends),
                    'sentiment_summary': self._get_product_sentiment(product_reviews),
                    'price_competitiveness': self._get_price_competitiveness(product, rating_patterns),
                    'category_performance': self._get_category_performance(product, category_insights),
                    'rating_consistency': self._get_rating_consistency(product_reviews),
                    'temporal_insights': self._get_temporal_insights(product_reviews, temporal_trends)
                }
            }
            
            # Create comprehensive content
            content_parts = []
            if doc['title']:
                content_parts.append(f"Product: {doc['title']}")
            if doc['description']:
                content_parts.append(f"Description: {doc['description']}")
            if doc['features']:
                content_parts.append(f"Features: {doc['features']}")
            if doc['categories']:
                content_parts.append(f"Categories: {' > '.join(doc['categories'])}")
            
            # Add analytics to content
            analytics = doc['analytics']
            if analytics['sentiment_summary']:
                content_parts.append(f"User sentiment: {analytics['sentiment_summary']['overall_sentiment']}")
            if analytics['price_competitiveness']:
                content_parts.append(f"Price competitiveness: {analytics['price_competitiveness']['tier']}")
            
            doc['content'] = ' '.join(content_parts)
            enhanced_documents.append(doc)
            
            # Enhanced review summary document
            if product_reviews:
                review_doc = {
                    'id': f"reviews_{parent_asin}",
                    'type': 'review_summary',
                    'parent_asin': parent_asin,
                    'product_title': doc['title'],
                    'review_analytics': {
                        'total_reviews': len(product_reviews),
                        'rating_distribution': dict(Counter(r.get('rating', 0) for r in product_reviews)),
                        'sentiment_breakdown': analytics['sentiment_summary'],
                        'temporal_pattern': analytics['temporal_insights'],
                        'key_themes': self._extract_review_themes(product_reviews)
                    }
                }
                
                # Create review content
                positive_reviews = [r for r in product_reviews if r.get('rating', 0) >= 4]
                negative_reviews = [r for r in product_reviews if r.get('rating', 0) <= 2]
                
                content_parts = [f"Reviews for {doc['title']}"]
                if positive_reviews:
                    pos_texts = [r.get('text', '')[:200] for r in positive_reviews[:3] if r.get('text')]
                    content_parts.append(f"Positive feedback: {' '.join(pos_texts)}")
                if negative_reviews:
                    neg_texts = [r.get('text', '')[:200] for r in negative_reviews[:3] if r.get('text')]
                    content_parts.append(f"Critical feedback: {' '.join(neg_texts)}")
                
                review_doc['content'] = ' '.join(content_parts)
                enhanced_documents.append(review_doc)
        
        return enhanced_documents
    
    async def _create_enhanced_rag_documents_async(self, 
                                                 products: List[Dict], 
                                                 reviews: List[Dict],
                                                 temporal_trends: Dict,
                                                 category_insights: Dict,
                                                 rating_patterns: Dict) -> List[Dict]:
        """Create enhanced RAG documents asynchronously."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_enhanced_rag_documents,
            products,
            reviews,
            temporal_trends,
            category_insights,
            rating_patterns
        )
    
    def _get_product_velocity(self, parent_asin: str, temporal_trends: Dict) -> Dict[str, Any]:
        """Get velocity metrics for a specific product."""
        velocity_data = temporal_trends.get('review_velocity', {}).get('product_velocities', {})
        return velocity_data.get(parent_asin, {})
    
    def _get_product_sentiment(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment for a specific product."""
        if not reviews:
            return {}
        
        ratings = [r.get('rating', 0) for r in reviews]
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for rating in ratings:
            if rating >= 4:
                sentiment_counts['positive'] += 1
            elif rating <= 2:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
        
        total = len(ratings)
        sentiment_ratios = {k: v/total for k, v in sentiment_counts.items()}
        
        # Determine overall sentiment
        if sentiment_ratios['positive'] > 0.6:
            overall = 'positive'
        elif sentiment_ratios['negative'] > 0.3:
            overall = 'mixed'
        elif sentiment_ratios['negative'] > 0.1:
            overall = 'neutral'
        else:
            overall = 'very_positive'
        
        return {
            'sentiment_counts': sentiment_counts,
            'sentiment_ratios': sentiment_ratios,
            'overall_sentiment': overall,
            'avg_rating': np.mean(ratings),
            'rating_variance': np.var(ratings)
        }
    
    def _get_price_competitiveness(self, product: Dict, rating_patterns: Dict) -> Dict[str, Any]:
        """Analyze price competitiveness."""
        price = product.get('price')
        if not price:
            return {}
        
        # Find products in similar rating range
        price_rating_data = rating_patterns.get('price_vs_rating', [])
        if not price_rating_data:
            return {}
        
        product_rating = product.get('average_rating', 0)
        similar_rated = [p for p in price_rating_data 
                        if abs(p['avg_rating'] - product_rating) <= 0.3]
        
        if similar_rated:
            prices = [p['price'] for p in similar_rated]
            price_percentile = (sum(1 for p in prices if p <= price) / len(prices)) * 100
            
            if price_percentile <= 25:
                tier = 'budget'
            elif price_percentile <= 50:
                tier = 'mid-range'
            elif price_percentile <= 75:
                tier = 'premium'
            else:
                tier = 'luxury'
            
            return {
                'price_percentile': price_percentile,
                'tier': tier,
                'compared_to_similar': len(similar_rated),
                'price_vs_market': {
                    'below_average': price < np.mean(prices),
                    'market_average': np.mean(prices),
                    'market_median': np.median(prices)
                }
            }
        
        return {}
    
    def _get_category_performance(self, product: Dict, category_insights: Dict) -> Dict[str, Any]:
        """Get category performance metrics."""
        categories = product.get('categories', [])
        if not categories:
            return {}
        
        category_data = category_insights.get('category_breakdown', {})
        performance_data = category_insights.get('category_performance', {})
        
        primary_category = ' > '.join(categories[:2]) if len(categories) >= 2 else categories[0]
        cat_data = category_data.get(primary_category, {})
        
        if not cat_data:
            return {}
        
        return {
            'primary_category': primary_category,
            'category_rank_by_products': self._get_category_rank(primary_category, performance_data, 'by_product_count'),
            'category_rank_by_reviews': self._get_category_rank(primary_category, performance_data, 'by_total_reviews'),
            'category_avg_rating': cat_data.get('avg_rating', 0),
            'category_review_density': cat_data.get('avg_reviews_per_product', 0),
            'category_sentiment': cat_data.get('review_sentiment', {})
        }
    
    def _get_category_rank(self, category: str, performance_data: Dict, metric: str) -> int:
        """Get category rank for specific metric."""
        rankings = performance_data.get(metric, [])
        try:
            return rankings.index(category) + 1
        except ValueError:
            return len(rankings) + 1
    
    def _get_rating_consistency(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze rating consistency for a product."""
        if len(reviews) < 3:
            return {}
        
        ratings = [r.get('rating', 0) for r in reviews]
        std_dev = np.std(ratings)
        mean_rating = np.mean(ratings)
        
        consistency_score = 1 - (std_dev / 2)  # Normalize to 0-1 scale
        
        return {
            'rating_std': std_dev,
            'consistency_score': max(0, consistency_score),
            'rating_range': max(ratings) - min(ratings),
            'coefficient_of_variation': std_dev / mean_rating if mean_rating > 0 else 0
        }
    
    def _get_temporal_insights(self, reviews: List[Dict], temporal_trends: Dict) -> Dict[str, Any]:
        """Get temporal insights for a product."""
        if not reviews:
            return {}
        
        # Convert timestamps to datetime for analysis
        review_dates = []
        for review in reviews:
            if 'timestamp' in review:
                try:
                    review_dates.append(datetime.fromtimestamp(review['timestamp']))
                except (ValueError, TypeError):
                    continue
        
        if not review_dates:
            return {}
        
        review_dates.sort()
        
        return {
            'first_review_date': review_dates[0].isoformat(),
            'latest_review_date': review_dates[-1].isoformat(),
            'review_span_days': (review_dates[-1] - review_dates[0]).days,
            'reviews_last_30_days': sum(1 for d in review_dates 
                                       if (datetime.now() - d).days <= 30),
            'reviews_last_90_days': sum(1 for d in review_dates 
                                       if (datetime.now() - d).days <= 90),
            'peak_review_month': max(Counter(d.strftime('%Y-%m') for d in review_dates).items(), 
                                   key=lambda x: x[1])[0] if review_dates else None
        }
    
    def _extract_review_themes(self, reviews: List[Dict]) -> List[str]:
        """Extract key themes from reviews using simple keyword analysis."""
        # Combine all review texts
        all_text = ' '.join([r.get('text', '') for r in reviews if r.get('text')])
        
        if not all_text:
            return []
        
        # Simple keyword extraction (in production, could use more advanced NLP)
        words = all_text.lower().split()
        
        # Filter for meaningful words and count frequencies
        meaningful_words = [w for w in words if len(w) > 4 and w.isalpha()]
        word_freq = Counter(meaningful_words)
        
        # Return top themes
        return [word for word, count in word_freq.most_common(10) if count >= 2]

    async def _load_jsonl_async(self, file_path: Path) -> List[Dict]:
        """Load JSONL file asynchronously."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._load_jsonl_sync,
            file_path
        )
    
    def _load_jsonl_sync(self, file_path: Path) -> List[Dict]:
        """Load JSONL file synchronously."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
        
        return data
    
    async def _save_results_async(self, output_dir: Path, results: Dict[str, Any]):
        """Save results asynchronously."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save different components in parallel
        tasks = []
        
        # Save products
        if results.get('products'):
            tasks.append(self._save_jsonl_async(
                output_dir / "electronics_top1000_products.jsonl",
                results['products']
            ))
        
        # Save reviews
        if results.get('reviews'):
            tasks.append(self._save_jsonl_async(
                output_dir / "electronics_top1000_products_reviews.jsonl",
                results['reviews']
            ))
        
        # Save enhanced documents
        if results.get('enhanced_documents'):
            tasks.append(self._save_jsonl_async(
                output_dir / "electronics_rag_documents.jsonl",
                results['enhanced_documents']
            ))
        
        # Save analytics
        analytics = {
            'temporal_trends': results.get('temporal_trends', {}),
            'category_insights': results.get('category_insights', {}),
            'rating_patterns': results.get('rating_patterns', {}),
            'processing_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_products': len(results.get('products', [])),
                'total_reviews': len(results.get('reviews', [])),
                'enhanced_documents': len(results.get('enhanced_documents', []))
            }
        }
        
        tasks.append(self._save_json_async(
            output_dir / "dataset_summary.json",
            analytics
        ))
        
        # Execute all save operations in parallel
        await asyncio.gather(*tasks)
    
    async def _save_jsonl_async(self, file_path: Path, data: List[Dict]):
        """Save JSONL file asynchronously."""
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._save_jsonl_sync,
            file_path,
            data
        )
    
    def _save_jsonl_sync(self, file_path: Path, data: List[Dict]):
        """Save JSONL file synchronously."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
    
    async def _save_json_async(self, file_path: Path, data: Dict):
        """Save JSON file asynchronously."""
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._save_json_sync,
            file_path,
            data
        )
    
    def _save_json_sync(self, file_path: Path, data: Dict):
        """Save JSON file synchronously."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
    
    def close(self):
        """Close executors and clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


def main():
    """Main execution function."""
    # Configuration
    config = ProcessingConfig(
        target_products=1000,
        embedding_model="all-MiniLM-L6-v2",
        temporal_window_days=365
    )
    
    # Initialize processor
    processor = AdvancedDataProcessor(config)
    
    # Set paths
    input_dir = Path("../data/processed")
    output_dir = Path("../data/processed/enhanced")
    
    # Create comprehensive dataset
    results = processor.create_comprehensive_dataset(input_dir, output_dir)
    
    print("Advanced data processing completed successfully!")
    print(f"Created {len(results['enhanced_documents'])} enhanced documents")
    print(f"Output files: {results['files_created']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()