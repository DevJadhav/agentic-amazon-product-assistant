"""
Interactive Visualizations and Analysis Dashboards
Comprehensive analysis dashboards with temporal trends, category insights, and rating patterns.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class InteractiveDashboard:
    """Interactive dashboard for comprehensive Amazon product analysis."""
    
    def __init__(self):
        self.data_loaded = False
        self.products_df = None
        self.reviews_df = None
        self.analysis_data = None
        
    def load_data(self, data_dir: Path) -> bool:
        """Load all necessary data for visualization."""
        try:
            # Load products data
            products_file = data_dir / "electronics_top1000_products.jsonl"
            if products_file.exists():
                products = []
                with open(products_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        products.append(json.loads(line))
                self.products_df = pd.DataFrame(products)
                logger.info(f"Loaded {len(self.products_df)} products")
            
            # Load reviews data
            reviews_file = data_dir / "electronics_top1000_products_reviews.jsonl"
            if reviews_file.exists():
                reviews = []
                with open(reviews_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        reviews.append(json.loads(line))
                self.reviews_df = pd.DataFrame(reviews)
                
                # Convert timestamps to datetime
                self.reviews_df['datetime'] = pd.to_datetime(
                    self.reviews_df['timestamp'], unit='s', errors='coerce'
                )
                logger.info(f"Loaded {len(self.reviews_df)} reviews")
            
            # Load analysis data if available
            analysis_file = data_dir / "enhanced" / "comprehensive_analysis.json"
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    self.analysis_data = json.load(f)
                logger.info("Loaded comprehensive analysis data")
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def create_temporal_trends_dashboard(self) -> Dict[str, go.Figure]:
        """Create temporal trends visualizations."""
        if not self.data_loaded or self.reviews_df is None:
            return {}
        
        figures = {}
        
        # 1. Review Volume Over Time
        monthly_reviews = self.reviews_df.groupby(
            self.reviews_df['datetime'].dt.to_period('M')
        ).size().reset_index()
        monthly_reviews['datetime'] = monthly_reviews['datetime'].dt.to_timestamp()
        
        fig_volume = px.line(
            monthly_reviews, 
            x='datetime', 
            y=0,
            title='Review Volume Over Time',
            labels={'0': 'Number of Reviews', 'datetime': 'Month'},
            template='plotly_white'
        )
        fig_volume.update_traces(line_color='#1f77b4', line_width=3)
        fig_volume.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Month",
            yaxis_title="Number of Reviews"
        )
        figures['review_volume'] = fig_volume
        
        # 2. Average Rating Trends
        monthly_ratings = self.reviews_df.groupby(
            self.reviews_df['datetime'].dt.to_period('M')
        )['rating'].agg(['mean', 'std', 'count']).reset_index()
        monthly_ratings['datetime'] = monthly_ratings['datetime'].dt.to_timestamp()
        
        fig_ratings = go.Figure()
        fig_ratings.add_trace(go.Scatter(
            x=monthly_ratings['datetime'],
            y=monthly_ratings['mean'] + monthly_ratings['std'],
            mode='lines',
            line=dict(width=0),
            name='Upper Bound',
            showlegend=False
        ))
        fig_ratings.add_trace(go.Scatter(
            x=monthly_ratings['datetime'],
            y=monthly_ratings['mean'] - monthly_ratings['std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='Standard Deviation',
            showlegend=True
        ))
        fig_ratings.add_trace(go.Scatter(
            x=monthly_ratings['datetime'],
            y=monthly_ratings['mean'],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            name='Average Rating'
        ))
        
        fig_ratings.update_layout(
            title='Average Rating Trends Over Time',
            xaxis_title='Month',
            yaxis_title='Average Rating',
            template='plotly_white',
            height=400,
            yaxis=dict(range=[1, 5])
        )
        figures['rating_trends'] = fig_ratings
        
        # 3. Seasonal Patterns
        self.reviews_df['month'] = self.reviews_df['datetime'].dt.month
        self.reviews_df['quarter'] = self.reviews_df['datetime'].dt.quarter
        
        seasonal_data = self.reviews_df.groupby('month').agg({
            'rating': ['count', 'mean'],
            'parent_asin': 'nunique'
        }).round(2)
        seasonal_data.columns = ['Review Count', 'Avg Rating', 'Unique Products']
        seasonal_data = seasonal_data.reset_index()
        seasonal_data['Month Name'] = seasonal_data['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        fig_seasonal = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Review Count', 'Monthly Avg Rating',
                          'Quarterly Distribution', 'Product Coverage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # Monthly review count
        fig_seasonal.add_trace(
            go.Bar(x=seasonal_data['Month Name'], y=seasonal_data['Review Count'],
                  name='Review Count', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Monthly average rating
        fig_seasonal.add_trace(
            go.Scatter(x=seasonal_data['Month Name'], y=seasonal_data['Avg Rating'],
                      mode='lines+markers', name='Avg Rating', line_color='orange'),
            row=1, col=2
        )
        
        # Quarterly pie chart
        quarterly_counts = self.reviews_df.groupby('quarter').size()
        fig_seasonal.add_trace(
            go.Pie(labels=[f'Q{q}' for q in quarterly_counts.index],
                  values=quarterly_counts.values, name="Quarterly"),
            row=2, col=1
        )
        
        # Product coverage by month
        fig_seasonal.add_trace(
            go.Bar(x=seasonal_data['Month Name'], y=seasonal_data['Unique Products'],
                  name='Unique Products', marker_color='lightgreen'),
            row=2, col=2
        )
        
        fig_seasonal.update_layout(
            height=800,
            title_text="Seasonal Patterns Analysis",
            template='plotly_white',
            showlegend=False
        )
        figures['seasonal_patterns'] = fig_seasonal
        
        # 4. Rating Distribution Evolution
        rating_evolution = self.reviews_df.groupby([
            self.reviews_df['datetime'].dt.to_period('Q'), 'rating'
        ]).size().unstack(fill_value=0)
        rating_evolution.index = rating_evolution.index.to_timestamp()
        
        # Calculate percentages
        rating_evolution_pct = rating_evolution.div(rating_evolution.sum(axis=1), axis=0) * 100
        
        fig_evolution = go.Figure()
        colors = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4']
        rating_names = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
        
        for i, rating in enumerate([1, 2, 3, 4, 5]):
            if rating in rating_evolution_pct.columns:
                fig_evolution.add_trace(go.Scatter(
                    x=rating_evolution_pct.index,
                    y=rating_evolution_pct[rating],
                    mode='lines+markers',
                    name=rating_names[i],
                    line=dict(color=colors[i], width=2),
                    stackgroup='one'
                ))
        
        fig_evolution.update_layout(
            title='Rating Distribution Evolution Over Time',
            xaxis_title='Quarter',
            yaxis_title='Percentage of Reviews',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        figures['rating_evolution'] = fig_evolution
        
        return figures
    
    def create_category_insights_dashboard(self) -> Dict[str, go.Figure]:
        """Create category insights visualizations."""
        if not self.data_loaded or self.products_df is None:
            return {}
        
        figures = {}
        
        # Prepare category data
        category_data = []
        for _, product in self.products_df.iterrows():
            categories = product.get('categories', [])
            for i, category in enumerate(categories[:3]):  # Top 3 levels
                category_path = ' > '.join(categories[:i+1])
                category_data.append({
                    'parent_asin': product['parent_asin'],
                    'category_level': i + 1,
                    'category': category,
                    'category_path': category_path,
                    'price': product.get('price'),
                    'average_rating': product.get('average_rating'),
                    'review_count': product.get('review_count', 0)
                })
        
        categories_df = pd.DataFrame(category_data)
        
        # 1. Category Tree Visualization
        level1_cats = categories_df[categories_df['category_level'] == 1]
        cat_summary = level1_cats.groupby('category').agg({
            'parent_asin': 'count',
            'price': 'mean',
            'average_rating': 'mean',
            'review_count': 'sum'
        }).round(2)
        cat_summary.columns = ['Product Count', 'Avg Price', 'Avg Rating', 'Total Reviews']
        cat_summary = cat_summary.sort_values('Product Count', ascending=True)
        
        fig_tree = px.treemap(
            cat_summary.reset_index(),
            path=['category'],
            values='Product Count',
            color='Avg Rating',
            color_continuous_scale='RdYlGn',
            title='Product Categories Overview (Size = Product Count, Color = Average Rating)'
        )
        fig_tree.update_layout(height=500)
        figures['category_tree'] = fig_tree
        
        # 2. Category Performance Matrix
        fig_matrix = px.scatter(
            cat_summary.reset_index(),
            x='Avg Price',
            y='Avg Rating',
            size='Product Count',
            color='Total Reviews',
            hover_name='category',
            title='Category Performance Matrix',
            labels={'Avg Price': 'Average Price ($)', 'Avg Rating': 'Average Rating (1-5)'},
            color_continuous_scale='viridis'
        )
        fig_matrix.update_layout(height=500, template='plotly_white')
        figures['category_matrix'] = fig_matrix
        
        # 3. Price Distribution by Category
        top_categories = cat_summary.head(10).index.tolist()
        price_data = []
        for cat in top_categories:
            cat_products = level1_cats[level1_cats['category'] == cat]
            valid_prices = cat_products.dropna(subset=['price'])
            for _, product in valid_prices.iterrows():
                price_data.append({
                    'category': cat,
                    'price': product['price']
                })
        
        if price_data:
            price_df = pd.DataFrame(price_data)
            fig_price = px.box(
                price_df,
                x='category',
                y='price',
                title='Price Distribution by Top Categories',
                labels={'price': 'Price ($)', 'category': 'Category'}
            )
            fig_price.update_xaxes(tickangle=45)
            fig_price.update_layout(height=500, template='plotly_white')
            figures['price_distribution'] = fig_price
        
        # 4. Category Rating Analysis
        rating_by_cat = level1_cats.groupby('category')['average_rating'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        rating_by_cat = rating_by_cat[rating_by_cat['count'] >= 5]  # Filter for sufficient data
        rating_by_cat = rating_by_cat.sort_values('mean', ascending=True)
        
        fig_rating = go.Figure()
        fig_rating.add_trace(go.Bar(
            x=rating_by_cat['mean'],
            y=rating_by_cat['category'],
            orientation='h',
            error_x=dict(array=rating_by_cat['std']),
            name='Average Rating',
            marker_color='lightblue'
        ))
        
        fig_rating.update_layout(
            title='Average Rating by Category (with Standard Deviation)',
            xaxis_title='Average Rating',
            yaxis_title='Category',
            height=max(400, len(rating_by_cat) * 30),
            template='plotly_white'
        )
        figures['category_ratings'] = fig_rating
        
        # 5. Category Market Share
        market_share = cat_summary['Product Count'].sort_values(ascending=False)
        
        fig_share = go.Figure(data=[
            go.Pie(
                labels=market_share.index,
                values=market_share.values,
                hole=0.4,
                textinfo='label+percent',
                textposition='auto'
            )
        ])
        fig_share.update_layout(
            title='Market Share by Category (Product Count)',
            height=500,
            template='plotly_white'
        )
        figures['market_share'] = fig_share
        
        return figures
    
    def create_rating_patterns_dashboard(self) -> Dict[str, go.Figure]:
        """Create rating patterns visualizations."""
        if not self.data_loaded:
            return {}
        
        figures = {}
        
        # 1. Overall Rating Distribution
        rating_dist = self.reviews_df['rating'].value_counts().sort_index()
        
        fig_dist = go.Figure(data=[
            go.Bar(
                x=rating_dist.index,
                y=rating_dist.values,
                marker_color=['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4'],
                text=rating_dist.values,
                textposition='auto'
            )
        ])
        fig_dist.update_layout(
            title='Overall Rating Distribution',
            xaxis_title='Rating (Stars)',
            yaxis_title='Number of Reviews',
            template='plotly_white',
            height=400
        )
        figures['rating_distribution'] = fig_dist
        
        # 2. Rating vs Price Analysis
        if 'price' in self.products_df.columns:
            # Merge products with their average ratings from reviews
            product_ratings = self.reviews_df.groupby('parent_asin')['rating'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            product_ratings.columns = ['parent_asin', 'avg_review_rating', 'rating_std', 'review_count']
            
            price_rating = self.products_df.merge(product_ratings, on='parent_asin', how='inner')
            price_rating = price_rating.dropna(subset=['price'])
            
            # Create price bins
            price_rating['price_bin'] = pd.cut(
                price_rating['price'], 
                bins=[0, 25, 50, 100, 200, 500, float('inf')],
                labels=['$0-25', '$25-50', '$50-100', '$100-200', '$200-500', '$500+']
            )
            
            price_rating_summary = price_rating.groupby('price_bin')['avg_review_rating'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            fig_price_rating = go.Figure()
            fig_price_rating.add_trace(go.Bar(
                x=price_rating_summary['price_bin'],
                y=price_rating_summary['mean'],
                error_y=dict(array=price_rating_summary['std']),
                name='Average Rating',
                marker_color='lightcoral'
            ))
            
            fig_price_rating.update_layout(
                title='Average Rating by Price Range',
                xaxis_title='Price Range',
                yaxis_title='Average Rating',
                template='plotly_white',
                height=400
            )
            figures['price_vs_rating'] = fig_price_rating
        
        # 3. Rating Consistency Analysis
        product_rating_stats = self.reviews_df.groupby('parent_asin')['rating'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).reset_index()
        product_rating_stats['rating_range'] = (
            product_rating_stats['max'] - product_rating_stats['min']
        )
        product_rating_stats['cv'] = (
            product_rating_stats['std'] / product_rating_stats['mean']
        )
        
        # Filter for products with sufficient reviews
        sufficient_reviews = product_rating_stats[product_rating_stats['count'] >= 5]
        
        fig_consistency = px.scatter(
            sufficient_reviews,
            x='mean',
            y='std',
            size='count',
            color='rating_range',
            hover_data=['parent_asin'],
            title='Rating Consistency Analysis (Mean vs Standard Deviation)',
            labels={'mean': 'Average Rating', 'std': 'Standard Deviation'},
            color_continuous_scale='viridis'
        )
        fig_consistency.update_layout(height=500, template='plotly_white')
        figures['rating_consistency'] = fig_consistency
        
        # 4. Review Volume vs Rating Quality
        volume_rating = product_rating_stats[product_rating_stats['count'] >= 3]
        
        # Create volume bins
        volume_rating['volume_bin'] = pd.cut(
            volume_rating['count'],
            bins=[0, 5, 10, 20, 50, float('inf')],
            labels=['3-5', '6-10', '11-20', '21-50', '50+']
        )
        
        volume_summary = volume_rating.groupby('volume_bin').agg({
            'mean': ['mean', 'std'],
            'std': 'mean',
            'parent_asin': 'count'
        }).round(3)
        volume_summary.columns = ['avg_rating_mean', 'avg_rating_std', 'consistency_mean', 'product_count']
        volume_summary = volume_summary.reset_index()
        
        fig_volume = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Rating by Review Volume', 'Rating Consistency by Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_volume.add_trace(
            go.Bar(x=volume_summary['volume_bin'], y=volume_summary['avg_rating_mean'],
                  error_y=dict(array=volume_summary['avg_rating_std']),
                  name='Avg Rating', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig_volume.add_trace(
            go.Bar(x=volume_summary['volume_bin'], y=volume_summary['consistency_mean'],
                  name='Avg Std Dev', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig_volume.update_layout(
            title='Review Volume Impact on Rating Quality',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        fig_volume.update_xaxes(title_text="Review Count Range", row=1, col=1)
        fig_volume.update_xaxes(title_text="Review Count Range", row=1, col=2)
        fig_volume.update_yaxes(title_text="Average Rating", row=1, col=1)
        fig_volume.update_yaxes(title_text="Rating Std Dev", row=1, col=2)
        
        figures['volume_vs_quality'] = fig_volume
        
        # 5. Top and Bottom Performers
        top_performers = product_rating_stats.nlargest(10, 'mean')[['parent_asin', 'mean', 'count']]
        bottom_performers = product_rating_stats.nsmallest(10, 'mean')[['parent_asin', 'mean', 'count']]
        
        # Get product titles
        if 'title' in self.products_df.columns:
            top_performers = top_performers.merge(
                self.products_df[['parent_asin', 'title']], on='parent_asin', how='left'
            )
            bottom_performers = bottom_performers.merge(
                self.products_df[['parent_asin', 'title']], on='parent_asin', how='left'
            )
            
            # Truncate long titles
            top_performers['short_title'] = top_performers['title'].str[:50] + '...'
            bottom_performers['short_title'] = bottom_performers['title'].str[:50] + '...'
            
            fig_performers = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Top 10 Highest Rated Products', 'Top 10 Lowest Rated Products'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_performers.add_trace(
                go.Bar(y=top_performers['short_title'], x=top_performers['mean'],
                      orientation='h', name='Top Rated', marker_color='green'),
                row=1, col=1
            )
            
            fig_performers.add_trace(
                go.Bar(y=bottom_performers['short_title'], x=bottom_performers['mean'],
                      orientation='h', name='Lowest Rated', marker_color='red'),
                row=1, col=2
            )
            
            fig_performers.update_layout(
                title='Product Performance Comparison',
                height=600,
                template='plotly_white',
                showlegend=False
            )
            fig_performers.update_xaxes(title_text="Average Rating", row=1, col=1)
            fig_performers.update_xaxes(title_text="Average Rating", row=1, col=2)
            
            figures['top_bottom_performers'] = fig_performers
        
        return figures
    
    def create_comprehensive_overview(self) -> Dict[str, go.Figure]:
        """Create comprehensive overview dashboard."""
        if not self.data_loaded:
            return {}
        
        figures = {}
        
        # 1. Key Metrics Summary
        total_products = len(self.products_df)
        total_reviews = len(self.reviews_df)
        avg_rating = self.reviews_df['rating'].mean()
        avg_price = self.products_df['price'].mean() if 'price' in self.products_df.columns else 0
        
        # Create KPI indicators
        fig_kpi = go.Figure()
        
        kpis = [
            {'label': 'Total Products', 'value': total_products, 'color': '#1f77b4'},
            {'label': 'Total Reviews', 'value': total_reviews, 'color': '#ff7f0e'},
            {'label': 'Avg Rating', 'value': f"{avg_rating:.2f}", 'color': '#2ca02c'},
            {'label': 'Avg Price', 'value': f"${avg_price:.2f}", 'color': '#d62728'}
        ]
        
        for i, kpi in enumerate(kpis):
            fig_kpi.add_trace(go.Indicator(
                mode="number",
                value=float(str(kpi['value']).replace('$', '')) if isinstance(kpi['value'], str) and '$' in kpi['value'] else kpi['value'],
                title={'text': kpi['label']},
                number={'font': {'color': kpi['color'], 'size': 40}},
                domain={'row': 0, 'column': i}
            ))
        
        fig_kpi.update_layout(
            grid={'rows': 1, 'columns': 4, 'pattern': "independent"},
            height=200,
            margin=dict(t=50, b=50),
            title="Key Performance Indicators"
        )
        figures['kpi_summary'] = fig_kpi
        
        # 2. Data Quality Overview
        quality_metrics = {
            'Products with Price': (self.products_df['price'].notna().sum() / len(self.products_df)) * 100,
            'Products with Reviews': (self.reviews_df['parent_asin'].nunique() / len(self.products_df)) * 100,
            'Reviews with Timestamps': (self.reviews_df['timestamp'].notna().sum() / len(self.reviews_df)) * 100,
            'Products with Categories': (self.products_df['categories'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum() / len(self.products_df)) * 100
        }
        
        fig_quality = go.Figure(data=[
            go.Bar(
                x=list(quality_metrics.keys()),
                y=list(quality_metrics.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                text=[f"{v:.1f}%" for v in quality_metrics.values()],
                textposition='auto'
            )
        ])
        fig_quality.update_layout(
            title='Data Quality Metrics',
            yaxis_title='Completeness (%)',
            template='plotly_white',
            height=400
        )
        figures['data_quality'] = fig_quality
        
        # 3. Dataset Distribution
        # Reviews per product distribution
        reviews_per_product = self.reviews_df.groupby('parent_asin').size()
        
        fig_distribution = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Reviews per Product', 'Price Distribution',
                          'Rating Distribution', 'Category Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Reviews per product histogram
        fig_distribution.add_trace(
            go.Histogram(x=reviews_per_product.values, nbinsx=20, name='Reviews per Product'),
            row=1, col=1
        )
        
        # Price distribution
        if 'price' in self.products_df.columns:
            valid_prices = self.products_df['price'].dropna()
            fig_distribution.add_trace(
                go.Histogram(x=valid_prices, nbinsx=30, name='Price Distribution'),
                row=1, col=2
            )
        
        # Rating distribution
        fig_distribution.add_trace(
            go.Bar(x=self.reviews_df['rating'].value_counts().sort_index().index,
                  y=self.reviews_df['rating'].value_counts().sort_index().values,
                  name='Rating Distribution'),
            row=2, col=1
        )
        
        # Category distribution (top level categories)
        if 'categories' in self.products_df.columns:
            top_categories = []
            for categories in self.products_df['categories']:
                if isinstance(categories, list) and len(categories) > 0:
                    top_categories.append(categories[0])
            
            if top_categories:
                cat_counts = Counter(top_categories)
                top_5_cats = dict(cat_counts.most_common(5))
                
                fig_distribution.add_trace(
                    go.Pie(labels=list(top_5_cats.keys()), values=list(top_5_cats.values()),
                          name="Top Categories"),
                    row=2, col=2
                )
        
        fig_distribution.update_layout(
            height=800,
            title_text="Dataset Distribution Analysis",
            template='plotly_white',
            showlegend=False
        )
        figures['dataset_distribution'] = fig_distribution
        
        return figures
    
    def render_streamlit_dashboard(self):
        """Render the complete dashboard in Streamlit."""
        st.set_page_config(
            page_title="Amazon Electronics Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 Amazon Electronics Analytics Dashboard")
        st.markdown("---")
        
        # Sidebar configuration
        st.sidebar.title("Dashboard Configuration")
        
        # Data loading section
        data_dir = Path("../data/processed")
        if st.sidebar.button("Load Data"):
            with st.spinner("Loading data..."):
                success = self.load_data(data_dir)
                if success:
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to load data. Please check the data directory.")
        
        if not self.data_loaded:
            st.warning("Please load data using the sidebar button to view analytics.")
            return
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", "⏰ Temporal Trends", 
            "🏷️ Category Insights", "⭐ Rating Patterns"
        ])
        
        with tab1:
            st.header("Comprehensive Overview")
            overview_figures = self.create_comprehensive_overview()
            
            if overview_figures:
                for title, fig in overview_figures.items():
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Temporal Trends Analysis")
            temporal_figures = self.create_temporal_trends_dashboard()
            
            if temporal_figures:
                for title, fig in temporal_figures.items():
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Category Insights")
            category_figures = self.create_category_insights_dashboard()
            
            if category_figures:
                for title, fig in category_figures.items():
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("Rating Patterns Analysis")
            rating_figures = self.create_rating_patterns_dashboard()
            
            if rating_figures:
                for title, fig in rating_figures.items():
                    st.plotly_chart(fig, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Amazon Electronics Analytics Dashboard** | "
            "Built with Streamlit and Plotly | "
            f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


def main():
    """Main function to run the dashboard."""
    dashboard = InteractiveDashboard()
    dashboard.render_streamlit_dashboard()


if __name__ == "__main__":
    main()