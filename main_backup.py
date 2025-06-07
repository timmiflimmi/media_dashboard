# main.py
"""
📰 MEDIA PULSE DASHBOARD
Deutsches Media Analytics Dashboard mit Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data.news_collector import GermanNewsCollector
    from src.analysis.sentiment_analyzer import GermanNLPAnalyzer
except ImportError:
    st.error("❌ Module nicht gefunden! Stelle sicher, dass src/data/ und src/analysis/ existieren.")
    st.stop()

# Page Config
st.set_page_config(
    page_title="📰 Media Pulse Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .news-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f9f9f9;
    }
    
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    
    .bias-left { color: #ff6b6b; }
    .bias-right { color: #ffeaa7; }
    .bias-center { color: #45b7d1; }
    .bias-populist { color: #fd79a8; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Initialize analyzers
@st.cache_resource
def load_analyzers():
    """Load News Collector and NLP Analyzer"""
    try:
        collector = GermanNewsCollector()
        nlp_analyzer = GermanNLPAnalyzer()
        return collector, nlp_analyzer
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Analyzer: {str(e)}")
        return None, None

@st.cache_data(ttl=1800)  # Cache für 30 Minuten
def collect_and_analyze_news(days_back=1):
    """Collect and analyze news with caching"""
    
    collector, nlp_analyzer = load_analyzers()
    if not collector or not nlp_analyzer:
        return None
    
    # Collect News
    with st.spinner("📰 Sammle aktuelle deutsche News..."):
        df = collector.collect_all_news(days_back=days_back)
    
    if df.empty:
        return None
    
    # NLP Analysis
    with st.spinner("🧠 Analysiere Sentiment und Bias..."):
        analyzed_articles = []
        total_articles = len(df)
        
        # Status container for updates
        status_container = st.empty()
        
        for idx, row in df.iterrows():
            # NLP Analysis
            analysis = nlp_analyzer.analyze_article(
                title=row.get('title', ''),
                content=row.get('description', '')
            )
            
            # Combine data
            article_data = {
                **row.to_dict(),
                'sentiment_score': analysis['sentiment']['compound'],
                'sentiment_class': analysis['sentiment']['classification'],
                'sentiment_confidence': analysis['sentiment']['confidence'],
                'bias_type': analysis['bias']['bias_type'],
                'bias_confidence': analysis['bias']['confidence'],
                'is_hamburg': analysis['hamburg_related']['is_hamburg'],
                'hamburg_keywords': analysis['hamburg_related']['found_keywords'],
                'keywords': [kw['word'] for kw in analysis['keywords'][:5]],
                'word_count': analysis['text_stats']['word_count']
            }
            
            analyzed_articles.append(article_data)
            
            # Update status every 5 articles
            if (idx + 1) % 5 == 0 or (idx + 1) == total_articles:
                status_container.text(f"🔄 Analysiert: {idx + 1}/{total_articles} Artikel")
        
        status_container.empty()
    
    return pd.DataFrame(analyzed_articles)

def show_overview_metrics(df):
    """📊 Overview Metrics"""
    
    if df is None or df.empty:
        st.warning("⚠️ Keine Daten verfügbar")
        return
    
    st.markdown("### 📊 Aktuelle Medien-Übersicht")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_articles = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_articles}</h3>
            <p>Artikel insgesamt</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sources_count = df['source'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{sources_count}</h3>
            <p>Medienquellen</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        hamburg_articles = df['is_hamburg'].sum()
        hamburg_percent = (hamburg_articles / total_articles * 100) if total_articles > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{hamburg_articles}</h3>
            <p>Hamburg-Bezug ({hamburg_percent:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😟"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{sentiment_emoji}</h3>
            <p>Ø Sentiment ({avg_sentiment:.2f})</p>
        </div>
        """, unsafe_allow_html=True)

def show_sentiment_analysis(df):
    """😊 Sentiment Analysis Charts"""
    
    if df is None or df.empty:
        return
    
    st.markdown("### 😊 Sentiment-Analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution
        sentiment_counts = df['sentiment_class'].value_counts()
        
        colors = {
            'positive': '#28a745',
            'negative': '#dc3545', 
            'neutral': '#6c757d'
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=[colors.get(label, '#cccccc') for label in sentiment_counts.index]
        )])
        
        fig_pie.update_layout(
            title="Sentiment-Verteilung",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment by Source
        sentiment_by_source = df.groupby(['source', 'sentiment_class']).size().unstack(fill_value=0)
        
        fig_bar = go.Figure()
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_by_source.columns:
                fig_bar.add_trace(go.Bar(
                    name=sentiment.title(),
                    x=sentiment_by_source.index,
                    y=sentiment_by_source[sentiment],
                    marker_color=colors.get(sentiment, '#cccccc')
                ))
        
        fig_bar.update_layout(
            title="Sentiment nach Quelle",
            xaxis_title="Medienquelle",
            yaxis_title="Anzahl Artikel",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

def show_bias_analysis(df):
    """🎯 Bias Analysis"""
    
    if df is None or df.empty:
        return
    
    st.markdown("### 🎯 Bias-Analyse")
    
    # Bias Distribution
    bias_counts = df['bias_type'].value_counts()
    
    bias_colors = {
        'left_leaning': '#ff6b6b',
        'right_leaning': '#ffeaa7',
        'center': '#45b7d1',
        'populist': '#fd79a8'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bias = go.Figure(data=[go.Bar(
            x=bias_counts.index,
            y=bias_counts.values,
            marker_color=[bias_colors.get(bias, '#cccccc') for bias in bias_counts.index]
        )])
        
        fig_bias.update_layout(
            title="Bias-Verteilung",
            xaxis_title="Bias-Typ",
            yaxis_title="Anzahl Artikel",
            height=400
        )
        
        st.plotly_chart(fig_bias, use_container_width=True)
    
    with col2:
        # Bias vs Sentiment
        bias_sentiment = df.groupby(['bias_type', 'sentiment_class']).size().unstack(fill_value=0)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=bias_sentiment.values,
            x=bias_sentiment.columns,
            y=bias_sentiment.index,
            colorscale='RdYlBu_r'
        ))
        
        fig_heatmap.update_layout(
            title="Bias vs Sentiment Heatmap",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def show_hamburg_analysis(df):
    """🏙️ Hamburg Analysis"""
    
    if df is None or df.empty:
        return
    
    st.markdown("### 🏙️ Hamburg-Fokus")
    
    hamburg_df = df[df['is_hamburg'] == True]
    
    if hamburg_df.empty:
        st.info("ℹ️ Keine Hamburg-spezifischen Artikel gefunden")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hamburg Articles by Source
        hamburg_by_source = hamburg_df['source'].value_counts()
        
        fig_hamburg = go.Figure(data=[go.Bar(
            x=hamburg_by_source.index,
            y=hamburg_by_source.values,
            marker_color='#17a2b8'
        )])
        
        fig_hamburg.update_layout(
            title=f"Hamburg-Artikel nach Quelle ({len(hamburg_df)} total)",
            height=400
        )
        st.plotly_chart(fig_hamburg, use_container_width=True)
    
    with col2:
        # Hamburg Sentiment
        hamburg_sentiment = hamburg_df['sentiment_class'].value_counts()
        
        fig_hamburg_sentiment = go.Figure(data=[go.Pie(
            labels=hamburg_sentiment.index,
            values=hamburg_sentiment.values
        )])
        
        fig_hamburg_sentiment.update_layout(
            title="Hamburg-Artikel Sentiment",
            height=400
        )
        st.plotly_chart(fig_hamburg_sentiment, use_container_width=True)

def show_latest_headlines(df, n=10):
    """🔥 Latest Headlines"""
    
    if df is None or df.empty:
        return
    
    st.markdown(f"### 🔥 Aktuelle Schlagzeilen (Top {n})")
    
    # Sort by publication date
    latest_df = df.sort_values('published_at', ascending=False).head(n)
    
    for idx, row in latest_df.iterrows():
        # Sentiment styling
        sentiment_class = row['sentiment_class']
        sentiment_style = f"sentiment-{sentiment_class}"
        
        # Bias styling  
        bias_class = row['bias_type'].replace('_', '-')
        bias_style = f"bias-{bias_class}"
        
        # Hamburg indicator
        hamburg_indicator = "🏙️" if row['is_hamburg'] else ""
        
        st.markdown(f"""
        <div class="news-card">
            <h4>{hamburg_indicator} {row['title']}</h4>
            <p><strong>Quelle:</strong> <span class="{bias_style}">{row['source']}</span> | 
               <strong>Sentiment:</strong> <span class="{sentiment_style}">{sentiment_class.title()}</span> 
               ({row['sentiment_score']:.2f}) | 
               <strong>Bias:</strong> <span class="{bias_style}">{row['bias_type'].replace('_', ' ').title()}</span></p>
            <p>{row.get('description', '')[:200]}...</p>
            <p><small>📅 {row['published_at']}</small></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """🚀 Main Dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Media Pulse Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### 🇩🇪 Deutsche Medien-Analyse mit KI-Power")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Einstellungen")
        
        # Data Collection Settings
        days_back = st.slider("📅 Tage zurück", 1, 7, 1)
        auto_refresh = st.checkbox("🔄 Auto-Refresh (30 Min)", value=True)
        
        # Manual Refresh Button
        if st.button("🔄 Daten neu laden", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        # Last Update Info
        if st.session_state.last_update:
            st.info(f"🕐 Letztes Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        st.markdown("**🎯 Features:**")
        st.markdown("- 📰 Deutsche Medien-APIs")
        st.markdown("- 🧠 NLP Sentiment-Analyse") 
        st.markdown("- 🎯 Bias-Detection")
        st.markdown("- 🏙️ Hamburg-Fokus")
        st.markdown("- 📊 Interactive Charts")
    
    # Load Data
    try:
        df = collect_and_analyze_news(days_back=days_back)
        st.session_state.news_data = df
        st.session_state.last_update = datetime.now()
        
        if df is None or df.empty:
            st.error("❌ Keine News-Daten verfügbar. Prüfe deine API-Keys in .env!")
            st.stop()
        
        # Dashboard Content
        show_overview_metrics(df)
        
        st.markdown("---")
        
        # Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["😊 Sentiment", "🎯 Bias", "🏙️ Hamburg", "🔥 Headlines"])
        
        with tab1:
            show_sentiment_analysis(df)
        
        with tab2:
            show_bias_analysis(df)
        
        with tab3:
            show_hamburg_analysis(df)
        
        with tab4:
            show_latest_headlines(df)
        
        # Footer
        st.markdown("---")
        st.markdown("**📊 Media Pulse Dashboard** | Powered by 🧠 German NLP & 📰 News APIs")
        
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {str(e)}")
        st.error("Stelle sicher, dass deine .env Datei korrekte API Keys enthält!")

if __name__ == "__main__":
    main()