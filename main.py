# main.py
"""
📰 MEDIA PULSE DASHBOARD - ERWEITERT
20+ Deutsche Medienquellen mit regionalen Filtern & Politik Bias Tracker
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
    from src.analysis.politik_analyzer import PolitikBiasTracker
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
    
    .filter-info {
        background: #e8f4fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
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
    """Load News Collector, NLP Analyzer und Politik Tracker"""
    try:
        collector = GermanNewsCollector()
        nlp_analyzer = GermanNLPAnalyzer()
        politik_tracker = PolitikBiasTracker()
        return collector, nlp_analyzer, politik_tracker
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Analyzer: {str(e)}")
        return None, None, None

@st.cache_data(ttl=1800)  # Cache für 30 Minuten
def collect_and_analyze_news_extended(days_back=1, region=None, media_type=None):
    """Collect and analyze news with EXTENDED filters"""
    
    collector, nlp_analyzer, politik_tracker = load_analyzers()
    if not collector or not nlp_analyzer or not politik_tracker:
        return None
    
    # Collect News mit Filtern
    filter_info = []
    if region:
        filter_info.append(f"Region: {region}")
    if media_type:
        filter_info.append(f"Typ: {media_type}")
    
    filter_text = f" ({', '.join(filter_info)})" if filter_info else ""
    
    with st.spinner(f"📰 Sammle deutsche News{filter_text}..."):
        df = collector.collect_all_news(
            days_back=days_back, 
            region=region, 
            media_type=media_type
        )
    
    if df.empty:
        return None
    
    # NLP Analysis + Politik Analysis
    with st.spinner("🧠 Analysiere Sentiment, Bias und Politik..."):
        analyzed_articles = []
        total_articles = len(df)
        
        # Status container for updates
        status_container = st.empty()
        
        for idx, row in df.iterrows():
            # Standard NLP Analysis
            nlp_analysis = nlp_analyzer.analyze_article(
                title=row.get('title', ''),
                content=row.get('description', '')
            )
            
            # Politik Analysis
            politik_analysis = politik_tracker.analyze_artikel_politik(
                title=row.get('title', ''),
                content=row.get('description', '')
            )
            
            # Combine data
            article_data = {
                **row.to_dict(),
                # Standard NLP
                'sentiment_score': nlp_analysis['sentiment']['compound'],
                'sentiment_class': nlp_analysis['sentiment']['classification'],
                'sentiment_confidence': nlp_analysis['sentiment']['confidence'],
                'bias_type': nlp_analysis['bias']['bias_type'],
                'bias_confidence': nlp_analysis['bias']['confidence'],
                'is_hamburg': nlp_analysis['hamburg_related']['is_hamburg'],
                'hamburg_keywords': nlp_analysis['hamburg_related']['found_keywords'],
                'keywords': [kw['word'] for kw in nlp_analysis['keywords'][:5]],
                'word_count': nlp_analysis['text_stats']['word_count'],
                # Politik Analysis
                'is_politik': politik_analysis['is_politik'],
                'parteien_mentions': politik_analysis['parteien_mentions'],
                'politik_themen': politik_analysis['politik_themen'],
                'partei_sentiments': politik_analysis['partei_sentiments'],
                'dominante_partei': politik_analysis['dominante_partei'],
                'dominantes_thema': politik_analysis['dominantes_thema'],
                'politik_relevanz_score': politik_analysis['politik_relevanz_score']
            }
            
            analyzed_articles.append(article_data)
            
            # Update status every 10 articles
            if (idx + 1) % 10 == 0 or (idx + 1) == total_articles:
                status_container.text(f"🔄 Analysiert: {idx + 1}/{total_articles} Artikel")
        
        status_container.empty()
    
    return pd.DataFrame(analyzed_articles)

def show_extended_overview_metrics(df, settings):
    """📊 Extended Overview Metrics mit regionalen Infos"""
    
    if df is None or df.empty:
        st.warning("⚠️ Keine Daten verfügbar")
        return
    
    # Filter Info anzeigen
    filter_info = []
    if settings.get('region'):
        filter_info.append(f"🗺️ Region: {settings['region'].title()}")
    if settings.get('media_type'):
        filter_info.append(f"📰 Typ: {settings['media_type'].title()}")
    
    if filter_info:
        st.markdown(f"""
        <div class="filter-info">
            <strong>🔍 Aktive Filter:</strong> {' | '.join(filter_info)}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Medien-Übersicht")
    
    # Erste Reihe - Hauptmetriken
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        politik_articles = df['is_politik'].sum()
        politik_percent = (politik_articles / total_articles * 100) if total_articles > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{politik_articles}</h3>
            <p>Politik ({politik_percent:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'region' in df.columns:
            regional_articles = len(df[df['region'] != 'national'])
            regional_percent = (regional_articles / total_articles * 100) if total_articles > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{regional_articles}</h3>
                <p>Regional ({regional_percent:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            hamburg_articles = df['is_hamburg'].sum()
            hamburg_percent = (hamburg_articles / total_articles * 100) if total_articles > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{hamburg_articles}</h3>
                <p>Hamburg ({hamburg_percent:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😟"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{sentiment_emoji}</h3>
            <p>Ø Sentiment ({avg_sentiment:.2f})</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Zweite Reihe - Erweiterte Metriken
    if 'media_type' in df.columns and 'region' in df.columns:
        st.markdown("#### 📊 Verteilung nach Typ & Region")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Medientyp-Verteilung
            type_counts = df['media_type'].value_counts()
            st.markdown("**📰 Nach Medientyp:**")
            for media_type, count in type_counts.head(5).items():
                percentage = (count / total_articles * 100)
                st.write(f"• **{media_type.title()}**: {count} ({percentage:.1f}%)")
        
        with col2:
            # Regional-Verteilung
            region_counts = df['region'].value_counts()
            st.markdown("**🗺️ Nach Region:**")
            for region, count in region_counts.head(5).items():
                percentage = (count / total_articles * 100)
                region_name = region.title() if region != 'national' else 'National'
                st.write(f"• **{region_name}**: {count} ({percentage:.1f}%)")

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

def show_politik_analysis(df):
    """🏛️ Politik Bias Analysis"""
    
    if df is None or df.empty:
        return
    
    st.markdown("### 🏛️ Politik Bias Tracker")
    
    # Politik Artikel filtern
    politik_df = df[df['is_politik'] == True]
    
    if politik_df.empty:
        st.info("ℹ️ Keine Politik-Artikel in diesem Zeitraum gefunden")
        return
    
    # Politik Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        politik_count = len(politik_df)
        politik_anteil = (politik_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Politik-Artikel", politik_count, f"{politik_anteil:.1f}% aller Artikel")
    
    with col2:
        unique_parteien = set()
        for _, row in politik_df.iterrows():
            unique_parteien.update(row.get('parteien_mentions', {}).keys())
        st.metric("Erwähnte Parteien", len(unique_parteien))
    
    with col3:
        avg_relevanz = politik_df['politik_relevanz_score'].mean()
        st.metric("Ø Politik-Relevanz", f"{avg_relevanz:.2f}")
    
    with col4:
        unique_themen = set()
        for _, row in politik_df.iterrows():
            unique_themen.update(row.get('politik_themen', {}).keys())
        st.metric("Politik-Themen", len(unique_themen))
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Parteien-Erwähnungen Chart
        st.markdown("#### 🏛️ Parteien-Erwähnungen")
        
        parteien_counts = {}
        parteien_info = {}
        
        for _, row in politik_df.iterrows():
            for partei, info in row.get('parteien_mentions', {}).items():
                if partei not in parteien_counts:
                    parteien_counts[partei] = 0
                    parteien_info[partei] = info
                parteien_counts[partei] += info['count']
        
        if parteien_counts:
            # Sortiere nach Häufigkeit
            sorted_parteien = sorted(parteien_counts.items(), key=lambda x: x[1], reverse=True)
            
            partei_names = [parteien_info[p[0]]['name'] for p in sorted_parteien]
            partei_values = [p[1] for p in sorted_parteien]
            partei_colors = [parteien_info[p[0]]['farbe'] for p in sorted_parteien]
            
            fig_parteien = go.Figure(data=[go.Bar(
                y=partei_names,
                x=partei_values,
                orientation='h',
                marker_color=partei_colors,
                text=partei_values,
                textposition='auto'
            )])
            
            fig_parteien.update_layout(
                title="Erwähnungen nach Parteien",
                xaxis_title="Anzahl Erwähnungen",
                height=400
            )
            
            st.plotly_chart(fig_parteien, use_container_width=True)
    
    with col2:
        # Sentiment by Partei
        st.markdown("#### 😊 Sentiment nach Parteien")
        
        partei_sentiments = {}
        
        for _, row in politik_df.iterrows():
            for partei, sentiment in row.get('partei_sentiments', {}).items():
                if partei not in partei_sentiments:
                    partei_sentiments[partei] = []
                partei_sentiments[partei].append(sentiment['score'])
        
        if partei_sentiments:
            avg_sentiments = {}
            for partei, scores in partei_sentiments.items():
                if scores:
                    avg_sentiments[partei] = sum(scores) / len(scores)
            
            if avg_sentiments:
                # Sortiere nach Sentiment
                sorted_sentiment = sorted(avg_sentiments.items(), key=lambda x: x[1], reverse=True)
                
                partei_names = [parteien_info[p[0]]['name'] for p in sorted_sentiment if p[0] in parteien_info]
                sentiment_values = [p[1] for p in sorted_sentiment if p[0] in parteien_info]
                
                # Farben basierend auf Sentiment
                colors = ['#28a745' if s > 0.1 else '#dc3545' if s < -0.1 else '#6c757d' for s in sentiment_values]
                
                fig_sentiment = go.Figure(data=[go.Bar(
                    y=partei_names,
                    x=sentiment_values,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{s:.2f}" for s in sentiment_values],
                    textposition='auto'
                )])
                
                fig_sentiment.update_layout(
                    title="Durchschnittliches Sentiment",
                    xaxis_title="Sentiment Score (-1 bis 1)",
                    height=400
                )
                
                st.plotly_chart(fig_sentiment, use_container_width=True)

def show_regional_analysis(df):
    """🗺️ Regional Analysis Dashboard"""
    
    if df is None or df.empty:
        return
    
    st.markdown("### 🗺️ Regionale Medienanalyse")
    
    # Prüfe ob Region-Spalte existiert
    if 'region' not in df.columns:
        st.info("ℹ️ Regionale Daten nicht verfügbar - verwende erweiterten News Collector")
        return
    
    # Filter auf regionale Artikel
    regional_df = df[df['region'] != 'national']
    
    if regional_df.empty:
        st.info("ℹ️ Keine regionalen Artikel in diesem Zeitraum")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional Distribution
        region_counts = regional_df['region'].value_counts()
        
        fig_regions = go.Figure(data=[go.Bar(
            x=region_counts.values,
            y=[r.title() for r in region_counts.index],
            orientation='h',
            marker_color='#17a2b8'
        )])
        
        fig_regions.update_layout(
            title="Artikel nach Regionen",
            xaxis_title="Anzahl Artikel",
            height=400
        )
        
        st.plotly_chart(fig_regions, use_container_width=True)
    
    with col2:
        # Regional Sentiment
        regional_sentiment = regional_df.groupby('region')['sentiment_score'].mean().sort_values(ascending=False)
        
        colors = ['#28a745' if s > 0.1 else '#dc3545' if s < -0.1 else '#6c757d' for s in regional_sentiment.values]
        
        fig_regional_sentiment = go.Figure(data=[go.Bar(
            x=regional_sentiment.values,
            y=[r.title() for r in regional_sentiment.index],
            orientation='h',
            marker_color=colors,
            text=[f"{s:.2f}" for s in regional_sentiment.values],
            textposition='auto'
        )])
        
        fig_regional_sentiment.update_layout(
            title="Durchschnittliches Sentiment nach Region",
            xaxis_title="Sentiment Score",
            height=400
        )
        
        st.plotly_chart(fig_regional_sentiment, use_container_width=True)

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
        
        # Politik indicator
        politik_indicator = "🏛️" if row['is_politik'] else ""
        
        # Regional indicator
        region_indicator = ""
        if 'region' in row and row['region'] != 'national':
            region_indicator = f"🗺️{row['region'].upper()}"
        
        st.markdown(f"""
        <div class="news-card">
            <h4>{hamburg_indicator}{politik_indicator}{region_indicator} {row['title']}</h4>
            <p><strong>Quelle:</strong> <span class="{bias_style}">{row['source']}</span> | 
               <strong>Sentiment:</strong> <span class="{sentiment_style}">{sentiment_class.title()}</span> 
               ({row['sentiment_score']:.2f}) | 
               <strong>Bias:</strong> <span class="{bias_style}">{row['bias_type'].replace('_', ' ').title()}</span></p>
            <p>{row.get('description', '')[:200]}...</p>
            <p><small>📅 {row['published_at']}</small></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """🚀 Main Dashboard mit erweiterten Filtern"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Media Pulse Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### 🇩🇪 Deutsche Medien-Analyse mit 20+ Quellen & regionalen Filtern")
    
    # Extended Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Einstellungen")
        
        # Data Collection Settings
        days_back = st.slider("📅 Tage zurück", 1, 7, 1)
        
        # Region Filter
        st.markdown("### 🗺️ Region")
        available_regions = ['alle', 'national', 'bayern', 'nrw', 'berlin', 'hamburg', 'sachsen', 'schleswig-holstein']
        selected_region = st.selectbox(
            "Wähle Region",
            available_regions,
            index=0,
            help="Filtere nach regionalen Medien"
        )
        
        # Medientyp Filter  
        st.markdown("### 📰 Medientyp")
        available_types = ['alle', 'quality', 'business', 'tech', 'public', 'regional', 'sport', 'alternative']
        selected_type = st.selectbox(
            "Wähle Medientyp", 
            available_types,
            index=0,
            help="Filtere nach Medienart"
        )
        
        # Advanced Settings
        with st.expander("🔧 Erweiterte Einstellungen"):
            auto_refresh = st.checkbox("🔄 Auto-Refresh (30 Min)", value=True)
            show_international = st.checkbox("🌍 Guardian einbeziehen", value=True)
        
        # Manual Refresh Button
        if st.button("🔄 Daten neu laden", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        # Last Update Info
        if st.session_state.last_update:
            st.info(f"🕐 Letztes Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        st.markdown("**🎯 Features:**")
        st.markdown("- 📰 20+ Deutsche Medien")
        st.markdown("- 🗺️ Regionale Filter")
        st.markdown("- 📊 Medientyp-Filter")
        st.markdown("- 🧠 NLP Sentiment-Analyse") 
        st.markdown("- 🎯 Bias-Detection")
        st.markdown("- 🏛️ Politik Bias Tracker")
        st.markdown("- 📊 Interactive Charts")
    
    # Settings zusammenfassen
    settings = {
        'days_back': days_back,
        'region': None if selected_region == 'alle' else selected_region,
        'media_type': None if selected_type == 'alle' else selected_type,
        'auto_refresh': auto_refresh,
        'show_international': show_international
    }
    
    # Load Data mit Filtern
    try:
        df = collect_and_analyze_news_extended(
            days_back=settings['days_back'],
            region=settings['region'],
            media_type=settings['media_type']
        )
        st.session_state.news_data = df
        st.session_state.last_update = datetime.now()
        
        if df is None or df.empty:
            st.error("❌ Keine News-Daten verfügbar. Prüfe Filter oder API-Keys!")
            st.stop()
        
        # Dashboard Content
        show_extended_overview_metrics(df, settings)
        
        st.markdown("---")
        
        # Analysis Tabs (ERWEITERT)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "😊 Sentiment", "🎯 Bias", "🏛️ Politik", 
            "🗺️ Regional", "🏙️ Hamburg", "🔥 Headlines"
        ])
        
        with tab1:
            show_sentiment_analysis(df)
        
        with tab2:
            show_bias_analysis(df)
        
        with tab3:
            show_politik_analysis(df)
        
        with tab4:
            show_regional_analysis(df)
        
        with tab5:
            show_hamburg_analysis(df)
        
        with tab6:
            show_latest_headlines(df)
        
        # Footer
        st.markdown("---")
        st.markdown("**📊 Media Pulse Dashboard** | 20+ Deutsche Medien | 🗺️ Regional | 🏛️ Politik Bias Tracker")
        
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {str(e)}")
        st.error("Stelle sicher, dass deine .env Datei korrekte API Keys enthält!")

if __name__ == "__main__":
    main()