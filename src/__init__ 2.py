# src/data/news_collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
import logging
from typing import Dict, List, Optional
import feedparser

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GermanNewsCollector:
    """
    🇩🇪 Deutscher News Collector
    Sammelt News von deutschen Medien via APIs und RSS
    """
    
    def __init__(self):
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.guardian_key = os.getenv('GUARDIAN_API_KEY')
        
        # Validate API Keys
        if not self.newsapi_key:
            raise ValueError("❌ NEWSAPI_KEY fehlt in .env!")
        if not self.guardian_key:
            raise ValueError("❌ GUARDIAN_API_KEY fehlt in .env!")
        
        # German Media Sources (NewsAPI IDs)
        self.german_sources = {
            'spiegel-online': {'name': 'SPIEGEL ONLINE', 'bias': 'center-left'},
            'bild': {'name': 'BILD', 'bias': 'right'},
            'zeit-online': {'name': 'ZEIT ONLINE', 'bias': 'center-left'},
            'focus': {'name': 'FOCUS Online', 'bias': 'center-right'},
            'handelsblatt': {'name': 'Handelsblatt', 'bias': 'center-right'},
            'welt': {'name': 'DIE WELT', 'bias': 'center-right'}
        }
        
        # RSS Feeds (als Backup/Ergänzung)
        self.rss_feeds = {
            'tagesschau': {
                'name': 'Tagesschau',
                'url': 'https://www.tagesschau.de/xml/rss2/',
                'bias': 'center'
            },
            'spiegel_rss': {
                'name': 'SPIEGEL RSS',
                'url': 'https://www.spiegel.de/schlagzeilen/index.rss',
                'bias': 'center-left'
            }
        }
        
        logger.info("✅ GermanNewsCollector initialisiert!")
    
    def collect_newsapi_articles(self, 
                               sources: List[str] = None, 
                               days_back: int = 1,
                               language: str = 'de') -> pd.DataFrame:
        """
        📰 Sammelt deutsche News von NewsAPI
        """
        logger.info(f"🔍 Sammle NewsAPI Artikel (letzte {days_back} Tage)...")
        
        articles = []
        
        # Default: alle deutschen Quellen
        if not sources:
            sources = list(self.german_sources.keys())
        
        # Date Range
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for source_id in sources:
            try:
                # NewsAPI Request
                url = "https://newsapi.org/v2/everything"
                params = {
                    'sources': source_id,
                    'from': from_date,
                    'language': language,
                    'sortBy': 'publishedAt',
                    'pageSize': 100,
                    'apiKey': self.newsapi_key
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] == 'ok':
                    source_info = self.german_sources.get(source_id, {'name': source_id, 'bias': 'unknown'})
                    
                    for article in data['articles']:
                        articles.append({
                            'title': article['title'],
                            'description': article['description'],
                            'content': article['content'],
                            'url': article['url'],
                            'source': source_info['name'],
                            'source_id': source_id,
                            'bias': source_info['bias'],
                            'published_at': article['publishedAt'],
                            'author': article['author'],
                            'url_to_image': article['urlToImage'],
                            'api_source': 'newsapi'
                        })
                    
                    logger.info(f"✅ {source_info['name']}: {len(data['articles'])} Artikel")
                else:
                    logger.error(f"❌ NewsAPI Error für {source_id}: {data.get('message', 'Unknown error')}")
                
                # Rate Limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Fehler bei {source_id}: {str(e)}")
                continue
        
        df = pd.DataFrame(articles)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.sort_values('published_at', ascending=False)
        
        logger.info(f"📊 NewsAPI: {len(df)} Artikel gesammelt")
        return df
    
    def collect_rss_articles(self, 
                           feeds: List[str] = None, 
                           max_articles: int = 50) -> pd.DataFrame:
        """
        📡 Sammelt deutsche News von RSS Feeds
        """
        logger.info("🔍 Sammle RSS Feed Artikel...")
        
        articles = []
        
        # Default: alle RSS Feeds
        if not feeds:
            feeds = list(self.rss_feeds.keys())
        
        for feed_id in feeds:
            try:
                feed_info = self.rss_feeds[feed_id]
                feed = feedparser.parse(feed_info['url'])
                
                for entry in feed.entries[:max_articles]:
                    # Parse Date
                    try:
                        published_date = datetime(*entry.published_parsed[:6])
                    except:
                        published_date = datetime.now()
                    
                    articles.append({
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'content': entry.get('content', [{'value': ''}])[0].get('value', '') if 'content' in entry else '',
                        'url': entry.get('link', ''),
                        'source': feed_info['name'],
                        'source_id': feed_id,
                        'bias': feed_info['bias'],
                        'published_at': published_date,
                        'author': entry.get('author', ''),
                        'url_to_image': '',
                        'api_source': 'rss'
                    })
                
                logger.info(f"✅ {feed_info['name']}: {len(feed.entries[:max_articles])} Artikel")
                
            except Exception as e:
                logger.error(f"❌ RSS Fehler bei {feed_id}: {str(e)}")
                continue
        
        df = pd.DataFrame(articles)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.sort_values('published_at', ascending=False)
        
        logger.info(f"📊 RSS: {len(df)} Artikel gesammelt")
        return df
    
    def collect_guardian_german_news(self, 
                                   query: str = "Germany", 
                                   days_back: int = 1) -> pd.DataFrame:
        """
        🇬🇧➡️🇩🇪 Guardian News über Deutschland
        """
        logger.info(f"🔍 Sammle Guardian News über '{query}'...")
        
        articles = []
        
        # Date Range
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        try:
            url = "https://content.guardianapis.com/search"
            params = {
                'q': query,
                'from-date': from_date,
                'page-size': 50,
                'show-fields': 'headline,byline,body,thumbnail',
                'api-key': self.guardian_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['response']['status'] == 'ok':
                for article in data['response']['results']:
                    fields = article.get('fields', {})
                    articles.append({
                        'title': fields.get('headline', article.get('webTitle', '')),
                        'description': fields.get('body', '')[:500] + '...' if fields.get('body') else '',
                        'content': fields.get('body', ''),
                        'url': article['webUrl'],
                        'source': 'The Guardian',
                        'source_id': 'guardian',
                        'bias': 'center-left',
                        'published_at': article['webPublicationDate'],
                        'author': fields.get('byline', ''),
                        'url_to_image': fields.get('thumbnail', ''),
                        'api_source': 'guardian'
                    })
                
                logger.info(f"✅ Guardian: {len(data['response']['results'])} Artikel")
            
        except Exception as e:
            logger.error(f"❌ Guardian API Fehler: {str(e)}")
        
        df = pd.DataFrame(articles)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.sort_values('published_at', ascending=False)
        
        logger.info(f"📊 Guardian: {len(df)} Artikel gesammelt")
        return df
    
    def collect_all_news(self, days_back: int = 1) -> pd.DataFrame:
        """
        🌍 Sammelt ALLE deutschen News aus allen Quellen
        """
        logger.info("🚀 Starte vollständige News-Sammlung...")
        
        all_articles = []
        
        # 1. NewsAPI Deutsche Medien
        try:
            newsapi_df = self.collect_newsapi_articles(days_back=days_back)
            if not newsapi_df.empty:
                all_articles.append(newsapi_df)
        except Exception as e:
            logger.error(f"❌ NewsAPI Collection Error: {str(e)}")
        
        # 2. RSS Feeds
        try:
            rss_df = self.collect_rss_articles()
            if not rss_df.empty:
                all_articles.append(rss_df)
        except Exception as e:
            logger.error(f"❌ RSS Collection Error: {str(e)}")
        
        # 3. Guardian Deutschland-News
        try:
            guardian_df = self.collect_guardian_german_news(days_back=days_back)
            if not guardian_df.empty:
                all_articles.append(guardian_df)
        except Exception as e:
            logger.error(f"❌ Guardian Collection Error: {str(e)}")
        
        # Combine all DataFrames
        if all_articles:
            combined_df = pd.concat(all_articles, ignore_index=True)
            
            # Remove Duplicates (based on title similarity)
            combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
            
            # Sort by publication date
            combined_df = combined_df.sort_values('published_at', ascending=False)
            
            logger.info(f"🎉 GESAMT: {len(combined_df)} einzigartige Artikel gesammelt!")
            return combined_df
        else:
            logger.warning("⚠️ Keine Artikel gesammelt!")
            return pd.DataFrame()
    
    def get_source_statistics(self, df: pd.DataFrame) -> Dict:
        """
        📊 Statistiken über gesammelte Artikel
        """
        if df.empty:
            return {}
        
        stats = {
            'total_articles': len(df),
            'sources_count': df['source'].nunique(),
            'date_range': {
                'from': df['published_at'].min().strftime('%Y-%m-%d %H:%M'),
                'to': df['published_at'].max().strftime('%Y-%m-%d %H:%M')
            },
            'by_source': df['source'].value_counts().to_dict(),
            'by_bias': df['bias'].value_counts().to_dict(),
            'by_api': df['api_source'].value_counts().to_dict()
        }
        
        return stats


# =================================
# 🧪 TEST FUNCTION
# =================================

def test_news_collector():
    """
    🧪 Teste den News Collector
    """
    print("🚀 TESTE GERMAN NEWS COLLECTOR")
    print("=" * 50)
    
    try:
        # Initialize Collector
        collector = GermanNewsCollector()
        
        # Test 1: NewsAPI
        print("\n📰 Test 1: NewsAPI Deutsche Medien...")
        newsapi_df = collector.collect_newsapi_articles(
            sources=['spiegel-online', 'bild'], 
            days_back=1
        )
        print(f"✅ NewsAPI: {len(newsapi_df)} Artikel")
        if not newsapi_df.empty:
            print(f"📊 Neueste Schlagzeile: {newsapi_df.iloc[0]['title']}")
        
        # Test 2: RSS Feeds
        print("\n📡 Test 2: RSS Feeds...")
        rss_df = collector.collect_rss_articles(feeds=['tagesschau'])
        print(f"✅ RSS: {len(rss_df)} Artikel")
        if not rss_df.empty:
            print(f"📊 Neueste Schlagzeile: {rss_df.iloc[0]['title']}")
        
        # Test 3: Guardian
        print("\n🇬🇧 Test 3: Guardian Deutschland-News...")
        guardian_df = collector.collect_guardian_german_news(query="Germany")
        print(f"✅ Guardian: {len(guardian_df)} Artikel")
        if not guardian_df.empty:
            print(f"📊 Neueste Schlagzeile: {guardian_df.iloc[0]['title']}")
        
        # Test 4: Alle Quellen
        print("\n🌍 Test 4: ALLE Quellen kombiniert...")
        all_df = collector.collect_all_news(days_back=1)
        print(f"✅ GESAMT: {len(all_df)} Artikel")
        
        # Statistiken
        if not all_df.empty:
            stats = collector.get_source_statistics(all_df)
            print(f"\n📊 STATISTIKEN:")
            print(f"   • Quellen: {stats['sources_count']}")
            print(f"   • Zeitraum: {stats['date_range']['from']} bis {stats['date_range']['to']}")
            print(f"   • Pro Quelle: {stats['by_source']}")
            print(f"   • Pro Bias: {stats['by_bias']}")
            
            # Top 3 Headlines
            print(f"\n🔥 TOP 3 AKTUELLE SCHLAGZEILEN:")
            for i, row in all_df.head(3).iterrows():
                print(f"   {i+1}. [{row['source']}] {row['title']}")
        
        print("\n🎉 ALLE TESTS ERFOLGREICH!")
        return True
        
    except Exception as e:
        print(f"❌ TEST FEHLER: {str(e)}")
        return False


if __name__ == "__main__":
    test_news_collector()