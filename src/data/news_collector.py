# src/data/news_collector.py
"""
🇩🇪 Deutscher News Collector - KOMPLETTE ERWEITERTE VERSION
20+ deutsche Medienquellen mit regionaler Abdeckung & Filtern
"""

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
    🇩🇪 Deutscher News Collector - MASSIV ERWEITERT
    20+ deutsche Medienquellen + Regionale Medien + Filter
    """
    
    def __init__(self):
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.guardian_key = os.getenv('GUARDIAN_API_KEY')
        
        # Validate API Keys
        if not self.newsapi_key:
            raise ValueError("❌ NEWSAPI_KEY fehlt in .env!")
        if not self.guardian_key:
            raise ValueError("❌ GUARDIAN_API_KEY fehlt in .env!")
        
        # ERWEITERTE German Media Sources (NewsAPI IDs)
        self.german_sources = {
            # Überregionale Qualitätsmedien
            'spiegel-online': {'name': 'SPIEGEL ONLINE', 'bias': 'center-left', 'type': 'quality', 'region': 'national'},
            'zeit-online': {'name': 'ZEIT ONLINE', 'bias': 'center-left', 'type': 'quality', 'region': 'national'},
            'handelsblatt': {'name': 'Handelsblatt', 'bias': 'center-right', 'type': 'business', 'region': 'national'},
            'welt': {'name': 'DIE WELT', 'bias': 'center-right', 'type': 'quality', 'region': 'national'},
            
            # Populäre Online-Medien
            'bild': {'name': 'BILD', 'bias': 'right', 'type': 'tabloid', 'region': 'national'},
            'focus': {'name': 'FOCUS Online', 'bias': 'center-right', 'type': 'news', 'region': 'national'},
            
            # Business & Wirtschaft
            'wirtschafts-woche': {'name': 'WirtschaftsWoche', 'bias': 'center-right', 'type': 'business', 'region': 'national'},
        }
        
        # MASSIV ERWEITERTE RSS Feeds
        self.rss_feeds = {
            # Öffentlich-Rechtliche
            'tagesschau': {
                'name': 'Tagesschau',
                'url': 'https://www.tagesschau.de/xml/rss2/',
                'bias': 'center',
                'type': 'public',
                'region': 'national'
            },
            'zdf_heute': {
                'name': 'ZDF heute',
                'url': 'https://www.zdf.de/rss/zdf/nachrichten',
                'bias': 'center',
                'type': 'public',
                'region': 'national'
            },
            'deutschlandfunk': {
                'name': 'Deutschlandfunk',
                'url': 'https://www.deutschlandfunk.de/die-nachrichten.353.de.rss',
                'bias': 'center',
                'type': 'public',
                'region': 'national'
            },
            
            # Printmedien Online
            'sueddeutsche': {
                'name': 'Süddeutsche Zeitung',
                'url': 'https://rss.sueddeutsche.de/app/service/rss/alles/rss.xml',
                'bias': 'center-left',
                'type': 'quality',
                'region': 'bayern'
            },
            'faz': {
                'name': 'FAZ',
                'url': 'https://www.faz.net/rss/aktuell/',
                'bias': 'center-right',
                'type': 'quality',
                'region': 'national'
            },
            'taz': {
                'name': 'taz',
                'url': 'https://taz.de/rss.xml',
                'bias': 'left',
                'type': 'alternative',
                'region': 'national'
            },
            'tagesspiegel': {
                'name': 'Der Tagesspiegel',
                'url': 'https://www.tagesspiegel.de/contentexport/feed/home',
                'bias': 'center-left',
                'type': 'quality',
                'region': 'berlin'
            },
            
            # Online-Medien
            'ntv': {
                'name': 'n-tv',
                'url': 'https://www.n-tv.de/rss',
                'bias': 'center-right',
                'type': 'news',
                'region': 'national'
            },
            't_online': {
                'name': 't-online',
                'url': 'https://feeds.t-online.de/rss/nachrichten',
                'bias': 'center',
                'type': 'portal',
                'region': 'national'
            },
            
            # Regionale Medien - Bayern
            'merkur': {
                'name': 'Merkur.de',
                'url': 'https://www.merkur.de/lokales/rssfeed.rdf',
                'bias': 'center-right',
                'type': 'regional',
                'region': 'bayern'
            },
            'abendzeitung_muenchen': {
                'name': 'Abendzeitung München',
                'url': 'https://www.abendzeitung-muenchen.de/rss.xml',
                'bias': 'center',
                'type': 'regional',
                'region': 'bayern'
            },
            
            # Regionale Medien - NRW
            'rp_online': {
                'name': 'RP ONLINE',
                'url': 'https://rp-online.de/feed.rss',
                'bias': 'center',
                'type': 'regional',
                'region': 'nrw'
            },
            'waz': {
                'name': 'WAZ',
                'url': 'https://www.waz.de/rss/nachrichten/',
                'bias': 'center',
                'type': 'regional',
                'region': 'nrw'
            },
            'express': {
                'name': 'EXPRESS',
                'url': 'https://www.express.de/feed/nachrichten',
                'bias': 'center-right',
                'type': 'regional',
                'region': 'nrw'
            },
            
            # Regionale Medien - Hamburg/Nord
            'ndr_hamburg': {
                'name': 'NDR Hamburg',
                'url': 'https://www.ndr.de/nachrichten/hamburg/index-rss.xml',
                'bias': 'center',
                'type': 'regional',
                'region': 'hamburg'
            },
            'abendblatt': {
                'name': 'Hamburger Abendblatt',
                'url': 'https://www.abendblatt.de/feed/rss/',
                'bias': 'center-right',
                'type': 'regional',
                'region': 'hamburg'
            },
            'mopo': {
                'name': 'Hamburger Morgenpost',
                'url': 'https://www.mopo.de/feed/',
                'bias': 'center',
                'type': 'regional',
                'region': 'hamburg'
            },
            'shz': {
                'name': 'shz.de',
                'url': 'https://www.shz.de/rss/nachrichten.xml',
                'bias': 'center',
                'type': 'regional',
                'region': 'schleswig-holstein'
            },
            
            # Regionale Medien - Berlin
            'berliner_zeitung': {
                'name': 'Berliner Zeitung',
                'url': 'https://www.berliner-zeitung.de/news.feed',
                'bias': 'center',
                'type': 'regional',
                'region': 'berlin'
            },
            
            # Regionale Medien - Sachsen/Ost
            'lvz': {
                'name': 'Leipziger Volkszeitung',
                'url': 'https://www.lvz.de/rss/feed/lvz_nachrichten',
                'bias': 'center',
                'type': 'regional',
                'region': 'sachsen'
            },
            'sz_online': {
                'name': 'Sächsische Zeitung',
                'url': 'https://www.saechsische.de/rss/nachrichten.xml',
                'bias': 'center',
                'type': 'regional',
                'region': 'sachsen'
            },
            
            # Wirtschaft Spezial
            'manager_magazin': {
                'name': 'manager magazin',
                'url': 'https://www.manager-magazin.de/news/index.rss',
                'bias': 'center-right',
                'type': 'business',
                'region': 'national'
            },
            'capital': {
                'name': 'Capital',
                'url': 'https://www.capital.de/feed/nachrichten',
                'bias': 'center-right',
                'type': 'business',
                'region': 'national'
            },
            
            # Sport
            'kicker': {
                'name': 'kicker',
                'url': 'https://www.kicker.de/news/fussball/bundesliga/rss.xml',
                'bias': 'center',
                'type': 'sport',
                'region': 'national'
            },
            'sport1': {
                'name': 'SPORT1',
                'url': 'https://www.sport1.de/news.rss',
                'bias': 'center',
                'type': 'sport',
                'region': 'national'
            },
            
            # Tech & IT
            'golem': {
                'name': 'Golem.de',
                'url': 'https://rss.golem.de/rss.php?feed=RSS2.0',
                'bias': 'center',
                'type': 'tech',
                'region': 'national'
            },
            'heise': {
                'name': 'heise online',
                'url': 'https://www.heise.de/rss/heise-atom.xml',
                'bias': 'center',
                'type': 'tech',
                'region': 'national'
            },
            'computerbild': {
                'name': 'COMPUTER BILD',
                'url': 'https://www.computerbild.de/rss/nachrichten.xml',
                'bias': 'center',
                'type': 'tech',
                'region': 'national'
            }
        }
        
        logger.info(f"✅ GermanNewsCollector initialisiert mit {len(self.german_sources)} NewsAPI + {len(self.rss_feeds)} RSS Quellen!")
    
    def get_sources_by_region(self, region: str = None) -> dict:
        """🗺️ Filtere Quellen nach Region"""
        if not region:
            return {**self.german_sources, **self.rss_feeds}
        
        filtered_sources = {}
        
        # NewsAPI Quellen
        for source_id, info in self.german_sources.items():
            if info.get('region') == region or region == 'national':
                filtered_sources[source_id] = info
        
        # RSS Quellen
        for source_id, info in self.rss_feeds.items():
            if info.get('region') == region or region == 'national':
                filtered_sources[source_id] = info
        
        return filtered_sources
    
    def get_sources_by_type(self, media_type: str = None) -> dict:
        """📰 Filtere Quellen nach Medientyp"""
        if not media_type:
            return {**self.german_sources, **self.rss_feeds}
        
        filtered_sources = {}
        
        # NewsAPI Quellen
        for source_id, info in self.german_sources.items():
            if info.get('type') == media_type:
                filtered_sources[source_id] = info
        
        # RSS Quellen
        for source_id, info in self.rss_feeds.items():
            if info.get('type') == media_type:
                filtered_sources[source_id] = info
        
        return filtered_sources
    
    def collect_newsapi_articles(self, 
                               sources: List[str] = None, 
                               days_back: int = 1,
                               language: str = 'de') -> pd.DataFrame:
        """📰 Sammelt deutsche News von NewsAPI (ERWEITERT)"""
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
                    'pageSize': 30,
                    'apiKey': self.newsapi_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] == 'ok':
                    source_info = self.german_sources.get(source_id, {'name': source_id, 'bias': 'unknown'})
                    
                    for article in data['articles']:
                        if article['title'] and article['title'] != '[Removed]':
                            articles.append({
                                'title': article['title'],
                                'description': article['description'],
                                'content': article['content'],
                                'url': article['url'],
                                'source': source_info['name'],
                                'source_id': source_id,
                                'bias': source_info['bias'],
                                'media_type': source_info.get('type', 'news'),
                                'region': source_info.get('region', 'national'),
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
            df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
            df = df.sort_values('published_at', ascending=False)
        
        logger.info(f"📊 NewsAPI: {len(df)} Artikel gesammelt")
        return df
    
    def collect_rss_articles(self, 
                           feeds: List[str] = None, 
                           max_articles: int = 15) -> pd.DataFrame:
        """📡 Sammelt deutsche News von RSS Feeds (MASSIV ERWEITERT)"""
        logger.info("🔍 Sammle RSS Feed Artikel...")
        
        articles = []
        
        # Default: alle RSS Feeds
        if not feeds:
            feeds = list(self.rss_feeds.keys())
        
        for feed_id in feeds:
            try:
                feed_info = self.rss_feeds[feed_id]
                
                # User-Agent für bessere RSS Kompatibilität
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                feed = feedparser.parse(feed_info['url'], request_headers=headers)
                
                # Fallback wenn Feed leer
                if not feed.entries:
                    logger.warning(f"⚠️ Kein Inhalt von {feed_info['name']}")
                    continue
                
                for entry in feed.entries[:max_articles]:
                    # Parse Date
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published_date = datetime(*entry.updated_parsed[:6])
                        else:
                            published_date = datetime.now()
                    except:
                        published_date = datetime.now()
                    
                    # Content extrahieren
                    content = ''
                    description = ''
                    
                    if hasattr(entry, 'content') and entry.content:
                        if isinstance(entry.content, list):
                            content = entry.content[0].get('value', '')
                        else:
                            content = str(entry.content)
                    
                    if hasattr(entry, 'summary'):
                        description = entry.summary[:500]
                    elif hasattr(entry, 'description'):
                        description = entry.description[:500]
                    
                    # Titel validieren
                    title = entry.get('title', '')
                    if not title or title == '[Removed]':
                        continue
                    
                    articles.append({
                        'title': title,
                        'description': description,
                        'content': content,
                        'url': entry.get('link', ''),
                        'source': feed_info['name'],
                        'source_id': feed_id,
                        'bias': feed_info['bias'],
                        'media_type': feed_info.get('type', 'news'),
                        'region': feed_info.get('region', 'national'),
                        'published_at': published_date,
                        'author': entry.get('author', ''),
                        'url_to_image': '',
                        'api_source': 'rss'
                    })
                
                logger.info(f"✅ {feed_info['name']}: {len(feed.entries[:max_articles])} Artikel")
                
                # Rate Limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ RSS Fehler bei {feed_id}: {str(e)}")
                continue
        
        df = pd.DataFrame(articles)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
            df = df.sort_values('published_at', ascending=False)
        
        logger.info(f"📊 RSS: {len(df)} Artikel gesammelt")
        return df
    
    def collect_guardian_german_news(self, 
                                   query: str = "Germany", 
                                   days_back: int = 1) -> pd.DataFrame:
        """🇬🇧➡️🇩🇪 Guardian News über Deutschland"""
        logger.info(f"🔍 Sammle Guardian News über '{query}'...")
        
        articles = []
        
        # Date Range
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        try:
            url = "https://content.guardianapis.com/search"
            params = {
                'q': query,
                'from-date': from_date,
                'page-size': 20,
                'show-fields': 'headline,byline,body,thumbnail',
                'api-key': self.guardian_key
            }
            
            response = requests.get(url, params=params, timeout=10)
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
                        'media_type': 'international',
                        'region': 'international',
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
            df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
            df = df.sort_values('published_at', ascending=False)
        
        logger.info(f"📊 Guardian: {len(df)} Artikel gesammelt")
        return df
    
    def collect_all_news(self, days_back: int = 1, region: str = None, media_type: str = None) -> pd.DataFrame:
        """🌍 Sammelt ALLE deutschen News aus allen Quellen (ERWEITERT mit Filtern)"""
        logger.info("🚀 Starte vollständige News-Sammlung...")
        
        all_articles = []
        
        # Filter anwenden
        if region or media_type:
            logger.info(f"📍 Filter: Region={region}, Type={media_type}")
        
        # 1. NewsAPI Deutsche Medien
        try:
            newsapi_sources = list(self.german_sources.keys())
            if region or media_type:
                filtered = self.get_sources_by_region(region) if region else self.german_sources
                if media_type:
                    filtered = {k: v for k, v in filtered.items() if v.get('type') == media_type}
                newsapi_sources = [k for k in newsapi_sources if k in filtered]
            
            if newsapi_sources:
                newsapi_df = self.collect_newsapi_articles(sources=newsapi_sources, days_back=days_back)
                if not newsapi_df.empty:
                    newsapi_df['published_at'] = pd.to_datetime(newsapi_df['published_at'], utc=True)
                    all_articles.append(newsapi_df)
        except Exception as e:
            logger.error(f"❌ NewsAPI Collection Error: {str(e)}")
        
        # 2. RSS Feeds
        try:
            rss_sources = list(self.rss_feeds.keys())
            if region or media_type:
                filtered = self.get_sources_by_region(region) if region else self.rss_feeds
                if media_type:
                    filtered = {k: v for k, v in filtered.items() if v.get('type') == media_type}
                rss_sources = [k for k in rss_sources if k in filtered]
            
            if rss_sources:
                rss_df = self.collect_rss_articles(feeds=rss_sources)
                if not rss_df.empty:
                    rss_df['published_at'] = pd.to_datetime(rss_df['published_at'], utc=True)
                    all_articles.append(rss_df)
        except Exception as e:
            logger.error(f"❌ RSS Collection Error: {str(e)}")
        
        # 3. Guardian Deutschland-News (nur wenn international erwünscht)
        if not media_type or media_type == 'international':
            try:
                guardian_df = self.collect_guardian_german_news(days_back=days_back)
                if not guardian_df.empty:
                    guardian_df['published_at'] = pd.to_datetime(guardian_df['published_at'], utc=True)
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
        """📊 Statistiken über gesammelte Artikel (ERWEITERT)"""
        if df.empty:
            return {}
        
        # Timezone-sicher konvertieren
        try:
            min_date = df['published_at'].min()
            max_date = df['published_at'].max()
            
            if hasattr(min_date, 'tz') and min_date.tz is not None:
                min_str = min_date.strftime('%Y-%m-%d %H:%M')
                max_str = max_date.strftime('%Y-%m-%d %H:%M')
            else:
                min_str = min_date.strftime('%Y-%m-%d %H:%M')
                max_str = max_date.strftime('%Y-%m-%d %H:%M')
        except:
            min_str = "Unknown"
            max_str = "Unknown"
        
        stats = {
            'total_articles': len(df),
            'sources_count': df['source'].nunique(),
            'date_range': {
                'from': min_str,
                'to': max_str
            },
            'by_source': df['source'].value_counts().to_dict(),
            'by_bias': df['bias'].value_counts().to_dict(),
            'by_api': df['api_source'].value_counts().to_dict(),
            'by_media_type': df['media_type'].value_counts().to_dict(),
            'by_region': df['region'].value_counts().to_dict()
        }
        
        return stats
    
    def get_available_regions(self) -> List[str]:
        """🗺️ Alle verfügbaren Regionen"""
        regions = set()
        for source_info in {**self.german_sources, **self.rss_feeds}.values():
            regions.add(source_info.get('region', 'national'))
        return sorted(list(regions))
    
    def get_available_media_types(self) -> List[str]:
        """📰 Alle verfügbaren Medientypen"""
        types = set()
        for source_info in {**self.german_sources, **self.rss_feeds}.values():
            types.add(source_info.get('type', 'news'))
        return sorted(list(types))


# =================================
# 🧪 TEST FUNCTION
# =================================

def test_extended_news_collector():
    """🧪 Teste den erweiterten News Collector"""
    
    print("🌍 TESTE ERWEITERTEN GERMAN NEWS COLLECTOR")
    print("=" * 60)
    
    try:
        # Initialize Collector
        collector = GermanNewsCollector()
        
        print(f"📊 Verfügbare Regionen: {collector.get_available_regions()}")
        print(f"📰 Verfügbare Medientypen: {collector.get_available_media_types()}")
        
        # Test 1: Nur Bayern
        print(f"\n🗺️ Test 1: Nur Bayern-Medien...")
        bayern_df = collector.collect_all_news(days_back=1, region='bayern')
        print(f"✅ Bayern: {len(bayern_df)} Artikel")
        if not bayern_df.empty:
            bayern_sources = bayern_df['source'].unique()
            print(f"   Quellen: {', '.join(bayern_sources)}")
        
        # Test 2: Nur Business
        print(f"\n💰 Test 2: Nur Business-Medien...")
        business_df = collector.collect_all_news(days_back=1, media_type='business')
        print(f"✅ Business: {len(business_df)} Artikel")
        if not business_df.empty:
            business_sources = business_df['source'].unique()
            print(f"   Quellen: {', '.join(business_sources)}")
        
        # Test 3: Bayern + Business
        print(f"\n🏢 Test 3: Bayern Business-Medien...")
        bayern_business_df = collector.collect_all_news(days_back=1, region='bayern', media_type='business')
        print(f"✅ Bayern Business: {len(bayern_business_df)} Artikel")
        
        # Test 4: Alle Quellen
        print(f"\n🌍 Test 4: ALLE Quellen...")
        all_df = collector.collect_all_news(days_back=1)
        print(f"✅ GESAMT: {len(all_df)} Artikel")
        
        if not all_df.empty:
            stats = collector.get_source_statistics(all_df)
            print(f"📊 Quellen: {stats['sources_count']}")
            print(f"📰 Nach Typ: {stats['by_media_type']}")
            print(f"🗺️ Nach Region: {stats['by_region']}")
            
            # Top 5 Quellen
            print(f"\n🔥 TOP 5 AKTIVSTE QUELLEN:")
            for i, (source, count) in enumerate(list(stats['by_source'].items())[:5], 1):
                print(f"   {i}. {source}: {count} Artikel")
        
        print(f"\n🎉 ERWEITERTE NEWS COLLECTION ERFOLGREICH!")
        print(f"🚀 Bereit für Dashboard-Integration!")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FEHLER: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_extended_news_collector()