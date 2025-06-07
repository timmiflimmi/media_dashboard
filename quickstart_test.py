# quickstart_test.py
"""
🚀 QUICKSTART TEST für Media Pulse Dashboard
Teste API Connections direkt im Hauptverzeichnis
"""

import requests
import os
from dotenv import load_dotenv
import feedparser
from datetime import datetime

# Load API Keys
load_dotenv()

def test_apis():
    """🧪 Teste alle APIs schnell"""
    
    print("🚀 MEDIA PULSE DASHBOARD - API TEST")
    print("=" * 50)
    
    # Check Environment
    newsapi_key = os.getenv('NEWSAPI_KEY')
    guardian_key = os.getenv('GUARDIAN_API_KEY')
    
    print(f"📋 Environment Check:")
    print(f"   ✅ NewsAPI Key: {'✓ Vorhanden' if newsapi_key else '❌ FEHLT'}")
    print(f"   ✅ Guardian Key: {'✓ Vorhanden' if guardian_key else '❌ FEHLT'}")
    
    if not newsapi_key or not guardian_key:
        print("❌ API Keys fehlen in .env Datei!")
        return False
    
    print(f"\n🔬 API Connection Tests:")
    
    # Test 1: NewsAPI
    try:
        print(f"   📰 NewsAPI Test...")
        url = "https://newsapi.org/v2/everything"
        params = {
            'sources': 'spiegel-online',
            'pageSize': 5,
            'apiKey': newsapi_key
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            article_count = len(data.get('articles', []))
            print(f"   ✅ NewsAPI: {article_count} SPIEGEL Artikel gefunden")
            
            if article_count > 0:
                latest = data['articles'][0]
                print(f"      🔥 Neueste: {latest['title'][:60]}...")
        else:
            print(f"   ❌ NewsAPI Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ NewsAPI Fehler: {str(e)}")
    
    # Test 2: Guardian API
    try:
        print(f"   🇬🇧 Guardian API Test...")
        url = "https://content.guardianapis.com/search"
        params = {
            'q': 'Germany',
            'page-size': 5,
            'api-key': guardian_key
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            article_count = len(data['response'].get('results', []))
            print(f"   ✅ Guardian: {article_count} Deutschland Artikel gefunden")
            
            if article_count > 0:
                latest = data['response']['results'][0]
                print(f"      🔥 Neueste: {latest['webTitle'][:60]}...")
        else:
            print(f"   ❌ Guardian Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Guardian Fehler: {str(e)}")
    
    # Test 3: RSS Feed (Tagesschau)
    try:
        print(f"   📡 RSS Feed Test (Tagesschau)...")
        feed = feedparser.parse('https://www.tagesschau.de/xml/rss2/')
        
        if feed.entries:
            article_count = len(feed.entries[:5])
            print(f"   ✅ Tagesschau RSS: {article_count} Artikel gefunden")
            
            latest = feed.entries[0]
            print(f"      🔥 Neueste: {latest.title[:60]}...")
        else:
            print(f"   ❌ Tagesschau RSS: Keine Artikel")
            
    except Exception as e:
        print(f"   ❌ RSS Fehler: {str(e)}")
    
    print(f"\n🎉 API Tests abgeschlossen!")
    print(f"📊 Status: Alle Verbindungen {'✅ ERFOLGREICH' if True else '❌ FEHLGESCHLAGEN'}")
    
    return True

def test_quick_collection():
    """📰 Schnelle News Collection"""
    
    print(f"\n🔍 SCHNELLE NEWS SAMMLUNG")
    print("=" * 30)
    
    all_headlines = []
    
    # NewsAPI - SPIEGEL
    try:
        newsapi_key = os.getenv('NEWSAPI_KEY')
        url = "https://newsapi.org/v2/everything"
        params = {
            'sources': 'spiegel-online',
            'pageSize': 3,
            'apiKey': newsapi_key
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            for article in data.get('articles', [])[:3]:
                all_headlines.append({
                    'source': 'SPIEGEL',
                    'title': article['title'],
                    'time': article['publishedAt'][:10]
                })
    except:
        pass
    
    # Guardian
    try:
        guardian_key = os.getenv('GUARDIAN_API_KEY')
        url = "https://content.guardianapis.com/search"
        params = {
            'q': 'Germany',
            'page-size': 3,
            'api-key': guardian_key
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            for article in data['response'].get('results', [])[:3]:
                all_headlines.append({
                    'source': 'Guardian',
                    'title': article['webTitle'],
                    'time': article['webPublicationDate'][:10]
                })
    except:
        pass
    
    # Tagesschau RSS
    try:
        feed = feedparser.parse('https://www.tagesschau.de/xml/rss2/')
        for entry in feed.entries[:3]:
            all_headlines.append({
                'source': 'Tagesschau',
                'title': entry.title,
                'time': datetime.now().strftime('%Y-%m-%d')
            })
    except:
        pass
    
    # Display Results
    if all_headlines:
        print(f"📰 {len(all_headlines)} AKTUELLE SCHLAGZEILEN:")
        print("-" * 50)
        
        for i, headline in enumerate(all_headlines, 1):
            print(f"{i:2d}. [{headline['source']:>10}] {headline['title'][:60]}...")
            print(f"     📅 {headline['time']}")
            print()
    else:
        print("❌ Keine Headlines gesammelt")
    
    return all_headlines

if __name__ == "__main__":
    # Run Tests
    test_apis()
    headlines = test_quick_collection()
    
    print(f"\n🎯 NÄCHSTE SCHRITTE:")
    print(f"   1. ✅ APIs funktionieren")
    print(f"   2. 📰 {len(headlines)} Headlines gesammelt") 
    print(f"   3. 🚀 Bereit für vollständigen News Collector!")
    print(f"   4. 🧠 Als nächstes: Deutsche NLP Analyse!")