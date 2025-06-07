# src/analysis/media_clash_detector.py
"""
⚔️ Media Clash Detector
Erkennt gleiche Stories in verschiedenen Medien und vergleicht Perspektiven
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MediaClashDetector:
    """
    ⚔️ Erkennt gleiche Stories in verschiedenen Medien
    und analysiert unterschiedliche Perspektiven
    """
    
    def __init__(self):
        # TF-IDF Vectorizer für deutsche Texte
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=self._get_german_stopwords(),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Medien-Gruppierungen für Clash Detection
        self.media_groups = {
            'links': {
                'medien': ['taz', 'Süddeutsche Zeitung', 'SPIEGEL ONLINE', 'ZEIT ONLINE'],
                'farbe': '#ff6b6b',
                'label': 'Links/Liberal'
            },
            'mitte': {
                'medien': ['Tagesschau', 'ZDF heute', 'Deutschlandfunk', 'FAZ'],
                'farbe': '#45b7d1', 
                'label': 'Mitte/Öffentlich-Rechtlich'
            },
            'rechts': {
                'medien': ['BILD', 'DIE WELT', 'FOCUS Online'],
                'farbe': '#ffeaa7',
                'label': 'Rechts/Konservativ'
            },
            'business': {
                'medien': ['Handelsblatt', 'manager magazin', 'Capital', 'WirtschaftsWoche'],
                'farbe': '#96CEB4',
                'label': 'Wirtschaft'
            },
            'regional': {
                'medien': ['Merkur.de', 'RP ONLINE', 'NDR Hamburg', 'Der Tagesspiegel'],
                'farbe': '#17a2b8',
                'label': 'Regional'
            }
        }
        
        # Clash-Schwellenwerte
        self.similarity_threshold = 0.3  # Mindest-Ähnlichkeit für Story-Clustering
        self.min_sources = 2  # Mindestanzahl Quellen für Clash
        self.max_time_diff = 24  # Max Stunden zwischen Artikeln
        
        logger.info("✅ MediaClashDetector initialisiert!")
    
    def _get_german_stopwords(self) -> List[str]:
        """🇩🇪 Deutsche Stopwords für TF-IDF"""
        return [
            'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
            'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine',
            'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass',
            'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch',
            'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur',
            'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein',
            'wurde', 'sei', 'kann', 'wenn', 'soll', 'da', 'ihr', 'seine', 'einem',
            'alle', 'zwei', 'drei', 'heute', 'jahren', 'jahr', 'zeit', 'ersten',
            'neue', 'neuen', 'gegen', 'bereits', 'sowie', 'unter', 'beim', 'seit'
        ]
    
    def _clean_text_for_similarity(self, text: str) -> str:
        """🧹 Text für Ähnlichkeitsvergleich bereinigen"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Entferne Sonderzeichen, behalte nur Buchstaben und Spaces
        text = re.sub(r'[^a-züäöß\s]', ' ', text)
        
        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def find_story_clusters(self, df: pd.DataFrame) -> Dict:
        """🔍 Finde ähnliche Stories zwischen verschiedenen Medien"""
        
        if df.empty:
            return {}
        
        logger.info(f"🔍 Analysiere {len(df)} Artikel auf Story-Cluster...")
        
        # Texte für Similarity vorbereiten
        texts = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            # Titel und Description kombinieren
            combined_text = f"{row.get('title', '')} {row.get('description', '')}"
            cleaned_text = self._clean_text_for_similarity(combined_text)
            
            if len(cleaned_text) > 20:  # Mindestlänge
                texts.append(cleaned_text)
                valid_indices.append(idx)
        
        if len(texts) < 2:
            logger.warning("⚠️ Zu wenige valide Texte für Clustering")
            return {}
        
        # TF-IDF Vektorisierung
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Cosine Similarity berechnen
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
        except Exception as e:
            logger.error(f"❌ TF-IDF Fehler: {str(e)}")
            return {}
        
        # Story Clusters finden
        clusters = {}
        processed = set()
        cluster_id = 0
        
        for i, idx_i in enumerate(valid_indices):
            if i in processed:
                continue
            
            # Ähnliche Artikel finden
            similar_articles = [idx_i]
            processed.add(i)
            
            for j, idx_j in enumerate(valid_indices):
                if i != j and j not in processed:
                    if similarity_matrix[i][j] >= self.similarity_threshold:
                        similar_articles.append(idx_j)
                        processed.add(j)
            
            # Nur Cluster mit mehreren Artikeln aus verschiedenen Quellen
            if len(similar_articles) >= self.min_sources:
                cluster_articles = df.loc[similar_articles]
                unique_sources = cluster_articles['source'].nunique()
                
                if unique_sources >= self.min_sources:
                    # Zeitfenster prüfen
                    time_diff = (cluster_articles['published_at'].max() - 
                               cluster_articles['published_at'].min()).total_seconds() / 3600
                    
                    if time_diff <= self.max_time_diff:
                        clusters[f"cluster_{cluster_id}"] = {
                            'articles': similar_articles,
                            'article_count': len(similar_articles),
                            'source_count': unique_sources,
                            'time_span_hours': round(time_diff, 1),
                            'main_topic': self._extract_main_topic(cluster_articles),
                            'avg_similarity': float(np.mean([similarity_matrix[i][j] 
                                                           for i in range(len(similar_articles)) 
                                                           for j in range(i+1, len(similar_articles))]))
                        }
                        cluster_id += 1
        
        logger.info(f"✅ {len(clusters)} Story-Cluster gefunden")
        return clusters
    
    def _extract_main_topic(self, articles: pd.DataFrame) -> str:
        """📝 Extrahiere Hauptthema aus Artikel-Cluster"""
        
        # Häufigste Wörter in Titeln
        all_titles = ' '.join(articles['title'].fillna(''))
        words = re.findall(r'\b[a-züäöß]{4,}\b', all_titles.lower())
        
        # Stopwords entfernen
        stopwords = set(self._get_german_stopwords())
        words = [w for w in words if w not in stopwords]
        
        if words:
            most_common = Counter(words).most_common(3)
            return ', '.join([word.title() for word, _ in most_common])
        
        return "Unbekanntes Thema"
    
    def analyze_media_perspectives(self, df: pd.DataFrame, cluster_info: Dict) -> Dict:
        """📊 Analysiere Medien-Perspektiven für jedes Cluster"""
        
        cluster_analyses = {}
        
        for cluster_id, cluster_data in cluster_info.items():
            article_indices = cluster_data['articles']
            cluster_articles = df.loc[article_indices]
            
            # Mediengruppen-Analyse
            media_perspectives = {}
            
            for group_name, group_info in self.media_groups.items():
                group_articles = cluster_articles[
                    cluster_articles['source'].isin(group_info['medien'])
                ]
                
                if not group_articles.empty:
                    # Sentiment-Analyse
                    avg_sentiment = group_articles['sentiment_score'].mean()
                    
                    # Bias-Analyse
                    bias_distribution = group_articles['bias_type'].value_counts().to_dict()
                    
                    # Politik-Fokus
                    politik_articles = group_articles[group_articles['is_politik'] == True]
                    politik_ratio = len(politik_articles) / len(group_articles)
                    
                    # Häufigste Keywords
                    all_keywords = []
                    for _, row in group_articles.iterrows():
                        if isinstance(row.get('keywords'), list):
                            all_keywords.extend(row['keywords'])
                    
                    top_keywords = Counter(all_keywords).most_common(5)
                    
                    media_perspectives[group_name] = {
                        'group_info': group_info,
                        'article_count': len(group_articles),
                        'sources': group_articles['source'].unique().tolist(),
                        'avg_sentiment': round(avg_sentiment, 3),
                        'bias_distribution': bias_distribution,
                        'politik_ratio': round(politik_ratio, 2),
                        'top_keywords': [kw for kw, _ in top_keywords],
                        'sample_headlines': group_articles['title'].head(3).tolist()
                    }
            
            # Bias-Gap Berechnung
            bias_gap = self._calculate_bias_gap(media_perspectives)
            
            # Sentiment-Spektrum
            sentiment_spectrum = self._calculate_sentiment_spectrum(media_perspectives)
            
            cluster_analyses[cluster_id] = {
                **cluster_data,
                'media_perspectives': media_perspectives,
                'bias_gap': bias_gap,
                'sentiment_spectrum': sentiment_spectrum,
                'clash_score': self._calculate_clash_score(media_perspectives)
            }
        
        return cluster_analyses
    
    def _calculate_bias_gap(self, perspectives: Dict) -> float:
        """🎯 Berechne Bias-Gap zwischen Mediengruppen"""
        
        sentiments = []
        for group_data in perspectives.values():
            sentiments.append(group_data['avg_sentiment'])
        
        if len(sentiments) >= 2:
            return round(max(sentiments) - min(sentiments), 3)
        return 0.0
    
    def _calculate_sentiment_spectrum(self, perspectives: Dict) -> Dict:
        """😊 Berechne Sentiment-Spektrum"""
        
        spectrum = {
            'most_positive': {'group': None, 'score': -2.0},
            'most_negative': {'group': None, 'score': 2.0},
            'range': 0.0
        }
        
        for group_name, group_data in perspectives.items():
            sentiment = group_data['avg_sentiment']
            
            if sentiment > spectrum['most_positive']['score']:
                spectrum['most_positive'] = {
                    'group': group_name,
                    'score': sentiment,
                    'label': group_data['group_info']['label']
                }
            
            if sentiment < spectrum['most_negative']['score']:
                spectrum['most_negative'] = {
                    'group': group_name,
                    'score': sentiment,
                    'label': group_data['group_info']['label']
                }
        
        spectrum['range'] = round(
            spectrum['most_positive']['score'] - spectrum['most_negative']['score'], 3
        )
        
        return spectrum
    
    def _calculate_clash_score(self, perspectives: Dict) -> float:
        """⚔️ Berechne Clash-Intensität (0-1)"""
        
        if len(perspectives) < 2:
            return 0.0
        
        # Faktoren für Clash-Score
        sentiment_variance = 0.0
        bias_diversity = len(set(
            bias for group_data in perspectives.values() 
            for bias in group_data['bias_distribution'].keys()
        ))
        
        sentiments = [group_data['avg_sentiment'] for group_data in perspectives.values()]
        if sentiments:
            sentiment_variance = np.var(sentiments)
        
        # Clash Score Formel
        clash_score = min(1.0, (sentiment_variance * 2 + bias_diversity * 0.1))
        
        return round(clash_score, 3)
    
    def get_top_clashes(self, cluster_analyses: Dict, top_n: int = 5) -> List[Dict]:
        """🔥 Hole die interessantesten Media Clashes"""
        
        clashes = []
        
        for cluster_id, analysis in cluster_analyses.items():
            clash_score = analysis['clash_score']
            bias_gap = analysis['bias_gap']
            media_count = len(analysis['media_perspectives'])
            
            if clash_score > 0.1 and media_count >= 2:
                clashes.append({
                    'cluster_id': cluster_id,
                    'topic': analysis['main_topic'],
                    'clash_score': clash_score,
                    'bias_gap': bias_gap,
                    'media_count': media_count,
                    'article_count': analysis['article_count'],
                    'time_span': analysis['time_span_hours'],
                    'sentiment_spectrum': analysis['sentiment_spectrum'],
                    'perspectives': analysis['media_perspectives']
                })
        
        # Sortiere nach Clash-Score
        clashes.sort(key=lambda x: x['clash_score'], reverse=True)
        
        return clashes[:top_n]
    
    def generate_clash_summary(self, df: pd.DataFrame) -> Dict:
        """📊 Generiere Media Clash Summary für Dashboard"""
        
        # Story Clusters finden
        clusters = self.find_story_clusters(df)
        
        if not clusters:
            return {
                'total_clusters': 0,
                'total_clashes': 0,
                'top_clashes': [],
                'avg_clash_score': 0.0,
                'most_controversial_topic': None
            }
        
        # Perspektiven analysieren
        cluster_analyses = self.analyze_media_perspectives(df, clusters)
        
        # Top Clashes
        top_clashes = self.get_top_clashes(cluster_analyses)
        
        # Statistiken
        clash_scores = [analysis['clash_score'] for analysis in cluster_analyses.values()]
        avg_clash_score = np.mean(clash_scores) if clash_scores else 0.0
        
        # Kontroversestes Thema
        most_controversial = None
        if top_clashes:
            most_controversial = top_clashes[0]
        
        return {
            'total_clusters': len(clusters),
            'total_clashes': len(top_clashes),
            'top_clashes': top_clashes,
            'avg_clash_score': round(avg_clash_score, 3),
            'most_controversial_topic': most_controversial,
            'cluster_analyses': cluster_analyses
        }


# =================================
# 🧪 TEST FUNCTION
# =================================

def test_media_clash_detector():
    """🧪 Teste den Media Clash Detector"""
    
    print("⚔️ TESTE MEDIA CLASH DETECTOR")
    print("=" * 50)
    
    # Erstelle Test-Daten
    test_data = {
        'title': [
            'Scholz kritisiert Opposition wegen Haushaltspolitik',
            'Kanzler Scholz unter Druck - Opposition attackiert Finanzpolitik',
            'Scholz verteidigt Haushaltsplan gegen massive Kritik',
            'Grüne fordern mehr Klimaschutz in neuer Energiepolitik',
            'Habeck: Energiewende muss beschleunigt werden',
            'AfD kritisiert Regierung scharf wegen Migrationspolitik',
            'Weidel attackiert Ampel-Koalition bei Migrationsdebatte'
        ],
        'description': [
            'Der Bundeskanzler wehrt sich gegen Vorwürfe der Opposition bezüglich der Finanzplanung',
            'Scharfe Kritik aus CDU und AfD an den Haushaltsplänen der Ampel-Regierung',
            'Olaf Scholz rechtfertigt die Ausgabenpolitik der Bundesregierung',
            'Die Grünen-Politiker fordern schnellere Maßnahmen für den Klimaschutz',
            'Wirtschaftsminister Habeck will die Energiewende vorantreiben',
            'Alice Weidel übt scharfe Kritik an der Migrationspolitik der Regierung',
            'AfD-Chefin attackiert die Ampel-Koalition in der Migrationsdebatte'
        ],
        'source': ['Tagesschau', 'BILD', 'SPIEGEL ONLINE', 'taz', 'Süddeutsche Zeitung', 'FOCUS Online', 'DIE WELT'],
        'sentiment_score': [0.1, -0.3, 0.0, 0.2, 0.1, -0.5, -0.4],
        'bias_type': ['center', 'right', 'center-left', 'left', 'center-left', 'center-right', 'center-right'],
        'is_politik': [True] * 7,
        'keywords': [['scholz', 'haushalt'], ['kritik', 'opposition'], ['verteidigung'], 
                    ['grüne', 'klima'], ['habeck', 'energie'], ['afd', 'migration'], ['weidel', 'ampel']],
        'published_at': [datetime.now() - timedelta(hours=i) for i in range(7)]
    }
    
    df = pd.DataFrame(test_data)
    
    print(f"📊 Test mit {len(df)} Artikeln:")
    for _, row in df.iterrows():
        print(f"   • {row['source']}: {row['title'][:40]}...")
    
    # Media Clash Detector testen
    detector = MediaClashDetector()
    
    # Clash-Analyse
    print(f"\n⚔️ Führe Clash-Analyse durch...")
    clash_summary = detector.generate_clash_summary(df)
    
    print(f"\n📊 CLASH SUMMARY:")
    print(f"   🔍 Story-Cluster: {clash_summary['total_clusters']}")
    print(f"   ⚔️ Media Clashes: {clash_summary['total_clashes']}")
    print(f"   📈 Ø Clash-Score: {clash_summary['avg_clash_score']}")
    
    if clash_summary['most_controversial_topic']:
        topic = clash_summary['most_controversial_topic']
        print(f"\n🔥 KONTROVERSESTES THEMA:")
        print(f"   📰 Topic: {topic['topic']}")
        print(f"   ⚔️ Clash-Score: {topic['clash_score']}")
        print(f"   🎯 Bias-Gap: {topic['bias_gap']}")
        print(f"   📊 Medien: {topic['media_count']}")
        
        print(f"\n📊 MEDIEN-PERSPEKTIVEN:")
        for group_name, perspective in topic['perspectives'].items():
            print(f"   {group_name.upper()}: {perspective['group_info']['label']}")
            print(f"      😊 Sentiment: {perspective['avg_sentiment']}")
            print(f"      📰 Quellen: {', '.join(perspective['sources'])}")
            print(f"      🔑 Keywords: {', '.join(perspective['top_keywords'][:3])}")
    
    print(f"\n🎉 MEDIA CLASH DETECTOR GETESTET!")
    print(f"🚀 Bereit für Dashboard-Integration!")
    
    return True

if __name__ == "__main__":
    test_media_clash_detector()