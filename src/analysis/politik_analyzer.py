# src/analysis/politik_analyzer.py
"""
🏛️ Politik Bias Tracker für deutsche Medien
Analysiert Parteien-Berichterstattung und politische Trends
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PolitikBiasTracker:
    """
    🏛️ Analysiert politische Berichterstattung deutscher Medien
    """
    
    def __init__(self):
        # Deutsche Parteien mit Variationen
        self.parteien = {
            'cdu': {
                'name': 'CDU/CSU',
                'keywords': [
                    'cdu', 'csu', 'union', 'christdemokraten', 'merkel', 'söder', 
                    'merz', 'friedrich merz', 'markus söder', 'armin laschet',
                    'christlich demokratische union', 'christlich soziale union'
                ],
                'farbe': '#000000',  # Schwarz
                'spektrum': 'mitte-rechts'
            },
            'spd': {
                'name': 'SPD',
                'keywords': [
                    'spd', 'sozialdemokraten', 'scholz', 'olaf scholz', 'lars klingbeil',
                    'saskia esken', 'sozialdemokratische partei', 'sozialdemokratie'
                ],
                'farbe': '#E3000F',  # Rot
                'spektrum': 'mitte-links'
            },
            'gruene': {
                'name': 'Bündnis 90/Die Grünen',
                'keywords': [
                    'grüne', 'bündnis 90', 'habeck', 'baerbock', 'robert habeck',
                    'annalena baerbock', 'cem özdemir', 'katrin göring-eckardt',
                    'bündnis 90 die grünen', 'grünenpolitiker'
                ],
                'farbe': '#1AA037',  # Grün
                'spektrum': 'links'
            },
            'fdp': {
                'name': 'FDP',
                'keywords': [
                    'fdp', 'freie demokraten', 'lindner', 'christian lindner',
                    'liberale', 'freie demokratische partei', 'fdp-politiker'
                ],
                'farbe': '#FFED00',  # Gelb
                'spektrum': 'liberal'
            },
            'linke': {
                'name': 'Die Linke',
                'keywords': [
                    'linke', 'die linke', 'wagenknecht', 'sahra wagenknecht',
                    'bartsch', 'dietmar bartsch', 'linkspartei', 'linken-politiker'
                ],
                'farbe': '#BE3075',  # Magenta
                'spektrum': 'links'
            },
            'afd': {
                'name': 'AfD',
                'keywords': [
                    'afd', 'alternative für deutschland', 'weidel', 'alice weidel',
                    'gauland', 'alexander gauland', 'höcke', 'björn höcke',
                    'chrupalla', 'tino chrupalla', 'afd-politiker'
                ],
                'farbe': '#009EE0',  # Blau
                'spektrum': 'rechts'
            },
            'bsw': {
                'name': 'BSW',
                'keywords': [
                    'bsw', 'bündnis sahra wagenknecht', 'wagenknecht partei',
                    'sahra wagenknecht partei'
                ],
                'farbe': '#722F87',  # Lila
                'spektrum': 'links-populistisch'
            }
        }
        
        # Politische Themen
        self.politik_themen = {
            'wirtschaft': [
                'inflation', 'konjunktur', 'arbeitslosigkeit', 'mindestlohn',
                'steuern', 'haushalt', 'schulden', 'wirtschaftspolitik',
                'sozialleistungen', 'rente'
            ],
            'klima': [
                'klimawandel', 'energiewende', 'kohleausstieg', 'atomausstieg',
                'erneuerbare energien', 'co2', 'emissionen', 'klimaschutz',
                'umweltpolitik', 'nachhaltigkeit'
            ],
            'migration': [
                'migration', 'flüchtlinge', 'asyl', 'integration', 'abschiebung',
                'einwanderung', 'ausländer', 'migrationspolitik', 'grenze'
            ],
            'außenpolitik': [
                'nato', 'eu', 'europa', 'ukraine', 'russland', 'china',
                'biden', 'putin', 'trump', 'außenpolitik', 'diplomatie'
            ],
            'innenpolitik': [
                'bundestag', 'koalition', 'regierung', 'opposition', 'wahlen',
                'demokratie', 'verfassung', 'grundgesetz', 'bundesrat'
            ],
            'gesundheit': [
                'gesundheitspolitik', 'krankenversicherung', 'pflege',
                'corona', 'pandemie', 'impfung', 'gesundheitssystem'
            ]
        }
        
        # Sentiment-Wörter für Politik
        self.politik_sentiment = {
            'positiv': [
                'erfolg', 'fortschritt', 'verbesserung', 'reform', 'innovation',
                'lösung', 'kompromiss', 'einigung', 'zusammenarbeit', 'stabilität'
            ],
            'negativ': [
                'skandal', 'affäre', 'korruption', 'versagen', 'krise',
                'streit', 'konflikt', 'chaos', 'desaster', 'rücktritt'
            ]
        }
        
        logger.info("✅ PolitikBiasTracker initialisiert!")
    
    def detect_parteien_mentions(self, text: str) -> dict:
        """
        🏛️ Erkennt Parteien-Erwähnungen in Text
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        mentions = {}
        
        for partei_id, partei_info in self.parteien.items():
            count = 0
            found_keywords = []
            
            for keyword in partei_info['keywords']:
                # Wort-Grenzen beachten für präzise Suche
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    count += matches
                    found_keywords.append(keyword)
            
            if count > 0:
                mentions[partei_id] = {
                    'name': partei_info['name'],
                    'count': count,
                    'keywords': found_keywords,
                    'farbe': partei_info['farbe'],
                    'spektrum': partei_info['spektrum']
                }
        
        return mentions
    
    def detect_politik_themen(self, text: str) -> dict:
        """
        📊 Erkennt politische Themen in Text
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        themen = {}
        
        for thema, keywords in self.politik_themen.items():
            count = 0
            found_keywords = []
            
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    count += matches
                    found_keywords.append(keyword)
            
            if count > 0:
                themen[thema] = {
                    'count': count,
                    'keywords': found_keywords
                }
        
        return themen
    
    def analyze_politik_sentiment(self, text: str, partei: str) -> dict:
        """
        😊 Analysiert Sentiment in Bezug auf spezifische Partei
        """
        if not text:
            return {'sentiment': 'neutral', 'score': 0.0, 'indicators': []}
        
        text_lower = text.lower()
        
        # Suche Partei-Kontext
        partei_keywords = self.parteien.get(partei, {}).get('keywords', [])
        partei_mentioned = any(keyword in text_lower for keyword in partei_keywords)
        
        if not partei_mentioned:
            return {'sentiment': 'neutral', 'score': 0.0, 'indicators': []}
        
        # Sentiment-Analyse
        positive_count = 0
        negative_count = 0
        found_indicators = []
        
        for word in self.politik_sentiment['positiv']:
            if word in text_lower:
                positive_count += 1
                found_indicators.append(f"+{word}")
        
        for word in self.politik_sentiment['negativ']:
            if word in text_lower:
                negative_count += 1
                found_indicators.append(f"-{word}")
        
        # Score berechnen
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return {'sentiment': 'neutral', 'score': 0.0, 'indicators': []}
        
        score = (positive_count - negative_count) / total_sentiment_words
        
        if score > 0.2:
            sentiment = 'positiv'
        elif score < -0.2:
            sentiment = 'negativ'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': round(score, 3),
            'indicators': found_indicators
        }
    
    def analyze_artikel_politik(self, title: str, content: str) -> dict:
        """
        📰 Vollständige Politik-Analyse eines Artikels
        """
        full_text = f"{title} {content}" if content else title
        
        # Parteien-Erwähnungen
        parteien_mentions = self.detect_parteien_mentions(full_text)
        
        # Politische Themen
        politik_themen = self.detect_politik_themen(full_text)
        
        # Sentiment pro Partei
        partei_sentiments = {}
        for partei_id in parteien_mentions.keys():
            sentiment = self.analyze_politik_sentiment(full_text, partei_id)
            partei_sentiments[partei_id] = sentiment
        
        # Ist es ein Politik-Artikel?
        is_politik = len(parteien_mentions) > 0 or len(politik_themen) > 0
        
        # Dominante Partei
        dominante_partei = None
        if parteien_mentions:
            dominante_partei = max(parteien_mentions.keys(), 
                                 key=lambda x: parteien_mentions[x]['count'])
        
        # Dominantes Thema
        dominantes_thema = None
        if politik_themen:
            dominantes_thema = max(politik_themen.keys(),
                                 key=lambda x: politik_themen[x]['count'])
        
        return {
            'is_politik': is_politik,
            'parteien_mentions': parteien_mentions,
            'politik_themen': politik_themen,
            'partei_sentiments': partei_sentiments,
            'dominante_partei': dominante_partei,
            'dominantes_thema': dominantes_thema,
            'politik_relevanz_score': self._calculate_politik_score(parteien_mentions, politik_themen)
        }
    
    def _calculate_politik_score(self, parteien: dict, themen: dict) -> float:
        """
        🎯 Berechnet Politik-Relevanz Score (0-1)
        """
        partei_score = min(sum(p['count'] for p in parteien.values()) * 0.1, 0.7)
        thema_score = min(sum(t['count'] for t in themen.values()) * 0.05, 0.3)
        return min(partei_score + thema_score, 1.0)
    
    def generate_politik_summary(self, df: pd.DataFrame) -> dict:
        """
        📊 Generiert Politik-Dashboard Summary
        """
        politik_articles = df[df['is_politik'] == True]
        
        if politik_articles.empty:
            return {
                'total_politik_articles': 0,
                'politik_anteil': 0.0,
                'top_parteien': {},
                'top_themen': {},
                'sentiment_by_partei': {},
                'bias_by_source': {}
            }
        
        # Statistiken
        total_politik = len(politik_articles)
        politik_anteil = total_politik / len(df) * 100
        
        # Top Parteien
        all_parteien = Counter()
        for _, row in politik_articles.iterrows():
            for partei, info in row.get('parteien_mentions', {}).items():
                all_parteien[partei] += info['count']
        
        # Top Themen
        all_themen = Counter()
        for _, row in politik_articles.iterrows():
            for thema, info in row.get('politik_themen', {}).items():
                all_themen[thema] += info['count']
        
        # Sentiment by Partei
        sentiment_by_partei = defaultdict(list)
        for _, row in politik_articles.iterrows():
            for partei, sentiment in row.get('partei_sentiments', {}).items():
                sentiment_by_partei[partei].append(sentiment['score'])
        
        # Durchschnittliches Sentiment
        avg_sentiment_by_partei = {}
        for partei, scores in sentiment_by_partei.items():
            if scores:
                avg_sentiment_by_partei[partei] = {
                    'avg_score': round(sum(scores) / len(scores), 3),
                    'article_count': len(scores),
                    'name': self.parteien[partei]['name'],
                    'farbe': self.parteien[partei]['farbe']
                }
        
        # Bias by Source
        bias_by_source = {}
        for source in politik_articles['source'].unique():
            source_articles = politik_articles[politik_articles['source'] == source]
            
            source_sentiments = []
            for _, row in source_articles.iterrows():
                for sentiment in row.get('partei_sentiments', {}).values():
                    source_sentiments.append(sentiment['score'])
            
            if source_sentiments:
                bias_by_source[source] = {
                    'avg_sentiment': round(sum(source_sentiments) / len(source_sentiments), 3),
                    'article_count': len(source_articles),
                    'std_deviation': round(pd.Series(source_sentiments).std(), 3)
                }
        
        return {
            'total_politik_articles': total_politik,
            'politik_anteil': round(politik_anteil, 1),
            'top_parteien': dict(all_parteien.most_common(5)),
            'top_themen': dict(all_themen.most_common(5)),
            'sentiment_by_partei': avg_sentiment_by_partei,
            'bias_by_source': bias_by_source
        }


# =================================
# 🧪 TEST FUNCTION
# =================================

def test_politik_analyzer():
    """🧪 Teste den Politik Bias Tracker"""
    
    print("🏛️ TESTE POLITIK BIAS TRACKER")
    print("=" * 50)
    
    analyzer = PolitikBiasTracker()
    
    # Test Artikel
    test_articles = [
        {
            'title': 'Scholz und Merz streiten über Wirtschaftspolitik',
            'content': 'SPD-Kanzler Olaf Scholz kritisiert CDU-Chef Friedrich Merz für seine Steuerpläne. Die Inflation belastet die Koalition.',
            'source': 'Tagesschau'
        },
        {
            'title': 'Grüne fordern schärfere Klimaschutzmaßnahmen',
            'content': 'Robert Habeck und Annalena Baerbock drängen auf schnelleren Kohleausstieg. Die Energiewende stockt.',
            'source': 'SPIEGEL'
        },
        {
            'title': 'AfD-Skandal erschüttert Bundestag',
            'content': 'Alice Weidel muss sich für umstrittene Äußerungen rechtfertigen. Die Opposition fordert Konsequenzen.',
            'source': 'BILD'
        }
    ]
    
    print("🔬 Teste Artikel-Analysen:")
    print("-" * 30)
    
    analyzed_articles = []
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n📰 Test {i}: {article['title'][:40]}...")
        
        # Politik-Analyse
        result = analyzer.analyze_artikel_politik(
            title=article['title'],
            content=article['content']
        )
        
        # Ergebnisse anzeigen
        print(f"   🏛️ Politik-Artikel: {'✅' if result['is_politik'] else '❌'}")
        print(f"   📊 Relevanz-Score: {result['politik_relevanz_score']:.2f}")
        
        if result['parteien_mentions']:
            parteien_str = ', '.join(result['parteien_mentions'].keys())
            print(f"   🏛️ Parteien: {parteien_str}")
        
        if result['dominante_partei']:
            partei_name = result['parteien_mentions'][result['dominante_partei']]['name']
            print(f"   👑 Dominante Partei: {partei_name}")
        
        if result['politik_themen']:
            themen_str = ', '.join(result['politik_themen'].keys())
            print(f"   📋 Themen: {themen_str}")
        
        # Sentiment
        for partei, sentiment in result['partei_sentiments'].items():
            partei_name = analyzer.parteien[partei]['name']
            print(f"   😊 {partei_name}: {sentiment['sentiment']} ({sentiment['score']})")
        
        # Für DataFrame vorbereiten
        article_data = {
            **article,
            **result
        }
        analyzed_articles.append(article_data)
    
    # Summary testen
    print(f"\n📊 POLITIK SUMMARY:")
    df = pd.DataFrame(analyzed_articles)
    summary = analyzer.generate_politik_summary(df)
    
    print(f"   📰 Politik-Artikel: {summary['total_politik_articles']}")
    print(f"   📈 Politik-Anteil: {summary['politik_anteil']}%")
    print(f"   🏛️ Top Parteien: {summary['top_parteien']}")
    print(f"   📋 Top Themen: {summary['top_themen']}")
    
    print(f"\n🎉 POLITIK BIAS TRACKER GETESTET!")
    print(f"🚀 Bereit für Dashboard-Integration!")
    
    return True

if __name__ == "__main__":
    test_politik_analyzer()