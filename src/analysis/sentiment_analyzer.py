# src/analysis/sentiment_analyzer.py
"""
🇩🇪 Deutsche NLP-Analyse für Media Pulse Dashboard
Sentiment Analysis, Bias Detection, Keyword Extraction
"""

import re
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GermanNLPAnalyzer:
    """
    🧠 Deutsche NLP-Analyse Engine
    """
    
    def __init__(self):
        """Initialize German NLP Tools"""
        
        # VADER Sentiment Analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Deutsche Sentiment-Wörter
        self.positive_words = {
            'stark': ['stark', 'stärker', 'stärkste', 'kraftvoll', 'mächtig'],
            'gut': ['gut', 'beste', 'besser', 'hervorragend', 'exzellent', 'toll', 'super'],
            'erfolg': ['erfolg', 'erfolgreich', 'gewinn', 'sieg', 'triumph', 'durchbruch'],
            'positiv': ['positiv', 'optimistisch', 'hoffnungsvoll', 'ermutigend', 'aufbauend'],
            'fortschritt': ['fortschritt', 'verbesserung', 'entwicklung', 'wachstum', 'innovation'],
            'stabilität': ['stabil', 'sicher', 'verlässlich', 'beständig', 'solide']
        }
        
        self.negative_words = {
            'schlecht': ['schlecht', 'schlimm', 'furchtbar', 'katastrophal', 'verheerend'],
            'krise': ['krise', 'problem', 'schwierigkeit', 'notstand', 'chaos'],
            'konflikt': ['konflikt', 'streit', 'kampf', 'krieg', 'auseinandersetzung'],
            'negativ': ['negativ', 'pessimistisch', 'hoffnungslos', 'bedrohlich', 'gefährlich'],
            'verlust': ['verlust', 'niederlage', 'rückgang', 'einbruch', 'kollaps'],
            'unsicher': ['unsicher', 'instabil', 'ungewiss', 'riskant', 'bedroht']
        }
        
        # Bias-Indikatoren für deutsche Medien
        self.bias_indicators = {
            'left_leaning': [
                'soziale gerechtigkeit', 'umverteilung', 'klimaschutz', 'diversität',
                'progressive', 'nachhaltig', 'solidarität', 'teilhabe'
            ],
            'right_leaning': [
                'tradition', 'sicherheit', 'ordnung', 'leistung', 'eigenverantwortung',
                'wirtschaftswachstum', 'stabilität', 'bewährte werte'
            ],
            'populist': [
                'das volk', 'die elite', 'establishment', 'mainstream', 'bürgernah',
                'volksnah', 'gegen das system', 'echte demokratie'
            ]
        }
        
        # Hamburg-spezifische Keywords
        self.hamburg_keywords = [
            'hamburg', 'hansestadt', 'elbe', 'hafencity', 'st. pauli', 'altona',
            'wandsbek', 'bergedorf', 'harburg', 'eimsbüttel', 'hamburg-mitte',
            'speicherstadt', 'landungsbrücken', 'reeperbahn', 'alster', 'blankenese'
        ]
        
        logger.info("✅ GermanNLPAnalyzer initialisiert!")
    
    def clean_text(self, text: str) -> str:
        """🧹 Text bereinigen"""
        if not text:
            return ""
        
        # HTML Tags entfernen
        text = re.sub(r'<[^>]+>', '', text)
        
        # URLs entfernen
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Mehrfache Leerzeichen
        text = re.sub(r'\s+', ' ', text)
        
        # Trimmen
        text = text.strip()
        
        return text
    
    def analyze_sentiment(self, text: str) -> dict:
        """😊 Sentiment-Analyse für deutschen Text"""
        
        if not text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'classification': 'neutral',
                'confidence': 0.0
            }
        
        # Text bereinigen
        clean_text = self.clean_text(text.lower())
        
        # VADER Sentiment (funktioniert teilweise auch für Deutsch)
        vader_scores = self.vader.polarity_scores(text)
        
        # Custom German Sentiment
        german_score = self._analyze_german_sentiment(clean_text)
        
        # TextBlob (Englisch, aber als Backup)
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
        except:
            textblob_polarity = 0.0
        
        # Ensemble Score (gewichteter Durchschnitt)
        compound_score = (
            vader_scores['compound'] * 0.4 +
            german_score * 0.5 +
            textblob_polarity * 0.1
        )
        
        # Klassifikation
        if compound_score >= 0.15:
            classification = 'positive'
            confidence = min(compound_score, 1.0)
        elif compound_score <= -0.15:
            classification = 'negative'
            confidence = min(abs(compound_score), 1.0)
        else:
            classification = 'neutral'
            confidence = 1.0 - abs(compound_score)
        
        return {
            'compound': round(compound_score, 3),
            'positive': round(max(0, compound_score), 3),
            'negative': round(max(0, -compound_score), 3),
            'neutral': round(1 - abs(compound_score), 3),
            'classification': classification,
            'confidence': round(confidence, 3),
            'vader_compound': round(vader_scores['compound'], 3),
            'german_custom': round(german_score, 3)
        }
    
    def _analyze_german_sentiment(self, text: str) -> float:
        """🇩🇪 Custom deutsche Sentiment-Analyse"""
        
        words = text.split()
        positive_count = 0
        negative_count = 0
        total_sentiment_words = 0
        
        for word in words:
            # Positive Wörter
            for category, word_list in self.positive_words.items():
                if any(pos_word in word for pos_word in word_list):
                    positive_count += 1
                    total_sentiment_words += 1
                    break
            
            # Negative Wörter
            for category, word_list in self.negative_words.items():
                if any(neg_word in word for neg_word in word_list):
                    negative_count += 1
                    total_sentiment_words += 1
                    break
        
        if total_sentiment_words == 0:
            return 0.0
        
        # Score berechnen
        sentiment_score = (positive_count - negative_count) / len(words)
        return max(-1.0, min(1.0, sentiment_score * 5))  # Skalierung
    
    def detect_bias(self, text: str) -> dict:
        """🎯 Bias Detection für deutschen Text"""
        
        if not text:
            return {
                'bias_type': 'unknown',
                'confidence': 0.0,
                'indicators': []
            }
        
        clean_text = self.clean_text(text.lower())
        words = clean_text.split()
        
        bias_scores = {
            'left_leaning': 0,
            'right_leaning': 0,
            'populist': 0
        }
        
        found_indicators = {
            'left_leaning': [],
            'right_leaning': [],
            'populist': []
        }
        
        # Bias-Indikatoren suchen
        for bias_type, indicators in self.bias_indicators.items():
            for indicator in indicators:
                if indicator in clean_text:
                    bias_scores[bias_type] += 1
                    found_indicators[bias_type].append(indicator)
        
        # Dominanten Bias bestimmen
        if all(score == 0 for score in bias_scores.values()):
            return {
                'bias_type': 'center',
                'confidence': 0.5,
                'indicators': [],
                'scores': bias_scores
            }
        
        max_bias = max(bias_scores, key=bias_scores.get)
        max_score = bias_scores[max_bias]
        total_indicators = sum(bias_scores.values())
        
        confidence = max_score / total_indicators if total_indicators > 0 else 0.0
        
        return {
            'bias_type': max_bias,
            'confidence': round(confidence, 3),
            'indicators': found_indicators[max_bias],
            'scores': bias_scores,
            'all_indicators': found_indicators
        }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> list:
        """🔑 Keyword-Extraktion für deutschen Text"""
        
        if not text:
            return []
        
        clean_text = self.clean_text(text.lower())
        
        # Stopwords (vereinfacht)
        german_stopwords = {
            'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
            'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine',
            'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass',
            'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch',
            'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur',
            'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein',
            'wurde', 'sei', 'in', 'kann', 'wenn', 'soll', 'da', 'er', 'ihr'
        }
        
        # Wörter extrahieren
        words = re.findall(r'\b[a-züäöß]{3,}\b', clean_text)
        
        # Stopwords entfernen
        keywords = [word for word in words if word not in german_stopwords]
        
        # Häufigkeit zählen
        keyword_counts = Counter(keywords)
        
        # Top Keywords
        top_keywords = keyword_counts.most_common(top_n)
        
        return [{'word': word, 'count': count} for word, count in top_keywords]
    
    def is_hamburg_related(self, text: str) -> dict:
        """🏙️ Hamburg-Bezug prüfen"""
        
        if not text:
            return {
                'is_hamburg': False,
                'confidence': 0.0,
                'found_keywords': []
            }
        
        clean_text = self.clean_text(text.lower())
        found_keywords = []
        
        for keyword in self.hamburg_keywords:
            if keyword in clean_text:
                found_keywords.append(keyword)
        
        is_hamburg = len(found_keywords) > 0
        confidence = min(len(found_keywords) / 3, 1.0)  # Max bei 3+ Keywords
        
        return {
            'is_hamburg': is_hamburg,
            'confidence': round(confidence, 3),
            'found_keywords': found_keywords
        }
    
    def analyze_article(self, title: str, content: str) -> dict:
        """📰 Vollständige Artikel-Analyse"""
        
        # Titel und Content kombinieren
        full_text = f"{title} {content}" if content else title
        
        return {
            'sentiment': self.analyze_sentiment(full_text),
            'bias': self.detect_bias(full_text),
            'keywords': self.extract_keywords(full_text),
            'hamburg_related': self.is_hamburg_related(full_text),
            'text_stats': {
                'char_count': len(full_text),
                'word_count': len(full_text.split()),
                'title_length': len(title) if title else 0
            }
        }


# =================================
# 🧪 TEST FUNCTION
# =================================

def test_german_nlp():
    """🧪 Teste die deutsche NLP-Analyse"""
    
    print("🧠 TESTE DEUTSCHE NLP-ANALYSE")
    print("=" * 50)
    
    # Initialize Analyzer
    analyzer = GermanNLPAnalyzer()
    
    # Test Texte
    test_texts = [
        {
            'title': 'Deutschland zeigt starke wirtschaftliche Entwicklung',
            'content': 'Die deutsche Wirtschaft verzeichnet hervorragende Erfolge und positive Wachstumszahlen.',
            'expected_sentiment': 'positive'
        },
        {
            'title': 'Krise und Chaos in der Politik',
            'content': 'Schwere Probleme und katastrophale Entwicklungen belasten das Land.',
            'expected_sentiment': 'negative'
        },
        {
            'title': 'Hamburg plant neue Hafenerweiterung in der Speicherstadt',
            'content': 'Die Hansestadt Hamburg investiert in die HafenCity und modernisiert die Elb-Infrastruktur.',
            'expected_hamburg': True
        },
        {
            'title': 'Soziale Gerechtigkeit und nachhaltige Entwicklung',
            'content': 'Progressive Politik für mehr Teilhabe und Klimaschutz in der Gesellschaft.',
            'expected_bias': 'left_leaning'
        }
    ]
    
    print("🔬 Teste Artikel-Analysen:")
    print("-" * 30)
    
    for i, test_case in enumerate(test_texts, 1):
        print(f"\n📰 Test {i}: {test_case['title'][:40]}...")
        
        # Analyse durchführen
        result = analyzer.analyze_article(
            title=test_case['title'],
            content=test_case['content']
        )
        
        # Sentiment
        sentiment = result['sentiment']
        print(f"   😊 Sentiment: {sentiment['classification']} (Score: {sentiment['compound']})")
        
        # Bias
        bias = result['bias']
        print(f"   🎯 Bias: {bias['bias_type']} (Confidence: {bias['confidence']})")
        
        # Hamburg
        hamburg = result['hamburg_related']
        if hamburg['is_hamburg']:
            print(f"   🏙️ Hamburg: ✅ ({', '.join(hamburg['found_keywords'])})")
        
        # Keywords
        top_keywords = result['keywords'][:3]
        if top_keywords:
            keywords_str = ', '.join([kw['word'] for kw in top_keywords])
            print(f"   🔑 Keywords: {keywords_str}")
        
        # Validierung
        if 'expected_sentiment' in test_case:
            expected = test_case['expected_sentiment']
            actual = sentiment['classification']
            status = "✅" if expected == actual else "❌"
            print(f"   📊 Sentiment Check: {status} (Erwartet: {expected}, Ist: {actual})")
        
        if 'expected_hamburg' in test_case:
            expected = test_case['expected_hamburg']
            actual = hamburg['is_hamburg']
            status = "✅" if expected == actual else "❌"
            print(f"   📊 Hamburg Check: {status} (Erwartet: {expected}, Ist: {actual})")
    
    print(f"\n🎉 DEUTSCHE NLP-ANALYSE GETESTET!")
    print(f"🚀 Bereit für Integration ins Dashboard!")
    
    return True

if __name__ == "__main__":
    test_german_nlp()