# 📰 Media Pulse Dashboard

Ein umfassendes Dashboard zur Analyse deutscher Medienlandschaft mit **20+ Nachrichtenquellen**, regionalen Filtern und erweiterten KI-Analysen.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

### 📊 **Medienanalyse**
- **20+ deutsche Nachrichtenquellen** (ARD, ZDF, Spiegel, Zeit, FAZ, SZ, etc.)
- **Regionale Filter** (Bayern, NRW, Berlin, Hamburg, Schleswig-Holstein, etc.)
- **Medientyp-Kategorisierung** (Quality, Business, Tech, Public, Regional, Sport)
- **Echtzeit-Datensammlung** mit automatischer Aktualisierung

### 🧠 **KI-Powered Analysis**
- **Sentiment-Analyse** mit deutscher NLP-Pipeline
- **Bias-Detection** (Links, Rechts, Zentrum, Populistisch)
- **Politik Bias Tracker** mit Parteien-Sentiment
- **Hamburg-Fokus** für lokale Berichterstattung
- **Keyword-Extraktion** und Themen-Clustering

### 📈 **Interactive Dashboards**
- **Live-Metriken** mit farbcodierten Cards
- **Plotly-Charts** für Sentiment-Verteilung
- **Bias-Heatmaps** nach Medienquellen
- **Politik-Tracker** mit Parteien-Erwähnungen
- **Regionale Analyse** mit geografischen Insights

## 🛠️ Installation

### 1. Repository klonen
```bash
git clone https://github.com/timmiflimmi/media_dashboard.git
cd media_dashboard
```

### 2. Python Environment
```bash
# Virtual Environment erstellen
python -m venv venv

# Aktivieren (Windows)
venv\Scripts\activate

# Aktivieren (macOS/Linux)
source venv/bin/activate
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. API Keys konfigurieren
Erstelle eine `.env` Datei im Root-Verzeichnis:
```env
# News API Keys
NEWS_API_KEY=your_newsapi_key_here
GUARDIAN_API_KEY=your_guardian_key_here

# Optional: NLP Service Keys
OPENAI_API_KEY=your_openai_key_here
```

## 🏃‍♂️ Quickstart

```bash
# Dashboard starten
streamlit run main.py
```

Das Dashboard öffnet sich automatisch im Browser unter `http://localhost:8501`

## 📁 Projektstruktur

```
media_pulse_dashboard/
├── main.py                 # 🚀 Hauptdashboard
├── src/
│   ├── data/
│   │   └── news_collector.py    # 📰 News Collection Engine
│   └── analysis/
│       ├── sentiment_analyzer.py   # 😊 NLP Sentiment Analysis
│       └── politik_analyzer.py     # 🏛️ Politik Bias Tracker
├── requirements.txt        # 📦 Python Dependencies
├── .env                   # 🔑 API Keys (nicht in Git!)
└── README.md              # 📖 Diese Datei
```

## 🎯 Usage

### Dashboard Navigation

#### **📊 Overview**
- **Gesamtmetriken** mit Artikel-Anzahl, Quellen, Politik-Anteil
- **Regionale Verteilung** und Medientyp-Statistiken
- **Sentiment-Durchschnitt** mit Emoji-Indikatoren

#### **😊 Sentiment Tab**
- Pie-Chart der Sentiment-Verteilung (Positiv/Negativ/Neutral)
- Bar-Chart des Sentiments nach Medienquelle
- Detaillierte Sentiment-Scores

#### **🎯 Bias Tab**
- Bias-Verteilung (Links/Rechts/Zentrum/Populistisch)
- Bias vs. Sentiment Heatmap
- Medienquellen-Klassifizierung

#### **🏛️ Politik Tab**
- Parteien-Erwähnungen (CDU, SPD, Grüne, FDP, AfD, Linke)
- Sentiment-Analyse pro Partei
- Politik-Relevanz-Scoring

#### **🗺️ Regional Tab**
- Artikel-Verteilung nach Bundesländern
- Regionales Sentiment-Mapping
- Local vs. National News

#### **🏙️ Hamburg Tab**
- Hamburg-spezifische Artikel
- Lokale Sentiment-Trends
- Keyword-Analyse für Hamburg

#### **🔥 Headlines Tab**
- Neueste Schlagzeilen mit Sentiment-Badges
- Bias-Indikatoren pro Artikel
- Zeitstempel und Quellen-Attribution

### Filter-Optionen

```python
# Regionale Filter
regions = ['alle', 'national', 'bayern', 'nrw', 'berlin', 'hamburg', 'sachsen', 'schleswig-holstein']

# Medientyp Filter
media_types = ['alle', 'quality', 'business', 'tech', 'public', 'regional', 'sport', 'alternative']

# Zeitraum
days_back = 1-7 Tage
```

## 🔧 Konfiguration

### Medienquellen erweitern
```python
# In src/data/news_collector.py
GERMAN_SOURCES = {
    'neue_quelle': {
        'name': 'Neue Medienquelle',
        'url': 'https://api.neue-quelle.de',
        'region': 'bayern',
        'type': 'regional'
    }
}
```

### NLP-Pipeline anpassen
```python
# In src/analysis/sentiment_analyzer.py
class GermanNLPAnalyzer:
    def __init__(self):
        # Custom German NLP models
        self.sentiment_model = "deutsche-telekom/gbert-large"
        self.bias_keywords = {...}
```

## 📊 Supported Media Sources

### **Quality Media**
- Der Spiegel, Die Zeit, FAZ, Süddeutsche, taz
- Handelsblatt, WirtschaftsWoche, Capital

### **Public Broadcasting**
- ARD Tagesschau, ZDF heute, Deutschlandfunk
- BR24, NDR, WDR, HR, RBB

### **Regional Media**
- Hamburger Abendblatt, Berliner Zeitung
- Bayern-spezifische Quellen
- NRW Regional News

### **Alternative Sources**
- Netzpolitik.org, Heise Online
- Golem.de, t3n Magazine

## 🚀 Advanced Features

### Auto-Refresh
```python
# 30-Minuten Cache mit automatischer Aktualisierung
@st.cache_data(ttl=1800)
def collect_and_analyze_news_extended():
    # News collection pipeline
```

### Real-time Bias Detection
```python
# KI-gestützte Bias-Erkennung
bias_analysis = {
    'bias_type': 'left_leaning',
    'confidence': 0.87,
    'indicators': ['social justice', 'climate action']
}
```

### Politik Sentiment Tracking
```python
# Parteien-spezifische Sentiment-Analyse
partei_sentiments = {
    'cdu': {'score': 0.15, 'trend': 'improving'},
    'spd': {'score': -0.05, 'trend': 'stable'},
    'gruene': {'score': 0.25, 'trend': 'positive'}
}
```

## 🔑 API Requirements

### News API
```bash
# Kostenlose Registrierung bei newsapi.org
# 500 Requests/Tag im Free Tier
NEWS_API_KEY=your_key_here
```

### Guardian API (Optional)
```bash
# Internationale Perspektive
GUARDIAN_API_KEY=your_key_here
```

## 📈 Performance

- **~50-100 Artikel** pro Sammlungsvorgang
- **~2-3 Sekunden** Analysezeit pro Artikel
- **30-Minuten Cache** für optimale Performance
- **Responsive Design** für Mobile & Desktop

## 🛠️ Development

### Local Development
```bash
# Development Mode mit Hot Reload
streamlit run main.py --server.runOnSave=true

# Debug Mode
streamlit run main.py --logger.level=debug
```

### Testing
```bash
# Unit Tests (falls vorhanden)
python -m pytest tests/

# Manual Testing der Module
python -c "from src.data.news_collector import GermanNewsCollector; print('✅ Import erfolgreich')"
```

## 🤝 Contributing

1. **Fork** das Repository
2. **Feature Branch** erstellen (`git checkout -b feature/amazing-feature`)
3. **Commit** deine Änderungen (`git commit -m 'Add amazing feature'`)
4. **Push** zum Branch (`git push origin feature/amazing-feature`)
5. **Pull Request** erstellen

## 📝 Roadmap

- [ ] **🔮 AI-Predictions** für Themen-Trends
- [ ] **📱 Mobile App** mit React Native
- [ ] **🔔 Alert-System** für Breaking News
- [ ] **📊 Export-Funktionen** (PDF, Excel)
- [ ] **🌍 Internationale Quellen** Integration
- [ ] **🤖 ChatBot** für News-Queries
- [ ] **📈 Historical Analysis** mit Trend-Prediction

## 📄 License

Dieses Projekt ist unter der **MIT License** lizenziert

## 🙏 Acknowledgments

- **Streamlit** für das fantastische Framework
- **Plotly** für interaktive Visualisierungen
- **spaCy** für deutsche NLP-Pipeline
- **NewsAPI** für Nachrichtenzugang
- **Deutsche Medienlandschaft** für qualitativ hochwertigen Journalismus

## 📞 Support

Bei Fragen oder Problemen:

- **Issues** auf GitHub erstellen
- **Pull Requests** für Verbesserungen
- **Discussions** für Feature-Requests

---

**🎯 Made with ❤️ for German Media Analysis**

*Entwickelt zur transparenten Analyse der deutschen Medienlandschaft mit modernsten KI-Technologien.*