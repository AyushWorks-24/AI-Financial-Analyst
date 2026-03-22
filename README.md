---
title: AI Financial Analyst
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.40.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# AI Financial Analyst

> An AI-powered real-time stock analysis web application built with Streamlit, LangGraph, Facebook Prophet, and SQLite.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🧠 What is this?

AI Financial Analyst is a full-stack web application that gives retail investors access to professional-grade stock analysis tools — for free. It combines live market data, machine learning forecasting, interactive backtesting, portfolio comparison, and a conversational AI agent all in one dark-themed dashboard.

Built as a **Minor Project** for B.Tech AI & ML at Oriental College of Technology, Bhopal.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Live Analysis** | Candlestick charts, SMA-based trend detection, volatility & Sharpe Ratio |
| 📈 **Fundamentals** | P/E ratio, EPS, market cap, 52W high/low, sector info |
| 📰 **News Feed** | Real-time financial news fetched from Google News |
| 🤖 **AI Forecast** | 14-day price prediction using Facebook Prophet |
| 📉 **Backtest** | Interactive MA crossover strategy vs Buy & Hold with live sliders |
| 💬 **AI Chat Agent** | Conversational agent powered by llama3.2 via Ollama + LangGraph |
| 📂 **Portfolio Tracker** | Compare up to 8 stocks with normalized charts & correlation heatmap |
| 📤 **Export Center** | Download full PDF reports with charts + CSV data exports |
| 🔍 **Smart Search** | Search by company name ("Apple") or ticker ("AAPL") across 4800+ stocks |
| 🗄️ **SQLite Cache** | Local caching layer — faster repeat loads + offline fallback |

---

## 🖥️ Screenshots

> *(Add screenshots here after deployment)*

---

## 🏗️ Project Structure

```
AI-Financial-Analyst/
│
├── app.py                  # Main Streamlit application
│
├── core/
│   ├── __init__.py
│   ├── engine.py           # Data fetching, analysis, caching, PDF generation
│   └── agent.py            # LangGraph ReAct agent + Ollama integration
│
├── data/
│   └── tickers.csv         # 4800+ stock tickers with company names
│
├── ui/
│   └── style.css           # Custom dark theme CSS
│
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | Python 3.10+ |
| **AI Agent** | LangGraph + Ollama (llama3.2) |
| **Forecasting** | Linear Regression |
| **Database** | SQLite (via Python's built-in `sqlite3`) |
| **Data Source** | yfinance (Yahoo Finance API) |
| **Charts** | Plotly |
| **PDF Export** | fpdf2 + kaleido |
| **Search** | RapidFuzz (fuzzy matching) |
| **News** | GNews |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- llama3.2 model pulled

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Financial-Analyst.git
cd AI-Financial-Analyst
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the AI model

```bash
ollama pull llama3.2
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit
yfinance
plotly
pandas
numpy
prophet
langchain-ollama
langchain-core
langgraph
gnews
rapidfuzz
fpdf2
kaleido
```

---

## 🗄️ How Data Works

This app uses a **two-layer data architecture:**

```
User Request
     ↓
Check SQLite Cache
     ↓
Fresh? → Return cached data instantly
Expired? → Fetch from Yahoo Finance API → Save to cache → Return
API Down? → Return stale cache with warning banner
```

- **Price data** cached for 6 hours
- **Company info** cached for 24 hours
- **News** always fetched live (real-time only)
- Cache stored locally as `cache.db` (auto-created on first run)

---

## 🤖 AI Agent Architecture

The AI Chat tab uses a **ReAct (Reasoning + Acting)** agent pattern:

```
User Question
     ↓
LangGraph ReAct Agent (llama3.2)
     ↓
Decides which tool to call
     ↓
┌─────────────────────────────────┐
│  get_stock_price_chart          │
│  get_fundamental_data           │
│  get_price_forecast             │
│  get_company_info               │
│  calculate_basic_risk_metrics   │
└─────────────────────────────────┘
     ↓
Synthesizes tool results into a response
     ↓
Answer displayed in chat
```

---

## ⚠️ Disclaimer

This application is built for **educational purposes only**. The AI forecasts and analysis provided are generated by statistical models and do **not** constitute financial advice. Always consult a certified financial advisor before making investment decisions.

---

## 🔭 Future Scope

- ☁️ Cloud deployment (Railway / Hugging Face Spaces)
- ⚡ Switch from Ollama to Groq API for cloud-hosted AI
- 🔐 User authentication and personal watchlists
- 📊 Sentiment analysis from news headlines
- 🌐 Expand to crypto and forex markets
- 📱 Mobile-responsive UI improvements

---

## 👨‍💻 Author

**Ayush** — B.Tech AI & ML, Oriental College of Technology, Bhopal

[![GitHub](https://img.shields.io/badge/GitHub-AyushWorks--24-black?style=flat-square&logo=github)](https://github.com/AyushWorks-24)

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.