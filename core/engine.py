import yfinance as yf
from gnews import GNews
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LinearRegression
import numpy as np
from fpdf import FPDF
import io
import tempfile
import os
import sqlite3
import json
import requests

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# ─────────────────────────────────────────
# SHARED SESSION (browser-like headers to avoid yfinance rate limiting on cloud IPs)
# ─────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
})

PERIOD_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180,
    "1y": 365, "2y": 730, "5y": 1825
}

# ─────────────────────────────────────────
# SQLITE CACHE LAYER
# ─────────────────────────────────────────
DB_PATH = "/tmp/cache.db"

CACHE_TTL = {
    "price":   60 * 60 * 6,   # 6 hours
    "info":    60 * 60 * 24,  # 24 hours
    "metrics": 60 * 60 * 6,   # 6 hours
}


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key  TEXT PRIMARY KEY,
            data_json  TEXT NOT NULL,
            cached_at  REAL NOT NULL,
            data_type  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _make_key(data_type: str, *args) -> str:
    parts = [data_type] + [str(a) for a in args]
    return "_".join(parts)


def _cache_get(cache_key: str, data_type: str):
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT data_json, cached_at FROM cache WHERE cache_key = ?",
            (cache_key,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        data_json, cached_at = row
        age_seconds = datetime.now().timestamp() - cached_at
        ttl = CACHE_TTL.get(data_type, 60 * 60 * 6)
        if age_seconds > ttl:
            return None
        return json.loads(data_json)
    except Exception:
        return None


def _cache_set(cache_key: str, data_type: str, data):
    try:
        conn = _get_db()
        conn.execute(
            """INSERT OR REPLACE INTO cache
               (cache_key, data_json, cached_at, data_type)
               VALUES (?, ?, ?, ?)""",
            (cache_key, json.dumps(data), datetime.now().timestamp(), data_type)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _cache_get_stale(cache_key: str):
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT data_json FROM cache WHERE cache_key = ?",
            (cache_key,)
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _cache_stats() -> dict:
    try:
        conn = _get_db()
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        by_type = conn.execute(
            "SELECT data_type, COUNT(*) FROM cache GROUP BY data_type"
        ).fetchall()
        oldest = conn.execute(
            "SELECT MIN(cached_at) FROM cache"
        ).fetchone()[0]
        conn.close()
        return {
            "total": total,
            "by_type": dict(by_type),
            "oldest": datetime.fromtimestamp(oldest).strftime("%d %b %H:%M") if oldest else "N/A"
        }
    except Exception:
        return {"total": 0, "by_type": {}, "oldest": "N/A"}


def clear_cache():
    try:
        conn = _get_db()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────
# YFINANCE FETCH HELPERS
# ─────────────────────────────────────────

def _yf_fetch_price(ticker_symbol: str, period=None, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Fetches OHLCV from yfinance using a shared session with browser-like headers
    to avoid rate limiting on cloud IPs (HuggingFace Spaces, Render, etc.)
    """
    ticker = yf.Ticker(ticker_symbol, session=_SESSION)

    if start_date and end_date:
        df = ticker.history(start=str(start_date), end=str(end_date), auto_adjust=True)
    else:
        df = ticker.history(period=period or "1y", auto_adjust=True)

    if df.empty:
        raise Exception(f"yfinance returned empty data for {ticker_symbol}")

    # Normalize column names
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep only OHLCV columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep]


def _yf_fetch_info(ticker_symbol: str) -> dict:
    """
    Fetches company overview from yfinance.
    Returns dict with standard keys used throughout engine.py.
    """
    ticker = yf.Ticker(ticker_symbol, session=_SESSION)

    # .fast_info is lighter and less likely to get rate limited
    try:
        fast = ticker.fast_info
        info = ticker.info  # full info for fundamentals
    except Exception:
        fast = {}
        info = {}

    def safe_get(d, key, fallback=None):
        val = d.get(key, fallback) if isinstance(d, dict) else getattr(d, key, fallback)
        return val if val not in (None, "None", "N/A", "") else fallback

    return {
        "longName":            safe_get(info, "longName", ticker_symbol),
        "sector":              safe_get(info, "sector", "N/A"),
        "longBusinessSummary": safe_get(info, "longBusinessSummary", ""),
        "marketCap":           safe_get(info, "marketCap"),
        "forwardPE":           safe_get(info, "forwardPE"),
        "trailingEps":         safe_get(info, "trailingEps"),
        "fiftyTwoWeekHigh":    safe_get(info, "fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow":     safe_get(info, "fiftyTwoWeekLow"),
        "dividendYield":       safe_get(info, "dividendYield"),
        "regularMarketPrice":  safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice"),
    }


# ─────────────────────────────────────────
# DATA FETCHING (yfinance + SQLite cache)
# ─────────────────────────────────────────

def fetch_price_data(ticker_symbol, period=None, start_date=None, end_date=None):
    """
    Fetches OHLCV price data with SQLite caching.
    Primary source: yfinance (with session headers).
    Falls back to stale cache if API fails.

    Returns (DataFrame, is_stale):
    - is_stale=False → fresh data
    - is_stale=True  → fallback stale cache
    - empty DataFrame → complete failure
    """
    if start_date and end_date:
        cache_key = _make_key("price", ticker_symbol, str(start_date), str(end_date))
    else:
        cache_key = _make_key("price", ticker_symbol, period or "1y")

    # 1. Try fresh cache
    cached = _cache_get(cache_key, "price")
    if cached is not None:
        df = pd.DataFrame(cached["data"])
        df.index = pd.to_datetime(cached["index"])
        return df, False

    # 2. Cache miss — fetch from yfinance
    try:
        df = _yf_fetch_price(ticker_symbol, period=period, start_date=start_date, end_date=end_date)

        if not df.empty:
            _cache_set(cache_key, "price", {
                "data": df.to_dict(),
                "index": [str(i) for i in df.index]
            })
            return df, False

    except Exception as e:
        logging.warning(f"yfinance fetch failed for {ticker_symbol}: {e}")

    # 3. Stale cache fallback
    stale = _cache_get_stale(cache_key)
    if stale:
        df = pd.DataFrame(stale["data"])
        df.index = pd.to_datetime(stale["index"])
        return df, True

    return pd.DataFrame(), False


def get_company_info(ticker_symbol: str) -> dict:
    """
    Fetches company info with SQLite caching.
    Primary source: yfinance.
    Falls back to stale cache if API fails.
    """
    cache_key = _make_key("info", ticker_symbol)

    # 1. Fresh cache
    cached = _cache_get(cache_key, "info")
    if cached is not None:
        return cached

    # 2. Live fetch from yfinance
    try:
        info = _yf_fetch_info(ticker_symbol)
        _cache_set(cache_key, "info", info)
        return info
    except Exception as e:
        logging.warning(f"yfinance info fetch failed for {ticker_symbol}: {e}")

    # 3. Stale cache fallback
    stale = _cache_get_stale(cache_key)
    if stale:
        return stale

    return {"error": f"Could not fetch data for {ticker_symbol}."}


def fetch_forecast_data(ticker_symbol):
    """Forecast uses 2 years of data — cached separately."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=730)
    df, _ = fetch_price_data(
        ticker_symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    return df


# ─────────────────────────────────────────
# FEATURE FUNCTIONS (unchanged from your original)
# ─────────────────────────────────────────

def get_news(ticker_symbol: str) -> str:
    try:
        info = get_company_info(ticker_symbol)
        company_name = info.get("longName", ticker_symbol)
        google_news = GNews(language="en", max_results=5)
        news_results = google_news.get_news(f'"{company_name}" stock finance')
        if not news_results:
            return "No recent news found."
        formatted_news = []
        for article in news_results:
            title = article.get("title", "No Title")
            date = article.get("published date", "N/A")
            url = article.get("url", "#")
            try:
                dt_obj = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
                date = dt_obj.strftime("%Y-%m-%d")
            except Exception:
                pass
            formatted_news.append(f"| {date} | [{title}]({url}) |")
        markdown_table = "| Date | Title |\n|---|---|\n" + "\n".join(formatted_news)
        return markdown_table
    except Exception as e:
        return f"News fetch failed: {e}"


def get_fundamental_data(ticker_symbol: str) -> str:
    try:
        info = get_company_info(ticker_symbol)

        def fmt(key, prefix="", suffix=""):
            val = info.get(key)
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)):
                return f"{prefix}{val:,.2f}{suffix}"
            return val

        data = [
            f"| Market Cap | {fmt('marketCap', '$')} |",
            f"| P/E Ratio | {fmt('forwardPE')} |",
            f"| EPS (TTM) | {fmt('trailingEps')} |",
            f"| 52W High | {fmt('fiftyTwoWeekHigh', '$')} |",
            f"| 52W Low | {fmt('fiftyTwoWeekLow', '$')} |",
            f"| Div Yield | {fmt('dividendYield', '', '%')} |",
            f"| Sector | {info.get('sector', 'N/A')} |",
        ]
        return "| Metric | Value |\n|---|---|\n" + "\n".join(data)
    except Exception as e:
        return f"Could not load fundamentals: {e}"


def get_stock_price_chart(ticker_symbol: str, period="1y", start_date=None, end_date=None):
    hist_data, is_stale = fetch_price_data(ticker_symbol, period, start_date, end_date)
    if hist_data.empty:
        return None, False
    if isinstance(hist_data.columns, pd.MultiIndex):
        hist_data.columns = hist_data.columns.get_level_values(0)
    required_cols = ["Open", "High", "Low", "Close"]
    for col in required_cols:
        if col not in hist_data.columns:
            return None, False
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data["Open"],
        high=hist_data["High"],
        low=hist_data["Low"],
        close=hist_data["Close"],
        name="Price",
    ))
    fig.update_layout(
        title=f"{ticker_symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=500,
    )
    return fig, is_stale


def get_price_forecast(ticker_symbol: str):
    """
    14-day price forecast using Linear Regression.
    Features: day index + 7-day and 21-day moving averages.
    """
    try:
        df = fetch_forecast_data(ticker_symbol)
        if df is None or df.empty or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].dropna().values.flatten()
        dates = df.index

        n = len(close)
        X, y = [], []

        for i in range(21, n):
            ma7  = close[i-7:i].mean()
            ma21 = close[i-21:i].mean()
            X.append([i, ma7, ma21])
            y.append(close[i])

        X = np.array(X)
        y = np.array(y)

        model = LinearRegression()
        model.fit(X, y)

        y_pred_train = model.predict(X)
        residual_std = np.std(y - y_pred_train)

        forecast_values = []
        for i in range(14):
            idx = n + i
            ma7  = np.append(close[-7+i:], forecast_values)[-7:].mean() if i < 7 else np.array(forecast_values[-7:]).mean()
            ma21 = np.append(close[-21+i:], forecast_values)[-21:].mean() if i < 21 else np.array(forecast_values[-21:]).mean()
            pred = model.predict([[idx, ma7, ma21]])[0]
            forecast_values.append(pred)

        last_date = dates[-1]
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        forecast_dates = pd.date_range(start=last_date, periods=15, freq="B")[1:]

        forecast_arr = np.array(forecast_values)
        upper = forecast_arr + 1.96 * residual_std
        lower = forecast_arr - 1.96 * residual_std

        fig = go.Figure()

        hist_dates = [d.tz_localize(None) if hasattr(d, 'tz') and d.tz else d for d in dates[-90:]]
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=close[-90:],
            mode="lines",
            name="Historical",
            line=dict(color="#00c6ff", width=2)
        ))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_arr,
            mode="lines",
            name="Forecast",
            line=dict(color="#7f5af0", width=2, dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(127, 90, 240, 0.15)",
            line=dict(width=0),
            name="95% Confidence"
        ))

        fig.update_layout(
            title=f"14-Day Price Forecast — {ticker_symbol}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        return fig

    except Exception as e:
        logging.error(f"Forecast error for {ticker_symbol}: {e}")
        return None


def analyze_stock(ticker_symbol: str, period="1y"):
    try:
        df, _ = fetch_price_data(ticker_symbol, period=period)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns or len(df) < 50:
            return None

        close = df["Close"].dropna()
        returns = close.pct_change().dropna()
        latest_price = close.iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        volatility = returns.std() * np.sqrt(252)

        if latest_price > sma_20 and sma_20 > sma_50:
            trend = "Uptrend"
        elif latest_price < sma_20 and sma_20 < sma_50:
            trend = "Downtrend"
        else:
            trend = "Sideways"

        risk_level = "High" if volatility > 0.4 else "Medium" if volatility > 0.25 else "Low"

        return f"""
### 📊 Automated Analysis Summary

- **Current Price:** ${latest_price:.2f}
- **Trend:** {trend}
- **Annualized Volatility:** {volatility:.2%}
- **Risk Level:** {risk_level}

The stock currently reflects a {trend.lower()} structure with {risk_level.lower()} volatility.
"""
    except Exception as e:
        logging.error(f"Analyze error for {ticker_symbol}: {e}")
        return None


def calculate_basic_risk_metrics(ticker, period="1y"):
    cache_key = _make_key("metrics", ticker, period)
    cached = _cache_get(cache_key, "metrics")
    if cached is not None:
        return cached

    df, _ = fetch_price_data(ticker, period=period)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    volatility = df["Returns"].std() * np.sqrt(252)
    sharpe = (df["Returns"].mean() * 252) / volatility
    cumulative = (1 + df["Returns"]).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    result = {
        "Volatility":    float(volatility.iloc[0])   if hasattr(volatility, 'iloc')   else float(volatility),
        "Sharpe Ratio":  float(sharpe.iloc[0])        if hasattr(sharpe, 'iloc')        else float(sharpe),
        "Max Drawdown":  float(max_drawdown.iloc[0])  if hasattr(max_drawdown, 'iloc')  else float(max_drawdown),
    }
    _cache_set(cache_key, "metrics", result)
    return result


def ma_crossover_backtest(ticker_symbol: str, short_window=20, long_window=50, period="1y"):
    df, _ = fetch_price_data(ticker_symbol, period=period)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None

    df = df.copy()
    df["Short_MA"] = df["Close"].rolling(short_window).mean()
    df["Long_MA"]  = df["Close"].rolling(long_window).mean()
    df.dropna(inplace=True)

    df["Signal"]          = 0
    df.loc[df["Short_MA"] > df["Long_MA"], "Signal"] = 1
    df["Position"]        = df["Signal"].shift(1)
    df["Market_Return"]   = df["Close"].pct_change()
    df["Strategy_Return"] = df["Market_Return"] * df["Position"]
    df.dropna(inplace=True)

    df["Cumulative_Market"]   = (1 + df["Market_Return"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()
    df["Drawdown"]            = df["Cumulative_Strategy"] / df["Cumulative_Strategy"].cummax() - 1
    return df


# ─────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────

def _fig_to_png_bytes(fig: go.Figure) -> bytes:
    fig_copy = fig.to_dict()
    export_fig = go.Figure(fig_copy)
    export_fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        width=800,
        height=400,
    )
    return export_fig.to_image(format="png", scale=2)


def generate_pdf(ticker_symbol: str, period: str = "1y") -> bytes:
    info          = get_company_info(ticker_symbol)
    company_name  = info.get("longName", ticker_symbol)
    sector        = info.get("sector", "N/A")
    current_price = info.get("regularMarketPrice") or info.get("currentPrice", "N/A")
    metrics       = calculate_basic_risk_metrics(ticker_symbol, period=period)
    analysis_text = analyze_stock(ticker_symbol, period=period)

    if analysis_text:
        clean_text = (
            analysis_text
            .replace("###", "").replace("**", "")
            .replace("- ", "  • ").replace("#", "")
            .strip()
        )
    else:
        clean_text = "Analysis not available."

    chart_fig, _ = get_stock_price_chart(ticker_symbol, period=period)
    chart_png = None
    if chart_fig:
        try:
            chart_png = _fig_to_png_bytes(chart_fig)
        except Exception as e:
            logging.warning(f"Chart image error: {e}")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_fill_color(15, 17, 23)
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 198, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "AI Financial Analyst — Stock Report", ln=True)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(180, 180, 180)
    pdf.set_xy(140, 10)
    pdf.cell(0, 6, f"Generated: {datetime.today().strftime('%d %b %Y, %H:%M')}")

    pdf.set_xy(10, 35)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 10, f"{ticker_symbol}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"{company_name}  |  Sector: {sector}", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 100, 180)
    price_str = f"${current_price:.2f}" if isinstance(current_price, (int, float)) else str(current_price)
    pdf.cell(0, 8, f"Current Price: {price_str}  |  Period: {period}", ln=True)

    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    if metrics:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, "Risk Metrics", ln=True)
        pdf.ln(2)
        col_w = 58
        pdf.set_fill_color(15, 17, 23)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 8, "  Metric",       border=0, fill=True)
        pdf.cell(col_w, 8, "  Volatility",   border=0, fill=True)
        pdf.cell(col_w, 8, "  Sharpe Ratio", border=0, fill=True)
        pdf.ln()
        pdf.set_fill_color(245, 247, 250)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(col_w, 8, "  Value",                             border=0, fill=True)
        pdf.cell(col_w, 8, f"  {metrics['Volatility']:.2%}",     border=0, fill=True)
        pdf.cell(col_w, 8, f"  {metrics['Sharpe Ratio']:.2f}",   border=0, fill=True)
        pdf.ln()
        pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_w, 8, "  Max Drawdown",                      border=0, fill=True)
        pdf.cell(col_w, 8, f"  {metrics['Max Drawdown']:.2%}",   border=0, fill=True)
        pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, "Automated Analysis Summary", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, clean_text)
    pdf.ln(6)

    if chart_png:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, f"Price Chart — {period}", ln=True)
        pdf.ln(2)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(chart_png)
            tmp_path = tmp.name
        try:
            pdf.image(tmp_path, x=10, w=190)
        finally:
            os.unlink(tmp_path)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This report is generated by an AI-powered statistical model for "
        "educational purposes only. It is NOT financial advice. Past performance does "
        "not guarantee future results. Always consult a certified financial advisor."
    )

    return pdf.output()