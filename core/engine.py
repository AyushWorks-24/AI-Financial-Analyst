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
import hashlib

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ─────────────────────────────────────────
# SQLITE CACHE LAYER
# ─────────────────────────────────────────
# The database file lives in the project root.
# SQLite creates it automatically if it doesn't exist.
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cache.db")

# How long cached data stays valid before we re-fetch
# Price data: 6 hours (markets update during trading hours)
# Company info: 24 hours (fundamentals change slowly)
CACHE_TTL = {
    "price":   60 * 60 * 6,   # 6 hours in seconds
    "info":    60 * 60 * 24,  # 24 hours
    "metrics": 60 * 60 * 6,   # 6 hours
}


def _get_db():
    """
    Opens (or creates) the SQLite database and ensures
    the cache table exists.

    Why a single table with JSON?
    Simpler than creating separate tables for each data type.
    We store any Python object as a JSON string and deserialize
    it when we read it back.

    Table schema:
    - cache_key: unique string identifying what was cached
                 (e.g. "price_AAPL_1y")
    - data_json: the actual data serialized as JSON
    - cached_at: Unix timestamp of when it was stored
    - data_type: "price", "info", or "metrics"
    """
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
    """
    Builds a unique cache key from the data type + arguments.
    e.g. _make_key("price", "AAPL", "1y") → "price_AAPL_1y"

    Why not just concatenate strings?
    If args contain special characters or spaces, concatenation
    could produce collisions. This approach is clean and readable.
    """
    parts = [data_type] + [str(a) for a in args]
    return "_".join(parts)


def _cache_get(cache_key: str, data_type: str):
    """
    Tries to read a value from the cache.

    Returns the deserialized data if it exists and is still fresh.
    Returns None if the cache is empty, expired, or corrupted.

    Why check TTL here instead of at write time?
    Because TTL can change between app versions. Checking at read
    time means old cached data naturally expires without needing
    a cleanup job.
    """
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT data_json, cached_at FROM cache WHERE cache_key = ?",
            (cache_key,)
        ).fetchone()
        conn.close()

        if row is None:
            return None  # cache miss — nothing stored yet

        data_json, cached_at = row
        age_seconds = datetime.now().timestamp() - cached_at
        ttl = CACHE_TTL.get(data_type, 60 * 60 * 6)

        if age_seconds > ttl:
            return None  # cache expired — needs refresh

        return json.loads(data_json)

    except Exception:
        return None  # if anything goes wrong, treat as cache miss


def _cache_set(cache_key: str, data_type: str, data):
    """
    Saves data to the cache.

    Uses INSERT OR REPLACE so if the key already exists,
    it gets overwritten with fresh data.

    We silently ignore errors here — if caching fails,
    the app should still work, just without caching.
    """
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
        pass  # caching is best-effort — never crash the app for it


def _cache_stats() -> dict:
    """
    Returns stats about the current cache.
    Used in the Streamlit sidebar to show users the cache status.
    """
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
    """Wipes all cached data. Called from the sidebar clear button."""
    try:
        conn = _get_db()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────
# DATA FETCHING (with cache)
# ─────────────────────────────────────────

def fetch_price_data(ticker_symbol, period=None, start_date=None, end_date=None):
    """
    Fetches OHLCV price data with SQLite caching.

    Cache key includes ticker + period (or date range) so
    AAPL/1y and AAPL/6mo are stored separately.

    If yfinance fails AND we have stale cache, we return
    the stale data with a warning flag so the UI can
    tell the user the data might be old.
    """
    # Build a cache key based on all the parameters
    if start_date and end_date:
        cache_key = _make_key("price", ticker_symbol, str(start_date), str(end_date))
    else:
        cache_key = _make_key("price", ticker_symbol, period or "1y")

    # Try cache first
    cached = _cache_get(cache_key, "price")
    if cached is not None:
        # Reconstruct DataFrame from the JSON we stored
        df = pd.DataFrame(cached["data"])
        df.index = pd.to_datetime(cached["index"])
        return df, False  # False = data is NOT stale

    # Cache miss — fetch from yfinance
    try:
        if start_date and end_date:
            data = yf.download(
                ticker_symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )
        else:
            data = yf.download(
                ticker_symbol,
                period=period,
                progress=False,
                auto_adjust=False,
            )

        if not data.empty:
            # Save to cache as JSON-serializable dict
            # We store index separately because DataFrames have
            # a DatetimeIndex which JSON can't serialize directly
            _cache_set(cache_key, "price", {
                "data": data.to_dict(),
                "index": [str(i) for i in data.index]
            })

        return data, False

    except Exception:
        # yfinance failed — try to return stale cache as fallback
        # We bypass TTL check here by reading directly from DB
        try:
            conn = _get_db()
            row = conn.execute(
                "SELECT data_json FROM cache WHERE cache_key = ?",
                (cache_key,)
            ).fetchone()
            conn.close()

            if row:
                cached_stale = json.loads(row[0])
                df = pd.DataFrame(cached_stale["data"])
                df.index = pd.to_datetime(cached_stale["index"])
                return df, True  # True = data IS stale (from fallback)
        except Exception:
            pass

        return pd.DataFrame(), False  # complete failure


def get_company_info(ticker_symbol: str) -> dict:
    """
    Fetches company info with SQLite caching.
    Company info changes slowly so we cache it for 24 hours.
    """
    cache_key = _make_key("info", ticker_symbol)
    cached = _cache_get(cache_key, "info")
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        if not info or "regularMarketPrice" not in info:
            if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO"):
                return get_company_info(f"{ticker_symbol}.NS")

        # Only cache JSON-serializable values
        # yfinance sometimes returns non-serializable objects
        safe_info = {}
        for k, v in info.items():
            try:
                json.dumps(v)
                safe_info[k] = v
            except (TypeError, ValueError):
                safe_info[k] = str(v)

        _cache_set(cache_key, "info", safe_info)
        return safe_info

    except Exception as e:
        # Try stale cache as fallback
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
        return {"error": f"{e}"}


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
# ALL EXISTING FUNCTIONS (unchanged logic,
# just updated to use cached fetch_price_data)
# ─────────────────────────────────────────

def get_news(ticker_symbol: str) -> str:
    # News is NOT cached — always fetch fresh
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
        return f"{e}"


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
        return f"{e}"


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
    Generates a 14-day price forecast using Linear Regression.

    Why not Prophet?
    Prophet requires cmdstan (C++ binary) which is unreliable
    on cloud servers. Linear Regression has zero binary dependencies
    and works everywhere.

    How it works:
    1. Take the last 60 days of closing prices
    2. Create features: day index + 7-day and 21-day moving averages
    3. Train a Linear Regression model on these features
    4. Predict the next 14 days
    5. Add confidence bands using the model's residual std
    """
    try:
        from sklearn.linear_model import LinearRegression

        df = fetch_forecast_data(ticker_symbol)
        print(f"LR Forecast - ticker={ticker_symbol} df_type={type(df)} empty={df.empty if hasattr(df,'empty') else 'N/A'} rows={len(df) if df is not None else 0}")
        if df is None or df.empty or len(df) < 60:
            print(f"LR Forecast - insufficient data: {len(df) if df is not None else 0} rows")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Use closing prices
        close = df["Close"].dropna().values.flatten()
        dates = df.index

        # Features: day index, 7-day MA, 21-day MA
        n = len(close)
        X = []
        y = []

        for i in range(21, n):
            ma7  = close[i-7:i].mean()
            ma21 = close[i-21:i].mean()
            X.append([i, ma7, ma21])
            y.append(close[i])

        X = np.array(X)
        y = np.array(y)

        model = LinearRegression()
        model.fit(X, y)

        # Calculate residual std for confidence bands
        y_pred_train = model.predict(X)
        residual_std = np.std(y - y_pred_train)

        # Predict next 14 days
        last_close = close[-21:]
        forecast_values = []

        for i in range(14):
            idx = n + i
            ma7  = np.append(close[-7+i:], forecast_values)[-7:].mean() if i < 7 else np.array(forecast_values[-7:]).mean()
            ma21 = np.append(close[-21+i:], forecast_values)[-21:].mean() if i < 21 else np.array(forecast_values[-21:]).mean()
            pred = model.predict([[idx, ma7, ma21]])[0]
            forecast_values.append(pred)

        # Build date range for forecast
        last_date = dates[-1]
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        forecast_dates = pd.date_range(start=last_date, periods=15, freq="B")[1:]

        forecast_arr = np.array(forecast_values)
        upper = forecast_arr + 1.96 * residual_std
        lower = forecast_arr - 1.96 * residual_std

        # Build chart
        fig = go.Figure()

        # Historical prices
        hist_dates = [d.tz_localize(None) if hasattr(d, 'tz') and d.tz else d for d in dates[-90:]]
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=close[-90:],
            mode="lines",
            name="Historical",
            line=dict(color="#00c6ff", width=2)
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_arr,
            mode="lines",
            name="Forecast",
            line=dict(color="#7f5af0", width=2, dash="dash")
        ))

        # Confidence band (upper)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))

        # Confidence band (lower + fill)
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
        print(f"Forecast error for {ticker_symbol}: {e}")
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
        print("Analyze Error:", e)
        return None


def calculate_basic_risk_metrics(ticker, period="1y"):
    """
    Returns Volatility, Sharpe Ratio, and Max Drawdown.
    Results are cached since they're computed from price data.
    """
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
    """
    Generates a full stock analysis PDF report.
    Steps: fetch data → convert chart to PNG → build PDF → return bytes.
    """
    info        = get_company_info(ticker_symbol)
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
            print(f"Chart image error: {e}")

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