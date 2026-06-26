"""
core/engine.py
──────────────
Data fetching, caching, analysis, forecasting, and PDF generation.
All yfinance calls use curl_cffi with Chrome impersonation to avoid
rate-limiting on cloud IPs (Render, etc.).
Cache: SQLite at /tmp/cache.db — 6hr TTL for price data, 24hr for company info.
"""

import os
import sqlite3
import json
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── yfinance with curl_cffi to avoid Yahoo Finance rate-limiting on cloud IPs ──
try:
    from curl_cffi import requests as cffi_requests
    _SESSION = cffi_requests.Session(impersonate="chrome")
except Exception:
    _SESSION = None

import yfinance as yf

# ── Cache config ──
_CACHE_DB = os.environ.get("CACHE_DB_PATH", "/tmp/cache.db")
_TTL_PRICE = 6 * 3600       # 6 hours for price data
_TTL_INFO  = 24 * 3600      # 24 hours for company info


# ─────────────────────────────────────────
# CACHE LAYER
# ─────────────────────────────────────────

def _get_conn():
    conn = sqlite3.connect(_CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key       TEXT PRIMARY KEY,
            value     TEXT,
            cached_at REAL
        )
    """)
    conn.commit()
    return conn


def _cache_get(key: str, ttl: int):
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT value, cached_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row and (time.time() - row[1]) < ttl:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _cache_set(key: str, value):
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, cached_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), time.time())
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _cache_stats():
    try:
        conn = _get_conn()
        row = conn.execute("SELECT COUNT(*), MIN(cached_at) FROM cache").fetchone()
        conn.close()
        total = row[0] or 0
        oldest = (
            datetime.fromtimestamp(row[1]).strftime("%Y-%m-%d %H:%M")
            if row[1] else "—"
        )
        return {"total": total, "oldest": oldest}
    except Exception:
        return {"total": 0, "oldest": "—"}


def clear_cache():
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────

def fetch_price_data(ticker: str, period: str = "1y",
                     start=None, end=None):
    """
    Returns (DataFrame, is_stale).
    is_stale=True means we returned cached data because live fetch failed.
    """
    cache_key = f"price:{ticker}:{period}:{start}:{end}"
    cached = _cache_get(cache_key, _TTL_PRICE)

    try:
        kwargs = dict(progress=False)
        if _SESSION:
            kwargs["session"] = _SESSION
        if start and end:
            kwargs["start"] = str(start)
            kwargs["end"]   = str(end)
        else:
            kwargs["period"] = period

        df = yf.download(ticker, **kwargs)

        if df is None or df.empty:
            raise ValueError("Empty dataframe returned")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        _cache_set(cache_key, df.reset_index().to_dict(orient="list"))
        return df, False

    except Exception:
        # Fall back to cache even if stale
        if cached:
            df = pd.DataFrame(cached)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            return df, True
        return pd.DataFrame(), False


def _get_ticker_info(ticker: str) -> dict:
    cache_key = f"info:{ticker}"
    cached = _cache_get(cache_key, _TTL_INFO)
    if cached:
        return cached
    try:
        kwargs = {"session": _SESSION} if _SESSION else {}
        stock = yf.Ticker(ticker, **kwargs)
        info = stock.info or {}
        _cache_set(cache_key, info)
        return info
    except Exception:
        return {}


# ─────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────

def get_stock_price_chart(ticker: str, period: str = "1y",
                          start=None, end=None):
    """Returns (plotly Figure, is_stale)."""
    df, is_stale = fetch_price_data(ticker, period, start, end)
    if df.empty:
        return None, is_stale

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return None, is_stale

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color="#00c6ff",
        decreasing_line_color="#ff4d4d",
    ), row=1, col=1)

    # SMA 20 & 50
    close = df["Close"].dropna()
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma20, name="SMA 20",
            line=dict(color="#f5a623", width=1.5)
        ), row=1, col=1)
    if len(close) >= 50:
        sma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma50, name="SMA 50",
            line=dict(color="#7f5af0", width=1.5)
        ), row=1, col=1)

    # Volume
    if "Volume" in df.columns:
        colors = [
            "#00c6ff" if c >= o else "#ff4d4d"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume", marker_color=colors, opacity=0.6
        ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Price Chart",
        xaxis_rangeslider_visible=False,
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",      row=2, col=1)

    return fig, is_stale


def get_fundamental_data(ticker: str) -> str:
    """Returns a markdown-formatted string of key fundamentals."""
    info = _get_ticker_info(ticker)
    if not info:
        return ""

    def fmt(val, prefix="", suffix="", billions=False):
        if val is None or val == "N/A":
            return "N/A"
        try:
            if billions:
                return f"{prefix}{float(val)/1e9:.2f}B{suffix}"
            return f"{prefix}{float(val):.2f}{suffix}"
        except Exception:
            return str(val)

    rows = [
        ("Company",        info.get("longName", ticker)),
        ("Sector",         info.get("sector", "N/A")),
        ("Industry",       info.get("industry", "N/A")),
        ("Market Cap",     fmt(info.get("marketCap"), prefix="$", billions=True)),
        ("P/E Ratio",      fmt(info.get("forwardPE"))),
        ("EPS (TTM)",      fmt(info.get("trailingEps"), prefix="$")),
        ("Revenue (TTM)",  fmt(info.get("totalRevenue"), prefix="$", billions=True)),
        ("Gross Margin",   fmt(info.get("grossMargins"), suffix="%")),
        ("52W High",       fmt(info.get("fiftyTwoWeekHigh"), prefix="$")),
        ("52W Low",        fmt(info.get("fiftyTwoWeekLow"), prefix="$")),
        ("Dividend Yield", fmt(info.get("dividendYield"), suffix="%")),
        ("Beta",           fmt(info.get("beta"))),
    ]

    lines = ["| Metric | Value |", "|--------|-------|"]
    for label, val in rows:
        lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def get_company_info(ticker: str) -> str:
    """Returns company name, sector, and description as plain text."""
    info = _get_ticker_info(ticker)
    name    = info.get("longName", ticker)
    sector  = info.get("sector", "N/A")
    desc    = info.get("longBusinessSummary", "No description available.")
    return f"**{name}** | Sector: {sector}\n\n{desc}"


def calculate_basic_risk_metrics(ticker: str, period: str = "1y") -> dict:
    """Returns dict with Volatility, Sharpe Ratio, Max Drawdown."""
    df, _ = fetch_price_data(ticker, period)
    if df.empty:
        return {}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return {}

    close   = df["Close"].dropna()
    returns = close.pct_change().dropna()

    if len(returns) < 10:
        return {}

    vol        = float(returns.std() * np.sqrt(252))
    sharpe     = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0
    cum        = (1 + returns).cumprod()
    roll_max   = cum.cummax()
    drawdown   = (cum - roll_max) / roll_max
    max_dd     = float(drawdown.min())

    return {
        "Volatility":   vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


def analyze_stock(ticker: str, period: str = "1y") -> str:
    """Returns a plain-text summary of trend, volatility, and risk."""
    df, _ = fetch_price_data(ticker, period)
    if df.empty:
        return f"Could not fetch data for {ticker}."

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close   = df["Close"].dropna()
    returns = close.pct_change().dropna()

    price  = float(close.iloc[-1])
    sma20  = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
    sma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
    vol    = float(returns.std() * np.sqrt(252))

    if price > sma20 > sma50:
        trend = "Uptrend"
    elif price < sma20 < sma50:
        trend = "Downtrend"
    else:
        trend = "Sideways"

    risk = "High" if vol > 0.4 else ("Medium" if vol > 0.25 else "Low")

    return (
        f"{ticker} is currently in a **{trend}**. "
        f"Price: ${price:.2f} | SMA20: ${sma20:.2f} | SMA50: ${sma50:.2f}. "
        f"Annualized volatility: {vol:.1%} ({risk} risk)."
    )


# ─────────────────────────────────────────
# NEWS
# ─────────────────────────────────────────

def get_news(ticker: str) -> str:
    """Fetches latest news via GNews. Returns HTML string."""
    try:
        from gnews import GNews
        gn = GNews(language="en", country="US", max_results=8)
        articles = gn.get_news(ticker)
        if not articles:
            return "<p>No recent news found.</p>"

        html = ""
        for a in articles:
            title     = a.get("title", "No title")
            url       = a.get("url", "#")
            publisher = a.get("publisher", {}).get("title", "")
            date      = a.get("published date", "")
            html += (
                f'<div style="margin-bottom:16px; padding:12px; '
                f'border-left:3px solid #00c6ff; background:#1c2331; border-radius:4px;">'
                f'<a href="{url}" target="_blank" style="color:#00c6ff; '
                f'font-weight:600; text-decoration:none;">{title}</a><br>'
                f'<small style="color:#8b949e;">{publisher} — {date}</small>'
                f'</div>'
            )
        return html
    except Exception as e:
        return f"<p>News unavailable: {e}</p>"


# ─────────────────────────────────────────
# FORECASTING — Linear Regression
# ─────────────────────────────────────────

def get_price_forecast(ticker: str) -> object:
    """
    14-day price forecast using Linear Regression with MA features.
    Returns a Plotly figure or None on failure.
    """
    try:
        from sklearn.linear_model import LinearRegression

        df, _ = fetch_price_data(ticker, period="2y")
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].dropna().reset_index(drop=True)
        if len(close) < 60:
            return None

        # Features: MA5, MA10, MA20, lag1, lag2
        data = pd.DataFrame({"close": close})
        data["ma5"]  = data["close"].rolling(5).mean()
        data["ma10"] = data["close"].rolling(10).mean()
        data["ma20"] = data["close"].rolling(20).mean()
        data["lag1"] = data["close"].shift(1)
        data["lag2"] = data["close"].shift(2)
        data = data.dropna()

        X = data[["ma5", "ma10", "ma20", "lag1", "lag2"]].values
        y = data["close"].values

        model = LinearRegression()
        model.fit(X, y)

        # Forecast 14 days ahead iteratively
        last_close = list(close.iloc[-20:])
        forecast_prices = []

        for _ in range(14):
            s = pd.Series(last_close)
            ma5  = float(s.rolling(5).mean().iloc[-1])
            ma10 = float(s.rolling(10).mean().iloc[-1])
            ma20 = float(s.rolling(20).mean().iloc[-1])
            lag1 = last_close[-1]
            lag2 = last_close[-2]
            pred = model.predict([[ma5, ma10, ma20, lag1, lag2]])[0]
            forecast_prices.append(pred)
            last_close.append(pred)

        last_date      = df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date, periods=15)[1:]

        hist_tail = df["Close"].iloc[-60:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail.values,
            mode="lines", name="Historical",
            line=dict(color="#00c6ff", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_prices,
            mode="lines+markers", name="Forecast",
            line=dict(color="#f5a623", width=2, dash="dash"),
            marker=dict(size=5)
        ))
        fig.update_layout(
            title=f"{ticker} — 14-Day Price Forecast (Linear Regression)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=450,
        )
        return fig

    except Exception:
        return None


# ─────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────

def ma_crossover_backtest(ticker: str, short_window: int = 20,
                          long_window: int = 50, period: str = "1y"):
    """
    MA crossover strategy backtest.
    Returns DataFrame with Cumulative_Market, Cumulative_Strategy, Drawdown.
    """
    df, _ = fetch_price_data(ticker, period)
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return None

    close = df["Close"].dropna()
    if len(close) < long_window + 10:
        return None

    bt = pd.DataFrame(index=close.index)
    bt["Close"]     = close
    bt["SMA_short"] = close.rolling(short_window).mean()
    bt["SMA_long"]  = close.rolling(long_window).mean()
    bt = bt.dropna()

    bt["Signal"]   = (bt["SMA_short"] > bt["SMA_long"]).astype(int)
    bt["Position"] = bt["Signal"].shift(1).fillna(0)
    bt["Returns"]  = bt["Close"].pct_change().fillna(0)

    bt["Strategy_Returns"]    = bt["Position"] * bt["Returns"]
    bt["Cumulative_Market"]   = (1 + bt["Returns"]).cumprod()
    bt["Cumulative_Strategy"] = (1 + bt["Strategy_Returns"]).cumprod()

    roll_max      = bt["Cumulative_Strategy"].cummax()
    bt["Drawdown"] = (bt["Cumulative_Strategy"] - roll_max) / roll_max

    return bt


# ─────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────

def generate_pdf(ticker: str, period: str = "1y") -> bytes:
    """
    Generates a PDF report for the given ticker.
    Returns raw bytes.
    """
    try:
        from fpdf import FPDF
        import tempfile

        metrics = calculate_basic_risk_metrics(ticker, period)
        info    = _get_ticker_info(ticker)
        name    = info.get("longName", ticker)
        sector  = info.get("sector", "N/A")
        summary = analyze_stock(ticker, period)

        # Try to generate and save chart image
        chart_path = None
        try:
            import plotly.io as pio
            fig, _ = get_stock_price_chart(ticker, period)
            if fig:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                pio.write_image(fig, tmp.name, width=900, height=400)
                chart_path = tmp.name
        except Exception:
            pass

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 12, f"AI Financial Analyst — {ticker}", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"{name} | Sector: {sector} | Period: {period}", ln=True)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(4)

        # Risk metrics table
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Risk Metrics", ln=True)
        pdf.set_font("Helvetica", "", 11)

        if metrics:
            for label, val in [
                ("Volatility",   f"{metrics['Volatility']:.2%}"),
                ("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}"),
                ("Max Drawdown", f"{metrics['Max Drawdown']:.2%}"),
            ]:
                pdf.cell(60, 8, label, border=1)
                pdf.cell(60, 8, val,   border=1, ln=True)
        else:
            pdf.cell(0, 8, "Metrics unavailable.", ln=True)

        pdf.ln(4)

        # Analysis summary
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Automated Analysis", ln=True)
        pdf.set_font("Helvetica", "", 10)
        # Strip markdown bold markers for plain PDF text
        plain_summary = summary.replace("**", "")
        pdf.multi_cell(0, 7, plain_summary)
        pdf.ln(4)

        # Chart image
        if chart_path and os.path.exists(chart_path):
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Price Chart", ln=True)
            pdf.image(chart_path, w=180)
            os.unlink(chart_path)

        return bytes(pdf.output())

    except Exception as e:
        raise RuntimeError(f"PDF generation failed: {e}\n{traceback.format_exc()}")