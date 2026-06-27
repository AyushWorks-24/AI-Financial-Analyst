import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from rapidfuzz import fuzz, process

from core.engine import (
    _SESSION,
    _cache_stats,
    _get_ticker_info,
    calculate_basic_risk_metrics,
    clear_cache,
    fetch_price_data,
    generate_pdf,
    get_fundamental_data,
    get_news,
    get_price_forecast,
    get_stock_price_chart,
    ma_crossover_backtest,
    analyze_stock,
)
from core.agent import create_agent_executor

st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────

def load_css():
    css_path = os.path.join("ui", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────

st.markdown("""
<div class="navbar">
  <div class="nav-title">📊 AI Financial Analyst</div>
  <div class="nav-items">
    <span>Analysis</span>
    <span>Forecast</span>
    <span>Backtest</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

@st.cache_data
def load_tickers():
    path = os.path.join("data", "tickers.csv")
    df = pd.read_csv(path)
    df["Ticker"] = df["Ticker"].str.upper()
    return df

tickers_df  = load_tickers()
ticker_list = tickers_df["Ticker"].tolist()

search_query = st.sidebar.text_input("🔍 Search Stock", placeholder="Type ticker...")

ticker = None
if search_query and len(search_query) >= 2:
    contains = [t for t in ticker_list if search_query.upper() in t]
    fuzzy    = process.extract(search_query.upper(), ticker_list, scorer=fuzz.WRatio, limit=10)
    fuzzy    = [m[0] for m in fuzzy if m[1] > 70]
    combined = list(dict.fromkeys(contains + fuzzy))[:15]
    if combined:
        ticker = st.sidebar.selectbox("Select Stock", combined)
    else:
        st.sidebar.warning("No matching tickers found.")
else:
    st.sidebar.info("Type at least 2 letters")

period = st.sidebar.selectbox(
    "Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3
)

use_date_filter = st.sidebar.checkbox("Use Custom Date Range")
start_date = end_date = None
if use_date_filter:
    start_date = st.sidebar.date_input("Start Date", date(2024, 1, 1))
    end_date   = st.sidebar.date_input("End Date", date.today())

# Cache panel
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🗄️ Data Cache")
stats = _cache_stats()
if stats["total"] > 0:
    st.sidebar.markdown(
        f"<small style='color:#8b949e;'>"
        f"📦 {stats['total']} records cached<br>"
        f"🕐 Oldest: {stats['oldest']}</small>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
        if clear_cache():
            st.sidebar.success("Cache cleared!")
            st.rerun()
else:
    st.sidebar.markdown(
        "<small style='color:#484f58;'>No cache yet — data will be<br>"
        "cached on first load for faster repeat visits.</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# LIVE PRICE HEADER
# ─────────────────────────────────────────

if ticker:
    # Uses _SESSION via fetch_ticker_history in engine — no raw yf calls here
    try:
        kwargs = {"session": _SESSION} if _SESSION else {}
        stock  = yf.Ticker(ticker, **kwargs)
        hist   = stock.history(period="2d")
    except Exception:
        hist = pd.DataFrame()

    if not hist.empty:
        price      = hist["Close"].iloc[-1]
        prev_price = hist["Close"].iloc[-2] if len(hist) >= 2 else price
        change     = price - prev_price
        change_pct = (change / prev_price) * 100
        arrow      = "▲" if change >= 0 else "▼"
        color      = "#00c6ff" if change >= 0 else "#ff4d4d"
        st.markdown(f"""
        <div class="stock-header">
          <h2>{ticker}</h2>
          <div class="price">${price:.2f}
            <span style="font-size:1rem; color:{color}; margin-left:12px;">
              {arrow} {change:+.2f} ({change_pct:+.2f}%)
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="stock-header">
          <h2>{ticker}</h2>
          <div class="price" style="color:#8b949e; font-size:0.9rem;">
            ⚠️ Live price unavailable (rate limit). Cached data will still load below.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Analysis", "📈 Fundamentals", "📰 News",
    "🤖 Forecast", "📉 Backtest", "💬 AI Chat",
    "📂 Portfolio", "📤 Export",
])

# ── TAB 1 — Analysis ──────────────────────

with tab1:
    if not ticker:
        st.info("👈 Search and select a stock from the sidebar to get started.")
    else:
        fig, is_stale = get_stock_price_chart(ticker, period, start_date, end_date)
        if is_stale:
            st.warning("⚠️ Showing cached data — Yahoo Finance is currently unreachable.")
        if fig:
            fig.update_layout(
                template="plotly_dark",
                hovermode="closest",
                hoverlabel=dict(bgcolor="#1c2331", font_size=14, font_color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

            raw_df, _ = fetch_price_data(ticker, period, start_date, end_date)
            if not raw_df.empty:
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                st.download_button(
                    "⬇️ Download Price Data (CSV)",
                    data=raw_df.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker}_price_{period}.csv",
                    mime="text/csv",
                )
        else:
            st.warning("Could not load price chart.")

        # Automated analysis card — uses selected period, not hardcoded 1y
        _df2, _ = fetch_price_data(ticker, period)
        if _df2 is not None and not _df2.empty:
            if isinstance(_df2.columns, pd.MultiIndex):
                _df2.columns = _df2.columns.get_level_values(0)
            if "Close" in _df2.columns and len(_df2) >= 50:
                _close    = _df2["Close"].dropna()
                _returns  = _close.pct_change().dropna()
                _price_val = float(_close.iloc[-1])
                _sma20     = float(_close.rolling(20).mean().iloc[-1])
                _sma50     = float(_close.rolling(50).mean().iloc[-1])
                _vol       = float(_returns.std() * np.sqrt(252))

                if _price_val > _sma20 > _sma50:
                    _trend, _tc = "Uptrend", "green"
                elif _price_val < _sma20 < _sma50:
                    _trend, _tc = "Downtrend", "red"
                else:
                    _trend, _tc = "Sideways", "purple"

                _risk = "High" if _vol > 0.4 else ("Medium" if _vol > 0.25 else "Low")
                _rc   = "red" if _risk == "High" else ("cyan" if _risk == "Medium" else "green")

                st.markdown(f"""
                <div class="section-card">
                  <div style="font-size:0.82rem;font-weight:600;color:var(--text-secondary);
                       text-transform:uppercase;letter-spacing:0.08em;margin-bottom:16px;">
                    Automated Analysis
                  </div>
                  <div style="display:flex;gap:36px;flex-wrap:wrap;align-items:flex-end;">
                    <div>
                      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;
                           letter-spacing:0.08em;margin-bottom:4px;">Current Price</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;
                           font-weight:600;color:var(--accent-cyan);">${_price_val:.2f}</div>
                    </div>
                    <div>
                      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;
                           letter-spacing:0.08em;margin-bottom:6px;">Trend</div>
                      <span class="badge {_tc}">{_trend}</span>
                    </div>
                    <div>
                      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;
                           letter-spacing:0.08em;margin-bottom:4px;">Annualized Volatility</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                           font-weight:600;color:var(--text-primary);">{_vol:.2%}</div>
                    </div>
                    <div>
                      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;
                           letter-spacing:0.08em;margin-bottom:6px;">Risk Level</div>
                      <span class="badge {_rc}">{_risk}</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        metrics = calculate_basic_risk_metrics(ticker)
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("📉 Volatility",   f"{metrics['Volatility']:.2%}")
            c2.metric("📐 Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            c3.metric("🕳️ Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            st.download_button(
                "⬇️ Download Risk Metrics (CSV)",
                data=pd.DataFrame([{
                    "Ticker": ticker, "Period": period,
                    "Volatility":   f"{metrics['Volatility']:.2%}",
                    "Sharpe Ratio": f"{metrics['Sharpe Ratio']:.2f}",
                    "Max Drawdown": f"{metrics['Max Drawdown']:.2%}",
                }]).to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_metrics_{period}.csv",
                mime="text/csv",
                key="metrics_csv_tab1",
            )

# ── TAB 2 — Fundamentals ──────────────────

with tab2:
    if not ticker:
        st.info("👈 Select a stock from the sidebar first.")
    else:
        st.subheader(f"📈 Fundamental Data — {ticker}")
        fund_data = get_fundamental_data(ticker)
        if fund_data:
            st.markdown(fund_data)
        else:
            st.warning("No fundamental data available.")

        # FIX: use _get_ticker_info instead of raw yf.Ticker — routes through
        # _SESSION and SQLite cache, avoids Yahoo rate-limiting on Render
        info = _get_ticker_info(ticker)

        desc = info.get("longBusinessSummary")
        if desc:
            with st.expander("📋 About the Company"):
                st.write(desc)

        fund_df = pd.DataFrame([{
            "Company":        info.get("longName", ticker),
            "Sector":         info.get("sector", "N/A"),
            "Market Cap":     info.get("marketCap", "N/A"),
            "P/E Ratio":      info.get("forwardPE", "N/A"),
            "EPS (TTM)":      info.get("trailingEps", "N/A"),
            "52W High":       info.get("fiftyTwoWeekHigh", "N/A"),
            "52W Low":        info.get("fiftyTwoWeekLow", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
        }])
        st.download_button(
            "⬇️ Download Fundamentals (CSV)",
            data=fund_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_fundamentals.csv",
            mime="text/csv",
        )

# ── TAB 3 — News ──────────────────────────

with tab3:
    if not ticker:
        st.info("👈 Select a stock from the sidebar first.")
    else:
        st.subheader(f"📰 Latest News — {ticker}")
        with st.spinner("Fetching latest news..."):
            news = get_news(ticker)
        st.markdown(news, unsafe_allow_html=True)

# ── TAB 4 — Forecast ──────────────────────

with tab4:
    if not ticker:
        st.info("👈 Select a stock from the sidebar first.")
    else:
        st.subheader(f"🤖 14-Day AI Price Forecast — {ticker}")
        st.caption("Linear Regression with moving average features — trained on 2 years of data.")
        with st.spinner("Running forecast model..."):
            try:
                forecast_fig = get_price_forecast(ticker)
            except Exception as e:
                forecast_fig = None
                st.error(f"Forecast error: {e}")

        if forecast_fig:
            forecast_fig.update_layout(template="plotly_dark")
            st.plotly_chart(forecast_fig, use_container_width=True)
            st.warning(
                "⚠️ Disclaimer: This forecast is for educational purposes only. "
                "Not financial advice."
            )
        else:
            st.warning("Forecast unavailable. Yahoo Finance may be rate-limiting.")

# ── TAB 5 — Backtest ──────────────────────

with tab5:
    if not ticker:
        st.info("👈 Select a stock from the sidebar first.")
    else:
        st.subheader(f"📉 MA Crossover Backtest — {ticker}")
        st.caption("Adjust the moving average windows — chart updates live.")

        cs, cl = st.columns(2)
        with cs:
            short_win = st.slider("Short MA (days)", 5, 50, 20, 1)
        with cl:
            long_win = st.slider("Long MA (days)", 20, 200, 50, 5)

        if short_win >= long_win:
            st.warning("⚠️ Short window must be smaller than Long window.")
        else:
            df_bt = ma_crossover_backtest(ticker, short_win, long_win, period)
            if df_bt is not None:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=df_bt.index, y=df_bt["Cumulative_Market"],
                    mode="lines", name="Buy & Hold",
                    line=dict(color="#00c6ff"),
                ))
                fig_eq.add_trace(go.Scatter(
                    x=df_bt.index, y=df_bt["Cumulative_Strategy"],
                    mode="lines", name=f"MA {short_win}/{long_win}",
                    line=dict(color="#7f5af0"),
                ))
                fig_eq.update_layout(
                    template="plotly_dark",
                    title=f"MA {short_win}/{long_win} vs Buy & Hold",
                    yaxis_title="Portfolio Value (start = 1.0)",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=df_bt.index, y=df_bt["Drawdown"],
                    mode="lines", fill="tozeroy",
                    name="Drawdown", line=dict(color="#ff4d4d"),
                ))
                fig_dd.update_layout(
                    template="plotly_dark", title="Strategy Drawdown",
                    yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig_dd, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("📈 Buy & Hold", f"{(df_bt['Cumulative_Market'].iloc[-1]-1)*100:.2f}%")
                c2.metric("🧠 Strategy",   f"{(df_bt['Cumulative_Strategy'].iloc[-1]-1)*100:.2f}%")
                c3.metric("🕳️ Max DD",     f"{df_bt['Drawdown'].min()*100:.2f}%")

                bt_export = df_bt[["Cumulative_Market", "Cumulative_Strategy", "Drawdown"]].copy()
                bt_export.columns = ["Buy & Hold", "MA Strategy", "Drawdown"]
                st.download_button(
                    "⬇️ Download Backtest (CSV)",
                    data=bt_export.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker}_backtest_{short_win}_{long_win}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Not enough data to run backtest.")

# ── TAB 6 — AI Chat ───────────────────────

with tab6:
    st.subheader("💬 AI Financial Analyst Chat")
    st.caption("Powered by LLaMA 3.3 70B via Groq.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about any stock — e.g. 'Analyze AAPL' or 'Risk of TSLA?'")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    agent  = create_agent_executor()
                    resp   = agent.invoke({"input": user_input})
                    answer = resp.get("output", "Sorry, no response.")
                except Exception as e:
                    answer = f"⚠️ Agent error: {e}"
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── TAB 7 — Portfolio ─────────────────────

with tab7:
    st.subheader("📂 Portfolio Tracker")
    st.caption("Add up to 8 stocks and compare performance side by side.")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    ci, cb, cp = st.columns([2, 1, 1])
    with ci:
        new_ticker = st.text_input(
            "ticker_input", placeholder="e.g. AAPL, TSLA",
            label_visibility="collapsed"
        )
    with cb:
        add_clicked = st.button("➕ Add Stock", use_container_width=True)
    with cp:
        portfolio_period = st.selectbox(
            "Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3, key="portfolio_period"
        )

    if add_clicked:
        cleaned = new_ticker.strip().upper()
        if not cleaned:
            st.warning("Enter a ticker symbol.")
        elif cleaned in st.session_state.portfolio:
            st.warning(f"{cleaned} already in portfolio.")
        elif len(st.session_state.portfolio) >= 8:
            st.warning("Maximum 8 stocks.")
        else:
            # FIX: use fetch_price_data instead of raw yf.download
            # routes through _SESSION and SQLite cache
            test, _ = fetch_price_data(cleaned, period="5d")
            if test.empty:
                st.error(f"Could not find data for '{cleaned}'.")
            else:
                st.session_state.portfolio.append(cleaned)
                st.rerun()

    if st.session_state.portfolio:
        chip_cols = st.columns(len(st.session_state.portfolio) + 1)
        for i, t in enumerate(st.session_state.portfolio):
            with chip_cols[i]:
                if st.button(f"❌ {t}", key=f"remove_{t}", use_container_width=True):
                    st.session_state.portfolio.remove(t)
                    st.rerun()
        with chip_cols[-1]:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.portfolio = []
                st.rerun()

        st.divider()

        @st.cache_data(ttl=300)
        def fetch_normalized(tickers: tuple, per: str):
            result = {}
            for t in tickers:
                # FIX: use fetch_price_data instead of raw yf.download
                # routes through _SESSION and SQLite cache
                df, _ = fetch_price_data(t, period=per)
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close = df["Close"].dropna()
                if len(close) == 0:
                    continue
                result[t] = (close / close.iloc[0]) * 100
            return pd.DataFrame(result)

        with st.spinner("Fetching portfolio data..."):
            norm_df = fetch_normalized(tuple(st.session_state.portfolio), portfolio_period)

        if norm_df.empty:
            st.error("Could not fetch data for any selected stocks.")
        else:
            colors = ["#00c6ff","#7f5af0","#ff6b6b","#ffd93d",
                      "#6bcb77","#ff922b","#cc5de8","#20c997"]
            fig_port = go.Figure()
            for i, col in enumerate(norm_df.columns):
                fig_port.add_trace(go.Scatter(
                    x=norm_df.index, y=norm_df[col],
                    mode="lines", name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig_port.update_layout(
                template="plotly_dark",
                title=f"Normalized Performance (base=100) — {portfolio_period}",
                yaxis_title="Normalized Price",
                hovermode="x unified", height=500,
            )
            st.plotly_chart(fig_port, use_container_width=True)

            st.subheader("📊 Side-by-Side Metrics")
            rows = []
            for t in st.session_state.portfolio:
                m = calculate_basic_risk_metrics(t, period=portfolio_period)
                if m:
                    ret = norm_df[t].iloc[-1] - 100 if t in norm_df.columns else None
                    rows.append({
                        "Ticker":       t,
                        "Return (%)":   f"{ret:.2f}%" if ret is not None else "N/A",
                        "Volatility":   f"{m['Volatility']:.2%}",
                        "Sharpe Ratio": f"{m['Sharpe Ratio']:.2f}",
                        "Max Drawdown": f"{m['Max Drawdown']:.2%}",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if len(norm_df.columns) >= 2:
                st.subheader("🔗 Correlation Matrix")
                corr = norm_df.pct_change().dropna().corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                ))
                fig_corr.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_corr, use_container_width=True)

            st.download_button(
                "⬇️ Download Portfolio Data (CSV)",
                data=norm_df.reset_index().to_csv(index=False).encode("utf-8"),
                file_name=f"portfolio_{portfolio_period}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("👆 Type a ticker and click ➕ Add Stock. Try AAPL, TSLA, GOOGL.")

# ── TAB 8 — Export ────────────────────────

with tab8:
    st.subheader("📤 Export Center")
    if not ticker:
        st.info("👈 Select a stock from the sidebar first.")
    else:
        st.markdown(f"### `{ticker}` — Period: `{period}`")
        st.divider()

        st.markdown("#### 📄 Full PDF Report")
        if st.button("🖨️ Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF..."):
                try:
                    pdf_bytes = generate_pdf(ticker, period)
                    st.success("✅ PDF ready!")
                    st.download_button(
                        f"⬇️ Download {ticker} Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"{ticker}_report_{period}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        st.divider()
        st.markdown("#### 📊 CSV Exports")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Price History**")
            raw_df, _ = fetch_price_data(ticker, period, start_date, end_date)
            if not raw_df.empty:
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                st.download_button(
                    "⬇️ Price Data (CSV)",
                    data=raw_df.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker}_price_{period}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="price_csv_export",
                )
        with c2:
            st.markdown("**Risk Metrics**")
            metrics = calculate_basic_risk_metrics(ticker, period)
            if metrics:
                st.download_button(
                    "⬇️ Risk Metrics (CSV)",
                    data=pd.DataFrame([{
                        "Ticker": ticker, "Period": period,
                        "Volatility":   f"{metrics['Volatility']:.2%}",
                        "Sharpe Ratio": f"{metrics['Sharpe Ratio']:.2f}",
                        "Max Drawdown": f"{metrics['Max Drawdown']:.2%}",
                    }]).to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker}_metrics_{period}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="metrics_csv_export",
                )