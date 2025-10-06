"""
app.py
------
Streamlit app for AI Stock Recommendation.

Features:
- Ticker + date inputs; optional news sentiment toggle (display-only if column exists)
- BUY/HOLD/SELL recommendation card with confidence and explainability
- Interactive Plotly charts: price with MAs & Bollinger bands, volume, RSI, MACD
- Optional sentiment timeline if 'sentiment_avg_compound' present
- Download computed dataset CSV and generate an executive brief (template-based)

Run:
  pip install -r requirements.txt
  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os

# Gemini (Google Generative AI)
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

from stock_logic import (
    fetch_price_data,
    compute_indicators,
    get_recommendation,
    build_feature_label_dataset,
    generate_brief,
    load_sample_tickers,
)

# News sentiment imports
try:
    from news_ingestion import fetch_newsapi_headlines, aggregate_daily_sentiment_vader
except ImportError:
    fetch_newsapi_headlines = None
    aggregate_daily_sentiment_vader = None


# -----------------------------
# Streamlit Config
# -----------------------------

st.set_page_config(
    page_title="AI Stock Recommendation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Caching Helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def cached_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    return fetch_price_data(ticker, start, end)


@st.cache_data(show_spinner=False)
def cached_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return compute_indicators(df)


@st.cache_data(show_spinner=False)
def cached_fetch_sentiment(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch and aggregate news sentiment for a ticker and date range."""
    if not api_key or not fetch_newsapi_headlines or not aggregate_daily_sentiment_vader:
        return pd.DataFrame(columns=["date", "sentiment_avg_compound", "headline_count"])
    
    try:
        # Fetch news headlines
        df_news = fetch_newsapi_headlines(
            query=ticker, 
            start_date=start, 
            end_date=end, 
            api_key=api_key
        )
        
        if df_news is None or df_news.empty:
            return pd.DataFrame(columns=["date", "sentiment_avg_compound", "headline_count"])
        
        # Aggregate daily sentiment
        daily_sentiment = aggregate_daily_sentiment_vader(df_news)
        return daily_sentiment
        
    except Exception as e:
        st.warning(f"News sentiment fetch failed: {e}")
        return pd.DataFrame(columns=["date", "sentiment_avg_compound", "headline_count"])


# -----------------------------
# UI Helpers
# -----------------------------

def _recommendation_badge(label: str, confidence: int) -> str:
    if label == "BUY":
        return f"🟢 BUY ({confidence})"
    if label == "SELL":
        return f"🔴 SELL ({confidence})"
    return f"🟡 HOLD ({confidence})"


def _styled_container(label: str):
    color = "#2ecc71" if label == "BUY" else ("#e74c3c" if label == "SELL" else "#f1c40f")
    return st.container(border=True)

def _fmt(value, fmt: str, fallback: str = "—") -> str:
    try:
        # If it's a Series or array, take the last element
        if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
            if len(value) == 0:
                return fallback
            value = value[-1]
        val = float(value)
        if np.isnan(val):
            return fallback
        return format(val, fmt)
    except Exception:
        return fallback


def _to_float(value) -> float:
    """Coerce potential Series/array/scalar to a float, or np.nan on failure."""
    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return np.nan
            value = value.iloc[-1]
        elif isinstance(value, (np.ndarray, list, tuple)):
            if len(value) == 0:
                return np.nan
            value = value[-1]
        return float(value)
    except Exception:
        return np.nan


def _call_gemini_for_recommendation(ticker: str, df: pd.DataFrame) -> tuple[str, int, dict]:
    """Use Gemini to generate BUY/HOLD/SELL, confidence, and explanation from the latest data.

    Returns (label, confidence, payload_dict). payload_dict contains 'explanation' text for UI.
    """
    # Ensure API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or genai is None:
        # Fallback minimal result
        return ("HOLD", 50, {"explanation": "Gemini not configured. Set GOOGLE_API_KEY to enable AI recommendations."})

    try:
        genai.configure(api_key=api_key)
        # Allow overriding the model via env; default to gemini-2.0-flash to match your key usage
        model_id = os.environ.get("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_id)
    except Exception:
        return ("HOLD", 50, {"explanation": "Failed initializing Gemini. Check GOOGLE_API_KEY."})

    # Prepare compact JSON-like context from latest row and a small history for momentum
    latest = df.iloc[-1]
    def safe(v, default=None):
        try:
            if isinstance(v, pd.DataFrame):
                try:
                    v = v.squeeze("columns")
                except Exception:
                    pass
            if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
                v = v[-1]
            if pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default

    history_len = min(30, len(df))
    recent = df.tail(history_len)

    # Column helpers to handle case/availability gracefully
    def pick_series(frame: pd.DataFrame, options: list[str]) -> pd.Series:
        # try exact matches first
        for name in options:
            if name in frame.columns:
                s = frame[name]
                if isinstance(s, pd.DataFrame):
                    # squeeze to one column if possible
                    if s.shape[1] > 0:
                        s = s.iloc[:, 0]
                    else:
                        continue
                return pd.to_numeric(s, errors="coerce")
        # case-insensitive / normalized search
        lower_map = {str(col).lower(): col for col in frame.columns}
        for name in options:
            key = name.lower()
            if key in lower_map:
                s = frame[lower_map[key]]
                if isinstance(s, pd.DataFrame):
                    if s.shape[1] > 0:
                        s = s.iloc[:, 0]
                    else:
                        continue
                return pd.to_numeric(s, errors="coerce")
        return pd.Series(dtype=float)

    close_series = pick_series(recent, ["Close", "close"])  # yfinance typically 'Close'
    rsi_series = pick_series(recent, ["rsi_14", "RSI", "rsi"]) if "rsi_14" in recent.columns or "RSI" in recent.columns or "rsi" in recent.columns else pd.Series(dtype=float)
    macd_series = pick_series(recent, ["macd", "MACD"]) if "macd" in recent.columns or "MACD" in recent.columns else pd.Series(dtype=float)
    sent_series = pick_series(recent, ["sentiment_avg_compound"]) if "sentiment_avg_compound" in recent.columns else pd.Series(dtype=float)

    payload = {
        "ticker": ticker,
        "latest": {
            "close": safe(latest.get("Close")),
            "sma_50": safe(latest.get("sma_50")),
            "sma_200": safe(latest.get("sma_200")),
            "rsi_14": safe(latest.get("rsi_14")),
            "macd": safe(latest.get("macd")),
            "macd_signal": safe(latest.get("macd_signal")),
            "atr_14": safe(latest.get("atr_14")),
            "vol_mean_21": safe(latest.get("vol_mean_21")),
            "daily_return": safe(latest.get("daily_return")),
            "sentiment_avg_compound": safe(latest.get("sentiment_avg_compound")),
        },
        "recent_closes": [safe(v) for v in close_series.dropna().tolist()],
        "recent_rsi": [safe(v) for v in rsi_series.dropna().tolist()],
        "recent_macd": [safe(v) for v in macd_series.dropna().tolist()],
        "recent_sentiment": [safe(v) for v in sent_series.dropna().tolist()],
    }

    system_instructions = (
        "You are an equity analyst. Analyze the provided technical snapshot and optional sentiment. "
        "Return a STRICT JSON with keys: label (BUY|HOLD|SELL), confidence (0-100 int), explanation (<=280 chars)."
    )
    user_prompt = (
        "Data:"\
        f"\n{payload}\n"\
        "Rules: Consider trend vs MAs, momentum via MACD/RSI, volatility via ATR, volume context, and sentiment. "
        "Do not include JSON comments."
    )

    try:
        resp = model.generate_content([
            {"role": "system", "parts": [system_instructions]},
            {"role": "user", "parts": [user_prompt]},
        ])
        text = (resp.text or "").strip()
        # Try to extract JSON
        import json, re
        # Find first JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        raw = match.group(0) if match else text
        obj = json.loads(raw)
        label = str(obj.get("label", "HOLD")).upper()
        if label not in ("BUY", "HOLD", "SELL"):
            label = "HOLD"
        try:
            conf = int(obj.get("confidence", 50))
        except Exception:
            conf = 50
        explanation = str(obj.get("explanation", "Generated by Gemini.")).strip()
        return (label, max(0, min(100, conf)), {"explanation": explanation})
    except Exception as e:
        return ("HOLD", 50, {"explanation": f"Gemini call failed: {e}"})


def plot_price_volume(df: pd.DataFrame) -> go.Figure:
    # Two-row subplot: price on top, volume below (shared x)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.75, 0.25])

    # Price traces
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="MA50", line=dict(color="#2ca02c", width=1)), row=1, col=1)
    if "sma_200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_200"], name="MA200", line=dict(color="#d62728", width=1)), row=1, col=1)

    # Bollinger band shading on price axis
    if set(["bb_upper_20", "bb_lower_20"]).issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper_20"], line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_lower_20"],
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.1)",
                line=dict(width=0),
                name="Bollinger 20",
            ),
            row=1,
            col=1,
        )

    # Volume bars on second row
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(128,128,128,0.5)"),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=600,
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", showgrid=False),
    )
    return fig


def plot_rsi_macd(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("RSI (14)", "MACD"))

    # RSI with 30/70 lines
    if "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI14", line=dict(color="#8e44ad")), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray", row=1, col=1)

    # MACD histogram and signal
    if set(["macd", "macd_signal", "macd_hist"]).issubset(df.columns):
        fig.add_trace(
            go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist", marker_color="rgba(52,152,219,0.5)"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#2980b9")), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#c0392b", dash="dot")),
            row=2,
            col=1,
        )

    fig.update_layout(height=500, margin=dict(l=30, r=30, t=30, b=30))
    return fig


def plot_sentiment(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sentiment_avg_compound"],
            name="Sentiment",
            line=dict(color="#16a085"),
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=30, b=30), yaxis=dict(title="VADER Compound"))
    return fig


# -----------------------------
# Main App
# -----------------------------

from plotly.subplots import make_subplots


def main():
    # Load .env for local dev if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    st.title("AI Stock Recommendation")
    st.caption("News Senitment based stock position recommendation using LLMs")

    # Sidebar inputs
    st.sidebar.header("Inputs")
    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=365)
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=default_end)
    include_sentiment = st.sidebar.checkbox("Include news sentiment (optional)", value=True)
    if include_sentiment and not os.environ.get("NEWSAPI_KEY"):
        st.sidebar.info("Set NEWSAPI_KEY in a .env file to enable news sentiment in dataset builds.")
    st.sidebar.write("Sample tickers:", ", ".join(load_sample_tickers()))
    run_btn = st.sidebar.button("Get Recommendation", use_container_width=True)

    # Early validation
    if run_btn:
        if not ticker.strip():
            st.error("Please enter a ticker symbol.")
            st.stop()
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

    # Trigger computation
    if run_btn:
        with st.spinner("Fetching data and computing indicators..."):
            # Fetch price data
            df_raw = cached_fetch(ticker.strip().upper(), str(start_date), str(end_date))
            if df_raw is None or df_raw.empty:
                st.error("No data found for the given inputs. Try another ticker or date range.")
                st.stop()
            
            # Compute technical indicators
            df = cached_indicators(df_raw)
            
            # Fetch news sentiment if enabled and API key available
            if include_sentiment:
                news_api_key = os.environ.get("NEWSAPI_KEY")
                if news_api_key:
                    with st.spinner("Fetching news sentiment..."):
                        sentiment_df = cached_fetch_sentiment(
                            ticker.strip().upper(), 
                            str(start_date), 
                            str(end_date), 
                            news_api_key
                        )
                        
                        # Merge sentiment data with price data
                        if not sentiment_df.empty:
                            # Convert sentiment date to datetime for merging
                            sentiment_df = sentiment_df.copy()
                            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                            
                            # Reset index of price data to get date as column
                            df_reset = df.reset_index()
                            if 'Date' in df_reset.columns:
                                df_reset = df_reset.rename(columns={'Date': 'date'})
                            elif df_reset.index.name == 'Date':
                                df_reset.index.name = 'date'
                                df_reset = df_reset.reset_index()
                            
                            # Merge on date
                            df_merged = df_reset.merge(sentiment_df, on='date', how='left')
                            
                            # Set date back as index
                            df_merged = df_merged.set_index('date')
                            df = df_merged
                            
                            # Show sentiment summary
                            sentiment_count = sentiment_df['headline_count'].sum()
                            avg_sentiment = sentiment_df['sentiment_avg_compound'].mean()
                            st.success(f"✅ Fetched {sentiment_count} headlines with avg sentiment: {avg_sentiment:.3f}")
                        else:
                            st.info("No news sentiment data found for this ticker and date range.")
                else:
                    st.warning("NewsAPI key not found. Set NEWSAPI_KEY in .env file to enable sentiment analysis.")
            
            # Warn if long windows not available
            if df["sma_200"].isna().sum() > 0:
                st.info("Some early dates have NaN for MA200. Consider selecting a wider date range to warm-up indicators.")

        latest_row = df.iloc[-1]
        # Use Gemini for recommendation (replaces rule-based logic)
        label, confidence, ai_payload = _call_gemini_for_recommendation(ticker.strip().upper(), df)

        # Top card
        st.subheader("Recommendation")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {_recommendation_badge(label, confidence)}")
        with col2:
            close_val = _to_float(latest_row.get('Close', np.nan))
            ret1d_val = _to_float(latest_row.get('daily_return', np.nan))
            close_display = f"{close_val:.2f}" if not np.isnan(close_val) else "—"
            ret_display = f"{ret1d_val*100:.2f}%" if not np.isnan(ret1d_val) else "—"
            st.metric("Close", close_display, delta=ret_display)

        # Two-column layout
        left, right = st.columns([3, 1])
        with left:
            try:
                # Debug: Show data info
                st.write(f"Data shape: {df.shape}")
                st.write(f"Columns: {list(df.columns)}")
                st.write(f"Index type: {type(df.index)}")
                
                price_fig = plot_price_volume(df)
                st.plotly_chart(price_fig, use_container_width=True, theme="streamlit")
            except Exception as e:
                st.warning(f"Unable to render price/volume chart: {e}")
                st.error(f"Error details: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        with right:
            st.markdown("#### Indicators (latest)")
            mini_cols = st.columns(2)
            mini_cols[0].metric("RSI14", _fmt(latest_row.get('rsi_14', np.nan), ".2f"))
            mini_cols[1].metric("MACD", _fmt(latest_row.get('macd', np.nan), ".4f"))
            mini_cols[0].metric("ATR14", _fmt(latest_row.get('atr_14', np.nan), ".2f"))
            mini_cols[1].metric("Volatility 21d", _fmt(latest_row.get('vol_21', np.nan), ".4f"))
            # Return display as percentage
            ret_display = "—"
            value = _to_float(latest_row.get('daily_return', np.nan))
            if not np.isnan(value):
                ret_val = value * 100.0
                ret_display = f"{ret_val:.2f}%"

            mini_cols[0].metric("Return 1d", ret_display)
            mini_cols[1].metric("Avg Volume 21d", _fmt(latest_row.get('vol_mean_21', np.nan), ".0f"))

            st.markdown("#### AI Rationale")
            st.write(ai_payload.get("explanation", "Generated by Gemini"))

        st.markdown("### Momentum & Oscillators")
        try:
            st.plotly_chart(plot_rsi_macd(df), use_container_width=True, theme="streamlit")
        except Exception as e:
            st.warning(f"Unable to render RSI/MACD chart: {e}")

        # Optional sentiment timeline if column exists and user opted in
        if include_sentiment and "sentiment_avg_compound" in df.columns and not df["sentiment_avg_compound"].isna().all():
            st.markdown("### News Sentiment (if available)")
            st.plotly_chart(plot_sentiment(df), use_container_width=True, theme="streamlit")
        elif include_sentiment:
            st.info("Sentiment column not found in data. Provide it from your backend to enable this panel.")

        # Dataset download & brief
        st.markdown("---")
        try:
            dataset = build_feature_label_dataset(df)
            csv_bytes = dataset.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Dataset (CSV)",
                data=csv_bytes,
                file_name=f"{ticker.upper()}_features_labels.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Unable to build downloadable dataset: {e}")

        if st.button("Generate Brief", use_container_width=True):
            try:
                brief = generate_brief(df, label, confidence)
                st.success(brief)
            except Exception as e:
                st.warning(f"Unable to generate brief: {e}")


if __name__ == "__main__":
    # Example usage instructions for local run
    print("Install deps: pip install -r requirements.txt")
    print("Run app: streamlit run app.py")
    main()


