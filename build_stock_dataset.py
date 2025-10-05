"""
README / Purpose
=================
This script builds a supervised dataset for a 3-class stock recommendation model (BUY / HOLD / SELL).

It fetches historical OHLCV data with yfinance, computes common technical indicators and volume features,
optionally augments with daily news sentiment (using NewsAPI + VADER), and produces:
- One CSV per ticker: datasets/<TICKER>_dataset.csv
- A combined CSV for all tickers: datasets/all_tickers_dataset.csv

Columns Overview
----------------
- date: Trading date (index reset to column)
- ticker: Stock ticker symbol
- open, high, low, close, adj_close, volume: Daily OHLCV
- Technical indicators (selected set):
  * sma_10, sma_20, sma_50, sma_200
  * ema_12, ema_26
  * macd, macd_signal, macd_hist
  * bb_middle_20, bb_upper_20, bb_lower_20, bb_bandwidth_20
  * rsi_14
  * atr_14
  * daily_return
  * rolling_volatility_21 (std dev of daily returns over 21 trading days)
  * volume_change (pct_change of volume)
  * avg_volume_21
- Sentiment features (if NewsAPI key provided and news available):
  * sentiment_avg_compound (daily average VADER compound score)
  * headline_count
- Targets:
  * next_day_return = (close_{t+1} - close_t) / close_t
  * label: BUY if next_day_return >= +0.005; SELL if <= -0.005; HOLD otherwise

Label Rules
-----------
- BUY  if next_day_return >= +0.5% (0.005)
- SELL if next_day_return <= -0.5% (-0.005)
- HOLD otherwise

Installation
------------
Install required packages (Python 3.9+ recommended):

pip install yfinance pandas numpy ta requests vaderSentiment

Notes
-----
- News sentiment is optional. Provide --news-api-key to enable NewsAPI usage. Without it, sentiment columns
  will be absent or NaN. The script skips news calls if the key is not provided.
- The script gracefully handles missing/non-trading days by computing indicators on available trading dates.

Extensibility Hooks
-------------------
- Additional data providers (Alpha Vantage / Finnhub / IEX) can be added in the fetch functions.
  See the placeholder functions and comments near fetch_news_sentiment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Third-party libraries
# yfinance for market data
import yfinance as yf

# ta for indicators
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator, MACD

# Optional sentiment
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional modular ingestion
try:
    from news_ingestion import fetch_newsapi_headlines, aggregate_daily_sentiment_vader
except Exception:
    fetch_newsapi_headlines = None  # type: ignore
    aggregate_daily_sentiment_vader = None  # type: ignore

# Load .env if available for API keys
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# -----------------------------
# Configuration & Logging Setup
# -----------------------------

DEFAULT_OUTPUT_DIR = os.path.join("datasets")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Utility Data Structures
# -----------------------------

@dataclass
class SentimentResult:
    date: dt.date
    avg_compound: float
    headline_count: int


# -----------------------------
# Input Parsing
# -----------------------------

def parse_tickers(tickers_arg: str | List[str]) -> List[str]:
    """Parse tickers from either:
    - A Python list of strings (if already a list)
    - A path to a CSV file containing a column named 'ticker' OR a single-column of tickers
    - A comma-separated string like "AAPL,MSFT,GOOGL"
    """
    if isinstance(tickers_arg, list):
        return [t.strip().upper() for t in tickers_arg if str(t).strip()]

    arg = str(tickers_arg).strip()
    if not arg:
        return []

    # CSV file path
    if os.path.exists(arg) and arg.lower().endswith(".csv"):
        try:
            df = pd.read_csv(arg)
            if "ticker" in df.columns:
                tickers = df["ticker"].dropna().astype(str).str.upper().str.strip().tolist()
            else:
                # Single column CSV
                first_col = df.columns[0]
                tickers = df[first_col].dropna().astype(str).str.upper().str.strip().tolist()
            return [t for t in tickers if t]
        except Exception as exc:
            logger.error("Failed to read tickers CSV '%s': %s", arg, exc)
            return []

    # Comma-separated list
    return [t.strip().upper() for t in arg.split(",") if t.strip()]


# -----------------------------
# Data Fetching
# -----------------------------

def fetch_ohlcv_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV using yfinance. Returns a DataFrame with normalized column names.

    Columns normalized to lowercase with underscores: open, high, low, close, adj_close, volume.
    Index is DatetimeIndex of trading days. Non-trading days are naturally excluded by the data source.
    """
    logger.info("Fetching OHLCV for %s from %s to %s", ticker, start_date, end_date)
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception as exc:
        logger.exception("yfinance download failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if data is None or data.empty:
        logger.warning("No data returned for %s", ticker)
        return pd.DataFrame()

    df = data.copy()
    # Normalize column names - handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns (e.g., ('Close', 'AAPL') -> 'close')
        df.columns = [col[0].lower().replace(" ", "_") for col in df.columns]
    else:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Ensure expected columns exist
    expected = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = expected.difference(set(df.columns))
    # Some tickers may not have adj_close; derive if possible, else set NaN
    if "adj_close" in missing and "adj_close" not in df.columns:
        df["adj_close"] = np.nan
        missing.discard("adj_close")
    if missing:
        logger.warning("%s missing columns: %s", ticker, ", ".join(sorted(missing)))

    return df


# -----------------------------
# Technical Indicators & Volume Features
# -----------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and volume features on a yfinance OHLCV DataFrame.

    Required columns: open, high, low, close, volume
    """
    if df.empty:
        return df

    df = df.copy()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # SMAs
    df["sma_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["sma_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(close=close, window=200).sma_indicator()

    # EMAs
    df["ema_12"] = EMAIndicator(close=close, window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(close=close, window=26).ema_indicator()

    # MACD
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    # Bollinger Bands (20-day)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_middle_20"] = bb.bollinger_mavg()
    df["bb_upper_20"] = bb.bollinger_hband()
    df["bb_lower_20"] = bb.bollinger_lband()
    # Bandwidth: (upper - lower) / middle
    with np.errstate(divide="ignore", invalid="ignore"):
        df["bb_bandwidth_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["bb_middle_20"]

    # RSI (14) - use smaller window if not enough data
    rsi_window = min(14, len(df))
    df["rsi_14"] = RSIIndicator(close=close, window=rsi_window).rsi()

    # ATR (14) - use smaller window if not enough data
    atr_window = min(14, len(df))
    df["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=atr_window).average_true_range()

    # Daily returns and rolling volatility (21-day)
    df["daily_return"] = close.pct_change()
    df["rolling_volatility_21"] = df["daily_return"].rolling(window=21, min_periods=10).std()

    # Volume features
    df["volume_change"] = volume.pct_change()
    df["avg_volume_21"] = volume.rolling(window=21, min_periods=10).mean()

    return df


# -----------------------------
# Optional News Sentiment (NewsAPI + VADER)
# -----------------------------

def _date_range(start_date: dt.date, end_date: dt.date) -> Iterable[dt.date]:
    current = start_date
    while current <= end_date:
        yield current
        current = current + dt.timedelta(days=1)


def fetch_news_sentiment(
    ticker: str,
    start_date: str,
    end_date: str,
    news_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily news headlines via NewsAPI and compute VADER compound sentiment averages per date.

    - If news_api_key is None or empty, returns empty DataFrame.
    - Aggregates by date (UTC). Columns: date, sentiment_avg_compound, headline_count
    - This function is written to be robust but simple; it queries per day to respect NewsAPI parameters.

    Hooks to extend: Replace/augment this with Alpha Vantage / Finnhub / IEX endpoints as needed.
    """
    # Prefer modular ingestion if available
    if fetch_newsapi_headlines is not None and aggregate_daily_sentiment_vader is not None and news_api_key:
        try:
            df_news = fetch_newsapi_headlines(query=ticker, start_date=start_date, end_date=end_date, api_key=news_api_key)
            if df_news is not None and not df_news.empty:
                daily = aggregate_daily_sentiment_vader(df_news)
                logger.info("Successfully fetched %d headlines for %s, aggregated to %d days", len(df_news), ticker, len(daily))
                return daily
            else:
                logger.info("No headlines found for %s in date range %s to %s", ticker, start_date, end_date)
                return pd.DataFrame(columns=["date", "sentiment_avg_compound", "headline_count"])
        except Exception as exc:
            logger.warning("Modular news ingestion failed for %s: %s; falling back to inline method.", ticker, exc)

    # Inline simple per-day NewsAPI + VADER fallback
    if not news_api_key:
        logger.info("No NewsAPI key provided, skipping news sentiment for %s", ticker)
        return pd.DataFrame(columns=["date", "sentiment_avg_compound", "headline_count"])  # empty

    logger.info("Fetching news sentiment for %s", ticker)
    analyzer = SentimentIntensityAnalyzer()

    try:
        start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()

    rows: List[SentimentResult] = []
    base_url = "https://newsapi.org/v2/everything"

    for day in _date_range(start_dt, end_dt):
        from_iso = dt.datetime.combine(day, dt.time.min).isoformat() + "Z"
        to_iso = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min).isoformat() + "Z"
        # Allow overriding sort/search/pageSize from env to match user's curl style
        sort_by = os.environ.get("NEWSAPI_SORT_BY", "popularity")
        search_in = os.environ.get("NEWSAPI_SEARCH_IN", "title,description")
        try:
            page_size = max(1, min(100, int(os.environ.get("NEWSAPI_PAGE_SIZE", "100"))))
        except Exception:
            page_size = 100

        params = {
            "q": ticker,
            "from": day.strftime("%Y-%m-%d"),
            "to": (day + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
            "language": "en",
            "searchIn": search_in,
            "sortBy": sort_by,
            "pageSize": page_size,
            "apiKey": news_api_key,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            if resp.status_code != 200:
                if os.environ.get("NEWSAPI_DEBUG") == "1":
                    try:
                        print(f"NewsAPI error status={resp.status_code} body={resp.text[:500]}")
                    except Exception:
                        pass
                continue
            payload = resp.json()
        except Exception as exc:
            if os.environ.get("NEWSAPI_DEBUG") == "1":
                try:
                    print(f"NewsAPI request exception: {exc}")
                except Exception:
                    pass
            continue

        if isinstance(payload, dict) and payload.get("status") == "error":
            if os.environ.get("NEWSAPI_DEBUG") == "1":
                try:
                    print(f"NewsAPI payload error: code={payload.get('code')} message={payload.get('message')}")
                except Exception:
                    pass
            continue

        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        if not articles:
            continue

        compounds: List[float] = []
        for art in articles:
            title = (art or {}).get("title") or ""
            description = (art or {}).get("description") or ""
            text = f"{title}. {description}".strip()
            if not text:
                continue
            try:
                score = analyzer.polarity_scores(text).get("compound", 0.0)
                compounds.append(float(score))
            except Exception:
                pass

        if compounds:
            rows.append(SentimentResult(date=day, avg_compound=float(np.mean(compounds)), headline_count=len(compounds)))

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment_avg_compound", "headline_count"])  # empty

    return pd.DataFrame({
        "date": [r.date for r in rows],
        "sentiment_avg_compound": [r.avg_compound for r in rows],
        "headline_count": [r.headline_count for r in rows],
    })


# -----------------------------
# Label Creation
# -----------------------------

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create forward-looking target and 3-class labels.

    next_day_return = (close_{t+1} - close_t) / close_t
    Labels:
      - BUY if next_day_return >= +0.005
      - SELL if next_day_return <= -0.005
      - HOLD otherwise
    """
    if df.empty:
        return df

    df = df.copy()
    df["next_day_return"] = df["close"].shift(-1) / df["close"] - 1.0

    def _labeler(x: float) -> str:
        if pd.isna(x):
            return np.nan  # will be dropped later
        if x >= 0.005:
            return "BUY"
        if x <= -0.005:
            return "SELL"
        return "HOLD"

    df["label"] = df["next_day_return"].apply(_labeler)
    # Drop last row where next_day_return is NaN due to shift(-1)
    df = df.dropna(subset=["next_day_return"])  # label may still include HOLD/BUY/SELL
    return df


# -----------------------------
# Save Helpers
# -----------------------------

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    try:
        df.to_csv(output_path, index=False)
        logger.info("Saved CSV: %s (rows=%d)", output_path, len(df))
    except Exception as exc:
        logger.exception("Failed to save CSV '%s': %s", output_path, exc)


# -----------------------------
# Orchestration per Ticker
# -----------------------------

def build_dataset_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    news_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Build a dataset DataFrame for a single ticker with indicators, optional sentiment, and labels."""
    ohlcv = fetch_ohlcv_yfinance(ticker=ticker, start_date=start_date, end_date=end_date)
    if ohlcv.empty:
        logger.warning("Skipping %s due to empty OHLCV.", ticker)
        return pd.DataFrame()

    features = compute_indicators(ohlcv)

    # Reset index to join on date, then re-index later
    features = features.reset_index().rename(columns={"index": "date"})
    if "Date" in features.columns:
        features = features.rename(columns={"Date": "date"})
    if not np.issubdtype(features["date"].dtype, np.datetime64):
        features["date"] = pd.to_datetime(features["date"])  # ensure datetime

    # Optional sentiment
    # Prefer param key; else fallback to env NEWSAPI_KEY
    key = news_api_key or os.environ.get("NEWSAPI_KEY")
    sentiment_df = fetch_news_sentiment(ticker, start_date, end_date, news_api_key=key)
    if not sentiment_df.empty:
        sentiment_df = sentiment_df.copy()
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        features = features.merge(sentiment_df, on="date", how="left")
    else:
        # If sentiment is not available, still include columns as NaN for schema consistency
        features["sentiment_avg_compound"] = np.nan
        features["headline_count"] = np.nan

    # Create labels
    features = create_labels(features)

    # Ticker column and clean-up
    features.insert(0, "ticker", ticker)
    # Rename OHLCV columns to consistent lowercase names if needed
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    features = features.rename(columns=rename_map)

    # Ensure expected base columns exist (if any were missing from source)
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in features.columns:
            features[col] = np.nan

    # Final ordering suggestion (not mandatory)
    preferred_order = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        # Indicators
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_middle_20",
        "bb_upper_20",
        "bb_lower_20",
        "bb_bandwidth_20",
        "rsi_14",
        "atr_14",
        "daily_return",
        "rolling_volatility_21",
        "volume_change",
        "avg_volume_21",
        # Sentiment (optional)
        "sentiment_avg_compound",
        "headline_count",
        # Targets
        "next_day_return",
        "label",
    ]

    # Ensure date is first and sorted ascending
    if "date" in features.columns:
        features = features.sort_values("date").reset_index(drop=True)

    # Reorder columns while keeping any extras
    existing_cols = [c for c in preferred_order if c in features.columns]
    extra_cols = [c for c in features.columns if c not in existing_cols]
    features = features[existing_cols + extra_cols]

    return features


# -----------------------------
# CLI / Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build supervised dataset for 3-class stock model.")
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help=(
            "Comma-separated list of tickers OR path to CSV with a 'ticker' column. "
            "Example: --tickers AAPL,MSFT or --tickers tickers.csv"
        ),
    )
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--news-api-key",
        type=str,
        default=None,
        help="Optional NewsAPI key to enable daily news sentiment features.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    tickers = parse_tickers(args.tickers)
    if not tickers:
        logger.error("No tickers parsed. Provide --tickers as CSV path or comma-separated list.")
        return

    ensure_output_dir(args.output_dir)

    all_frames: List[pd.DataFrame] = []
    for t in tickers:
        try:
            df_t = build_dataset_for_ticker(
                ticker=t,
                start_date=args.start_date,
                end_date=args.end_date,
                news_api_key=args.news_api_key,
            )
        except Exception as exc:
            logger.exception("Failed building dataset for %s: %s", t, exc)
            df_t = pd.DataFrame()

        if df_t.empty:
            logger.warning("No rows generated for %s", t)
            continue

        # Save per-ticker
        per_ticker_path = os.path.join(args.output_dir, f"{t}_dataset.csv")
        save_csv(df_t, per_ticker_path)

        all_frames.append(df_t)

    if all_frames:
        combined = pd.concat(all_frames, axis=0, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "all_tickers_dataset.csv")
        save_csv(combined, combined_path)
    else:
        logger.warning("No combined dataset created because no per-ticker data was available.")


if __name__ == "__main__":
    # Example usage:
    # python build_stock_dataset.py --tickers AAPL,MSFT,GOOGL --start-date 2020-01-01 --end-date 2023-12-31 \
    #   --news-api-key YOUR_NEWSAPI_KEY --output-dir datasets
    #
    # Or with a CSV file containing a 'ticker' column:
    # python build_stock_dataset.py --tickers tickers.csv --start-date 2020-01-01 --end-date 2023-12-31
    #
    # Note: News sentiment is optional. If --news-api-key is omitted, sentiment features are skipped.
    main()


