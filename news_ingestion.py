"""
news_ingestion.py
-----------------
Lightweight ingestion utilities for headlines from NewsAPI and RSS feeds.

Goals for v1:
- Provide simple functions to fetch headlines for a ticker and date range
- Normalize to a standard schema
- Dedupe by URL/content hash

Notes:
- For NewsAPI, you need an API key. Respect their terms and rate limits.
- RSS support is minimal: we accept pre-known feed URLs (e.g., IR/press).
"""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Iterable, List, Optional
import os
import argparse

import pandas as pd
import requests


def _hash_content(url: str | None, title: str | None, description: str | None) -> str:
    h = hashlib.sha256()
    h.update((url or "").encode("utf-8", errors="ignore"))
    h.update((title or "").encode("utf-8", errors="ignore"))
    h.update((description or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _newsapi_defaults_from_env() -> dict:
    sort_by = os.environ.get("NEWSAPI_SORT_BY", "popularity")  # popularity | publishedAt | relevancy
    search_in = os.environ.get("NEWSAPI_SEARCH_IN", "title,description")
    page_size_env = os.environ.get("NEWSAPI_PAGE_SIZE", "100")
    try:
        page_size = max(1, min(100, int(page_size_env)))
    except Exception:
        page_size = 100
    return {"sortBy": sort_by, "searchIn": search_in, "pageSize": page_size}


def fetch_newsapi_headlines(
    query: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str],
    language: str = "en",
) -> pd.DataFrame:
    """Fetch headlines via NewsAPI 'everything' endpoint for a query (e.g., ticker or company name).

    Returns normalized DataFrame columns:
      - published_at (datetime64[ns, UTC])
      - source
      - title
      - description
      - url
      - hash
    """
    if not api_key:
        return pd.DataFrame(columns=["published_at", "source", "title", "description", "url", "hash"])  # empty

    base_url = "https://newsapi.org/v2/everything"
    # Normalize dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # NewsAPI supports ISO8601 with Z; we will paginate by date windows if needed (v1: single call)
    defaults = _newsapi_defaults_from_env()
    params = {
        "q": query,
        "from": start_dt.strftime("%Y-%m-%d"),  # accept plain date per curl style
        "to": end_dt.strftime("%Y-%m-%d"),
        "language": language,
        "searchIn": defaults["searchIn"],
        "sortBy": defaults["sortBy"],
        "pageSize": defaults["pageSize"],
        "apiKey": api_key,
    }
    try:
        resp = requests.get(base_url, params=params, timeout=20)
        if resp.status_code != 200:
            if os.environ.get("NEWSAPI_DEBUG") == "1":
                print(f"NewsAPI error status={resp.status_code} body={resp.text[:500]}")
            return pd.DataFrame(columns=["published_at", "source", "title", "description", "url", "hash"])
        payload = resp.json()
    except Exception:
        return pd.DataFrame(columns=["published_at", "source", "title", "description", "url", "hash"])

    articles = payload.get("articles") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and payload.get("status") == "error":
        if os.environ.get("NEWSAPI_DEBUG") == "1":
            print(f"NewsAPI payload error: code={payload.get('code')} message={payload.get('message')}")
        return pd.DataFrame(columns=["published_at", "source", "title", "description", "url", "hash"])
    if not isinstance(articles, list) or not articles:
        return pd.DataFrame(columns=["published_at", "source", "title", "description", "url", "hash"])

    rows = []
    for art in articles:
        source_name = ((art or {}).get("source") or {}).get("name")
        title = (art or {}).get("title")
        description = (art or {}).get("description")
        url = (art or {}).get("url")
        published_at = pd.to_datetime((art or {}).get("publishedAt"), errors="coerce", utc=True)
        rows.append(
            {
                "published_at": published_at,
                "source": source_name,
                "title": title,
                "description": description,
                "url": url,
                "hash": _hash_content(url, title, description),
            }
        )

    df = pd.DataFrame(rows)
    # Dedupe by hash, keep latest published_at
    if not df.empty:
        df = df.sort_values("published_at").drop_duplicates(subset=["hash"], keep="last").reset_index(drop=True)
    return df


def normalize_rss_items(items: List[dict]) -> pd.DataFrame:
    """Normalize a list of RSS-like items to the standard schema.

    Each item can contain keys: published_at (datetime or str), title, summary/description, link, source.
    """
    rows = []
    for it in items or []:
        title = it.get("title")
        description = it.get("summary") or it.get("description")
        url = it.get("link")
        published_at = pd.to_datetime(it.get("published_at") or it.get("pubDate"), errors="coerce", utc=True)
        source = it.get("source") or it.get("feed")
        rows.append(
            {
                "published_at": published_at,
                "source": source,
                "title": title,
                "description": description,
                "url": url,
                "hash": _hash_content(url, title, description),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published_at").drop_duplicates(subset=["hash"], keep="last").reset_index(drop=True)
    return df


def aggregate_daily_sentiment_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-day average VADER compound sentiment over title+description.

    Expects columns: published_at, title, description.
    Returns columns: date, sentiment_avg_compound, headline_count
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "sentiment_avg_compound", "headline_count"])

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    df_local = df.copy()
    texts = (df_local["title"].fillna("") + ". " + df_local["description"].fillna("")).str.strip()
    scores = []
    for txt in texts:
        try:
            scores.append(float(analyzer.polarity_scores(txt).get("compound", 0.0)))
        except Exception:
            scores.append(0.0)
    df_local["compound"] = scores
    df_local["date"] = pd.to_datetime(df_local["published_at"], utc=True).dt.tz_convert("UTC").dt.date
    agg = df_local.groupby("date").agg(sentiment_avg_compound=("compound", "mean"), headline_count=("compound", "size")).reset_index()
    return agg


def _cli() -> None:
    # Load .env if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Test NewsAPI ingestion and daily sentiment aggregation")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker or query string for NewsAPI")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--api-key", type=str, default=os.environ.get("NEWSAPI_KEY"), help="NewsAPI key (or set NEWSAPI_KEY env or .env)")
    args = parser.parse_args()

    df_news = fetch_newsapi_headlines(query=args.ticker, start_date=args.start, end_date=args.end, api_key=args.api_key)
    if df_news.empty:
        print("No headlines fetched. Check API key, dates, or query.")
        return
    print(f"Fetched {len(df_news)} headlines. Sample:")
    print(df_news.head(5)[["published_at", "source", "title"]])

    daily = aggregate_daily_sentiment_vader(df_news)
    if daily.empty:
        print("No sentiment aggregated.")
        return
    print("\nDaily sentiment (date, avg_compound, count):")
    print(daily.to_string(index=False))


if __name__ == "__main__":
    _cli()


