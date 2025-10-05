"""
fetch_news.py
--------------
Standalone script to fetch headlines from NewsAPI for a query (e.g., ticker or company name).

Usage (after setting NEWSAPI_KEY in .env):
  python fetch_news.py --query AAPL --start 2025-09-25 --end 2025-10-02 --out aapl_news.csv

If --out is omitted, a preview is printed to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
from typing import Dict, Iterable


def main() -> None:
    # Load .env if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    # Lazy import to avoid hard dependency if module not needed elsewhere
    try:
        from news_ingestion import fetch_newsapi_headlines, aggregate_daily_sentiment_vader  # type: ignore
    except Exception as exc:
        print(f"Failed to import news_ingestion: {exc}", file=sys.stderr)
        sys.exit(1)

    # Optional entity linking
    alias_index: Dict[str, set] = {}
    try:
        from entity_linking import build_alias_index, find_tickers_in_text  # type: ignore
    except Exception:
        build_alias_index = None  # type: ignore
        find_tickers_in_text = None  # type: ignore

    parser = argparse.ArgumentParser(description="Fetch headlines from NewsAPI and print or save to CSV")
    parser.add_argument("--query", required=True, type=str, help="Search query (e.g., ticker or company name)")
    parser.add_argument("--start", required=True, type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--api-key", type=str, default=os.environ.get("NEWSAPI_KEY"), help="Override API key (defaults to env NEWSAPI_KEY)")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save CSV of normalized headlines")
    parser.add_argument("--count-only", action="store_true", help="Only print the number of headlines found and exit")
    parser.add_argument("--aliases-csv", type=str, default=None, help="Optional CSV with columns: ticker,alias for entity linking")
    args = parser.parse_args()

    if not args.api_key:
        print("NEWSAPI_KEY not set. Put it in .env or pass --api-key.", file=sys.stderr)
        sys.exit(2)

    df = fetch_newsapi_headlines(query=args.query, start_date=args.start, end_date=args.end, api_key=args.api_key)
    if df is None or df.empty:
        print("No headlines found (or request failed).", file=sys.stderr)
        sys.exit(3)

    # If aliases CSV provided and entity linking available, build index
    if args.aliases_csv and build_alias_index and find_tickers_in_text:
        try:
            alias_df = pd.read_csv(args.aliases_csv)
            if not set(["ticker", "alias"]).issubset(alias_df.columns):
                print("aliases CSV must have columns: ticker,alias", file=sys.stderr)
            else:
                mapping: Dict[str, set] = {}
                for _, row in alias_df.iterrows():
                    t = str(row["ticker"]).strip().upper()
                    a = str(row["alias"]).strip()
                    if not t or not a:
                        continue
                    mapping.setdefault(t, set()).add(a)
                alias_index = build_alias_index({k: list(v) for k, v in mapping.items()})
        except Exception as exc:
            print(f"Failed to load aliases CSV: {exc}", file=sys.stderr)

    # Print a compact per-article preview with detected tickers if alias index exists
    cols = [c for c in ["published_at", "source", "title", "url"] if c in df.columns]
    print("Preview (first 10 articles):")
    if alias_index and 'title' in df.columns and find_tickers_in_text:
        for _, row in df.head(10).iterrows():
            title = str(row.get("title") or "")
            desc = str(row.get("description") or "")
            text = (title + ". " + desc).strip()
            tickers = find_tickers_in_text(text, alias_index)
            pub = row.get("published_at")
            src = row.get("source")
            url = row.get("url")
            print(f"- {pub} | {src} | {title} | tickers={tickers} | {url}")
    else:
        # Fallback: no alias index, just print the preview table
        print(df.head(10)[cols].to_string(index=False))

    # Print daily sentiment summary
    try:
        daily = aggregate_daily_sentiment_vader(df)
        if not daily.empty:
            print("\nDaily sentiment (date, avg_compound, count):")
            print(daily.to_string(index=False))
    except Exception as exc:
        print(f"Sentiment aggregation failed: {exc}", file=sys.stderr)

    if args.count_only:
        print(f"Found {len(df)} headlines")
        return

    if args.out:
        try:
            df.to_csv(args.out, index=False)
            print(f"Saved {len(df)} headlines to {args.out}")
        except Exception as exc:
            print(f"Failed to save CSV: {exc}", file=sys.stderr)
            sys.exit(4)
    else:
        # Print a concise preview
        cols = [c for c in ["published_at", "source", "title", "url"] if c in df.columns]
        print(df.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()


