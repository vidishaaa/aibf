# AI Stock Recommendation App

Streamlit application that fetches stock data with `yfinance`, computes technical indicators, and generates a BUY / HOLD / SELL recommendation with an explainability panel and interactive charts.

## Features
- Price chart with MA50/MA200, Bollinger Bands, and volume
- RSI and MACD subplots
- Optional news sentiment timeline (if `sentiment_avg_compound` provided by your backend)
- Downloadable dataset (features + forward return + label + confidence)
- Executive brief generator (template-based)

## Indicators
- SMA: 10/20/50/200
- EMA: 12/26
- MACD, Signal, Histogram
- RSI(14), ATR(14)
- Daily return, 21-day volatility, 21-day average volume

## Recommendation Rules (hybrid rules + score)
- Price > MA200 → +2
- Price > MA50 → +1
- MA50 > MA200 (golden cross) → +3
- RSI > 70 → -3
- RSI < 30 → +2
- MACD > Signal → +1
- Volume > 1.5 × mean21 → +1
- Return 1d > 0 → +1
- Optional: sentiment > 0.3 → +1; sentiment < -0.3 → -1

Score mapping:
- Score ≥ 3 → BUY (confidence ~60–95)
- -2 ≤ Score < 3 → HOLD (confidence ~40–70)
- Score < -2 → SELL (confidence ~60–95)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The app does not call paid/rate-limited APIs. Provide sentiment from your backend if available.
- For long-window indicators (e.g., MA200), early rows may contain NaNs if the date range is short.


