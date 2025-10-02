"""
stock_logic.py
---------------
Helper module for AI Stock Recommendation Streamlit app.

Functions:
- fetch_price_data(ticker, start_date, end_date): fetches daily OHLCV with yfinance
- compute_indicators(df): computes technical indicators/volume features
- get_recommendation(df_row): hybrid rules + score -> (label, confidence, fired_rules)

Notes:
- No external paid APIs are called here. Hooks can be added for Alpha Vantage / Finnhub / IEX.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV using yfinance. Returns DataFrame indexed by date with standard columns.

    Columns: Open, High, Low, Close, Adj Close, Volume (as returned by yfinance)
    """
    if not ticker:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators on a yfinance OHLCV DataFrame.

    Adds: sma_10/20/50/200, ema_12/26, macd, macd_signal, macd_hist,
          bb_middle_20/upper/lower, atr_14, rsi_14,
          daily_return, vol_21 (rolling std), vol_mean_21 (volume mean 21)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # Ensure 1D Series for indicators (yfinance can sometimes produce (n,1) DataFrames)
    def _ensure_series(s, name: str):
        if isinstance(s, pd.DataFrame):
            s = s.squeeze("columns")
        if not isinstance(s, pd.Series):
            s = pd.Series(np.asarray(s).ravel(), index=out.index, name=name)
        s = pd.to_numeric(s, errors="coerce").astype(float)
        return s

    close = _ensure_series(out.get("Close"), "Close")
    high = _ensure_series(out.get("High"), "High")
    low = _ensure_series(out.get("Low"), "Low")
    volume = _ensure_series(out.get("Volume"), "Volume")

    # SMAs
    out["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    out["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    out["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    out["sma_200"] = SMAIndicator(close, window=200).sma_indicator()

    # EMAs
    out["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    out["ema_26"] = EMAIndicator(close, window=26).ema_indicator()

    # MACD
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()
    out["macd_hist"] = macd_ind.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    out["bb_middle_20"] = bb.bollinger_mavg()
    out["bb_upper_20"] = bb.bollinger_hband()
    out["bb_lower_20"] = bb.bollinger_lband()

    # ATR
    out["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # RSI
    out["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

    # Returns & volatility
    out["daily_return"] = close.pct_change()
    out["vol_21"] = out["daily_return"].rolling(21, min_periods=10).std()

    # Volume mean 21
    out["vol_mean_21"] = volume.rolling(21, min_periods=10).mean()

    return out


def _map_score_to_label_and_confidence(score: int, sentiment: float | None = None) -> Tuple[str, int]:
    """Map total score to label and confidence in range 0-100.

    Sentiment (if provided) nudges confidence by +/- up to 5 points.
    """
    conf = 50
    if score >= 3:
        # Map 3..10 -> 60..95 (clamped)
        conf = min(95, 60 + (score - 3) * 5)
        label = "BUY"
    elif score < -2:
        # Map -10..-3 -> 95..60 (reverse)
        conf = min(95, 60 + (abs(score) - 3) * 5)
        label = "SELL"
    else:
        # HOLD - wider middle band
        conf = 40 + int((score + 2) * 5)  # -2..2 -> 40..65
        conf = min(max(conf, 40), 70)
        label = "HOLD"

    if sentiment is not None and not np.isnan(sentiment):
        conf = int(np.clip(conf + 5 * np.sign(sentiment), 0, 100))

    return label, int(conf)


def get_recommendation(df_row: pd.Series) -> Tuple[str, int, Dict[str, dict]]:
    """Hybrid rules + score.

    Returns: (label, confidence, fired_rules_dict)
    fired_rules_dict: { rule_name: {"fired": bool, "contribution": int, "explanation": str} }
    """
    fired = {}
    score = 0

    def add_rule(name: str, fired_bool: bool, points: int, explanation: str):
        nonlocal score
        fired[name] = {"fired": bool(fired_bool), "contribution": points if fired_bool else 0, "explanation": explanation}
        if fired_bool:
            score += points

    close = float(df_row.get("Close", np.nan))
    ma50 = float(df_row.get("sma_50", np.nan))
    ma200 = float(df_row.get("sma_200", np.nan))
    rsi = float(df_row.get("rsi_14", np.nan))
    macd_v = float(df_row.get("macd", np.nan))
    macd_sig = float(df_row.get("macd_signal", np.nan))
    vol = float(df_row.get("Volume", np.nan))
    vol_mean_21 = float(df_row.get("vol_mean_21", np.nan))
    ret1d = float(df_row.get("daily_return", np.nan))
    sentiment = df_row.get("sentiment_avg_compound", np.nan)
    try:
        sentiment = float(sentiment)
    except Exception:
        sentiment = np.nan

    # Rules
    add_rule("Price > MA200", (close > ma200) if not np.isnan(ma200) else False, 2, f"Price {close:.2f} vs MA200 {ma200:.2f}")
    add_rule("Price > MA50", (close > ma50) if not np.isnan(ma50) else False, 1, f"Price {close:.2f} vs MA50 {ma50:.2f}")
    add_rule("MA50 > MA200 (golden)", (ma50 > ma200) if not (np.isnan(ma50) or np.isnan(ma200)) else False, 3, f"MA50 {ma50:.2f} vs MA200 {ma200:.2f}")
    add_rule("RSI > 70 (overbought)", (rsi > 70) if not np.isnan(rsi) else False, -3, f"RSI {rsi:.2f}")
    add_rule("RSI < 30 (oversold)", (rsi < 30) if not np.isnan(rsi) else False, 2, f"RSI {rsi:.2f}")
    add_rule("MACD > Signal (momentum)", (macd_v > macd_sig) if not (np.isnan(macd_v) or np.isnan(macd_sig)) else False, 1, f"MACD {macd_v:.4f} vs Signal {macd_sig:.4f}")
    add_rule(
        "High Volume Spike",
        (vol > vol_mean_21 * 1.5) if not (np.isnan(vol) or np.isnan(vol_mean_21)) else False,
        1,
        f"Vol {vol:.0f} vs mean21 {vol_mean_21:.0f}",
    )
    add_rule("Positive Return 1d", (ret1d > 0) if not np.isnan(ret1d) else False, 1, f"Return1d {ret1d:.4f}")

    # Optional: sentiment rule
    if not np.isnan(sentiment):
        add_rule("Sentiment > 0.3", sentiment > 0.3, 1, f"Sentiment {sentiment:.3f}")
        add_rule("Sentiment < -0.3", sentiment < -0.3, -1, f"Sentiment {sentiment:.3f}")

    label, conf = _map_score_to_label_and_confidence(score, sentiment)
    return label, conf, fired


def build_feature_label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Utility: from indicator-enriched df, compute next_day_return and label per the same rules used elsewhere.
    This supports the Download Dataset button in the app.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out.reset_index().rename(columns={"Date": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["next_day_return"] = out["Close"].shift(-1) / out["Close"] - 1.0

    labels = []
    confs = []
    rules = []
    for _, row in out.iterrows():
        label, conf, fired = get_recommendation(row)
        labels.append(label)
        confs.append(conf)
        rules.append(fired)
    out["label"] = labels
    out["confidence"] = confs
    out["rules_json"] = [str(r) for r in rules]

    # Drop last row where forward return is NaN
    out = out.dropna(subset=["next_day_return"]).reset_index(drop=True)
    return out


def generate_brief(df: pd.DataFrame, label: str, confidence: int) -> str:
    """Produce a short 3–4 sentence executive brief using simple templates.
    Avoids any external LLM calls.
    """
    if df is None or df.empty:
        return "Insufficient data to generate a brief."

    latest = df.iloc[-1]
    close = latest.get("Close", np.nan)
    ma50 = latest.get("sma_50", np.nan)
    ma200 = latest.get("sma_200", np.nan)
    rsi = latest.get("rsi_14", np.nan)
    macd_v = latest.get("macd", np.nan)
    macd_sig = latest.get("macd_signal", np.nan)
    vol = latest.get("Volume", np.nan)
    vol_mean_21 = latest.get("vol_mean_21", np.nan)
    sentiment = latest.get("sentiment_avg_compound", np.nan)

    parts = []
    parts.append(f"Latest close is {close:.2f}. 50-day MA at {ma50:.2f}, 200-day MA at {ma200:.2f}.")
    if not (np.isnan(macd_v) or np.isnan(macd_sig)):
        parts.append(f"MACD is {'above' if macd_v > macd_sig else 'below'} signal, suggesting momentum {'up' if macd_v > macd_sig else 'down'}.")
    if not np.isnan(rsi):
        if rsi > 70:
            parts.append(f"RSI at {rsi:.1f} indicates overbought conditions.")
        elif rsi < 30:
            parts.append(f"RSI at {rsi:.1f} indicates oversold conditions.")
        else:
            parts.append(f"RSI at {rsi:.1f} is neutral.")
    if not (np.isnan(vol) or np.isnan(vol_mean_21)):
        spike = vol > vol_mean_21 * 1.5
        parts.append(f"Volume is {'elevated' if spike else 'normal'} relative to its 21-day average.")
    if not np.isnan(sentiment):
        parts.append(f"News sentiment is {('positive' if sentiment > 0.1 else ('negative' if sentiment < -0.1 else 'mixed'))}.")
    parts.append(f"Overall recommendation: {label} with confidence {confidence}.")

    return " " .join(parts)


def load_sample_tickers():
    return ["AAPL", "MSFT", "GOOGL"]


