from stock_logic import get_recommendation
import pandas as pd


def test_buy_signal_basic():
    # Strong bullish indicators
    row = pd.Series({
        'Close': 110,
        'sma_50': 105,
        'sma_200': 100,
        'rsi_14': 55,
        'macd': 0.5,
        'macd_signal': 0.2,
        'Volume': 2000000,
        'vol_mean_21': 1000000,
        'daily_return': 0.01,
    })
    label, conf, rules = get_recommendation(row)
    assert label in ['BUY', 'HOLD', 'SELL']
    assert label == 'BUY'


def test_sell_signal_overbought():
    row = pd.Series({
        'Close': 90,
        'sma_50': 100,
        'sma_200': 110,
        'rsi_14': 75,  # overbought
        'macd': -0.1,
        'macd_signal': 0.0,
        'Volume': 500000,
        'vol_mean_21': 700000,
        'daily_return': -0.02,
    })
    label, conf, rules = get_recommendation(row)
    assert label in ['BUY', 'HOLD', 'SELL']
    assert label == 'SELL' or label == 'HOLD'  # could be borderline, but not BUY


