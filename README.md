# AI Stock Recommendation App 📈

A comprehensive Streamlit application that analyzes stock data using technical indicators and news sentiment to generate BUY/HOLD/SELL recommendations with confidence scores and interactive visualizations.

## 🚀 Features

### Core Functionality
- **Real-time Stock Analysis**: Fetch live data using yfinance
- **Technical Indicators**: Comprehensive technical analysis with 15+ indicators
- **News Sentiment Integration**: Optional sentiment analysis using NewsAPI + VADER
- **Interactive Charts**: Plotly-based visualizations with price, volume, RSI, MACD
- **Recommendation Engine**: Rule-based scoring system with confidence levels
- **Dataset Export**: Download computed features and labels for ML training
- **Executive Briefs**: Automated summary generation

### Supported Markets
- **US Stocks**: AAPL, MSFT, GOOGL, etc.
- **Indian Stocks**: RELIANCE.NS, TCS.NS, HDFC.NS, etc. (NSE format)
- **International**: Any ticker supported by Yahoo Finance

## 📊 Technical Indicators

### Trend Indicators
- **SMA**: 10, 20, 50, 200-day Simple Moving Averages
- **EMA**: 12, 26-day Exponential Moving Averages
- **MACD**: Moving Average Convergence Divergence with Signal and Histogram

### Momentum Indicators
- **RSI(14)**: Relative Strength Index with overbought/oversold levels
- **ATR(14)**: Average True Range for volatility measurement

### Volume & Volatility
- **Bollinger Bands**: 20-day with 2 standard deviations
- **Volume Analysis**: 21-day average and volume spikes
- **Daily Returns**: Price change calculations
- **Volatility**: 21-day rolling standard deviation

## 🎯 Recommendation System

### Scoring Rules (Hybrid Approach)
- **Price > MA200**: +2 points (bullish trend)
- **Price > MA50**: +1 point (short-term bullish)
- **Golden Cross (MA50 > MA200)**: +3 points (strong bullish signal)
- **RSI > 70**: -3 points (overbought condition)
- **RSI < 30**: +2 points (oversold opportunity)
- **MACD > Signal**: +1 point (momentum confirmation)
- **Volume Spike**: +1 point (unusual activity)
- **Positive Daily Return**: +1 point (immediate momentum)
- **News Sentiment > 0.3**: +1 point (positive news)
- **News Sentiment < -0.3**: -1 point (negative news)

### Confidence Mapping
- **Score ≥ 3**: BUY (60-95% confidence)
- **Score -2 to 2**: HOLD (40-70% confidence)
- **Score < -2**: SELL (60-95% confidence)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd aibf

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Optional: News Sentiment Setup
To enable news sentiment analysis:

1. Get a free API key from [NewsAPI](https://newsapi.org/)
2. Create a `.env` file in the project root:
```bash
NEWSAPI_KEY=your_api_key_here
```

## 📱 How to Use

### 1. Launch the App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### 2. Configure Analysis
- **Ticker Symbol**: Enter stock symbol (e.g., AAPL, RELIANCE.NS)
- **Date Range**: Select start and end dates (default: last 365 days)
- **News Sentiment**: Toggle to include/exclude news analysis

### 3. Get Recommendations
- Click "Get Recommendation" to analyze the stock
- View interactive charts and technical indicators
- Review the BUY/HOLD/SELL recommendation with confidence score
- Check the explainability panel for rule breakdown

### 4. Export Data
- Download the computed dataset as CSV
- Generate executive briefs for reports

## 🔧 Advanced Usage

### Building Datasets
```bash
# Build dataset for multiple tickers
python build_stock_dataset.py \
  --tickers AAPL,MSFT,GOOGL,RELIANCE.NS,TCS.NS \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --news-api-key YOUR_KEY

# Build from CSV file
python build_stock_dataset.py \
  --tickers tickers.csv \
  --start-date 2020-01-01 \
  --end-date 2023-12-31
```

### Fetching News Data
```bash
# Fetch news for a specific ticker
python fetch_news.py \
  --query AAPL \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --out aapl_news.csv
```

## 📁 Project Structure

```
aibf/
├── app.py                 # Main Streamlit application
├── stock_logic.py         # Core recommendation engine
├── build_stock_dataset.py # Dataset generation script
├── fetch_news.py         # News data fetching utility
├── news_ingestion.py     # News API integration
├── entity_linking.py     # Ticker mapping utilities
├── test_stock_logic.py   # Unit tests
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── datasets/            # Generated datasets
│   ├── AAPL_dataset.csv
│   └── all_tickers_dataset.csv
└── venv/               # Virtual environment
```

## 🧪 Testing

```bash
# Run unit tests
python test_stock_logic.py

# Test news fetching
python news_ingestion.py \
  --ticker AAPL \
  --start 2024-01-01 \
  --end 2024-01-07
```

## 📈 Example Outputs

### Recommendation Example
```
🟢 BUY (85%)
Close: $150.25 (+2.3%)
RSI: 65.2 | MACD: 0.45
```

### Explainability Panel
```
✅ Price > MA200 (contrib +2) — Price 150.25 vs MA200 145.30
✅ Golden Cross (contrib +3) — MA50 148.20 vs MA200 145.30
✅ MACD > Signal (contrib +1) — MACD 0.45 vs Signal 0.20
✅ Volume Spike (contrib +1) — Vol 2.5M vs mean21 1.8M
```

## 🔍 Troubleshooting

### Common Issues

1. **Empty Charts**: Check if ticker symbol is valid and data is available
2. **News API Errors**: Verify API key and rate limits
3. **Import Errors**: Ensure all dependencies are installed
4. **Date Range Issues**: Use wider date ranges for long-term indicators

### Debug Mode
The app includes debugging information to help identify data issues.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **yfinance**: Stock data provider
- **NewsAPI**: News data provider
- **Streamlit**: Web application framework
- **Plotly**: Interactive charting library
- **VADER**: Sentiment analysis

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the example outputs
- Test with known tickers (AAPL, MSFT, GOOGL)

---

**⚠️ Disclaimer**: This application is for educational and research purposes only. Stock recommendations should not be considered as financial advice. Always consult with qualified financial advisors before making investment decisions.


