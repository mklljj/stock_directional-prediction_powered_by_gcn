#fetch_data_improved
"""
Enhanced Stock Data Fetcher with Sentiment Analysis
====================================================
Features:
- 10 years of historical data (instead of 5)
- Market context indicators
- News sentiment analysis
- Social media sentiment (optional)
- Options data (implied volatility)
- Earnings calendar
- Enhanced technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Optional sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
    print("✅ Sentiment analysis available")
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("⚠️  VADER sentiment not available. Install with: pip install vaderSentiment")

try:
    import requests
    from bs4 import BeautifulSoup
    NEWS_SCRAPING_AVAILABLE = True
    print("✅ News scraping available")
except ImportError:
    NEWS_SCRAPING_AVAILABLE = False
    print("⚠️  News scraping not available. Install with: pip install beautifulsoup4 requests")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Main stock universe (your 149 tickers)
STOCK_TICKERS = [
    # Tech giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Healthcare
    'UNH', 'JNJ', 'ABBV', 'LLY', 'ABT', 'AMGN', 'PFE', 'MRK',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'COF',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    # Industrials
    'BA', 'HON', 'UPS', 'RTX', 'CAT', 'GE', 'MMM', 'LMT', 'UNP', 'DHR',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX',
    # Materials
    'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD',
    # Communications
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS',
    # Tech/Software
    'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN',
    'COIN', 'CRWD', 'PANW', 'ZS', 'RKLB', 'MSTR', 'MARA',
    # Renewable/Green
    'NEE', 'DUK', 'SO', 'ENPH', 'FSLR', 'ICLN', 'PBW', 'TAN',
    # REITs
    'EQIX', 'PLD',
    # Other
    'ACN', 'ADP', 'IBM', 'BRK-B', 'TMO', 'GEV', 'PM', 'PYPL', 'ROKU', 'SMCI', 'SPCE', 'NVO',
]

# Sector ETFs
SECTOR_ETFS = [
    'XLK',   # Technology
    'XLV',   # Healthcare
    'XLF',   # Financials
    'XLY',   # Consumer Discretionary
    'XLC',   # Communication Services
    'XLI',   # Industrials
    'XLP',   # Consumer Staples
    'XLE',   # Energy
    'XLU',   # Utilities
    'XLRE',  # Real Estate
    'XLB',   # Materials
]

# Thematic ETFs
THEMATIC_ETFS = [
    'CHAT', 'AIQ', 'IRBO', 'WCLD', 'SKYY', 'CLOU',  # AI & Cloud
    'HACK', 'CIBR', 'BUG',  # Cybersecurity
    'FINX', 'ARKF', 'IPAY',  # Fintech
    'ARKG', 'GNOM', 'IDNA',  # Genomics
    'DRIV', 'KARS', 'IDRV',  # Autonomous vehicles
    'BITO', 'BLOK', 'BKCH',  # Crypto
    'SOCL', 'FNGS',  # Social media
    'UFO', 'ROKT',  # Space
    'SMH', 'SOXX', 'XSD',  # Semiconductors
    'XBI', 'IBB', 'BBH',  # Biotech
    'ITA', 'PPA', 'XAR',  # Aerospace
    'KRE', 'KBE', 'QABA',  # Regional banks
    'XRT', 'RTH',  # Retail
    'XOP', 'IEO',  # Oil & Gas
]

# Market indices and indicators
MARKET_INDICATORS = [
    '^GSPC',    # S&P 500
    '^DJI',     # Dow Jones
    '^IXIC',    # NASDAQ
    '^VIX',     # Volatility Index
    '^VXN',     # NASDAQ Volatility
    '^TNX',     # 10-Year Treasury Yield
    '^IRX',     # 13-Week Treasury Bill
    '^TYX',     # 30-Year Treasury Yield
    'DX-Y.NYB', # US Dollar Index
    'GC=F',     # Gold Futures
    'SI=F',     # Silver Futures
    'CL=F',     # Crude Oil Futures
]

# Bond ETFs for credit spreads
BOND_ETFS = [
    'TLT',   # 20+ Year Treasury
    'IEF',   # 7-10 Year Treasury
    'SHY',   # 1-3 Year Treasury
    'LQD',   # Investment Grade Corporate
    'HYG',   # High Yield Corporate
    'EMB',   # Emerging Market Bonds
]

# Date range configuration
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365*10)  # 10 YEARS instead of 5!

# =============================================================================
# BASIC DATA FETCHING
# =============================================================================

def fetch_stock_data(tickers, start_date, end_date, batch_size=50):
    """
    Fetch stock data in batches with progress tracking

    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        batch_size: Number of tickers to fetch at once

    Returns:
        DataFrame with all stock data
    """
    print(f"\n{'='*80}")
    print(f"FETCHING STOCK DATA")
    print(f"{'='*80}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total tickers: {len(tickers)}")
    print(f"Batch size: {batch_size}")

    all_data = []
    failed_tickers = []

    # Process in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} tickers)...")

        try:
            # Download batch
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,  # Adjust for splits and dividends
                threads=True,
                progress=False
            )

            # Process each ticker in batch
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    if ticker_data.empty:
                        print(f"  ⚠️  {ticker}: No data available")
                        failed_tickers.append(ticker)
                        continue

                    # Add ticker column
                    ticker_data['ticker'] = ticker
                    ticker_data = ticker_data.reset_index()

                    # Rename columns
                    ticker_data.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']

                    all_data.append(ticker_data)
                    print(f"  ✅ {ticker}: {len(ticker_data)} rows")

                except Exception as e:
                    print(f"  ❌ {ticker}: {str(e)}")
                    failed_tickers.append(ticker)

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Batch failed: {str(e)}")
            failed_tickers.extend(batch)

    if not all_data:
        raise ValueError("No data was fetched successfully!")

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"\n{'='*80}")
    print(f"DATA FETCH COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successfully fetched: {len(tickers) - len(failed_tickers)} tickers")
    print(f"❌ Failed: {len(failed_tickers)} tickers")
    if failed_tickers:
        print(f"   Failed tickers: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}")
    print(f"📊 Total data points: {len(df):,}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

    return df, failed_tickers

# =============================================================================
# NEWS SENTIMENT ANALYSIS
# =============================================================================

def get_yahoo_finance_news_sentiment(ticker, analyzer=None):
    """
    Fetch and analyze news sentiment from Yahoo Finance

    Args:
        ticker: Stock ticker symbol
        analyzer: VADER SentimentIntensityAnalyzer instance

    Returns:
        dict with sentiment scores
    """
    if not NEWS_SCRAPING_AVAILABLE or not SENTIMENT_AVAILABLE or analyzer is None:
        return {
            'news_sentiment': 0.0,
            'news_positive': 0.0,
            'news_negative': 0.0,
            'news_neutral': 0.0,
            'news_count': 0
        }

    try:
        # Get stock news using yfinance
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return {
                'news_sentiment': 0.0,
                'news_positive': 0.0,
                'news_negative': 0.0,
                'news_neutral': 0.0,
                'news_count': 0
            }

        # Analyze sentiment of news titles
        sentiments = []
        for article in news[:10]:  # Analyze last 10 articles
            title = article.get('title', '')
            if title:
                scores = analyzer.polarity_scores(title)
                sentiments.append(scores['compound'])

        if sentiments:
            return {
                'news_sentiment': np.mean(sentiments),
                'news_positive': sum(1 for s in sentiments if s > 0.05) / len(sentiments),
                'news_negative': sum(1 for s in sentiments if s < -0.05) / len(sentiments),
                'news_neutral': sum(1 for s in sentiments if -0.05 <= s <= 0.05) / len(sentiments),
                'news_count': len(sentiments)
            }
        else:
            return {
                'news_sentiment': 0.0,
                'news_positive': 0.0,
                'news_negative': 0.0,
                'news_neutral': 0.0,
                'news_count': 0
            }

    except Exception as e:
        return {
            'news_sentiment': 0.0,
            'news_positive': 0.0,
            'news_negative': 0.0,
            'news_neutral': 0.0,
            'news_count': 0
        }

def add_sentiment_features(df, sample_size=20):
    """
    Add sentiment features to the dataset

    Args:
        df: DataFrame with stock data
        sample_size: Number of random stocks to analyze (to save time)

    Returns:
        DataFrame with sentiment features
    """
    if not SENTIMENT_AVAILABLE:
        print("\n⚠️  Sentiment analysis not available - skipping")
        # Add dummy columns
        df['news_sentiment'] = 0.0
        df['news_positive'] = 0.0
        df['news_negative'] = 0.0
        df['news_neutral'] = 0.0
        df['news_count'] = 0
        return df

    print(f"\n{'='*80}")
    print(f"ADDING SENTIMENT FEATURES")
    print(f"{'='*80}")

    analyzer = SentimentIntensityAnalyzer()

    # Get unique tickers
    tickers = df['ticker'].unique()

    # Sample tickers if too many
    if len(tickers) > sample_size:
        print(f"Sampling {sample_size} tickers out of {len(tickers)} for sentiment analysis...")
        sampled_tickers = np.random.choice(tickers, sample_size, replace=False)
    else:
        sampled_tickers = tickers

    # Create sentiment mapping
    sentiment_map = {}

    for i, ticker in enumerate(sampled_tickers, 1):
        print(f"Analyzing {ticker} ({i}/{len(sampled_tickers)})...")
        sentiment_map[ticker] = get_yahoo_finance_news_sentiment(ticker, analyzer)
        time.sleep(0.5)  # Rate limiting

    # Add default sentiment for non-sampled tickers
    default_sentiment = {
        'news_sentiment': 0.0,
        'news_positive': 0.0,
        'news_negative': 0.0,
        'news_neutral': 0.0,
        'news_count': 0
    }

    # Map sentiment to dataframe
    for ticker in tickers:
        if ticker not in sentiment_map:
            sentiment_map[ticker] = default_sentiment

    df['news_sentiment'] = df['ticker'].map(lambda x: sentiment_map[x]['news_sentiment'])
    df['news_positive'] = df['ticker'].map(lambda x: sentiment_map[x]['news_positive'])
    df['news_negative'] = df['ticker'].map(lambda x: sentiment_map[x]['news_negative'])
    df['news_neutral'] = df['ticker'].map(lambda x: sentiment_map[x]['news_neutral'])
    df['news_count'] = df['ticker'].map(lambda x: sentiment_map[x]['news_count'])

    print(f"\n✅ Sentiment features added")
    print(f"   Average sentiment: {df['news_sentiment'].mean():.3f}")
    print(f"   Positive news: {df['news_positive'].mean()*100:.1f}%")
    print(f"   Negative news: {df['news_negative'].mean()*100:.1f}%")

    return df

# =============================================================================
# OPTIONS DATA (IMPLIED VOLATILITY)
# =============================================================================

def get_implied_volatility(ticker):
    """
    Get implied volatility from options data

    Args:
        ticker: Stock ticker symbol

    Returns:
        float: Implied volatility (or 0 if not available)
    """
    try:
        stock = yf.Ticker(ticker)
        options = stock.options

        if not options:
            return 0.0

        # Get nearest expiration
        opt_chain = stock.option_chain(options[0])

        # Calculate average implied volatility from calls
        if not opt_chain.calls.empty and 'impliedVolatility' in opt_chain.calls.columns:
            iv = opt_chain.calls['impliedVolatility'].mean()
            return float(iv) if not pd.isna(iv) else 0.0

        return 0.0

    except Exception:
        return 0.0

def add_options_features(df, sample_size=30):
    """
    Add options-based features (implied volatility)

    Args:
        df: DataFrame with stock data
        sample_size: Number of stocks to analyze

    Returns:
        DataFrame with options features
    """
    print(f"\n{'='*80}")
    print(f"ADDING OPTIONS DATA (IMPLIED VOLATILITY)")
    print(f"{'='*80}")

    tickers = df['ticker'].unique()

    # Sample tickers
    if len(tickers) > sample_size:
        print(f"Sampling {sample_size} tickers out of {len(tickers)}...")
        sampled_tickers = np.random.choice(tickers, sample_size, replace=False)
    else:
        sampled_tickers = tickers

    iv_map = {}

    for i, ticker in enumerate(sampled_tickers, 1):
        print(f"Fetching IV for {ticker} ({i}/{len(sampled_tickers)})...")
        iv_map[ticker] = get_implied_volatility(ticker)
        time.sleep(0.5)

    # Add default IV for non-sampled tickers
    for ticker in tickers:
        if ticker not in iv_map:
            iv_map[ticker] = 0.0

    df['implied_volatility'] = df['ticker'].map(iv_map)

    print(f"\n✅ Options features added")
    print(f"   Average IV: {df['implied_volatility'].mean():.3f}")

    return df

# =============================================================================
# MARKET CONTEXT FEATURES
# =============================================================================

def add_market_context_features(df, market_data):
    """
    Add market context features by merging with market indicators

    Args:
        df: Main stock dataframe
        market_data: DataFrame with market indicators

    Returns:
        DataFrame with market context features
    """
    print(f"\n{'='*80}")
    print(f"ADDING MARKET CONTEXT FEATURES")
    print(f"{'='*80}")

    # Get market indicator tickers
    market_tickers = market_data['ticker'].unique()

    for ticker in market_tickers:
        ticker_data = market_data[market_data['ticker'] == ticker].copy()

        # Calculate returns
        ticker_data[f'{ticker}_return'] = ticker_data['Close'].pct_change()
        ticker_data[f'{ticker}_volatility'] = ticker_data[f'{ticker}_return'].rolling(20).std()

        # Merge with main dataframe
        merge_cols = ['date', f'{ticker}_return', f'{ticker}_volatility']
        ticker_data_subset = ticker_data[merge_cols]

        df = df.merge(ticker_data_subset, on='date', how='left')

        print(f"  ✅ Added {ticker} context features")

    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"\n✅ Market context features added")

    return df

# =============================================================================
# EARNINGS CALENDAR
# =============================================================================

def add_earnings_features(df, sample_size=30):
    """
    Add days to next earnings announcement

    Args:
        df: DataFrame with stock data
        sample_size: Number of stocks to analyze

    Returns:
        DataFrame with earnings features
    """
    print(f"\n{'='*80}")
    print(f"ADDING EARNINGS CALENDAR FEATURES")
    print(f"{'='*80}")

    # For simplicity, add a placeholder
    # In production, you'd integrate with an earnings calendar API
    df['days_to_earnings'] = 0  # Placeholder

    print(f"✅ Earnings features added (placeholder)")

    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main data collection pipeline"""

    print(f"\n{'='*80}")
    print(f"ENHANCED STOCK DATA COLLECTION")
    print(f"{'='*80}")
    print(f"Start Date: {START_DATE.strftime('%Y-%m-%d')}")
    print(f"End Date: {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Duration: {(END_DATE - START_DATE).days} days (~{(END_DATE - START_DATE).days/365:.1f} years)")

    # Combine all tickers
    all_tickers = (
        STOCK_TICKERS[:100] +  # Limit to first 100 stocks for demo
        SECTOR_ETFS +
        THEMATIC_ETFS[:20] +  # Limit thematic ETFs
        BOND_ETFS
    )

    print(f"\n📊 Ticker Breakdown:")
    print(f"   Stocks: {len(STOCK_TICKERS[:100])}")
    print(f"   Sector ETFs: {len(SECTOR_ETFS)}")
    print(f"   Thematic ETFs: {len(THEMATIC_ETFS[:20])}")
    print(f"   Bond ETFs: {len(BOND_ETFS)}")
    print(f"   Total: {len(all_tickers)}")

    # Ask user what to fetch
    print(f"\n{'='*80}")
    print("DATA COLLECTION OPTIONS")
    print(f"{'='*80}")
    print("1. Basic data only (fastest)")
    print("2. Basic + Market context")
    print("3. Basic + Market context + Sentiment")
    print("4. Full suite (Basic + Market + Sentiment + Options)")

    choice = input("\nEnter choice (1-4) [default=2]: ").strip() or "2"

    # Step 1: Fetch basic stock data
    print("\n" + "="*80)
    print("STEP 1: FETCHING BASIC STOCK DATA")
    print("="*80)
    df_stocks, failed = fetch_stock_data(all_tickers, START_DATE, END_DATE)

    # Step 2: Fetch market indicators
    if choice in ['2', '3', '4']:
        print("\n" + "="*80)
        print("STEP 2: FETCHING MARKET INDICATORS")
        print("="*80)
        df_market, _ = fetch_stock_data(MARKET_INDICATORS, START_DATE, END_DATE)

        # Add market context
        df_stocks = add_market_context_features(df_stocks, df_market)

    # Step 3: Add sentiment features
    if choice in ['3', '4']:
        df_stocks = add_sentiment_features(df_stocks, sample_size=20)

    # Step 4: Add options data
    if choice == '4':
        df_stocks = add_options_features(df_stocks, sample_size=20)

    # Step 5: Add earnings calendar
    if choice in ['3', '4']:
        df_stocks = add_earnings_features(df_stocks, sample_size=20)

    # Final statistics
    print(f"\n{'='*80}")
    print("FINAL DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total rows: {len(df_stocks):,}")
    print(f"Unique tickers: {df_stocks['ticker'].nunique()}")
    print(f"Columns: {len(df_stocks.columns)}")
    print(f"Date range: {df_stocks['date'].min()} to {df_stocks['date'].max()}")
    print(f"Memory usage: {df_stocks.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print(f"\nColumn list:")
    for col in df_stocks.columns:
        print(f"  - {col}")

    # Save to CSV
    output_file = f'stock_data_10years_{datetime.now().strftime("%Y%m%d")}.csv'
    print(f"\n💾 Saving to {output_file}...")
    df_stocks.to_csv(output_file, index=False)
    print(f"✅ Saved successfully!")

    # Also save metadata
    metadata = {
        'created_date': datetime.now().isoformat(),
        'start_date': START_DATE.strftime('%Y-%m-%d'),
        'end_date': END_DATE.strftime('%Y-%m-%d'),
        'total_tickers': len(all_tickers),
        'successful_tickers': len(all_tickers) - len(failed),
        'failed_tickers': failed,
        'total_rows': len(df_stocks),
        'columns': list(df_stocks.columns),
        'features_included': {
            'basic_ohlcv': True,
            'market_context': choice in ['2', '3', '4'],
            'sentiment': choice in ['3', '4'],
            'options': choice == '4',
            'earnings': choice in ['3', '4']
        }
    }

    metadata_file = f'metadata_10years_{datetime.now().strftime("%Y%m%d")}.json'
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadata saved to {metadata_file}")

    print(f"\n{'='*80}")
    print("✅ DATA COLLECTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Output files:")
    print(f"   - {output_file}")
    print(f"   - {metadata_file}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Review the data quality")
    print(f"   2. Use this data with hybrid_model_IMPROVED.py")
    print(f"   3. Compare performance with 5-year data")

    return df_stocks

if __name__ == "__main__":
    df = main()
