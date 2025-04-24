from flask import Flask, render_template, request, jsonify
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import yfinance as yf
import feedparser
import random

load_dotenv()

nltk.download('vader_lexicon')

app = Flask(__name__)

# Configure Reddit API (replace with your credentials)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

sentiment_analyzer = SentimentIntensityAnalyzer()

# Simple in-memory cache for ticker sentiment
sentiment_cache = {}
CACHE_DURATION = 600  # seconds (10 minutes)

# Helper: Get trending tickers from r/wallstreetbets

def get_trending_stocks(limit=10):
    subreddit = reddit.subreddit('wallstreetbets')
    posts = subreddit.hot(limit=100)
    tickers = {}
    for post in posts:
        words = post.title.split()
        for word in words:
            if word.isupper() and 2 <= len(word) <= 5 and word in POPULAR_TICKERS:
                tickers[word] = tickers.get(word, 0) + 1
    trending = sorted(tickers.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [t[0] for t in trending]

# Helper: Get sentiment for a ticker, return posts

def get_sentiment_for_ticker(ticker, limit=30):
    now = time.time()
    # Check cache
    if ticker in sentiment_cache:
        cached = sentiment_cache[ticker]
        if now - cached['timestamp'] < CACHE_DURATION:
            return cached['avg_sentiment'], cached['count'], cached['post_details']
    subreddit = reddit.subreddit('wallstreetbets')
    query = ticker
    posts = list(subreddit.search(query, limit=limit))
    sentiments = []
    post_details = []
    for post in posts:
        text = post.title + ' ' + post.selftext
        score = sentiment_analyzer.polarity_scores(text)
        sentiments.append(score['compound'])
        post_details.append({
            'title': post.title,
            'selftext': post.selftext,
            'created_utc': datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M'),
            'score': score['compound'],
            'url': post.url
        })
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
    else:
        avg_sentiment = 0
    # Store in cache
    sentiment_cache[ticker] = {
        'avg_sentiment': avg_sentiment,
        'count': len(sentiments),
        'post_details': post_details,
        'timestamp': now
    }
    return avg_sentiment, len(sentiments), post_details

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('regularMarketPrice')
        currency = stock.info.get('currency', 'USD')
        return price, currency
    except Exception:
        return None, None

def get_company_logo_url(ticker):
    # Use Clearbit Logo API or similar (public, free for most use cases)
    return f"https://logo.clearbit.com/{ticker.lower()}.com"

def get_sparkline_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        closes = hist['Close'].tolist() if 'Close' in hist else []
        return closes
    except Exception:
        return []

def get_market_overview():
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'Dow Jones': '^DJI'
    }
    overview = []
    for name, symbol in indices.items():
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            price = info.get('regularMarketPrice')
            prev = info.get('regularMarketPreviousClose')
            if price is not None and prev is not None:
                change = price - prev
                pct = (change / prev) * 100 if prev else 0
            else:
                change = pct = None
            overview.append({
                'name': name,
                'symbol': symbol,
                'price': price,
                'change': change,
                'pct': pct
            })
        except Exception:
            overview.append({
                'name': name,
                'symbol': symbol,
                'price': None,
                'change': None,
                'pct': None
            })
    return overview

def get_news_for_ticker(ticker):
    # Yahoo Finance RSS feed for news (no API key required)
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    try:
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries[:3]:
            news.append({
                'title': entry.title,
                'link': entry.link
            })
        return news
    except Exception:
        return []

def get_sentiment_history(ticker):
    # Simulate 7 days of sentiment scores between -1 and 1
    return [round(random.uniform(-0.5, 0.8), 2) for _ in range(7)]

# List of popular tickers to scan for movers
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'BRK-B',
    'JPM', 'V', 'UNH', 'DIS', 'PYPL', 'INTC', 'CSCO', 'PFE', 'BA', 'WMT',
    'T', 'KO', 'PEP', 'MCD', 'ADBE', 'CRM', 'ORCL', 'ABNB', 'UBER', 'SNAP',
    'SHOP', 'PLTR', 'SOFI', 'RIVN', 'NIO', 'XOM', 'CVX', 'GM', 'F', 'LCID',
    'BABA', 'JD', 'BIDU', 'ZM', 'SPOT', 'SQ', 'COIN', 'ROKU', 'DOCU', 'TWLO'
]

# Helper: Get top gainers/losers from a set of tickers

def get_top_movers(limit=10):
    tickers = yf.Tickers(' '.join(POPULAR_TICKERS))
    movers = []
    for symbol, stock in tickers.tickers.items():
        try:
            info = stock.info
            price = info.get('regularMarketPrice')
            prev_close = info.get('regularMarketPreviousClose')
            name = info.get('shortName', symbol)
            if price is not None and prev_close is not None and prev_close > 0:
                pct_change = round(100 * (price - prev_close) / prev_close, 2)
                movers.append({'ticker': symbol, 'name': name, 'price': price, 'pct_change': pct_change})
        except Exception:
            continue
    gainers = sorted([m for m in movers if m['pct_change'] > 0], key=lambda x: x['pct_change'], reverse=True)[:limit]
    losers = sorted([m for m in movers if m['pct_change'] < 0], key=lambda x: x['pct_change'])[:limit]
    return gainers, losers

@app.route('/')
def index():
    trending_raw = get_trending_stocks(40)  # Fetch even more to maximize pool
    print('Raw trending tickers:', trending_raw)
    trending_data = []
    for ticker in trending_raw:
        price, currency = get_stock_price(ticker)
        logo_url = get_company_logo_url(ticker)
        sparkline = get_sparkline_data(ticker)
        news = get_news_for_ticker(ticker)
        sentiment_history = get_sentiment_history(ticker)
        avg_sentiment = sum(sentiment_history)/len(sentiment_history) if sentiment_history else 0
        # No filtering at all, show every ticker from Reddit
        trending_data.append({
            'ticker': ticker,
            'price': price,
            'currency': currency,
            'logo_url': logo_url,
            'sparkline': sparkline,
            'news': news,
            'sentiment_history': sentiment_history,
            'avg_sentiment': avg_sentiment
        })
    print('Filtered trending_data count:', len(trending_data))
    trending_data.sort(key=lambda x: x['avg_sentiment'])
    trending_data = trending_data[:20]
    market_overview = get_market_overview()
    return render_template('index.html', trending=trending_data, market_overview=market_overview)

@app.route('/search')
def search():
    q = request.args.get('q', '').upper()
    avg_sent, count, post_details = get_sentiment_for_ticker(q)
    price, currency = get_stock_price(q)
    return jsonify({'ticker': q, 'avg_sentiment': avg_sent, 'mentions': count, 'price': price, 'currency': currency, 'posts': post_details})

@app.route('/top-movers')
def top_movers():
    gainers, losers = get_top_movers()
    return jsonify({'gainers': gainers, 'losers': losers})

if __name__ == '__main__':
    app.run(debug=True, port=5010)
