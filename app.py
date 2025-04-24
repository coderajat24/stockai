from flask import Flask, render_template, request, jsonify
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import os

nltk.download('vader_lexicon')

app = Flask(__name__)

# Configure Reddit API (replace with your credentials)
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', 'stockai-app')

reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

sentiment_analyzer = SentimentIntensityAnalyzer()

# Helper: Get trending tickers from r/wallstreetbets

def get_trending_stocks(limit=10):
    subreddit = reddit.subreddit('wallstreetbets')
    posts = subreddit.hot(limit=100)
    tickers = {}
    for post in posts:
        words = post.title.split()
        for word in words:
            if word.isupper() and 2 <= len(word) <= 5:
                tickers[word] = tickers.get(word, 0) + 1
    trending = sorted(tickers.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [t[0] for t in trending]

# Helper: Get sentiment for a ticker

def get_sentiment_for_ticker(ticker, limit=30):
    subreddit = reddit.subreddit('wallstreetbets')
    query = ticker
    posts = subreddit.search(query, limit=limit)
    sentiments = []
    for post in posts:
        score = sentiment_analyzer.polarity_scores(post.title + ' ' + post.selftext)
        sentiments.append(score['compound'])
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
    else:
        avg_sentiment = 0
    return avg_sentiment, len(sentiments)

@app.route('/')
def index():
    trending = get_trending_stocks()
    sentiments = []
    for ticker in trending:
        avg_sent, count = get_sentiment_for_ticker(ticker)
        sentiments.append({'ticker': ticker, 'sentiment': avg_sent, 'mentions': count})
    return render_template('index.html', trending=sentiments)

@app.route('/search')
def search():
    q = request.args.get('q', '').upper()
    avg_sent, count = get_sentiment_for_ticker(q)
    return jsonify({'ticker': q, 'avg_sentiment': avg_sent, 'mentions': count})

if __name__ == '__main__':
    app.run(debug=True, port=5010)
