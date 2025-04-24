# STOCKAI

A web app that analyzes trending stocks on Reddit and provides live sentiment analysis using machine learning.

## Features
- Fetches trending stocks from Reddit (r/wallstreetbets)
- Analyzes sentiment of posts mentioning each stock
- Displays trending stocks with sentiment bars
- Search for any stock ticker and get live Reddit sentiment

## Getting Started
1. Install dependencies:
   ```sh
   pip3 install -r requirements.txt
   ```
2. Set up Reddit API credentials (see below)
3. Run the app:
   ```sh
   python3 app.py
   ```
4. Visit [http://127.0.0.1:5010](http://127.0.0.1:5010)

## Reddit API Setup
- Get credentials at https://www.reddit.com/prefs/apps
- Set environment variables:
  - `REDDIT_CLIENT_ID`
  - `REDDIT_CLIENT_SECRET`
  - `REDDIT_USER_AGENT`

## Notes
- Uses VADER sentiment for fast prototyping (can upgrade to transformers later)
- For demo/learning purposes only

---

Built with Flask, Reddit API, and ML sentiment analysis.
