import praw
import string
import re
import pandas as pd
import numpy as np
from collections import Counter
import yfinance as yf

# --- Helper: Validate ticker by checking if yfinance returns data ---
def is_valid_ticker(ticker, period='1mo'):
    try:
        df = yf.Ticker(ticker).history(period=period)
        return not df.empty
    except Exception:
        return False

# --- Reddit API setup ---
reddit = praw.Reddit(
    client_id="TzEynEpYuoBZ72HpKVfsTA",
    client_secret="A_A_OI-MyHEYxZKLYqYsW34XhE0r8A",
    user_agent="Reddit_stock_scraping",
)

def extract_tickers(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    tickers = [word for word in words if word.isupper() and 1 <= len(word) <= 5]
    return tickers

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def sentiment_analysis(text):
    positive_words = ['buy', 'bullish', 'gain', 'profit']
    negative_words = ['sell', 'bearish', 'loss', 'drop']
    text = clean_text(text)
    if any(word in text for word in positive_words):
        return 1
    elif any(word in text for word in negative_words):
        return -1
    else:
        return 0

# --- Scrape Reddit and build ticker list ---
subreddit = reddit.subreddit('IndianStockMarket')
trending_tickers = []
ticker_sentiment = {}

print("Scraping Reddit for tickers...")
for submission in subreddit.top(limit=50):  # Reduce for faster testing
    submission_tickers = extract_tickers(submission.title)
    sentiment_score = sentiment_analysis(submission.title)
    trending_tickers.extend(submission_tickers)
    for ticker in submission_tickers:
        if ticker in ticker_sentiment:
            ticker_sentiment[ticker].append(sentiment_score)
        else:
            ticker_sentiment[ticker] = [sentiment_score]

ticker_avg_sentiment = {ticker: sum(scores)/len(scores) for ticker, scores in ticker_sentiment.items()}
ticker_counts = Counter(trending_tickers)
ticker_df = pd.DataFrame(ticker_counts.items(), columns=['Ticker', 'Count'])
ticker_df['Sentiment'] = ticker_df['Ticker'].map(ticker_avg_sentiment)

print("\nExtracted tickers from Reddit:")
print(ticker_df['Ticker'].tolist())

# --- Add suffix for Indian tickers and filter valid tickers ---
def format_and_validate_ticker(ticker):
    nse_ticker = ticker + '.NS'
    if is_valid_ticker(nse_ticker):
        return nse_ticker
    bse_ticker = ticker + '.BO'
    if is_valid_ticker(bse_ticker):
        return bse_ticker
    if is_valid_ticker(ticker):  # US tickers
        return ticker
    return None

print("\nValidating tickers with Yahoo Finance...")
ticker_df['Yahoo_Ticker'] = ticker_df['Ticker'].apply(format_and_validate_ticker)
valid_tickers = ticker_df[~ticker_df['Yahoo_Ticker'].isnull()]['Yahoo_Ticker'].tolist()
invalid_tickers = ticker_df[ticker_df['Yahoo_Ticker'].isnull()]['Ticker'].tolist()

print(f"\nValid tickers: {valid_tickers}")
print(f"Invalid tickers: {invalid_tickers}")

if len(valid_tickers) == 0:
    print("No valid tickers found! Exiting.")
    exit()

# --- Fetch historical stock data for valid tickers ---
def get_stock_data(ticker, period='1mo'):
    try:
        stock_data = yf.Ticker(ticker).history(period=period)
        if stock_data.empty:
            return None
        return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return None

stock_data_frames = []
for ticker in valid_tickers:
    stock_data = get_stock_data(ticker)
    if stock_data is not None:
        stock_data['Ticker'] = ticker
        stock_data_frames.append(stock_data)

if not stock_data_frames:
    print("No stock data could be fetched for any valid ticker. Exiting.")
    exit()

historical_data_df = pd.concat(stock_data_frames)
print("\nShape of historical_data_df:", historical_data_df.shape)
if historical_data_df.empty:
    print("historical_data_df is empty after fetching. Exiting.")
    exit()

combined_df = pd.merge(historical_data_df, ticker_df, left_on='Ticker', right_on='Yahoo_Ticker', how='left')
combined_df.fillna(method='ffill', inplace=True)
print("Shape of combined_df after merge:", combined_df.shape)
if combined_df.empty:
    print("combined_df is empty after merging. Exiting.")
    exit()

# --- Feature Engineering ---
combined_df['Moving_Avg_10'] = combined_df['Close'].rolling(window=10).mean()
combined_df['Volatility'] = combined_df['High'] - combined_df['Low']
combined_df = combined_df.fillna(0)

# --- Model Training ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

X = combined_df.drop(['Close'], axis=1)
y = combined_df['Close']

print("X shape before split:", X.shape)
print("y shape before split:", y.shape)

if X.empty or y.empty:
    print("No data available for training/testing. Exiting.")
    exit()

label_encoder = LabelEncoder()
X['Ticker'] = label_encoder.fit_transform(X['Ticker'])

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except ValueError as e:
    print("Error during train_test_split:", e)
    exit()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

joblib.dump(model, 'stock_prediction_model.pkl')
print("Model saved as 'stock_prediction_model.pkl'")

# Save the combined dataset to CSV
combined_df.to_csv('combined_stock_data_indian.csv', index=False)

print("Pipeline completed successfully!")
