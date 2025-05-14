import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import praw
import string
import re
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Streamlit UI setup
st.title('Stock Dashboard')

# Sidebar inputs
ticker = st.sidebar.text_input('Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2021-01-01'))

# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date)
fig = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Stock Price')
st.plotly_chart(fig)

# Tabs for content display
pricing_data, fundamental_data, news, prediction = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News", "Prediction"])

# Pricing Data Tab
with pricing_data:
    st.header('Price Movements')
    data['% Change'] = data['Adj Close'].pct_change() * 100
    data.dropna(inplace=True)
    st.write(data)
    
    annual_return = data['% Change'].mean() * 252
    st.write(f'Annual Return: {annual_return:.2f}%')
    
    stdev = np.std(data['% Change']) * np.sqrt(252)
    st.write(f'Standard Deviation: {stdev:.2f}%')
    
    risk_adj_return = annual_return / stdev
    st.write(f'Risk-Adjusted Return: {risk_adj_return:.2f}')

# Fundamental Data Tab (commented as the functionality is incomplete)


from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
    st.write("fundamental")
    # key = 'OW1639L63B5UCYYL'
    # fd = FundamentalData(key, output_format = 'pandas')
    # st.subheader('Balance Sheet')
    # balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    # bs = balance_sheet.T[2:]
    # bs.columns = list(balance_sheet.T.iloc[0])
    # st.write(bs)
    # st.subheader('Income Statement')
    # income_statement = fd.get_income_statement_annual (ticker) [0]
    # is1 = income_statement.T[2:]
    # is1.columns = list(income_statement.T.iloc[0])
    # st.write(is1)
    # st.subheader('Cash Flow Statement')
    # cash_flow = fd.get_cash_flow_annual (ticker)[0]
    # cf = cash_flow.T[2:]
    # cf.columns = list(cash_flow.T.iloc[0])
    # st.write(cf)

# from stocknews import StockNews
with news:
    st.write("news")
#     st.header(f'News of {ticker}')
#     sn = StockNews(ticker, save_news=False)
#     df_news = sn.read_rss()
#     for i in range(10):
#         st.subheader(f'News {i+1}')
#         st.write(df_news['published'][i])
#         st.write(df_news['title'][i])
#         st.write(df_news['summary'][i])
#         title_sentiment = df_news['sentiment_title'][i]
#         st.write(f'Title Sentiment {title_sentiment}')
#         news_sentiment = df_news['sentiment_summary'][i]
#         st.write(f'News Sentiment {news_sentiment}')

# Prediction Tab
with prediction:
    # Streamlit configuration and app title
   
    st.title("Stock Sentiment Analysis and Prediction")


    # Directly hardcoding the credentials (not secure for production)
    client_id = "TzEynEpYuoBZ72HpKVfsTA"
    client_secret = "A_A_OI-MyHEYxZKLYqYsW34XhE0r8A"
    user_agent = "Reddit_stock_scraping"

    # Initialize Reddit API
    reddit = praw.Reddit(
        client_id="TzEynEpYuoBZ72HpKVfsTA",
        client_secret="A_A_OI-MyHEYxZKLYqYsW34XhE0r8A",
        user_agent="Reddit_stock_scraping"
    )

    # Function to extract potential stock tickers from text
    def extract_tickers(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        tickers = [word for word in words if word.isupper() and len(word) >= 1 and len(word) <= 5]
        return tickers

    # Function to clean and preprocess text for sentiment analysis
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    # Basic keyword matching for sentiment analysis
    def sentiment_analysis(text):
        positive_words = ['buy', 'bullish', 'gain', 'profit']
        negative_words = ['sell', 'bearish', 'loss', 'drop']
        text = clean_text(text)
        if any(word in text for word in positive_words):
            return 1  # Positive sentiment
        elif any(word in text for word in negative_words):
            return -1  # Negative sentiment
        else:
            return 0  # Neutral sentiment

    # Sidebar configuration for selecting subreddit and limit
    subreddit_name = st.sidebar.text_input("Subreddit", "IndianStockMarket")
    post_limit = st.sidebar.slider("Number of Posts to Analyze", 100, 1000, 500)

    # Scrape Reddit data
    @st.cache_data
    def fetch_reddit_data(subreddit_name, post_limit):
        subreddit = reddit.subreddit(subreddit_name)
        trending_tickers = []
        ticker_sentiment = {}
        for submission in subreddit.top(limit=post_limit):
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
        return pd.DataFrame(ticker_counts.items(), columns=['Ticker', 'Count']).assign(Sentiment=lambda df: df['Ticker'].map(ticker_avg_sentiment))

    ticker_df = fetch_reddit_data(subreddit_name, post_limit)
    st.write("Reddit Data Analysis:", ticker_df)

    # Function to fetch historical stock price data
    @st.cache_data
    def get_stock_data(ticker, period='1mo'):
        try:
            stock_data = yf.Ticker(ticker).history(period=period)
            return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            st.error(f"Could not retrieve data for {ticker}: {e}")
            return None

    # Add historical stock data for each ticker
    @st.cache_data
    def compile_stock_data(ticker_df):
        stock_data_frames = []
        for ticker in ticker_df['Ticker']:
            stock_data = get_stock_data(ticker)
            if stock_data is not None:
                stock_data['Ticker'] = ticker
                stock_data_frames.append(stock_data)
        return pd.concat(stock_data_frames)

    historical_data_df = compile_stock_data(ticker_df)
    combined_df = pd.merge(historical_data_df, ticker_df, on='Ticker', how='left')
    combined_df.fillna(method='ffill', inplace=True)

    # Display combined data
    st.write("Combined Stock and Sentiment Data:", combined_df)

    # Plotting functions
    def plot_boxplots(df):
        st.subheader("Boxplot Analysis for Outliers")
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for i, col in enumerate(columns):
            sns.boxplot(data=df[col], ax=axs[i//3, i%3])
            axs[i//3, i%3].set_title(f'Boxplot of {col}')
        st.pyplot(fig)

    def plot_sentiment_vs_price(df):
        st.subheader("Sentiment vs Closing Price")
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Sentiment', y='Close', hue='Sentiment', palette='coolwarm', marker='o')
        plt.title('Sentiment vs Closing Price')
        st.pyplot(fig)

    plot_boxplots(combined_df)
    plot_sentiment_vs_price(combined_df)

    # Outlier Detection
    combined_df['Open_zscore'] = zscore(combined_df['Open'])
    outliers = combined_df[abs(combined_df['Open_zscore']) > 3]
    st.write("Outliers in 'Open' column:", outliers)

    # Feature Engineering
    combined_df['Moving_Avg_10'] = combined_df['Close'].rolling(window=10).mean()
    combined_df['Volatility'] = combined_df['High'] - combined_df['Low']

    # Train-Test Split
    X = combined_df.drop(['Close'], axis=1)  # Features
    y = combined_df['Close']  # Target variable
    label_encoder = LabelEncoder()
    X['Ticker'] = label_encoder.fit_transform(X['Ticker'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    st.subheader("Model Evaluation")
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared:", r2_score(y_test, y_pred))

    # Plot Actual vs Predicted
    st.subheader("Actual vs Predicted Closing Prices")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # Save Model
    joblib.dump(model, 'stock_prediction_model.pkl')
    st.write("Model saved as 'stock_prediction_model.pkl'")

