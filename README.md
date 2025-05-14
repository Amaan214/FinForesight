# Stock Dashboard: Price, Sentiment, and Prediction

A powerful Streamlit dashboard for stock analysis, combining price data, Reddit sentiment scraping, and machine learning-based price prediction.

---

## Features

- 📈 **Stock Price Visualization**: Interactive charts using Yahoo Finance data.
- 🧾 **Fundamental Data**: (Optional) Balance sheet, income statement, and cash flow (requires Alpha Vantage API).
- 📰 **Reddit Sentiment Analysis**: Scrapes Reddit posts for ticker mentions and sentiment.
- 🤖 **Machine Learning Prediction**: Predicts stock closing prices using Random Forest regression.
- 📊 **Outlier and Feature Analysis**: Boxplots, volatility, moving averages, and more.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

Install required packages:
```bash
pip install -r requirements.txt
```

**Sample `requirements.txt`:**
```bash
streamlit
pandas
numpy
yfinance
plotly
praw
scikit-learn
seaborn
matplotlib
scipy
joblib
```

### 3. Configure Reddit API

Obtain Reddit API credentials from [Reddit Apps](https://www.reddit.com/prefs/apps) and update them in the code:

```bash
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
user_agent = "YOUR_USER_AGENT"
```


### 4. (Optional) Configure Alpha Vantage

For fundamental data, get an API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and insert it where indicated.

---

## Running the App

Start the Streamlit app:
```bash
streamlit run your_ui_file.py
```

*(Replace `your_ui_file.py` with your main Streamlit script, e.g., `app.py` or `scrapping_testing.ipynb` after conversion to `.py`.)*

---

## Usage

- **Sidebar**: Select stock ticker, date range, subreddit, and number of Reddit posts.
- **Tabs**:
  - **Pricing Data**: View price changes, annual returns, and risk metrics.
  - **Fundamental Data**: (If enabled) View financial statements.
  - **Top 10 News**: (If enabled) Latest news headlines.
  - **Prediction**: See Reddit sentiment, feature analysis, outlier detection, model training, and price predictions.

---

## Model

- **Algorithm**: Random Forest Regressor
- **Features**: Price data, Reddit sentiment, moving averages, volatility, etc.
- **Evaluation**: MAE, MSE, R², and actual vs predicted plots.
- **Model Saving**: Trained model is saved as `stock_prediction_model.pkl`.

---

## Notes

- **Security**: Never share API keys or secrets publicly.
- **Performance**: For large Reddit queries, increase Streamlit's cache or use smaller post limits.
- **Extensibility**: Add more features, switch ML models, or improve sentiment analysis as needed.

---

## License

[MIT](LICENSE)

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Reddit](https://www.reddit.com/)
- [scikit-learn](https://scikit-learn.org/)
- [Alpha Vantage](https://www.alphavantage.co/)

---

*Happy analyzing! 🚀*
