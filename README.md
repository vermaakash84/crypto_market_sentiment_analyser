# Market Sentiment Prediction Web App

This Streamlit web application predicts market sentiment categories (Fear, Greed, Neutral, etc.)  
based on historical trading data using a champion XGBoost model.  
The champion model is automatically selected based on the best Fear recall and overall accuracy from multiple trained models.

---

## 🚀 Features
- Interactive input form for trading metrics
- Real-time prediction of market sentiment
- Built with Streamlit for fast web interaction

---

## ⚡ How to Use

1. Deploy the app on Streamlit Cloud or run it locally.
2. Enter the required input fields:
    - Mean Execution Price
    - Total USD Traded
    - Total Tokens Traded
    - Average Fee
    - Total PnL
    - Buy Ratio
    - Crossed Ratio
    - Execution Price Volatility
    - Fee Volatility
    - Price Efficiency
    - Fee Efficiency
    - Risk-Reward Ratio
    - Aggressiveness
    - Smart Money
3. Click **Predict Sentiment**.
4. View the predicted market sentiment.

---

## 📂 Data Download

The large dataset `historical_data.csv` (46 MB) is available for download here:  
[Google Drive Link](https://drive.google.com/file/d/1BWlLp26bUZ8prT8L-e6BXhNcBRwCTpgC/view?usp=sharing)

👉 Make sure to replace `https://drive.google.com/your_file_link_here`  
with your actual file link from Google Drive (use the "share" option and get a public link).

---

## ✅ Project Structure

