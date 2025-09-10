 Crypto Market Sentiment & Trader Behavior Analyzer

This project explores and analyzes the relationship between **trader behavior** and **market sentiment** in the cryptocurrency market.  
It predicts market sentiment (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) based on key trading behavior features.

---

## 🎯 Objective

To analyze how trading behavior metrics (profitability, risk, volume, leverage) align or diverge from overall market sentiment (Fear vs Greed),  
and identify hidden trends or signals for smarter trading strategies.

---

## 🚀 Features

- Predict market sentiment based on input trading behavior features.
- Uses an optimized XGBoost model trained with advanced engineered features and sample weights.
- Visualizes key insights such as feature importance and correlation (in notebook).
- Designed for easy deployment on Streamlit Cloud.

---

## ⚙️ How It Works

- Users input trading metrics (e.g., mean execution price, total USD volume, avg fee, risk-reward ratio).
- The app loads a pre-trained model (`production_model_final.pkl`) and predicts the market sentiment.
- Displays optimized weights used for training.

---

## ✅ Files Included

| File | Description |
|------|-------------|
| `app.py` | Streamlit app entry point for deployment. |
| `production_model_final.pkl` | Trained XGBoost classifier model. |
| `optimal_weights_final.pkl` | Optimized class sample weights used in training. |
| `requirements.txt` | Python dependencies required to run the app. |
| `README.md` | This file. |

---

## 📋 Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
