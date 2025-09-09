import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# Auto-download large dataset if not present
DATA_URL = 'https://drive.google.com/file/d/1BWlLp26bUZ8prT8L-e6BXhNcBRwCTpgC/view?usp=sharing' 
DATA_PATH = 'historical_data.csv'

if not os.path.exists(DATA_PATH):
    st.info('Downloading large dataset...')
    gdown.download(DATA_URL, DATA_PATH, quiet=False)

# Load datasets
df1 = pd.read_csv('fear_greed_index.csv')
df2 = pd.read_csv(DATA_PATH)

# Load the champion model
model = joblib.load('champion_model.pkl')

st.title('Market Sentiment Prediction App (Optimized Champion Model)')

st.write("""
This app predicts market sentiment (Fear, Greed, Neutral, etc.) based on user input trading data.
""")

# Input fields
mean_exec_price = st.number_input('Mean Execution Price', value=1000.0)
total_usd = st.number_input('Total USD Traded', value=50000.0)
total_tokens = st.number_input('Total Tokens Traded', value=1000.0)
avg_fee = st.number_input('Average Fee', value=0.01)
total_pnl = st.number_input('Total PnL', value=1000.0)
buy_ratio = st.number_input('Buy Ratio', value=0.5)
crossed_ratio = st.number_input('Crossed Ratio', value=0.5)
exec_price_volatility = st.number_input('Execution Price Volatility', value=1.0)
fee_volatility = st.number_input('Fee Volatility', value=0.1)
price_efficiency = st.number_input('Price Efficiency', value=1.0)
fee_efficiency = st.number_input('Fee Efficiency', value=0.01)
risk_reward = st.number_input('Risk-Reward Ratio', value=0.5)
aggressiveness = st.number_input('Aggressiveness', value=0.5)
smart_money = st.number_input('Smart Money', value=0.5)

if st.button('Predict Sentiment'):
    input_data = pd.DataFrame({
        'mean_exec_price': [mean_exec_price],
        'total_usd': [total_usd],
        'total_tokens': [total_tokens],
        'avg_fee': [avg_fee],
        'total_pnl': [total_pnl],
        'buy_ratio': [buy_ratio],
        'crossed_ratio': [crossed_ratio],
        'exec_price_volatility': [exec_price_volatility],
        'fee_volatility': [fee_volatility],
        'price_efficiency': [price_efficiency],
        'fee_efficiency': [fee_efficiency],
        'risk_reward': [risk_reward],
        'aggressiveness': [aggressiveness],
        'smart_money': [smart_money],
    })

    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Market Sentiment: {prediction}')
