# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load champion model ---
model = joblib.load("champion_model.pkl")

# --- Feature columns expected by your model ---
feature_cols = [
    'mean_exec_price', 'total_usd', 'total_tokens', 'avg_fee', 'total_pnl',
    'buy_ratio', 'crossed_ratio', 'exec_price_volatility', 'fee_volatility',
    'price_efficiency', 'fee_efficiency', 'risk_reward', 'aggressiveness',
    'smart_money', 'pnl_per_usd', 'enhanced_fee_efficiency', 'smart_fee_ratio',
    'risk_adjusted_efficiency', 'aggressive_efficiency', 'volatility_ratio',
    'stable_efficiency', 'top_features_interaction'
]

st.title("🚀 Crypto Market Sentiment Analyser")

st.markdown("""
This app lets you input trades dynamically, calculates all features your model needs, 
and gives real-time predictions. Top trades, summary stats, and extreme values are highlighted.
""")

# --- Editable trades table ---
if 'trades' not in st.session_state:
    # initialize with 5 empty rows
    st.session_state.trades = pd.DataFrame({
        'price': [0.0]*5,
        'size': [0.0]*5,
        'side': ['buy','buy','buy','buy','buy']
    })

edited_df = st.experimental_data_editor(st.session_state.trades, num_rows="dynamic")
st.session_state.trades = edited_df.copy()

# --- Feature calculation function ---
def calculate_features(trades_df):
    df = trades_df.copy()
    if df.empty:
        return pd.DataFrame(columns=feature_cols)
    
    # Basic example feature calculations
    df['buy_flag'] = (df['side']=='buy').astype(int)
    df['sell_flag'] = (df['side']=='sell').astype(int)
    total_usd = (df['price']*df['size']).sum()
    total_tokens = df['size'].sum()
    avg_fee = 0.001  # placeholder, you can calculate actual fees
    total_pnl = 0.0  # placeholder
    buy_ratio = df['buy_flag'].mean()
    crossed_ratio = buy_ratio  # placeholder
    exec_price_volatility = df['price'].std() if len(df)>1 else 0.0
    fee_volatility = 0.0
    price_efficiency = 1.0
    fee_efficiency = 1.0
    risk_reward = 1.0
    aggressiveness = df['size'].max() if not df.empty else 0
    smart_money = 0.5
    pnl_per_usd = total_pnl / total_usd if total_usd != 0 else 0
    enhanced_fee_efficiency = 1.0
    smart_fee_ratio = 1.0
    risk_adjusted_efficiency = 1.0
    aggressive_efficiency = 1.0
    volatility_ratio = 1.0
    stable_efficiency = 1.0
    top_features_interaction = 1.0
    
    features = pd.DataFrame([[
        df['price'].mean(), total_usd, total_tokens, avg_fee, total_pnl,
        buy_ratio, crossed_ratio, exec_price_volatility, fee_volatility,
        price_efficiency, fee_efficiency, risk_reward, aggressiveness,
        smart_money, pnl_per_usd, enhanced_fee_efficiency, smart_fee_ratio,
        risk_adjusted_efficiency, aggressive_efficiency, volatility_ratio,
        stable_efficiency, top_features_interaction
    ]], columns=feature_cols)
    
    return features

# --- Calculate features dynamically ---
features_for_model = calculate_features(st.session_state.trades)

# --- Make predictions ---
if not features_for_model.empty:
    predictions = model.predict(features_for_model)[0]
else:
    predictions = None

st.subheader("📊 Predictions")
st.write(f"Predicted Sentiment Score: **{predictions}**")

# --- Top 3 trades ---
st.subheader("🏆 Top 3 Trades by Size")
if not st.session_state.trades.empty:
    top_trades = st.session_state.trades.nlargest(3, 'size')
    st.table(top_trades)

# --- Summary statistics ---
st.subheader("📈 Summary Statistics")
if not st.session_state.trades.empty:
    summary_stats = st.session_state.trades.describe()
    st.table(summary_stats)

# --- Highlight extremes ---
st.subheader("⚡ Extreme Trades Highlighted")
if not st.session_state.trades.empty:
    df_highlighted = st.session_state.trades.copy()
    # highlight trades with price > mean+2*std
    price_threshold = df_highlighted['price'].mean() + 2*df_highlighted['price'].std()
    df_highlighted['extreme'] = df_highlighted['price'] > price_threshold
    st.table(df_highlighted)
