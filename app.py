import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load champion model
with open('champion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize session state for trades
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=['price', 'size', 'side', 'fee', 'timestamp'])

st.title("Crypto Market Sentiment Analyzer")

# Editable trades table
st.subheader("Editable Trades Table")
edited_df = st.experimental_data_editor(st.session_state.trades, num_rows="dynamic", use_container_width=True)
st.session_state.trades = edited_df.copy()

# Feature calculation function
def calculate_features(df):
    if df.empty:
        return pd.DataFrame(columns=[
            'mean_exec_price', 'total_usd', 'total_tokens', 'avg_fee', 'total_pnl', 'buy_ratio',
            'crossed_ratio', 'exec_price_volatility', 'fee_volatility', 'price_efficiency',
            'fee_efficiency', 'risk_reward', 'aggressiveness', 'smart_money', 'pnl_per_usd',
            'enhanced_fee_efficiency', 'smart_fee_ratio', 'risk_adjusted_efficiency',
            'aggressive_efficiency', 'volatility_ratio', 'stable_efficiency',
            'top_features_interaction'
        ])
    mean_exec_price = df['price'].mean()
    total_usd = (df['price'] * df['size']).sum()
    total_tokens = df['size'].sum()
    avg_fee = df['fee'].mean()
    total_pnl = ((df['side'] == 'buy') * df['price'] - (df['side'] == 'sell') * df['price']).sum()
    buy_ratio = (df['side'] == 'buy').mean()
    crossed_ratio = (df['price'].diff().fillna(0) < 0).mean()
    exec_price_volatility = df['price'].std()
    fee_volatility = df['fee'].std()
    price_efficiency = mean_exec_price / (df['price'].max() + 1e-6)
    fee_efficiency = avg_fee / (df['fee'].max() + 1e-6)
    risk_reward = total_pnl / (exec_price_volatility + 1e-6)
    aggressiveness = buy_ratio * crossed_ratio
    smart_money = buy_ratio - crossed_ratio
    pnl_per_usd = total_pnl / (total_usd + 1e-6)
    enhanced_fee_efficiency = avg_fee / (exec_price_volatility + 1e-6)
    smart_fee_ratio = smart_money / (avg_fee + 1e-6)
    risk_adjusted_efficiency = risk_reward * smart_money
    aggressive_efficiency = aggressiveness * fee_efficiency
    volatility_ratio = exec_price_volatility / (fee_volatility + 1e-6)
    stable_efficiency = 1 / (exec_price_volatility + 1e-6)
    top_features_interaction = mean_exec_price * total_tokens

    feature_vector = pd.DataFrame([[
        mean_exec_price, total_usd, total_tokens, avg_fee, total_pnl, buy_ratio,
        crossed_ratio, exec_price_volatility, fee_volatility, price_efficiency,
        fee_efficiency, risk_reward, aggressiveness, smart_money, pnl_per_usd,
        enhanced_fee_efficiency, smart_fee_ratio, risk_adjusted_efficiency,
        aggressive_efficiency, volatility_ratio, stable_efficiency,
        top_features_interaction
    ]], columns=[
        'mean_exec_price', 'total_usd', 'total_tokens', 'avg_fee', 'total_pnl', 'buy_ratio',
        'crossed_ratio', 'exec_price_volatility', 'fee_volatility', 'price_efficiency',
        'fee_efficiency', 'risk_reward', 'aggressiveness', 'smart_money', 'pnl_per_usd',
        'enhanced_fee_efficiency', 'smart_fee_ratio', 'risk_adjusted_efficiency',
        'aggressive_efficiency', 'volatility_ratio', 'stable_efficiency',
        'top_features_interaction'
    ])
    return feature_vector

features_df = calculate_features(st.session_state.trades)

# Make predictions
if not features_df.empty:
    pred_scores = model.predict_proba(features_df)
    sentiment_labels = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    predicted_sentiment = sentiment_labels[np.argmax(pred_scores)]
else:
    predicted_sentiment = "N/A"

# Display prediction
st.subheader("Predicted Market Sentiment")
st.write(f"**Sentiment:** {predicted_sentiment}")

# Summary Stats Panel
st.subheader("Summary Statistics")
total_buys = (st.session_state.trades['side'] == 'buy').sum()
total_sells = (st.session_state.trades['side'] == 'sell').sum()
largest_trade = (st.session_state.trades['price'] * st.session_state.trades['size']).max() if not st.session_state.trades.empty else 0

st.write(f"Total Buys: {total_buys}")
st.write(f"Total Sells: {total_sells}")
st.write(f"Largest Trade USD Value: {largest_trade:.2f}")

# Top 3 Largest Trades Panel
st.subheader("Top 3 Largest Trades (USD)")
st.write("🔼 Buy | 🔽 Sell")

if not st.session_state.trades.empty:
    st.session_state.trades['usd_value'] = st.session_state.trades['price'] * st.session_state.trades['size']
    top_trades = st.session_state.trades.nlargest(3, 'usd_value')
    for i, row in top_trades.iterrows():
        icon = "🔼" if row['side'] == 'buy' else "🔽"
        st.write(f"{icon} Price: {row['price']}, Size: {row['size']}, USD Value: {row['usd_value']:.2f}")

# Highlight extreme trades in table
def highlight_extremes(row):
    usd_value = row['price'] * row['size']
    if usd_value > 100000:  # Threshold for huge trades
        return ['background-color: yellow'] * len(row)
    return [''] * len(row)

st.subheader("Trades with Highlighted Extremes")
if not st.session_state.trades.empty:
    styled_df = st.session_state.trades.style.apply(highlight_extremes, axis=1)
    st.dataframe(styled_df, height=300)
else:
    st.write("No trades to display.")
