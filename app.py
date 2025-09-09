import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load champion model ---
model = joblib.load("champion_model.pkl")

# --- Define the features your model expects ---
model_features = [
    'mean_exec_price', 'total_usd', 'total_tokens', 'avg_fee', 'total_pnl',
    'buy_ratio', 'crossed_ratio', 'exec_price_volatility', 'fee_volatility',
    'price_efficiency', 'fee_efficiency', 'risk_reward', 'aggressiveness',
    'smart_money', 'pnl_per_usd', 'enhanced_fee_efficiency', 'smart_fee_ratio',
    'risk_adjusted_efficiency', 'aggressive_efficiency', 'volatility_ratio',
    'stable_efficiency', 'top_features_interaction'
]

# --- Feature calculation function ---
def calculate_features(df):
    df_feat = pd.DataFrame()
    df_feat['mean_exec_price'] = df['price'].rolling(2, min_periods=1).mean()
    df_feat['total_usd'] = df['price'] * df['size']
    df_feat['total_tokens'] = df['size']
    df_feat['avg_fee'] = 0.001
    df_feat['total_pnl'] = df_feat['total_usd'].cumsum()
    df_feat['buy_ratio'] = (df['side'] == 'buy').cumsum() / (np.arange(len(df)) + 1)
    df_feat['crossed_ratio'] = 0.5
    df_feat['exec_price_volatility'] = df['price'].rolling(2, min_periods=1).std().fillna(0)
    df_feat['fee_volatility'] = 0.0
    df_feat['price_efficiency'] = df['price'] / df['price'].mean()
    df_feat['fee_efficiency'] = 1.0
    df_feat['risk_reward'] = 1.0
    df_feat['aggressiveness'] = (df['side'] == 'buy').astype(int)
    df_feat['smart_money'] = 0.0
    df_feat['pnl_per_usd'] = df_feat['total_pnl'] / (df_feat['total_usd'] + 1e-9)
    df_feat['enhanced_fee_efficiency'] = 1.0
    df_feat['smart_fee_ratio'] = 0.0
    df_feat['risk_adjusted_efficiency'] = 1.0
    df_feat['aggressive_efficiency'] = 1.0
    df_feat['volatility_ratio'] = df_feat['exec_price_volatility'] / (df['price'] + 1e-9)
    df_feat['stable_efficiency'] = 1.0
    df_feat['top_features_interaction'] = df_feat['mean_exec_price'] * df_feat['buy_ratio']
    return df_feat

# --- Streamlit App ---
st.title("Crypto Market Sentiment Analyser (Live Dashboard)")

# --- Editable trades table with dynamic rows ---
st.subheader("Editable Trades Table")
if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame({
        "price": [100, 102, 101],
        "size": [1, 2, 1.5],
        "side": ["buy", "sell", "buy"]
    })

editable_trades = st.experimental_data_editor(
    st.session_state.trades,
    num_rows="dynamic",
    key="trades_editor"
)

# --- Update session state ---
st.session_state.trades = editable_trades.copy()

# --- Compute features dynamically ---
features_df = calculate_features(editable_trades)

# --- Ensure all model features exist ---
for col in model_features:
    if col not in features_df.columns:
        features_df[col] = 0

X = features_df[model_features]

# --- Predict dynamically ---
features_df["prediction"] = model.predict(X)

# --- Display predictions and stats ---
st.subheader("Predictions")
st.dataframe(features_df.style
             .highlight_max(subset=["prediction"], color="lightgreen")
             .highlight_min(subset=["prediction"], color="lightcoral"))

st.subheader("Top 3 Trades")
st.table(features_df.nlargest(3, "prediction")[["price", "size", "side", "prediction"]])

st.subheader("Summary Statistics")
st.write(features_df.describe())
