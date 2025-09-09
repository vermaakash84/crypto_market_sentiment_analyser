import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load final champion model ---
model = joblib.load("production_model_final.pkl")
optimal_weights = joblib.load("optimal_weights_final.pkl")  # Optional if needed

# --- Feature calculation function ---
def calculate_features(df):
    df = df.copy()
    df["price_change"] = df["price"].diff().fillna(0)
    df["size_change"] = df["size"].diff().fillna(0)
    df["buy_flag"] = (df["side"].str.lower() == "buy").astype(int)
    df["sell_flag"] = (df["side"].str.lower() == "sell").astype(int)
    df["cumulative_buy"] = df["size"] * df["buy_flag"]
    df["cumulative_sell"] = df["size"] * df["sell_flag"]
    df["rolling_mean_price"] = df["price"].rolling(3, min_periods=1).mean()
    df["rolling_std_price"] = df["price"].rolling(3, min_periods=1).std().fillna(0)
    df["rolling_mean_size"] = df["size"].rolling(3, min_periods=1).mean()
    df["rolling_std_size"] = df["size"].rolling(3, min_periods=1).std().fillna(0)
    df["buy_sell_ratio"] = df["cumulative_buy"] / (df["cumulative_sell"] + 1e-6)
    df["price_volatility"] = df["price"].rolling(5, min_periods=1).std().fillna(0)
    df["size_volatility"] = df["size"].rolling(5, min_periods=1).std().fillna(0)
    df["trade_count"] = np.arange(1, len(df)+1)
    df["buy_count"] = df["buy_flag"].cumsum()
    df["sell_count"] = df["sell_flag"].cumsum()
    df["buy_sell_diff"] = df["buy_count"] - df["sell_count"]
    df["price_momentum"] = df["price"] - df["price"].shift(3).fillna(0)
    df["size_momentum"] = df["size"] - df["size"].shift(3).fillna(0)
    df["large_trade_flag"] = (df["size"] > df["size"].mean() + 2*df["size"].std()).astype(int)
    df["price_spike_flag"] = (df["price_change"].abs() > 2*df["price_change"].std()).astype(int)
    df["recent_buy_ratio"] = df["buy_flag"].rolling(5, min_periods=1).mean()
    df["recent_sell_ratio"] = df["sell_flag"].rolling(5, min_periods=1).mean()
    df["price_slope"] = df["price"].diff().rolling(3, min_periods=1).mean().fillna(0)
    df["size_slope"] = df["size"].diff().rolling(3, min_periods=1).mean().fillna(0)
    return df

# --- Top 3 largest trades ---
def top_3_trades(df):
    if df.empty:
        return pd.DataFrame(columns=["Rank", "Price", "Size", "Side", "Indicator"])
    df_sorted = df.sort_values("size", ascending=False).head(3).reset_index(drop=True)
    df_sorted["Rank"] = df_sorted.index + 1
    df_sorted["Indicator"] = df_sorted["side"].apply(lambda x: "🔼" if x.lower() == "buy" else "🔽")
    return df_sorted[["Rank", "Price", "Size", "Side", "Indicator"]]

# --- Streamlit UI ---
st.set_page_config(page_title="Crypto Market Sentiment Analyzer", layout="wide")
st.title("Crypto Market Sentiment Analyzer")

# Sidebar with champion model metrics
st.sidebar.header("Champion Model Performance")
st.sidebar.metric("Accuracy", "59%")
st.sidebar.metric("Fear Recall", "60%")
st.sidebar.metric("Extreme Fear Recall", "100%")

st.write("Enter trades below to see live predictions, summary stats, and top trades.")

# Editable trade table
st.subheader("Trade Input Table")
default_data = {
    "price": [100, 102, 101],
    "size": [5, 10, 7],
    "side": ["Buy", "Sell", "Buy"]
}
df_raw = st.data_editor(
    pd.DataFrame(default_data),
    num_rows="dynamic",
    use_container_width=True
)

# Calculate features and predictions in real-time
df_features = calculate_features(df_raw)
feature_cols = [col for col in df_features.columns if col not in ["price", "size", "side"]]
df_features_for_model = df_features[feature_cols]

if not df_features_for_model.empty:
    df_raw["prediction"] = model.predict(df_features_for_model)
else:
    df_raw["prediction"] = []

# Highlight extreme trades
def highlight_extremes(row):
    if row["size"] > df_raw["size"].mean() + 2*df_raw["size"].std():
        return ["background-color: #ffcccc"]*len(row)
    else:
        return [""]*len(row)

st.subheader("Trades with Predictions")
st.dataframe(df_raw.style.apply(highlight_extremes, axis=1), use_container_width=True)

# Summary stats
st.subheader("Summary Stats")
total_buys = df_raw[df_raw["side"].str.lower()=="buy"]["size"].sum()
total_sells = df_raw[df_raw["side"].str.lower()=="sell"]["size"].sum()
largest_trade = df_raw["size"].max() if not df_raw.empty else 0
col1, col2, col3 = st.columns(3)
col1.metric("Total Buys", total_buys)
col2.metric("Total Sells", total_sells)
col3.metric("Largest Trade", largest_trade)

# Top 3 largest trades
st.subheader("Top 3 Largest Trades")
st.table(top_3_trades(df_raw))
