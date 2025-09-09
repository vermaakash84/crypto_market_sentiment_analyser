import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load champion model ---
model = joblib.load("champion_model.pkl")

# --- Load precomputed features ---
# This CSV must have columns exactly matching the features your model was trained on
precomputed_features = pd.read_csv("precomputed_features.csv")

# --- Load original trades corresponding to precomputed features ---
# This CSV is optional, for editable UI and matching predictions
trade_inputs = pd.read_csv("trade_inputs.csv")

st.title("Crypto Market Sentiment Analyzer (Champion Model)")
st.write("This app uses the champion model for predictions.")

# --- Editable Trade Table ---
st.subheader("Trade Input Table")
df_raw = st.data_editor(trade_inputs, num_rows="dynamic", width="stretch")

# --- Map edits to precomputed features ---
# For simplicity, we assume the user edits don't change feature names
# In practice, you'd recalculate features here
df_features_for_model = precomputed_features.loc[df_raw.index]

# --- Make predictions ---
if not df_features_for_model.empty:
    df_raw["prediction"] = model.predict(df_features_for_model)
else:
    df_raw["prediction"] = []

# --- Highlight extreme trades ---
def highlight_extremes(row):
    if row["size"] > df_raw["size"].mean() + 2*df_raw["size"].std():
        return ["background-color: #ffcccc"]*len(row)
    else:
        return [""]*len(row)

st.subheader("Trades with Predictions")
st.dataframe(df_raw.style.apply(highlight_extremes, axis=1), width="stretch")

# --- Summary Stats ---
st.subheader("Summary Stats")
total_buys = df_raw[df_raw["side"].str.lower()=="buy"]["size"].sum()
total_sells = df_raw[df_raw["side"].str.lower()=="sell"]["size"].sum()
largest_trade = df_raw["size"].max() if not df_raw.empty else 0
col1, col2, col3 = st.columns(3)
col1.metric("Total Buys", total_buys)
col2.metric("Total Sells", total_sells)
col3.metric("Largest Trade", largest_trade)

# --- Top 3 Largest Trades ---
def top_3_trades(df):
    if df.empty:
        return pd.DataFrame(columns=["Rank", "Price", "Size", "Side", "Indicator"])
    df_sorted = df.sort_values("size", ascending=False).head(3).reset_index(drop=True)
    df_sorted["Rank"] = df_sorted.index + 1
    df_sorted["Indicator"] = df_sorted["side"].apply(lambda x: "🔼" if x.lower() == "buy" else "🔽")
    return df_sorted[["Rank", "Price", "Size", "Side", "Indicator"]]

st.subheader("Top 3 Largest Trades")
st.table(top_3_trades(df_raw))
