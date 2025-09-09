import streamlit as st
import os
import pandas as pd
import joblib
import gdown
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# File paths
# -----------------------------
FEAR_GREED_FILE = "fear_greed_index.csv"
HISTORICAL_DATA_FILE = "historical_data.csv"
MODEL_FILE = "champion_model.pkl"

st.set_page_config(page_title="Crypto Market Sentiment Analyzer", layout="wide")

st.title("📈 Crypto Market Sentiment Analyzer")

# -----------------------------
# Load fear_greed_index.csv
# -----------------------------
if os.path.exists(FEAR_GREED_FILE):
    df_fgi = pd.read_csv(FEAR_GREED_FILE)
else:
    st.error(f"{FEAR_GREED_FILE} not found! Please upload it to the repo.")
    st.stop()

# -----------------------------
# Download historical_data.csv from Google Drive if not present
# -----------------------------
if not os.path.exists(HISTORICAL_DATA_FILE):
    gdrive_url = "https://drive.google.com/uc?id=1BWlLp26bUZ8prT8L-e6BXhNcBRwCTpgC"
    st.info("Downloading historical_data.csv from Google Drive...")
    gdown.download(gdrive_url, HISTORICAL_DATA_FILE, quiet=False)

# Load historical data
try:
    df_hist = pd.read_csv(HISTORICAL_DATA_FILE)
except pd.errors.ParserError:
    st.error("Error reading historical_data.csv. Check if the file is a valid CSV.")
    st.stop()

# -----------------------------
# Load XGBoost model
# -----------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error(f"{MODEL_FILE} not found! Please upload it to the repo.")
    st.stop()

st.success("✅ All data and model loaded successfully!")

# -----------------------------
# Sidebar options
# -----------------------------
st.sidebar.header("Options")
show_fgi = st.sidebar.checkbox("Show Fear & Greed Index Data", value=True)
show_hist = st.sidebar.checkbox("Show Historical Data", value=True)
show_charts = st.sidebar.checkbox("Show Charts", value=True)
predict = st.sidebar.checkbox("Run Predictions", value=True)

# -----------------------------
# Display Fear & Greed Index
# -----------------------------
if show_fgi:
    st.subheader("📊 Fear & Greed Index Data")
    st.dataframe(df_fgi.head(20))

# -----------------------------
# Display Historical Data
# -----------------------------
if show_hist:
    st.subheader("📊 Historical Data")
    st.dataframe(df_hist.head(20))

# -----------------------------
# Charts
# -----------------------------
if show_charts:
    st.subheader("📈 Charts")

    # Example: line chart of historical closing prices
    if "Close" in df_hist.columns:
        st.line_chart(df_hist["Close"])
    else:
        st.warning("Column 'Close' not found in historical_data.csv.")

    # Example: correlation heatmap
    numeric_cols = df_hist.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        st.write("Correlation Heatmap")
        corr = df_hist[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -----------------------------
# Predictions
# -----------------------------
if predict:
    st.subheader("🤖 Model Predictions")

    # Check if model expects certain columns
    features = df_hist.drop(columns=[col for col in df_hist.columns if col not in df_hist.select_dtypes(include='number')], errors='ignore')

    if features.shape[1] == 0:
        st.warning("No numeric features found in historical_data.csv for prediction.")
    else:
        # Example: predict on first 10 rows
        st.write("Predictions on first 10 rows:")
        preds = model.predict(features.head(10))
        st.write(preds)

st.info("App is interactive! Use the sidebar to toggle data, charts, and predictions.")
