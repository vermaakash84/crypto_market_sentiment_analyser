import streamlit as st
import os
import pandas as pd
import joblib
import gdown

# File paths
FEAR_GREED_FILE = "fear_greed_index.csv"
HISTORICAL_DATA_FILE = "historical_data.csv"
MODEL_FILE = "champion_model.pkl"

# Load fear_greed_index.csv
if os.path.exists(FEAR_GREED_FILE):
    df1 = pd.read_csv(FEAR_GREED_FILE)
else:
    st.error(f"{FEAR_GREED_FILE} not found! Please upload it to the repo.")
    st.stop()

# Download historical_data.csv from Google Drive if not present
if not os.path.exists(HISTORICAL_DATA_FILE):
    gdrive_url = "https://drive.google.com/uc?id=1BWlLp26bUZ8prT8L-e6BXhNcBRwCTpgC"
    st.info("Downloading historical_data.csv from Google Drive...")
    gdown.download(gdrive_url, HISTORICAL_DATA_FILE, quiet=False)

# Load historical data
try:
    df2 = pd.read_csv(HISTORICAL_DATA_FILE)
except pd.errors.ParserError:
    st.error("Error reading historical_data.csv. Check if the file is a valid CSV.")
    st.stop()

# Load champion model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error(f"{MODEL_FILE} not found! Please upload it to the repo.")
    st.stop()

st.success("All data and model loaded successfully!")
