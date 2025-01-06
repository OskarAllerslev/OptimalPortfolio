import numpy as np
import pandas as pd
import streamlit as st

def preprocess_data(data, spy_data, min_non_na_ratio=0.8):
    """
    Clean and prepare data:
      - Drop assets with too few observations.
      - Align portfolio and SPY data for comparison.
    """
    # Minimum valid rows per column
    min_non_na = int(min_non_na_ratio * len(data))
    data = data.dropna(axis=1, thresh=min_non_na)

    # Resample to monthly, compute monthly returns
    monthly_data = data.resample("M").last().dropna()
    spy_data = spy_data.resample("M").last().dropna()  # Match portfolio resampling

    # Monthly log returns
    monthly_returns = np.log(monthly_data / monthly_data.shift(1)).dropna()
    spy_returns = np.log(spy_data / spy_data.shift(1)).dropna()

    # Drop zero-variance columns
    monthly_returns = monthly_returns.loc[:, monthly_returns.var() > 0]

    # Align SPY and portfolio dates
    monthly_returns = monthly_returns.loc[spy_returns.index]

    if monthly_returns.empty:
        st.error("No valid assets remain after cleaning. Check data or lookback years.")
        st.stop()

    return monthly_returns, spy_returns
