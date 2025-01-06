import numpy as np
import pandas as pd
import streamlit as st

def preprocess_data(
    data: pd.DataFrame, 
    spy_data: pd.Series,
    min_non_na_ratio: float = 0.80
):
    """
    Clean and prepare data:
      1. Possibly drop columns with too many NaNs.
      2. Resample to monthly, compute monthly returns.
      3. Drop zero-variance columns.
      4. Return cleaned monthly returns, plus the same for SPY.
    """
    # 1. (Optional) Drop columns with < min_non_na_ratio non-NaN values
    min_non_na = int(min_non_na_ratio * len(data))
    data = data.dropna(axis=1, thresh=min_non_na)

    # 2. Resample to monthly data and drop remaining NaNs
    monthly_data = data.resample("M").last().dropna()

    # 3. Calculate monthly returns
    monthly_returns = np.log(monthly_data / monthly_data.shift(1)).dropna()

    # 4. Drop zero-variance columns
    monthly_returns = monthly_returns.loc[:, monthly_returns.var() > 0]

    # 5. Do the same for SPY
    spy_returns = np.log(spy_data / spy_data.shift(1)).dropna()

    if monthly_returns.empty:
        st.error(
            "No valid assets remain after data cleaning (NaNs, zero variance, etc.). "
            "Try reducing lookback years or adjusting filters."
        )
        st.stop()

    return monthly_returns, spy_returns
