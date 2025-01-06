import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import streamlit as st

def plot_portfolio_vs_spy(daily_data, spy_data, chosen_assets, chosen_weights):
    """
    Plot normalized performance of the chosen portfolio vs SPY.
    """
    chosen_data_daily = daily_data[chosen_assets].dropna()
    spy_data_daily = spy_data.dropna()

    normalized_chosen_data_daily = chosen_data_daily / chosen_data_daily.iloc[0]
    normalized_spy_data_daily = spy_data_daily / spy_data_daily.iloc[0]

    # Daily portfolio performance with the chosen weights
    portfolio_values_daily = (normalized_chosen_data_daily * np.array(chosen_weights)).sum(axis=1)

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each chosen asset
    for i, asset in enumerate(chosen_assets):
        ax.plot(
            normalized_chosen_data_daily.index, 
            normalized_chosen_data_daily[asset],
            label=asset, 
            color=cm.Blues(np.linspace(0.4, 0.9, len(chosen_assets)))[i]
        )

    # Plot portfolio & SPY
    ax.plot(
        portfolio_values_daily.index, 
        portfolio_values_daily, 
        label="Portfolio (Weighted)", 
        color="green", 
        linewidth=2.5
    )
    ax.plot(
        normalized_spy_data_daily.index, 
        normalized_spy_data_daily, 
        label="SPY (Benchmark)", 
        color="orange", 
        linewidth=2
    )

    ax.set_title("Portfolio and SPY Performance (Daily Normalized)", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.5)

    st.pyplot(fig)
