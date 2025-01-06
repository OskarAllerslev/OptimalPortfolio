import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import streamlit as st

def plot_portfolio_vs_spy(daily_data, spy_data, chosen_assets, chosen_weights):
    """
    Plot normalized performance of the chosen portfolio vs SPY using daily data.
    """
    chosen_data_daily = daily_data[chosen_assets].dropna(how="any")
    spy_data_daily = spy_data.dropna()

    if len(chosen_data_daily) == 0 or len(spy_data_daily) == 0:
        st.write("Not enough data to plot. Possibly due to missing daily data.")
        return

    # Normalize
    chosen_data_daily_norm = chosen_data_daily / chosen_data_daily.iloc[0]
    spy_data_daily_norm = spy_data_daily / spy_data_daily.iloc[0]

    # Weighted portfolio series
    portfolio_values_daily = (chosen_data_daily_norm * np.array(chosen_weights)).sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each chosen asset
    colors = cm.Blues(np.linspace(0.4, 0.9, len(chosen_assets)))
    for i, asset in enumerate(chosen_assets):
        ax.plot(
            chosen_data_daily_norm.index, 
            chosen_data_daily_norm[asset], 
            label=asset,
            color=colors[i]
        )

    # Plot portfolio and SPY
    ax.plot(portfolio_values_daily.index, portfolio_values_daily, 
            label="Portfolio (Weighted)", color="green", linewidth=2.2)
    ax.plot(spy_data_daily_norm.index, spy_data_daily_norm, 
            label="SPY (Benchmark)", color="orange", linewidth=2)

    ax.set_title("Portfolio vs SPY (Normalized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
