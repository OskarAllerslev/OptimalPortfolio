import streamlit as st
import numpy as np
import pandas as pd

from data_loader import download_data, download_spy_data
from data_preprocessing import preprocess_data
from optimization import optimize_portfolio
from plotting import plot_portfolio_vs_spy

def main():
    st.title("Advanced Portfolio Optimization")

    # Sidebar
    st.sidebar.header("Input Parameters")

    lookback_years = st.sidebar.slider("Lookback Period (Years)", 1, 4, 2) 
    annual_target_return = st.sidebar.slider("Target Annual Return (%)", 1, 30, 10) / 100
    K = st.sidebar.slider("Max Number of Assets in Portfolio", 5, 30, 20)
    min_weight = st.sidebar.slider("Min Weight for Chosen Assets (%)", 0.0, 5.0, 2.0) / 100

    # Optimization method
    method = st.sidebar.selectbox("Optimization Method", 
                                  ["mean_variance", "robust", "cvar"])

    # CVaR alpha explanation
    st.sidebar.markdown(
        "**CVaR alpha** controls how far into the tail risk we look. "
        "For example, alpha=0.95 means we focus on the worst 5% of outcomes."
    )
    alpha = st.sidebar.slider("CVaR alpha (if using CVaR)", 0.80, 0.99, 0.95)

    # Convert annual target return to monthly
    monthly_target_return = (1 + annual_target_return) ** (1/12) - 1

    # Load data
    data = download_data(lookback_years)
    spy_data = download_spy_data(lookback_years)

    # Preprocess to get monthly returns
    monthly_returns, spy_returns = preprocess_data(data, spy_data, 0.8)

    # For plotting, we’ll keep daily data
    daily_data = data.dropna()

    # Construct scenario matrix if using CVaR
    scenario_matrix = monthly_returns.values  # shape (T, N)

    # Optimize
    result = optimize_portfolio(
        monthly_returns=monthly_returns,
        monthly_target_return=monthly_target_return,
        K=K,
        min_weight=min_weight,
        method=method,
        alpha=alpha,
        # Drawdown constraint removed
        drawdown_constraint=False,
        max_drawdown=0.1,      # Not used now
        scenarios=scenario_matrix
    )

    if result["status"] == "optimal":
        chosen_assets = result["chosen_assets"]
        chosen_weights = result["chosen_weights"]

        st.session_state["chosen_assets"] = chosen_assets
        st.session_state["chosen_weights"] = chosen_weights

        mu_values = result["mu"]
        Sigma_values = result["Sigma"]

        # Build full weight vector for stats
        w_full = np.zeros_like(mu_values)
        idx_map = {asset: i for i, asset in enumerate(monthly_returns.columns)}
        for asset, w_val in zip(chosen_assets, chosen_weights):
            w_full[idx_map[asset]] = w_val

        # Portfolio stats
        real_return_monthly = float(mu_values @ w_full)
        real_return_annual = (1 + real_return_monthly)**12 - 1
        real_variance = float(w_full @ Sigma_values @ w_full)
        portfolio_vol_monthly = np.sqrt(real_variance)
        portfolio_vol_annual = portfolio_vol_monthly * np.sqrt(12)
        sharpe_ratio = (real_return_annual - 0.02) / portfolio_vol_annual

        # SPY stats (convert .mean() / .std() to float to avoid Series-format error)
        spy_monthly_return = float(spy_returns.mean())
        spy_annual_return = (1 + spy_monthly_return)**12 - 1
        spy_monthly_vol = float(spy_returns.std())
        spy_annual_vol = spy_monthly_vol * np.sqrt(12)

        st.subheader("Optimal Portfolio Results")
        st.write("**Chosen Assets:**", chosen_assets)
        st.write("**Weights:**", 
                 {a: round(w, 4) for a, w in zip(chosen_assets, chosen_weights)})

        # Create a table comparing Portfolio vs. SPY
        df_stats = pd.DataFrame({
            "Monthly Return": [real_return_monthly, spy_monthly_return],
            "Annual Return": [real_return_annual, spy_annual_return],
            "Monthly Volatility": [portfolio_vol_monthly, spy_monthly_vol],
            "Annual Volatility": [portfolio_vol_annual, spy_annual_vol],
            "Sharpe (Annual)": [sharpe_ratio, None]  # None for SPY's Sharpe, or compute if you like
        }, index=["Portfolio", "SPY"])

        # Format as percentages where appropriate
        format_dict = {
            "Monthly Return": "{:,.2%}",
            "Annual Return": "{:,.2%}",
            "Monthly Volatility": "{:,.2%}",
            "Annual Volatility": "{:,.2%}",
            "Sharpe (Annual)": "{:,.3f}"
        }
        st.dataframe(df_stats.style.format(format_dict))

        # Plot
        st.subheader("Performance Chart (Daily Data)")
        plot_portfolio_vs_spy(daily_data, spy_data, chosen_assets, chosen_weights)

    else:
        st.warning(f"Optimization not successful. Status: {result['status']}")

if __name__ == "__main__":
    main()
