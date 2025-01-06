import streamlit as st
import numpy as np

# Import from our new modules
from data_loader import download_data, download_spy_data
from data_preprocessing import preprocess_data
from optimization import optimize_portfolio
from plotting import plot_portfolio_vs_spy

# ------------------------------------------------------
# Streamlit App
# ------------------------------------------------------
st.title("Portfolio Optimization App")
st.sidebar.header("Input Parameters")

# Sidebar parameters
lookback_years = st.sidebar.slider("Lookback Period (Years)", 1, 10, 5)
annual_target_return = st.sidebar.slider("Target Annual Return (%)", 1, 30, 10, step=1) / 100
K = st.sidebar.slider("Max Number of Assets in Portfolio", 5, 30, 20)
min_weight = st.sidebar.slider("Minimum Weight for Chosen Assets (%)", 0.0, 5.0, 2.0) / 100

# Convert annual target return to monthly
monthly_target_return = (1 + annual_target_return) ** (1 / 12) - 1

# ------------------------------------------------------
# 1. Download data
# ------------------------------------------------------
data = download_data(lookback_years)
spy_data = download_spy_data(lookback_years)

# ------------------------------------------------------
# 2. Preprocess data
# ------------------------------------------------------
monthly_returns, spy_returns = preprocess_data(data, spy_data, min_non_na_ratio=0.80)

# ------------------------------------------------------
# 3. Optimize portfolio
# ------------------------------------------------------
result = optimize_portfolio(monthly_returns, monthly_target_return, K, min_weight)

if result["status"] == "optimal":
    chosen_assets = result["chosen_assets"]
    chosen_weights = result["chosen_weights"]
    mu_values = result["mu"]
    Sigma_values = result["Sigma"]

    # Compute some metrics
    # Reconstruct full portfolio weights (including unchosen) if needed
    # For metrics, we can do the entire weight vector if needed.
    w_full = np.zeros(len(mu_values))
    idx_chosen = 0
    for i, _ in enumerate(mu_values):
        if i in [monthly_returns.columns.get_loc(a) for a in chosen_assets]:
            w_full[i] = chosen_weights[idx_chosen]
            idx_chosen += 1

    real_return_monthly = mu_values @ w_full
    real_return_annual = (1 + real_return_monthly) ** 12 - 1
    real_variance = w_full @ Sigma_values @ w_full
    portfolio_vol_monthly = np.sqrt(real_variance)
    portfolio_vol_annual = portfolio_vol_monthly * np.sqrt(12)
    sharpe_ratio = (real_return_annual - 0.02) / portfolio_vol_annual  # 2% risk-free

    # ------------------------------------------------------
    # 4. Display results
    # ------------------------------------------------------
    st.subheader("Optimal Portfolio")
    st.write("**Chosen Assets:**", chosen_assets)
    st.write("**Weights:**", {asset: weight for asset, weight in zip(chosen_assets, chosen_weights)})
    st.write(f"**Expected Return (Monthly):** {real_return_monthly:.4%}")
    st.write(f"**Expected Return (Annual):** {real_return_annual:.4%}")
    st.write(f"**Portfolio Volatility (Monthly):** {portfolio_vol_monthly:.4%}")
    st.write(f"**Portfolio Volatility (Annual):** {portfolio_vol_annual:.4%}")
    st.write(f"**Sharpe Ratio (Annual):** {sharpe_ratio:.4f}")

    # ------------------------------------------------------
    # 5. Plot results
    # ------------------------------------------------------
    plot_portfolio_vs_spy(data, spy_data, chosen_assets, chosen_weights)
else:
    st.warning(f"Solver did not find an optimal solution. Status: {result['status']}")
