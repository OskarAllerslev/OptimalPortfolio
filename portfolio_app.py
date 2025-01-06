# Save as `portfolio_app.py` and run with `streamlit run portfolio_app.py`
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
import matplotlib.cm as cm

# ------------------------------------------------------
# Streamlit Application Header
# ------------------------------------------------------
st.title("Portfolio Optimization App")
st.sidebar.header("Input Parameters")

# Sidebar Parameters
lookback_years = st.sidebar.slider("Lookback Period (Years)", 1, 10, 5)
annual_target_return = st.sidebar.slider("Target Annual Return (%)", 1, 30, 10, step=1) / 100
K = st.sidebar.slider("Max Number of Assets in Portfolio", 5, 30, 20)
min_weight = st.sidebar.slider("Minimum Weight for Chosen Assets (%)", 0.0, 5.0, 2.0) / 100

# Convert annual target return to monthly
monthly_target_return = (1 + annual_target_return) ** (1 / 12) - 1

# ------------------------------------------------------
# Download Data
# ------------------------------------------------------
@st.cache_data
def download_data(lookback_years):
    ucits_tickers = [
    "SWDA.L", "SSAC.L", "EMIM.L", "CSP1.L", "VUSA.L", "VUAA.L", "EQQQ.L", "VFEM.L",
    "VWRL.L", "XDWD.L", "IMEU.L", "SWRD.L", "IEMA.L", "VERX.L", "SPY5.L", "IUSA.L",
    "IWVL.L", "IWMO.L", "INRG.L", "HEAL.L", "GOAT.L", "GDX.L", "IUIT.L", "BNKS.L",
    "HMWO.L", "HMEF.L", "AGGG.L", "IBTS.L", "IBTM.L",
    "LQDE.L", "IEMB.L", "IGLH.L", "AGGU.L",
    "SGLN.L", "SSLN.L", "SGLD.L", "SSIL.L",
    "XGIG.L", "IUKP.L", "IWDP.L", "SPY4.L", "VHYL.L",
    "VMID.L", "XLFS.L", "CSPX.L", "V3AA.L", "V3AM.L",
    "SGLP.L"
]
    raw_data = yf.download(ucits_tickers, period=f"{lookback_years}y", interval="1d")["Adj Close"]
    return raw_data

data = download_data(lookback_years)

@st.cache_data
def download_spy_data(lookback_years):
    return yf.download("SPY", period=f"{lookback_years}y", interval="1d")["Adj Close"]

spy_data = download_spy_data(lookback_years)

# ------------------------------------------------------
# Data Cleaning and Preparation
# ------------------------------------------------------
min_non_na = 0.80 * len(data)


# Resample to monthly data and drop remaining NaNs
monthly_data = data.resample("M").last().dropna()



# Calculate monthly returns
monthly_returns = np.log(monthly_data / monthly_data.shift(1)).dropna()
spy_returns = np.log(spy_data / spy_data.shift(1)).dropna()
# Drop tickers with zero variance
monthly_returns = monthly_returns.loc[:, monthly_returns.var() > 0]



if monthly_returns.empty:
    st.error("No valid assets remain after data cleaning (NaNs, zero variance, etc.). "
             "Try reducing lookback years or adjusting filters.")
    st.stop()

# Calculate mean returns and covariance matrix
mu = monthly_returns.mean()
Sigma = monthly_returns.cov()

# Regularize and validate covariance matrix
Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(len(Sigma))

if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
    st.error("Covariance matrix contains invalid values after cleaning. Check input data.")
    st.stop()

# Store the final list of tickers after cleaning
final_tickers = monthly_returns.columns
mu_values = mu.values
Sigma_values = Sigma.values


# ------------------------------------------------------
# Optimization
# ------------------------------------------------------
n = len(final_tickers)
w = cp.Variable(n)
z = cp.Variable(n, boolean=True)
portfolio_variance = cp.quad_form(w, psd_wrap(Sigma_values))
objective = cp.Minimize(portfolio_variance)
constraints = [
    cp.sum(w) == 1,
    w >= 0,
    w @ mu_values >= monthly_target_return,
    cp.sum(z) <= K,
    *[w[i] <= z[i] for i in range(n)],
    *[w[i] >= min_weight * z[i] for i in range(n)],
]
problem = cp.Problem(objective, constraints)
result = problem.solve(solver=cp.ECOS_BB)

# ------------------------------------------------------
# Display Results
# ------------------------------------------------------
if problem.status == "optimal":
    optimal_weights = w.value
    z_values = z.value
    chosen_indices = [i for i in range(n) if z_values[i] > 0.5]
    chosen_assets = [final_tickers[i] for i in chosen_indices]
    chosen_weights = [optimal_weights[i] for i in chosen_indices]
    
    # Portfolio metrics
    real_return_monthly = mu_values @ optimal_weights
    real_return_annual = (1 + real_return_monthly) ** 12 - 1
    real_variance = optimal_weights @ Sigma_values @ optimal_weights
    portfolio_vol_monthly = np.sqrt(real_variance)
    portfolio_vol_annual = portfolio_vol_monthly * np.sqrt(12)
    sharpe_ratio = (real_return_annual - 0.02) / portfolio_vol_annual  # Annual RF of 2%


    st.subheader("Optimal Portfolio")
    st.write("**Chosen Assets:**", chosen_assets)
    st.write("**Weights:**", {asset: weight for asset, weight in zip(chosen_assets, chosen_weights)})
    st.write(f"**Expected Return (Monthly):** {real_return_monthly:.4%}")
    st.write(f"**Expected Return (Annual):** {real_return_annual:.4%}")
    st.write(f"**Portfolio Volatility (Monthly):** {portfolio_vol_monthly:.4%}")
    st.write(f"**Portfolio Volatility (Annual):** {portfolio_vol_annual:.4%}")
    st.write(f"**Sharpe Ratio (Annual):** {sharpe_ratio:.4f}")

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
# ------------------------------------------------------
# Plot: Daily Data with Normalization
# ------------------------------------------------------
# Normalize daily data for chosen assets and SPY
chosen_data_daily = data[chosen_assets].dropna()
spy_data_daily = spy_data.dropna()

normalized_chosen_data_daily = chosen_data_daily / chosen_data_daily.iloc[0]
normalized_spy_data_daily = spy_data_daily / spy_data_daily.iloc[0]

# Calculate portfolio daily performance using daily weights
portfolio_values_daily = (normalized_chosen_data_daily * np.array(chosen_weights)).sum(axis=1)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each chosen asset
for i, asset in enumerate(chosen_assets):
    ax.plot(normalized_chosen_data_daily.index, normalized_chosen_data_daily[asset], 
            label=asset, color=cm.Blues(np.linspace(0.4, 0.9, len(chosen_assets)))[i])

# Plot portfolio and SPY
ax.plot(portfolio_values_daily.index, portfolio_values_daily, 
        label="Portfolio (Weighted)", color="green", linewidth=2.5)
ax.plot(normalized_spy_data_daily.index, normalized_spy_data_daily, 
        label="SPY (Benchmark)", color="orange", linewidth=2)

# Title, labels, and legend
ax.set_title("Portfolio and SPY Performance (Daily Normalized)", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Normalized Value", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.5)

# Display the plot
st.pyplot(fig)
