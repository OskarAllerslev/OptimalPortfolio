import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap

# ------------------------------------------------------
# 1. Define UCITS Tickers (no 'BAC' or 'F', etc.)
# ------------------------------------------------------
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

print("Downloading UCITS data...")
raw_data = yf.download(ucits_tickers, period="5y", interval="1d")["Adj Close"]

# ------------------------------------------------------
# 2. Clean & Prepare Data
#    - Drop columns with too many NaNs
#    - Drop leftover NaN rows
#    - Compute monthly log returns
#    - Shrink & symmetrize covariance
# ------------------------------------------------------
# Keep tickers with at least 80% valid data
min_non_na = 0.8 * len(raw_data)
data = raw_data.dropna(axis=1, thresh=int(min_non_na))

# Drop rows still containing NaNs
data = data.dropna(axis=0)

# Resample to monthly
monthly_data = data.resample("M").last()

# Monthly log returns
monthly_returns = np.log(monthly_data / monthly_data.shift(1)).dropna()

mu = monthly_returns.mean()   # monthly average returns
Sigma = monthly_returns.cov() # covariance matrix

# Symmetrize & add small diagonal
Sigma = 0.5*(Sigma + Sigma.T)
epsilon = 1e-8
Sigma += epsilon * np.eye(len(Sigma))

# Convert to arrays
mu_values = mu.values
Sigma_values = Sigma.values
final_tickers = monthly_returns.columns
n = len(final_tickers)

print(f"\nUsed {n} tickers after cleaning:")
print(final_tickers.tolist())

# ------------------------------------------------------
# 3. Mean–Variance Formulation (Convex)
#    Minimize portfolio variance subject to a target return
# ------------------------------------------------------
w = cp.Variable(n)
z = cp.Variable(n, boolean=True)

# For demonstration, pick a monthly return target of, e.g., 1% (~12.7% annual)
target_return = 0.01142

# Objective: Minimize w^T Sigma w
# but we pass Sigma through psd_wrap to avoid ARPACK PSD check issues
Sigma_psd = psd_wrap(Sigma_values)
portfolio_variance = cp.quad_form(w, Sigma_psd)
objective = cp.Minimize(portfolio_variance)

constraints = []
# a) sum of weights == 1
constraints.append(cp.sum(w) == 1)
# b) no short selling
constraints.append(w >= 0)
# c) portfolio return >= target_return
constraints.append(w @ mu_values >= target_return)
# d) cardinality <= K
K = 20
constraints.append(cp.sum(z) <= K)
# e) link w and z
for i in range(n):
    constraints.append(w[i] <= z[i])  # Big-M can be 1.0 or so, if sum(w)=1 => w[i] can't exceed 1 anyway
# f) minimum weight for chosen assets (e.g., 2%)
min_weight = 0.02
for i in range(n):
    constraints.append(w[i] >= min_weight * z[i])

# ------------------------------------------------------
# 4. Solve the Mixed-Integer Quadratic Problem
# ------------------------------------------------------
problem = cp.Problem(objective, constraints)
result = problem.solve(solver=cp.ECOS_BB)

print("\nSolver status:", problem.status)
print("Optimal objective value (portfolio variance):", result)

optimal_weights = w.value
z_values = z.value

chosen_indices = [i for i in range(n) if z_values[i] > 0.5]
chosen_assets = [final_tickers[i] for i in chosen_indices]
chosen_weights = [optimal_weights[i] for i in chosen_indices]

# ------------------------------------------------------
# 5. Final Portfolio Metrics
# ------------------------------------------------------
# We have minimized variance subject to w^T mu >= target_return
real_return = mu_values @ optimal_weights
real_variance = optimal_weights @ Sigma_values @ optimal_weights
portfolio_vol = np.sqrt(real_variance)

print(f"\nActual portfolio return (monthly): {real_return:.4%}")
print(f"Target return (monthly):          {target_return:.4%}")
print(f"Portfolio volatility (monthly):   {portfolio_vol:.4%}")

# Sharpe ratio vs. 2% annual RF => monthly RF ~ 0.001667
monthly_rfr = 0.02 / 12
portfolio_sharpe = (real_return - monthly_rfr) / portfolio_vol
print(f"Portfolio Sharpe ratio (monthly, 2% RF): {portfolio_sharpe:.4f}")

# ------------------------------------------------------
# 6. Compare with SPY (Optional Benchmark)
# ------------------------------------------------------
print("\nDownloading SPY for benchmark comparison...")
spy_data = yf.download(["SPY"], period="5y", interval="1d")["Adj Close"]
spy_monthly = spy_data.resample("M").last()
spy_returns = np.log(spy_monthly / spy_monthly.shift(1)).dropna()

spy_mu_series = spy_returns.mean()
spy_sigma_series = spy_returns.std()
spy_mu = spy_mu_series.item()
spy_sigma = spy_sigma_series.item()

spy_sharpe = (spy_mu - monthly_rfr) / spy_sigma

print(f"SPY monthly return:         {spy_mu:.4%}")
print(f"SPY monthly volatility:     {spy_sigma:.4%}")
print(f"SPY Sharpe ratio (monthly): {spy_sharpe:.4f}")

# ------------------------------------------------------
# 7. Display the Chosen Assets
# ------------------------------------------------------
print("\nChosen assets and weights:")
for asset, weight in zip(chosen_assets, chosen_weights):
    print(f"{asset:15s}  weight = {weight:.4f}")

print(f"\nSum of chosen weights = {sum(chosen_weights):.4f}")
print(f"Number of chosen assets = {len(chosen_assets)}")
