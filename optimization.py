import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.wraps import psd_wrap

def optimize_portfolio(
    monthly_returns, 
    target_monthly_return, 
    K, 
    min_weight
):
    """
    Run the portfolio optimization with the user constraints:
      - Sum of weights = 1
      - Weights >= 0
      - Weighted monthly return >= target_monthly_return
      - At most K assets
      - Each chosen asset has weight >= min_weight
    Returns the chosen assets, optimal weights, and stats.
    """
    # Calculate mean returns and covariance
    mu = monthly_returns.mean()
    Sigma = monthly_returns.cov()

    # Regularize / ensure symmetrical
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(len(Sigma))

    # Final set of tickers
    final_tickers = monthly_returns.columns
    n = len(final_tickers)

    # Define variables
    w = cp.Variable(n)
    z = cp.Variable(n, boolean=True)

    # Objective: Minimize variance
    portfolio_variance = cp.quad_form(w, psd_wrap(Sigma))
    objective = cp.Minimize(portfolio_variance)

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w @ mu.values >= target_monthly_return,
        cp.sum(z) <= K,
        *[w[i] <= z[i] for i in range(n)],
        *[w[i] >= min_weight * z[i] for i in range(n)],
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.ECOS_BB)

    # If the solution is optimal, gather results
    if problem.status == "optimal":
        optimal_weights = w.value
        z_values = z.value
        chosen_indices = [i for i in range(n) if z_values[i] > 0.5]
        chosen_assets = [final_tickers[i] for i in chosen_indices]
        chosen_weights = [optimal_weights[i] for i in chosen_indices]

        return {
            "status": "optimal",
            "chosen_assets": chosen_assets,
            "chosen_weights": chosen_weights,
            "mu": mu.values,    # used for stats
            "Sigma": Sigma.values
        }
    else:
        return {
            "status": problem.status
        }
