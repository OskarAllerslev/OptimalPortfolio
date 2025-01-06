import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.wraps import psd_wrap

def optimize_portfolio(
    monthly_returns, 
    monthly_target_return,
    K=20,
    min_weight=0.0,
    method="mean_variance",  # or "robust", "cvar"
    alpha=0.95,
    drawdown_constraint=False,
    max_drawdown=0.1,
    scenarios=None
):
    """
    A flexible optimization that can do:
      - standard mean-variance ("mean_variance"),
      - robust mean-variance ("robust"),
      - CVaR ("cvar").
    Also can optionally add a (basic) drawdown constraint if 'drawdown_constraint=True'.
    
    Args:
        monthly_returns: pd.DataFrame of monthly returns. shape: (T, N)
        monthly_target_return: float, target monthly return.
        K: max number of assets
        min_weight: minimum weight for chosen assets
        method: which approach to use ("mean_variance", "robust", "cvar")
        alpha: significance for CVaR, e.g., 0.95
        drawdown_constraint: bool. If True, tries to constrain drawdown
        max_drawdown: maximum allowed drawdown in scenario approach (0.1 -> 10%)
        scenarios: (optional) if you want scenario-based logic for CVaR or drawdown

    Returns:
        dict with keys: "status", "chosen_assets", "chosen_weights", "mu", "Sigma"
    """

    final_tickers = monthly_returns.columns
    n = len(final_tickers)

    # Basic stats
    mu = monthly_returns.mean().values  # shape (n,)
    Sigma = monthly_returns.cov().values  # shape (n,n)

    # "Regularize" Sigma
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(n)

    # Vars
    w = cp.Variable(n)
    z = cp.Variable(n, boolean=True)  # for cardinality constraints

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.sum(z) <= K,
        *[w[i] <= z[i] for i in range(n)],
        *[w[i] >= min_weight * z[i] for i in range(n)]
    ]

    # 1. Basic or robust mean-variance
    if method in ["mean_variance", "robust"]:
        # For mean-variance, objective = Minimize w^T Sigma w
        # For robust, we can do a small "shrink" on mu or inflate Sigma, etc.
        # Let's do a simple approach: if robust, shrink mu by some fraction
        # or just define a "worst-case mu" approach.
        if method == "robust":
            # e.g., shrink mu by a fraction (0.5) to emulate uncertainty
            # or you could do an ellipsoidal set.
            # For demonstration, let's just do a simple shrink:
            mu_robust = 0.5 * mu
            constraints += [w @ mu_robust >= monthly_target_return]
        else:
            # Normal mean-variance
            constraints += [w @ mu >= monthly_target_return]

        portfolio_variance = cp.quad_form(w, psd_wrap(Sigma))
        objective = cp.Minimize(portfolio_variance)

    # 2. CVaR approach
    elif method == "cvar":
        # We need scenario-based data to do CVaR properly. 
        # If user didn't pass 'scenarios', we can build them from the monthly_returns itself.
        # Let's interpret each row as one scenario of returns.
        if scenarios is None:
            scenarios = monthly_returns.values  # shape (T, N)

        T = scenarios.shape[0]
        # We want to ensure w meets the target return on average, or something similar.
        # Actually, for CVaR, we typically do *loss* = negative returns. 
        # So let's define scenario losses as L_i = -r_i^T w.

        # We'll also add a constraint that the expected return is >= monthly_target_return 
        # => w @ avg(mu) >= monthly_target_return
        constraints += [w @ mu >= monthly_target_return]

        # CVaR alpha
        xi = cp.Variable(T, nonneg=True)
        zeta = cp.Variable()  # scalar
        # L_i(w) = -(scenario[i] @ w). i.e. negative of realized return
        # We'll do: L_i(w) = -scenarios[i].dot(w)
        # Our objective is to minimize zeta + (1 / (1-alpha)*T) * sum(xi_i)
        # subject to xi_i >= L_i(w) - zeta, xi_i >= 0
        objective = cp.Minimize(zeta + 1.0 / ((1 - alpha) * T) * cp.sum(xi))

        # Add those constraints
        for i in range(T):
            loss_i = - scenarios[i, :] @ w
            constraints += [xi[i] >= loss_i - zeta]

    # 3. Else unknown method
    else:
        return {
            "status": f"Unknown method {method}. Choose from mean_variance, robust, cvar."
        }

    # (Optional) 4. Add a basic drawdown constraint if desired
    if drawdown_constraint:
        # We'll do a scenario-based approach, which again requires daily or monthly scenario returns.
        # If not given, let's approximate from monthly_returns. 
        # We'll treat each row as a scenario (like above).
        if scenarios is None:
            scenarios = monthly_returns.values  # shape (T, N)
        T = scenarios.shape[0]

        # Let's define a portfolio value time-series: V_0=1, then V_t = V_{t-1} * (1 + r_{t}^p).
        # We'll approximate the portfolio return in scenario row t as w^T scenarios[t].
        # This is a single path approach, but let's treat each row as consecutive time steps 
        # for demonstration. Real approach might need multiple scenario paths.

        # Create a variable for portfolio value each time-step
        V = cp.Variable(T+1, nonneg=True)
        # Constrain V[0] = 1 (start with 1)
        constraints += [V[0] == 1]

        for t in range(1, T+1):
            # V[t] = V[t-1] * (1 + w^T r_t)
            constraints += [V[t] == V[t-1] * (1 + scenarios[t-1, :] @ w)]

        # Now define drawdown_t = max_{0..t}(V[s]) - V[t]
        # In practice, we need big-M or an additional variable + constraints to track the running max.
        # We'll define M[t], the running max up to t.
        M = cp.Variable(T+1, nonneg=True)
        constraints += [M[0] == 1]
        for t in range(1, T+1):
            constraints += [M[t] >= M[t-1], M[t] >= V[t]]

        # The drawdown at time t is D[t] = M[t] - V[t].
        # We want all D[t] <= max_drawdown * M[t].
        # or we want (M[t] - V[t]) / M[t] <= max_drawdown => 1 - V[t]/M[t] <= max_drawdown => V[t]/M[t] >= 1 - max_drawdown
        # We'll do the simpler: M[t] - V[t] <= max_drawdown (assuming M[t] = 1 baseline). 
        # But that might not scale. Let's do the ratio version (need a nonlinear constraint though).
        # For demonstration, let's do a linear approach: M[t] - V[t] <= max_drawdown. 
        # That implies an absolute drawdown limit, not a % drawdown. 
        # If M[t] can grow above 1, we might want a different approach. 
        # We'll do absolute for simplicity:

        for t in range(T+1):
            constraints += [M[t] - V[t] <= max_drawdown]

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        result = problem.solve(solver=cp.ECOS_BB)  # or cp.MOSEK, etc.
    except Exception as e:
        return {
            "status": f"Error solving problem: {str(e)}"
        }

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return {
            "status": f"Solver did not find an optimal solution. Status: {problem.status}"
        }

    # Gather results
    optimal_weights = w.value
    z_values = z.value if z.value is not None else np.ones(n)
    chosen_indices = [i for i in range(n) if z_values[i] > 0.5]
    chosen_assets = [final_tickers[i] for i in chosen_indices]
    chosen_weights = [optimal_weights[i] for i in chosen_indices]

    return {
        "status": "optimal",
        "chosen_assets": chosen_assets,
        "chosen_weights": chosen_weights,
        "mu": mu,
        "Sigma": Sigma
    }
