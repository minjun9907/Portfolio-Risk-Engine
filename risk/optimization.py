"""Portfolio optimization.

Implements classic Markowitz optimization, max-Sharpe tangency portfolio,
risk parity (equal risk contribution), and the efficient frontier.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import Portfolio


def min_variance_portfolio(returns: pd.DataFrame) -> Portfolio:
    """Global minimum variance portfolio.

    Solves: min w' * Sigma * w  subject to sum(w) = 1, w >= 0.
    """
    cov = returns.cov().values
    n = cov.shape[0]

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(portfolio_variance, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return Portfolio(
        tickers=list(returns.columns),
        weights=result.x.tolist(),
        name="Min Variance",
    )


def max_sharpe_portfolio(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> Portfolio:
    """Maximum Sharpe ratio (tangency) portfolio.

    Solves: max (mu' * w - rf) / sqrt(w' * Sigma * w)  subject to sum(w) = 1, w >= 0.
    Equivalent to minimizing the negative Sharpe ratio.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 0.0
        return -(port_return - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return Portfolio(
        tickers=list(returns.columns),
        weights=result.x.tolist(),
        name="Max Sharpe",
    )


def risk_parity_portfolio(returns: pd.DataFrame) -> Portfolio:
    """Risk parity — each asset contributes equal risk.

    Minimizes the variance of risk contributions:
        RC_i = w_i * (Sigma * w)_i
    Target: RC_i = total_variance / n for all i.
    """
    cov = returns.cov().values
    n = cov.shape[0]

    def risk_parity_objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib
        target = port_var / n
        return float(np.sum((risk_contrib - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.001, 1.0)] * n  # avoid zero weights for stability
    x0 = np.ones(n) / n

    result = minimize(risk_parity_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return Portfolio(
        tickers=list(returns.columns),
        weights=result.x.tolist(),
        name="Risk Parity",
    )


def efficient_frontier(returns: pd.DataFrame, n_points: int = 20) -> list[Portfolio]:
    """Compute the efficient frontier by sweeping target returns.

    For each target return, find the minimum-variance portfolio that achieves it.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    n = len(mu)

    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    frontier = []

    for target in target_returns:
        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(portfolio_variance, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            frontier.append(
                Portfolio(
                    tickers=list(returns.columns),
                    weights=result.x.tolist(),
                    name=f"EF target={target:.4f}",
                )
            )

    return frontier
