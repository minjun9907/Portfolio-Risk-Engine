"""Stress testing — historical, hypothetical, and reverse stress tests.

Historical scenarios replay major crises on the current portfolio.
Hypothetical scenarios apply custom shocks. Reverse stress tests find
the scenario that would cause a target loss, a Basel III requirement.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import Portfolio, StressResult


HISTORICAL_SCENARIOS: dict[str, dict[str, float]] = {
    # Approximate single-day shocks by asset class during major episodes
    "2008_gfc": {
        "equity": -0.09,
        "credit": -0.05,
        "rates": 0.015,
        "vol": 2.5,
    },
    "2020_covid": {
        "equity": -0.12,
        "credit": -0.04,
        "rates": -0.005,
        "vol": 3.0,
    },
    "2022_rate_hikes": {
        "equity": -0.04,
        "credit": -0.02,
        "rates": 0.004,
        "vol": 1.5,
    },
    "2023_svb": {
        "equity": -0.02,
        "credit": -0.03,
        "rates": -0.002,
        "vol": 1.8,
    },
}


def hypothetical_scenario(
    portfolio: Portfolio,
    shocks: dict[str, float],
    scenario_name: str = "custom",
) -> StressResult:
    """Apply custom per-asset shocks and compute portfolio P&L.

    Args:
        portfolio: Portfolio to stress.
        shocks: Dict mapping ticker -> return shock (e.g. -0.10 = -10%).
            Tickers not in the dict are assumed unshocked.

    Returns:
        StressResult with total and per-asset P&L contributions.
    """
    asset_pnls: dict[str, float] = {}
    for ticker, weight in zip(portfolio.tickers, portfolio.weights):
        shock = shocks.get(ticker, 0.0)
        asset_pnls[ticker] = float(weight * shock)

    total = sum(asset_pnls.values())
    return StressResult(
        scenario_name=scenario_name,
        portfolio_pnl=float(total),
        asset_pnls=asset_pnls,
    )


def sensitivity_analysis(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    weight_perturbation: float = 0.05,
) -> pd.DataFrame:
    """How much does portfolio volatility change when each weight shifts?

    For each asset, increase its weight by the perturbation amount
    (rebalancing others proportionally) and compute the resulting change
    in portfolio volatility. Highlights which positions dominate risk.
    """
    cov = returns.cov().values
    w = np.array(portfolio.weights)
    base_vol = float(np.sqrt(w @ cov @ w))

    results = []
    for i, ticker in enumerate(portfolio.tickers):
        w_new = w.copy()
        w_new[i] += weight_perturbation
        # Normalize to sum to 1
        w_new /= w_new.sum()
        new_vol = float(np.sqrt(w_new @ cov @ w_new))
        results.append(
            {
                "ticker": ticker,
                "base_vol": base_vol,
                "new_vol": new_vol,
                "vol_change": new_vol - base_vol,
            }
        )

    return pd.DataFrame(results)


def reverse_stress_test(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    target_loss: float = -0.20,
) -> dict[str, float]:
    """Find the smallest coordinated shock that causes a target loss.

    Solves for shock vector s minimizing ||s|| subject to w' * s = target_loss.
    The optimal solution is the shock direction closest to the portfolio's
    weight vector, giving the "most plausible" disaster scenario.
    """
    w = np.array(portfolio.weights)
    n = len(w)

    def objective(shocks: np.ndarray) -> float:
        return float(np.sum(shocks ** 2))

    constraints = [{"type": "eq", "fun": lambda s: float(w @ s) - target_loss}]
    x0 = np.ones(n) * target_loss

    result = minimize(objective, x0, method="SLSQP", constraints=constraints)
    return {ticker: float(s) for ticker, s in zip(portfolio.tickers, result.x)}
