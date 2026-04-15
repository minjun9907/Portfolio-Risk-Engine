"""VaR decomposition: marginal, component, and incremental VaR.

Answers the question 'where is the risk coming from?' by attributing
portfolio VaR to individual positions. Uses Euler's theorem to ensure
component contributions sum exactly to total VaR.
"""

import numpy as np
import pandas as pd
from scipy import stats


def marginal_var(
    returns: pd.DataFrame,
    weights: list[float],
    confidence: float = 0.95,
) -> np.ndarray:
    """Marginal VaR: sensitivity of portfolio VaR to each weight.

    Marginal VaR_i = z_alpha * (Sigma * w)_i / sigma_p
    where sigma_p is the portfolio volatility.

    Interpreted as: the change in VaR if weight_i increases by 1 unit.
    """
    cov = returns.cov().values
    w = np.array(weights)
    port_vol = float(np.sqrt(w @ cov @ w))
    z = stats.norm.ppf(confidence)

    return z * (cov @ w) / port_vol


def component_var(
    returns: pd.DataFrame,
    weights: list[float],
    confidence: float = 0.95,
) -> np.ndarray:
    """Component VaR: contribution of each position to total VaR.

    Component VaR_i = w_i * Marginal VaR_i

    By Euler's theorem on homogeneous functions, component VaRs sum
    exactly to total portfolio VaR. This is the key property used for
    risk budgeting and position limit allocation.
    """
    w = np.array(weights)
    mvar = marginal_var(returns, weights, confidence)
    return w * mvar


def incremental_var(
    returns: pd.DataFrame,
    weights: list[float],
    confidence: float = 0.95,
) -> np.ndarray:
    """Incremental VaR: risk impact of removing each position entirely.

    IVaR_i = VaR(full portfolio) - VaR(portfolio without position i)

    Unlike component VaR, incremental VaR does not sum to total VaR —
    it measures what happens if a position is liquidated.
    """
    w = np.array(weights)
    cov = returns.cov().values
    z = stats.norm.ppf(confidence)

    total_var = z * float(np.sqrt(w @ cov @ w))

    ivars = np.zeros(len(w))
    for i in range(len(w)):
        # Portfolio with position i removed, weights renormalized
        w_minus = w.copy()
        w_minus[i] = 0
        s = w_minus.sum()
        if s > 0:
            w_minus /= s
        var_minus = z * float(np.sqrt(w_minus @ cov @ w_minus))
        ivars[i] = total_var - var_minus

    return ivars
