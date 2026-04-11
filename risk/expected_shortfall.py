"""Expected Shortfall (Conditional VaR) computation.

ES measures the average loss in the tail beyond VaR — the expected loss
given that VaR has been breached. Preferred over VaR by Basel III for
its coherence and sub-additivity properties.
"""

import numpy as np
import pandas as pd
from scipy import stats


def historical_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall.

    Average of all losses beyond the historical VaR threshold.
    """
    losses = -returns
    var_threshold = float(np.percentile(losses, confidence * 100))
    tail_losses = losses[losses >= var_threshold]
    return float(tail_losses.mean())


def parametric_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric Expected Shortfall assuming normal distribution.

    ES = -mu + sigma * phi(z_alpha) / (1 - alpha)
    where phi is the standard normal PDF and z_alpha is the normal quantile.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(confidence)
    return float(-mu + sigma * stats.norm.pdf(z) / (1 - confidence))


def monte_carlo_es(
    returns: pd.DataFrame,
    weights: list[float],
    confidence: float = 0.95,
    n_sims: int = 10_000,
) -> float:
    """Monte Carlo Expected Shortfall.

    Average of simulated portfolio losses beyond the Monte Carlo VaR.
    Uses Cholesky decomposition for correlated return generation.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    w = np.array(weights)

    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal((n_sims, len(w)))
    sim_returns = z @ L.T + mu

    port_returns = sim_returns @ w
    losses = -port_returns
    var_threshold = float(np.percentile(losses, confidence * 100))
    tail_losses = losses[losses >= var_threshold]
    return float(tail_losses.mean())


def evt_es(
    returns: pd.Series,
    confidence: float = 0.99,
    threshold_quantile: float = 0.90,
) -> float:
    """EVT Expected Shortfall using GPD tail estimate.

    ES_GPD = VaR_GPD / (1 - xi) + (beta - xi * u) / (1 - xi)
    where xi is the shape parameter, beta the scale, and u the threshold.
    """
    losses = -returns
    n = len(losses)
    u = float(np.percentile(losses, threshold_quantile * 100))

    exceedances = losses[losses > u] - u
    n_exceed = len(exceedances)

    xi, loc, beta = stats.genpareto.fit(exceedances, floc=0)

    ratio = n / n_exceed * (1 - confidence)
    var = u + (beta / xi) * (ratio ** (-xi) - 1)
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    return float(es)
