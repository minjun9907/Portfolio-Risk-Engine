"""Value-at-Risk computation: 5 methods.

Implements Historical, Parametric (normal/Student-t), Monte Carlo,
Cornish-Fisher (skewness/kurtosis adjusted), and EVT (Peaks-Over-Threshold
with Generalized Pareto Distribution).
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import VaRResult, VaRMethod


def historical_var(returns: pd.Series, confidence: float = 0.95) -> VaRResult:
    """Historical simulation VaR.

    Non-parametric approach using the empirical distribution of returns.
    VaR is the (1 - confidence) quantile of the loss distribution.
    """
    losses = -returns
    var = float(np.percentile(losses, confidence * 100))
    es = float(losses[losses >= var].mean())

    return VaRResult(
        var=var,
        es=es,
        confidence=confidence,
        method=VaRMethod.HISTORICAL,
        details={"n_observations": len(returns)},
    )


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    distribution: str = "normal",
) -> VaRResult:
    """Parametric (variance-covariance) VaR.

    Assumes returns follow a known distribution:
        normal: VaR = -mu + z_alpha * sigma
        student-t: VaR = -mu + t_alpha(nu) * sigma * sqrt((nu-2)/nu)

    Args:
        distribution: "normal" or "student-t".
    """
    mu = returns.mean()
    sigma = returns.std()

    if distribution == "normal":
        z = stats.norm.ppf(confidence)
        var = -mu + z * sigma
        es = -mu + sigma * stats.norm.pdf(z) / (1 - confidence)

        return VaRResult(
            var=float(var),
            es=float(es),
            confidence=confidence,
            method=VaRMethod.PARAMETRIC,
            details={"distribution": "normal", "mu": float(mu), "sigma": float(sigma)},
        )

    elif distribution == "student-t":
        # Fit Student-t to estimate degrees of freedom
        nu, loc, scale = stats.t.fit(returns)
        t_quantile = stats.t.ppf(confidence, df=nu)
        var = -loc + t_quantile * scale
        # ES for Student-t: E[X | X > VaR]
        es_factor = (stats.t.pdf(t_quantile, df=nu) / (1 - confidence)) * ((nu + t_quantile ** 2) / (nu - 1))
        es = -loc + scale * es_factor

        return VaRResult(
            var=float(var),
            es=float(es),
            confidence=confidence,
            method=VaRMethod.PARAMETRIC,
            details={"distribution": "student-t", "nu": float(nu), "loc": float(loc), "scale": float(scale)},
        )

    else:
        raise ValueError(f"distribution must be 'normal' or 'student-t', got '{distribution}'")


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: list[float],
    confidence: float = 0.95,
    n_sims: int = 10_000,
) -> VaRResult:
    """Monte Carlo VaR with correlated simulations.

    Generates correlated random returns via Cholesky decomposition
    of the covariance matrix, then computes portfolio VaR from
    the simulated P&L distribution.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    w = np.array(weights)

    # Cholesky decomposition for correlated samples
    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal((n_sims, len(w)))
    sim_returns = z @ L.T + mu

    # Portfolio P&L
    port_returns = sim_returns @ w
    losses = -port_returns
    var = float(np.percentile(losses, confidence * 100))
    es = float(losses[losses >= var].mean())

    return VaRResult(
        var=var,
        es=es,
        confidence=confidence,
        method=VaRMethod.MONTE_CARLO,
        details={"n_sims": n_sims},
    )


def cornish_fisher_var(returns: pd.Series, confidence: float = 0.95) -> VaRResult:
    """Cornish-Fisher expansion VaR.

    Adjusts the normal quantile for skewness (S) and excess kurtosis (K):
        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3*z)*K/24 - (2*z^3 - 5*z)*S^2/36

    Lightweight tail adjustment without full distribution fitting.
    """
    mu = returns.mean()
    sigma = returns.std()
    S = float(stats.skew(returns))
    K = float(stats.kurtosis(returns))  # excess kurtosis

    z = stats.norm.ppf(confidence)
    z_cf = (
        z
        + (z ** 2 - 1) * S / 6
        + (z ** 3 - 3 * z) * K / 24
        - (2 * z ** 3 - 5 * z) * S ** 2 / 36
    )

    var = -mu + z_cf * sigma

    # ES approximation: use historical ES as fallback
    losses = -returns
    historical_threshold = float(np.percentile(losses, confidence * 100))
    tail = losses[losses >= historical_threshold]
    es = float(tail.mean()) if len(tail) > 0 else var

    return VaRResult(
        var=float(var),
        es=es,
        confidence=confidence,
        method=VaRMethod.CORNISH_FISHER,
        details={"skewness": S, "excess_kurtosis": K, "z_cf": float(z_cf)},
    )


def evt_var(
    returns: pd.Series,
    confidence: float = 0.99,
    threshold_quantile: float = 0.90,
) -> VaRResult:
    """Extreme Value Theory VaR using Peaks-Over-Threshold.

    Fits a Generalized Pareto Distribution (GPD) to exceedances
    above a high threshold, then extrapolates to the desired
    confidence level.

    Args:
        threshold_quantile: Quantile for the POT threshold (e.g. 0.90
            uses the 90th percentile of losses as the threshold).
    """
    losses = -returns
    n = len(losses)
    u = float(np.percentile(losses, threshold_quantile * 100))

    # Exceedances over threshold
    exceedances = losses[losses > u] - u
    n_exceed = len(exceedances)

    # Fit GPD to exceedances via MLE
    xi, loc, beta = stats.genpareto.fit(exceedances, floc=0)

    # VaR using GPD tail estimate
    # VaR_p = u + (beta / xi) * ((n / n_exceed * (1 - confidence))^(-xi) - 1)
    ratio = n / n_exceed * (1 - confidence)
    var = u + (beta / xi) * (ratio ** (-xi) - 1)

    # ES using GPD
    # ES = VaR / (1 - xi) + (beta - xi * u) / (1 - xi)
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi)

    return VaRResult(
        var=float(var),
        es=float(es),
        confidence=confidence,
        method=VaRMethod.EVT,
        details={
            "xi": float(xi),
            "beta": float(beta),
            "threshold": float(u),
            "n_exceedances": n_exceed,
        },
    )
