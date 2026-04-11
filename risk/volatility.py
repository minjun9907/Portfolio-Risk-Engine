"""Volatility modeling: EWMA and GARCH(1,1).

Provides time-varying volatility estimates that capture volatility clustering,
a key input for conditional VaR models.
"""

import numpy as np
import pandas as pd
from arch import arch_model


def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """Exponentially weighted moving average volatility.

    Applies the RiskMetrics EWMA model:
        sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

    Args:
        returns: Daily return series.
        lambda_: Decay factor (default 0.94 per RiskMetrics).

    Returns:
        Series of conditional volatility estimates.
    """
    variance = pd.Series(np.zeros(len(returns)), index=returns.index)
    variance.iloc[0] = returns.var()

    for t in range(1, len(returns)):
        variance.iloc[t] = lambda_ * variance.iloc[t - 1] + (1 - lambda_) * returns.iloc[t - 1] ** 2

    return np.sqrt(variance)


def garch_volatility(returns: pd.Series) -> pd.Series:
    """GARCH(1,1) conditional volatility via maximum likelihood.

    Fits a GARCH(1,1) model using the arch library:
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Returns:
        Series of conditional volatility estimates.
    """
    scaled = returns * 100  # arch library expects percentage returns
    model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
    result = model.fit(disp="off")
    cond_vol = result.conditional_volatility / 100  # scale back
    return cond_vol


def compare_vol_models(returns: pd.Series) -> dict[str, pd.Series]:
    """Compare static, EWMA, and GARCH volatility models.

    Returns:
        Dict mapping model name to its volatility series:
        {"static": ..., "ewma": ..., "garch": ...}
    """
    static_vol = pd.Series(returns.std(), index=returns.index)
    ewma_vol = ewma_volatility(returns)
    garch_vol = garch_volatility(returns)

    return {
        "static": static_vol,
        "ewma": ewma_vol,
        "garch": garch_vol,
    }
