"""Data ingestion and return computation.

Handles price fetching via yfinance, return calculation (log/simple),
portfolio return aggregation, and covariance matrix estimation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

from . import Portfolio, CovarianceMethod


def fetch_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Fetch adjusted close prices for given tickers.

    Uses yfinance to download historical price data.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices[tickers].dropna()
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute asset returns from price series.

    Args:
        prices: DataFrame of adjusted close prices.
        method: "log" for log returns, "simple" for arithmetic returns.

    Returns:
        DataFrame of daily returns with first row dropped.
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        return prices.pct_change().dropna()
    else:
        raise ValueError(f"method must be 'log' or 'simple', got '{method}'")


def portfolio_returns(returns: pd.DataFrame, weights: list[float]) -> pd.Series:
    """Compute weighted portfolio return series."""
    w = np.array(weights)
    return returns @ w


def compute_covariance(
    returns: pd.DataFrame,
    method: CovarianceMethod = CovarianceMethod.SAMPLE,
    **kwargs,
) -> pd.DataFrame:
    """Estimate covariance matrix of asset returns.

    Methods:
        SAMPLE: Standard sample covariance.
        EWMA: Exponentially weighted with decay factor lambda_ (default 0.94).
        LEDOIT_WOLF: Shrinkage estimator for better conditioning.
    """
    if method == CovarianceMethod.SAMPLE:
        return returns.cov()

    elif method == CovarianceMethod.EWMA:
        lambda_ = kwargs.get("lambda_", 0.94)
        n = len(returns)
        weights = np.array([(1 - lambda_) * lambda_ ** i for i in range(n - 1, -1, -1)])
        weights /= weights.sum()
        demeaned = returns.values - returns.values.mean(axis=0)
        # Element-wise multiply each row by its weight, then matrix multiply
        # Avoids creating a full N x N diagonal matrix
        cov = (demeaned * weights[:, np.newaxis]).T @ demeaned
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

    elif method == CovarianceMethod.LEDOIT_WOLF:
        lw = LedoitWolf().fit(returns.values)
        return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

    else:
        raise ValueError(f"Unknown covariance method: {method}")
