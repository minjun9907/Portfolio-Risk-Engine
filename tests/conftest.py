"""Shared pytest fixtures for the Portfolio Risk Engine test suite."""

import numpy as np
import pandas as pd
import pytest

from risk import Portfolio


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """4-asset equal-weight portfolio."""
    return Portfolio(
        tickers=["AAPL", "GOOGL", "MSFT", "JPM"],
        weights=[0.25, 0.25, 0.25, 0.25],
    )


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """500 days of synthetic daily returns (~1% daily vol, seed=42)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    data = np.random.randn(500, 4) * 0.01
    return pd.DataFrame(data, index=dates, columns=["AAPL", "GOOGL", "MSFT", "JPM"])


@pytest.fixture
def sample_portfolio_returns(sample_returns, sample_portfolio) -> pd.Series:
    """Equal-weighted portfolio return series from sample data."""
    weights = np.array(sample_portfolio.weights)
    return sample_returns @ weights


@pytest.fixture
def sample_covariance(sample_returns) -> pd.DataFrame:
    """Sample covariance matrix from synthetic returns."""
    return sample_returns.cov()
