"""Tests for risk/data.py — data ingestion and return computation."""

import numpy as np
import pandas as pd
import pytest

from risk import Portfolio, CovarianceMethod
from risk.data import fetch_prices, compute_returns, portfolio_returns, compute_covariance


class TestFetchPrices:
    @pytest.mark.slow
    def test_returns_dataframe_with_correct_columns(self):
        """Should return DataFrame with one column per ticker."""
        prices = fetch_prices(["AAPL", "MSFT"], period="1mo")
        assert isinstance(prices, pd.DataFrame)
        assert list(prices.columns) == ["AAPL", "MSFT"]

    @pytest.mark.slow
    def test_no_missing_values(self):
        """Returned prices should have no NaN values."""
        prices = fetch_prices(["AAPL"], period="1mo")
        assert not prices.isna().any().any()


class TestComputeReturns:
    def test_log_returns_shape(self):
        """Log returns should have one fewer row than prices."""
        prices = pd.DataFrame({"A": [100, 101, 102]})
        returns = compute_returns(prices, method="log")
        assert len(returns) == 2

    def test_simple_returns_known_value(self):
        """Simple return of 100 -> 110 should be 0.10."""
        prices = pd.DataFrame({"A": [100.0, 110.0]})
        returns = compute_returns(prices, method="simple")
        assert abs(returns.iloc[0, 0] - 0.10) < 1e-10

    def test_log_returns_known_value(self):
        """Log return of 100 -> 110 should be ln(1.10)."""
        prices = pd.DataFrame({"A": [100.0, 110.0]})
        returns = compute_returns(prices, method="log")
        assert abs(returns.iloc[0, 0] - np.log(1.10)) < 1e-10


class TestPortfolioReturns:
    def test_equal_weight_is_mean(self, sample_returns):
        """Equal-weight portfolio return should equal row mean."""
        weights = [0.25, 0.25, 0.25, 0.25]
        port_ret = portfolio_returns(sample_returns, weights)
        expected = sample_returns.mean(axis=1)
        pd.testing.assert_series_equal(port_ret, expected, check_names=False)

    def test_single_asset_portfolio(self, sample_returns):
        """100% weight in one asset should return that asset's returns."""
        weights = [1.0, 0.0, 0.0, 0.0]
        port_ret = portfolio_returns(sample_returns, weights)
        pd.testing.assert_series_equal(port_ret, sample_returns.iloc[:, 0], check_names=False)


class TestComputeCovariance:
    def test_sample_covariance_is_symmetric(self, sample_returns):
        """Sample covariance matrix should be symmetric."""
        cov = compute_covariance(sample_returns, method=CovarianceMethod.SAMPLE)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_covariance_shape(self, sample_returns):
        """Covariance should be n_assets x n_assets."""
        cov = compute_covariance(sample_returns)
        assert cov.shape == (4, 4)

    def test_ewma_covariance_differs_from_sample(self, sample_returns):
        """EWMA covariance should differ from sample covariance."""
        cov_sample = compute_covariance(sample_returns, method=CovarianceMethod.SAMPLE)
        cov_ewma = compute_covariance(sample_returns, method=CovarianceMethod.EWMA)
        assert not np.allclose(cov_sample.values, cov_ewma.values)
