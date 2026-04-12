"""Tests for risk/factor_model.py — Fama-French regression and attribution."""

import numpy as np
import pandas as pd
import pytest

from risk import FactorExposure
from risk.factor_model import (
    factor_regression,
    factor_risk_attribution,
    sector_concentration,
)


@pytest.fixture
def sample_factors(sample_returns) -> pd.DataFrame:
    """Synthetic 3-factor dataset for deterministic testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.randn(len(sample_returns)) * 0.01,
            "SMB": np.random.randn(len(sample_returns)) * 0.005,
            "HML": np.random.randn(len(sample_returns)) * 0.005,
        },
        index=sample_returns.index,
    )


class TestFactorRegression:
    def test_returns_factor_exposure(self, sample_portfolio_returns, sample_factors):
        """Should return a FactorExposure dataclass."""
        result = factor_regression(sample_portfolio_returns, sample_factors)
        assert isinstance(result, FactorExposure)

    def test_factor_names_match(self, sample_portfolio_returns, sample_factors):
        """Factor names in result should match input columns."""
        result = factor_regression(sample_portfolio_returns, sample_factors)
        assert result.factor_names == list(sample_factors.columns)

    def test_r_squared_in_valid_range(self, sample_portfolio_returns, sample_factors):
        """R-squared should be between 0 and 1."""
        result = factor_regression(sample_portfolio_returns, sample_factors)
        assert 0 <= result.r_squared <= 1

    def test_residual_vol_positive(self, sample_portfolio_returns, sample_factors):
        """Residual volatility should be positive."""
        result = factor_regression(sample_portfolio_returns, sample_factors)
        assert result.residual_vol > 0


class TestFactorRiskAttribution:
    def test_returns_dict(self):
        """Should return a dict with expected keys."""
        weights = [0.25, 0.25, 0.25, 0.25]
        betas = np.array([[1.0, 0.5], [0.8, -0.3], [1.2, 0.1], [0.9, 0.2]])
        factor_cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        result = factor_risk_attribution(weights, betas, factor_cov)
        assert "total_factor_variance" in result
        assert "contributions" in result
        assert "portfolio_betas" in result

    def test_total_variance_nonnegative(self):
        """Total factor variance should be non-negative."""
        weights = [0.5, 0.5]
        betas = np.array([[1.0, 0.3], [0.7, 0.5]])
        factor_cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        result = factor_risk_attribution(weights, betas, factor_cov)
        assert result["total_factor_variance"] >= 0


class TestSectorConcentration:
    @pytest.mark.slow
    def test_returns_dict(self):
        """Should return a dict of sector weights."""
        result = sector_concentration(["AAPL", "MSFT"], [0.5, 0.5])
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_weights_approximately_sum_to_one(self):
        """Sector weights should sum to approximately 1."""
        result = sector_concentration(["AAPL", "MSFT"], [0.5, 0.5])
        assert abs(sum(result.values()) - 1.0) < 1e-6
