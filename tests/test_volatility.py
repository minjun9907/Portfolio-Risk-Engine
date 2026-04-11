"""Tests for risk/volatility.py — EWMA and GARCH volatility models."""

import numpy as np
import pandas as pd
import pytest

from risk.volatility import ewma_volatility, garch_volatility, compare_vol_models


class TestEWMAVolatility:
    def test_returns_series(self, sample_portfolio_returns):
        """Should return a pandas Series."""
        vol = ewma_volatility(sample_portfolio_returns)
        assert isinstance(vol, pd.Series)

    def test_all_positive(self, sample_portfolio_returns):
        """Volatility estimates should all be positive."""
        vol = ewma_volatility(sample_portfolio_returns)
        assert (vol > 0).all()

    def test_higher_lambda_smoother(self, sample_portfolio_returns):
        """Higher lambda should produce smoother (less variable) vol series."""
        vol_low = ewma_volatility(sample_portfolio_returns, lambda_=0.90)
        vol_high = ewma_volatility(sample_portfolio_returns, lambda_=0.97)
        assert vol_high.std() < vol_low.std()

    def test_same_length_as_input(self, sample_portfolio_returns):
        """Output should have same length as input returns."""
        vol = ewma_volatility(sample_portfolio_returns)
        assert len(vol) == len(sample_portfolio_returns)


class TestGARCHVolatility:
    def test_returns_series(self, sample_portfolio_returns):
        """Should return a pandas Series."""
        vol = garch_volatility(sample_portfolio_returns)
        assert isinstance(vol, pd.Series)

    def test_all_positive(self, sample_portfolio_returns):
        """Volatility estimates should all be positive."""
        vol = garch_volatility(sample_portfolio_returns)
        assert (vol > 0).all()


class TestCompareVolModels:
    def test_returns_three_models(self, sample_portfolio_returns):
        """Should return dict with static, ewma, and garch keys."""
        result = compare_vol_models(sample_portfolio_returns)
        assert set(result.keys()) == {"static", "ewma", "garch"}

    def test_static_is_constant(self, sample_portfolio_returns):
        """Static vol should be a constant series."""
        result = compare_vol_models(sample_portfolio_returns)
        assert result["static"].nunique() == 1
