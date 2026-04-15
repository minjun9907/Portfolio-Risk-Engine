"""Tests for risk/var_decomposition.py — marginal, component, and incremental VaR."""

import numpy as np
import pandas as pd
import pytest

from risk.var_decomposition import marginal_var, component_var, incremental_var


class TestMarginalVaR:
    def test_returns_array(self, sample_returns, sample_portfolio):
        """Should return a numpy array with length = number of assets."""
        result = marginal_var(sample_returns, sample_portfolio.weights)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_portfolio.weights)

    def test_all_finite(self, sample_returns, sample_portfolio):
        """All marginal VaR values should be finite numbers."""
        result = marginal_var(sample_returns, sample_portfolio.weights)
        assert np.all(np.isfinite(result))


class TestComponentVaR:
    def test_sums_to_total_var(self, sample_returns, sample_portfolio):
        """Component VaRs should sum to total parametric VaR (Euler's theorem)."""
        from scipy import stats

        components = component_var(sample_returns, sample_portfolio.weights)
        w = np.array(sample_portfolio.weights)
        cov = sample_returns.cov().values
        z = stats.norm.ppf(0.95)
        total_var = z * float(np.sqrt(w @ cov @ w))

        assert abs(components.sum() - total_var) < 1e-10

    def test_length_matches_portfolio(self, sample_returns, sample_portfolio):
        """One component per asset."""
        result = component_var(sample_returns, sample_portfolio.weights)
        assert len(result) == len(sample_portfolio.weights)


class TestIncrementalVaR:
    def test_returns_array(self, sample_returns, sample_portfolio):
        """Should return a numpy array with one value per asset."""
        result = incremental_var(sample_returns, sample_portfolio.weights)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_portfolio.weights)

    def test_all_finite(self, sample_returns, sample_portfolio):
        """All incremental VaR values should be finite."""
        result = incremental_var(sample_returns, sample_portfolio.weights)
        assert np.all(np.isfinite(result))
