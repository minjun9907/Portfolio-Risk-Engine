"""Tests for risk/var.py — 5 Value-at-Risk methods."""

import numpy as np
import pandas as pd
import pytest

from risk import VaRResult, VaRMethod
from risk.var import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    cornish_fisher_var,
    evt_var,
)


class TestHistoricalVaR:
    def test_returns_var_result(self, sample_portfolio_returns):
        """Should return a VaRResult dataclass."""
        result = historical_var(sample_portfolio_returns)
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL

    def test_var_is_positive(self, sample_portfolio_returns):
        """VaR should be positive (represents a loss)."""
        result = historical_var(sample_portfolio_returns)
        assert result.var > 0

    def test_higher_confidence_higher_var(self, sample_portfolio_returns):
        """99% VaR should be larger than 95% VaR."""
        var_95 = historical_var(sample_portfolio_returns, confidence=0.95)
        var_99 = historical_var(sample_portfolio_returns, confidence=0.99)
        assert var_99.var > var_95.var

    def test_es_exceeds_var(self, sample_portfolio_returns):
        """Expected Shortfall should be >= VaR."""
        result = historical_var(sample_portfolio_returns)
        assert result.es >= result.var


class TestParametricVaR:
    def test_normal_returns_var_result(self, sample_portfolio_returns):
        """Normal parametric VaR should return VaRResult."""
        result = parametric_var(sample_portfolio_returns, distribution="normal")
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC

    def test_student_t_returns_valid(self, sample_portfolio_returns):
        """Student-t parametric VaR should return valid positive result."""
        result = parametric_var(sample_portfolio_returns, confidence=0.99, distribution="student-t")
        assert isinstance(result, VaRResult)
        assert result.var > 0
        assert result.details["distribution"] == "student-t"
        assert result.details["nu"] > 0

    def test_var_is_positive(self, sample_portfolio_returns):
        """VaR should be positive."""
        result = parametric_var(sample_portfolio_returns)
        assert result.var > 0


class TestMonteCarloVaR:
    def test_returns_var_result(self, sample_returns, sample_portfolio):
        """Should return a VaRResult dataclass."""
        result = monte_carlo_var(sample_returns, sample_portfolio.weights)
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.MONTE_CARLO

    def test_more_sims_reduces_variance(self, sample_returns, sample_portfolio):
        """VaR estimate should stabilize with more simulations."""
        np.random.seed(42)
        results = [
            monte_carlo_var(sample_returns, sample_portfolio.weights, n_sims=5000).var
            for _ in range(3)
        ]
        assert np.std(results) < np.mean(results) * 0.5

    def test_var_is_positive(self, sample_returns, sample_portfolio):
        """VaR should be positive."""
        result = monte_carlo_var(sample_returns, sample_portfolio.weights)
        assert result.var > 0


class TestCornishFisherVaR:
    def test_returns_var_result(self, sample_portfolio_returns):
        """Should return VaRResult with CORNISH_FISHER method."""
        result = cornish_fisher_var(sample_portfolio_returns)
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.CORNISH_FISHER

    def test_close_to_parametric_for_normal_data(self, sample_portfolio_returns):
        """For near-normal data, CF VaR should be close to parametric VaR."""
        cf = cornish_fisher_var(sample_portfolio_returns)
        param = parametric_var(sample_portfolio_returns)
        assert abs(cf.var - param.var) / param.var < 0.15

    def test_details_contain_skew_kurt(self, sample_portfolio_returns):
        """Details dict should include skewness and kurtosis."""
        result = cornish_fisher_var(sample_portfolio_returns)
        assert "skewness" in result.details
        assert "excess_kurtosis" in result.details


class TestEVTVaR:
    def test_returns_var_result(self, sample_portfolio_returns):
        """Should return VaRResult with EVT method."""
        result = evt_var(sample_portfolio_returns)
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.EVT

    def test_var_is_positive(self, sample_portfolio_returns):
        """EVT VaR should be positive."""
        result = evt_var(sample_portfolio_returns)
        assert result.var > 0

    def test_details_contain_gpd_params(self, sample_portfolio_returns):
        """Details should include GPD shape (xi) and scale (beta)."""
        result = evt_var(sample_portfolio_returns)
        assert "xi" in result.details
        assert "beta" in result.details
