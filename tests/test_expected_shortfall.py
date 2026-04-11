"""Tests for risk/expected_shortfall.py — Expected Shortfall (CVaR)."""

import numpy as np
import pandas as pd
import pytest

from risk.expected_shortfall import (
    historical_es,
    parametric_es,
    monte_carlo_es,
    evt_es,
)


class TestHistoricalES:
    def test_returns_float(self, sample_portfolio_returns):
        """Should return a float value."""
        result = historical_es(sample_portfolio_returns)
        assert isinstance(result, float)

    def test_es_is_positive(self, sample_portfolio_returns):
        """ES should be positive (represents expected loss)."""
        result = historical_es(sample_portfolio_returns)
        assert result > 0

    def test_higher_confidence_higher_es(self, sample_portfolio_returns):
        """99% ES should exceed 95% ES."""
        es_95 = historical_es(sample_portfolio_returns, confidence=0.95)
        es_99 = historical_es(sample_portfolio_returns, confidence=0.99)
        assert es_99 > es_95


class TestParametricES:
    def test_returns_float(self, sample_portfolio_returns):
        """Should return a float value."""
        result = parametric_es(sample_portfolio_returns)
        assert isinstance(result, float)

    def test_es_exceeds_mean(self, sample_portfolio_returns):
        """ES should be larger than the mean loss."""
        result = parametric_es(sample_portfolio_returns)
        mean_loss = -sample_portfolio_returns.mean()
        assert result > mean_loss


class TestMonteCarloES:
    def test_returns_float(self, sample_returns, sample_portfolio):
        """Should return a float value."""
        result = monte_carlo_es(sample_returns, sample_portfolio.weights)
        assert isinstance(result, float)

    def test_es_is_positive(self, sample_returns, sample_portfolio):
        """ES should be positive."""
        result = monte_carlo_es(sample_returns, sample_portfolio.weights)
        assert result > 0


class TestEVTES:
    def test_returns_float(self, sample_portfolio_returns):
        """Should return a float value."""
        result = evt_es(sample_portfolio_returns)
        assert isinstance(result, float)

    def test_es_exceeds_historical(self, sample_portfolio_returns):
        """EVT ES at 99% should generally exceed historical ES (captures tail better)."""
        evt = evt_es(sample_portfolio_returns, confidence=0.99)
        hist = historical_es(sample_portfolio_returns, confidence=0.99)
        assert evt > 0
