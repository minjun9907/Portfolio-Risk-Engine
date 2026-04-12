"""Tests for risk/optimization.py — portfolio optimization."""

import numpy as np
import pytest

from risk import Portfolio
from risk.optimization import (
    min_variance_portfolio,
    max_sharpe_portfolio,
    risk_parity_portfolio,
    efficient_frontier,
)


class TestMinVariancePortfolio:
    def test_returns_portfolio(self, sample_returns):
        """Should return a Portfolio dataclass."""
        p = min_variance_portfolio(sample_returns)
        assert isinstance(p, Portfolio)

    def test_weights_sum_to_one(self, sample_returns):
        """Weights should sum to 1."""
        p = min_variance_portfolio(sample_returns)
        assert abs(sum(p.weights) - 1.0) < 1e-6

    def test_weights_nonnegative(self, sample_returns):
        """All weights should be non-negative (long-only)."""
        p = min_variance_portfolio(sample_returns)
        assert all(w >= -1e-8 for w in p.weights)


class TestMaxSharpePortfolio:
    def test_returns_portfolio(self, sample_returns):
        """Should return a Portfolio dataclass."""
        p = max_sharpe_portfolio(sample_returns)
        assert isinstance(p, Portfolio)

    def test_weights_sum_to_one(self, sample_returns):
        """Weights should sum to 1."""
        p = max_sharpe_portfolio(sample_returns)
        assert abs(sum(p.weights) - 1.0) < 1e-6


class TestRiskParityPortfolio:
    def test_returns_portfolio(self, sample_returns):
        """Should return a Portfolio dataclass."""
        p = risk_parity_portfolio(sample_returns)
        assert isinstance(p, Portfolio)

    def test_weights_sum_to_one(self, sample_returns):
        """Weights should sum to 1."""
        p = risk_parity_portfolio(sample_returns)
        assert abs(sum(p.weights) - 1.0) < 1e-6

    def test_risk_contributions_approximately_equal(self, sample_returns):
        """Each asset should contribute approximately equal risk."""
        p = risk_parity_portfolio(sample_returns)
        w = np.array(p.weights)
        cov = sample_returns.cov().values
        risk_contrib = w * (cov @ w)
        # Contributions should be close to each other
        assert risk_contrib.std() / risk_contrib.mean() < 0.15


class TestEfficientFrontier:
    def test_returns_list_of_portfolios(self, sample_returns):
        """Should return a list of Portfolio objects."""
        frontier = efficient_frontier(sample_returns, n_points=10)
        assert isinstance(frontier, list)
        assert all(isinstance(p, Portfolio) for p in frontier)

    def test_frontier_has_points(self, sample_returns):
        """Should return multiple frontier points."""
        frontier = efficient_frontier(sample_returns, n_points=10)
        assert len(frontier) > 0
