"""Tests for risk/liquidity.py — liquidity-adjusted VaR."""

import pytest

from risk.liquidity import (
    bid_ask_cost,
    liquidity_adjusted_var,
    liquidity_stress_var,
)


class TestBidAskCost:
    @pytest.mark.slow
    def test_returns_dict(self):
        """Should return a dict mapping tickers to cost estimates."""
        result = bid_ask_cost(["AAPL", "MSFT"])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"AAPL", "MSFT"}

    @pytest.mark.slow
    def test_costs_positive(self):
        """All spread cost estimates should be positive."""
        result = bid_ask_cost(["AAPL"])
        assert all(v > 0 for v in result.values())


class TestLiquidityAdjustedVar:
    def test_exceeds_unadjusted_var(self):
        """LVaR should always be greater than plain VaR."""
        var = 0.02
        lvar = liquidity_adjusted_var(var, liquidity_cost=0.001, holding_period=10)
        assert lvar > var

    def test_scales_with_holding_period(self):
        """Longer holding period should increase LVaR."""
        var = 0.02
        short = liquidity_adjusted_var(var, 0.001, holding_period=1)
        long = liquidity_adjusted_var(var, 0.001, holding_period=20)
        assert long > short

    def test_zero_liquidity_cost_equals_var(self):
        """With zero liquidity cost, LVaR equals VaR."""
        var = 0.02
        lvar = liquidity_adjusted_var(var, liquidity_cost=0.0, holding_period=10)
        assert abs(lvar - var) < 1e-12


class TestLiquidityStressVar:
    def test_exceeds_normal_lvar(self):
        """Stressed LVaR should exceed normal LVaR."""
        var = 0.02
        normal = liquidity_adjusted_var(var, 0.001)
        stressed = liquidity_stress_var(var, 0.001, stress_spread_multiplier=3.0)
        assert stressed > normal

    def test_default_multiplier_applied(self):
        """The 3x default multiplier should triple the spread cost contribution."""
        var = 0.0
        cost = 0.001
        horizon = 10
        normal_excess = liquidity_adjusted_var(var, cost, horizon) - var
        stressed_excess = liquidity_stress_var(var, cost, 3.0, horizon) - var
        assert abs(stressed_excess - 3 * normal_excess) < 1e-12
