"""Tests for risk/stress_testing.py — historical, hypothetical, reverse stress."""

import numpy as np
import pandas as pd
import pytest

from risk import StressResult
from risk.stress_testing import (
    HISTORICAL_SCENARIOS,
    hypothetical_scenario,
    sensitivity_analysis,
    reverse_stress_test,
)


class TestHistoricalScenarios:
    def test_scenarios_populated(self):
        """The HISTORICAL_SCENARIOS dict should contain at least 4 scenarios."""
        assert len(HISTORICAL_SCENARIOS) >= 4
        assert "2008_gfc" in HISTORICAL_SCENARIOS
        assert "2020_covid" in HISTORICAL_SCENARIOS


class TestHypotheticalScenario:
    def test_returns_stress_result(self, sample_portfolio):
        """Should return a StressResult dataclass."""
        shocks = {"AAPL": -0.10, "GOOGL": -0.08}
        result = hypothetical_scenario(sample_portfolio, shocks)
        assert isinstance(result, StressResult)

    def test_unshocked_assets_zero_pnl(self, sample_portfolio):
        """Assets not in the shocks dict should contribute zero P&L."""
        shocks = {"AAPL": -0.10}
        result = hypothetical_scenario(sample_portfolio, shocks)
        for ticker in sample_portfolio.tickers:
            if ticker != "AAPL":
                assert result.asset_pnls[ticker] == 0

    def test_portfolio_pnl_sums_assets(self, sample_portfolio):
        """Total portfolio PnL equals sum of asset-level PnLs."""
        shocks = {t: -0.05 for t in sample_portfolio.tickers}
        result = hypothetical_scenario(sample_portfolio, shocks)
        assert abs(result.portfolio_pnl - sum(result.asset_pnls.values())) < 1e-12


class TestSensitivityAnalysis:
    def test_returns_dataframe(self, sample_portfolio, sample_returns):
        """Should return a DataFrame with one row per asset."""
        result = sensitivity_analysis(sample_portfolio, sample_returns)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_portfolio.tickers)

    def test_has_expected_columns(self, sample_portfolio, sample_returns):
        """Result should have ticker, base_vol, new_vol, vol_change columns."""
        result = sensitivity_analysis(sample_portfolio, sample_returns)
        assert set(result.columns) == {"ticker", "base_vol", "new_vol", "vol_change"}


class TestReverseStressTest:
    def test_returns_dict_per_asset(self, sample_portfolio, sample_returns):
        """Should return a dict with one shock per ticker."""
        result = reverse_stress_test(sample_portfolio, sample_returns, target_loss=-0.10)
        assert set(result.keys()) == set(sample_portfolio.tickers)

    def test_shocks_achieve_target_loss(self, sample_portfolio, sample_returns):
        """Weighted sum of shocks should equal the target loss."""
        target = -0.15
        shocks = reverse_stress_test(sample_portfolio, sample_returns, target_loss=target)
        achieved = sum(
            w * shocks[t]
            for t, w in zip(sample_portfolio.tickers, sample_portfolio.weights)
        )
        assert abs(achieved - target) < 1e-6
