"""Tests for risk/backtesting.py — VaR model validation."""

import numpy as np
import pandas as pd
import pytest

from risk import VaRMethod, BacktestResult
from risk.backtesting import (
    rolling_var_backtest,
    kupiec_pof_test,
    christoffersen_test,
    traffic_light_test,
    compare_var_models,
)


@pytest.fixture
def long_returns() -> pd.Series:
    """600 days of synthetic returns for backtesting (need >250 for rolling window)."""
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=600, freq="B")
    return pd.Series(np.random.randn(600) * 0.01, index=dates)


class TestRollingVarBacktest:
    def test_returns_backtest_result(self, long_returns):
        """Should return a BacktestResult dataclass."""
        result = rolling_var_backtest(long_returns, confidence=0.95, window=250)
        assert isinstance(result, BacktestResult)

    def test_breach_rate_reasonable(self, long_returns):
        """Breach rate should be roughly near the expected rate (5% for 95% VaR)."""
        result = rolling_var_backtest(long_returns, confidence=0.95, window=250)
        # Allow wide range — with 350 test days, some variance is expected
        assert 0.0 < result.breach_rate < 0.20

    def test_n_observations_correct(self, long_returns):
        """Number of test observations should be total - window."""
        result = rolling_var_backtest(long_returns, confidence=0.95, window=250)
        assert result.n_observations == len(long_returns) - 250

    def test_traffic_light_is_valid(self, long_returns):
        """Traffic light should be green, yellow, or red."""
        result = rolling_var_backtest(long_returns, confidence=0.95, window=250)
        assert result.traffic_light in ("green", "yellow", "red")


class TestKupiecPOFTest:
    def test_perfect_model_high_pvalue(self):
        """A model with exactly the expected breach rate should have high p-value."""
        # 250 days, 5% expected rate = 12.5 breaches → 12 or 13
        p = kupiec_pof_test(250, 12, 0.95)
        assert p > 0.05

    def test_too_many_breaches_low_pvalue(self):
        """Way too many breaches should give low p-value (model rejected)."""
        p = kupiec_pof_test(250, 30, 0.95)
        assert p < 0.05

    def test_zero_breaches(self):
        """Zero breaches should return p-value of 1.0 (edge case)."""
        p = kupiec_pof_test(250, 0, 0.95)
        assert p == 1.0


class TestChristoffersenTest:
    def test_independent_breaches_high_pvalue(self):
        """Randomly scattered breaches should have high p-value."""
        np.random.seed(42)
        breaches = pd.Series(np.random.choice([0, 1], size=500, p=[0.95, 0.05]))
        p = christoffersen_test(breaches)
        assert p > 0.05

    def test_clustered_breaches_low_pvalue(self):
        """Breaches that cluster together should have low p-value."""
        # Create clustered breaches: alternating blocks of breaches and calm
        breaches = pd.Series(
            ([1] * 10 + [0] * 40) * 10  # 10 breaches then 40 calm, repeated 10 times
        )
        p = christoffersen_test(breaches)
        assert p < 0.05


class TestTrafficLightTest:
    def test_green(self):
        """0-4 breaches in 250 days should be green."""
        assert traffic_light_test(250, 3) == "green"

    def test_yellow(self):
        """5-9 breaches in 250 days should be yellow."""
        assert traffic_light_test(250, 7) == "yellow"

    def test_red(self):
        """10+ breaches in 250 days should be red."""
        assert traffic_light_test(250, 12) == "red"

    def test_scales_with_window(self):
        """Thresholds should scale proportionally for non-250 day windows."""
        assert traffic_light_test(500, 8) == "green"  # 8 breaches in 500 days ~ 4 in 250


class TestCompareVarModels:
    def test_returns_dataframe(self, long_returns):
        """Should return a DataFrame with one row per method."""
        result = compare_var_models(long_returns, confidence=0.95, window=250)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3

    def test_has_expected_columns(self, long_returns):
        """Result should have method, breach_rate, kupiec_pvalue, traffic_light."""
        result = compare_var_models(long_returns, confidence=0.95, window=250)
        expected_cols = {"method", "n_breaches", "breach_rate", "expected_rate", "breach_ratio", "kupiec_pvalue", "traffic_light"}
        assert expected_cols == set(result.columns)
