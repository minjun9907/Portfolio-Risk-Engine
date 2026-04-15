"""Tests for risk/greeks.py — portfolio Greeks and option scenario P&L."""

import numpy as np
import pytest

from risk import OptionPosition
from risk.greeks import portfolio_greeks, greeks_scenario_pnl, greeks_heatmap


@pytest.fixture
def sample_option_book() -> list[OptionPosition]:
    """A small options book: long call, short put."""
    return [
        OptionPosition(
            ticker="AAPL",
            strike=150.0,
            expiry_years=0.25,
            position_size=10,
            is_call=True,
            implied_vol=0.30,
        ),
        OptionPosition(
            ticker="AAPL",
            strike=140.0,
            expiry_years=0.25,
            position_size=-5,
            is_call=False,
            implied_vol=0.32,
        ),
    ]


@pytest.fixture
def sample_spot_prices() -> dict[str, float]:
    return {"AAPL": 155.0}


class TestPortfolioGreeks:
    def test_returns_all_greeks(self, sample_option_book, sample_spot_prices):
        """Should return dict with delta, gamma, theta, vega, rho."""
        result = portfolio_greeks(sample_option_book, sample_spot_prices)
        assert set(result.keys()) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_long_call_positive_delta(self, sample_spot_prices):
        """A pure long call position should have positive delta."""
        book = [
            OptionPosition(
                ticker="AAPL",
                strike=150.0,
                expiry_years=0.25,
                position_size=1,
                is_call=True,
                implied_vol=0.30,
            )
        ]
        result = portfolio_greeks(book, sample_spot_prices)
        assert result["delta"] > 0

    def test_gamma_nonnegative_for_long_options(self, sample_spot_prices):
        """Long options should have positive gamma."""
        book = [
            OptionPosition(
                ticker="AAPL",
                strike=150.0,
                expiry_years=0.25,
                position_size=1,
                is_call=True,
                implied_vol=0.30,
            )
        ]
        result = portfolio_greeks(book, sample_spot_prices)
        assert result["gamma"] > 0


class TestGreeksScenarioPnl:
    def test_returns_float(self, sample_option_book, sample_spot_prices):
        """Should return a float P&L value."""
        pnl = greeks_scenario_pnl(sample_option_book, sample_spot_prices, 0.05, 0.01)
        assert isinstance(pnl, float)

    def test_zero_shock_zero_pnl(self, sample_option_book, sample_spot_prices):
        """No shock should produce approximately zero P&L."""
        pnl = greeks_scenario_pnl(sample_option_book, sample_spot_prices, 0.0, 0.0)
        assert abs(pnl) < 1e-6


class TestGreeksHeatmap:
    def test_returns_2d_array(self, sample_option_book, sample_spot_prices):
        """Should return a 2D numpy array of the requested grid size."""
        grid = greeks_heatmap(sample_option_book, sample_spot_prices, grid_size=5)
        assert grid.shape == (5, 5)

    def test_all_finite(self, sample_option_book, sample_spot_prices):
        """All grid values should be finite."""
        grid = greeks_heatmap(sample_option_book, sample_spot_prices, grid_size=5)
        assert np.all(np.isfinite(grid))
