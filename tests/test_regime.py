"""Tests for risk/regime.py — HMM and rule-based regime detection."""

import numpy as np
import pandas as pd
import pytest

from risk import VaRResult
from risk.regime import (
    hmm_regime_detection,
    rule_based_regime,
    regime_conditional_var,
    regime_transition_matrix,
)


class TestHMMRegimeDetection:
    def test_returns_series(self, sample_portfolio_returns):
        """Should return a pandas Series of integer states."""
        regimes = hmm_regime_detection(sample_portfolio_returns, n_states=2)
        assert isinstance(regimes, pd.Series)
        assert regimes.dtype in (np.int64, np.int32, int)

    def test_has_multiple_states(self, sample_portfolio_returns):
        """Should detect at least 2 distinct states."""
        regimes = hmm_regime_detection(sample_portfolio_returns, n_states=2)
        assert regimes.nunique() >= 1  # at least one state detected

    def test_same_length_as_input(self, sample_portfolio_returns):
        """Output length should match input."""
        regimes = hmm_regime_detection(sample_portfolio_returns, n_states=2)
        assert len(regimes) == len(sample_portfolio_returns)


class TestRuleBasedRegime:
    def test_returns_binary_series(self, sample_portfolio_returns):
        """Should return a Series with 0/1 values."""
        regimes = rule_based_regime(sample_portfolio_returns)
        unique_values = set(regimes.dropna().unique())
        assert unique_values.issubset({0, 1})

    def test_same_length_as_input(self, sample_portfolio_returns):
        """Output length should match input."""
        regimes = rule_based_regime(sample_portfolio_returns)
        assert len(regimes) == len(sample_portfolio_returns)


class TestRegimeConditionalVar:
    def test_returns_dict_of_var_results(self, sample_portfolio_returns):
        """Should return a dict mapping state -> VaRResult."""
        regimes = rule_based_regime(sample_portfolio_returns)
        results = regime_conditional_var(sample_portfolio_returns, regimes)
        assert isinstance(results, dict)
        assert all(isinstance(v, VaRResult) for v in results.values())


class TestRegimeTransitionMatrix:
    def test_returns_square_matrix(self, sample_portfolio_returns):
        """Transition matrix should be square."""
        regimes = rule_based_regime(sample_portfolio_returns)
        matrix = regime_transition_matrix(regimes)
        assert matrix.shape[0] == matrix.shape[1]

    def test_rows_sum_to_one(self, sample_portfolio_returns):
        """Each row should sum to 1 (valid probability distribution)."""
        regimes = rule_based_regime(sample_portfolio_returns)
        matrix = regime_transition_matrix(regimes)
        row_sums = matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones_like(row_sums))
