"""Tests for risk/correlation.py — rolling correlation and PCA."""

import numpy as np
import pandas as pd
import pytest

from risk.correlation import (
    rolling_correlation,
    eigenvalue_decomposition,
    correlation_regime_indicator,
)


class TestRollingCorrelation:
    def test_returns_dataframe(self, sample_returns):
        """Should return a DataFrame with one column."""
        result = rolling_correlation(sample_returns, window=60)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1

    def test_first_window_is_nan(self, sample_returns):
        """Days before the first full window should be NaN."""
        result = rolling_correlation(sample_returns, window=60)
        assert result.iloc[:59].isna().all().all()

    def test_values_in_valid_range(self, sample_returns):
        """Correlations should be within [-1, 1]."""
        result = rolling_correlation(sample_returns, window=60).dropna()
        assert (result >= -1).all().all()
        assert (result <= 1).all().all()


class TestEigenvalueDecomposition:
    def test_eigenvalues_sorted_descending(self, sample_covariance):
        """Eigenvalues should be returned in descending order."""
        eigenvalues, _ = eigenvalue_decomposition(sample_covariance)
        assert np.all(np.diff(eigenvalues) <= 0)

    def test_eigenvectors_orthonormal(self, sample_covariance):
        """Eigenvectors should be orthonormal (V @ V.T = I)."""
        _, eigenvectors = eigenvalue_decomposition(sample_covariance)
        identity = eigenvectors @ eigenvectors.T
        np.testing.assert_array_almost_equal(identity, np.eye(len(eigenvectors)))

    def test_sum_of_eigenvalues_equals_trace(self, sample_covariance):
        """Sum of eigenvalues should equal the trace of the covariance matrix."""
        eigenvalues, _ = eigenvalue_decomposition(sample_covariance)
        assert abs(eigenvalues.sum() - np.trace(sample_covariance.values)) < 1e-10


class TestCorrelationRegimeIndicator:
    def test_returns_series(self, sample_returns):
        """Should return a pandas Series."""
        result = correlation_regime_indicator(sample_returns)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, sample_returns):
        """Output length matches input length."""
        result = correlation_regime_indicator(sample_returns)
        assert len(result) == len(sample_returns)
