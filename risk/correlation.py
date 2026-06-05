"""Correlation analysis and PCA decomposition.

Rolling correlations reveal regime shifts (correlations spike in crises).
PCA identifies principal risk factors driving portfolio variance.
"""

import numpy as np
import pandas as pd


def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Compute rolling pairwise correlation average.

    For each day, calculates the average pairwise correlation across all
    asset pairs over the trailing window. High values indicate regime stress
    when assets move together.
    """
    n = returns.shape[1]
    # Stack rolling pairwise correlations, then average the upper triangle
    rolling_corr_stack = returns.rolling(window).corr()
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    avg_corr = (
        rolling_corr_stack
        .groupby(level=0)
        .apply(lambda m: m.values[mask].mean() if not np.isnan(m.values).all() else np.nan)
    )
    avg_corr.name = "avg_correlation"
    return avg_corr.to_frame()


def eigenvalue_decomposition(cov_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """PCA via eigenvalue decomposition of the covariance matrix.

    Returns eigenvalues (sorted descending) and eigenvectors.
    The largest eigenvalue represents the dominant risk factor —
    in equity markets, this is typically "the market."
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix.values)
    # Sort descending by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def correlation_regime_indicator(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Average pairwise correlation as a stress indicator.

    When average correlation spikes, diversification breaks down —
    a classic signature of market stress (e.g., 2008, COVID).
    """
    return rolling_correlation(returns, window=window).iloc[:, 0]
