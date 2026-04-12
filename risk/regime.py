"""Market regime detection.

Identifies distinct market states (e.g., low-vol vs high-vol) using
Hidden Markov Models and rule-based thresholds. Regime-conditional VaR
adapts risk estimates to the current market environment.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from . import VaRResult
from .var import historical_var


def hmm_regime_detection(returns: pd.Series, n_states: int = 2) -> pd.Series:
    """HMM-based regime classification.

    Fits a Gaussian HMM to returns and labels each day with its most likely
    hidden state. States are reordered so state 0 is the lowest-vol regime.
    """
    X = returns.values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    states = model.predict(X)

    # Reorder states so state 0 = lowest volatility
    state_vols = [np.sqrt(model.covars_[i][0, 0]) for i in range(n_states)]
    order = np.argsort(state_vols)
    remapping = {old: new for new, old in enumerate(order)}
    relabeled = np.array([remapping[s] for s in states])

    return pd.Series(relabeled, index=returns.index, name="regime")


def rule_based_regime(returns: pd.Series, vol_threshold: float | None = None) -> pd.Series:
    """Threshold-based regime using realized volatility.

    Classifies each day as high-vol (1) or low-vol (0) based on a rolling
    20-day realized volatility compared to a threshold. If threshold is None,
    uses the median of the realized vol series.
    """
    realized_vol = returns.rolling(20).std()
    if vol_threshold is None:
        vol_threshold = float(realized_vol.median())

    regime = (realized_vol > vol_threshold).astype(int)
    regime.name = "regime"
    return regime


def regime_conditional_var(
    returns: pd.Series,
    regimes: pd.Series,
    confidence: float = 0.95,
) -> dict[int, VaRResult]:
    """Compute VaR separately for each regime state.

    Splits returns by regime label and runs historical VaR within each state.
    Useful to show how tail risk differs between calm and stressed periods.
    """
    results = {}
    aligned = pd.concat([returns, regimes], axis=1).dropna()
    returns_col = aligned.columns[0]
    regime_col = aligned.columns[1]

    for state in sorted(aligned[regime_col].unique()):
        state_returns = aligned.loc[aligned[regime_col] == state, returns_col]
        if len(state_returns) > 10:
            results[int(state)] = historical_var(state_returns, confidence=confidence)

    return results


def regime_transition_matrix(regimes: pd.Series) -> np.ndarray:
    """Empirical transition probabilities between regime states.

    Returns an n_states x n_states matrix where element [i, j] is the
    probability of moving from state i to state j on the next day.
    """
    clean = regimes.dropna().astype(int).values
    n_states = int(clean.max()) + 1
    matrix = np.zeros((n_states, n_states))

    for i in range(len(clean) - 1):
        matrix[clean[i], clean[i + 1]] += 1

    # Normalize each row to probabilities
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    return matrix / row_sums
