"""VaR backtesting and model validation.

Tests whether a VaR model's predictions are accurate by comparing
forecasted VaR against realized losses. Includes statistical tests
(Kupiec, Christoffersen) and the Basel traffic light classification.
This is literally the job of a risk model validation analyst.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import VaRMethod, BacktestResult
from .var import historical_var, parametric_var, monte_carlo_var, cornish_fisher_var, evt_var


def rolling_var_backtest(
    returns: pd.Series,
    confidence: float = 0.95,
    method: VaRMethod = VaRMethod.HISTORICAL,
    window: int = 250,
) -> BacktestResult:
    """Rolling window VaR backtest.

    For each day after the initial window, compute VaR using the trailing
    'window' days, then check if the next day's actual loss exceeded VaR.
    A breach means the model underestimated risk that day.
    """
    losses = -returns
    n_test = len(returns) - window
    breaches = 0
    var_series = []
    breach_series = []

    for t in range(window, len(returns)):
        train = returns.iloc[t - window : t]

        if method == VaRMethod.HISTORICAL:
            result = historical_var(train, confidence)
        elif method == VaRMethod.PARAMETRIC:
            result = parametric_var(train, confidence)
        elif method == VaRMethod.CORNISH_FISHER:
            result = cornish_fisher_var(train, confidence)
        else:
            result = historical_var(train, confidence)

        var_value = result.var
        actual_loss = float(losses.iloc[t])
        is_breach = actual_loss > var_value

        var_series.append(var_value)
        breach_series.append(is_breach)

        if is_breach:
            breaches += 1

    breach_rate = breaches / n_test if n_test > 0 else 0
    expected_rate = 1 - confidence
    p_value = kupiec_pof_test(n_test, breaches, confidence)
    light = traffic_light_test(n_test, breaches)

    return BacktestResult(
        method=method,
        n_observations=n_test,
        n_breaches=breaches,
        breach_rate=breach_rate,
        expected_rate=expected_rate,
        kupiec_pvalue=p_value,
        traffic_light=light,
    )


def kupiec_pof_test(n_observations: int, n_breaches: int, confidence: float) -> float:
    """Kupiec Proportion of Failures test.

    Tests H0: the observed breach rate equals the expected rate.
    Uses a likelihood ratio test with chi-squared(1) distribution.
    A low p-value (<0.05) means the VaR model is rejected — it's
    producing too many or too few breaches.
    """
    p = 1 - confidence  # expected breach rate
    n = n_observations
    x = n_breaches

    if x == 0 or x == n:
        # Edge case: perfect or total failure
        return 1.0 if x == 0 else 0.0

    p_hat = x / n  # observed breach rate

    # Log-likelihood ratio
    lr = 2 * (
        x * np.log(p_hat / p) + (n - x) * np.log((1 - p_hat) / (1 - p))
    )

    # LR ~ chi-squared(1) under H0
    return float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_test(breach_series: pd.Series) -> float:
    """Christoffersen conditional coverage test.

    Tests whether breaches are independent (not clustered).
    If VaR breaches cluster together (e.g., 5 breaches in a row),
    the model fails to capture volatility dynamics. A low p-value
    indicates breach clustering — the model needs time-varying inputs.
    """
    breaches = breach_series.astype(int).values
    n = len(breaches)

    # Count transitions: 00, 01, 10, 11
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = breaches[i - 1], breaches[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return 1.0  # not enough transitions to test

    p01 = n01 / (n00 + n01)  # P(breach | no breach yesterday)
    p11 = n11 / (n10 + n11)  # P(breach | breach yesterday)

    # Under independence, both should equal the unconditional probability
    p = (n01 + n11) / n

    if p == 0 or p == 1 or p01 == 0 or p11 == 0:
        return 1.0

    # Log-likelihood under independence
    ll_ind = (n01 + n11) * np.log(p) + (n00 + n10) * np.log(1 - p)

    # Log-likelihood under dependence
    ll_dep = 0
    if n01 > 0:
        ll_dep += n01 * np.log(p01)
    if n00 > 0:
        ll_dep += n00 * np.log(1 - p01)
    if n11 > 0:
        ll_dep += n11 * np.log(p11)
    if n10 > 0:
        ll_dep += n10 * np.log(1 - p11)

    lr = 2 * (ll_dep - ll_ind)
    return float(1 - stats.chi2.cdf(max(lr, 0), df=1))


def traffic_light_test(n_observations: int, n_breaches: int) -> str:
    """Basel traffic light classification.

    Based on the number of VaR breaches in a 250-day window:
        Green:  0-4 breaches   (model is fine)
        Yellow: 5-9 breaches   (model needs attention, capital surcharge)
        Red:    10+ breaches   (model is rejected, major capital penalty)

    These thresholds assume 99% VaR over 250 trading days.
    """
    # Scale thresholds for non-250 day windows
    scale = n_observations / 250

    green_max = int(4 * scale)
    yellow_max = int(9 * scale)

    if n_breaches <= green_max:
        return "green"
    elif n_breaches <= yellow_max:
        return "yellow"
    else:
        return "red"


def compare_var_models(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 250,
) -> pd.DataFrame:
    """Run backtests across all VaR methods and compare performance.

    Returns a summary DataFrame ranking methods by breach rate accuracy,
    Kupiec p-value, and Basel traffic light. This is the key output for
    model selection — which VaR method works best for this portfolio?
    """
    methods = [
        VaRMethod.HISTORICAL,
        VaRMethod.PARAMETRIC,
        VaRMethod.CORNISH_FISHER,
    ]

    results = []
    for method in methods:
        bt = rolling_var_backtest(returns, confidence, method, window)
        results.append(
            {
                "method": method.value,
                "n_breaches": bt.n_breaches,
                "breach_rate": round(bt.breach_rate, 4),
                "expected_rate": round(bt.expected_rate, 4),
                "breach_ratio": round(bt.breach_rate / bt.expected_rate, 2) if bt.expected_rate > 0 else 0,
                "kupiec_pvalue": round(bt.kupiec_pvalue, 4),
                "traffic_light": bt.traffic_light,
            }
        )

    return pd.DataFrame(results)
