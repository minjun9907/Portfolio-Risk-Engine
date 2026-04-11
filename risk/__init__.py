"""Portfolio Risk Engine — core types and dataclasses."""

from dataclasses import dataclass, field
from enum import Enum


class VaRMethod(Enum):
    """Supported Value-at-Risk computation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    EVT = "evt"


class CovarianceMethod(Enum):
    """Supported covariance estimation methods."""
    SAMPLE = "sample"
    EWMA = "ewma"
    LEDOIT_WOLF = "ledoit_wolf"


@dataclass
class Portfolio:
    """Multi-asset portfolio definition."""
    tickers: list[str]          # Asset ticker symbols
    weights: list[float]        # Portfolio weights (sum to 1)
    name: str = "Portfolio"     # Display name


@dataclass
class OptionPosition:
    """Single option position for Greeks integration."""
    ticker: str                 # Underlying ticker
    strike: float               # Strike price
    expiry_years: float         # Time to expiry in years
    position_size: int          # Number of contracts (negative = short)
    is_call: bool = True        # True for call, False for put
    implied_vol: float = 0.0    # Implied volatility (annualized)


@dataclass
class VaRResult:
    """Container for Value-at-Risk computation output."""
    var: float                  # VaR estimate (positive = loss)
    es: float                   # Expected Shortfall estimate
    confidence: float           # Confidence level (e.g. 0.95)
    method: VaRMethod           # Method used
    details: dict = field(default_factory=dict)  # Method-specific extras


@dataclass
class StressResult:
    """Container for stress test output."""
    scenario_name: str          # Name of scenario
    portfolio_pnl: float        # Portfolio P&L under scenario
    asset_pnls: dict[str, float] = field(default_factory=dict)  # Per-asset P&L


@dataclass
class BacktestResult:
    """Container for VaR backtest output."""
    method: VaRMethod           # VaR method tested
    n_observations: int         # Total observations
    n_breaches: int             # Number of VaR breaches
    breach_rate: float          # Actual breach rate
    expected_rate: float        # Expected breach rate (1 - confidence)
    kupiec_pvalue: float        # Kupiec POF test p-value
    traffic_light: str          # "green", "yellow", or "red"


@dataclass
class FactorExposure:
    """Container for factor model regression output."""
    factor_names: list[str]     # Factor names
    betas: list[float]          # Factor loadings
    alpha: float                # Regression intercept (annualized)
    r_squared: float            # Goodness of fit
    residual_vol: float         # Idiosyncratic volatility
