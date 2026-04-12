"""Factor risk decomposition.

Fama-French factor regression attributes portfolio returns to systematic
factors (market, size, value, etc.). Sector concentration analysis reveals
industry exposure risks.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import FactorExposure


def fetch_ff_factors(period: str = "5y") -> pd.DataFrame:
    """Fetch Fama-French 5-factor daily data from Kenneth French's library.

    Returns a DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF.
    Returns are in decimal form (not percentage).
    """
    from pandas_datareader import data as pdr

    n_years = int(period.rstrip("y"))
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=n_years)
    ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)[0]
    return ff / 100  # convert percentage to decimal


def factor_regression(returns: pd.Series, factors: pd.DataFrame) -> FactorExposure:
    """OLS regression of returns on factors.

    Returns betas (factor loadings), alpha (intercept, annualized),
    R-squared, and residual volatility.
    """
    # Align on dates
    aligned = pd.concat([returns, factors], axis=1, join="inner").dropna()
    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]

    # Subtract risk-free rate if present to get excess returns
    if "RF" in X.columns:
        y = y - X["RF"]
        X = X.drop(columns=["RF"])

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    factor_names = list(X.columns)
    betas = [float(model.params[name]) for name in factor_names]
    alpha_daily = float(model.params["const"])
    residual_vol_daily = float(np.sqrt(model.mse_resid))

    return FactorExposure(
        factor_names=factor_names,
        betas=betas,
        alpha=alpha_daily * 252,  # annualized
        r_squared=float(model.rsquared),
        residual_vol=residual_vol_daily * np.sqrt(252),  # annualized
    )


def factor_risk_attribution(
    weights: list[float],
    betas: np.ndarray,
    factor_cov: np.ndarray,
) -> dict[str, float]:
    """Decompose portfolio variance into factor contributions.

    Portfolio factor exposure: B_p = w' * B (where B is n_assets x n_factors)
    Factor variance contribution: B_p' * Sigma_F * B_p
    """
    w = np.array(weights)
    portfolio_betas = w @ betas  # shape: (n_factors,)
    factor_variance = portfolio_betas @ factor_cov @ portfolio_betas
    factor_contributions = portfolio_betas * (factor_cov @ portfolio_betas)

    return {
        "total_factor_variance": float(factor_variance),
        "contributions": factor_contributions.tolist(),
        "portfolio_betas": portfolio_betas.tolist(),
    }


def sector_concentration(tickers: list[str], weights: list[float]) -> dict[str, float]:
    """Portfolio exposure by sector via yfinance metadata.

    Aggregates position weights by sector. High concentration in one sector
    is a red flag for idiosyncratic risk.
    """
    import yfinance as yf

    sector_weights: dict[str, float] = {}
    for ticker, weight in zip(tickers, weights):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    return sector_weights
