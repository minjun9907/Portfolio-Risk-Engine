"""Liquidity-adjusted risk.

Textbook VaR assumes instant liquidation at mid prices. Real positions
pay bid-ask spreads and take time to unwind. This module extends VaR
to include liquidation cost, a requirement under Basel III FRTB.
"""

import numpy as np
import pandas as pd


DEFAULT_SPREAD_ESTIMATES: dict[str, float] = {
    # Rough typical bid-ask spreads as a fraction of price
    "mega_cap": 0.0005,   # 5 bps  (AAPL, MSFT, GOOGL, SPY)
    "large_cap": 0.0010,  # 10 bps (most S&P 500 names)
    "mid_cap": 0.0025,    # 25 bps
    "small_cap": 0.0075,  # 75 bps
    "illiquid": 0.02,     # 2%    (micro caps, OTC)
}


def bid_ask_cost(tickers: list[str]) -> dict[str, float]:
    """Estimate bid-ask spread for each ticker.

    Uses yfinance to fetch current bid/ask when available.
    Falls back to a large-cap default if data is missing.
    """
    import yfinance as yf

    costs: dict[str, float] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            bid = info.get("bid", 0)
            ask = info.get("ask", 0)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                costs[ticker] = (ask - bid) / mid
            else:
                costs[ticker] = DEFAULT_SPREAD_ESTIMATES["large_cap"]
        except Exception:
            costs[ticker] = DEFAULT_SPREAD_ESTIMATES["large_cap"]

    return costs


def liquidity_adjusted_var(
    var: float,
    liquidity_cost: float,
    holding_period: int = 10,
) -> float:
    """Adjust VaR to include liquidation cost.

    LVaR = VaR + (holding_period / 2) * liquidity_cost

    The factor of 1/2 represents the average position held during
    an orderly liquidation — you unwind linearly so on average half
    the position is still exposed to spread costs.
    """
    return float(var + (holding_period / 2) * liquidity_cost)


def liquidity_stress_var(
    var: float,
    liquidity_cost: float,
    stress_spread_multiplier: float = 3.0,
    holding_period: int = 10,
) -> float:
    """Liquidity-adjusted VaR under stressed market conditions.

    In a crisis, bid-ask spreads widen dramatically (typically 3-5x).
    This function applies a stress multiplier to the spread cost before
    computing the adjusted VaR. Based on the 2008 and 2020 episodes.
    """
    stressed_cost = liquidity_cost * stress_spread_multiplier
    return liquidity_adjusted_var(var, stressed_cost, holding_period)
