"""Portfolio-level Greeks for derivatives books.

Aggregates option-level Greeks from the Option-Pricing-Engine across a
book of positions. Also computes scenario P&L under joint spot and
volatility moves — the core of option risk management.
"""

import sys
from pathlib import Path

import numpy as np

from . import OptionPosition


# Add Option-Pricing-Engine to sys.path so we can import its pricer package.
# A cleaner long-term solution would be `pip install -e ../Option-Pricing-Engine`.
_PRICING_ENGINE_PATH = Path(__file__).resolve().parents[2] / "Option-Pricing-Engine"
if str(_PRICING_ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(_PRICING_ENGINE_PATH))

from pricer import OptionParams, OptionType  # noqa: E402
from pricer.black_scholes import bs_price  # noqa: E402
from pricer.greeks import analytical_greeks  # noqa: E402


def _to_params(position: OptionPosition, spot: float, rate: float = 0.04) -> OptionParams:
    """Convert our OptionPosition to the pricer's OptionParams format."""
    return OptionParams(
        S=spot,
        K=position.strike,
        T=position.expiry_years,
        r=rate,
        sigma=position.implied_vol,
        q=0.0,
    )


def portfolio_greeks(
    option_positions: list[OptionPosition],
    spot_prices: dict[str, float],
    rate: float = 0.04,
) -> dict[str, float]:
    """Aggregate Greeks across an options book.

    Multiplies each option's Greeks by its position size and underlying
    spot, then sums into portfolio-level delta, gamma, theta, vega, rho.

    Args:
        option_positions: List of OptionPosition dataclasses.
        spot_prices: Dict mapping underlying ticker to current spot price.
        rate: Risk-free rate for pricing.

    Returns:
        Dict of aggregated Greeks: delta, gamma, theta, vega, rho.
    """
    totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    for pos in option_positions:
        spot = spot_prices[pos.ticker]
        params = _to_params(pos, spot, rate)
        option_type = OptionType.CALL if pos.is_call else OptionType.PUT
        g = analytical_greeks(params, option_type)

        totals["delta"] += pos.position_size * g.delta
        totals["gamma"] += pos.position_size * g.gamma
        totals["theta"] += pos.position_size * g.theta
        totals["vega"] += pos.position_size * g.vega
        totals["rho"] += pos.position_size * g.rho

    return totals


def greeks_scenario_pnl(
    option_positions: list[OptionPosition],
    spot_prices: dict[str, float],
    spot_shift: float,
    vol_shift: float,
    time_shift_days: float = 0.0,
    rate: float = 0.04,
) -> float:
    """P&L of the options book under a joint scenario.

    Re-prices every position with shifted spot, implied vol, and time,
    then returns the total P&L against the baseline prices.

    Args:
        spot_shift: Fractional change in spot (e.g. -0.10 = -10%).
        vol_shift: Absolute change in vol (e.g. 0.05 = +5 vol points).
        time_shift_days: Days to age the positions by.

    Returns:
        Total dollar P&L of the book under the scenario.
    """
    total_pnl = 0.0
    for pos in option_positions:
        spot = spot_prices[pos.ticker]
        base_params = _to_params(pos, spot, rate)
        option_type = OptionType.CALL if pos.is_call else OptionType.PUT
        base_price = bs_price(base_params, option_type)

        new_params = OptionParams(
            S=spot * (1 + spot_shift),
            K=pos.strike,
            T=max(pos.expiry_years - time_shift_days / 365, 1e-6),
            r=rate,
            sigma=max(pos.implied_vol + vol_shift, 1e-6),
            q=0.0,
        )
        new_price = bs_price(new_params, option_type)
        total_pnl += pos.position_size * (new_price - base_price)

    return float(total_pnl)


def greeks_heatmap(
    option_positions: list[OptionPosition],
    spot_prices: dict[str, float],
    spot_range: tuple[float, float] = (-0.20, 0.20),
    vol_range: tuple[float, float] = (-0.10, 0.10),
    grid_size: int = 11,
    rate: float = 0.04,
) -> np.ndarray:
    """2D P&L grid across spot and volatility shocks.

    Returns a grid_size x grid_size array where rows sweep spot shifts
    and columns sweep vol shifts. Useful for visualizing whether the
    book is net long or short gamma/vega under different conditions.
    """
    spot_shifts = np.linspace(spot_range[0], spot_range[1], grid_size)
    vol_shifts = np.linspace(vol_range[0], vol_range[1], grid_size)

    grid = np.zeros((grid_size, grid_size))
    for i, ds in enumerate(spot_shifts):
        for j, dv in enumerate(vol_shifts):
            grid[i, j] = greeks_scenario_pnl(
                option_positions, spot_prices, ds, dv, rate=rate
            )

    return grid
