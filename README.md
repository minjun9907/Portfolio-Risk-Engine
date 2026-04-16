# Portfolio Risk Engine

> Multi-asset portfolio risk engine aligned with FRM Part 2 Book 4 curriculum — VaR, stress testing, factor models, regime detection, and more.

## What It Does

- **5 VaR methods** — Historical, Parametric (normal/Student-t), Monte Carlo (Cholesky), Cornish-Fisher, EVT (GPD)
- **Expected Shortfall** — Tail average for each VaR method (Basel III preferred measure)
- **Volatility models** — EWMA (RiskMetrics) and GARCH(1,1) conditional volatility
- **Factor attribution** — Fama-French regression, sector concentration analysis
- **Portfolio optimization** — Min-variance, max-Sharpe, risk parity, efficient frontier
- **Regime detection** — HMM-based and rule-based, with regime-conditional VaR
- **Stress testing** — Historical scenarios (2008, COVID, 2022, SVB), hypothetical shocks, reverse stress test
- **VaR decomposition** — Marginal, component (Euler), incremental VaR
- **Greeks integration** — Portfolio-level Greeks with scenario P&L heatmap via [Option Pricing Engine](https://github.com/minjun9907/Option-Pricing-Engine)
- **Liquidity risk** — Bid-ask cost estimation, liquidity-adjusted VaR, stressed LVaR

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/
```

```python
from risk.data import fetch_prices, compute_returns, portfolio_returns
from risk.var import historical_var, parametric_var, evt_var

tickers = ["AAPL", "GOOGL", "MSFT", "JPM"]
weights = [0.30, 0.25, 0.25, 0.20]

prices = fetch_prices(tickers, period="5y")
returns = compute_returns(prices, method="log")
port_ret = portfolio_returns(returns, weights)

hist = historical_var(port_ret, confidence=0.95)
param = parametric_var(port_ret, confidence=0.95)
evt = evt_var(port_ret, confidence=0.99)

print(f"Historical 95% VaR: {hist.var:.4f}")
print(f"Parametric 95% VaR: {param.var:.4f}")
print(f"EVT 99% VaR:        {evt.var:.4f}")
```

## Examples

### VaR Comparison (5 Methods)

```python
from risk.var import historical_var, parametric_var, cornish_fisher_var, evt_var

hist = historical_var(port_ret, confidence=0.95)
param = parametric_var(port_ret, confidence=0.95)
cf = cornish_fisher_var(port_ret, confidence=0.95)
evt = evt_var(port_ret, confidence=0.99)
```
```
Historical 95% VaR:     0.0128   ES: 0.0167
Parametric 95% VaR:     0.0129   ES: 0.0161
Cornish-Fisher 95% VaR: 0.0125
EVT 99% VaR:            0.0192   (xi=-0.110)
```

### VaR Decomposition — Where Is the Risk?

```python
from risk.var_decomposition import component_var

components = component_var(returns, weights)
```
```
AAPL:  0.0042   (32.6%)
GOOGL: 0.0031   (23.9%)
MSFT:  0.0028   (22.0%)
JPM:   0.0028   (21.5%)
Sum:   0.0129   = total portfolio VaR (Euler's theorem)
```

### Regime Detection — Adapt Risk to Market State

```python
from risk.regime import rule_based_regime, regime_conditional_var

regimes = rule_based_regime(port_ret)
rcvar = regime_conditional_var(port_ret, regimes)
```
```
State 0 (low-vol):  639 days   95% VaR: 0.0115
State 1 (high-vol): 620 days   95% VaR: 0.0144
Transition: stay low-vol = 0.92, switch to high-vol = 0.08
```

### Stress Testing — What If 2008 Happens Again?

```python
from risk.stress_testing import hypothetical_scenario, reverse_stress_test

result = hypothetical_scenario(portfolio, {"AAPL": -0.10, "GOOGL": -0.08, "MSFT": -0.06, "JPM": -0.12})
```
```
Portfolio P&L: -0.0890
  AAPL:  -0.0300
  GOOGL: -0.0200
  MSFT:  -0.0150
  JPM:   -0.0240
```
```python
# Reverse stress test: what shock causes a 10% loss?
shocks = reverse_stress_test(portfolio, returns, target_loss=-0.10)
```
```
AAPL: -0.1176   GOOGL: -0.0980   MSFT: -0.0980   JPM: -0.0784
```

### Volatility Models — Static vs EWMA vs GARCH

```python
from risk.volatility import compare_vol_models

vols = compare_vol_models(port_ret)
```
```
Static: 0.007823   (constant — ignores regime changes)
EWMA:   0.007677   (adapts to recent volatility)
GARCH:  0.007803   (mean-reverting conditional vol)
```

## Architecture

```
risk/
├── __init__.py            Core dataclasses (Portfolio, VaRResult, FactorExposure, ...)
├── data.py                Price fetching, returns, covariance (sample, EWMA, Ledoit-Wolf)
├── volatility.py          EWMA, GARCH(1,1), model comparison
├── var.py                 Historical, Parametric, Monte Carlo, Cornish-Fisher, EVT
├── expected_shortfall.py  ES for each VaR method
├── factor_model.py        Fama-French factor regression, sector concentration
├── optimization.py        Min-variance, max-Sharpe, risk parity, efficient frontier
├── correlation.py         Rolling correlation, PCA, stress indicator
├── regime.py              HMM regime detection, regime-conditional VaR
├── stress_testing.py      Historical, hypothetical, and reverse stress tests
├── var_decomposition.py   Marginal, component, incremental VaR
├── greeks.py              Portfolio Greeks via Option-Pricing-Engine
└── liquidity.py           Bid-ask cost, LVaR, stressed LVaR

tests/
├── conftest.py            Shared fixtures (sample portfolio, returns, covariance)
└── test_*.py              99 tests across all modules
```

## Modules

| Module | Description | FRM Topic |
|--------|-------------|-----------|
| `var.py` | Historical, Parametric (normal/Student-t), Monte Carlo, Cornish-Fisher, EVT | VaR methods, EVT/GPD |
| `expected_shortfall.py` | ES for historical, parametric, Monte Carlo, and EVT | Expected Shortfall |
| `volatility.py` | EWMA (RiskMetrics), GARCH(1,1) conditional volatility | GARCH/EWMA volatility |
| `factor_model.py` | Fama-French 5-factor regression, sector concentration | Factor models |
| `optimization.py` | Min-variance, max-Sharpe, risk parity, efficient frontier | Portfolio optimization |
| `correlation.py` | Rolling pairwise correlation, PCA of covariance matrix | Correlation/PCA |
| `regime.py` | HMM-based and rule-based regime detection, regime-conditional VaR | — |
| `stress_testing.py` | 2008 GFC, COVID, 2022 rate hikes, SVB; hypothetical shocks; reverse stress test | Stress testing |
| `var_decomposition.py` | Marginal VaR, component VaR (Euler), incremental VaR | VaR decomposition |
| `greeks.py` | Portfolio-level Greeks aggregation, scenario P&L, 2D heatmap | Options risk |
| `liquidity.py` | Bid-ask spread estimation, liquidity-adjusted VaR, stressed LVaR | Liquidity risk |

## FRM Part 2 Alignment

| FRM Topic | Module |
|-----------|--------|
| VaR methods (historical, parametric, MC) | `risk/var.py` |
| Expected Shortfall | `risk/expected_shortfall.py` |
| Extreme Value Theory / GPD | `risk/var.py` (evt_var) |
| GARCH / EWMA volatility | `risk/volatility.py` |
| VaR decomposition (marginal, component, incremental) | `risk/var_decomposition.py` |
| Stress testing + reverse stress testing | `risk/stress_testing.py` |
| Liquidity-adjusted VaR | `risk/liquidity.py` |
| Correlation / PCA | `risk/correlation.py` |
| Portfolio optimization | `risk/optimization.py` |
| Factor models | `risk/factor_model.py` |
| Options risk (Greeks) | `risk/greeks.py` |

## Related Projects

- [Option Pricing Engine](https://github.com/minjun9907/Option-Pricing-Engine) — Black-Scholes, Binomial, Monte Carlo pricing with Greeks. Used by `risk/greeks.py` for portfolio-level derivatives risk.
