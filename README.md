# Portfolio Risk Engine

Multi-asset equity portfolio risk engine with VaR (5 methods), Expected Shortfall, EVT tail modeling, GARCH conditional volatility, factor risk decomposition, options Greeks integration, liquidity-adjusted risk, stress testing, regime detection, and VaR backtesting with Basel compliance.

Aligned with **FRM Part 2 Book 4** topics.

---

## Architecture

```
risk/
├── __init__.py            Core dataclasses (Portfolio, VaRResult, ...)
├── data.py                Price fetching, returns, covariance
├── volatility.py          EWMA, GARCH(1,1), model comparison
├── var.py                 Historical, Parametric, MC, Cornish-Fisher, EVT
├── expected_shortfall.py  ES counterparts for each VaR method
├── factor_model.py        Fama-French regression, factor risk attribution
├── optimization.py        Min-var, max-Sharpe, risk parity, efficient frontier
├── correlation.py         Rolling correlation, PCA, stress indicator
├── regime.py              HMM regime detection, regime-conditional VaR
├── stress_testing.py      Historical / hypothetical / reverse stress tests
├── var_decomposition.py   Marginal, component, incremental VaR
├── greeks.py              Portfolio Greeks via Option-Pricing-Engine
├── liquidity.py           Bid-ask cost, LVaR, stressed LVaR
└── backtesting.py         Rolling backtest, Kupiec, Christoffersen, Basel

app/
├── dashboard.py           Streamlit dashboard (6 tabs)
└── report.py              HTML/PDF daily risk report generator

notebooks/
└── full_analysis.ipynb    End-to-end research notebook
```

## Modules

| Module | Description | FRM Topic |
|--------|-------------|-----------|
| `var.py` | Historical, Parametric, Monte Carlo, Cornish-Fisher, EVT VaR | VaR methods, EVT/GPD |
| `expected_shortfall.py` | ES for each VaR method | Expected Shortfall |
| `volatility.py` | EWMA, GARCH(1,1) conditional volatility | GARCH/EWMA volatility |
| `factor_model.py` | Fama-French regression, sector concentration | Factor models |
| `optimization.py` | Min-variance, max-Sharpe, risk parity, factor-constrained | Portfolio optimization |
| `correlation.py` | Rolling correlation, PCA of covariance | Correlation/PCA |
| `regime.py` | HMM regime detection, regime-conditional VaR | — |
| `stress_testing.py` | Historical/hypothetical/reverse stress tests | Stress testing |
| `var_decomposition.py` | Marginal, component, incremental VaR | VaR decomposition |
| `greeks.py` | Portfolio-level Greeks, scenario P&L | Options risk |
| `liquidity.py` | Liquidity-adjusted VaR | Liquidity-adjusted VaR |
| `backtesting.py` | Kupiec, Christoffersen, Basel traffic light | Backtesting VaR |

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/
```

## Run Dashboard

```bash
streamlit run app/dashboard.py
```

## Generate Risk Report

```python
from risk import Portfolio
from app.report import generate_daily_report

portfolio = Portfolio(tickers=["AAPL", "GOOGL", "MSFT", "JPM"], weights=[0.25, 0.25, 0.25, 0.25])
generate_daily_report(portfolio, output_path="report.html")
```
