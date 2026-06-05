"""
Real historical scenario backtest using actual market data.
Runs the Portfolio Risk Engine against 4 crisis windows to produce
defensible numbers for the resume.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from risk.data import compute_returns, portfolio_returns
from risk.var import historical_var, parametric_var, evt_var
from risk.volatility import compare_vol_models

TICKERS  = ["AAPL", "GOOGL", "MSFT", "JPM"]
WEIGHTS  = [0.30, 0.25, 0.25, 0.20]

# 6-month pre-crisis window for VaR calibration + the crisis window itself
SCENARIOS = {
    "2008 GFC":        {"calibrate": ("2007-07-01", "2008-09-14"), "crisis": ("2008-09-15", "2009-03-31")},
    "COVID-19":        {"calibrate": ("2019-08-01", "2020-02-19"), "crisis": ("2020-02-20", "2020-03-23")},
    "2022 Rate Hikes": {"calibrate": ("2021-06-01", "2021-12-31"), "crisis": ("2022-01-01", "2022-10-14")},
    "2023 SVB":        {"calibrate": ("2022-09-01", "2023-03-08"), "crisis": ("2023-03-09", "2023-03-24")},
}

def fetch(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw = raw[tickers].dropna()
    return raw

def max_drawdown(ret_series):
    cum = (1 + ret_series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def print_separator():
    print("=" * 64)

print_separator()
print("PORTFOLIO: AAPL 30% | GOOGL 25% | MSFT 25% | JPM 20%")
print_separator()

summary_rows = []

for name, windows in SCENARIOS.items():
    print(f"\n{'─'*64}")
    print(f"  {name}")
    print(f"{'─'*64}")

    # --- fetch calibration data ---
    try:
        cal_prices = fetch(TICKERS, windows["calibrate"][0], windows["calibrate"][1])
    except Exception as e:
        print(f"  [SKIP] Could not fetch calibration data: {e}")
        continue

    cal_rets   = compute_returns(cal_prices, method="log")
    cal_port   = portfolio_returns(cal_rets, WEIGHTS)

    # --- pre-crisis VaR (95% and 99%) ---
    var_95 = historical_var(cal_port, confidence=0.95)
    var_99 = historical_var(cal_port, confidence=0.99)
    print(f"  Pre-crisis  95% VaR  : {var_95.var*100:.2f}%  |  ES: {var_95.es*100:.2f}%")
    print(f"  Pre-crisis  99% VaR  : {var_99.var*100:.2f}%  |  ES: {var_99.es*100:.2f}%")

    # --- fetch crisis data ---
    try:
        cr_prices = fetch(TICKERS, windows["crisis"][0], windows["crisis"][1])
    except Exception as e:
        print(f"  [SKIP] Could not fetch crisis data: {e}")
        continue

    cr_rets  = compute_returns(cr_prices, method="log")
    cr_port  = portfolio_returns(cr_rets, WEIGHTS)

    # --- crisis stats ---
    worst_day   = cr_port.min()
    total_ret   = (1 + cr_port).prod() - 1
    mdd         = max_drawdown(cr_port)
    n_days      = len(cr_port)
    breach_95   = (cr_port < -var_95.var).sum()
    breach_99   = (cr_port < -var_99.var).sum()
    breach_pct  = breach_95 / n_days * 100

    print(f"  Crisis window        : {windows['crisis'][0]}  →  {windows['crisis'][1]}  ({n_days} trading days)")
    print(f"  Total return         : {total_ret*100:.1f}%")
    print(f"  Max drawdown         : {mdd*100:.1f}%")
    print(f"  Worst single day     : {worst_day*100:.2f}%")
    print(f"  95% VaR breaches     : {breach_95}/{n_days}  ({breach_pct:.1f}%)   [expected ~5%]")
    print(f"  99% VaR breaches     : {breach_99}/{n_days}")

    # --- vol comparison ---
    vol_stats = compare_vol_models(cal_port)
    static_vol = float(vol_stats["static"].iloc[-1]) * np.sqrt(252) * 100
    garch_vol  = float(vol_stats["garch"].iloc[-1])  * np.sqrt(252) * 100
    print(f"  Pre-crisis annl. vol : static={static_vol:.1f}%  GARCH={garch_vol:.1f}%")

    summary_rows.append({
        "Scenario":        name,
        "Window":          f"{windows['crisis'][0]} → {windows['crisis'][1]}",
        "Total Return":    f"{total_ret*100:.1f}%",
        "Max Drawdown":    f"{mdd*100:.1f}%",
        "Worst Day":       f"{worst_day*100:.2f}%",
        "95% VaR Breaches":f"{breach_95}/{n_days} ({breach_pct:.1f}%)",
    })

print(f"\n{'='*64}")
print("SUMMARY TABLE")
print(f"{'='*64}")
df = pd.DataFrame(summary_rows).set_index("Scenario")
print(df.to_string())
print(f"{'='*64}\n")
