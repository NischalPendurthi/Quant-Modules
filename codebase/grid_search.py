"""
grid_search.py
--------------
Fast grid search around current optimum config values.
Loads data + signal ONCE, then sweeps backtest parameters only.
Run: python grid_search.py

Searches 6 parameters with 3 values each = smart subset of combinations.
Uses the saved model (no retraining).
"""

import numpy as np
import pandas as pd
import pickle, json, sys, itertools, time
from pathlib import Path

sys.path.insert(0, ".")
from data import load_data, create_target_variable
from features import engineer_features

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH    = str(Path(__file__).parent.parent / "moccm_intraday_blackbox.csv")
GRADER_ROWS  = 94_500
FEE_BPS      = 0.0010
INTERVALS_PER_YEAR = 75 * 252

FEATURES = [
    "f1_mom1","f1_rzsc_3","f1_rzsc_6","f1_rzsc_12",
    "f1_mom3","f1_rzsc_24","f1_mom6","f1_mom12","f1_mom24",
]

# ── Grid — 3 values per param centred on current best ─────────────────
GRID = {
    "lo_entry"      : [0.85, 0.95, 0.97],      # LO entry percentile
    "lo_exit"       : [0.15, 0.42, 0.30],       # LO exit percentile
    "lo_invest_frac": [0.85, 0.90, 0.95],       # fraction of capital per LO trade
    "ls_long_pct"   : [0.93, 0.95, 0.97],       # LS long entry percentile
    "ls_long_frac"  : [0.08, 0.10, 0.12],       # LS long position fraction
    "ls_short_frac" : [0.01, 0.02, 0.03],       # LS short position fraction
}

# To keep runtime manageable, search LO and LS params independently
# then combine the best of each.
LO_GRID = list(itertools.product(
    GRID["lo_entry"], GRID["lo_exit"], GRID["lo_invest_frac"]
))  # 27 combos
LS_GRID = list(itertools.product(
    GRID["ls_long_pct"], GRID["ls_long_frac"], GRID["ls_short_frac"]
))  # 27 combos

print(f"LO grid: {len(LO_GRID)} combos | LS grid: {len(LS_GRID)} combos")

# ── Load data & signal ONCE ───────────────────────────────────────────
print("\nLoading data (once)...")
t0 = time.time()
df = load_data(DATA_PATH, timestamp_col="Timestamp",
               ticker_col="Ticker", price_col="Close", volume_col="Volume")
df = df.ffill().bfill()
df = create_target_variable(df, price_col="Close", horizon=1,
                            target_type="return", group_by_ticker=False)
df_eng = engineer_features(df, windows=(6,12,24), lags=(1,3,6,12,24),
                           add_regime=True, add_cross_sectional=True)
df_eng = df_eng.dropna(subset=["y"])

with open("artifacts/best_model.pkl","rb") as f:
    model = pickle.load(f)
with open("artifacts/signal_sign.json") as f:
    sign = json.load(f)["signal_sign"]

sub      = df_eng.iloc[-GRADER_ROWS:]
prices   = sub["Close"].values.astype(float)
timestamps = sub.index
X_sub    = sub[FEATURES].fillna(0).values
signal   = model.predict(X_sub) * sign

print(f"Data ready in {time.time()-t0:.1f}s. Signal shape: {signal.shape}")

# ── Regime IC array (precompute for speed) ────────────────────────────
print("Precomputing regime IC array...")
from scipy.stats import spearmanr
REGIME_WINDOW    = 2000
REGIME_THRESHOLD = -0.02
price_returns = np.diff(prices, prepend=prices[0]) / (np.abs(prices) + 1e-12)

regime_ic_arr = np.ones(len(signal))
for t in range(REGIME_WINDOW + 1, len(signal)):
    s_w = signal[t - REGIME_WINDOW - 1: t - 1]
    r_w = price_returns[t - REGIME_WINDOW: t]
    ic, _ = spearmanr(s_w, r_w)
    regime_ic_arr[t] = float(ic) if np.isfinite(ic) else 0.0

regime_ok_arr = regime_ic_arr >= REGIME_THRESHOLD
print(f"Regime gate: {regime_ok_arr.sum():,}/{len(regime_ok_arr):,} bars open ({regime_ok_arr.mean()*100:.1f}%)")

# ── Sharpe helper ─────────────────────────────────────────────────────
def grader_sharpe(gross_nav, turnover):
    cum_fees = (turnover * FEE_BPS).cumsum()
    net_nav  = gross_nav - cum_fees
    r = pd.Series(net_nav).pct_change().fillna(0)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if r.std() == 0: return 0.0, net_nav[-1], cum_fees[-1]
    sh = round((r.mean() / r.std()) * np.sqrt(INTERVALS_PER_YEAR), 4)
    return sh, net_nav[-1], cum_fees[-1]

# ── Fast LO backtest ──────────────────────────────────────────────────
ATOL = 0.01
LO_CAP  = 1_000_000.0
LS_CAP  = 2_000_000.0
MAX_TO_FRAC = 0.02   # fixed turnover cap

def run_lo(lo_entry, lo_exit, lo_invest_frac):
    entry_thr = np.percentile(signal, lo_entry * 100)
    exit_thr  = np.percentile(signal, lo_exit  * 100)
    n = len(signal)
    cash, shares = LO_CAP, 0.0
    entry_bar = -3
    max_to = LO_CAP * MAX_TO_FRAC

    gross_nav_arr = np.empty(n)
    turnover_arr  = np.zeros(n)
    cash_arr      = np.empty(n)

    for t in range(n):
        p = prices[t]
        s = signal[t]
        is_last = (t == n - 1)
        prev = shares

        if t == 0 or is_last:
            tgt = 0.0
        elif not regime_ok_arr[t]:
            tgt = 0.0
        elif shares == 0 and s >= entry_thr and t < n - 4:
            invest = min(cash * lo_invest_frac, LO_CAP * lo_invest_frac)
            tgt = invest / p if p > 1e-10 else 0.0
            entry_bar = t
        elif shares > 0 and s < exit_thr:
            tgt = 0.0
        else:
            tgt = shares

        delta = tgt - shares
        to    = abs(delta) * p
        if to > max_to:
            delta = delta * max_to / to
            to    = max_to
            tgt   = prev + delta

        cc = -delta * p
        if cc < 0 and (-cc) > cash + ATOL:
            delta = cash / p - prev
            if delta < 0: delta = 0.0
            tgt = prev + delta
            cc  = -delta * p
            to  = abs(delta) * p

        cash   += cc
        shares  = tgt
        cash    = max(cash, 0.0)

        ge = abs(shares) * p
        if ge > LO_CAP + ATOL:
            shares = LO_CAP / p
            cash  += (ge - LO_CAP)
            ge     = LO_CAP

        if is_last and abs(shares) > 1e-10:
            cash  += shares * p
            shares = 0.0

        gross_nav_arr[t] = cash + shares * p
        turnover_arr[t]  = to
        cash_arr[t]      = cash

    return gross_nav_arr, turnover_arr

def run_ls(ls_long_pct, ls_long_frac, ls_short_frac):
    upper_thr = np.percentile(signal, ls_long_pct  * 100)
    lower_thr = np.percentile(signal, (1 - ls_long_pct) * 100)
    n = len(signal)
    cash, shares = LS_CAP, 0.0
    max_to = LS_CAP * MAX_TO_FRAC

    gross_nav_arr = np.empty(n)
    turnover_arr  = np.zeros(n)

    for t in range(n):
        p = prices[t]
        s = signal[t]
        is_last = (t == n - 1)
        prev = shares

        if t == 0 or is_last:
            tgt = 0.0
        elif not regime_ok_arr[t]:
            tgt = 0.0
        elif shares == 0:
            if s > upper_thr and t < n - 4:
                alloc = min(cash * ls_long_frac, LS_CAP * ls_long_frac)
                tgt = alloc / p if p > 1e-10 else 0.0
            elif s < lower_thr and t < n - 4:
                alloc = min(cash * ls_short_frac, LS_CAP * ls_short_frac)
                tgt = -(alloc / p) if p > 1e-10 else 0.0
            else:
                tgt = shares
        elif shares > 0 and s < lower_thr:
            tgt = 0.0
        elif shares < 0 and s > upper_thr:
            tgt = 0.0
        else:
            tgt = shares

        delta = tgt - shares
        to    = abs(delta) * p
        if to > max_to:
            delta = delta * max_to / to
            to    = max_to
            tgt   = prev + delta

        cc = -delta * p
        if delta > 0 and cc < 0 and (-cc) > cash + ATOL:
            delta = cash / p
            cc    = -delta * p
            to    = abs(delta) * p
            tgt   = prev + delta

        cash   += cc
        shares  = tgt
        cash    = max(cash, 0.0)

        ge = abs(shares) * p
        if ge > LS_CAP + ATOL:
            scale  = LS_CAP / ge
            cash  += abs(shares) * (1 - scale) * p
            shares = shares * scale

        if is_last and abs(shares) > 1e-10:
            cash  += shares * p
            shares = 0.0

        gross_nav_arr[t] = cash + shares * p
        turnover_arr[t]  = to

    return gross_nav_arr, turnover_arr

# ── Run LO grid ───────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"LONG-ONLY GRID  ({len(LO_GRID)} combos)")
print(f"{'─'*70}")
print(f"{'lo_entry':>8} {'lo_exit':>7} {'invest':>7} │ {'Sharpe':>7} {'NetRet%':>8} {'Fees$':>10}")
print(f"{'─'*70}")

lo_results = []
for lo_entry, lo_exit, lo_invest in LO_GRID:
    gn, to = run_lo(lo_entry, lo_exit, lo_invest)
    sh, net_end, fees = grader_sharpe(gn, to)
    net_ret = (net_end / LO_CAP - 1) * 100
    lo_results.append((sh, lo_entry, lo_exit, lo_invest, net_ret, fees))
    print(f"{lo_entry:>8.2f} {lo_exit:>7.2f} {lo_invest:>7.2f} │ {sh:>7.4f} {net_ret:>7.2f}% ${fees:>10,.0f}")

lo_results.sort(reverse=True)
best_lo = lo_results[0]
print(f"\n★ Best LO: entry={best_lo[1]} exit={best_lo[2]} invest={best_lo[3]}  →  Sharpe={best_lo[0]}  NetRet={best_lo[4]:.2f}%")

# ── Run LS grid ───────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"LONG-SHORT GRID  ({len(LS_GRID)} combos)")
print(f"{'─'*70}")
print(f"{'ls_pct':>6} {'long_f':>7} {'short_f':>8} │ {'Sharpe':>7} {'NetRet%':>8} {'Fees$':>10}")
print(f"{'─'*70}")

ls_results = []
for ls_pct, ls_lf, ls_sf in LS_GRID:
    gn, to = run_ls(ls_pct, ls_lf, ls_sf)
    sh, net_end, fees = grader_sharpe(gn, to)
    net_ret = (net_end / LS_CAP - 1) * 100
    ls_results.append((sh, ls_pct, ls_lf, ls_sf, net_ret, fees))
    print(f"{ls_pct:>6.2f} {ls_lf:>7.3f} {ls_sf:>8.3f} │ {sh:>7.4f} {net_ret:>7.2f}% ${fees:>10,.0f}")

ls_results.sort(reverse=True)
best_ls = ls_results[0]
print(f"\n★ Best LS: pct={best_ls[1]} long_f={best_ls[2]} short_f={best_ls[3]}  →  Sharpe={best_ls[0]}  NetRet={best_ls[4]:.2f}%")

# ── Blended summary ───────────────────────────────────────────────────
blended = round((best_lo[0] + best_ls[0]) / 2, 4)
print(f"\n{'='*70}")
print(f"BEST BLENDED SHARPE: ({best_lo[0]} + {best_ls[0]}) / 2 = {blended}")
print(f"{'='*70}")
print(f"\nPaste these into main.py Config:")
print(f"  LO_ENTRY_PERCENTILE       = {best_lo[1]}")
print(f"  LO_EXIT_PERCENTILE        = {best_lo[2]}")
print(f"  LO_INVEST_FRAC            = {best_lo[3]}   # use in backtest invest line")
print(f"  LS_LONG_PERCENTILE        = {best_ls[1]}")
print(f"  LS_LONG_POSITION_FRACTION = {best_ls[2]}")
print(f"  LS_SHORT_POSITION_FRACTION= {best_ls[3]}")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")