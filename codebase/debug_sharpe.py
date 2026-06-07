"""
debug_sharpe.py
--------------
Reads the actual submission CSVs and tells you EXACTLY why Sharpe is negative.
Run from your project root:  python debug_sharpe.py
"""

import numpy as np
import pandas as pd

TEAM = "ragasofrevenge"
LO_FILE = f"submissions/{TEAM}_longonly_results.csv"
LS_FILE = f"submissions/{TEAM}_longshort_results.csv"
FEE_BPS = 0.0010
INTERVALS_PER_YEAR = 75 * 252   # 18900

# ── helpers ──────────────────────────────────────────────────────────────────

def add_net_nav(df):
    df = df.copy()
    df["Friction_Costs"]  = df["Interval_Turnover"] * FEE_BPS
    df["Cumulative_Fees"] = df["Friction_Costs"].cumsum()
    df["Net_NAV"]         = df["Gross_NAV"] - df["Cumulative_Fees"]
    df["Return"]          = df["Net_NAV"].pct_change().fillna(0)
    return df

def grader_sharpe(df):
    r = df["Return"].replace([np.inf, -np.inf], np.nan).dropna()
    mean_r = r.mean()
    std_r  = r.std()
    if std_r == 0 or np.isnan(std_r):
        return 0.0
    return round((mean_r / std_r) * np.sqrt(INTERVALS_PER_YEAR), 4)

def diagnose(path, label, cap):
    print(f"\n{'='*70}")
    print(f"  {label}  —  {path}")
    print(f"{'='*70}")

    df = pd.read_csv(path)
    df = add_net_nav(df)
    n  = len(df)

    # ── 1. Basic shape ────────────────────────────────────────────────
    print(f"\n[1] Shape & period")
    print(f"    Rows            : {n:,}  (expected 94,500)")
    print(f"    First timestamp : {df['Timestamp'].iloc[0]}")
    print(f"    Last  timestamp : {df['Timestamp'].iloc[-1]}")

    # ── 2. NAV journey ───────────────────────────────────────────────
    gross_start = df["Gross_NAV"].iloc[0]
    gross_end   = df["Gross_NAV"].iloc[-1]
    net_start   = df["Net_NAV"].iloc[0]
    net_end     = df["Net_NAV"].iloc[-1]
    total_fees  = df["Cumulative_Fees"].iloc[-1]

    print(f"\n[2] NAV journey")
    print(f"    Gross_NAV start : ${gross_start:>14,.2f}")
    print(f"    Gross_NAV end   : ${gross_end:>14,.2f}")
    print(f"    Gross return    : {(gross_end/gross_start - 1)*100:+.4f}%")
    print(f"    Total fees      : ${total_fees:>14,.2f}")
    print(f"    Net_NAV start   : ${net_start:>14,.2f}")
    print(f"    Net_NAV end     : ${net_end:>14,.2f}")
    print(f"    Net return      : {(net_end/net_start - 1)*100:+.4f}%")

    # ── 3. Activity (are we actually trading?) ────────────────────────
    active = (df["Gross_Exposure"] > 0.01).sum()
    trades = (df["Interval_Turnover"] > 0.01).sum()
    print(f"\n[3] Activity")
    print(f"    Rows with open position : {active:,}  ({active/n*100:.1f}%)")
    print(f"    Rows with turnover      : {trades:,}  ({trades/n*100:.1f}%)")
    print(f"    Total turnover $        : ${df['Interval_Turnover'].sum():,.2f}")

    # ── 4. Return distribution ────────────────────────────────────────
    r = df["Return"]
    pos = (r > 0).sum()
    neg = (r < 0).sum()
    zer = (r == 0).sum()
    print(f"\n[4] Return distribution  (all {n:,} rows)")
    print(f"    Positive returns : {pos:,}  ({pos/n*100:.1f}%)")
    print(f"    Zero returns     : {zer:,}  ({zer/n*100:.1f}%)")
    print(f"    Negative returns : {neg:,}  ({neg/n*100:.1f}%)")
    print(f"    Mean return/bar  : {r.mean():.8f}")
    print(f"    Std  return/bar  : {r.std():.8f}")
    print(f"    Min  return      : {r.min():.8f}  @ row {r.idxmin()}")
    print(f"    Max  return      : {r.max():.8f}  @ row {r.idxmax()}")

    # ── 5. WHERE are negative returns happening? ──────────────────────
    # Split into: (a) rows where we hold a position  (b) idle rows
    hold_mask = df["Gross_Exposure"].shift(1).fillna(0) > 0.01
    r_hold  = r[hold_mask]
    r_idle  = r[~hold_mask]

    print(f"\n[5] Returns split: holding vs idle")
    print(f"    ▸ While HOLDING position ({hold_mask.sum():,} rows):")
    print(f"      mean={r_hold.mean():.8f}  std={r_hold.std():.8f}  "
          f"neg={( r_hold < 0).sum():,}  pos={(r_hold > 0).sum():,}")
    print(f"    ▸ While IDLE / flat ({(~hold_mask).sum():,} rows):")
    print(f"      mean={r_idle.mean():.8f}  std={r_idle.std():.8f}  "
          f"neg={(r_idle < 0).sum():,}  pos={(r_idle > 0).sum():,}")

    # ── 6. Fee-drag on idle rows ──────────────────────────────────────
    idle_neg_return_rows = r_idle[r_idle < 0]
    print(f"\n[6] Fee-drag analysis (idle rows with negative return)")
    print(f"    Count  : {len(idle_neg_return_rows):,}")
    if len(idle_neg_return_rows) > 0:
        # check if those negative returns are purely from fee cumulation
        # on idle rows the Net_NAV drop = Friction_Costs[t], Gross_NAV is flat
        sample = df[~hold_mask & (r < 0)][["Timestamp","Gross_NAV","Cumulative_Fees","Net_NAV","Return"]].head(10)
        print(f"    Sample rows:\n{sample.to_string(index=False)}")

    # ── 7. Temporal breakdown — where in the 94,500 rows is the damage? ──
    print(f"\n[7] Cumulative Net_NAV at quartiles")
    qs = [0, 0.25, 0.5, 0.75, 1.0]
    for q in qs:
        idx = min(int(q * (n-1)), n-1)
        ts  = df["Timestamp"].iloc[idx]
        nav = df["Net_NAV"].iloc[idx]
        ret = (nav / net_start - 1) * 100
        print(f"    {int(q*100):3d}%  row {idx:6d}  {ts}  Net_NAV=${nav:>14,.2f}  ({ret:+.2f}%)")

    # ── 8. Grader Sharpe ─────────────────────────────────────────────
    sharpe = grader_sharpe(df)
    print(f"\n[8] Grader Sharpe : {sharpe}")

    # ── 9. THE KEY DIAGNOSTIC — what Sharpe would be if we only scored ──
    #       active rows (what you *think* you're measuring)
    r_only_active = r[hold_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if r_only_active.std() > 0:
        sharpe_active_only = round(
            (r_only_active.mean() / r_only_active.std()) * np.sqrt(INTERVALS_PER_YEAR), 4
        )
    else:
        sharpe_active_only = 0.0
    print(f"    Sharpe if only active rows counted : {sharpe_active_only}")
    print(f"    ← Gap between these two = the idle-row dilution effect")

    # ── 10. Hypothesis check: is the problem in early vs late rows? ───
    mid = n // 2
    r_first_half = r.iloc[:mid]
    r_second_half = r.iloc[mid:]
    def sharpe_of(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.std() == 0: return 0.0
        return round((s.mean() / s.std()) * np.sqrt(INTERVALS_PER_YEAR), 4)

    print(f"\n[10] First-half vs second-half Sharpe")
    print(f"    First  half (rows 0–{mid})   : {sharpe_of(r_first_half)}")
    print(f"    Second half (rows {mid}–{n}) : {sharpe_of(r_second_half)}")
    print(f"    ← If first half is very negative → early-period regime issue")
    print(f"    ← If both halves are ~0 → idle-row dilution is the whole story")

    # ── 11. Root-cause verdict ────────────────────────────────────────
    print(f"\n[ROOT CAUSE VERDICT]")
    mean_r = r.mean()
    if mean_r < 0:
        print(f"  ❌ mean_return is NEGATIVE ({mean_r:.8f})")
        print(f"     → Your strategy LOSES money on Net_NAV basis over the full window.")
        pct_idle = (~hold_mask).sum() / n
        if pct_idle > 0.9:
            print(f"     → You are idle {pct_idle*100:.0f}% of the time.")
            print(f"        Fee drag on turnover bars is creating a persistent negative drift.")
    elif mean_r >= 0 and sharpe < 0:
        print(f"  ⚠️  mean_return is positive ({mean_r:.8f}) but Sharpe is negative")
        print(f"     → This is IMPOSSIBLE with std > 0. Check for inf/nan in returns.")
    else:
        print(f"  ✅ mean_return = {mean_r:.8f}")
        idle_pct = (~hold_mask).sum() / n * 100
        print(f"  ⚠️  You are idle {idle_pct:.0f}% of the time.")
        print(f"     Idle rows have Return≈0 but std≠0 → Sharpe gets diluted.")
        print(f"     Trade more frequently to push mean_return up vs std.")

# ── Run ───────────────────────────────────────────────────────────────────────
diagnose(LO_FILE, "LONG-ONLY", 1_000_000)
diagnose(LS_FILE, "LONG-SHORT", 2_000_000)