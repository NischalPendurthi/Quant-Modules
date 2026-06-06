"""
plots.py - Publication-Quality Visualizations
==============================================
Generate 11 publication-ready performance plots:
1. Cumulative PnL
2. Daily PnL
3. Drawdown Curve
4. Rolling Sharpe Ratio
5. Rolling Information Coefficient (IC)
6. Signal Distribution
7. Actual vs Predicted
8. Feature Importance
9. Monthly Returns Heatmap
10. Exposure Through Time
11. Turnover Through Time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ══════════════════════════════════════════════════════════════════════
# Plot 1: Cumulative PnL
# ══════════════════════════════════════════════════════════════════════

def plot_cumulative_pnl(
    backtest_results: pd.DataFrame,
    title: str = "Cumulative NAV",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot cumulative portfolio value over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(backtest_results["timestamp"], backtest_results["nav"], 
            linewidth=2, label="NAV", color="darkblue")
    ax.fill_between(backtest_results["timestamp"], 
                     backtest_results["nav"], 
                     alpha=0.3, color="lightblue")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 2: Daily PnL
# ══════════════════════════════════════════════════════════════════════

def plot_daily_pnl(
    backtest_results: pd.DataFrame,
    title: str = "Daily Realized PnL",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot daily realized profit/loss."""
    daily_pnl = backtest_results.groupby(
        backtest_results["timestamp"].dt.date
    )["realized_pnl"].sum()

    colors = ["green" if x > 0 else "red" for x in daily_pnl.values]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Daily PnL ($)", fontsize=11)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 3: Drawdown Curve
# ══════════════════════════════════════════════════════════════════════

def plot_drawdown(
    nav: np.ndarray,
    timestamps: pd.Index,
    title: str = "Underwater Plot (Drawdown)",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot drawdown curve."""
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / (running_max + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(timestamps, drawdown, alpha=0.5, color="red", label="Drawdown")
    ax.plot(timestamps, drawdown, color="darkred", linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 4: Rolling Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════

def plot_rolling_sharpe(
    backtest_results: pd.DataFrame,
    window: int = 63,
    title: str = "Rolling Sharpe Ratio",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot rolling Sharpe ratio."""
    nav = backtest_results["nav"].values
    returns = np.diff(nav) / (nav[:-1] + 1e-12)
    
    rolling_sharpe = []
    for i in range(window, len(returns)):
        ret_w = returns[i - window: i]
        mean_ret = np.mean(ret_w)
        std_ret = np.std(ret_w)
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = 0
        rolling_sharpe.append(sharpe)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(backtest_results["timestamp"].iloc[window:], rolling_sharpe, 
            linewidth=2, label="Rolling Sharpe", color="darkgreen")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.fill_between(backtest_results["timestamp"].iloc[window:], rolling_sharpe, 
                     alpha=0.3, color="lightgreen")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 5: Rolling Information Coefficient
# ══════════════════════════════════════════════════════════════════════

def plot_rolling_ic(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 252,
    title: str = "Rolling Information Coefficient",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot rolling IC."""
    from scipy.stats import spearmanr
    
    ic_vals = []
    idx_vals = []
    
    # Ensure we have enough data
    if len(y_true) < window:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Insufficient data for rolling IC (need {window}, have {len(y_true)})", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax
    
    for i in range(window, len(y_true)):
        y_w = y_true.iloc[i - window: i]
        p_w = y_pred.iloc[i - window: i]
        valid = (~y_w.isna()) & (~p_w.isna())
        if valid.sum() >= 10:
            rho, _ = spearmanr(y_w[valid], p_w[valid])
            ic_vals.append(rho)
            idx_vals.append(y_true.index[i - 1])
    
    fig, ax = plt.subplots(figsize=figsize)
    if idx_vals:
        ax.plot(idx_vals, ic_vals, linewidth=2, label="Rolling IC", color="darkorange")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.fill_between(idx_vals, ic_vals, alpha=0.3, color="lightyellow")
    else:
        ax.text(0.5, 0.5, "No valid IC values computed", ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Information Coefficient", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 6: Signal Distribution
# ══════════════════════════════════════════════════════════════════════

def plot_signal_distribution(
    signal: np.ndarray,
    title: str = "Signal Distribution",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
):
    """Plot histogram and KDE of signal values."""
    valid = signal[np.isfinite(signal)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(valid, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_title("Signal Histogram", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Signal Value", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Box plot
    ax2.boxplot(valid, vert=True)
    ax2.set_title("Signal Box Plot", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Signal Value", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


# ══════════════════════════════════════════════════════════════════════
# Plot 7: Actual vs Predicted
# ══════════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Target",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
):
    """Scatter plot of actual vs predicted."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[valid]
    y_p = y_pred[valid]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_t, y_p, alpha=0.5, s=20, color="steelblue")
    
    # Add diagonal reference line
    if len(y_t) > 0 and len(y_p) > 0:
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Target", fontsize=11)
    ax.set_ylabel("Predicted Target", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 8: Feature Importance
# ══════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    top_k: int = 15,
    title: str = "Feature Importance (IC Ranking)",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    """Bar plot of feature importance."""
    # Ensure lengths match
    min_len = min(len(feature_names), len(importances))
    df_imp = pd.DataFrame({
        "feature": feature_names[:min_len],
        "importance": importances[:min_len],
    }).sort_values("importance", key=abs, ascending=False).head(top_k)
    
    if len(df_imp) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No feature importance data available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax
    
    colors = ["green" if x > 0 else "red" for x in df_imp["importance"]]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df_imp["feature"], df_imp["importance"], color=colors, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (IC)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 9: Monthly Returns Heatmap
# ══════════════════════════════════════════════════════════════════════

def plot_monthly_returns_heatmap(
    backtest_results: pd.DataFrame,
    title: str = "Monthly Returns Heatmap",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
):
    """Heatmap of returns by month and year."""
    backtest_results_copy = backtest_results.copy()
    backtest_results_copy["date"] = pd.to_datetime(backtest_results_copy["timestamp"]).dt.date
    
    # Group by month
    monthly_pnl = backtest_results_copy.groupby(
        pd.to_datetime(backtest_results_copy["date"]).dt.to_period("M")
    )["realized_pnl"].sum()
    
    # Reshape to year x month
    monthly_pnl.index = pd.to_datetime(monthly_pnl.index.to_timestamp())
    monthly_pnl_pivot = pd.DataFrame({
        "year": monthly_pnl.index.year,
        "month": monthly_pnl.index.month,
        "return": monthly_pnl.values,
    }).pivot(index="year", columns="month", values="return")
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(monthly_pnl_pivot, annot=True, fmt=".0f", cmap="RdYlGn", 
                center=0, cbar_kws={"label": "PnL ($)"}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Year", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 10: Exposure Through Time
# ══════════════════════════════════════════════════════════════════════

def plot_exposure(
    backtest_results: pd.DataFrame,
    title: str = "Market Exposure Through Time",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot portfolio exposure (position size) over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(backtest_results["timestamp"], backtest_results["position"], 
            linewidth=1.5, label="Position", color="purple", alpha=0.8)
    ax.fill_between(backtest_results["timestamp"], backtest_results["position"], 
                     alpha=0.3, color="plum")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Position Size", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Plot 11: Turnover Through Time
# ══════════════════════════════════════════════════════════════════════

def plot_turnover(
    backtest_results: pd.DataFrame,
    title: str = "Portfolio Turnover Through Time",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot cumulative turnover and transaction costs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Turnover
    # Use a bar width that's appropriate for the number of points
    bar_width = max(0.01, 1.0 / max(len(backtest_results), 1))
    ax1.bar(backtest_results["timestamp"], backtest_results["turnover"], 
            color="steelblue", alpha=0.7, width=bar_width)
    ax1.set_title("Daily Turnover", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Turnover ($)", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    # Cumulative costs
    cum_costs = backtest_results["transaction_cost"].cumsum()
    ax2.plot(backtest_results["timestamp"], cum_costs, 
             linewidth=2, label="Cumulative Costs", color="red")
    ax2.fill_between(backtest_results["timestamp"], cum_costs, 
                     alpha=0.3, color="lightcoral")
    ax2.set_title("Cumulative Transaction Costs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Cumulative Costs ($)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


# ══════════════════════════════════════════════════════════════════════
# Master Plotting Function
# ══════════════════════════════════════════════════════════════════════

def generate_all_plots(
    backtest_results: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    signal: np.ndarray,
    feature_names: List[str],
    feature_ics: List[float],
    output_dir: str = "plots",
) -> dict:
    """
    Generate all 11 publication-quality plots.

    Returns
    -------
    plot_dict : dict with paths to saved plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    nav = backtest_results["nav"].values
    timestamps = pd.to_datetime(backtest_results["timestamp"])
    
    plot_paths = {}
    
    print("[plots] Generating publication-quality visualizations...")
    
    plot_paths["cumulative_pnl"] = f"{output_dir}/01_cumulative_pnl.png"
    plot_cumulative_pnl(backtest_results, save_path=plot_paths["cumulative_pnl"])
    
    plot_paths["daily_pnl"] = f"{output_dir}/02_daily_pnl.png"
    plot_daily_pnl(backtest_results, save_path=plot_paths["daily_pnl"])
    
    plot_paths["drawdown"] = f"{output_dir}/03_drawdown.png"
    plot_drawdown(nav, timestamps, save_path=plot_paths["drawdown"])
    
    plot_paths["rolling_sharpe"] = f"{output_dir}/04_rolling_sharpe.png"
    plot_rolling_sharpe(backtest_results, save_path=plot_paths["rolling_sharpe"])
    
    plot_paths["rolling_ic"] = f"{output_dir}/05_rolling_ic.png"
    plot_rolling_ic(y_true, y_pred, save_path=plot_paths["rolling_ic"])
    
    plot_paths["signal_dist"] = f"{output_dir}/06_signal_distribution.png"
    plot_signal_distribution(signal, save_path=plot_paths["signal_dist"])
    
    plot_paths["actual_vs_pred"] = f"{output_dir}/07_actual_vs_predicted.png"
    plot_actual_vs_predicted(y_true.values, y_pred.values, save_path=plot_paths["actual_vs_pred"])
    
    plot_paths["feature_imp"] = f"{output_dir}/08_feature_importance.png"
    plot_feature_importance(feature_names, feature_ics, save_path=plot_paths["feature_imp"])
    
    plot_paths["monthly_returns"] = f"{output_dir}/09_monthly_returns_heatmap.png"
    plot_monthly_returns_heatmap(backtest_results, save_path=plot_paths["monthly_returns"])
    
    plot_paths["exposure"] = f"{output_dir}/10_exposure.png"
    plot_exposure(backtest_results, save_path=plot_paths["exposure"])
    
    plot_paths["turnover"] = f"{output_dir}/11_turnover.png"
    plot_turnover(backtest_results, save_path=plot_paths["turnover"])
    
    print("[plots] All plots saved to:", output_dir)
    return plot_paths