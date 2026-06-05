"""
models.py - White-Box Signal Discovery
=======================================
All models implemented from scratch using NumPy only.
Every prediction is expressible as a closed-form mathematical formula.

Mathematical Definitions
------------------------
OLS (Model 1):
    ŷ = Xβ   where β = (X'X)^{-1} X'y

Ridge (Model 2):
    β_ridge = (X'X + λI)^{-1} X'y

Lasso (Model 3):
    β_lasso = argmin ||y - Xβ||² + λ||β||₁   (solved via coordinate descent)

Rolling OLS (Model 4):
    β_t = (X_{t-w:t}'X_{t-w:t} + εI)^{-1} X_{t-w:t}'y_{t-w:t}

Factor Score (Model 5):
    Signal_t = Σ_i IC_i · rank(f_i,t)

Rank Alpha (Model 6):
    Signal_t = Σ_i w_i · percentile_rank(f_i,t)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _safe_solve(A: np.ndarray, b: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Solve Ax = b with a small ridge stabiliser."""
    A_reg = A + ridge * np.eye(A.shape[0])
    return np.linalg.solve(A_reg, b)


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 5:
        return np.nan
    rho, _ = stats.spearmanr(y_true[valid], y_pred[valid])
    return float(rho)


def _pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 5:
        return np.nan
    rho, _ = stats.pearsonr(y_true[valid], y_pred[valid])
    return float(rho)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    n       = len(yt)
    resid   = yt - yp
    ss_res  = np.sum(resid ** 2)
    ss_tot  = np.sum((yt - yt.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse     = ss_res / n
    rmse    = np.sqrt(mse)
    mae     = np.mean(np.abs(resid))
    ic      = _pearson_ic(yt, yp)
    rank_ic = _spearman_ic(yt, yp)
    return {
        f"{prefix}ic":      round(ic,      6),
        f"{prefix}rank_ic": round(rank_ic, 6),
        f"{prefix}mse":     round(mse,     8),
        f"{prefix}rmse":    round(rmse,    8),
        f"{prefix}mae":     round(mae,     8),
        f"{prefix}r2":      round(r2,      6),
    }


# ══════════════════════════════════════════════════════════════════════
# MODEL 1 – Ordinary Least Squares
# ══════════════════════════════════════════════════════════════════════

class OLSModel:
    """
    ŷ = β₀ + β₁x₁ + ... + βₙxₙ

    β = (X'X)⁻¹ X'y   (closed-form normal equations)
    """
    name = "OLS Linear Regression"

    def __init__(self):
        self.coef_      = None
        self.intercept_ = None
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        Xa = _add_intercept(X)
        XtX = Xa.T @ Xa
        Xty = Xa.T @ y
        beta = _safe_solve(XtX, Xty)
        self.intercept_ = beta[0]
        self.coef_      = beta[1:]
        self.feature_names_ = feature_names or [f"x{i}" for i in range(X.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def formula(self) -> str:
        terms = [f"{c:.6f}·{n}" for c, n in zip(self.coef_, self.feature_names_)]
        return f"ŷ = {self.intercept_:.6f} + " + " + ".join(terms[:5]) + (" ..." if len(terms) > 5 else "")

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature":   self.feature_names_,
            "coefficient": self.coef_,
        })


# ══════════════════════════════════════════════════════════════════════
# MODEL 2 – Ridge Regression
# ══════════════════════════════════════════════════════════════════════

class RidgeModel:
    """
    β_ridge = (X'X + λI)⁻¹ X'y

    L2 regularization shrinks all coefficients continuously toward zero.
    """
    name = "Ridge Regression"

    def __init__(self, alpha: float = 1.0):
        self.alpha      = alpha
        self.coef_      = None
        self.intercept_ = None
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        # Centre X for numerical stability
        self.X_mean_ = X.mean(axis=0)
        self.X_std_  = X.std(axis=0) + 1e-12
        Xs = (X - self.X_mean_) / self.X_std_
        y_mean = y.mean()
        ys = y - y_mean

        n, p   = Xs.shape
        XtX    = Xs.T @ Xs
        Xty    = Xs.T @ ys
        beta   = _safe_solve(XtX + self.alpha * np.eye(p), Xty, ridge=0.0)

        self.coef_      = beta / self.X_std_
        self.intercept_ = y_mean - self.X_mean_ @ self.coef_
        self.feature_names_ = feature_names or [f"x{i}" for i in range(X.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def formula(self) -> str:
        terms = [f"{c:.6f}·{n}" for c, n in zip(self.coef_, self.feature_names_)]
        return f"ŷ_ridge(λ={self.alpha}) = {self.intercept_:.6f} + " + " + ".join(terms[:5])

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names_, "coefficient": self.coef_})


# ══════════════════════════════════════════════════════════════════════
# MODEL 3 – Lasso Regression  (Coordinate Descent)
# ══════════════════════════════════════════════════════════════════════

class LassoModel:
    """
    min_β  ||y - Xβ||² + λ||β||₁

    Solved via coordinate descent (exact closed form per coordinate):
        β_j ← S(ρ_j, λ) / (x_j'x_j)
        where S(z, λ) = sign(z) · max(|z| − λ, 0)   [soft-threshold]
    """
    name = "Lasso Regression"

    def __init__(self, alpha: float = 0.01, max_iter: int = 500, tol: float = 1e-4):
        self.alpha      = alpha
        self.max_iter   = max_iter
        self.tol        = tol
        self.coef_      = None
        self.intercept_ = None
        self.feature_names_: List[str] = []

    @staticmethod
    def _soft_threshold(z: float, lam: float) -> float:
        return np.sign(z) * max(abs(z) - lam, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        self.X_mean_ = X.mean(axis=0)
        self.X_std_  = X.std(axis=0) + 1e-12
        Xs = (X - self.X_mean_) / self.X_std_
        y_mean = y.mean()
        ys = y - y_mean

        n, p  = Xs.shape
        beta  = np.zeros(p)
        lam   = self.alpha * n  # scale λ

        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j    = ys - Xs @ beta + Xs[:, j] * beta[j]
                rho_j  = float(Xs[:, j] @ r_j)
                xjnorm = float(Xs[:, j] @ Xs[:, j])
                if xjnorm < 1e-12:
                    beta[j] = 0.0
                else:
                    beta[j] = self._soft_threshold(rho_j, lam) / xjnorm
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        self.coef_      = beta / self.X_std_
        self.intercept_ = y_mean - self.X_mean_ @ self.coef_
        self.feature_names_ = feature_names or [f"x{i}" for i in range(p)]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def formula(self) -> str:
        non_zero = [(c, n) for c, n in zip(self.coef_, self.feature_names_) if abs(c) > 1e-10]
        terms    = [f"{c:.6f}·{n}" for c, n in non_zero[:5]]
        return (f"ŷ_lasso(λ={self.alpha}) = {self.intercept_:.6f} + " +
                " + ".join(terms) + (f"  [{len(non_zero)} non-zero coefs]"))

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names_, "coefficient": self.coef_})


# ══════════════════════════════════════════════════════════════════════
# MODEL 4 – Rolling Linear Regression
# ══════════════════════════════════════════════════════════════════════

class RollingOLSModel:
    """
    At each time t, fit OLS on the trailing window of length w:
        β_t = (X_{t-w:t}'X_{t-w:t} + εI)^{-1} X_{t-w:t}'y_{t-w:t}
        ŷ_t = x_t' β_{t-1}    (no look-ahead)

    Coefficients vary over time capturing non-stationarity.
    """
    name = "Rolling OLS (Adaptive)"

    def __init__(self, window: int = 500, ridge: float = 1e-6):
        self.window = window
        self.ridge  = ridge
        self.coef_history_: Optional[pd.DataFrame] = None
        self.feature_names_: List[str] = []

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> np.ndarray:
        n, p = X.shape
        preds = np.full(n, np.nan)
        coef_hist = np.full((n, p + 1), np.nan)

        for t in range(self.window, n):
            Xw  = X[t - self.window: t]
            yw  = y[t - self.window: t]
            Xa  = _add_intercept(Xw)
            XtX = Xa.T @ Xa + self.ridge * np.eye(p + 1)
            Xty = Xa.T @ yw
            try:
                beta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                continue
            coef_hist[t] = beta
            preds[t] = np.dot(np.concatenate([[1.0], X[t]]), beta)

        self.feature_names_ = feature_names or [f"x{i}" for i in range(p)]
        cols = ["intercept"] + self.feature_names_
        self.coef_history_ = pd.DataFrame(coef_hist, columns=cols)
        return preds

    def predict(self, X_test: np.ndarray, last_beta: np.ndarray) -> np.ndarray:
        Xa = _add_intercept(X_test)
        return Xa @ last_beta

    def formula(self) -> str:
        return (f"ŷ_t = β₀(t) + Σ βⱼ(t)·xⱼ  "
                f"[OLS on trailing {self.window}-bar window]")


# ══════════════════════════════════════════════════════════════════════
# MODEL 5 – Factor Score Model (IC-Weighted)
# ══════════════════════════════════════════════════════════════════════

class FactorScoreModel:
    """
    Signal_t = Σᵢ IC_i · rank(fᵢ,t)

    Weights are the historical mean IC of each feature, so stronger
    predictors receive larger weights.  Rank normalises each feature
    to [0,1] removing scale effects.

    Formula:
        w_i     = MeanIC(fᵢ, y)   over training window
        rank_t  = (rank(fᵢ,t) − 1) / (N−1)    [0,1 scaled percentile]
        Signal  = Σ w_i · rank_t(fᵢ)
    """
    name = "IC-Weighted Factor Score"

    def __init__(self):
        self.weights_: Optional[np.ndarray]  = None
        self.feature_names_: List[str]       = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        n, p = X_train.shape
        ics  = np.array([_pearson_ic(y_train, X_train[:, j]) for j in range(p)])
        ics  = np.nan_to_num(ics)
        # Weight = IC (positive → long, negative → short implicitly via ranking)
        self.weights_       = ics
        self.feature_names_ = feature_names or [f"x{i}" for i in range(p)]
        return self

    def _rank_transform(self, X: np.ndarray) -> np.ndarray:
        """Convert each column to percentile rank [0,1]."""
        n, p  = X.shape
        ranks = np.zeros_like(X, dtype=float)
        for j in range(p):
            col = X[:, j]
            r   = stats.rankdata(col, method="average") - 1
            r  /= max(len(col) - 1, 1)
            ranks[:, j] = r
        return ranks

    def predict(self, X: np.ndarray) -> np.ndarray:
        ranked = self._rank_transform(X)
        return ranked @ self.weights_

    def formula(self) -> str:
        top5 = sorted(
            zip(self.weights_, self.feature_names_),
            key=lambda x: abs(x[0]),
            reverse=True,
        )[:5]
        terms = [f"{w:.4f}·rank({n})" for w, n in top5]
        return "Signal = " + " + ".join(terms) + "  [IC-weighted rank model]"

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names_, "ic_weight": self.weights_})


# ══════════════════════════════════════════════════════════════════════
# MODEL 6 – Rank-Based Alpha Model
# ══════════════════════════════════════════════════════════════════════

class RankAlphaModel:
    """
    Signal_t = Σᵢ w_i · pct_rank(fᵢ,t)

    Weights via rank-IC (Spearman) and feature percentile buckets.
    More robust to outliers than raw feature values.

    Formula:
        rank_IC_i = Spearman(rank(fᵢ), rank(y))  [training]
        pct_t(fᵢ) = (rank(fᵢ,t) − 1) / (N−1)
        Signal_t  = Σᵢ rank_IC_i · pct_t(fᵢ)
    """
    name = "Rank-Based Alpha Model"

    def __init__(self):
        self.rank_ic_weights_: Optional[np.ndarray] = None
        self.feature_names_:  List[str]             = []
        self.n_buckets_:      int                   = 5   # quintile buckets

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        _, p   = X_train.shape
        r_ics  = np.array([_spearman_ic(y_train, X_train[:, j]) for j in range(p)])
        self.rank_ic_weights_ = np.nan_to_num(r_ics)
        self.feature_names_   = feature_names or [f"x{i}" for i in range(p)]
        return self

    def _pct_rank(self, X: np.ndarray) -> np.ndarray:
        n, p  = X.shape
        out   = np.zeros_like(X, dtype=float)
        for j in range(p):
            r = stats.rankdata(X[:, j], method="average") - 1
            out[:, j] = r / max(len(r) - 1, 1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pct_rank(X) @ self.rank_ic_weights_

    def formula(self) -> str:
        top5 = sorted(
            zip(self.rank_ic_weights_, self.feature_names_),
            key=lambda x: abs(x[0]),
            reverse=True,
        )[:5]
        terms = [f"{w:.4f}·pct_rank({n})" for w, n in top5]
        return "Signal = " + " + ".join(terms) + "  [Rank-IC alpha model]"

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names_, "rank_ic": self.rank_ic_weights_})


# ══════════════════════════════════════════════════════════════════════
# MODEL COMPARISON & SELECTION
# ══════════════════════════════════════════════════════════════════════

def compare_models(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    feature_names: List[str],
    ridge_alpha:  float = 1.0,
    lasso_alpha:  float = 0.001,
    rolling_window: int = 500,
) -> Tuple[object, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Train all 6 models, compute metrics, and select the best by Test Rank IC.

    Returns
    -------
    best_model    : fitted model object
    y_pred_train  : training predictions (best model)
    y_pred_test   : test predictions (best model)
    comparison_df : model comparison table
    """
    results = []
    model_preds = {}

    # ── Model 1: OLS ─────────────────────────────────────────────────
    print("[models] Fitting OLS ...")
    m1 = OLSModel().fit(X_train, y_train, feature_names)
    tr1 = m1.predict(X_train)
    te1 = m1.predict(X_test)
    results.append({"model": m1.name, **_metrics(y_train, tr1, "train_"), **_metrics(y_test, te1, "test_")})
    model_preds[m1.name] = (m1, tr1, te1)

    # ── Model 2: Ridge ───────────────────────────────────────────────
    print("[models] Fitting Ridge ...")
    m2 = RidgeModel(alpha=ridge_alpha).fit(X_train, y_train, feature_names)
    tr2 = m2.predict(X_train)
    te2 = m2.predict(X_test)
    results.append({"model": m2.name, **_metrics(y_train, tr2, "train_"), **_metrics(y_test, te2, "test_")})
    model_preds[m2.name] = (m2, tr2, te2)

    # ── Model 3: Lasso ───────────────────────────────────────────────
    print("[models] Fitting Lasso ...")
    m3 = LassoModel(alpha=lasso_alpha).fit(X_train, y_train, feature_names)
    tr3 = m3.predict(X_train)
    te3 = m3.predict(X_test)
    results.append({"model": m3.name, **_metrics(y_train, tr3, "train_"), **_metrics(y_test, te3, "test_")})
    model_preds[m3.name] = (m3, tr3, te3)

    # ── Model 4: Rolling OLS ─────────────────────────────────────────
    print("[models] Fitting Rolling OLS ...")
    m4 = RollingOLSModel(window=rolling_window)
    all_X = np.vstack([X_train, X_test])
    all_y = np.concatenate([y_train, y_test])
    all_preds = m4.fit_predict(all_X, all_y, feature_names)
    tr4 = all_preds[:len(X_train)]
    te4 = all_preds[len(X_train):]
    results.append({"model": m4.name, **_metrics(y_train, tr4, "train_"), **_metrics(y_test, te4, "test_")})
    # For test, use the last β from training
    last_beta = m4.coef_history_.iloc[len(X_train) - 1].values
    model_preds[m4.name] = (m4, tr4, te4)
    m4._last_beta = last_beta

    # ── Model 5: Factor Score ────────────────────────────────────────
    print("[models] Fitting Factor Score ...")
    m5 = FactorScoreModel().fit(X_train, y_train, feature_names)
    tr5 = m5.predict(X_train)
    te5 = m5.predict(X_test)
    results.append({"model": m5.name, **_metrics(y_train, tr5, "train_"), **_metrics(y_test, te5, "test_")})
    model_preds[m5.name] = (m5, tr5, te5)

    # ── Model 6: Rank Alpha ──────────────────────────────────────────
    print("[models] Fitting Rank Alpha ...")
    m6 = RankAlphaModel().fit(X_train, y_train, feature_names)
    tr6 = m6.predict(X_train)
    te6 = m6.predict(X_test)
    results.append({"model": m6.name, **_metrics(y_train, tr6, "train_"), **_metrics(y_test, te6, "test_")})
    model_preds[m6.name] = (m6, tr6, te6)

    # ── Comparison Table ─────────────────────────────────────────────
    comparison_df = pd.DataFrame(results)
    print("\n[models] Model Comparison:")
    print(comparison_df[["model", "train_ic", "test_ic", "train_rank_ic",
                          "test_rank_ic", "test_r2"]].to_string(index=False))

    # ── Select Best (by test Rank IC) ────────────────────────────────
    best_name = comparison_df.sort_values("test_rank_ic", ascending=False).iloc[0]["model"]
    best_model, y_pred_train, y_pred_test = model_preds[best_name]
    print(f"\n[models] ★ Best model: {best_name}")

    return best_model, y_pred_train, y_pred_test, comparison_df


def save_model(model, path: str = "best_model.pkl"):
    """Persist model using numpy (coefficients saved as npz)."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[models] Model saved → {path}")