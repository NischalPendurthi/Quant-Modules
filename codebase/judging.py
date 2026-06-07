from pathlib import Path
import sys
import json
import pickle
import time

import numpy as np
import pandas as pd

from data import load_data, create_target_variable
from features import engineer_features
from backtest import (
    LongOnlyBacktest,
    LongShortBacktest,
    validate_output,
)

# =============================================================================
# CONFIG
# =============================================================================

TEAM_NAME = "ragasofrevenge"

# True on Sunday
FINAL_SUBMISSION = True

# Used only when testing on 10-year training file
TEST_ROWS = 94500

FEATURES = [
    "f1_mom1",
    "f1_rzsc_3",
    "f1_rzsc_6",
    "f1_rzsc_12",
    "f1_mom3",
    "f1_rzsc_24",
    "f1_mom6",
    "f1_mom12",
    "f1_mom24",
]

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_saved_model():
    with open("artifacts/best_model.pkl", "rb") as f:
        return pickle.load(f)


def load_sign():
    with open("artifacts/signal_sign.json") as f:
        return json.load(f)["signal_sign"]


# =============================================================================
# MAIN
# =============================================================================

def engineer_submission_features(df):
    f1 = df["f1"]

    # Momentum
    df["f1_mom1"] = f1 - f1.shift(1)
    df["f1_mom3"] = f1 - f1.shift(3)
    df["f1_mom6"] = f1 - f1.shift(6)
    df["f1_mom12"] = f1 - f1.shift(12)
    df["f1_mom24"] = f1 - f1.shift(24)

    # Rolling z-scores
    for w in [3, 6, 12, 24]:
        mu = f1.rolling(w).mean()
        sd = f1.rolling(w).std()

        df[f"f1_rzsc_{w}"] = (
            (f1 - mu) / (sd + 1e-12)
        )

    return df

def main(csv_path):

    t0 = time.time()

    print("=" * 80)
    print("JUDGING MODE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    df = load_data(csv_path)

    # -------------------------------------------------------------------------
    # TRAINING-ONLY COMPONENTS
    # -------------------------------------------------------------------------
    #
    # Needed when testing locally against the 10-year dataset because
    # engineer_features() was originally designed around the training flow.
    #
    # Skip entirely on the final judging file.
    #
    # -------------------------------------------------------------------------

    if not FINAL_SUBMISSION:

        print("[LOCAL TEST MODE] Creating target variable")

        df = create_target_variable(
            df,
            price_col="Close",
            horizon=1,
        )

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------------

    print("\n[FEATURE ENGINEERING]")

    if FINAL_SUBMISSION:
        df = engineer_submission_features(df)
    else:
        df = engineer_features(
            df,
            windows=(6,12,24),
            lags=(1,3,6,12,24),
            add_regime=True,
            add_cross_sectional=True,
        )

    print(f"Feature engineering complete: {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    print("\n[MODEL] Loading saved model")

    model = load_saved_model()
    signal_sign = load_sign()

    print(f"Signal sign: {signal_sign}")

    # -------------------------------------------------------------------------
    # FEATURE MATRIX
    # -------------------------------------------------------------------------

    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        raise ValueError(
            f"Missing engineered features:\n{missing}"
        )

    X = df[FEATURES].fillna(0)

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    print("\n[PREDICTION]")

    signal = model.predict(X.values)

    signal = signal * signal_sign

    prices = df["Close"].values
    timestamps = df.index

    print(f"Rows available: {len(prices):,}")

    # -------------------------------------------------------------------------
    # LOCAL TEST MODE
    # -------------------------------------------------------------------------

    if not FINAL_SUBMISSION:

        print(
            f"[LOCAL TEST MODE] Using last {TEST_ROWS:,} rows "
            "to mimic judging dataset"
        )

        prices = prices[-TEST_ROWS:]
        timestamps = timestamps[-TEST_ROWS:]
        signal = signal[-TEST_ROWS:]

    print(f"Rows used: {len(prices):,}")

    # -------------------------------------------------------------------------
    # BACKTESTERS
    # -------------------------------------------------------------------------

    bt_lo = LongOnlyBacktest(
        initial_capital=1_000_000,
        entry_percentile=0.95,
        exit_percentile=0.60,
    )

    bt_ls = LongShortBacktest(
        initial_capital=2_000_000,
        long_percentile=0.95,
        short_percentile=0.05,
        long_position_fraction=0.12,
        short_position_fraction=0.04,
    )

    # -------------------------------------------------------------------------
    # LONG ONLY
    # -------------------------------------------------------------------------

    print("\n[LONG ONLY]")

    lo = bt_lo.backtest(
        timestamps=timestamps,
        prices=prices,
        signal=signal,
        force_sign=1,
    )

    # -------------------------------------------------------------------------
    # LONG SHORT
    # -------------------------------------------------------------------------

    print("\n[LONG SHORT]")

    ls = bt_ls.backtest(
        timestamps=timestamps,
        prices=prices,
        signal=signal,
        force_sign=1,
    )

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    Path("submissions").mkdir(exist_ok=True)

    lo_file = f"submissions/{TEAM_NAME}_longonly_results.csv"
    ls_file = f"submissions/{TEAM_NAME}_longshort_results.csv"

    lo.to_csv(lo_file, index=False)
    ls.to_csv(ls_file, index=False)

    print("\n[FILES WRITTEN]")
    print(lo_file)
    print(ls_file)

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    if not FINAL_SUBMISSION:

        print("\n[LOCAL VALIDATION]")

        try:
            validate_output(
                lo,
                "LONG_ONLY",
                1_000_000,
            )

            validate_output(
                ls,
                "LONG_SHORT",
                2_000_000,
            )

            print("Local validation passed")

        except Exception as e:

            print(f"Validation warning: {e}")

    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    print(
        f"Runtime: "
        f"{time.time() - t0:.2f} seconds"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python judging.py judging_dataset.csv"
        )

        sys.exit(1)

    main(sys.argv[1])