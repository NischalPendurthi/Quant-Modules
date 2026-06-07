from pathlib import Path
import sys
import json
import pickle
import numpy as np
import pandas as pd

from data import load_data, create_target_variable
from features import engineer_features
from backtest import (
    LongOnlyBacktest,
    LongShortBacktest,
    validate_output,
)

TEAM_NAME = "ragasofrevenge"

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


def load_saved_model():
    with open("artifacts/best_model.pkl", "rb") as f:
        return pickle.load(f)


def load_sign():
    with open("artifacts/signal_sign.json") as f:
        return json.load(f)["signal_sign"]


def main(csv_path):

    print("=" * 80)
    print("JUDGING MODE")
    print("=" * 80)

    df = load_data(csv_path)

    df = create_target_variable(
        df,
        price_col="Close",
        # method="return",
        horizon=1,
    )

    df = engineer_features(
        df,
        windows=(6, 12, 24),
        lags=(1, 3, 6, 12, 24),
        add_regime=True,
        add_cross_sectional=True,
    )

    model = load_saved_model()
    signal_sign = load_sign()

    X = df[FEATURES].fillna(0)

    signal = model.predict(X.values)
    signal = signal * signal_sign

    prices = df["Close"].values
    timestamps = df.index

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

    lo = bt_lo.backtest(
        timestamps=timestamps,
        prices=prices,
        signal=signal,
        force_sign=1,
    )

    ls = bt_ls.backtest(
        timestamps=timestamps,
        prices=prices,
        signal=signal,
        force_sign=1,
    )

    Path("submissions").mkdir(exist_ok=True)

    lo_file = f"submissions/{TEAM_NAME}_longonly_results.csv"
    ls_file = f"submissions/{TEAM_NAME}_longshort_results.csv"

    lo.to_csv(lo_file, index=False)
    ls.to_csv(ls_file, index=False)

    validate_output(lo, "LONG_ONLY", 1_000_000)
    validate_output(ls, "LONG_SHORT", 2_000_000)

    print()
    print("Generated:")
    print(lo_file)
    print(ls_file)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python judging.py judging_dataset.csv"
        )
        sys.exit(1)

    main(sys.argv[1])