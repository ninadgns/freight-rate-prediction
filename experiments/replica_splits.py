"""How much does tree flattening actually cost, across both replica splits?

The real task trains through day 30 of a quarter and predicts days 31-91 of that
same quarter, two months forward. That shape occurs twice inside the training
data, so it can be measured rather than argued about.

Run: python experiments/replica_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbm_extrapolation import HistGBM, features, parametric, prep

SPLITS = {
    "Q2 replica": ("2025-05-01", "2025-07-01"),   # train Jan-Apr, test May-Jun
    "Q3 replica": ("2025-08-01", "2025-10-01"),   # train Jan-Jul, test Aug-Sep
}


def hybrid(train, test):
    """GBM on the residual of the parametric trend, so the trend can extrapolate
    while the trees handle everything that does not need to."""
    p_tr = parametric(train, train)
    g = HistGBM().fit(features(train, False), train["y"].values - p_tr)
    return parametric(train, test) + g.predict(features(test, False))


def main():
    d = prep()
    print(f"{'split':<12} {'model':<28} {'MAE':>8} {'RMSE':>8} {'bias':>9} "
          f"{'bias wk1':>9} {'bias wk8':>9}")
    print("-" * 88)
    for name, (start, end) in SPLITS.items():
        train = d[d["date"] < start]
        test = d[(d["date"] >= start) & (d["date"] < end)]
        wk1 = test["date"] < pd.Timestamp(start) + pd.Timedelta(days=7)
        wk8 = test["date"] >= pd.Timestamp(end) - pd.Timedelta(days=14)

        models = {
            "GBM (with day_of_year)":
                lambda: HistGBM().fit(features(train, True), train["y"].values)
                                 .predict(features(test, True)),
            "GBM (no day_of_year)":
                lambda: HistGBM().fit(features(train, False), train["y"].values)
                                 .predict(features(test, False)),
            "parametric (linear trend)": lambda: parametric(train, test),
            "hybrid (parametric + GBM)": lambda: hybrid(train, test),
        }
        y = test["y"].values
        for k, fn in models.items():
            err = y - fn()
            print(f"{name:<12} {k:<28} {np.abs(err).mean():8.4f} "
                  f"{np.sqrt((err**2).mean()):8.4f} {err.mean():+9.4f} "
                  f"{err[wk1.values].mean():+9.4f} {err[wk8.values].mean():+9.4f}")
        print("-" * 88)


if __name__ == "__main__":
    main()
