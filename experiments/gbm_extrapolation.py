"""Why a plain gradient-boosted tree underprices December.

Trees are piecewise constant. A split can only be placed at a value seen in
training, so for any feature whose test range lies *beyond* its training range,
every test row falls into the same terminal bin as the last training rows and
receives their prediction. The fitted trend flattens at the boundary.

`day_of_year` is exactly such a feature here: training covers 1-304, the holdout
is 305-365. `day_of_quarter` is not, because Q1-Q3 cover its full 0-91 range.

This script demonstrates the effect on the Q3 replica split, which has the same
shape as the real task (train to day 30 of a quarter, predict days 31-91), and
compares three models. Run: python experiments/gbm_extrapolation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gbm import HistGBM  # noqa: E402

OUT = Path("eda_out")


# --------------------------------------------------------------------------
def prep():
    t = pd.read_csv("train-test.csv", parse_dates=["date"])
    t["weight"] = t["weight"].abs()
    t["weight"] = t["weight"].fillna(t["weight"].median())
    t["market_index"] = t.groupby("date")["market_index"].transform(
        lambda s: s.fillna(s.mean()))
    t["y"] = np.log(t["posted_rate"] / t["distance"])
    t["doy"] = t["date"].dt.dayofyear
    t["doq"] = (t["date"] - t["date"].dt.to_period("Q").dt.start_time).dt.days
    # drop the 1.4% corrupted labels
    return t[(t["y"] > np.log(1.5)) & (t["y"] < np.log(3.5))].reset_index(drop=True)


def features(d, with_doy=True):
    ld = np.log(d["distance"])
    cols = [ld, d["weight"] / 1e4,
            (d["equipment"] == "Flatbed") * 1.0, (d["equipment"] == "Reefer") * 1.0,
            d["market_index"], d["pickup_lat"], d["pickup_lon"],
            d["delivery_lat"], d["delivery_lon"], d["doq"]]
    if with_doy:
        cols.append(d["doy"])
    return np.column_stack(cols)


def parametric(train, test):
    """Linear in logs, with the trend and the quarter-end/equipment interaction explicit."""
    def design(d):
        ld = np.log(d["distance"])
        fb = (d["equipment"] == "Flatbed") * 1.0
        rf = (d["equipment"] == "Reefer") * 1.0
        ramp = np.clip(d["doq"] - 59, 0, None) / 10
        return np.column_stack([ld, ld ** 2, d["weight"] / 1e4, fb, rf, d["market_index"],
                                d["pickup_lat"] / 10, d["delivery_lat"] / 10,
                                d["doy"] / 100, ramp, fb * ramp, rf * ramp, np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(design(train), train["y"].values, rcond=None)
    return design(test) @ beta


def main():
    d = prep()
    # Q3 replica: train through day 30 of Q3, predict days 31-91. Same shape as the real task.
    train = d[d["date"] < "2025-08-01"]
    test = d[(d["date"] >= "2025-08-01") & (d["date"] < "2025-10-01")]
    print(f"train {len(train):,} rows (doy {train.doy.min()}-{train.doy.max()})")
    print(f"test  {len(test):,} rows (doy {test.doy.min()}-{test.doy.max()})\n")

    preds = {}
    print("fitting GBM with day_of_year ...")
    g1 = HistGBM().fit(features(train, True), train["y"].values)
    preds["GBM (with day_of_year)"] = g1.predict(features(test, True))

    print("fitting GBM without day_of_year ...")
    g2 = HistGBM().fit(features(train, False), train["y"].values)
    preds["GBM (no day_of_year)"] = g2.predict(features(test, False))

    print("fitting parametric ...\n")
    preds["parametric (linear trend)"] = parametric(train, test)

    y = test["y"].values
    print(f"{'model':<28} {'MAE':>8} {'RMSE':>8} {'bias':>9} {'bias, last 2wk':>15}")
    last = test["date"] >= "2025-09-17"
    rows = {}
    for k, p in preds.items():
        err = y - p
        rows[k] = err
        print(f"{k:<28} {np.abs(err).mean():8.4f} {np.sqrt((err**2).mean()):8.4f} "
              f"{err.mean():+9.4f} {err[last.values].mean():+15.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    colors = {"GBM (with day_of_year)": "#C2453B", "GBM (no day_of_year)": "#C58A22",
              "parametric (linear trend)": "#064A56"}
    for k, err in rows.items():
        s = pd.Series(err, index=test["date"].values).groupby(level=0).mean()
        ax[0].plot(s.index, s.rolling(7, center=True).mean(), color=colors[k], lw=2, label=k)
    ax[0].axhline(0, color="#9DAFB3", lw=1)
    ax[0].set(title="prediction bias grows into the holdout\n(actual minus predicted, log scale)",
              ylabel="bias")
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].legend(fontsize=8)

    # What the models think the trend is, holding everything else fixed.
    grid = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-09-30")})
    grid["doy"] = grid["date"].dt.dayofyear
    grid["doq"] = (grid["date"] - grid["date"].dt.to_period("Q").dt.start_time).dt.days
    for c, v in (("distance", 360.0), ("weight", 32000.0), ("equipment", "Dry Van"),
                 ("market_index", 1.0), ("pickup_lat", 36.99), ("pickup_lon", -85.0),
                 ("delivery_lat", 41.32), ("delivery_lon", -85.36)):
        grid[c] = v
    ax[1].plot(grid["date"], g1.predict(features(grid, True)),
               color="#C2453B", lw=2, label="GBM (with day_of_year)")
    ax[1].plot(grid["date"], parametric(train, grid), color="#064A56", lw=2,
               label="parametric (linear trend)")
    ax[1].axvline(pd.Timestamp("2025-08-01"), color="#9DAFB3", ls="--")
    ax[1].text(pd.Timestamp("2025-08-03"), ax[1].get_ylim()[0], " holdout starts", fontsize=8,
               color="#455A60", va="bottom")
    ax[1].set(title="fitted curve on one fixed lane\n(only the date varies)", ylabel="log $/mi")
    ax[1].tick_params(axis="x", rotation=30)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "06_gbm_extrapolation.png", dpi=140)
    print(f"\nfigure written to {OUT}/06_gbm_extrapolation.png")


if __name__ == "__main__":
    main()
