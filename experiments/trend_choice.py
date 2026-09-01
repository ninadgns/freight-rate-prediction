"""How should the trend be extrapolated into the holdout?

The baseline level is not linear over the year: it climbs Jan-Jul, plateaus
Jul-Sep, then steps up in October. So a straight line fitted on all history and
a straight line fitted on recent history disagree by ~2% over a two-month
horizon, which is the single largest open choice in the model.

Estimated in two stages so the window is easy to vary: fit the structured model
with no day_of_year term, then fit a trend to the daily means of its residual
over the last W days and extrapolate that.

Run: python experiments/trend_choice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbm_extrapolation import HistGBM, features, prep
from model_v2 import city_effects, design, fit_knots, knn_fill, shrunk_means
from replica_splits import SPLITS


def design_nodoy(d, knots):
    A = design(d, knots)
    for j in range(A.shape[1]):
        if np.allclose(A[:, j], d["doy"].values / 100):
            return np.delete(A, j, axis=1)
    raise RuntimeError("day_of_year column not found")


def fit(train, test, window, damp=1.0):
    knots = fit_knots(train)
    Xtr, Xte = design_nodoy(train, knots), design_nodoy(test, knots)
    beta, *_ = np.linalg.lstsq(Xtr, train["y"].values, rcond=None)
    r_tr = train["y"].values - Xtr @ beta
    pred = Xte @ beta

    coords = {}
    for d in (train, test):
        for a, la, lo in (("pickup", "pickup_lat", "pickup_lon"),
                          ("delivery", "delivery_lat", "delivery_lon")):
            coords.update(dict(zip(d[a], zip(d[la], d[lo]))))
    ce = city_effects(train, r_tr)
    ce = knn_fill(ce, coords, set(test["pickup"]) | set(test["delivery"]) | set(ce.index))
    r_tr = r_tr - train["pickup"].map(ce).values - train["delivery"].map(ce).values
    pred = pred + test["pickup"].map(ce).values + test["delivery"].map(ce).values

    lt = (train["pickup"] + ">" + train["delivery"]).values
    le = shrunk_means(lt, r_tr, 0.0112 ** 2, float(np.var(r_tr)))
    r_tr = r_tr - pd.Series(lt).map(le).fillna(0).values
    pred = pred + pd.Series((test["pickup"] + ">" + test["delivery"]).values).map(le).fillna(0).values

    # --- the trend, fitted on daily means of what is left ---
    day = pd.Series(r_tr, index=train["doy"].values).groupby(level=0).mean()
    w = day.index >= day.index.max() - window if window else np.ones(len(day), bool)
    A = np.column_stack([day.index[w], np.ones(w.sum())])
    slope, intercept = np.linalg.lstsq(A, day.values[w], rcond=None)[0]
    anchor = day.index.max()
    level_at_anchor = slope * anchor + intercept
    r_tr = r_tr - (slope * train["doy"].values + intercept)
    pred = pred + level_at_anchor + damp * slope * (test["doy"].values - anchor)

    g = HistGBM().fit(features(train, False), r_tr)
    return pred + g.predict(features(test, False))


def main():
    d = prep()
    rows = []
    for name, (start, end) in SPLITS.items():
        train = d[d["date"] < start]
        test = d[(d["date"] >= start) & (d["date"] < end)]
        wk8 = (test["date"] >= pd.Timestamp(end) - pd.Timedelta(days=14)).values
        y = test["y"].values
        for window in (60, 90, 120, 180, None):
            for damp in (1.0, 0.5):
                if damp != 1.0 and window not in (60, 90):
                    continue
                e = y - fit(train, test, window, damp)
                rows.append({"split": name, "window": window or "all", "damp": damp,
                             "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                             "bias_wk8": e[wk8].mean()})
    r = pd.DataFrame(rows)
    print(r.pivot_table(index=["window", "damp"], columns="split",
                        values=["MAE", "bias_wk8"]).round(4).to_string())
    print()
    agg = r.groupby(["window", "damp"]).agg(MAE=("MAE", "mean"),
                                            worst_bias=("bias_wk8", lambda s: s.abs().max()))
    print("averaged over both replicas, ranked by worst-case bias:")
    print(agg.sort_values("worst_bias").round(4).to_string())


if __name__ == "__main__":
    main()
