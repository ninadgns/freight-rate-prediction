"""Choose the trend handling on more than two folds.

The two quarter-aligned replicas are the most faithful tests, but two folds is
far too thin to pick a hyper-parameter on. This walks the cut date forward
monthly, always predicting the following two months, which is the real task's
horizon. The GBM residual stage is omitted here because it is fitted with
day_of_year withheld and so contributes almost nothing to trend bias, which is
what this is selecting on.

Run: python experiments/rolling_origin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbm_extrapolation import prep
from model_v2 import city_effects, design, fit_knots, knn_fill, shrunk_means
from trend_choice import design_nodoy

CUTS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]


def core(train, test, use_doy, window=None, damp=1.0):
    """Everything except the GBM stage. use_doy=True fits the trend jointly."""
    knots = fit_knots(train)
    build = design if use_doy else design_nodoy
    Xtr, Xte = build(train, knots), build(test, knots)
    beta, *_ = np.linalg.lstsq(Xtr, train["y"].values, rcond=None)
    r = train["y"].values - Xtr @ beta
    pred = Xte @ beta

    coords = {}
    for d in (train, test):
        for a, la, lo in (("pickup", "pickup_lat", "pickup_lon"),
                          ("delivery", "delivery_lat", "delivery_lon")):
            coords.update(dict(zip(d[a], zip(d[la], d[lo]))))
    ce = city_effects(train, r)
    ce = knn_fill(ce, coords, set(test["pickup"]) | set(test["delivery"]) | set(ce.index))
    r = r - train["pickup"].map(ce).values - train["delivery"].map(ce).values
    pred = pred + test["pickup"].map(ce).values + test["delivery"].map(ce).values

    lt = (train["pickup"] + ">" + train["delivery"]).values
    le = shrunk_means(lt, r, 0.0112 ** 2, float(np.var(r)))
    r = r - pd.Series(lt).map(le).fillna(0).values
    pred = pred + pd.Series((test["pickup"] + ">" + test["delivery"]).values).map(le).fillna(0).values

    if not use_doy:
        day = pd.Series(r, index=train["doy"].values).groupby(level=0).mean()
        w = (day.index >= day.index.max() - window) if window else np.ones(len(day), bool)
        A = np.column_stack([day.index[w], np.ones(w.sum())])
        slope, icpt = np.linalg.lstsq(A, day.values[w], rcond=None)[0]
        anchor = day.index.max()
        pred = pred + (slope * anchor + icpt) + damp * slope * (test["doy"].values - anchor)
    return pred


def main():
    d = prep()
    variants = [("joint linear doy", dict(use_doy=True))]
    variants += [(f"window {w}d, damp {dm}", dict(use_doy=False, window=w, damp=dm))
                 for w in (90, 120, 180, None) for dm in (1.0, 0.5)]

    rows = []
    for cut in CUTS:
        cut_ts = pd.Timestamp(cut)
        train = d[d["date"] < cut_ts]
        test = d[(d["date"] >= cut_ts) & (d["date"] < cut_ts + pd.DateOffset(months=2))]
        if len(test) == 0:
            continue
        late = (test["date"] >= test["date"].max() - pd.Timedelta(days=14)).values
        y = test["y"].values
        for name, kw in variants:
            e = y - core(train, test, **kw)
            rows.append({"cut": cut[:7], "variant": name, "MAE": np.abs(e).mean(),
                         "bias_late": e[late].mean()})
    r = pd.DataFrame(rows)
    print("bias in the final 2 weeks of each holdout (the December-equivalent position):")
    print(r.pivot(index="variant", columns="cut", values="bias_late").round(4).to_string())
    print()
    agg = r.groupby("variant").agg(mean_MAE=("MAE", "mean"),
                                   mean_abs_bias=("bias_late", lambda s: s.abs().mean()),
                                   worst_bias=("bias_late", lambda s: s.abs().max()))
    print(agg.sort_values("mean_abs_bias").round(4).to_string())


if __name__ == "__main__":
    main()
