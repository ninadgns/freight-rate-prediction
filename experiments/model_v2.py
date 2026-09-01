"""Can we beat the plain hybrid? Adds the structure the EDA already established.

Three additions over `hybrid` in replica_splits.py:

1. Linear splines on log(distance), weight and the quarter-end ramp, since all
   three are known to be nonlinear. Linear beyond the outer knots, so the
   extrapolation behaviour that matters for day_of_year is preserved.
2. Explicit shrunk city effects. One symmetric premium per city applied at both
   ends (section 7.1), with kNN-on-lat/lon fallback for cities absent from
   training. The GBM otherwise has to rediscover this through coordinate splits.
3. Explicit shrunk lane effects, empirical-Bayes weighted by lane volume.

Run: python experiments/model_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbm_extrapolation import HistGBM, features, parametric, prep
from replica_splits import SPLITS


def hinges(x, knots):
    """Linear spline basis: linear inside, and linear (not wild) beyond the ends."""
    return np.column_stack([x] + [np.clip(x - k, 0, None) for k in knots])


def design(d, knots):
    ld = np.log(d["distance"]).values
    w = d["weight"].values / 1e4
    fb = (d["equipment"] == "Flatbed").values * 1.0
    rf = (d["equipment"] == "Reefer").values * 1.0
    ramp = hinges(d["doq"].values.astype(float), [60, 70, 80])
    ramp = np.clip(ramp - ramp[:, :1] * 0, 0, None)          # keep the raw hinges
    ramp = np.column_stack([np.clip(d["doq"].values - k, 0, None) / 10 for k in (55, 65, 75, 84)])
    return np.column_stack([
        hinges(ld, knots["ld"]),
        hinges(w, knots["w"]),
        fb, rf,
        d["market_index"].values,
        d["doy"].values / 100,                                # linear: must extrapolate
        ramp,
        ramp * fb[:, None], ramp * rf[:, None],
        np.ones(len(d)),
    ])


def fit_knots(train):
    return {"ld": list(np.quantile(np.log(train["distance"]), [.15, .35, .55, .75, .9])),
            "w": list(np.quantile(train["weight"] / 1e4, [.2, .4, .6, .8]))}


def shrunk_means(keys, resid, prior_var, sigma2):
    """Empirical-Bayes: pull small-sample group means toward zero."""
    df = pd.DataFrame({"k": keys, "r": resid})
    g = df.groupby("k")["r"].agg(["size", "mean"])
    w = prior_var / (prior_var + sigma2 / g["size"])
    return (g["mean"] * w)


def city_effects(train, resid):
    """One symmetric premium per city, estimated from both ends of every lane."""
    keys = np.concatenate([train["pickup"].values, train["delivery"].values])
    r = np.concatenate([resid, resid]) / 2          # each load contributes half to each end
    raw = pd.DataFrame({"k": keys, "r": r}).groupby("k")["r"].agg(["size", "mean"])
    sigma2 = float(np.var(resid))
    prior = max(float(np.var(raw["mean"])) - sigma2 / raw["size"].mean(), 1e-6)
    w = prior / (prior + sigma2 / raw["size"])
    return raw["mean"] * w * 2                       # undo the halving


def knn_fill(effects, coords, want, k=5):
    """Recover an unseen city's premium from its nearest known neighbours."""
    known = np.array([coords[c] for c in effects.index])
    vals = effects.values
    out = {}
    for city in want:
        if city in effects.index:
            out[city] = effects[city]
            continue
        d = np.sqrt(((known - np.array(coords[city])) ** 2).sum(1))
        out[city] = vals[np.argsort(d)[:k]].mean()
    return pd.Series(out)


def structured(train, test):
    knots = fit_knots(train)
    Xtr, Xte = design(train, knots), design(test, knots)
    beta, *_ = np.linalg.lstsq(Xtr, train["y"].values, rcond=None)
    r_tr = train["y"].values - Xtr @ beta
    pred = Xte @ beta

    coords = {}
    for d in (train, test):
        for a, la, lo in (("pickup", "pickup_lat", "pickup_lon"),
                          ("delivery", "delivery_lat", "delivery_lon")):
            coords.update(dict(zip(d[a], zip(d[la], d[lo]))))

    ce = city_effects(train, r_tr)
    want = set(test["pickup"]) | set(test["delivery"]) | set(ce.index)
    ce = knn_fill(ce, coords, want)
    r_tr = r_tr - train["pickup"].map(ce).values - train["delivery"].map(ce).values
    pred = pred + test["pickup"].map(ce).values + test["delivery"].map(ce).values

    lane_tr = (train["pickup"] + ">" + train["delivery"]).values
    lane_te = (test["pickup"] + ">" + test["delivery"]).values
    sigma2 = float(np.var(r_tr))
    le = shrunk_means(lane_tr, r_tr, max(0.0112 ** 2, 1e-6), sigma2)
    r_tr = r_tr - pd.Series(lane_tr).map(le).fillna(0).values
    pred = pred + pd.Series(lane_te).map(le).fillna(0).values

    g = HistGBM().fit(features(train, False), r_tr)
    return pred + g.predict(features(test, False))


def hybrid(train, test):
    p_tr = parametric(train, train)
    g = HistGBM().fit(features(train, False), train["y"].values - p_tr)
    return parametric(train, test) + g.predict(features(test, False))


def main():
    d = prep()
    print(f"{'split':<12} {'model':<26} {'MAE':>8} {'RMSE':>8} {'bias wk1':>9} {'bias wk8':>9}")
    print("-" * 78)
    for name, (start, end) in SPLITS.items():
        train = d[d["date"] < start]
        test = d[(d["date"] >= start) & (d["date"] < end)]
        wk1 = (test["date"] < pd.Timestamp(start) + pd.Timedelta(days=7)).values
        wk8 = (test["date"] >= pd.Timestamp(end) - pd.Timedelta(days=14)).values
        y = test["y"].values
        for k, p in (("hybrid (baseline)", hybrid(train, test)),
                     ("structured hybrid", structured(train, test))):
            e = y - p
            print(f"{name:<12} {k:<26} {np.abs(e).mean():8.4f} "
                  f"{np.sqrt((e**2).mean()):8.4f} {e[wk1].mean():+9.4f} {e[wk8].mean():+9.4f}")
        print("-" * 78)


if __name__ == "__main__":
    main()
