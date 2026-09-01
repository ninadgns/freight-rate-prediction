"""Cleaning, features and the freight rate model.

Design decisions here are all justified in FINDINGS.md; the section references
below point at the evidence rather than repeating it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gbm import HistGBM

TRAIN_CSV = "train-test.csv"
VAL_CSV = "validation.csv"

# quote_signal is deliberately absent from every feature list. It matches the
# target almost exactly in five months and is worthless in the other five, and
# the validation window sits in the worthless regime (FINDINGS.md section 5).
OUTLIER_Z = 5.0
LANE_PRIOR_SD = 0.0112          # section 7.4
RAMP_KNOTS = (55, 65, 75, 84)   # quarter-end ramp, section 6


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Repairs the defects found in section 3. Runs on train and inference alike."""
    d = df.copy()
    d["weight"] = d["weight"].abs()                      # 3.2, sign-flipped values
    d["weight"] = d["weight"].fillna(d["weight"].median())
    # market_index is effectively a daily series (daily means explain 98% of its
    # variance), so the same-day mean beats a global median. Section 3.3.
    d["market_index"] = d.groupby("date")["market_index"].transform(lambda s: s.fillna(s.mean()))
    d["market_index"] = d["market_index"].fillna(d["market_index"].median())
    d["doy"] = d["date"].dt.dayofyear
    d["doq"] = (d["date"] - d["date"].dt.to_period("Q").dt.start_time).dt.days
    return d


def load_training() -> pd.DataFrame:
    d = clean(pd.read_csv(TRAIN_CSV, parse_dates=["date"]))
    d["y"] = np.log(d["posted_rate"] / d["distance"])
    return d


def drop_corrupted(d: pd.DataFrame) -> pd.DataFrame:
    """Removes the 1.41% of multiplicatively corrupted labels (section 3.1).

    The core is separated from the corrupted lobes by an entirely empty gap, so
    a coarse first pass is enough to find the threshold robustly.
    """
    rpm = np.exp(d["y"])
    coarse = d[(rpm > 1.5) & (rpm < 3.5)]
    ld = np.log(d["distance"])
    X = np.column_stack([ld, ld ** 2, d["weight"] / 1e4,
                         (d["equipment"] == "Flatbed") * 1.0,
                         (d["equipment"] == "Reefer") * 1.0,
                         d["market_index"], np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(X[d.index.isin(coarse.index)], coarse["y"].values, rcond=None)
    resid = d["y"].values - X @ beta
    keep = np.abs(resid) < OUTLIER_Z * resid[d.index.isin(coarse.index)].std()
    return d[keep]


def city_coords(*frames) -> dict:
    """One fixed coordinate pair per city; verified constant in section 3.5."""
    out = {}
    for f in frames:
        for name, la, lo in (("pickup", "pickup_lat", "pickup_lon"),
                             ("delivery", "delivery_lat", "delivery_lon")):
            if name in f.columns and la in f.columns:
                out.update(dict(zip(f[name], zip(f[la], f[lo]))))
    return out


def _hinges(x, knots):
    """Linear spline: linear between knots and linear beyond the outer ones, so
    extrapolation stays tame."""
    return np.column_stack([x] + [np.clip(x - k, 0, None) for k in knots])


class FreightRateModel:
    """Structured model, selected on replica splits in FINDINGS.md section 8.

    Stages, each fitted on the residual of the one before:
      1. parametric core, splines plus a joint linear trend that can extrapolate
      2. symmetric per-city premium, applied at both ends of the lane
      3. per-lane effect
      4. a thin gradient-boosted correction, with day_of_year withheld
    """

    def __init__(self, use_gbm: bool = True):
        self.use_gbm = use_gbm

    # ---- feature blocks -------------------------------------------------
    def _core_design(self, d):
        ld = np.log(d["distance"]).values
        fb = (d["equipment"] == "Flatbed").values * 1.0
        rf = (d["equipment"] == "Reefer").values * 1.0
        ramp = np.column_stack([np.clip(d["doq"].values - k, 0, None) / 10 for k in RAMP_KNOTS])
        return np.column_stack([
            _hinges(ld, self.knots_["ld"]),
            _hinges(d["weight"].values / 1e4, self.knots_["w"]),
            fb, rf,
            d["market_index"].values,
            d["doy"].values / 100,                 # linear, so it extrapolates (section 8.2)
            ramp,
            ramp * fb[:, None], ramp * rf[:, None],   # section 6
            np.ones(len(d)),
        ])

    def _gbm_features(self, d):
        """day_of_year withheld: trees cannot extrapolate it (section 8.2)."""
        return np.column_stack([
            np.log(d["distance"]), d["weight"] / 1e4,
            (d["equipment"] == "Flatbed") * 1.0, (d["equipment"] == "Reefer") * 1.0,
            d["market_index"], d["pickup_lat"], d["pickup_lon"],
            d["delivery_lat"], d["delivery_lon"], d["doq"],
        ])

    # ---- fit ------------------------------------------------------------
    def fit(self, train: pd.DataFrame):
        self.coords_ = city_coords(train)
        self.knots_ = {
            "ld": list(np.quantile(np.log(train["distance"]), [.15, .35, .55, .75, .9])),
            "w": list(np.quantile(train["weight"] / 1e4, [.2, .4, .6, .8])),
        }
        y = train["y"].values
        X = self._core_design(train)
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ self.beta_

        self.city_ = self._fit_city(train, r)
        r = r - self._city_of(train)

        lane = (train["pickup"] + ">" + train["delivery"]).values
        self.lane_ = self._shrink(lane, r, LANE_PRIOR_SD ** 2)
        r = r - pd.Series(lane).map(self.lane_).fillna(0).values

        self.gbm_ = HistGBM().fit(self._gbm_features(train), r) if self.use_gbm else None
        self.resid_sd_ = float(r.std())
        return self

    @staticmethod
    def _shrink(keys, resid, prior_var):
        """Empirical Bayes: small groups are pulled toward zero."""
        g = pd.DataFrame({"k": keys, "r": resid}).groupby("k")["r"].agg(["size", "mean"])
        sigma2 = float(np.var(resid))
        return g["mean"] * (prior_var / (prior_var + sigma2 / g["size"]))

    def _fit_city(self, train, resid):
        """One premium per city, estimated from both ends (section 7.1)."""
        keys = np.concatenate([train["pickup"].values, train["delivery"].values])
        halves = np.concatenate([resid, resid]) / 2
        g = pd.DataFrame({"k": keys, "r": halves}).groupby("k")["r"].agg(["size", "mean"])
        sigma2 = float(np.var(resid))
        prior = max(float(np.var(g["mean"])) - sigma2 / g["size"].mean(), 1e-6)
        return g["mean"] * (prior / (prior + sigma2 / g["size"])) * 2

    def _city_effect(self, cities, lats, lons):
        """Premium per city, with kNN on lat/lon for cities absent from training.

        Coordinates come from the rows being scored rather than a lookup table,
        so an unseen city needs no prior registration. Leave-one-city-out puts
        this at R2 0.889 for a held-out city (section 7.3).
        """
        known = np.array([self.coords_[c] for c in self.city_.index])
        vals = self.city_.values
        cache, out = {}, np.empty(len(cities))
        for i, c in enumerate(cities):
            if c not in cache:
                if c in self.city_.index:
                    cache[c] = self.city_[c]
                else:
                    dist = np.sqrt(((known - np.array([lats[i], lons[i]])) ** 2).sum(1))
                    cache[c] = vals[np.argsort(dist)[:5]].mean()
            out[i] = cache[c]
        return out

    def _city_of(self, d):
        return (self._city_effect(d["pickup"].values,
                                  d["pickup_lat"].values, d["pickup_lon"].values)
                + self._city_effect(d["delivery"].values,
                                    d["delivery_lat"].values, d["delivery_lon"].values))

    # ---- predict --------------------------------------------------------
    def predict_log_rpm(self, d: pd.DataFrame) -> np.ndarray:
        p = self._core_design(d) @ self.beta_ + self._city_of(d)
        lane = (d["pickup"] + ">" + d["delivery"]).values
        p = p + pd.Series(lane).map(self.lane_).fillna(0).values
        if self.gbm_ is not None:
            p = p + self.gbm_.predict(self._gbm_features(d))
        return p

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        """Dollar rate. Residual sd is ~1.6%, so the lognormal mean/median gap
        is under 0.02% and not worth a smearing correction."""
        return np.exp(self.predict_log_rpm(d)) * d["distance"].values
