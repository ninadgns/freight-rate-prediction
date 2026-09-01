"""Exploratory data analysis for the Spotter freight-rate challenge.

Run:  python eda.py
Writes figures to eda_out/ and prints the findings table to stdout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("eda_out")
GOOD_MONTHS = (1, 2, 3, 6, 9)          # months where quote_signal tracks the target
BAD_MONTHS = (4, 5, 7, 8, 10)


def load():
    train = pd.read_csv("train-test.csv", parse_dates=["date"])
    val = pd.read_csv("validation.csv", parse_dates=["date"])
    return train, val


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Repairs the two mechanical defects: sign-flipped weights and missing values."""
    out = df.copy()
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(out["weight"].median())
    out["market_index"] = out["market_index"].fillna(out["market_index"].median())
    out["doy"] = out["date"].dt.dayofyear
    out["doq"] = (out["date"] - out["date"].dt.to_period("Q").dt.start_time).dt.days
    out["dow"] = out["date"].dt.dayofweek
    return out


def design(df: pd.DataFrame) -> np.ndarray:
    """Transparent structural basis used only to expose residual signal, not to predict."""
    ld = np.log(df["distance"])
    return np.column_stack([
        ld, ld ** 2,
        df["weight"] / 10_000,
        (df["equipment"] == "Flatbed").astype(float),
        (df["equipment"] == "Reefer").astype(float),
        df["market_index"],
        df["doy"] / 100,
        np.clip(df["doq"] - 60, 0, None) / 10,     # quarter-end ramp
        np.ones(len(df)),
    ])


def fit_structural(train: pd.DataFrame):
    """Fits on the uncorrupted core so the 1.4% of bad labels do not bend the fit."""
    y = np.log(train["posted_rate"] / train["distance"])
    core = train[(np.exp(y) > 1.5) & (np.exp(y) < 3.5)]
    beta, *_ = np.linalg.lstsq(design(core), np.log(core["posted_rate"] / core["distance"]), rcond=None)
    resid = y - design(train) @ beta
    sigma = resid.loc[core.index].std()
    return beta, resid, sigma


def fig_target(train, resid, sigma):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rpm = train["posted_rate"] / train["distance"]
    ax[0].hist(train["posted_rate"], bins=120, color="#064A56")
    ax[0].set(title="posted_rate", xlabel="$", ylabel="loads")
    ax[1].hist(rpm, bins=np.linspace(0, 8, 160), color="#064A56")
    ax[1].set(title="rate per mile: tight core, two outlier lobes", xlabel="$/mi")
    z = resid / sigma
    # Clipped so the two corrupted lobes, which run past |z|=50, land in the edge bins.
    ax[2].hist(z.clip(-40, 40), bins=np.linspace(-40, 40, 161), color="#064A56")
    for c in (-5, 5):
        ax[2].axvline(c, color="#C2453B", ls="--", lw=1.2)
    ax[2].set(title=f"structural residual: {(z.abs() > 5).sum()} corrupted labels\nsit past a clean empty gap at |z|=5",
              xlabel="z (clipped to +/-40)", yscale="log")
    fig.tight_layout()
    fig.savefig(OUT / "01_target_and_outliers.png", dpi=140)
    plt.close(fig)


def fig_quote_signal(train, val):
    rpm = train["posted_rate"] / train["distance"]
    lr = np.log(train["quote_signal"] / rpm)
    month = train["date"].dt.month

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    frac = (lr.abs() < 0.02).groupby(train["date"].dt.to_period("M")).mean()
    colors = ["#1B7A4B" if p.month in GOOD_MONTHS else "#C2453B" for p in frac.index]
    ax[0].bar([str(p)[-2:] for p in frac.index], frac.values, color=colors)
    ax[0].set(title="share of loads where quote_signal is within 2% of $/mi",
              xlabel="month of 2025", ylim=(0, 1))

    for m, c, lab in ((1, "#1B7A4B", "Jan (reliable)"), (10, "#C2453B", "Oct (degraded)")):
        ax[1].hist(lr[month == m], bins=np.linspace(-1, 0.6, 120), alpha=0.65, color=c, label=lab)
    ax[1].set(title="log(quote_signal / actual $/mi)", xlabel="log ratio")
    ax[1].legend()

    # The marginal shape is the only tell available on unlabelled data.
    for m, c, lab in ((1, "#1B7A4B", "Jan (reliable)"), (8, "#C2453B", "Aug (degraded)")):
        ax[2].hist(train.loc[month == m, "quote_signal"], bins=np.linspace(0.6, 3.7, 90),
                   histtype="step", lw=2, color=c, label=lab, density=True)
    ax[2].hist(val["quote_signal"], bins=np.linspace(0.6, 3.7, 90), histtype="step", lw=2.4,
               color="#1B1B1B", ls="--", label="Nov-Dec (validation)", density=True)
    ax[2].set(title="validation quote_signal matches the degraded regime", xlabel="quote_signal")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "02_quote_signal_regime.png", dpi=140)
    plt.close(fig)


def fig_structure(train, resid):
    core = train[resid.abs() < 5 * resid.std()]
    rpm = core["posted_rate"] / core["distance"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for eq, c in (("Dry Van", "#064A56"), ("Flatbed", "#C58A22"), ("Reefer", "#1B7A4B")):
        m = core["equipment"] == eq
        b = pd.cut(core.loc[m, "distance"], np.geomspace(70, 3440, 22))
        ax[0].plot([i.mid for i in b.cat.categories], rpm[m].groupby(b, observed=False).median(),
                   marker="o", ms=3, color=c, label=eq)
    ax[0].set(title="$/mi falls with length of haul", xlabel="miles", ylabel="$/mi", xscale="log")
    ax[0].legend()

    # Base model deliberately excludes market_index and the calendar terms, so the
    # next two panels show those effects rather than what is left after removing them.
    ld = np.log(core["distance"])
    base = np.column_stack([ld, ld ** 2, core["weight"] / 10_000,
                            (core["equipment"] == "Flatbed").astype(float),
                            (core["equipment"] == "Reefer").astype(float), np.ones(len(core))])
    y = np.log(rpm)
    bb, *_ = np.linalg.lstsq(base, y, rcond=None)
    raw = pd.Series(y - base @ bb, index=core.index)

    q = pd.qcut(core["market_index"], 12)
    mid = [i.mid for i in q.cat.categories]
    ax[1].plot(mid, raw.groupby(q, observed=False).mean(), marker="o", color="#064A56",
               label="observed")
    slope = np.polyfit(core["market_index"], raw, 1)
    ax[1].plot(mid, np.polyval(slope, mid), color="#C2453B", ls="--",
               label=f"slope {slope[0]:.3f}/unit")
    ax[1].set(title="market_index moves rate", xlabel="market_index", ylabel="log residual")
    ax[1].legend()

    # Net of market_index, what is left is pure calendar: trend plus quarter-end ramp.
    net = pd.Series(raw - np.polyval(slope, core["market_index"]), index=core.index)
    for series, c, lab in ((raw, "#9DAFB3", "before market_index"),
                           (net, "#064A56", "net of market_index")):
        d = series.groupby(core["date"]).mean().rolling(7, center=True).mean()
        ax[2].plot(d.index, d.values, color=c, lw=2, label=lab)
    for q_start in pd.to_datetime(["2025-04-01", "2025-07-01", "2025-10-01"]):
        ax[2].axvline(q_start, color="#C2453B", ls="--", lw=1)
    ax[2].set(title="calendar: rising trend + quarter-end ramp\n(dashed = quarter start)",
              ylabel="log residual")
    ax[2].legend(fontsize=8)
    ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "03_rate_structure.png", dpi=140)
    plt.close(fig)


def fig_geo_and_december(train, val, resid):
    core = train[resid.abs() < 5 * resid.std()]
    r = resid.loc[core.index]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    city = (pd.DataFrame({"city": core["pickup"], "lat": core["pickup_lat"],
                          "lon": core["pickup_lon"], "r": r})
            .groupby(["city", "lat", "lon"]).agg(n=("r", "size"), r=("r", "mean")).reset_index())
    city = city[city["n"] > 150]
    sc = ax[0].scatter(city["lon"], city["lat"], c=city["r"], cmap="coolwarm", s=60,
                       vmin=-0.04, vmax=0.04, edgecolor="#333", lw=0.4)
    unseen = {"Allentown", "Charlotte", "Chicago", "Jackson", "Knoxville",
              "Laredo", "Norfolk", "San Diego"}
    u = val[val["pickup"].isin(unseen)].drop_duplicates("pickup")
    ax[0].scatter(u["pickup_lon"], u["pickup_lat"], marker="x", s=90, color="#000",
                  label="8 cities absent from train")
    fig.colorbar(sc, ax=ax[0], label="origin premium (log)")
    ax[0].set(title="origin premium is a smooth geographic\ngradient, so lat/lon covers unseen cities",
              xlabel="lon", ylabel="lat")
    ax[0].legend(loc="lower left", fontsize=8)

    mi = pd.concat([train, val]).groupby("date")["market_index"].mean()
    ax[1].plot(mi.index, mi.values, color="#064A56", lw=0.9)
    ax[1].axvline(pd.Timestamp("2025-11-01"), color="#C2453B", ls="--")
    ax[1].set(title="market_index is known through Dec 31\n(validation.csv carries it)", ylabel="market_index")
    ax[1].tick_params(axis="x", rotation=30)

    dec = val[val["date"] >= "2025-12-01"].groupby("date")["market_index"].mean()
    ax[2].plot(dec.index, dec.values, marker="o", ms=3, color="#064A56")
    ax[2].set(title="December market_index: a weekly sawtooth\n(this is the chart's wiggle)", ylabel="market_index")
    ax[2].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "04_geography_and_december.png", dpi=140)
    plt.close(fig)


def fig_equipment_ramp(train, val):
    """The equipment premium is flat for 60 days then ramps to quarter end.

    December is the only quarter-end month with no labels, so this interaction has
    to be extrapolated rather than learned in place.
    """
    core = train[np.abs(np.log(train["posted_rate"] / train["distance"])
                        - np.log(2.2)) < 0.5].copy()
    y = np.log(core["posted_rate"] / core["distance"])
    ld = np.log(core["distance"])
    base = np.column_stack([ld, ld ** 2, core["weight"] / 10_000, core["market_index"],
                            core["pickup_lat"] / 10, core["delivery_lat"] / 10,
                            core["doy"] / 100, np.ones(len(core))])
    dv = core["equipment"] == "Dry Van"
    beta, *_ = np.linalg.lstsq(base[dv.values], y[dv], rcond=None)   # Dry Van sets the baseline
    core["prem"] = y - base @ beta

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    edges = [-1, 9, 19, 29, 39, 49, 59, 69, 74, 79, 84, 92]
    b = pd.cut(core["doq"], edges)
    for eq, c in (("Dry Van", "#064A56"), ("Flatbed", "#C58A22"), ("Reefer", "#1B7A4B")):
        m = core["equipment"] == eq
        ax[0].plot([i.mid for i in b.cat.categories],
                   core.loc[m, "prem"].groupby(b[m], observed=False).mean(),
                   marker="o", color=c, label=eq)
    ax[0].axvspan(60, 92, color="#C2453B", alpha=0.08)
    ax[0].set(title="equipment premium is flat for 60 days,\nthen ramps into quarter end",
              xlabel="day of quarter", ylabel="premium over Dry Van (log)")
    ax[0].legend(fontsize=8)

    # Q4's ramp region is exactly December, and it carries no labels.
    cov = pd.concat([train.assign(src="train"), val.assign(src="validation")])
    for src, c, m in (("train", "#064A56", "o"), ("validation", "#C2453B", "x")):
        d = cov[cov["src"] == src]
        ax[1].scatter(d["doq"], d["date"].dt.quarter, s=2, color=c, marker=m, label=src)
    ax[1].axvspan(60, 92, color="#C2453B", alpha=0.08)
    ax[1].set(title="the ramp is only ever labelled in Q1-Q3;\nQ4's ramp is December",
              xlabel="day of quarter", ylabel="quarter", yticks=[1, 2, 3, 4])
    ax[1].legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT / "05_equipment_quarter_end.png", dpi=140)
    plt.close(fig)


def report(train, val, resid, sigma):
    z = resid / sigma
    unseen = (set(val["pickup"]) | set(val["delivery"])) - (set(train["pickup"]) | set(train["delivery"]))
    touched = val["pickup"].isin(unseen) | val["delivery"].isin(unseen)
    rows = [
        ("corrupted labels (|z|>5)", f"{(z.abs() > 5).sum()} rows, {100 * (z.abs() > 5).mean():.2f}%"),
        ("  scaled up 2.2-5.4x", f"{(z > 5).sum()} rows"),
        ("  scaled down 0.17-0.44x", f"{(z < -5).sum()} rows"),
        ("sign-flipped weight", f"train {(pd.read_csv('train-test.csv')['weight'] < 0).sum()}, "
                               f"val {(pd.read_csv('validation.csv')['weight'] < 0).sum()}"),
        ("quote_signal reliable months", f"{GOOD_MONTHS} (corr 0.95 with target)"),
        ("quote_signal degraded months", f"{BAD_MONTHS} (corr ~0.00)"),
        ("cities in val, absent from train", f"{len(unseen)}: {sorted(unseen)}"),
        ("val rows touching those cities", f"{touched.sum()} ({100 * touched.mean():.1f}%)"),
        ("structural residual sigma", f"{sigma:.4f} (~{100 * sigma:.1f}% of rate)"),
    ]
    width = max(len(k) for k, _ in rows)
    print("\n" + "=" * 78)
    for k, v in rows:
        print(f"{k:<{width}}  {v}")
    print("=" * 78)


def main():
    OUT.mkdir(exist_ok=True)
    raw_train, raw_val = load()
    train, val = clean(raw_train), clean(raw_val)
    beta, resid, sigma = fit_structural(train)
    fig_target(train, resid, sigma)
    fig_quote_signal(train, val)
    fig_structure(train, resid)
    fig_geo_and_december(train, val, resid)
    fig_equipment_ramp(train, val)
    report(raw_train, raw_val, resid, sigma)
    print(f"figures written to {OUT}/")


if __name__ == "__main__":
    main()
