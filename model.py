"""Train the freight rate model and write both prediction files.

Run:  python model.py

Outputs:
  validation_predictions.csv        load_id,predicted_rate for all 12,000 loads
  december-chart-inputs.csv         predicted_rate column filled in place

Then:
  python score.py --predictions validation_predictions.csv \
                  --december-predictions december-chart-inputs.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import (FreightRateModel, TRAIN_CSV, VAL_CSV, city_coords, clean,
                      drop_corrupted, load_training)

DECEMBER_CSV = Path("december-chart-inputs.csv")
TEMPLATE_CSV = Path("validation-predictions-template.csv")
PREDICTIONS_CSV = Path("validation_predictions.csv")

# Each holds out two months, the real task's horizon. The two quarter-aligned
# replicas match its shape exactly. See FINDINGS.md section 8.
FOLDS = {
    "Q2 replica (May-Jun)": "2025-05-01",
    "Jun-Jul": "2025-06-01",
    "Jul-Aug": "2025-07-01",
    "Q3 replica (Aug-Sep)": "2025-08-01",
    "Sep-Oct": "2025-09-01",
}


def backtest(train: pd.DataFrame) -> None:
    print("Backtest: train to the cut, predict the next two months\n")
    print(f"{'fold':<22} {'n test':>7} {'MAE':>8} {'RMSE':>8} {'MAPE':>7} {'bias, last 2wk':>15}")
    print("-" * 74)
    maes = []
    for name, cut in FOLDS.items():
        cut_ts = pd.Timestamp(cut)
        tr = train[train["date"] < cut_ts]
        te = train[(train["date"] >= cut_ts) & (train["date"] < cut_ts + pd.DateOffset(months=2))]
        if te.empty:
            continue
        pred = FreightRateModel().fit(tr).predict_log_rpm(te)
        err = te["y"].values - pred
        late = (te["date"] >= te["date"].max() - pd.Timedelta(days=14)).values
        mape = np.abs(np.expm1(-err)).mean() * 100
        maes.append(np.abs(err).mean())
        print(f"{name:<22} {len(te):>7,} {np.abs(err).mean():8.4f} "
              f"{np.sqrt((err ** 2).mean()):8.4f} {mape:6.2f}% {err[late].mean():+15.4f}")
    print("-" * 74)
    print(f"{'mean':<22} {'':>7} {np.mean(maes):8.4f}\n")
    print("MAE and RMSE are in log space; MAPE is percent error on the dollar rate.\n")


def december_frame(model: FreightRateModel, val: pd.DataFrame) -> pd.DataFrame:
    """The December file carries only 7 columns, so coordinates and market_index
    have to be supplied. Both are known: coordinates are fixed per city, and
    validation.csv already covers every December date."""
    dec = pd.read_csv(DECEMBER_CSV)
    original_columns = list(dec.columns)
    d = dec.copy()
    d["date"] = pd.to_datetime(d["date"])

    coords = model.coords_ | city_coords(val)
    for side in ("pickup", "delivery"):
        d[f"{side}_lat"] = [coords[c][0] for c in d[side]]
        d[f"{side}_lon"] = [coords[c][1] for c in d[side]]

    daily = val.groupby("date")["market_index"].mean()
    d["market_index"] = d["date"].map(daily)
    if d["market_index"].isna().any():
        missing = d.loc[d["market_index"].isna(), "date"].dt.date.tolist()
        raise SystemExit(f"no market_index available for December dates: {missing}")

    d["doy"] = d["date"].dt.dayofyear
    d["doq"] = (d["date"] - d["date"].dt.to_period("Q").dt.start_time).dt.days
    dec["predicted_rate"] = np.round(model.predict(d), 2)
    return dec[original_columns]


def main() -> None:
    train_all = load_training()
    train = drop_corrupted(train_all)
    print(f"Training rows: {len(train):,} of {len(train_all):,} "
          f"({len(train_all) - len(train)} corrupted labels removed)\n")

    backtest(train)

    print("Fitting on all of Jan-Oct ...")
    model = FreightRateModel().fit(train)
    print(f"in-sample residual sd {model.resid_sd_:.4f}\n")

    val = clean(pd.read_csv(VAL_CSV, parse_dates=["date"]))
    val_pred = model.predict(val)

    template = pd.read_csv(TEMPLATE_CSV)
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(val_pred, 2)}),
        on="load_id", how="left")
    if out["predicted_rate"].isna().any():
        raise SystemExit("some template load_ids got no prediction")
    out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Wrote {PREDICTIONS_CSV} ({len(out):,} rows), "
          f"rate ${out.predicted_rate.min():,.0f} to ${out.predicted_rate.max():,.0f}, "
          f"median ${out.predicted_rate.median():,.0f}")

    dec = december_frame(model, val)
    dec.to_csv(DECEMBER_CSV, index=False)
    lo, hi = dec["predicted_rate"].iloc[0], dec["predicted_rate"].iloc[-1]
    print(f"Wrote {DECEMBER_CSV}: Dec 1 ${lo:,.0f} to Dec 31 ${hi:,.0f} "
          f"({100 * (hi / lo - 1):+.1f}% across the month)")


if __name__ == "__main__":
    main()
