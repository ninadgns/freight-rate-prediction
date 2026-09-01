# Freight Rate Prediction

Predicts `posted_rate` for 12,000 loads in `validation.csv` (Nov-Dec 2025) from 48,000
labelled loads in `train-test.csv` (Jan-Oct 2025), plus a fixed-input December chart.

Full analysis and the reasoning behind every modelling decision is in **[FINDINGS.md](FINDINGS.md)**.

## Setup and run

Requires Python 3.10+. No dependencies beyond the three the provided scorer already needs.

```bash
python -m pip install -r requirements.txt

python model.py          # backtests, then writes both prediction files (~50s)

python score.py --predictions validation_predictions.csv \
                --december-predictions december-chart-inputs.csv
```

`model.py` writes `validation_predictions.csv` and fills the `predicted_rate` column of
`december-chart-inputs.csv`. `score.py` validates both and writes
`scorer_results/candidate_december.png`.

To regenerate the exploratory figures in `eda_out/`:

```bash
python eda.py
```

## Results

Backtest, training to each cut and predicting the following two months, which is the
real task's horizon:

| Fold | MAE (log) | RMSE (log) | MAPE |
|---|---|---|---|
| Q2 replica (May-Jun) | 0.0126 | 0.0158 | 1.26% |
| Jun-Jul | 0.0131 | 0.0164 | 1.31% |
| Jul-Aug | 0.0163 | 0.0201 | 1.65% |
| Q3 replica (Aug-Sep) | 0.0137 | 0.0172 | 1.38% |
| Sep-Oct | 0.0137 | 0.0170 | 1.36% |
| **mean** | **0.0139** | | **1.39%** |

The two "replica" folds match the real task's shape exactly: train through day 30 of a
quarter, predict days 31-91 of that same quarter. No random cross-validation is used
anywhere, for reasons in the next section.

## The two things that decide this problem

**`quote_signal` is a trap.** It reproduces the target's rate-per-mile almost exactly in
Jan, Feb, Mar, Jun and Sep (correlation 0.95) and is worthless in the other five months
(correlation ~0.00). Under random cross-validation it looks like a jackpot, because half
the training rows carry the answer. A KS test against each training month shows the
validation window matches August, a degraded month, 6x more closely than any alternative.
**The feature is excluded from the model entirely.** It cannot be rescued row-wise either:
gating on agreement with a model prediction leaves error unchanged.

**The equipment premium ramps into quarter end, and Q4's ramp is never labelled.** Flatbed
goes from +0.068 to +0.140 and Reefer from +0.113 to +0.160 over the last 30 days of a
quarter, repeating identically in Q1, Q2 and Q3. Training stops on Oct 31, day 30 of Q4, so
November falls entirely before the ramp and December entirely inside it. Half the holdout is
therefore an extrapolation into a region with no training coverage, which needs an explicit
day-of-quarter feature rather than a feature to drop. Modelling it cuts residual sigma by 26%.

## How it works

`pipeline.py` cleans, then fits four stages, each on the residual of the one before:

1. **Parametric core.** Linear splines on `log(distance)` and `weight`, equipment dummies,
   `market_index`, a **linear** day-of-year trend, and a quarter-end ramp interacted with
   equipment. The trend is linear on purpose: training covers day-of-year 1-304 and the
   holdout is 305-365, and trees cannot extrapolate beyond their training range.
2. **City premium.** One symmetric premium per city, applied at *both* ends of the lane. A
   load pays the full premium at origin and destination (regression slopes 1.013 and 1.015),
   and reversing a lane does not change the price. Empirical-Bayes shrunk, with kNN on
   lat/lon for the 8 cities that appear only in validation.
3. **Lane effect.** Per-lane, shrunk by volume.
4. **Gradient-boosted correction** on what remains, with day-of-year withheld.

Stage 4 is worth only about 1.4%: once the structure is explicit, the trees have little left
to find. The model is effectively a well-specified GLM with shrunk fixed effects.

Cleaning removes 677 corrupted labels (1.41%, separated from the clean core by an empty gap),
repairs sign-flipped weights with `abs()`, and imputes `market_index` from the same-day mean
rather than a global median, since daily means explain 98% of its variance.

## Layout

```
model.py          entry point: backtest, fit, write both prediction files
pipeline.py       cleaning, features, the model
gbm.py            minimal histogram gradient booster (numpy only)
eda.py            exploratory analysis, writes eda_out/
experiments/      model selection: architecture, trend handling, replica splits
FINDINGS.md       full analysis and the evidence for every decision
```
