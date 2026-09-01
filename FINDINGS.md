# Freight Rate Prediction: Running Findings

Working notes for the Spotter ML assessment. Appended to as work proceeds.
Everything here is reproducible with `python eda.py` (figures land in `eda_out/`).

**Status:** Model built and both prediction files generated. Report and Loom outstanding.
**Last updated:** 2026-09-02

---

## 1. The task in one line

Predict `posted_rate` for 12,000 unlabelled loads (`validation.csv`, 2025-11-01 to 2025-12-31)
from 48,000 labelled loads (`train-test.csv`, 2025-01-01 to 2025-10-31), plus a 31-row
December chart where only the date varies.

The split is forward in time, so time-based validation is required. Random CV is actively
misleading here for reasons in section 5.

---

## 2. Headline conclusions

| # | Finding | Consequence |
|---|---|---|
| 1 | `quote_signal` is a near-perfect copy of the target in 5 of 10 training months and worthless in the other 5. Validation sits in the worthless regime. | Drop it. It is the single biggest way to fail this assessment. |
| 2 | Equipment premiums are flat for 60 days of a quarter then ramp hard into quarter end. Q4's ramp region is exactly December and is never labelled. | Needs an explicit day-of-quarter feature, otherwise December is underpriced. |
| 3 | 677 labels (1.41%) are multiplicatively corrupted, cleanly separable at 5 sigma. | Filter before training. |
| 4 | 8 cities appear only in validation, touching 12.1% of rows. | Use lat/lon, not city-name encodings. |
| 5 | Validation is about twice as dirty as training, as a discrete step. | Cleaning must run on the inference path and be robust, not tuned. |

---

## 3. Data quality

### 3.1 Corrupted labels (677 rows, 1.41%)

Residuals from a structural fit form a tight core at `|z| < 5`, then a **completely empty gap**,
then two symmetric lobes. There is no ambiguity about where to set the threshold.

- 340 rows scaled **up** by 2.2x to 5.4x
- 337 rows scaled **down** by 0.17x to 0.44x
- Spread evenly across months and equipment types, so missing-completely-at-random

Figure: `eda_out/01_target_and_outliers.png`

### 3.2 Sign-flipped weights

292 train rows and 145 validation rows carry negative `weight`. Absolute values fall in the
normal range and those rows then price like their peers, so `abs()` is the correct repair.

### 3.3 Missing values

`weight` and `market_index` only. MCAR: uniform across months, and missing-flagged rows show
no residual signature (mean `|res|` 0.0431 vs 0.0430 overall).

**`market_index` should be imputed with the same-day mean, not a global median.** It is
effectively a daily series: daily means explain 98% of its variance. Every validation day has
at least 159 non-null values, so this always works. Matters more in validation, where it is
2.0% missing versus 0.8% in training.

### 3.4 Defect rates step up at the train/val boundary

| Defect | Train (all 10 months) | Nov-Dec |
|---|---|---|
| `weight` missing | 0.6% | 1.4% |
| `weight` negative | 0.6% | 1.2% |
| `market_index` missing | 0.8% | 2.0% |

This is a discrete step, not a gradual drift, so the validation defect rate cannot be
extrapolated from a training trend.

### 3.5 Non-issues (checked and cleared)

- `distance` is floored at 70 miles, which is why very short lanes show odd
  distance/haversine ratios. Not corruption.
- `weight` is clipped to [5000, 47500] with 2.4% at the cap. The apparent -1.3% residual at
  the cap is **weight-effect curvature**, not censoring. A tree or spline absorbs it.
- City coordinates are perfectly constant per city (std 0.0). They are synthetic, not real US
  coordinates, but internally consistent.

---

## 4. Structure of the rate

A 9-term linear model on `log($/mi)` reaches R2 = 0.90 with sigma = 3.2%, so the signal is
mostly low-dimensional. Adding the equipment-by-quarter-end interaction (section 6) takes
sigma to 2.38%.

| Driver | Effect |
|---|---|
| `log(distance)` | dominant. $/mi falls from ~2.9 at 100 mi to ~1.9 at 2500 mi |
| Equipment | Reefer +12%, Flatbed +8% over Dry Van (but see section 6) |
| `weight` | +3% per 10,000 lb, **saturating** at both ends |
| `market_index` | genuine, monotone, **linear** (flat residual across 12 bins), elasticity 0.131/unit |
| Calendar | +0.62%/30d trend, plus a quarter-end ramp |
| Geography | origin/destination premium, +/- 3% |

Day-of-week has no direct effect. It acts entirely through `market_index`
(Mon 1.00, Thu 1.18, Sun 0.99), which is why raw day-of-week looks significant and then
vanishes once `market_index` is controlled for.

Figure: `eda_out/03_rate_structure.png`

---

## 5. Trap 1: `quote_signal` is regime-switched

`quote_signal` equals the target's rate-per-mile, but only in some months.

| Months | Loads within 2% of actual $/mi | Correlation with target |
|---|---|---|
| Jan, Feb, Mar, Jun, Sep | 92-94% | **0.95** |
| Apr, May, Jul, Aug, Oct | 7-10% | **~0.00** |

The switch lands exactly on month boundaries.

### Why this matters

On a random CV split this feature looks like a jackpot, because half the training rows have
the answer written on them. On validation it is worth nothing.

### Which regime is validation in?

Determinable without labels. The degraded months have a distinct marginal shape, capped near
2.5 with a long low tail. A KS test of validation against each training month:

| Feature | Best-matching month | Next best |
|---|---|---|
| **`quote_signal`** | **Aug 0.009** | Jan 0.052 |
| `market_index` | Jan 0.039 | Oct 0.152 |
| `distance` | 0.007 | (all months <= 0.017) |
| `weight` | 0.008 | (all months <= 0.027) |
| all 4 lat/lon | 0.010-0.024 | no shift |

August is a degraded month, and the match is 6x tighter than the runner-up. **Validation is
in the degraded regime.**

### It cannot be partially salvaged

I tested whether the ~10% of degraded-month rows where `quote_signal` happens to agree with a
model prediction are trustworthy. They are not: residual sd 0.0230 versus 0.0231 baseline,
correlation -0.047. Row-wise gating buys nothing. Drop the feature.

Figure: `eda_out/02_quote_signal_regime.png`

---

## 6. Trap 2: equipment premium ramps into quarter end

Fitting the model independently per month, distance, weight, and lat/lon coefficients are
stable to within a few percent. **Equipment is not.** Flatbed and Reefer premiums jump in
March, June, and September, and only those months.

The cause: the equipment premium is flat for the first 60 days of a quarter, then ramps.

| | day 0-59 | day 84-92 |
|---|---|---|
| Flatbed premium | +0.068 | **+0.140** |
| Reefer premium | +0.113 | **+0.160** |
| Dry Van (general ramp) | 0.000 | +0.022 |

It repeats identically in Q1, Q2, Q3. Adding this one interaction cuts residual sigma from
0.0321 to 0.0238, a 26% reduction.

### Why this is a trap

**Q4's ramp region is never labelled.** Training stops Oct 31, which is day 30 of Q4. The
split falls exactly on the boundary:

- **November** = days 31-60 of Q4, entirely *before* the ramp
- **December** = days 61-91, entirely *inside* it

So half the validation set tests the base model and the other half tests an extrapolation
into a region with zero training coverage. Without an explicit day-of-quarter feature, roughly
2,660 December Flatbed/Reefer loads are underpriced by 3-5%, and every December load by a
further ~2%.

This is a different failure mode from `quote_signal`: not leakage, but unobserved-region
extrapolation.

Figure: `eda_out/05_equipment_quarter_end.png`

---

## 7. Geography: one city premium, applied at both ends

### 7.1 A load pays the premium at BOTH ends, additively

**Method note.** The first pass grouped per-load residuals by `pickup` and separately by
`delivery`, giving two marginal averages. That alone cannot show a single load pays at both
ends, because lane pairing confounds it: if Houston ships mostly to other high-premium cities,
the origin average absorbs the destination effect. Re-estimated with **joint origin and
destination fixed effects** (64 + 64 dummies, fitted simultaneously), which decontaminates it.
The marginal estimates survive: correlation 0.9994 with the joint ones, sd 0.0177 vs 0.0179.
Lane pairing in this dataset happens to be close to random, so the original numbers held.

Three tests, all on the per-load residual after a no-geography base model:

| Test | Result | Reading |
|---|---|---|
| Regress residual on `premium(origin)` and `premium(destination)` jointly | slopes **1.013** and **1.015** | a load pays the **full** premium at each end |
| Regress (forward - reverse lane) on the premium gap between the two cities | slope **+0.022** | symmetric: 1.0 would mean origin-only |
| Joint FE: corr(origin FE, destination FE) | **0.992**, slope 0.982 | one premium, not two |

Concrete check, predicting a lane's residual as `prem(A) + prem(B)`:

| Lane | Predicted | Actual |
|---|---|---|
| Houston to Dallas (high + high) | +0.0539 | +0.0522 |
| Houston to Syracuse (high + low) | -0.0075 | -0.0072 |

So a Houston to Dallas load carries **both** premiums, about +5.4% in total, while Houston to
Syracuse nets out to roughly zero. Reversing a lane does not change the price.

Modelling both ends drops the residual sd from 0.0316 to **0.0198**, so this is the single
largest remaining effect after distance, equipment and calendar.

### 7.2 The gradient is latitude, not longitude

Longitude contributes essentially nothing. Its coefficient is ~85x smaller than latitude's,
which over the observed ranges is 0.002 of effect versus 0.065. The map reads as a
northeast/southwest split only because latitude and longitude happen to be correlated across
this city set. The real effect is **purely north-south**: Great Lakes and Northeast about -3%,
Texas and South-central about +3%.

### 7.3 8 cities exist only in validation

Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego.
They touch **1,447 validation rows (12.1%)**, and all 8 sit inside the region covered by
training cities.

How well can their premium be recovered from coordinates alone? Leave-one-city-out on the 64
training cities is the honest simulation:

| Method | R2 on held-out city | Residual |
|---|---|---|
| global plane in lat/lon | 0.708 | 0.96% |
| quadratic in lat/lon | 0.833 | 0.72% |
| **kNN on lat/lon (k=1 to 5)** | **0.889** | **0.58%** |

The premium field is smooth and **locally** predictable, so local interpolation beats any
global surface. A tree model splitting on lat/lon does this naturally. Conclusion:
**lat/lon is genuinely sufficient for the unseen cities**, leaving only ~0.6% residual error
on their 12.1% of rows. A city-name encoding recovers nothing for them.

### 7.4 Lane effects survive the additive city model, and they are directional

After removing the symmetric additive city premium, real per-lane structure remains.
Noise-corrected variance decomposition over 2,711 lanes with 8+ loads:

- observed variance of lane means 0.000154, expected from noise 0.000030
- **implied true lane sd 0.0112 (1.1% of rate)**
- forward-minus-reverse differences sit 316% above noise, so the leftover lane effect is
  **direction-specific**, unlike the city premium

It replicates rather than being a composition artifact: estimating lane effects on two
disjoint halves of the calendar (before and after Jun 1) gives correlation **+0.82** across
1,333 lanes, and +0.73 restricting to Dry Van only.

Practical value: median lane has 10 training loads, giving SE 0.0062 against a true sd of
0.0112, so a shrinkage factor of 0.76. That recovers sd 0.0098 and would take the residual
from 0.0198 to 0.0172 on seen lanes, about 13%. Against that, **1,465 validation rows (12.2%)
sit on lanes absent from training** and get nothing. Worth doing with shrinkage, worth nothing
without it.

Figure: `eda_out/04_geography_and_december.png`

---

## 8. Model choice: measured, not assumed

Two replica splits inside the training data have the *same shape* as the real task, namely
train through day 30 of a quarter and predict days 31-91 of that same quarter, two months
forward:

| Split | Train | Holdout | Quarters fully covering the ramp |
|---|---|---|---|
| Q2 replica | Jan 1 - Apr 30 | May 1 - Jun 30 | Q1 only |
| Q3 replica | Jan 1 - Jul 31 | Aug 1 - Sep 30 | Q1, Q2 |
| *real task* | *Jan 1 - Oct 31* | *Nov 1 - Dec 31* | *Q1, Q2, Q3* |

The replicas bracket the real task and are both slightly harder than it, since each has fewer
prior quarters to learn the ramp from.

### 8.1 Results

All models fitted on `log($/mi)`, corrupted labels removed, same feature set.

| Split | Model | MAE | RMSE | bias wk 1 | bias wk 8 |
|---|---|---|---|---|---|
| Q2 | GBM (with day_of_year) | 0.0216 | 0.0270 | +0.0022 | **+0.0322** |
| Q2 | GBM (no day_of_year) | 0.0220 | 0.0274 | +0.0057 | +0.0315 |
| Q2 | parametric (linear trend) | 0.0190 | 0.0240 | +0.0044 | +0.0040 |
| Q2 | **hybrid** | **0.0140** | **0.0175** | +0.0022 | +0.0040 |
| Q3 | GBM (with day_of_year) | 0.0151 | 0.0191 | -0.0004 | +0.0024 |
| Q3 | GBM (no day_of_year) | 0.0303 | 0.0347 | +0.0256 | +0.0301 |
| Q3 | parametric (linear trend) | 0.0196 | 0.0245 | -0.0065 | -0.0078 |
| Q3 | **hybrid** | **0.0149** | **0.0187** | -0.0059 | -0.0096 |

### 8.2 What this settles

**Tree flattening is real, but its cost is not fixed.** Training covers day-of-year 1-304 and
the holdout is 305-365, so every test row falls in the top `day_of_year` bin and inherits the
last training days' value. The fitted curve visibly flattens: on the Q3 replica the GBM's
Q3 quarter-end peak lands *below* its own Q2 peak, which cannot be right.

The damage depends entirely on whether the underlying level is moving during the holdout:

- **Q2 replica: 3.2% by week 8.** May-June was a steeply rising market, and the GBM missed all
  of it.
- **Q3 replica: 0.24% by week 8.** August-September was flat, so flattening cost almost
  nothing.

I had guessed 1-2%. The truth is that it varies from negligible to severe depending on the
regime, and **which case December resembles is not knowable in advance.**

**The diagnostic is bias growth, not average bias.** The GBM's bias grows into the holdout
(+0.0022 to +0.0322 on Q2), while parametric and hybrid stay flat (+0.0044 to +0.0040). Average
bias hides this. December is the far end of the holdout, so bias growth is exactly what hurts.

**Dropping `day_of_year` is worse than flattening it.** Q3 bias +0.0292 versus +0.0015. The
feature carries the seasonal baseline; freezing it beats not having it.

**The linear trend alone over-extrapolates.** Parametric bias is -0.0075 on Q3, so a naive
straight line overshoots.

### 8.3 Going further: structure beats capacity

The plain hybrid was not the end of it. Adding the structure the EDA had already established
improves it consistently:

1. **Linear splines** on `log(distance)`, `weight` and the quarter-end ramp. All three are known
   to be nonlinear. Linear beyond the outer knots, so `day_of_year` extrapolation is unaffected.
2. **Explicit shrunk city effects.** One symmetric premium per city applied at both ends
   (section 7.1), empirical-Bayes shrunk, with kNN-on-lat/lon fallback for the 8 cities absent
   from training. The GBM otherwise has to rediscover this through coordinate splits.
3. **Explicit shrunk lane effects**, weighted by lane volume (section 7.4).

| Split | plain hybrid MAE | structured MAE | gain |
|---|---|---|---|
| Q2 replica | 0.0140 | **0.0126** | 10% |
| Q3 replica | 0.0149 | **0.0136** | 9% |

### 8.4 Three things that did *not* work

Recorded because the negative results are as load-bearing as the positive ones.

**Tuning the trend window made it worse.** Since the baseline is not linear (it climbs Jan-Jul,
plateaus Jul-Sep, then steps up in October), a trend fitted on recent history rather than all
history looked promising: full-year slope is +0.56%/30d, last-60-days is +1.61%/30d, a ~2%
disagreement over a two-month horizon. Tested on **five** rolling-origin folds, each predicting
two months forward, because two replicas is far too thin to tune on:

| Variant | mean MAE | mean abs bias (late) | worst bias |
|---|---|---|---|
| **joint linear day_of_year** | **0.0140** | **0.0100** | **0.0180** |
| window 90d, damp 1.0 | 0.0153 | 0.0120 | 0.0213 |
| window 120d, damp 1.0 | 0.0156 | 0.0130 | 0.0201 |
| window 180d / all, damped | 0.0172-0.0188 | 0.0166-0.0202 | 0.0256-0.0269 |

The joint fit wins on every metric, and it is the only variant whose bias alternates sign
across folds rather than being systematically positive. Estimating the trend separately from
the ramp prevents the two from co-adjusting, and every two-stage variant under-predicts.

**More GBM capacity hurts.** On the Q3 replica: 200 trees 0.0136, 400 trees 0.0138, depth 7
0.0139, 800 trees at lr 0.03 0.0138.

**The GBM itself is now nearly redundant.** Once the structure is explicit it adds 0.0002 MAE,
about 1.4%, consistent across all five folds but marginal:

| cut | no GBM | + GBM |
|---|---|---|
| May-Jun | 0.0129 | 0.0126 |
| Jun-Jul | 0.0133 | 0.0131 |
| Jul-Aug | 0.0166 | 0.0164 |
| Aug-Sep | 0.0136 | 0.0136 |
| Sep-Oct | 0.0137 | 0.0136 |
| **mean** | **0.0140** | **0.0139** |

So roughly 99% of the performance comes from the explicit specification, not the trees. The
final model is essentially a well-specified GLM with shrunk fixed effects, plus a thin
gradient-boosted correction.

### 8.5 How close to the floor?

Fitting in-sample with *saturated* lane fixed effects, an optimistic bound, leaves residual sd
0.0142. Holdout MAE of 0.0136 implies sd about 0.0170. So the model sits within roughly 20% of
a bound that is itself optimistic.

Remaining error is concentrated in three places, none of which look reducible with this data:

- **~1% bias at the December-equivalent horizon**, alternating sign across folds. This is trend
  uncertainty about an unobservable future.
- **12.2% of validation rows on unseen lanes**, which get no lane effect.
- **~0.6% floor on the 8 unseen cities**, the part of their premium coordinates cannot recover.

### 8.6 Decision

Structured model: splines + shrunk city effects + shrunk lane effects + joint linear trend +
quarter-end ramp with equipment interaction, then a thin GBM on the residual with
`day_of_year` withheld.

Scripts: `experiments/gbm_extrapolation.py`, `replica_splits.py`, `model_v2.py`,
`trend_choice.py`, `rolling_origin.py`. Figure: `eda_out/06_gbm_extrapolation.png`.
Still numpy/pandas/matplotlib only.

---

## 8b. The December chart

Fixed inputs: Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb. Only the date varies.

Three things drive the shape:

1. **Quarter-end ramp.** All of December sits at day-of-quarter 61-91, inside the ramp.
   Roughly +3.4% across the month.
2. **Linear trend.** +0.62%/30d.
3. **`market_index` weekly sawtooth.** Already known for every December date, because
   `validation.csv` covers Nov 1 to Dec 31. No forecasting needed. Thursday peaks ~1.03,
   Sunday troughs ~0.83, which at elasticity 0.131 is a +/- 1.3% ripple.

Back-of-envelope from the structural model: **$837 on Dec 1 rising to $900 on Dec 31 (+7.5%)**,
with a weekly ripple. Sanity anchor: the 32 real Lexington to Fort Wayne loads in training
span $758-974 (mean $857) across Jan-Oct.

**A flat December line means the time structure was missed.** That chart is the qualitative
test in this assessment.

Caveat: the quarter-end ramp magnitude varies somewhat by quarter for Dry Van
(+0.021, +0.022, +0.016 at day 84-92 for Q1, Q2, Q3), which is real uncertainty for Q4.

---

## 8c. Final model and results

`model.py` backtests, fits on all of Jan-Oct, and writes both prediction files.

| Fold | n test | MAE (log) | RMSE (log) | MAPE | bias, last 2wk |
|---|---|---|---|---|---|
| Q2 replica (May-Jun) | 9,571 | 0.0126 | 0.0158 | 1.26% | +0.0034 |
| Jun-Jul | 9,567 | 0.0131 | 0.0164 | 1.31% | -0.0093 |
| Jul-Aug | 9,538 | 0.0163 | 0.0201 | 1.65% | -0.0176 |
| Q3 replica (Aug-Sep) | 9,297 | 0.0137 | 0.0172 | 1.38% | -0.0106 |
| Sep-Oct | 9,379 | 0.0137 | 0.0170 | 1.36% | +0.0113 |
| **mean** | | **0.0139** | | **1.39%** | |

Trained on 47,323 rows after removing 677 corrupted labels. In-sample residual sd 0.0145.

### Sanity checks on the output

| Check | Result |
|---|---|
| predicted rate distribution | median $2,072 vs $2,031 in training; p05/p95 $628/$4,985 vs $600/$4,954 |
| predicted $/mi | median 2.177 vs 2.145 in training |
| Dec vs Nov $/mi | 2.206 vs 2.146, i.e. +2.8%, the quarter-end ramp |
| equipment premium, Nov (pre-ramp) | Flatbed +7.1%, Reefer +12.0% |
| equipment premium, Dec (in ramp) | Flatbed +10.3%, Reefer +13.9% |
| the 8 unseen cities | $/mi median 2.181 vs 2.177 for the rest, no outlier behaviour |
| format | 12,000 unique ids, all positive, no NaN; scorer passes |

The November and December equipment premia land on the values measured in section 6, which
confirms the quarter-end interaction is being applied rather than merely declared.

### The December chart

$832 on Dec 1 rising to $885 on Dec 31, **+6.3%** across the month, with a weekly ripple from
the `market_index` sawtooth. Close to the section 8b back-of-envelope of $837 to $900, and
inside the $758-974 range of the 32 real Lexington to Fort Wayne loads in training.

The line rises because December sits entirely inside the quarter-end ramp. A flat line would
have meant the time structure was missed.

---

## 8d. Freight domain seasonality: produce season and retail peak

Checked because the calendar work so far was mechanical (quarter-end, trend, weekly) and never
tested the named industry seasons directly.

### Spring produce season (Apr-Jul): present, and already captured

`market_index` averaged by month over all 12 months:

| Month | 1 | 2 | 3 | **4** | **5** | **6** | **7** | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.93 | 1.00 | 1.07 | **1.21** | **1.30** | **1.28** | **1.20** | 0.98 | 0.89 | 0.96 | 0.92 | 0.93 |

Apr-Jul averages **1.248** against 0.944 for Aug-Oct. That is a textbook produce-season shape,
peaking in exactly the window the season occupies.

It needs no extra modelling, for two reasons: `market_index` is already a model feature, and
its Nov-Dec values are **given** in `validation.csv`. The seasonal cycle arrives for free and
never has to be forecast.

### Retail peak (Aug-Oct): not present in this data

Aug-Oct is the annual **trough** (0.944), not a peak. Whatever generated this dataset modelled
a produce-driven cycle and not an import/holiday-driven one.

### It does not act through equipment or region

If a produce season were operating on top of the market-wide cycle, Reefer would be lifted in
Apr-Jul. It is not. Premium relative to Dry Van, by month, after the model:

| Equipment | Apr | May | Jun | Jul | vs Jan-Mar |
|---|---|---|---|---|---|
| Reefer | -0.0007 | +0.0004 | +0.0010 | -0.0009 | flat within 0.1% |
| Flatbed | +0.0000 | +0.0009 | +0.0007 | -0.0002 | flat within 0.2% |

Nor by region. Splitting origins into produce states, port cities and everything else, crossed
with the three season blocks, the spread across all nine cells is **0.0022** against a model
residual sd of 0.0145. Produce origins in Apr-Jul sit at +0.0010 versus +0.0002 for others.

### Modelling the cycle explicitly makes it worse

An annual sin/cos pair, the smooth extrapolable version of a seasonal cycle, was tested on the
five rolling folds:

| Fold | base | + annual harmonic |
|---|---|---|
| Q2 replica | 0.0129 | **0.0260** |
| Jun-Jul | 0.0133 | 0.0132 |
| Jul-Aug | 0.0166 | 0.0129 |
| Q3 replica | 0.0136 | 0.0132 |
| Sep-Oct | 0.0137 | **0.0233** |
| **mean** | **0.0140** | **0.0177** |

It helps on three folds and destroys two. Fitting an annual cycle from a partial year and
extrapolating past the observed window is unstable, which is precisely the situation Nov-Dec
puts us in. Rejected.

### Leftover monthly wobble

Residual month effects after the full model are real but tiny, at most 0.5% (Aug -0.0048,
Oct +0.0057). The shape, a plateau through Aug-Sep then an October step, matches the
trend misspecification noted in section 8.4 rather than any seasonal story.

---

## 9. Modelling decisions carried forward

- [x] **Drop `quote_signal`.** Not gated, not row-wise, dropped.
- [x] **Time-based split.** Use the Q2 and Q3 replicas (section 8), which reproduce the real
      task's shape. Never random CV.
- [x] **Filter 677 corrupted labels** at 5 sigma before training.
- [x] **Repair on the inference path**: `abs()` on weight, same-day mean for `market_index`.
- [x] **Explicit calendar features**: day-of-quarter ramp, day-of-year trend, and the
      day-of-quarter by equipment interaction.
- [x] **Geography via lat/lon**, not city-name encodings. One symmetric city premium,
      driven by latitude. Local interpolation recovers ~89% of it for unseen cities.
- [x] **Model `log(rate/mile)`**, not raw rate, given the multiplicative structure throughout.
- [x] **Structured model**: splines + shrunk city and lane effects + joint linear trend.
      The GBM stage is worth only ~1.4%. See section 8.
- [x] Validate the December curve is rising, roughly $840 to $900.

## 10. Open questions

- Does the Q4 ramp match Q1-Q3 in magnitude? Unobservable. Q1-Q3 spread is 0.016-0.022 for
  Dry Van, so there is irreducible uncertainty in the December level.
- Is the lane-level residual structure worth modelling, given 1,461 validation rows sit on
  lanes never seen in training?

---

## Changelog

- **2026-09-01** EDA complete. Both traps identified and characterized. Sections 1-10 written.
- **2026-09-01** Section 7 expanded: origin and destination premiums shown to be the same
  symmetric city effect (corr 0.992), driven by latitude alone; leave-one-city-out kNN
  quantifies what is recoverable for the 8 unseen cities.
- **2026-09-02** Section 8d added: tested the named freight seasons. Spring produce season is
  present but arrives entirely through `market_index`, whose Nov-Dec values are given, so it
  needs no modelling. Retail peak is absent (Aug-Oct is the trough). No equipment or region
  interaction. An explicit annual harmonic was tested and rejected as unstable.
- **2026-09-02** Model built (`model.py`, `pipeline.py`, `gbm.py`). Both prediction files
  written and the scorer passes. Backtest mean MAE 0.0139 (1.39% MAPE) over five folds.
  December chart rises $832 to $885. Section 8c added.
- **2026-09-02** Sections 8.3-8.6 added: structured model beats the plain hybrid by ~10%.
  Trend-window tuning and extra GBM capacity both tested and rejected; the GBM stage turns out
  to be worth only 1.4% once structure is explicit. Model is within ~20% of an optimistic
  noise floor.
- **2026-09-02** Section 8 added: replica-split evaluation of four model architectures.
  Corrects an earlier guess that tree flattening would cost 1-2%; measured cost ranges from
  0.24% to 3.2% depending on regime. Hybrid chosen on evidence.
- **2026-09-01** Section 7.1 re-derived with joint origin/destination fixed effects after the
  marginal-average method was questioned. Confirms a load pays the full premium at both ends
  (slopes 1.013 / 1.015) and that the effect is direction-symmetric. Section 7.4 now
  noise-corrects and replication-tests the leftover lane effect.
