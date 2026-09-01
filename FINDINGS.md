# Freight Rate Prediction: Running Findings

Working notes for the Spotter ML assessment. Appended to as work proceeds.
Everything here is reproducible with `python eda.py` (figures land in `eda_out/`).

**Status:** EDA complete. Modelling not started.
**Last updated:** 2026-09-01

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

## 8. The December chart

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

## 9. Modelling decisions carried forward

- [ ] **Drop `quote_signal`.** Not gated, not row-wise, dropped.
- [ ] **Time-based split.** Hold out Aug and Oct as degraded-regime proxies for validation.
      Never random CV.
- [ ] **Filter 677 corrupted labels** at 5 sigma before training.
- [ ] **Repair on the inference path**: `abs()` on weight, same-day mean for `market_index`.
- [ ] **Explicit calendar features**: day-of-quarter ramp, day-of-year trend, and the
      day-of-quarter by equipment interaction.
- [ ] **Geography via lat/lon**, not city-name encodings. One symmetric city premium,
      driven by latitude. Local interpolation recovers ~89% of it for unseen cities.
- [ ] **Model `log(rate/mile)`**, not raw rate, given the multiplicative structure throughout.
- [ ] Validate the December curve is rising, roughly $840 to $900.

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
- **2026-09-01** Section 7.1 re-derived with joint origin/destination fixed effects after the
  marginal-average method was questioned. Confirms a load pays the full premium at both ends
  (slopes 1.013 / 1.015) and that the effect is direction-symmetric. Section 7.4 now
  noise-corrects and replication-tests the leftover lane effect.
