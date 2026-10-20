# Module 3 implementation plan

## Initial assessment

Module 3 builds on, rather than replaces, the frozen Module 1 cohort and Module
2 continuous-discrete OU state-space engine. Reusable components are:

- `cohort_renal_bounded.csv`: the bounded 2,500-patient, six-analyte cohort.
- Module 1 canonicalization and `POLICY_A_STORETIME` arbitration.
- Frozen patient-level train/validation/test partition (1,750/375/375).
- Training-only scaler parameters, including log transforms for creatinine, BUN,
  and glucose.
- `PatientInferenceService`, which supplies causal operational/biological
  eligibility and the fitted three-state filter.
- The Module 2 observation matrix `H`, propagated covariance `P`, and
  measurement covariance `R`.

The available processed interface contains serum creatinine, BUN, potassium,
sodium, bicarbonate, glucose, timestamps, arbitration metadata, age, sex, and
admission metadata. It does **not** establish cystatin C, albuminuria, blood
pressure, medications, CKD diagnoses, or adjudicated progression outcomes.
Consequently, Module 3 will not expose a deterioration-risk score initially.

## Forecasting problem

Primary target: future observed serum creatinine. Secondary targets are BUN,
potassium, sodium, bicarbonate, and glucose. Forecasts originate at a supplied
clinical wall-clock query time and use only observations eligible under the
selected operational (reported-result) or biological (collection-time,
retrospective) semantics.

The initial supported horizons are 24, 72, and 168 hours. They are configurable
only through this fixed, documented set until the feasibility audit records the
actual follow-up support. A forecast is an observation-model forecast at
`origin + horizon`, not an invented continuously observed trajectory. Retrospective
evaluation pairs an origin with the first eligible target observation in a
predeclared ±25% horizon window; absent targets are excluded and reported.

Minimum history is one finite, eligible observation that can update the filter.
No eligible observation returns an explicit `INSUFFICIENT_HISTORY` response.

## Uncertainty and baselines

At a horizon, state covariance is propagated using the existing OU equations.
For analyte observation row `H`, predictive observation variance is
`H P Hᵀ + R`; latent uncertainty `H P Hᵀ` is retained as a separate quantity.
Prediction intervals are computed in transformed space and inverse transformed
at their endpoints. LOCF and a training-fitted population mean are required
baselines; all evaluation masks must contain finite predictions from every
compared method.

Predefined acceptance gate: on the frozen test set, the state-space forecast
must beat LOCF in RMSE for creatinine at a supported horizon *and* report its
per-analyte/horizon coverage. Failure does not block an API response, but blocks
claims of predictive superiority and scenario simulation.

## Ordered phases

1. Generate a feasibility audit: follow-up, gaps, usable origins/targets, and
   deterioration-endpoint variable availability.
2. Implement a causal forecast engine with state/observation uncertainty and
   explicit unavailable statuses.
3. Add synthetic tests for horizons, irregular times, time semantics, inverse
   transforms, missing history, and split isolation.
4. Add typed backend forecasting endpoints. Risk remains unavailable unless a
   documented endpoint is feasible.
5. Run validation then one frozen-test evaluation. Generate machine-readable and
   scientific reports before any frontend scenario capability.
6. Add a focused frontend view only after the backend contract and evaluation
   report are stable.

## Current blockers and assumptions

- The existing Module 2 full-test audit found OU weaker than LOCF for creatinine
  and BUN. Module 3 will report, not tune away, that limitation.
- No independent deterioration endpoint is currently available. A future
  lab-defined proxy would require its own leakage/censoring audit and cannot be
  presented as a clinical diagnosis.
- The feasibility report is the authority for final horizon support; no raw data,
  split, scaler, or Module 2 parameter will be altered by this work.
