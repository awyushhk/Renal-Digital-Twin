# Module 3 methodology: observable-biomarker forecasts

## Scope

Module 3 forecasts future **observable laboratory biomarkers**, beginning with
serum creatinine. It reuses the Module 2 three-state OU filter and makes no
claim to forecast a direct measure of nephron count, renal reserve, treatment
effect, or clinical diagnosis.

## Causal origin

At forecast origin `t`, operational forecasts use only effective records with
`storetime <= t` and no negative reporting-delay anomaly. Biological forecasts
use `charttime <= t` and are explicitly retrospective. All eligible records are
replayed chronologically, then the filter is propagated from the origin.

## Prediction and uncertainty

For analyte observation row `H_i`, propagated state mean and covariance are
`x(t+h)` and `P(t+h)`. The reported point prediction is the inverse transformed
`H_i x(t+h)`. The predictive observation variance is:

`S_i(t+h) = H_i P(t+h) H_iᵀ + R_ii`.

The API reports 90% and 95% predictive intervals derived from this variance,
plus the projected latent standard deviation excluding `R_ii`. Log-space
analytes are inverse transformed at each interval endpoint, yielding correctly
ordered intervals in original units.

## Statuses and restrictions

- `INSUFFICIENT_HISTORY`: no finite origin-eligible update; no numeric forecast.
- `NUMERICAL_UNAVAILABLE`: a transformation or covariance result is non-finite.
- `AVAILABLE`: a model-supported prediction, not a validated clinical forecast.
- `UNAVAILABLE_NO_VALIDATED_ENDPOINT`: deterioration risk is not implemented;
  the processed cohort lacks an independently documented progression outcome.

## Evaluation protocol

The real-data evaluation must use chronological origins and the first future
target in a predeclared ±25% horizon window. It must compare an identical mask
against LOCF and a training-fitted population baseline, select nothing on the
frozen test split, and report original-unit RMSE/MAE, patient counts, coverage,
interval width, and Winkler score by analyte and horizon.
