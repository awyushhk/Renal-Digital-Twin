# Module 3 scientific evaluation status

## Implemented and tested

The causal forecast engine and typed API contract are implemented. Synthetic
tests cover operational versus biological eligibility, no eligible history,
supported horizons, log-scale inverse transformation, ordered intervals, and
the distinction between latent and observation predictive uncertainty. The
real-cohort audit evaluates the same existing Module 3 forecast engine; it does
not alter Module 2 parameters, transforms, partitions, or source data.

## Evaluation protocol

The operational-time evaluation used the frozen 375-patient validation split to
retain the prespecified 24, 72, and 168-hour horizons (each had more than 100
creatinine targets and 25 patients), then evaluated that unchanged set once on
the frozen 375-patient test split. For every patient and horizon, it uses one
latest shared forecast origin and the first future target per analyte in the
fixed `[0.75h, 1.25h]` window. This limits repeated measurements from a single
patient from dominating pooled metrics.

Operational history and targets are keyed on result availability (`storetime`);
biological feasibility is reported separately using collection time
(`charttime`). The common comparison mask contains a finite target plus finite
OU, causal LOCF, and training-fitted population-mean predictions.

## Evaluation-correctness audit (2026-10-20)

The audit inspected `src/engine/forecast_evaluation.py`,
`src/engine/forecasting.py`, the frozen split loader, the generated JSON, and
focused regression tests. The evaluator chooses the latest shared origin per
patient/horizon only when a future target exists, then gives every comparator
the same target, origin, and finite common mask. The observed future target is
used only to determine retrospective evaluation eligibility; it never enters
the filter, OU forecast, LOCF history, or population baseline.

Operational forecasts admit results filed by (`storetime <= origin`) and mask
negative reporting delays; biological feasibility uses `charttime <= origin`.
Patient IDs are loaded from the frozen disjoint 1,750/375/375 split; the
population baseline is the training-fitted scaler mean. Metrics are calculated
in original units after inverse transformation. Predictive coverage uses the
observation predictive interval (projected latent variance plus observation
noise), not latent-state uncertainty alone. The 95% Winkler score uses the
prespecified alpha=0.05 penalty. The fixed target window `[0.75h, 1.25h]`
matches this module's documented scientific objective and now has explicit
boundary regression coverage.

No demonstrated evaluation defect required a scientific-result correction.
The extant report records the frozen test result; a subsequent report-command
attempt in this constrained session did not confirm a completed rewrite, so it
is not represented as an independent reproduction.

## Follow-up feasibility

All horizons have substantial cohort support. Under operational timing, the
number of patients with a latest evaluable pair ranges from 1,810–1,858 at 24 h,
1,548–1,603 at 72 h, and 1,118–1,246 at 168 h across the six analytes. Biological
timing is nearly identical (1,810–1,849; 1,547–1,605; 1,115–1,237). Exact
counts, observation, gap, and follow-up distributions are in
`reports/generated/module3_followup_feasibility.json`.

## Frozen test results

All values below are original units. Each cell is `OU RMSE / LOCF RMSE`, followed
by OU 90%/95% predictive-interval coverage.

| Analyte | 24 h | 72 h | 168 h |
|---|---|---|---|
| Bicarbonate | 3.661 / 2.601; 71.2%/78.0% | 3.816 / 3.499; 77.6%/84.8% | 4.172 / 4.365; 79.0%/85.2% |
| Creatinine | 0.541 / 0.298; 97.3%/98.9% | 0.604 / 0.342; 96.9%/98.7% | 0.741 / 0.568; 92.1%/95.5% |
| Glucose | 43.731 / 36.747; 49.2%/56.9% | 38.330 / 41.068; 53.5%/64.5% | 39.178 / 48.499; 62.8%/73.2% |
| Potassium | 0.511 / 0.548; 93.4%/96.5% | 0.582 / 0.726; 96.5%/97.8% | 0.516 / 0.642; 95.3%/98.8% |
| Sodium | 2.838 / 2.441; 68.4%/77.9% | 3.491 / 3.682; 73.2%/80.4% | 4.286 / 4.051; 69.9%/77.7% |
| BUN | 14.276 / 4.596; 96.1%/96.9% | 20.856 / 9.372; 96.0%/99.1% | 11.777 / 11.830; 98.8%/100.0% |

Target counts range from 153–260 unique patients per analyte/horizon. The
machine-readable evaluation report includes MAE, RMSE versus both baselines,
unique patient count, 95% width, Winkler score, patient-level median absolute
error, and exclusions for every cell.

## Acceptance outcome

**FAILED:** the prespecified primary gate requires lower creatinine RMSE than
LOCF at at least one validation-supported horizon. OU is worse than LOCF at all
three horizons (ratios 1.815, 1.769, and 1.304). It performs better than LOCF
for potassium at all three horizons, glucose at 72/168 h, bicarbonate at 168 h,
and sodium at 72 h, but this does not establish primary-target superiority.

Calibration is also limited: creatinine/BUN intervals are generally too wide,
while bicarbonate, glucose, and sodium under-cover at several horizons. These
are empirical predictive intervals, not clinically calibrated risk intervals.
Scenario simulation remains blocked.

## Deterioration-risk feasibility

The current bounded processed interface contains six labs plus limited
demographics/admission metadata, not a verified CKD progression or outcome
label. Deterioration risk is deliberately returned as
`UNAVAILABLE_NO_VALIDATED_ENDPOINT`. A future laboratory-defined proxy would
need a separate prespecified endpoint, baseline, follow-up/censoring audit,
and patient-level validation before exposure.

## Required next evaluation

No further test-set tuning is justified. The scientifically justified next step
is validation-only work on a different, explicitly prespecified forecast model
or an external outcome-bearing dataset; neither is part of the current scope.
