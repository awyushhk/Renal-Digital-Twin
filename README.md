# Nephron-Aware Renal Digital Twin

Research prototype for auditable, irregular-time analysis of renal laboratory trajectories from MIMIC-IV. It combines a bounded cohort and dual-timeline preprocessing pipeline (Module 1), a three-state latent renal-signal engine (Module 2), and observable-biomarker forecasting (Module 3).

> This is a research system, not a clinical device. MIMIC-IV timestamps are de-identified hospital wall-clock dates; they are not real calendar dates and must not be used for patient care.

## Objectives and scope

The project studies how to represent irregular laboratory streams without daily-grid expansion, cross-patient leakage, or operational lookahead. Current scope: a 2,500-patient MIMIC-IV cohort with six analytes: creatinine (mg/dL), urea nitrogen/BUN (mg/dL), potassium (mEq/L), sodium (mEq/L), bicarbonate (mEq/L), and glucose (mg/dL).

The system does not directly measure nephron count or true renal reserve, and does not provide a diagnosis, treatment-effect estimate, or validated deterioration-risk endpoint.

## Implemented modules

### Module 1 — data foundation and temporal representation

- Read-only MIMIC-IV audit and deterministic bounded cohort extraction.
- 303,460 laboratory observations, frozen patient-level splits (1,750/375/375), and training-only scalers.
- Unit harmonisation, canonical events, duplicate/conflict arbitration, and separate biological (charttime) and operational (storetime) information sets.
- The historical Module 1 end-to-end acceptance report is reports/generated/m1_4_end_to_end_validation_report.md.

### Module 2 — latent-state engine

- FastAPI endpoints for patient browsing, trajectories, latent trajectories, and query-time state estimation.
- Continuous-discrete three-state OU model with model-inferred filtration-associated, tubular/electrolyte/acid-base-associated, and workload/avidity-associated proxies.
- These latent states are empirical model outputs, not direct physiological measurements.

### Module 3 — biomarker forecasting

- Causal forecasts of observable biomarkers at 24, 72, and 168 hours.
- Operational forecasts use records reported by the origin; biological mode uses samples collected by the origin and is retrospective.
- Original-unit point forecasts with 90% and 95% observation-prediction intervals, explicit INSUFFICIENT_HISTORY and NUMERICAL_UNAVAILABLE statuses, and a Forecast dashboard for the selected patient.
- Deterioration-risk status remains UNAVAILABLE_NO_VALIDATED_ENDPOINT.

## Architecture and data flow

~~~text
Read-only MIMIC-IV CSVs
  -> Module 1 audit, bounded cohort, canonicalisation, and arbitration
  -> frozen splits + training-only scalers + fitted Module 2 parameters
  -> PatientInferenceService (operational or biological eligibility)
  -> FastAPI endpoints
  -> Next.js patient dashboard and Forecast panel
~~~

The forecast engine replays origin-eligible records through the existing Module 2 filter, propagates the fitted OU state and covariance to a supported horizon, and projects to each observable biomarker. It does not create a dense imputed trajectory.

## Technology stack

- Python, pandas, NumPy, SciPy, PyArrow, PyYAML, FastAPI, Pydantic, Uvicorn, HTTPX, and pytest (requirements.txt).
- Next.js 16.4.2, React 19.3.0, TypeScript, ESLint, and Tailwind CSS tooling (frontend/package.json).
- Credentialed MIMIC-IV 3.1 data, subject to the applicable PhysioNet data-use agreement.

## Repository layout

~~~text
backend/                    FastAPI application, routes, and schemas
configs/                    paths, frozen splits, scalers, and fitted parameters
data/raw/                   read-only MIMIC-IV source files (ignored)
data/interim/               temporary extraction files (ignored)
data/processed/             bounded cohort files (ignored)
docs/                       Module 1 protocols and Module 3 documentation
frontend/                   Next.js dashboard, API client, and components
reports/generated/          generated audit and evaluation artifacts
src/engine/                 Module 2 inference and Module 3 forecasting
src/ingestion/, temporal/   Module 1 extraction and temporal representation
src/validation/             Module 1 validation harnesses
tests/                      Module 1–3 tests and fixtures
~~~

## Prerequisites and Windows PowerShell setup

Use supported native Windows Python and Node.js installations. Do not use a WSL-to-Windows Node bridge. Put credentialed MIMIC-IV files in data/raw only; they must remain read-only and must never be committed.

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Set-Location frontend
npm ci
Set-Location ..
~~~

npm ci uses the committed frontend/package-lock.json; do not delete or regenerate the lockfile merely to start the project.

## Configuration

Create frontend/.env.local locally, without committing it:

~~~text
NEXT_PUBLIC_API_URL=http://localhost:8000
~~~

This is the browser-visible API base URL and must not have a trailing slash. Do not place credentials or MIMIC-IV paths in it. Data paths and cohort item IDs are in configs/paths.yaml.

## Run the application

Start the backend from the repository root:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
~~~

In a second PowerShell window:

~~~powershell
Set-Location frontend
npm run dev
~~~

Open http://localhost:3000, select a patient, then choose Forecast. For the production build/server:

~~~powershell
Set-Location frontend
npm run build
npm run start
~~~

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | /health | service health and version |
| GET | /api/v1/cohort/summary | bounded-cohort summary and scaler metadata |
| GET | /api/v1/patients | paginated patient list; paging, admission filter, and search |
| GET | /api/v1/patients/{subject_id}/trajectory | effective laboratory trajectory |
| GET | /api/v1/patients/{subject_id}/latent-trajectory | replayed latent-state points |
| POST | /api/v1/state/estimate | query-time Module 2 state estimate |
| POST | /api/v1/forecast | Module 3 causal observable-biomarker forecasts |

POST /api/v1/forecast accepts subject_id, forecast_origin, supported horizons_hours (24, 72, and/or 168), timeline_mode, and an optional analyte subset. It returns target time, a point prediction, 90% and 95% prediction intervals, units, eligibility/initialisation status, and the unavailable deterioration-risk status. Unknown patients return 404; unsupported horizons and invalid forecast requests return 400.

## Tests and build checks

Focused current regression command executed in the project’s WSL environment:

~~~bash
./.venv-wsl/bin/python -m pytest tests/test_module2_units.py tests/test_module3_forecasting.py -q
~~~

Latest recorded result: **47 passed in 42.06s**. The requested Module 3/backend source files also passed py_compile there.

The Windows frontend commands are defined by frontend/package.json:

~~~powershell
Set-Location frontend
npm run build
npm run lint
~~~

The latest Windows PowerShell verification reported a successful production build. This constrained WSL environment cannot execute the native Windows Node runtime, so its failed bridge invocation is not evidence of a Windows build failure. See docs/module3_integration_report.md for verification provenance and remaining gaps.

## Data provenance and reproducibility

MIMIC-IV is restricted-access clinical data. Raw, interim, and processed patient-level files are excluded from Git and must not be shared. Researchers with the required PhysioNet access can use the configured data directories and Module 1 workflows to reproduce the bounded-cohort artifacts. Do not alter frozen splits, fitted parameters, or reports to improve Module 3 results.

## Scientific evaluation summary

The frozen-test evaluation is in docs/module3_scientific_evaluation.md and reports/generated/module3_forecasting_evaluation.json.

- The OU forecast did **not** outperform causal LOCF for the primary creatinine target at 24, 72, or 168 hours.
- Prediction-interval calibration varies by analyte and horizon: some intervals under-cover, while creatinine and BUN intervals are generally too wide.
- Secondary-analyte improvements at some horizons do not establish primary-target superiority or clinical utility.
- Forecasts are research predictions of laboratory observations, not diagnoses, treatment-effect estimates, or validated deterioration-risk predictions.

## Limitations and future research

- No independently validated CKD progression/deterioration endpoint is available in the current processed interface.
- The model uses six labs plus limited demographics/admission metadata; it lacks medication exposure, blood pressure, albuminuria, cystatin C, and adjudicated outcomes.
- Biological-time mode is retrospective and is not a real-time decision-support input.
- Read endpoint HTTP and browser verification status from the latest integration report; do not infer it from source inspection.

Future work needs a prespecified endpoint, leakage-resistant follow-up and censoring design, validation-only model development, and external or outcome-bearing validation data. It must not tune the frozen test split.

## Further documentation

- docs/m1_4_end_to_end_integration_protocol.md
- docs/module3_implementation_plan.md
- docs/module3_methodology.md
- docs/module3_scientific_evaluation.md
- docs/module3_integration_report.md
