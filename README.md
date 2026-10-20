# Nephron-Aware Renal Digital Twin

> An irregular-time physiological modeling and machine-learning framework for renal laboratory trajectory representation, latent-state estimation, and short-horizon biomarker forecasting using MIMIC-IV 3.1 clinical observational data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Data](https://img.shields.io/badge/MIMIC--IV-3.1%20Audited-orange.svg)
![Tests](https://img.shields.io/badge/Module%202%2F3%20Tests-47%20Passed-brightgreen.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%2B%20Next.js-0A7EA4.svg)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-blue.svg)

> **Research disclaimer:** This project is a research prototype, not a clinically validated digital twin or clinical decision-support system. Latent states are model-derived statistical scores, not direct measurements of nephron count, glomerular filtration capacity, or renal functional reserve. Forecast performance is mixed and does not consistently outperform simple baselines.

---

## 1. Executive Summary and Research Motivation

Clinical laboratory measurements are inherently irregular. Patients may have several measurements within a single day, followed by long intervals without additional observations. Converting these histories into uniform daily sequences can introduce artificial time steps, discard temporal information, and contaminate historical predictions with information that would not have been available at the time.

The **Nephron-Aware Renal Digital Twin** addresses these challenges through three integrated modules:

- **Module 1 — Data Foundation and Temporal Representation:** Audits MIMIC-IV, extracts a bounded renal laboratory cohort, resolves duplicate and conflicting measurements, and establishes separate biological and operational timelines.
- **Module 2 — Physiology-Informed Latent-State Engine:** Estimates three latent renal-associated signals from irregular laboratory observations using a continuous-discrete Ornstein–Uhlenbeck (OU) state-space model.
- **Module 3 — Patient-Specific Biomarker Forecasting:** Propagates patient-specific latent states to generate 24-, 72-, and 168-hour laboratory forecasts, with model-based prediction intervals.

The system combines an auditable data-processing pipeline, temporal information controls, mathematical state estimation, a forecasting API, and an interactive dashboard.

### 1.1 Problems Addressed

**Irregular temporal sampling**

Daily-grid resampling can expand sparse clinical histories into large numbers of artificial time steps. Multiple measurements collected on the same day can also be collapsed into a single value, hiding changes between measurements.

**Historical information leakage**

Interpolation, retrospective aggregation, and improperly fitted preprocessing transformations can introduce future information into historical estimates. Operational predictions must account for when laboratory results actually became available, not only when specimens were collected.

**Patient-boundary contamination**

Elapsed-time features, rolling statistics, and temporal transformations must be reset correctly at patient boundaries to prevent information from one patient's history from entering another patient's representation.

**Limitations of simple forecasting assumptions**

A continuous-time state-space model provides a structured way to propagate latent states across irregular intervals. However, its mean-reverting dynamics may not capture persistent or nonstationary laboratory trajectories. The project therefore evaluates the model against causal baselines rather than assuming that mathematical complexity guarantees better predictions.

---

## 2. Core Architectural Pillars

### A. Dual-Timeline Information Model

Clinical laboratory records contain two distinct chronological dimensions.

**1. Biological timeline — `charttime`**

- Represents specimen collection time.
- Supports retrospective reconstruction of laboratory trajectories.
- Preserves the timing of observed measurements without requiring a regular daily grid.
- Is explicitly distinguished from information availability in a live workflow.

**2. Operational timeline — `storetime`**

- Represents the time a result became available in the electronic health record.
- Supports state estimation at an explicit query time `t_query`.
- Enforces the eligibility condition: `storetime <= t_query`.

- Excludes results reported after the query time.
- Prevents late-arriving results from retroactively changing what was available to an earlier operational query.

Biological-time reconstruction is retrospective. Operational eligibility is necessary for realistic information-availability constraints, but does not by itself establish readiness for clinical deployment.

### B. Irregular-Time Representation and Leakage Controls

The data-processing pipeline includes:

- Canonical laboratory event representation.
- Unit harmonization and provenance-aware record processing.
- Patient-boundary resets for temporal features.
- Explicit handling of duplicate and conflicting observations.
- Causal query-time filtering for operational inference.
- Comparisons between regular-grid and irregular-observation representations.

The representation preserves observed timestamps rather than requiring a dense daily trajectory. These controls address specific leakage mechanisms; they do not establish that every possible source of bias has been eliminated.

### C. Auditable Conflict Arbitration

Five configurable policies are implemented for handling duplicate and conflicting laboratory records:

- `POLICY_0_RAW` — preserve source records without arbitration.
- `POLICY_A_STORETIME` — resolve conflicts using reporting-time precedence and deterministic tie-breaking.
- `POLICY_B_STAT` — prioritize STAT measurements where applicable, with fallback behavior.
- `POLICY_C_MEAN` — aggregate numeric conflicts using a mean.
- `POLICY_D_EXTREMUM` — select an analyte-specific risk-oriented extremum.

The extremum policy is an experimental data-processing strategy, not a validated clinical decision rule.

### D. Latent-State Estimation and Uncertainty

Module 2 estimates three model-derived latent signals from laboratory observations using an irregular-time state-space model. The filter propagates state estimates and covariance across actual elapsed intervals.

Module 3 reuses this model to generate observable-biomarker forecasts and prediction intervals. Both latent-state and observation-prediction uncertainty are represented, subject to the model's assumptions and calibration limitations.

---

## 3. Repository Architecture

```text
nephron-aware-renal-digital-twin/
├── backend/
│   ├── main.py                       # FastAPI application
│   ├── routes.py                     # API routes and inference endpoints
│   └── schemas.py                    # Typed request/response schemas
├── configs/
│   ├── paths.yaml                    # Local data and project paths
│   ├── cohort_patient_splits.json    # Frozen patient-level splits
│   └── module2_scaler_parameters.json
├── data/
│   ├── raw/                          # Credentialed source data (git-ignored)
│   ├── interim/                      # Intermediate artifacts (git-ignored)
│   └── processed/                    # Bounded cohort (git-ignored)
├── docs/                             # Research protocols and evaluation reports
├── frontend/
│   ├── src/
│   │   ├── app/                      # Application pages
│   │   ├── components/               # Patient, timeline and forecast UI
│   │   ├── services/                 # Backend API client
│   │   └── types/                    # TypeScript API types
│   ├── package.json
│   └── package-lock.json
├── reports/
│   └── generated/                    # Audit and evaluation artifacts
├── src/
│   ├── ingestion/                    # Module 1 dataset audit and extraction
│   ├── temporal/                     # Module 1 schema and temporal processing
│   ├── validation/                   # Module 1 validation harnesses
│   └── engine/
│       ├── state_space.py            # Module 2 continuous-discrete model
│       ├── trajectory_stability.py   # Trajectory Stability Index
│       ├── data_partition.py         # Patient-level partitioning
│       ├── preprocessing.py          # Model preprocessing
│       ├── patient_inference.py      # Patient-specific state inference
│       └── forecasting.py            # Module 3 forecasting engine
├── tests/
├── requirements.txt
└── README.md
```

This is a high-level representation; the checked-out repository is authoritative for the exact files and current implementation.

---

## 4. Module 1 — Data Foundation and Temporal Representation

Module 1 establishes the data and temporal-processing foundation for the modeling pipeline.

### 4.1 Dataset Audit and Bounded Cohort Extraction

- Memory-conscious, chunked audit of the MIMIC-IV 3.1 laboratory-event dataset.
- Deterministic, bounded cohort extraction targeting at most 2,500 eligible patients.
- Source-data quality and laboratory-item inventories.
- Provenance-aware processing of the selected observations.
- Generated audit reports and validation artifacts.

The extracted cohort contains **303,460 source observations before arbitration**. The selection procedure is bounded and deterministic, not designed to produce a population-representative sample.

### 4.2 Core Laboratory Analytes

The current modeling pipeline uses six continuous laboratory variables.

| Analyte | MIMIC-IV item ID | Modeling unit |
|---|---:|---|
| Serum creatinine | `50912` | mg/dL |
| Blood urea nitrogen (BUN) | `51006` | mg/dL |
| Potassium | `50971` | mEq/L |
| Sodium | `50983` | mEq/L |
| Bicarbonate | `50882` | mEq/L |
| Glucose | `50931` | mg/dL |

These measurements provide complementary information about renal-associated laboratory patterns, electrolyte balance, acid-base status, and metabolic context. They do not uniquely determine the underlying physiological mechanisms.

### 4.3 Canonical Events and Arbitration

The temporal pipeline converts laboratory records into a canonical representation containing typed values, timestamps, and relevant provenance information.

The arbitration engine applies the selected conflict policy deterministically and retains the information needed to audit how conflicting records were handled.

### 4.4 Temporal Representation Evaluation

The Module 1 evaluation compares conventional regular-grid representation with irregular-observation approaches.

The audited cohort demonstrated:

- **2,648,183** steps in the daily-grid representation.
- **8.7266×** grid expansion relative to the 303,460 source observations.
- **98.2059%** empty daily-grid steps.
- **43,363** active observations collapsed by the evaluated daily aggregation.
- **103,354** records lost in the audited flawed inner-join control.

These results demonstrate the representational cost of the evaluated daily-grid and join strategies for this cohort. They do not establish that every regular-grid approach will exhibit identical behavior.

The irregular-observation representation retains source observation timestamps and avoids introducing artificial daily steps solely to regularize the timeline.

---

## 5. Module 2 — Physiology-Informed Latent-State Engine

Module 2 estimates latent signals from irregular laboratory observations and exposes query-time inference through the backend.

### 5.1 Three-State Representation

The engine estimates three model-derived latent signals.

| State | Intended interpretation | Scientific qualification |
|---|---|---|
| `x_filt` | Filtration-associated latent score | Not a direct GFR measurement |
| `x_tub` | Electrolyte and acid-base-associated latent score | Not a direct measurement of tubular function |
| `x_work` | Workload/avidity-associated proxy | Not a validated physiological workload measurement |

These states are inferred from laboratory observations through the fitted observation model. They depend on the model structure, parameter estimates, preprocessing, and available measurements.

The implementation does not directly estimate nephron count, single-nephron GFR, or true renal functional reserve.

### 5.2 Continuous-Discrete OU State-Space Model

The model describes latent-state evolution in continuous time. An OU-style formulation is:

```text
dx_t = A(mu - x_t) dt + L dW_t
```

where:

- `x_t` is the latent-state vector.
- `A` controls mean-reverting dynamics.
- `mu` is the model's long-run reference state.
- `L` controls process-noise intensity.
- `dW_t` represents stochastic perturbations.

The discrete filtering steps account for elapsed time between eligible observations instead of assuming that every observation occurs exactly one day after the previous observation.

An observation model connects the latent state to the observed laboratory variables, with observation noise and covariance propagation.

### 5.3 Query-Time Inference

The inference engine reconstructs a patient's latent state from observations eligible at the requested query time. Its outputs include:

- Estimated latent-state values.
- Model-conditional uncertainty.
- Laboratory histories and contributing observations.
- Reconstructed latent-state trajectories.
- Query-time estimates under the selected timeline mode.

A latent-state estimate is a model inference, not a directly measured clinical quantity.

### 5.4 Trajectory Stability Index

The system calculates a separate Trajectory Stability Index (TSI) as an additional empirical summary.

TSI is not a validated clinical stability score and should not be interpreted as a diagnosis, prognosis, or direct measure of renal functional reserve.

### 5.5 Module 2 Evaluation

The held-out evaluation compared the OU model against causal Last Observation Carried Forward (LOCF) and a linear-interpolation baseline.

The OU model did not outperform LOCF across all six analytes. In particular, creatinine and BUN prediction remained challenging, and prediction-interval coverage varied across analytes.

Module 2 is therefore presented as an irregular-time latent-state estimation engine, not as a demonstrated replacement for simpler laboratory forecasting methods.

---

## 6. Module 3 — Patient-Specific Biomarker Forecasting

Module 3 extends the existing state-space engine to generate forecasts of observable laboratory measurements.

### 6.1 Forecasting Pipeline

For a selected patient and forecast origin, the engine:

1. Determines which records are eligible at the origin under the selected timeline mode.
2. Reconstructs the latent state using the existing Module 2 inference procedure.
3. Propagates the fitted OU state and covariance to the requested forecast horizons.
4. Projects the propagated state into observable laboratory space.
5. Calculates point forecasts and observation-prediction intervals.
6. Returns explicit availability or initialization statuses when a forecast cannot be produced reliably.

### 6.2 Supported Horizons

| Horizon | Target time |
|---|---|
| 24 hours | Forecast origin + 24 hours |
| 72 hours | Forecast origin + 72 hours |
| 168 hours | Forecast origin + 168 hours (7 days) |

### 6.3 Forecast Outputs

For each requested analyte and horizon, the forecasting engine provides:

- Forecast origin and target timestamp.
- Point prediction in the original laboratory unit.
- 90% observation-prediction interval.
- 95% observation-prediction interval.
- Relevant eligibility and initialization statuses.
- Latent-state uncertainty and observation-prediction uncertainty.
- Explicit deterioration-risk availability status.

Prediction intervals are conditional on the fitted model and its assumptions. They do not guarantee that future measurements will fall within the reported bounds.

### 6.4 Biological and Operational Forecasting

**Operational mode**

Uses records available by the forecast origin according to reporting-time eligibility. This is the relevant mode for evaluating whether a forecast could have been produced using information available at that time.

**Biological mode**

Uses specimens collected by the biological forecast origin. This mode supports retrospective analysis and must not be treated as equivalent to a real-time operational forecast.

### 6.5 Deterioration-Risk Availability

The current interface reports:

`UNAVAILABLE_NO_VALIDATED_ENDPOINT`

An independently verified CKD progression or deterioration endpoint is not available in the current processed interface. Consequently, the system does not claim to provide validated CKD progression probabilities, AKI risk estimates, or future clinical deterioration probabilities.

### 6.6 Forecast Evaluation Design

The forecasting evaluation uses a frozen patient-level test split, common eligible origins and targets, and comparable baseline predictions.

The OU model's performance is evaluated against causal LOCF using RMSE. The evaluation does not assume that the proposed model is superior, and the frozen test set must not be used for iterative model tuning.

The results below show that forecast performance depends on both the analyte and the horizon.

---

## 7. Quantitative Evaluation Results

### 7.1 Module 2 — Held-Out State-Space Benchmark

Lower RMSE indicates lower prediction error.

| Analyte | OU RMSE | LOCF RMSE | Better RMSE |
|---|---:|---:|---|
| Bicarbonate | 3.9719 | 2.6474 | LOCF |
| Creatinine | 0.7419 | 0.3643 | LOCF |
| Glucose | 59.5207 | 63.6853 | OU |
| Potassium | 0.5385 | 0.5844 | OU |
| Sodium | 2.9420 | 2.9021 | LOCF |
| BUN | 12.5725 | 6.5198 | LOCF |

The OU model outperformed LOCF for glucose and potassium but performed worse for the other four analytes in this evaluation.

Prediction-interval coverage was inadequate for some analytes, particularly bicarbonate and glucose. Creatinine and BUN forecasting also showed limitations that are important when interpreting the model's behavior.

### 7.2 Module 3 — Forecast RMSE by Horizon

Values are RMSE in each analyte's original unit. Lower is better.

| Analyte | Horizon | OU RMSE | LOCF RMSE | Better RMSE |
|---|---:|---:|---:|---|
| Creatinine | 24 h | 0.541 | 0.298 | LOCF |
| Creatinine | 72 h | 0.604 | 0.342 | LOCF |
| Creatinine | 168 h | 0.741 | 0.568 | LOCF |
| BUN | 24 h | 14.276 | 4.596 | LOCF |
| BUN | 72 h | 20.856 | 9.372 | LOCF |
| BUN | 168 h | 11.777 | 11.830 | OU |
| Potassium | 24 h | 0.511 | 0.548 | OU |
| Potassium | 72 h | 0.582 | 0.726 | OU |
| Potassium | 168 h | 0.516 | 0.642 | OU |
| Sodium | 24 h | 2.838 | 2.441 | LOCF |
| Sodium | 72 h | 3.491 | 3.682 | OU |
| Sodium | 168 h | 4.286 | 4.051 | LOCF |
| Bicarbonate | 24 h | 3.661 | 2.601 | LOCF |
| Bicarbonate | 72 h | 3.816 | 3.499 | LOCF |
| Bicarbonate | 168 h | 4.172 | 4.365 | OU |
| Glucose | 24 h | 43.731 | 36.747 | LOCF |
| Glucose | 72 h | 38.330 | 41.068 | OU |
| Glucose | 168 h | 39.178 | 48.499 | OU |

### 7.3 Interpretation

The evaluation demonstrates that the OU model does not consistently outperform LOCF. Creatinine forecasts have higher RMSE than LOCF at all three horizons. BUN also performs substantially worse at 24 and 72 hours, while potassium shows lower OU RMSE across all three horizons in this particular evaluation.

These results support a measured conclusion: the system implements irregular-time state estimation and patient-specific forecasting, but overall predictive superiority and clinical utility have not been established.

A plausible-looking forecast is not necessarily accurate, well-calibrated, or clinically meaningful.

---

## 8. Technology Stack

### Backend and Modeling

- Python
- NumPy and pandas
- SciPy
- FastAPI and Pydantic
- Uvicorn
- Pytest
- PyYAML and other dependencies listed in `requirements.txt`

### Frontend

- Next.js
- React
- TypeScript
- CSS and the frontend tooling configured in `frontend/package.json`

Use the versions recorded in `frontend/package.json` and `frontend/package-lock.json` as the source of truth for frontend dependencies.

### Data

- MIMIC-IV 3.1, subject to PhysioNet access requirements and its applicable data-use agreement.

---

## 9. Setup and Execution

### 9.1 Prerequisites

- Python 3.10 or newer, compatible with the pinned project dependencies.
- Node.js and npm compatible with the frontend lockfile.
- Authorized MIMIC-IV access to regenerate cohort and audit artifacts.

Use native Windows Python and Node.js when running on Windows. Avoid using a WSL-to-Windows Node runtime bridge.

### 9.2 Python Environment

From the repository root, in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If virtual-environment activation is unavailable, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Review `configs/paths.yaml` before running data extraction or other workflows that require local dataset artifacts.

### 9.3 Frontend Dependencies

From the repository root:

```powershell
Set-Location frontend
npm ci
```

`npm ci` installs dependencies from the committed lockfile. Do not delete or regenerate the lockfile merely to start the application.

### 9.4 Local Environment Configuration

If required by the current frontend implementation, create `frontend/.env.local`:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Keep `.env.local` out of version control. Do not put credentials or private dataset paths in frontend environment variables.

### 9.5 Start the Backend

From the repository root, in the first PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

For development auto-reload, append `--reload`.

Check the health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The backend requires the configured data artifacts and model parameters to be available.

### 9.6 Start the Frontend

In a second PowerShell window:

```powershell
Set-Location frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), select a patient, and explore the estimated states, patient timeline, and Forecast view.

The backend API documentation may be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), depending on the application's configuration.

### 9.7 Production Frontend Build

From the `frontend` directory:

```powershell
npm run build
npm run start
```

The production frontend requires the backend to be available at the configured API URL. Review authentication, CORS, and deployment settings before exposing any server publicly.

---

## 10. API Reference

The following endpoints are implemented in the application. Request and response details are defined by `backend/routes.py` and `backend/schemas.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `GET` | `/api/v1/cohort/summary` | Bounded-cohort summary and relevant metadata |
| `GET` | `/api/v1/patients` | Paginated patient listing and supported filters |
| `GET` | `/api/v1/patients/{subject_id}/trajectory` | Patient laboratory trajectory |
| `GET` | `/api/v1/patients/{subject_id}/latent-trajectory` | Reconstructed latent-state trajectory |
| `POST` | `/api/v1/state/estimate` | Query-time Module 2 state estimation |
| `POST` | `/api/v1/forecast` | Module 3 observable-biomarker forecasting |

### Forecast Request

The forecast endpoint accepts a patient identifier, forecast origin, supported horizons, timeline mode, and an optional analyte subset.

Supported horizons are 24, 72, and 168 hours. The response includes point predictions, prediction intervals, units, target times, and relevant availability or initialization statuses.

The current Pydantic schemas and API implementation are authoritative for the precise request and response fields.

---

## 11. Tests and Verification

### 11.1 Module 2 and Module 3 Regression Tests

Run the focused regression suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_module2_units.py tests/test_module3_forecasting.py -q
```

The latest documented focused regression run reported **47 passing tests in 42.06 seconds**.

### 11.2 Full Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The focused Module 2/3 result does not establish that every test in the repository passes. Run the full suite in an environment with the necessary dependencies, data, and configuration.

### 11.3 Frontend Build

From the `frontend` directory:

```powershell
npm run build
```

A successful Windows PowerShell production build was reported during development. Re-run the build after subsequent frontend changes to verify the current checkout.

### 11.4 Verification Boundaries

Unit-test success, frontend build success, backend HTTP integration, and browser behavior are separate forms of evidence.

Backend HTTP integration testing was inconclusive in the restricted development sandbox. Local endpoint responses and browser behavior were reported during Windows development, while some edge cases remain unverified.

Do not interpret an environment-blocked or unexecuted integration test as a passing test.

---

## 12. Data Privacy and Governance

MIMIC-IV is restricted-access clinical data distributed through PhysioNet.

- Complete the required credentialing and training and comply with the applicable Data Use Agreement.
- Do not publish raw or patient-level MIMIC-IV records.
- Keep raw, interim, and processed patient-level files out of Git.
- Preserve the read-only status of raw source data.
- Do not commit credentials, `.env.local`, virtual environments, build output, or local caches.
- Confirm that sharing aggregate outputs is permitted before publishing them.
- Preserve the relevant cohort configuration and provenance for reproducibility.

The cohort and model results are conditional on the bounded cohort, selected laboratory variables, preprocessing rules, and model configuration used in the repository.

---

## 13. Scientific Limitations

### Data Limitations

- The 2,500-patient cohort is bounded and is not established as representative of the wider patient population.
- The core model uses six laboratory analytes.
- The processed interface does not provide an independently validated CKD progression or deterioration endpoint.
- Medication exposure, blood pressure, albuminuria, cystatin C, and adjudicated outcomes are not comprehensively incorporated into the current modeling interface.

### Modeling Limitations

- Latent states are empirical model-derived signals, not direct physiological measurements.
- OU mean-reverting dynamics may be unsuitable for persistent or nonstationary laboratory trajectories.
- Forecast accuracy varies by analyte and horizon.
- The OU model does not consistently outperform causal LOCF.
- Prediction-interval calibration varies across analytes.
- Model-conditional uncertainty does not capture every source of clinical, measurement, and distributional uncertainty.

### Clinical Limitations

- Forecasts are not diagnoses.
- Deterioration-risk estimation is unavailable without a validated endpoint.
- Causal treatment effects are not estimated.
- Nephron count, true renal functional reserve, and single-nephron GFR are not directly measured or validated.
- Clinical utility, prospective safety, and external clinical validity have not been established.

These limitations are part of the research findings and must be considered when interpreting model outputs.

---

## 14. Future Research

Potential future directions, subject to data availability and a prespecified evaluation design, include:

1. **Improved forecasting baselines:** Evaluate alternative models against the same patient splits, forecast origins, target definitions, and evaluation masks.
2. **Uncertainty calibration:** Define calibration criteria on validation data and evaluate interval coverage on the frozen test set.
3. **Observation-process modeling:** Investigate how irregular sampling and missingness affect renal laboratory trajectory estimation.
4. **Outcome-defined deterioration modeling:** Establish a defensible endpoint with explicit follow-up, censoring, and leakage controls before developing a risk model.
5. **External validation:** Evaluate generalization on an appropriate independent cohort before making broader claims.
6. **Scenario analysis:** Investigate counterfactual or intervention simulation only after its causal assumptions and evaluation requirements are established.

These are proposed research directions, not completed functionality. Future model selection must not use the frozen test set for iterative tuning.

---

## 15. Project Summary

The Nephron-Aware Renal Digital Twin integrates an auditable irregular-time data pipeline, a continuous-discrete latent-state model, patient-specific biomarker forecasting, and an interactive interface.

Its primary contribution is the implementation and evaluation of a complete research workflow that explicitly addresses temporal representation, information availability, state estimation, and forecast uncertainty.

The empirical results show both the strengths and limitations of the approach: irregular-time state estimation and forecasting are implemented, but the OU model does not consistently outperform causal baselines, and clinical validity has not been established.

**Project:** Nephron-Aware Renal Digital Twin  
**Research focus:** Irregular-time renal laboratory representation, latent-state estimation, and patient-specific biomarker forecasting  
**Data source:** MIMIC-IV 3.1  
**Status:** Research prototype; clinical validation not established

---

## 16. Citation

When using this project in academic work, cite relevant methods and data-source publications and follow the citation requirements specified by PhysioNet for MIMIC-IV.
