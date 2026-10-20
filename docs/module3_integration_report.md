# Module 3 integration report

## Scope and contract

POST /api/v1/forecast uses the existing cached Module 2 inference service. It accepts a subject ID, a de-identified hospital-wall-clock forecast origin, operational or biological eligibility semantics, supported horizons (24, 72, and 168 hours), and an optional analyte subset. It returns only predictions based on origin-eligible records, including original-unit 90% and 95% observation-prediction intervals and initialisation status.

Forecasts are research predictions of observable biomarkers. They are not measurements of nephron count or renal reserve, diagnoses, treatment-effect estimates, or a validated deterioration-risk model. Deterioration risk remains UNAVAILABLE_NO_VALIDATED_ENDPOINT.

## Evidence categories

This record distinguishes direct local checks from Windows/browser outcomes supplied by the project operator. Source inspection is not substituted for HTTP or browser verification.

### Local source and regression checks

- The current page imports ForecastPanel, exposes a dedicated Forecast control for a selected patient, and renders ForecastPanel when the active view is forecast.
- ForecastPanel sends the selected patient ID, current query origin, requested horizon, and selected timeline mode through apiClient.forecast(), which posts to /api/v1/forecast. No forecast values are hardcoded.
- Executed locally: ./.venv-wsl/bin/python -m pytest tests/test_module2_units.py tests/test_module3_forecasting.py -q — 47 passed in 42.06s.
- Executed locally: ./.venv-wsl/bin/python -m py_compile src/engine/forecast_evaluation.py src/engine/forecasting.py backend/routes.py backend/schemas.py — passed with no output.
- Earlier direct production-path diagnostic in this environment: PatientInferenceService preload read 2,500 patients and 303,460 labs in 2.38 seconds; a 24-hour operational forecast for subject 10000032 completed in 0.32 seconds with six forecasts and first-item status AVAILABLE.

### Windows PowerShell and browser verification reported by the project operator

The following outcomes were reported from the Windows environment during finalisation. No command transcript or response body was captured by this constrained WSL session.

- The FastAPI backend started successfully in Windows PowerShell.
- Successful responses were observed for GET /health, GET /api/v1/patients, and GET /api/v1/cohort/summary.
- The Windows Next.js production build, run as npm run build from frontend, completed successfully.
- The Forecast dashboard is visible in the browser after selecting a patient.
- Forecasts were displayed for the tested patient at 24-hour, 72-hour, and 168-hour horizons.

These observations verify the reported happy path only. They do not establish all API error paths, interval calibration, or clinical validity.

## Remaining unverified or constrained checks

- The local focused backend HTTP test command, ./.venv-wsl/bin/python -m pytest tests/test_backend_integration.py::TestForecastEndpoint -q -vv, did not complete in this sandbox. The TestClient lifespan path stalled and local loopback socket creation is blocked by the sandbox. It is not counted as passed here.
- The browser was not automated or independently inspected by this execution environment. The dashboard visibility and horizon display above are explicitly user-reported Windows/browser outcomes.
- The no-history, numerical-unavailable, invalid-request, and unsupported-horizon paths remain covered by focused/source-level tests where applicable, but they have not all been re-verified through the final Windows browser session.
- This WSL environment cannot execute the native Windows Node runtime; that limitation does not negate the reported successful Windows production build.

## Scientific limitations

The frozen-test evaluation found that the OU model did not outperform causal LOCF for the primary creatinine target at 24, 72, or 168 hours. Prediction-interval calibration varies by analyte and horizon; creatinine and BUN intervals are generally too wide, while several other analyte/horizon combinations under-cover. These outputs are not clinically calibrated risk intervals.

See docs/module3_methodology.md and docs/module3_scientific_evaluation.md for semantics and the complete frozen-test evaluation.

