"""Causal, uncertainty-aware Module 3 observable-biomarker forecasting.

This module deliberately reuses the fitted Module 2 filter.  It makes an
observation forecast at a requested future horizon; it does not impute a dense
daily trajectory, infer a clinical diagnosis, or estimate a causal effect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.engine.patient_inference import PatientInferenceService
from src.engine.preprocessing import unstandardize_observation, standardize_observation
from src.engine.state_space import ANALYTE_INDEX_MAP, ContinuousDiscreteRenalFilter


SUPPORTED_HORIZONS_HOURS = (24, 72, 168)
Z_90 = 1.6448536269514722
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class BiomarkerForecast:
    analyte_name: str
    unit: str
    horizon_hours: int
    target_time: str
    predicted_value: Optional[float]
    predictive_interval_90: Optional[tuple[float, float]]
    predictive_interval_95: Optional[tuple[float, float]]
    latent_projected_sd_standardized: Optional[float]
    predictive_sd_standardized: Optional[float]
    status: str


@dataclass(frozen=True)
class ForecastResult:
    subject_id: int
    forecast_origin: str
    timeline_mode: str
    eligibility_semantics: str
    eligible_observation_count: int
    initialization_status: str
    forecasts: List[BiomarkerForecast]
    model_identifier: str = "module2_ou_observation_forecast"
    baseline_identifiers: tuple[str, str] = ("locf", "training_population_mean")
    deterioration_risk_status: str = "UNAVAILABLE_NO_VALIDATED_ENDPOINT"
    caveat: str = (
        "Research forecast only. It is an observable-biomarker prediction conditional on the fitted "
        "Module 2 model, not a clinical diagnosis, nephron-count measurement, or treatment-effect estimate."
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            **asdict(self),
            "forecasts": [asdict(item) for item in self.forecasts],
        }


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _eligible_records(
    service: PatientInferenceService,
    subject_id: int,
    origin: pd.Timestamp,
    timeline_mode: str,
) -> pd.DataFrame:
    """Apply the same causal eligibility rules as Module 2 inference."""
    records = service.get_patient_records(subject_id)
    mode = timeline_mode.strip().upper()
    if mode == "OPERATIONAL":
        mask = (
            records["storetime"].notna()
            & (records["storetime"] <= origin)
            & (records["delay_anomaly_flag"] != "NEGATIVE_REPORTING_DELAY")
        )
    elif mode == "BIOLOGICAL":
        mask = records["charttime"].notna() & (records["charttime"] <= origin)
    else:
        raise ValueError("timeline_mode must be OPERATIONAL or BIOLOGICAL")
    return records.loc[mask].sort_values(["charttime", "storetime"], kind="mergesort")


def forecast_patient_biomarkers(
    service: PatientInferenceService,
    subject_id: int,
    forecast_origin: str | pd.Timestamp,
    horizons_hours: Iterable[int] = SUPPORTED_HORIZONS_HOURS,
    timeline_mode: str = "OPERATIONAL",
    analytes: Optional[Iterable[str]] = None,
) -> ForecastResult:
    """Forecast supported observable biomarkers using only origin-eligible history.

    The filter is reconstructed from effective, causally eligible events, then
    propagated from the query origin separately for each requested horizon.
    """
    origin = pd.to_datetime(forecast_origin)
    if getattr(origin, "tz", None) is not None:
        origin = origin.tz_localize(None)
    horizon_values = tuple(sorted({int(hours) for hours in horizons_hours}))
    if not horizon_values or any(hours not in SUPPORTED_HORIZONS_HOURS for hours in horizon_values):
        raise ValueError(f"Unsupported horizon; supported values are {SUPPORTED_HORIZONS_HOURS} hours")
    selected = tuple(analytes) if analytes is not None else tuple(ANALYTE_INDEX_MAP)
    unknown = set(selected) - set(ANALYTE_INDEX_MAP)
    if unknown:
        raise ValueError(f"Unsupported analytes: {sorted(unknown)}")

    records = _eligible_records(service, subject_id, origin, timeline_mode)
    filter_ = ContinuousDiscreteRenalFilter(params=service.params)
    last_time: Optional[pd.Timestamp] = None
    eligible_count = 0

    for charttime, epoch in records.groupby("charttime", sort=True):
        timestamp = pd.Timestamp(charttime)
        dt = 0.0 if last_time is None else max(0.0, (timestamp - last_time).total_seconds() / 3600.0)
        filter_.predict(dt)
        observations: Dict[int, float] = {}
        for _, row in epoch.iterrows():
            analyte = str(row.get("analyte_name", ""))
            value = row.get("valuenum_effective")
            if analyte not in ANALYTE_INDEX_MAP or not _finite(value):
                continue
            standardized = standardize_observation(float(value), analyte, service.scaler_config)
            if _finite(standardized):
                observations[ANALYTE_INDEX_MAP[analyte]] = float(standardized)
                eligible_count += 1
        if observations:
            filter_.update(observations)
        last_time = timestamp

    semantics = (
        "operational: results reported by the forecast origin only"
        if timeline_mode.strip().upper() == "OPERATIONAL"
        else "biological: samples collected by the forecast origin only (retrospective)"
    )
    if eligible_count == 0:
        return ForecastResult(
            subject_id=subject_id,
            forecast_origin=origin.isoformat(),
            timeline_mode=timeline_mode.strip().upper(),
            eligibility_semantics=semantics,
            eligible_observation_count=0,
            initialization_status="INSUFFICIENT_HISTORY",
            forecasts=[
                BiomarkerForecast(a, service.scaler_config.analytes[a].unit, h, (origin + pd.Timedelta(hours=h)).isoformat(), None, None, None, None, None, "INSUFFICIENT_HISTORY")
                for h in horizon_values for a in selected
            ],
        )

    if last_time is not None:
        filter_.predict(max(0.0, (origin - last_time).total_seconds() / 3600.0))
    origin_state, origin_cov = filter_.state.copy(), filter_.cov.copy()
    H, R = service.params.get_full_h_matrix(), service.params.get_full_r_matrix()
    forecasts: List[BiomarkerForecast] = []
    for horizon in horizon_values:
        projected = ContinuousDiscreteRenalFilter(service.params, origin_state, origin_cov)
        projected.predict(float(horizon))
        for analyte in selected:
            index = ANALYTE_INDEX_MAP[analyte]
            row = H[index, :]
            mean_std = float(row @ projected.state)
            latent_variance = max(0.0, float(row @ projected.cov @ row.T))
            predictive_variance = max(0.0, latent_variance + float(R[index, index]))
            latent_sd, predictive_sd = np.sqrt(latent_variance), np.sqrt(predictive_variance)
            point = unstandardize_observation(mean_std, analyte, service.scaler_config)
            interval = lambda z: (
                unstandardize_observation(mean_std - z * predictive_sd, analyte, service.scaler_config),
                unstandardize_observation(mean_std + z * predictive_sd, analyte, service.scaler_config),
            )
            lower90, upper90 = interval(Z_90)
            lower95, upper95 = interval(Z_95)
            status = "AVAILABLE" if all(_finite(v) for v in (point, lower90, upper90, lower95, upper95)) else "NUMERICAL_UNAVAILABLE"
            forecasts.append(BiomarkerForecast(
                analyte, service.scaler_config.analytes[analyte].unit, horizon,
                (origin + pd.Timedelta(hours=horizon)).isoformat(),
                float(point) if _finite(point) else None,
                (float(lower90), float(upper90)) if status == "AVAILABLE" else None,
                (float(lower95), float(upper95)) if status == "AVAILABLE" else None,
                float(latent_sd), float(predictive_sd), status,
            ))

    return ForecastResult(
        subject_id=subject_id,
        forecast_origin=origin.isoformat(),
        timeline_mode=timeline_mode.strip().upper(),
        eligibility_semantics=semantics,
        eligible_observation_count=eligible_count,
        initialization_status="OBSERVATION_UPDATED_FORECAST",
        forecasts=forecasts,
    )
