"""Focused Module 3 causal forecasting tests using synthetic patient records."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.engine.forecasting import SUPPORTED_HORIZONS_HOURS, forecast_patient_biomarkers
from src.engine.forecast_evaluation import _metrics, _target_after_origin
from src.engine.preprocessing import load_cohort_scalers
from src.engine.state_space import StateSpaceParameters


class FakeService:
    def __init__(self, records: pd.DataFrame) -> None:
        self.records = records
        self.scaler_config = load_cohort_scalers()
        self.params = StateSpaceParameters()

    def get_patient_records(self, subject_id: int) -> pd.DataFrame:
        assert subject_id == 1
        return self.records.copy()


def _records() -> pd.DataFrame:
    return pd.DataFrame({
        "charttime": pd.to_datetime(["2180-01-01 00:00", "2180-01-02 00:00", "2180-01-03 00:00"]),
        "storetime": pd.to_datetime(["2180-01-01 01:00", "2180-01-02 01:00", "2180-01-05 01:00"]),
        "delay_anomaly_flag": ["NONE", "NONE", "NONE"],
        "analyte_name": ["creatinine", "creatinine", "creatinine"],
        "valuenum_effective": [1.0, 1.2, 9.0],
    })


def test_operational_forecast_does_not_use_later_reported_record() -> None:
    service = FakeService(_records())
    result = forecast_patient_biomarkers(
        service, 1, "2180-01-03 12:00", horizons_hours=[24], analytes=["creatinine"], timeline_mode="OPERATIONAL"
    )
    assert result.eligible_observation_count == 2
    assert result.initialization_status == "OBSERVATION_UPDATED_FORECAST"
    assert result.forecasts[0].status == "AVAILABLE"


def test_biological_forecast_can_use_collected_record_retrospectively() -> None:
    service = FakeService(_records())
    result = forecast_patient_biomarkers(
        service, 1, "2180-01-03 12:00", horizons_hours=[24], analytes=["creatinine"], timeline_mode="BIOLOGICAL"
    )
    assert result.eligible_observation_count == 3
    assert "retrospective" in result.eligibility_semantics


def test_insufficient_history_is_explicit() -> None:
    service = FakeService(_records())
    result = forecast_patient_biomarkers(service, 1, "2179-12-31", horizons_hours=[24])
    assert result.initialization_status == "INSUFFICIENT_HISTORY"
    assert all(item.status == "INSUFFICIENT_HISTORY" for item in result.forecasts)


def test_intervals_and_log_inverse_transform_are_ordered() -> None:
    service = FakeService(_records())
    result = forecast_patient_biomarkers(service, 1, "2180-01-03 12:00", horizons_hours=[24], analytes=["creatinine"])
    item = result.forecasts[0]
    assert item.predicted_value is not None and item.predicted_value > 0
    assert item.predictive_interval_90 is not None and item.predictive_interval_95 is not None
    assert item.predictive_interval_95[0] <= item.predictive_interval_90[0] <= item.predicted_value
    assert item.predicted_value <= item.predictive_interval_90[1] <= item.predictive_interval_95[1]
    assert item.predictive_sd_standardized >= item.latent_projected_sd_standardized >= 0


def test_rejects_unsupported_horizon() -> None:
    with pytest.raises(ValueError, match="Unsupported horizon"):
        forecast_patient_biomarkers(FakeService(_records()), 1, "2180-01-03", horizons_hours=[48])


def test_nonfinite_model_output_is_explicitly_unavailable() -> None:
    service = FakeService(_records())
    service.params = StateSpaceParameters(r_stds=[float("nan")] * 6)
    result = forecast_patient_biomarkers(
        service, 1, "2180-01-03 12:00", horizons_hours=[24], analytes=["creatinine"]
    )
    assert result.forecasts[0].status == "NUMERICAL_UNAVAILABLE"
    assert result.forecasts[0].predicted_value is None


def test_forecast_response_schema_preserves_unavailable_risk_status() -> None:
    from backend.schemas import ForecastResponse

    result = forecast_patient_biomarkers(FakeService(_records()), 1, "2180-01-03 12:00", horizons_hours=[24])
    response = ForecastResponse(**result.to_dict())
    assert response.deterioration_risk_status == "UNAVAILABLE_NO_VALIDATED_ENDPOINT"
    assert response.forecasts[0].target_time.startswith("2180-01-04")


def test_target_window_includes_bounds_and_excludes_outside_observations() -> None:
    """The evaluation target is the first target in the documented ±25% window."""
    origin = pd.Timestamp("2180-01-01 00:00")
    records = pd.DataFrame({
        "analyte_name": ["creatinine"] * 3,
        "storetime": pd.to_datetime([
            "2180-01-01 17:59",  # one minute before the 18 h lower bound
            "2180-01-01 18:00",  # lower bound for a 24 h target
            "2180-01-02 06:01",  # one minute after the 30 h upper bound
        ]),
        "charttime": pd.to_datetime(["2180-01-01"] * 3),
        "valuenum_effective": [7.0, 1.2, 9.0],
    })
    target = _target_after_origin(records, "creatinine", origin, 24, "OPERATIONAL")
    assert target is not None
    assert target["valuenum_effective"] == 1.2


def test_metrics_use_only_the_explicit_common_comparison_mask() -> None:
    """Non-finite targets/predictions must not reach metric aggregation."""
    rows = pd.DataFrame({
        "analyte": ["creatinine"], "horizon_hours": [24], "patient_id": [1],
        "actual": [1.0], "ou": [1.1], "locf": [1.0], "population": [1.2],
        "pi90": [(0.8, 1.4)], "pi95": [(0.7, 1.5)],
    })
    metric = _metrics(rows)[0]
    assert metric["target_count"] == 1
    assert metric["unique_patients"] == 1
    assert metric["coverage_90_pct"] == 100.0
