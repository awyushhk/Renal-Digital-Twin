"""Module 2: Typed Pydantic Schemas for FastAPI Renal Latent-State Engine.

All request and response contracts strictly enforce types and boundaries.
Any example values are illustrative only and never hardcoded in production responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CohortAnalyteScale(BaseModel):
    itemid: int
    analyte_name: str
    is_log_scale: bool
    train_mean: float
    train_std: float
    unit: str


class CohortSummaryResponse(BaseModel):
    total_patients: int
    total_observations: int
    patient_id_range: Dict[str, int]
    gender_distribution: Dict[str, int]
    admission_metrics: Dict[str, Any]
    analyte_scalers: Dict[str, CohortAnalyteScale]
    schema_version: str = "2.0.0"
    verification_hash: str


class PatientSummaryItem(BaseModel):
    subject_id: int
    gender: str
    anchor_age: float
    anchor_year_group: Optional[str] = None
    has_admission: bool
    admission_count: int
    usable_creatinine_count: int
    total_target_labs_count: int


class PatientListResponse(BaseModel):
    total_patients: int
    page: int
    limit: int
    total_pages: int
    patients: List[PatientSummaryItem]


class ObservationRecord(BaseModel):
    itemid: int
    analyte_name: str
    charttime: str
    storetime: str
    reporting_delay_hours: float
    valuenum_effective: float
    unit_harmonized: str
    arbitration_role: str
    standardized_value: Optional[float] = None


class PatientTrajectoryResponse(BaseModel):
    subject_id: int
    patient_metadata: Dict[str, Any]
    arbitration_policy: str
    total_observations: int
    observations: List[ObservationRecord]


class StateEstimateRequest(BaseModel):
    subject_id: int = Field(..., description="MIMIC-IV subject_id in bounded cohort")
    query_time: str = Field(..., description="ISO 8601 clinical query timestamp (e.g. 2180-05-06T14:30:00Z)")
    timeline_mode: str = Field("OPERATIONAL", description="'OPERATIONAL' (respects storetime) or 'BIOLOGICAL' (charttime)")
    policy_id: str = Field("POLICY_A_STORETIME", description="Arbitration policy identifier")
    noise_scale_factor: float = Field(1.0, ge=0.1, le=10.0, description="Noise scaling factor for sensitivity analysis")


class FilteredLatentStates(BaseModel):
    filtration_state_z: float = Field(..., description="Glomerular filtration latent score (cohort z-score)")
    tubular_stress_z: float = Field(..., description="Electrolyte/acid-base tubular stress latent score (cohort z-score)")
    workload_avidity_z: float = Field(..., description="Neurohormonal / prerenal workload proxy (cohort z-score)")


class StateUncertaintySD(BaseModel):
    filtration_sd: float = Field(..., description="Model-conditional conditional standard error of filtration state")
    tubular_sd: float = Field(..., description="Model-conditional conditional standard error of tubular stress state")
    workload_sd: float = Field(..., description="Model-conditional conditional standard error of workload state")


class CalculatedEmpiricalIndices(BaseModel):
    trajectory_stability_index: Optional[float] = Field(None, description="Trajectory Stability Index in (0.0, 1.0]")
    trajectory_stability_status: str
    derived_egfr_ckd_epi: Optional[float] = Field(None, description="2021 race-free CKD-EPI derived eGFR proxy (mL/min/1.73m^2)")
    derived_egfr_interpretation: str = "ALGEBRAIC DERIVED PROXY (NON-INDEPENDENT)"
    cotemporal_bun_cr_ratio: Optional[float] = Field(None, description="Co-temporal BUN/Creatinine ratio")
    cotemporal_ratio_status: str


class ResponseProvenance(BaseModel):
    active_arbitration_policy: str
    operational_lookahead_violations_count: int
    is_snapshot_immutable: bool
    research_disclaimer: str = (
        "RESEARCH PROTOTYPE ONLY — NOT A CLINICAL DIAGNOSTIC DEVICE. "
        "Latent scores represent empirical mathematical signals conditional on the model, "
        "not direct measurements of nephron count or true physiological reserve."
    )


class StateEstimateResponse(BaseModel):
    subject_id: int
    query_time: str
    timeline_mode: str
    hours_since_last_observation: Optional[float]
    contributing_observations_count: int
    filtered_latent_states: FilteredLatentStates
    conditional_uncertainty_sd: StateUncertaintySD
    calculated_empirical_indices: CalculatedEmpiricalIndices
    contributing_observations: List[ObservationRecord]
    provenance: ResponseProvenance
    initialization_status: str = "OBSERVATION_UPDATED_ESTIMATE"
    eligibility_explanation: Optional[str] = None
    first_eligible_time: Optional[str] = None
    latest_eligible_time: Optional[str] = None


class LatentTrajectoryPoint(BaseModel):
    charttime: str
    storetime_max: str
    dt_hours: float
    filtration_state_z: float
    tubular_stress_z: float
    workload_avidity_z: float
    filtration_sd: float
    tubular_sd: float
    workload_sd: float
    observed_analytes: List[str]
    step_type: str = "FILTERED_OBSERVATION"


class LatentTrajectoryResponse(BaseModel):
    subject_id: int
    total_epochs: int
    points: List[LatentTrajectoryPoint]

