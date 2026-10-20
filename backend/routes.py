"""Module 2: FastAPI Route Definitions for Renal Latent-State Engine.

Provides typed endpoints for cohort metadata, patient browsing, longitudinal
trajectories, and causal query-time latent state estimation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.schemas import (
    CohortAnalyteScale,
    CohortSummaryResponse,
    LatentTrajectoryPoint,
    LatentTrajectoryResponse,
    ObservationRecord,
    PatientListResponse,
    PatientSummaryItem,
    PatientTrajectoryResponse,
    ResponseProvenance,
    StateEstimateRequest,
    StateEstimateResponse,
    ForecastRequest,
    ForecastResponse,
)
from src.engine.forecasting import forecast_patient_biomarkers
from src.engine.patient_inference import PatientInferenceService
from src.engine.preprocessing import load_cohort_scalers

router = APIRouter()


def get_inference_service(request: Request) -> PatientInferenceService:
    """Dependency injection for cached patient inference service."""
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Inference service not initialized")
    return service


@router.get("/health")
def health_check() -> Dict[str, str]:
    """Service health probe."""
    return {
        "status": "healthy",
        "service": "Nephron-Aware Renal Latent State Engine",
        "version": "2.0.0",
        "milestone": "Module 2 Production Engine",
    }


@router.get("/api/v1/cohort/summary", response_model=CohortSummaryResponse)
def get_cohort_summary(
    service: PatientInferenceService = Depends(get_inference_service),
) -> CohortSummaryResponse:
    """Return frozen cohort summary statistics and training scaler parameters."""
    summary_json_path = Path("reports/generated/cohort_summary.json")
    if not summary_json_path.exists():
        raise HTTPException(status_code=404, detail="cohort_summary.json not found")

    with open(summary_json_path, "r", encoding="utf-8") as f:
        cdata = json.load(f)

    scalers = load_cohort_scalers()
    scaler_items = {
        name: CohortAnalyteScale(
            itemid=p.itemid,
            analyte_name=p.analyte_name,
            is_log_scale=p.is_log_scale,
            train_mean=p.train_mean,
            train_std=p.train_std,
            unit=p.unit,
        )
        for name, p in scalers.analytes.items()
    }

    c_metrics = cdata.get("cohort_metrics", {})
    return CohortSummaryResponse(
        total_patients=c_metrics.get("total_patients", 2500),
        total_observations=c_metrics.get("total_lab_observations", 303460),
        patient_id_range=c_metrics.get("patient_id_range", {"min": 10000032, "max": 10138195}),
        gender_distribution=c_metrics.get("gender_distribution", {"F": 1287, "M": 1213}),
        admission_metrics=c_metrics.get("admission_metrics", {}),
        analyte_scalers=scaler_items,
        schema_version="2.0.0",
        verification_hash="9534adfda88896433cae38e87efcd66008bf558595a5734d5251181a615b4d35",
    )


@router.get("/api/v1/patients", response_model=PatientListResponse)
def list_patients(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(25, ge=1, le=200, description="Items per page"),
    has_admission: Optional[str] = Query("ALL", description="'ALL', 'INPATIENT', or 'OUTPATIENT'"),
    search: Optional[str] = Query(None, description="Search by subject ID string"),
    service: PatientInferenceService = Depends(get_inference_service),
) -> PatientListResponse:
    """Return paginated list of bounded cohort patients with demographic context and search."""
    service.preload()
    df = service._patients_df
    if df is None:
        raise HTTPException(status_code=500, detail="Patients database not loaded")

    # Filter by admission status if requested
    filtered_df = df
    status_filter = has_admission.strip().upper() if has_admission else "ALL"
    if status_filter == "INPATIENT":
        filtered_df = df[df["has_admission"] == True]
    elif status_filter == "OUTPATIENT":
        filtered_df = df[df["has_admission"] == False]

    # Search filter by subject_id substring
    if search and search.strip():
        q_str = search.strip()
        filtered_df = filtered_df[filtered_df["subject_id"].astype(str).str.contains(q_str)]

    total_count = len(filtered_df)
    total_pages = max(1, (total_count + limit - 1) // limit)

    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    page_df = filtered_df.iloc[start_idx:end_idx]

    patients = [
        PatientSummaryItem(
            subject_id=int(r["subject_id"]),
            gender=str(r["gender"]),
            anchor_age=float(r["anchor_age"]),
            anchor_year_group=str(r["anchor_year_group"]) if pd.notna(r["anchor_year_group"]) else None,
            has_admission=bool(r["has_admission"]),
            admission_count=int(r["admission_count"]),
            usable_creatinine_count=int(r["usable_creatinine_count"]),
            total_target_labs_count=int(r["total_target_labs_count"]),
        )
        for _, r in page_df.iterrows()
    ]

    return PatientListResponse(
        total_patients=total_count,
        page=page,
        limit=limit,
        total_pages=total_pages,
        patients=patients,
    )


@router.get("/api/v1/patients/{subject_id}/latent-trajectory", response_model=LatentTrajectoryResponse)
def get_patient_latent_trajectory(
    subject_id: int,
    policy_id: str = Query("POLICY_A_STORETIME", description="Arbitration policy"),
    service: PatientInferenceService = Depends(get_inference_service),
) -> LatentTrajectoryResponse:
    """Return chronological latent state trajectory replay for patient graph visualization."""
    meta = service.get_patient_metadata(subject_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found in bounded cohort")

    raw_points = service.compute_latent_trajectory(subject_id, policy_id=policy_id)
    points = [LatentTrajectoryPoint(**p) for p in raw_points]
    return LatentTrajectoryResponse(
        subject_id=subject_id,
        total_epochs=len(points),
        points=points,
    )


@router.get("/api/v1/patients/{subject_id}/trajectory", response_model=PatientTrajectoryResponse)
def get_patient_trajectory(
    subject_id: int,
    policy_id: str = Query("POLICY_A_STORETIME", description="Arbitration policy"),
    service: PatientInferenceService = Depends(get_inference_service),
) -> PatientTrajectoryResponse:
    """Return complete longitudinal laboratory trajectory for a single patient."""
    meta = service.get_patient_metadata(subject_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found in bounded cohort")

    records = service.get_patient_records(subject_id, policy_id=policy_id)
    if records.empty:
        return PatientTrajectoryResponse(
            subject_id=subject_id,
            patient_metadata=meta,
            arbitration_policy=policy_id,
            total_observations=0,
            observations=[],
        )

    obs_records = [
        ObservationRecord(
            itemid=int(r["itemid"]),
            analyte_name=str(r["analyte_name"]),
            charttime=pd.to_datetime(r["charttime"]).isoformat(),
            storetime=pd.to_datetime(r["storetime"]).isoformat() if pd.notna(r["storetime"]) else "",
            reporting_delay_hours=float(r["reporting_delay_hours"]) if pd.notna(r["reporting_delay_hours"]) else 0.0,
            valuenum_effective=round(float(r["valuenum_effective"]), 4),
            unit_harmonized=str(r["unit_harmonized"]),
            arbitration_role=str(r["arbitration_role"]),
        )
        for _, r in records.iterrows()
    ]

    return PatientTrajectoryResponse(
        subject_id=subject_id,
        patient_metadata=meta,
        arbitration_policy=policy_id,
        total_observations=len(obs_records),
        observations=obs_records,
    )


@router.post("/api/v1/state/estimate", response_model=StateEstimateResponse)
def estimate_state(
    request: StateEstimateRequest,
    service: PatientInferenceService = Depends(get_inference_service),
) -> StateEstimateResponse:
    """Estimate multivariate latent state at query_time with strict causality and uncertainty intervals."""
    meta = service.get_patient_metadata(request.subject_id)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {request.subject_id} not found in bounded cohort",
        )

    try:
        est = service.estimate_state(
            subject_id=request.subject_id,
            query_time=request.query_time,
            timeline_mode=request.timeline_mode,
            policy_id=request.policy_id,
            noise_scale_factor=request.noise_scale_factor,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Estimation failed: {str(e)}")

    obs_records = [
        ObservationRecord(
            itemid=o.itemid,
            analyte_name=o.analyte_name,
            charttime=o.charttime,
            storetime=o.storetime,
            reporting_delay_hours=o.reporting_delay_hours,
            valuenum_effective=o.valuenum_effective,
            unit_harmonized=o.unit_harmonized,
            arbitration_role=o.arbitration_role,
            standardized_value=o.standardized_value,
        )
        for o in est.contributing_observations
    ]

    return StateEstimateResponse(
        subject_id=est.subject_id,
        query_time=est.query_time,
        timeline_mode=est.timeline_mode,
        hours_since_last_observation=est.hours_since_last_observation,
        contributing_observations_count=est.contributing_observations_count,
        filtered_latent_states={
            "filtration_state_z": est.filtration_state_z,
            "tubular_stress_z": est.tubular_stress_z,
            "workload_avidity_z": est.workload_avidity_z,
        },
        conditional_uncertainty_sd={
            "filtration_sd": est.filtration_sd,
            "tubular_sd": est.tubular_sd,
            "workload_sd": est.workload_sd,
        },
        calculated_empirical_indices={
            "trajectory_stability_index": est.trajectory_stability_index,
            "trajectory_stability_status": est.trajectory_stability_status,
            "derived_egfr_ckd_epi": est.derived_egfr_ckd_epi,
            "derived_egfr_interpretation": est.derived_egfr_interpretation,
            "cotemporal_bun_cr_ratio": est.cotemporal_bun_cr_ratio,
            "cotemporal_ratio_status": est.cotemporal_ratio_status,
        },
        contributing_observations=obs_records,
        provenance=ResponseProvenance(
            active_arbitration_policy=est.active_arbitration_policy,
            operational_lookahead_violations_count=est.operational_lookahead_violations_count,
            is_snapshot_immutable=est.is_snapshot_immutable,
        ),
        initialization_status=est.initialization_status,
        eligibility_explanation=est.eligibility_explanation,
        first_eligible_time=est.first_eligible_time,
        latest_eligible_time=est.latest_eligible_time,
    )


@router.post("/api/v1/forecast", response_model=ForecastResponse)
def forecast_biomarkers(
    request: ForecastRequest,
    service: PatientInferenceService = Depends(get_inference_service),
) -> ForecastResponse:
    """Return causal, uncertainty-aware future observable-biomarker forecasts.

    This endpoint never includes later observed values. Deterioration risk is
    explicitly unavailable until a separately validated endpoint exists.
    """
    if service.get_patient_metadata(request.subject_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient {request.subject_id} not found in bounded cohort")
    try:
        result = forecast_patient_biomarkers(
            service=service,
            subject_id=request.subject_id,
            forecast_origin=request.forecast_origin,
            horizons_hours=request.horizons_hours,
            timeline_mode=request.timeline_mode,
            analytes=request.analytes,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ForecastResponse(**result.to_dict())
