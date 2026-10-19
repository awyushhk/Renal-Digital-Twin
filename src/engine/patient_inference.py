"""Module 2: Patient Trajectory Inference and Dual-Timeline State Estimator.

This module provides causal, patient-isolated inference over longitudinal
renal observation trajectories from `cohort_renal_bounded.csv`.

Dual-Timeline Semantics:
1. Operational Mode (F_ops):
   - Causal visibility gate: storetime <= query_time AND delay_anomaly_flag != 'NEGATIVE_REPORTING_DELAY'.
   - Records with storetime > query_time are mathematically inaccessible.
   - Snapshot immutability: re-evaluating at query_time produces an identical state vector.
2. Biological Mode (F_bio):
   - Evaluated strictly using specimen collection time: charttime <= query_time.
   - Tagged as retrospective-only (non-deployable for live decision points).

Output Taxonomy:
- Filtered Latent States: x_filt, x_tub, x_work (accompanied by conditional uncertainty SDs).
- Calculated Empirical Indices: Trajectory Stability Index (TSI), co-temporal BUN/Cr ratio,
  and derived 2021 race-free CKD-EPI eGFR proxy.

Research Boundary:
These are lab-derived latent scores, not direct measurements of nephron mass,
true renal functional reserve, or biopsy-level pathology.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.engine.preprocessing import (
    CohortScalerConfig,
    load_cohort_scalers,
    standardize_observation,
)
from src.engine.state_space import (
    ANALYTE_INDEX_MAP,
    ContinuousDiscreteRenalFilter,
    StateSpaceParameters,
)
from src.engine.trajectory_stability import (
    TrajectoryStabilityResult,
    compute_trajectory_stability_index,
)
from src.temporal.canonical_schema import (
    TARGET_ITEM_IDS,
    create_canonical_event_frame,
)
from src.temporal.conflict_arbitration import (
    POLICY_A_STORETIME,
    arbitrate_canonical_frame,
    get_effective_events,
)
from src.temporal.temporal_features import compute_cotemporal_bun_creatinine_ratio

DEFAULT_COHORT_LABS_PATH = Path("data/processed/cohort_renal_bounded.csv")
DEFAULT_COHORT_PATIENTS_PATH = Path("data/processed/cohort_patients.csv")


def calculate_ckd_epi_2021(
    scr_mg_dl: float,
    anchor_age: float,
    gender: str,
) -> Optional[float]:
    """Compute 2021 Race-Free CKD-EPI Creatinine Equation eGFR (mL/min/1.73 m^2).

    Reference: Inker LA, Eneanya ND, Coresh J, et al. NEJM 2021; 385:1737-1749.
    Designated strictly as an algebraic proxy, not an independent validation ground truth.
    """
    if pd.isna(scr_mg_dl) or scr_mg_dl <= 0 or pd.isna(anchor_age) or anchor_age <= 0:
        return None

    is_female = str(gender).strip().upper() == "F"

    if is_female:
        kappa = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        sex_factor = 1.0

    scr_ratio = scr_mg_dl / kappa
    min_term = min(scr_ratio, 1.0) ** alpha
    max_term = max(scr_ratio, 1.0) ** -1.200
    age_term = 0.9938 ** anchor_age

    egfr = 142.0 * min_term * max_term * age_term * sex_factor
    return round(float(egfr), 2)


@dataclass
class ContributingObservation:
    """Detailed record of an observation contributing to the state at query_time."""
    itemid: int
    analyte_name: str
    charttime: str
    storetime: str
    reporting_delay_hours: float
    valuenum_effective: float
    unit_harmonized: str
    arbitration_role: str
    standardized_value: Optional[float]


@dataclass
class LatentStateEstimate:
    """Comprehensive output of patient state estimation at query_time."""
    subject_id: int
    query_time: str
    timeline_mode: str
    hours_since_last_observation: Optional[float]
    contributing_observations_count: int

    # Filtered Latent States (Track 1)
    filtration_state_z: float
    tubular_stress_z: float
    workload_avidity_z: float

    # Model-Conditional Uncertainty (Track 1)
    filtration_sd: float
    tubular_sd: float
    workload_sd: float

    # Calculated Empirical Indices (Track 2)
    trajectory_stability_index: Optional[float]
    trajectory_stability_status: str
    derived_egfr_ckd_epi: Optional[float]
    derived_egfr_interpretation: str
    cotemporal_bun_cr_ratio: Optional[float]
    cotemporal_ratio_status: str

    # Context & Provenance
    contributing_observations: List[ContributingObservation]
    active_arbitration_policy: str
    operational_lookahead_violations_count: int
    is_snapshot_immutable: bool

    # Initialization & Explanatory Metadata
    initialization_status: str = "OBSERVATION_UPDATED_ESTIMATE"
    eligibility_explanation: Optional[str] = None
    first_eligible_time: Optional[str] = None
    latest_eligible_time: Optional[str] = None


class PatientInferenceService:
    """In-memory cached inference service for rapid API responses and trajectory replay."""

    def __init__(
        self,
        cohort_labs_path: Path = DEFAULT_COHORT_LABS_PATH,
        cohort_patients_path: Path = DEFAULT_COHORT_PATIENTS_PATH,
        scaler_config: Optional[CohortScalerConfig] = None,
        state_space_params: Optional[StateSpaceParameters] = None,
    ) -> None:
        self.cohort_labs_path = cohort_labs_path
        self.cohort_patients_path = cohort_patients_path
        self.scaler_config = scaler_config or load_cohort_scalers()
        self.params = state_space_params or StateSpaceParameters()

        # In-memory indices for sub-25ms response latency
        self._patients_df: Optional[pd.DataFrame] = None
        self._raw_labs_df: Optional[pd.DataFrame] = None
        self._patient_effective_cache: Dict[int, pd.DataFrame] = {}

    def preload(self) -> None:
        """Load datasets once into memory and index by patient ID."""
        if self._patients_df is None and self.cohort_patients_path.exists():
            self._patients_df = pd.read_csv(self.cohort_patients_path)

        if self._raw_labs_df is None and self.cohort_labs_path.exists():
            self._raw_labs_df = pd.read_csv(self.cohort_labs_path)

    def get_patient_metadata(self, subject_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve patient demographics and summary metadata."""
        self.preload()
        if self._patients_df is None:
            return None

        match = self._patients_df[self._patients_df["subject_id"] == subject_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        return {
            "subject_id": int(row["subject_id"]),
            "gender": str(row["gender"]),
            "anchor_age": float(row["anchor_age"]),
            "anchor_year": int(row["anchor_year"]) if pd.notna(row["anchor_year"]) else None,
            "anchor_year_group": str(row["anchor_year_group"]) if pd.notna(row["anchor_year_group"]) else None,
            "usable_creatinine_count": int(row["usable_creatinine_count"]),
            "total_target_labs_count": int(row["total_target_labs_count"]),
            "has_admission": bool(row["has_admission"]),
            "admission_count": int(row["admission_count"]),
        }

    def get_patient_records(
        self,
        subject_id: int,
        policy_id: str = POLICY_A_STORETIME,
    ) -> pd.DataFrame:
        """Retrieve arbitrated, effective records for a patient from memory cache."""
        if subject_id in self._patient_effective_cache:
            return self._patient_effective_cache[subject_id]

        self.preload()
        if self._raw_labs_df is None:
            raise RuntimeError("Cohort labs DataFrame not loaded")

        sub_raw = self._raw_labs_df[self._raw_labs_df["subject_id"] == subject_id]
        if sub_raw.empty:
            empty_df = pd.DataFrame()
            self._patient_effective_cache[subject_id] = empty_df
            return empty_df

        canonical = create_canonical_event_frame(sub_raw)
        arbitrated, _ = arbitrate_canonical_frame(canonical, policy_id=policy_id)
        effective = get_effective_events(arbitrated)

        # Pre-parse timestamps
        effective["charttime"] = pd.to_datetime(effective["charttime"])
        effective["storetime"] = pd.to_datetime(effective["storetime"])
        effective = effective.sort_values(["charttime", "storetime"]).reset_index(drop=True)

        self._patient_effective_cache[subject_id] = effective
        return effective

    def estimate_state(
        self,
        subject_id: int,
        query_time: str | pd.Timestamp,
        timeline_mode: str = "OPERATIONAL",
        policy_id: str = POLICY_A_STORETIME,
        noise_scale_factor: float = 1.0,
    ) -> LatentStateEstimate:
        q_dt = pd.to_datetime(query_time)
        if hasattr(q_dt, "tz") and q_dt.tz is not None:
            q_dt = q_dt.tz_localize(None)
        mode = timeline_mode.strip().upper()

        records = self.get_patient_records(subject_id, policy_id=policy_id)
        patient_meta = self.get_patient_metadata(subject_id)

        # Causal filtering
        if mode == "OPERATIONAL":
            # Only records filed by query_time without negative reporting delay
            mask = (
                (records["storetime"] <= q_dt)
                & (records["storetime"].notna())
                & (records["delay_anomaly_flag"] != "NEGATIVE_REPORTING_DELAY")
            )
            visible_records = records[mask].copy()
            # Order deterministically by charttime for physiological stepping
            visible_records = visible_records.sort_values(["charttime", "storetime"])
        elif mode == "BIOLOGICAL":
            mask = (records["charttime"] <= q_dt) & (records["charttime"].notna())
            visible_records = records[mask].copy()
            visible_records = visible_records.sort_values(["charttime", "storetime"])
        else:
            raise ValueError(f"Invalid timeline_mode: {timeline_mode}. Must be 'OPERATIONAL' or 'BIOLOGICAL'")

        # Configure filter instance
        params = StateSpaceParameters(
            tau_hours=self.params.tau_hours,
            q_scales=self.params.q_scales,
            h_21=self.params.h_21,
            h_42=self.params.h_42,
            h_43=self.params.h_43,
            h_52=self.params.h_52,
            h_63=self.params.h_63,
            r_stds=self.params.r_stds,
            noise_scale_factor=noise_scale_factor,
        )
        kf = ContinuousDiscreteRenalFilter(params=params)

        contributing_obs: List[ContributingObservation] = []

        # Determine temporal bounds for context
        if mode == "OPERATIONAL":
            valid_times = records[records["storetime"].notna()]["storetime"]
        else:
            valid_times = records[records["charttime"].notna()]["charttime"]

        first_avail = valid_times.min().isoformat() if not valid_times.empty else None
        latest_avail = valid_times.max().isoformat() if not valid_times.empty else None

        if visible_records.empty:
            # Return prior state and stationary covariance
            p_inf = kf.compute_stationary_cov()
            explanation = (
                f"Query time ({q_dt.strftime('%Y-%m-%d %H:%M:%S')}) precedes first eligible observation "
                f"({first_avail or 'None'}) under {mode} timeline constraints. "
                "Filter state remains uninitialized at cohort population prior (0.000) with stationary uncertainty."
            )
            return LatentStateEstimate(
                subject_id=subject_id,
                query_time=q_dt.isoformat(),
                timeline_mode=mode,
                hours_since_last_observation=None,
                contributing_observations_count=0,
                filtration_state_z=0.0,
                tubular_stress_z=0.0,
                workload_avidity_z=0.0,
                filtration_sd=round(float(np.sqrt(p_inf[0, 0])), 4),
                tubular_sd=round(float(np.sqrt(p_inf[1, 1])), 4),
                workload_sd=round(float(np.sqrt(p_inf[2, 2])), 4),
                trajectory_stability_index=None,
                trajectory_stability_status="NO_OBSERVATIONS_BY_QUERY_TIME",
                derived_egfr_ckd_epi=None,
                derived_egfr_interpretation="NO_CREATININE_AVAILABLE",
                cotemporal_bun_cr_ratio=None,
                cotemporal_ratio_status="NO_OBSERVATIONS",
                contributing_observations=[],
                active_arbitration_policy=policy_id,
                operational_lookahead_violations_count=0,
                is_snapshot_immutable=True,
                initialization_status="UNINITIALIZED_PRIOR_STATE",
                eligibility_explanation=explanation,
                first_eligible_time=first_avail,
                latest_eligible_time=latest_avail,
            )

        # Step through epochs chronologically
        last_epoch_time: Optional[pd.Timestamp] = None
        latest_creat_val: Optional[float] = None
        latest_bun_val: Optional[float] = None

        for ctime, epoch_df in visible_records.groupby("charttime", sort=False):
            c_ts = pd.to_datetime(ctime)
            dt_h = 0.0 if last_epoch_time is None else max(0.0, (c_ts - last_epoch_time).total_seconds() / 3600.0)
            last_epoch_time = c_ts

            kf.predict(dt_h)

            obs_dict: Dict[int, float] = {}
            for _, row in epoch_df.iterrows():
                analyte = str(row["analyte_name"])
                val = float(row["valuenum_effective"])
                itemid = int(row["itemid"])

                std_val: Optional[float] = None
                if analyte in ANALYTE_INDEX_MAP:
                    std_val = standardize_observation(val, analyte, self.scaler_config)
                    if not pd.isna(std_val):
                        obs_dict[ANALYTE_INDEX_MAP[analyte]] = float(std_val)

                if analyte == "creatinine":
                    latest_creat_val = val
                elif analyte == "urea_nitrogen":
                    latest_bun_val = val

                contributing_obs.append(
                    ContributingObservation(
                        itemid=itemid,
                        analyte_name=analyte,
                        charttime=pd.to_datetime(row["charttime"]).isoformat(),
                        storetime=pd.to_datetime(row["storetime"]).isoformat() if pd.notna(row["storetime"]) else "",
                        reporting_delay_hours=float(row["reporting_delay_hours"]) if pd.notna(row["reporting_delay_hours"]) else 0.0,
                        valuenum_effective=round(val, 4),
                        unit_harmonized=str(row["unit_harmonized"]),
                        arbitration_role=str(row["arbitration_role"]),
                        standardized_value=round(float(std_val), 4) if std_val is not None else None,
                    )
                )

            if obs_dict:
                kf.update(obs_dict)

        # Compute elapsed time since the most recent observation
        if mode == "OPERATIONAL":
            last_time = visible_records["storetime"].max()
        else:
            last_time = visible_records["charttime"].max()
        hours_since = max(0.0, (q_dt - last_time).total_seconds() / 3600.0)

        # Predict forward from last observation up to query_time
        if hours_since > 0.0:
            kf.predict(hours_since)

        # Track 2: Trajectory Stability Index
        creat_subset = visible_records[visible_records["analyte_name"] == "creatinine"]
        tsi_res = compute_trajectory_stability_index(creat_subset)

        # Track 2: Derived eGFR (CKD-EPI 2021)
        egfr_val: Optional[float] = None
        if latest_creat_val is not None and patient_meta is not None:
            egfr_val = calculate_ckd_epi_2021(
                scr_mg_dl=latest_creat_val,
                anchor_age=patient_meta["anchor_age"],
                gender=patient_meta["gender"],
            )

        # Track 2: Co-temporal BUN/Cr ratio
        bun_cr_ratio, ratio_status = compute_cotemporal_bun_creatinine_ratio(
            bun_val=latest_bun_val,
            cr_val=latest_creat_val,
        )

        return LatentStateEstimate(
            subject_id=subject_id,
            query_time=q_dt.isoformat(),
            timeline_mode=mode,
            hours_since_last_observation=round(float(hours_since), 2),
            contributing_observations_count=len(contributing_obs),
            filtration_state_z=round(float(kf.state[0]), 4),
            tubular_stress_z=round(float(kf.state[1]), 4),
            workload_avidity_z=round(float(kf.state[2]), 4),
            filtration_sd=round(float(np.sqrt(max(1e-6, kf.cov[0, 0]))), 4),
            tubular_sd=round(float(np.sqrt(max(1e-6, kf.cov[1, 1]))), 4),
            workload_sd=round(float(np.sqrt(max(1e-6, kf.cov[2, 2]))), 4),
            trajectory_stability_index=tsi_res.tsi_score,
            trajectory_stability_status=tsi_res.status,
            derived_egfr_ckd_epi=egfr_val,
            derived_egfr_interpretation="ALGEBRAIC DERIVED PROXY (NON-INDEPENDENT)",
            cotemporal_bun_cr_ratio=bun_cr_ratio if not pd.isna(bun_cr_ratio) else None,
            cotemporal_ratio_status=ratio_status,
            contributing_observations=contributing_obs,
            active_arbitration_policy=policy_id,
            operational_lookahead_violations_count=0,
            is_snapshot_immutable=True,
            initialization_status="OBSERVATION_UPDATED_ESTIMATE",
            eligibility_explanation=f"{len(contributing_obs)} observations successfully assimilated up to {q_dt.strftime('%Y-%m-%d %H:%M:%S')}.",
            first_eligible_time=first_avail,
            latest_eligible_time=latest_avail,
        )

    def compute_latent_trajectory(
        self,
        subject_id: int,
        policy_id: str = POLICY_A_STORETIME,
    ) -> List[Dict[str, Any]]:
        """Compute chronological latent state replay for trajectory graphing."""
        records = self.get_patient_records(subject_id, policy_id=policy_id)
        if records.empty:
            return []

        params = StateSpaceParameters(
            tau_hours=self.params.tau_hours,
            q_scales=self.params.q_scales,
            h_21=self.params.h_21,
            h_42=self.params.h_42,
            h_43=self.params.h_43,
            h_52=self.params.h_52,
            h_63=self.params.h_63,
            r_stds=self.params.r_stds,
        )
        kf = ContinuousDiscreteRenalFilter(params=params)
        last_time: Optional[pd.Timestamp] = None
        trajectory_points: List[Dict[str, Any]] = []

        for ctime, epoch_df in records.groupby("charttime", sort=True):
            c_ts = pd.to_datetime(ctime)
            dt_h = 0.0 if last_time is None else max(0.0, (c_ts - last_time).total_seconds() / 3600.0)
            last_time = c_ts

            kf.predict(dt_h)

            obs_dict: Dict[int, float] = {}
            analytes_seen: List[str] = []
            for _, row in epoch_df.iterrows():
                analyte = str(row["analyte_name"])
                if analyte in ANALYTE_INDEX_MAP:
                    val = float(row["valuenum_effective"])
                    std_val = standardize_observation(val, analyte, self.scaler_config)
                    if not pd.isna(std_val):
                        obs_dict[ANALYTE_INDEX_MAP[analyte]] = float(std_val)
                        analytes_seen.append(analyte)

            if obs_dict:
                kf.update(obs_dict)

            cov_diag = kf.cov.diagonal()
            max_store = epoch_df["storetime"].max()
            trajectory_points.append({
                "charttime": c_ts.isoformat(),
                "storetime_max": max_store.isoformat() if pd.notna(max_store) else "",
                "dt_hours": round(dt_h, 2),
                "filtration_state_z": round(float(kf.state[0]), 4),
                "tubular_stress_z": round(float(kf.state[1]), 4),
                "workload_avidity_z": round(float(kf.state[2]), 4),
                "filtration_sd": round(float(np.sqrt(max(1e-6, cov_diag[0]))), 4),
                "tubular_sd": round(float(np.sqrt(max(1e-6, cov_diag[1]))), 4),
                "workload_sd": round(float(np.sqrt(max(1e-6, cov_diag[2]))), 4),
                "observed_analytes": analytes_seen,
                "step_type": "FILTERED_OBSERVATION",
            })

        return trajectory_points
