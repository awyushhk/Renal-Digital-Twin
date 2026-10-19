/**
 * API type definitions matching the FastAPI Pydantic schemas.
 * All types are read-only to prevent accidental mutation.
 */

export interface CohortAnalyteScale {
  readonly itemid: number;
  readonly analyte_name: string;
  readonly is_log_scale: boolean;
  readonly train_mean: number;
  readonly train_std: number;
  readonly unit: string;
}

export interface CohortSummaryResponse {
  readonly total_patients: number;
  readonly total_observations: number;
  readonly patient_id_range: { readonly min: number; readonly max: number };
  readonly gender_distribution: Record<string, number>;
  readonly admission_metrics: Record<string, unknown>;
  readonly analyte_scalers: Record<string, CohortAnalyteScale>;
  readonly schema_version: string;
  readonly verification_hash: string;
}

export interface PatientSummaryItem {
  readonly subject_id: number;
  readonly gender: string;
  readonly anchor_age: number;
  readonly anchor_year_group: string | null;
  readonly has_admission: boolean;
  readonly admission_count: number;
  readonly usable_creatinine_count: number;
  readonly total_target_labs_count: number;
}

export interface PatientListResponse {
  readonly total_patients: number;
  readonly page: number;
  readonly limit: number;
  readonly total_pages: number;
  readonly patients: readonly PatientSummaryItem[];
}

export interface ObservationRecord {
  readonly itemid: number;
  readonly analyte_name: string;
  readonly charttime: string;
  readonly storetime: string;
  readonly reporting_delay_hours: number;
  readonly valuenum_effective: number;
  readonly unit_harmonized: string;
  readonly arbitration_role: string;
  readonly standardized_value?: number | null;
}

export interface PatientTrajectoryResponse {
  readonly subject_id: number;
  readonly patient_metadata: Record<string, unknown>;
  readonly arbitration_policy: string;
  readonly total_observations: number;
  readonly observations: readonly ObservationRecord[];
}

export interface StateEstimateRequest {
  readonly subject_id: number;
  readonly query_time: string;
  readonly timeline_mode: "OPERATIONAL" | "BIOLOGICAL";
  readonly policy_id: string;
  readonly noise_scale_factor: number;
}

export interface FilteredLatentStates {
  readonly filtration_state_z: number;
  readonly tubular_stress_z: number;
  readonly workload_avidity_z: number;
}

export interface StateUncertaintySD {
  readonly filtration_sd: number;
  readonly tubular_sd: number;
  readonly workload_sd: number;
}

export interface CalculatedEmpiricalIndices {
  readonly trajectory_stability_index: number | null;
  readonly trajectory_stability_status: string;
  readonly derived_egfr_ckd_epi: number | null;
  readonly derived_egfr_interpretation: string;
  readonly cotemporal_bun_cr_ratio: number | null;
  readonly cotemporal_ratio_status: string;
}

export interface ResponseProvenance {
  readonly active_arbitration_policy: string;
  readonly operational_lookahead_violations_count: number;
  readonly is_snapshot_immutable: boolean;
  readonly research_disclaimer: string;
}

export interface StateEstimateResponse {
  readonly subject_id: number;
  readonly query_time: string;
  readonly timeline_mode: string;
  readonly hours_since_last_observation: number | null;
  readonly contributing_observations_count: number;
  readonly filtered_latent_states: FilteredLatentStates;
  readonly conditional_uncertainty_sd: StateUncertaintySD;
  readonly calculated_empirical_indices: CalculatedEmpiricalIndices;
  readonly contributing_observations: readonly ObservationRecord[];
  readonly provenance: ResponseProvenance;
  readonly initialization_status: string;
  readonly eligibility_explanation?: string | null;
  readonly first_eligible_time?: string | null;
  readonly latest_eligible_time?: string | null;
}

export interface LatentTrajectoryPoint {
  readonly charttime: string;
  readonly storetime_max: string;
  readonly dt_hours: number;
  readonly filtration_state_z: number;
  readonly tubular_stress_z: number;
  readonly workload_avidity_z: number;
  readonly filtration_sd: number;
  readonly tubular_sd: number;
  readonly workload_sd: number;
  readonly observed_analytes: readonly string[];
  readonly step_type: string;
}

export interface LatentTrajectoryResponse {
  readonly subject_id: number;
  readonly total_epochs: number;
  readonly points: readonly LatentTrajectoryPoint[];
}

