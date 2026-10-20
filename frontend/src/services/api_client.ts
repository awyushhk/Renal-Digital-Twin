/**
 * Typed API client for the Nephron-Aware Renal Latent-State Engine backend.
 *
 * All requests are causal — the caller supplies query_time and timeline_mode.
 * The base URL is read from NEXT_PUBLIC_API_URL or falls back to localhost:8000.
 */

import type {
  CohortSummaryResponse,
  ForecastRequest,
  ForecastResponse,
  LatentTrajectoryResponse,
  PatientListResponse,
  PatientTrajectoryResponse,
  StateEstimateRequest,
  StateEstimateResponse,
} from "@/types/api";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const apiClient = {
  /** GET /health */
  health(): Promise<{ status: string; service: string; version: string }> {
    return apiFetch("/health");
  },

  /** GET /api/v1/cohort/summary */
  cohortSummary(): Promise<CohortSummaryResponse> {
    return apiFetch("/api/v1/cohort/summary");
  },

  /** GET /api/v1/patients?page=&limit=&has_admission=&search= */
  listPatients(
    page = 1,
    limit = 25,
    admissionFilter: "ALL" | "INPATIENT" | "OUTPATIENT" = "ALL",
    search?: string
  ): Promise<PatientListResponse> {
    const params = new URLSearchParams({
      page: String(page),
      limit: String(limit),
      has_admission: admissionFilter,
    });
    if (search && search.trim()) {
      params.set("search", search.trim());
    }
    return apiFetch(`/api/v1/patients?${params}`);
  },

  /** GET /api/v1/patients/{subject_id}/trajectory */
  patientTrajectory(
    subjectId: number,
    policyId = "POLICY_A_STORETIME"
  ): Promise<PatientTrajectoryResponse> {
    const params = new URLSearchParams({ policy_id: policyId });
    return apiFetch(`/api/v1/patients/${subjectId}/trajectory?${params}`);
  },

  /** GET /api/v1/patients/{subject_id}/latent-trajectory */
  latentTrajectory(
    subjectId: number,
    policyId = "POLICY_A_STORETIME"
  ): Promise<LatentTrajectoryResponse> {
    const params = new URLSearchParams({ policy_id: policyId });
    return apiFetch(`/api/v1/patients/${subjectId}/latent-trajectory?${params}`);
  },

  /** POST /api/v1/state/estimate */
  estimateState(req: StateEstimateRequest): Promise<StateEstimateResponse> {
    return apiFetch("/api/v1/state/estimate", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },

  /** POST /api/v1/forecast; returns causal future biomarker predictions only. */
  forecast(req: ForecastRequest): Promise<ForecastResponse> {
    return apiFetch("/api/v1/forecast", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },
};
