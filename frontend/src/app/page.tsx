"use client";

import { useEffect, useState, useCallback } from "react";
import { apiClient } from "@/services/api_client";
import type {
  CohortSummaryResponse,
  LatentTrajectoryPoint,
  ObservationRecord,
  PatientSummaryItem,
  StateEstimateResponse,
} from "@/types/api";
import PatientSelector from "@/components/PatientSelector";
import DualTimelineScrubber, { TimelineMode } from "@/components/DualTimelineScrubber";
import LatentStateGauges from "@/components/LatentStateGauges";
import ContributingLabsTable from "@/components/ContributingLabsTable";
import ForecastPanel from "@/components/ForecastPanel";
import PatientGraphs from "@/components/PatientGraphs";
import { clinicalTimeToMs, formatClinicalTimestamp } from "@/lib/clinical_time";

type BackendStatus = "loading" | "healthy" | "offline";
type ActiveTab = "gauges" | "graphs" | "forecast" | "labs" | "provenance";

export default function DashboardPage() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("loading");
  const [cohortSummary, setCohortSummary] = useState<CohortSummaryResponse | null>(null);
  const [selectedPatient, setSelectedPatient] = useState<PatientSummaryItem | null>(null);

  // Patient longitudinal data
  const [trajectoryRecords, setTrajectoryRecords] = useState<readonly ObservationRecord[]>([]);
  const [latentPoints, setLatentPoints] = useState<readonly LatentTrajectoryPoint[]>([]);
  const [loadingTrajectory, setLoadingTrajectory] = useState(false);

  // Timeline & state estimation
  const [timelineMode, setTimelineMode] = useState<TimelineMode>("OPERATIONAL");
  const [currentQueryTime, setCurrentQueryTime] = useState<string>("");
  const [estimate, setEstimate] = useState<StateEstimateResponse | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [estimateError, setEstimateError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>("gauges");

  // First & latest observation timestamps for selected patient
  const [firstObsTime, setFirstObsTime] = useState<string | null>(null);
  const [latestObsTime, setLatestObsTime] = useState<string | null>(null);

  // ── Backend health check ──
  useEffect(() => {
    apiClient.health()
      .then(() => setBackendStatus("healthy"))
      .catch(() => setBackendStatus("offline"));
  }, []);

  // ── Cohort summary ──
  useEffect(() => {
    if (backendStatus !== "healthy") return;
    apiClient.cohortSummary()
      .then(setCohortSummary)
      .catch(() => { /* non-fatal */ });
  }, [backendStatus]);

  // ── Estimate handler ──
  const handleEstimate = useCallback(
    async (queryTime: string, mode: TimelineMode) => {
      if (!selectedPatient) return;
      setEstimating(true);
      setEstimateError(null);
      setCurrentQueryTime(queryTime);

      try {
        const result = await apiClient.estimateState({
          subject_id: selectedPatient.subject_id,
          query_time: queryTime,
          timeline_mode: mode,
          policy_id: "POLICY_A_STORETIME",
          noise_scale_factor: 1.0,
        });
        setEstimate(result);
      } catch (e) {
        setEstimateError(e instanceof Error ? e.message : "State estimation failed");
      } finally {
        setEstimating(false);
      }
    },
    [selectedPatient]
  );

  // ── Patient selection: load full trajectory & default query time to latest observation ──
  const handleSelectPatient = useCallback(
    async (patient: PatientSummaryItem) => {
      setSelectedPatient(patient);
      setEstimate(null);
      setEstimateError(null);
      setLoadingTrajectory(true);

      try {
        const [trajData, latentData] = await Promise.all([
          apiClient.patientTrajectory(patient.subject_id),
          apiClient.latentTrajectory(patient.subject_id).catch(() => ({ total_epochs: 0, points: [] })),
        ]);

        const obs = [...trajData.observations].sort(
          (left, right) => clinicalTimeToMs(left.charttime) - clinicalTimeToMs(right.charttime)
        );
        const latent = [...latentData.points].sort(
          (left, right) => clinicalTimeToMs(left.charttime) - clinicalTimeToMs(right.charttime)
        );
        setTrajectoryRecords(obs);
        setLatentPoints(latent);

        if (obs.length > 0) {
          const first = obs[0].charttime;
          const last = obs[obs.length - 1].charttime;
          setFirstObsTime(first);
          setLatestObsTime(last);

          const defaultQueryTime = last.length === 16 ? `${last}:00` : last;
          setCurrentQueryTime(defaultQueryTime);

          void handleEstimate(defaultQueryTime, timelineMode);
        } else {
          setFirstObsTime(null);
          setLatestObsTime(null);
          setCurrentQueryTime("");
        }
      } catch (e) {
        setEstimateError(e instanceof Error ? e.message : "Failed to load patient records");
      } finally {
        setLoadingTrajectory(false);
      }
    },
    [handleEstimate, timelineMode]
  );

  // ── Mode change ──
  const handleModeChange = useCallback(
    (newMode: TimelineMode) => {
      setTimelineMode(newMode);
      if (selectedPatient && currentQueryTime) {
        void handleEstimate(currentQueryTime, newMode);
      }
    },
    [selectedPatient, currentQueryTime, handleEstimate]
  );

  const navItems: { key: ActiveTab; label: string }[] = [
    { key: "gauges", label: "Estimated states" },
    { key: "graphs", label: "Patient timeline" },
    { key: "forecast", label: "Forecast" },
    { key: "labs", label: `Eligible labs (${estimate?.contributing_observations_count ?? 0})` },
    { key: "provenance", label: "Details" },
  ];

  return (
    <div className="dashboard-shell">
      {/* ── Topbar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="topbar-brand-icon" aria-hidden="true">🫘</div>
          <div>
            <div className="topbar-title">Nephron-Aware Renal Digital Twin</div>
            <div className="topbar-subtitle">Module 2 · Physiology-Informed Latent State Engine</div>
          </div>
        </div>

        <nav className="nav-pills" aria-label="Clinical views">
          {navItems.map((item) => (
            <button
              key={item.key}
              type="button"
              className={`nav-pill ${activeTab === item.key ? "active" : ""} ${item.key === "forecast" ? "forecast-tab" : ""}`}
              disabled={!selectedPatient || loadingTrajectory}
              aria-current={activeTab === item.key ? "page" : undefined}
              onClick={() => setActiveTab(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="topbar-status-row">
          {cohortSummary && (
            <span className="topbar-cohort">
              {cohortSummary.total_patients.toLocaleString()} patients · {cohortSummary.total_observations.toLocaleString()} observations
            </span>
          )}
          <span
            id="backend-status-badge"
            className={`status-badge ${backendStatus}`}
            aria-live="polite"
            aria-label={`Backend status: ${backendStatus}`}
          >
            <span className={`status-dot ${backendStatus === "healthy" ? "pulse" : ""}`} />
            {backendStatus === "loading" ? "Connecting…" : backendStatus === "healthy" ? "Backend online" : "Backend offline"}
          </span>
        </div>
      </header>

      {/* ── Sidebar ── */}
      <aside className="sidebar" aria-label="Patient selection panel">
        {cohortSummary && (
          <div className="sidebar-section">
            <div className="sidebar-section-title">Cohort Overview</div>
            <div className="cohort-stat">
              <span className="cohort-stat-label">Total patients</span>
              <span className="cohort-stat-value" style={{ color: "var(--accent-blue)" }}>
                {cohortSummary.total_patients.toLocaleString()}
              </span>
            </div>
            <div className="cohort-stat">
              <span className="cohort-stat-label">Lab observations</span>
              <span className="cohort-stat-value" style={{ color: "var(--accent-cyan)" }}>
                {cohortSummary.total_observations.toLocaleString()}
              </span>
            </div>
            <div className="cohort-stat">
              <span className="cohort-stat-label">Gender (F / M)</span>
              <span className="cohort-stat-value" style={{ color: "var(--text-secondary)" }}>
                {cohortSummary.gender_distribution["F"] ?? "—"} / {cohortSummary.gender_distribution["M"] ?? "—"}
              </span>
            </div>
            <div className="cohort-stat">
              <span className="cohort-stat-label">Engine schema</span>
              <span className="cohort-stat-value" style={{ color: "var(--accent-teal)" }}>
                v{cohortSummary.schema_version}
              </span>
            </div>
          </div>
        )}

        <div className="sidebar-section" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div className="sidebar-section-title">Select patient</div>
          {backendStatus === "offline" ? (
            <div className="error-banner">⚠ Backend is offline. Start the FastAPI server on port 8000.</div>
          ) : (
            <PatientSelector
              selectedId={selectedPatient?.subject_id ?? null}
              onSelect={handleSelectPatient}
            />
          )}
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="main-content" id="main-panel">
        <div className="page-hero">
          <div>
            <h1 className="page-hero-title">Renal State Overview</h1>
            <p className="page-hero-sub">
              Causal latent-state estimates and forecasts from irregular laboratory data. Research prototype — not for clinical use.
            </p>
          </div>
          {selectedPatient && (
            <span id="selected-patient-info" className="patient-chip">
              Patient <strong>#{selectedPatient.subject_id}</strong> · {selectedPatient.gender} · {Math.round(selectedPatient.anchor_age)} yr · {selectedPatient.total_target_labs_count} source labs
              {trajectoryRecords.length > 0 ? ` · ${trajectoryRecords.length} effective records` : ""}
            </span>
          )}
        </div>

        {/* Timeline query panel */}
        <section className="panel" aria-labelledby="timeline-panel-title">
          <div className="panel-header">
            <h2 className="panel-title" id="timeline-panel-title">
              <span className="panel-title-icon">⏱</span>
              Patient Timeline &amp; Query Time
            </h2>
          </div>

          <DualTimelineScrubber
            patientId={selectedPatient?.subject_id ?? null}
            mode={timelineMode}
            onModeChange={handleModeChange}
            onEstimate={handleEstimate}
            isLoading={estimating || loadingTrajectory}
            initialQueryTime={currentQueryTime}
            firstObservationTime={firstObsTime}
            latestObservationTime={latestObsTime}
            totalObservationsCount={trajectoryRecords.length}
          />

          {estimateError && <div className="error-banner" style={{ marginTop: 12 }}>⚠ {estimateError}</div>}
        </section>

        {/* Empty state */}
        {!selectedPatient && (
          <div className="panel">
            <div className="empty-state">
              <div className="empty-state-icon">🧬</div>
              <h2 className="empty-state-title">Select a patient to begin</h2>
              <p className="empty-state-sub">
                Choose any patient from the left sidebar (or search by subject ID, e.g. 10006029).
                The engine will replay the patient&apos;s longitudinal labs and run the continuous-discrete latent state filter.
              </p>
            </div>
          </div>
        )}

        {/* Loading */}
        {selectedPatient && loadingTrajectory && (
          <div className="panel">
            <div className="empty-state">
              <div className="spinner" />
              <div className="empty-state-title" style={{ marginTop: 14 }}>
                Loading patient #{selectedPatient.subject_id} longitudinal records…
              </div>
              <p className="empty-state-sub">Fetching laboratory trajectory and pre-computing filter steps.</p>
            </div>
          </div>
        )}

        {/* Active patient views */}
        {selectedPatient && !loadingTrajectory && (
          <>
            {activeTab === "gauges" && (
              <section id="tabpanel-gauges" className="panel" role="tabpanel">
                <div className="panel-header">
                  <h2 className="panel-title">
                    <span className="panel-title-icon">🫀</span>
                    Estimated Kidney States
                  </h2>
                  {estimate && (
                    <span className="panel-meta">
                      Mode: <strong>{estimate.timeline_mode}</strong> · Query: {formatClinicalTimestamp(estimate.query_time)}
                    </span>
                  )}
                </div>
                {estimate ? (
                  <LatentStateGauges estimate={estimate} history={latentPoints} />
                ) : (
                  <div style={{ textAlign: "center", padding: "32px 0", color: "var(--text-muted)" }}>
                    Click «Estimate State» to evaluate filter at the chosen query time.
                  </div>
                )}
              </section>
            )}

            {activeTab === "gauges" && trajectoryRecords.length > 0 && (
              <div id="primary-patient-graphs" aria-label="Patient graphs">
                <PatientGraphs
                  observations={trajectoryRecords}
                  latentPoints={latentPoints}
                  currentEstimate={estimate}
                  queryTime={currentQueryTime}
                  timelineMode={timelineMode}
                  onSelectTime={(t) => handleEstimate(t, timelineMode)}
                />
              </div>
            )}

            {activeTab === "graphs" && (
              <PatientGraphs
                observations={trajectoryRecords}
                latentPoints={latentPoints}
                currentEstimate={estimate}
                queryTime={currentQueryTime}
                timelineMode={timelineMode}
                onSelectTime={(t) => handleEstimate(t, timelineMode)}
              />
            )}

            {activeTab === "forecast" && (
              <div id="tabpanel-forecast" role="tabpanel">
                <ForecastPanel
                  patientId={selectedPatient.subject_id}
                  queryTime={currentQueryTime}
                  timelineMode={timelineMode}
                />
              </div>
            )}

            {activeTab === "labs" && (
              <section id="tabpanel-labs" className="panel" role="tabpanel">
                <div className="panel-header">
                  <h2 className="panel-title">
                    <span className="panel-title-icon">🔬</span>
                    Contributing Laboratory Observations
                  </h2>
                  <span className="panel-meta">
                    {estimate?.contributing_observations_count ?? 0} records causally visible at query time
                  </span>
                </div>
                {estimate && estimate.contributing_observations.length > 0 ? (
                  <ContributingLabsTable observations={estimate.contributing_observations} />
                ) : (
                  <div style={{ textAlign: "center", padding: "32px 0", color: "var(--text-muted)" }}>
                    No observations were eligible or filed before the selected query time under {timelineMode} mode.
                  </div>
                )}
              </section>
            )}

            {activeTab === "provenance" && estimate && (
              <section id="tabpanel-provenance" className="panel" role="tabpanel">
                <div className="panel-header">
                  <h2 className="panel-title">
                    <span className="panel-title-icon">📋</span>
                    Response Provenance &amp; Policy Audit
                  </h2>
                </div>
                <div className="provenance-box">
                  <div className="provenance-row">
                    <span className="provenance-key">Subject ID</span>
                    <span className="provenance-val">{estimate.subject_id}</span>
                  </div>
                  <div className="provenance-row">
                    <span className="provenance-key">Initialization Status</span>
                    <span
                      className="provenance-val"
                      style={{
                        color:
                          estimate.initialization_status === "OBSERVATION_UPDATED_ESTIMATE"
                            ? "var(--accent-green)"
                            : "var(--accent-amber)",
                      }}
                    >
                      {estimate.initialization_status}
                    </span>
                  </div>
                  {estimate.eligibility_explanation && (
                    <div className="provenance-row">
                      <span className="provenance-key">Eligibility Context</span>
                      <span className="provenance-val">{estimate.eligibility_explanation}</span>
                    </div>
                  )}
                  <div className="provenance-row">
                    <span className="provenance-key">Query Instant (Hospital Wall-Clock)</span>
                    <span className="provenance-val">{formatClinicalTimestamp(estimate.query_time)}</span>
                  </div>
                  <div className="provenance-row">
                    <span className="provenance-key">Timeline Mode</span>
                    <span className="provenance-val">{estimate.timeline_mode}</span>
                  </div>
                  <div className="provenance-row">
                    <span className="provenance-key">Active Arbitration Policy</span>
                    <span className="provenance-val">{estimate.provenance.active_arbitration_policy}</span>
                  </div>
                  <div className="provenance-row">
                    <span className="provenance-key">Causal Lookahead Violations Masked</span>
                    <span
                      className="provenance-val"
                      style={{
                        color:
                          estimate.provenance.operational_lookahead_violations_count > 0
                            ? "var(--accent-amber)"
                            : "var(--accent-green)",
                      }}
                    >
                      {estimate.provenance.operational_lookahead_violations_count}
                    </span>
                  </div>
                  <div className="provenance-row">
                    <span className="provenance-key">Contributing Observations Count</span>
                    <span className="provenance-val">{estimate.contributing_observations_count}</span>
                  </div>
                  <div className="provenance-row">
                    <span className="provenance-key">Hours Since Most Recent Observation</span>
                    <span className="provenance-val">
                      {estimate.hours_since_last_observation !== null
                        ? `${estimate.hours_since_last_observation.toFixed(2)} h`
                        : "N/A (Pre-observation period)"}
                    </span>
                  </div>
                  <div className="provenance-row" style={{ borderBottom: 0 }}>
                    <span className="provenance-key">Snapshot Immutability</span>
                    <span className="provenance-val" style={{ color: "var(--accent-green)" }}>Deterministic ✓</span>
                  </div>
                  <div className="disclaimer-text">{estimate.provenance.research_disclaimer}</div>
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}