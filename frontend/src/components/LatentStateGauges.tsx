"use client";

import type { StateEstimateResponse } from "@/types/api";

interface LatentStateGaugesProps {
  estimate: StateEstimateResponse;
}

interface GaugeConfig {
  key: "filtration_state_z" | "tubular_stress_z" | "workload_avidity_z";
  sdKey: "filtration_sd" | "tubular_sd" | "workload_sd";
  label: string;
  icon: string;
  color: string;
  glowColor: string;
  description: string;
  biologyAnchor: string;
}

const GAUGE_CONFIGS: GaugeConfig[] = [
  {
    key: "filtration_state_z",
    sdKey: "filtration_sd",
    label: "Filtration score",
    icon: "🫀",
    color: "var(--accent-cyan)",
    glowColor: "rgba(0,212,255,0.25)",
    description: "Glomerular clearance latent score (z-score)",
    biologyAnchor: "Observed primarily via serum creatinine & BUN",
  },
  {
    key: "tubular_stress_z",
    sdKey: "tubular_sd",
    label: "Tubular / acid-base score",
    icon: "⚡",
    color: "var(--accent-violet)",
    glowColor: "rgba(168,85,247,0.25)",
    description: "Electrolyte handling & acid-base stress score (z-score)",
    biologyAnchor: "Observed via potassium, sodium & bicarbonate",
  },
  {
    key: "workload_avidity_z",
    sdKey: "workload_sd",
    label: "Workload / Avidity Proxy",
    icon: "📈",
    color: "var(--accent-amber)",
    glowColor: "rgba(245,158,11,0.25)",
    description: "Neurohormonal / prerenal workload proxy (z-score)",
    biologyAnchor: "Observed via BUN, sodium & glucose",
  },
];

/** Map a z-score in [-4, +4] to a 0–100 bar fill percentage. */
function zToPercent(z: number): number {
  return Math.min(100, Math.max(0, ((z + 4) / 8) * 100));
}

export default function LatentStateGauges({ estimate }: LatentStateGaugesProps) {
  const {
    filtered_latent_states: fls,
    conditional_uncertainty_sd: sd,
    calculated_empirical_indices: idx,
    contributing_observations_count: obsCount,
    initialization_status: initStatus,
    eligibility_explanation: explanation,
    hours_since_last_observation: hoursSince,
  } = estimate;

  const isUninitialized = obsCount === 0 || initStatus === "UNINITIALIZED_PRIOR_STATE";

  return (
    <>
      {/* ── Uninitialized / No Observations Explicit Banner ── */}
      {isUninitialized && (
        <div
          id="no-observations-alert"
          style={{
            padding: "16px 20px",
            background: "rgba(245, 158, 11, 0.08)",
            border: "1px solid rgba(245, 158, 11, 0.35)",
            borderRadius: "var(--radius-md)",
            marginBottom: 20,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <span style={{ fontSize: 20 }}>⚠</span>
            <div style={{ fontWeight: 700, color: "var(--accent-amber)", fontSize: 14 }}>
              NO ELIGIBLE OBSERVATIONS AT QUERY TIME
            </div>
          </div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            {explanation ||
              `The selected query time precedes any eligible observations for this patient. Under strict causality, the filter has not performed any measurement updates.`}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
            Status: <code style={{ color: "var(--accent-amber)" }}>UNINITIALIZED_PRIOR_STATE</code> ·
            Values displayed below represent the <strong>unconditioned population prior mean (0.000)</strong> with
            asymptotic stationary uncertainty, NOT patient-specific filtered estimates.
          </div>
        </div>
      )}

      <div className="research-note">
        <span aria-hidden="true">ℹ</span>
        <span>Research scores, not direct clinical measurements or diagnoses.</span>
        <details>
          <summary>About these estimates</summary>
          <p>They are continuous-discrete Ornstein–Uhlenbeck scores standardized against the MIMIC-IV cohort; they are not validated measurements of nephron count, true GFR, or biopsy pathology.</p>
        </details>
      </div>

      {/* ── Latent state gauges ── */}
      <div className="gauges-grid">
        {GAUGE_CONFIGS.map((cfg) => {
          const value = fls[cfg.key];
          const sdVal = sd[cfg.sdKey];
          const pct = zToPercent(value);

          return (
            <div
              key={cfg.key}
              id={`gauge-${cfg.key}`}
              className="gauge-card"
              style={{
                borderColor: isUninitialized ? "rgba(255, 255, 255, 0.1)" : "var(--border-subtle)",
                borderStyle: isUninitialized ? "dashed" : "solid",
              }}
              title={`${cfg.description}`}
            >
              <div className="gauge-header">
                <div>
                  <div className="gauge-name">{cfg.label}</div>
                </div>
                <span className="gauge-icon" aria-hidden="true">{cfg.icon}</span>
              </div>

              <div>
                <div className="gauge-value-row">
                  <span
                    className="gauge-value"
                    style={{ color: isUninitialized ? "var(--text-muted)" : cfg.color }}
                  >
                    {value >= 0 ? "+" : ""}
                    {value.toFixed(2)}
                  </span>
                  <span className="gauge-unit">z-score</span>
                </div>

                <div className="gauge-ci">
                  Estimated uncertainty: ±{sdVal.toFixed(2)} SD
                </div>

                {/* Progress bar with centered zero-reference marker */}
                <div className="gauge-track">
                  <div
                    className="gauge-fill"
                    style={{
                      width: `${pct}%`,
                      background: isUninitialized
                        ? "rgba(255, 255, 255, 0.15)"
                        : `linear-gradient(90deg, ${cfg.color}88, ${cfg.color})`,
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      left: "50%",
                      top: 0,
                      bottom: 0,
                      width: 2,
                      background: "rgba(255, 255, 255, 0.35)",
                      zIndex: 2,
                    }}
                    title="Cohort Population Zero Reference"
                  />
                </div>
                <details className="score-details">
                  <summary>Details</summary>
                  <p>{cfg.biologyAnchor}. 1 SD range: [{Math.max(-4, value - sdVal).toFixed(2)}, {Math.min(4, value + sdVal).toFixed(2)}]; 2 SD range: [{Math.max(-4, value - 2 * sdVal).toFixed(2)}, {Math.min(4, value + 2 * sdVal).toFixed(2)}].</p>
                </details>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Track 2 Derived Empirical Indices ── */}
      <div className="track2-section">
        <div className="track2-title">
          <span>Additional indicators</span>
          <span>Calculated from eligible observations</span>
        </div>

        <div className="indices-grid">
          {/* Trajectory Stability Index */}
          <div className="index-card" id="index-tsi">
            <div className="index-card-header">
              <span className="index-card-name">Trajectory Stability Index (TSI)</span>
              <span className="index-card-icon" aria-hidden="true">🎯</span>
            </div>
            <div className="index-value" style={{ color: "var(--accent-teal)" }}>
              {idx.trajectory_stability_index !== null
                ? idx.trajectory_stability_index.toFixed(4)
                : "—"}
            </div>
            <div className="index-status"><code style={{ color: "var(--accent-teal)" }}>{idx.trajectory_stability_status}</code></div>
            <details className="index-details"><summary>Details</summary><p>Historical creatinine trajectory statistic, bounded in (0, 1]. It is not a clinical diagnostic confidence score.</p></details>
          </div>

          {/* Derived CKD-EPI 2021 eGFR Proxy */}
          <div className="index-card" id="index-egfr">
            <div className="index-card-header">
              <span className="index-card-name">Creatinine-derived eGFR estimate (CKD-EPI 2021)</span>
              <span className="index-card-icon" aria-hidden="true">🧪</span>
            </div>
            <div className="index-value" style={{ color: "var(--accent-cyan)" }}>
              {idx.derived_egfr_ckd_epi !== null ? (
                <>
                  {idx.derived_egfr_ckd_epi.toFixed(1)}
                  <span style={{ fontSize: 13, color: "var(--text-muted)", marginLeft: 6 }}>
                    mL/min/1.73m²
                  </span>
                </>
              ) : (
                "—"
              )}
            </div>
            <div className="index-status"><code style={{ color: "var(--accent-cyan)" }}>{idx.derived_egfr_interpretation}</code></div>
            <details className="index-details"><summary>Details</summary><p>CKD-EPI 2021 creatinine-derived calculation from the latest eligible creatinine, age, and sex; not an independent measurement or validation target.</p></details>
          </div>

          {/* Co-temporal BUN/Cr Ratio */}
          <div className="index-card" id="index-bun-cr">
            <div className="index-card-header">
              <span className="index-card-name">Co-temporal BUN / Creatinine</span>
              <span className="index-card-icon" aria-hidden="true">⚖</span>
            </div>
            <div className="index-value" style={{ color: "var(--accent-amber)" }}>
              {idx.cotemporal_bun_cr_ratio !== null ? (
                <>
                  {idx.cotemporal_bun_cr_ratio.toFixed(1)}
                  <span style={{ fontSize: 13, color: "var(--text-muted)", marginLeft: 6 }}>
                    mass ratio
                  </span>
                </>
              ) : (
                "—"
              )}
            </div>
            <div className="index-status"><code style={{ color: "var(--accent-amber)" }}>{idx.cotemporal_ratio_status}</code></div>
            <details className="index-details"><summary>Details</summary><p>Calculated only from BUN and creatinine draws collected within two hours; values are not synthesized across visits.</p></details>
          </div>
        </div>
      </div>
    </>
  );
}
