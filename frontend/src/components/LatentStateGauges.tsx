"use client";

import type { LatentTrajectoryPoint, StateEstimateResponse } from "@/types/api";
import { clinicalTimeToMs } from "@/lib/clinical_time";

interface LatentStateGaugesProps {
  estimate: StateEstimateResponse;
  /** Optional real latent trajectory, used only to draw a small trend sparkline. */
  history?: readonly LatentTrajectoryPoint[];
}

interface GaugeConfig {
  key: "filtration_state_z" | "tubular_stress_z" | "workload_avidity_z";
  sdKey: "filtration_sd" | "tubular_sd" | "workload_sd";
  label: string;
  icon: string;
  color: string;
  description: string;
  biologyAnchor: string;
}

const GAUGE_CONFIGS: GaugeConfig[] = [
  {
    key: "filtration_state_z",
    sdKey: "filtration_sd",
    label: "Filtration score",
    icon: "🫀",
    color: "#2563eb",
    description: "Glomerular clearance latent score (z-score)",
    biologyAnchor: "Observed primarily via serum creatinine & BUN",
  },
  {
    key: "tubular_stress_z",
    sdKey: "tubular_sd",
    label: "Tubular / acid-base score",
    icon: "⚡",
    color: "#7c3aed",
    description: "Electrolyte handling & acid-base stress score (z-score)",
    biologyAnchor: "Observed via potassium, sodium & bicarbonate",
  },
  {
    key: "workload_avidity_z",
    sdKey: "workload_sd",
    label: "Workload / Avidity Proxy",
    icon: "📈",
    color: "#d97706",
    description: "Neurohormonal / prerenal workload proxy (z-score)",
    biologyAnchor: "Observed via BUN, sodium & glucose",
  },
];

const Z_RANGE = 4;

/** Map a z-score in [-4, +4] to a 0–100 position. */
function zToPercent(z: number): number {
  return Math.min(100, Math.max(0, ((z + Z_RANGE) / (2 * Z_RANGE)) * 100));
}

function Sparkline({ values, color, id }: { values: number[]; color: string; id: string }) {
  if (values.length < 2) return null;
  const w = 120;
  const h = 44;
  const pad = 4;
  const x = (i: number) => pad + (i / (values.length - 1)) * (w - pad * 2);
  const y = (v: number) => {
    const clamped = Math.max(-Z_RANGE, Math.min(Z_RANGE, v));
    return h - pad - ((clamped + Z_RANGE) / (2 * Z_RANGE)) * (h - pad * 2);
  };
  const line = values.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
  const area = `${line} L ${x(values.length - 1).toFixed(1)} ${h - pad} L ${x(0).toFixed(1)} ${h - pad} Z`;
  return (
    <svg className="gauge-spark" viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(values.length - 1)} cy={y(values[values.length - 1])} r="3" fill={color} />
    </svg>
  );
}

export default function LatentStateGauges({ estimate, history = [] }: LatentStateGaugesProps) {
  const {
    filtered_latent_states: fls,
    conditional_uncertainty_sd: sd,
    calculated_empirical_indices: idx,
    contributing_observations_count: obsCount,
    initialization_status: initStatus,
    eligibility_explanation: explanation,
  } = estimate;

  const isUninitialized = obsCount === 0 || initStatus === "UNINITIALIZED_PRIOR_STATE";

  // Recent real estimates up to the query time (display only).
  const queryMs = clinicalTimeToMs(estimate.query_time);
  const recent = history
    .filter((p) => Number.isNaN(queryMs) || clinicalTimeToMs(p.charttime) <= queryMs)
    .slice(-30);

  return (
    <>
      {isUninitialized && (
        <div id="no-observations-alert" className="uninit-alert">
          <div className="uninit-alert-title">⚠ NO ELIGIBLE OBSERVATIONS AT QUERY TIME</div>
          <div className="uninit-alert-body">
            {explanation ||
              `The selected query time precedes any eligible observations for this patient. Under strict causality, the filter has not performed any measurement updates.`}
          </div>
          <div className="uninit-alert-foot">
            Status: <code style={{ color: "var(--accent-amber)" }}>UNINITIALIZED_PRIOR_STATE</code> · Values below
            represent the <strong>unconditioned population prior mean (0.000)</strong> with asymptotic stationary
            uncertainty, NOT patient-specific filtered estimates.
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

      <div className="gauges-grid">
        {GAUGE_CONFIGS.map((cfg) => {
          const value = fls[cfg.key];
          const sdVal = sd[cfg.sdKey];
          const pct = zToPercent(value);
          const lo = zToPercent(value - sdVal);
          const hi = zToPercent(value + sdVal);
          const fillColor = isUninitialized ? "#c3cbdc" : cfg.color;

          return (
            <div
              key={cfg.key}
              id={`gauge-${cfg.key}`}
              className="gauge-card"
              style={{ borderStyle: isUninitialized ? "dashed" : "solid" }}
              title={cfg.description}
            >
              <div className="gauge-header">
                <span className="gauge-icon" aria-hidden="true">{cfg.icon}</span>
                <div>
                  <div className="gauge-name">{cfg.label}</div>
                  <div className="gauge-sub">z-score vs. MIMIC-IV cohort</div>
                </div>
              </div>

              <div className="gauge-body">
                <div>
                  <div className="gauge-value-row">
                    <span className="gauge-value" style={{ color: isUninitialized ? "var(--text-muted)" : cfg.color }}>
                      {value >= 0 ? "+" : ""}
                      {value.toFixed(2)}
                    </span>
                    <span className="gauge-unit">z</span>
                  </div>
                  <div className="gauge-ci">Estimated uncertainty: ±{sdVal.toFixed(2)} SD</div>
                </div>
                {!isUninitialized && <Sparkline values={recent.map((p) => p[cfg.key])} color={cfg.color} id={`spark-${cfg.key}`} />}
              </div>

              {/* Diverging bar: fills from the cohort zero line toward the estimate; band = ±1 SD */}
              <div>
                <div className="gauge-track">
                  <div className="gauge-band" style={{ left: `${lo}%`, width: `${Math.max(0, hi - lo)}%`, background: fillColor }} />
                  <div
                    className="gauge-fill"
                    style={{ left: `${Math.min(50, pct)}%`, width: `${Math.abs(pct - 50)}%`, background: fillColor }}
                  />
                  <div className="gauge-zero" title="Cohort Population Zero Reference" />
                </div>
                <div className="gauge-scale">
                  <span>−{Z_RANGE}</span>
                  <span>0</span>
                  <span>+{Z_RANGE}</span>
                </div>
              </div>

              <details className="score-details">
                <summary>Details</summary>
                <p>
                  {cfg.biologyAnchor}. 1 SD range: [{Math.max(-Z_RANGE, value - sdVal).toFixed(2)}, {Math.min(Z_RANGE, value + sdVal).toFixed(2)}]; 2 SD range: [{Math.max(-Z_RANGE, value - 2 * sdVal).toFixed(2)}, {Math.min(Z_RANGE, value + 2 * sdVal).toFixed(2)}].
                </p>
              </details>
            </div>
          );
        })}
      </div>

      <div className="track2-section">
        <div className="track2-title">
          <span>Additional indicators</span>
          <span>Calculated from eligible observations</span>
        </div>

        <div className="indices-grid">
          <div className="index-card" id="index-tsi">
            <div className="index-card-header">
              <span className="index-card-name">Trajectory Stability Index (TSI)</span>
              <span className="index-card-icon" aria-hidden="true">🎯</span>
            </div>
            <div className="index-value" style={{ color: "var(--accent-teal)" }}>
              {idx.trajectory_stability_index !== null ? idx.trajectory_stability_index.toFixed(4) : "—"}
            </div>
            <div className="index-status"><code style={{ color: "var(--accent-teal)" }}>{idx.trajectory_stability_status}</code></div>
            <details className="index-details"><summary>Details</summary><p>Historical creatinine trajectory statistic, bounded in (0, 1]. It is not a clinical diagnostic confidence score.</p></details>
          </div>

          <div className="index-card" id="index-egfr">
            <div className="index-card-header">
              <span className="index-card-name">Creatinine-derived eGFR estimate (CKD-EPI 2021)</span>
              <span className="index-card-icon" aria-hidden="true">🧪</span>
            </div>
            <div className="index-value" style={{ color: "var(--accent-blue)" }}>
              {idx.derived_egfr_ckd_epi !== null ? (
                <>
                  {idx.derived_egfr_ckd_epi.toFixed(1)}
                  <span style={{ fontSize: 13, color: "var(--text-muted)", marginLeft: 6, fontWeight: 600 }}>mL/min/1.73m²</span>
                </>
              ) : (
                "—"
              )}
            </div>
            <div className="index-status"><code style={{ color: "var(--accent-blue)" }}>{idx.derived_egfr_interpretation}</code></div>
            <details className="index-details"><summary>Details</summary><p>CKD-EPI 2021 creatinine-derived calculation from the latest eligible creatinine, age, and sex; not an independent measurement or validation target.</p></details>
          </div>

          <div className="index-card" id="index-bun-cr">
            <div className="index-card-header">
              <span className="index-card-name">Co-temporal BUN / Creatinine</span>
              <span className="index-card-icon" aria-hidden="true">⚖</span>
            </div>
            <div className="index-value" style={{ color: "var(--accent-amber)" }}>
              {idx.cotemporal_bun_cr_ratio !== null ? (
                <>
                  {idx.cotemporal_bun_cr_ratio.toFixed(1)}
                  <span style={{ fontSize: 13, color: "var(--text-muted)", marginLeft: 6, fontWeight: 600 }}>mass ratio</span>
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