"use client";

import { useState, useMemo } from "react";
import {
  clinicalTimeToMs,
  formatClinicalTimestamp,
  formatClinicalTimestampFromMs,
} from "@/lib/clinical_time";
import type {
  LatentTrajectoryPoint,
  ObservationRecord,
  StateEstimateResponse,
} from "@/types/api";

interface PatientGraphsProps {
  observations: readonly ObservationRecord[];
  latentPoints: readonly LatentTrajectoryPoint[];
  currentEstimate: StateEstimateResponse | null;
  queryTime: string;
  timelineMode: "OPERATIONAL" | "BIOLOGICAL";
  onSelectTime?: (isoString: string) => void;
}

const ANALYTE_COLORS: Record<string, { stroke: string; fill: string; unit: string; label: string }> = {
  creatinine: { stroke: "#00d4ff", fill: "rgba(0,212,255,0.15)", unit: "mg/dL", label: "Creatinine" },
  urea_nitrogen: { stroke: "#3b82f6", fill: "rgba(59,130,246,0.15)", unit: "mg/dL", label: "BUN" },
  potassium: { stroke: "#a855f7", fill: "rgba(168,85,247,0.15)", unit: "mEq/L", label: "Potassium" },
  sodium: { stroke: "#00f5c4", fill: "rgba(0,245,196,0.15)", unit: "mEq/L", label: "Sodium" },
  bicarbonate: { stroke: "#10b981", fill: "rgba(16,185,129,0.15)", unit: "mEq/L", label: "Bicarbonate" },
  glucose: { stroke: "#f59e0b", fill: "rgba(245,158,11,0.15)", unit: "mg/dL", label: "Glucose" },
};

export default function PatientGraphs({
  observations,
  latentPoints,
  currentEstimate,
  queryTime,
  timelineMode,
  onSelectTime,
}: PatientGraphsProps) {
  const [activeTab, setActiveTab] = useState<"all" | "labs" | "latent" | "availability">("all");
  // One analyte is the readable default. The scaled overlay remains available for comparison.
  const [selectedAnalyte, setSelectedAnalyte] = useState<string>("creatinine");
  const [hoveredPoint, setHoveredPoint] = useState<{
    x: number;
    y: number;
    title: string;
    lines: string[];
  } | null>(null);

  // One canonical wall-clock axis is shared by all graph panels and the query marker.
  const { tMin, tMax, queryMs, hasValidTimeSpan, axisStartLabel, axisEndLabel } = useMemo(() => {
    const times: number[] = [];
    observations.forEach((o) => {
      times.push(clinicalTimeToMs(o.charttime));
      times.push(clinicalTimeToMs(o.storetime));
    });
    latentPoints.forEach((p) => {
      times.push(clinicalTimeToMs(p.charttime));
    });

    const qMs = clinicalTimeToMs(queryTime);
    if (!isNaN(qMs)) times.push(qMs);

    const valid = times.filter((t) => !isNaN(t));
    if (valid.length === 0) {
      return { tMin: 0, tMax: 1, queryMs: 0, hasValidTimeSpan: false, axisStartLabel: "—", axisEndLabel: "—" };
    }

    const min = Math.min(...valid);
    const max = Math.max(...valid);
    const span = max - min || 3600000;
    // Add small 2% padding on edges
    return {
      tMin: min - span * 0.02,
      tMax: max + span * 0.02,
      queryMs: isNaN(qMs) ? min : qMs,
      hasValidTimeSpan: true,
      axisStartLabel: formatClinicalTimestampFromMs(min),
      axisEndLabel: formatClinicalTimestampFromMs(max),
    };
  }, [observations, latentPoints, queryTime]);

  const timeToX = (tStr: string, width: number, padLeft = 50, padRight = 20) => {
    const ms = clinicalTimeToMs(tStr);
    if (isNaN(ms) || tMax <= tMin) return padLeft;
    const ratio = Math.max(0, Math.min(1, (ms - tMin) / (tMax - tMin)));
    return padLeft + ratio * (width - padLeft - padRight);
  };

  const msToX = (ms: number, width: number, padLeft = 50, padRight = 20) => {
    if (isNaN(ms) || tMax <= tMin) return padLeft;
    const ratio = Math.max(0, Math.min(1, (ms - tMin) / (tMax - tMin)));
    return padLeft + ratio * (width - padLeft - padRight);
  };

  if (!hasValidTimeSpan || observations.length === 0) {
    return (
      <div className="panel" style={{ textAlign: "center", padding: "32px 20px" }}>
        <div style={{ fontSize: 24, marginBottom: 8 }}>📊</div>
        <div style={{ color: "var(--text-secondary)", fontSize: 13 }}>
          No longitudinal observations available to graph for this patient.
        </div>
      </div>
    );
  }

  // Group observations by analyte
  const analyteGroups = useMemo(() => {
    const map: Record<string, ObservationRecord[]> = {};
    Object.keys(ANALYTE_COLORS).forEach((k) => (map[k] = []));
    observations.forEach((o) => {
      if (map[o.analyte_name]) {
        map[o.analyte_name].push(o);
      }
    });
    // Sort each group chronologically
    Object.keys(map).forEach((k) => {
      map[k].sort((a, b) => clinicalTimeToMs(a.charttime) - clinicalTimeToMs(b.charttime));
    });
    return map;
  }, [observations]);

  const svgWidth = 960;
  const graphAHeight = 250;
  const graphBHeight = 260;
  const graphCHeight = 210;

  const qX = msToX(queryMs, svgWidth);

  return (
    <section className="panel" aria-label="Longitudinal patient graphs" style={{ marginTop: 24 }}>
      {/* ── Section Header & Sub-Navigation ── */}
      <div className="panel-header" style={{ flexWrap: "wrap", gap: 12 }}>
        <div>
          <h2 className="panel-title" style={{ fontSize: 15 }}>
            <span className="panel-title-icon">📈</span>
            Patient Timeline
          </h2>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
            Observed labs, model estimates, and reporting delays · hospital local clock
          </div>
        </div>

        {/* View Tabs */}
        <div className="filter-row" style={{ margin: 0 }}>
          <button
            type="button"
            className={`filter-btn ${activeTab === "all" ? "active" : ""}`}
            onClick={() => setActiveTab("all")}
          >
            All charts
          </button>
          <button
            type="button"
            className={`filter-btn ${activeTab === "labs" ? "active" : ""}`}
            onClick={() => setActiveTab("labs")}
          >
            Labs
          </button>
          <button
            type="button"
            className={`filter-btn ${activeTab === "latent" ? "active" : ""}`}
            onClick={() => setActiveTab("latent")}
          >
            State estimates
          </button>
          <button
            type="button"
            className={`filter-btn ${activeTab === "availability" ? "active" : ""}`}
            onClick={() => setActiveTab("availability")}
          >
            Reporting time
          </button>
        </div>
      </div>

      {/* ── Query Line Legend ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 12,
          padding: "8px 12px",
          background: "rgba(255, 255, 255, 0.02)",
          borderRadius: "var(--radius-sm)",
          border: "1px solid rgba(255, 255, 255, 0.06)",
          fontSize: 11,
          marginBottom: 16,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 14, height: 2, background: "#ef4444", borderTop: "2px dashed #ef4444" }} />
          <span style={{ color: "var(--text-secondary)" }}>
            Query time:{" "}
            <strong style={{ color: "#ef4444", fontFamily: "monospace" }}>
              {formatClinicalTimestamp(queryTime)}
            </strong>{" "}
            ({timelineMode.toLowerCase()} view)
          </span>
        </div>
        <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
          Hover a point for its value and timestamp.
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* ── GRAPH A: Laboratory Trajectories ── */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {(activeTab === "all" || activeTab === "labs") && (
        <div style={{ marginBottom: 24 }}>
          <div className="graph-heading">
            <div>
              <span style={{ fontWeight: 600, fontSize: 13, color: "var(--text-primary)" }}>
                Laboratory results
              </span>
              <span style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: 8 }}>
                Each dot is an observed result; lines never bridge gaps longer than 14 days.
              </span>
            </div>

            {/* Analyte filter buttons */}
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              <button
                type="button"
                className={`filter-btn ${selectedAnalyte === "all" ? "active" : ""}`}
                style={{ fontSize: 10, padding: "2px 6px" }}
                onClick={() => setSelectedAnalyte("all")}
              >
                All (scaled)
              </button>
              {Object.entries(ANALYTE_COLORS).map(([key, cfg]) => (
                <button
                  key={key}
                  type="button"
                  className={`filter-btn ${selectedAnalyte === key ? "active" : ""}`}
                  style={{
                    fontSize: 10,
                    padding: "2px 6px",
                    color: selectedAnalyte === key ? "#fff" : cfg.stroke,
                  }}
                  onClick={() => setSelectedAnalyte(key)}
                >
                  {cfg.label} ({analyteGroups[key]?.length || 0})
                </button>
              ))}
            </div>
          </div>

          <div className="clinical-chart-scroll">
            <svg
              viewBox={`0 0 ${svgWidth} ${graphAHeight}`}
              className="clinical-chart"
            >
              {/* Background grid lines */}
              <line x1="50" y1="20" x2={svgWidth - 20} y2="20" stroke="rgba(255,255,255,0.06)" />
              <line x1="50" y1={graphAHeight / 2} x2={svgWidth - 20} y2={graphAHeight / 2} stroke="rgba(255,255,255,0.06)" />
              <line x1="50" y1={graphAHeight - 25} x2={svgWidth - 20} y2={graphAHeight - 25} stroke="rgba(255,255,255,0.12)" />

              {/* Causal query gate vertical line */}
              <line
                x1={qX}
                y1="10"
                x2={qX}
                y2={graphAHeight - 25}
                stroke="#ef4444"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />
              <text x={Math.min(qX + 4, svgWidth - 54)} y="18" fill="#ef4444" fontSize="11">
                Query
              </text>

              {/* Render each analyte's points */}
              {Object.entries(analyteGroups).map(([analyte, records]) => {
                if (records.length === 0) return null;
                if (selectedAnalyte !== "all" && selectedAnalyte !== analyte) return null;

                const cfg = ANALYTE_COLORS[analyte];
                const vals = records.map((r) => r.valuenum_effective).filter(Number.isFinite);
                if (vals.length === 0) return null;
                const minV = Math.min(...vals);
                const maxV = Math.max(...vals);
                const spanV = maxV - minV || 1;

                const valToY = (v: number) => {
                  const ratio = (v - minV) / spanV;
                  return graphAHeight - 35 - ratio * (graphAHeight - 65);
                };

                // Build line segments between points that are close in time (< 7 days)
                const segments: string[] = [];
                for (let i = 0; i < records.length; i++) {
                  const x = timeToX(records[i].charttime, svgWidth);
                  const y = valToY(records[i].valuenum_effective);
                  if (i === 0) {
                    segments.push(`M ${x} ${y}`);
                  } else {
                    const prevX = timeToX(records[i - 1].charttime, svgWidth);
                    const dtDays = (clinicalTimeToMs(records[i].charttime) - clinicalTimeToMs(records[i - 1].charttime)) / 86400000;
                    if (dtDays <= 14) {
                      segments.push(`L ${x} ${y}`);
                    } else {
                      segments.push(`M ${x} ${y}`);
                    }
                  }
                }

                return (
                  <g key={analyte}>
                    {/* Trajectory path */}
                    <path
                      d={segments.join(" ")}
                      fill="none"
                      stroke={cfg.stroke}
                      strokeWidth="1.5"
                      strokeOpacity="0.65"
                    />

                    {/* Individual points */}
                    {records.map((r, i) => {
                      const x = timeToX(r.charttime, svgWidth);
                      const y = valToY(r.valuenum_effective);
                      const isFutureStoretime =
                        timelineMode === "OPERATIONAL" &&
                        clinicalTimeToMs(r.storetime) > queryMs;

                      return (
                        <circle
                          key={i}
                          cx={x}
                          cy={y}
                          r={isFutureStoretime ? 3 : 4}
                          fill={isFutureStoretime ? "none" : cfg.stroke}
                          stroke={cfg.stroke}
                          strokeWidth={isFutureStoretime ? 1.5 : 1}
                          strokeDasharray={isFutureStoretime ? "2 1" : undefined}
                          opacity={isFutureStoretime ? 0.45 : 0.9}
                          style={{ cursor: "pointer" }}
                          onMouseEnter={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            setHoveredPoint({
                              x: rect.x + 8,
                              y: rect.y - 10,
                              title: `${cfg.label}: ${r.valuenum_effective} ${r.unit_harmonized}`,
                              lines: [
                                `Charttime: ${formatClinicalTimestamp(r.charttime)}`,
                                `Storetime: ${r.storetime ? formatClinicalTimestamp(r.storetime) : "N/A"}`,
                                `Delay: +${r.reporting_delay_hours.toFixed(2)}h`,
                                `Arbitration: ${r.arbitration_role}`,
                                isFutureStoretime ? "⚠ Causally masked in Operational mode" : "✓ Visible at query time",
                              ],
                            });
                          }}
                          onMouseLeave={() => setHoveredPoint(null)}
                          onClick={() => onSelectTime && onSelectTime(r.charttime)}
                        />
                      );
                    })}
                  </g>
                );
              })}

              {selectedAnalyte === "all" && (
                <text x="50" y="18" fill="var(--text-muted)" fontSize="10">
                  Each analyte is scaled to its own observed range; select one analyte for values.
                </text>
              )}
              {/* Time axis label */}
              <text x="50" y={graphAHeight - 8} fill="var(--text-muted)" fontSize="11">
                {axisStartLabel.slice(0, 10)}
              </text>
              <text x={svgWidth - 90} y={graphAHeight - 8} fill="var(--text-muted)" fontSize="11">
                {axisEndLabel.slice(0, 10)}
              </text>
            </svg>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* ── GRAPH B: Latent-State Trajectories ── */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {(activeTab === "all" || activeTab === "latent") && (
        <div style={{ marginBottom: 24 }}>
          <div className="graph-heading">
            <div>
              <span style={{ fontWeight: 600, fontSize: 13, color: "var(--text-primary)" }}>
                Estimated state trajectories
              </span>
              <span style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: 8 }}>
                Model estimates at observation epochs; gaps longer than 14 days are not connected.
              </span>
            </div>
            <div style={{ display: "flex", gap: 12, fontSize: 10 }}>
              <span style={{ color: "var(--accent-cyan)" }}>● Filtration</span>
              <span style={{ color: "var(--accent-violet)" }}>● Tubular / acid-base</span>
              <span style={{ color: "var(--accent-amber)" }}>● Workload / avidity</span>
            </div>
          </div>

          <div className="clinical-chart-scroll">
            <svg
              viewBox={`0 0 ${svgWidth} ${graphBHeight}`}
              className="clinical-chart"
            >
              {/* Zero-reference horizontal line */}
              <line
                x1="50"
                y1={graphBHeight / 2}
                x2={svgWidth - 20}
                y2={graphBHeight / 2}
                stroke="rgba(255,255,255,0.2)"
                strokeDasharray="2 2"
              />
              <text x={Math.min(qX + 4, svgWidth - 54)} y="18" fill="#ef4444" fontSize="11">
                Query
              </text>
              <text x="15" y={graphBHeight / 2 + 4} fill="var(--text-muted)" fontSize="11">
                0.0 z
              </text>
              <text x="15" y="24" fill="var(--text-muted)" fontSize="11">
                +3.0 z
              </text>
              <text x="15" y={graphBHeight - 32} fill="var(--text-muted)" fontSize="11">
                -3.0 z
              </text>

              {/* Causal query gate line */}
              <line
                x1={qX}
                y1="10"
                x2={qX}
                y2={graphBHeight - 25}
                stroke="#ef4444"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />

              {latentPoints.length > 0 && (() => {
                // Map z in [-3, +3] to y
                const zToY = (z: number) => {
                  const clamped = Math.max(-3, Math.min(3, z));
                  const ratio = (clamped + 3) / 6;
                  return graphBHeight - 30 - ratio * (graphBHeight - 50);
                };

                const tracks = [
                  { key: "filtration_state_z", sdKey: "filtration_sd", color: "#00d4ff", name: "Filtration" },
                  { key: "tubular_stress_z", sdKey: "tubular_sd", color: "#a855f7", name: "Tubular" },
                  { key: "workload_avidity_z", sdKey: "workload_sd", color: "#f59e0b", name: "Workload" },
                ] as const;

                return tracks.map((tr) => {
                  const pts = latentPoints.map((p) => ({
                    x: timeToX(p.charttime, svgWidth),
                    y: zToY(p[tr.key]),
                    raw: p,
                  }));

                  const linePath = pts.map((p, i) => {
                    const previous = latentPoints[i - 1];
                    const gapDays = previous
                      ? (clinicalTimeToMs(p.raw.charttime) - clinicalTimeToMs(previous.charttime)) / 86400000
                      : 0;
                    return `${i === 0 || gapDays > 14 ? "M" : "L"} ${p.x} ${p.y}`;
                  }).join(" ");

                  return (
                    <g key={tr.key}>
                      <path d={linePath} fill="none" stroke={tr.color} strokeWidth="1.8" />
                      {pts.map((p, i) => (
                        <circle
                          key={i}
                          cx={p.x}
                          cy={p.y}
                          r="3"
                          fill={tr.color}
                          opacity="0.85"
                          style={{ cursor: "pointer" }}
                          onMouseEnter={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            setHoveredPoint({
                              x: rect.x + 8,
                              y: rect.y - 10,
                              title: `${tr.name} State: ${p.raw[tr.key] >= 0 ? "+" : ""}${p.raw[tr.key].toFixed(3)} z`,
                              lines: [
                                `Epoch: ${formatClinicalTimestamp(p.raw.charttime)}`,
                                `Conditional SD: ±${p.raw[tr.sdKey].toFixed(3)}`,
                                `Observed: ${p.raw.observed_analytes.join(", ") || "Predict-only"}`,
                                `dt: +${p.raw.dt_hours}h`,
                              ],
                            });
                          }}
                          onMouseLeave={() => setHoveredPoint(null)}
                        />
                      ))}
                    </g>
                  );
                });
              })()}

              <text x="50" y={graphBHeight - 8} fill="var(--text-muted)" fontSize="11">
                {axisStartLabel.slice(0, 10)}
              </text>
              <text x={svgWidth - 90} y={graphBHeight - 8} fill="var(--text-muted)" fontSize="11">
                {axisEndLabel.slice(0, 10)}
              </text>
            </svg>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* ── GRAPH C: Observation and Availability Timeline ── */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {(activeTab === "all" || activeTab === "availability") && (
        <div>
          <div className="graph-heading">
            <div>
              <span style={{ fontWeight: 600, fontSize: 13, color: "var(--text-primary)" }}>
                Lab collection vs reporting time
              </span>
              <span style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: 8 }}>
                A line shows the delay between collection and when the result entered the record.
              </span>
            </div>
            <div style={{ display: "flex", gap: 10, fontSize: 10 }}>
              <span style={{ color: "var(--accent-teal)" }}>● Reported by query time</span>
              <span style={{ color: "#ef4444" }}>○ Reported after query time</span>
            </div>
          </div>

          <div className="clinical-chart-scroll">
            <svg
              viewBox={`0 0 ${svgWidth} ${graphCHeight}`}
              className="clinical-chart"
            >
              <line x1="50" y1="20" x2={svgWidth - 20} y2="20" stroke="rgba(255,255,255,0.06)" />
              <line x1="50" y1={graphCHeight / 2} x2={svgWidth - 20} y2={graphCHeight / 2} stroke="rgba(255,255,255,0.06)" />
              <line x1="50" y1={graphCHeight - 25} x2={svgWidth - 20} y2={graphCHeight - 25} stroke="rgba(255,255,255,0.12)" />

              {/* Causal query gate line */}
              <line
                x1={qX}
                y1="10"
                x2={qX}
                y2={graphCHeight - 25}
                stroke="#ef4444"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />
              <text x={Math.min(qX + 4, svgWidth - 54)} y="18" fill="#ef4444" fontSize="11">
                Query
              </text>

              {/* Render horizontal delay lines for distinct draw epochs */}
              {observations.slice(0, 120).map((obs, i) => {
                const x1 = timeToX(obs.charttime, svgWidth);
                const hasValidStoretime = Boolean(obs.storetime) && Number.isFinite(clinicalTimeToMs(obs.storetime));
                const x2 = hasValidStoretime ? timeToX(obs.storetime, svgWidth) : x1;
                const rowY = 30 + (i % 8) * 12;

                const isMaskedInOps =
                  timelineMode === "OPERATIONAL" &&
                  hasValidStoretime &&
                  clinicalTimeToMs(obs.storetime) > queryMs;

                const strokeColor = isMaskedInOps ? "#ef4444" : "#00f5c4";

                return (
                  <g key={i}>
                    {/* Connecting delay bar */}
                    {x2 > x1 && (
                      <line
                        x1={x1}
                        y1={rowY}
                        x2={x2}
                        y2={rowY}
                        stroke={strokeColor}
                        strokeWidth="1.5"
                        opacity={isMaskedInOps ? 0.35 : 0.6}
                      />
                    )}
                    {/* Specimen draw point */}
                    <circle
                      cx={x1}
                      cy={rowY}
                      r="2.5"
                      fill={strokeColor}
                      opacity={isMaskedInOps ? 0.5 : 0.85}
                    />
                    {/* Storetime filed point */}
                    {hasValidStoretime && (
                      <circle
                        cx={x2}
                        cy={rowY}
                        r="3.5"
                        fill={isMaskedInOps ? "none" : strokeColor}
                        stroke={strokeColor}
                        strokeWidth="1.2"
                        onMouseEnter={(e) => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          setHoveredPoint({
                            x: rect.x + 8,
                            y: rect.y - 10,
                            title: `${obs.analyte_name.toUpperCase()} Draw`,
                            lines: [
                              `Charttime (drawn): ${formatClinicalTimestamp(obs.charttime)}`,
                              `Storetime (filed): ${formatClinicalTimestamp(obs.storetime)}`,
                              `Turnaround Delay: +${obs.reporting_delay_hours.toFixed(2)} hours`,
                              isMaskedInOps
                                ? `❌ CAUSALLY MASKED in Operational mode (Filed after query)`
                                : `✔ Available to Operational Filter`,
                            ],
                          });
                        }}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                    )}
                  </g>
                );
              })}

              <text x="50" y={graphCHeight - 8} fill="var(--text-muted)" fontSize="11">
                {axisStartLabel.slice(0, 10)}
              </text>
              <text x={svgWidth - 90} y={graphCHeight - 8} fill="var(--text-muted)" fontSize="11">
                {axisEndLabel.slice(0, 10)}
              </text>
            </svg>
          </div>
        </div>
      )}

      {/* Floating Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: "fixed",
            left: hoveredPoint.x,
            top: hoveredPoint.y,
            transform: "translate(-50%, -100%)",
            background: "rgba(10, 15, 29, 0.95)",
            border: "1px solid var(--accent-cyan)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 12px",
            fontSize: 11,
            pointerEvents: "none",
            zIndex: 1000,
            boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
            maxWidth: 280,
          }}
        >
          <div style={{ fontWeight: 700, color: "var(--accent-cyan)", marginBottom: 4 }}>
            {hoveredPoint.title}
          </div>
          {hoveredPoint.lines.map((l, i) => (
            <div key={i} style={{ color: "var(--text-secondary)", fontSize: 10 }}>
              {l}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
