"use client";

import type { ObservationRecord } from "@/types/api";
import { formatClinicalTimestamp } from "@/lib/clinical_time";

interface ContributingLabsTableProps {
  observations: readonly ObservationRecord[];
}

const ANALYTE_COLORS: Record<number, { bg: string; text: string }> = {
  50912: { bg: "rgba(0,212,255,0.12)", text: "var(--accent-cyan)" },     // creatinine
  51006: { bg: "rgba(245,158,11,0.12)", text: "var(--accent-amber)" },   // BUN
  50971: { bg: "rgba(168,85,247,0.12)", text: "var(--accent-violet)" },  // potassium
  50983: { bg: "rgba(99,140,255,0.12)", text: "var(--accent-blue)" },    // sodium
  50882: { bg: "rgba(45,212,191,0.12)", text: "var(--accent-teal)" },    // bicarbonate
  50931: { bg: "rgba(244,63,94,0.12)", text: "var(--accent-rose)" },     // glucose
};

function formatDate(iso: string): string {
  return formatClinicalTimestamp(iso);
}

export default function ContributingLabsTable({ observations }: ContributingLabsTableProps) {
  if (observations.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🔬</div>
        <div className="empty-state-title">No contributing observations</div>
        <div className="empty-state-sub">
          No lab records were causally available at the selected query time.
        </div>
      </div>
    );
  }

  return (
    <div className="labs-table-wrapper">
      <table className="labs-table" aria-label="Contributing laboratory observations">
        <thead>
          <tr>
            <th scope="col">Analyte</th>
            <th scope="col">Chart time</th>
            <th scope="col">Store time</th>
            <th scope="col" style={{ textAlign: "right" }}>Value</th>
            <th scope="col">Unit</th>
            <th scope="col" style={{ textAlign: "right" }}>z-score</th>
            <th scope="col" style={{ textAlign: "right" }}>Delay (h)</th>
            <th scope="col">Role</th>
          </tr>
        </thead>
        <tbody>
          {observations.map((obs, idx) => {
            const color = ANALYTE_COLORS[obs.itemid];
            return (
              <tr key={`${obs.itemid}-${obs.charttime}-${idx}`}>
                <td>
                  <span
                    className="analyte-pill"
                    style={color
                      ? { background: color.bg, color: color.text }
                      : { background: "var(--bg-elevated)", color: "var(--text-secondary)" }
                    }
                  >
                    {obs.analyte_name}
                  </span>
                </td>
                <td style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  {formatDate(obs.charttime)}
                </td>
                <td style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  {obs.storetime ? formatDate(obs.storetime) : "—"}
                </td>
                <td style={{ textAlign: "right", color: "var(--text-primary)", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                  {obs.valuenum_effective.toFixed(3)}
                </td>
                <td style={{ color: "var(--text-muted)", fontSize: 11 }}>
                  {obs.unit_harmonized}
                </td>
                <td style={{ textAlign: "right", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  {obs.standardized_value !== null && obs.standardized_value !== undefined
                    ? (obs.standardized_value >= 0 ? "+" : "") + obs.standardized_value.toFixed(3)
                    : <span style={{ color: "var(--text-muted)" }}>—</span>}
                </td>
                <td style={{ textAlign: "right", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  {obs.reporting_delay_hours.toFixed(1)}
                </td>
                <td>
                  <span className="arbitration-role-tag">{obs.arbitration_role}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
