"use client";

import { useState, useEffect } from "react";
import {
  datetimeLocalToClinicalTime,
  formatClinicalTimestamp,
  midpointClinicalTime,
  toDatetimeLocal,
} from "@/lib/clinical_time";

export type TimelineMode = "OPERATIONAL" | "BIOLOGICAL";

interface DualTimelineScrubberProps {
  patientId: number | null;
  mode: TimelineMode;
  onModeChange: (mode: TimelineMode) => void;
  onEstimate: (queryTime: string, mode: TimelineMode) => void;
  isLoading: boolean;
  initialQueryTime?: string | null;
  firstObservationTime?: string | null;
  latestObservationTime?: string | null;
  totalObservationsCount?: number;
}

export default function DualTimelineScrubber({
  patientId,
  mode,
  onModeChange,
  onEstimate,
  isLoading,
  initialQueryTime,
  firstObservationTime,
  latestObservationTime,
  totalObservationsCount = 0,
}: DualTimelineScrubberProps) {
  const [dateValue, setDateValue] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Sync dateValue when initialQueryTime changes (e.g. on patient selection)
  useEffect(() => {
    if (initialQueryTime) {
      setDateValue(toDatetimeLocal(initialQueryTime));
    }
  }, [initialQueryTime]);

  function handleEstimate() {
    setError(null);
    if (!patientId) {
      setError("Please select a patient from the sidebar first.");
      return;
    }
    if (!dateValue) {
      setError("Please enter or select a valid clinical query time.");
      return;
    }
    const iso = datetimeLocalToClinicalTime(dateValue);
    onEstimate(iso, mode);
  }

  function jumpToFirst() {
    if (firstObservationTime) {
      const v = toDatetimeLocal(firstObservationTime);
      setDateValue(v);
      onEstimate(datetimeLocalToClinicalTime(v), mode);
    }
  }

  function jumpToLatest() {
    if (latestObservationTime) {
      const v = toDatetimeLocal(latestObservationTime);
      setDateValue(v);
      onEstimate(datetimeLocalToClinicalTime(v), mode);
    }
  }

  function jumpToMid() {
    if (firstObservationTime && latestObservationTime) {
      const midpoint = midpointClinicalTime(firstObservationTime, latestObservationTime);
      if (midpoint) {
        const v = toDatetimeLocal(midpoint);
        setDateValue(v);
        onEstimate(midpoint, mode);
      }
    }
  }

  return (
    <div className="timeline-section">
      {/* Mode toggle */}
      <div className="timeline-mode-toggle" role="tablist" aria-label="Causal timeline modes">
        <button
          id="timeline-mode-operational"
          className={`mode-btn ${mode === "OPERATIONAL" ? "active operational" : ""}`}
          onClick={() => onModeChange("OPERATIONAL")}
          aria-pressed={mode === "OPERATIONAL"}
          type="button"
        >
          Operational (reported results)
        </button>
        <button
          id="timeline-mode-biological"
          className={`mode-btn ${mode === "BIOLOGICAL" ? "active biological" : ""}`}
          onClick={() => onModeChange("BIOLOGICAL")}
          aria-pressed={mode === "BIOLOGICAL"}
          type="button"
        >
          Biological (collection time)
        </button>
      </div>

      {/* Mode description */}
      <div className="mode-desc">
        {mode === "OPERATIONAL" ? (
          <>
            <span className="op">Operational:</span>{" "}
            use only results reported by the selected hospital-clock time.
          </>
        ) : (
          <>
            <span className="bio">Biological:</span>{" "}
            use samples collected by that time; intended for retrospective comparison.
          </>
        )}
      </div>

      {/* Patient timeline span & jump controls */}
      {patientId && firstObservationTime && latestObservationTime && (
        <div className="timeline-span-row">
          <div>
            <span className="muted">{totalObservationsCount} records: </span>
            <span className="from">{formatClinicalTimestamp(firstObservationTime)}</span>
            <span className="muted"> → </span>
            <span className="to">{formatClinicalTimestamp(latestObservationTime)}</span>
          </div>
          <div className="timeline-jumps">
            <button
              id="jump-first-btn"
              type="button"
              className="filter-btn"
              onClick={jumpToFirst}
              title="Jump to patient's first eligible laboratory observation"
            >
              First
            </button>
            <button
              id="jump-mid-btn"
              type="button"
              className="filter-btn"
              onClick={jumpToMid}
              title="Jump to midpoint of patient's trajectory"
            >
              Middle
            </button>
            <button
              id="jump-latest-btn"
              type="button"
              className="filter-btn"
              onClick={jumpToLatest}
              title="Jump to patient's latest eligible laboratory observation"
            >
              Latest
            </button>
          </div>
        </div>
      )}

      <div className="deidentified-time-note">
        Dates use the de-identified MIMIC-IV hospital clock; they are not current calendar dates.
      </div>

      {/* DateTime picker and execute button */}
      <div className="timeline-datetime-row">
        <div className="input-group" style={{ flex: 1, minWidth: 220 }}>
          <label htmlFor="query-datetime-input" className="input-label">
            Query time (hospital local clock)
          </label>
          <input
            id="query-datetime-input"
            type="datetime-local"
            className="input-field"
            value={dateValue}
            onChange={(e) => setDateValue(e.target.value)}
            aria-label="Select clinical query time"
          />
        </div>
        <button
          id="estimate-state-btn"
          type="button"
          className={`estimate-btn ${isLoading ? "loading" : ""}`}
          onClick={handleEstimate}
          disabled={isLoading || !patientId}
          aria-label="Run latent state estimation at chosen query time"
        >
          {isLoading ? "Running Filter…" : "Estimate State"}
        </button>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}
    </div>
  );
}