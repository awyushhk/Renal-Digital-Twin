"use client";

import { useEffect, useMemo, useState } from "react";
import { apiClient } from "@/services/api_client";
import type { ForecastResponse } from "@/types/api";
import {
  datetimeLocalToClinicalTime,
  formatClinicalTimestamp,
  toDatetimeLocal,
} from "@/lib/clinical_time";
import type { TimelineMode } from "@/components/DualTimelineScrubber";

const HORIZONS = [24, 72, 168] as const;

function labelForAnalyte(name: string): string {
  return name === "urea_nitrogen" ? "BUN" : name.replaceAll("_", " ");
}

function displayValue(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
}

interface ForecastPanelProps {
  patientId: number;
  queryTime: string;
  timelineMode: TimelineMode;
}

/** A deliberately compact clinical-research forecast request and response view. */
export default function ForecastPanel({ patientId, queryTime, timelineMode }: ForecastPanelProps) {
  const [originInput, setOriginInput] = useState("");
  const [horizon, setHorizon] = useState<(typeof HORIZONS)[number]>(24);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setOriginInput(toDatetimeLocal(queryTime));
    setResult(null);
    setError(null);
  }, [patientId, queryTime, timelineMode]);

  const available = useMemo(
    () => result?.forecasts.filter((item) => item.status === "AVAILABLE") ?? [],
    [result]
  );

  async function requestForecast() {
    if (!originInput) {
      setError("Choose a valid hospital-clock forecast origin.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const forecast = await apiClient.forecast({
        subject_id: patientId,
        forecast_origin: datetimeLocalToClinicalTime(originInput),
        horizons_hours: [horizon],
        timeline_mode: timelineMode,
      });
      setResult(forecast);
    } catch (reason) {
      setResult(null);
      setError(reason instanceof Error ? reason.message : "Forecast request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel forecast-panel" aria-labelledby="forecast-panel-title">
      <div className="panel-header">
        <div>
          <h2 className="panel-title" id="forecast-panel-title">
            <span className="panel-title-icon">🔭</span>
            Biomarker Forecast
          </h2>
          <p className="forecast-subtitle">Future laboratory predictions from the current fitted research model.</p>
        </div>
        <span className="forecast-mode">{timelineMode.toLowerCase()} eligibility</span>
      </div>

      <div className="forecast-controls">
        <label className="input-group">
          <span className="input-label">Forecast origin (hospital clock)</span>
          <input
            className="input-field"
            type="datetime-local"
            value={originInput}
            onChange={(event) => setOriginInput(event.target.value)}
            aria-label="Forecast origin"
          />
        </label>
        <div className="input-group">
          <span className="input-label">Horizon</span>
          <div className="forecast-horizons" role="group" aria-label="Forecast horizon">
            {HORIZONS.map((hours) => (
              <button
                type="button"
                key={hours}
                className={`filter-btn ${horizon === hours ? "active" : ""}`}
                onClick={() => setHorizon(hours)}
                aria-pressed={horizon === hours}
              >
                {hours} h
              </button>
            ))}
          </div>
        </div>
        <button
          type="button"
          className={`estimate-btn ${loading ? "loading" : ""}`}
          onClick={requestForecast}
          disabled={loading || !originInput}
        >
          {loading ? "Forecasting…" : "Run Forecast"}
        </button>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      {result && (
        <div className="forecast-result" aria-live="polite">
          <div className="forecast-meta">
            {result.initialization_status === "INSUFFICIENT_HISTORY"
              ? "No eligible history at this origin."
              : `${result.eligible_observation_count} eligible observations · ${result.model_identifier}`}
          </div>
          {available.length > 0 ? (
            <div className="forecast-grid">
              {available.map((item) => (
                <article className="forecast-card" key={`${item.analyte_name}-${item.horizon_hours}`}>
                  <div className="forecast-card-name">{labelForAnalyte(item.analyte_name)}</div>
                  <div className="forecast-value">
                    {displayValue(item.predicted_value)} <span>{item.unit}</span>
                  </div>
                  <div className="forecast-interval">
                    95% prediction interval: {displayValue(item.predictive_interval_95?.[0] ?? null)}–{displayValue(item.predictive_interval_95?.[1] ?? null)} {item.unit}
                  </div>
                  <div className="forecast-target">Target: {formatClinicalTimestamp(item.target_time)}</div>
                </article>
              ))}
            </div>
          ) : (
            <div className="forecast-empty">
              {result.initialization_status === "INSUFFICIENT_HISTORY"
                ? "Forecast unavailable: no eligible historical observations at this origin."
                : "Forecast unavailable: the model returned no finite biomarker prediction."}
            </div>
          )}
          <p className="forecast-caveat">{result.caveat} Deterioration risk: unavailable (no validated endpoint).</p>
        </div>
      )}
    </section>
  );
}
