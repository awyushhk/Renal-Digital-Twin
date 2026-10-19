"use client";

import { useEffect, useState, useCallback } from "react";
import { apiClient } from "@/services/api_client";
import type { PatientSummaryItem } from "@/types/api";

type AdmissionFilter = "ALL" | "INPATIENT" | "OUTPATIENT";

interface PatientSelectorProps {
  selectedId: number | null;
  onSelect: (patient: PatientSummaryItem) => void;
}

export default function PatientSelector({ selectedId, onSelect }: PatientSelectorProps) {
  const [patients, setPatients] = useState<PatientSummaryItem[]>([]);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [filter, setFilter] = useState<AdmissionFilter>("ALL");
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search.trim());
      setPage(1);
    }, 250);
    return () => clearTimeout(timer);
  }, [search]);

  const loadPatients = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.listPatients(page, 10, filter, debouncedSearch);
      setPatients(data.patients as PatientSummaryItem[]);
      setTotalPages(data.total_pages);
      setTotalCount(data.total_patients);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load patients");
    } finally {
      setLoading(false);
    }
  }, [page, filter, debouncedSearch]);

  useEffect(() => {
    void loadPatients();
  }, [loadPatients]);

  const filteredPatients = patients;

  return (
    <>
      {/* Search */}
      <div className="patient-search-box">
        <span style={{ color: "var(--text-muted)", fontSize: 14 }}>🔍</span>
        <input
          id="patient-search-input"
          type="text"
          placeholder="Search by subject ID…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search patient by ID"
        />
      </div>

      {/* Admission filter */}
      <div className="filter-row">
        {(["ALL", "INPATIENT", "OUTPATIENT"] as AdmissionFilter[]).map((f) => (
          <button
            key={f}
            id={`filter-btn-${f.toLowerCase()}`}
            className={`filter-btn ${filter === f ? "active" : ""}`}
            onClick={() => {
              setFilter(f);
              setPage(1);
            }}
            aria-pressed={filter === f}
          >
            {f === "ALL" ? "All" : f === "INPATIENT" ? "Inpt." : "Outpt."}
          </button>
        ))}
      </div>

      {/* Patient list */}
      <div className="patient-list" role="listbox" aria-label="Patient list">
        {loading
          ? Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="skeleton"
                style={{ height: 56, borderRadius: "var(--radius-md)" }}
              />
            ))
          : error
          ? <div className="error-banner">⚠ {error}</div>
          : filteredPatients.length === 0
          ? (
            <div style={{ textAlign: "center", padding: "24px 0", color: "var(--text-muted)", fontSize: 12 }}>
              No patients match the current filter
            </div>
          )
          : filteredPatients.map((p) => (
              <div
                key={p.subject_id}
                id={`patient-card-${p.subject_id}`}
                className={`patient-card ${selectedId === p.subject_id ? "selected" : ""}`}
                role="option"
                aria-selected={selectedId === p.subject_id}
                tabIndex={0}
                onClick={() => onSelect(p)}
                onKeyDown={(e) => e.key === "Enter" && onSelect(p)}
              >
                <div>
                  <div className="patient-card-id">#{p.subject_id}</div>
                  <div className="patient-card-meta">
                    {p.gender} · {Math.round(p.anchor_age)} yr · {p.total_target_labs_count} labs
                  </div>
                </div>
                <span
                  className={`patient-card-badge ${p.has_admission ? "inpatient" : "outpatient"}`}
                >
                  {p.has_admission ? "Inpt" : "Outpt"}
                </span>
              </div>
            ))}
      </div>

      {/* Pagination */}
      {!loading && !error && (
        <div className="pagination-row">
          <span className="pagination-info">
            {totalCount.toLocaleString()} patients · pg {page}/{totalPages}
          </span>
          <div className="pagination-controls">
            <button
              className="page-btn"
              id="pagination-prev"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              aria-label="Previous page"
            >
              ‹
            </button>
            <button
              className="page-btn"
              id="pagination-next"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              aria-label="Next page"
            >
              ›
            </button>
          </div>
        </div>
      )}
    </>
  );
}
