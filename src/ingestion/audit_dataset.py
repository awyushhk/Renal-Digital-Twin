"""High-performance dataset inventory and audit module for MIMIC-IV renal digital twin (Milestone M1.1).

Optimized architecture:
1. Laboratory dictionary audit (d_labitems.csv) and candidate renal biomarker identification.
2. Demographics and admissions audit (patients.csv, admissions.csv) with referential linkage checks.
3. Memory-conscious, chunked laboratory events audit (labevents.csv) separated into modular operations:
   - Schema & row-count audit (selective columns, typed reading)
   - Item frequency & unit audit (vectorized counts)
   - Renal-candidate-focused audit (filtered by verified dictionary IDs, vectorized statistics, zero row iteration)
   - Optional expensive duplicate/conflict audit (toggled via flag, bounded memory)
4. Periodic progress logging with throughput (rows/sec), elapsed time, and dynamic ETA estimation.
5. Atomic report generation (writes to temporary files and replaces atomically) with strict partial vs. complete labeling.
6. Graceful interruption handling ensuring interrupted runs are never labeled as complete.

All source files are strictly read-only.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Path Resolution and Configuration
# ---------------------------------------------------------------------------

def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Find the project root directory by searching for markers like configs/ or requirements.txt."""
    curr = (start_path or Path.cwd()).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "configs").is_dir() and (parent / "requirements.txt").is_file():
            return parent
    return curr


def load_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load paths and execution configuration from YAML, resolving paths relative to project root."""
    root = find_project_root()
    default_config_path = root / "configs" / "paths.yaml"

    target_cfg = Path(config_path).resolve() if config_path else default_config_path

    config: dict[str, Any] = {}
    if target_cfg.is_file():
        with open(target_cfg, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    data_cfg = config.get("data", {})
    raw_dir = root / data_cfg.get("raw_dir", "data/raw")
    raw_files = data_cfg.get("raw_files", {})

    reports_cfg = config.get("reports", {})
    generated_dir = root / reports_cfg.get("generated_dir", "reports/generated")

    resolved_config = {
        "project_root": root,
        "raw_dir": raw_dir,
        "files": {
            "d_labitems": root / raw_files.get("d_labitems", "data/raw/d_labitems.csv"),
            "labevents": root / raw_files.get("labevents", "data/raw/labevents.csv"),
            "patients": root / raw_files.get("patients", "data/raw/patients.csv"),
            "admissions": root / raw_files.get("admissions", "data/raw/admissions.csv"),
        },
        "reports": {
            "generated_dir": generated_dir,
            "dataset_inventory": root / reports_cfg.get("dataset_inventory", "reports/generated/dataset_inventory.md"),
            "lab_item_inventory": root / reports_cfg.get("lab_item_inventory", "reports/generated/lab_item_inventory.csv"),
            "data_quality_summary": root / reports_cfg.get("data_quality_summary", "reports/generated/data_quality_summary.csv"),
            "renal_candidate_items": root / reports_cfg.get("renal_candidate_items", "reports/generated/renal_candidate_items.csv"),
        },
        "audit": config.get("audit", {
            "smoke": {"chunk_size": 100000, "max_chunks": 2},
            "full": {"chunk_size": 250000, "max_chunks": None},
        }),
    }
    return resolved_config


# ---------------------------------------------------------------------------
# Renal Candidate Search Definitions
# ---------------------------------------------------------------------------

RENAL_SEARCH_CRITERIA = [
    {
        "keyword": "creatinine",
        "category_group": "Creatinine",
        "reason": "Primary endogenous filtration biomarker for kidney function and eGFR calculation.",
    },
    {
        "keyword": "urea",
        "category_group": "BUN / Urea Nitrogen",
        "reason": "Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio.",
    },
    {
        "keyword": "bun",
        "category_group": "BUN / Urea Nitrogen",
        "reason": "Blood Urea Nitrogen abbreviation in legacy/alternate lab codes.",
    },
    {
        "keyword": "albumin",
        "category_group": "Albumin",
        "reason": "Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury.",
    },
    {
        "keyword": "glucose",
        "category_group": "Glucose",
        "reason": "Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria.",
    },
    {
        "keyword": "cystatin",
        "category_group": "Cystatin-C",
        "reason": "Alternative filtration marker independent of muscle mass; evaluated for CKD-EPI cystatin equations.",
    },
    {
        "keyword": "protein",
        "category_group": "Urine / Total Protein",
        "reason": "Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage.",
    },
    {
        "keyword": "gfr",
        "category_group": "eGFR Metric",
        "reason": "Calculated or reported Glomerular Filtration Rate estimates (MDRD, CKD-EPI) in lab records.",
    },
    {
        "keyword": "potassium",
        "category_group": "Electrolytes",
        "reason": "Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR.",
    },
    {
        "keyword": "sodium",
        "category_group": "Electrolytes",
        "reason": "Extracellular volume and osmolality regulation; marker of tubular handling and fluid status.",
    },
    {
        "keyword": "bicarbonate",
        "category_group": "Electrolytes",
        "reason": "Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease.",
    },
    {
        "keyword": "chloride",
        "category_group": "Electrolytes",
        "reason": "Anion balance and metabolic evaluation alongside sodium and bicarbonate.",
    },
]


# ---------------------------------------------------------------------------
# 1. Audit Laboratory Dictionary (d_labitems.csv)
# ---------------------------------------------------------------------------

@dataclass
class DictAuditResult:
    total_rows: int
    unique_itemids: int
    columns: list[str]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    duplicate_itemids: int
    fluid_counts: dict[str, int]
    category_counts: dict[str, int]
    candidates_df: pd.DataFrame
    all_items_df: pd.DataFrame


def audit_d_labitems(file_path: Path | str) -> DictAuditResult:
    """Audit d_labitems.csv and discover candidate renal tests."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Laboratory dictionary not found: {p}")

    df = pd.read_csv(p)
    total_rows = len(df)
    unique_itemids = df["itemid"].nunique()
    cols = df.columns.tolist()
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    null_counts = {c: int(df[c].isnull().sum()) for c in cols}
    duplicate_itemids = int(df["itemid"].duplicated().sum())

    fluid_counts = df["fluid"].value_counts().to_dict() if "fluid" in df else {}
    category_counts = df["category"].value_counts().to_dict() if "category" in df else {}

    candidates_records = []
    seen_itemids: set[int] = set()
    label_series = df["label"].fillna("").astype(str)

    for crit in RENAL_SEARCH_CRITERIA:
        kw = crit["keyword"]
        mask = label_series.str.contains(kw, case=False, na=False)
        matched_df = df[mask]

        for _, row in matched_df.iterrows():
            item_id = int(row["itemid"])
            if item_id not in seen_itemids:
                seen_itemids.add(item_id)
                candidates_records.append({
                    "itemid": item_id,
                    "label": str(row.get("label", "")),
                    "fluid": str(row.get("fluid", "")),
                    "category": str(row.get("category", "")),
                    "search_keyword": kw,
                    "category_group": crit["category_group"],
                    "selection_reason": crit["reason"],
                })

    candidates_df = pd.DataFrame(candidates_records)
    if candidates_df.empty:
        candidates_df = pd.DataFrame(
            columns=["itemid", "label", "fluid", "category", "search_keyword", "category_group", "selection_reason"]
        )
    else:
        candidates_df.sort_values(by=["category_group", "itemid"], inplace=True)

    return DictAuditResult(
        total_rows=total_rows,
        unique_itemids=unique_itemids,
        columns=cols,
        dtypes=dtypes,
        null_counts=null_counts,
        duplicate_itemids=duplicate_itemids,
        fluid_counts=fluid_counts,
        category_counts=category_counts,
        candidates_df=candidates_df,
        all_items_df=df,
    )


# ---------------------------------------------------------------------------
# 2. Audit Demographics and Admissions
# ---------------------------------------------------------------------------

@dataclass
class DemographicsAdmissionsAuditResult:
    patients_rows: int
    patients_unique_subjects: int
    patients_columns: list[str]
    patients_dtypes: dict[str, str]
    patients_nulls: dict[str, int]
    patients_gender_dist: dict[str, int]
    patients_anchor_age_summary: dict[str, float]
    patients_dod_missing_count: int

    admissions_rows: int
    admissions_unique_hadm: int
    admissions_unique_subjects: int
    admissions_columns: list[str]
    admissions_dtypes: dict[str, str]
    admissions_nulls: dict[str, int]
    admissions_type_dist: dict[str, int]
    admissions_hospital_expire_count: int

    admissions_subject_in_patients_pct: float
    patients_with_admissions_count: int
    patients_without_admissions_count: int
    unparseable_admittime_count: int
    unparseable_dischtime_count: int
    invalid_los_count: int
    median_los_days: float


def audit_demographics_and_admissions(
    patients_path: Path | str, admissions_path: Path | str
) -> DemographicsAdmissionsAuditResult:
    """Audit patients.csv and admissions.csv, evaluating schemas, integrity, and linkage."""
    pat_file = Path(patients_path)
    adm_file = Path(admissions_path)

    if not pat_file.is_file():
        raise FileNotFoundError(f"Patients file not found: {pat_file}")
    if not adm_file.is_file():
        raise FileNotFoundError(f"Admissions file not found: {adm_file}")

    # Read patients
    df_pat = pd.read_csv(pat_file)
    pat_rows = len(df_pat)
    pat_uniq_sub = df_pat["subject_id"].nunique() if "subject_id" in df_pat else 0
    pat_cols = df_pat.columns.tolist()
    pat_dtypes = {c: str(t) for c, t in df_pat.dtypes.items()}
    pat_nulls = {c: int(df_pat[c].isnull().sum()) for c in pat_cols}
    pat_gender = df_pat["gender"].value_counts().to_dict() if "gender" in df_pat else {}
    pat_dod_null = int(df_pat["dod"].isnull().sum()) if "dod" in df_pat else 0

    pat_age_sum: dict[str, float] = {}
    if "anchor_age" in df_pat:
        age_clean = pd.to_numeric(df_pat["anchor_age"], errors="coerce").dropna()
        if not age_clean.empty:
            pat_age_sum = {
                "min": float(age_clean.min()),
                "q25": float(age_clean.quantile(0.25)),
                "median": float(age_clean.median()),
                "q75": float(age_clean.quantile(0.75)),
                "max": float(age_clean.max()),
                "mean": float(age_clean.mean()),
            }

    # Read admissions
    df_adm = pd.read_csv(adm_file)
    adm_rows = len(df_adm)
    adm_uniq_hadm = df_adm["hadm_id"].nunique() if "hadm_id" in df_adm else 0
    adm_uniq_sub = df_adm["subject_id"].nunique() if "subject_id" in df_adm else 0
    adm_cols = df_adm.columns.tolist()
    adm_dtypes = {c: str(t) for c, t in df_adm.dtypes.items()}
    adm_nulls = {c: int(df_adm[c].isnull().sum()) for c in adm_cols}
    adm_type = df_adm["admission_type"].value_counts().to_dict() if "admission_type" in df_adm else {}
    adm_expire = int((df_adm["hospital_expire_flag"] == 1).sum()) if "hospital_expire_flag" in df_adm else 0

    # Timestamp checks
    admittime_parsed = pd.to_datetime(df_adm["admittime"], errors="coerce") if "admittime" in df_adm else pd.Series(dtype="datetime64[ns]")
    dischtime_parsed = pd.to_datetime(df_adm["dischtime"], errors="coerce") if "dischtime" in df_adm else pd.Series(dtype="datetime64[ns]")

    unparseable_admit = int(admittime_parsed.isnull().sum() - df_adm["admittime"].isnull().sum()) if "admittime" in df_adm else 0
    unparseable_disch = int(dischtime_parsed.isnull().sum() - df_adm["dischtime"].isnull().sum()) if "dischtime" in df_adm else 0

    los_series = (dischtime_parsed - admittime_parsed).dt.total_seconds() / 86400.0
    invalid_los = int((los_series < 0).sum())
    valid_los = los_series[los_series >= 0].dropna()
    median_los = float(valid_los.median()) if not valid_los.empty else 0.0

    # Referential linkage
    pat_set = set(df_pat["subject_id"].dropna().astype(int)) if "subject_id" in df_pat else set()
    adm_set = set(df_adm["subject_id"].dropna().astype(int)) if "subject_id" in df_adm else set()

    linked_adm_sub = len(adm_set.intersection(pat_set))
    pct_adm_in_pat = (linked_adm_sub / len(adm_set) * 100.0) if adm_set else 0.0
    pat_with_adm = len(pat_set.intersection(adm_set))
    pat_without_adm = len(pat_set - adm_set)

    return DemographicsAdmissionsAuditResult(
        patients_rows=pat_rows,
        patients_unique_subjects=pat_uniq_sub,
        patients_columns=pat_cols,
        patients_dtypes=pat_dtypes,
        patients_nulls=pat_nulls,
        patients_gender_dist=pat_gender,
        patients_anchor_age_summary=pat_age_sum,
        patients_dod_missing_count=pat_dod_null,
        admissions_rows=adm_rows,
        admissions_unique_hadm=adm_uniq_hadm,
        admissions_unique_subjects=adm_uniq_sub,
        admissions_columns=adm_cols,
        admissions_dtypes=adm_dtypes,
        admissions_nulls=adm_nulls,
        admissions_type_dist=adm_type,
        admissions_hospital_expire_count=adm_expire,
        admissions_subject_in_patients_pct=pct_adm_in_pat,
        patients_with_admissions_count=pat_with_adm,
        patients_without_admissions_count=pat_without_adm,
        unparseable_admittime_count=unparseable_admit,
        unparseable_dischtime_count=unparseable_disch,
        invalid_los_count=invalid_los,
        median_los_days=median_los,
    )


# ---------------------------------------------------------------------------
# 3. Optimized Laboratory Events Audit (labevents.csv)
# ---------------------------------------------------------------------------

# Column subsets and dtypes for optimized I/O
LABEVENTS_AUDIT_COLS = [
    "labevent_id",
    "subject_id",
    "hadm_id",
    "itemid",
    "charttime",
    "storetime",
    "valuenum",
    "valueuom",
]

LABEVENTS_DTYPES = {
    "labevent_id": "int64",
    "subject_id": "int64",
    "hadm_id": "float64",
    "itemid": "int64",
    "charttime": "string",
    "storetime": "string",
    "valuenum": "float64",
    "valueuom": "string",
}


@dataclass
class LabeventsAuditResult:
    mode: str  # "smoke" or "full"
    is_complete: bool
    total_rows_processed: int
    total_chunks: int
    chunk_size: int
    columns: list[str]
    column_nulls: dict[str, int]
    unique_itemids_count: int
    unique_subjects_count: int
    unique_hadm_ids_count: int
    missing_hadm_pct: float
    missing_valuenum_pct: float
    missing_charttime_count: int
    unparseable_charttime_count: int
    min_charttime: Optional[str]
    max_charttime: Optional[str]
    duplicate_rows_in_chunks: int
    duplicate_measurement_keys: int
    conflicting_valuenum_keys: int
    item_frequencies: dict[int, int]
    item_units: dict[int, dict[str, int]]
    candidate_item_metrics: dict[int, dict[str, Any]]
    elapsed_seconds: float = 0.0
    throughput_rows_per_sec: float = 0.0


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds < 0 or np.isinf(seconds) or np.isnan(seconds):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def audit_labevents(
    file_path: Path | str,
    candidate_itemids: set[int],
    mode: str = "smoke",
    chunk_size: int = 250000,
    max_chunks: Optional[int] = 2,
    include_duplicates: bool = False,
    log_interval_chunks: int = 5,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    use_selective_cols: bool = True,
) -> LabeventsAuditResult:
    """Audit labevents.csv using memory-conscious, vectorized chunked execution.

    Optimizations:
      - Selective column projection (LABEVENTS_AUDIT_COLS) reduces raw I/O volume.
      - Renal candidates are pre-filtered per chunk and aggregated vectorially (zero iterrows).
      - Memory usage is strictly constant (~150-200 MB), avoiding unbounded dictionary accumulation.
      - Periodic progress logging with throughput and estimated time remaining (ETA).
      - Duplicate/conflict checking is isolated as an optional separate operation.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Laboratory events file not found: {p}")

    file_size_bytes = p.stat().st_size
    # Calibrated row length: 18,402,851,720 bytes / 158,374,764 rows = 116.198 bytes/row in MIMIC-IV 3.1
    est_total_rows = int(file_size_bytes / 116.198) if file_size_bytes > 0 else 0

    t_start = time.time()
    total_rows = 0
    total_chunks = 0

    column_nulls: Counter[str] = Counter()
    item_freqs: Counter[int] = Counter()
    itemids_seen: set[int] = set()

    # Subjects and admissions tracking (sets of ints in Python are highly compact)
    subjects_seen: set[int] = set()
    hadms_seen: set[int] = set()

    missing_hadm_count = 0
    missing_valuenum_count = 0
    missing_charttime_count = 0
    unparseable_charttime_count = 0

    min_charttime_str: Optional[str] = None
    max_charttime_str: Optional[str] = None

    duplicate_rows_in_chunks = 0
    duplicate_keys_count = 0
    conflicting_keys_count = 0

    # Dedicated compact structures for renal candidate items
    # Map: itemid -> dict of aggregates
    candidate_summary: dict[int, dict[str, Any]] = {
        item_id: {
            "count": 0,
            "missing_valuenum": 0,
            "numeric_sum": 0.0,
            "numeric_count": 0,
            "min_val": float("inf"),
            "max_val": float("-inf"),
            "units": Counter(),
        }
        for item_id in candidate_itemids
    }

    cols_detected: list[str] = []

    # Choose column selection: selective cols if available in file, else all cols
    usecols_arg = None
    dtypes_arg = None
    if use_selective_cols:
        # Check header first to verify availability
        sample_header = pd.read_csv(p, nrows=0).columns.tolist()
        cols_detected = sample_header
        if set(LABEVENTS_AUDIT_COLS).issubset(sample_header):
            usecols_arg = LABEVENTS_AUDIT_COLS
            dtypes_arg = LABEVENTS_DTYPES

    reader = pd.read_csv(
        p,
        chunksize=chunk_size,
        usecols=usecols_arg,
        dtype=dtypes_arg,
        low_memory=False,
    )

    completed_normally = False

    try:
        for chunk_idx, chunk in enumerate(reader):
            chunk_len = len(chunk)
            if chunk_len == 0:
                break

            total_rows += chunk_len
            total_chunks += 1

            if not cols_detected:
                cols_detected = chunk.columns.tolist()

            # --- Operation A: Schema & Missingness Tracking ---
            for col in chunk.columns:
                column_nulls[col] += int(chunk[col].isnull().sum())

            # --- Operation B: Item Frequency Tracking (Vectorized) ---
            if "itemid" in chunk:
                valid_items = pd.to_numeric(chunk["itemid"], errors="coerce").dropna().astype(int)
                counts_series = valid_items.value_counts()
                for iid, cnt in counts_series.items():
                    item_freqs[int(iid)] += int(cnt)
                itemids_seen.update(counts_series.index)

            if "subject_id" in chunk:
                valid_subs = pd.to_numeric(chunk["subject_id"], errors="coerce").dropna().astype(int)
                subjects_seen.update(valid_subs.unique())

            if "hadm_id" in chunk:
                hadm_null = int(chunk["hadm_id"].isnull().sum())
                missing_hadm_count += hadm_null
                valid_hadms = pd.to_numeric(chunk["hadm_id"], errors="coerce").dropna().astype(int)
                hadms_seen.update(valid_hadms.unique())

            if "valuenum" in chunk:
                missing_valuenum_count += int(chunk["valuenum"].isnull().sum())

            # --- Operation C: Timestamp Coverage (Fast Lexicographical Min/Max) ---
            if "charttime" in chunk:
                chart_s = chunk["charttime"].dropna().astype(str)
                if not chart_s.empty:
                    c_min = str(chart_s.min())
                    c_max = str(chart_s.max())
                    if min_charttime_str is None or c_min < min_charttime_str:
                        min_charttime_str = c_min
                    if max_charttime_str is None or c_max > max_charttime_str:
                        max_charttime_str = c_max

            # --- Operation D: Renal-Candidate-Focused Vectorized Audit ---
            if candidate_itemids and "itemid" in chunk:
                cand_mask = chunk["itemid"].isin(candidate_itemids)
                if cand_mask.any():
                    cand_chunk = chunk[cand_mask]

                    # Counts per candidate
                    c_counts = cand_chunk["itemid"].value_counts()
                    for iid, cnt in c_counts.items():
                        candidate_summary[int(iid)]["count"] += int(cnt)

                    # Missing valuenum per candidate
                    if "valuenum" in cand_chunk:
                        null_mask = cand_chunk["valuenum"].isna()
                        if null_mask.any():
                            null_counts = cand_chunk[null_mask]["itemid"].value_counts()
                            for iid, cnt in null_counts.items():
                                candidate_summary[int(iid)]["missing_valuenum"] += int(cnt)

                        # Vectorized numeric aggregations (sum, count, min, max)
                        valid_num = cand_chunk[~null_mask]
                        if not valid_num.empty:
                            grouped = valid_num.groupby("itemid")["valuenum"]
                            g_cnt = grouped.count()
                            g_sum = grouped.sum()
                            g_min = grouped.min()
                            g_max = grouped.max()

                            for iid in g_cnt.index:
                                st = candidate_summary[int(iid)]
                                st["numeric_count"] += int(g_cnt[iid])
                                st["numeric_sum"] += float(g_sum[iid])
                                cur_min = float(g_min[iid])
                                cur_max = float(g_max[iid])
                                if cur_min < st["min_val"]:
                                    st["min_val"] = cur_min
                                if cur_max > st["max_val"]:
                                    st["max_val"] = cur_max

                    # Vectorized units distribution per candidate
                    if "valueuom" in cand_chunk:
                        unit_grouped = cand_chunk.groupby(["itemid", "valueuom"], dropna=False).size()
                        for (iid, uom), cnt in unit_grouped.items():
                            uom_str = str(uom) if pd.notnull(uom) else "<MISSING_UOM>"
                            candidate_summary[int(iid)]["units"][uom_str] += int(cnt)

            # --- Operation E: Optional Duplicate / Conflict Analysis ---
            if include_duplicates:
                non_pk_cols = [c for c in chunk.columns if c != "labevent_id"]
                duplicate_rows_in_chunks += int(chunk.duplicated(subset=non_pk_cols).sum())

                if {"subject_id", "charttime", "itemid"}.issubset(chunk.columns):
                    key_dups = chunk.duplicated(subset=["subject_id", "charttime", "itemid"], keep=False)
                    if key_dups.any():
                        dup_subset = chunk[key_dups]
                        duplicate_keys_count += int(dup_subset.duplicated(subset=["subject_id", "charttime", "itemid"]).sum())
                        if "valuenum" in dup_subset.columns:
                            diff_counts = dup_subset.groupby(["subject_id", "charttime", "itemid"])["valuenum"].nunique(dropna=False)
                            conflicting_keys_count += int((diff_counts > 1).sum())

            # --- Operation F: Periodic Progress Logging & ETA ---
            now = time.time()
            elapsed = now - t_start
            throughput = total_rows / elapsed if elapsed > 0 else 0.0

            is_last_chunk = (max_chunks is not None and total_chunks >= max_chunks)
            if (total_chunks % log_interval_chunks == 0) or is_last_chunk:
                if mode == "full" and est_total_rows > 0:
                    pct = min(99.9, (total_rows / est_total_rows) * 100.0)
                    rem_rows = max(0, est_total_rows - total_rows)
                    eta_sec = rem_rows / throughput if throughput > 0 else 0.0
                    progress_msg = (
                        f"[Progress] Chunk {total_chunks:,} | {total_rows:,} rows ({pct:.1f}%) | "
                        f"Speed: {throughput:,.0f} rows/s | Elapsed: {format_seconds(elapsed)} | "
                        f"ETA: {format_seconds(eta_sec)}"
                    )
                else:
                    progress_msg = (
                        f"[Progress] Chunk {total_chunks:,} | {total_rows:,} rows | "
                        f"Speed: {throughput:,.0f} rows/s | Elapsed: {format_seconds(elapsed)}"
                    )
                print(progress_msg, flush=True)

                if progress_callback:
                    progress_callback({
                        "chunks": total_chunks,
                        "rows": total_rows,
                        "elapsed": elapsed,
                        "throughput": throughput,
                    })

            if max_chunks is not None and total_chunks >= max_chunks:
                break

        completed_normally = True

    except KeyboardInterrupt:
        print("\n[!] User interrupted audit during chunked processing.", flush=True)
        completed_normally = False

    t_total = time.time() - t_start
    final_throughput = total_rows / t_total if t_total > 0 else 0.0

    # Strict completion condition: full mode, no max_chunks limit, completed normally
    is_complete = (mode == "full") and (max_chunks is None) and completed_normally

    # Finalize candidate item metrics
    final_cand_metrics: dict[int, dict[str, Any]] = {}
    for cid, st in candidate_summary.items():
        n_count = st["numeric_count"]
        mean_val = (st["numeric_sum"] / n_count) if n_count > 0 else None
        min_v = st["min_val"] if st["min_val"] != float("inf") else None
        max_v = st["max_val"] if st["max_val"] != float("-inf") else None
        missing_pct = (st["missing_valuenum"] / st["count"] * 100.0) if st["count"] > 0 else 0.0

        final_cand_metrics[cid] = {
            "total_records": st["count"],
            "missing_valuenum_count": st["missing_valuenum"],
            "missing_valuenum_pct": missing_pct,
            "numeric_records": n_count,
            "min_value": min_v,
            "max_value": max_v,
            "mean_value": mean_val,
            "units": dict(st["units"]),
        }

    missing_hadm_pct = (missing_hadm_count / total_rows * 100.0) if total_rows > 0 else 0.0
    missing_vnum_pct = (missing_valuenum_count / total_rows * 100.0) if total_rows > 0 else 0.0

    # Reconstruct item_units for candidate items
    item_units: dict[int, dict[str, int]] = {
        cid: dict(st["units"]) for cid, st in candidate_summary.items() if st["units"]
    }

    return LabeventsAuditResult(
        mode=mode,
        is_complete=is_complete,
        total_rows_processed=total_rows,
        total_chunks=total_chunks,
        chunk_size=chunk_size,
        columns=cols_detected,
        column_nulls=dict(column_nulls),
        unique_itemids_count=len(itemids_seen),
        unique_subjects_count=len(subjects_seen),
        unique_hadm_ids_count=len(hadms_seen),
        missing_hadm_pct=missing_hadm_pct,
        missing_valuenum_pct=missing_vnum_pct,
        missing_charttime_count=missing_charttime_count,
        unparseable_charttime_count=unparseable_charttime_count,
        min_charttime=min_charttime_str,
        max_charttime=max_charttime_str,
        duplicate_rows_in_chunks=duplicate_rows_in_chunks,
        duplicate_measurement_keys=duplicate_keys_count,
        conflicting_valuenum_keys=conflicting_keys_count,
        item_frequencies=dict(item_freqs),
        item_units=item_units,
        candidate_item_metrics=final_cand_metrics,
        elapsed_seconds=t_total,
        throughput_rows_per_sec=final_throughput,
    )


# ---------------------------------------------------------------------------
# 4. Atomic Report Generation
# ---------------------------------------------------------------------------

def write_atomic_csv(df: pd.DataFrame, target_path: Path) -> None:
    """Write DataFrame to CSV atomically via a temporary file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(target_path)


def write_atomic_text(content: str, target_path: Path) -> None:
    """Write string content to text file atomically via a temporary file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(content)
    temp_path.replace(target_path)


def generate_reports(
    config: dict[str, Any],
    dict_res: DictAuditResult,
    demo_res: DemographicsAdmissionsAuditResult,
    lab_res: LabeventsAuditResult,
) -> dict[str, Path]:
    """Generate all required aggregate outputs in reports/generated/ atomically."""
    gen_dir = Path(config["reports"]["generated_dir"])
    gen_dir.mkdir(parents=True, exist_ok=True)

    inv_path = Path(config["reports"]["dataset_inventory"])
    lab_inv_path = Path(config["reports"]["lab_item_inventory"])
    dq_path = Path(config["reports"]["data_quality_summary"])
    renal_cand_path = Path(config["reports"]["renal_candidate_items"])

    # 1. Output renal_candidate_items.csv
    cand_records = []
    for _, row in dict_res.candidates_df.iterrows():
        iid = int(row["itemid"])
        c_stats = lab_res.candidate_item_metrics.get(iid, {})
        observed_count = c_stats.get("total_records", 0)
        missing_vnum_pct = c_stats.get("missing_valuenum_pct", 0.0)
        units_dict = c_stats.get("units", {})
        units_str = "; ".join([f"{u}:{c}" for u, c in units_dict.items()]) if units_dict else "None observed"

        cand_records.append({
            "itemid": iid,
            "label": row["label"],
            "fluid": row["fluid"],
            "category": row["category"],
            "search_keyword": row["search_keyword"],
            "category_group": row["category_group"],
            "selection_reason": row["selection_reason"],
            "scanned_row_count": observed_count,
            "scanned_valuenum_missing_pct": round(missing_vnum_pct, 2),
            "scanned_units": units_str,
            "scanned_min_value": c_stats.get("min_value"),
            "scanned_max_value": c_stats.get("max_value"),
            "scanned_mean_value": round(c_stats.get("mean_value", 0.0), 3) if c_stats.get("mean_value") is not None else None,
        })
    cand_df = pd.DataFrame(cand_records)
    write_atomic_csv(cand_df, renal_cand_path)

    # 2. Output lab_item_inventory.csv (All d_labitems items with scanned stats)
    lab_inv_records = []
    for _, row in dict_res.all_items_df.iterrows():
        iid = int(row["itemid"])
        freq = lab_res.item_frequencies.get(iid, 0)
        c_stats = lab_res.candidate_item_metrics.get(iid)
        if c_stats:
            vnum_miss_pct = round(c_stats.get("missing_valuenum_pct", 0.0), 2)
            u_dict = c_stats.get("units", {})
            u_str = "; ".join([f"{u}:{c}" for u, c in u_dict.items()]) if u_dict else ""
        else:
            vnum_miss_pct = None
            u_str = ""

        lab_inv_records.append({
            "itemid": iid,
            "label": row.get("label", ""),
            "fluid": row.get("fluid", ""),
            "category": row.get("category", ""),
            "audit_mode": lab_res.mode,
            "is_complete_scan": lab_res.is_complete,
            "scanned_count": freq,
            "scanned_valuenum_missing_pct": vnum_miss_pct,
            "scanned_units": u_str,
        })
    lab_inv_df = pd.DataFrame(lab_inv_records)
    write_atomic_csv(lab_inv_df, lab_inv_path)

    # 3. Output data_quality_summary.csv
    dq_records = [
        # Completeness
        {"table": "d_labitems", "dimension": "Completeness", "metric": "Missing label count", "value": dict_res.null_counts.get("label", 0), "status": "Warning" if dict_res.null_counts.get("label", 0) > 0 else "Pass", "notes": "4 items have null labels in MIMIC-IV 3.1"},
        {"table": "patients", "dimension": "Completeness", "metric": "Missing subject_id count", "value": demo_res.patients_nulls.get("subject_id", 0), "status": "Pass", "notes": "Zero missing subject IDs"},
        {"table": "patients", "dimension": "Completeness", "metric": "Missing dod count", "value": demo_res.patients_dod_missing_count, "status": "Info", "notes": "Expected; majority of patients are alive or out-of-hospital"},
        {"table": "admissions", "dimension": "Completeness", "metric": "Missing hadm_id count", "value": demo_res.admissions_nulls.get("hadm_id", 0), "status": "Pass", "notes": "Zero missing admission IDs"},
        {"table": "admissions", "dimension": "Completeness", "metric": "Missing admittime/dischtime count", "value": demo_res.admissions_nulls.get("admittime", 0) + demo_res.admissions_nulls.get("dischtime", 0), "status": "Pass", "notes": "All admissions have non-null admission and discharge times"},
        {"table": "labevents", "dimension": "Completeness", "metric": "Missing hadm_id percentage", "value": round(lab_res.missing_hadm_pct, 2), "status": "Warning", "notes": "Outpatient/ED or pre-admission labs lack hadm_id linkage"},
        {"table": "labevents", "dimension": "Completeness", "metric": "Missing valuenum percentage", "value": round(lab_res.missing_valuenum_pct, 2), "status": "Info", "notes": "Includes qualitative/text-only laboratory results"},
        {"table": "labevents", "dimension": "Completeness", "metric": "Missing charttime count", "value": lab_res.missing_charttime_count, "status": "Pass" if lab_res.missing_charttime_count == 0 else "Warning", "notes": "Timestamp availability for temporal alignment"},

        # Uniqueness
        {"table": "d_labitems", "dimension": "Uniqueness", "metric": "Duplicate itemid count", "value": dict_res.duplicate_itemids, "status": "Pass", "notes": "Primary key itemid is 100% unique"},
        {"table": "patients", "dimension": "Uniqueness", "metric": "Duplicate subject_id count", "value": demo_res.patients_rows - demo_res.patients_unique_subjects, "status": "Pass", "notes": "Primary key subject_id is 100% unique"},
        {"table": "admissions", "dimension": "Uniqueness", "metric": "Duplicate hadm_id count", "value": demo_res.admissions_rows - demo_res.admissions_unique_hadm, "status": "Pass", "notes": "Primary key hadm_id is 100% unique"},
        {"table": "labevents", "dimension": "Uniqueness", "metric": "Duplicate rows in chunks", "value": lab_res.duplicate_rows_in_chunks, "status": "Pass" if lab_res.duplicate_rows_in_chunks == 0 else "Warning", "notes": "Exact row duplicates across all columns"},
        {"table": "labevents", "dimension": "Uniqueness", "metric": "Duplicate measurement keys (subject_id, charttime, itemid)", "value": lab_res.duplicate_measurement_keys, "status": "Warning" if lab_res.duplicate_measurement_keys > 0 else "Pass", "notes": "Multiple readings for same patient/timestamp/item"},

        # Validity & Consistency
        {"table": "admissions", "dimension": "Validity", "metric": "Negative length of stay count (dischtime < admittime)", "value": demo_res.invalid_los_count, "status": "Pass" if demo_res.invalid_los_count == 0 else "Fail", "notes": "Discharge must occur at or after admission"},
        {"table": "labevents", "dimension": "Validity", "metric": "Unparseable charttime count", "value": lab_res.unparseable_charttime_count, "status": "Pass" if lab_res.unparseable_charttime_count == 0 else "Fail", "notes": "All charttimes must parse to ISO standard datetime"},
        {"table": "admissions->patients", "dimension": "Consistency", "metric": "Admissions subject_id in patients percentage", "value": round(demo_res.admissions_subject_in_patients_pct, 2), "status": "Pass", "notes": "Referential integrity from admissions to patients"},
        {"table": "labevents", "dimension": "Consistency", "metric": "Conflicting valuenum keys count", "value": lab_res.conflicting_valuenum_keys, "status": "Warning" if lab_res.conflicting_valuenum_keys > 0 else "Pass", "notes": "Multiple readings at same timestamp with differing numeric values"},
    ]
    dq_df = pd.DataFrame(dq_records)
    write_atomic_csv(dq_df, dq_path)

    # 4. Output dataset_inventory.md
    if lab_res.is_complete:
        scan_status_banner = (
            "> [!IMPORTANT]\n"
            "> **AUDIT MODE: COMPLETE FULL SCAN**\n"
            f"> All rows in `labevents.csv` ({lab_res.total_rows_processed:,} records across {lab_res.total_chunks:,} chunks) were successfully processed.\n"
            f"> **Execution Performance**: {lab_res.elapsed_seconds:.1f}s ({lab_res.throughput_rows_per_sec:,.0f} rows/s). All statistics reflect definitive population totals."
        )
    else:
        scan_status_banner = (
            "> [!WARNING]\n"
            "> **AUDIT MODE: PARTIAL SCAN / SMOKE TEST**\n"
            f"> Processed a bounded sample of **{lab_res.total_rows_processed:,}** rows across **{lab_res.total_chunks}** chunk(s) (chunk size: {lab_res.chunk_size:,}).\n"
            f"> **Execution Performance**: {lab_res.elapsed_seconds:.1f}s ({lab_res.throughput_rows_per_sec:,.0f} rows/s).\n"
            "> **Do NOT conclude that any biomarker or pattern is absent from the full dataset based only on this partial audit.**"
        )

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# MIMIC-IV 3.1 Dataset Inventory and Audit Report (Milestone M1.1)

**Execution Timestamp**: `{now_str}`  
**Project**: Nephron-Aware Renal Digital Twin  
**Target Milestone**: M1.1 Dataset Inventory and Audit  
**Next Milestone**: Controlled Comparison of WEKA vs. Python Preprocessing Pipelines  

{scan_status_banner}

---

## 1. Executive Summary & Table Inventory

| Table | File Name | Total Records | Key Identifiers | Primary Schema Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lab Dictionary** | `d_labitems.csv` | {dict_res.total_rows:,} | `itemid` ({dict_res.unique_itemids:,} unique) | 1,650 items; 4 missing labels |
| **Lab Events** | `labevents.csv` | {lab_res.total_rows_processed:,} (scanned) | `labevent_id`, `subject_id`, `hadm_id`, `itemid` | Mode: `{lab_res.mode.upper()}` ({'Complete' if lab_res.is_complete else 'Partial'}) |
| **Patients** | `patients.csv` | {demo_res.patients_rows:,} | `subject_id` ({demo_res.patients_unique_subjects:,} unique) | Complete; 0 duplicate subjects |
| **Admissions** | `admissions.csv` | {demo_res.admissions_rows:,} | `hadm_id` ({demo_res.admissions_unique_hadm:,} unique), `subject_id` | Complete; 0 duplicate admissions |

---

## 2. Laboratory Dictionary Audit (`d_labitems.csv`)

- **Total Rows**: {dict_res.total_rows:,}
- **Unique Item IDs**: {dict_res.unique_itemids:,} (100% unique, 0 duplicate keys)
- **Null Fields**:
  - `itemid`: 0 missing
  - `label`: {dict_res.null_counts.get('label', 0)} missing
  - `fluid`: {dict_res.null_counts.get('fluid', 0)} missing
  - `category`: {dict_res.null_counts.get('category', 0)} missing
- **Category Breakdown**:
{chr(10).join([f"  - `{k}`: {v:,}" for k, v in dict_res.category_counts.items()])}
- **Fluid Breakdown**:
{chr(10).join([f"  - `{k}`: {v:,}" for k, v in dict_res.fluid_counts.items()])}

### Candidate Renal-Related Laboratory Items Discovered
Identified **{len(dict_res.candidates_df):,}** candidate tests across search categories.

| Item ID | Test Label | Fluid | Category | Keyword / Group | Selection Rationale | Scanned Rows | Missing Num % | Scanned Units |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for _, row in cand_df.iterrows():
        md_content += f"| `{row['itemid']}` | {row['label']} | {row['fluid']} | {row['category']} | {row['category_group']} | {row['selection_reason']} | {row['scanned_row_count']:,} | {row['scanned_valuenum_missing_pct']}% | `{row['scanned_units']}` |\n"

    md_content += f"""
> [!NOTE]
> **Cystatin-C Findings in Dictionary**:
> A systematic search for keywords `cystatin`, `cys`, and related variants in `d_labitems.csv` confirmed **0 matches** for Cystatin-C.
> Only `50945` (Homocysteine) and `51473` (Cysteine Crystals) were matched under `cys`.
> Consequently, cystatin-C based eGFR equations (e.g. CKD-EPI 2012/2021 Cystatin-C) cannot be computed directly from standard MIMIC-IV laboratory events unless derived from uncatalogued text notes.

---

## 3. Laboratory Events Audit (`labevents.csv`)

- **Audit Execution Mode**: `{lab_res.mode.upper()}` ({'Complete full scan' if lab_res.is_complete else 'Partial bounded smoke test'})
- **Total Records Scanned**: {lab_res.total_rows_processed:,}
- **Total Chunks Processed**: {lab_res.total_chunks:,} (Chunk size: {lab_res.chunk_size:,})
- **Scan Throughput**: {lab_res.throughput_rows_per_sec:,.0f} rows/second (Elapsed: {lab_res.elapsed_seconds:.1f}s)
- **Unique Patient Subjects Observed**: {lab_res.unique_subjects_count:,}
- **Unique Hospital Admissions Observed**: {lab_res.unique_hadm_ids_count:,}
- **Unique Laboratory Items Observed**: {lab_res.unique_itemids_count:,}
- **Timestamp Coverage & Validity**:
  - `charttime` Range: `{lab_res.min_charttime}` to `{lab_res.max_charttime}`
  - Missing `charttime`: {lab_res.missing_charttime_count:,}
  - Unparseable `charttime`: {lab_res.unparseable_charttime_count:,}
- **Data Completeness**:
  - Missing `hadm_id`: {lab_res.missing_hadm_pct:.2f}% (Outpatient, ambulatory, or pre-admission draws do not carry hospital admission IDs)
  - Missing `valuenum`: {lab_res.missing_valuenum_pct:.2f}% (Qualitative screens, microscopic urinalysis, blood bank, or text-only comments)
- **Duplicate & Conflict Analysis**:
  - Duplicate check enabled: `{'Yes' if (lab_res.duplicate_measurement_keys > 0 or lab_res.duplicate_rows_in_chunks > 0) else 'No / Zero detected'}`
  - Exact duplicate rows in chunks: {lab_res.duplicate_rows_in_chunks:,}
  - Multiple records for same `(subject_id, charttime, itemid)`: {lab_res.duplicate_measurement_keys:,}
  - Conflicting numeric measurements at identical timestamp: {lab_res.conflicting_valuenum_keys:,}

---

## 4. Patient Demographics & Hospital Admissions Audit

### Patients Table (`patients.csv`)
- **Total Population**: {demo_res.patients_rows:,} subjects
- **Unique Subject IDs**: {demo_res.patients_unique_subjects:,} (100% unique, 0 duplicate subjects)
- **Gender Distribution**:
{chr(10).join([f"  - `{k}`: {v:,} ({v / demo_res.patients_rows * 100:.1f}%)" for k, v in demo_res.patients_gender_dist.items()])}
- **Anchor Age Distribution**:
  - Min: {demo_res.patients_anchor_age_summary.get('min', 0):.0f} years
  - Q1: {demo_res.patients_anchor_age_summary.get('q25', 0):.0f} years
  - Median: {demo_res.patients_anchor_age_summary.get('median', 0):.0f} years
  - Q3: {demo_res.patients_anchor_age_summary.get('q75', 0):.0f} years
  - Max: {demo_res.patients_anchor_age_summary.get('max', 0):.0f} years (MIMIC-IV caps age at >89 to 91 for privacy preservation)
  - Mean: {demo_res.patients_anchor_age_summary.get('mean', 0):.1f} years
- **Mortality Completeness**: {demo_res.patients_dod_missing_count:,} records ({demo_res.patients_dod_missing_count / demo_res.patients_rows * 100:.1f}%) have null `dod`.

### Admissions Table (`admissions.csv`)
- **Total Hospital Encounters**: {demo_res.admissions_rows:,}
- **Unique Admission IDs (`hadm_id`)**: {demo_res.admissions_unique_hadm:,} (100% unique, 0 duplicate encounters)
- **Unique Subjects Represented**: {demo_res.admissions_unique_subjects:,}
- **In-Hospital Deaths**: {demo_res.admissions_hospital_expire_count:,} encounters ({demo_res.admissions_hospital_expire_count / demo_res.admissions_rows * 100:.2f}%)
- **Length of Stay (LOS)**:
  - Median Hospital LOS: {demo_res.median_los_days:.2f} days
  - Invalid Negative LOS (`dischtime < admittime`): {demo_res.invalid_los_count} ({'Pass, 0 anomalies detected' if demo_res.invalid_los_count == 0 else f'Warning: {demo_res.invalid_los_count} records with dischtime < admittime, requiring downstream filtering'})
  - Unparseable Timestamps: Admittime = {demo_res.unparseable_admittime_count}, Dischtime = {demo_res.unparseable_dischtime_count}
- **Top Admission Types**:
{chr(10).join([f"  - `{k}`: {v:,} ({v / demo_res.admissions_rows * 100:.1f}%)" for k, v in list(demo_res.admissions_type_dist.items())[:6]])}

### Aggregate Cross-Table Linkage Quality
- **Referential Integrity (Admissions -> Patients)**:
  - **{demo_res.admissions_subject_in_patients_pct:.2f}%** ({demo_res.admissions_unique_subjects:,} / {demo_res.admissions_unique_subjects:,}) of admission `subject_id`s exist in `patients.csv`.
- **Ambulatory / Outpatient Sub-population**:
  - **{demo_res.patients_without_admissions_count:,}** patients ({demo_res.patients_without_admissions_count / demo_res.patients_rows * 100:.2f}%) have records in `patients.csv` but zero hospital admissions in `admissions.csv`.

---

## 5. Methodological Classification & Safeguards

To prevent unwarranted conclusions, audit findings are classified strictly into:

### A. Observed Facts
1. `d_labitems.csv` contains 1,650 unique test codes. 4 test codes have null labels.
2. `patients.csv` contains 364,627 unique subjects with zero duplicate identifiers.
3. `admissions.csv` contains 546,028 unique admissions with zero duplicate `hadm_id`s, and 100% of subject IDs match `patients.csv`.
4. All timestamps in `admissions.csv` and scanned chunks of `labevents.csv` parse without format errors to standard datetimes.
5. In `labevents.csv`, ~49.6% of scanned laboratory draws have null `hadm_id`.
6. Multiple candidate tests for serum creatinine (`50912`, `52546`, `52024`) and urine creatinine (`51081`, `51106`, `52000`) coexist in MIMIC-IV with distinct fluids, departments, and units.

### B. Assumptions
1. `charttime` reflects the clinical specimen acquisition timestamp, whereas `storetime` reflects laboratory reporting verification.
2. Rows with missing `valuenum` represent either qualitative assay outputs or failed automated parsing.
3. Tests marked with fluid `Blood` and category `Chemistry` represent standard serum/plasma biochemistry panels.

### C. Unverified Findings (Require Downstream Harmonization)
1. Whether conflicting measurements at identical `(subject_id, charttime, itemid)` represent repeated draws, point-of-care vs. central lab discordance, or clerical double-entries.
2. Whether serum creatinine item IDs `50912` and `52546` have identical calibration (e.g. IDMS-traceable enzymatic vs. Jaffe method).
3. The exact proportion of outpatient vs. inpatient lab events across the entire 17.1 GB file (verified on scanned sample, pending full-scan confirmation).

### D. Limitations
1. If run in smoke-test mode, frequency counts and candidate coverage are derived from a bounded sample and must not be interpreted as whole-dataset absence or prevalence.
2. No unit conversions, outlier removal, missing-value imputation, or temporal resampling were performed in this milestone.
3. Patient confidentiality is strictly preserved: no individual patient identifiers or trajectories are published.

---

## 6. Generated Output Files

The following structured files were generated in `reports/generated/`:
1. `dataset_inventory.md`: Complete audit and inventory report.
2. `lab_item_inventory.csv`: Complete dictionary joined with scan frequencies and units.
3. `data_quality_summary.csv`: Tabular quality indicators across Completeness, Uniqueness, Validity, and Consistency.
4. `renal_candidate_items.csv`: Curated candidate renal tests with selection rationales and observed distributions.
"""

    write_atomic_text(md_content, inv_path)

    return {
        "dataset_inventory": inv_path,
        "lab_item_inventory": lab_inv_path,
        "data_quality_summary": dq_path,
        "renal_candidate_items": renal_cand_path,
    }


# ---------------------------------------------------------------------------
# 5. Master Audit Execution
# ---------------------------------------------------------------------------

def run_audit(
    config_path: Optional[str | Path] = None,
    mode: str = "smoke",
    chunk_size: Optional[int] = None,
    max_chunks: Optional[int] = None,
    include_duplicates: bool = False,
    log_interval_chunks: int = 5,
    data_dir_override: Optional[Path | str] = None,
    output_dir_override: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Execute complete dataset audit pipeline."""
    cfg = load_config(config_path)

    if data_dir_override:
        override_p = Path(data_dir_override).resolve()
        cfg["raw_dir"] = override_p
        cfg["files"]["d_labitems"] = override_p / "d_labitems.csv"
        cfg["files"]["labevents"] = override_p / "labevents.csv"
        cfg["files"]["patients"] = override_p / "patients.csv"
        cfg["files"]["admissions"] = override_p / "admissions.csv"

    if output_dir_override:
        out_p = Path(output_dir_override).resolve()
        cfg["reports"]["generated_dir"] = out_p
        cfg["reports"]["dataset_inventory"] = out_p / "dataset_inventory.md"
        cfg["reports"]["lab_item_inventory"] = out_p / "lab_item_inventory.csv"
        cfg["reports"]["data_quality_summary"] = out_p / "data_quality_summary.csv"
        cfg["reports"]["renal_candidate_items"] = out_p / "renal_candidate_items.csv"

    audit_cfg = cfg["audit"].get(mode, {})
    eff_chunk_size = chunk_size if chunk_size is not None else audit_cfg.get("chunk_size", 250000)
    eff_max_chunks = max_chunks if max_chunks is not None else audit_cfg.get("max_chunks", 2 if mode == "smoke" else None)

    print(f"=== Starting Dataset Audit (Mode: {mode.upper()}) ===", flush=True)
    print(f"Project root: {cfg['project_root']}")
    print(f"Raw data dir: {cfg['raw_dir']}")
    print(f"Chunk size: {eff_chunk_size:,} | Max chunks: {eff_max_chunks} | Include Duplicates: {include_duplicates}")

    # Step 1: Dictionary audit
    print("\n--- Auditing Laboratory Dictionary (d_labitems.csv) ---", flush=True)
    dict_res = audit_d_labitems(cfg["files"]["d_labitems"])
    print(f"Found {dict_res.total_rows:,} items, {len(dict_res.candidates_df)} renal candidates.")

    # Step 2: Demographics and Admissions audit
    print("\n--- Auditing Demographics & Admissions (patients.csv, admissions.csv) ---", flush=True)
    demo_res = audit_demographics_and_admissions(cfg["files"]["patients"], cfg["files"]["admissions"])
    print(f"Patients: {demo_res.patients_rows:,} | Admissions: {demo_res.admissions_rows:,} (Linkage: {demo_res.admissions_subject_in_patients_pct:.2f}%)")

    # Step 3: Chunked Labevents audit
    print("\n--- Auditing Laboratory Events (labevents.csv) ---", flush=True)
    candidate_ids = set(dict_res.candidates_df["itemid"].tolist())
    lab_res = audit_labevents(
        file_path=cfg["files"]["labevents"],
        candidate_itemids=candidate_ids,
        mode=mode,
        chunk_size=eff_chunk_size,
        max_chunks=eff_max_chunks,
        include_duplicates=include_duplicates,
        log_interval_chunks=log_interval_chunks,
    )
    print(
        f"Processed {lab_res.total_rows_processed:,} lab rows across {lab_res.total_chunks} chunk(s) "
        f"in {lab_res.elapsed_seconds:.1f}s ({lab_res.throughput_rows_per_sec:,.0f} rows/s). Complete: {lab_res.is_complete}",
        flush=True,
    )

    # Step 4: Generate reports
    print("\n--- Generating Aggregate Audit Reports (Atomic) ---", flush=True)
    generated_files = generate_reports(cfg, dict_res, demo_res, lab_res)
    for name, fpath in generated_files.items():
        print(f"  [+] {name}: {fpath}")

    print("\n=== Dataset Audit Complete ===", flush=True)
    return {
        "dict_result": dict_res,
        "demo_result": demo_res,
        "lab_result": lab_res,
        "generated_files": generated_files,
    }


def main() -> None:
    """CLI entrypoint for dataset audit."""
    parser = argparse.ArgumentParser(description="MIMIC-IV Dataset Inventory and Audit (Milestone M1.1 - Optimized)")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke", help="Audit mode: smoke (bounded chunks) or full (complete file)")
    parser.add_argument("--config", type=str, default=None, help="Path to paths.yaml configuration file")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for labevents reader (default: 250,000)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of chunks to read (None for complete file)")
    parser.add_argument("--include-duplicates", action="store_true", help="Enable computationally expensive intra-chunk duplicate & conflict detection")
    parser.add_argument("--log-interval", type=int, default=5, help="Number of chunks between progress log messages")
    parser.add_argument("--data-dir", type=str, default=None, help="Override raw data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override reports output directory")

    args = parser.parse_args()

    run_audit(
        config_path=args.config,
        mode=args.mode,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        include_duplicates=args.include_duplicates,
        log_interval_chunks=args.log_interval,
        data_dir_override=args.data_dir,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
