"""Comprehensive unit and integration tests for dataset audit module (Milestone M1.1).

Tests use synthetic CSV fixtures to verify:
- Missing files and configuration errors
- Chunked processing behavior and chunk boundaries
- Smoke-test versus full-scan flags and labels
- Candidate renal test discovery, pre-filtering, and reason attribution
- Missing numeric values and missingness accounting
- Timestamp parseability and logical validation
- Duplicate identifier and conflict detection (isolated operation)
- Progress callback and throughput tracking
- Atomic report writing and interrupted / partial run labeling
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.ingestion.audit_dataset import (
    audit_d_labitems,
    audit_demographics_and_admissions,
    audit_labevents,
    generate_reports,
    load_config,
    run_audit,
    find_project_root,
    write_atomic_csv,
    write_atomic_text,
)


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Create a temporary synthetic MIMIC-IV dataset environment using validated DataFrames."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic d_labitems.csv
    d_labitems_df = pd.DataFrame([
        {"itemid": 50912, "label": "Creatinine", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 51006, "label": "Urea Nitrogen", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 50862, "label": "Albumin", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 50931, "label": "Glucose", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 51081, "label": "Urine Creatinine", "fluid": "Urine", "category": "Chemistry"},
        {"itemid": 51102, "label": "Total Protein, Urine", "fluid": "Urine", "category": "Chemistry"},
        {"itemid": 50920, "label": "Estimated GFR (MDRD equation)", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 50971, "label": "Potassium", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 50983, "label": "Sodium", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 50882, "label": "Bicarbonate", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 50902, "label": "Chloride", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 51221, "label": "Hematocrit", "fluid": "Blood", "category": "Hematology"},
        {"itemid": 59999, "label": None, "fluid": "Blood", "category": "Chemistry"},  # Null label test
    ])
    d_labitems_df.to_csv(data_dir / "d_labitems.csv", index=False)

    # 2. Synthetic patients.csv
    patients_df = pd.DataFrame([
        {"subject_id": 1001, "gender": "M", "anchor_age": 65, "anchor_year": 2150, "anchor_year_group": "2014 - 2016", "dod": "2152-01-01"},
        {"subject_id": 1002, "gender": "F", "anchor_age": 45, "anchor_year": 2160, "anchor_year_group": "2017 - 2019", "dod": None},
        {"subject_id": 1003, "gender": "M", "anchor_age": 72, "anchor_year": 2170, "anchor_year_group": "2014 - 2016", "dod": None},
        {"subject_id": 1004, "gender": "F", "anchor_age": 30, "anchor_year": 2180, "anchor_year_group": "2020 - 2022", "dod": None},  # Patient without admission
    ])
    patients_df.to_csv(data_dir / "patients.csv", index=False)

    # 3. Synthetic admissions.csv
    admissions_df = pd.DataFrame([
        {
            "subject_id": 1001,
            "hadm_id": 2001,
            "admittime": "2150-05-10 10:00:00",
            "dischtime": "2150-05-15 14:00:00",
            "deathtime": None,
            "admission_type": "EW EMER.",
            "admit_provider_id": "P1",
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "HOME",
            "insurance": "Medicare",
            "language": "English",
            "marital_status": "MARRIED",
            "race": "WHITE",
            "edregtime": "2150-05-10 08:00:00",
            "edouttime": "2150-05-10 11:00:00",
            "hospital_expire_flag": 0,
        },
        {
            "subject_id": 1002,
            "hadm_id": 2002,
            "admittime": "2160-03-01 12:00:00",
            "dischtime": "2160-03-04 11:00:00",
            "deathtime": None,
            "admission_type": "URGENT",
            "admit_provider_id": "P2",
            "admission_location": "PHYSICIAN REFERRAL",
            "discharge_location": "HOME",
            "insurance": "Other",
            "language": "English",
            "marital_status": "SINGLE",
            "race": "ASIAN",
            "edregtime": None,
            "edouttime": None,
            "hospital_expire_flag": 0,
        },
        {
            "subject_id": 1003,
            "hadm_id": 2003,
            "admittime": "2170-08-20 09:00:00",
            "dischtime": "2170-08-22 17:00:00",
            "deathtime": "2170-08-22 17:00:00",
            "admission_type": "DIRECT EMER.",
            "admit_provider_id": "P3",
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "DIED",
            "insurance": "Medicare",
            "language": "English",
            "marital_status": "WIDOWED",
            "race": "BLACK",
            "edregtime": None,
            "edouttime": None,
            "hospital_expire_flag": 1,
        },
    ])
    admissions_df.to_csv(data_dir / "admissions.csv", index=False)

    # 4. Synthetic labevents.csv
    labevents_df = pd.DataFrame([
        # Row 1: Valid creatinine
        {
            "labevent_id": 1,
            "subject_id": 1001,
            "hadm_id": 2001,
            "specimen_id": 501,
            "itemid": 50912,
            "order_provider_id": "P1",
            "charttime": "2150-05-10 11:00:00",
            "storetime": "2150-05-10 12:00:00",
            "value": "1.2",
            "valuenum": 1.2,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.6,
            "ref_range_upper": 1.3,
            "flag": None,
            "priority": "ROUTINE",
            "comments": None,
        },
        # Row 2: Valid BUN
        {
            "labevent_id": 2,
            "subject_id": 1001,
            "hadm_id": 2001,
            "specimen_id": 501,
            "itemid": 51006,
            "order_provider_id": "P1",
            "charttime": "2150-05-10 11:00:00",
            "storetime": "2150-05-10 12:00:00",
            "value": "18",
            "valuenum": 18.0,
            "valueuom": "mg/dL",
            "ref_range_lower": 8.0,
            "ref_range_upper": 20.0,
            "flag": None,
            "priority": "ROUTINE",
            "comments": None,
        },
        # Row 3: Outpatient lab (no hadm_id), missing valuenum (qualitative/text)
        {
            "labevent_id": 3,
            "subject_id": 1004,
            "hadm_id": None,
            "specimen_id": 502,
            "itemid": 50912,
            "order_provider_id": "P4",
            "charttime": "2180-01-15 08:30:00",
            "storetime": "2180-01-15 09:30:00",
            "value": "ERROR",
            "valuenum": None,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.6,
            "ref_range_upper": 1.3,
            "flag": None,
            "priority": "ROUTINE",
            "comments": None,
        },
        # Row 4: Duplicate measurement key with identical valuenum
        {
            "labevent_id": 4,
            "subject_id": 1001,
            "hadm_id": 2001,
            "specimen_id": 501,
            "itemid": 50912,
            "order_provider_id": "P1",
            "charttime": "2150-05-10 11:00:00",
            "storetime": "2150-05-10 12:00:00",
            "value": "1.2",
            "valuenum": 1.2,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.6,
            "ref_range_upper": 1.3,
            "flag": None,
            "priority": "ROUTINE",
            "comments": None,
        },
        # Row 5: Duplicate measurement key with conflicting valuenum (1.5 vs 1.2)
        {
            "labevent_id": 5,
            "subject_id": 1001,
            "hadm_id": 2001,
            "specimen_id": 503,
            "itemid": 50912,
            "order_provider_id": "P1",
            "charttime": "2150-05-10 11:00:00",
            "storetime": "2150-05-10 13:00:00",
            "value": "1.5",
            "valuenum": 1.5,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.6,
            "ref_range_upper": 1.3,
            "flag": "abnormal",
            "priority": "ROUTINE",
            "comments": None,
        },
        # Row 6: Non-renal test (Hematocrit)
        {
            "labevent_id": 6,
            "subject_id": 1002,
            "hadm_id": 2002,
            "specimen_id": 504,
            "itemid": 51221,
            "order_provider_id": "P2",
            "charttime": "2160-03-01 13:00:00",
            "storetime": "2160-03-01 14:00:00",
            "value": "38.5",
            "valuenum": 38.5,
            "valueuom": "%",
            "ref_range_lower": 36.0,
            "ref_range_upper": 46.0,
            "flag": None,
            "priority": "ROUTINE",
            "comments": None,
        },
    ])
    labevents_df.to_csv(data_dir / "labevents.csv", index=False)

    return data_dir


def test_missing_files_error(tmp_path: Path):
    """Test that missing raw files raise descriptive FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Laboratory dictionary not found"):
        audit_d_labitems(tmp_path / "nonexistent_d_labitems.csv")

    with pytest.raises(FileNotFoundError, match="Patients file not found"):
        audit_demographics_and_admissions(
            tmp_path / "nonexistent_patients.csv", tmp_path / "nonexistent_adm.csv"
        )

    with pytest.raises(FileNotFoundError, match="Laboratory events file not found"):
        audit_labevents(tmp_path / "nonexistent_labevents.csv", candidate_itemids={50912})


def test_audit_d_labitems(synthetic_data_dir: Path):
    """Test d_labitems audit: schema, null labels, unique IDs, and candidate extraction."""
    d_labitems_file = synthetic_data_dir / "d_labitems.csv"
    res = audit_d_labitems(d_labitems_file)

    assert res.total_rows == 13
    assert res.unique_itemids == 13
    assert res.duplicate_itemids == 0
    assert res.null_counts.get("label") == 1  # itemid 59999 has null label

    # Check candidates
    cand_ids = set(res.candidates_df["itemid"].tolist())
    assert 50912 in cand_ids  # Creatinine
    assert 51006 in cand_ids  # Urea Nitrogen
    assert 50862 in cand_ids  # Albumin
    assert 50931 in cand_ids  # Glucose
    assert 51081 in cand_ids  # Urine Creatinine
    assert 51102 in cand_ids  # Total Protein, Urine
    assert 50920 in cand_ids  # Estimated GFR
    assert 51221 not in cand_ids  # Hematocrit (non-renal)

    # Verify every candidate has selection_reason and category_group
    for _, row in res.candidates_df.iterrows():
        assert len(row["selection_reason"]) > 0
        assert len(row["category_group"]) > 0
        assert len(row["search_keyword"]) > 0


def test_audit_demographics_and_admissions(synthetic_data_dir: Path):
    """Test patients and admissions schemas, timestamp validity, and linkage metrics."""
    pat_file = synthetic_data_dir / "patients.csv"
    adm_file = synthetic_data_dir / "admissions.csv"

    res = audit_demographics_and_admissions(pat_file, adm_file)

    assert res.patients_rows == 4
    assert res.patients_unique_subjects == 4
    assert res.patients_dod_missing_count == 3  # 3 alive / unrecorded
    assert res.admissions_rows == 3
    assert res.admissions_unique_hadm == 3
    assert res.admissions_unique_subjects == 3

    # Linkage: All 3 admissions belong to known patients (100%)
    assert res.admissions_subject_in_patients_pct == 100.0
    # 1 patient (1004) has no admissions
    assert res.patients_without_admissions_count == 1
    assert res.patients_with_admissions_count == 3

    # Timestamp validity: No negative length of stay
    assert res.invalid_los_count == 0
    assert res.unparseable_admittime_count == 0
    assert res.unparseable_dischtime_count == 0
    assert res.median_los_days > 0


def test_chunked_labevents_smoke_mode(synthetic_data_dir: Path):
    """Test chunked processing in smoke mode with bounded chunks."""
    lab_file = synthetic_data_dir / "labevents.csv"
    candidate_ids = {50912, 51006, 50862, 50931}

    # Bounded smoke test with chunk_size=2 and max_chunks=2 (should read at most 4 rows)
    res_smoke = audit_labevents(
        file_path=lab_file,
        candidate_itemids=candidate_ids,
        mode="smoke",
        chunk_size=2,
        max_chunks=2,
    )

    assert res_smoke.mode == "smoke"
    assert res_smoke.is_complete is False
    assert res_smoke.total_rows_processed == 4
    assert res_smoke.total_chunks == 2
    assert res_smoke.chunk_size == 2


def test_chunked_labevents_full_mode(synthetic_data_dir: Path):
    """Test chunked processing in full scan mode covering all rows."""
    lab_file = synthetic_data_dir / "labevents.csv"
    candidate_ids = {50912, 51006, 50862, 50931, 51221}

    # Full scan with small chunks
    res_full = audit_labevents(
        file_path=lab_file,
        candidate_itemids=candidate_ids,
        mode="full",
        chunk_size=2,
        max_chunks=None,
    )

    assert res_full.mode == "full"
    assert res_full.is_complete is True
    assert res_full.total_rows_processed == 6
    assert res_full.total_chunks == 3

    # Missing valuenum accounting
    assert res_full.column_nulls["valuenum"] == 1
    # Missing hadm_id accounting (row 3 is outpatient)
    assert res_full.column_nulls["hadm_id"] == 1
    assert res_full.missing_hadm_pct == pytest.approx(100.0 / 6, 0.1)

    # Candidate tracking
    creat_stats = res_full.candidate_item_metrics[50912]
    assert creat_stats["total_records"] == 4
    assert creat_stats["missing_valuenum_count"] == 1  # Row 3 missing valuenum
    assert creat_stats["numeric_records"] == 3
    assert creat_stats["min_value"] == 1.2
    assert creat_stats["max_value"] == 1.5
    assert creat_stats["mean_value"] == pytest.approx((1.2 + 1.2 + 1.5) / 3, 0.01)

    # Unit tracking
    assert creat_stats["units"]["mg/dL"] == 4


def test_duplicate_and_conflict_detection(synthetic_data_dir: Path):
    """Test detection of duplicate measurement keys and conflicting numeric values when enabled."""
    lab_file = synthetic_data_dir / "labevents.csv"
    candidate_ids = {50912}

    # Read with include_duplicates=True
    res = audit_labevents(
        file_path=lab_file,
        candidate_itemids=candidate_ids,
        mode="full",
        chunk_size=10,
        max_chunks=None,
        include_duplicates=True,
    )

    # Row 1 and Row 4 are identical rows
    assert res.duplicate_rows_in_chunks >= 1
    # Duplicate measurement keys count (1001, 2150-05-10 11:00:00, 50912)
    assert res.duplicate_measurement_keys >= 1
    # Conflicting valuenum keys (1.2 vs 1.5)
    assert res.conflicting_valuenum_keys >= 1


def test_progress_callback(synthetic_data_dir: Path):
    """Test that progress callback receives valid progress and throughput events."""
    lab_file = synthetic_data_dir / "labevents.csv"
    updates = []

    def callback(info: dict[str, Any]):
        updates.append(info)

    res = audit_labevents(
        file_path=lab_file,
        candidate_itemids={50912},
        mode="smoke",
        chunk_size=2,
        max_chunks=3,
        log_interval_chunks=1,
        progress_callback=callback,
    )

    assert len(updates) > 0
    assert "chunks" in updates[0]
    assert "rows" in updates[0]
    assert "elapsed" in updates[0]
    assert "throughput" in updates[0]
    assert res.throughput_rows_per_sec > 0


def test_filtering_by_candidate_item_ids(synthetic_data_dir: Path):
    """Verify that only candidate item IDs are processed for renal metrics."""
    lab_file = synthetic_data_dir / "labevents.csv"
    candidate_ids = {50912}  # Only creatinine

    res = audit_labevents(
        file_path=lab_file,
        candidate_itemids=candidate_ids,
        mode="full",
        chunk_size=3,
        max_chunks=None,
    )

    # BUN (51006) was present in file but NOT in candidate_ids
    assert 51006 not in res.candidate_item_metrics
    # Creatinine (50912) was tracked
    assert 50912 in res.candidate_item_metrics
    assert res.candidate_item_metrics[50912]["total_records"] == 4


def test_interrupted_run_labeling(synthetic_data_dir: Path, tmp_path: Path):
    """Verify that bounded/interrupted runs are labeled PARTIAL and never COMPLETE."""
    out_dir = tmp_path / "reports_interrupted"

    config = {
        "project_root": tmp_path,
        "reports": {
            "generated_dir": out_dir,
            "dataset_inventory": out_dir / "dataset_inventory.md",
            "lab_item_inventory": out_dir / "lab_item_inventory.csv",
            "data_quality_summary": out_dir / "data_quality_summary.csv",
            "renal_candidate_items": out_dir / "renal_candidate_items.csv",
        },
    }

    dict_res = audit_d_labitems(synthetic_data_dir / "d_labitems.csv")
    demo_res = audit_demographics_and_admissions(
        synthetic_data_dir / "patients.csv", synthetic_data_dir / "admissions.csv"
    )

    # max_chunks=1 in full mode should result in is_complete=False
    lab_res = audit_labevents(
        file_path=synthetic_data_dir / "labevents.csv",
        candidate_itemids={50912},
        mode="full",
        chunk_size=2,
        max_chunks=1,
    )

    assert lab_res.is_complete is False
    reports = generate_reports(config, dict_res, demo_res, lab_res)

    md_text = reports["dataset_inventory"].read_text(encoding="utf-8")
    assert "PARTIAL SCAN" in md_text
    assert "COMPLETE FULL SCAN" not in md_text


def test_atomic_write_safety(tmp_path: Path):
    """Test atomic writing of text and CSV files."""
    csv_target = tmp_path / "atomic_test.csv"
    text_target = tmp_path / "atomic_test.md"

    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
    write_atomic_csv(df, csv_target)
    assert csv_target.is_file()
    assert not csv_target.with_suffix(".tmp").exists()
    assert len(pd.read_csv(csv_target)) == 2

    write_atomic_text("# Hello Atomic", text_target)
    assert text_target.is_file()
    assert not text_target.with_suffix(".tmp").exists()
    assert "# Hello Atomic" in text_target.read_text(encoding="utf-8")


def test_generate_reports_pipeline(synthetic_data_dir: Path, tmp_path: Path):
    """Test generation of all 4 aggregate report files and verify structure."""
    out_dir = tmp_path / "reports" / "generated"

    config = {
        "project_root": tmp_path,
        "reports": {
            "generated_dir": out_dir,
            "dataset_inventory": out_dir / "dataset_inventory.md",
            "lab_item_inventory": out_dir / "lab_item_inventory.csv",
            "data_quality_summary": out_dir / "data_quality_summary.csv",
            "renal_candidate_items": out_dir / "renal_candidate_items.csv",
        },
    }

    dict_res = audit_d_labitems(synthetic_data_dir / "d_labitems.csv")
    demo_res = audit_demographics_and_admissions(
        synthetic_data_dir / "patients.csv", synthetic_data_dir / "admissions.csv"
    )
    cand_ids = set(dict_res.candidates_df["itemid"].tolist())
    lab_res = audit_labevents(
        file_path=synthetic_data_dir / "labevents.csv",
        candidate_itemids=cand_ids,
        mode="smoke",
        chunk_size=3,
        max_chunks=2,
    )

    reports = generate_reports(config, dict_res, demo_res, lab_res)

    for r_name, r_path in reports.items():
        assert r_path.is_file(), f"Expected report {r_name} to exist at {r_path}"

    # Verify dataset_inventory.md
    md_content = reports["dataset_inventory"].read_text(encoding="utf-8")
    assert "MIMIC-IV 3.1 Dataset Inventory and Audit Report" in md_content
    assert "PARTIAL SCAN / SMOKE TEST" in md_content
    assert "Candidate Renal-Related Laboratory Items Discovered" in md_content
    assert "Methodological Classification & Safeguards" in md_content
    assert "Observed Facts" in md_content
    assert "Assumptions" in md_content
    assert "Unverified Findings" in md_content
    assert "Limitations" in md_content

    # Verify renal_candidate_items.csv
    cand_df = pd.read_csv(reports["renal_candidate_items"])
    assert "itemid" in cand_df.columns
    assert "label" in cand_df.columns
    assert "selection_reason" in cand_df.columns
    assert "scanned_row_count" in cand_df.columns
    assert len(cand_df) > 0

    # Verify data_quality_summary.csv
    dq_df = pd.read_csv(reports["data_quality_summary"])
    assert "table" in dq_df.columns
    assert "dimension" in dq_df.columns
    assert "metric" in dq_df.columns
    assert "status" in dq_df.columns
    assert "Pass" in dq_df["status"].values

    # Verify lab_item_inventory.csv
    inv_df = pd.read_csv(reports["lab_item_inventory"])
    assert len(inv_df) == 13
    assert "audit_mode" in inv_df.columns
    assert "is_complete_scan" in inv_df.columns
