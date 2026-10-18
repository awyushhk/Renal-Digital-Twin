"""Unit and integration tests for M1.2 Phase B cohort extraction.

Verifies:
- Cohort determinism
- Patient cap enforcement (<= 2500)
- Eligibility filtering (>=3 usable serum creatinine observations)
- Target variable filtering (6 blood chemistry items)
- Relational joins (demographics and admissions)
- Preservation of null admission identifiers (outpatient labs)
- Unit mappings and flagging (standard, converted, missing, unknown)
- Timestamp preservation without truncation
- Exact duplicate vs conflicting measurement detection
- Preservation of missing observations (no silent forward-fill or drop)
- Monotonic early-exit streaming logic
"""

import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.ingestion.extract_cohort import (
    CohortConfig,
    TARGET_ITEM_IDS,
    STANDARD_UNITS,
    UNIT_CONVERSION_RULES,
    identify_cohort_patients,
    extract_cohort_labevents,
    perform_relational_joins,
    audit_duplicates_and_conflicts,
    harmonize_units,
    generate_cohort_summary_and_reports,
    run_cohort_extraction,
)


@pytest.fixture
def synthetic_raw_environment(tmp_path: Path):
    """Creates a complete synthetic raw environment with known ground truth."""
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    reports_dir = tmp_path / "reports" / "generated"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    # 1. Patients: 5 patients
    # P1001: 4 usable creat -> eligible
    # P1002: 3 usable creat -> eligible
    # P1003: 2 usable creat -> NOT eligible (< 3)
    # P1004: 3 creat (1 is NaN, 1 is 0.0, 1 is 1.5) -> NOT eligible (only 1 usable)
    # P1005: 5 usable creat -> eligible
    patients_data = {
        "subject_id": [1001, 1002, 1003, 1004, 1005],
        "gender": ["F", "M", "F", "M", "F"],
        "anchor_age": [54, 62, 45, 71, 38],
        "anchor_year": [2150, 2152, 2148, 2160, 2155],
        "anchor_year_group": ["2014 - 2016", "2014 - 2016", "2011 - 2013", "2017 - 2019", "2014 - 2016"],
        "dod": [np.nan, "2155-04-12", np.nan, np.nan, np.nan],
    }
    patients_df = pd.DataFrame(patients_data)
    patients_csv = raw_dir / "patients.csv"
    patients_df.to_csv(patients_csv, index=False)

    # 2. Admissions: P1001 (2 adms), P1002 (1 adm), P1005 (0 adms - outpatient only)
    admissions_data = {
        "subject_id": [1001, 1001, 1002],
        "hadm_id": [20001, 20002, 20003],
        "admittime": ["2150-01-10 08:00:00", "2150-06-15 14:30:00", "2152-03-20 10:15:00"],
        "dischtime": ["2150-01-15 12:00:00", "2150-06-20 18:00:00", "2152-03-25 11:00:00"],
        "deathtime": [np.nan, np.nan, np.nan],
        "admission_type": ["EW EMER.", "URGENT", "DIRECT EMER."],
        "admit_provider_id": ["P001", "P002", "P003"],
        "admission_location": ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "EMERGENCY ROOM"],
        "discharge_location": ["HOME", "HOME", "REHAB"],
        "insurance": ["Medicare", "Medicaid", "Other"],
        "language": ["English", "English", "English"],
        "marital_status": ["MARRIED", "SINGLE", "DIVORCED"],
        "race": ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN"],
        "edregtime": [np.nan, np.nan, np.nan],
        "edouttime": [np.nan, np.nan, np.nan],
        "hospital_expire_flag": [0, 0, 0],
    }
    admissions_df = pd.DataFrame(admissions_data)
    admissions_csv = raw_dir / "admissions.csv"
    admissions_df.to_csv(admissions_csv, index=False)

    # 3. Labevents with known edge cases:
    # - null hadm_id (outpatient / ED)
    # - exact duplicate record
    # - conflicting measurement (same timestamp, different valuenum)
    # - units: standard mg/dL, converted umol/L, mmol/L, unknown unit, missing unit
    # - missing observation (valuenum = NaN)
    # - irrelevant itemid (e.g. 50862 albumin)
    labs_rows = [
        # P1001: 4 usable creat, 2 BUN, 1 glucose, 1 exact dup, 1 conflict, 1 outpatient null hadm_id
        {"labevent_id": 1, "subject_id": 1001, "hadm_id": 20001, "itemid": 50912, "charttime": "2150-01-10 09:00:00", "storetime": "2150-01-10 10:00:00", "value": "1.2", "valuenum": 1.2, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 2, "subject_id": 1001, "hadm_id": 20001, "itemid": 50912, "charttime": "2150-01-11 06:00:00", "storetime": "2150-01-11 07:00:00", "value": "1.4", "valuenum": 1.4, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        # Conflicting measurement at 2150-01-12 06:00:00: value 1.6 vs 1.9
        {"labevent_id": 3, "subject_id": 1001, "hadm_id": 20001, "itemid": 50912, "charttime": "2150-01-12 06:00:00", "storetime": "2150-01-12 07:00:00", "value": "1.6", "valuenum": 1.6, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 4, "subject_id": 1001, "hadm_id": 20001, "itemid": 50912, "charttime": "2150-01-12 06:00:00", "storetime": "2150-01-12 07:05:00", "value": "1.9", "valuenum": 1.9, "valueuom": "mg/dL", "flag": None, "priority": "STAT"},
        # Outpatient creatinine (hadm_id is null/NaN) -> 4th usable creat
        {"labevent_id": 5, "subject_id": 1001, "hadm_id": np.nan, "itemid": 50912, "charttime": "2150-04-01 11:00:00", "storetime": "2150-04-01 12:00:00", "value": "1.1", "valuenum": 1.1, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        # BUN with exact duplicate
        {"labevent_id": 6, "subject_id": 1001, "hadm_id": 20001, "itemid": 51006, "charttime": "2150-01-10 09:00:00", "storetime": "2150-01-10 10:00:00", "value": "22", "valuenum": 22.0, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 7, "subject_id": 1001, "hadm_id": 20001, "itemid": 51006, "charttime": "2150-01-10 09:00:00", "storetime": "2150-01-10 10:00:00", "value": "22", "valuenum": 22.0, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},  # Exact dup of 6
        # Irrelevant itemid (albumin 50862) -> should be filtered out
        {"labevent_id": 8, "subject_id": 1001, "hadm_id": 20001, "itemid": 50862, "charttime": "2150-01-10 09:00:00", "storetime": "2150-01-10 10:00:00", "value": "3.8", "valuenum": 3.8, "valueuom": "g/dL", "flag": None, "priority": "ROUTINE"},

        # P1002: 3 usable creat (one in umol/L), 1 K, 1 Na
        {"labevent_id": 9, "subject_id": 1002, "hadm_id": 20003, "itemid": 50912, "charttime": "2152-03-20 11:00:00", "storetime": "2152-03-20 12:00:00", "value": "0.9", "valuenum": 0.9, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 10, "subject_id": 1002, "hadm_id": 20003, "itemid": 50912, "charttime": "2152-03-21 06:00:00", "storetime": "2152-03-21 07:00:00", "value": "1.0", "valuenum": 1.0, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        # 3rd creat in umol/L: 88.42 umol/L == 1.0 mg/dL
        {"labevent_id": 11, "subject_id": 1002, "hadm_id": 20003, "itemid": 50912, "charttime": "2152-03-22 06:00:00", "storetime": "2152-03-22 07:00:00", "value": "88.42", "valuenum": 88.42, "valueuom": "umol/L", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 12, "subject_id": 1002, "hadm_id": 20003, "itemid": 50971, "charttime": "2152-03-20 11:00:00", "storetime": "2152-03-20 12:00:00", "value": "4.2", "valuenum": 4.2, "valueuom": "mEq/L", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 13, "subject_id": 1002, "hadm_id": 20003, "itemid": 50983, "charttime": "2152-03-20 11:00:00", "storetime": "2152-03-20 12:00:00", "value": "139", "valuenum": 139.0, "valueuom": "mEq/L", "flag": None, "priority": "ROUTINE"},

        # P1003: only 2 usable creat -> ineligible
        {"labevent_id": 14, "subject_id": 1003, "hadm_id": np.nan, "itemid": 50912, "charttime": "2148-05-01 08:00:00", "storetime": "2148-05-01 09:00:00", "value": "0.8", "valuenum": 0.8, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 15, "subject_id": 1003, "hadm_id": np.nan, "itemid": 50912, "charttime": "2148-05-02 08:00:00", "storetime": "2148-05-02 09:00:00", "value": "0.9", "valuenum": 0.9, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},

        # P1004: 3 creat but 1 NaN and 1 zero -> only 1 usable -> ineligible
        {"labevent_id": 16, "subject_id": 1004, "hadm_id": np.nan, "itemid": 50912, "charttime": "2160-01-01 08:00:00", "storetime": "2160-01-01 09:00:00", "value": "CANCELLED", "valuenum": np.nan, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 17, "subject_id": 1004, "hadm_id": np.nan, "itemid": 50912, "charttime": "2160-01-02 08:00:00", "storetime": "2160-01-02 09:00:00", "value": "0.0", "valuenum": 0.0, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 18, "subject_id": 1004, "hadm_id": np.nan, "itemid": 50912, "charttime": "2160-01-03 08:00:00", "storetime": "2160-01-03 09:00:00", "value": "1.5", "valuenum": 1.5, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},

        # P1005: Outpatient only (no admissions), 3 usable creat, 1 glucose with unknown unit, 1 glucose with missing unit, 1 bicarb
        {"labevent_id": 19, "subject_id": 1005, "hadm_id": np.nan, "itemid": 50912, "charttime": "2155-07-01 10:00:00", "storetime": "2155-07-01 11:00:00", "value": "0.7", "valuenum": 0.7, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 20, "subject_id": 1005, "hadm_id": np.nan, "itemid": 50912, "charttime": "2155-07-02 10:00:00", "storetime": "2155-07-02 11:00:00", "value": "0.8", "valuenum": 0.8, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 21, "subject_id": 1005, "hadm_id": np.nan, "itemid": 50912, "charttime": "2155-07-03 10:00:00", "storetime": "2155-07-03 11:00:00", "value": "0.75", "valuenum": 0.75, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"},
        {"labevent_id": 22, "subject_id": 1005, "hadm_id": np.nan, "itemid": 50931, "charttime": "2155-07-01 10:00:00", "storetime": "2155-07-01 11:00:00", "value": "100", "valuenum": 100.0, "valueuom": "furlongs", "flag": None, "priority": "ROUTINE"},  # Unknown unit
        {"labevent_id": 23, "subject_id": 1005, "hadm_id": np.nan, "itemid": 50931, "charttime": "2155-07-02 10:00:00", "storetime": "2155-07-02 11:00:00", "value": "110", "valuenum": 110.0, "valueuom": np.nan, "flag": None, "priority": "ROUTINE"},  # Missing unit
        {"labevent_id": 24, "subject_id": 1005, "hadm_id": np.nan, "itemid": 50882, "charttime": "2155-07-01 10:00:00", "storetime": "2155-07-01 11:00:00", "value": "24", "valuenum": 24.0, "valueuom": "mEq/L", "flag": None, "priority": "ROUTINE"},
    ]
    labs_df = pd.DataFrame(labs_rows)
    labs_csv = raw_dir / "labevents.csv"
    labs_df.to_csv(labs_csv, index=False)

    # d_labitems dummy
    d_labitems_df = pd.DataFrame([{"itemid": k, "label": v} for k, v in TARGET_ITEM_IDS.items()])
    d_labitems_csv = raw_dir / "d_labitems.csv"
    d_labitems_df.to_csv(d_labitems_csv, index=False)

    config = CohortConfig(
        patient_cap=2,  # Cap at 2 to test cap truncation (eligible: 1001, 1002, 1005 -> select 1001, 1002)
        min_creatinine_obs=3,
        anchor_itemid=50912,
        chunk_size=10,
        labevents_path=str(labs_csv),
        patients_path=str(patients_csv),
        admissions_path=str(admissions_csv),
        d_labitems_path=str(d_labitems_csv),
        cohort_labs_path=str(processed_dir / "cohort_renal_bounded.csv"),
        cohort_patients_path=str(processed_dir / "cohort_patients.csv"),
        quality_report_path=str(reports_dir / "cohort_quality_report.md"),
        summary_csv_path=str(reports_dir / "cohort_summary.csv"),
        summary_json_path=str(reports_dir / "cohort_summary.json"),
        enforce_monotonic_early_exit=False,  # Disable early exit for small multi-chunk test
    )
    return config, labs_df, patients_df, admissions_df


def test_cohort_determinism_and_cap(synthetic_raw_environment):
    """Test 1: Verification of deterministic patient ranking and strict cap enforcement."""
    config, _, _, _ = synthetic_raw_environment

    # Run identification twice
    patients_run1, counts1, stats1 = identify_cohort_patients(
        labevents_path=config.labevents_path,
        patient_cap=2,
        min_creatinine_obs=3,
        chunk_size=5,
        enforce_monotonic_early_exit=False,
    )
    patients_run2, counts2, stats2 = identify_cohort_patients(
        labevents_path=config.labevents_path,
        patient_cap=2,
        min_creatinine_obs=3,
        chunk_size=5,
        enforce_monotonic_early_exit=False,
    )

    # Determinism: identical across runs
    assert patients_run1 == patients_run2
    assert counts1 == counts2

    # Cap: exactly 2 patients selected (even though 1001, 1002, 1005 qualify)
    assert len(patients_run1) == 2
    # Deterministic ascending order: first 2 eligible are 1001 and 1002 (1005 excluded by cap)
    assert patients_run1 == [1001, 1002]
    assert 1005 not in patients_run1


def test_eligibility_criteria_filtering(synthetic_raw_environment):
    """Test 2: Verification of >=3 usable creatinine measurements rule."""
    config, _, _, _ = synthetic_raw_environment

    # With cap=10 (higher than total eligible), check all qualifying patients
    patients, counts, stats = identify_cohort_patients(
        labevents_path=config.labevents_path,
        patient_cap=10,
        min_creatinine_obs=3,
        chunk_size=5,
        enforce_monotonic_early_exit=False,
    )

    # P1001 (4 usable), P1002 (3 usable), P1005 (3 usable) must qualify
    assert 1001 in patients
    assert 1002 in patients
    assert 1005 in patients

    # P1003 (only 2 creat) must NOT qualify
    assert 1003 not in patients

    # P1004 (3 creat, but 1 NaN and 1 0.0 -> only 1 usable) must NOT qualify
    assert 1004 not in patients


def test_variable_filtering_excludes_extraneous_items(synthetic_raw_environment):
    """Test 3: Verification that only the 6 validated target blood chemistry items are extracted."""
    config, _, _, _ = synthetic_raw_environment
    selected_patients = [1001, 1002]

    labs_extracted, stats = extract_cohort_labevents(
        labevents_path=config.labevents_path,
        selected_patients=selected_patients,
        target_items=config.target_item_ids,
        chunk_size=5,
        enforce_monotonic_early_exit=False,
    )

    extracted_items = set(labs_extracted["itemid"].unique())
    target_item_set = set(TARGET_ITEM_IDS.values())

    # All extracted items must belong to target items
    assert extracted_items.issubset(target_item_set)

    # Albumin (itemid 50862) was present in P1001's raw labs; must be excluded
    assert 50862 not in extracted_items

    # P1003 and P1004 rows must not be present in extracted labs
    assert not any(labs_extracted["subject_id"].isin([1003, 1004]))


def test_relational_joins_and_null_hadm_preservation(synthetic_raw_environment):
    """Test 4: Relational join integrity and outpatient null hadm_id preservation."""
    config, _, _, _ = synthetic_raw_environment
    selected_patients = [1001, 1002, 1005]

    labs_extracted, _ = extract_cohort_labevents(
        labevents_path=config.labevents_path,
        selected_patients=selected_patients,
        target_items=config.target_item_ids,
        chunk_size=10,
        enforce_monotonic_early_exit=False,
    )

    joined_df, patient_summary, join_stats = perform_relational_joins(
        labs_df=labs_extracted,
        patients_path=config.patients_path,
        admissions_path=config.admissions_path,
        selected_patients=selected_patients,
    )

    # 1. Row count must be perfectly preserved across joins
    assert len(joined_df) == len(labs_extracted)

    # 2. Demographics must be populated for 100% of rows
    assert joined_df["gender"].notna().all()
    assert joined_df["anchor_age"].notna().all()

    # 3. Null hadm_id rows must be preserved without loss
    null_hadm_rows = joined_df[joined_df["hadm_id"].isna()]
    assert len(null_hadm_rows) > 0  # P1001 outpatient lab + P1005 labs
    # For null hadm_id, admission columns should be NaN
    assert null_hadm_rows["admission_type"].isna().all()

    # 4. Non-null hadm_id rows must have joined admission data
    non_null_hadm = joined_df[joined_df["hadm_id"].notna()]
    assert non_null_hadm["admission_type"].notna().all()

    # 5. P1005 has no admissions in admissions.csv, but is retained in patient summary
    p1005_summary = patient_summary[patient_summary["subject_id"] == 1005].iloc[0]
    assert p1005_summary["has_admission"] == False
    assert p1005_summary["admission_count"] == 0



def test_duplicate_and_conflict_detection(synthetic_raw_environment):
    """Test 5: Detection and separate flagging of exact duplicates vs conflicting measurements."""
    config, _, _, _ = synthetic_raw_environment
    selected_patients = [1001, 1002]

    labs_extracted, _ = extract_cohort_labevents(
        labevents_path=config.labevents_path,
        selected_patients=selected_patients,
        target_items=config.target_item_ids,
        chunk_size=10,
        enforce_monotonic_early_exit=False,
    )

    audited_df, dup_stats = audit_duplicates_and_conflicts(labs_extracted)

    # Exact duplicates: P1001 BUN (labevent_id 6 and 7) are identical across clinical cols
    exact_dups = audited_df[audited_df["is_exact_duplicate"]]
    assert len(exact_dups) == 2
    assert (exact_dups["itemid"] == 51006).all()
    assert (exact_dups["charttime"] == "2150-01-10 09:00:00").all()

    # Conflicting measurements: P1001 creat at 2150-01-12 06:00:00 has values 1.6 and 1.9
    conflicts = audited_df[audited_df["is_conflicting_measurement"]]
    assert len(conflicts) == 2
    assert (conflicts["itemid"] == 50912).all()
    assert set(conflicts["valuenum"].values) == {1.6, 1.9}

    # An exact duplicate should NOT be flagged as conflicting if values are identical
    bun_dups = audited_df[audited_df["itemid"] == 51006]
    assert not bun_dups["is_conflicting_measurement"].any()


def test_unit_harmonization_and_flagging(synthetic_raw_environment):
    """Test 6: Explicit unit harmonization, conversion rules, and error flagging."""
    # Synthetic dataset with standard, converted, unknown, and missing units
    test_df = pd.DataFrame([
        {"itemid": 50912, "valuenum": 1.2, "valueuom": "mg/dL"},      # Standard
        {"itemid": 50912, "valuenum": 88.42, "valueuom": "umol/L"},   # Converted (88.42 / 88.42 = 1.0)
        {"itemid": 51006, "valuenum": 10.0, "valueuom": "mmol/L"},    # Converted (10 * 2.801 = 28.01)
        {"itemid": 50971, "valuenum": 4.5, "valueuom": "mEq/L"},      # Standard
        {"itemid": 50931, "valuenum": 100.0, "valueuom": "furlongs"}, # Unknown unit
        {"itemid": 50931, "valuenum": 110.0, "valueuom": np.nan},     # Missing unit
    ])

    harmonized_df, unit_stats = harmonize_units(test_df)

    # 1. Standard creatinine
    row0 = harmonized_df.iloc[0]
    assert row0["unit_flag"] == "STANDARD"
    assert row0["valuenum_harmonized"] == 1.2
    assert row0["unit_harmonized"] == "mg/dL"

    # 2. Converted creatinine (umol/L -> mg/dL)
    row1 = harmonized_df.iloc[1]
    assert row1["unit_flag"] == "CONVERTED"
    assert pytest.approx(row1["valuenum_harmonized"], 0.001) == 1.0
    assert row1["unit_harmonized"] == "mg/dL"

    # 3. Converted BUN (mmol/L -> mg/dL)
    row2 = harmonized_df.iloc[2]
    assert row2["unit_flag"] == "CONVERTED"
    assert pytest.approx(row2["valuenum_harmonized"], 0.01) == 28.01

    # 4. Standard Potassium
    row3 = harmonized_df.iloc[3]
    assert row3["unit_flag"] == "STANDARD"
    assert row3["valuenum_harmonized"] == 4.5
    assert row3["unit_harmonized"] == "mEq/L"

    # 5. Unknown unit must be flagged, not guessed, valuenum_harmonized must be NaN
    row4 = harmonized_df.iloc[4]
    assert row4["unit_flag"] == "UNKNOWN_UNIT"
    assert pd.isna(row4["valuenum_harmonized"])

    # 6. Missing UOM must be flagged
    row5 = harmonized_df.iloc[5]
    assert row5["unit_flag"] == "MISSING_UOM"
    assert pd.isna(row5["valuenum_harmonized"])

    # Original valuenum and valueuom must remain completely unchanged
    assert test_df["valuenum"].equals(harmonized_df["valuenum"])
    assert test_df["valueuom"].equals(harmonized_df["valueuom"])


def test_timestamp_preservation_fidelity(synthetic_raw_environment):
    """Test 7: Verification that event timestamps retain exact second-level precision."""
    config, _, _, _ = synthetic_raw_environment
    selected_patients = [1001]

    labs_extracted, _ = extract_cohort_labevents(
        labevents_path=config.labevents_path,
        selected_patients=selected_patients,
        target_items=config.target_item_ids,
        chunk_size=5,
        enforce_monotonic_early_exit=False,
    )

    # Check that timestamps contain full YYYY-MM-DD HH:MM:SS
    for charttime in labs_extracted["charttime"]:
        assert len(charttime) == 19
        assert charttime[10] == " "
        assert charttime[13] == ":"
        assert charttime[16] == ":"


def test_missing_observation_preservation(synthetic_raw_environment):
    """Test 8: Verification that missing observations (valuenum=NaN) are preserved without imputation."""
    config, _, _, _ = synthetic_raw_environment
    selected_patients = [1005]  # P1005 has lab with missing unit and another with missing valuenum if present

    # Test directly with DataFrame containing NaN
    df_with_nan = pd.DataFrame([
        {"labevent_id": 99, "subject_id": 1005, "hadm_id": np.nan, "itemid": 50912, "charttime": "2155-08-01 10:00:00", "storetime": "2155-08-01 11:00:00", "value": None, "valuenum": np.nan, "valueuom": "mg/dL", "flag": None, "priority": "ROUTINE"}
    ])

    audited, _ = audit_duplicates_and_conflicts(df_with_nan)
    harmonized, _ = harmonize_units(audited)

    # Must preserve row
    assert len(harmonized) == 1
    # Must preserve NaN valuenum
    assert pd.isna(harmonized["valuenum"].iloc[0])
    assert pd.isna(harmonized["valuenum_harmonized"].iloc[0])


def test_monotonic_early_exit_efficiency(tmp_path: Path):
    """Test 9: Verification that monotonic streaming terminates as soon as cap patient's history closes."""
    # Create 4 chunks with strictly monotonic subject_ids:
    # Chunk 1: sid 101, 102 (both have 3 creat)
    # Chunk 2: sid 103 (has 3 creat)
    # Chunk 3: sid 104 (has 3 creat)
    # Chunk 4: sid 105 (has 3 creat)
    rows = []
    for sid in [101, 102, 103, 104, 105]:
        for i in range(3):
            rows.append({
                "subject_id": sid,
                "itemid": 50912,
                "valuenum": 1.0 + i * 0.1,
            })
    stream_df = pd.DataFrame(rows)
    stream_csv = tmp_path / "stream_monotonic.csv"
    stream_df.to_csv(stream_csv, index=False)

    # Set patient_cap = 2 with chunk_size = 3 (each chunk has 1 patient)
    patients, counts, stats = identify_cohort_patients(
        labevents_path=str(stream_csv),
        patient_cap=2,
        min_creatinine_obs=3,
        chunk_size=3,
        enforce_monotonic_early_exit=True,
    )

    # Cap was 2, so only 101 and 102 should be selected
    assert patients == [101, 102]
    # Early exit must have triggered before reading all 15 rows (5 chunks of 3)
    assert stats["early_exit_triggered"] is True
    # Rows streamed must be strictly less than total rows (15)
    assert stats["total_streamed_rows"] < len(stream_df)


def test_end_to_end_cohort_pipeline_execution(synthetic_raw_environment):
    """Test 10: End-to-end execution of run_cohort_extraction and artifact creation."""
    config, _, _, _ = synthetic_raw_environment

    result = run_cohort_extraction(config)

    assert result["status"] == "SUCCESS"
    assert result["cohort_patients_count"] == 2
    assert result["cohort_lab_rows_count"] > 0
    assert Path(result["cohort_labs_path"]).exists()
    assert Path(result["cohort_patients_path"]).exists()
    assert Path(result["quality_report_path"]).exists()
    assert Path(result["summary_csv_path"]).exists()
    assert Path(result["summary_json_path"]).exists()

    # Verify quality report content
    with open(result["quality_report_path"], "r", encoding="utf-8") as f:
        report_text = f.read()

    assert "MIMIC-IV Bounded Cohort Extraction Quality Report" in report_text
    assert "Creatinine" in report_text
    assert "Exact Duplicate Rows" in report_text
    assert "Conflicting Measurement Rows" in report_text
    assert "Selection Bias & Epidemiological Limitations" in report_text


def test_exact_duplicate_with_null_hadm_id():
    """Test 11: Regression test for duplicate detection when hadm_id is null (outpatient duplicate)."""
    df = pd.DataFrame([
        {"subject_id": 9999, "hadm_id": np.nan, "itemid": 50912, "charttime": "2150-01-01 08:00:00", "valuenum": 1.1, "valueuom": "mg/dL"},
        {"subject_id": 9999, "hadm_id": np.nan, "itemid": 50912, "charttime": "2150-01-01 08:00:00", "valuenum": 1.1, "valueuom": "mg/dL"},
        {"subject_id": 9999, "hadm_id": np.nan, "itemid": 50912, "charttime": "2150-01-02 08:00:00", "valuenum": 1.2, "valueuom": "mg/dL"},
    ])

    audited, stats = audit_duplicates_and_conflicts(df)
    assert audited["is_exact_duplicate"].sum() == 2
    assert audited["is_conflicting_measurement"].sum() == 0

    # Grouping with dropna=False must preserve the null-hadm duplicate group
    dup_cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    groups = audited[audited["is_exact_duplicate"]].groupby(dup_cols, dropna=False)
    assert groups.ngroups == 1
    assert groups.size().iloc[0] == 2


def test_conflict_and_duplicate_overlap_semantics():
    """Test 12: Regression test verifying independent flagging when identical and conflicting rows coexist."""
    # A 3-row cluster at the same timestamp: two rows have valuenum=1.0 (exact duplicates of each other),
    # and one row has valuenum=1.5 (creates conflict across the timestamp key)
    df = pd.DataFrame([
        {"subject_id": 8888, "hadm_id": 100, "itemid": 50912, "charttime": "2150-05-01 08:00:00", "valuenum": 1.0, "valueuom": "mg/dL"},
        {"subject_id": 8888, "hadm_id": 100, "itemid": 50912, "charttime": "2150-05-01 08:00:00", "valuenum": 1.0, "valueuom": "mg/dL"},
        {"subject_id": 8888, "hadm_id": 100, "itemid": 50912, "charttime": "2150-05-01 08:00:00", "valuenum": 1.5, "valueuom": "mg/dL"},
    ])

    audited, stats = audit_duplicates_and_conflicts(df)

    # First two rows are exact duplicates of each other
    assert audited["is_exact_duplicate"].tolist() == [True, True, False]
    # All three rows belong to the conflicting key (2 distinct values at same timestamp)
    assert audited["is_conflicting_measurement"].tolist() == [True, True, True]


def test_early_stop_disabled_fallback_behavior(tmp_path: Path):
    """Test 13: Regression test verifying full scan fallback when enforce_monotonic_early_exit is False."""
    rows = []
    # 4 patients across 4 chunks
    for sid in [201, 202, 203, 204]:
        for i in range(3):
            rows.append({"subject_id": sid, "itemid": 50912, "valuenum": 1.0 + i})
    stream_df = pd.DataFrame(rows)
    stream_csv = tmp_path / "fallback_stream.csv"
    stream_df.to_csv(stream_csv, index=False)

    # Cap=2 but early-exit disabled: must scan all rows
    patients, counts, stats = identify_cohort_patients(
        labevents_path=str(stream_csv),
        patient_cap=2,
        min_creatinine_obs=3,
        chunk_size=3,
        enforce_monotonic_early_exit=False,
    )

    assert patients == [201, 202]
    assert stats["early_exit_triggered"] is False
    assert stats["total_streamed_rows"] == len(stream_df)

