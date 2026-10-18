"""Focused tests for Module 1.3 Step 1: Canonical Renal-Event Schema.

Verifies:
- Creation of canonical renal event DataFrame
- Schema validation rules and non-null constraints
- Preservation of raw source values and units without mutation
- Preservation of outpatient null hadm_id records
- Accurate reporting delay calculation and delay anomaly flagging
- Rejection of invalid item IDs or malformed schemas
- Deterministic event sorting by (subject_id, charttime, itemid, labevent_id)
"""

import numpy as np
import pandas as pd
import pytest

from src.temporal.canonical_schema import (
    TARGET_ITEM_IDS,
    ITEM_TO_ANALYTE,
    CANONICAL_UNITS,
    CANONICAL_COLUMNS,
    PERMITTED_UNIT_FLAGS,
    PERMITTED_DELAY_FLAGS,
    PERMITTED_ARBITRATION_ROLES,
    CanonicalSchemaError,
    validate_canonical_schema,
    create_canonical_event_frame,
)


@pytest.fixture
def synthetic_extracted_cohort_df():
    """Returns a synthetic DataFrame matching the M1.2 Phase B cohort structure."""
    return pd.DataFrame([
        {
            "labevent_id": 101,
            "subject_id": 10001,
            "hadm_id": 20001.0,
            "itemid": 50912,
            "charttime": "2180-05-01 10:00:00",
            "storetime": "2180-05-01 11:30:00",  # Delay = 1.5h
            "value": "1.2",
            "valuenum": 1.2,
            "valueuom": "mg/dL",
            "flag": None,
            "priority": "ROUTINE",
            "gender": "F",
            "anchor_age": 58,
            "anchor_year": 2180,
            "anchor_year_group": "2014 - 2016",
            "dod": None,
            "admittime": "2180-05-01 08:00:00",
            "dischtime": "2180-05-05 18:00:00",
            "deathtime": None,
            "admission_type": "EW EMER.",
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "HOME",
            "insurance": "Medicare",
            "race": "WHITE",
            "hospital_expire_flag": 0,
            "is_exact_duplicate": False,
            "is_conflicting_measurement": False,
            "valuenum_harmonized": 1.2,
            "unit_harmonized": "mg/dL",
            "unit_flag": "STANDARD",
        },
        # Outpatient lab (hadm_id is NaN)
        {
            "labevent_id": 102,
            "subject_id": 10001,
            "hadm_id": np.nan,
            "itemid": 51006,
            "charttime": "2180-03-15 09:00:00",
            "storetime": "2180-03-15 13:00:00",  # Delay = 4.0h
            "value": "24",
            "valuenum": 24.0,
            "valueuom": "mg/dL",
            "flag": None,
            "priority": "ROUTINE",
            "gender": "F",
            "anchor_age": 58,
            "anchor_year": 2180,
            "anchor_year_group": "2014 - 2016",
            "dod": None,
            "admittime": None,
            "dischtime": None,
            "deathtime": None,
            "admission_type": None,
            "admission_location": None,
            "discharge_location": None,
            "insurance": None,
            "race": None,
            "hospital_expire_flag": None,
            "is_exact_duplicate": False,
            "is_conflicting_measurement": False,
            "valuenum_harmonized": 24.0,
            "unit_harmonized": "mg/dL",
            "unit_flag": "STANDARD",
        },
        # Missing observation (valuenum is NaN)
        {
            "labevent_id": 103,
            "subject_id": 10001,
            "hadm_id": 20001.0,
            "itemid": 50971,
            "charttime": "2180-05-01 10:00:00",
            "storetime": "2180-05-01 11:30:00",  # Co-temporal with 101
            "value": "CANCELLED",
            "valuenum": np.nan,
            "valueuom": "mEq/L",
            "flag": None,
            "priority": "STAT",
            "gender": "F",
            "anchor_age": 58,
            "anchor_year": 2180,
            "anchor_year_group": "2014 - 2016",
            "dod": None,
            "admittime": "2180-05-01 08:00:00",
            "dischtime": "2180-05-05 18:00:00",
            "deathtime": None,
            "admission_type": "EW EMER.",
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "HOME",
            "insurance": "Medicare",
            "race": "WHITE",
            "hospital_expire_flag": 0,
            "is_exact_duplicate": False,
            "is_conflicting_measurement": False,
            "valuenum_harmonized": np.nan,
            "unit_harmonized": "mEq/L",
            "unit_flag": "STANDARD",
        },
    ])


def test_canonical_schema_constants_and_mappings():
    """Verify item ID definitions, canonical units, and permitted categorical flags."""
    assert len(TARGET_ITEM_IDS) == 6
    assert TARGET_ITEM_IDS["creatinine"] == 50912
    assert TARGET_ITEM_IDS["urea_nitrogen"] == 51006
    assert TARGET_ITEM_IDS["potassium"] == 50971
    assert TARGET_ITEM_IDS["sodium"] == 50983
    assert TARGET_ITEM_IDS["bicarbonate"] == 50882
    assert TARGET_ITEM_IDS["glucose"] == 50931

    for analyte, itemid in TARGET_ITEM_IDS.items():
        assert ITEM_TO_ANALYTE[itemid] == analyte
        assert analyte in CANONICAL_UNITS

    assert "STANDARD" in PERMITTED_UNIT_FLAGS
    assert "NEGATIVE_REPORTING_DELAY" in PERMITTED_DELAY_FLAGS
    assert "PRIMARY_SELECTED" in PERMITTED_ARBITRATION_ROLES


def test_create_canonical_event_frame(synthetic_extracted_cohort_df):
    """Test creation and full validation of canonical event frame."""
    canonical_df = create_canonical_event_frame(synthetic_extracted_cohort_df)

    is_valid, errors = validate_canonical_schema(canonical_df)
    assert is_valid, f"Validation failed with errors: {errors}"
    assert len(errors) == 0

    # Verify column presence
    for col in CANONICAL_COLUMNS:
        assert col in canonical_df.columns

    # Verify datatypes
    assert pd.api.types.is_datetime64_any_dtype(canonical_df["charttime"])
    assert pd.api.types.is_datetime64_any_dtype(canonical_df["storetime"])
    assert pd.api.types.is_float_dtype(canonical_df["reporting_delay_hours"])
    assert pd.api.types.is_bool_dtype(canonical_df["is_exact_duplicate"])
    assert pd.api.types.is_bool_dtype(canonical_df["is_conflicting_measurement"])


def test_raw_values_and_source_fidelity_unmodified(synthetic_extracted_cohort_df):
    """Verify that source values, strings, units, and timestamps are not altered or masked."""
    canonical_df = create_canonical_event_frame(synthetic_extracted_cohort_df)

    # Sort both to align rows
    orig = synthetic_extracted_cohort_df.sort_values("labevent_id").reset_index(drop=True)
    canon = canonical_df.sort_values("labevent_id").reset_index(drop=True)

    assert orig["value"].tolist() == canon["raw_value_text"].tolist()
    assert np.allclose(orig["valuenum"].fillna(-999), canon["raw_valuenum"].fillna(-999))
    assert orig["valueuom"].tolist() == canon["raw_valueuom"].tolist()
    assert pd.to_datetime(orig["charttime"]).tolist() == canon["charttime"].tolist()
    assert pd.to_datetime(orig["storetime"]).tolist() == canon["storetime"].tolist()



def test_reporting_delay_and_anomaly_flags():
    """Verify delay calculation in hours and anomaly flag classification."""
    test_df = pd.DataFrame([
        # Positive delay: 2 hours
        {"labevent_id": 1, "subject_id": 1, "hadm_id": np.nan, "itemid": 50912, "charttime": "2180-01-01 10:00:00", "storetime": "2180-01-01 12:00:00", "value": "1.0", "valuenum": 1.0, "valueuom": "mg/dL", "valuenum_harmonized": 1.0, "unit_harmonized": "mg/dL", "unit_flag": "STANDARD", "is_exact_duplicate": False, "is_conflicting_measurement": False, "gender": "F", "anchor_age": 50},
        # Zero delay: exact same timestamp
        {"labevent_id": 2, "subject_id": 1, "hadm_id": np.nan, "itemid": 50912, "charttime": "2180-01-02 10:00:00", "storetime": "2180-01-02 10:00:00", "value": "1.1", "valuenum": 1.1, "valueuom": "mg/dL", "valuenum_harmonized": 1.1, "unit_harmonized": "mg/dL", "unit_flag": "STANDARD", "is_exact_duplicate": False, "is_conflicting_measurement": False, "gender": "F", "anchor_age": 50},
        # Negative delay: storetime precedes charttime (e.g. clock desync / pre-dated entry)
        {"labevent_id": 3, "subject_id": 1, "hadm_id": np.nan, "itemid": 50912, "charttime": "2180-01-03 10:00:00", "storetime": "2180-01-03 09:30:00", "value": "1.2", "valuenum": 1.2, "valueuom": "mg/dL", "valuenum_harmonized": 1.2, "unit_harmonized": "mg/dL", "unit_flag": "STANDARD", "is_exact_duplicate": False, "is_conflicting_measurement": False, "gender": "F", "anchor_age": 50},
    ])

    canonical_df = create_canonical_event_frame(test_df)

    r1 = canonical_df[canonical_df["labevent_id"] == 1].iloc[0]
    assert r1["reporting_delay_hours"] == 2.0
    assert r1["delay_anomaly_flag"] == "NORMAL_DELAY"

    r2 = canonical_df[canonical_df["labevent_id"] == 2].iloc[0]
    assert r2["reporting_delay_hours"] == 0.0
    assert r2["delay_anomaly_flag"] == "ZERO_DELAY"

    r3 = canonical_df[canonical_df["labevent_id"] == 3].iloc[0]
    assert r3["reporting_delay_hours"] == -0.5
    assert r3["delay_anomaly_flag"] == "NEGATIVE_REPORTING_DELAY"


def test_outpatient_hadm_id_null_preservation(synthetic_extracted_cohort_df):
    """Verify that records with null hadm_id are preserved without schema error."""
    canonical_df = create_canonical_event_frame(synthetic_extracted_cohort_df)

    outpatient_row = canonical_df[canonical_df["labevent_id"] == 102].iloc[0]
    assert pd.isna(outpatient_row["hadm_id"])
    assert pd.isna(outpatient_row["admission_type"])
    assert outpatient_row["analyte_name"] == "urea_nitrogen"


def test_schema_validation_catches_unsupported_itemid():
    """Verify that unsupported item IDs (e.g. albumin 50862) raise CanonicalSchemaError."""
    bad_df = pd.DataFrame([{
        "labevent_id": 1,
        "subject_id": 1,
        "hadm_id": np.nan,
        "itemid": 50862,  # Albumin, not in target 6
        "charttime": "2180-01-01 10:00:00",
        "storetime": "2180-01-01 11:00:00",
        "value": "3.5",
        "valuenum": 3.5,
        "valueuom": "g/dL",
        "valuenum_harmonized": 3.5,
        "unit_harmonized": "g/dL",
        "unit_flag": "STANDARD",
        "is_exact_duplicate": False,
        "is_conflicting_measurement": False,
        "gender": "M",
        "anchor_age": 60,
    }])

    with pytest.raises(CanonicalSchemaError, match="unsupported item IDs"):
        create_canonical_event_frame(bad_df)


def test_deterministic_sorting_of_canonical_frame():
    """Verify that canonical frame sorts deterministically by (subject_id, charttime, itemid, labevent_id)."""
    unsorted_df = pd.DataFrame([
        {"labevent_id": 20, "subject_id": 102, "hadm_id": np.nan, "itemid": 50912, "charttime": "2180-01-02 10:00:00", "storetime": "2180-01-02 11:00:00", "value": "1.0", "valuenum": 1.0, "valueuom": "mg/dL", "valuenum_harmonized": 1.0, "unit_harmonized": "mg/dL", "unit_flag": "STANDARD", "is_exact_duplicate": False, "is_conflicting_measurement": False, "gender": "F", "anchor_age": 50},
        {"labevent_id": 10, "subject_id": 101, "hadm_id": np.nan, "itemid": 50912, "charttime": "2180-01-01 10:00:00", "storetime": "2180-01-01 11:00:00", "value": "1.0", "valuenum": 1.0, "valueuom": "mg/dL", "valuenum_harmonized": 1.0, "unit_harmonized": "mg/dL", "unit_flag": "STANDARD", "is_exact_duplicate": False, "is_conflicting_measurement": False, "gender": "F", "anchor_age": 50},
    ])

    canonical_df = create_canonical_event_frame(unsorted_df)
    assert canonical_df.iloc[0]["subject_id"] == 101
    assert canonical_df.iloc[1]["subject_id"] == 102


def test_canonical_frame_against_real_extracted_cohort_slice():
    """Integration test: Validate canonical conversion on actual M1.2 Phase B derived file."""
    cohort_path = "data/processed/cohort_renal_bounded.csv"
    slice_df = pd.read_csv(cohort_path, nrows=500)

    canonical_df = create_canonical_event_frame(slice_df)
    is_valid, errors = validate_canonical_schema(canonical_df)

    assert is_valid, f"Validation on real cohort slice failed: {errors}"
    assert len(canonical_df) == 500
    assert canonical_df["reporting_delay_hours"].min() >= 0.0
    assert (canonical_df["delay_anomaly_flag"] == "NORMAL_DELAY").all()
