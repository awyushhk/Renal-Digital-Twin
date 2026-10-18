"""Focused tests for Module 1.3 Step 2: Conflict & Duplicate Arbitration.

Verifies:
1. Exact duplicates, including null hadm_id
2. Conflicting values with different storetimes
3. Equal storetimes and deterministic tie-breaking
4. STAT, ROUTINE, missing and unknown priority handling
5. Analytical mean aggregation with source provenance links
6. Policy D for all supported analytes and explicit rejection/flagging for sodium and glucose
7. Missing numerical values handling without silent dropping
8. Mixed hadm_id values within a conflict group reporting and retention
9. Raw input immutability and repeat-run determinism
10. Policy 0 retaining every source row without arbitration
"""

import numpy as np
import pandas as pd
import pytest

from src.temporal.canonical_schema import (
    create_canonical_event_frame,
    validate_canonical_schema,
)
from src.temporal.conflict_arbitration import (
    POLICY_0_RAW,
    POLICY_A_STORETIME,
    POLICY_B_STAT,
    POLICY_C_MEAN,
    POLICY_D_EXTREMUM,
    UnsupportedPolicyAnalyteError,
    arbitrate_canonical_frame,
    get_effective_events,
)


def _build_test_canonical_frame(rows: list[dict]) -> pd.DataFrame:
    """Helper to convert synthetic row dictionaries into a validated canonical DataFrame."""
    base_rows = []
    for i, r in enumerate(rows):
        row_dict = {
            "labevent_id": r.get("labevent_id", 1000 + i),
            "subject_id": r.get("subject_id", 10001),
            "hadm_id": r.get("hadm_id", 20001.0),
            "itemid": r.get("itemid", 50912),
            "charttime": r.get("charttime", "2180-05-01 10:00:00"),
            "storetime": r.get("storetime", "2180-05-01 11:30:00"),
            "value": str(r.get("valuenum", "1.0")),
            "valuenum": r.get("valuenum", 1.0),
            "valueuom": r.get("valueuom", "mg/dL"),
            "valuenum_harmonized": r.get("valuenum_harmonized", r.get("valuenum", 1.0)),
            "unit_harmonized": r.get("unit_harmonized", r.get("valueuom", "mg/dL")),
            "unit_flag": "STANDARD",
            "flag": None,
            "priority": r.get("priority", "ROUTINE"),
            "is_exact_duplicate": r.get("is_exact_duplicate", False),
            "is_conflicting_measurement": r.get("is_conflicting_measurement", False),
            "gender": "F",
            "anchor_age": 60,
        }
        base_rows.append(row_dict)

    raw_df = pd.DataFrame(base_rows)
    return create_canonical_event_frame(raw_df)


# --------------------------------------------------------------------------
# Test 1: Policy 0 Retaining Every Source Row
# --------------------------------------------------------------------------
def test_policy_0_retains_every_source_row():
    """Verify that POLICY_0_RAW preserves all records without deduplication or resolution."""
    df = _build_test_canonical_frame([
        {"labevent_id": 1, "charttime": "2180-05-01 10:00:00", "valuenum": 1.2, "is_exact_duplicate": True},
        {"labevent_id": 2, "charttime": "2180-05-01 10:00:00", "valuenum": 1.2, "is_exact_duplicate": True},
        {"labevent_id": 3, "charttime": "2180-05-01 12:00:00", "valuenum": 1.5, "is_conflicting_measurement": True},
        {"labevent_id": 4, "charttime": "2180-05-01 12:00:00", "valuenum": 2.1, "is_conflicting_measurement": True},
    ])

    res_df, summary = arbitrate_canonical_frame(df, policy_id=POLICY_0_RAW)

    assert len(res_df) == 4
    assert summary.total_rows_input == 4
    assert summary.total_rows_output == 4
    assert summary.retained_rows_count == 4
    assert summary.superseded_duplicate_count == 0
    assert summary.superseded_conflict_count == 0
    assert summary.synthesized_aggregate_count == 0
    assert (res_df["arbitration_role"] == "UNARBITRATED_RAW").all()
    assert (res_df["arbitration_policy_id"] == POLICY_0_RAW).all()
    assert res_df["valuenum_effective"].notna().all()


# --------------------------------------------------------------------------
# Test 2: Exact Duplicates Including Null hadm_id
# --------------------------------------------------------------------------
def test_exact_duplicates_including_null_hadm_id():
    """Verify exact deduplication selects one primary and marks others as superseded, even for outpatient null hadm_id."""
    df = _build_test_canonical_frame([
        # Inpatient duplicate pair
        {"labevent_id": 10, "hadm_id": 2001.0, "charttime": "2180-01-01 10:00:00", "storetime": "2180-01-01 11:00:00", "valuenum": 1.0, "is_exact_duplicate": True},
        {"labevent_id": 11, "hadm_id": 2001.0, "charttime": "2180-01-01 10:00:00", "storetime": "2180-01-01 12:00:00", "valuenum": 1.0, "is_exact_duplicate": True},
        # Outpatient duplicate pair (hadm_id is NaN)
        {"labevent_id": 20, "hadm_id": np.nan, "charttime": "2180-02-01 10:00:00", "storetime": "2180-02-01 11:30:00", "valuenum": 2.5, "is_exact_duplicate": True},
        {"labevent_id": 21, "hadm_id": np.nan, "charttime": "2180-02-01 10:00:00", "storetime": "2180-02-01 11:00:00", "valuenum": 2.5, "is_exact_duplicate": True},
    ])

    res_df, summary = arbitrate_canonical_frame(df, policy_id=POLICY_A_STORETIME)

    assert len(res_df) == 4
    assert summary.exact_duplicate_groups_count == 2
    assert summary.superseded_duplicate_count == 2
    assert summary.primary_selected_count == 2

    # Inpatient pair: storetime 11:00 is earlier than 12:00 -> labevent 10 is primary
    r10 = res_df[res_df["labevent_id"] == 10].iloc[0]
    r11 = res_df[res_df["labevent_id"] == 11].iloc[0]
    assert r10["arbitration_role"] == "PRIMARY_SELECTED"
    assert r10["valuenum_effective"] == 1.0
    assert r11["arbitration_role"] == "SUPERSEDED_DUPLICATE"
    assert pd.isna(r11["valuenum_effective"])

    # Outpatient pair: storetime 11:00 is earlier than 11:30 -> labevent 21 is primary
    r20 = res_df[res_df["labevent_id"] == 20].iloc[0]
    r21 = res_df[res_df["labevent_id"] == 21].iloc[0]
    assert r21["arbitration_role"] == "PRIMARY_SELECTED"
    assert r21["valuenum_effective"] == 2.5
    assert r20["arbitration_role"] == "SUPERSEDED_DUPLICATE"
    assert pd.isna(r20["valuenum_effective"])

    # Effective events view
    eff_df = get_effective_events(res_df)
    assert len(eff_df) == 2
    assert sorted(eff_df["labevent_id"].tolist()) == [10, 21]


# --------------------------------------------------------------------------
# Test 3: Conflicting Values with Different Storetimes
# --------------------------------------------------------------------------
def test_conflicting_values_with_different_storetimes():
    """Verify Policy A selects latest storetime for conflicting measurements."""
    df = _build_test_canonical_frame([
        # Row 1 filed earlier at 11:00 (valuenum=1.2)
        {"labevent_id": 101, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.2, "is_conflicting_measurement": True},
        # Row 2 filed later at 13:00 (valuenum=1.8 - e.g. analyzer re-run)
        {"labevent_id": 102, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 13:00:00", "valuenum": 1.8, "is_conflicting_measurement": True},
    ])

    res_df, summary = arbitrate_canonical_frame(df, policy_id=POLICY_A_STORETIME)

    assert summary.conflict_groups_count == 1
    assert summary.superseded_conflict_count == 1
    assert summary.primary_selected_count == 1

    r101 = res_df[res_df["labevent_id"] == 101].iloc[0]
    r102 = res_df[res_df["labevent_id"] == 102].iloc[0]

    assert r102["arbitration_role"] == "PRIMARY_SELECTED"
    assert r102["valuenum_effective"] == 1.8
    assert r101["arbitration_role"] == "SUPERSEDED_CONFLICT"
    assert pd.isna(r101["valuenum_effective"])


# --------------------------------------------------------------------------
# Test 4: Equal Storetimes and Deterministic Tie-Breaking
# --------------------------------------------------------------------------
def test_equal_storetimes_and_deterministic_tie_breaking():
    """Verify deterministic tie-breaking (lowest labevent_id) when storetimes are identical."""
    df = _build_test_canonical_frame([
        {"labevent_id": 300, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.4, "is_conflicting_measurement": True},
        {"labevent_id": 200, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.9, "is_conflicting_measurement": True},
    ])

    # Run in forward and reversed order to verify row-order independence
    res_df1, _ = arbitrate_canonical_frame(df, policy_id=POLICY_A_STORETIME)
    res_df2, _ = arbitrate_canonical_frame(df.iloc[::-1], policy_id=POLICY_A_STORETIME)

    # In both cases, labevent_id 200 (lower ID) must win the tie-break
    r200_1 = res_df1[res_df1["labevent_id"] == 200].iloc[0]
    r300_1 = res_df1[res_df1["labevent_id"] == 300].iloc[0]
    assert r200_1["arbitration_role"] == "PRIMARY_SELECTED"
    assert r300_1["arbitration_role"] == "SUPERSEDED_CONFLICT"

    r200_2 = res_df2[res_df2["labevent_id"] == 200].iloc[0]
    assert r200_2["arbitration_role"] == "PRIMARY_SELECTED"


# --------------------------------------------------------------------------
# Test 5: STAT, ROUTINE, Missing and Unknown Priority
# --------------------------------------------------------------------------
def test_priority_stat_routine_missing_and_unknown():
    """Verify Policy B priority arbitration and fallback behaviors."""
    # Subtest 5A: STAT vs ROUTINE (STAT wins despite earlier storetime)
    df_stat_routine = _build_test_canonical_frame([
        {"labevent_id": 1, "storetime": "2180-01-01 11:00:00", "valuenum": 1.2, "priority": "STAT", "is_conflicting_measurement": True},
        {"labevent_id": 2, "storetime": "2180-01-01 13:00:00", "valuenum": 1.8, "priority": "ROUTINE", "is_conflicting_measurement": True},
    ])
    res_a, _ = arbitrate_canonical_frame(df_stat_routine, policy_id=POLICY_B_STAT)
    assert res_a[res_a["labevent_id"] == 1].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"
    assert res_a[res_a["labevent_id"] == 2].iloc[0]["arbitration_role"] == "SUPERSEDED_CONFLICT"

    # Subtest 5B: STAT vs Missing priority (STAT wins)
    df_stat_missing = _build_test_canonical_frame([
        {"labevent_id": 3, "storetime": "2180-01-01 11:00:00", "valuenum": 1.2, "priority": "STAT", "is_conflicting_measurement": True},
        {"labevent_id": 4, "storetime": "2180-01-01 13:00:00", "valuenum": 1.8, "priority": None, "is_conflicting_measurement": True},
    ])
    res_b, _ = arbitrate_canonical_frame(df_stat_missing, policy_id=POLICY_B_STAT)
    assert res_b[res_b["labevent_id"] == 3].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # Subtest 5C: STAT vs Unknown string priority (STAT wins)
    df_stat_unknown = _build_test_canonical_frame([
        {"labevent_id": 5, "storetime": "2180-01-01 11:00:00", "valuenum": 1.2, "priority": "STAT", "is_conflicting_measurement": True},
        {"labevent_id": 6, "storetime": "2180-01-01 13:00:00", "valuenum": 1.8, "priority": "RESEARCH_PROTOCOL", "is_conflicting_measurement": True},
    ])
    res_c, _ = arbitrate_canonical_frame(df_stat_unknown, policy_id=POLICY_B_STAT)
    assert res_c[res_c["labevent_id"] == 5].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # Subtest 5D: ROUTINE vs ROUTINE (Tied: fall back to Policy A latest storetime)
    df_routine_routine = _build_test_canonical_frame([
        {"labevent_id": 7, "storetime": "2180-01-01 11:00:00", "valuenum": 1.2, "priority": "ROUTINE", "is_conflicting_measurement": True},
        {"labevent_id": 8, "storetime": "2180-01-01 13:00:00", "valuenum": 1.8, "priority": "ROUTINE", "is_conflicting_measurement": True},
    ])
    res_d, _ = arbitrate_canonical_frame(df_routine_routine, policy_id=POLICY_B_STAT)
    assert res_d[res_d["labevent_id"] == 8].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # Subtest 5E: ROUTINE vs Missing (Neither is STAT: fall back to Policy A latest storetime)
    df_routine_missing = _build_test_canonical_frame([
        {"labevent_id": 9, "storetime": "2180-01-01 14:00:00", "valuenum": 1.2, "priority": None, "is_conflicting_measurement": True},
        {"labevent_id": 10, "storetime": "2180-01-01 11:00:00", "valuenum": 1.8, "priority": "ROUTINE", "is_conflicting_measurement": True},
    ])
    res_e, _ = arbitrate_canonical_frame(df_routine_missing, policy_id=POLICY_B_STAT)
    assert res_e[res_e["labevent_id"] == 9].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"


# --------------------------------------------------------------------------
# Test 6: Analytical Mean Aggregation with Source Provenance
# --------------------------------------------------------------------------
def test_mean_aggregation_with_source_provenance():
    """Verify Policy C synthesizes a mean value, marks source rows superseded, and links provenance."""
    df = _build_test_canonical_frame([
        {"labevent_id": 101, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.2, "is_conflicting_measurement": True},
        {"labevent_id": 102, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 12:00:00", "valuenum": 1.8, "is_conflicting_measurement": True},
    ])

    res_df, summary = arbitrate_canonical_frame(df, policy_id=POLICY_C_MEAN)

    assert summary.total_rows_input == 2
    assert summary.total_rows_output == 3  # 2 source rows + 1 synthesized
    assert summary.superseded_conflict_count == 2
    assert summary.synthesized_aggregate_count == 1
    assert summary.retained_rows_count == 1

    # Source rows must be superseded
    r101 = res_df[res_df["labevent_id"] == 101].iloc[0]
    r102 = res_df[res_df["labevent_id"] == 102].iloc[0]
    assert r101["arbitration_role"] == "SUPERSEDED_CONFLICT"
    assert pd.isna(r101["valuenum_effective"])
    assert r102["arbitration_role"] == "SUPERSEDED_CONFLICT"
    assert pd.isna(r102["valuenum_effective"])

    # Synthesized row
    synth_row = res_df[res_df["arbitration_role"] == "SYNTHESIZED_AGGREGATE"].iloc[0]
    assert synth_row["valuenum_harmonized"] == 1.5
    assert synth_row["valuenum_effective"] == 1.5
    assert synth_row["labevent_id"] < 0  # Deterministic negative ID
    assert synth_row["source_labevent_ids"] == "101,102"
    assert "SYNTHESIZED_MEAN(1.2,1.8)" in synth_row["raw_value_text"]
    assert pd.isna(synth_row["raw_valuenum"])


# --------------------------------------------------------------------------
# Test 7: Policy D for Supported Analytes and Rejection for Sodium / Glucose
# --------------------------------------------------------------------------
def test_policy_d_all_supported_analytes_and_rejection_for_sodium_glucose():
    """Verify Policy D extremum rules for Creatinine, BUN, Bicarbonate, Potassium, and rejection of Sodium/Glucose."""
    # 1. Creatinine (max)
    df_cr = _build_test_canonical_frame([
        {"labevent_id": 1, "itemid": 50912, "valuenum": 1.2, "is_conflicting_measurement": True},
        {"labevent_id": 2, "itemid": 50912, "valuenum": 2.4, "is_conflicting_measurement": True},
    ])
    res_cr, _ = arbitrate_canonical_frame(df_cr, policy_id=POLICY_D_EXTREMUM)
    assert res_cr[res_cr["labevent_id"] == 2].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # 2. Urea Nitrogen (max)
    df_bun = _build_test_canonical_frame([
        {"labevent_id": 3, "itemid": 51006, "valuenum": 35.0, "is_conflicting_measurement": True},
        {"labevent_id": 4, "itemid": 51006, "valuenum": 20.0, "is_conflicting_measurement": True},
    ])
    res_bun, _ = arbitrate_canonical_frame(df_bun, policy_id=POLICY_D_EXTREMUM)
    assert res_bun[res_bun["labevent_id"] == 3].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # 3. Bicarbonate (min)
    df_hco3 = _build_test_canonical_frame([
        {"labevent_id": 5, "itemid": 50882, "valuenum": 24.0, "is_conflicting_measurement": True},
        {"labevent_id": 6, "itemid": 50882, "valuenum": 16.0, "is_conflicting_measurement": True},
    ])
    res_hco3, _ = arbitrate_canonical_frame(df_hco3, policy_id=POLICY_D_EXTREMUM)
    assert res_hco3[res_hco3["labevent_id"] == 6].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # 4. Potassium (furthest from 4.0)
    df_k = _build_test_canonical_frame([
        {"labevent_id": 7, "itemid": 50971, "valuenum": 3.6, "is_conflicting_measurement": True},  # |3.6-4.0| = 0.4
        {"labevent_id": 8, "itemid": 50971, "valuenum": 5.2, "is_conflicting_measurement": True},  # |5.2-4.0| = 1.2
    ])
    res_k, _ = arbitrate_canonical_frame(df_k, policy_id=POLICY_D_EXTREMUM)
    assert res_k[res_k["labevent_id"] == 8].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # Potassium equidistant tie: 3.5 vs 4.5 (|dist| = 0.5) -> prefer higher value (4.5)
    df_k_tie = _build_test_canonical_frame([
        {"labevent_id": 9, "itemid": 50971, "valuenum": 3.5, "is_conflicting_measurement": True},
        {"labevent_id": 10, "itemid": 50971, "valuenum": 4.5, "is_conflicting_measurement": True},
    ])
    res_k_tie, _ = arbitrate_canonical_frame(df_k_tie, policy_id=POLICY_D_EXTREMUM)
    assert res_k_tie[res_k_tie["labevent_id"] == 10].iloc[0]["arbitration_role"] == "PRIMARY_SELECTED"

    # 5. Sodium (Explicitly Unsupported -> Must Raise Error)
    df_na = _build_test_canonical_frame([
        {"labevent_id": 11, "itemid": 50983, "valuenum": 135.0, "is_conflicting_measurement": True},
        {"labevent_id": 12, "itemid": 50983, "valuenum": 142.0, "is_conflicting_measurement": True},
    ])
    with pytest.raises(UnsupportedPolicyAnalyteError, match="explicitly unsupported under Policy D"):
        arbitrate_canonical_frame(df_na, policy_id=POLICY_D_EXTREMUM, raise_on_unsupported=True)

    # 6. Glucose (Explicitly Unsupported -> Must Raise Error)
    df_glu = _build_test_canonical_frame([
        {"labevent_id": 13, "itemid": 50931, "valuenum": 95.0, "is_conflicting_measurement": True},
        {"labevent_id": 14, "itemid": 50931, "valuenum": 180.0, "is_conflicting_measurement": True},
    ])
    with pytest.raises(UnsupportedPolicyAnalyteError, match="explicitly unsupported under Policy D"):
        arbitrate_canonical_frame(df_glu, policy_id=POLICY_D_EXTREMUM, raise_on_unsupported=True)

    # 7. Flagging Mode (raise_on_unsupported=False)
    res_flag, summary_flag = arbitrate_canonical_frame(df_na, policy_id=POLICY_D_EXTREMUM, raise_on_unsupported=False)
    assert summary_flag.unsupported_analyte_conflicts_count == 1
    assert (res_flag["arbitration_role"] == "UNSUPPORTED_POLICY_D_ANALYTE").all()


# --------------------------------------------------------------------------
# Test 8: Missing Numerical Values Handling
# --------------------------------------------------------------------------
def test_missing_numerical_values_handling():
    """Verify that rows with missing numerical values are never silently discarded."""
    df = _build_test_canonical_frame([
        {"labevent_id": 1, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": np.nan, "is_conflicting_measurement": True},
        {"labevent_id": 2, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 12:00:00", "valuenum": 2.0, "is_conflicting_measurement": True},
    ])

    # Under Policy C, mean is synthesized from eligible numeric values (2.0)
    res_c, summary_c = arbitrate_canonical_frame(df, policy_id=POLICY_C_MEAN)
    assert len(res_c) == 3  # Both source rows preserved + 1 synthesized
    assert summary_c.retained_rows_count == 1

    synth_row = res_c[res_c["arbitration_role"] == "SYNTHESIZED_AGGREGATE"].iloc[0]
    assert synth_row["valuenum_effective"] == 2.0

    # Under Policy A, missing value row is retained as superseded, not dropped
    res_a, summary_a = arbitrate_canonical_frame(df, policy_id=POLICY_A_STORETIME)
    assert len(res_a) == 2
    assert 1 in res_a["labevent_id"].values
    assert 2 in res_a["labevent_id"].values


# --------------------------------------------------------------------------
# Test 9: Mixed hadm_id Values Within a Conflict Group
# --------------------------------------------------------------------------
def test_mixed_hadm_id_within_conflict_group():
    """Verify that conflict groups with disparate hadm_id are tracked and reported without dropping."""
    df = _build_test_canonical_frame([
        # Row 1 inpatient (hadm_id = 20001.0)
        {"labevent_id": 1, "hadm_id": 20001.0, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.2, "is_conflicting_measurement": True},
        # Row 2 outpatient / ED (hadm_id = np.nan)
        {"labevent_id": 2, "hadm_id": np.nan, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 12:00:00", "valuenum": 1.8, "is_conflicting_measurement": True},
    ])

    res_df, summary = arbitrate_canonical_frame(df, policy_id=POLICY_A_STORETIME)

    assert summary.groups_with_mixed_hadm_id == 1
    assert len(summary.mixed_hadm_details) == 1
    assert summary.mixed_hadm_details[0]["subject_id"] == 10001
    assert len(res_df) == 2


# --------------------------------------------------------------------------
# Test 10: Raw Input Immutability and Repeat-Run Determinism
# --------------------------------------------------------------------------
def test_raw_input_immutability_and_repeat_run_determinism():
    """Verify input DataFrame is never modified and repeat runs yield identical outputs."""
    df = _build_test_canonical_frame([
        {"labevent_id": 10, "charttime": "2180-01-01 10:00:00", "valuenum": 1.0, "is_exact_duplicate": True},
        {"labevent_id": 11, "charttime": "2180-01-01 10:00:00", "valuenum": 1.0, "is_exact_duplicate": True},
        {"labevent_id": 20, "charttime": "2180-02-01 10:00:00", "valuenum": 1.2, "is_conflicting_measurement": True},
        {"labevent_id": 21, "charttime": "2180-02-01 10:00:00", "valuenum": 1.8, "is_conflicting_measurement": True},
    ])

    # Snapshot of input
    input_snapshot = df.copy(deep=True)

    # First run
    res1, sum1 = arbitrate_canonical_frame(df, policy_id=POLICY_A_STORETIME)

    # Assert input was NOT mutated
    pd.testing.assert_frame_equal(df, input_snapshot)

    # Second run
    res2, sum2 = arbitrate_canonical_frame(df, policy_id=POLICY_A_STORETIME)

    # Assert outputs are strictly identical
    pd.testing.assert_frame_equal(res1, res2)
    assert sum1.to_dict() == sum2.to_dict()


# --------------------------------------------------------------------------
# Test 11: Integration Slice on Real Extracted Cohort
# --------------------------------------------------------------------------
def test_arbitration_on_real_extracted_cohort_slice():
    """Integration test: Run all 4 general policies on a 500-row slice of the real extracted cohort."""
    cohort_path = "data/processed/cohort_renal_bounded.csv"
    raw_slice = pd.read_csv(cohort_path, nrows=500)
    canonical_df = create_canonical_event_frame(raw_slice)

    for pol in [POLICY_0_RAW, POLICY_A_STORETIME, POLICY_B_STAT, POLICY_C_MEAN]:
        res_df, summary = arbitrate_canonical_frame(canonical_df, policy_id=pol)
        is_valid, errors = validate_canonical_schema(res_df)
        assert is_valid, f"Policy {pol} failed schema validation: {errors}"
        assert summary.total_rows_input == 500


# --------------------------------------------------------------------------
# Test 12: Policy C Multiple Aggregates Uniqueness and Source ID Isolation
# --------------------------------------------------------------------------
def test_policy_c_multiple_aggregates_uniqueness():
    """Verify that multiple synthesized aggregate rows receive unique, strictly negative IDs that do not collide."""
    df = _build_test_canonical_frame([
        # Group 1: Creatinine at 10:00
        {"labevent_id": 101, "itemid": 50912, "charttime": "2180-05-01 10:00:00", "valuenum": 1.2, "is_conflicting_measurement": True},
        {"labevent_id": 102, "itemid": 50912, "charttime": "2180-05-01 10:00:00", "valuenum": 1.8, "is_conflicting_measurement": True},
        # Group 2: BUN at 10:00
        {"labevent_id": 201, "itemid": 51006, "charttime": "2180-05-01 10:00:00", "valuenum": 20.0, "is_conflicting_measurement": True},
        {"labevent_id": 202, "itemid": 51006, "charttime": "2180-05-01 10:00:00", "valuenum": 30.0, "is_conflicting_measurement": True},
        # Group 3: Creatinine at 16:00
        {"labevent_id": 301, "itemid": 50912, "charttime": "2180-05-01 16:00:00", "valuenum": 1.4, "is_conflicting_measurement": True},
        {"labevent_id": 302, "itemid": 50912, "charttime": "2180-05-01 16:00:00", "valuenum": 2.0, "is_conflicting_measurement": True},
    ])

    res_df, summary = arbitrate_canonical_frame(df, policy_id=POLICY_C_MEAN)

    assert summary.synthesized_aggregate_count == 3
    synth_rows = res_df[res_df["arbitration_role"] == "SYNTHESIZED_AGGREGATE"]
    synth_ids = synth_rows["labevent_id"].tolist()

    # All synthesized IDs must be strictly unique
    assert len(synth_ids) == 3
    assert len(set(synth_ids)) == 3

    # All synthesized IDs must be strictly negative
    assert (synth_rows["labevent_id"] < 0).all()

    # No overlap with any source labevent_id
    source_ids = set(df["labevent_id"])
    assert len(set(synth_ids) & source_ids) == 0


# --------------------------------------------------------------------------
# Test 13: Policy C Collision Detection Explicit Failure
# --------------------------------------------------------------------------
def test_policy_c_synthetic_id_collision_detection(monkeypatch):
    """Verify that collision detection explicitly raises ArbitrationError if synthetic IDs collide or overlap source IDs."""
    from src.temporal.conflict_arbitration import ArbitrationError
    import src.temporal.conflict_arbitration as ca

    df = _build_test_canonical_frame([
        {"labevent_id": 101, "itemid": 50912, "charttime": "2180-05-01 10:00:00", "valuenum": 1.2, "is_conflicting_measurement": True},
        {"labevent_id": 102, "itemid": 50912, "charttime": "2180-05-01 10:00:00", "valuenum": 1.8, "is_conflicting_measurement": True},
        {"labevent_id": 201, "itemid": 51006, "charttime": "2180-05-01 10:00:00", "valuenum": 20.0, "is_conflicting_measurement": True},
        {"labevent_id": 202, "itemid": 51006, "charttime": "2180-05-01 10:00:00", "valuenum": 30.0, "is_conflicting_measurement": True},
    ])

    # Case A: Monkeypatch ID generator to simulate an internal hash collision between two groups
    monkeypatch.setattr(ca, "_generate_synthetic_labevent_id", lambda sid, itemid, ctime: -999999)
    with pytest.raises(ArbitrationError, match="Collision detected among synthesized aggregate"):
        arbitrate_canonical_frame(df, policy_id=POLICY_C_MEAN)

    # Case B: Monkeypatch ID generator to return distinct synthetic IDs where one collides with source ID 101
    monkeypatch.setattr(ca, "_generate_synthetic_labevent_id", lambda sid, itemid, ctime: 101 if itemid == 50912 else -88888)
    with pytest.raises(ArbitrationError, match="overlaps with existing source labevent_id"):
        arbitrate_canonical_frame(df, policy_id=POLICY_C_MEAN)

