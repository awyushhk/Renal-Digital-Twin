"""Tests for Module 1.4: End-to-End Integration and Final Verification Harness.

Verifies:
1. Cohort reference artifact presence, dimensions, and SHA-256 hash integrity check.
2. Exact arbitration superseded counts on the FULL 303,460-row bounded cohort:
   - 94 superseded duplicates
   - 134 superseded conflicts
   - 228 total superseded records
   - 303,232 active effective records
   - 0 mixed hadm_id collision groups
3. Fast 5,000-row integration test for rapid regression coverage.
4. Operational query-time filtration & immutability:
   - Result drawn at 10:00, filed at 12:30 is absent at 11:00, visible at 12:30.
   - Later arriving results never retroactively rewrite past snapshots.
5. Strict patient boundary isolation and co-temporal ratio gating.
6. Dynamic negative controls (Control 2A leakage, Control 2B bleed, Control 2C outpatient loss).
7. Full-cohort source row accounting (259,869 cells, 43,591 raw collapsed rows).
8. End-to-end acceptance suite execution verdict is PASS.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest

from src.temporal.canonical_schema import (
    create_canonical_event_frame,
    validate_canonical_schema,
)
from src.temporal.conflict_arbitration import (
    POLICY_A_STORETIME,
    arbitrate_canonical_frame,
    get_effective_events,
)
from src.temporal.temporal_features import (
    generate_biological_features,
    generate_operational_replay_snapshot,
)
from src.temporal.representation_pipeline import (
    calculate_source_row_accounting,
    run_three_way_comparison_pipeline,
)
from src.validation.end_to_end_validation import (
    DOCUMENTED_COHORT_SHA256,
    ValidationConfig,
    _build_synthetic_test_frame,
    compute_file_sha256,
    verify_cohort_reference_integrity,
    verify_arbitration_superseded_counts,
    verify_operational_query_time_lookahead_and_immutability,
    verify_cotemporal_ratio_and_patient_boundary_isolation,
    verify_dynamic_negative_controls_and_retention,
    verify_three_arm_comparative_pipeline,
    execute_full_end_to_end_acceptance,
)

BOUNDED_COHORT_PATH = Path("data/processed/cohort_renal_bounded.csv")
ADMISSIONS_PATH = Path("data/raw/admissions.csv")


def test_reference_artifact_integrity_and_hashes():
    """Test 1: Verify presence, dimensions, and reference SHA-256 integrity check.

    Targeted requirement:
    Retain the documented cohort SHA-256 as a reference-artifact integrity check.
    If it differs, report a clear mismatch and investigate; do not treat a hardcoded
    hash as a universal validity rule.
    """
    config = ValidationConfig()
    res = verify_cohort_reference_integrity(config)

    assert res["passed"] is True
    assert res["actual_rows"] == 303460
    assert res["actual_patients"] == 2500
    assert res["actual_cols"] == 30

    # Verify SHA-256 reference check
    assert res["actual_sha256"] == DOCUMENTED_COHORT_SHA256
    assert res["hash_match"] is True


def test_arbitration_superseded_counts_match_report():
    """Test 2: Run Policy A arbitration on the FULL 303,460-row bounded cohort.

    Targeted requirement:
    Run test_arbitration_superseded_counts_match_report against the full 303,460-row
    bounded cohort. Verify the reported 94 superseded duplicates, 134 superseded conflicts,
    and 228 total against the same policy and input conventions used to generate the report.
    Do not use proportional slice assertions for these exact cohort totals.
    """
    config = ValidationConfig()
    df_raw = pd.read_csv(BOUNDED_COHORT_PATH)
    assert len(df_raw) == 303460

    canonical_df = create_canonical_event_frame(df_raw)
    arbitrated_df, summary = arbitrate_canonical_frame(
        canonical_df, policy_id=POLICY_A_STORETIME
    )

    # Exact Count Assertions on Full Cohort
    assert summary.total_rows_input == 303460
    assert summary.total_rows_output == 303460
    assert summary.retained_rows_count == 303232
    assert summary.superseded_duplicate_count == 94
    assert summary.superseded_conflict_count == 134
    assert summary.superseded_duplicate_count + summary.superseded_conflict_count == 228
    assert summary.groups_with_mixed_hadm_id == 0

    # Value counts of arbitration_role
    vc = arbitrated_df["arbitration_role"].value_counts().to_dict()
    assert vc.get("PRIMARY_SELECTED") == 303232
    assert vc.get("SUPERSEDED_DUPLICATE") == 94
    assert vc.get("SUPERSEDED_CONFLICT") == 134
    assert vc.get("UNARBITRATED_RAW", 0) == 0


def test_operational_timeline_explicit_query_time_lookahead():
    """Test 3: Operational availability filtration and immutability.

    Targeted requirement:
    Test where a result is collected at 10:00, becomes available at 12:30, and is
    absent from operational features at 11:00 but available from 12:30 onward.
    Also verifies that late-arriving results entering at 12:30 do not retroactively
    alter the operational snapshot previously computed for 11:00.
    """
    res = verify_operational_query_time_lookahead_and_immutability()
    assert res["passed"] is True
    assert res["absent_at_1100"] is True
    assert res["present_at_1230"] is True
    assert res["immutability_preserved"] is True
    assert res["retrospective_only_flag"] is True
    assert res["velocity_ops_present"] is True


def test_cotemporal_ratio_and_patient_boundary_resets():
    """Test 4: Strict patient boundary resets and co-temporal ratio gating.

    Targeted requirement:
    Distinguish biological rates from operational reporting rates.
    Compute the operational BUN/creatinine ratio only when both active measurements
    belong to the same patient and exact charttime epoch and both results have
    storetime <= query_time, with non-negative reporting delays.
    Otherwise return NaN with an explicit reason flag.
    """
    res = verify_cotemporal_ratio_and_patient_boundary_isolation()
    assert bool(res["passed"]) is True
    assert bool(res["ratio_epoch1_valid"]) is True
    assert bool(res["ratio_epoch2_nan"]) is True
    assert bool(res["flag_epoch2_correct"]) is True
    assert bool(res["pat_b_elapsed_zero"]) is True
    assert bool(res["pat_b_velocity_nan"]) is True


def test_fast_5000_row_integration_pipeline():
    """Test 5: Fast 5,000-row integration test across Arms 1, 2, and 3.

    Targeted requirement:
    Keep the 5,000-row integration test for fast regression coverage.
    """
    config = ValidationConfig()
    res = verify_three_arm_comparative_pipeline(sample_rows=5000, config=config)
    assert res["passed"] is True
    assert res["runtime_seconds"] < 60.0

    summary = res["summary"]
    # Arm 1 conventional grid checks
    assert summary.arm1_distinct_grid_days > 0
    assert summary.arm1_intra_day_collapsed_rows > 0

    # Arm 2 flawed controls checks
    assert summary.arm2_control_2a_leakage_drift > 0.0
    assert summary.arm2_control_2b_bleed_detected is True
    assert summary.arm2_control_2c_rows_lost > 0

    # Arm 3 proposed representation checks
    assert summary.arm3_intra_day_collapsed_rows == 0
    assert summary.arm3_future_drift == 0.0
    assert summary.arm3_boundary_violations == 0
    assert summary.arm3_zero_gap_singularities == 0
    assert summary.arm3_operational_lookahead_violations == 0


def test_dynamic_negative_controls_and_source_accounting():
    """Test 6: Verify dynamic negative controls and full-cohort source row accounting.

    Verifies:
    - Normalization leakage drift > 0
    - Cross-patient boundary bleed detected = True
    - Admissions inner join loss = 103,354 rows (34.0585%) and 378 patients (15.1200%)
    - Distinct cells = 259,869
    - Raw collapsed observations = 43,591
    """
    config = ValidationConfig()
    res = verify_dynamic_negative_controls_and_retention(config)
    assert res["passed"] is True
    assert res["leakage_detected"] is True
    assert res["bleed_detected"] is True
    assert res["properly_reset_arm3"] is True
    assert res["rows_lost"] == 103354
    assert res["pts_lost"] == 378
    assert np.isclose(res["loss_pct"], 34.0585, atol=1e-3)
    assert res["distinct_cells"] == 259869
    assert res["raw_collapsed_rows"] == 43591


def test_full_end_to_end_acceptance_verdict_pass():
    """Test 7: Execute full acceptance harness and assert overall PASS verdict."""
    config = ValidationConfig()
    results = execute_full_end_to_end_acceptance(config)

    assert results["overall_status"] == "PASS"
    assert results["total_runtime_seconds"] < 60.0
    assert config.report_output_path.is_file()
