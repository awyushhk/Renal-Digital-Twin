"""Focused tests for Module 1.3 Step 3: Irregular Temporal Features and Dual-Timeline Engine.

Verifies:
1. Dual timelines: biological (F_bio) vs operational (F_ops) availability separation.
2. Prevention of operational lookahead: storetime > query_time is excluded.
3. Prevention of retrospective rewriting: advancing replay does not alter historical snapshots.
4. Manually calculated unequal-interval acceleration test with verified units (unit/h^2).
5. Zero-interval safety: co-temporal collisions map to NaN with zero-gap flags without division-by-zero.
6. Co-temporal BUN/Creatinine ratio: evaluates only on co-temporal draws; rejects missing/non-positive values.
7. Quarantine of anomalous negative reporting delays and missing storetimes from operational timeline.
8. Patient boundary isolation: strict reset of continuous clocks, running stats, and velocities.
9. Outpatient preservation: null hadm_id records retained with is_inpatient=False and dt_admission=NaN.
10. Causality: future perturbations at t > T produce 0.0 shift on past features at t <= T.
11. Integration test on real extracted cohort slice.
12. 3-arm baseline generation tests (Arm 1 LOCF, Arm 2 Flawed Controls).
"""

import numpy as np
import pandas as pd
import pytest

from src.temporal.canonical_schema import create_canonical_event_frame
from src.temporal.conflict_arbitration import arbitrate_canonical_frame, POLICY_A_STORETIME
from src.temporal.temporal_features import (
    calculate_irregular_velocity,
    calculate_unequal_interval_acceleration,
    compute_cotemporal_bun_creatinine_ratio,
    generate_biological_features,
    generate_operational_replay_snapshot,
    generate_arm1_conventional_locf_grid,
    generate_arm2_flawed_controls,
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
    canonical_df = create_canonical_event_frame(raw_df)
    arbitrated_df, _ = arbitrate_canonical_frame(canonical_df, policy_id=POLICY_A_STORETIME)
    return arbitrated_df


# --------------------------------------------------------------------------
# Test 1: Dual-Timeline Availability Separation
# --------------------------------------------------------------------------
def test_dual_timeline_availability_separation():
    """Verify that a result drawn at 10:00 and filed at 12:30 is absent at 11:00 but present at 12:30."""
    df = _build_test_canonical_frame([
        {
            "labevent_id": 101,
            "subject_id": 1001,
            "itemid": 50912,
            "charttime": "2180-05-01 10:00:00",
            "storetime": "2180-05-01 12:30:00",
            "valuenum": 1.5,
        }
    ])

    # 1. Biological timeline: available immediately at charttime 10:00:00
    bio_df, _ = generate_biological_features(df)
    assert len(bio_df) == 1
    assert bio_df.iloc[0]["hours_since_first_observed_event_bio"] == 0.0
    assert bool(bio_df.iloc[0]["is_retrospective_only"]) is True

    # 2. Operational timeline at query_time = 11:00:00 (Result is NOT yet filed)
    ops_1100 = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 11:00:00"))
    assert len(ops_1100) == 0  # Completely absent

    # 3. Operational timeline at query_time = 12:30:00 (Result is NOW filed)
    ops_1230 = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 12:30:00"))
    assert len(ops_1230) == 1
    assert ops_1230.iloc[0]["labevent_id"] == 101
    assert ops_1230.iloc[0]["hours_since_first_available_event_ops"] == 0.0


# --------------------------------------------------------------------------
# Test 2: No Retrospective Rewriting of Operational Replay Snapshots
# --------------------------------------------------------------------------
def test_no_retrospective_rewriting_of_operational_snapshots():
    """Verify that late-arriving results entering at 16:00 do not retroactively alter snapshots at 12:00."""
    df = _build_test_canonical_frame([
        # Test A: Drawn at 09:00, filed at 11:00
        {"labevent_id": 1, "subject_id": 1001, "itemid": 50912, "charttime": "2180-05-01 09:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.2},
        # Test B: Drawn at 10:00, filed late at 16:00 (e.g. send-out / dilution re-run)
        {"labevent_id": 2, "subject_id": 1001, "itemid": 50912, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 16:00:00", "valuenum": 2.5},
    ])

    # Snapshot at 12:00:00
    snap_1200 = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 12:00:00"))
    assert len(snap_1200) == 1
    assert snap_1200.iloc[0]["labevent_id"] == 1

    # Snapshot at 17:00:00 (both are now available)
    snap_1700 = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 17:00:00"))
    assert len(snap_1700) == 2

    # Re-evaluating at 12:00:00 must yield strictly identical snapshot as before
    snap_1200_recheck = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 12:00:00"))
    pd.testing.assert_frame_equal(snap_1200, snap_1200_recheck)


# --------------------------------------------------------------------------
# Test 3: Manually Calculated Unequal-Interval Acceleration
# --------------------------------------------------------------------------
def test_manually_calculated_unequal_interval_acceleration():
    """Verify non-uniform divided-difference acceleration against verified manual calculation.

    Reference calculation:
    - Epoch 1: c1 = 0h, x1 = 1.0 mg/dL
    - Epoch 2: c2 = 4h, x2 = 1.8 mg/dL -> dt12 = 4h, v12 = (1.8-1.0)/4 = 0.2000 mg/dL/h (midpoint = 2h)
    - Epoch 3: c3 = 10h, x3 = 3.6 mg/dL -> dt23 = 6h, v23 = (3.6-1.8)/6 = 0.3000 mg/dL/h (midpoint = 7h)
    - Midpoint separation: (4 + 6)/2 = 5.0h
    - Acceleration: a = (0.3000 - 0.2000) / 5.0 = 0.0200 mg/dL/h^2
    """
    # 1. Pure function test
    a_calc, a_status = calculate_unequal_interval_acceleration(
        x_prev2=1.0,
        x_prev1=1.8,
        x_curr=3.6,
        dt12_hours=4.0,
        dt23_hours=6.0,
    )
    assert a_status == "VALID"
    assert pytest.approx(a_calc, abs=1e-6) == 0.0200

    # 2. DataFrame pipeline test
    df = _build_test_canonical_frame([
        {"labevent_id": 1, "charttime": "2180-05-01 00:00:00", "storetime": "2180-05-01 01:00:00", "valuenum": 1.0},
        {"labevent_id": 2, "charttime": "2180-05-01 04:00:00", "storetime": "2180-05-01 05:00:00", "valuenum": 1.8},
        {"labevent_id": 3, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 3.6},
    ])

    bio_df, _ = generate_biological_features(df)
    r1 = bio_df.iloc[0]
    r2 = bio_df.iloc[1]
    r3 = bio_df.iloc[2]

    # Row 1: First observation
    assert pd.isna(r1["velocity_biological"])
    assert pd.isna(r1["acceleration_biological"])

    # Row 2: Velocity valid, acceleration insufficient history (k=2)
    assert pytest.approx(r2["velocity_biological"], abs=1e-6) == 0.2000
    assert r2["velocity_biological_midpoint_hours"] == 2.0
    assert pd.isna(r2["acceleration_biological"])
    assert r2["acceleration_biological_flag"] == "INSUFFICIENT_HISTORY"

    # Row 3: Acceleration equals manual expectation 0.0200 mg/dL/h^2
    assert pytest.approx(r3["velocity_biological"], abs=1e-6) == 0.3000
    assert r3["velocity_biological_midpoint_hours"] == 7.0
    assert pytest.approx(r3["acceleration_biological"], abs=1e-6) == 0.0200
    assert r3["acceleration_biological_flag"] == "VALID"


# --------------------------------------------------------------------------
# Test 4: Zero-Interval Safety (Singularity Avoidance)
# --------------------------------------------------------------------------
def test_zero_interval_safety_and_singularity_avoidance():
    """Verify that co-temporal observations with dt=0 map to NaN with COLLISION_ZERO_DELTA without division by zero."""
    v, v_flag = calculate_irregular_velocity(1.0, 2.0, dt_hours=0.0)
    assert pd.isna(v)
    assert v_flag == "COLLISION_ZERO_DELTA"

    a, a_flag = calculate_unequal_interval_acceleration(1.0, 2.0, 3.0, dt12_hours=0.0, dt23_hours=2.0)
    assert pd.isna(a)
    assert a_flag == "COLLISION_ZERO_DELTA"


# --------------------------------------------------------------------------
# Test 5: Co-Temporal BUN / Creatinine Ratio
# --------------------------------------------------------------------------
def test_cotemporal_bun_creatinine_ratio():
    """Verify BUN/Creatinine evaluated only on co-temporal draws; rejects missing/non-positive values."""
    # Valid draw: BUN=28, Creatinine=1.4 -> Ratio = 20.0
    r, flag = compute_cotemporal_bun_creatinine_ratio(28.0, 1.4)
    assert flag == "VALID_COTEMPORAL_RATIO"
    assert pytest.approx(r, abs=1e-4) == 20.0

    # Missing constituent: BUN=28, Creatinine=None
    r_miss, flag_miss = compute_cotemporal_bun_creatinine_ratio(28.0, None)
    assert pd.isna(r_miss)
    assert flag_miss == "MISSING_CONSTITUENT"

    # Non-positive creatinine: Creatinine = 0.0 -> Non-physiological, must not divide by zero
    r_zero, flag_zero = compute_cotemporal_bun_creatinine_ratio(28.0, 0.0)
    assert pd.isna(r_zero)
    assert flag_zero == "NON_POSITIVE_CREATININE"

    # Non-positive creatinine: Creatinine = -0.5
    r_neg, flag_neg = compute_cotemporal_bun_creatinine_ratio(28.0, -0.5)
    assert pd.isna(r_neg)
    assert flag_neg == "NON_POSITIVE_CREATININE"


# --------------------------------------------------------------------------
# Test 6: Operational BUN/Cr Ratio Availability
# --------------------------------------------------------------------------
def test_operational_bun_creatinine_ratio_availability():
    """Verify operational BUN/Cr ratio evaluates only when both constituents are filed <= query_time."""
    df = _build_test_canonical_frame([
        # BUN drawn at 10:00, filed at 11:30
        {"labevent_id": 1, "itemid": 51006, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:30:00", "valuenum": 30.0},
        # Creatinine drawn at 10:00, filed later at 13:00
        {"labevent_id": 2, "itemid": 50912, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 13:00:00", "valuenum": 1.5},
    ])

    # At 12:00:00: BUN is filed, but Creatinine is not yet filed
    snap_1200 = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 12:00:00"))
    assert len(snap_1200) == 1
    assert pd.isna(snap_1200.iloc[0]["bun_creatinine_ratio_ops"])
    assert snap_1200.iloc[0]["bun_creatinine_ratio_ops_flag"] == "RESULT_NOT_YET_FILED"

    # At 13:30:00: Both are filed
    snap_1330 = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 13:30:00"))
    assert len(snap_1330) == 2
    r_bun = snap_1330[snap_1330["itemid"] == 51006].iloc[0]
    assert pytest.approx(r_bun["bun_creatinine_ratio_ops"], abs=1e-4) == 20.0
    assert r_bun["bun_creatinine_ratio_ops_flag"] == "VALID_COTEMPORAL_RATIO"


# --------------------------------------------------------------------------
# Test 7: Quarantine of Anomalous Negative Delays from Operational Timeline
# --------------------------------------------------------------------------
def test_quarantine_negative_delays_from_operational_timeline():
    """Verify that records with negative reporting delay are barred from operational timeline."""
    df = _build_test_canonical_frame([
        # Normal record: delay = 1.0h
        {"labevent_id": 1, "charttime": "2180-05-01 10:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.2},
        # Anomalous record: storetime precedes charttime by 1.0h (clock skew)
        {"labevent_id": 2, "charttime": "2180-05-01 12:00:00", "storetime": "2180-05-01 11:00:00", "valuenum": 1.8},
    ])

    # Biological timeline retains both with anomaly flag
    bio_df, _ = generate_biological_features(df)
    assert len(bio_df) == 2
    r_anom = bio_df[bio_df["labevent_id"] == 2].iloc[0]
    assert r_anom["delay_anomaly_flag"] == "NEGATIVE_REPORTING_DELAY"

    # Operational timeline at query_time = 13:00:00 must quarantine the anomalous record
    ops_df = generate_operational_replay_snapshot(df, query_time=pd.Timestamp("2180-05-01 13:00:00"))
    assert len(ops_df) == 1
    assert ops_df.iloc[0]["labevent_id"] == 1


# --------------------------------------------------------------------------
# Test 8: Patient Boundary Isolation
# --------------------------------------------------------------------------
def test_patient_boundary_isolation():
    """Verify that running stats, clocks, and velocities reset strictly at patient boundaries."""
    df = _build_test_canonical_frame([
        # Patient 1
        {"labevent_id": 1, "subject_id": 101, "charttime": "2180-05-01 10:00:00", "valuenum": 1.0},
        {"labevent_id": 2, "subject_id": 101, "charttime": "2180-05-01 14:00:00", "valuenum": 2.0},
        # Patient 2 (starts after patient 1)
        {"labevent_id": 3, "subject_id": 102, "charttime": "2180-05-01 16:00:00", "valuenum": 3.0},
    ])

    bio_df, _ = generate_biological_features(df)

    p1_first = bio_df[bio_df["subject_id"] == 101].iloc[0]
    p1_second = bio_df[bio_df["subject_id"] == 101].iloc[1]
    p2_first = bio_df[bio_df["subject_id"] == 102].iloc[0]

    # Patient 1 first: 0.0 elapsed lifetime
    assert p1_first["hours_since_first_observed_event_bio"] == 0.0
    assert pd.isna(p1_first["velocity_biological"])

    # Patient 1 second: 4.0 elapsed lifetime, velocity = (2.0 - 1.0)/4 = 0.25
    assert p1_second["hours_since_first_observed_event_bio"] == 4.0
    assert pytest.approx(p1_second["velocity_biological"], abs=1e-6) == 0.25

    # Patient 2 first: MUST reset to 0.0 elapsed lifetime and NaN velocity (no bleed from Patient 1)
    assert p2_first["hours_since_first_observed_event_bio"] == 0.0
    assert pd.isna(p2_first["velocity_biological"])
    assert pd.isna(p2_first["dt_previous_event_hours_bio"])


# --------------------------------------------------------------------------
# Test 9: Outpatient Null hadm_id Preservation
# --------------------------------------------------------------------------
def test_outpatient_hadm_id_preservation():
    """Verify that outpatient records with null hadm_id are retained with is_inpatient=False and dt_admission=NaN."""
    df = _build_test_canonical_frame([
        {"labevent_id": 1, "hadm_id": np.nan, "charttime": "2180-05-01 10:00:00", "valuenum": 1.2},
    ])

    bio_df, _ = generate_biological_features(df)
    r = bio_df.iloc[0]
    assert pd.isna(r["hadm_id"])
    assert bool(r["is_inpatient"]) is False
    assert pd.isna(r["dt_admission_hours_bio"])


# --------------------------------------------------------------------------
# Test 10: Future Perturbation Invariance
# --------------------------------------------------------------------------
def test_future_perturbation_invariance():
    """Verify modifying observations at t > T produces 0.0 shift on past features at t <= T."""
    base_rows = [
        {"labevent_id": 1, "subject_id": 1001, "charttime": "2180-05-01 10:00:00", "valuenum": 1.0},
        {"labevent_id": 2, "subject_id": 1001, "charttime": "2180-05-01 14:00:00", "valuenum": 1.5},
        {"labevent_id": 3, "subject_id": 1001, "charttime": "2180-05-01 20:00:00", "valuenum": 2.0},  # Future event
    ]

    perturbed_rows = [
        {"labevent_id": 1, "subject_id": 1001, "charttime": "2180-05-01 10:00:00", "valuenum": 1.0},
        {"labevent_id": 2, "subject_id": 1001, "charttime": "2180-05-01 14:00:00", "valuenum": 1.5},
        {"labevent_id": 3, "subject_id": 1001, "charttime": "2180-05-01 20:00:00", "valuenum": 99.9},  # Severely perturbed
    ]

    df_base = _build_test_canonical_frame(base_rows)
    df_pert = _build_test_canonical_frame(perturbed_rows)

    bio_base, _ = generate_biological_features(df_base)
    bio_pert, _ = generate_biological_features(df_pert)

    # Historical cutoff T = 14:00:00
    past_base = bio_base[bio_base["charttime"] <= "2180-05-01 14:00:00"]
    past_pert = bio_pert[bio_pert["charttime"] <= "2180-05-01 14:00:00"]

    # Past running mean and velocity must be bitwise identical
    assert past_base["running_mean_creatinine"].tolist() == past_pert["running_mean_creatinine"].tolist()
    assert np.allclose(
        past_base["velocity_biological"].fillna(-999),
        past_pert["velocity_biological"].fillna(-999),
    )


# --------------------------------------------------------------------------
# Test 11: Integration Slice on Real Extracted Cohort
# --------------------------------------------------------------------------
def test_temporal_features_on_real_extracted_cohort_slice():
    """Integration test: Generate biological and operational features on a 500-row slice of real cohort."""
    cohort_path = "data/processed/cohort_renal_bounded.csv"
    raw_slice = pd.read_csv(cohort_path, nrows=500)
    canonical_df = create_canonical_event_frame(raw_slice)
    arbitrated_df, _ = arbitrate_canonical_frame(canonical_df, policy_id=POLICY_A_STORETIME)

    # 1. Biological timeline
    bio_df, summary = generate_biological_features(arbitrated_df)
    assert len(bio_df) > 0
    assert summary.patient_count > 0
    assert bio_df["hours_since_first_observed_event_bio"].min() == 0.0

    # 2. Operational replay snapshot
    mid_storetime = arbitrated_df["storetime"].median()
    ops_df = generate_operational_replay_snapshot(arbitrated_df, query_time=pd.Timestamp(mid_storetime))
    assert len(ops_df) <= len(arbitrated_df)


# --------------------------------------------------------------------------
# Test 12: 3-Arm Comparative Baseline Generation
# --------------------------------------------------------------------------
def test_three_arm_baseline_generators():
    """Verify Arm 1 regular 24h LOCF grid and Arm 2 flawed controls generate without error."""
    df = _build_test_canonical_frame([
        {"labevent_id": 1, "subject_id": 101, "charttime": "2180-05-01 10:00:00", "itemid": 50912, "valuenum": 1.0},
        {"labevent_id": 2, "subject_id": 101, "charttime": "2180-05-03 10:00:00", "itemid": 50912, "valuenum": 1.4},
        {"labevent_id": 3, "subject_id": 102, "charttime": "2180-05-01 12:00:00", "itemid": 50912, "valuenum": 2.0},
    ])

    # Arm 1: 24h LOCF grid
    arm1 = generate_arm1_conventional_locf_grid(df, resample_hours=24)
    assert len(arm1) > 0
    assert "creatinine" in arm1.columns
    assert "grid_charttime" in arm1.columns

    # Arm 2: Flawed controls
    arm2 = generate_arm2_flawed_controls(df)
    assert "control_2a_leakage" in arm2
    assert "control_2b_bleed" in arm2
    assert "control_2c_loss" in arm2
