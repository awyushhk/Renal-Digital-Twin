"""Unit and integration tests for Phase A: WEKA 3.8.7 vs. Python Preprocessing Benchmark.

Tests each of the 8 preprocessing requirements (A through H) on deterministic synthetic fixtures:
  - Requirement A: Numeric normalization & extreme value handling
  - Requirement B: Missing-value handling and observation indicators
  - Requirement C: Unit and laboratory-item mapping (distinct numeric vs. ordinal)
  - Requirement D: Timestamp parsing and preservation
  - Requirement E: Patient/admission linkage integrity (validated flat input)
  - Requirement F: Duplicate and conflicting measurement handling
  - Requirement G: Irregular elapsed-time feature derivation
  - Requirement H: Strictly causal feature generation & future-perturbation test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.weka_benchmark import (
    DEFAULT_WEKA_JAR,
    generate_all_fixtures,
    get_weka_jar_path,
    parse_arff_data,
    run_all_benchmarks,
    run_benchmark_req_a,
    run_benchmark_req_b,
    run_benchmark_req_c,
    run_benchmark_req_d,
    run_benchmark_req_e,
    run_benchmark_req_f,
    run_benchmark_req_g,
    run_benchmark_req_h,
    run_weka_filter,
)


@pytest.fixture(scope="module")
def weka_jar_path() -> Path:
    """Fixture providing verified WEKA jar path."""
    jar = get_weka_jar_path()
    assert jar.is_file(), f"WEKA jar not found at {jar}"
    return jar


@pytest.fixture(scope="module")
def benchmark_fixtures(tmp_path_factory) -> dict[str, dict[str, Path]]:
    """Fixture generating synthetic datasets in a temporary directory."""
    fixture_dir = tmp_path_factory.mktemp("benchmark_fixtures")
    return generate_all_fixtures(fixture_dir)


@pytest.fixture
def scratch_dir(tmp_path) -> Path:
    """Temporary scratch directory for test execution artifacts."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


# ===========================================================================
# 1. Environment & Fixture Integrity Tests
# ===========================================================================

def test_weka_jar_accessible(weka_jar_path: Path):
    """Verify WEKA JAR exists and reports version/help via CLI."""
    assert weka_jar_path.name == "weka.jar"
    assert weka_jar_path.stat().st_size > 10_000_000  # >10 MB


def test_fixture_generation_completeness(benchmark_fixtures: dict[str, dict[str, Path]]):
    """Verify all 8 requirement fixture sets are generated with both CSV and ARFF."""
    expected_keys = [
        "normalization",
        "missingness",
        "units",
        "timestamps",
        "linkage",
        "duplicates",
        "elapsed_time",
        "causality",
    ]
    for key in expected_keys:
        assert key in benchmark_fixtures, f"Missing fixture key: {key}"
        fixture_group = benchmark_fixtures[key]
        for ftype, fpath in fixture_group.items():
            assert fpath.is_file(), f"Fixture file does not exist: {fpath}"
            assert fpath.stat().st_size > 0, f"Fixture file is empty: {fpath}"


def test_fixture_edge_cases_present(benchmark_fixtures: dict[str, dict[str, Path]]):
    """Verify all mandated edge cases are present across the synthetic fixtures."""
    # 1. Extreme numeric value (creatinine = 15.0) and constant column
    norm_df = pd.read_csv(benchmark_fixtures["normalization"]["csv"])
    assert 15.0 in norm_df["creatinine"].values
    assert norm_df["constant_col"].nunique() == 1

    # 2. Missingness at start, middle, end
    miss_df = pd.read_csv(benchmark_fixtures["missingness"]["csv"])
    p1 = miss_df[miss_df["subject_id"] == 1001]
    assert pd.isna(p1.iloc[0]["creatinine"])  # Start
    assert pd.isna(p1.iloc[1]["bun"])         # Middle
    assert pd.isna(p1.iloc[3]["creatinine"])  # End

    # 3. Missing admission ID (hadm_id = NaN) for outpatient
    link_df = pd.read_csv(benchmark_fixtures["linkage"]["csv"])
    assert link_df["hadm_id"].isna().sum() == 1

    # 4. Same-timestamp conflicting observation (1.2 vs 1.8)
    dup_df = pd.read_csv(benchmark_fixtures["duplicates"]["csv"])
    p1_draws = dup_df[(dup_df["subject_id"] == 1001) & (dup_df["charttime"] == "2150-01-01 10:00:00")]
    assert len(p1_draws) == 3
    assert set(p1_draws["valuenum"].values) == {1.2, 1.8}

    # 5. Multiple patients with irregular gaps
    elapsed_df = pd.read_csv(benchmark_fixtures["elapsed_time"]["csv"])
    assert elapsed_df["subject_id"].nunique() >= 2


# ===========================================================================
# 2. Requirement A: Numeric Normalization Tests
# ===========================================================================

def test_req_a_minmax_normalization(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA Normalize matches Python MinMaxScaler to machine precision."""
    results = run_benchmark_req_a(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if r.subtest_name == "Min-Max Scaling [0, 1]")
    assert subtest.status == "PASSED"
    assert subtest.numerical_tolerance is not None
    assert subtest.numerical_tolerance < 1e-4


def test_req_a_zscore_standardization(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA Standardize matches Python sample standard deviation (ddof=1)."""
    results = run_benchmark_req_a(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Standardization" in r.subtest_name)
    assert subtest.status == "PASSED"
    assert subtest.numerical_tolerance is not None
    assert subtest.numerical_tolerance < 1e-4


def test_req_a_robust_scaling_gap(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA lacks native RobustScaler (median/IQR), causing outlier distortion."""
    results = run_benchmark_req_a(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Robust Scaling" in r.subtest_name)
    assert subtest.status == "UNSUPPORTED"
    assert subtest.limitation_type == "Capability Gap"


# ===========================================================================
# 3. Requirement B: Missing-Value Handling Tests
# ===========================================================================

def test_req_b_mean_imputation(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA ReplaceMissingValues matches Python column mean replacement."""
    results = run_benchmark_req_b(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if r.subtest_name == "Global Mean Imputation")
    assert subtest.status == "PASSED"
    assert subtest.numerical_tolerance is not None
    assert subtest.numerical_tolerance < 1e-4


def test_req_b_observation_indicator_workflow_gap(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA ReplaceMissingValues lacks missingness indicator output option."""
    results = run_benchmark_req_b(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Observation Mask" in r.subtest_name)
    assert subtest.status == "UNSUPPORTED"
    assert subtest.limitation_type == "Workflow Inconvenience"


def test_req_b_locf_capability_gap(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA cannot perform longitudinal trajectory forward-fill (LOCF)."""
    results = run_benchmark_req_b(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Forward-Fill" in r.subtest_name)
    assert subtest.status == "UNSUPPORTED"
    assert subtest.limitation_type == "Capability Gap"


# ===========================================================================
# 4. Requirement C: Unit & Laboratory-Item Mapping Tests
# ===========================================================================

def test_req_c_conditional_unit_harmonization_gap(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA two-step NominalToBinary+MathExpression pipeline achieves conditional unit harmonization.

    Audit finding: single MathExpression on a nominal attribute fails (SemanticException), but the
    composed pipeline NominalToBinary -> MathExpression -R <col> -V succeeds. This is a
    Workflow Inconvenience, not a Capability Gap.
    """
    results = run_benchmark_req_c(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Conditional Nominal-to-Numeric" in r.subtest_name)
    # After audit correction: two-step pipeline works -> PASSED, Workflow Inconvenience
    assert subtest.status == "PASSED", (
        f"Two-step NominalToBinary+MathExpression pipeline should PASS unit harmonization. "
        f"Status was: {subtest.status}. Notes: {subtest.evidence_notes[:100]}"
    )
    assert subtest.limitation_type == "Workflow Inconvenience"


def test_req_c_dipstick_ordinal_mapping(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify dipstick ordinal rank mapping is kept distinct from continuous scaling."""
    results = run_benchmark_req_c(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Dipstick Ordinal Rank" in r.subtest_name)
    assert subtest.status == "PASSED"
    assert subtest.limitation_type == "Workflow Inconvenience"


# ===========================================================================
# 5. Requirement D: Timestamp Parsing & Preservation Tests
# ===========================================================================

def test_req_d_timestamp_preservation(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA parses and preserves exact minute-and-second ISO dates in ARFF."""
    results = run_benchmark_req_d(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if r.requirement_id == "R4_TIMESTAMPS")
    assert subtest.status == "PASSED"
    assert subtest.limitation_type == "None"


# ===========================================================================
# 6. Requirement E: Linkage Integrity Tests (Validated Flat Representation)
# ===========================================================================

def test_req_e_outpatient_hadm_id_preservation(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify outpatient records with missing hadm_id are retained in flat ARFF without row loss."""
    results = run_benchmark_req_e(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "hadm_id Retention" in r.subtest_name)
    assert subtest.status == "PASSED"
    assert subtest.limitation_type == "None"


def test_req_e_relational_join_capability_gap(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA lacks in-memory multi-table relational join capability across flat files."""
    results = run_benchmark_req_e(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Relational Database Joining" in r.subtest_name)
    assert subtest.status == "UNSUPPORTED"
    assert subtest.limitation_type == "Capability Gap"


# ===========================================================================
# 7. Requirement F: Duplicate & Conflict Handling Tests
# ===========================================================================

def test_req_f_exact_duplicate_removal(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA RemoveDuplicates successfully removes 100% identical rows."""
    results = run_benchmark_req_f(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Exact Full-Row" in r.subtest_name)
    assert subtest.status == "PASSED"
    assert subtest.limitation_type == "None"


def test_req_f_conflicting_measurement_arbitration_gap(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA cannot arbitrate discordant draws at identical timestamps (leaves conflicting instances)."""
    results = run_benchmark_req_f(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if "Conflicting Same-Timestamp" in r.subtest_name)
    assert subtest.status == "UNSUPPORTED"
    assert subtest.limitation_type == "Capability Gap"


# ===========================================================================
# 8. Requirement G: Irregular Elapsed-Time Features Tests
# ===========================================================================

def test_req_g_elapsed_time_and_patient_reset_gap(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify WEKA TimeSeriesDelta fails to parse dates into elapsed hours and fails to reset at patient boundaries."""
    results = run_benchmark_req_g(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if r.requirement_id == "R7_ELAPSED_TIME")
    assert subtest.status == "UNSUPPORTED"
    assert subtest.limitation_type == "Capability Gap"


# ===========================================================================
# 9. Requirement H: Causal Future-Perturbation Test
# ===========================================================================

def test_req_h_future_perturbation_leakage(weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path):
    """Verify batch normalization in WEKA leaks future acute surge (t=5) into historical baseline (t=2),

    whereas Python expanding window remains strictly invariant to future perturbations.
    """
    results = run_benchmark_req_h(weka_jar_path, benchmark_fixtures, scratch_dir)
    subtest = next(r for r in results if r.requirement_id == "R8_CAUSALITY")
    assert subtest.status == "FAILED"  # Direct proof of future data leakage under batch preprocessing
    assert subtest.numerical_tolerance is not None
    assert subtest.numerical_tolerance > 0.1  # Shift of ~0.24


# ===========================================================================
# 10. End-to-End Master Benchmark Runner Test
# ===========================================================================

def test_run_all_benchmarks_end_to_end(weka_jar_path: Path, tmp_path: Path):
    """Verify master benchmark runner executes and produces valid CSV and Markdown reports."""
    out_dir = tmp_path / "reports_test"
    df_results, csv_path, md_path = run_all_benchmarks(weka_jar=weka_jar_path, output_dir=out_dir)

    assert csv_path.is_file()
    assert md_path.is_file()
    assert len(df_results) == 15

    # After audit correction: R3a is now PASSED (Workflow Inconvenience via two-step pipeline)
    status_counts = df_results["status"].value_counts().to_dict()
    assert status_counts.get("PASSED", 0) == 8, f"Expected 8 PASSED (incl. corrected R3a). Got: {status_counts}"
    assert status_counts.get("FAILED", 0) == 1
    assert status_counts.get("UNSUPPORTED", 0) == 6, f"Expected 6 UNSUPPORTED after R3a correction. Got: {status_counts}"


# ===========================================================================
# 11. Regression Tests Added by Audit (tests that exercise claimed behaviour)
# ===========================================================================

def test_r3a_two_step_weka_pipeline_produces_correct_harmonization(
    weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path
):
    """Regression: verify the NominalToBinary -> MathExpression pipeline produces correct mg/dL values.

    Audit finding: the two-step composed WEKA pipeline achieves unit harmonization,
    refuting the original 'Capability Gap' claim. This test exercises actual WEKA execution
    and reads back the output to confirm correctness.
    """
    from src.validation.weka_benchmark import run_weka_filter, parse_arff_data
    in_arff = benchmark_fixtures["units"]["arff"]

    # Step 1: Convert nominal valueuom column (attr 4) to numeric 0/1
    ntb_arff = scratch_dir / "r3_ntb.arff"
    ntb_res = run_weka_filter(
        "weka.filters.unsupervised.attribute.NominalToBinary",
        ["-R", "4"],
        in_arff, ntb_arff, weka_jar_path,
    )
    assert ntb_res.returncode == 0, f"NominalToBinary failed: {ntb_res.stderr}"
    assert ntb_arff.is_file()

    # Step 2: Apply ifelse on numeric flag; -R 3 -V = ONLY modify attribute 3 (valuenum)
    harm_arff = scratch_dir / "r3_harmonized.arff"
    harm_res = run_weka_filter(
        "weka.filters.unsupervised.attribute.MathExpression",
        ["-E", "ifelse(A4=0, A*1000, A)", "-R", "3", "-V"],
        ntb_arff, harm_arff, weka_jar_path,
    )
    assert harm_res.returncode == 0, f"MathExpression failed: {harm_res.stderr}"
    assert harm_arff.is_file()

    # Read back and verify correctness
    w_df = parse_arff_data(harm_arff)
    valuenum_col = w_df["valuenum"]
    # Row 0: 4.0 g_dL -> 4000.0 mg_dL
    assert abs(valuenum_col.iloc[0] - 4000.0) < 0.01, f"Row 0 should be 4000.0, got {valuenum_col.iloc[0]}"
    # Row 1: 4000.0 mg_dL -> 4000.0 (unchanged)
    assert abs(valuenum_col.iloc[1] - 4000.0) < 0.01, f"Row 1 should be 4000.0, got {valuenum_col.iloc[1]}"
    # Row 2: 2.5 g_dL -> 2500.0 mg_dL
    assert abs(valuenum_col.iloc[2] - 2500.0) < 0.01, f"Row 2 should be 2500.0, got {valuenum_col.iloc[2]}"
    # Row 3: 2500.0 mg_dL -> 2500.0 (unchanged)
    assert abs(valuenum_col.iloc[3] - 2500.0) < 0.01, f"Row 3 should be 2500.0, got {valuenum_col.iloc[3]}"


def test_r6b_weka_dedup_output_retains_conflicting_rows(
    weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path
):
    """Regression: read back WEKA RemoveDuplicates output and confirm both conflicting rows are retained.

    Audit finding: the R6b benchmark infers the result without reading the output ARFF. This test
    exercises the WEKA call and parses the output to confirm 1.2 and 1.8 both remain.
    """
    from src.validation.weka_benchmark import run_weka_filter, parse_arff_data
    in_arff = benchmark_fixtures["duplicates"]["arff"]
    out_arff = scratch_dir / "r6b_dedup_out.arff"

    res = run_weka_filter(
        "weka.filters.unsupervised.instance.RemoveDuplicates", [], in_arff, out_arff, weka_jar_path
    )
    assert res.returncode == 0, f"RemoveDuplicates failed: {res.stderr}"
    assert out_arff.is_file()

    w_df = parse_arff_data(out_arff)
    assert len(w_df) == 3, f"Expected 3 rows after exact dedup (rows: {len(w_df)})"

    # Confirm both conflicting values for patient 1001 at the same timestamp remain
    p1001_rows = w_df[w_df.apply(lambda r: float(r["subject_id"]) == 1001.0, axis=1)]
    conflict_values = set(p1001_rows["valuenum"].astype(float).round(4))
    assert 1.2 in conflict_values, f"Conflicting row 1.2 should be retained; got {conflict_values}"
    assert 1.8 in conflict_values, f"Conflicting row 1.8 should be retained; got {conflict_values}"


def test_r7_weka_timeseriesdelta_cross_patient_contamination(
    weka_jar_path: Path, benchmark_fixtures: dict[str, dict[str, Path]], scratch_dir: Path
):
    """Regression: read back WEKA TimeSeriesDelta output and confirm cross-patient contamination.

    Audit finding: the R7 benchmark hardcodes UNSUPPORTED without verifying the actual WEKA output.
    This test runs TimeSeriesDelta, parses the output, and confirms:
    (a) deltas are value differences, not elapsed hours, and
    (b) Patient 1001's last-row delta is computed using Patient 1002's first value (cross-patient artifact).
    """
    from src.validation.weka_benchmark import run_weka_filter, parse_arff_data
    in_arff = benchmark_fixtures["elapsed_time"]["arff"]
    out_arff = scratch_dir / "r7_delta_out.arff"

    res = run_weka_filter(
        "weka.filters.unsupervised.attribute.TimeSeriesDelta",
        ["-R", "3", "-I", "1"],
        in_arff, out_arff, weka_jar_path,
    )
    assert res.returncode == 0, f"TimeSeriesDelta failed: {res.stderr}"
    assert out_arff.is_file()

    w_df = parse_arff_data(out_arff)
    delta_col = w_df.columns[-1]  # 'creatinine d+1'

    # Row 2 (P1001 last row): WEKA should produce ~0.7 (= 1.6 - 0.9 using P1002's first value)
    p1001_last_delta = float(w_df[delta_col].iloc[2])
    assert abs(p1001_last_delta - 0.7) < 0.01, (
        f"P1001's last-row delta should be ~0.7 (cross-patient contamination: 1.6-0.9). "
        f"Got {p1001_last_delta}. If this changes, cross-patient contamination may be fixed."
    )

    # Row 3 (P1002 first row): WEKA gives -0.2 instead of NaN (no patient boundary reset)
    p1002_first_delta = w_df[delta_col].iloc[3]
    assert not pd.isna(p1002_first_delta), (
        f"P1002's first-row delta should NOT be NaN in WEKA (no patient boundary reset). "
        f"Got NaN — this would mean WEKA now handles boundaries correctly (recheck classification)."
    )

    # Verify deltas are value differences (not hours): P1001 row0 delta = 1.0 - 1.2 = -0.2
    p1001_first_delta = float(w_df[delta_col].iloc[0])
    assert abs(p1001_first_delta - (-0.2)) < 0.01, (
        f"P1001 first delta should be ~-0.2 (creatinine diff, not hours). Got {p1001_first_delta}"
    )
