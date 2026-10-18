"""Module 1.4: Validation & Benchmark Suite."""

from .weka_benchmark import (
    BenchmarkResult,
    run_all_benchmarks,
)
from .end_to_end_validation import (
    DOCUMENTED_COHORT_SHA256,
    ValidationConfig,
    compute_file_sha256,
    verify_cohort_reference_integrity,
    verify_arbitration_superseded_counts,
    verify_operational_query_time_lookahead_and_immutability,
    verify_cotemporal_ratio_and_patient_boundary_isolation,
    verify_dynamic_negative_controls_and_retention,
    verify_three_arm_comparative_pipeline,
    execute_full_end_to_end_acceptance,
    generate_end_to_end_validation_report,
)

__all__ = [
    "BenchmarkResult",
    "run_all_benchmarks",
    "DOCUMENTED_COHORT_SHA256",
    "ValidationConfig",
    "compute_file_sha256",
    "verify_cohort_reference_integrity",
    "verify_arbitration_superseded_counts",
    "verify_operational_query_time_lookahead_and_immutability",
    "verify_cotemporal_ratio_and_patient_boundary_isolation",
    "verify_dynamic_negative_controls_and_retention",
    "verify_three_arm_comparative_pipeline",
    "execute_full_end_to_end_acceptance",
    "generate_end_to_end_validation_report",
]
