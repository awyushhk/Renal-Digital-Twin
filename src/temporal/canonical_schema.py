"""Module 1.3: Canonical Renal-Event Schema Definition and Validation.

This module defines the typed canonical schema for renal laboratory events,
enforcing preservation of raw clinical observations, unit harmonization provenance,
audit quality flags, and dual-timestamp reporting delays according to
`docs/m1_3_temporal_representation_protocol.md`.

Source files and earlier audit artifacts are strictly read-only and immutable.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

# Target blood chemistry analytes validated in M1.1 and extracted in M1.2 Phase B
TARGET_ITEM_IDS: Dict[str, int] = {
    "creatinine": 50912,
    "urea_nitrogen": 51006,
    "potassium": 50971,
    "sodium": 50983,
    "bicarbonate": 50882,
    "glucose": 50931,
}

ITEM_TO_ANALYTE: Dict[int, str] = {v: k for k, v in TARGET_ITEM_IDS.items()}
ANALYTE_TO_ITEM: Dict[str, int] = TARGET_ITEM_IDS.copy()

# Canonical target units
CANONICAL_UNITS: Dict[str, str] = {
    "creatinine": "mg/dL",
    "urea_nitrogen": "mg/dL",
    "potassium": "mEq/L",
    "sodium": "mEq/L",
    "bicarbonate": "mEq/L",
    "glucose": "mg/dL",
}

# Permitted quality and audit categorical flags
PERMITTED_UNIT_FLAGS: Set[str] = {"STANDARD", "CONVERTED", "UNKNOWN_UNIT", "MISSING_UOM"}
PERMITTED_DELAY_FLAGS: Set[str] = {"NORMAL_DELAY", "ZERO_DELAY", "NEGATIVE_REPORTING_DELAY"}
PERMITTED_ARBITRATION_ROLES: Set[str] = {
    "PRIMARY_SELECTED",
    "SUPERSEDED_DUPLICATE",
    "SUPERSEDED_CONFLICT",
    "SYNTHESIZED_AGGREGATE",
    "UNARBITRATED_RAW",
    "UNSUPPORTED_POLICY_D_ANALYTE",
}

# Canonical schema columns and nullability definitions
CANONICAL_COLUMNS: List[str] = [
    # Core event identifiers
    "labevent_id",
    "subject_id",
    "hadm_id",
    "itemid",
    "analyte_name",
    # Dual timestamps and reporting delay
    "charttime",
    "storetime",
    "reporting_delay_hours",
    "delay_anomaly_flag",
    # Raw source observations (immutable)
    "raw_value_text",
    "raw_valuenum",
    "raw_valueuom",
    # Harmonized values and units
    "valuenum_harmonized",
    "unit_harmonized",
    "unit_flag",
    # Source clinical flags
    "flag_abnormal",
    "priority",
    # Audit quality flags
    "is_exact_duplicate",
    "is_conflicting_measurement",
    # Arbitration state (experimental)
    "arbitration_policy_id",
    "arbitration_role",
    "valuenum_effective",
    # Patient demographics (from patients.csv)
    "gender",
    "anchor_age",
    "anchor_year",
    "anchor_year_group",
    "dod",
    # Admission context (from admissions.csv, null for outpatient)
    "admittime",
    "dischtime",
    "deathtime",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "race",
    "hospital_expire_flag",
]

NON_NULLABLE_COLUMNS: Set[str] = {
    "labevent_id",
    "subject_id",
    "itemid",
    "analyte_name",
    "charttime",
    "storetime",
    "reporting_delay_hours",
    "delay_anomaly_flag",
    "unit_harmonized",
    "unit_flag",
    "is_exact_duplicate",
    "is_conflicting_measurement",
    "arbitration_policy_id",
    "arbitration_role",
    "gender",
    "anchor_age",
}


class CanonicalSchemaError(Exception):
    """Raised when DataFrame fails canonical renal event schema validation."""
    pass


def validate_canonical_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that a DataFrame conforms to the canonical renal-event schema contract.

    Checks:
    - Presence of all required canonical columns.
    - Non-null constraints on mandatory columns.
    - Datetime types on charttime and storetime.
    - Permitted values for categorical flags.
    - Target analyte restriction.

    Returns:
        (is_valid, list_of_validation_errors)
    """
    errors: List[str] = []

    # 1. Column presence check
    missing_cols = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required canonical columns: {sorted(missing_cols)}")

    if errors:
        return False, errors

    # 2. Non-null constraints check
    for col in NON_NULLABLE_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' violates non-null constraint: {null_count} nulls found.")

    # 3. Datetime types check
    for dt_col in ["charttime", "storetime"]:
        if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            errors.append(f"Column '{dt_col}' must be datetime64 dtype, found: {df[dt_col].dtype}")

    # 4. Item ID and analyte consistency
    invalid_items = set(df["itemid"].unique()) - set(ITEM_TO_ANALYTE.keys())
    if invalid_items:
        errors.append(f"Unsupported item IDs found in canonical frame: {invalid_items}")

    for itemid, analyte in df[["itemid", "analyte_name"]].drop_duplicates().values:
        expected_analyte = ITEM_TO_ANALYTE.get(itemid)
        if analyte != expected_analyte:
            errors.append(
                f"Item ID {itemid} mapped to '{analyte}', expected canonical '{expected_analyte}'."
            )

    # 5. Categorical flag validity
    invalid_unit_flags = set(df["unit_flag"].dropna().unique()) - PERMITTED_UNIT_FLAGS
    if invalid_unit_flags:
        errors.append(f"Invalid unit flags: {invalid_unit_flags}")

    invalid_delay_flags = set(df["delay_anomaly_flag"].dropna().unique()) - PERMITTED_DELAY_FLAGS
    if invalid_delay_flags:
        errors.append(f"Invalid delay flags: {invalid_delay_flags}")

    invalid_arbitration_roles = set(df["arbitration_role"].dropna().unique()) - PERMITTED_ARBITRATION_ROLES
    if invalid_arbitration_roles:
        errors.append(f"Invalid arbitration roles: {invalid_arbitration_roles}")

    # 6. Boolean flag dtypes
    for bool_col in ["is_exact_duplicate", "is_conflicting_measurement"]:
        if not pd.api.types.is_bool_dtype(df[bool_col]):
            errors.append(f"Column '{bool_col}' must be boolean dtype, found: {df[bool_col].dtype}")

    is_valid = len(errors) == 0
    return is_valid, errors


def create_canonical_event_frame(
    df: pd.DataFrame,
    policy_id: str = "POLICY_0_RAW",
    copy: bool = True,
) -> pd.DataFrame:
    """Transform extracted cohort DataFrame into the standardized canonical renal-event schema.

    Parameters:
        df: Input DataFrame (e.g. loaded from `data/processed/cohort_renal_bounded.csv`).
        policy_id: Initial arbitration policy identifier (default: 'POLICY_0_RAW').
        copy: If True, deep copies the input DataFrame before transformation.

    Returns:
        Canonical DataFrame with standardized column names, types, dual timestamps,
        reporting delay, delay anomaly flag, and arbitration metadata.
    """
    if copy:
        df = df.copy()

    # Map input source column aliases if present
    val_text_col = "raw_value_text" if "raw_value_text" in df.columns else "value"
    val_num_col = "raw_valuenum" if "raw_valuenum" in df.columns else "valuenum"
    val_uom_col = "raw_valueuom" if "raw_valueuom" in df.columns else "valueuom"
    flag_col = "flag_abnormal" if "flag_abnormal" in df.columns else "flag"

    # Coerce timestamps to datetime64[ns]
    charttime_dt = pd.to_datetime(df["charttime"])
    storetime_dt = pd.to_datetime(df["storetime"])

    # Compute reporting delay in hours: storetime - charttime
    delays_seconds = (storetime_dt - charttime_dt).dt.total_seconds()
    delays_hours = np.round(delays_seconds / 3600.0, 4)

    # Classify reporting delay anomaly flag
    delay_flags = []
    for d in delays_hours:
        if d > 0.0:
            delay_flags.append("NORMAL_DELAY")
        elif d == 0.0:
            delay_flags.append("ZERO_DELAY")
        else:
            delay_flags.append("NEGATIVE_REPORTING_DELAY")

    # Map analyte names
    analyte_names = df["itemid"].map(ITEM_TO_ANALYTE)
    if analyte_names.isna().any():
        unmapped = df.loc[analyte_names.isna(), "itemid"].unique().tolist()
        raise CanonicalSchemaError(f"Found unsupported item IDs cannot be mapped to analyte names: {unmapped}")

    # Build canonical frame
    canonical = pd.DataFrame()
    canonical["labevent_id"] = df["labevent_id"].astype("int64")
    canonical["subject_id"] = df["subject_id"].astype("int64")
    canonical["hadm_id"] = df["hadm_id"]  # Preserved nullable (float64 or Int64)
    canonical["itemid"] = df["itemid"].astype("int64")
    canonical["analyte_name"] = analyte_names.astype("string")

    canonical["charttime"] = charttime_dt
    canonical["storetime"] = storetime_dt
    canonical["reporting_delay_hours"] = delays_hours
    canonical["delay_anomaly_flag"] = pd.Categorical(delay_flags, categories=list(PERMITTED_DELAY_FLAGS))

    canonical["raw_value_text"] = df[val_text_col].astype("string")
    canonical["raw_valuenum"] = df[val_num_col].astype("float64")
    canonical["raw_valueuom"] = df[val_uom_col].astype("string")

    canonical["valuenum_harmonized"] = df["valuenum_harmonized"].astype("float64")
    canonical["unit_harmonized"] = df["unit_harmonized"].astype("string")
    canonical["unit_flag"] = pd.Categorical(df["unit_flag"], categories=list(PERMITTED_UNIT_FLAGS))

    canonical["flag_abnormal"] = df[flag_col].astype("string") if flag_col in df.columns else None
    canonical["priority"] = df["priority"].astype("string") if "priority" in df.columns else None

    canonical["is_exact_duplicate"] = df["is_exact_duplicate"].astype("bool")
    canonical["is_conflicting_measurement"] = df["is_conflicting_measurement"].astype("bool")

    # Arbitration metadata (initialized to raw/unarbitrated state)
    canonical["arbitration_policy_id"] = str(policy_id)
    canonical["arbitration_role"] = pd.Categorical(
        ["UNARBITRATED_RAW"] * len(df), categories=list(PERMITTED_ARBITRATION_ROLES)
    )
    canonical["valuenum_effective"] = df["valuenum_harmonized"].astype("float64")

    # Demographics
    for demo_col in ["gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"]:
        if demo_col in df.columns:
            canonical[demo_col] = df[demo_col]
        else:
            canonical[demo_col] = None

    # Admission context
    for adm_col in [
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "race",
        "hospital_expire_flag",
    ]:
        if adm_col in df.columns:
            canonical[adm_col] = df[adm_col]
        else:
            canonical[adm_col] = None

    # Sort deterministically by subject_id, charttime, itemid, labevent_id
    canonical.sort_values(
        by=["subject_id", "charttime", "itemid", "labevent_id"],
        ascending=[True, True, True, True],
        inplace=True,
    )
    canonical.reset_index(drop=True, inplace=True)

    # Validate schema
    is_valid, validation_errors = validate_canonical_schema(canonical)
    if not is_valid:
        raise CanonicalSchemaError(f"Canonical schema validation failed:\n" + "\n".join(validation_errors))

    return canonical
