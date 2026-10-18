"""Module 1.3 Step 2: Conflict and Duplicate Arbitration Engine.

This module implements deterministic, auditable arbitration policies for duplicate
and conflicting laboratory measurements in the canonical renal-event schema according to
`docs/m1_3_temporal_representation_protocol.md`.

Policies:
- POLICY_0_RAW: Preserves all records without deduplication or conflict resolution.
- POLICY_A_STORETIME: Operational storetime precedence; selects latest storetime for
  conflicts, earlier storetime for exact duplicates, with deterministic tie-breaking.
- POLICY_B_STAT: Clinical priority precedence; prefers STAT over ROUTINE for conflicts
  when priorities are unambiguous; falls back to Policy A when tied or missing.
- POLICY_C_MEAN: Analytical mean aggregation for numeric conflicts; creates a synthesized
  aggregate record linked to all source records, marking constituents as superseded.
- POLICY_D_EXTREMUM: Worst-case physiological risk extremum for creatinine, BUN,
  bicarbonate, and potassium only. Explicitly rejects or flags sodium and glucose conflicts.

All raw source values, timestamps, and audit flags are strictly immutable.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.temporal.canonical_schema import (
    CANONICAL_COLUMNS,
    PERMITTED_ARBITRATION_ROLES,
    TARGET_ITEM_IDS,
    ITEM_TO_ANALYTE,
    validate_canonical_schema,
)

# Supported arbitration policy identifiers
POLICY_0_RAW: str = "POLICY_0_RAW"
POLICY_A_STORETIME: str = "POLICY_A_STORETIME"
POLICY_B_STAT: str = "POLICY_B_STAT"
POLICY_C_MEAN: str = "POLICY_C_MEAN"
POLICY_D_EXTREMUM: str = "POLICY_D_EXTREMUM"

SUPPORTED_POLICIES: Set[str] = {
    POLICY_0_RAW,
    POLICY_A_STORETIME,
    POLICY_B_STAT,
    POLICY_C_MEAN,
    POLICY_D_EXTREMUM,
}

# Policy D analyte coverage constraints
SUPPORTED_POLICY_D_ANALYTES: Set[str] = {"creatinine", "urea_nitrogen", "bicarbonate", "potassium"}
UNSUPPORTED_POLICY_D_ANALYTES: Set[str] = {"sodium", "glucose"}


class ArbitrationError(Exception):
    """Base exception for arbitration failures."""
    pass


class UnsupportedPolicyAnalyteError(ArbitrationError):
    """Raised when Policy D is invoked on an unsupported analyte conflict (e.g. sodium or glucose)."""
    pass


@dataclass
class ArbitrationSummary:
    """Structured audit metrics produced during conflict arbitration."""
    policy_id: str
    total_rows_input: int
    total_rows_output: int
    retained_rows_count: int
    superseded_duplicate_count: int
    superseded_conflict_count: int
    synthesized_aggregate_count: int
    primary_selected_count: int
    unarbitrated_raw_count: int
    exact_duplicate_groups_count: int
    conflict_groups_count: int
    groups_with_mixed_hadm_id: int
    mixed_hadm_details: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_analyte_conflicts_count: int = 0
    unsupported_analyte_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary dataclass to dictionary."""
        return {
            "policy_id": self.policy_id,
            "total_rows_input": self.total_rows_input,
            "total_rows_output": self.total_rows_output,
            "retained_rows_count": self.retained_rows_count,
            "superseded_duplicate_count": self.superseded_duplicate_count,
            "superseded_conflict_count": self.superseded_conflict_count,
            "synthesized_aggregate_count": self.synthesized_aggregate_count,
            "primary_selected_count": self.primary_selected_count,
            "unarbitrated_raw_count": self.unarbitrated_raw_count,
            "exact_duplicate_groups_count": self.exact_duplicate_groups_count,
            "conflict_groups_count": self.conflict_groups_count,
            "groups_with_mixed_hadm_id": self.groups_with_mixed_hadm_id,
            "mixed_hadm_details": self.mixed_hadm_details,
            "unsupported_analyte_conflicts_count": self.unsupported_analyte_conflicts_count,
            "unsupported_analyte_details": self.unsupported_analyte_details,
        }


def _get_priority_rank(priority: Any) -> int:
    """Map ordering priority to an integer rank.
    
    STAT -> 2 (highest urgency)
    ROUTINE -> 1 (standard)
    Missing / Null / Unknown -> 0 (unspecified)
    """
    if pd.isna(priority):
        return 0
    p_str = str(priority).strip().upper()
    if p_str == "STAT":
        return 2
    elif p_str == "ROUTINE":
        return 1
    return 0


def _generate_synthetic_labevent_id(subject_id: int, itemid: int, charttime_str: str) -> int:
    """Generate a deterministic negative int64 ID for synthesized aggregate records."""
    key = f"{subject_id}_{itemid}_{charttime_str}"
    hash_hex = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Mask to positive 48-bit int, offset by 1 to guarantee strictly negative ID
    return -(int(hash_hex[:12], 16) + 1)


def arbitrate_canonical_frame(
    df: pd.DataFrame,
    policy_id: str = POLICY_A_STORETIME,
    raise_on_unsupported: bool = True,
    copy: bool = True,
) -> Tuple[pd.DataFrame, ArbitrationSummary]:
    """Apply an experimental arbitration policy to a canonical renal-event DataFrame.

    Parameters:
        df: Input canonical DataFrame conforming to validate_canonical_schema.
        policy_id: One of POLICY_0_RAW, POLICY_A_STORETIME, POLICY_B_STAT,
                   POLICY_C_MEAN, or POLICY_D_EXTREMUM.
        raise_on_unsupported: If True, raises UnsupportedPolicyAnalyteError when Policy D
                              encounters sodium or glucose conflicts. If False, flags them.
        copy: If True, deep copies the input DataFrame to guarantee immutability.

    Returns:
        (arbitrated_df, summary)
    """
    if policy_id not in SUPPORTED_POLICIES:
        raise ArbitrationError(
            f"Unsupported policy_id '{policy_id}'. Must be one of: {sorted(SUPPORTED_POLICIES)}"
        )

    # 1. Deep copy to protect input immutability
    if copy:
        work_df = df.copy(deep=True)
    else:
        work_df = df

    # Validate canonical schema contract on input
    is_valid, validation_errors = validate_canonical_schema(work_df)
    if not is_valid:
        raise ArbitrationError(
            f"Input DataFrame fails canonical schema validation:\n" + "\n".join(validation_errors)
        )

    n_input = len(work_df)
    if n_input == 0:
        empty_summary = ArbitrationSummary(
            policy_id=policy_id,
            total_rows_input=0,
            total_rows_output=0,
            retained_rows_count=0,
            superseded_duplicate_count=0,
            superseded_conflict_count=0,
            synthesized_aggregate_count=0,
            primary_selected_count=0,
            unarbitrated_raw_count=0,
            exact_duplicate_groups_count=0,
            conflict_groups_count=0,
            groups_with_mixed_hadm_id=0,
        )
        return work_df, empty_summary

    # Ensure source_labevent_ids column exists for provenance tracking
    if "source_labevent_ids" not in work_df.columns:
        work_df["source_labevent_ids"] = work_df["labevent_id"].astype(str)

    # Coerce timestamps for consistent chronological operations
    work_df["charttime"] = pd.to_datetime(work_df["charttime"])
    work_df["storetime"] = pd.to_datetime(work_df["storetime"])

    # Ensure duplicate and conflict audit flags reflect the data
    # (If passed a synthetic test frame with all False, auto-compute flags)
    has_dups = work_df["is_exact_duplicate"].any()
    has_conflicts = work_df["is_conflicting_measurement"].any()
    if not has_dups and not has_conflicts:
        key_cols = ["subject_id", "itemid", "charttime"]
        any_co_temporal = work_df.duplicated(subset=key_cols, keep=False).any()
        if any_co_temporal:
            exact_cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum_harmonized", "unit_harmonized"]
            exact_mask = work_df.duplicated(subset=exact_cols, keep=False)
            work_df["is_exact_duplicate"] = exact_mask

            key_dups = work_df[work_df.duplicated(subset=key_cols, keep=False)]
            if not key_dups.empty:
                conflicting_keys = (
                    key_dups.groupby(key_cols)["valuenum_harmonized"]
                    .nunique()
                    .loc[lambda x: x > 1]
                    .index
                )
                conflicting_set = set(conflicting_keys)
                current_keys = list(zip(work_df["subject_id"], work_df["itemid"], work_df["charttime"]))
                work_df["is_conflicting_measurement"] = [k in conflicting_set for k in current_keys]

    # Initialize arbitration metadata
    work_df["arbitration_policy_id"] = policy_id

    # -------------------------------------------------------------
    # POLICY 0: RAW BASELINE (NO ARBITRATION)
    # -------------------------------------------------------------
    if policy_id == POLICY_0_RAW:
        work_df["arbitration_role"] = pd.Categorical(
            ["UNARBITRATED_RAW"] * n_input,
            categories=list(PERMITTED_ARBITRATION_ROLES),
        )
        work_df["valuenum_effective"] = work_df["valuenum_harmonized"]
        work_df["source_labevent_ids"] = work_df["labevent_id"].astype(str)

        # Sort deterministically
        work_df.sort_values(
            by=["subject_id", "charttime", "itemid", "labevent_id"],
            ascending=[True, True, True, True],
            inplace=True,
        )
        work_df.reset_index(drop=True, inplace=True)

        summary = ArbitrationSummary(
            policy_id=POLICY_0_RAW,
            total_rows_input=n_input,
            total_rows_output=n_input,
            retained_rows_count=n_input,
            superseded_duplicate_count=0,
            superseded_conflict_count=0,
            synthesized_aggregate_count=0,
            primary_selected_count=0,
            unarbitrated_raw_count=n_input,
            exact_duplicate_groups_count=int(work_df["is_exact_duplicate"].sum() // 2) if work_df["is_exact_duplicate"].any() else 0,
            conflict_groups_count=int(work_df["is_conflicting_measurement"].sum() // 2) if work_df["is_conflicting_measurement"].any() else 0,
            groups_with_mixed_hadm_id=0,
        )
        return work_df, summary

    # -------------------------------------------------------------
    # POLICIES A, B, C, D: ACTIVE ARBITRATION PIPELINES
    # -------------------------------------------------------------
    # Default baseline for unique, non-colliding records: PRIMARY_SELECTED
    roles = pd.Series("PRIMARY_SELECTED", index=work_df.index, dtype=object)
    valuenum_eff = work_df["valuenum_harmonized"].copy()
    source_ids = work_df["labevent_id"].astype(str).copy()

    # 1. Audit and report mixed hadm_id within conflict groups
    conflict_key_cols = ["subject_id", "itemid", "charttime"]
    mixed_hadm_details: List[Dict[str, Any]] = []
    conflict_rows = work_df[work_df["is_conflicting_measurement"]]

    if not conflict_rows.empty:
        for (sid, itemid, ctime), grp in conflict_rows.groupby(conflict_key_cols, dropna=False):
            unique_hadms = grp["hadm_id"].dropna().unique().tolist()
            has_null = grp["hadm_id"].isna().any()
            distinct_hadm_count = len(unique_hadms) + (1 if has_null and len(unique_hadms) > 0 else 0)
            if distinct_hadm_count > 1:
                mixed_hadm_details.append({
                    "subject_id": int(sid),
                    "itemid": int(itemid),
                    "charttime": str(ctime),
                    "distinct_hadm_ids": unique_hadms + ([None] if has_null else []),
                    "group_size": len(grp),
                })

    # 2. Arbitrate Exact Duplicates
    # Definition: identical (subject_id, hadm_id, itemid, charttime, valuenum_harmonized, unit_harmonized)
    exact_dup_cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum_harmonized", "unit_harmonized"]
    dup_rows = work_df[work_df["is_exact_duplicate"]]
    exact_dup_groups_count = 0
    superseded_dup_count = 0

    if not dup_rows.empty:
        for _, dup_grp in dup_rows.groupby(exact_dup_cols, dropna=False):
            exact_dup_groups_count += 1
            # Primary selection rule: earlier storetime preferred; tie-break by lowest labevent_id
            sorted_indices = dup_grp.sort_values(
                by=["storetime", "labevent_id"],
                ascending=[True, True],
            ).index.tolist()

            primary_idx = sorted_indices[0]
            superseded_indices = sorted_indices[1:]

            roles.loc[primary_idx] = "PRIMARY_SELECTED"
            for sup_idx in superseded_indices:
                roles.loc[sup_idx] = "SUPERSEDED_DUPLICATE"
                valuenum_eff.loc[sup_idx] = np.nan
                superseded_dup_count += 1

    # 3. Arbitrate Conflicting Measurements
    conflict_groups_count = 0
    superseded_conflict_count = 0
    synthesized_rows: List[Dict[str, Any]] = []
    unsupported_analyte_details: List[Dict[str, Any]] = []

    if not conflict_rows.empty:
        for (sid, itemid, ctime), grp in conflict_rows.groupby(conflict_key_cols, dropna=False):
            conflict_groups_count += 1
            analyte_name = grp["analyte_name"].iloc[0]

            # ---------------------------------------------------------
            # POLICY A: LATEST STORETIME
            # ---------------------------------------------------------
            if policy_id == POLICY_A_STORETIME:
                # Latest storetime wins; deterministic tie-break by lowest labevent_id
                sorted_grp = grp.sort_values(
                    by=["storetime", "labevent_id"],
                    ascending=[False, True],
                )
                winner_idx = sorted_grp.index[0]
                loser_indices = sorted_grp.index[1:]

                roles.loc[winner_idx] = "PRIMARY_SELECTED"
                for l_idx in loser_indices:
                    roles.loc[l_idx] = "SUPERSEDED_CONFLICT"
                    valuenum_eff.loc[l_idx] = np.nan
                    superseded_conflict_count += 1

            # ---------------------------------------------------------
            # POLICY B: CLINICAL PRIORITY (STAT PREFERENCE)
            # ---------------------------------------------------------
            elif policy_id == POLICY_B_STAT:
                # Priority rank mapping: STAT=2, ROUTINE=1, Unknown/Missing=0
                ranks = grp["priority"].apply(_get_priority_rank)
                max_rank = ranks.max()

                # Unambiguous STAT: exactly one record with STAT rank 2
                stat_candidates = grp[ranks == 2]
                if len(stat_candidates) == 1:
                    winner_idx = stat_candidates.index[0]
                elif len(stat_candidates) > 1:
                    # Multiple STATs: ambiguous -> fallback to Policy A among STAT candidates
                    winner_idx = stat_candidates.sort_values(
                        by=["storetime", "labevent_id"],
                        ascending=[False, True],
                    ).index[0]
                else:
                    # No STATs (both ROUTINE, both missing, or ROUTINE vs missing):
                    # Priority does not resolve conflict -> fallback to Policy A across all candidates
                    winner_idx = grp.sort_values(
                        by=["storetime", "labevent_id"],
                        ascending=[False, True],
                    ).index[0]

                roles.loc[winner_idx] = "PRIMARY_SELECTED"
                for l_idx in grp.index:
                    if l_idx != winner_idx:
                        roles.loc[l_idx] = "SUPERSEDED_CONFLICT"
                        valuenum_eff.loc[l_idx] = np.nan
                        superseded_conflict_count += 1

            # ---------------------------------------------------------
            # POLICY C: ANALYTICAL MEAN SYNTHESIS
            # ---------------------------------------------------------
            elif policy_id == POLICY_C_MEAN:
                numeric_candidates = grp[grp["valuenum_harmonized"].notna()]

                # Mark all original source records in the group as superseded conflicts
                for l_idx in grp.index:
                    roles.loc[l_idx] = "SUPERSEDED_CONFLICT"
                    valuenum_eff.loc[l_idx] = np.nan
                    superseded_conflict_count += 1

                if not numeric_candidates.empty:
                    mean_val = float(np.round(numeric_candidates["valuenum_harmonized"].mean(), 4))
                    constituent_ids = sorted(grp["labevent_id"].tolist())
                    rep_row = grp.sort_values(by=["storetime", "labevent_id"], ascending=[False, True]).iloc[0]

                    # Synthesize aggregate event
                    latest_storetime = grp["storetime"].max()
                    charttime_val = pd.to_datetime(ctime)
                    delay_hours = np.round((latest_storetime - charttime_val).total_seconds() / 3600.0, 4)
                    if delay_hours > 0.0:
                        delay_flag = "NORMAL_DELAY"
                    elif delay_hours == 0.0:
                        delay_flag = "ZERO_DELAY"
                    else:
                        delay_flag = "NEGATIVE_REPORTING_DELAY"

                    synthetic_id = _generate_synthetic_labevent_id(int(sid), int(itemid), str(ctime))
                    
                    # Retain hadm_id: use first non-null hadm_id if available
                    hadm_val = grp["hadm_id"].dropna().iloc[0] if not grp["hadm_id"].dropna().empty else np.nan

                    synth_dict: Dict[str, Any] = {
                        "labevent_id": synthetic_id,
                        "subject_id": int(sid),
                        "hadm_id": hadm_val,
                        "itemid": int(itemid),
                        "analyte_name": analyte_name,
                        "charttime": charttime_val,
                        "storetime": latest_storetime,
                        "reporting_delay_hours": delay_hours,
                        "delay_anomaly_flag": delay_flag,
                        "raw_value_text": f"SYNTHESIZED_MEAN({','.join(str(v) for v in sorted(numeric_candidates['valuenum_harmonized']))})",
                        "raw_valuenum": np.nan,
                        "raw_valueuom": rep_row["unit_harmonized"],
                        "valuenum_harmonized": mean_val,
                        "unit_harmonized": rep_row["unit_harmonized"],
                        "unit_flag": "STANDARD",
                        "flag_abnormal": None,
                        "priority": None,
                        "is_exact_duplicate": False,
                        "is_conflicting_measurement": True,
                        "arbitration_policy_id": POLICY_C_MEAN,
                        "arbitration_role": "SYNTHESIZED_AGGREGATE",
                        "valuenum_effective": mean_val,
                        "gender": rep_row.get("gender"),
                        "anchor_age": rep_row.get("anchor_age"),
                        "anchor_year": rep_row.get("anchor_year"),
                        "anchor_year_group": rep_row.get("anchor_year_group"),
                        "dod": rep_row.get("dod"),
                        "admittime": rep_row.get("admittime"),
                        "dischtime": rep_row.get("dischtime"),
                        "deathtime": rep_row.get("deathtime"),
                        "admission_type": rep_row.get("admission_type"),
                        "admission_location": rep_row.get("admission_location"),
                        "discharge_location": rep_row.get("discharge_location"),
                        "insurance": rep_row.get("insurance"),
                        "race": rep_row.get("race"),
                        "hospital_expire_flag": rep_row.get("hospital_expire_flag"),
                        "source_labevent_ids": ",".join(str(i) for i in constituent_ids),
                    }
                    synthesized_rows.append(synth_dict)

            # ---------------------------------------------------------
            # POLICY D: WORST-CASE PHYSIOLOGICAL EXTREMUM
            # ---------------------------------------------------------
            elif policy_id == POLICY_D_EXTREMUM:
                if analyte_name in UNSUPPORTED_POLICY_D_ANALYTES:
                    err_msg = (
                        f"Analyte '{analyte_name}' (itemid={itemid}) has conflicting measurements at "
                        f"charttime '{ctime}' for subject_id {sid}, but is explicitly unsupported under Policy D. "
                        f"Protocol restricts Policy D to creatinine, urea_nitrogen, bicarbonate, and potassium."
                    )
                    unsupported_analyte_details.append({
                        "subject_id": int(sid),
                        "itemid": int(itemid),
                        "analyte_name": analyte_name,
                        "charttime": str(ctime),
                    })
                    if raise_on_unsupported:
                        raise UnsupportedPolicyAnalyteError(err_msg)
                    else:
                        # Quarantine without resolution
                        for l_idx in grp.index:
                            roles.loc[l_idx] = "UNSUPPORTED_POLICY_D_ANALYTE"
                            valuenum_eff.loc[l_idx] = np.nan
                        continue

                # Supported Policy D analytes
                numeric_candidates = grp[grp["valuenum_harmonized"].notna()]
                if numeric_candidates.empty:
                    # All missing: fallback to Policy A
                    winner_idx = grp.sort_values(
                        by=["storetime", "labevent_id"],
                        ascending=[False, True],
                    ).index[0]
                else:
                    if analyte_name in {"creatinine", "urea_nitrogen"}:
                        # Maximum value reflects worst azotemia / renal failure
                        target_val = numeric_candidates["valuenum_harmonized"].max()
                        best_candidates = numeric_candidates[numeric_candidates["valuenum_harmonized"] == target_val]
                    elif analyte_name == "bicarbonate":
                        # Minimum value reflects worst metabolic acidosis
                        target_val = numeric_candidates["valuenum_harmonized"].min()
                        best_candidates = numeric_candidates[numeric_candidates["valuenum_harmonized"] == target_val]
                    elif analyte_name == "potassium":
                        # Bi-directional distance from normal 4.0 mEq/L
                        dist = (numeric_candidates["valuenum_harmonized"] - 4.0).abs()
                        max_dist = dist.max()
                        best_candidates = numeric_candidates[dist == max_dist]
                        if len(best_candidates) > 1:
                            # Equidistant tie (e.g. 3.5 and 4.5): prefer higher value (hyperkalemic risk)
                            max_k = best_candidates["valuenum_harmonized"].max()
                            best_candidates = best_candidates[best_candidates["valuenum_harmonized"] == max_k]
                    else:
                        best_candidates = numeric_candidates

                    # Tie-break equal extremum values via Policy A (latest storetime, lowest labevent_id)
                    winner_idx = best_candidates.sort_values(
                        by=["storetime", "labevent_id"],
                        ascending=[False, True],
                    ).index[0]

                roles.loc[winner_idx] = "PRIMARY_SELECTED"
                for l_idx in grp.index:
                    if l_idx != winner_idx:
                        roles.loc[l_idx] = "SUPERSEDED_CONFLICT"
                        valuenum_eff.loc[l_idx] = np.nan
                        superseded_conflict_count += 1

    # Apply assigned roles and effective values
    work_df["arbitration_role"] = pd.Categorical(
        roles,
        categories=list(PERMITTED_ARBITRATION_ROLES),
    )
    work_df["valuenum_effective"] = valuenum_eff
    work_df["source_labevent_ids"] = source_ids

    # Append synthesized records if Policy C generated any
    if synthesized_rows:
        synth_df = pd.DataFrame(synthesized_rows)

        # Explicit collision detection strategy: fail explicitly on any duplicate or overlap
        synth_ids = synth_df["labevent_id"].tolist()
        if len(synth_ids) != len(set(synth_ids)):
            raise ArbitrationError(
                f"Collision detected among synthesized aggregate labevent_ids: "
                f"{len(synth_ids)} total vs {len(set(synth_ids))} unique."
            )
        source_id_set = set(work_df["labevent_id"])
        overlapping_ids = set(synth_ids) & source_id_set
        if overlapping_ids:
            raise ArbitrationError(
                f"Synthesized labevent_id overlaps with existing source labevent_id: {overlapping_ids}"
            )

        # Ensure categorical dtypes match
        synth_df["delay_anomaly_flag"] = pd.Categorical(
            synth_df["delay_anomaly_flag"],
            categories=work_df["delay_anomaly_flag"].cat.categories,
        )
        synth_df["unit_flag"] = pd.Categorical(
            synth_df["unit_flag"],
            categories=work_df["unit_flag"].cat.categories,
        )
        synth_df["arbitration_role"] = pd.Categorical(
            synth_df["arbitration_role"],
            categories=list(PERMITTED_ARBITRATION_ROLES),
        )
        result_df = pd.concat([work_df, synth_df], ignore_index=True)
    else:
        result_df = work_df

    # Deterministic sort order
    result_df.sort_values(
        by=["subject_id", "charttime", "itemid", "labevent_id"],
        ascending=[True, True, True, True],
        inplace=True,
    )
    result_df.reset_index(drop=True, inplace=True)

    # Validate output schema contract
    is_valid_out, validation_errors_out = validate_canonical_schema(result_df)
    if not is_valid_out:
        raise ArbitrationError(
            f"Arbitrated DataFrame failed canonical schema validation:\n" + "\n".join(validation_errors_out)
        )

    # Compile audit summary
    role_counts = result_df["arbitration_role"].value_counts()
    n_primary = int(role_counts.get("PRIMARY_SELECTED", 0))
    n_sup_dup = int(role_counts.get("SUPERSEDED_DUPLICATE", 0))
    n_sup_conflict = int(role_counts.get("SUPERSEDED_CONFLICT", 0))
    n_synth = int(role_counts.get("SYNTHESIZED_AGGREGATE", 0))
    n_raw = int(role_counts.get("UNARBITRATED_RAW", 0))
    n_retained = n_primary + n_synth + n_raw

    summary = ArbitrationSummary(
        policy_id=policy_id,
        total_rows_input=n_input,
        total_rows_output=len(result_df),
        retained_rows_count=n_retained,
        superseded_duplicate_count=n_sup_dup,
        superseded_conflict_count=n_sup_conflict,
        synthesized_aggregate_count=n_synth,
        primary_selected_count=n_primary,
        unarbitrated_raw_count=n_raw,
        exact_duplicate_groups_count=exact_dup_groups_count,
        conflict_groups_count=conflict_groups_count,
        groups_with_mixed_hadm_id=len(mixed_hadm_details),
        mixed_hadm_details=mixed_hadm_details,
        unsupported_analyte_conflicts_count=len(unsupported_analyte_details),
        unsupported_analyte_details=unsupported_analyte_details,
    )

    return result_df, summary


def get_effective_events(df: pd.DataFrame) -> pd.DataFrame:
    """Filter an arbitrated canonical DataFrame to retain only active effective events.

    Retains records where arbitration_role is PRIMARY_SELECTED, SYNTHESIZED_AGGREGATE,
    or UNARBITRATED_RAW, and excludes SUPERSEDED_DUPLICATE and SUPERSEDED_CONFLICT.

    Parameters:
        df: Arbitrated canonical DataFrame.

    Returns:
        Filtered DataFrame with superseded records excluded.
    """
    effective_roles = {"PRIMARY_SELECTED", "SYNTHESIZED_AGGREGATE", "UNARBITRATED_RAW"}
    mask = df["arbitration_role"].isin(effective_roles)
    return df[mask].copy().reset_index(drop=True)
