"""Module 1.3 Step 3: Irregular Temporal Features and Dual-Timeline Engine.

This module implements deterministic, irregular-time, causal feature generation across
two strictly separated temporal information sets according to
`docs/m1_3_temporal_representation_protocol.md`:

1. Biological Timeline (F_bio):
   Specimen draw time (`charttime`) for retrospective biological trajectory reconstruction.
   Explicitly tagged as non-deployable for real-time alerting.
   Features include biological elapsed hours, irregular velocity, and unequal-interval acceleration.

2. Operational Timeline (F_ops):
   EHR filing time (`storetime`) representing operational availability.
   Evaluated at explicit query/prediction times.
   Results with storetime > query_time are strictly excluded (no operational lookahead).
   Late-arriving results entering at later query times never retroactively rewrite past snapshots.
   Reporting-time rates of change are explicitly documented as reporting arrival cadences,
   never inferred as physiological dynamics.

3. Co-temporal Epochs and Invariants:
   Multi-analyte draws at identical charttime form co-temporal epochs with causal symmetry.
   Ratios (BUN/Creatinine) are computed only on co-temporal draws with non-negative delays
   and positive creatinine, never silently substituted.
   All clocks and running statistics reset strictly at patient boundaries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.temporal.canonical_schema import (
    TARGET_ITEM_IDS,
    ITEM_TO_ANALYTE,
    CANONICAL_UNITS,
    validate_canonical_schema,
)
from src.temporal.conflict_arbitration import get_effective_events

# Canonical analytes list
TARGET_ANALYTES: List[str] = [
    "creatinine",
    "urea_nitrogen",
    "potassium",
    "sodium",
    "bicarbonate",
    "glucose",
]


class TemporalFeatureError(Exception):
    """Base exception for temporal feature generation errors."""
    pass


@dataclass
class DualTimelineSummary:
    """Summary metrics of dual-timeline feature generation."""
    total_input_records: int
    total_effective_records: int
    biological_epochs_count: int
    patient_count: int
    zero_delta_velocity_collisions: int
    zero_delta_acceleration_collisions: int
    insufficient_history_accelerations: int
    valid_bun_creatinine_ratios: int
    missing_constituent_ratios: int
    non_positive_creatinine_ratios: int
    quarantined_negative_reporting_delays: int
    quarantined_missing_storetimes: int


def calculate_irregular_velocity(
    x_prev: float,
    x_curr: float,
    dt_hours: float,
) -> Tuple[float, str]:
    """Calculate causal discrete velocity on irregular time intervals.

    Parameters:
        x_prev: Analyte value at previous epoch.
        x_curr: Analyte value at current epoch.
        dt_hours: Elapsed time difference in decimal hours.

    Returns:
        (velocity, status_flag)
    """
    if pd.isna(x_prev) or pd.isna(x_curr) or pd.isna(dt_hours):
        return np.nan, "MISSING_DATA"
    if dt_hours <= 0.0:
        return np.nan, "COLLISION_ZERO_DELTA"
    v = (x_curr - x_prev) / dt_hours
    return float(v), "VALID"


def calculate_unequal_interval_acceleration(
    x_prev2: float,
    x_prev1: float,
    x_curr: float,
    dt12_hours: float,
    dt23_hours: float,
) -> Tuple[float, str]:
    """Calculate non-uniform divided-difference acceleration across unequal intervals.

    Formula:
        a = 2 * ((x_curr - x_prev1)/dt23 - (x_prev1 - x_prev2)/dt12) / (dt12 + dt23)
        where (dt12 + dt23)/2 is the exact separation between velocity midpoints.

    Parameters:
        x_prev2: Value at epoch k-2.
        x_prev1: Value at epoch k-1.
        x_curr: Value at epoch k.
        dt12_hours: Elapsed hours between epoch k-2 and k-1.
        dt23_hours: Elapsed hours between epoch k-1 and k.

    Returns:
        (acceleration, status_flag)
    """
    if pd.isna(x_prev2) or pd.isna(x_prev1) or pd.isna(x_curr):
        return np.nan, "MISSING_DATA"
    if pd.isna(dt12_hours) or pd.isna(dt23_hours):
        return np.nan, "MISSING_DATA"
    if dt12_hours <= 0.0 or dt23_hours <= 0.0 or (dt12_hours + dt23_hours) <= 0.0:
        return np.nan, "COLLISION_ZERO_DELTA"

    v12 = (x_prev1 - x_prev2) / dt12_hours
    v23 = (x_curr - x_prev1) / dt23_hours
    midpoint_separation = (dt12_hours + dt23_hours) / 2.0
    a = (v23 - v12) / midpoint_separation
    return float(a), "VALID"


def compute_cotemporal_bun_creatinine_ratio(
    bun_val: Optional[float],
    cr_val: Optional[float],
) -> Tuple[float, str]:
    """Compute co-temporal BUN/Creatinine ratio strictly within the same patient and epoch.

    Parameters:
        bun_val: Urea nitrogen effective value.
        cr_val: Creatinine effective value.

    Returns:
        (ratio, status_flag)
    """
    if bun_val is None or cr_val is None or pd.isna(bun_val) or pd.isna(cr_val):
        return np.nan, "MISSING_CONSTITUENT"
    if cr_val <= 0.0:
        return np.nan, "NON_POSITIVE_CREATININE"
    ratio = bun_val / cr_val
    return float(np.round(ratio, 4)), "VALID_COTEMPORAL_RATIO"


def generate_biological_features(
    df: pd.DataFrame,
    copy: bool = True,
) -> Tuple[pd.DataFrame, DualTimelineSummary]:
    """Generate irregular temporal features on the retrospective biological timeline (F_bio).

    - Driven by specimen collection time `charttime`.
    - Features represent retrospective continuous-time physiological trajectory reconstruction.
    - Explicitly flagged with `is_retrospective_only = True`.

    Parameters:
        df: Input canonical or arbitrated DataFrame.
        copy: If True, deep copies the input DataFrame.

    Returns:
        (feature_df, summary)
    """
    if copy:
        work_df = df.copy(deep=True)
    else:
        work_df = df

    # Retain active/effective records only
    effective_df = get_effective_events(work_df)

    # Sort deterministically by subject_id, charttime, itemid, labevent_id
    effective_df["charttime"] = pd.to_datetime(effective_df["charttime"])
    effective_df["storetime"] = pd.to_datetime(effective_df["storetime"])
    effective_df.sort_values(
        by=["subject_id", "charttime", "itemid", "labevent_id"],
        ascending=[True, True, True, True],
        inplace=True,
    )
    effective_df.reset_index(drop=True, inplace=True)

    # Output containers
    hours_since_first = []
    dt_admission = []
    dt_prev_event = []
    dt_prev_analyte = []
    analyte_v_bio = []
    analyte_v_bio_midpoint_hours = []
    analyte_v_bio_flag = []
    analyte_a_bio = []
    analyte_a_bio_flag = []
    cum_obs_counts = {analyte: [] for analyte in TARGET_ANALYTES}
    running_means = {analyte: [] for analyte in TARGET_ANALYTES}
    running_mins = {analyte: [] for analyte in TARGET_ANALYTES}
    running_maxs = {analyte: [] for analyte in TARGET_ANALYTES}
    bun_cr_ratios = []
    bun_cr_ratio_flags = []

    # Summary metric counters
    zero_v_collisions = 0
    zero_a_collisions = 0
    insufficient_h_acc = 0
    valid_bun_cr = 0
    missing_bun_cr = 0
    non_pos_cr = 0
    quarantined_neg_delay = int((effective_df["delay_anomaly_flag"] == "NEGATIVE_REPORTING_DELAY").sum())
    quarantined_null_store = int(effective_df["storetime"].isna().sum())

    # Process grouped by patient to guarantee strict boundary isolation
    for subject_id, patient_df in effective_df.groupby("subject_id", sort=False):
        t_first_bio = patient_df["charttime"].min()
        last_event_time = None

        # Patient-level state tracking
        analyte_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = {a: [] for a in TARGET_ANALYTES}

        # Group by observation epoch (same charttime) to enforce co-temporal symmetry
        for ctime, epoch_df in patient_df.groupby("charttime", sort=False):
            # 1. Biological elapsed lifetime clock from first eligible charttime
            tau_hours = (ctime - t_first_bio).total_seconds() / 3600.0

            # 2. Inter-event gap across any analyte
            if last_event_time is None:
                dt_event = np.nan
            else:
                dt_event = (ctime - last_event_time).total_seconds() / 3600.0
            last_event_time = ctime

            # 3. Check co-temporal BUN and Creatinine in this epoch
            epoch_measurements = dict(zip(epoch_df["analyte_name"], epoch_df["valuenum_effective"]))
            bun_val = epoch_measurements.get("urea_nitrogen")
            cr_val = epoch_measurements.get("creatinine")
            ratio, ratio_flag = compute_cotemporal_bun_creatinine_ratio(bun_val, cr_val)
            if ratio_flag == "VALID_COTEMPORAL_RATIO":
                valid_bun_cr += len(epoch_df)
            elif ratio_flag == "NON_POSITIVE_CREATININE":
                non_pos_cr += len(epoch_df)
            else:
                missing_bun_cr += len(epoch_df)

            # Process each observation row within the epoch
            for _, row in epoch_df.iterrows():
                hours_since_first.append(tau_hours)
                dt_prev_event.append(dt_event)

                # Admission elapsed time
                if pd.notna(row.get("hadm_id")) and pd.notna(row.get("admittime")):
                    adm_dt = pd.to_datetime(row["admittime"])
                    dt_adm_hours = (ctime - adm_dt).total_seconds() / 3600.0
                else:
                    dt_adm_hours = np.nan
                dt_admission.append(dt_adm_hours)

                # Co-temporal ratio assigned to every row in epoch
                bun_cr_ratios.append(ratio)
                bun_cr_ratio_flags.append(ratio_flag)

                # Analyte specific history
                current_analyte = row["analyte_name"]
                current_val = row["valuenum_effective"]
                hist = analyte_history[current_analyte]

                # Analyte gap
                if len(hist) == 0:
                    dt_analyte = np.nan
                    v_bio = np.nan
                    v_mid_hours = np.nan
                    v_flag = "FIRST_OBSERVATION"
                    a_bio = np.nan
                    a_flag = "INSUFFICIENT_HISTORY"
                    insufficient_h_acc += 1
                else:
                    prev_time, prev_val = hist[-1]
                    dt_analyte = (ctime - prev_time).total_seconds() / 3600.0
                    v_bio, v_flag = calculate_irregular_velocity(prev_val, current_val, dt_analyte)
                    if v_flag == "COLLISION_ZERO_DELTA":
                        zero_v_collisions += 1
                        v_mid_hours = np.nan
                    else:
                        v_mid_hours = ((prev_time + (ctime - prev_time) / 2) - t_first_bio).total_seconds() / 3600.0

                    # Acceleration requires k >= 3 epochs
                    if len(hist) < 2:
                        a_bio = np.nan
                        a_flag = "INSUFFICIENT_HISTORY"
                        insufficient_h_acc += 1
                    else:
                        prev2_time, prev2_val = hist[-2]
                        dt12 = (prev_time - prev2_time).total_seconds() / 3600.0
                        a_bio, a_flag = calculate_unequal_interval_acceleration(
                            prev2_val, prev_val, current_val, dt12, dt_analyte
                        )
                        if a_flag == "COLLISION_ZERO_DELTA":
                            zero_a_collisions += 1

                dt_prev_analyte.append(dt_analyte)
                analyte_v_bio.append(v_bio)
                analyte_v_bio_midpoint_hours.append(v_mid_hours)
                analyte_v_bio_flag.append(v_flag)
                analyte_a_bio.append(a_bio)
                analyte_a_bio_flag.append(a_flag)

                # Update running statistics for each target analyte
                for a in TARGET_ANALYTES:
                    a_hist = analyte_history[a]
                    # If this row is analyte a, include current_val in running stats
                    vals = [v for _, v in a_hist if pd.notna(v)]
                    if a == current_analyte and pd.notna(current_val):
                        vals = vals + [current_val]
                    if len(vals) == 0:
                        cum_obs_counts[a].append(0)
                        running_means[a].append(np.nan)
                        running_mins[a].append(np.nan)
                        running_maxs[a].append(np.nan)
                    else:
                        cum_obs_counts[a].append(len(vals))
                        running_means[a].append(float(np.mean(vals)))
                        running_mins[a].append(float(np.min(vals)))
                        running_maxs[a].append(float(np.max(vals)))

            # Append epoch observations to analyte history after processing whole epoch
            for analyte, val in epoch_measurements.items():
                if analyte in analyte_history:
                    analyte_history[analyte].append((ctime, val))

    # Assemble output DataFrame
    out_df = effective_df.copy()
    out_df["is_retrospective_only"] = True
    out_df["hours_since_first_observed_event_bio"] = hours_since_first
    out_df["dt_admission_hours_bio"] = dt_admission
    out_df["is_inpatient"] = out_df["hadm_id"].notna()
    out_df["dt_previous_event_hours_bio"] = dt_prev_event
    out_df["dt_previous_analyte_hours_bio"] = dt_prev_analyte

    out_df["velocity_biological"] = analyte_v_bio
    out_df["velocity_biological_midpoint_hours"] = analyte_v_bio_midpoint_hours
    out_df["velocity_biological_flag"] = analyte_v_bio_flag
    out_df["acceleration_biological"] = analyte_a_bio
    out_df["acceleration_biological_flag"] = analyte_a_bio_flag

    out_df["bun_creatinine_ratio_bio"] = bun_cr_ratios
    out_df["bun_creatinine_ratio_bio_flag"] = bun_cr_ratio_flags

    # Running statistics and cumulative observation counts
    for a in TARGET_ANALYTES:
        out_df[f"cum_obs_count_{a}"] = cum_obs_counts[a]
        out_df[f"running_mean_{a}"] = running_means[a]
        out_df[f"running_min_{a}"] = running_mins[a]
        out_df[f"running_max_{a}"] = running_maxs[a]

    # Synchronous multi-analyte presence mask
    for a in TARGET_ANALYTES:
        out_df[f"mask_{a}"] = (out_df["analyte_name"] == a).astype(int)

    summary = DualTimelineSummary(
        total_input_records=len(df),
        total_effective_records=len(out_df),
        biological_epochs_count=out_df.groupby(["subject_id", "charttime"]).ngroups,
        patient_count=out_df["subject_id"].nunique(),
        zero_delta_velocity_collisions=zero_v_collisions,
        zero_delta_acceleration_collisions=zero_a_collisions,
        insufficient_history_accelerations=insufficient_h_acc,
        valid_bun_creatinine_ratios=valid_bun_cr,
        missing_constituent_ratios=missing_bun_cr,
        non_positive_creatinine_ratios=non_pos_cr,
        quarantined_negative_reporting_delays=quarantined_neg_delay,
        quarantined_missing_storetimes=quarantined_null_store,
    )

    return out_df, summary


def generate_operational_replay_snapshot(
    df: pd.DataFrame,
    query_time: pd.Timestamp,
    subject_id: Optional[int] = None,
) -> pd.DataFrame:
    """Generate an immutable operational snapshot for a clinical query time (F_ops).

    Strict Operational Rules:
    - Only records with storetime <= query_time and non-negative reporting delay are visible.
    - Records with storetime > query_time are strictly excluded (no operational lookahead).
    - Rates calculated using storetime differences represent reporting-arrival velocity/acceleration,
      never inferred as biological dynamics.
    - Emitted snapshots are immutable; later query times never retroactively rewrite earlier snapshots.

    Parameters:
        df: Input canonical or arbitrated DataFrame.
        query_time: Clinical clock timestamp at which predictions/features are requested.
        subject_id: Optional patient filter.

    Returns:
        DataFrame representing the operational state known at query_time.
    """
    effective_df = get_effective_events(df)
    effective_df["charttime"] = pd.to_datetime(effective_df["charttime"])
    effective_df["storetime"] = pd.to_datetime(effective_df["storetime"])

    if subject_id is not None:
        effective_df = effective_df[effective_df["subject_id"] == subject_id]

    query_dt = pd.to_datetime(query_time)

    # Operational visibility filter: storetime <= query_time AND delay >= 0
    ops_mask = (
        (effective_df["storetime"] <= query_dt)
        & (effective_df["storetime"].notna())
        & (effective_df["delay_anomaly_flag"] != "NEGATIVE_REPORTING_DELAY")
    )
    visible_df = effective_df[ops_mask].copy()

    if visible_df.empty:
        # Return empty operational frame with expected schema
        visible_df["query_time"] = query_dt
        visible_df["is_retrospective_only"] = False
        visible_df["hours_since_first_available_event_ops"] = []
        visible_df["velocity_operational_reporting"] = []
        visible_df["acceleration_operational_reporting"] = []
        visible_df["bun_creatinine_ratio_ops"] = []
        visible_df["bun_creatinine_ratio_ops_flag"] = []
        return visible_df

    # Sort deterministically by storetime, subject_id, itemid, labevent_id
    visible_df.sort_values(
        by=["subject_id", "storetime", "charttime", "labevent_id"],
        ascending=[True, True, True, True],
        inplace=True,
    )
    visible_df.reset_index(drop=True, inplace=True)

    # Process operational features per patient
    ops_elapsed = []
    ops_v_rep = []
    ops_a_rep = []
    ops_bun_cr = []
    ops_bun_cr_flag = []

    for sid, pat_df in visible_df.groupby("subject_id", sort=False):
        t_first_ops = pat_df["storetime"].min()
        pat_analyte_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = {a: [] for a in TARGET_ANALYTES}

        # Check co-temporal BUN and Creatinine filed by query_time
        # (Both must share the exact same charttime epoch and both filed <= query_time)
        cotemporal_groups = pat_df.groupby("charttime")

        for _, row in pat_df.iterrows():
            stime = row["storetime"]
            ctime = row["charttime"]
            # 1. Operational elapsed hours since first available event
            tau_ops = (stime - t_first_ops).total_seconds() / 3600.0
            ops_elapsed.append(tau_ops)

            # 2. Operational BUN/Cr ratio
            epoch_rows = cotemporal_groups.get_group(ctime)
            epoch_analytes = dict(zip(epoch_rows["analyte_name"], epoch_rows["valuenum_effective"]))
            bun_v = epoch_analytes.get("urea_nitrogen")
            cr_v = epoch_analytes.get("creatinine")

            if bun_v is not None and cr_v is not None:
                r, r_flag = compute_cotemporal_bun_creatinine_ratio(bun_v, cr_v)
            else:
                r, r_flag = np.nan, "RESULT_NOT_YET_FILED"
            ops_bun_cr.append(r)
            ops_bun_cr_flag.append(r_flag)

            # 3. Operational reporting velocity and acceleration
            analyte = row["analyte_name"]
            val = row["valuenum_effective"]
            hist = pat_analyte_history[analyte]

            if len(hist) == 0:
                v_rep = np.nan
                a_rep = np.nan
            else:
                prev_stime, prev_v = hist[-1]
                dt_rep = (stime - prev_stime).total_seconds() / 3600.0
                v_rep, _ = calculate_irregular_velocity(prev_v, val, dt_rep)

                if len(hist) < 2:
                    a_rep = np.nan
                else:
                    prev2_stime, prev2_v = hist[-2]
                    dt12_rep = (prev_stime - prev2_stime).total_seconds() / 3600.0
                    a_rep, _ = calculate_unequal_interval_acceleration(prev2_v, prev_v, val, dt12_rep, dt_rep)

            ops_v_rep.append(v_rep)
            ops_a_rep.append(a_rep)

            # Append to history
            pat_analyte_history[analyte].append((stime, val))

    visible_df["query_time"] = query_dt
    visible_df["is_retrospective_only"] = False
    visible_df["hours_since_first_available_event_ops"] = ops_elapsed
    visible_df["velocity_operational_reporting"] = ops_v_rep
    visible_df["acceleration_operational_reporting"] = ops_a_rep
    visible_df["bun_creatinine_ratio_ops"] = ops_bun_cr
    visible_df["bun_creatinine_ratio_ops_flag"] = ops_bun_cr_flag

    return visible_df


# --------------------------------------------------------------------------
# Baseline Generation Engines for Comparative Evaluation (Arms 1 and 2)
# --------------------------------------------------------------------------
def generate_arm1_conventional_locf_grid(
    df: pd.DataFrame,
    resample_hours: int = 24,
) -> pd.DataFrame:
    """Arm 1: Fair Conventional Baseline (24-hour regular grid with LOCF).

    - Resamples time series to regular 24-hour intervals per patient.
    - Forward-fills missing days using Last Observation Carried Forward (LOCF).
    - Resets strictly at patient boundaries.
    - Preserves outpatient records.
    """
    effective_df = get_effective_events(df)
    effective_df["charttime"] = pd.to_datetime(effective_df["charttime"])

    grid_rows = []
    freq_str = f"{resample_hours}h"

    for sid, pat_df in effective_df.groupby("subject_id", sort=False):
        pat_min = pat_df["charttime"].min().floor("h")
        pat_max = pat_df["charttime"].max().ceil("h")
        if pat_min == pat_max:
            pat_max = pat_min + pd.Timedelta(hours=resample_hours)

        date_grid = pd.date_range(start=pat_min, end=pat_max, freq=freq_str)

        # Pivot to analyte columns
        pivot_pat = pat_df.pivot_table(
            index="charttime",
            columns="analyte_name",
            values="valuenum_effective",
            aggfunc="last",
        ).reindex(columns=TARGET_ANALYTES)

        # Reindex to regular date grid and apply LOCF within patient
        grid_pat = pivot_pat.reindex(pivot_pat.index.union(date_grid)).sort_index()
        grid_locf = grid_pat.ffill()
        grid_sampled = grid_locf.loc[date_grid].copy()

        grid_sampled["subject_id"] = sid
        grid_sampled["grid_charttime"] = date_grid
        grid_rows.append(grid_sampled.reset_index(drop=True))

    arm1_df = pd.concat(grid_rows, ignore_index=True)
    return arm1_df


def generate_arm2_flawed_controls(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Arm 2: Deliberately Flawed Controls demonstrating standard ML pitfalls.

    - Control 2A: Full-Batch Normalization Leakage (MinMax fit across past + future).
    - Control 2B: Cross-Patient Delta Bleed (Naive diff across un-reset DataFrame).
    - Control 2C: Admissions Inner-Join Drop (Drops all outpatient null hadm_id labs).
    """
    effective_df = get_effective_events(df)
    effective_df["charttime"] = pd.to_datetime(effective_df["charttime"])

    # Control 2A: Full-batch minmax leakage
    c2a_df = effective_df.copy()
    for a in TARGET_ANALYTES:
        mask = c2a_df["analyte_name"] == a
        vals = c2a_df.loc[mask, "valuenum_effective"]
        if not vals.empty and vals.max() > vals.min():
            c2a_df.loc[mask, f"{a}_global_leaked_norm"] = (vals - vals.min()) / (vals.max() - vals.min())

    # Control 2B: Cross-patient delta bleed
    c2b_df = effective_df.sort_values(by=["charttime"]).copy()
    c2b_df["unreset_time_delta_hours"] = c2b_df["charttime"].diff().dt.total_seconds() / 3600.0

    # Control 2C: Admissions inner-join drop
    c2c_df = effective_df[effective_df["hadm_id"].notna()].copy()

    return {
        "control_2a_leakage": c2a_df,
        "control_2b_bleed": c2b_df,
        "control_2c_loss": c2c_df,
    }
