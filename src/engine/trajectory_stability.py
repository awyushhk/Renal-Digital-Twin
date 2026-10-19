"""Module 2: Trajectory Stability Index (TSI) Engine.

Calculates an empirical kinetic smoothness index from observed serum creatinine
history outside the latent state filter.

Mathematical properties and edge-case handling:
1. Minimum history: Requires at least 2 distinct temporal observation epochs
   for velocity; 3 epochs for divided-difference acceleration.
2. Missing / insufficient history: Returns status 'INSUFFICIENT_HISTORY' with
   score=None. Does NOT invent arbitrary values.
3. Duplicate timestamps and zero elapsed time: Groups by distinct charttime epoch
   or skips zero-delta collisions (dt <= 0). Never divides by zero.
4. Scale: Bounded in (0.0, 1.0], where 1.0 indicates zero observed kinetic drift.
5. Research boundary: Designated as an empirical kinetic curvature index.
   It is NOT a validated diagnostic measurement of nephron functional reserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TrajectoryStabilityResult:
    """Structured output of Trajectory Stability Index calculation."""
    tsi_score: Optional[float]
    status: str
    available_epochs_count: int
    most_recent_velocity_mg_dl_per_day: Optional[float]
    most_recent_acceleration_mg_dl_per_day2: Optional[float]
    hours_spanned: Optional[float]
    message: str


def compute_trajectory_stability_index(
    creatinine_records: pd.DataFrame,
    v_scale_mg_dl_per_day: float = 0.3,
    a_scale_mg_dl_per_day2: float = 0.3,
) -> TrajectoryStabilityResult:
    """Compute empirical Trajectory Stability Index (TSI) from historical creatinine records.

    Parameters:
        creatinine_records: DataFrame containing 'charttime' and 'valuenum' (or 'valuenum_effective').
        v_scale_mg_dl_per_day: Normalization scale for velocity (default 0.3 mg/dL / 24h,
                               aligned with KDIGO absolute rise reference scale).
        a_scale_mg_dl_per_day2: Normalization scale for acceleration.

    Returns:
        TrajectoryStabilityResult with score in (0, 1] or None if insufficient history.
    """
    if creatinine_records.empty:
        return TrajectoryStabilityResult(
            tsi_score=None,
            status="EMPTY_HISTORY",
            available_epochs_count=0,
            most_recent_velocity_mg_dl_per_day=None,
            most_recent_acceleration_mg_dl_per_day2=None,
            hours_spanned=None,
            message="No creatinine observations provided",
        )

    val_col = (
        "valuenum_effective"
        if "valuenum_effective" in creatinine_records.columns
        else "valuenum"
    )

    # Filter to non-null positive creatinine values
    df_clean = creatinine_records[
        creatinine_records[val_col].notna() & (creatinine_records[val_col] > 0)
    ].copy()

    if df_clean.empty:
        return TrajectoryStabilityResult(
            tsi_score=None,
            status="NO_VALID_CREATININE",
            available_epochs_count=0,
            most_recent_velocity_mg_dl_per_day=None,
            most_recent_acceleration_mg_dl_per_day2=None,
            hours_spanned=None,
            message="No valid positive creatinine values found",
        )

    # Coerce charttime to datetime and sort
    df_clean["charttime"] = pd.to_datetime(df_clean["charttime"])
    df_clean = df_clean.sort_values("charttime")

    # Group by distinct charttime epoch (resolves duplicate timestamps cleanly)
    epoch_series = df_clean.groupby("charttime")[val_col].median()
    times = epoch_series.index
    values = epoch_series.values

    n_epochs = len(times)
    if n_epochs < 2:
        return TrajectoryStabilityResult(
            tsi_score=None,
            status="INSUFFICIENT_HISTORY",
            available_epochs_count=n_epochs,
            most_recent_velocity_mg_dl_per_day=None,
            most_recent_acceleration_mg_dl_per_day2=None,
            hours_spanned=0.0,
            message=f"Requires at least 2 distinct temporal epochs; found {n_epochs}",
        )

    # Compute inter-epoch elapsed hours and velocities (converted to per-day: 24h)
    dt_hours = np.array(
        [(times[i] - times[i - 1]).total_seconds() / 3600.0 for i in range(1, n_epochs)]
    )

    # Filter out any non-positive delta-t anomalies (safety guard)
    valid_step_mask = dt_hours > 0.0
    if not np.any(valid_step_mask):
        return TrajectoryStabilityResult(
            tsi_score=None,
            status="ZERO_ELAPSED_INTERVALS",
            available_epochs_count=n_epochs,
            most_recent_velocity_mg_dl_per_day=None,
            most_recent_acceleration_mg_dl_per_day2=None,
            hours_spanned=0.0,
            message="All inter-observation intervals have non-positive elapsed time",
        )

    dv = np.diff(values)
    # Velocity in mg/dL per 24 hours
    velocities_per_day = (dv / dt_hours) * 24.0

    recent_v = float(velocities_per_day[-1])
    recent_a: Optional[float] = None

    if len(velocities_per_day) >= 2 and (dt_hours[-2] + dt_hours[-1]) > 0.0:
        # Unequal-interval acceleration
        midpoint_sep_days = (dt_hours[-2] + dt_hours[-1]) / (2.0 * 24.0)
        recent_a = float((velocities_per_day[-1] - velocities_per_day[-2]) / midpoint_sep_days)

    # Calculate kinetic curvature penalty
    penalty = abs(recent_v) / v_scale_mg_dl_per_day
    if recent_a is not None:
        penalty += abs(recent_a) / a_scale_mg_dl_per_day2

    tsi = 1.0 / (1.0 + penalty)
    total_hours = float((times[-1] - times[0]).total_seconds() / 3600.0)

    return TrajectoryStabilityResult(
        tsi_score=round(float(tsi), 4),
        status="COMPUTED",
        available_epochs_count=n_epochs,
        most_recent_velocity_mg_dl_per_day=round(recent_v, 4),
        most_recent_acceleration_mg_dl_per_day2=round(recent_a, 4) if recent_a is not None else None,
        hours_spanned=round(total_hours, 2),
        message="Trajectory Stability Index computed successfully",
    )
