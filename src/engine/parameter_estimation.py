"""Module 2: Parameter Estimation and Identifiability Engine.

Fits state-space model parameters strictly on Training patients (1,750 cohort patients)
using Gaussian prediction error decomposition maximum likelihood.

Identifiability and scientific boundaries:
1. Identifying constraints:
   - Factor 1 (x_filt) is anchored to Creatinine with fixed loading -1.0.
   - Factor 2 (x_tub) is anchored to Potassium with fixed loading +1.0.
   - Factor 3 (x_work) is anchored to BUN with fixed loading +1.0.
2. Free estimated parameters:
   - Time constants: tau_1, tau_2, tau_3 in [12.0, 720.0] hours.
   - Diffusion intensities: q_1, q_2, q_3 in [0.01, 1.0].
   - Cross-loadings: h_21, h_42, h_43, h_52, h_63 in [0.05, 3.0].
3. Convergence tracking:
   - Records optimizer convergence flag, message, iteration count, and boundary hits.
   - Never silently substitutes fabricated fitted parameters.
   - If any parameter hits a boundary, it is explicitly flagged in fitting metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.engine.data_partition import load_patient_splits
from src.engine.preprocessing import CohortScalerConfig, load_cohort_scalers, standardize_observation
from src.engine.state_space import (
    ANALYTE_INDEX_MAP,
    ContinuousDiscreteRenalFilter,
    StateSpaceParameters,
)
from src.temporal.canonical_schema import create_canonical_event_frame
from src.temporal.conflict_arbitration import POLICY_A_STORETIME, arbitrate_canonical_frame, get_effective_events

DEFAULT_FITTED_PARAMS_PATH = Path("configs/module2_fitted_parameters.json")


@dataclass
class FittingReport:
    """Detailed record of parameter estimation and convergence diagnostics."""
    success: bool
    status_message: str
    iterations: int
    negative_log_likelihood: float
    boundary_hits: List[str]
    training_patient_count: int
    training_steps_evaluated: int
    fitted_parameters: StateSpaceParameters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status_message": self.status_message,
            "iterations": self.iterations,
            "negative_log_likelihood": round(float(self.negative_log_likelihood), 4),
            "boundary_hits": self.boundary_hits,
            "training_patient_count": self.training_patient_count,
            "training_steps_evaluated": self.training_steps_evaluated,
            "fitted_parameters": self.fitted_parameters.to_dict(),
        }


def compute_patient_negative_log_likelihood(
    patient_df: pd.DataFrame,
    filter_instance: ContinuousDiscreteRenalFilter,
    scaler_config: CohortScalerConfig,
) -> Tuple[float, int]:
    """Compute total Gaussian negative log-likelihood for one patient's trajectory.

    -ln L = 0.5 * sum_k [ ln|S_k| + nu_k^T S_k^-1 nu_k + m_k * ln(2*pi) ]
    """
    total_nll = 0.0
    step_count = 0

    # Ensure chronological sort by charttime
    p_df = patient_df.sort_values("charttime").copy()

    # Reset filter state to prior for new patient (strict patient boundary isolation)
    filter_instance.state = np.zeros(3, dtype=float)
    filter_instance.cov = filter_instance.compute_stationary_cov()

    last_time: Optional[pd.Timestamp] = None

    # Group by observation epoch (same charttime)
    for ctime, epoch_df in p_df.groupby("charttime", sort=False):
        c_ts = pd.to_datetime(ctime)
        if last_time is None:
            dt_hours = 0.0
        else:
            dt_hours = max(0.0, (c_ts - last_time).total_seconds() / 3600.0)
        last_time = c_ts

        # Continuous prediction over dt
        filter_instance.predict(dt_hours)

        # Standardize available observations in this epoch
        obs_dict: Dict[int, float] = {}
        for _, row in epoch_df.iterrows():
            analyte = str(row["analyte_name"])
            val = float(row["valuenum_effective"])
            if analyte in ANALYTE_INDEX_MAP:
                std_v = standardize_observation(val, analyte, scaler_config)
                if not pd.isna(std_v):
                    obs_dict[ANALYTE_INDEX_MAP[analyte]] = std_v

        if obs_dict:
            res = filter_instance.update(obs_dict)
            if res is not None:
                # Log-likelihood contribution
                S = res.innovation_cov
                m_k = len(res.observed_analyte_indices)
                sign, logdet = np.linalg.slogdet(S)
                if sign > 0:
                    nll_step = 0.5 * (logdet + res.normalized_innovation_squared + m_k * np.log(2.0 * np.pi))
                    total_nll += float(nll_step)
                    step_count += 1

    return total_nll, step_count


def fit_state_space_parameters(
    cohort_labs_path: Path = Path("data/processed/cohort_renal_bounded.csv"),
    splits_json_path: Optional[Path] = None,
    output_params_path: Path = DEFAULT_FITTED_PARAMS_PATH,
    max_training_patients: int = 150,
    seed: int = 42,
) -> FittingReport:
    """Fit free state-space parameters on Training split patients using L-BFGS-B."""
    scaler_config = load_cohort_scalers()
    splits = load_patient_splits(splits_json_path) if splits_json_path else load_patient_splits()
    train_sids = sorted(list(splits["train"]))

    # Subsample training patients deterministically for tractability
    rng = np.random.default_rng(seed)
    selected_sids = rng.choice(
        train_sids,
        size=min(max_training_patients, len(train_sids)),
        replace=False,
    )
    selected_set = set(selected_sids)

    # Load and arbitrate training records
    raw_df = pd.read_csv(cohort_labs_path)
    train_raw = raw_df[raw_df["subject_id"].isin(selected_set)].copy()

    canonical_df = create_canonical_event_frame(train_raw)
    arbitrated_df, _ = arbitrate_canonical_frame(canonical_df, policy_id=POLICY_A_STORETIME)
    effective_df = get_effective_events(arbitrated_df)

    # Pre-group training patients to avoid repeated groupby inside optimizer
    patient_trajectories: List[pd.DataFrame] = [
        p_df.copy() for _, p_df in effective_df.groupby("subject_id")
    ]

    # Initial parameter vector:
    # 0: tau_1, 1: tau_2, 2: tau_3  (time constants in hours)
    # 3: q_1,   4: q_2,   5: q_3    (diffusion intensities)
    # 6: h_21,  7: h_42,  8: h_43, 9: h_52, 10: h_63 (cross-loadings)
    init_params = StateSpaceParameters()
    x0 = np.array([
        init_params.tau_hours[0],
        init_params.tau_hours[1],
        init_params.tau_hours[2],
        init_params.q_scales[0],
        init_params.q_scales[1],
        init_params.q_scales[2],
        init_params.h_21,
        init_params.h_42,
        init_params.h_43,
        init_params.h_52,
        init_params.h_63,
    ], dtype=float)

    bounds = [
        (12.0, 720.0),   # tau_1
        (12.0, 720.0),   # tau_2
        (12.0, 720.0),   # tau_3
        (0.01, 1.0),     # q_1
        (0.01, 1.0),     # q_2
        (0.01, 1.0),     # q_3
        (0.05, 3.0),     # h_21
        (0.05, 3.0),     # h_42
        (0.05, 3.0),     # h_43
        (0.05, 3.0),     # h_52
        (0.05, 3.0),     # h_63
    ]

    param_names = [
        "tau_1", "tau_2", "tau_3",
        "q_1", "q_2", "q_3",
        "h_21", "h_42", "h_43", "h_52", "h_63",
    ]

    def _objective(vec: np.ndarray) -> float:
        params = StateSpaceParameters(
            tau_hours=[float(vec[0]), float(vec[1]), float(vec[2])],
            q_scales=[float(vec[3]), float(vec[4]), float(vec[5])],
            h_21=float(vec[6]),
            h_42=float(vec[7]),
            h_43=float(vec[8]),
            h_52=float(vec[9]),
            h_63=float(vec[10]),
        )
        filter_inst = ContinuousDiscreteRenalFilter(params=params)

        total_nll = 0.0
        for p_traj in patient_trajectories:
            nll, _ = compute_patient_negative_log_likelihood(p_traj, filter_inst, scaler_config)
            total_nll += nll
        return total_nll

    # Run bounded optimization
    opt_res = minimize(
        _objective,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 35, "disp": False, "ftol": 1e-4},
    )

    opt_vec = opt_res.x

    # Detect boundary solutions
    boundary_hits: List[str] = []
    for i, (val, (b_low, b_high)) in enumerate(zip(opt_vec, bounds)):
        name = param_names[i]
        if np.isclose(val, b_low, atol=1e-3):
            boundary_hits.append(f"{name}_HIT_LOWER_BOUND({round(val, 4)})")
        elif np.isclose(val, b_high, atol=1e-3):
            boundary_hits.append(f"{name}_HIT_UPPER_BOUND({round(val, 4)})")

    fitted_params = StateSpaceParameters(
        tau_hours=[round(float(opt_vec[0]), 2), round(float(opt_vec[1]), 2), round(float(opt_vec[2]), 2)],
        q_scales=[round(float(opt_vec[3]), 4), round(float(opt_vec[4]), 4), round(float(opt_vec[5]), 4)],
        h_21=round(float(opt_vec[6]), 4),
        h_42=round(float(opt_vec[7]), 4),
        h_43=round(float(opt_vec[8]), 4),
        h_52=round(float(opt_vec[9]), 4),
        h_63=round(float(opt_vec[10]), 4),
    )

    # Count total steps evaluated
    total_steps = sum(len(p) for p in patient_trajectories)

    report = FittingReport(
        success=bool(opt_res.success),
        status_message=str(opt_res.message),
        iterations=int(opt_res.nit),
        negative_log_likelihood=float(opt_res.fun),
        boundary_hits=boundary_hits,
        training_patient_count=len(patient_trajectories),
        training_steps_evaluated=total_steps,
        fitted_parameters=fitted_params,
    )

    output_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_params_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    return report


def load_state_space_parameters(
    params_path: Path = DEFAULT_FITTED_PARAMS_PATH,
) -> StateSpaceParameters:
    """Load fitted parameters or fit on demand if file does not exist."""
    if not params_path.exists():
        fit_state_space_parameters(output_params_path=params_path)

    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "fitted_parameters" in data:
        return StateSpaceParameters.from_dict(data["fitted_parameters"])
    return StateSpaceParameters.from_dict(data)
