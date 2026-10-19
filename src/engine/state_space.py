"""Module 2: 3D Continuous-Discrete State-Space Model for Renal Latent Signals.

Latent States:
- Index 0: x_filt  (filtration-associated latent signal)
- Index 1: x_tub   (electrolyte/acid-base-associated latent signal)
- Index 2: x_work  (BUN/glucose-associated workload proxy)

Mathematical formulation:
Between irregular observations: Continuous-time Ornstein-Uhlenbeck (OU) process:
  dx(t) = F (x(t) - mu) dt + L dW(t)
  where F = -diag(1/tau_1, 1/tau_2, 1/tau_3) with time constants tau_i in hours.
  Note: tau_i are continuous-time constants (e^{-t/tau}), NOT half-lives (t_{1/2} = tau * ln(2)).

At observation epochs: Discrete Kalman measurement update:
  y_k = H_k x(t_k) + eps_k,  eps_k ~ N(0, R_k)
  where H_k is dynamically selected for available analytes at t_k.

Cross-covariance note:
  A potassium-only draw has direct observation loading ONLY on x_tub ([0.0, 1.0, 0.0]).
  However, non-zero cross-covariance in P(t) legitimately updates x_filt and x_work
  via the Kalman gain K = P H^T S^{-1}.

Research classification:
  These are lab-derived latent scores, not direct measurements of nephron mass,
  true renal functional reserve, or biopsy-level tubular pathology.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ANALYTE_INDEX_MAP = {
    "creatinine": 0,
    "urea_nitrogen": 1,
    "potassium": 2,
    "sodium": 3,
    "bicarbonate": 4,
    "glucose": 5,
}

INDEX_TO_ANALYTE = {v: k for k, v in ANALYTE_INDEX_MAP.items()}
STATE_NAMES = ["x_filt", "x_tub", "x_work"]


@dataclass
class StateSpaceParameters:
    """Parameters governing continuous-time dynamics and observation projections."""
    # Continuous-time constants in hours: tau_i in [12.0, 720.0]
    tau_hours: List[float] = field(default_factory=lambda: [72.0, 48.0, 96.0])

    # Diffusion scale intensities: q_i
    q_scales: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.15])

    # Free observation loadings
    # Row 0: Cr  = [-1.0,  0.0,  0.0] (anchored)
    # Row 1: BUN = [-h_21, 0.0, +1.0] (anchored on work)
    # Row 2: K   = [ 0.0, +1.0,  0.0] (anchored on tub)
    # Row 3: Na  = [ 0.0, -h_42, +h_43]
    # Row 4: HCO3= [ 0.0, -h_52, 0.0]
    # Row 5: Glu = [ 0.0,  0.0, +h_63]
    h_21: float = 0.85
    h_42: float = 0.30
    h_43: float = 0.40
    h_52: float = 0.60
    h_63: float = 0.35

    # Measurement noise standard deviations in standardized space (EFLM provisional priors)
    r_stds: List[float] = field(
        default_factory=lambda: [
            0.15,  # creatinine
            0.25,  # urea_nitrogen
            0.20,  # potassium
            0.10,  # sodium
            0.20,  # bicarbonate
            0.22,  # glucose
        ]
    )

    # Global noise scaling factor for sensitivity analyses
    noise_scale_factor: float = 1.0

    def get_time_constants(self) -> np.ndarray:
        return np.array(self.tau_hours, dtype=float)

    def get_decay_rates(self) -> np.ndarray:
        """Return decay rates theta = 1 / tau."""
        return 1.0 / np.array(self.tau_hours, dtype=float)

    def get_full_h_matrix(self) -> np.ndarray:
        """Return 6x3 observation loading matrix."""
        H = np.zeros((6, 3), dtype=float)
        # Creatinine: filtration only (negative loading: higher Cr = lower filtration)
        H[0, 0] = -1.0
        # BUN: filtration and workload
        H[1, 0] = -self.h_21
        H[1, 2] = +1.0
        # Potassium: tubular only (no direct loading on filt or work)
        H[2, 1] = +1.0
        # Sodium: tubular and workload
        H[3, 1] = -self.h_42
        H[3, 2] = +self.h_43
        # Bicarbonate: tubular only
        H[4, 1] = -self.h_52
        # Glucose: workload / metabolic load
        H[5, 2] = +self.h_63
        return H

    def get_full_r_matrix(self) -> np.ndarray:
        """Return 6x6 diagonal measurement noise covariance matrix."""
        diag_vars = [(s * self.noise_scale_factor) ** 2 for s in self.r_stds]
        return np.diag(diag_vars).astype(float)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateSpaceParameters:
        return cls(**data)


@dataclass
class FilterStepResult:
    """Detailed output of a single continuous-discrete filter step."""
    time_elapsed_hours: float
    prior_state: np.ndarray
    prior_cov: np.ndarray
    posterior_state: np.ndarray
    posterior_cov: np.ndarray
    observed_analyte_indices: List[int]
    innovations: np.ndarray
    innovation_cov: np.ndarray
    normalized_innovation_squared: float


class ContinuousDiscreteRenalFilter:
    """Continuous-Discrete 3D Kalman Filter for irregular renal trajectories."""

    def __init__(
        self,
        params: Optional[StateSpaceParameters] = None,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
    ) -> None:
        self.params = params or StateSpaceParameters()

        # Prior mean is 0.0 in standardized space
        self.state = (
            initial_state.copy().astype(float)
            if initial_state is not None
            else np.zeros(3, dtype=float)
        )

        # Prior covariance initialized to stationary process covariance P_inf
        if initial_cov is not None:
            self.cov = initial_cov.copy().astype(float)
        else:
            self.cov = self.compute_stationary_cov()

    def compute_stationary_cov(self) -> np.ndarray:
        """Compute asymptotic stationary covariance P_inf = q_i^2 / (2 * theta_i)."""
        theta = self.params.get_decay_rates()
        q = np.array(self.params.q_scales, dtype=float)
        p_inf_diag = (q ** 2) / (2.0 * theta)
        return np.diag(p_inf_diag).astype(float)

    def predict(self, dt_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        """Continuous-time prediction step over irregular interval dt_hours.

        x^-(t) = Phi(dt) * x^+(t_prev)
        P^-(t) = Phi(dt) * P^+(t_prev) * Phi(dt)^T + Q(dt)
        """
        if dt_hours < 0.0:
            raise ValueError(f"Elapsed time dt_hours must be non-negative, got {dt_hours}")

        if dt_hours == 0.0:
            return self.state.copy(), self.cov.copy()

        theta = self.params.get_decay_rates()
        q = np.array(self.params.q_scales, dtype=float)

        # Analytical transition matrix: exp(-theta * dt)
        phi_diag = np.exp(-theta * dt_hours)
        Phi = np.diag(phi_diag)

        # Analytical integrated process covariance
        # Q_ii = (q_i^2 / 2*theta_i) * (1 - exp(-2*theta_i * dt))
        q_diag = (q ** 2) / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt_hours))
        Q = np.diag(q_diag)

        # Propagate state and covariance
        state_pred = Phi @ self.state
        cov_pred = Phi @ self.cov @ Phi.T + Q

        # Numerical symmetrization
        cov_pred = 0.5 * (cov_pred + cov_pred.T)

        self.state = state_pred
        self.cov = cov_pred
        return self.state.copy(), self.cov.copy()

    def update(
        self,
        observations: Dict[int, float],
    ) -> Optional[FilterStepResult]:
        """Discrete measurement update with dynamic observation row selection.

        Parameters:
            observations: Dict mapping analyte_index (0..5) -> standardized value.

        Returns:
            FilterStepResult containing innovations and updated state/cov,
            or None if no valid observations provided.
        """
        valid_obs = {
            idx: float(val)
            for idx, val in observations.items()
            if idx in ANALYTE_INDEX_MAP.values() and not pd.isna(val)
        }

        if not valid_obs:
            return None

        obs_indices = sorted(valid_obs.keys())
        y_vec = np.array([valid_obs[i] for i in obs_indices], dtype=float)

        H_full = self.params.get_full_h_matrix()
        R_full = self.params.get_full_r_matrix()

        # Dynamic row selection
        H_k = H_full[obs_indices, :]
        R_k = R_full[np.ix_(obs_indices, obs_indices)]

        # Innovation: nu = y - H * x^-
        y_pred = H_k @ self.state
        nu = y_vec - y_pred

        # Innovation covariance: S = H * P^- * H^T + R
        S = H_k @ self.cov @ H_k.T + R_k
        S = 0.5 * (S + S.T)

        # Invert S with fallback regularization if ill-conditioned
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_reg = S + 1e-6 * np.eye(len(obs_indices))
            S_inv = np.linalg.inv(S_reg)

        # Kalman gain: K = P^- * H^T * S^-1
        K = self.cov @ H_k.T @ S_inv

        # Posterior state: x^+ = x^- + K * nu
        state_post = self.state + K @ nu

        # Joseph-form covariance update for numerical symmetry & positive definiteness
        I_KH = np.eye(3) - K @ H_k
        cov_post = I_KH @ self.cov @ I_KH.T + K @ R_k @ K.T
        cov_post = 0.5 * (cov_post + cov_post.T)

        # Normalized Innovation Squared (NIS): nu^T * S^-1 * nu
        nis = float(nu.T @ S_inv @ nu)

        prior_state = self.state.copy()
        prior_cov = self.cov.copy()

        self.state = state_post
        self.cov = cov_post

        return FilterStepResult(
            time_elapsed_hours=0.0,
            prior_state=prior_state,
            prior_cov=prior_cov,
            posterior_state=self.state.copy(),
            posterior_cov=self.cov.copy(),
            observed_analyte_indices=obs_indices,
            innovations=nu,
            innovation_cov=S,
            normalized_innovation_squared=nis,
        )

    def step(
        self,
        dt_hours: float,
        observations: Optional[Dict[int, float]] = None,
    ) -> Optional[FilterStepResult]:
        """Perform combined prediction over dt_hours followed by optional measurement update."""
        self.predict(dt_hours)
        if observations:
            return self.update(observations)
        return None
