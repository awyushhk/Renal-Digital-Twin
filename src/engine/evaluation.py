"""Module 2: Rigorous Scientific Evaluation and Benchmark Audit Engine.

Evaluates continuous-discrete latent state engine against LOCF and linear baselines
on held-out test patients from the frozen data partition.

Strict Scientific Constraints:
1. Exact Target Comparability: All metrics evaluate identical targets at identical horizons.
2. Proper Predictive Uncertainty: Predictive intervals use total observation predictive
   variance S_k = H_k P_k^- H_k^T + R_k, contrasting with latent-only intervals H P H^T.
3. Monotonic Inverse Transformation: Log-space intervals map to original analyte units
   via exponential transformation without synthetic distortion.
4. Stratification: Evaluates performance across short (<=24h), medium (24-168h),
   and long (>168h) observation gaps to audit OU mean reversion.
5. Honest Acceptance Reporting: Explicitly records failed acceptance gates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.engine.data_partition import load_patient_splits
from src.engine.parameter_estimation import load_state_space_parameters
from src.engine.patient_inference import PatientInferenceService
from src.engine.preprocessing import (
    load_cohort_scalers,
    standardize_observation,
    unstandardize_observation,
)
from src.engine.state_space import (
    ANALYTE_INDEX_MAP,
    ContinuousDiscreteRenalFilter,
    StateSpaceParameters,
)


@dataclass
class AnalyteMetrics:
    analyte_name: str
    unit: str
    total_candidate_points: int
    model_eligible_points: int
    excluded_from_common_mask_count: int
    evaluated_targets_count: int
    rmse_model: float
    rmse_locf: float
    rmse_linear: float
    mae_model: float
    mae_locf: float
    mae_linear: float
    coverage_95_predictive_pct: float
    coverage_90_predictive_pct: float
    coverage_95_latent_only_pct: float
    mean_interval_width_95: float
    winkler_score_95: float
    model_beats_locf_rmse: bool


@dataclass
class StratifiedGapMetrics:
    stratum: str
    analyte_name: str
    sample_count: int
    rmse_model: float
    rmse_locf: float
    mae_model: float
    mae_locf: float
    rmse_difference: float
    model_beats_locf: bool


def run_scientific_evaluation(
    test_subject_ids: Optional[List[int]] = None,
    output_json_path: Path = Path("reports/generated/module2_scientific_evaluation_audit.json"),
) -> Dict[str, Any]:
    """Execute rigorous benchmark evaluation across held-out test patients."""
    scalers = load_cohort_scalers()
    params = load_state_space_parameters()
    service = PatientInferenceService(scaler_config=scalers, state_space_params=params)
    service.preload()

    if test_subject_ids is None:
        # Acceptance metrics must cover the frozen test partition, not an
        # illustrative subset of it.  Candidate selection remains outside this
        # evaluator and must use training/validation data only.
        test_subject_ids = sorted(load_patient_splits()["test"])

    candidate_points = 0
    candidate_points_by_analyte: Dict[str, int] = {name: 0 for name in ANALYTE_INDEX_MAP}
    records_all: List[Dict[str, Any]] = []

    for pid in test_subject_ids:
        recs = service.get_patient_records(pid)
        if recs.empty:
            continue

        kf = ContinuousDiscreteRenalFilter(params=params)
        last_time: Optional[pd.Timestamp] = None
        locf_dict: Dict[str, float] = {}
        history_dict: Dict[str, List[Tuple[pd.Timestamp, float]]] = {a: [] for a in ANALYTE_INDEX_MAP}

        for ctime, epoch_df in recs.groupby("charttime", sort=True):
            c_ts = pd.to_datetime(ctime)
            dt_h = 0.0 if last_time is None else max(0.0, (c_ts - last_time).total_seconds() / 3600.0)
            last_time = c_ts

            kf.predict(dt_h)
            prior_state = kf.state.copy()
            prior_cov = kf.cov.copy()

            H_full = params.get_full_h_matrix()
            R_full = params.get_full_r_matrix()

            obs_dict: Dict[int, float] = {}
            for _, row in epoch_df.iterrows():
                analyte = str(row["analyte_name"])
                if analyte not in ANALYTE_INDEX_MAP:
                    continue
                candidate_points += 1
                candidate_points_by_analyte[analyte] += 1

                val = float(row["valuenum_effective"])
                a_idx = ANALYTE_INDEX_MAP[analyte]
                std_val = standardize_observation(val, analyte, scalers)
                if pd.isna(std_val):
                    continue

                h_analyte = history_dict[analyte]
                dt_analyte_hours = (c_ts - h_analyte[-1][0]).total_seconds() / 3600.0 if h_analyte else np.nan

                # Model prediction in standardized space
                pred_std = float(H_full[a_idx, :] @ prior_state)
                pred_val = unstandardize_observation(pred_std, analyte, scalers)

                # Predictive variance: S_k = H P H^T + R_k
                pred_var_std = float(H_full[a_idx, :] @ prior_cov @ H_full[a_idx, :].T + R_full[a_idx, a_idx])
                pred_sd_std = np.sqrt(max(1e-6, pred_var_std))

                # Latent-state projected variance (excluding measurement noise R_k)
                latent_var_std = float(H_full[a_idx, :] @ prior_cov @ H_full[a_idx, :].T)
                latent_sd_std = np.sqrt(max(1e-6, latent_var_std))

                # 95% predictive interval (with R_k)
                pi95_lower = unstandardize_observation(pred_std - 1.95996 * pred_sd_std, analyte, scalers)
                pi95_upper = unstandardize_observation(pred_std + 1.95996 * pred_sd_std, analyte, scalers)

                # 90% predictive interval (with R_k)
                pi90_lower = unstandardize_observation(pred_std - 1.64485 * pred_sd_std, analyte, scalers)
                pi90_upper = unstandardize_observation(pred_std + 1.64485 * pred_sd_std, analyte, scalers)

                # 95% latent credible interval (WITHOUT R_k)
                ci95_latent_lower = unstandardize_observation(pred_std - 1.95996 * latent_sd_std, analyte, scalers)
                ci95_latent_upper = unstandardize_observation(pred_std + 1.95996 * latent_sd_std, analyte, scalers)

                # LOCF prediction
                locf_val = locf_dict.get(analyte, np.nan)

                # Linear baseline prediction
                if len(h_analyte) >= 2:
                    t0, v0 = h_analyte[-2]
                    t1, v1 = h_analyte[-1]
                    dt_prev = (t1 - t0).total_seconds() / 3600.0
                    dt_curr = (c_ts - t1).total_seconds() / 3600.0
                    if dt_prev > 0 and dt_curr < 720.0:
                        linear_val = v1 + ((v1 - v0) / dt_prev) * dt_curr
                    else:
                        linear_val = v1
                elif len(h_analyte) == 1:
                    linear_val = h_analyte[-1][1]
                else:
                    linear_val = np.nan

                in_pi95 = (pi95_lower <= val <= pi95_upper) if (pd.notna(pi95_lower) and pd.notna(pi95_upper)) else False
                in_pi90 = (pi90_lower <= val <= pi90_upper) if (pd.notna(pi90_lower) and pd.notna(pi90_upper)) else False
                in_ci95_latent = (ci95_latent_lower <= val <= ci95_latent_upper) if (pd.notna(ci95_latent_lower) and pd.notna(ci95_latent_upper)) else False

                # Winkler interval score (95%, alpha = 0.05)
                alpha_95 = 0.05
                width_95 = pi95_upper - pi95_lower
                pen_l = (2.0 / alpha_95) * (pi95_lower - val) if val < pi95_lower else 0.0
                pen_u = (2.0 / alpha_95) * (val - pi95_upper) if val > pi95_upper else 0.0
                winkler_95 = width_95 + pen_l + pen_u

                records_all.append({
                    "pid": pid,
                    "charttime": c_ts.isoformat(),
                    "analyte": analyte,
                    "actual": val,
                    "pred_model": pred_val,
                    "pred_locf": locf_val,
                    "pred_linear": linear_val,
                    "pred_sd_std": pred_sd_std,
                    "latent_sd_std": latent_sd_std,
                    "in_pi95": in_pi95,
                    "in_pi90": in_pi90,
                    "in_ci95_latent": in_ci95_latent,
                    "width_95": width_95,
                    "winkler_95": winkler_95,
                    "dt_analyte_hours": dt_analyte_hours,
                })

                obs_dict[a_idx] = float(std_val)
                locf_dict[analyte] = val
                history_dict[analyte].append((c_ts, val))

            if obs_dict:
                kf.update(obs_dict)

    df = pd.DataFrame(records_all)
    # Filter to strictly comparable targets where all 3 models made a valid prediction
    comp = df.dropna(subset=["actual", "pred_locf", "pred_model", "pred_linear"]).copy()
    comparable_cols = ["actual", "pred_locf", "pred_model", "pred_linear"]
    comp = comp[np.isfinite(comp[comparable_cols]).all(axis=1)].copy()

    # Per-analyte metrics on the one common mask: an observed target with a
    # finite causal prediction from OU, LOCF, and linear baselines.
    analyte_metrics_list: List[AnalyteMetrics] = []
    per_analyte_masks: List[Dict[str, Any]] = []
    for analyte, grp in comp.groupby("analyte"):
        candidate_count = candidate_points_by_analyte[analyte]
        model_eligible_count = len(df[df["analyte"] == analyte])
        eval_count = len(grp)
        dropped_count = model_eligible_count - eval_count
        per_analyte_masks.append({
            "analyte_name": analyte,
            "raw_candidate_count": int(candidate_count),
            "model_ineligible_count": int(candidate_count - model_eligible_count),
            "model_eligible_candidate_count": int(model_eligible_count),
            "strictly_comparable_target_count": int(eval_count),
            "excluded_from_common_mask_count": int(dropped_count),
            "common_mask": "finite actual plus finite causal OU, LOCF, and linear predictions",
        })

        rmse_m = float(np.sqrt(np.mean((grp["actual"] - grp["pred_model"]) ** 2)))
        rmse_l = float(np.sqrt(np.mean((grp["actual"] - grp["pred_locf"]) ** 2)))
        rmse_lin = float(np.sqrt(np.mean((grp["actual"] - grp["pred_linear"]) ** 2)))

        mae_m = float(np.mean(np.abs(grp["actual"] - grp["pred_model"])))
        mae_l = float(np.mean(np.abs(grp["actual"] - grp["pred_locf"])))
        mae_lin = float(np.mean(np.abs(grp["actual"] - grp["pred_linear"])))

        cov95 = float(grp["in_pi95"].mean() * 100.0)
        cov90 = float(grp["in_pi90"].mean() * 100.0)
        cov_latent95 = float(grp["in_ci95_latent"].mean() * 100.0)
        mean_width = float(grp["width_95"].mean())
        mean_winkler = float(grp["winkler_95"].mean())

        unit = scalers.analytes[analyte].unit
        beats_locf = rmse_m < rmse_l

        analyte_metrics_list.append(
            AnalyteMetrics(
                analyte_name=analyte,
                unit=unit,
                total_candidate_points=candidate_count,
                model_eligible_points=model_eligible_count,
                excluded_from_common_mask_count=dropped_count,
                evaluated_targets_count=eval_count,
                rmse_model=round(rmse_m, 4),
                rmse_locf=round(rmse_l, 4),
                rmse_linear=round(rmse_lin, 4),
                mae_model=round(mae_m, 4),
                mae_locf=round(mae_l, 4),
                mae_linear=round(mae_lin, 4),
                coverage_95_predictive_pct=round(cov95, 2),
                coverage_90_predictive_pct=round(cov90, 2),
                coverage_95_latent_only_pct=round(cov_latent95, 2),
                mean_interval_width_95=round(mean_width, 4),
                winkler_score_95=round(mean_winkler, 4),
                model_beats_locf_rmse=beats_locf,
            )
        )

    # Gap stratification
    def get_stratum(dt: float) -> str:
        if dt <= 24.0:
            return "Short (<=24h)"
        elif dt <= 168.0:
            return "Medium (24h-168h)"
        else:
            return "Long (>168h)"

    comp["stratum"] = comp["dt_analyte_hours"].apply(get_stratum)
    stratified_metrics_list: List[StratifiedGapMetrics] = []

    for stratum in ["Short (<=24h)", "Medium (24h-168h)", "Long (>168h)"]:
        sub = comp[comp["stratum"] == stratum]
        for analyte in sorted(ANALYTE_INDEX_MAP.keys()):
            grp = sub[sub["analyte"] == analyte]
            if len(grp) == 0:
                continue
            rmse_m = float(np.sqrt(np.mean((grp["actual"] - grp["pred_model"]) ** 2)))
            rmse_l = float(np.sqrt(np.mean((grp["actual"] - grp["pred_locf"]) ** 2)))
            mae_m = float(np.mean(np.abs(grp["actual"] - grp["pred_model"])))
            mae_l = float(np.mean(np.abs(grp["actual"] - grp["pred_locf"])))
            diff = rmse_m - rmse_l

            stratified_metrics_list.append(
                StratifiedGapMetrics(
                    stratum=stratum,
                    analyte_name=analyte,
                    sample_count=len(grp),
                    rmse_model=round(rmse_m, 4),
                    rmse_locf=round(rmse_l, 4),
                    mae_model=round(mae_m, 4),
                    mae_locf=round(mae_l, 4),
                    rmse_difference=round(diff, 4),
                    model_beats_locf=diff < 0,
                )
            )

    winning_analytes = [m.analyte_name for m in analyte_metrics_list if m.model_beats_locf_rmse]
    losing_analytes = [m.analyte_name for m in analyte_metrics_list if not m.model_beats_locf_rmse]
    H_full = params.get_full_h_matrix()
    R_full = params.get_full_r_matrix()
    information = H_full.T @ np.linalg.inv(R_full) @ H_full
    information_eigenvalues = np.linalg.eigvalsh(information)
    observability_condition = float(np.linalg.cond(H_full))

    # Summary audit record
    output_audit = {
        "metadata": {
            "evaluation_title": "Module 2 Scientific Benchmark & Uncertainty Calibration Audit",
            "evaluated_test_patients": test_subject_ids,
            "evaluated_test_patient_count": len(test_subject_ids),
            "total_candidate_observations": candidate_points,
            "total_model_eligible_candidates": int(len(df)),
            "total_model_ineligible_candidates": int(candidate_points - len(df)),
            "total_excluded_from_common_mask": int(len(df) - len(comp)),
            "total_strictly_comparable_targets": len(comp),
            "reconciliation_explanation": (
                f"The {candidate_points} candidate observations include {len(df)} targets with a finite OU prediction; "
                f"{candidate_points - len(df)} are model-ineligible after preprocessing. The strict common mask excludes "
                f"{len(df) - len(comp)} model-eligible targets because at least one causal baseline "
                f"(LOCF or linear) lacks a finite antecedent prediction. The remaining {len(comp)} targets are the only "
                "ones used for all RMSE, MAE, coverage, and interval-score comparisons."
            ),
        },
        "evaluation_masks_per_analyte": per_analyte_masks,
        "acceptance_gates": {
            "all_analyte_superiority_vs_locf": {
                "verdict": "PASSED" if len(winning_analytes) == len(ANALYTE_INDEX_MAP) else "FAILED",
                "reason": (
                    "OU has lower RMSE than LOCF for all six analytes."
                    if len(winning_analytes) == len(ANALYTE_INDEX_MAP)
                    else f"OU beats LOCF only for: {', '.join(winning_analytes) or 'none'}; "
                    f"LOCF is lower for: {', '.join(losing_analytes)}."
                ),
            },
            "uncertainty_calibration_gate": {
                "verdict": "NOT_ESTABLISHED",
                "reason": (
                    "No prespecified coverage tolerance is encoded in the project. Per-analyte 90%/95% observation "
                    "predictive coverage and 95% Winkler scores are reported below; calibration is not accepted as a "
                    "single aggregate claim."
                ),
            },
            "practical_identifiability_gate": {
                "verdict": "PASSED" if np.linalg.matrix_rank(H_full) == 3 and np.all(information_eigenvalues > 0) else "FAILED",
                "reason": (
                    f"Measurement loading matrix rank={np.linalg.matrix_rank(H_full)}, condition number={observability_condition:.2f}; "
                    f"minimum eigenvalue of H^T R^-1 H={float(information_eigenvalues.min()):.4f}."
                ),
            },
        },
        "per_analyte_metrics": [asdict(m) for m in analyte_metrics_list],
        "stratified_by_gap": [asdict(s) for s in stratified_metrics_list],
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_audit, f, indent=2)

    return output_audit


if __name__ == "__main__":
    res = run_scientific_evaluation()
    print("Scientific evaluation audit completed successfully.")
    print("Reconciliation: Candidates =", res["metadata"]["total_candidate_observations"],
          "Excluded from common mask =", res["metadata"]["total_excluded_from_common_mask"],
          "Comparable =", res["metadata"]["total_strictly_comparable_targets"])
