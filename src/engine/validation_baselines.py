"""Validation-only causal baseline selection for Module 2.

This module deliberately never loads the frozen test split.  It evaluates the
existing OU estimator and simple causal baselines on validation patients only.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.engine.data_partition import load_patient_splits
from src.engine.parameter_estimation import load_state_space_parameters
from src.engine.preprocessing import CohortScalerConfig, load_cohort_scalers, standardize_observation, unstandardize_observation
from src.engine.state_space import ANALYTE_INDEX_MAP, ContinuousDiscreteRenalFilter, StateSpaceParameters
from src.temporal.canonical_schema import create_canonical_event_frame
from src.temporal.conflict_arbitration import POLICY_A_STORETIME, arbitrate_canonical_frame, get_effective_events


CANDIDATES = ("locf", "ou", "prior_median", "trailing3_median")


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _rmse(actual: pd.Series, prediction: pd.Series) -> float:
    values = (actual.to_numpy(dtype=float) - prediction.to_numpy(dtype=float)) ** 2
    result = float(np.sqrt(np.mean(values)))
    return result if np.isfinite(result) else float("nan")


def evaluate_baseline_candidates_from_records(
    records_by_patient: Mapping[int, pd.DataFrame],
    scalers: CohortScalerConfig,
    params: StateSpaceParameters,
) -> Dict[str, Any]:
    """Score causal predictors from already effective, patient-indexed records.

    Each patient frame is sorted once.  At an epoch the filter predicts first;
    all rows at that timestamp see the same pre-assimilation history, and valid
    measurements are assimilated only after scoring the complete epoch.
    """
    exclusions: Counter[str] = Counter()
    rows: List[Dict[str, Any]] = []
    patient_count = 0

    for patient_id, raw_patient in records_by_patient.items():
        if raw_patient.empty:
            continue
        patient_count += 1
        patient = raw_patient.copy()
        patient["charttime"] = pd.to_datetime(patient["charttime"], errors="coerce")
        if "storetime" in patient:
            patient["storetime"] = pd.to_datetime(patient["storetime"], errors="coerce")
        else:
            patient["storetime"] = pd.NaT
        patient = patient.sort_values(["charttime", "storetime"], kind="mergesort")

        kf = ContinuousDiscreteRenalFilter(params=params)
        histories: Dict[str, List[float]] = {name: [] for name in ANALYTE_INDEX_MAP}
        last_time: Optional[pd.Timestamp] = None

        for charttime, epoch in patient.groupby("charttime", sort=False, dropna=False):
            if pd.isna(charttime):
                exclusions["invalid_charttime"] += len(epoch)
                continue
            timestamp = pd.Timestamp(charttime)
            dt_hours = 0.0 if last_time is None else max(0.0, (timestamp - last_time).total_seconds() / 3600.0)
            last_time = timestamp
            kf.predict(dt_hours)
            prior_state = kf.state.copy()
            observations: Dict[int, float] = {}
            pending_history: List[tuple[str, float]] = []

            for _, row in epoch.iterrows():
                analyte = str(row.get("analyte_name", ""))
                if analyte not in ANALYTE_INDEX_MAP:
                    exclusions["unknown_analyte"] += 1
                    continue
                target = row.get("valuenum_effective")
                if not _finite(target):
                    exclusions["nonfinite_target"] += 1
                    continue
                target = float(target)
                standardized = standardize_observation(target, analyte, scalers)
                if not _finite(standardized):
                    exclusions["invalid_observation_for_history"] += 1
                    continue

                history = histories[analyte]
                model_std = float(params.get_full_h_matrix()[ANALYTE_INDEX_MAP[analyte], :] @ prior_state)
                predictions = {
                    "locf": history[-1] if history else float("nan"),
                    "ou": unstandardize_observation(model_std, analyte, scalers),
                    "prior_median": float(np.median(history)) if len(history) >= 3 else float("nan"),
                    "trailing3_median": float(np.median(history[-3:])) if len(history) >= 3 else float("nan"),
                }
                valid = {name: _finite(value) for name, value in predictions.items()}
                for name, is_valid in valid.items():
                    if not is_valid:
                        exclusions[f"nonfinite_or_ineligible_prediction_{name}"] += 1
                if all(valid.values()):
                    rows.append({
                        "patient_id": int(patient_id),
                        "charttime": timestamp,
                        "analyte": analyte,
                        "actual": target,
                        # Retained only in the in-memory regression/debug frame;
                        # serialized reports omit scored_records.
                        "history_values_used": tuple(history),
                        **predictions,
                    })
                else:
                    exclusions["not_identical_scoring_target"] += 1

                # Current-epoch values are queued, never used by another row at
                # this timestamp; history is updated only after scoring the epoch.
                observations[ANALYTE_INDEX_MAP[analyte]] = float(standardized)
                pending_history.append((analyte, target))
            if observations:
                kf.update(observations)
            for analyte, target in pending_history:
                histories[analyte].append(target)

    scored = pd.DataFrame(rows)
    metrics: List[Dict[str, Any]] = []
    for analyte in ANALYTE_INDEX_MAP:
        group = scored[scored["analyte"] == analyte] if not scored.empty else pd.DataFrame()
        item: Dict[str, Any] = {"analyte": analyte, "target_count": int(len(group))}
        if group.empty:
            item.update({name: None for name in CANDIDATES})
            item.update({f"{name}_rmse_ratio_to_locf": None for name in CANDIDATES})
        else:
            locf_rmse = _rmse(group["actual"], group["locf"])
            for name in CANDIDATES:
                value = _rmse(group["actual"], group[name])
                item[name] = value if _finite(value) else None
                item[f"{name}_rmse_ratio_to_locf"] = value / locf_rmse if _finite(value) and _finite(locf_rmse) and locf_rmse > 0 else None
        metrics.append(item)

    candidate_means: Dict[str, Optional[float]] = {}
    for name in CANDIDATES:
        ratios = [m[f"{name}_rmse_ratio_to_locf"] for m in metrics]
        candidate_means[name] = float(np.mean(ratios)) if len(ratios) == len(ANALYTE_INDEX_MAP) and all(_finite(v) for v in ratios) else None

    valid_candidates = {name: score for name, score in candidate_means.items() if score is not None}
    # LOCF remains selected at exact ties, including its 1.0 reference score.
    selected: Optional[str] = None
    if valid_candidates:
        best_score = min(valid_candidates.values())
        tied = [name for name, score in valid_candidates.items() if np.isclose(score, best_score, rtol=0.0, atol=1e-12)]
        selected = tied[0] if len(tied) == 1 else "locf"

    return {
        "patient_count": patient_count,
        "target_count": int(len(scored)),
        "finite_prediction_counts": {name: int(scored[name].map(_finite).sum()) if not scored.empty else 0 for name in CANDIDATES},
        "exclusions": dict(sorted(exclusions.items())),
        "per_analyte": metrics,
        "candidate_mean_rmse_ratios": candidate_means,
        "selected_candidate": selected,
        "scored_records": scored,
    }


def run_validation_baseline_selection(
    cohort_labs_path: Path = Path("data/processed/cohort_renal_bounded.csv"),
    output_report_path: Path = Path("reports/generated/module2_validation_baseline_selection.json"),
    freeze_config_path: Path = Path("configs/module2_selected_baseline.json"),
    max_patients: Optional[int] = None,
) -> Dict[str, Any]:
    """Run selection exclusively on the frozen validation split; never touch test IDs."""
    started = time.perf_counter()
    splits = load_patient_splits()
    validation_ids = sorted(splits["val"])
    if max_patients is not None:
        validation_ids = validation_ids[:max_patients]
    if set(validation_ids) & set(splits["test"]):
        raise RuntimeError("Validation selection attempted to include frozen test patients")

    raw = pd.read_csv(cohort_labs_path)
    validation_raw = raw[raw["subject_id"].isin(validation_ids)].copy()
    canonical = create_canonical_event_frame(validation_raw)
    arbitrated, _ = arbitrate_canonical_frame(canonical, policy_id=POLICY_A_STORETIME)
    effective = get_effective_events(arbitrated)
    indexed = {int(pid): frame for pid, frame in effective.groupby("subject_id", sort=False)}

    result = evaluate_baseline_candidates_from_records(indexed, load_cohort_scalers(), load_state_space_parameters())
    result["elapsed_seconds"] = time.perf_counter() - started
    result["metadata"] = {
        "split": "validation",
        "validation_patient_ids_requested": len(validation_ids),
        "test_patient_ids_accessed": 0,
        "candidate_methods": list(CANDIDATES),
        "primary_scoring_rule": "equal-weight mean of six per-analyte RMSE ratios relative to LOCF",
        "tie_rule": "ties retain LOCF",
    }
    serializable = {key: value for key, value in result.items() if key != "scored_records"}
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, default=str)
    freeze_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(freeze_config_path, "w", encoding="utf-8") as handle:
        json.dump({"selection_split": "validation", "selected_candidate": result["selected_candidate"], "selection_status": "selected" if result["selected_candidate"] else "no_valid_candidate", "candidate_mean_rmse_ratios": result["candidate_mean_rmse_ratios"], "tie_rule": "ties retain LOCF"}, handle, indent=2)
    return serializable
