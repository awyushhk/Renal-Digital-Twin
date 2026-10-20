"""Module 3 feasibility and leakage-resistant real-cohort forecast evaluation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.engine.data_partition import load_patient_splits
from src.engine.forecasting import SUPPORTED_HORIZONS_HOURS, forecast_patient_biomarkers
from src.engine.patient_inference import PatientInferenceService
from src.engine.preprocessing import load_cohort_scalers, standardize_observation
from src.engine.state_space import ANALYTE_INDEX_MAP
from src.temporal.canonical_schema import create_canonical_event_frame
from src.temporal.conflict_arbitration import POLICY_A_STORETIME, arbitrate_canonical_frame, get_effective_events


WINDOW_FRACTION = 0.25
MIN_VALIDATION_PATIENTS = 25
MIN_VALIDATION_TARGETS = 100


def _preindex_effective_records(service: PatientInferenceService, patient_ids: Optional[Iterable[int]] = None) -> None:
    """Build the evaluator's patient index once; do not alter inference semantics."""
    service.preload()
    if service._raw_labs_df is None:
        raise RuntimeError("Cohort laboratory data were not loaded")
    raw_source = service._raw_labs_df
    if patient_ids is not None:
        raw_source = raw_source[raw_source["subject_id"].isin(set(patient_ids))]
    for patient_id, raw in raw_source.groupby("subject_id", sort=False):
        if int(patient_id) in service._patient_effective_cache:
            continue
        canonical = create_canonical_event_frame(raw)
        arbitrated, _ = arbitrate_canonical_frame(canonical, policy_id=POLICY_A_STORETIME)
        effective = get_effective_events(arbitrated).copy()
        effective["charttime"] = pd.to_datetime(effective["charttime"], errors="coerce")
        effective["storetime"] = pd.to_datetime(effective["storetime"], errors="coerce")
        service._patient_effective_cache[int(patient_id)] = effective.sort_values(["charttime", "storetime"], kind="mergesort").reset_index(drop=True)


def _split_service(patient_ids: Iterable[int]) -> PatientInferenceService:
    """Use an audit-local raw-data subset; source files and production cache stay untouched."""
    service = PatientInferenceService(scaler_config=load_cohort_scalers())
    service.preload()
    if service._raw_labs_df is None:
        raise RuntimeError("Cohort laboratory data were not loaded")
    service._raw_labs_df = service._raw_labs_df[service._raw_labs_df["subject_id"].isin(set(patient_ids))].copy()
    return service


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _time_column(mode: str) -> str:
    return "storetime" if mode.upper() == "OPERATIONAL" else "charttime"


def _valid_records(records: pd.DataFrame, mode: str) -> pd.DataFrame:
    time_col = _time_column(mode)
    data = records.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    if mode.upper() == "OPERATIONAL":
        data = data[data["delay_anomaly_flag"] != "NEGATIVE_REPORTING_DELAY"]
    data = data[data[time_col].notna() & data["valuenum_effective"].map(_finite)]
    return data.sort_values([time_col, "charttime", "storetime"], kind="mergesort")


def _candidate_pairs(records: pd.DataFrame, horizon: int, mode: str) -> Dict[str, tuple[pd.Timestamp, pd.Series]]:
    """Return latest one-patient origin/target pair per analyte within the fixed window."""
    time_col = _time_column(mode)
    result: Dict[str, tuple[pd.Timestamp, pd.Series]] = {}
    for analyte, group in records.groupby("analyte_name", sort=False):
        if analyte not in ANALYTE_INDEX_MAP:
            continue
        group = group.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        times = list(pd.to_datetime(group[time_col]))
        low_delta = pd.Timedelta(hours=horizon * (1 - WINDOW_FRACTION))
        high_delta = pd.Timedelta(hours=horizon * (1 + WINDOW_FRACTION))
        chosen: Optional[tuple[pd.Timestamp, pd.Series]] = None
        for index, origin_time in enumerate(times[:-1]):
            candidates = [target_index for target_index in range(index + 1, len(times)) if times[target_index] >= origin_time + low_delta]
            if candidates and times[candidates[0]] <= origin_time + high_delta:
                chosen = (pd.Timestamp(origin_time), group.iloc[candidates[0]])
        if chosen is not None:
            result[str(analyte)] = chosen
    return result


def _summary(values: pd.Series) -> Dict[str, Optional[float]]:
    if values.empty:
        return {"median": None, "p25": None, "p75": None, "p90": None, "max": None}
    return {key: round(float(values.quantile(q)), 3) for key, q in (("median", .5), ("p25", .25), ("p75", .75), ("p90", .9), ("max", 1.0))}


def run_followup_feasibility_audit(
    output_path: Path = Path("reports/generated/module3_followup_feasibility.json"),
) -> Dict[str, Any]:
    """Audit actual available event/availability follow-up without changing data."""
    # Feasibility uses the actual bounded processed rows and both timestamps.
    # It does not score forecasts, so it does not need to materialize the
    # slower per-patient arbitration cache used by the evaluator.
    raw = pd.read_csv("data/processed/cohort_renal_bounded.csv")
    raw = raw[raw["itemid"].isin([50912, 51006, 50971, 50983, 50882, 50931])].copy()
    names = {50912: "creatinine", 51006: "urea_nitrogen", 50971: "potassium", 50983: "sodium", 50882: "bicarbonate", 50931: "glucose"}
    raw["analyte_name"] = raw["itemid"].map(names)
    raw["valuenum_effective"] = raw["valuenum_harmonized"]
    raw["delay_anomaly_flag"] = "NONE"
    raw["charttime"] = pd.to_datetime(raw["charttime"], errors="coerce")
    raw["storetime"] = pd.to_datetime(raw["storetime"], errors="coerce")
    raw_groups = {int(patient_id): frame for patient_id, frame in raw.groupby("subject_id", sort=False)}
    splits = load_patient_splits()
    output: Dict[str, Any] = {
        "cohort_patient_count": 2500,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "horizons_hours": list(SUPPORTED_HORIZONS_HOURS),
        "target_window": "first future observation in [0.75*horizon, 1.25*horizon]",
        "modes": {},
        "deterioration_risk": {
            "status": "UNAVAILABLE_NO_VALIDATED_ENDPOINT",
            "reason": "Processed cohort interface has laboratory trajectories and limited demographics/admission metadata, not an independently adjudicated CKD-progression outcome or required outcome ascertainment data.",
        },
    }
    for mode in ("OPERATIONAL", "BIOLOGICAL"):
        time_col = _time_column(mode)
        valid = raw[raw[time_col].notna() & raw["valuenum_effective"].map(_finite)].copy()
        key = ["subject_id", "analyte_name"]
        counts = valid.groupby(key).size().rename("count")
        spans_frame = valid.groupby(key)[time_col].agg(["min", "max"])
        spans_frame["span"] = (spans_frame["max"] - spans_frame["min"]).dt.total_seconds() / 3600
        ordered = valid.sort_values(key + [time_col])
        ordered["gap"] = ordered.groupby(key)[time_col].diff().dt.total_seconds() / 3600
        support: Dict[str, Dict[str, Dict[str, int]]] = {}
        excluded = Counter()
        for horizon in SUPPORTED_HORIZONS_HOURS:
            origins = valid[key + [time_col]].rename(columns={time_col: "origin"}).copy()
            origins["lower"] = origins["origin"] + pd.to_timedelta(horizon * (1 - WINDOW_FRACTION), unit="h")
            targets = valid[key + [time_col]].rename(columns={time_col: "target"})
            merged = pd.merge_asof(
                origins.sort_values("lower"), targets.sort_values("target"),
                left_on="lower", right_on="target", by=key, direction="forward",
                tolerance=pd.to_timedelta(horizon * 2 * WINDOW_FRACTION, unit="h"),
            )
            matched = merged[merged["target"].notna()]
            latest = matched.sort_values("origin").groupby(key, as_index=False).tail(1)
            by_analyte = latest.groupby("analyte_name")["subject_id"].nunique().to_dict()
            support[str(horizon)] = {a: {"patients": int(by_analyte.get(a, 0)), "latest_pairs": int(by_analyte.get(a, 0))} for a in ANALYTE_INDEX_MAP}
            for analyte in ANALYTE_INDEX_MAP:
                excluded[f"no_{horizon}h_target_{analyte}"] = 2500 - int(by_analyte.get(analyte, 0))
        output["modes"][mode.lower()] = {
            "analyte_observation_count_per_patient": {a: _summary(counts.xs(a, level="analyte_name") if a in counts.index.get_level_values("analyte_name") else pd.Series(dtype=float)) for a in ANALYTE_INDEX_MAP},
            "followup_hours": {a: _summary(spans_frame.xs(a, level="analyte_name")["span"] if a in spans_frame.index.get_level_values("analyte_name") else pd.Series(dtype=float)) for a in ANALYTE_INDEX_MAP},
            "interobservation_gap_hours": {a: _summary(ordered.loc[ordered.analyte_name == a, "gap"].dropna()) for a in ANALYTE_INDEX_MAP},
            "horizon_support": support,
            "patient_exclusions": dict(excluded),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _population_baseline(analyte: str, service: PatientInferenceService) -> float:
    scale = service.scaler_config.analytes[analyte]
    return float(np.exp(scale.train_mean)) if scale.is_log_scale else float(scale.train_mean)


def _target_after_origin(records: pd.DataFrame, analyte: str, origin: pd.Timestamp, horizon: int, mode: str) -> Optional[pd.Series]:
    time_col = _time_column(mode)
    group = records[records["analyte_name"] == analyte].sort_values(time_col)
    if group.empty:
        return None
    lower = origin + pd.Timedelta(hours=horizon * (1 - WINDOW_FRACTION))
    upper = origin + pd.Timedelta(hours=horizon * (1 + WINDOW_FRACTION))
    candidates = group[(group[time_col] >= lower) & (group[time_col] <= upper)]
    return candidates.iloc[0] if not candidates.empty else None


def _evaluate_split(service: PatientInferenceService, patient_ids: Iterable[int], mode: str, horizons: Iterable[int]) -> tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    excluded = Counter()
    for patient_id in sorted(patient_ids):
        records = _valid_records(service.get_patient_records(patient_id), mode)
        if records.empty:
            excluded["no_valid_history"] += 1
            continue
        for horizon in horizons:
            pairs = _candidate_pairs(records, horizon, mode)
            if not pairs:
                excluded["no_supported_future_target"] += 1
                continue
            # One latest origin per patient/horizon keeps the real-cohort
            # summary patient-balanced rather than rewarding dense monitoring.
            origin = max(origin for origin, _ in pairs.values())
            forecasts = forecast_patient_biomarkers(service, patient_id, origin, [horizon], mode, ANALYTE_INDEX_MAP).forecasts
            by_analyte = {item.analyte_name: item for item in forecasts}
            for analyte in ANALYTE_INDEX_MAP:
                target = _target_after_origin(records, analyte, origin, horizon, mode)
                if target is None:
                    excluded[f"no_target_in_window_{analyte}"] += 1
                    continue
                target_value = float(target["valuenum_effective"])
                forecast = by_analyte[analyte]
                history = records[(records["analyte_name"] == analyte) & (records[_time_column(mode)] <= origin)]
                locf = float(history.iloc[-1]["valuenum_effective"]) if not history.empty else np.nan
                pop = _population_baseline(analyte, service)
                if forecast.status != "AVAILABLE" or not all(_finite(v) for v in (target_value, forecast.predicted_value, locf, pop)):
                    excluded["noncomparable_prediction_or_target"] += 1
                    continue
                elapsed = (pd.Timestamp(target[_time_column(mode)]) - origin).total_seconds() / 3600
                rows.append({
                    "patient_id": patient_id, "analyte": analyte, "horizon_hours": horizon,
                    "origin": origin.isoformat(), "target_time": pd.Timestamp(target[_time_column(mode)]).isoformat(),
                    "actual": target_value, "ou": forecast.predicted_value, "locf": locf, "population": pop,
                    "pi90": forecast.predictive_interval_90, "pi95": forecast.predictive_interval_95,
                    "actual_gap_hours": elapsed,
                })
    return pd.DataFrame(rows), dict(excluded)


def _metrics(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for (analyte, horizon), group in frame.groupby(["analyte", "horizon_hours"], sort=True):
        actual = group["actual"].to_numpy(float)
        def metric(name: str) -> tuple[float, float]:
            error = actual - group[name].to_numpy(float)
            return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))
        mae_ou, rmse_ou = metric("ou"); mae_locf, rmse_locf = metric("locf"); mae_pop, rmse_pop = metric("population")
        pi90 = np.array(group["pi90"].tolist(), float); pi95 = np.array(group["pi95"].tolist(), float)
        covered90 = (actual >= pi90[:, 0]) & (actual <= pi90[:, 1]); covered95 = (actual >= pi95[:, 0]) & (actual <= pi95[:, 1])
        width95 = pi95[:, 1] - pi95[:, 0]
        winkler = width95 + np.where(actual < pi95[:, 0], 40 * (pi95[:, 0] - actual), 0) + np.where(actual > pi95[:, 1], 40 * (actual - pi95[:, 1]), 0)
        result.append({
            "analyte": analyte, "horizon_hours": int(horizon), "target_count": int(len(group)), "unique_patients": int(group.patient_id.nunique()),
            "mae_ou": round(mae_ou, 4), "rmse_ou": round(rmse_ou, 4), "mae_locf": round(mae_locf, 4), "rmse_locf": round(rmse_locf, 4), "mae_population": round(mae_pop, 4), "rmse_population": round(rmse_pop, 4),
            "rmse_ratio_ou_to_locf": round(rmse_ou / rmse_locf, 4) if rmse_locf else None,
            "coverage_90_pct": round(float(covered90.mean() * 100), 2), "coverage_95_pct": round(float(covered95.mean() * 100), 2),
            "mean_interval_width_95": round(float(width95.mean()), 4), "winkler_score_95": round(float(winkler.mean()), 4),
            "patient_level_median_absolute_error": round(float(group.assign(abs_error=np.abs(actual - group.ou)).groupby("patient_id").abs_error.mean().median()), 4),
        })
    return result


def run_forecasting_evaluation(
    output_path: Path = Path("reports/generated/module3_forecasting_evaluation.json"),
    mode: str = "OPERATIONAL",
) -> Dict[str, Any]:
    """Validate fixed horizons, then evaluate that unchanged set once on frozen test patients."""
    splits = load_patient_splits()
    validation_service = _split_service(splits["val"])
    validation, validation_exclusions = _evaluate_split(validation_service, splits["val"], mode, SUPPORTED_HORIZONS_HOURS)
    validation_metrics = _metrics(validation) if not validation.empty else []
    selected = sorted({m["horizon_hours"] for m in validation_metrics if m["analyte"] == "creatinine" and m["target_count"] >= MIN_VALIDATION_TARGETS and m["unique_patients"] >= MIN_VALIDATION_PATIENTS})
    test_service = _split_service(splits["test"])
    test, test_exclusions = _evaluate_split(test_service, splits["test"], mode, selected)
    test_metrics = _metrics(test) if not test.empty else []
    creatinine = [m for m in test_metrics if m["analyte"] == "creatinine"]
    acceptance = "PASSED" if creatinine and any(m["rmse_ou"] < m["rmse_locf"] for m in creatinine) else "FAILED"
    report = {
        "protocol": {"mode": mode.upper(), "horizons_prespecified": list(SUPPORTED_HORIZONS_HOURS), "selected_on_validation": selected, "target_definition": "latest shared origin per patient/horizon with first future analyte target in ±25% window", "common_mask": "finite actual plus finite OU, LOCF, and training-fitted population predictions"},
        "validation": {"patient_count": len(splits["val"]), "exclusions": validation_exclusions, "metrics": validation_metrics},
        "frozen_test": {"patient_count": len(splits["test"]), "exclusions": test_exclusions, "metrics": test_metrics},
        "acceptance": {"creatinine_ou_beats_locf_at_any_selected_horizon": acceptance, "criterion": "OU RMSE lower than LOCF for creatinine at one validation-supported horizon; this is not a clinical-validity criterion."},
        "deterioration_risk_status": "UNAVAILABLE_NO_VALIDATED_ENDPOINT",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
