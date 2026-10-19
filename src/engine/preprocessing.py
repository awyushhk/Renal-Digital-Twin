"""Module 2: Training-Only Preprocessing and Standardization Pipeline.

Fits scaling parameters (means, standard deviations) strictly on the Training
patient split (1,750 patients). Test and validation patients are standardized
exclusively using the frozen training constants, preventing evaluation leakage.

Analytes:
- Creatinine (50912): Log-space standardized
- Blood Urea Nitrogen (51006): Log-space standardized
- Potassium (50971): Linear standardized
- Sodium (50983): Linear standardized
- Bicarbonate (50882): Linear standardized
- Glucose (50931): Log-space standardized

Source terminology note:
Standardized values reflect the empirical Cohort Reference Distribution
(derived from MIMIC-IV 3.1 hospital cohort), not a healthy physiological baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.engine.data_partition import load_patient_splits
from src.temporal.canonical_schema import TARGET_ITEM_IDS, ITEM_TO_ANALYTE

DEFAULT_COHORT_LABS_PATH = Path("data/processed/cohort_renal_bounded.csv")
DEFAULT_SCALER_CONFIG_PATH = Path("configs/module2_scaler_parameters.json")

# Analytes modeled in log-space due to right-skewed positive concentrations
LOG_TRANSFORM_ANALYTES = {"creatinine", "urea_nitrogen", "glucose"}
LINEAR_ANALYTES = {"potassium", "sodium", "bicarbonate"}


@dataclass
class AnalyteScaleParam:
    """Scaling parameters for an individual analyte."""
    itemid: int
    analyte_name: str
    is_log_scale: bool
    train_mean: float
    train_std: float
    unit: str


@dataclass
class CohortScalerConfig:
    """Complete collection of scaling parameters fit on training split."""
    training_patient_count: int
    training_observation_count: int
    analytes: Dict[str, AnalyteScaleParam]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_patient_count": self.training_patient_count,
            "training_observation_count": self.training_observation_count,
            "analytes": {
                name: asdict(param) for name, param in self.analytes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CohortScalerConfig:
        analytes = {
            name: AnalyteScaleParam(**p)
            for name, p in data["analytes"].items()
        }
        return cls(
            training_patient_count=data["training_patient_count"],
            training_observation_count=data["training_observation_count"],
            analytes=analytes,
        )


def fit_cohort_scalers(
    cohort_labs_path: Path = DEFAULT_COHORT_LABS_PATH,
    splits_json_path: Optional[Path] = None,
    output_config_path: Path = DEFAULT_SCALER_CONFIG_PATH,
) -> CohortScalerConfig:
    """Compute empirical scaling parameters strictly on training patients."""
    if splits_json_path is not None:
        splits = load_patient_splits(splits_json_path)
    else:
        splits = load_patient_splits()

    train_sids = splits["train"]

    # Read labs CSV and filter strictly to training patients
    df = pd.read_csv(cohort_labs_path)
    train_df = df[df["subject_id"].isin(train_sids)].copy()

    # Retain non-null valuenum rows for scaling
    train_df = train_df[train_df["valuenum"].notna() & (train_df["valuenum"] > 0)]

    scalers: Dict[str, AnalyteScaleParam] = {}

    for analyte_name, itemid in TARGET_ITEM_IDS.items():
        subset = train_df[train_df["itemid"] == itemid]
        if subset.empty:
            raise ValueError(f"No observations found in training set for {analyte_name} (item {itemid})")

        vals = subset["valuenum"].to_numpy(dtype=float)
        unit = str(subset["valueuom"].iloc[0]) if "valueuom" in subset.columns else ""

        if analyte_name in LOG_TRANSFORM_ANALYTES:
            log_vals = np.log(vals)
            mu = float(np.mean(log_vals))
            sigma = float(np.std(log_vals, ddof=1))
            is_log = True
        else:
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1))
            is_log = False

        scalers[analyte_name] = AnalyteScaleParam(
            itemid=itemid,
            analyte_name=analyte_name,
            is_log_scale=is_log,
            train_mean=round(mu, 6),
            train_std=round(sigma, 6),
            unit=unit,
        )

    config = CohortScalerConfig(
        training_patient_count=len(train_sids),
        training_observation_count=len(train_df),
        analytes=scalers,
    )

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    return config


def load_cohort_scalers(config_path: Path = DEFAULT_SCALER_CONFIG_PATH) -> CohortScalerConfig:
    """Load previously fitted scaling parameters."""
    if not config_path.exists():
        return fit_cohort_scalers(output_config_path=config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CohortScalerConfig.from_dict(data)


def standardize_observation(
    valuenum: float,
    analyte_name: str,
    scaler_config: CohortScalerConfig,
) -> float:
    """Transform a raw measurement into standardized coordinate space using training constants."""
    if pd.isna(valuenum) or not np.isfinite(valuenum) or valuenum <= 0:
        return np.nan

    if analyte_name not in scaler_config.analytes:
        raise KeyError(f"Analyte {analyte_name} not found in scaler config")

    p = scaler_config.analytes[analyte_name]
    val = float(valuenum)

    if not np.isfinite(p.train_mean) or not np.isfinite(p.train_std) or p.train_std <= 0:
        raise ValueError(f"Invalid scaler parameters for {analyte_name}")

    if p.is_log_scale:
        log_val = np.log(val)
        result = (log_val - p.train_mean) / p.train_std
    else:
        result = (val - p.train_mean) / p.train_std
    return float(result) if np.isfinite(result) else np.nan


def unstandardize_observation(
    std_val: float,
    analyte_name: str,
    scaler_config: CohortScalerConfig,
) -> float:
    """Convert standardized value back to natural unit scale."""
    if pd.isna(std_val) or not np.isfinite(std_val):
        return np.nan

    p = scaler_config.analytes[analyte_name]
    if not np.isfinite(p.train_mean) or not np.isfinite(p.train_std) or p.train_std <= 0:
        raise ValueError(f"Invalid scaler parameters for {analyte_name}")

    if p.is_log_scale:
        log_val = (std_val * p.train_std) + p.train_mean
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.exp(log_val))
    else:
        result = float((std_val * p.train_std) + p.train_mean)
    return result if np.isfinite(result) else np.nan
