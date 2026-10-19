"""Module 2: Patient-Level Deterministic Stratified Data Partitioning.

This module partitions the 2,500 bounded cohort patients into:
- 70% Training Split (1,750 patients)
- 15% Validation Split (375 patients)
- 15% Test Split (375 patients)

Stratification is performed on:
1. `has_admission` (preserving the exact ~9.0% outpatient-only proportion).
2. `gender` (preserving sex balance).
3. `creatinine_count_quartile` (ensuring balanced longitudinal observation density).

Patient-level isolation guarantees:
- All cohort-level preprocessing parameters (means, standard deviations, scalers)
  are fit strictly on Training patients.
- Zero temporal leakage: within every patient, time ordering is preserved.
- Train, validation, and test patient sets are mutually disjoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

DEFAULT_COHORT_PATIENTS_PATH = Path("data/processed/cohort_patients.csv")
DEFAULT_SPLITS_PATH = Path("configs/cohort_patient_splits.json")


def generate_stratified_patient_splits(
    patients_csv_path: Path = DEFAULT_COHORT_PATIENTS_PATH,
    output_json_path: Path = DEFAULT_SPLITS_PATH,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Deterministically partition bounded patients into stratified train/val/test splits."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    df = pd.read_csv(patients_csv_path)
    total_patients = len(df)
    if total_patients != 2500:
        raise ValueError(f"Expected 2,500 cohort patients, found {total_patients}")

    # Create stratification strata
    # 1. has_admission (bool)
    # 2. gender (F/M)
    # 3. creatinine quartile (0, 1, 2, 3)
    q_labels = [0, 1, 2, 3]
    df["creat_quartile"] = pd.qcut(
        df["usable_creatinine_count"], q=4, labels=q_labels, duplicates="drop"
    )
    df["stratum"] = (
        df["has_admission"].astype(str)
        + "_"
        + df["gender"].astype(str)
        + "_q"
        + df["creat_quartile"].astype(str)
    )

    rng = np.random.default_rng(seed)

    train_ids: List[int] = []
    val_ids: List[int] = []
    test_ids: List[int] = []

    # Allocate deterministically within each stratum
    strata = sorted(df["stratum"].unique())
    for s in strata:
        s_df = df[df["stratum"] == s].sort_values("subject_id")
        s_sids = s_df["subject_id"].to_numpy().copy()
        rng.shuffle(s_sids)

        n_s = len(s_sids)
        n_train = int(np.floor(n_s * train_ratio))
        n_val = int(np.floor(n_s * val_ratio))
        # Remaining go to test to ensure exact preservation
        n_test = n_s - n_train - n_val

        train_ids.extend(s_sids[:n_train].tolist())
        val_ids.extend(s_sids[n_train : n_train + n_val].tolist())
        test_ids.extend(s_sids[n_train + n_val :].tolist())

    # Adjust rounding differences to hit exact 1750 / 375 / 375 targets
    target_train = int(total_patients * train_ratio)  # 1750
    target_val = int(total_patients * val_ratio)      # 375
    target_test = total_patients - target_train - target_val  # 375

    # Rebalance if floor rounding caused small discrepancy
    while len(train_ids) < target_train and len(test_ids) > target_test:
        train_ids.append(test_ids.pop())
    while len(val_ids) < target_val and len(test_ids) > target_test:
        val_ids.append(test_ids.pop())
    while len(train_ids) > target_train:
        test_ids.append(train_ids.pop())
    while len(val_ids) > target_val:
        test_ids.append(val_ids.pop())

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()

    # Integrity assertions
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    assert len(train_ids) == target_train, f"Train count {len(train_ids)} != {target_train}"
    assert len(val_ids) == target_val, f"Val count {len(val_ids)} != {target_val}"
    assert len(test_ids) == target_test, f"Test count {len(test_ids)} != {target_test}"
    assert len(set_train & set_val) == 0, "Overlap between train and val"
    assert len(set_train & set_test) == 0, "Overlap between train and test"
    assert len(set_val & set_test) == 0, "Overlap between val and test"
    assert len(set_train | set_val | set_test) == total_patients, "Missing patients in split"

    # Compute balance diagnostics
    def _compute_strata_props(sub_df: pd.DataFrame) -> Dict[str, float]:
        adm_pct = float(sub_df["has_admission"].mean() * 100.0)
        fem_pct = float((sub_df["gender"] == "F").mean() * 100.0)
        med_cr = float(sub_df["usable_creatinine_count"].median())
        return {
            "count": len(sub_df),
            "pct_with_admissions": round(adm_pct, 2),
            "pct_female": round(fem_pct, 2),
            "median_creatinine_obs": med_cr,
        }

    train_df = df[df["subject_id"].isin(set_train)]
    val_df = df[df["subject_id"].isin(set_val)]
    test_df = df[df["subject_id"].isin(set_test)]

    summary = {
        "metadata": {
            "source": str(patients_csv_path),
            "total_patients": total_patients,
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "diagnostics": {
            "overall": _compute_strata_props(df),
            "train": _compute_strata_props(train_df),
            "val": _compute_strata_props(val_df),
            "test": _compute_strata_props(test_df),
        },
        "splits": {
            "train_subject_ids": train_ids,
            "val_subject_ids": val_ids,
            "test_subject_ids": test_ids,
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_patient_splits(splits_json_path: Path = DEFAULT_SPLITS_PATH) -> Dict[str, Set[int]]:
    """Load train/val/test subject_id sets from JSON split file."""
    if not splits_json_path.exists():
        generate_stratified_patient_splits(output_json_path=splits_json_path)

    with open(splits_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "train": set(data["splits"]["train_subject_ids"]),
        "val": set(data["splits"]["val_subject_ids"]),
        "test": set(data["splits"]["test_subject_ids"]),
    }


def get_patient_split(subject_id: int, splits_json_path: Path = DEFAULT_SPLITS_PATH) -> str:
    """Return 'train', 'val', or 'test' for a given subject_id."""
    splits = load_patient_splits(splits_json_path)
    if subject_id in splits["train"]:
        return "train"
    if subject_id in splits["val"]:
        return "val"
    if subject_id in splits["test"]:
        return "test"
    raise KeyError(f"subject_id {subject_id} not found in cohort patient splits")
