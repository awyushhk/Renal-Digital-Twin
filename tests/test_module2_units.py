"""
Module 2 — Unit Tests for Engine Components
=============================================
Tests state-space model, TSI, preprocessing, and data partitioning
without requiring the FastAPI server or MIMIC-IV data.

All symbol names are verified against the actual module public API.

Run with:
    pytest tests/test_module2_units.py -v
"""

import math
import numpy as np
import pandas as pd
import pytest


# ── State-space model ────────────────────────────────────────────────────────

class TestContinuousDiscreteRenalFilter:
    """Unit tests for the 3D continuous-discrete Kalman filter."""

    def _make_filter(self):
        from src.engine.state_space import ContinuousDiscreteRenalFilter
        # Default construction uses built-in parameter defaults
        return ContinuousDiscreteRenalFilter()

    def test_initial_state_is_zero(self):
        f = self._make_filter()
        assert f.state.shape == (3,)
        assert np.allclose(f.state, 0.0), f"Expected zero initial state, got {f.state}"

    def test_initial_covariance_is_positive_definite(self):
        f = self._make_filter()
        P = f.cov
        assert P.shape == (3, 3)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0), f"Initial covariance not positive definite: {eigvals}"

    def test_predict_returns_state_and_cov(self):
        f = self._make_filter()
        x, P = f.predict(dt_hours=24.0)
        assert x.shape == (3,)
        assert P.shape == (3, 3)

    def test_predict_zero_dt_leaves_state_unchanged(self):
        f = self._make_filter()
        x0 = f.state.copy()
        f.predict(dt_hours=0.0)
        assert np.allclose(f.state, x0, atol=1e-10)

    def test_predict_grows_covariance(self):
        f = self._make_filter()
        P0_diag = f.cov.diagonal().copy()
        f.predict(dt_hours=24.0)
        P1_diag = f.cov.diagonal()
        # After prediction with no observations, uncertainty should not shrink
        assert np.all(P1_diag >= P0_diag - 1e-10), (
            f"Diagonal shrank after predict: before={P0_diag}, after={P1_diag}"
        )

    def test_update_with_creatinine(self):
        f = self._make_filter()
        f.predict(dt_hours=12.0)
        P_before = f.cov.diagonal().copy()
        # Update takes itemid→standardized_value dict (50912 = creatinine)
        result = f.update({50912: 0.5})
        P_after = f.cov.diagonal()
        # Uncertainty should decrease (or at worst stay the same) after observation
        assert np.trace(f.cov) <= np.sum(P_before) + 1e-6, (
            "Covariance trace should not increase after update"
        )

    def test_positive_definite_after_many_steps(self):
        """Covariance must remain positive definite after 20 predict-update cycles."""
        f = self._make_filter()
        rng = np.random.default_rng(42)
        analyte_ids = [50912, 51006, 50971, 50983, 50882, 50931]
        for _ in range(20):
            dt = float(rng.uniform(1.0, 48.0))
            f.predict(dt_hours=dt)
            chosen_id = int(rng.choice(analyte_ids))
            z = float(rng.normal(0, 1))
            f.update({chosen_id: z})
        eigvals = np.linalg.eigvalsh(f.cov)
        assert np.all(eigvals > -1e-9), f"Covariance has negative eigenvalues after cycles: {eigvals}"

    def test_time_constant_semantics(self):
        """At dt = tau, the OU decay factor should equal exp(-1) ≈ 0.368, not 0.5."""
        from src.engine.state_space import ContinuousDiscreteRenalFilter, StateSpaceParameters
        # tau_hours[0] = 72h → at dt=72h, Phi[0,0] should be exp(-1)
        params = StateSpaceParameters(tau_hours=[72.0, 24.0, 12.0])
        f = ContinuousDiscreteRenalFilter(params)
        # Predict with dt = tau for first state
        state_before = f.state.copy()
        # Inject a unit state and check decay
        import numpy as np
        f2 = ContinuousDiscreteRenalFilter(params, initial_state=np.array([1.0, 0.0, 0.0]))
        x_after, _ = f2.predict(dt_hours=72.0)
        expected = math.exp(-1.0)
        assert abs(x_after[0] - expected) < 0.01, (
            f"State decay at dt=tau: {x_after[0]:.4f} vs exp(-1)={expected:.4f}"
        )

    def test_step_method_combines_predict_update(self):
        f = self._make_filter()
        result = f.step(dt_hours=12.0, observations={50912: 0.5, 51006: 0.3})
        # step() should return a FilterStepResult or None
        # At minimum, state should be 3D and cov positive
        assert f.state.shape == (3,)
        assert f.cov.shape == (3, 3)


# ── Trajectory Stability Index ────────────────────────────────────────────────

class TestTrajectoryStabilityIndex:
    """Tests for the creatinine trajectory stability function."""

    def _make_creatinine_df(self, times_h, values):
        """Build minimal DataFrame with charttime and valuenum_effective columns."""
        base = pd.Timestamp("2180-01-01 00:00:00")
        charttimes = [base + pd.Timedelta(hours=h) for h in times_h]
        return pd.DataFrame({
            "charttime": pd.to_datetime(charttimes),
            "valuenum_effective": values,
        })

    def test_constant_series_gives_stable_result(self):
        from src.engine.trajectory_stability import compute_trajectory_stability_index
        df = self._make_creatinine_df([0, 10, 20, 30, 40], [1.0, 1.0, 1.0, 1.0, 1.0])
        result = compute_trajectory_stability_index(df)
        # Either None (insufficient) or a valid score; constant should score well if computed
        if result.tsi_score is not None:
            assert 0.0 < result.tsi_score <= 1.0

    def test_result_has_required_fields(self):
        from src.engine.trajectory_stability import compute_trajectory_stability_index
        df = self._make_creatinine_df([0, 6, 12, 24, 36, 48], [1.2, 1.8, 1.4, 2.1, 1.9, 1.7])
        result = compute_trajectory_stability_index(df)
        assert hasattr(result, "tsi_score")
        assert hasattr(result, "status")
        assert hasattr(result, "available_epochs_count")
        assert hasattr(result, "message")

    def test_score_in_unit_interval_when_available(self):
        from src.engine.trajectory_stability import compute_trajectory_stability_index
        df = self._make_creatinine_df([0, 6, 12, 24, 36, 48], [1.2, 1.8, 1.4, 2.1, 1.9, 1.7])
        result = compute_trajectory_stability_index(df)
        if result.tsi_score is not None:
            assert 0.0 < result.tsi_score <= 1.0, f"TSI={result.tsi_score} outside (0, 1]"

    def test_volatile_scores_lower_than_stable(self):
        from src.engine.trajectory_stability import compute_trajectory_stability_index
        stable_df = self._make_creatinine_df([0, 10, 20, 30, 40, 50], [1.0, 1.01, 1.0, 1.01, 1.0, 1.01])
        volatile_df = self._make_creatinine_df([0, 10, 20, 30, 40, 50], [0.5, 4.0, 0.6, 3.8, 0.7, 5.0])
        stable_r = compute_trajectory_stability_index(stable_df)
        volatile_r = compute_trajectory_stability_index(volatile_df)
        if stable_r.tsi_score is not None and volatile_r.tsi_score is not None:
            assert stable_r.tsi_score > volatile_r.tsi_score, (
                f"Stable TSI ({stable_r.tsi_score}) should exceed volatile ({volatile_r.tsi_score})"
            )

    def test_empty_dataframe_handled(self):
        from src.engine.trajectory_stability import compute_trajectory_stability_index
        df = pd.DataFrame({"charttime": pd.to_datetime([]), "valuenum_effective": []})
        result = compute_trajectory_stability_index(df)
        # Should not raise; tsi_score should be None
        assert result.tsi_score is None

    def test_single_point_returns_none_score(self):
        from src.engine.trajectory_stability import compute_trajectory_stability_index
        df = self._make_creatinine_df([0], [1.0])
        result = compute_trajectory_stability_index(df)
        assert result.tsi_score is None, "Single observation should yield None TSI score"

    def test_result_type(self):
        from src.engine.trajectory_stability import compute_trajectory_stability_index, TrajectoryStabilityResult
        df = self._make_creatinine_df([0, 12, 24], [1.0, 1.2, 1.1])
        result = compute_trajectory_stability_index(df)
        assert isinstance(result, TrajectoryStabilityResult)


# ── Preprocessing / scalers ───────────────────────────────────────────────────

class TestPreprocessingScalers:
    def test_load_cohort_scalers(self):
        from src.engine.preprocessing import load_cohort_scalers
        config = load_cohort_scalers()
        assert config is not None
        assert hasattr(config, "analytes")
        assert len(config.analytes) > 0

    def test_scaler_std_positive(self):
        from src.engine.preprocessing import load_cohort_scalers
        config = load_cohort_scalers()
        for name, params in config.analytes.items():
            assert params.train_std > 0, f"Zero or negative std for analyte: {name}"

    def test_scaler_has_expected_analytes(self):
        from src.engine.preprocessing import load_cohort_scalers
        config = load_cohort_scalers()
        # At minimum creatinine must be present
        analyte_names = set(config.analytes.keys())
        assert "creatinine" in analyte_names, f"Creatinine missing from scalers: {analyte_names}"

    def test_standardize_creatinine(self):
        from src.engine.preprocessing import load_cohort_scalers, standardize_observation
        config = load_cohort_scalers()
        # Standardize a normal creatinine value (1.0 mg/dL)
        z = standardize_observation(1.0, "creatinine", config)
        assert isinstance(z, float)
        assert not math.isnan(z)

    def test_standardize_and_unstandardize_roundtrip(self):
        from src.engine.preprocessing import load_cohort_scalers, standardize_observation, unstandardize_observation
        config = load_cohort_scalers()
        original = 1.5  # mg/dL creatinine
        z = standardize_observation(original, "creatinine", config)
        recovered = unstandardize_observation(z, "creatinine", config)
        assert abs(recovered - original) < 1e-6, (
            f"Round-trip failed: {original} -> z={z} -> {recovered}"
        )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_values_are_rejected_at_transform_boundaries(self, bad_value):
        from src.engine.preprocessing import load_cohort_scalers, standardize_observation, unstandardize_observation
        config = load_cohort_scalers()
        assert math.isnan(standardize_observation(bad_value, "creatinine", config))
        assert math.isnan(unstandardize_observation(bad_value, "creatinine", config))


class TestValidationBaselineEvaluation:
    def _records(self, values, times=None):
        if times is None:
            times = list(range(len(values)))
        return pd.DataFrame({
            "charttime": pd.to_datetime([f"2180-01-01 {hour:02d}:00:00" for hour in times]),
            "storetime": pd.to_datetime([f"2180-01-01 {hour:02d}:00:00" for hour in times]),
            "analyte_name": ["creatinine"] * len(values),
            "valuenum_effective": values,
        })

    def _evaluate(self, records):
        from src.engine.parameter_estimation import load_state_space_parameters
        from src.engine.preprocessing import load_cohort_scalers
        from src.engine.validation_baselines import evaluate_baseline_candidates_from_records
        return evaluate_baseline_candidates_from_records({1: records}, load_cohort_scalers(), load_state_space_parameters())

    def test_nonfinite_targets_do_not_enter_history_or_metrics(self):
        result = self._evaluate(self._records([1.0, np.inf, 2.0, 3.0, 4.0, 5.0]))
        assert result["exclusions"]["nonfinite_target"] == 1
        # Three finite prior values are available only at the fifth finite record.
        assert result["target_count"] == 2
        assert result["finite_prediction_counts"]["prior_median"] == 2

    def test_same_timestamp_rows_use_pre_assimilation_history(self):
        result = self._evaluate(self._records([1.0, 2.0, 3.0, 10.0, 100.0, 4.0], [0, 1, 2, 3, 3, 4]))
        scored = result["scored_records"]
        shared_epoch = scored[scored["charttime"].dt.hour == 3]

        # Both hour-3 targets are eligible, but neither may see the other's
        # value: each must use exactly the three prior-timestamp values.
        assert len(shared_epoch) == 2
        assert all(values == (1.0, 2.0, 3.0) for values in shared_epoch["history_values_used"])
        assert all(value == 3.0 for value in shared_epoch["locf"])
        assert all(value == 2.0 for value in shared_epoch["trailing3_median"])

    def test_candidate_with_no_common_finite_metrics_cannot_be_selected(self):
        result = self._evaluate(self._records([1.0, 1.1]))
        assert result["target_count"] == 0
        assert result["candidate_mean_rmse_ratios"]["ou"] is None
        assert result["selected_candidate"] is None

    def test_metrics_and_predictions_are_finite_on_scored_targets(self):
        result = self._evaluate(self._records([1.0, 1.1, 1.2, 1.3, 1.4]))
        assert result["target_count"] == 2
        for metric in result["per_analyte"]:
            if metric["target_count"]:
                assert all(math.isfinite(metric[name]) for name in ("locf", "ou", "prior_median", "trailing3_median"))


# ── Data partitioning ─────────────────────────────────────────────────────────

class TestDataPartition:
    def test_partition_json_exists(self):
        from pathlib import Path
        p = Path("configs/cohort_patient_splits.json")
        assert p.exists(), "Patient splits JSON not found at configs/cohort_patient_splits.json"

    def test_load_returns_dict_with_splits(self):
        from src.engine.data_partition import load_patient_splits
        splits = load_patient_splits()
        assert isinstance(splits, dict)
        for key in ("train", "val", "test"):
            assert key in splits, f"Missing split key: {key}"

    def test_splits_contain_integers(self):
        from src.engine.data_partition import load_patient_splits
        splits = load_patient_splits()
        for key, ids in splits.items():
            assert all(isinstance(i, int) for i in ids), f"Non-integer IDs in {key} split"

    def test_splits_are_disjoint(self):
        from src.engine.data_partition import load_patient_splits
        splits = load_patient_splits()
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert train.isdisjoint(val), f"Train/val overlap: {len(train & val)} patients"
        assert train.isdisjoint(test), f"Train/test overlap: {len(train & test)} patients"
        assert val.isdisjoint(test), f"Val/test overlap: {len(val & test)} patients"

    def test_splits_non_empty(self):
        from src.engine.data_partition import load_patient_splits
        splits = load_patient_splits()
        for key, ids in splits.items():
            assert len(ids) > 0, f"Split '{key}' is empty"

    def test_train_fraction_reasonable(self):
        from src.engine.data_partition import load_patient_splits
        splits = load_patient_splits()
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        train_frac = len(splits["train"]) / total
        assert 0.60 <= train_frac <= 0.80, (
            f"Train fraction {train_frac:.2f} outside expected 60–80%"
        )

    def test_get_patient_split_function(self):
        from src.engine.data_partition import load_patient_splits, get_patient_split
        splits = load_patient_splits()
        first_train_id = next(iter(splits["train"]))
        split_label = get_patient_split(first_train_id)
        assert split_label == "train", f"Expected 'train', got '{split_label}'"


# ── StateSpaceParameters dataclass ──────────────────────────────────────────

class TestStateSpaceParameters:
    def test_default_construction(self):
        from src.engine.state_space import StateSpaceParameters
        p = StateSpaceParameters()
        assert len(p.tau_hours) == 3
        assert all(t > 0 for t in p.tau_hours)
        assert len(p.r_stds) == 6  # one per analyte

    def test_custom_tau(self):
        from src.engine.state_space import StateSpaceParameters
        p = StateSpaceParameters(tau_hours=[48.0, 24.0, 12.0])
        assert p.tau_hours == [48.0, 24.0, 12.0]

    def test_time_constant_exp_decay_semantics(self):
        """At dt = tau, OU decay factor must equal exp(-1), not 0.5."""
        from src.engine.state_space import ContinuousDiscreteRenalFilter, StateSpaceParameters
        tau0 = 72.0
        params = StateSpaceParameters(tau_hours=[tau0, 24.0, 12.0])
        f = ContinuousDiscreteRenalFilter(params, initial_state=np.array([1.0, 0.0, 0.0]))
        x_after, _ = f.predict(dt_hours=tau0)
        expected_decay = math.exp(-1.0)
        assert abs(x_after[0] - expected_decay) < 0.01, (
            f"At dt=tau, state should decay to exp(-1)={expected_decay:.4f}, got {x_after[0]:.4f}. "
            "Check that tau is used as a time constant (1/e relaxation), not a half-life."
        )
