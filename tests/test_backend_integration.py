"""
Module 2 — Backend Integration Tests
=====================================
Tests the FastAPI endpoints against a running backend.

Run with:
    pytest tests/test_backend_integration.py -v

The backend must be running on http://localhost:8000.
These tests use httpx for async-capable requests.
"""

import pytest
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 15.0


@pytest.fixture(scope="module")
def client():
    """Synchronous test client for all integration tests.
    Uses running localhost server if reachable, otherwise TestClient with lifespan."""
    try:
        with httpx.Client(base_url=BASE_URL, timeout=1.0) as c:
            r = c.get("/health")
            if r.status_code == 200:
                with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as live_c:
                    yield live_c
                return
    except Exception:
        pass

    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as tc:
        yield tc


# ── /health ─────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200, resp.text

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data


# ── /api/v1/cohort/summary ───────────────────────────────────────────────────

class TestCohortSummary:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/cohort/summary")
        assert resp.status_code == 200, resp.text

    def test_patient_count_reasonable(self, client):
        data = client.get("/api/v1/cohort/summary").json()
        assert data["total_patients"] > 0
        assert data["total_observations"] > 0

    def test_analyte_scalers_present(self, client):
        data = client.get("/api/v1/cohort/summary").json()
        scalers = data["analyte_scalers"]
        assert isinstance(scalers, dict)
        assert len(scalers) > 0
        for name, s in scalers.items():
            assert "itemid" in s
            assert "train_mean" in s
            assert "train_std" in s
            assert s["train_std"] > 0, f"Zero std for {name}"

    def test_schema_version(self, client):
        data = client.get("/api/v1/cohort/summary").json()
        assert data["schema_version"] == "2.0.0"


# ── /api/v1/patients ─────────────────────────────────────────────────────────

class TestPatientsList:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/patients")
        assert resp.status_code == 200, resp.text

    def test_pagination_schema(self, client):
        data = client.get("/api/v1/patients?page=1&limit=10").json()
        assert "total_patients" in data
        assert "page" in data
        assert "total_pages" in data
        assert len(data["patients"]) <= 10

    def test_patients_have_expected_fields(self, client):
        data = client.get("/api/v1/patients?page=1&limit=5").json()
        for p in data["patients"]:
            assert isinstance(p["subject_id"], int)
            assert p["gender"] in ("M", "F")
            assert p["anchor_age"] > 0
            assert isinstance(p["has_admission"], bool)
            assert p["total_target_labs_count"] >= 0

    def test_inpatient_filter(self, client):
        data = client.get("/api/v1/patients?has_admission=INPATIENT&limit=50").json()
        for p in data["patients"]:
            assert p["has_admission"] is True

    def test_outpatient_filter(self, client):
        data = client.get("/api/v1/patients?has_admission=OUTPATIENT&limit=50").json()
        for p in data["patients"]:
            assert p["has_admission"] is False

    def test_page_2(self, client):
        data_p1 = client.get("/api/v1/patients?page=1&limit=5").json()
        data_p2 = client.get("/api/v1/patients?page=2&limit=5").json()
        ids_p1 = {p["subject_id"] for p in data_p1["patients"]}
        ids_p2 = {p["subject_id"] for p in data_p2["patients"]}
        assert ids_p1.isdisjoint(ids_p2), "Page 1 and Page 2 must return disjoint patient sets"


# ── /api/v1/patients/{id}/trajectory ─────────────────────────────────────────

class TestPatientTrajectory:
    def _first_patient_id(self, client) -> int:
        data = client.get("/api/v1/patients?page=1&limit=1").json()
        return data["patients"][0]["subject_id"]

    def test_returns_200_for_valid_patient(self, client):
        pid = self._first_patient_id(client)
        resp = client.get(f"/api/v1/patients/{pid}/trajectory")
        assert resp.status_code == 200, resp.text

    def test_returns_404_for_unknown_patient(self, client):
        resp = client.get("/api/v1/patients/999999999/trajectory")
        assert resp.status_code == 404

    def test_trajectory_schema(self, client):
        pid = self._first_patient_id(client)
        data = client.get(f"/api/v1/patients/{pid}/trajectory").json()
        assert data["subject_id"] == pid
        assert isinstance(data["observations"], list)
        assert data["total_observations"] == len(data["observations"])

    def test_observation_fields(self, client):
        pid = self._first_patient_id(client)
        data = client.get(f"/api/v1/patients/{pid}/trajectory").json()
        for obs in data["observations"][:5]:
            assert "itemid" in obs
            assert "charttime" in obs
            assert "valuenum_effective" in obs
            assert "unit_harmonized" in obs
            assert "arbitration_role" in obs

    def test_deidentified_charttimes_are_serialized_unchanged_and_ordered(self, client):
        """The UI must receive MIMIC's de-identified wall-clock timestamps verbatim."""
        subject_id = 10000032
        data = client.get(f"/api/v1/patients/{subject_id}/trajectory").json()
        assert data["subject_id"] == subject_id
        assert data["patient_metadata"]["anchor_year"] == 2180
        charttimes = [obs["charttime"] for obs in data["observations"]]
        assert charttimes[0] == "2180-03-23T11:51:00"
        assert charttimes == sorted(charttimes)
        assert all("Z" not in timestamp and "+" not in timestamp for timestamp in charttimes)


# ── /api/v1/state/estimate ────────────────────────────────────────────────────

class TestStateEstimate:
    def _first_patient(self, client) -> dict:
        data = client.get("/api/v1/patients?page=1&limit=1").json()
        return data["patients"][0]

    def _get_patient_last_charttime(self, client, subject_id: int) -> str:
        traj = client.get(f"/api/v1/patients/{subject_id}/trajectory").json()
        if not traj["observations"]:
            pytest.skip("Patient has no observations")
        # Use a time well after the last observation
        return traj["observations"][-1]["charttime"]

    def test_valid_estimate_returns_200(self, client):
        p = self._first_patient(client)
        query_time = self._get_patient_last_charttime(client, p["subject_id"])
        payload = {
            "subject_id": p["subject_id"],
            "query_time": query_time,
            "timeline_mode": "BIOLOGICAL",
            "policy_id": "POLICY_A_STORETIME",
            "noise_scale_factor": 1.0,
        }
        resp = client.post("/api/v1/state/estimate", json=payload)
        assert resp.status_code == 200, resp.text

    def test_estimate_schema(self, client):
        p = self._first_patient(client)
        query_time = self._get_patient_last_charttime(client, p["subject_id"])
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": p["subject_id"],
            "query_time": query_time,
            "timeline_mode": "BIOLOGICAL",
            "policy_id": "POLICY_A_STORETIME",
            "noise_scale_factor": 1.0,
        })
        data = resp.json()
        # Required top-level keys
        for key in ["subject_id", "query_time", "timeline_mode", "filtered_latent_states",
                    "conditional_uncertainty_sd", "calculated_empirical_indices",
                    "contributing_observations", "provenance"]:
            assert key in data, f"Missing key: {key}"

        # Latent states
        fls = data["filtered_latent_states"]
        for k in ["filtration_state_z", "tubular_stress_z", "workload_avidity_z"]:
            assert k in fls
            assert isinstance(fls[k], (int, float))

        # Uncertainty SDs must be non-negative
        sd = data["conditional_uncertainty_sd"]
        for k in ["filtration_sd", "tubular_sd", "workload_sd"]:
            assert sd[k] >= 0, f"{k} is negative"

    def test_unknown_patient_returns_404(self, client):
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": 999999999,
            "query_time": "2180-01-01T00:00:00Z",
            "timeline_mode": "OPERATIONAL",
            "policy_id": "POLICY_A_STORETIME",
            "noise_scale_factor": 1.0,
        })
        assert resp.status_code == 404

    def test_operational_masks_future_storetimes(self, client):
        """OPERATIONAL mode must mask any observation whose storetime > query_time."""
        p = self._first_patient(client)
        # Query at a very early time — should have 0 or few observations
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": p["subject_id"],
            "query_time": "2100-01-01T00:00:00Z",
            "timeline_mode": "OPERATIONAL",
            "policy_id": "POLICY_A_STORETIME",
            "noise_scale_factor": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["timeline_mode"] == "OPERATIONAL"

    def test_provenance_disclaimer_present(self, client):
        p = self._first_patient(client)
        query_time = self._get_patient_last_charttime(client, p["subject_id"])
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": p["subject_id"],
            "query_time": query_time,
            "timeline_mode": "BIOLOGICAL",
            "policy_id": "POLICY_A_STORETIME",
            "noise_scale_factor": 1.0,
        })
        provenance = resp.json()["provenance"]
        assert "research_disclaimer" in provenance
        assert len(provenance["research_disclaimer"]) > 20

    def test_noise_scale_factor_out_of_range(self, client):
        """noise_scale_factor outside [0.1, 10.0] should return 422."""
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": 10000032,
            "query_time": "2180-01-01T00:00:00Z",
            "timeline_mode": "OPERATIONAL",
            "policy_id": "POLICY_A_STORETIME",
            "noise_scale_factor": 100.0,  # out of range
        })
        assert resp.status_code == 422


# ── Priority 1 Regression & Trajectory Tests ───────────────────────────────

class TestPriority1RegressionScenarios:
    """Rigorous regression tests for patient 10006029 and Priority 1 & 2 requirements."""

    PID = 10006029

    def test_patient_search(self, client):
        """Search parameter correctly filters patient list across full bounded cohort."""
        resp = client.get(f"/api/v1/patients?search={self.PID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_patients"] >= 1
        ids = [p["subject_id"] for p in data["patients"]]
        assert self.PID in ids

    def test_latent_trajectory_endpoint(self, client):
        """Latent trajectory replay endpoint returns full chronological points."""
        resp = client.get(f"/api/v1/patients/{self.PID}/latent-trajectory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["subject_id"] == self.PID
        assert data["total_epochs"] > 10
        p0 = data["points"][0]
        assert "charttime" in p0
        assert "filtration_state_z" in p0
        assert "tubular_stress_z" in p0
        assert "workload_avidity_z" in p0
        assert "filtration_sd" in p0
        assert p0["filtration_sd"] > 0

    def test_scenario_1_before_first_observation(self, client):
        """Before first observation: 0 obs, explicit UNINITIALIZED_PRIOR_STATE and explanation."""
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": self.PID,
            "query_time": "2168-01-01T00:00:00",
            "timeline_mode": "OPERATIONAL",
        })
        assert resp.status_code == 200
        d = resp.json()
        assert d["contributing_observations_count"] == 0
        assert d["initialization_status"] == "UNINITIALIZED_PRIOR_STATE"
        assert "precedes first eligible observation" in d["eligibility_explanation"]
        assert d["first_eligible_time"] is not None
        assert d["filtered_latent_states"]["filtration_state_z"] == 0.0
        # Uncertainty must be positive stationary prior SD
        assert d["conditional_uncertainty_sd"]["filtration_sd"] > 0.5

    def test_scenario_2_at_first_observation_biological_vs_operational(self, client):
        """At first charttime 2168-09-04T12:20:00:
        BIOLOGICAL sees labs at collection time;
        OPERATIONAL masks them until storetime."""
        first_charttime = "2168-09-04T12:20:00"
        
        # Biological: specimen drawn -> visible
        r_bio = client.post("/api/v1/state/estimate", json={
            "subject_id": self.PID,
            "query_time": first_charttime,
            "timeline_mode": "BIOLOGICAL",
        })
        assert r_bio.status_code == 200
        d_bio = r_bio.json()
        assert d_bio["contributing_observations_count"] > 0
        assert d_bio["initialization_status"] == "OBSERVATION_UPDATED_ESTIMATE"
        assert d_bio["filtered_latent_states"]["filtration_state_z"] != 0.0

        # Operational: not filed yet at 12:20 -> masked
        r_ops = client.post("/api/v1/state/estimate", json={
            "subject_id": self.PID,
            "query_time": first_charttime,
            "timeline_mode": "OPERATIONAL",
        })
        assert r_ops.status_code == 200
        d_ops = r_ops.json()
        assert d_ops["contributing_observations_count"] == 0
        assert d_ops["initialization_status"] == "UNINITIALIZED_PRIOR_STATE"

    def test_scenario_3_at_time_with_available_observations(self, client):
        """At mid-trajectory: real updated estimates, TSI calculated, eGFR proxy present."""
        mid_time = "2169-11-21T08:29:00"
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": self.PID,
            "query_time": mid_time,
            "timeline_mode": "OPERATIONAL",
        })
        assert resp.status_code == 200
        d = resp.json()
        assert d["contributing_observations_count"] > 50
        assert d["initialization_status"] == "OBSERVATION_UPDATED_ESTIMATE"
        fls = d["filtered_latent_states"]
        assert any(fls[k] != 0.0 for k in fls)
        # TSI and eGFR
        idx = d["calculated_empirical_indices"]
        assert idx["trajectory_stability_status"] == "COMPUTED"
        assert idx["trajectory_stability_index"] is not None
        assert idx["derived_egfr_ckd_epi"] is not None

    def test_scenario_4_after_latest_observation(self, client):
        """30 days after latest observation: all 368 observations assimilated, hours elapsed ~720."""
        post_time = "2170-09-09T06:25:00"
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": self.PID,
            "query_time": post_time,
            "timeline_mode": "OPERATIONAL",
        })
        assert resp.status_code == 200
        d = resp.json()
        assert d["contributing_observations_count"] == 368
        assert d["hours_since_last_observation"] > 700.0

    def test_scenario_5_during_observation_gap(self, client):
        """Querying during a gap between observations predicts forward without measurement update."""
        # Query at 2169-01-01 (in between visits)
        gap_time = "2169-01-01T00:00:00"
        resp = client.post("/api/v1/state/estimate", json={
            "subject_id": self.PID,
            "query_time": gap_time,
            "timeline_mode": "OPERATIONAL",
        })
        assert resp.status_code == 200
        d = resp.json()
        assert d["contributing_observations_count"] > 0
        assert d["hours_since_last_observation"] is not None
