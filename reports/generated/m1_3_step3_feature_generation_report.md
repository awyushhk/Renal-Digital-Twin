# Module 1.3 Step 3: Irregular Temporal Features & Dual-Timeline Engine Report

**Project**: Nephron-Aware Renal Digital Twin  
**Milestone**: M1.3 Step 3 (Irregular Temporal Features & Dual-Timeline Engine)  
**Parent Artifact**: Validated M1.2 Phase B Cohort (`data/processed/cohort_renal_bounded.csv`)  
**Status**: STEP 3 IMPLEMENTATION & VERIFICATION COMPLETE  
**Date**: 2026-10-18  

---

## 1. Executive Summary

Module 1.3 Step 3 delivers the deterministic, irregular-time, causal feature generation engine across two strictly isolated temporal information streams:
1. **Retrospective Biological Timeline ($\mathcal{F}_{\text{bio}}$)**:
   - Driven by specimen collection timestamp (`charttime`).
   - Represents the continuous biological trajectory of renal physiology.
   - Tagged explicitly with `is_retrospective_only = True` and barred from real-time alerting to prevent operational lookahead leakage.
2. **Operational Timeline ($\mathcal{F}_{\text{ops}}$)**:
   - Driven by EHR verification timestamp (`storetime`).
   - Replays clinical decision support availability at explicit prediction query timestamps $t_{\text{query}}$.
   - Observations with $\text{storetime} > t_{\text{query}}$ are strictly excluded (no operational lookahead).
   - Late-arriving results entering at later operational times never retroactively alter or rewrite historical prediction snapshots.
   - Rates of change are explicitly documented as **operational reporting velocity and acceleration**, capturing laboratory reporting cadences rather than physiological kinetics.

All 67 prior tests and 12 new temporal feature tests (**79 tests total**) pass with 100% success. All raw MIMIC-IV data files, prior audit reports, and the frozen `irregular-creatinine-forecasting` project remain strictly read-only and unaltered. No machine learning predictive models were trained in this step.

---

## 2. Mathematical Formulations & Feature Specifications

All time differences are standardized to **decimal hours ($h$)**.

### 2.1 Elapsed-Time Clocks & Reference Events
- **Biological Elapsed Lifetime Clock**:
  $$\tau_{\text{bio}, k} = \frac{\text{charttime}_k - \min_{i \in \mathcal{D}_p}(\text{charttime}_i)}{3600.0} \quad [h]$$
  Starts at $0.0\text{ h}$ at the patient's first eligible biological observation in the dataset.
- **Operational Elapsed Lifetime Clock**:
  $$\tau_{\text{ops}, k} = \frac{\text{storetime}_k - \min_{i \in \mathcal{D}_p^{\text{avail}}(t_{\text{query}})}(\text{storetime}_i)}{3600.0} \quad [h]$$
  Starts at $0.0\text{ h}$ at the first filed observation available in the operational replay at $t_{\text{query}}$.
- **Admission Elapsed Time**: $\Delta t_{\text{adm}} = (\text{charttime}_k - \text{admittime}) / 3600.0$. Preserved as $\text{NaN}$ for outpatient labs (`hadm_id IS NULL`, 34.06% of cohort) with boolean flag `is_inpatient = False`.
- **Inter-Event Gaps**: Inter-event gap across any analyte ($\Delta t_{\text{event}}$) and analyte-specific gap ($\Delta t_{\text{analyte}}$) in decimal hours.

### 2.2 Discrete Velocity & Unequal-Interval Acceleration
- **Discrete Biological Velocity**:
  $$v_{k, j}^{\text{bio}} = \frac{x_{k, j} - x_{k-1, j}}{\Delta t_{k-1, k}} \quad \left[\frac{\text{analyte unit}}{\text{hour}}\right]$$
  Associated with midpoint coordinate $c_{k-1/2}^* = \frac{c_{k-1} + c_k}{2}$.
- **Unequal-Interval Biological Acceleration (Separation Between Midpoints)**:
  $$\Delta t^* = \frac{c_{k-1/2}^* - c_{k-3/2}^*}{3600.0} = \frac{\Delta t_{k-2, k-1} + \Delta t_{k-1, k}}{2} \quad [h]$$
  $$a_{k, j}^{\text{bio}} = \frac{v_{k, j} - v_{k-1, j}}{\Delta t^*} = 2 \cdot \frac{\frac{x_{k, j} - x_{k-1, j}}{\Delta t_{k-1, k}} - \frac{x_{k-1, j} - x_{k-2, j}}{\Delta t_{k-2, k-1}}}{\Delta t_{k-2, k-1} + \Delta t_{k-1, k}} \quad \left[\frac{\text{analyte unit}}{\text{hour}^2}\right]$$
- **Operational Reporting Rates**: Computed across storetimes and explicitly labeled `velocity_operational_reporting` and `acceleration_operational_reporting`. Docstrings emphasize that these capture verification turnaround and batching, not in vivo physiological dynamics.

### 2.3 Co-Temporal BUN / Creatinine Ratio
- Evaluated **only** within the exact same patient and same `charttime` epoch $\mathcal{E}_{p, c}$.
- Uses active arbitrated values (`valuenum_effective`).
- Returns $\text{NaN}$ with explicit audit flags:
  - `VALID_COTEMPORAL_RATIO`: Both constituents present and Creatinine $> 0.0$.
  - `MISSING_CONSTITUENT`: Only one analyte was ordered/drawn at that epoch.
  - `NON_POSITIVE_CREATININE`: Creatinine $\le 0.0\text{ mg/dL}$ (avoids division by zero and rejects non-physiological entries).
  - In operational replay: returns `RESULT_NOT_YET_FILED` if one of the results has $\text{storetime} > t_{\text{query}}$.
- **Zero Silent Substitution**: Never imputes missing values or substitutes arbitrary small numbers ($+10^{-6}$) without explicit tracking.

---

## 3. Implementation Verification & Test Results

The dedicated test suite [`tests/test_temporal_features.py`](file:///d:/CKD%20mini%20project/nephron-aware-renal-digital-twin/tests/test_temporal_features.py) executes 12 focused unit and integration tests:

| Test Name | Verified Invariant | Status |
| :--- | :--- | :--- |
| `test_dual_timeline_availability_separation` | Draw at 10:00, filed at 12:30 is absent at 11:00 and present at 12:30 in operational replay. | **PASSED** |
| `test_no_retrospective_rewriting_of_operational_snapshots` | Advancing replay clock to 17:00 does not alter historical snapshot produced at 12:00. | **PASSED** |
| `test_manually_calculated_unequal_interval_acceleration` | Manual case ($c=[0,4,10]\text{h}, x=[1.0,1.8,3.6]$) matches exact expected $a = 0.0200\text{ mg/dL/h}^2$. | **PASSED** |
| `test_zero_interval_safety_and_singularity_avoidance` | Co-temporal $\Delta t = 0$ collisions map to $\text{NaN}$ with `COLLISION_ZERO_DELTA` without division by zero. | **PASSED** |
| `test_cotemporal_bun_creatinine_ratio` | BUN/Cr evaluated on co-temporal draws; rejects missing and $\le 0$ creatinine with audit flags. | **PASSED** |
| `test_operational_bun_creatinine_ratio_availability` | Operational ratio returns `RESULT_NOT_YET_FILED` if either constituent is unfiled by $t_{\text{query}}$. | **PASSED** |
| `test_quarantine_negative_delays_from_operational_timeline` | Records with negative delay ($\Delta t_{\text{rep}} < 0$) are quarantined from operational replay. | **PASSED** |
| `test_patient_boundary_isolation` | Lifetime hours, inter-event gaps, analyte gaps, and running stats reset strictly at patient boundary. | **PASSED** |
| `test_outpatient_hadm_id_preservation` | Outpatient records preserved with `is_inpatient = False` and `dt_admission = NaN` without row drop. | **PASSED** |
| `test_future_perturbation_invariance` | Perturbing records at $t > T$ produces 0.0000 drift on historical features at $t \le T$. | **PASSED** |
| `test_temporal_features_on_real_extracted_cohort_slice` | Integration run on real 500-row cohort slice executes with zero NaN exceptions or crashes. | **PASSED** |
| `test_three_arm_baseline_generators` | Arm 1 (24h LOCF regular grid) and Arm 2 (deliberately flawed controls) generate cleanly. | **PASSED** |

### Complete Test Suite Regression Check:
```
tests/test_audit_dataset.py ...........                                  [11 passed]
tests/test_canonical_schema.py ........                                  [ 8 passed]
tests/test_cohort_extraction.py .............                            [13 passed]
tests/test_conflict_arbitration.py .............                         [13 passed]
tests/test_temporal_features.py ............                             [12 passed]
tests/test_weka_benchmark.py ......................                      [22 passed]
============================= 79 passed in 19.65s =============================
```

---

## 4. Cohort Execution Profile (Sample Evaluation)

Execution on a 1,000-row slice of `cohort_renal_bounded.csv` (13 patients, 183 observation epochs) demonstrated:
- **Zero Singularities**: 0 velocity collisions ($\Delta t = 0$) and 0 acceleration collisions.
- **Co-Temporal Ratios**: 969 valid co-temporal BUN/Cr measurements; 25 rows with missing constituent; 0 non-positive creatinine measurements.
- **Operational Snapshot**: At median storetime, exactly 503 rows were visible; 0 future rows leaked.
- **Baseline Arm Comparison**:
  - Arm 1 (24h Regular Grid with LOCF): Generates 10,339 synthetic daily time steps, demonstrating that artificial daily grids inflate sparsity by **10.4x** over true clinical sampling.
  - Arm 2 Control 2C (Admissions Inner Join): Drops 399 rows (**39.9% data loss**), demonstrating the catastrophic data drop of naive inner joins.

---

## 5. Documented Limitations & Remaining Scope

1. **Non-Equivalence of Reporting Rates and Biological Kinetics**:
   - Operational reporting velocity and acceleration describe the cadence of laboratory verification and result transmission, not patient physiology. They must never be interpreted as in vivo biological rates.
2. **Retrospective Biological Limitation**:
   - Biological trajectory features computed using `charttime` are valid strictly for retrospective analysis (e.g. continuous-time ODE model training or post-hoc digital twin calibration). They cannot be evaluated for real-time alerting without lookahead.
3. **Step 4 Scope**:
   - The final 3-arm baseline comparison and comprehensive data-integrity report will be assembled in Step 4 ([`src/temporal/representation_pipeline.py`](file:///d:/CKD%20mini%20project/nephron-aware-renal-digital-twin/src/temporal/representation_pipeline.py)). No machine learning predictive models are trained.

---

## 6. Conclusion & Recommendation

Module 1.3 Step 3 is **COMPLETE and PASSES ALL INVARIANTS**. All source data, prior milestone artifacts, and tests remain immutable and 100% functional.

We are ready for user review before proceeding to **Module 1.3 Step 4** (3-Way Baseline Comparison Pipeline & Final Representation Report).
