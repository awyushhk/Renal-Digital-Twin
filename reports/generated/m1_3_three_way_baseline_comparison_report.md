# Module 1.3 Step 4: Three-Way Baseline Comparison & Representation Audit Report

**Milestone:** Module 1.3 Step 4 — Three-Way Baseline Comparison Pipeline & Final Representation Audit  
**Date:** 2026-10-18  
**Dataset Evaluated:** Validated Bounded MIMIC-IV Renal Cohort (`data/processed/cohort_renal_bounded.csv`)  
**SHA-256 Hash:** `9534adfda88896433cae38e87efcd66008bf558595a5734d5251181a615b4d35`  
**Total Source Records:** 303,460 rows | **Total Patients:** 2,500 unique individuals  
**Repository State:** Frozen baseline code, prior milestone artifacts, and raw MIMIC-IV source files strictly preserved and immutable.

---

## 1. Executive Summary & Scope Declaration

Module 1.3 Step 4 establishes a rigorous, quantitative data-integrity comparison across three distinct temporal representation paradigms for clinical laboratory trajectories:

1. **Arm 1: Fair Conventional Baseline (24-Hour Regular Grid with LOCF)**  
   Resamples observation epochs onto regular 24-hour calendar-day steps, applies Last Observation Carried Forward (LOCF) within patient boundaries, and retains outpatient records through preserved left joins.
2. **Arm 2: Deliberately Flawed Controls (Common ML Pitfalls)**  
   - *Control 2A (Normalization Leakage):* Full-batch normalization computing global bounds across past and future observation windows.
   - *Control 2B (Cross-Patient Contamination):* Naive time-series differences across un-reset patient boundaries.
   - *Control 2C (Admissions Inner-Join Loss):* Relational inner join against `admissions.csv` dropping all outpatient records lacking an `hadm_id`.
3. **Arm 3: Proposed Renal-Aware Representation**  
   Irregular-time point process preserving every clinical observation epoch, dual-timeline filtration separating biological retrospective trajectory ($\mathcal{F}_{\text{bio}}$) from operational clinical availability ($\mathcal{F}_{\text{ops}}$), co-temporal multi-analyte BUN/creatinine ratios, and non-uniform velocity/acceleration.

### Explicit Scope & Non-Claims Constraint
In accordance with approved protocols:
- **No predictive models were trained.**
- **No claims of clinical superiority, enhanced diagnostic accuracy, or downstream predictive performance are made.**
- Evaluation is restricted strictly to **structural data integrity, temporal distortion, division-by-zero avoidance, patient-boundary isolation, and operational lookahead leakage.**

---

## 2. Definitive Reconciliation & Source-Row Accounting

To resolve discrepancies across prior planning documents, all figures have been recalculated dynamically across the full 2,500-patient bounded cohort ($N=303,460$). Denominators and accounting perspectives are explicitly defined below.

### Explicit Denominator Definitions
- **Total Source Records ($N_{\text{source}} = 303,460$):** Total valid lab rows extracted from MIMIC-IV in Module 1.2 meeting renal bounding criteria.
- **Active Policy A Records ($N_{\text{active}} = 303,232$):** Retained records after Policy A conflict arbitration (preferring latest storetime and earliest labevent ID). Exactly **228 records** are superseded:
  - 94 exact duplicate records (`SUPERSEDED_DUPLICATE`)
  - 134 conflicting measurement records (`SUPERSEDED_CONFLICT`)
- **Distinct Patient-Day-Analyte Cells ($N_{\text{cells}} = 259,869$):** Unique combinations of `(subject_id, date, analyte)` in the cohort. This constitutes the maximum theoretical resolution of any 24-hour daily grid.
- **Intra-Day Collapsed Records ($N_{\text{collapsed}}$):** Observations that share the same patient, calendar day, and analyte, and are consequently merged or discarded when reducing observations to a single daily scalar.

### Exact Mathematical Reconciliation
$$\begin{aligned}
N_{\text{source}} &= N_{\text{cells}} + N_{\text{raw\_collapsed}} = 259,869 + 43,591 = \mathbf{303,460} \\
N_{\text{active}} &= N_{\text{cells}} + N_{\text{active\_collapsed}} = 259,869 + 43,363 = \mathbf{303,232} \\
\Delta_{\text{collapsed}} &= N_{\text{raw\_collapsed}} - N_{\text{active\_collapsed}} = 43,591 - 43,363 = \mathbf{228} \equiv N_{\text{superseded}}
\end{aligned}$$

| Metric Description | Raw Source Accounting | Active Policy A Accounting | Mathematical Difference |
| :--- | :---: | :---: | :---: |
| **Total Cohort Denominator** | **303,460** (100.0%) | **303,232** (100.0%) | 228 superseded rows |
| **Distinct Patient-Day-Analyte Cells** | 259,869 | 259,869 | 0 (identical grid cells) |
| **Cells with Exactly 1 Observation** | 226,940 (87.33%) | 227,118 (87.40%) | +178 single-obs cells |
| **Cells with >1 Multi-Observations** | 32,929 (12.67%) | 32,751 (12.60%) | -178 multi-obs cells |
| **Total Rows in Multi-Observation Cells** | 76,520 | 76,114 | -406 multi-obs rows |
| **Intra-Day Collapsed Observations** | **43,591** (14.3647%) | **43,363** (14.3003%) | **228 records** |
| **Individual Value Retention Rate** | **85.6353%** | **85.6997%** | +0.0644% |
| **Source Provenance Traceability** | **100.0%** (303,460/303,460) | **100.0%** (303,232/303,232) | 0 lost identifiers |

### Source Traceability vs. Value Preservation
> [!IMPORTANT]
> **Source traceability is not value preservation.** While Arm 1 destination grid cells can retain lists of source `labevent_id`s (achieving 100% ID traceability), **43,591 raw clinical measurements (14.36%) and 43,363 active clinical measurements (14.30%) are completely dropped from the feature representation.** In acute hospital settings where multiple measurements track rapid renal decline within a single day, collapsing to a single daily value discards vital kinetic information. In contrast, Arm 3 preserves 100% of active values ($N=303,232$) with 0 intra-day collapses.

---

## 3. Daily-Grid Anchoring Convention & Subgroup Density Metrics

### Anchoring Convention
To ensure fair comparison, the conventional 24-hour grid in Arm 1 is anchored per patient:
- **Start:** $\lfloor\min(\text{charttime})\rfloor_{\text{day}}$ (00:00:00 on the day of the patient's first observation)
- **End:** $\lfloor\max(\text{charttime})\rfloor_{\text{day}}$ (00:00:00 on the day of the patient's last observation)
- **Span:** Every intermediate calendar day is included in the patient's grid.

### Grid Terminology Correction
The metric previously termed "grid expansion ratio" has been formally renamed to **daily-grid-to-source-record ratio** ($\text{Grid Steps} / \text{Source Records}$), because the ratio is **below 1.0** for acute inpatients experiencing frequent monitoring.

### Empirical Density Metrics across Cohort Subgroups

| Cohort Subgroup | Patient Count | Source Records | Occupied Patient-Days | Total 24h Grid Steps | Daily-Grid-to-Source-Record Ratio | Empty Grid Fraction | Within-Day Multi-Obs Ratio | Distinct Epochs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Inpatient** | 208 | 58,819 | 7,458 | 31,192 | **0.5303x** | **76.0908%** | 7.8867 | 9,985 |
| **Pure Outpatient** | 378 | 15,317 | 3,236 | 707,685 | **46.2026x** | **99.5427%** | 4.7333 | 3,367 |
| **Mixed (In + Out)** | 1,914 | 229,324 | 36,818 | 1,909,306 | **8.3258x** | **98.0717%** | 6.2286 | 43,663 |
| **Full Bounded Cohort** | **2,500** | **303,460** | **47,512** | **2,648,183** | **8.7266x** | **98.2059%** | **6.3870** | **57,015** |

### Reconciliation of Step 3 Sample Metrics
- In the Step 3 preliminary run, a sample of 13 ascending-ID patients generated **10,339 daily grid steps** across **994 post-arbitration records**, yielding a ratio of **10.4014x** and an empty fraction of **98.2300%**.
- Across the complete 2,500-patient cohort, the verified empirical ratio is **8.7266x** and empty fraction is **98.2059%**.
- The acute inpatient ratio of **0.5303x** proves that resampling to a daily grid compresses observation density by nearly half for intensive care patients, while creating massive 99.54% empty grid matrices for outpatients over multi-year monitoring spans.

---

## 4. Dynamic Negative-Control Audit (Arm 2 Evaluation)

All negative controls were evaluated dynamically against mathematical assertions rather than static string matching:

### Control 2A: Full-Batch Normalization Leakage
- **Mechanism:** Global MinMax normalization fitted across all timestamps $[0, T]$. A future observation at $t_{\text{future}} > t_{\text{cutoff}}$ was perturbed ($x_{\text{pert}} = 10 \times x_{\max}$).
- **Historical Drift Assertion:** For all $t \le t_{\text{cutoff}}$, historical drift was calculated as $\max_{t \le t_{\text{cutoff}}} |x_{\text{norm}}^{\text{base}}(t) - x_{\text{norm}}^{\text{pert}}(t)|$.
- **Empirical Results:**
  - Control 2A exhibited measurable historical drift: $\mathbf{\Delta_{\text{drift}} > 0.0}$ (dynamically verified in automated test suite).
  - Arm 1 (causal train-split scaler) and Arm 3 (causal retrospective features) exhibited identically $\mathbf{\Delta_{\text{drift}} \equiv 0.0}$.
- **Implication:** Full-batch scaling silently allows future clinical severity to alter historical normalized inputs, creating severe lookahead bias in retrospective ML benchmarks.

### Control 2B: Cross-Patient Delta Bleed
- **Mechanism:** Naive `diff()` across an un-reset DataFrame sorted purely by timestamp.
- **Boundary Transition Assertion:** At transition rows where `subject_id` switches from Patient $A$ to Patient $B$:
  - Control 2B computed an inter-patient time delta $\Delta t = (t_{B, 0} - t_{A, \text{end}}) > 0$, erroneously coupling unrelated patients.
  - Arm 3 dynamically verified that the patient clock resets strictly:
    - $\text{hours\_since\_first\_observed\_event\_bio} \equiv 0.0$ (patient clock initialized, not NaN).
    - $\text{dt\_previous\_event\_hours\_bio} \equiv \text{NaN}$ (no cross-patient inter-event interval).
    - $\text{velocity\_biological} \equiv \text{NaN}$ (0 cross-patient boundary violations).
- **Implication:** Un-reset time-series transformations in standard tools (e.g. Weka TimeSeriesDelta) contaminate initial patient observations with preceding patient data.

### Control 2C: Admissions Inner-Join Outpatient Loss
- **Mechanism:** Standard relational `merge(..., on="hadm_id", how="inner")` commonly found in ICU-centric pipelines.
- **Empirical Loss Audit:**
  - **Rows Lost:** Exactly **103,354 records** (**34.0585%** of the bounded cohort).
  - **Patients Lost:** Exactly **378 patients** (**15.1200%** of the bounded cohort) had zero inpatient admissions and were completely eliminated.
  - **Verification:** Rows lost dynamically matches `(df['hadm_id'].isna()).sum() == 103,354`.
  - *Discrepancy Note:* Earlier planning estimates cited 225 patients lost based on a preliminary subset; the verified full-cohort count is **378 pure outpatient individuals**.

---

## 5. Three-Way Baseline Comparison Matrix

The table below summarizes the structural data integrity and temporal distortion across all three arms:

| Dimension | Arm 1: Fair Conventional Baseline (24h Grid + LOCF) | Arm 2: Deliberately Flawed Controls (ML Pitfalls) | Arm 3: Proposed Renal-Aware Representation |
| :--- | :--- | :--- | :--- |
| **Grid Structure** | Regular 24-hour resampled grid | Variable / Un-reset tabular | Continuous irregular point process |
| **Intra-Day Resolution** | **Collapsed** (43,363 active rows / 14.30% discarded) | Dependent on pipeline flaws | **Preserved** (0 rows collapsed; 100% active retention) |
| **Observation Density Ratio** | 8.7266x full cohort (0.5303x in acute, 46.20x in outpt) | 1.0x (un-resampled) | 1.0x (exact observation epochs) |
| **Empty Matrix Sparsity** | **98.2059%** artificial missing grid steps | 0.0% artificial steps | **0.0%** artificial steps |
| **Imputation Artifacts** | Step-function LOCF plateau distortion | None / Leaked global imputation | Point-process elapsed-time representations |
| **Outpatient Preservation** | Preserved via left joins (0 patients lost) | **Severe loss** (Control 2C: 34.06% rows, 15.12% pts lost) | **Preserved** (0 records or patients lost) |
| **Boundary Isolation** | Preserved via patient-level grouping | **Contaminated** (Control 2B: cross-patient bleed) | **Strictly isolated** (0 boundary violations) |
| **Operational Lookahead** | Vulnerable if using charttime as availability | **Severe leakage** (Control 2A: future bound drift) | **Strictly prevented** via dual timelines ($\mathcal{F}_{\text{bio}} \perp \mathcal{F}_{\text{ops}}$) |
| **Kinetic Dynamics** | Daily step-deltas (coarse, biased by LOCF) | Unscaled index diffs | Unequal-interval velocity and non-uniform acceleration |
| **Multi-Analyte Ratios** | Daily cross-analyte ratios (coarse) | Unchecked ratios (division-by-zero hazard) | Synchronous co-temporal epochs with zero-division safety |
| **Provenance Traceability** | Cell-level list mapping (values collapsed) | Broken if un-reset or dropped | Complete bidirectional source-ID audit links |

---

## 6. Scientific Limitations & Disclosures

1. **Retrospective Trajectory vs. Operational Availability:**  
   Features computed on the biological timeline ($\mathcal{F}_{\text{bio}}$) using `charttime` represent retrospective clinical reconstruction (e.g. for disease phenotyping or clinical audit). They must **never** be used for prospective operational alerting at `charttime`, because lab turnaround delays (averaging 1.63 hours in MIMIC-IV) render results operationally invisible until `storetime`.
2. **Kinetic Approximation:**  
   Irregular-time biological velocity ($v_k$) and unequal-interval acceleration ($a_k$) are finite-difference discrete approximations of underlying non-linear physiological processes. Acceleration requires a minimum of 3 consecutive observation epochs and is flagged as `INSUFFICIENT_HISTORY` when fewer observations are present.
3. **No Downstream Predictive Superiority Claimed:**  
   This audit establishes structural data-hygiene superiority (zero collapsed observations, zero cross-patient contamination, zero future leakage, zero outpatient join loss). It does **not** assert that models trained on Arm 3 representations will achieve higher AUC-ROC or lower RMSE than Arm 1 models. Empirical predictive performance remains a question for downstream model benchmarking.

---

## 7. Verification Test Suite Results

The comprehensive test suite was executed across all components using `pytest`:

```
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\CKD mini project\nephron-aware-renal-digital-twin

tests/test_audit_dataset.py ...........                                  [ 12%]
tests/test_canonical_schema.py ........                                  [ 21%]
tests/test_cohort_extraction.py .............                             [ 36%]
tests/test_conflict_arbitration.py .............                          [ 51%]
tests/test_representation_pipeline.py ........                           [ 60%]
tests/test_temporal_features.py ............                             [ 74%]
tests/test_weka_benchmark.py .......................                     [100%]

============================= 87 passed in 21.29s =============================
```

### Verified Hash Invariants
- `data/processed/cohort_renal_bounded.csv`: `9534adfda88896433cae38e87efcd66008bf558595a5734d5251181a615b4d35` (Bitwise immutable, unmodified).
- Frozen forecasting repository `irregular-creatinine-forecasting`: Untouched and preserved.
- All prior milestone reports (M1.1, M1.2 Phase A/B, M1.3 Steps 1–3): Intact.
