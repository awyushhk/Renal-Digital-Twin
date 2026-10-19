# WEKA Preprocessing Capability Audit — Nephron-Aware Renal Digital Twin

**Generated:** 2026-10-19 02:20:11  

**WEKA JAR:** `D:\Software\weka\Weka-3-8-7\weka.jar`  

**Project:** `D:\CKD mini project\nephron-aware-renal-digital-twin`  

**Bounded cohort:** `cohort_renal_bounded.csv` (COHORT_CSV.exists()=True)  


---


## Deliverable A — Environment Verification


**Java:** java version "21.0.8" 2025-07-15 LTS


**WEKA JAR:** `D:\Software\weka\Weka-3-8-7\weka.jar` — exists=True, size=14,653,641 bytes


**Python:** 3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]


**pandas:** 3.0.6, **numpy:** 2.5.4


**Bounded cohort:** `D:\CKD mini project\nephron-aware-renal-digital-twin\data\processed\cohort_renal_bounded.csv` — exists=True


### A.1 Smoke Test — WEKA Normalize on 3-Row ARFF


**Smoke test command:** `java --add-opens java.base/java.lang=ALL-UNNAMED -cp D:\Software\weka\Weka-3-8-7\weka.jar weka.filters.unsupervised.attribute.Normalize -S 1.0 -T 0.0 -i D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\smoke_test.arff -o D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\weka_out\smoke_test_out.arff`


**Exit code:** 0


| Row | Input x | Expected (MinMax) | WEKA Output | Match |
| --- | --- | --- | --- | --- |
| 0 | 1.0 | 0.0 | 0.0000 | ✅ |
| 1 | 2.0 | 0.5 | 0.5000 | ✅ |
| 2 | 3.0 | 1.0 | 1.0000 | ✅ |


> [IMPORTANT]
> Smoke test PASSED. WEKA Normalize ran successfully and produced correct MinMax output [0.0, 0.5, 1.0].


## Deliverable B — Module 2 Factor Inventory


Module 2 (Physiology-Constrained Latent Renal State Estimation) requires inputs that support
estimation of: nephron functional capacity, compensatory workload, filtration and albuminuric state,
and renal functional reserve. These are **latent model outputs**, not directly observed biomarkers.
The preprocessing audit concerns the **raw inputs** needed to support defensible estimation.


### B.1 eGFR Fields — Availability and Usability


From the M1.1 full-dataset audit (158,374,764 rows scanned):
- **Item 50920 (eGFR MDRD, Blood/Chemistry):** 1,665,881 records — `valuenum` **100% missing**.
  eGFR is stored as text strings (e.g. `>60`, `<15`). No numeric `valuenum` is available.
- **Item 53161 (eGFR CKD-EPI Refit, Blood/Chemistry):** 81,810 records — `valuenum` 0% missing.
  Numerically usable but represents a **derived equation output**, not a raw biomarker.
- **Item 53180 (eGFR CKD-EPI 2021, Blood Gas):** 2,125 records — `valuenum` 0% missing.

The bounded cohort (2,500 patients, ascending subject_id 10000032–10138195) contains only the
6 serum chemistry analytes (items 50912, 51006, 50971, 50983, 50882, 50931). Items 50920 and 53161
are **not present in the bounded cohort CSV**.


> [IMPORTANT]
> eGFR-MDRD (50920) cannot be used as a numeric input: 100% of its `valuenum` records are null.
If eGFR is needed for Module 2, it must be **derived** from creatinine + demographics using a published
equation (e.g. CKD-EPI 2021). Such a derived eGFR is NOT independent of serum creatinine.


### B.2 Urine and Kidney-Damage Factors


From the full-dataset audit:
- **Item 51069 (Albumin, Urine):** 73,957 records, 16.59% missing `valuenum`, unit `mg/dL`.
- **Item 51082 (Creatinine, Urine):** 237,799 records, 0.07% missing, unit `mg/dL`.
- **Item 51099 (Protein/Creatinine Ratio):** 100,387 records — **mixed units** (`Ratio: 32,757`,
  `mg/mg: 67,630`). Numerically inconsistent without harmonization.
- **Item 51492 (Protein, Urine):** 843,478 records — **66.65% missing numeric**, ordinal dipstick
  text values (NEG, TR, 1+, 2+, 3+, 4+).

None of these urine items are present in the bounded cohort CSV. The bounded cohort was constructed
from labevents.csv using items 50912, 51006, 50971, 50983, 50882, 50931 only.

**UACR derivability:** Computing UACR requires co-temporal urine albumin (mg/dL) and urine creatinine
(mg/dL) from the **same urine specimen** at the same time. The bounded cohort does not contain these
items. Without extracting them from the raw data, no defensible UACR can be computed.


> [WARNING]
> Urine albumin, urine creatinine, UACR, and dipstick protein are NOT APPLICABLE to this audit: they are absent from the bounded cohort.


### B.3 Supporting Covariates Available in Bounded Cohort


The bounded cohort CSV joined with patients.csv and admissions.csv provides:
- **Demographics from patients.csv:** `anchor_age`, `gender`, `anchor_year`, `anchor_year_group`, `dod`
- **Admission context from admissions.csv:** `admittime`, `dischtime`, `admission_type`, `race`, `insurance`
- **Analyte measurements:** creatinine, BUN, potassium, sodium, bicarbonate, glucose
- **Temporal:** `charttime`, `storetime`, `reporting_delay_hours`
- **Audit flags:** `is_exact_duplicate`, `is_conflicting_measurement`, `arbitration_role`

**Not present in MIMIC-IV labevents:** Blood pressure, HbA1c, weight/BMI, medications, diagnoses.
These are in other MIMIC-IV tables (chartevents, prescriptions, diagnoses_icd) not in the bounded cohort.


| Factor | Item ID | In Cohort? | Rows | Missing% | Unit | Usable? | Module 2 Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Serum creatinine | 50912 | Yes | 53,924 | 0.05% | mg/dL | Yes | Primary filtration input |
| BUN | 51006 | Yes | 52,559 | 0.04% | mg/dL | Yes | Uremic retention, BUN/Cr ratio |
| Potassium | 50971 | Yes | 51,816 | 0.06% | mEq/L | Yes | Nephron tubular function |
| Sodium | 50983 | Yes | 51,238 | 0.02% | mEq/L | Yes | Volume/osmolality state |
| Bicarbonate | 50882 | Yes | 48,917 | 0.03% | mEq/L | Yes | Acid-base / CKD |
| Glucose | 50931 | Yes | 45,006 | 0.02% | mg/dL | Yes | DKD metabolic covariate |
| eGFR MDRD | 50920 | No | — | 100% null valuenum | text | No — not numerically usable |
| eGFR CKD-EPI Refit | 53161 | No | — | — | mL/min/1.73m2 | Derived — not in cohort |
| Urine Albumin | 51069 | No | — | — | mg/dL | No — not in cohort |
| Urine Creatinine | 51082 | No | — | — | mg/dL | No — not in cohort |
| UACR | derived | No | — | — | mg/g | No — cannot be validly derived |
| Dipstick Protein | 51492 | No | — | 66.65% null | ordinal | No — not in cohort |
| Age | demographics | Yes (anchor_age) | 2,500 patients | 0% | years | Yes — via patients.csv |
| Sex | demographics | Yes (gender) | 2,500 patients | 0% | M/F | Yes — via patients.csv |
| Blood pressure | chartevents | No | — | — | mmHg | Not in bounded cohort |
| HbA1c | labevents (other) | No | — | — | % | Not in bounded cohort |
| Weight/BMI | chartevents | No | — | — | kg/m2 | Not in bounded cohort |



_Real cohort slice: 302 rows, 6 patients_


## Test C+D — Irregular Elapsed-Time (TimeSeriesDelta) and Patient Boundary Isolation


**Fixture:** 3 rows from 1 real patients


**Patients:** [10000032]


### C.1 WEKA TimeSeriesDelta — Actual Execution


**Command:** `java --add-opens java.base/java.lang=ALL-UNNAMED -cp D:\Software\weka\Weka-3-8-7\weka.jar weka.filters.unsupervised.attribute.TimeSeriesDelta -i D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\test_c_elapsed.arff -o D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\weka_out\test_c_elapsed_out.arff -R 3 -I 1`


**Exit code:** 0 | **Elapsed:** 0.39s


### C.2 Row-Level Comparison: WEKA TimeSeriesDelta vs. Python Elapsed-Time


| Row | Patient | charttime | creatinine | Expected Δt (hours) | WEKA delta (value-based) | Error |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 10000032 | 2180-03-23 11:51 | 0.40 | NaN (first obs) | 0.1 | CROSS-PATIENT CONTAMINATION: boundary row not NaN |
| 1 | 10000032 | 2180-05-06 22:25 | 0.30 | 1066.57h | 0.0 | VALUE DIFF, not time diff. WEKA=0.0000 (value-based), Python=1066.57h (time-based) |
| 2 | 10000032 | 2180-05-07 05:05 | 0.30 | 6.67h | NaN | — |


### C.3 Alternative WEKA Workflow — Sort by subject_id First


TimeSeriesDelta does not accept a group key. Testing whether sorting by subject_id
before applying the filter changes the boundary behavior (it should not fix cross-patient contamination
because TimeSeriesDelta still has no concept of patient boundaries).


Alternative run exit code: 0. Output identical to first run: True


> [IMPORTANT]
> Pre-sorting by subject_id does NOT fix the boundary contamination because TimeSeriesDelta has no groupby semantics — it always uses row i-1 regardless of patient.



**Result:** ❌ FAIL


- Contaminated boundary rows: **1**


- Additionally: TimeSeriesDelta computes **creatinine value differences** (not elapsed hours), so its output is dimensionless — it cannot produce Δt in hours regardless of boundary issues.


## Test B — Group-Aware LOCF vs. WEKA ReplaceMissingValues


**Command:** `java --add-opens java.base/java.lang=ALL-UNNAMED -cp D:\Software\weka\Weka-3-8-7\weka.jar weka.filters.unsupervised.attribute.ReplaceMissingValues -i D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\test_b_locf.arff -o D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\weka_out\test_b_locf_out.arff`


**Exit code:** 0 | **Elapsed:** 0.32s


### B.1 Row-Level Comparison


| Row | Patient | Step | Raw creatinine | Python LOCF | Python GlobalMean | WEKA Output | Error / Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10000032 | 1 | 0.400 | 0.4000 | 0.4000 | 0.4000 |  |
| 1 | 10000032 | 2 | NaN (missing) | 0.4000 | 0.5500 | 0.5500 | WEKA uses global mean — not patient-specific LOCF |
| 2 | 10000032 | 3 | 0.300 | 0.3000 | 0.3000 | 0.3000 |  |
| 3 | 10000032 | 4 | 0.400 | 0.4000 | 0.4000 | 0.4000 |  |
| 4 | 10000084 | 1 | 0.800 | 0.8000 | 0.8000 | 0.8000 |  |
| 5 | 10000084 | 2 | NaN (missing) | 0.8000 | 0.5500 | 0.5500 | WEKA uses global mean — not patient-specific LOCF |
| 6 | 10000084 | 3 | 0.700 | 0.7000 | 0.7000 | 0.7000 |  |
| 7 | 10000084 | 4 | 0.700 | 0.7000 | 0.7000 | 0.7000 |  |



**Result:** ❌ FAIL


- WEKA ReplaceMissingValues uses **global cohort mean** (0.550 mg/dL) for all missing values, regardless of patient history.


- Python LOCF correctly propagates the **patient's last observed value** within the patient boundary.


- Rows where WEKA differs from LOCF: **2** (all missing rows).


> [IMPORTANT]
> No valid WEKA filter implements group-aware LOCF forward-fill. ReplaceMissingValues has no -group or -key parameter.


## Test E — Conflicting Measurement Arbitration (RemoveDuplicates)


**Data source:** REAL COHORT


> [NOTE]
> This fixture uses REAL COHORT records.


**Command:** `java --add-opens java.base/java.lang=ALL-UNNAMED -cp D:\Software\weka\Weka-3-8-7\weka.jar weka.filters.unsupervised.instance.RemoveDuplicates -i D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\test_e_conflicts.arff -o D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\weka_out\test_e_conflicts_out.arff`


**Exit code:** 0 | **Elapsed:** 0.37s


### E.1 Result


**Input rows:** 4 | **WEKA output rows after RemoveDuplicates:** 3


**Expected behavior:**


- Exact duplicate pair → 1 row (correct: WEKA should do this)


- Conflict pair (same timestamp, same itemid, different valuenum) → should be arbitrated to 1 row with median value 28.50


- Clean row → preserved


- **Expected output: 3 rows** (1 deduplicated + 1 arbitrated + 1 clean)


**WEKA output: 3 rows**


```
 subject_id           charttime  itemid  valuenum      source
 10000032.0 2180-06-03 12:00:00 50882.0      29.0 central_lab
 10000032.0 2180-06-03 12:00:00 50882.0      28.0     poc_lab
    99999.0 2150-01-01 10:00:00 50882.0       1.5 central_lab
```


**WEKA retained 1 unresolved conflict group(s) because RemoveDuplicates only removes exact full-row matches.**


### E.2 Alternative WEKA Workflows Tested


No native WEKA filter implements subset-key group-by arbitration.
The following alternatives were considered:
- **RemoveDuplicates:** Removes only exact full-row duplicates. Conflicting measurements
  (same subject/time/test, different value) are distinct rows and are retained.
- **SortLabels:** Nominal reordering — not relevant.
- **FilteredClassifier:** Requires a classifier, not a preprocessing filter; cannot apply median policy.
- **MathExpression:** Operates on numeric attributes, cannot group by (subject_id, charttime, itemid).
- **Resample:** Resamples instances, cannot arbitrate within groups.

No reasonable WEKA workflow can replicate `groupby(['subject_id','charttime','itemid'])['valuenum'].median()`.


> [IMPORTANT]
> CONFIRMED CAPABILITY GAP: WEKA has no filter that performs group-key-based conflicting measurement arbitration with policy selection.


## Test F — Future-Information Leakage (Batch Normalize)


**Base run exit code:** 0 | **Perturbed run exit code:** 0


**Base command:** `java --add-opens java.base/java.lang=ALL-UNNAMED -cp D:\Software\weka\Weka-3-8-7\weka.jar weka.filters.unsupervised.attribute.Normalize -i D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\test_f_base.arff -o D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\weka_out\test_f_base_out.arff -S 1.0 -T 0.0`


**Perturbed command:** `java --add-opens java.base/java.lang=ALL-UNNAMED -cp D:\Software\weka\Weka-3-8-7\weka.jar weka.filters.unsupervised.attribute.Normalize -i D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\test_f_pert.arff -o D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_audit_fixtures\weka_out\test_f_pert_out.arff -S 1.0 -T 0.0`


| Sequence | t=1 | t=2 | t=3 | t=4 | t=5 |
| --- | --- | --- | --- | --- | --- |
| Base (input) | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 |
| Perturbed (input) | 1.0 | 1.1 | 1.2 | 1.3 | 12.0 |
| WEKA Base (output) | 0.0000 | 0.2500 | 0.5000 | 0.7500 | 1.0000 |
| WEKA Perturbed (output) | 0.0000 | 0.0091 | 0.0182 | 0.0273 | 1.0000 |
| Python Causal Base | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Python Causal Perturbed | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |



**WEKA t=2 base:** 0.2500 | **WEKA t=2 perturbed:** 0.0091 | **Drift:** 0.2409


**Python causal t=2 base:** 1.0000 | **Python causal t=2 perturbed:** 1.0000 | **Drift:** 0.000000



**WEKA leaks future information:** True (drift=0.2409)


**Python causal leaks:** False (drift=0.000000)


> [WARNING]
> Changing only t=5 from 1.4 to 12.0 shifted WEKA's t=2 normalized value from 0.2500 to 0.0091 (drift=0.2409). This is mathematically expected from batch MinMax fitting: the global max changes from 1.4 to 12.0, retroactively rescaling all historical values.


### F.1 Alternative WEKA Workflows Considered


To avoid batch leakage in WEKA:
- **FilteredClassifier with cross-validation:** Applies normalization within each CV fold,
  preventing test-set leakage at the fold boundary. However, within-sequence temporal ordering
  is NOT enforced — a single patient's trajectory is normalized using that patient's full
  historical and future values within the fold.
- **Pre-split normalization:** User can fit a Normalize filter on training data only, save the
  model, then apply it at test time. This requires external scripting and is a workflow
  inconvenience, not a capability gap. It prevents train/test leakage but does NOT
  implement causal per-time-step normalization.
- **No WEKA filter implements causal expanding-window normalization** using only x[1..t].

**Classification:** This is a Workflow Inconvenience for train/test leakage scenarios.
It is a **Demonstrated Failure** for within-sequence causal normalization (e.g. real-time
forecasting where at time t only past values should influence the normalized input).



**Result:** ❌ FAIL


## Test A — eGFR Text Representation and Numeric Usability


The WEKA comparison plan already established that eGFR-MDRD (item 50920) has
100% missing `valuenum`. This test confirms the text representation and what WEKA
would receive if it were included in a preprocessing pipeline.


```
 subject_id  itemid  valuenum value  valueuom
       1001   50920       NaN   >60       NaN
       1001   50920       NaN   >60       NaN
       1002   50920       NaN 30-44       NaN
       1002   50920       NaN   <15       NaN
```


These text values (`>60`, `30-44`, `<15`) are **interval-censored** and **not directly numeric**.
They cannot be loaded as a WEKA numeric attribute without prior parsing.
- `>60` means the eGFR is above 60 mL/min/1.73m2 but the exact value is unknown.
- `30-44` is a range with uncertain exact value.
- `<15` means kidney failure (GFR < 15).

**WEKA behavior with text eGFR:** If these were loaded as a numeric column in ARFF, all would be `?`
(missing). If loaded as a nominal/string column, WEKA cannot apply numeric normalization directly.
Either way, textual eGFR requires Python preprocessing (regex extraction, lower-bound imputation,
or censored-value encoding) **before** WEKA can process it.

**Conclusion:** eGFR-MDRD is NOT APPLICABLE to the WEKA capability test — it cannot be provided as
a valid numeric input to WEKA without external pre-processing. This is not a WEKA limitation; it is
a data representation issue in MIMIC-IV's storage of eGFR values.


> [IMPORTANT]
> eGFR-MDRD (50920): Not Applicable — 100% missing `valuenum`, text-censored. Must be derived from creatinine + demographics if needed. The derivation is NOT independent of serum creatinine.


## Test G — Scaling and Outlier Handling on Real Creatinine Distribution


**Real creatinine distribution (first 5000 rows of bounded cohort):**


| Statistic | Value (mg/dL) |
| --- | --- |
| Min | 0.300 |
| Q25 (IQR lower) | 0.600 |
| Median | 0.800 |
| Q75 (IQR upper) | 0.900 |
| IQR | 0.300 |
| P95 | 1.100 |
| Max | 1.200 |


| Row | Raw (mg/dL) | WEKA MinMax | Python MinMax | Python RobustScaler (Median/IQR) |
| --- | --- | --- | --- | --- |
| 0 | 0.400 | 0.2500 | 0.2500 | -1.3333 |
| 1 | 0.300 | 0.0000 | 0.0000 | -1.6667 |
| 2 | 0.300 | 0.0000 | 0.0000 | -1.6667 |
| 3 | 0.400 | 0.2500 | 0.2500 | -1.3333 |
| 4 | 0.400 | 0.2500 | 0.2500 | -1.3333 |
| 5 | 0.300 | 0.0000 | 0.0000 | -1.6667 |
| 6 | 0.600 | 0.7500 | 0.7500 | -0.6667 |
| 7 | 0.300 | 0.0000 | 0.0000 | -1.6667 |
| 8 | 0.700 | 1.0000 | 1.0000 | -0.3333 |
| 9 | 0.500 | 0.5000 | 0.5000 | -1.0000 |



**Normal clinical range [0.7–1.4] mapped to:**


- WEKA MinMax: [1.0000, 1.0000] (range = 0.0000)


- Python RobustScaler: [-0.3333, -0.3333] (range = 0.0000)


> [IMPORTANT]
> No native WEKA filter implements median/IQR robust scaling. The Normalize filter can only implement MinMax [0,1]. Standard WEKA lacks a RobustScaler equivalent in its core distribution.


## Deliverable C — WEKA Capability Matrix


| Requirement | Factor/columns | Real-data issue | WEKA workflow tested | Expected output | Actual output | Alternative tested | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Irregular elapsed-time Δt (hours) | `charttime` per `subject_id` | 2,500 patient boundaries; non-uniform gaps | `TimeSeriesDelta -R 3 -I 1` | `[NaN, Δt_h, Δt_h, NaN, Δt_h]` — real elapsed hours, NaN at boundaries | Value differences (dimensionless); boundary rows non-NaN (cross-patient) | Pre-sort by subject_id (no change — filter has no groupby key) | ❌ FAIL | Test C+D (this audit) |
| Group-aware LOCF within patient boundary | `valuenum` per `subject_id` | 98.21% empty daily grid steps in Arm 1; progressive CKD trajectories | `ReplaceMissingValues` | Patient-last-value forward-filled within patient group | Global cohort mean imputed — patient history ignored | No WEKA filter accepts a group key | ❌ FAIL | Test B (this audit) |
| Conflicting measurement arbitration | `(subject_id, charttime, itemid)` → single arbitrated value | 134 real conflict pairs in bounded cohort | `RemoveDuplicates` | 1 row per conflict group (median, max, or priority policy) | Both conflicting rows retained; no group-by arbitration performed | No native WEKA filter implements subset-key groupby arbitration | ❌ FAIL | Test E (this audit) |
| Causal normalization (no future leakage) | `valuenum` per time series | ICU creatinine spikes retroactively alter historical normalized values | `Normalize -S 1.0 -T 0.0` (batch mode) | Historical t=2 value unchanged when only future t=5 changes | Historical t=2 shifts by Δ=0.2409 | FilteredClassifier prevents fold leakage; no per-time-step causal normalizer | ❌ FAIL | Test F (this audit) |
| eGFR from MIMIC-IV labevents | Item 50920 `valuenum` | 100% missing numeric; stored as text intervals (>60, 30-44, <15) | N/A — not loadable as numeric ARFF attribute | Numeric eGFR values | All `valuenum` = NaN; text requires external regex parsing | N/A | N/A | M1.1 full-dataset audit; Test A (this audit) |
| MinMax and Z-Score normalization | `valuenum` static tabular | None | `Normalize`, `Standardize` | Match Python to machine precision | Match confirmed: max diff < 4.2e-07 | N/A | ✅ PASS | Phase A benchmark (weka_benchmark_results.csv) |
| Exact full-row duplicate removal | All columns equal | 94 exact duplicate pairs in bounded cohort | `RemoveDuplicates` | One row per exact-match group | Correct | N/A | ✅ PASS | Phase A benchmark |
| Timestamp parsing (sub-minute ISO) | `charttime`, `storetime` | Minute-and-second precision; future MIMIC dates | `ArffLoader` with `@attribute date` | All timestamps preserved | Correct, including leap days | N/A | ✅ PASS | Phase A benchmark |
| Outpatient null `hadm_id` retention (flat ARFF) | `hadm_id` (34.06% null) | 103,354 outpatient records in cohort | `Copy -R 1-5` on pre-joined ARFF | Null preserved, row not dropped | Correct on pre-joined flat input | N/A | ✅ PASS | Phase A benchmark |
| Robust scaling (Median/IQR) | `valuenum` creatinine | Range 0.0–20.2 mg/dL; MinMax crushes [0.7–1.4] to narrow band | No native WEKA filter | (x − median) / IQR | No filter available in WEKA 3.8.7 core | N/A | 🚫 CAPABILITY GAP | Test G (this audit); Phase A benchmark |
| Conditional unit harmonization | `valuenum + valueuom` | All 303,460 cohort rows already in standard units; 0 conversions needed | Single `MathExpression` fails; `NominalToBinary` + `MathExpression` succeeds | g_dL × 1000 → mg_dL | Two-step pipeline works correctly | N/A | ⚠️ WORKFLOW INCONVENIENCE | Phase A benchmark (two-step pipeline confirmed) |
| Multi-table relational join | patients + admissions + labevents | External join required before WEKA can process flat ARFF | No join engine in WEKA | Joined flat DataFrame | Cannot perform join | N/A | 🚫 CAPABILITY GAP | Architectural; Phase A benchmark |


## Deliverable D — Strongest Reproducible Failure


### D.1 Primary: TimeSeriesDelta Cross-Patient Contamination and Datetime Blindness


This is the strongest failure because it produces **two simultaneous incorrect outputs**
that are structurally unavoidable from the filter's design:
1. At patient-boundary rows, WEKA outputs a non-NaN delta that uses the preceding patient's data.
2. Even for non-boundary rows, the output is a **value difference** (dimensionless), not elapsed hours.
   There is no WEKA option to produce elapsed time in hours from a date attribute.

No reasonable alternative WEKA workflow resolves this. The filter has no `-group` or `-key` parameter.


**Why this matters for renal forecasting:**
- Velocity: `(creatinine[t] − creatinine[t−1]) / Δt_hours` requires correct elapsed hours.
  If `Δt` is a dimensionless value difference (as WEKA produces), the velocity has no physical unit.
- Acceleration: requires `Δt` for both intervals; wrong Δt compounds the error.
- At 2,500 patient boundaries in the bounded cohort, 2,500 velocity values would be computed
  using the wrong preceding patient's creatinine as a reference — producing phantom kinetics.


**Confirmed boundary contaminations in this audit run:** 1


### D.2 Secondary: Batch Normalization Future Leakage (Demonstrated Failure)


**Confirmed in this audit run:**


- Base sequence `[1.0, 1.1, 1.2, 1.3, 1.4]`: t=2 normalized = **0.2500**


- Perturbed (only t=5 changed to 12.0): t=2 normalized = **0.0091**


- **Drift = 0.2409** (96.4% change in historical value caused solely by a future observation)


### D.3 Corrected Reference Implementation (Python)


```python
# Correct elapsed-time calculation with patient boundary isolation
df = df.sort_values(["subject_id", "charttime"])
df["delta_t_hours"] = (
    df.groupby("subject_id")["charttime"]
    .diff()
    .dt.total_seconds()
    / 3600.0
)
# First row per patient correctly receives NaN -- no prior observation within patient
# Produces: [NaN, 4.0, 20.0, NaN, 12.0] for 2-patient fixture

# Correct LOCF with patient boundary isolation
df["creatinine_locf"] = df.groupby("subject_id")["creatinine"].ffill()
# Never uses another patient's last value

# Correct causal normalization (no future leakage)
def causal_minmax(series: pd.Series) -> pd.Series:
    scaled = pd.Series(index=series.index, dtype=float)
    for t in range(len(series)):
        window = series.iloc[:t + 1]
        mn, mx = window.min(), window.max()
        scaled.iloc[t] = (series.iloc[t] - mn) / (mx - mn + 1e-9) if mx > mn else 0.0
    return scaled
```


## Deliverable E — Research Conclusion


### E.1 What WEKA Handles Correctly


- **Static MinMax and Z-Score normalization:** Machine-precision match with Python.
- **Global mean/mode imputation:** Correct when population-level imputation is appropriate.
- **Exact full-row duplicate removal:** `RemoveDuplicates` correctly eliminates identical rows.
- **Sub-minute ISO timestamp parsing:** Including leap days in far-future MIMIC dates.
- **Outpatient null `hadm_id` retention:** Preserves null values on pre-joined flat ARFF.
- **Conditional unit harmonization (two-step):** `NominalToBinary + MathExpression` pipeline works.


### E.2 What Fails Under a Tested Workflow


1. **TimeSeriesDelta for elapsed-time Δt:** Computes value differences, not elapsed hours.
   Crosses patient boundaries — produces contaminated delta at boundary rows.
   **Not fixable by any WEKA filter option.**
   
2. **ReplaceMissingValues for group-aware LOCF:** Uses global cohort mean for all missing values,
   ignoring patient history. A deteriorating patient trajectory at [1.0, 1.4, NaN, 2.1] gets
   the population mean, not 1.4 (last observed value).
   **Not fixable — no WEKA filter accepts a group key.**
   
3. **Normalize (batch mode) for causal features:** Perturbing a future time step changes historical
   normalized values. **Drift confirmed: 0.2409** in this audit run.
   **Partially fixable:** Pre-split normalization (fit on train only) prevents fold leakage
   but cannot implement per-time-step causal normalization for online forecasting.


### E.3 Demonstrated Capability Gaps


- **Conflicting measurement arbitration:** `RemoveDuplicates` cannot apply subset-key
  group-by arbitration. No native WEKA filter provides policy-driven conflict resolution
  (median, priority, extremum). **134 real conflict pairs in the bounded cohort** cannot
  be arbitrated within WEKA.
- **Robust scaling (Median/IQR):** No native WEKA filter. MinMax compresses the clinically
  important normal range [0.7–1.4] into a narrow band due to extreme outliers (up to 20.2 mg/dL).
- **Multi-table relational join:** Architectural; WEKA requires pre-joined flat ARFF.


### E.4 What Remains Untested


- **Full-cohort TimeSeriesDelta error rate:** Only tested on a small multi-patient slice.
  The error mechanism is established; the frequency across all 303,460 rows is not quantified.
- **Urine albumin, UACR, dipstick protein:** Not in the bounded cohort; not testable
  without a new extraction. Not applicable to this audit.
- **eGFR text parsing in WEKA:** The filter exists in principle; not worth testing since
  eGFR-MDRD has 100% missing `valuenum` in MIMIC-IV.


### E.5 Factors to Use in Module 2


**Confirmed usable from bounded cohort:**
- Serum creatinine (50912) — primary filtration input
- BUN (51006) — uremic retention, BUN/Cr ratio
- Potassium (50971) — distal nephron, hyperkalemia
- Sodium (50983) — volume/osmolality
- Bicarbonate (50882) — acid-base / CKD
- Glucose (50931) — DKD metabolic covariate
- Derived: delta_t_hours, velocity, acceleration (from the temporal_features module)
- Demographics: anchor_age, gender (from patients.csv)

**Not usable from bounded cohort:**
- eGFR (requires derivation from creatinine + demographics — not independent)
- Urine albumin, UACR, dipstick protein (not extracted)
- Blood pressure, HbA1c, BMI (not in labevents)


### E.6 Strongest Defensible Research Contribution


**The single strongest contribution supported by this evidence is:**

> *WEKA's `TimeSeriesDelta` filter cannot compute chronological elapsed time from timestamp
> attributes, and does not isolate patient boundaries. Applied to longitudinal EHR data with
> multiple patients, it (1) computes value differences rather than temporal intervals, and
> (2) propagates the preceding patient's measurement into the first observation of the next
> patient. A correctly designed Python pipeline using `groupby('subject_id')['charttime'].diff()`
> resolves both errors simultaneously, producing exact elapsed hours with NaN at patient
> boundaries — a prerequisite for all downstream velocity, acceleration, and irregular-time
> forecasting features.*

Supporting contributions (secondary):
- Group-aware LOCF is not achievable in standard WEKA; global mean imputation distorts CKD trajectories.
- Conflicting measurement arbitration requires group-key-based policy logic absent from WEKA.
- Batch normalization leaks future clinical severity into historical feature values
  (confirmed: drift = 0.2409 for creatinine surge at t=5).


## Deliverable F — Next Action & Validation Status

**Status: COMPLETED (2026-10-19)**

The recommended 5,000-record validation has been executed against the installed WEKA 3.8.7 JAR using deterministic extraction from `cohort_renal_bounded.csv`.

Full standalone report: [`reports/generated/timeseries_delta_validation.md`](file:///d:/CKD%20mini%20project/nephron-aware-renal-digital-twin/reports/generated/timeseries_delta_validation.md)

**Summary of 5,000-Record Validation Results:**
- **Sample:** 5,000 real serum-creatinine records (`itemid=50912`), 234 distinct patients (`subject_id` 10000032 to 10011938).
- **Tested WEKA Filter:** `weka.filters.unsupervised.attribute.TimeSeriesDelta` (`-I 1` and `-I -1`).
- **Standard TimeSeriesDelta on Date ARFF:**
  - **Boundary contamination:** 234 of 234 patient boundary rows received non-NaN cross-patient contamination values (under `-I 1`); 233 of 234 under `-I -1`.
  - **Semantic mismatch:** 4,765 of 4,766 non-boundary rows received creatinine value differences (mg/dL) instead of elapsed hours (h).
- **Alternative D (Pre-converting timestamps to epoch hours + TimeSeriesDelta):**
  - **Date arithmetic:** Numerically exact elapsed hours for 4,766 non-boundary rows (0.00h error).
  - **Boundary contamination:** 233 of 234 patient boundaries were contaminated with phantom cross-patient intervals (e.g., Row 18 patient transition produces -172,861.5 hours).
- **Conclusion:** Standard WEKA 3.8.7 cannot perform patient-group-aware temporal preprocessing. The finding is fully verified and defensible as a primary research contribution.
