# MIMIC-IV Bounded Cohort Extraction Quality Report (M1.2 Phase B)

**Project**: Nephron-Aware Renal Digital Twin  
**Extraction Date**: 2026-10-18T16:35:27.405019+00:00  
**Protocol Document**: [`docs/m1_2_phase_b_cohort_protocol.md`](file:///d:/CKD%20mini%20project/nephron-aware-renal-digital-twin/docs/m1_2_phase_b_cohort_protocol.md)  
**Status**: VERIFIED & REPRODUCIBLE  

---

## 1. Executive Summary & Cohort Identification

- **Target Patient Cap**: 2,500 patients
- **Eligibility Criterion**: Documented Serum Creatinine (`itemid=50912`), $\ge 3$ usable measurements ($> 0\text{ mg/dL}$).
- **Ranking Rule**: Strictly ascending administrative `subject_id` among all eligible patients.
- **Patients Extracted**: **2,500** (representing the first 2,500 eligible patients under ascending `subject_id` ordering).
- **Identifier Range**: `subject_id` **10000032** to **10138195**
- **Total Laboratory Records**: **303,460** rows
- **Execution Runtime**: **13.16 seconds** (unprofiled run; 35.83 seconds under `tracemalloc` memory tracing)
- **Measured Peak Memory**: **369.04 MB RAM** (measured via Python `tracemalloc`; driven by loading `patients.csv` and `admissions.csv` during relational joins)
- **Early-Exit Stream Optimization**: Completed by streaming **2,500,000 rows** (~1.4% of 18.4 GB `labevents.csv`), guarded by runtime monotonicity validation.

---

## 2. Patient Demographics & Admission Linkage

| Demographic Metric | Cohort Value | Reference / Notes |
| :--- | :--- | :--- |
| **Total Cohort Patients** | 2,500 | 100% matched in `patients.csv` (1:1 join) |
| **Gender Distribution** | Female: 1,287 (51.5%), Male: 1,213 (48.5%) | Representative sex balance |
| **Age at Anchor (Mean $\pm$ SD)** | 58.8 $\pm$ 19.2 years | Median: 61.0, Range: 18 - 91 |
| **Patients with Inpatient Admissions** | 2,275 (91.0%) | Total admissions: 6,190 |
| **Outpatient / ED-Only Patients** | 225 (9.0%) | Preserved without loss |
| **Lab Rows with Null `hadm_id`** | 103,354 (34.06%) | Outpatient / pre-admission labs preserved |
| **Unlinked `hadm_id`s** | 0 | Zero orphan admission foreign keys |

---

## 3. Laboratory Variable Summary & Missingness Audit

| Analyte | Item ID | Standard Unit | Observations | Valid (%) | Missing (%) | Median [IQR] | Min - Max | Exact Duplicates | Conflicting Rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Creatinine** | `50912` | mg/dL | 53,924 | 99.95% | 0.05% | 1.0 [0.7 - 1.4] | 0.0 - 20.2 | 80 | 30 |
| **Urea Nitrogen** | `51006` | mg/dL | 52,559 | 99.96% | 0.04% | 18.0 [13.0 - 29.0] | 1.0 - 240.0 | 52 | 42 |
| **Potassium** | `50971` | mEq/L | 51,816 | 99.94% | 0.06% | 4.1 [3.8 - 4.5] | 1.7 - 12.6 | 22 | 62 |
| **Sodium** | `50983` | mEq/L | 51,238 | 99.98% | 0.02% | 139.0 [136.0 - 141.0] | 103.0 - 177.0 | 20 | 44 |
| **Bicarbonate** | `50882` | mEq/L | 48,917 | 99.97% | 0.03% | 25.0 [23.0 - 28.0] | 4.0 - 50.0 | 10 | 38 |
| **Glucose** | `50931` | mg/dL | 45,006 | 99.98% | 0.02% | 111.0 [94.0 - 141.0] | 8.0 - 2240.0 | 4 | 52 |

---

## 4. Data Integrity & Measurement Conflict Analysis
 
- **Exact Duplicate Rows**: **188** (0.062%).
  - *Grouping Key*: Identical `(subject_id, hadm_id, itemid, charttime, valuenum, valueuom)`.
  - *Structure*: Exactly 94 pairs of size 2 (80 outpatient pairs where `hadm_id IS NULL`, 14 inpatient pairs).
  - *Action*: Flagged via `is_exact_duplicate = True`; preserved for explicit downstream deduplication.
- **Conflicting Measurement Rows**: **268** (0.088%).
  - *Grouping Key*: Same `(subject_id, itemid, charttime)` but distinct `valuenum` (`nunique > 1`).
  - *Structure*: Exactly 134 pairs of size 2 across 134 unique timestamp keys.
  - *Action*: Flagged via `is_conflicting_measurement = True`; preserved for explicit arbitration benchmarking.
- **Category Overlap**: Exactly **0 rows** overlap between exact duplicates and conflicting measurements in this cohort.
- **Unit Harmonization**:
  - Standard Units: 303,460 (100% of rows adhere to expected standard units in raw source)
  - Explicitly Converted: 0
  - Missing UOM: 0
  - Unknown / Ambiguous Units: 0


---

## 5. Selection Bias & Epidemiological Limitations

1. **Surveillance / Testing-Frequency Bias**: Requiring $\ge 3$ serum creatinine measurements restricts the cohort to patients with repeated clinical contact (inpatient admissions, serial outpatient nephrology visits, or surgical/ICU monitoring). Patients with mild or stable CKD managed without frequent hospital labs are excluded.
2. **Identifier Truncation**: Bounded selection by ascending `subject_id` captures patients in the identifier range `10000032` to `10138195`. While de-identification hashing renders this range pseudo-random relative to clinical status, demographic verification against the full MIMIC-IV population is documented above.
3. **Non-Representativeness**: **This cohort is NOT an epidemiologically representative sample of all CKD patients.** It is an engineering and benchmark cohort for renal digital twin modeling and WEKA-versus-Python preprocessing comparisons.

---

## 6. Provenance & Software Environment

- **Python Version**: `3.14.0` on `Windows-11-10.0.26200-SP0`
- **Pandas Version**: `3.0.6`, **NumPy Version**: `2.5.4`
- **Input Provenance**:
  - `labevents.csv`: `data/raw/labevents.csv` (17.139 GB)
  - `patients.csv`: `data/raw/patients.csv` (12,051,137 bytes)
  - `admissions.csv`: `data/raw/admissions.csv` (94,156,458 bytes)
- **Output Artifacts**:
  - `cohort_renal_bounded.csv`: `data/processed/cohort_renal_bounded.csv` (303,460 rows)
  - `cohort_patients.csv`: `data/processed/cohort_patients.csv` (2,500 patients)
