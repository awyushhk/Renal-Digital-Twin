# M1.2 Phase B: Bounded MIMIC-IV Cohort Extraction Protocol

**Project**: Nephron-Aware Renal Digital Twin  
**Milestone**: M1.2 Phase B (Bounded Reproducible Cohort Extraction)  
**Target Dataset**: MIMIC-IV 3.1 (`data/raw/`)  
**Protocol Status**: APPROVED / ACTIVE  
**Last Updated**: 2026-10-18  

---

## 1. Executive Summary & Purpose

The objective of Milestone 1.2 Phase B is to define, specify, and implement a bounded, deterministic, and fully reproducible cohort extraction from MIMIC-IV 3.1. This cohort provides the standardized empirical substrate for:
1. Benchmarking Python versus WEKA preprocessing pipelines on real-world clinical records without exceeding computational memory limits.
2. Validating relational joins, irregular time series, unit harmonization, and measurement conflict handling on real intensive care and hospital laboratory data.
3. Supplying clean, audited longitudinal renal and metabolic trajectories for subsequent Renal Digital Twin modeling modules.

All source raw data in `data/raw/` remain **strictly read-only**. All derived datasets are written exclusively to `data/processed/` and are excluded from Git version control.

---

## 2. Cohort Definition & Selection Methodology

### 2.1 Target Patient Cap
The cohort is strictly bounded to at most **2,500 patients**. This size guarantees sufficient statistical diversity and clinical trajectory complexity (>300,000 lab events) while maintaining rapid end-to-end pipeline execution (<60 seconds) and low memory usage (<250 MB RAM).

### 2.2 Eligibility Anchor: Serum Creatinine (`itemid = 50912`)
Renal digital twin models require longitudinal assessment of glomerular filtration function. Serum creatinine serves as the universal clinical anchor:
- **Laboratory Item**: Creatinine, Blood, Chemistry (`itemid = 50912`).
- **Eligibility Gate**: A patient must have **at least three (≥3) usable serum creatinine observations**.
- **Definition of "Usable Observation"**:
  - `valuenum` is not null (`valuenum IS NOT NULL`).
  - `valuenum` is strictly positive (`valuenum > 0.0 mg/dL`).
  - Timestamp (`charttime`) is recorded and parseable.

Patients with fewer than 3 usable measurements, or only non-numeric / missing entries, are excluded from eligibility.

### 2.3 Deterministic Ranking and Selection Procedure
When the number of eligible patients exceeds the 2,500 cap (MIMIC-IV contains over 100,000 patients with ≥3 creatinine tests), selection is executed according to the following deterministic rule:
1. All patients meeting the eligibility criterion are identified.
2. Eligible patients are ranked strictly by **ascending administrative identifier** (`subject_id` non-decreasing).
3. The first **2,500 patients** in this ordered sequence are selected as the final cohort.

**Clarification on Cohort Definition**:
The extracted cohort represents **the first 2,500 eligible patients under the ascending administrative identifier selection rule**, bounded and accelerated by the physical ordering of the source table. It is not an enumeration of all ~100,000 eligible patients across all 158 million rows of MIMIC-IV.

**Scientific Justification for Identifier-Based Selection**:
- `subject_id` is a randomly assigned de-identified administrative integer generated during MIMIC-IV compilation.
- It possesses zero clinical correlation with patient severity, renal pathology, age, length of stay, or mortality.
- It completely eliminates outcome-based selection bias: **no future deterioration, laboratory trajectory, diagnosis code, or survival status is utilized to choose patients**.
- It guarantees 100% determinism: identical patient sets are extracted on every run across any computing platform.

### 2.4 Selection Bias & Epidemiological Caveats
The user and modelers must account for the following documented selection biases:
1. **Frequency-of-Testing (Surveillance) Bias**: Requiring ≥3 serum creatinine measurements enriches the cohort for patients who were admitted to inpatient wards, stayed in the intensive care unit (ICU), or presented with acute or complex chronic illness requiring serial monitoring. Mild or ambulatory outpatients who received only a single routine screening test are systematically excluded.
2. **Identifier Range Truncation**: Because MIMIC-IV `labevents.csv` is sorted in ascending `subject_id` order, selecting the first 2,500 eligible patients encompasses `subject_id`s between `10000032` and `10138195`. While de-identification hashing renders this range pseudo-random relative to clinical status, demographic verification against the full MIMIC-IV population is conducted to confirm representative distributions.
3. **Non-Representativeness Statement**: **This cohort is NOT an epidemiologically representative sample of the entire chronic kidney disease (CKD) population.** It does not represent early asymptomatic CKD managed in outpatient clinics without serial hospital labs. The cohort is intended solely for algorithmic benchmarking and digital twin prototype engineering.

---

## 3. Target Continuous Laboratory Variables

Phase B focuses strictly on the **six continuous blood chemistry analytes** validated during the M1.1 full dataset audit:

| Analyte | Source `itemid` | Fluid | Category | Standard Unit | Physiological Role in Renal Twin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Serum Creatinine** | `50912` | Blood | Chemistry | `mg/dL` | Primary endogenous marker of glomerular filtration rate (GFR). |
| **Blood Urea Nitrogen (BUN)** | `51006` | Blood | Chemistry | `mg/dL` | Urea clearance, catabolic status, and BUN/Cr volume-depletion ratio. |
| **Serum Potassium ($K^+$)** | `50971` | Blood | Chemistry | `mEq/L` | Distal tubule electrolyte regulation; hyperkalemia risk in low GFR. |
| **Serum Sodium ($Na^+$)** | `50983` | Blood | Chemistry | `mEq/L` | Extracellular volume, tubular handling, and osmoregulation. |
| **Serum Bicarbonate ($HCO_3^-$)** | `50882` | Blood | Chemistry | `mEq/L` | Acid-base balance and metabolic acidosis evaluation in renal decline. |
| **Serum Glucose** | `50931` | Blood | Chemistry | `mg/dL` | Diabetic kidney disease covariate and osmotic filtration load. |

### Excluded Variables (Deferred to Subsequent Phases)
The following candidate renal biomarkers are explicitly **excluded** from the initial Phase B extraction:
- **Urine Creatinine (`51082`, `51106`)**: Excluded until 24-hour versus spot urine collection semantics and volume normalization are validated.
- **Urine Albumin (`51069`) & Albumin/Creatinine Ratio (`51070`)**: Excluded until microalbuminuria reporting limits and unit ratios are validated.
- **Estimated GFR (`50920`, `53161`)**: Excluded from raw inputs; eGFR must be causally computed within the pipeline using standard 2021 CKD-EPI equations from harmonized serum creatinine, age, and sex rather than relying on legacy unharmonized MDRD electronic medical record strings.
- **Categorical Urine Protein Dipstick (`51492`)**: Excluded until non-numeric ordinal parsing and dipstick missingness protocols are audited.

---

## 4. Extraction Architecture & Data Integrity Standards

### 4.1 Streaming Ingestion, Ordering Validation & Measured Memory Bounds
`labevents.csv` is an 18.4 GB file with 158,374,764 records. Under no circumstances should the entire file be read into RAM:
- Reads are performed in chunks of **250,000 rows**.
- Chunk filtering applies immediately on `subject_id` and `itemid`.
- **Validation of Source Ordering**:
  - In MIMIC-IV 3.1, records in `labevents.csv` are exported from PostgreSQL ordered by `subject_id`.
  - Contiguity and monotonicity were empirically audited across the first 10,000,000 rows (20 chunks), confirming 0 non-monotonic boundary transitions and 0 non-contiguous patient reappearances.
  - Runtime validation checks `sids[0] < last_subject_id` and `(sids[:-1] <= sids[1:]).all()` on every streamed chunk.
- **Monotonic Early-Exit Optimization**: Because the stream is non-decreasing, once the maximum `subject_id` of the 2,500 selected cohort patients (`10138195`) is exceeded by the minimum `subject_id` of a subsequent chunk, the streaming reader terminates immediately. No patient with `subject_id <= 10138195` can ever appear in subsequent chunks. This achieves complete cohort extraction by scanning only 2.50 million rows (~1.4% of `labevents.csv`) in ~13.16 seconds.
- **Safe Fallback**: If source data ordering is non-monotonic or if `enforce_monotonic_early_exit = False`, the pipeline executes a safe full-stream scan across all 158 million rows (~12–15 minutes).
- **Measured Peak Memory**: Measured via Python `tracemalloc` at **369.04 MB RAM** (driven primarily by loading `patients.csv` with 364k rows and `admissions.csv` with 546k rows during relational joins).

### 4.2 Preservation of Source Data & Immutability
- Raw files in `data/raw/` are accessed in read-only mode (`mode='r'`).
- Original values (`value`, `valuenum`, `valueuom`, `charttime`, `storetime`, `flag`, `priority`) are **never overwritten, masked, or silently altered**.
- All derived variables (`valuenum_harmonized`, `unit_harmonized`, `unit_flag`, `is_exact_duplicate`, `is_conflicting_measurement`) are stored in distinct, dedicated columns.

### 4.3 Relational Joins & Null Identifier Preservation
1. **Patient Demographics Join (`patients.csv`)**:
   - Joined on `subject_id` (1:1 relation).
   - Validates that 100% of cohort `subject_id`s match a record in `patients.csv`.
   - Incorporates `gender`, `anchor_age`, `anchor_year`, `anchor_year_group`, and `dod`.
2. **Hospital Admission Join (`admissions.csv`)**:
   - Performed as a **LEFT JOIN** on `hadm_id` (or `[subject_id, hadm_id]`).
   - **Handling of Null `hadm_id`**: In MIMIC-IV, exactly 103,354 laboratory measurements (34.06%) occur in outpatient clinics or the emergency department before hospital admission, resulting in null `hadm_id`.
   - **Crucial Integrity Rule**: Rows with `hadm_id IS NULL` **must be fully preserved**. They must not be dropped by an inner join.
   - For rows with valid `hadm_id`, admission metadata (`admittime`, `dischtime`, `admission_type`, `race`, `hospital_expire_flag`) are attached. 100% of non-null `hadm_id` values resolve to `admissions.csv`.

### 4.4 Duplicate & Conflict Detection
Row duplicates are classified into two distinct clinical categories:
- **Exact Duplicates**: Rows sharing identical `(subject_id, hadm_id, itemid, charttime, valuenum, valueuom)`. Exactly 188 rows forming 94 pairs of size 2. Grouping requires `dropna=False` because 80 of the 94 pairs have null `hadm_id`. Flagged via boolean column `is_exact_duplicate`.
- **Conflicting Measurements**: Rows sharing identical `(subject_id, itemid, charttime)` but having disparate `valuenum` values. Exactly 268 rows forming 134 pairs of size 2 across 134 distinct timestamp keys. Flagged via boolean column `is_conflicting_measurement`.
- **Mutual Exclusivity in Extracted Cohort**: In this dataset, no timestamp key contains both identical and disparate values; overlap between exact duplicates and conflicting measurements is exactly 0 rows. Both types are flagged and preserved.


### 4.5 Unit Harmonization Policy
Units are harmonized using strict, tested conversion rules:
- **50912 (Creatinine)**: Standard unit is `mg/dL`. Conversion: `umol/L` $\div 88.42 \to \text{mg/dL}$.
- **51006 (Urea Nitrogen)**: Standard unit is `mg/dL`. Conversion: $\text{mmol/L} \times 2.801 \to \text{mg/dL}$.
- **50971 (Potassium)**: Standard unit is `mEq/L`. Conversion: $\text{mmol/L} \times 1.0 \to \text{mEq/L}$.
- **50983 (Sodium)**: Standard unit is `mEq/L`. Conversion: $\text{mmol/L} \times 1.0 \to \text{mEq/L}$.
- **50882 (Bicarbonate)**: Standard unit is `mEq/L`. Conversion: $\text{mmol/L} \times 1.0 \to \text{mEq/L}$.
- **50931 (Glucose)**: Standard unit is `mg/dL`. Conversion: $\text{mmol/L} \times 18.0182 \to \text{mg/dL}$.
- **Unknown or Missing Units**: If `valueuom` is unrecognized or null, `valuenum_harmonized` is set to null, and `unit_flag` is tagged as `UNKNOWN_UNIT` or `MISSING_UOM`. The pipeline **never guesses or substitutes arbitrary units**.

### 4.6 Missingness Visibility
- No missing values are interpolated, forward-filled, or resampled in Phase B.
- True missing observations (e.g. cancelled tests where `valuenum` is null) are retained in the dataset and reported in the quality audit.

---

## 5. Output Schemas & Deliverable Artifacts

### 5.1 Primary Extracted Dataset: `data/processed/cohort_renal_bounded.csv`
Contains all observed laboratory events for the 2,500 cohort patients across the 6 blood chemistry items:
- `labevent_id`: Unique MIMIC-IV laboratory event identifier.
- `subject_id`: De-identified patient identifier.
- `hadm_id`: Hospital admission identifier (nullable).
- `itemid`: Source laboratory test code.
- `charttime`: Timestamp of blood draw.
- `storetime`: Timestamp of laboratory result filing.
- `value`: Original text representation.
- `valuenum`: Original numerical value.
- `valueuom`: Original unit of measure.
- `flag`: Clinical abnormality flag (e.g. `abnormal` or null).
- `priority`: Test order priority (`ROUTINE`, `STAT`).
- `gender`, `anchor_age`, `anchor_year`, `anchor_year_group`, `dod`: Patient demographic metadata from `patients.csv`.
- `admittime`, `dischtime`, `admission_type`, `race`, `hospital_expire_flag`: Admission metadata from `admissions.csv` (nullable for outpatient labs).
- `valuenum_harmonized`: Standardized numerical value.
- `unit_harmonized`: Target standardized unit string.
- `unit_flag`: Integrity flag (`STANDARD`, `CONVERTED`, `UNKNOWN_UNIT`, `MISSING_UOM`).
- `is_exact_duplicate`: Boolean indicator for identical repeat rows.
- `is_conflicting_measurement`: Boolean indicator for disparate values at the same timestamp.

### 5.2 Patient Summary Dataset: `data/processed/cohort_patients.csv`
Contains patient-level attributes for the 2,500 cohort members:
- `subject_id`, `usable_creatinine_count`, `total_target_labs_count`, `anchor_age`, `gender`, `anchor_year`, `dod`, `has_admission`, `admission_count`.

### 5.3 Quality & Audit Deliverables
- `reports/generated/cohort_quality_report.md`: Comprehensive Markdown audit documenting cohort counts, join verification, missingness, duplicate rates, and selection bias statistics.
- `reports/generated/cohort_summary.csv`: Machine-readable tabular summary of variable counts, valid/missing percentages, and value ranges.
- `reports/generated/cohort_summary.json`: Provenance metadata including timestamps, software environment, and input file checksums.

---

## 6. Scope Boundaries & Unresolved Decisions

1. **No Prediction Models**: No risk prediction, time-to-event, or trajectory model is trained in Phase B.
2. **No Clinical Generalizability Claims**: The cohort is bounded to hospital/ICU-tested patients in MIMIC-IV and does not generalize to broader community CKD epidemiology.
3. **Phase A WEKA Capabilities**: The five unsupported Phase A benchmark requirements reflect the absence of direct equivalent transformation filters in standard WEKA 3.8.7 core, not proven mathematical impossibility.
4. **Causal MinMax Normalization**: Remains an unresolved benchmark question until a dedicated streaming window comparison between Python and WEKA is executed.
5. **Conflict Arbitration**: Disparate values at identical timestamps are flagged; specific clinical arbitration rules (e.g. latest storetime vs mean vs STAT priority) are left for downstream preprocessing module testing.
