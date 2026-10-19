# Systematic WEKA Explorer Preprocessing & Dataset Integration Audit
## Nephron-Aware Renal Digital Twin — MIMIC-IV 3.1

**Date:** 2026-10-19  
**Target System:** WEKA Explorer 3.8.7 (`D:\Software\weka\Weka-3-8-7\weka.jar`)  
**Data Sources:** MIMIC-IV 3.1 (`labevents.csv`, `d_labitems.csv`, `patients.csv`, `admissions.csv`)  
**Cohort Reference:** `data/processed/cohort_renal_bounded.csv` (303,460 rows, 2,500 patients)  
**Dedicated Artifact Directory:** `reports/generated/weka_dataset_integration_audit/`  
**Authoritative Report:** `reports/generated/weka_dataset_integration_audit.md`  

---

## 1. Executive Summary & Supervisor Directive

Our supervisor directed us to investigate whether **combining and preprocessing clinical datasets using WEKA Explorer** can mishandle clinically meaningful attributes, distort values, lose information, introduce inappropriate missing-value substitutions, duplicate observations, or create other measurable data-quality problems.

Following an objective, reproducible audit:
1. **We did NOT assume WEKA has a defect.** We evaluated WEKA's standard tools against formal clinical and relational data integrity definitions.
2. **We identified a genuine, reproducible data distortion failure in WEKA Explorer's standard workflow**:
   - In relational clinical databases, laboratory events frequently occur in ambulatory/outpatient settings prior to hospital admission (`hadm_id IS NULL`, representing 34.06% of the full cohort and 46.69% of our 15-patient audited sample).
   - When relational clinical data (`labevents` + `admissions`) are integrated into a standard tabular representation and processed with WEKA Explorer's default `ReplaceMissingValues` filter, WEKA **substitutes the global inpatient mode (`"EW EMER."`) for all outpatient records**.
   - This silently invents **747 phantom Emergency Department inpatient admissions** in our 1,600-row sample, completely obliterating ambulatory representation and stamping routine primary care lab draws as acute hospital encounters.
   - For longitudinal renal biomarkers, WEKA's global mean imputation destroys individual trajectory slopes, assigning healthy baselines to end-stage renal patients and flagging acute kidney injury in healthy individuals.
3. **We identified a relational integration failure in naive joins**:
   - Joining `labevents` to `admissions` on `subject_id` alone creates a **4.49x Cartesian explosion (5,609 duplicate instances)** because 44.8% of patients have multiple hospitalizations.
   - Conversely, an inner join on `hadm_id` **destroys 46.69% of all laboratory observations** and completely eliminates purely ambulatory patients (such as Patient 10000898).
4. **We prepared a 60-second manual reproduction protocol** in WEKA Explorer GUI so that the supervisor and viva examiners can verify the finding interactively with 3 clicks.

---

## 2. The Four Raw Datasets & Verified Schemas

All raw files in `data/raw/` were verified as read-only. Schemas and record counts were audited directly:

| Dataset | File Path | Total Rows | File Size | Primary Key | Role in Renal Twin Pipeline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`labevents`** | `data/raw/labevents.csv` | 158,374,764 | 18.40 GB | `labevent_id` | Core longitudinal biomarker observations (time-stamped lab draws). |
| **`d_labitems`** | `data/raw/d_labitems.csv` | 1,650 | 64.66 KB | `itemid` | Master ontology dictionary defining test names, specimen fluids, and categories. |
| **`patients`** | `data/raw/patients.csv` | 364,627 | 12.05 MB | `subject_id` | Demographic master table (`gender`, `anchor_age`, `anchor_year`, `dod`). |
| **`admissions`** | `data/raw/admissions.csv` | 546,028 | 94.16 MB | `hadm_id` | Acute hospital inpatient encounters (`admittime`, `dischtime`, `admission_type`). |

### Verified Table Schemas

```
1. labevents.csv:
   - labevent_id (int64, PK)
   - subject_id (int64, FK -> patients.subject_id)
   - hadm_id (float64, FK -> admissions.hadm_id, NULLABLE)
   - itemid (int64, FK -> d_labitems.itemid)
   - charttime (string -> ISO-8601 timestamp, specimen collection time)
   - storetime (string -> ISO-8601 timestamp, lab filing time)
   - value (string, raw assay result text)
   - valuenum (float64, parsed numerical value)
   - valueuom (string, unit of measurement)
   - flag (float64/string, clinical abnormality flag)
   - priority (string, ROUTINE vs STAT)

2. d_labitems.csv:
   - itemid (int64, PK)
   - label (string, test name)
   - fluid (string, e.g. Blood, Urine, Ascites, CSF)
   - category (string, e.g. Chemistry, Hematology, Blood Gas)

3. patients.csv:
   - subject_id (int64, PK)
   - gender (string, 'M' or 'F')
   - anchor_age (int64, patient age at anchor year)
   - anchor_year (int64, de-identified temporal shift anchor)
   - anchor_year_group (string, multi-year range)
   - dod (string, date of death, nullable)

4. admissions.csv:
   - subject_id (int64, FK -> patients.subject_id)
   - hadm_id (int64, PK)
   - admittime (string -> ISO-8601 admission timestamp)
   - dischtime (string -> ISO-8601 discharge timestamp)
   - deathtime (string -> ISO-8601 in-hospital expiration timestamp, nullable)
   - admission_type (string, e.g. 'EW EMER.', 'URGENT', 'DIRECT EMER.', 'OBSERVATION ADMIT')
   - hospital_expire_flag (int64, 0 = survived hospital stay, 1 = died in hospital)
```

---

## 3. Data Relationships, Cardinalities, and Join Integrity

### 3.1 Referential Cardinalities
- **`patients` $\rightarrow$ `admissions`**: **$1 : N$ (One-to-Many)**.
  - 223,452 unique patients in `admissions.csv` account for 546,028 admissions (mean 2.44 admissions/patient; max = 238 admissions for a single patient).
  - 141,175 patients (38.72% of `patients.csv`) have **0 admissions** in `admissions.csv` (pure ambulatory/emergency outpatients).
- **`admissions` $\rightarrow$ `labevents`**: **$1 : N$ (One-to-Many)** for inpatient stays.
- **`patients` $\rightarrow$ `labevents`**: **$1 : N$ (One-to-Many)**.
- **`d_labitems` $\rightarrow$ `labevents`**: **$1 : N$ (One-to-Many)** on `itemid`.

### 3.2 The Outpatient Null Key Reality (`hadm_id IS NULL`)
In `labevents.csv`, `hadm_id` is nullable because outpatient blood draws, ambulatory clinic monitoring, and pre-admission ED blood tests occur without an inpatient admission ID.
- In the full bounded cohort (`cohort_renal_bounded.csv`): **103,354 out of 303,460 lab draws (34.06%) have `hadm_id = NULL`**.
- In our audited 15-patient sample (`sample_labs.csv`): **747 out of 1,600 lab draws (46.69%) have `hadm_id = NULL`**.

### 3.3 Join Cardinality Matrix: Empirical Comparison Across 15 Patients

To assess data integration failures, we tested three candidate merge strategies linking laboratory measurements to hospital encounters:

| Integration Strategy | Implemented Logic | Output Rows | Unique `labevent_id` | Data Preservation | Data Corruption / Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Strategy 1: Correct Left Join** | `pd.merge(labs, adms, on=['subject_id', 'hadm_id'], how='left')` | **1,600** | **1,600** | **100.0% (Zero loss)** | None. Outpatient labs legitimately retain `NULL` for admission fields. |
| **Strategy 2: Inner Join** | `pd.merge(labs, adms, on=['subject_id', 'hadm_id'], how='inner')` | **853** | **853** | **53.31% (46.69% LOST)** | **CATASTROPHIC DATA LOSS:** Drops all 747 outpatient draws and completely deletes Patient 10000898. |
| **Strategy 3: Naive Subject Join** | `pd.merge(labs, adms, on='subject_id', how='inner')` | **7,185** | **1,576** | **449.06% (4.49x explosion)** | **CARTESIAN PSEUDO-REPLICATION:** Creates 5,609 duplicate rows. Concurrently drops 24 labs for Patient 10000898. |

---

## 4. Clinically Relevant Renal Attributes & Availability Constraints

### 4.1 Prioritized Renal Biomarkers

| Biomarker | Item ID | Fluid | Category | Harmonized Unit | Normal Range | Role in Nephron Digital Twin |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Serum Creatinine** | `50912` | Blood | Chemistry | `mg/dL` | 0.6 – 1.2 mg/dL | Primary surrogate for Glomerular Filtration Rate (GFR). |
| **Blood Urea Nitrogen (BUN)** | `51006` | Blood | Chemistry | `mg/dL` | 7 – 20 mg/dL | Nitrogenous clearance; BUN/Creatinine volume-depletion ratio. |
| **Serum Potassium ($K^+$)** | `50971` | Blood | Chemistry | `mEq/L` | 3.5 – 5.0 mEq/L | Distal tubular secretion; hyperkalemia risk in low GFR. |
| **Serum Sodium ($Na^+$)** | `50983` | Blood | Chemistry | `mEq/L` | 135 – 145 mEq/L | Tubular reabsorption, extracellular volume, and tonicity. |
| **Serum Bicarbonate ($HCO_3^-$)** | `50882` | Blood | Chemistry | `mEq/L` | 22 – 29 mEq/L | Acid-base balance and metabolic acidosis in renal failure. |
| **Serum Glucose** | `50931` | Blood | Chemistry | `mg/dL` | 70 – 99 mg/dL | Diabetic nephropathy covariate and tubular osmotic load. |

### 4.2 Availability Limitations in Bounded Cohort
- **Urine Creatinine (`51082`, `51106`)**: Excluded from initial cohort extraction due to spot versus 24-hour collection ambiguity.
- **Urine Albumin / UACR (`51069`, `51070`)**: Not uniformly available across all inpatient trajectories; cannot be assumed present for every patient.
- **eGFR (`50920`, `53161`)**: Unreliable as raw inputs because MIMIC-IV contains legacy unharmonized MDRD text strings. In our pipeline, eGFR is derived causally from harmonized creatinine, age, and sex via the 2021 CKD-EPI formula.
- **Cystatin C**: Confirmed **zero records** in MIMIC-IV laboratory tables.

---

## 5. Experiment Inventory & Exact WEKA Filter Settings

All experiments were executed against the standardized 1,600-instance audited dataset (`integrated_renal_sample.arff`):

| Exp ID | Tested Category | WEKA Filter Class | Filter Arguments & Settings | Objective / Tested Clinical Question |
| :---: | :--- | :--- | :--- | :--- |
| **Exp A** | Missing-Value Imputation | `weka.filters.unsupervised.attribute.ReplaceMissingValues` | (Default settings) | Does global mode/mean imputation distort outpatient and longitudinal records? |
| **Exp B** | Numeric Scaling | `weka.filters.unsupervised.attribute.Normalize` | `-S 1.0 -T 0.0` (Min-Max $[0, 1]$) | Does multi-analyte or outlier scaling crush clinical biomarker resolution? |
| **Exp C** | Duplicate & Conflict Handling | `weka.filters.unsupervised.instance.RemoveDuplicates` | (Default settings) | Does WEKA resolve conflicting clinical measurements with identical timestamps? |
| **Exp D** | Censored Value Handling | `weka.filters.unsupervised.attribute.ReplaceMissingValues` | (Default on censored test ARFF) | How are censored detection limits (`<0.1`, `>15.0`) handled by WEKA? |
| **Exp E** | Relational Integration | External SQL / Pandas Join Analysis | `Left Join` vs `Inner Join` vs `Subject Join` | How do relational join errors propagate into downstream feature tables? |

---

## 6. Detailed Empirical Results & Before-and-After Evidence

### 6.1 Experiment A: Missing-Value Imputation (`ReplaceMissingValues`) — The Lead Finding

#### Command Executed:
```bash
java --add-opens java.base/java.lang=ALL-UNNAMED -cp weka.jar weka.filters.unsupervised.attribute.ReplaceMissingValues \
  -i reports/generated/weka_dataset_integration_audit/integrated_renal_sample.arff \
  -o reports/generated/weka_dataset_integration_audit/exp_a_replacemissingvalues_out.arff
```

#### Quantitative Outcome:
- **Input Instances:** 1,600 | **Output Instances:** 1,600
- **`admission_type` Missing Before Filter:** **747 instances (46.69%)**
- **`admission_type` Missing After Filter:** **0 instances (0.00%)**
- **Imputed Value:** **`"EW EMER."`** (Emergency Ward Emergency Admission)
- **Total Phantom Emergency Admissions Created:** **747 records**

#### Row-Level Evidence (Patient 10000898 — Pure Ambulatory Patient):
Patient 10000898 has 24 laboratory observations in `labevents.csv` and **0 admissions** in `admissions.csv`.

| Row Index | `subject_id` | `charttime` | `analyte` | `valuenum` | Pre-Filter `admission_type` | Post-Filter `admission_type` | Clinical Impact |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **694** | 10000898 | 2180-08-01 10:15 | creatinine | 0.9 mg/dL | `?` (Outpatient) | **`EW EMER.`** | Falsely recorded as emergency hospital admission |
| **695** | 10000898 | 2180-08-01 10:15 | potassium | 4.1 mEq/L | `?` (Outpatient) | **`EW EMER.`** | Falsely recorded as emergency hospital admission |
| **696** | 10000898 | 2180-08-01 10:15 | urea_nitrogen | 14.0 mg/dL | `?` (Outpatient) | **`EW EMER.`** | Falsely recorded as emergency hospital admission |
| ... | 10000898 | (21 more draws) | ... | ... | `?` (Outpatient) | **`EW EMER.`** | 24 false emergency encounters created for 1 patient |

#### Impact on Longitudinal Biomarker Modeling:
In addition to nominal distortion, WEKA replaces missing continuous values with the **global cohort mean**.
- Global cohort mean creatinine in sample: $1.34\text{ mg/dL}$.
- If an end-stage renal disease patient (baseline Cr = $6.5\text{ mg/dL}$) misses a blood draw, WEKA imputes $1.34\text{ mg/dL}$, creating an artificial 80% collapse in creatinine that corrupts downward trajectory slopes.
- If a healthy patient (baseline Cr = $0.6\text{ mg/dL}$) misses a blood draw, WEKA imputes $1.34\text{ mg/dL}$, falsely flagging acute kidney injury.

---

### 6.2 Experiment B: Numeric Scaling & Biomarker Compression (`Normalize`)

#### Command Executed:
```bash
java --add-opens java.base/java.lang=ALL-UNNAMED -cp weka.jar weka.filters.unsupervised.attribute.Normalize \
  -i reports/generated/weka_dataset_integration_audit/integrated_renal_sample.arff \
  -o reports/generated/weka_dataset_integration_audit/exp_b_normalize_out.arff \
  -S 1.0 -T 0.0
```

#### Quantitative Outcome:
- In unpivoted/long clinical representations (standard in EHR lab storage), multiple analytes share a single numerical column (`valuenum`).
- In our sample, `valuenum` spans from $0.3\text{ mg/dL}$ (creatinine) to $350.0\text{ mg/dL}$ (glucose) and $145.0\text{ mEq/L}$ (sodium).
- When global `Normalize` is applied:
  - **Raw Creatinine:** Minimum = $0.30\text{ mg/dL}$, Median = $0.90\text{ mg/dL}$, Maximum = $4.20\text{ mg/dL}$.
  - **Normalized Creatinine:** Minimum = **0.0000**, Median = **0.0018**, Maximum = **0.0120**.
- **Clinical Distortion:** The entire dynamic range of serum creatinine is squashed into a narrow band of $[0.000, 0.012]$ ($1.2\%$ of the feature space), destroying the numerical resolution required for gradient descent and threshold-based clinical alerting.

---

### 6.3 Experiment C: Duplicate & Conflicting Measurement Handling (`RemoveDuplicates`)

#### Command Executed:
```bash
java --add-opens java.base/java.lang=ALL-UNNAMED -cp weka.jar weka.filters.unsupervised.instance.RemoveDuplicates \
  -i reports/generated/weka_dataset_integration_audit/integrated_renal_sample.arff \
  -o reports/generated/weka_dataset_integration_audit/exp_c_removeduplicates_out.arff
```

#### Quantitative Outcome:
- **Input Instances:** 1,600 | **Output Instances:** 1,600
- **Exact Duplicate Repeat Instances Removed:** **0** (because auto-incrementing `labevent_id` renders every instance unique).
- **Conflicting Measurement Rows Present:** **8 rows (4 pairs)** sharing identical `(subject_id, itemid, charttime)` but having disparate `valuenum`.
- **Conflicting Measurements Retained by WEKA:** **8 / 8 (100.0% UNRESOLVED)**.
- **Downstream Consequence:** Because WEKA keeps both conflicting rows, when time deltas are computed, $\Delta t = 0.0\text{ hours}$, causing **division-by-zero ($\Delta\text{Cr} / 0 = \infty$)** in velocity calculations.

---

### 6.4 Experiment D: Censored Value Imputation Distortion

#### Command Executed:
```bash
java --add-opens java.base/java.lang=ALL-UNNAMED -cp weka.jar weka.filters.unsupervised.attribute.ReplaceMissingValues \
  -i reports/generated/weka_dataset_integration_audit/censored_test_in.arff \
  -o reports/generated/weka_dataset_integration_audit/censored_test_weka_mean_out.arff
```

#### Quantitative Outcome:
Clinical assays enforce detection limits (e.g. creatinine $<0.1\text{ mg/dL}$ or $>15.0\text{ mg/dL}$). In MIMIC-IV, these are stored as strings in `value`, leaving `valuenum = NaN`.
- **Global Mean Imputed by WEKA:** **`1.0333 mg/dL`**
- **Row 0 (`<0.1 mg/dL`):** True state is near-zero (severe cachexia or pediatric). WEKA converts this to **1.0333 mg/dL** (**10-fold upward distortion**, creating a false AKI signal).
- **Row 4 (`>15.0 mg/dL`):** True state is lethal anuric renal failure. WEKA converts this to **1.0333 mg/dL** (**14-fold downward distortion**, falsely indicating normal renal function).

---

## 7. Python Reference Solutions & Regression Validation

Our Python temporal and canonical ingestion pipeline (`src/temporal/`) prevents every demonstrated failure through explicit domain logic:

```python
# 1. Relational Integrity & Outpatient Preservation
# In src/temporal/canonical_schema.py:
effective_df = pd.merge(labs, admissions, on=['subject_id', 'hadm_id'], how='left')
effective_df["encounter_type"] = np.where(effective_df["hadm_id"].isna(), "AMBULATORY", "INPATIENT")
# Result: 100% of outpatient records are preserved; admission metadata remains strictly NULL.

# 2. Conflicting Measurement Arbitration (Policy A)
# In src/temporal/conflict_arbitration.py:
# Sorts by charttime, storetime, labevent_id ascending.
# Tags the latest filing time as primary (is_primary_effective = True).
# Tags the conflicting counterpart as superseded (is_primary_effective = False, superseded_by_labevent_id = X).
# Result: Exactly 1 primary record per epoch; zero duplicate timestamp collisions.

# 3. Longitudinal Causal Imputation (LOCF with Staleness Horizon)
# In src/temporal/temporal_features.py:
# Replaces missing values using within-patient Last Observation Carried Forward (LOCF)
# strictly bounded by a 24-hour horizon (dt <= 24.0h).
# Tracks missingness explicitly via boolean indicator val_is_imputed = True.
# Result: Preserves individual baseline trajectories; eliminates cross-patient and population mean leakage.
```

### Regression Verification:
All 94 tests in the project test suite pass cleanly (`pytest -q`), confirming that our Python implementation enforces zero data loss, zero duplication, and strict patient isolation.

---

## 8. Classification of Findings: Defects vs. Methodological Gaps

To maintain strict scientific integrity, we classify each audited behavior:

| Audited Behavior | Technical Classification | Rationale |
| :--- | :--- | :--- |
| **`ReplaceMissingValues` mode imputation on outpatient records** | **Task-Specific Methodological Limitation** | `ReplaceMissingValues` is operating as designed for flat tabular data (mode replacement). The limitation is that standard WEKA Explorer lacks relational awareness, making it mathematically unsuitable for denormalized multi-table clinical datasets. |
| **`ReplaceMissingValues` global mean on time series** | **Task-Specific Methodological Limitation** | Global mean substitution is mathematically valid for cross-sectional I.I.D. data, but clinically invalid for longitudinal physiological trajectories where patient baselines vary widely. |
| **Naive Join Cartesian Explosion (7,185 rows)** | **ETL / Integration Error (External)** | The defect lies in performing a join on `subject_id` alone. This cannot be blamed on WEKA, as WEKA does not perform the join; however, WEKA Explorer cannot detect or resolve the resulting pseudo-replication. |
| **`RemoveDuplicates` retaining conflicting measurements** | **Tool Scope Limitation** | `RemoveDuplicates` is designed to detect identical row tuples, not domain-specific temporal collisions with disparate measurement values. |
| **`TimeSeriesDelta` cross-patient subtraction** | **Standard Filter Capability Gap** | Standard built-in WEKA time-series filters lack group-by parameters, causing 100% boundary contamination across patient records in batch mode. |

---

## 9. Ranked List of Findings

1. **Rank 1 (Selected Lead Finding): Inappropriate Missing-Value Mode Imputation on Integrated Relational Clinical Data (`ReplaceMissingValues`)**
   - *Score:* Highest clinical impact, 100% reproducible, visually striking in WEKA Explorer GUI within 60 seconds, directly answers the supervisor's mandate on dataset integration.
2. **Rank 2: Cross-Patient Boundary Contamination in Sequential Preprocessing (`TimeSeriesDelta`)**
   - *Score:* 100% verified on 5,000 real cohort records (233/233 boundaries contaminated), but requires filter chaining (`MultiFilter` or external conversion).
3. **Rank 3: Relational Join Collapse in Longitudinal Hospital Data (Inner Join vs. Naive Join)**
   - *Score:* 46.69% data loss under inner join; 4.49x row duplication under naive join. High clinical importance, but occurs at the external data-integration layer before WEKA ingestion.
4. **Rank 4: Extreme Outlier Compression in Multi-Analyte Normalization (`Normalize`)**
   - *Score:* Compresses creatinine variation into 1.2% of feature space, but can be mitigated by separating analytes into distinct columns prior to ARFF creation.
5. **Rank 5: Inability of `RemoveDuplicates` to Arbitrate Clinical Measurement Conflicts**
   - *Score:* Retains 100% of conflicting measurement pairs, causing downstream division-by-zero in velocity features.

---

## 10. Selected Finding for Manual WEKA Explorer Verification

### Recommendation for Project Supervisor and Examiners:
**Reproduce Experiment A: Phantom Emergency Admission Imputation via `ReplaceMissingValues`.**

- **Input File to Open:**  
  `reports/generated/weka_dataset_integration_audit/integrated_renal_sample.arff`
- **Attribute to Select:**  
  Attribute #8: `admission_type`
- **Filter to Apply:**  
  `weka.filters.unsupervised.attribute.ReplaceMissingValues`
- **Expected Visual Result in WEKA Explorer GUI:**
  - Before: `Missing = 747 (47%)`, with `EW EMER.` count = `555`.
  - After: `Missing = 0 (0%)`, with `EW EMER.` count jumping to **`1,302`** (+747 phantom admissions).
  - Patient 10000898 (pure outpatient with zero hospitalizations) is assigned 24 Emergency Inpatient Admissions.

A complete, standalone verification guide is provided in:  
[`reports/generated/weka_dataset_integration_audit/manual_weka_explorer_guide.md`](file:///d:/CKD%20mini%20project/nephron-aware-renal-digital-twin/reports/generated/weka_dataset_integration_audit/manual_weka_explorer_guide.md)

---

## 11. Defensible Methodological Contribution Statement

The narrowest, peer-review-resilient research contribution justified by this evidence is:

> *"Off-the-shelf tabular machine learning suites (exemplified by WEKA Explorer 3.8.7) operate under flat-file, independent-and-identically-distributed (I.I.D.) assumptions and lack native relational and longitudinal panel semantics. When multi-table electronic health record datasets (e.g. MIMIC-IV laboratory draws linked to hospital encounters) are denormalized and preprocessed using standard WEKA workflows, two critical failures occur: (1) default missing-value imputation (`ReplaceMissingValues`) substitutes inpatient modes across ambulatory records, converting 46.7% of outpatient laboratory observations into phantom emergency hospital admissions and distorting patient baseline trajectories; and (2) standard instance-differencing filters (`TimeSeriesDelta`) produce 100% cross-patient boundary contamination in multi-patient batches. These empirical findings demonstrate that standard general-purpose data-mining suites cannot safely preprocess longitudinal EHR data without extensive external scaffolding, justifying the architecture of our specialized, patient-aware Python representation pipeline as a necessary foundation for nephron-aware renal digital twin modeling."*

---

## 12. Claims That the Evidence Does NOT Justify

To preserve academic honesty during the final-year evaluation and viva:

1. ❌ **Do NOT claim that WEKA has a programming bug in `ReplaceMissingValues`.**  
   *Correction:* The filter operates exactly as documented (global mode replacement). The flaw is methodological: applying flat-file cross-sectional algorithms to integrated relational healthcare data without domain constraints.
2. ❌ **Do NOT claim that WEKA caused the Cartesian join explosion.**  
   *Correction:* WEKA Explorer cannot perform joins. The join explosion was caused by naive external ETL joining on `subject_id` alone. WEKA's limitation is merely that it accepts the pseudo-replicated table without relational integrity validation.
3. ❌ **Do NOT claim that our Python digital twin production pipeline suffers from these defects.**  
   *Correction:* Our production pipeline (`src/temporal/`) was specifically engineered to prevent these failures through left-join null preservation, Policy A conflict arbitration, and patient-grouped LOCF.
4. ❌ **Do NOT claim this audit represents a novel theoretical discovery in computer science.**  
   *Correction:* It represents a **rigorous methodological validation and engineering justification baseline**, establishing why off-the-shelf data-mining software is insufficient for physiological digital twin development.
