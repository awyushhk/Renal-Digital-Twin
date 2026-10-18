# Milestone M1.2: Controlled Comparison Plan (WEKA 3.8 vs. Python Pipeline)

**Project**: Nephron-Aware Renal Digital Twin  
**Milestone**: M1.2 (Controlled Preprocessing Benchmark & Evaluation Plan)  
**Preceding Milestone**: M1.1 (Dataset Inventory and Audit – Complete Population Scan: 158,374,764 records)  
**Target Environment**: WEKA 3.8.7 (`D:\Software\weka\Weka-3-8-7\weka.jar`, Java 21 LTS) vs. Python 3.14 (`pandas`, `numpy`, `scikit-learn`, `pyarrow`)  
**Status**: Phase A Completed (Controlled Synthetic Benchmark Verified); Phase B Bounded Cohort Pending Review

---

## 1. Executive Summary & Experimental Objectives

The goal of Milestone M1.2 is to conduct a fair, controlled, and methodologically sound comparison between standard machine-learning preprocessing workflows in **WEKA 3.8.7** and a custom **Python pipeline** tailored for the Nephron-Aware Renal Digital Twin.

### Core Scientific Questions

1. Can standard WEKA filters process irregular, longitudinal EHR laboratory time series without introducing subtle future-data leakage or distorting extreme acute kidney injury (AKI) signals?
2. Where does WEKA exhibit true **capability gaps** (operations fundamentally impossible without custom Java programming), where are limitations merely **configuration issues** (solvable through correct parameter tuning or package installation), and where are they **workflow inconveniences** (requiring extensive manual GUI/KnowledgeFlow chaining or external preprocessing)?
3. What is the exact operational boundary between preprocessing tasks best handled by relational Python engines (e.g. denormalization, conditional unit harmonization, irregular elapsed-time derivation) versus tabular feature transformations suitable for WEKA?

---

## 2. Selected Renal Laboratory Subset (Evidence-Based from M1.1)

Based on the completed full scan of **158,374,764 rows** across MIMIC-IV 3.1 `labevents.csv`, we establish a focused, high-coverage panel of **8 core variables** (6 serum chemistry biomarkers + 2 urine biomarkers). This panel captures glomerular filtration, nitrogenous waste retention, acid-base homeostasis, electrolyte stability, and proteinuria barrier function.

### Recommended Experimental Variables

| Biomarker               | Item ID | Dictionary Label    | Specimen & Category | Scanned Records in MIMIC-IV | Missing Num % | Standard Unit | Inclusion Rationale                                                                                           |
| :---------------------- | :------ | :------------------ | :------------------ | :-------------------------- | :------------ | :------------ | :------------------------------------------------------------------------------------------------------------ |
| **Serum Creatinine**    | `50912` | `Creatinine`        | Blood / Chemistry   | **4,319,091**               | 0.04%         | `mg/dL`       | Primary endogenous filtration biomarker for KDIGO AKI staging and baseline eGFR calculation.                  |
| **Blood Urea Nitrogen** | `51006` | `Urea Nitrogen`     | Blood / Chemistry   | **4,202,807**               | 0.05%         | `mg/dL`       | Key indicator of uremic retention, protein catabolism, and prerenal azotemia (BUN/Creatinine ratio).          |
| **Serum Potassium**     | `50971` | `Potassium`         | Blood / Chemistry   | **4,149,507**               | 0.06%         | `mEq/L`       | Distal nephron tubular electrolyte; critical safety marker for hyperkalemic cardiotoxicity in renal failure.  |
| **Serum Sodium**        | `50983` | `Sodium`            | Blood / Chemistry   | **4,111,289**               | 0.01%         | `mEq/L`       | Extracellular osmolality, volume status regulator, and component of fractional excretion of sodium (FeNa).    |
| **Serum Bicarbonate**   | `50882` | `Bicarbonate`       | Blood / Chemistry   | **3,934,240**               | 0.02%         | `mEq/L`       | Acid-base balance; essential for metabolic acidosis assessment in CKD and calculation of the serum anion gap. |
| **Serum Glucose**       | `50931` | `Glucose`           | Blood / Chemistry   | **3,621,769**               | 0.01%         | `mg/dL`       | Key metabolic covariate for diabetic kidney disease (DKD), glycemic stress, and osmotic tubular load.         |
| **Urine Creatinine**    | `51082` | `Creatinine, Urine` | Urine / Chemistry   | **237,799**                 | 0.07%         | `mg/dL`       | Primary spot urine creatinine concentration; necessary denominator for urine albumin/creatinine ratio (uACR). |
| **Urine Albumin**       | `51069` | `Albumin, Urine`    | Urine / Chemistry   | **73,957**                  | 16.59%        | `mg/dL`       | Quantitative biomarker of glomerular podocyte and endothelial barrier injury (microalbuminuria).              |

### Justification for Excluded or Secondary Biomarkers

| Excluded Item ID | Label                      | Specimen    | Count     | Rationale for Exclusion from Initial Comparison                                                                                                                                                          |
| :--------------- | :------------------------- | :---------- | :-------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `52546`          | `Creatinine`               | Blood       | 1,273     | Rare secondary chemistry catalog code (0.03% of serum draws). Should be unified with `50912` downstream, but kept separate in initial baseline.                                                          |
| `52024`          | `Creatinine, Whole Blood`  | Blood (Gas) | 15,175    | Arterial/venous point-of-care blood gas analyzer. Calibration and heparin matrix systematically differ from central lab serum chemistry.                                                                 |
| `51081`          | `Creatinine, Serum`        | Urine       | 717       | Mislabeled dictionary entry (fluid is marked `Urine` despite label); low sample size.                                                                                                                    |
| `50822`          | `Potassium, Whole Blood`   | Blood (Gas) | 333,554   | Whole blood analyzer subject to in vitro hemolysis artifacts; distinct clinical precision from serum potassium `50971`.                                                                                  |
| `50920`          | `Estimated GFR (MDRD)`     | Blood       | 1,665,881 | **100.00% missing numeric values (`valuenum`)**. MIMIC reports MDRD eGFR as string text/comments (e.g. `>60`), making it unsuited for numeric normalization testing without prior regex text extraction. |
| `53161`          | `Estimated GFR (CKD-EPI)`  | Blood       | 81,810    | Fully numeric (0.0% missing), but represents an engineered downstream target equation rather than raw physiological phlebotomy input.                                                                    |
| `51492`          | `Protein`                  | Urine       | 843,478   | **66.65% missing numeric values**. Represents semi-quantitative dipstick test (`NEG`, `TR`, `1+`, `2+`), requiring discrete ordinal encoding rather than continuous scaling.                             |
| `51099`          | `Protein/Creatinine Ratio` | Urine       | 100,387   | Mixed reporting units in raw data (`Ratio: 32,757`, `mg/mg: 67,630`), requiring multi-unit parsing before ARFF loading.                                                                                  |
| `N/A`            | `Cystatin-C`               | All         | **0**     | **Absent from MIMIC-IV laboratory dictionary** (verified across all 1,650 items).                                                                                                                        |

---

## 3. Detailed Evaluation Framework for the 8 Pipeline Requirements

For each of the 8 preprocessing requirements, the comparative benchmark will evaluate WEKA 3.8.7 against Python using a two-tier evaluation protocol:

- **Tier 1 (Synthetic Fixture)**: Deterministic, edge-case micro-datasets with known ground-truth outputs.
- **Tier 2 (Bounded MIMIC Extract)**: A bounded cohort of 1,000–5,000 real patient trajectories using the 8 selected variables.

---

### Requirement 1: Numeric Normalization & Extreme Value Handling

#### Context for Renal Dataset

Renal biomarkers exhibit severe positive skewness with long, clinically critical right tails:

- Serum Creatinine: baseline 0.8–1.2 mg/dL, but escalating to 8.0–15.0+ mg/dL in acute renal failure or ESRD.
- Serum Potassium: tightly regulated (3.5–5.0 mEq/L); values >6.5 mEq/L represent life-threatening hyperkalemia, while values >10.0 mEq/L represent extreme outliers or hemolyzed draws.
  Standard min-max scaling to $[0, 1]$ compresses 95% of normal physiological dynamics into a narrow band $[0, 0.05]$ due to extreme outliers, while standard z-score standardization assumes Gaussianity and distorts non-normal skewed biomarker distributions.

#### Synthetic Dataset Testable?

**Yes**. A 10-row fixture containing baseline values, standard deviations, extreme uremic outliers, and constant/zero-variance attributes.

#### WEKA Workflow (WEKA 3.8.7)

- Filter: `weka.filters.unsupervised.attribute.Normalize` (scales all numeric attributes to $[0, 1]$).
- Filter: `weka.filters.unsupervised.attribute.Standardize` (zero mean, unit variance: $z = \frac{x - \mu}{\sigma}$).
- Parameters: `-S <scale>`, `-T <translation>`, `-unset-class-temporarily`.

#### Equivalent Python Operation

- `sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`
- `sklearn.preprocessing.StandardScaler()`
- `sklearn.preprocessing.RobustScaler(quantile_range=(25.0, 75.0))` (scales using median and IQR).

#### Expected Output & Correctness Checks

- Output values match mathematical definitions to machine precision ($10^{-6}$).
- Constant columns (zero variance) must not trigger divide-by-zero errors or produce `NaN`.
- Outlier test: Assess whether extreme creatinine (e.g. 15.0 mg/dL) obliterates the separation between Stage 1 AKI (1.5 mg/dL) and normal baseline (0.9 mg/dL).

#### Limitation Evidence & Classification

- **Evidence of Limitation**: WEKA standard filters provide only global MinMax and Z-Score; no built-in quantile/robust scaling filter exists in core WEKA without external plugins.
- **Classification**: **Capability Gap** — not a configuration issue. No parameter combination of `Normalize` or `Standardize` can implement median/IQR scaling. External WEKA plugins exist but are outside the core WEKA 3.8.7 distribution tested.
- _**Phase A Correction** (Rationale)_: Original text said "Configuration / Algorithmic limitation." Corrected to "Capability Gap" because (a) the benchmark confirmed no native filter accepts IQR parameters; and (b) under the 5-row synthetic fixture with creatinine outlier = 15.0, MinMax compresses the 0.8–1.2 physiological baseline into the band [0.000, 0.028], empirically confirming the clinical impact.

---

### Requirement 2: Missing-Value Handling and Observation Indicators

#### Context for Renal Dataset

In observational EHRs, phlebotomy is non-random (MNAR – Missing Not At Random). A clinician orders creatinine or blood gases specifically when acute deterioration is suspected. Therefore:

1. Imputing with the cohort mean (e.g. replacing missing creatinine with 1.34 mg/dL) causes a patient with worsening acute kidney failure to abruptly revert to "normal" during unmeasured periods.
2. The presence or absence of a test is itself an informative predictor. An observation mask ($M_{i,t} \in \{0, 1\}$) must be generated alongside any imputed value.

#### Synthetic Dataset Testable?

**Yes**. A multi-patient fixture with missing values at start, middle, and end of trajectories.

#### WEKA Workflow (WEKA 3.8.7)

- Filter: `weka.filters.unsupervised.attribute.ReplaceMissingValues` (replaces missing numeric values with training-set mean).
- Filter: Chaining `weka.filters.unsupervised.attribute.AddExpression` or `NumericToBinary` / `NominalToBinary` to construct observation indicators.

#### Equivalent Python Operation

- `SimpleImputer(strategy="mean", add_indicator=True)`
- Causal temporal forward-fill: `df.groupby('subject_id')[features].ffill()`
- Missingness indicator generation: `df[f"{col}_mask"] = df[col].notna().astype(int)`.

#### Expected Output & Correctness Checks

- For every biomarker $X$, an indicator $X_{\text{mask}} \in \{0, 1\}$ is created.
- Temporal forward-fill propagates the last known physiological state forward without backward lookahead.
- Mean imputation must strictly use training-fold statistics, not the test-fold mean.

#### Limitation Evidence & Classification

- **Evidence of Limitation**: WEKA's `ReplaceMissingValues` filter has no native `add_indicator` option; generating indicators requires complex multi-step filter chaining. Furthermore, `ReplaceMissingValues` is static/global and cannot perform trajectory-aware forward-fill (`LOCF`) across longitudinal patient stays.
- **Classification**: **Capability Gap** for temporal forward-fill (LOCF is architecturally impossible in core WEKA filters); **Workflow Inconvenience** for observation mask generation (achievable in principle by duplicating attributes in a multi-filter KnowledgeFlow pipeline, but not via any single standard filter).
- _**Phase A Correction** (Rationale)_: Original text was consistent. Classification terminology normalized to match benchmark vocabulary (capitalised exact terms). Mean imputation verified numerically: WEKA mean and Python `fillna(mean)` agree to < 10⁻⁴ on the synthetic fixture.

---

### Requirement 3: Renal Laboratory Item and Unit Harmonization

#### Context for Renal Dataset

MIMIC-IV stores measurements as fragmented item codes and occasionally distinct units:

- Albumin: `50862` (Serum Albumin, `g/dL`) vs. `53085` (`mg/dL`, 1000x difference) vs. `51069` (Urine Albumin, `mg/dL`).
- Spot UPCR: `51099` reported both as dimensionless `Ratio` and `mg/mg`.
  A pipeline must translate item codes and conditionally rescale based on `valueuom`.

#### Synthetic Dataset Testable?

**Yes**. A small dataset containing records with mixed item IDs, units, and conversion factors. Numeric/continuous (albumin g/dL vs mg/dL) and semi-quantitative ordinal (dipstick protein grades NEG/TR/1+/2+/3+) are **kept as separate subtests** with distinct expected outputs — continuous values are not converted to arbitrary floats for the WEKA comparison.

_**Phase A Correction** (Rationale)_: Original plan section did not distinguish dipstick ordinal from numeric unit conversion. The benchmark tests them as separate experiments (C1 and C2) following the user's methodological directive.

#### WEKA Workflow (WEKA 3.8.7)

- Filter: `weka.filters.unsupervised.attribute.MathExpression` (applies mathematical operations to attributes).
- Filter: `weka.filters.unsupervised.attribute.MergeManyValues` (combines discrete nominal levels).
- Filter: `weka.filters.unsupervised.attribute.RenameAttribute`.

#### Equivalent Python Operation

- Declarative mapping table:
  ```python
  CONVERSIONS = {("Albumin", "g/dL"): 1.0, ("Albumin", "mg/dL"): 0.001}
  df["val_harmonized"] = df["valuenum"] * df.set_index(["label", "valueuom"]).index.map(CONVERSIONS)
  ```

#### Expected Output & Correctness Checks

- Units mapped to standard SI / clinical base units.
- Categorical fluid/code distinctions preserved so urine creatinine is never accidentally merged into serum creatinine.

#### Limitation Evidence & Classification

- **Evidence of Limitation**: WEKA's `MathExpression` operates on attributes across all instances uniformly (e.g. `A1 * 1000`). It cannot execute conditional row-wise logic based on a second nominal attribute (e.g. `IF valueuom == 'mg/dL' THEN valuenum * 0.001 ELSE valuenum`) without external scripting or partitioning the dataset.
- **Classification**: **Capability Gap** in standard WEKA GUI attribute filters.
- _**Phase A Confirmation**_: WEKA CLI returned `SemanticException: equal is only applicable to doubles!` when `ifelse(A4=0, A*1000, A)` was tested on the synthetic ARFF containing nominal column `valueuom`. This confirms the gap is not a misconfiguration.

---

### Requirement 4: Exact Timestamp Preservation and Granularity

#### Context for Renal Dataset

AKI progression (KDIGO stage 1 $\rightarrow$ 2 $\rightarrow$ 3) is clinically defined by relative creatinine rises within specific time windows (e.g. $\ge 0.3$ mg/dL within 48 hours, or $\ge 1.5\times$ baseline within 7 days). Minute-level timestamp preservation is essential.

#### Synthetic Dataset Testable?

**Yes**. A fixture with ISO timestamps spanning consecutive minutes, hours, days, leap years, and daylight saving transitions.

#### WEKA Workflow (WEKA 3.8.7)

- ARFF Header Specification:
  ```arff
  @attribute charttime date "yyyy-MM-dd HH:mm:ss"
  @attribute storetime date "yyyy-MM-dd HH:mm:ss"
  ```
- Loader: `weka.core.converters.ArffLoader` and `weka.core.converters.CSVLoader` with `-D "yyyy-MM-dd HH:mm:ss"`.

#### Equivalent Python Operation

- `pd.to_datetime(df["charttime"], format="%Y-%m-%d %H:%M:%S")`
- Storage as PyArrow timestamp `timestamp[ns]` or 64-bit datetime.

#### Expected Output & Correctness Checks

- Exact preservation of hours, minutes, and seconds upon loading, filtering, and re-exporting.
- Time difference calculation between consecutive instances matches ground truth down to the minute.

#### Limitation Evidence & Classification

- **Phase A Result**: **No limitation found**. WEKA correctly parses and round-trips all test timestamps (consecutive minutes, seconds, leap-day `2152-02-29`) when the ARFF header declares `@attribute charttime date "yyyy-MM-dd HH:mm:ss"`. All four synthetic timestamps survived the `Copy` filter with zero drift.
- **Classification**: **None** — WEKA fully supports exact ISO minute-and-second datetime preservation given correct ARFF header declaration.
- _**Phase A Correction** (Rationale)_: Original text described hypothetical failures (truncation, timezone shift). Replaced with empirical evidence of correct behavior to avoid misleading Phase B implementation.

---

### Requirement 5: Patient/Admission Linkage Integrity

#### Context for Renal Dataset

In MIMIC-IV, **46.58%** of laboratory events lack `hadm_id` (representing outpatient or pre-admission draws), while 100% of admission `subject_id`s link to `patients.csv`. A valid preprocessing pipeline must maintain referential integrity without dropping ambulatory baseline renal measurements.

#### Synthetic Dataset Testable?

**Yes**. Using a **validated flat pre-joined representation** (not a multi-table relational triad). The benchmark tests linkage on a Python-produced flat ARFF with `hadm_id = NaN` for the outpatient row. This isolates the preprocessing comparison from database joining, consistent with the methodological directive.

_**Phase A Correction** (Rationale)_: Original text planned testing a three-table relational triad inside WEKA. That design conflated the relational join task with the flat-table preprocessing task. Testing on a pre-joined flat table correctly isolates whether WEKA handles null numeric IDs correctly — which it does.

#### WEKA Workflow (WEKA 3.8.7)

- Investigation: Does WEKA provide a native in-memory relational join mechanism for CSV/ARFF files, or must the dataset be denormalized prior to WEKA ingestion?
- Tools: WEKA KnowledgeFlow `DatabaseLoader` with SQL JOIN vs. flat ARFF loading.

#### Equivalent Python Operation

- `pd.merge(lab_df, patients_df, on="subject_id", how="left")`
- `pd.merge_asof(lab_df, admissions_df, left_on="charttime", right_on="admittime", by="subject_id")`.

#### Expected Output & Correctness Checks

- Outpatient lab events are preserved with `hadm_id = null`.
- No row multiplication or cross-product inflation occurs.
- Patient demographics (age, sex) are correctly attributed to all lab events.

#### Limitation Evidence & Classification

- **Evidence of Limitation**: WEKA Explorer, Experimenter, and standard CLI tools require single flat tabular files (`.arff` or `.csv`). They cannot execute relational foreign-key joins directly on filesystem CSVs. Outpatient rows with missing `hadm_id` **are preserved** correctly when WEKA loads an already-flat ARFF (4 rows in, 4 rows out; 1 null `hadm_id` retained).
- **Classification**: **Capability Gap** for native multi-table relational joins; **None** for null-ID preservation on flat tables.
- _**Phase A Correction** (Rationale)_: Linkage test split into two subtests (R5_LINKAGE × 2): (1) flat-table null-ID retention (PASSED), and (2) relational join capability (UNSUPPORTED). Previously conflated into one result.

---

### Requirement 6: Duplicate and Conflicting Measurement Handling

#### Context for Renal Dataset

The M1.1 audit verified that concurrent draws at identical timestamps `(subject_id, charttime, itemid)` occur in MIMIC-IV (e.g. blood drawn simultaneously for central lab and point-of-care verification), with frequent discordant values. These must be arbitrated deterministically (e.g. median, central lab precedence, or exclusion) to avoid non-deterministic model inputs.

#### Synthetic Dataset Testable?

**Yes**. A fixture containing exact duplicate rows, concurrent draws with identical values, and concurrent draws with discordant values.

#### WEKA Workflow (WEKA 3.8.7)

- Filter: `weka.filters.unsupervised.instance.RemoveDuplicates` (removes identical instances across all attributes).

#### Equivalent Python Operation

- Deterministic group-by aggregation:
  ```python
  clean_df = (
      df.sort_values(["subject_id", "charttime", "itemid", "priority_rank"])
      .groupby(["subject_id", "charttime", "itemid"])
      .agg({"valuenum": "median", "valueuom": "first"})
      .reset_index()
  )
  ```

#### Expected Output & Correctness Checks

- Exact duplicates removed.
- Discordant values at identical timestamps reduced to a single deterministic value per patient-timestamp-test tuple.

#### Limitation Evidence & Classification

- **Evidence of Limitation**: WEKA's `RemoveDuplicates` only eliminates instances that are 100% identical across all columns. It cannot detect partial-key duplicates or arbitrate conflicting numeric measurements (e.g. averaging or taking the median of two differing creatinine draws).
- **Classification**: **Capability Gap** in core instance filters.
- _**Phase A Confirmation**_: Benchmark fixture contained rows `(1001, 2150-01-01 10:00:00, 50912, valuenum=1.2)` and `(1001, 2150-01-01 10:00:00, 50912, valuenum=1.8)`. After `RemoveDuplicates`, both discordant rows remained (3 → 3). Python `groupby(...).median()` correctly reduced to 1 row with `valuenum=1.5`.

---

### Requirement 7: Irregular Elapsed-Time Feature Derivation

#### Context for Renal Dataset

Clinical phlebotomy intervals are highly irregular (e.g. every 4 hours in the ICU, daily on the floor, months apart in outpatient clinics). The elapsed time since last observation ($\Delta t = t_k - t_{k-1}$) is a critical feature: a creatinine ordered after 4 hours conveys rapid clinical deterioration, whereas a repeat draw after 6 months indicates routine chronic monitoring.

#### Synthetic Dataset Testable?

**Yes**. A fixture with multiple patients having irregular time intervals between draws.

#### WEKA Workflow (WEKA 3.8.7)

- Extension / Filter: `weka.filters.unsupervised.attribute.TimeSeriesDelta` or `TimeSeriesTranslate`.
- Package: WEKA `timeSeriesForecasting` package.

#### Equivalent Python Operation

- Group-by chronological time delta:
  ```python
  df["delta_hours"] = (
      df.sort_values(["subject_id", "charttime"])
      .groupby("subject_id")["charttime"]
      .diff()
      .dt.total_seconds() / 3600.0
  )
  ```

#### Expected Output & Correctness Checks

- Correct elapsed hours computed between consecutive records for the same patient.
- $\Delta t$ resets to `NaN` or a sentinel value at patient boundaries (first draw of Patient B must not compute $\Delta t$ relative to the last draw of Patient A).

#### Limitation Evidence & Classification

- **Evidence of Limitation**: `TimeSeriesDelta` in standard WEKA operates on instance row index offsets ($i - 1$), assuming uniform sampling intervals. It does not compute time deltas from timestamp attributes and does not respect patient boundary grouping (`subject_id`).
- **Classification**: **Capability Gap** for patient-grouped irregular longitudinal time series.
- _**Phase A Confirmation**_: Benchmark fixture placed Patient 1002's first draw at row index 4 immediately after Patient 1001's last draw. `TimeSeriesDelta` subtracted Patient 1001's creatinine (1.6) from Patient 1002's creatinine (0.9), yielding −0.7 — a clinically meaningless cross-patient artifact. Python `groupby('subject_id').diff()` correctly returned `NaN` for Patient 1002's first draw.

---

### Requirement 8: Strictly Causal Feature Generation & Leakage Prevention

#### Context for Renal Dataset

A digital twin predicting renal recovery or AKI progression must operate strictly causally in real time. Any calculation at timestamp $t$ (e.g. baseline creatinine, cumulative fluid balance, rolling mean) must only utilize information available at or before $t$.

#### Synthetic Dataset Testable?

**Yes**. A fixture designed with known train/test splits where test-set future outcomes are engineered to distort training statistics if leakage occurs.

#### WEKA Workflow (WEKA 3.8.7)

- Workflow: Using `weka.classifiers.meta.FilteredClassifier` to encapsulate preprocessing within cross-validation folds.
- Comparison: Evaluating whether batch filtering in the WEKA Preprocess GUI applies whole-cohort statistics across both train and test partitions.

#### Equivalent Python Operation

- Scikit-learn `Pipeline` fitted strictly on `X_train` and evaluated on `X_test`.
- Strictly causal rolling windows: `df.groupby('subject_id')['valuenum'].expanding().mean()`.

#### Expected Output & Correctness Checks

- Training-set scaling statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) are completely independent of test instances.
- Zero future-data leakage into past time points.
- **Future-perturbation test** (empirical causality check): modifying any observation at time $t > t_{\text{eval}}$ must leave the feature computed at $t_{\text{eval}}$ completely unchanged.

#### Limitation Evidence & Classification

- **Evidence of Limitation**: Batch normalization in WEKA recomputes global min/max across all rows. Changing the future row at $t_5$ from 1.4 to 12.0 caused the normalized value at historical row $t_2$ to shift from **0.2500 → 0.0091** (absolute shift = 0.2409). Python causal expanding windows remained exactly **1.0500** in both base and perturbed datasets — zero leakage.
- **Classification**: **Workflow Inconvenience** for batch normalization specifically (preventable using `FilteredClassifier` in cross-validation, but a systematic hazard in interactive WEKA batch mode). Note: `FilteredClassifier` encapsulation does NOT prevent leakage within a single patient's temporal sequence — it only prevents train/test fold contamination.
- _**Phase A Correction** (Rationale)_: Original text said `FilteredClassifier` prevents the hazard. Clarified that `FilteredClassifier` guards train/test boundaries but does NOT enforce strict within-sequence causal ordering. The future-perturbation test was implemented as a direct empirical proof rather than a theoretical argument.

---

## 4. Controlled Experimental Protocol & Execution Matrix

### Phase A: Synthetic Correctness Benchmark (Unit Verification)

**Status: COMPLETED** — All 15 subtests executed; results recorded in [`reports/generated/weka_benchmark_results.csv`](../reports/generated/weka_benchmark_results.csv).

| Requirement # | Actual Fixture File | WEKA Evaluation Target | Python Evaluation Target | Phase A Result |
| :--- | :--- | :--- | :--- | :--- |
| **R1a (Min-Max)** | `tests/fixtures/weka_benchmark/r1_norm_fixture.arff` | `Normalize` | `(x - min) / (max - min)` | **PASSED** (tol < 4.2×10⁻⁷) |
| **R1b (Z-Score)** | `tests/fixtures/weka_benchmark/r1_norm_fixture.arff` | `Standardize` | `(x - mean) / std(ddof=1)` | **PASSED** (tol < 3.3×10⁻⁷) |
| **R1c (Robust)** | (Python-only) | None in WEKA 3.8.7 | `(x - median) / IQR` | **UNSUPPORTED** (Capability Gap) |
| **R2a (Mean Imp.)** | `tests/fixtures/weka_benchmark/r2_missing_fixture.arff` | `ReplaceMissingValues` | `fillna(mean)` | **PASSED** (tol = 0.0) |
| **R2b (Obs. Mask)** | (concept test) | No equivalent option | `notna().astype(int)` | **UNSUPPORTED** (Workflow Inconvenience) |
| **R2c (LOCF)** | `tests/fixtures/weka_benchmark/r2_missing_fixture.arff` | No equivalent filter | `groupby().ffill()` | **UNSUPPORTED** (Capability Gap) |
| **R3a (Unit Conv.)** | `tests/fixtures/weka_benchmark/r3_units_fixture.arff` | `MathExpression` (fails) | `apply(lambda r: ...)` | **UNSUPPORTED** (Capability Gap) |
| **R3b (Dipstick)** | `tests/fixtures/weka_benchmark/r3_units_fixture.arff` | `NominalToString` (manual) | `series.map({...})` | **PASSED** (Workflow Inconvenience) |
| **R4 (Timestamps)** | `tests/fixtures/weka_benchmark/r4_timestamp_fixture.arff` | `Copy` filter + date header | `pd.to_datetime` | **PASSED** (tol = 0.0) |
| **R5a (Flat Null-ID)** | `tests/fixtures/weka_benchmark/r5_linkage_fixture.arff` | `Copy` filter | pandas preservation | **PASSED** (tol = 0.0) |
| **R5b (Relational)** | N/A (flat-table isolation) | No join engine in WEKA | `pd.merge` | **UNSUPPORTED** (Capability Gap) |
| **R6a (Exact Dup.)** | `tests/fixtures/weka_benchmark/r6_duplicate_fixture.arff` | `RemoveDuplicates` | `drop_duplicates()` | **PASSED** (tol = 0.0) |
| **R6b (Conflict)** | `tests/fixtures/weka_benchmark/r6_duplicate_fixture.arff` | `RemoveDuplicates` (retains) | `groupby().median()` | **UNSUPPORTED** (Capability Gap) |
| **R7 (Δt Reset)** | `tests/fixtures/weka_benchmark/r7_elapsed_fixture.arff` | `TimeSeriesDelta` (wrong) | `groupby().diff()` | **UNSUPPORTED** (Capability Gap) |
| **R8 (Causality)** | `tests/fixtures/weka_benchmark/r8_causal_base.arff` + pert | `Normalize` (batch leaks) | `expanding().mean()` | **FAILED** (shift = 0.241) |

---

### Phase B: Bounded Empirical MIMIC Cohort Extract

To benchmark execution scalability and real-world behavior on clinical data without overloading WEKA:

- **Cohort Size**: Bounded extract of **2,500 unique patients** having at least 3 longitudinal serum creatinine measurements.
- **Features Extracted**: The 8 selected variables (6 serum chemistry + 2 urine markers) plus patient demographics (`anchor_age`, `gender`) and admission timings.
- **Size on Disk**: Estimated ~15–25 MB (easily loaded into WEKA Explorer without Java OutOfMemory exceptions).
- **Export Format**: Synchronized flat files (`cohort_renal_bounded.csv` and `cohort_renal_bounded.arff`).

---

## 5. Methodological Safeguards & Scope Boundaries

To prevent research drift and adhere strictly to project milestones:

1. **Source Data Immutability**: Raw MIMIC-IV 3.1 files in `data/raw/` remain strictly read-only.
2. **No Arbitrary Clinical Imputation**: No clinical values will be interpolated or imputed in the raw tables.
3. **No Prediction Modeling in M1.2**: No machine learning classifiers, neural networks, or survival models will be trained in this milestone. The milestone evaluates **preprocessing operations exclusively**.
4. **No Premature Model Performance Claims**: No claims regarding "superior CKD prediction" will be asserted. The output of M1.2 is an empirical audit of preprocessing correctness, data integrity, and pipeline ergonomics.

---

## 6. Open Decisions — Resolved by Phase A Evidence

The three design decisions raised before Phase A are now resolved by empirical evidence:

1. **WEKA Invocation Mode** → **Decision: Option A (Headless CLI)**  
   Phase A executed all WEKA subtests using `java --add-opens java.base/java.lang=ALL-UNNAMED -cp "D:\Software\weka\Weka-3-8-7\weka.jar" weka.filters...` via Python subprocess. All 8 filter classes executed correctly or returned interpretable errors. No GUI interaction was required or used.

2. **Urine Protein Variable Prioritization** → **Decision: Both, as distinct experiments**  
   Phase A tested them as separate subtests (R3a numeric unit conversion, R3b dipstick ordinal mapping). The benchmark confirmed dipstick grades must **not** be converted to arbitrary continuous values and must be handled with an explicit ordinal rank map. Both subtests are now implemented and verified.

3. **Denormalization Responsibility** → **Decision: Mandatory preliminary Python stage**  
   Phase A confirmed WEKA has no native multi-table relational join engine. The benchmark tests linkage on validated Python-produced flat ARFF files. Denormalization is an external Python prerequisite, not a WEKA task.

---

## 7. Phase A Evidence Summary & Recommendation for Phase B

Phase A synthetic benchmark produced **15 subtests** across 8 preprocessing requirements. Results:

- **7 PASSED** (46.7%) — WEKA is equivalent to Python for static linear transformations.
- **1 FAILED** (6.7%) — WEKA batch normalization leaks future values (empirically proven, shift = 0.241 units).
- **7 UNSUPPORTED** (46.7%) — WEKA has true Capability Gaps for longitudinal EHR-specific operations.

### Capability Gaps (Must Be Implemented in Python)

| Gap | Clinical Consequence |
| :--- | :--- |
| No RobustScaler | Uremic outliers (creatinine 15 mg/dL) compress 95% of clinical range into 2.8% of output space under MinMax |
| No grouped LOCF forward-fill | Missing draws replaced with cohort mean, breaking deteriorating patient trajectories |
| No row-conditional unit rescaling | g/dL vs mg/dL conversions cannot be automated (1000× error if ignored) |
| No clinical key-based arbitration | Concurrent discordant lab draws remain as two separate instances |
| No chronological Δt with patient reset | Cross-patient Δt artifacts introduced at every patient boundary |
| No relational multi-table joining | `hadm_id` denormalization must occur before any WEKA processing |

### Safe WEKA Operations for Static Tabular Use

- MinMax normalization, Z-Score standardization (numerically equivalent to Python)
- ISO timestamp preservation (with correct ARFF `date` header)
- Full-row exact duplicate removal
- Null numeric ID preservation in flat tables

### Prerequisite for Phase B

Phase B (bounded MIMIC cohort extract) **may proceed** only after this Phase A summary is reviewed and approved. The 2,500-patient cohort extract, clinical pipeline implementation, and production imputation logic remain out of scope until explicit authorization.
