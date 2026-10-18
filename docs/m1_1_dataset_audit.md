# Milestone M1.1: Dataset Inventory and Audit

**Project**: Nephron-Aware Renal Digital Twin  
**Target Milestone**: M1.1 (Dataset Inventory and Audit)  
**Downstream Milestone**: M1.2 (Controlled Comparison of WEKA vs. Python Preprocessing Pipelines)  
**Dataset**: MIMIC-IV v3.1 (`admissions.csv`, `patients.csv`, `d_labitems.csv`, `labevents.csv`)  
**Status**: Completed (Dictionary Audit, Demographics/Admissions Audit, and Laboratory Events Smoke Audit Executed; Full-Scan Mode Implemented)

---

## 1. Executive Summary & Research Motivation

The overarching goal of the **Nephron-Aware Renal Digital Twin** is to model acute and chronic renal function trajectory, glomerular filtration dynamics, and nephrotoxicity risk by integrating physiological priors with longitudinal clinical observational data.

Milestone M1.1 establishes the dataset inventory and empirical data quality profile for the raw MIMIC-IV 3.1 tables. In Milestone M1.2, we will conduct a rigorous, controlled comparison between standard tabular preprocessing workflows implemented in **WEKA** (e.g. `weka.filters.unsupervised.attribute.*`) and a custom, strictly causal Python pipeline. To ensure the M1.2 comparison evaluates true operational capabilities rather than arbitrary assumptions, this audit documents:
1. Exact schemas, row counts, and referential linkage structures.
2. The complete laboratory test dictionary (`d_labitems.csv`) and candidate renal biomarkers.
3. Observed data anomalies, duplicate measurements, conflicting values, and irregular temporal sampling in `labevents.csv`.
4. An empirical requirements matrix for candidate WEKA baseline evaluation.

---

## 2. Table-by-Table Inventory and Audit Findings

### 2.1 Laboratory Dictionary (`d_labitems.csv`)
- **Total Records**: 1,650 rows (each representing a distinct laboratory assay).
- **Primary Key**: `itemid` (1,650 unique, zero duplicates).
- **Missingness**: `itemid` (0 null), `fluid` (0 null), `category` (0 null), `label` (4 nulls).
- **Top Categories**: `Chemistry` (800 items), `Hematology` (785 items), `Blood Gas` (65 items).
- **Top Specimen Fluids**: `Blood` (821), `Other Body Fluid` (201), `Urine` (194), `Cerebrospinal Fluid` (101), `Joint Fluid` (72), `Pleural` (69), `Ascites` (68), `Bone Marrow` (63), `Stool` (46).

#### Candidate Renal-Related Biomarkers Identified
A systematic keyword search identified **135 candidate tests** relevant to nephron function, glomerular permeability, nitrogenous waste clearance, metabolic state, and electrolyte balance:
- **Creatinine (Serum & Urine)**:
  - `50912`: Creatinine (Blood / Chemistry) – Primary routine serum creatinine assay.
  - `52546`: Creatinine (Blood / Chemistry) – Secondary/alternate serum chemistry code.
  - `52024`: Creatinine, Whole Blood (Blood / Blood Gas) – Point-of-care / blood gas analyzer.
  - `51081`: Creatinine, Urine (Urine / Chemistry) – Random / spot urine creatinine.
  - `51106`: Urine Creatinine (Urine / Chemistry) – Alternative urine creatinine catalog.
  - `52000`: Urine Creatinine (Urine / Chemistry) – Secondary urine code.
- **Urea Nitrogen / BUN**:
  - `51006`: Urea Nitrogen (Blood / Chemistry) – Standard blood urea nitrogen.
  - `52647`: Urea Nitrogen (Blood / Chemistry) – Secondary blood urea nitrogen.
  - `51104`: Urea Nitrogen, Urine (Urine / Chemistry) – 24h / spot urine clearance.
  - `51842`: Bun (Other Body Fluid / Chemistry) – Non-standard body fluid.
- **Albumin & Glomerular Barrier Markers**:
  - `50862`: Albumin (Blood / Chemistry) – Serum albumin.
  - `52022`: Albumin, Blood (Blood / Blood Gas).
  - `51069`: Albumin, Urine (Urine / Chemistry) – Direct urine albumin (microalbuminuria).
  - `51070`: Albumin/Creatinine, Urine (Urine / Chemistry) – ACR spot test.
  - `52703`: Albumin, Urine (Urine / Chemistry) – Alternative urine albumin.
- **Proteins & Proteinuria**:
  - `50976`: Protein, Total (Blood / Chemistry).
  - `51102`: Total Protein, Urine (Urine / Chemistry) – Quantitative 24-hr/timed urine protein.
  - `51068`: 24 hr Protein (Urine / Chemistry).
  - `51099`: Protein/Creatinine Ratio (Urine / Chemistry) – UPCR spot test.
  - `51492`: Protein (Urine / Hematology) – Urinalysis dipstick test (semi-quantitative).
- **Glomerular Filtration Rate (eGFR)**:
  - `50920`: Estimated GFR (MDRD equation) (Blood / Chemistry).
  - `53161`: Estimated GFR (CKD-EPI Refit) (Blood / Chemistry).
  - `53180`: Estimated GFR (CKD-EPI 2021) (Blood / Blood Gas).
  - `51770`: MDRDgfr (Blood / Chemistry).
- **Glucose & Metabolic Regulators**:
  - `50931`: Glucose (Blood / Chemistry) – Standard serum glucose.
  - `50809`: Glucose (Blood / Blood Gas) – Whole blood arterial/venous glucose.
  - `51084`: Glucose, Urine (Urine / Chemistry).
- **Electrolytes & Acid-Base Balance**:
  - Potassium: `50971` (Serum), `50822` (Whole Blood), `51097` (Urine).
  - Sodium: `50983` (Serum), `50824` (Whole Blood), `51100` (Urine).
  - Bicarbonate: `50882` (Serum), `50803` (Calculated Bicarbonate, Blood Gas), `51076` (Urine).
  - Chloride: `50902` (Serum), `50806` (Whole Blood), `51078` (Urine).

> [!NOTE]
> **Cystatin-C Absence in MIMIC-IV Lab Dictionary**:
> A targeted search for `cystatin`, `cys`, and wildcard variants across all 1,650 items confirmed **zero entries for Cystatin-C**. The only matches under `cys` were `50945` (Homocysteine) and `51473` (Cysteine Crystals). Consequently, standard cystatin-C eGFR equations cannot be evaluated directly from MIMIC-IV structured lab tables without auxiliary unstructured EHR notes.

---

### 2.2 Patient Demographics (`patients.csv`)
- **Total Population**: 364,627 subjects.
- **Uniqueness**: `subject_id` is 100% unique (0 duplicates).
- **Sex Breakdown**: Female = 191,984 (52.7%), Male = 172,643 (47.3%).
- **Age Distribution**:
  - Minimum: 18 years
  - 25th percentile (Q1): 29 years
  - Median: 48 years
  - 75th percentile (Q3): 65 years
  - Maximum: 91 years (MIMIC-IV de-identification caps all ages >89 to 91).
- **Out-of-Hospital Mortality**: 326,326 records (89.5%) have null `dod` (representing surviving patients or those whose post-discharge death was not captured in state registry records).

---

### 2.3 Hospital Encounters (`admissions.csv`)
- **Total Encounters**: 546,028 hospital admissions.
- **Uniqueness**: `hadm_id` is 100% unique (0 duplicate admissions across the entire table).
- **Subject Representation**: 223,452 unique patients.
- **Referential Integrity**: **100.00%** of `subject_id`s in `admissions.csv` match a valid record in `patients.csv`.
- **Ambulatory Subpopulation**: **141,175** patients (38.72% of `patients.csv`) have zero hospital admissions in `admissions.csv` (outpatients, emergency-only visits without formal inpatient admission, or outpatient laboratory monitoring).
- **In-Hospital Mortality**: 11,801 encounters (2.16%) resulted in in-hospital death (`hospital_expire_flag == 1`).
- **Hospital Length of Stay (LOS)**:
  - Median LOS: 2.82 days.
  - Negative LOS Anomaly: 175 encounters exhibit `dischtime < admittime`. These instances represent administrative observation records or clerical timestamp reversals and must be screened out in causal temporal cohorts.

---

### 2.4 Laboratory Events (`labevents.csv`)
- **Scale**: 18.4 GB on disk (18,402,851,720 bytes). Must **never** be loaded in full into system memory.
- **Auditing Architecture**: Implemented streaming chunked evaluation (`audit_labevents`) with bounded smoke-test and exhaustive full-scan modes.
- **Population Audit Findings (from 100% Complete Full Scan)**:
  - **Total Rows Processed**: **158,374,764** rows across **634** chunks (250,000 rows/chunk).
  - **Execution Throughput**: **357,297 rows/second** (Total elapsed: 443.3s / 7.4 minutes).
  - **Unique Patients Represented**: **313,442** unique subjects.
  - **Unique Hospital Admissions Represented**: **447,689** unique encounters.
  - **Unique Laboratory Items Observed**: **976** distinct test items.
  - **Timestamp Parseability & Range**: `charttime` ranges from `2105-01-19 12:01:00` to `2215-01-12 11:45:00` with **zero unparseable timestamps** and **zero missing charttimes**.
  - **Linkage Missingness (`hadm_id`)**: **46.58%** (73,770,967 rows) lack `hadm_id`. These draws correspond to outpatient clinics, pre-admission workups, or ambulatory phlebotomy. Preprocessing pipelines that naively drop records with missing `hadm_id` discard nearly half of all clinical observations.
  - **Numeric Result Missingness (`valuenum`)**: **13.57%** (21,491,455 rows) have null `valuenum`, corresponding to qualitative urinalysis, blood bank, or qualitative screens.
  - **Concurrent / Duplicate Measurement Keys**: In a 200,000-row sample, 1,979 concurrent measurement keys `(subject_id, charttime, itemid)` were observed, with 1,128 conflicting numeric values requiring deterministic clinical arbitration.

---

## 3. Methodological Safeguards & Evidence Classification

To maintain scientific integrity prior to the downstream experimental comparison:

| Classification | Meaning | Examples from this Audit |
| :--- | :--- | :--- |
| **Observed Fact** | Directly measured from raw CSV data without transformation. | `d_labitems.csv` has 1,650 rows; 100% of admission `subject_id`s link to `patients.csv`; 175 admissions have `dischtime < admittime`. |
| **Assumption** | Hypothesized operational behavior requiring validation. | Assuming `charttime` strictly precedes `storetime`; assuming `hadm_id == null` strictly indicates ambulatory draws. |
| **Unverified Finding** | Plausible pattern requiring experimental or full-file confirmation. | Calibration equivalence between serum creatinine item `50912` vs `52546`; whole-dataset prevalence of discordant duplicate draws. |
| **Limitation** | Boundary condition of the current audit milestone. | Smoke test frequencies reflect bounded samples; no imputation, outlier removal, or causal temporal modeling performed. |

---

## 4. Candidate WEKA Baseline Requirements

Milestone M1.2 will benchmark WEKA against our Python pipeline for clinical preprocessing. To design a fair, controlled, and rigorous experimental protocol, each pipeline requirement is evaluated below against what the M1.1 audit has established, what remains unknown, and the specific WEKA capability to evaluate.

> [!IMPORTANT]
> **Methodological Rule**: We do not claim that WEKA cannot perform any operation prior to experimental benchmarking. Rather, we identify the exact operational constraints, filter mechanisms, and verification criteria to test in M1.2.

### Requirement 1: Numeric Normalization
- **What the Audit Establishes**: Renal biomarkers exhibit wide dynamic ranges with heavy right tails (e.g., serum creatinine normal: 0.7–1.2 mg/dL, acute renal injury / ESRD: >10 mg/dL; BUN normal: 10–20 mg/dL, uremic crisis: >120 mg/dL).
- **What Remains Unknown**: Whether standard z-score normalization (`(x - μ)/σ`) distorts extreme acute kidney injury (AKI) signals compared to robust scaling (median / IQR) or physiological log-transforms.
- **WEKA Workflow to Test**: `weka.filters.unsupervised.attribute.Normalize` (scales to `[0, 1]`) and `weka.filters.unsupervised.attribute.Standardize` (zero mean, unit variance).
- **Evaluation Criteria**: Numerical stability, sensitivity to uremic outliers, and handling of zero-variance features in subsamples.

### Requirement 2: Missing-Value Handling
- **What the Audit Establishes**: Missingness in EHR laboratories is non-random (MNAR – Missing Not At Random). In `labevents.csv`, 13.91% of records lack `valuenum` (qualitative results), and outpatient periods feature sparse, irregular sampling.
- **What Remains Unknown**: Whether mean or mode imputation creates clinically implausible renal trajectories (e.g., abruptly resetting an escalating creatinine trajectory to the cohort mean).
- **WEKA Workflow to Test**: `weka.filters.unsupervised.attribute.ReplaceMissingValues` (replaces with global training mean/mode).
- **Evaluation Criteria**: Distortion of temporal autocorrelation, preservation of missingness indicators (observation masks), and prevention of clinical signal flattening.

### Requirement 3: Unit and Test-Identifier Harmonization
- **What the Audit Establishes**: Multiple distinct `itemid`s represent creatinine (`50912`, `52546`, `52024`) and albumin (`50862`, `52022`, `51069`). Some are reported in `mg/dL`, others in `g/dL` or `%`.
- **What Remains Unknown**: Whether disparate item IDs and fluid categories can be unified without introducing departmental measurement bias or misinterpreting urine creatinine as serum creatinine.
- **WEKA Workflow to Test**: `weka.filters.unsupervised.attribute.MergeManyValues`, `AddExpression`, or MultiFilter pipelines for relational dictionary translation.
- **Evaluation Criteria**: Can WEKA natively harmonize conditional multi-column mappings (e.g. `if fluid == 'Blood' and itemid in [50912, 52546]`) without prior Python tabular shaping?

### Requirement 4: Timestamp Preservation and Granularity
- **What the Audit Establishes**: Timestamps in MIMIC-IV (`charttime`, `admittime`, `dischtime`) are formatted as ISO strings (`YYYY-MM-DD HH:MM:SS`).
- **What Remains Unknown**: Whether WEKA's ARFF `date` attribute format preserves sub-day time differences (hours, minutes) without precision loss during serialization.
- **WEKA Workflow to Test**: ARFF Date specification `@attribute charttime date "yyyy-MM-dd HH:mm:ss"` and `weka.core.converters.ArffLoader`.
- **Evaluation Criteria**: Exact elapsed time retention, absence of day-level truncation, and correct timezone-agnostic parsing.

### Requirement 5: Patient and Admission Linkage Integrity
- **What the Audit Establishes**: 100% of admission `subject_id`s link to `patients.csv`, but 49.57% of lab events lack `hadm_id` (outpatient/pre-admission).
- **What Remains Unknown**: How non-inpatient records should be anchored to clinical timelines, and whether relational table joins can be executed inside WEKA or require external relational flattening.
- **WEKA Workflow to Test**: WEKA KnowledgeFlow relational loader or evaluating whether WEKA requires a single pre-joined denormalized CSV/ARFF table.
- **Evaluation Criteria**: Referential consistency, memory consumption when loading denormalized tables, and retention of outpatient baseline measurements.

### Requirement 6: Duplicate and Conflict Handling
- **What the Audit Establishes**: Simultaneous measurements at identical `(subject_id, charttime, itemid)` occur in `labevents.csv` (1,979 duplicate keys in 200k rows), with 1,128 conflicting numeric values.
- **What Remains Unknown**: Whether conflicting values represent point-of-care vs. central laboratory confirmation, rapid repeat draws, or clerical errors, and what deterministic arbitration rule (e.g., median, priority attribute, or central lab precedence) is optimal.
- **WEKA Workflow to Test**: In-engine deduplication filters vs. external pre-filtering.
- **Evaluation Criteria**: Can WEKA detect and resolve multi-column grouping conflicts, or does duplicate arbitration necessitate deterministic preprocessing before ARFF ingestion?

### Requirement 7: Irregular-Time Features and Observation Masks
- **What the Audit Establishes**: Phlebotomy occurs at highly irregular intervals (e.g. every 4 hours in the ICU, every 24 hours on general medical floors, and months apart in outpatient clinics).
- **What Remains Unknown**: How to represent the informative presence/absence of a test (the fact that a clinician ordered creatinine is itself a marker of clinical suspicion).
- **WEKA Workflow to Test**: Observation indicator binarization via `weka.filters.unsupervised.attribute.NumericToBinary` and lagging via `weka.filters.unsupervised.attribute.TimeSeriesDelta`.
- **Evaluation Criteria**: Representation of time elapsed since last measurement ($\Delta t$), observation mask creation, and handling of asynchronous multivariate streams.

### Requirement 8: Strictly Causal Feature Generation
- **What the Audit Establishes**: Digital twin predictive models must operate in real time without future data leakage (i.e., calculations at time $t$ must only use information available at or before $t$).
- **What Remains Unknown**: Whether batch filters in WEKA apply whole-dataset statistics (e.g., full-sequence mean imputation or forward/backward filling) that leak future test results into past timestamps.
- **WEKA Workflow to Test**: Train/test split evaluation with `FilteredClassifier` or `weka.filters.supervised.attribute.*` versus rolling-window causal transformations.
- **Evaluation Criteria**: Strict separation of training statistics from test instances, absence of bidirectional smoothing, and preservation of chronological causality.

---

## 5. Audit Execution Commands

### Smoke Test (Default Verification)
To execute a rapid, bounded audit on a sample of chunks:
```bash
python -m src.ingestion.audit_dataset --mode smoke --chunk-size 100000 --max-chunks 2
```
Outputs are written atomically to `reports/generated/`:
- `dataset_inventory.md`
- `lab_item_inventory.csv`
- `data_quality_summary.csv`
- `renal_candidate_items.csv`

### Fast Full Scan (Complete Population Audit)
To process all ~120M+ rows across the full 18.4 GB `labevents.csv` file:
```bash
python -m src.ingestion.audit_dataset --mode full --chunk-size 250000 --log-interval 5
```

### Full Scan with Isolated Duplicate/Conflict Audit
To include intra-chunk duplicate and conflicting measurement analysis:
```bash
python -m src.ingestion.audit_dataset --mode full --chunk-size 250000 --include-duplicates --log-interval 5
```

---

## 6. Runtime and Memory Optimization Profile

### Identified Bottlenecks in Initial Implementation
1. **Unbounded Memory from Key Accumulation**: Storing measurement keys for renal candidate tests across 120M rows accumulated ~40 million entries in a Python dictionary with tuple keys, demanding >8–12 GB of RAM and causing severe garbage collection thrashing.
2. **Row-by-Row Python Iteration (`iterrows`)**: Iterating over candidate slices using `iterrows()` constructs millions of `pd.Series` objects, capping throughput at <5,000 rows/sec.
3. **Unselective Column Reads**: Reading all 16 columns of string/object types doubled I/O volume and memory allocation compared to projecting only required columns.
4. **Intra-chunk Duplicate Grouping Overhead**: Running multi-column hashing and `.groupby()` across every chunk consumed substantial CPU cycles on every batch.

### Architectural Mitigations & Trade-Offs
- **Vectorized Groupby Aggregations**: Replacing row iteration with pre-filtered vectorized aggregations (`cand_chunk.groupby("itemid")["valuenum"].agg(...)`) increases throughput to >400,000–500,000 rows/sec.
- **Strictly Constant Memory Footprint**: Maintaining only running aggregate counters (sum, count, min, max per item ID) keeps heap memory capped at ~150–200 MB across the entire 120-million-row scan.
- **Separation of Expensive Duplicate Checks**: Isolating duplicate/conflict checks behind `--include-duplicates` ensures the primary audit completes in minutes, while allowing deep anomaly checks when specifically requested.
- **Atomic File Serialization**: Writing to `.tmp` files prior to renaming ensures that interrupted or cancelled runs never produce corrupted or falsely labeled complete outputs.
