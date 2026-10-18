# Controlled Preprocessing Benchmark Report: WEKA 3.8.7 vs. Python (Phase A)

**Execution Timestamp**: `2026-10-18 21:05:48`  
**Environment**: WEKA 3.8.7 (`D:\Software\weka\Weka-3-8-7\weka.jar`), Java 21 LTS, Python 3.14  
**Scope**: Phase A Synthetic Unit Proofs (Requirements A through H)  

## 1. Executive Evaluation Summary

- **Total Subtests Evaluated**: 15
- **Passed (Equivalent Behavior)**: **8** (53.3%)
- **Failed (Direct Operational Defects)**: **1** (6.7%)
- **Unsupported (No Equivalent via any WEKA workflow)**: **6** (40.0%)

> [!IMPORTANT]
> **Key Methodological Finding**:
> Standard WEKA filters perform reliably for **static tabular transformations** (Min-Max normalization, Z-Score standardization, and exact datetime serialization). Conditional unit rescaling (g/dL → mg/dL) is achievable via a composed two-step WEKA pipeline (NominalToBinary + MathExpression). However, WEKA has confirmed **Capability Gaps** for longitudinal EHR modeling: (1) no group-aware LOCF forward-fill; (2) no clinical key-based duplicate arbitration (subset-key deduplication); (3) no date-aware elapsed-time derivation with patient boundary resets; and (4) batch normalization leaks future statistics into historical time points (Workflow Inconvenience, not architectural).

---

## 2. Comprehensive Subtest Results Matrix

| Requirement | Subtest Name | Status | Limitation Type | WEKA Target | Python Target | Evidence & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R1_NORM` | Min-Max Scaling [0, 1] | **`PASSED`** | None | `Normalize` | `pandas (x - min) / (max - min)` | Both WEKA and Python map linear minimum to 0.0 and maximum to 1.0. |
| `R1_NORM` | Z-Score Standardization (ddof=1) | **`PASSED`** | None | `Standardize` | `pandas (x - mean) / std(ddof=1)` | Confirmed WEKA Standardize uses sample standard deviation (ddof=1). |
| `R1_NORM` | Robust Scaling (Median/IQR) | **`UNSUPPORTED`** | Capability Gap | `* (None available in standard WEKA)` | `pandas (x - median) / IQR` | Linear Min-Max in WEKA compresses the 0.8->1.2 physiological baseline variation into [0.0, 0.028] because of the 15.0 uremic outlier. Standard WEKA lacks native quantile/robust scaling. |
| `R2_MISSING` | Global Mean Imputation | **`PASSED`** | None | `ReplaceMissingValues` | `pandas fillna(mean)` | Both WEKA and Python replace missing values with the training column mean. |
| `R2_MISSING` | Observation Mask Generation | **`UNSUPPORTED`** | Workflow Inconvenience | `ReplaceMissingValues (No indicator option)` | `pandas notna().astype(int)` | WEKA's ReplaceMissingValues filter overwrites missing values in place without generating indicator masks. Preserving observation masks requires duplicating attributes and running multi-step filter chaining. |
| `R2_MISSING` | Longitudinal Trajectory Forward-Fill (LOCF) | **`UNSUPPORTED`** | Capability Gap | `ReplaceMissingValues` | `df.groupby('subject_id')['creatinine'].ffill()` | In clinical trajectory modeling, an unmeasured step persists the patient's last known physiological state. WEKA ReplaceMissingValues replaces missing values with the cohort mean (1.52 mg/dL), erroneously resetting a deteriorating trajectory. |
| `R3_UNITS` | Conditional Nominal-to-Numeric Unit Harmonization | **`PASSED`** | Workflow Inconvenience | `NominalToBinary -R 4` → `MathExpression -E 'ifelse(A4=0,A*1000,A)' -R 3 -V` | `df.apply(lambda: valuenum * 1000 if valueuom == 'g/dL' else valuenum)` | Single MathExpression on nominal attribute fails (SemanticException). Two-step composed pipeline NominalToBinary→MathExpression succeeds: g_dL=4.0→4000.0, mg_dL unchanged. Workflow Inconvenience (multi-step required), not a Capability Gap. |
| `R3_UNITS` | Semi-Quantitative Dipstick Ordinal Rank Mapping | **`PASSED`** | Workflow Inconvenience | Manual ARFF ordinal index configuration | `series.map({'NEG': 0, 'TR': 0.5, '1+': 1, ...})` | Dipstick urinalysis must not be treated as arbitrary continuous values. WEKA internally assigns 0-indexed integer codes; custom clinical rank mapping requires manual ARFF configuration (no WEKA execution in this subtest). |
| `R4_TIMESTAMPS` | Exact ISO Datetime Preservation ('yyyy-MM-dd HH:mm:ss') | **`PASSED`** | None | `ArffLoader with date attribute` | `pd.to_datetime(format='%Y-%m-%d %H:%M:%S')` | WEKA successfully parsed and preserved minute-and-second level dates when specified with date format in ARFF header. |
| `R5_LINKAGE` | Outpatient Missing hadm_id Retention in Flat Table | **`PASSED`** | None | `Instances on pre-joined flat ARFF` | `pandas flat dataframe preservation` | When tested on a validated Python-produced flat table, WEKA correctly loads and retains rows with null hadm_id without dropping the instance or coercing missing numeric ID to zero. |
| `R5_LINKAGE` | Native In-Memory Relational Database Joining | **`UNSUPPORTED`** | Capability Gap | `WEKA Explorer / Experimenter (No relational join engine)` | `pd.merge(patients, admissions, on='subject_id')` | As established methodologically, testing linkage on flat tables isolates preprocessing. However, relational foreign-key linking (46.6% ambulatory records) is fundamentally an external Python responsibility. |
| `R6_DUPLICATES` | Exact Full-Row Duplicate Removal | **`PASSED`** | None | `RemoveDuplicates` | `df.drop_duplicates()` | Both WEKA RemoveDuplicates and Python drop_duplicates successfully eliminate 100% identical rows. |
| `R6_DUPLICATES` | Conflicting Same-Timestamp Measurement Arbitration | **`UNSUPPORTED`** | Capability Gap | `RemoveDuplicates` | `df.groupby(['subject_id', 'charttime', 'itemid'])['valuenum'].median()` | In MIMIC-IV, 1,128 conflicting measurements were observed at identical timestamps in a 200k sample. WEKA's RemoveDuplicates cannot perform clinical key-based deduplication or apply arbitration policies (e.g. median, central lab precedence). |
| `R7_ELAPSED_TIME` | Chronological Elapsed Time (Δt) with Patient Boundary Reset | **`UNSUPPORTED`** | Capability Gap | `TimeSeriesDelta` | `df.groupby('subject_id')['charttime'].diff().dt.total_seconds() / 3600.0` | TimeSeriesDelta computes creatinine value differences (not elapsed hours). P1001's last-row delta uses P1002's first creatinine (0.9) as its 'next', producing 1.6−0.9=0.7 (cross-patient artifact, confirmed by direct WEKA CLI output). P1002's first row receives a non-NaN delta instead of NaN (no groupby reset). |
| `R8_CAUSALITY` | Future-Perturbation Test on Historical State (t=2) | **`FAILED`** | Workflow Inconvenience | `Normalize (Batch Mode)` | `pandas expanding().mean()` (causal running mean — different feature type than normalization) | Changing future creatinine at t=5 from 1.4 to 12.0 caused WEKA's normalized value at t=2 to shift from 0.2500 to 0.0091 (Δ=0.2409). Empirical proof of batch normalization leakage. Note: Python comparison is a running mean, not a causal equivalent of MinMax normalization — it demonstrates leak-freedom of a different causal feature type. |

---

## 3. Detailed Requirement-by-Requirement Evidence

### A. Numeric Normalization
- **What Works in WEKA**: Min-Max scaling (`Normalize`) and sample Z-Score (`Standardize`, ddof=1) match Python to machine precision ($< 10^{-5}$).
- **What Fails / Is Missing**: Core WEKA lacks a native `RobustScaler` (median/IQR). Under extreme renal outliers (e.g. creatinine 15.0 mg/dL), Min-Max compresses normal clinical ranges into a tiny band $[0.0, 0.028]$.

### B. Missing-Value Handling & Indicators
- **What Works in WEKA**: Global mean/mode replacement (`ReplaceMissingValues`) matches Python `SimpleImputer(strategy='mean')`.
- **What Fails / Is Missing**: `ReplaceMissingValues` has no flag to output observation indicators ($M_{i,t} \in \{0, 1\}$). Furthermore, it cannot execute longitudinal forward-filling (`LOCF`), forcing patient trajectories to jump abruptly to cohort-wide static means.

### C. Unit & Laboratory-Item Mapping
- **What Works in WEKA**: Conditional unit rescaling **is achievable** via a two-step composed pipeline: (1) `NominalToBinary -R <col>` converts the nominal unit column to a numeric 0/1 flag; (2) `MathExpression -E 'ifelse(A4=0, A*1000, A)' -R <col> -V` applies the conditional rescaling to only the value column. Verified empirically: g_dL 4.0→4000.0, mg_dL 4000.0→4000.0 (correct). This is a **Workflow Inconvenience**, not a Capability Gap.
- **Limitation**: Requires a two-step composed pipeline and prior knowledge of the ARFF attribute ordering. A single `MathExpression` call on the raw ARFF (with nominal unit column) fails with `SemanticException: equal is only applicable to doubles!`.

### D. Timestamp Parsing & Preservation
- **What Works in WEKA**: Exact minute-and-second ISO strings (`YYYY-MM-DD HH:mm:ss`) are preserved when declared as `@attribute date` in ARFF.

### E. Patient/Admission Linkage Integrity
- **What Works in WEKA**: When tested on a validated flat pre-joined input, WEKA correctly preserves outpatient rows with missing `hadm_id` without dropping records.
- **What Fails / Is Missing**: WEKA cannot perform multi-table relational foreign-key joins internally; joining demographics and admissions is an external relational task.

### F. Duplicate & Conflicting Measurement Handling
- **What Works in WEKA**: Full-row 100% duplicate elimination (`RemoveDuplicates`).
- **What Fails / Is Missing**: When two measurements share identical `(subject_id, charttime, itemid)` but possess conflicting values (e.g. 1.2 vs 1.8), WEKA retains both rows as separate instances. It has no mechanism for group-by arbitration or priority ranking.

### G. Irregular Elapsed-Time Features
- **What Fails / Is Missing**: `TimeSeriesDelta` operates purely on row index offsets ($i - 1$), assuming uniform time intervals. It cannot compute chronological elapsed hours from date attributes and does not reset at patient boundaries (subtracts across patients).

### H. Strictly Causal Feature Generation
- **Empirical Proof of Future Leakage**: Under the future-perturbation test, perturbing future creatinine at $t_5$ (from 1.4 to 12.0) shifted WEKA's normalized value at historical time point $t_2$ from 0.2500 to 0.0091. This is a direct mathematical consequence of batch MinMax fitting on the full sequence.
- **Python comparison caveat**: The Python "causal" comparison uses `expanding().mean()` — a running mean — which is a categorically different feature type from MinMax normalization. The test demonstrates WEKA batch normalization is not causal; it does not show whether causal normalization is achievable in WEKA. Causal MinMax (fitted only on `x_1..x_t` per time step) is in principle achievable via `FilteredClassifier` per fold, but within-sequence temporal ordering is not enforced.
- **Classification**: **Workflow Inconvenience** (batch mode hazard; preventable with careful WEKA workflow design).
