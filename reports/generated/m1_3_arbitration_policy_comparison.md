# Module 1.3 Step 2: Conflict & Duplicate Arbitration Policy Comparison Report

**Project**: Nephron-Aware Renal Digital Twin  
**Milestone**: M1.3 Step 2 (Conflict and Duplicate Arbitration Engine)  
**Parent Artifact**: Validated M1.2 Phase B Cohort (`data/processed/cohort_renal_bounded.csv`)  
**Status**: COMPLETED AUDIT REPORT  
**Date**: 2026-10-18  

---

## 1. Executive Summary & Objectives

Module 1.3 Step 2 implements and audits five configurable, auditable arbitration policies for duplicate and conflicting laboratory measurements in the canonical renal-event schema:
- **`POLICY_0_RAW`**: Raw unarbitrated baseline preserving all source records without mutation.
- **`POLICY_A_STORETIME`**: Operational EHR filing precedence (latest storetime for conflicts, earlier storetime for exact duplicates, deterministic tie-breaking by lowest `labevent_id`).
- **`POLICY_B_STAT`**: Clinical priority precedence (prefers STAT over ROUTINE when unambiguous; falls back to Policy A when priorities are identical or unspecified).
- **`POLICY_C_MEAN`**: Analytical mean aggregation for numeric conflicts (synthesizes single aggregate with full provenance linkage to constituent `labevent_id`s; supersedes original rows).
- **`POLICY_D_EXTREMUM`**: Worst-case physiological risk extremum (Creatinine: max, BUN: max, Bicarbonate: min, Potassium: $|K^+ - 4.0|$ extremum). Explicitly restricts scope to these 4 analytes and rejects/flags Sodium and Glucose.

All five policies were benchmarked against the full validated M1.2 Phase B cohort (**2,500 patients, 303,460 rows**). Every source measurement, unit, timestamp, duplicate flag, and conflict flag is strictly preserved without mutation or silent data loss.

---

## 2. Cohort Collision Profile

The Phase B audit established the following baseline collision profile across the 303,460 cohort records:
- **Exact Duplicate Rows**: **188 rows** (94 distinct pairs of size 2; 80 outpatient, 14 inpatient).
- **Conflicting Measurement Rows**: **268 rows** (134 distinct collision epochs of size 2 across identical `(subject_id, itemid, charttime)`).
- **Overlap between Duplicates and Conflicts**: **0 rows** (strictly disjoint sets).
- **Unique Non-Colliding Observations**: **303,004 rows** (99.85% of total cohort).
- **Groups with Mixed `hadm_id`**: **0 groups** in the extracted cohort (all collisions occurred strictly within the same admission or entirely outpatient).

### Collision Distribution by Target Analyte
| Analyte | Item ID | Canonical Unit | Total Rows | Exact Dup Rows (Pairs) | Conflicting Rows (Pairs) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Potassium** | 50971 | mEq/L | 57,014 | 36 (18) | 62 (31) |
| **Glucose** | 50931 | mg/dL | 53,745 | 36 (18) | 52 (26) |
| **Sodium** | 50983 | mEq/L | 52,999 | 36 (18) | 44 (22) |
| **Urea Nitrogen** | 51006 | mg/dL | 50,296 | 28 (14) | 42 (21) |
| **Creatinine** | 50912 | mg/dL | 49,602 | 26 (13) | 30 (15) |
| **Bicarbonate** | 50882 | mEq/L | 39,804 | 26 (13) | 38 (19) |
| **Total** | — | — | **303,460** | **188 (94)** | **268 (134)** |

---

## 3. Policy Execution Comparison Matrix

Each policy was applied to the canonicalized cohort. All records retain their original provenance, with effective numerical values passed forward via `valuenum_effective` and role annotations captured in `arbitration_role`.

| Policy Metric | Policy 0 (Raw) | Policy A (Storetime) | Policy B (Priority) | Policy C (Mean) | Policy D (Extremum, Flag Mode) | Policy D (Supported 4 Analytes Only) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Policy Identifier** | `POLICY_0_RAW` | `POLICY_A_STORETIME` | `POLICY_B_STAT` | `POLICY_C_MEAN` | `POLICY_D_EXTREMUM` | `POLICY_D_EXTREMUM` |
| **Input Rows** | 303,460 | 303,460 | 303,460 | 303,460 | 303,460 | 207,216 |
| **Output Rows** | 303,460 | 303,460 | 303,460 | 303,594 | 303,460 | 207,216 |
| **Retained / Active Events** | **303,460** | **303,232** | **303,232** | **303,232** | **303,184** | **207,048** |
| • `PRIMARY_SELECTED` | 0 | 303,232 | 303,232 | 303,098 | 303,184 | 207,048 |
| • `SYNTHESIZED_AGGREGATE` | 0 | 0 | 0 | 134 | 0 | 0 |
| • `UNARBITRATED_RAW` | 303,460 | 0 | 0 | 0 | 0 | 0 |
| **Superseded Records** | **0** | **228** | **228** | **362** | **228** | **168** |
| • `SUPERSEDED_DUPLICATE` | 0 | 94 | 94 | 94 | 94 | 82 |
| • `SUPERSEDED_CONFLICT` | 0 | 134 | 134 | 268 | 86 | 86 |
| • `UNSUPPORTED_POLICY_D_ANALYTE` | 0 | 0 | 0 | 0 | 96 (48 pairs) | 0 (excluded upfront) |
| **Execution Time** | **0.29 s** | **0.56 s** | **0.62 s** | **0.76 s** | **0.59 s** | **0.38 s** |

*Note on Retained / Active Events and Reconciliation*:
- For Policies A, B, and C, exactly **303,232** effective observations are retained (303,460 raw minus 94 duplicate rows minus 134 superseded conflict rows).
- In Policy C, the 134 conflict pairs (268 rows) are all marked `SUPERSEDED_CONFLICT`, and 134 new `SYNTHESIZED_AGGREGATE` records are appended, preserving total active event count at 303,232 (303,098 primary + 134 synthesized).
- In Policy D (Flag Mode), 96 collision rows belonging to Glucose (52 rows, 26 pairs) and Sodium (44 rows, 22 pairs) across 48 collision epochs are quarantined as `UNSUPPORTED_POLICY_D_ANALYTE`. The 303,460 rows reconcile exactly: 303,184 primary selected + 94 superseded duplicates + 86 superseded conflicts + 96 quarantined unsupported rows = 303,460 rows. Under strict mode (`raise_on_unsupported=True`), Policy D deliberately and safely raises `UnsupportedPolicyAnalyteError`.

---

## 4. Policy Concordance and Disagreement Analysis

Because arbitration rules differ in their clinical and operational assumptions, comparing their selected winners across the 134 conflict groups reveals key empirical trade-offs:

### 4.1 Policy A (Storetime) vs. Policy B (Clinical Priority)
- **Total Conflict Pairs Tested**: 134 pairs.
- **Pairs with Identical Priority**: **126 pairs** (72 Routine/Routine, 54 STAT/STAT).
  - For these 126 pairs, Policy B unambiguously falls back to Policy A (latest storetime, lowest `labevent_id`).
- **Pairs with Mixed Priority (STAT vs. ROUTINE)**: **8 pairs**.
  - In all 8 pairs, the ROUTINE test was filed into the EHR database *after* the STAT test (e.g. STAT filed at 13:59, ROUTINE filed at 15:39).
  - **Policy A** selected the ROUTINE record (latest storetime = 15:39).
  - **Policy B** selected the STAT record (clinical urgency priority).
  - **Discordance Rate**: $8 / 134 = 5.97\%$.

### 4.2 Policy A (Storetime) vs. Policy D (Physiological Extremum)
- **Supported Analyte Pairs**: 86 pairs (Potassium: 31, BUN: 21, Bicarbonate: 19, Creatinine: 15).
- **Concordant Selections**: **57 pairs** (66.28%).
- **Discordant Selections**: **29 pairs** (33.72%).
  - In 29 collision pairs, the worst-case physiological value was filed *earlier* than the less severe value.
  - Policy A selected the later verification result, whereas Policy D selected the higher azotemia/acidosis/dyskalemia value.

---

## 5. Scientific & Operational Neutrality (No Clinical Superiority Claims)

In accordance with good scientific practice, **no arbitration policy is asserted to be clinically superior or universally correct**:

1. **Policy 0 (Raw)**: Preserves the authentic, unadulterated point process but exposes downstream velocity formulations to zero-gap ($\Delta t = 0$) singularities.
2. **Policy A (Operational Storetime)**: Faithfully models the EHR update sequence available to clinical staff, but risks selecting an aged sample re-run or an administrative re-file over an initial valid assay.
3. **Policy B (Clinical Priority)**: Captures urgent clinical attention, but point-of-care (POC) devices frequently used for STAT draws often exhibit wider analytical coefficients of variation than central laboratory wet-chemistry analyzers.
4. **Policy C (Analytical Mean)**: Reduces Gaussian measurement noise if both aliquots represent independent valid draws, but averages in contaminated artifacts if one aliquot suffered hemolysis or line dilution.
5. **Policy D (Physiological Extremum)**: Prioritizes patient safety-envelope monitoring for acute kidney injury and dyskalemia, but is highly sensitive to spurious spikes, bubbles, and pre-analytical artifacts, and cannot be extended to Sodium or Glucose without introducing arbitrary non-physiological heuristics.

The pipeline architecture maintains all five policies as **configurable, parameter-driven options** so downstream sensitivity analyses can evaluate the impact of arbitration choices on renal trajectory representations.

---

## 6. Audit Invariants & Quality Verification

| Requirement | Audit Verification Check | Result |
| :--- | :--- | :--- |
| **Raw Data Immutability** | Input `cohort_renal_bounded.csv` and input DataFrames are deep-copied and never modified in place. | **VERIFIED** |
| **No Silent Data Loss** | Every source record is retained in the output DataFrame with an explicit `arbitration_role`. | **VERIFIED** |
| **Null hadm_id Preservation** | All 80 outpatient duplicate records and all outpatient conflict records are preserved with nullable `hadm_id`. | **VERIFIED** |
| **Deterministic Tie-Breaking** | Identical storetimes or tied priorities are resolved deterministically by lowest integer `labevent_id`. | **VERIFIED** |
| **Full Provenance Linkage** | Column `source_labevent_ids` deterministically links every output record to its original source row(s). | **VERIFIED** |
| **Repeat-Run Idempotence** | Executing any policy repeatedly on identical input produces bitwise identical DataFrames. | **VERIFIED** |
| **Canonical Schema Compliance** | All output DataFrames pass strict `validate_canonical_schema` checks. | **VERIFIED** |
