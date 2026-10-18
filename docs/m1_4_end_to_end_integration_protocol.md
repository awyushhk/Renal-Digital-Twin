# Module 1.4: End-to-End Integration & Final Validation Protocol

**Project**: Nephron-Aware Renal Digital Twin  
**Milestone**: M1.4 End-to-End Integration and Validation  
**Parent Artifacts**: 
- Validated M1.2 Phase B Cohort (`data/processed/cohort_renal_bounded.csv`)
- M1.3 Canonical Schema, Arbitration Engine, Dual-Timeline Features, & Three-Way Baseline Pipeline
**Status**: APPROVED PROTOCOL SPECIFICATION  
**Author**: Antigravity AI & Digital Twin Engineering Team  
**Date**: 2026-10-18  

---

## 1. Overview & Objectives

Module 1.4 establishes the formal integration and validation standard synthesizing all modules across Phase 1 (Data Foundation, Representation, and Audit):
1. **Module 1.1 (Dataset Audit)**: Verification of raw source file inventories, renal laboratory item discovery (item IDs 50912, 51006, 50971, 50983, 50882, 50931), and schema properties.
2. **Module 1.2 (Cohort Extraction & WEKA Benchmark)**: Deterministic bounded cohort extraction (2,500 patients, 303,460 observations), demographic and admission foreign-key linkages, standard unit harmonization, and synthetic baseline benchmark parity.
3. **Module 1.3 (Canonical Representation & Dual-Timeline Engine)**:
   - Canonical 14-column typed event schema.
   - Five configurable conflict and duplicate arbitration policies.
   - Strict separation between retrospective biological trajectories ($F_{bio}$) and operational EHR availability ($F_{ops}$).
   - Mathematical proof of divergence across Arms 1, 2, and 3 (demonstrating observation collapse, future leakage, cross-patient bleed, and outpatient loss).
4. **Module 1.4 (End-to-End Acceptance)**: Automated verification gates ensuring 100% reproducibility, zero regressions, and exact alignment between empirical code outputs and published milestone reports.

---

## 2. Invariants & Verification Gates

### Gate 1: Cohort Reference Artifact Integrity
- **Target File**: `data/processed/cohort_renal_bounded.csv`
- **Dimensionality**: Exactly 303,460 rows, 2,500 unique `subject_id`s, 30 canonical and demographic columns.
- **SHA-256 Hash Check**: Evaluates full file SHA-256 against reference hash:
  `9534adfda88896433cae38e87efcd66008bf558595a5734d5251181a615b4d35`
  *Protocol Requirement*: If a discrepancy is observed, the system logs a detailed diagnostic report; hardcoded hashes are treated as reference integrity checkpoints rather than arbitrary validity barriers.

### Gate 2: Full-Cohort Policy A Arbitration Accounting
- **Input**: Complete 303,460 bounded cohort rows converted via `create_canonical_event_frame()`.
- **Policy**: `POLICY_A_STORETIME` (Operational filing precedence with lowest `labevent_id` deterministic tie-breaking).
- **Exact Numeric Invariants**:
  * Total Input Rows: `303,460`
  * Total Output Rows: `303,460` (100.0% retention; zero raw source loss)
  * Active Effective Records (`PRIMARY_SELECTED`): `303,232` (99.9249%)
  * Superseded Exact Duplicates (`SUPERSEDED_DUPLICATE`): `94` (0.0310%)
  * Superseded Numeric Conflicts (`SUPERSEDED_CONFLICT`): `134` (0.0442%)
  * Total Superseded Records: `228` (0.0751%)
  * Mixed `hadm_id` Collisions: `0` (all collisions strictly intra-admission or pure outpatient)
- **Traceability Requirement**: Every superseded record maintains full pointer linkage via `superseded_by_labevent_id` to its primary selected counterpart.

### Gate 3: Dual-Timeline Operational Filtration & Immutability
- **Principle**: Real-time operational models evaluated at query time $t_{query}$ must never observe results filed after $t_{query}$ (`storetime > t_{query}`), regardless of when the biological specimen was collected (`charttime <= t_{query}`).
- **Empirical Replay Invariant**:
  * Specimen drawn at 10:00:00, filed at 12:30:00.
  * Snapshot at 11:00:00 must exclude the record (count = 0).
  * Snapshot at 12:30:00 must include the record (count = 1).
  * Re-querying at 11:00:00 must produce an identical DataFrame (immutability; late-arriving results never retroactively rewrite previously emitted feature vectors).
- **Reporting Velocity Designation**: Rates of change derived from `storetime` differences are designated as reporting arrival velocities, never inferred as physiological kinetics.

### Gate 4: Patient Boundary Isolation & Co-temporal Ratio Gating
- **Patient Boundary Reset**: All continuous-time elapsed clocks ($\tau = 0$), inter-event gaps ($\Delta t = \text{NaN}$), and running analyte statistics must strictly reset at `subject_id` boundaries.
- **Co-temporal BUN/Creatinine Ratio**: Computed if and only if both analytes belong to the exact same patient and observation epoch (`charttime`), with non-negative reporting delay and positive creatinine. When non-co-temporal, the system returns `NaN` with explicit status flag `MISSING_CONSTITUENT`.

### Gate 5: Dynamic Negative Controls & Three-Way Divergence
The pipeline evaluates three distinct representation paradigms:
1. **Arm 1 (Conventional 24-Hour Regular Grid with LOCF)**:
   - Measures daily grid expansion ratio (8.7266x full cohort) and empty grid fraction (98.2059%).
   - Demonstrates intra-day collapse: 43,363 active observations collapsed to daily means.
2. **Arm 2 (Flawed Modeling Controls)**:
   - *Control 2A (Normalization Leakage)*: Future extreme observation shifts global min-max scaling, causing measurable historical value drift ($> 0$).
   - *Control 2B (Patient Boundary Bleed)*: Un-reset time series delta computes cross-patient transition jump ($> 0$).
   - *Control 2C (Admissions Inner-Join Loss)*: Inner join against admissions table irreversibly drops 103,354 records (34.0585%) and 378 pure outpatient patients (15.1200%).
3. **Arm 3 (Proposed Renal-Aware Representation)**:
   - 0 intra-day collapse, 0 future-information leakage, 0 cross-patient bleed, 0 outpatient loss, 0 zero-gap singularities, 0 operational lookahead violations.

---

## 3. Architecture & Code Structure

The implementation is organized across validated, modular components:

```
src/
├── ingestion/
│   ├── audit_dataset.py           # M1.1 Raw MIMIC-IV audit
│   └── extract_cohort.py          # M1.2 Phase B bounded cohort extraction
├── temporal/
│   ├── canonical_schema.py        # M1.3 Step 1 Canonical 14-column event schema
│   ├── conflict_arbitration.py    # M1.3 Step 2 Configurable arbitration engine
│   ├── temporal_features.py       # M1.3 Step 3 Irregular features & dual-timeline engine
│   └── representation_pipeline.py # M1.3 Step 4 Three-way comparative baseline pipeline
└── validation/
    ├── __init__.py                # Package exports
    ├── weka_benchmark.py          # M1.2 Phase A WEKA synthetic benchmark harness
    └── end_to_end_validation.py   # M1.4 End-to-end integration & acceptance runner

tests/
├── test_audit_dataset.py
├── test_canonical_schema.py
├── test_cohort_extraction.py
├── test_conflict_arbitration.py
├── test_temporal_features.py
├── test_representation_pipeline.py
├── test_weka_benchmark.py
└── test_end_to_end_validation.py  # M1.4 End-to-end regression & acceptance test suite
```

---

## 4. Execution & Verification Commands

### Automated Test Suite Execution
Execute the complete test suite including unit tests and M1.4 end-to-end verification:
```powershell
.venv\Scripts\pytest.exe -v
```

### Fast Regression Test (M1.4 Only)
```powershell
.venv\Scripts\pytest.exe tests/test_end_to_end_validation.py -v
```

### Full-Cohort Acceptance Run & Report Generation
Execute the standalone acceptance harness to verify all full-cohort numbers and re-generate `reports/generated/m1_4_end_to_end_validation_report.md`:
```powershell
.venv\Scripts\python.exe -m src.validation.end_to_end_validation
```

---

## 5. Acceptance Checklist

- [x] Reference cohort artifact exists with 303,460 rows, 2,500 patients, 30 columns.
- [x] SHA-256 hash verified: `9534adfda88896433cae38e87efcd66008bf558595a5734d5251181a615b4d35`.
- [x] Exact Policy A arbitration verified on full cohort: 303,232 active, 94 dupes, 134 conflicts, 228 total superseded.
- [x] Operational timeline lookahead prevention and snapshot immutability verified.
- [x] Dynamic negative controls (leakage, bleed, outpatient loss) verified.
- [x] Fast 5,000-row integration pipeline verified in < 20 seconds.
- [x] Comprehensive Markdown audit report generated at `reports/generated/m1_4_end_to_end_validation_report.md`.
- [x] Raw data files in `data/raw` remain unmodified.
- [x] 100% of test suite passes cleanly.
