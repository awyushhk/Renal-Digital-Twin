# Nephron-Aware Renal Digital Twin

> A continuous-time physiological and machine-learning framework for modeling dynamic renal function kinetics, chronic kidney disease (CKD) progression, and acute kidney injury (AKI) risk trajectories using MIMIC-IV clinical observational data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Pytest-94%20Passed-brightgreen.svg)]()
[![Data Integrity](https://img.shields.io/badge/MIMIC--IV-3.1%20Audited-orange.svg)]()
[![Milestone](https://img.shields.io/badge/Milestone-M1.4%20Verified%20(PASS)-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## 1. Executive Summary & Vision

Traditional clinical machine learning benchmarks typically resample longitudinal laboratory measurements onto uniform 24-hour regular grids with Last Observation Carried Forward (`LOCF`) imputation. In renal clinical monitoring, this standard practice introduces profound distortions:
- **Intra-Day Measurement Collapse**: In acute clinical episodes (e.g., ICU resuscitation or rapidly evolving AKI), patients receive multiple lab draws per day. Daily grids collapse multiple readings into single daily aggregates, obliterating rapid creatinine surges and kinetic inflection points.
- **Extreme Sparsity & Redundant Grid Expansion**: In outpatient and chronic care trajectories, intervals between clinic visits span weeks or months. Resampling onto a 24-hour grid produces an **8.7x artificial step expansion** with an **empty-grid step fraction exceeding 98.2%**.
- **Pervasive Historical Data Leakage**: Standard global normalizers (such as full-batch min-max scaling) introduce future-information contamination into historical time points, while un-reset rolling deltas leak features across patient boundaries.
- **Outpatient Erasure**: Blind inner joins on hospital encounters (`hadm_id`) silently drop outpatient observations, discarding **34.1% of lab records and 15.1% of entire patient histories**.

The **Nephron-Aware Renal Digital Twin** addresses these foundational challenges through a continuous-time, irregular point-process representation featuring strict **Dual-Timeline Information Filtering**, auditable conflict arbitration, and mathematical anti-leakage guarantees.

---

## 2. Core Architectural Pillars

### A. Dual-Timeline Engine
Clinical laboratory data inherently inhabit two separate chronological dimensions:
1. **Biological Timeline ($F_{bio}$)**:
   - Driven by specimen draw time (`charttime`).
   - Reconstructs continuous retrospective physiological kinetics (irregular velocity $\Delta v / \Delta t$ and unequal-interval acceleration $\Delta a / \Delta t$).
   - Explicitly tagged as `is_retrospective_only = True` (non-deployable for live clinical alerts).
2. **Operational Timeline ($F_{ops}$)**:
   - Driven by electronic health record filing time (`storetime`).
   - Generates immutable feature snapshots evaluated at explicit clinical decision points $t_{query}$ (`storetime <= t_{query}`).
   - Results with `storetime > t_{query}` are strictly excluded (zero operational lookahead).
   - Late-arriving results entering at subsequent query times never retroactively rewrite historical snapshots.
   - Reporting cadence rates are explicitly documented as reporting velocities, never conflated with physiological kinetics.

### B. Provable Anti-Leakage & Patient Boundary Isolation
- **Co-temporal Symmetry**: Multi-analyte laboratory draws at identical collection times form co-temporal observation epochs. Ratios (e.g., BUN / Creatinine) are evaluated exclusively within synchronous epochs with non-negative reporting delays.
- **Strict Boundary Resets**: Continuous-time elapsed clocks ($\tau = 0$), inter-event gaps ($\Delta t = \text{NaN}$), and rolling summary statistics reset completely at patient ID transitions.
- **Causal Local Normalization**: Eliminates future-to-past contamination demonstrated in conventional full-batch scalers.

### C. Configurable & Auditable Conflict Arbitration
Accommodates duplicate and colliding laboratory draws across 5 configurable policies:
- `POLICY_0_RAW`: Raw baseline preserving all source records without mutation.
- `POLICY_A_STORETIME`: Operational EHR filing precedence (latest storetime for conflicts, earlier storetime for exact duplicates, deterministic tie-breaking by lowest `labevent_id`).
- `POLICY_B_STAT`: Clinical priority precedence (prefers STAT over ROUTINE when unambiguous; falls back to Policy A).
- `POLICY_C_MEAN`: Analytical mean aggregation for numeric conflicts with complete provenance linkage.
- `POLICY_D_EXTREMUM`: Worst-case physiological risk extremum (Creatinine: max, BUN: max, Bicarbonate: min, Potassium: $|K^+ - 4.0|$ extremum).

---

## 3. Repository Architecture

```text
nephron-aware-renal-digital-twin/
├── configs/
│   └── paths.yaml                       # Configurable paths for data, reports, and parameters
├── data/
│   ├── raw/                             # Read-only raw MIMIC-IV 3.1 CSV files (git-ignored)
│   ├── interim/                         # Intermediate extraction artifacts (git-ignored)
│   └── processed/                       # Validated bounded cohorts (git-ignored)
├── docs/
│   ├── m1_1_dataset_audit.md            # Milestone M1.1 audit report & raw schema inventory
│   ├── m1_2_phase_b_cohort_protocol.md  # Milestone M1.2 Phase B cohort extraction protocol
│   ├── m1_2_weka_comparison_plan.md     # Phase A WEKA 3.8.7 benchmark comparison plan
│   ├── m1_3_temporal_representation_protocol.md # Milestone M1.3 representation protocol
│   └── m1_4_end_to_end_integration_protocol.md  # Milestone M1.4 integration & verification protocol
├── reports/
│   └── generated/                       # Audited milestone reports & aggregate summaries
│       ├── dataset_inventory.md         # M1.1 Raw MIMIC-IV data quality & distribution profile
│       ├── lab_item_inventory.csv       # Scanned laboratory items and candidate frequencies
│       ├── renal_candidate_items.csv    # 6 core renal laboratory items (item IDs 50912, 51006, etc.)
│       ├── cohort_quality_report.md     # M1.2 Demographic, admission, and unit audit
│       ├── cohort_summary.csv           # M1.2 Aggregate cohort summary metrics
│       ├── cohort_summary.json          # M1.2 Provenance metadata & reference SHA-256 hashes
│       ├── weka_benchmark_summary.md    # M1.2 Phase A WEKA 3.8.7 vs. Python benchmark report
│       ├── weka_benchmark_results.csv   # M1.2 Machine-readable benchmark subtest outputs
│       ├── m1_3_arbitration_policy_comparison.md # M1.3 Step 2 Policy 0/A/B/C/D evaluation
│       ├── m1_3_step3_feature_generation_report.md # M1.3 Step 3 Feature derivation report
│       ├── m1_3_three_way_baseline_comparison_report.md # M1.3 Step 4 Three-arm audit report
│       └── m1_4_end_to_end_validation_report.md # M1.4 Final acceptance audit report (PASS)
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── audit_dataset.py             # Memory-conscious chunked dataset auditor
│   │   └── extract_cohort.py            # Bounded cohort extraction & relational join engine
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── canonical_schema.py          # 14-column canonical event schema & validator
│   │   ├── conflict_arbitration.py      # Multi-policy arbitration & provenance engine
│   │   ├── temporal_features.py         # Dual-timeline continuous feature derivation
│   │   └── representation_pipeline.py   # Three-way comparative baseline audit pipeline
│   └── validation/
│       ├── __init__.py
│       ├── weka_benchmark.py            # WEKA 3.8.7 controlled synthetic benchmark harness
│       └── end_to_end_validation.py     # Comprehensive end-to-end acceptance runner
├── tests/
│   ├── __init__.py
│   ├── fixtures/weka_benchmark/         # Deterministic synthetic test fixtures
│   ├── test_audit_dataset.py            # M1.1 Ingestion audit tests
│   ├── test_cohort_extraction.py        # M1.2 Cohort extraction & unit harmonization tests
│   ├── test_weka_benchmark.py           # M1.2 WEKA comparative benchmark tests
│   ├── test_canonical_schema.py         # M1.3 Step 1 Canonical schema tests
│   ├── test_conflict_arbitration.py     # M1.3 Step 2 Collision arbitration tests
│   ├── test_temporal_features.py        # M1.3 Step 3 Dual-timeline feature tests
│   ├── test_representation_pipeline.py  # M1.3 Step 4 Three-arm comparative tests
│   └── test_end_to_end_validation.py    # M1.4 End-to-end integration & acceptance tests
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project overview & documentation
```

---

## 4. Phase 1 Milestones & Verification Achievements

| Milestone | Scope & Deliverable | Core Quantitative Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **M1.1 Dataset Audit** | Memory-conscious scan of full 18.4 GB MIMIC-IV `labevents.csv`. | 6 candidate renal analytes isolated; duplicate rates cataloged. | **COMPLETED** |
| **M1.2 Phase B Cohort** | Deterministic bounded cohort extraction (2,500 patients). | 303,460 observations; 100% standard unit harmonization; zero data loss. | **COMPLETED** |
| **M1.2 Phase A WEKA Benchmark** | Controlled synthetic evaluation of WEKA 3.8.7 across 8 requirements. | Parity verified on standard filters; 6 architectural capability gaps proven. | **COMPLETED** |
| **M1.3 Step 1 Canonical Schema** | 14-column typed event schema with reporting delays and flags. | 100% schema compliance; non-negative delay anomaly isolation. | **COMPLETED** |
| **M1.3 Step 2 Arbitration Engine** | Five configurable collision policies benchmarked on full cohort. | Exact collision accounting: 94 dupes, 134 conflicts, 228 total superseded. | **COMPLETED** |
| **M1.3 Step 3 Dual Timelines** | Biological ($F_{bio}$) vs. Operational ($F_{ops}$) feature derivation. | Causal query-time filtering; snapshot immutability; 0 zero-gap singularities. | **COMPLETED** |
| **M1.3 Step 4 Three-Arm Pipeline** | Comparative audit: Conventional Grid vs. Flawed Controls vs. Arm 3. | 8.7x grid expansion proven; 43,363 collapsed observations preserved. | **COMPLETED** |
| **M1.4 End-to-End Acceptance** | Comprehensive integration harness & reference hash verification. | 94/94 passing tests; full cohort verification verdict: **PASS**. | **VERIFIED** |

---

## 5. Three-Way Comparative Representation Matrix

Empirical comparison across the validated 303,460-row renal cohort:

| Representation Dimension | Arm 1 (Conventional 24h Grid) | Arm 2 (Deliberately Flawed Controls) | Arm 3 (Proposed Digital Twin) |
| :--- | :--- | :--- | :--- |
| **Temporal Granularity** | Discrete 24-hour steps with LOCF | Flawed time-series offsets | **Continuous irregular point process** |
| **Grid Expansion Ratio** | **8.7266x** (2,648,183 steps / 303,460 obs) | Unaligned | **1.0x** (exact point observations) |
| **Empty Grid Step Fraction** | **98.2059%** empty resampled days | Unaligned | **0.0%** (zero artificial step padding) |
| **Intra-Day Observation Preservation** | **43,363 active observations collapsed** | Arbitrary overwrite | **100.0% preserved** (0 collapsed rows) |
| **Future Normalization Leakage** | 0.0 (local) | **Historical value drift $> 0$** | **0.0 (strictly causal)** |
| **Patient Boundary Isolation** | Clean resets | **Cross-patient delta bleed detected** | **Strict resets ($\Delta t = \text{NaN}, \tau = 0$)** |
| **Outpatient Record Retention** | 100% retained (left joins) | **103,354 rows (34.1%) & 378 pts (15.1%) lost** | **100% retained** (all visits preserved) |
| **Operational Lookahead Gate** | N/A (retrospective daily) | N/A | **Zero lookahead ($storetime \le t_{query}$)** |
| **Snapshot Immutability** | N/A | Retrospective rewriting | **Byte-for-byte immutable replay** |

---

## 6. Getting Started

### 1. Prerequisites & Environment Setup
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Recommended: Windows PowerShell, Linux Bash, or macOS Terminal

Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install pinned dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Running Automated Tests
Execute the complete test suite (94 passing tests across all 8 modules):
```powershell
pytest -v
```

Execute only the M1.4 End-to-End Integration & Acceptance test suite:
```powershell
pytest tests/test_end_to_end_validation.py -v
```

### 3. Running Standalone Acceptance Validation
Execute the standalone acceptance harness to verify all full-cohort numerical invariants and re-generate `reports/generated/m1_4_end_to_end_validation_report.md`:
```powershell
python -m src.validation.end_to_end_validation
```

### 4. Running Cohort Extraction (Smoke vs. Full)
The cohort extraction engine operates in memory-conscious streaming chunks:
```powershell
# Smoke test on bounded sample (fast verification)
python -m src.ingestion.audit_dataset --mode smoke --chunk-size 100000 --max-chunks 2

# Full population audit
python -m src.ingestion.audit_dataset --mode full --chunk-size 250000
```

---

## 7. Data Privacy, Governance & PhysioNet Credentialing

This project uses clinical data from the **MIMIC-IV** database (Medical Information Mart for Intensive Care, version 3.1), provided by the MIT Laboratory for Computational Physiology and Beth Israel Deaconess Medical Center via PhysioNet.

- **Data Use Agreement (DUA)**: MIMIC-IV is restricted-access credentialed clinical health data under HIPAA regulations. Direct sharing or publishing of raw patient-level records to public version control repositories is strictly prohibited.
- **Repository Isolation**: All patient-level data files (`data/raw/*`, `data/interim/*`, `data/processed/*`) are strictly excluded via `.gitignore`.
- **Reproducibility**: Researchers with credentialed PhysioNet access can place raw MIMIC-IV CSVs into `data/raw/` and regenerate identical bounded cohorts and audit reports using the deterministic extraction scripts.

---

## 8. Roadmap: Transition to Phase 2

With the completion of **Phase 1 (Data Foundation, Representation, and Audit)**, the data pipeline is fully prepared for **Phase 2: Digital Twin Modeling & Nephron-Aware Differential Equations**:
1. **Physiological ODE/SDE Formulation**:
   - Single-nephron glomerular filtration rate (SNGFR) dynamics.
   - Creatinine generation kinetics ($G_{Cr}$), volume of distribution ($V_d$), and tubuloglomerular feedback (TGF).
2. **Hybrid Physics-Informed Neural Networks (PINNs)**:
   - Coupling continuous-time neural ordinary differential equations with renal physiology constraints.
   - Modeling acute kidney injury (AKI) insult and chronic nephron loss trajectories.
3. **Clinical Counterfactual Simulation**:
   - Simulating nephrotoxic drug challenges, fluid resuscitation, and renal functional reserve (RFR) depletion.

---

## 9. Authors & Citation

**Digital Twin Engineering & Renal Analytics Research Team**  
Developed for Nephron-Aware Renal Function Trajectory Modeling and Clinical Decision Support.
