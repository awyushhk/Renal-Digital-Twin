# Module 1.3: Renal-Aware Temporal Representation Protocol

**Project**: Nephron-Aware Renal Digital Twin  
**Milestone**: M1.3 (Renal-Aware Temporal Representation)  
**Parent Artifact**: Validated M1.2 Phase B Cohort (`data/processed/cohort_renal_bounded.csv`)  
**Status**: APPROVED DESIGN SPECIFICATION  
**Last Updated**: 2026-10-18  

---

## 1. Scientific & Engineering Objectives

The objective of Module 1.3 is to construct a **deterministic, irregular-time, observation-aware temporal representation** for the six validated blood chemistry analytes from MIMIC-IV 3.1.

Unlike standard machine learning pipelines that impose artificial 24-hour daily resampling grids, forward-fill missing data across unobserved periods, and leak future normalization statistics, Module 1.3:
1. Preserves true biological observation timestamps (`charttime`) and irregular elapsed times.
2. Formalizes a dual-time filtration separating biological physiological time (`charttime`) from operational EHR filing time (`storetime`), precluding real-time deployment leakage.
3. Formulates simultaneous multi-analyte observations as co-temporal epochs and prevents zero-gap ($\Delta t = 0$) rate-of-change singularities.
4. Preserves all raw clinical records and separates raw audit flags from experimental arbitration policies.
5. Implements a 3-way evaluation protocol (fair conventional baseline, deliberate flawed controls, and proposed representation).
6. Separates data-integrity and temporal-correctness metrics from predictive performance claims.

All raw MIMIC-IV source files in `data/raw/` and all M1.1/M1.2 audit artifacts remain **strictly read-only and immutable**.

---

## 2. Canonical Renal-Event Schema

The canonical schema establishes an immutable, typed contract for laboratory records. It preserves all source fields, data types, units, and flags, while appending standardized dual timestamps, reporting delays, delay anomaly flags, and arbitration provenance.

### 2.1 Schema Definition & Invariant Rules

| Field Name | Type | Nullable | Source | Clinical & Engineering Invariants |
| :--- | :--- | :--- | :--- | :--- |
| **`labevent_id`** | `int64` | No | `labevents.csv` | Primary key of the source laboratory event in MIMIC-IV. |
| **`subject_id`** | `int64` | No | `labevents.csv` | De-identified administrative patient identifier. |
| **`hadm_id`** | `float64` / `Int64` | **Yes** | `labevents.csv` | Hospital admission identifier. **Null for outpatient / ED labs** (103,354 rows; 34.06%). Must be preserved via LEFT JOIN without row loss. |
| **`itemid`** | `int64` | No | `labevents.csv` | One of six target analyte IDs (`50912`, `51006`, `50971`, `50983`, `50882`, `50931`). |
| **`analyte_name`** | `string` | No | Target mapping | Canonical name: `creatinine`, `urea_nitrogen`, `potassium`, `sodium`, `bicarbonate`, `glucose`. |
| **`charttime`** | `datetime64[ns]` | No | `labevents.csv` | **Physiological event timestamp**: Blood specimen draw time in vivo. |
| **`storetime`** | `datetime64[ns]` | No | `labevents.csv` | **Operational event timestamp**: Lab assay verification and EHR database filing time. |
| **`reporting_delay_hours`** | `float64` | No | Derived | $\Delta t_{\text{rep}} = \frac{\text{storetime} - \text{charttime}}{3600}$. In MIMIC cohort: median $1.38\text{ h}$, range $0.0\text{ h}$ to $144.3\text{ h}$. |
| **`delay_anomaly_flag`** | `category` | No | Derived | `NORMAL_DELAY` ($\Delta t_{\text{rep}} > 0$), `ZERO_DELAY` ($= 0$), or `NEGATIVE_REPORTING_DELAY` ($< 0$, clock skew / clerical error). |
| **`raw_value_text`** | `string` | Yes | `labevents.csv` (`value`) | Unmodified original string output from laboratory instrument. |
| **`raw_valuenum`** | `float64` | Yes | `labevents.csv` (`valuenum`) | Unmodified original numeric float (NaN for 110 missing observations). |
| **`raw_valueuom`** | `string` | Yes | `labevents.csv` (`valueuom`) | Unmodified original unit of measure string. |
| **`valuenum_harmonized`** | `float64` | Yes | Tested conversion | Standardized numerical value in canonical unit (never overwrites raw). |
| **`unit_harmonized`** | `string` | No | Tested conversion | Canonical unit (`mg/dL` for Creatinine, BUN, Glucose; `mEq/L` for $K^+$, $Na^+$, $HCO_3^-$). |
| **`unit_flag`** | `category` | No | Audit pipeline | `STANDARD` (100% in extracted cohort), `CONVERTED`, `UNKNOWN_UNIT`, `MISSING_UOM`. |
| **`flag_abnormal`** | `category` | Yes | `labevents.csv` (`flag`) | Clinical abnormality flag (`abnormal` or null). |
| **`priority`** | `category` | Yes | `labevents.csv` (`priority`) | Clinical test ordering priority: `ROUTINE`, `STAT`, or null. |
| **`is_exact_duplicate`** | `bool` | No | Audit pipeline | True if identical `(subject_id, hadm_id, itemid, charttime, valuenum, valueuom)`. |
| **`is_conflicting_measurement`**| `bool` | No | Audit pipeline | True if same `(subject_id, itemid, charttime)` has disparate values (`nunique > 1`). |
| **`arbitration_policy_id`** | `string` | No | Module 1.3 | Experimental policy applied (`POLICY_0_RAW`, `POLICY_A_STORETIME`, `POLICY_B_STAT`, `POLICY_C_MEAN`, `POLICY_D_RISK`). |
| **`arbitration_role`** | `category` | No | Module 1.3 | `PRIMARY_SELECTED`, `SUPERSEDED_DUPLICATE`, `SUPERSEDED_CONFLICT`, `SYNTHESIZED_AGGREGATE`, `UNARBITRATED_RAW`. |
| **`valuenum_effective`** | `float64` | Yes | Module 1.3 | Effective numeric value passed forward to the temporal feature layer under the active policy. |
| **Patient Demographics** | | | `patients.csv` | `gender`, `anchor_age`, `anchor_year`, `anchor_year_group`, `dod`. |
| **Admission Context** | | | `admissions.csv` | `admittime`, `dischtime`, `admission_type`, `race`, `hospital_expire_flag` (null for outpatient). |

---

## 3. Dual-Time Filtration: Biological vs. Operational Information Sets

Let $\mathcal{D}_p = \{(x_i, c_i, s_i, j_i, u_i)\}_{i=1}^{M_p}$ denote all laboratory events for patient $p$, where $x_i$ is the value, $c_i$ is `charttime`, $s_i$ is `storetime`, $j_i$ is the analyte, and $u_i$ is the unit.

```
Biological Timeline (charttime): Specimen drawn in vivo
--------------------------------+---------------------------------------------------->
                                c_i = 11:51:00
                                [True physiological state frozen]
                                
Operational Timeline (storetime): Assay filed into EHR database
---------------------------------------------------+--------------------------------->
                                                   s_i = 16:40:00 (Delay = 4.82 h)
                                                   [Result becomes visible to clinicians]
```

### 3.1 Formal Mathematical Filtrations
1. **Biological-Time Information Set ($\mathcal{F}_{\text{bio}}(t)$)**:
   $$\mathcal{F}_{\text{bio}}(t) = \{(x_i, c_i, j_i, u_i) \in \mathcal{D}_p \mid c_i \le t\}$$
   - **Clinical & Modeling Semantics**: Represents the retrospective biological trajectory of the patient's renal physiology up to specimen collection timestamp $t$.
   - **Crucial Non-Deployability Caveat**: **$\mathcal{F}_{\text{bio}}(t)$ is NOT deployable as a real-time clinical feature at clock time $t$.** Because laboratory assays require physical transport, centrifugation, analyzer processing, and electronic verification, result $x_i$ is physically absent from the EHR at $t = c_i$. Computing features from $\mathcal{F}_{\text{bio}}(t)$ for real-time alerting at clock time $t$ constitutes **operational lookahead leakage**. Its valid use is strictly retrospective (continuous-time biological trajectory modeling, mechanistic ODE digital twin fitting, or post-discharge phenotyping).

2. **Operational-Time Information Set ($\mathcal{F}_{\text{ops}}(t)$)**:
   $$\mathcal{F}_{\text{ops}}(t) = \{(x_i, c_i, s_i, j_i, u_i) \in \mathcal{D}_p \mid s_i \le t\}$$
   - **Clinical & Modeling Semantics**: Represents the information state accessible to bedside clinicians and real-time clinical decision support algorithms at clock time $t$.
   - An observation drawn at $c_i = 11:51:00$ and filed at $s_i = 16:40:00$ enters $\mathcal{F}_{\text{ops}}(t)$ strictly when $t \ge 16:40:00$. At $t = 12:00:00$, this observation cannot inform any causal operational feature.

3. **Validation of Negative Reporting Delays**:
   - The reporting delay is defined as $\Delta t_{\text{rep}} = s_i - c_i$.
   - The pipeline **does not assume negative delays are impossible**. If $\Delta t_{\text{rep}} < 0$, the record is flagged with `delay_anomaly_flag = 'NEGATIVE_REPORTING_DELAY'`. Such rows are quarantined from operational-timeline features to prevent causality violations arising from clock desynchronization or pre-dated entries.

---

## 4. Simultaneous Observations, Collisions, and Zero-Gap ($\Delta t = 0$) Dynamics

In clinical laboratory data, multiple observations frequently share the exact same `charttime`. We define explicit mathematical handling for all collision types:

### 4.1 Observation Epochs & Multi-Analyte Symmetry
- For patient $p$, all records sharing the exact same `charttime` $c$ form a single **observation epoch** $\mathcal{E}_{p, c}$.
- **Causal Symmetry Principle**: Within $\mathcal{E}_{p, c}$, there is no temporal precedence between different analytes (e.g. Creatinine and BUN drawn in the same tube).
- Event sorting within $\mathcal{E}_{p, c}$ (e.g. by `itemid` or `labevent_id`) must **never create artificial causal sequential dependencies** between co-temporal measurements. Historical summary features for epoch $\mathcal{E}_{p, c}$ depend strictly on prior epochs:
  $$\mathcal{H}_{p, c} = \bigcup_{c' < c} \mathcal{E}_{p, c'}$$
- Co-temporal cross-analyte ratios (such as BUN/Creatinine) are computed directly from the simultaneous measurements within $\mathcal{E}_{p, c}$.

### 4.2 Same-Analyte Collisions & Zero-Gap ($\Delta t = 0$) Singularity Prevention
When two measurements of the *same analyte* $j$ occur at the exact same `charttime` $c$:
- Elapsed time between them is zero: $\Delta t = c - c = 0.0\text{ hours}$.
- Evaluating velocity naively produces a division by zero:
  $$v = \frac{x_2 - x_1}{\Delta t} = \frac{x_2 - x_1}{0} \longrightarrow \pm\infty \text{ or NaN}$$
- **Mathematical Resolution Protocol**:
  1. *At the Unarbitrated Record Level*: Velocity across identical timestamps ($\Delta t = 0$) is formally undefined and mapped to `NaN`, accompanied by `velocity_flag = 'COLLISION_ZERO_DELTA'`.
  2. *At the Arbitrated Feature Level*: The arbitration policy resolves same-analyte collisions at epoch $c$ into a single effective observation value $x^*_{p, c, j}$ *prior* to inter-epoch rate-of-change calculation.
  3. *Inter-Epoch Velocity*: Rate of change is evaluated strictly across strictly distinct epoch timestamps:
     $$v_{k, j} = \frac{x^*_{c_k, j} - x^*_{c_{k-1}, j}}{c_k - c_{k-1}} \quad \text{where } c_k > c_{k-1} > 0$$
  Division by zero is thereby mathematically precluded.

---

## 5. Deterministic Irregular-Time Feature Suite

Synthetic daily resampling, artificial forward-fill grids, and unobserved synthetic days are **strictly prohibited**. The representation preserves the irregular point-process structure of clinical care.

### 5.1 Elapsed-Time Features
All elapsed time features reset strictly at patient boundaries and never cross from one patient to another:

1. **`hours_since_first_observed_event`**:
   $$\tau_{p, i} = \frac{\text{charttime}_i - \min_{k \in \mathcal{D}_p}(\text{charttime}_k)}{3600}$$
   Monotonically non-decreasing continuous clock tracking the patient's observation span in the extracted dataset. Set to $0.0$ at the patient's first recorded event.
2. **`dt_admission_hours`**:
   $$\Delta t_{\text{adm}, i} = \frac{\text{charttime}_i - \text{admittime}_i}{3600}$$
   Elapsed hours within current hospital admission. Set to `NaN` for outpatient / ED labs (`hadm_id IS NULL`), accompanied by indicator `is_inpatient = False`.
3. **`dt_previous_event_hours`**:
   $$\Delta t_{\text{event}, k} = \frac{\text{charttime}_k - \text{charttime}_{k-1}}{3600}$$
   Time elapsed since the patient's immediate preceding laboratory draw of *any* analyte. Reset to `NaN` on the first observation of each patient.
4. **`dt_previous_analyte_hours`** (Analyte-Specific Gap):
   $$\Delta t_{\text{analyte}, k, j} = \frac{\text{charttime}_{k, j} - \text{charttime}_{k-\Delta, j}}{3600}$$
   Elapsed time since *this specific analyte* was last observed for this patient. Reset to `NaN` on the analyte's first observation for that patient.

### 5.2 Irregular-Time Velocity and Acceleration Formulations

#### 1. Discrete Velocity on Irregular Grids
For distinct epochs $c_{k-1} < c_k$:
$$v_{k, j} = \frac{x_{k, j} - x_{k-1, j}}{c_k - c_{k-1}}$$
The velocity estimate $v_{k, j}$ is associated with the time interval $(c_{k-1}, c_k)$ and its natural midpoint coordinate:
$$c_{k - 1/2}^* = \frac{c_{k-1} + c_k}{2}$$

#### 2. Discrete Irregular-Time Acceleration (Non-Uniform Divided Difference)
Consecutive velocity estimates $v_{k-1, j}$ and $v_{k, j}$ span **unequal time intervals**:
- $v_{k-1, j}$ spans $(c_{k-2}, c_{k-1})$ with midpoint $c_{k - 3/2}^* = \frac{c_{k-2} + c_{k-1}}{2}$.
- $v_{k, j}$ spans $(c_{k-1}, c_k)$ with midpoint $c_{k - 1/2}^* = \frac{c_{k-1} + c_k}{2}$.

The time distance between these two velocity estimates is:
$$\Delta c^* = c_{k - 1/2}^* - c_{k - 3/2}^* = \frac{c_{k-1} + c_k}{2} - \frac{c_{k-2} + c_{k-1}}{2} = \frac{c_k - c_{k-2}}{2}$$

The rigorous discrete non-uniform acceleration (second divided difference) is therefore formulated as:
$$a_{k, j} = \frac{v_{k, j} - v_{k-1, j}}{c_{k - 1/2}^* - c_{k - 3/2}^*} = 2 \cdot \frac{\frac{x_{k, j} - x_{k-1, j}}{c_k - c_{k-1}} - \frac{x_{k-1, j} - x_{k-2, j}}{c_{k-1} - c_{k-2}}}{c_k - c_{k-2}}$$
*(Requires $k \ge 3$ distinct epochs; undefined / NaN for $k < 3$).*

---

## 6. Conflict & Duplicate Handling: Experimental Policies and Risk Analysis

The Phase B audit established that the 2,500-patient cohort contains:
- **188 exact duplicate rows** (94 pairs of size 2; 80 outpatient, 14 inpatient).
- **268 conflicting measurement rows** (134 pairs of size 2 across 134 unique timestamp keys).
- **0 overlapping rows** between the two categories.

Arbitration policies are implemented as **configurable experimental pipelines**, not asserted clinical truths.

### 6.1 Policy Specifications and Risk Matrix

| Policy | Deduplication Rule | Conflict Resolution Rule | Analyte Coverage | Underlying Assumption | Documented Clinical & Operational Risks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Policy 0** *(Raw Baseline)* | Retain both rows (`RAW_TAGGED`). | Retain both rows (`RAW_TAGGED`). | All 6 Analytes | Preserves unmanipulated raw data stream for baseline comparison. | Sequence models may encounter zero-gap collisions and ambiguous states. |
| **Policy A** *(Operational Storetime)* | Keep earlier `storetime`; flag second as `SUPERSEDED_DUPLICATE`. | Select record with **latest `storetime`**. | All 6 Analytes | Assumes later filing reflects analyzer re-run, dilution check, or senior pathologist verification. | An add-on order run on an aged/degraded aliquot, or an administrative batch re-file, could supersede a valid initial assay. |
| **Policy B** *(Clinical Priority)* | Keep earlier `storetime`. | Select record with **`priority == 'STAT'`** over `'ROUTINE'`. Fallback to Policy A. | All 6 Analytes | Assumes STAT orders receive immediate clinical oversight and urgent processing. | Point-of-care (POC) blood gas instruments used for STAT tests often exhibit higher analytical variance than automated wet-chemistry central lab analyzers. |
| **Policy C** *(Analytical Mean)* | Keep one row. | Synthesize single observation: $x^* = \frac{x_1 + x_2}{2}$. Flag as `ARBITRATED_MEAN`. | All 6 Analytes | Assumes both aliquots are valid samples of a noisy physiological state (Gaussian noise reduction). | If one tube suffered gross in vitro hemolysis or line contamination, averaging incorporates the contaminated artifact into the patient's record. |
| **Policy D** *(Physiological Extremum)* | Keep one row. | Creatinine: $\max(x_1, x_2)$<br>BUN: $\max(x_1, x_2)$<br>Bicarbonate: $\min(x_1, x_2)$<br>Potassium: Bi-directional $|x - 4.0|$ | **Creatinine, BUN, Bicarbonate, Potassium ONLY**.<br>*(Explicitly **UNSUPPORTED** for Sodium and Glucose).* | Assumes safety-envelope monitoring should prioritize detection of acute pathology (azotemia, acidosis, dyskalemia). | Highly vulnerable to spurious positive assay spikes, bubbles, or laboratory pre-analytical errors. |

### 6.2 Policy D Analyte Scope Invariant
- **Policy D is strictly restricted to Creatinine, BUN, Bicarbonate, and Potassium.**
- **Sodium and Glucose are explicitly UNSUPPORTED under Policy D**. In clinical physiology, sodium and glucose do not possess a single monotonic "worst-case" direction applicable across all renal patients (e.g. hyponatremia vs hypernatremia; hypoglycemia vs severe hyperosmolar hyperglycemia).
- If Policy D is invoked on a dataset containing conflicting observations for Sodium or Glucose, the pipeline **must raise an `UnsupportedPolicyAnalyteError` or leave them explicitly tagged as `UNSUPPORTED_POLICY_D_ANALYTE`**. Arbitrary clinical rules are strictly prohibited.

---

## 7. 3-Way Baseline Comparison Experiment Protocol

To provide an objective benchmark, we establish a **3-way comparative evaluation**:

```
                       [Validated MIMIC-IV Cohort Data]
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
[Arm 1: Fair Conventional]   [Arm 2: Flawed Controls]     [Arm 3: Proposed Renal]
[Preprocessing Baseline]     [Negative Demonstrations]    [Representation]
- 24h Fixed Resampling (LOCF)- 2A: Full-Batch Norm Leakage - Irregular point process
- Causal Train-Fold Scalers  - 2B: Cross-Patient Delta Bleed- Exact charttime & gaps
- Patient Boundary Resets    - 2C: Inner-Join Drop of      - Synchronous mask (delta)
- Outpatient Left-Join Preserved   Outpatient Labs         - Causal running stats
                                                           - Tested arbitration
```

### 7.1 Protocol Arms
1. **Arm 1: Fair Conventional Baseline (Sound LOCF Grid)**:
   - Resampled to a 24-hour fixed regular grid per patient.
   - Missing daily intervals imputed via Last Observation Carried Forward (LOCF).
   - **Leakage-Safe Scaling**: Normalization scalers (StandardScaler / MinMax) are fit **strictly on historical training splits**, never across the entire cohort or future windows.
   - **Boundary-Aware**: LOCF forward-fill and temporal differencing reset strictly at patient boundaries (zero cross-patient contamination).
   - Preserves outpatient null `hadm_id` records via left join.
2. **Arm 2: Deliberately Flawed Controls (Ablation Negative Checks)**:
   - **Control 2A (Full-Dataset Normalization Leakage)**: Fits global MinMax scaling parameters across the entire dataset (past + future), demonstrating leakage when future records are perturbed.
   - **Control 2B (Cross-Patient Contamination)**: Applies naive time-series lag/delta operations across the un-reset DataFrame, demonstrating artificial inter-patient transition spikes.
   - **Control 2C (Admissions Inner-Join Data Loss)**: Performs an `INNER JOIN` against `admissions.csv`, demonstrating the silent loss of 103,354 records (34.06%) and 225 patients.
3. **Arm 3: Proposed Renal-Aware Representation**:
   - Irregular point process preserving true `charttime` and irregular $\Delta t$.
   - Dual-timeline filtration ($\mathcal{F}_{\text{bio}}$ and $\mathcal{F}_{\text{ops}}$).
   - Elapsed time clock (`hours_since_first_observed_event`) and analyte gaps.
   - Synchronous observation indicator mask ($\delta_j$).
   - Strictly causal running statistics (mean, min, max, velocity, acceleration).
   - Documented duplicate and conflict arbitration policies (A, B, C, D).

### 7.2 Structural Data-Integrity Evaluation Metrics (No Predictive Claims)
Direct comparisons evaluate the exact same eligible source observations. Observation loss is reported separately:

| Metric Category | Specific Data-Integrity Metric | Mathematical Definition | Target / Benchmark Expectation |
| :--- | :--- | :--- | :--- |
| **Observation Retention** | Retained Observation Rate | $\frac{N_{\text{retained}}}{N_{\text{raw}}} \times 100\%$ | Arm 1: 100%, Control 2C: **65.94%** (34.06% loss), Arm 3: 100%. |
| **Future Perturbation Invariance** | Historical Shift under Future Perturbation | $\max_{t \le T} \|F_t^{(\text{base})} - F_t^{(\text{perturbed})}\|$ | Arm 1: 0.0000, Control 2A: **$> 0$ (leaks)**, Arm 3: 0.0000. |
| **Boundary Isolation** | Inter-Patient Transition Violations | Count of features at $t_{p, 0}$ depending on $t_{p-1, \text{end}}$ | Arm 1: 0, Control 2B: **$> 0$ (bleeds)**, Arm 3: 0. |
| **Operational Leakage** | EHR Lookahead Violation Rate | Fraction of features at $t$ using data where $s_i > t$ | Biological features: **Documented non-deployable**, Operational: 0.00%. |
| **Singularity Avoidance** | Division-by-Zero Count ($\Delta t = 0$) | Count of infinite / NaN velocities at colliding timestamps | Arm 3: **0** (handled by collision logic). |
| **Conflict Auditability** | Audit Provenance Traceability | Fraction of collisions with explicit arbitration role flag | Arm 3: **100%** traceable. |

---

## 8. Implementation Steps & Scope

Module 1.3 is executed in focused, reviewable steps:
- **Step 1 (Current)**: Implement `src/temporal/canonical_schema.py` and its focused test suite. Stop for review.
- **Step 2**: Implement conflict arbitration policies (`src/temporal/conflict_arbitration.py`).
- **Step 3**: Implement irregular temporal feature suite and dual causal timeline engine (`src/temporal/temporal_features.py`).
- **Step 4**: Implement 3-way baseline comparison pipeline and generate the quality report (`src/temporal/representation_pipeline.py`).
