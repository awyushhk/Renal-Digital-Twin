# Final WEKA Capability Audit: Empirical Evaluation of Patient-Boundary Preprocessing for Longitudinal Renal Trajectories
## Nephron-Aware Renal Digital Twin — MIMIC-IV 3.1

**Date:** 2026-10-19  
**Execution Environment:** Java 21.0.8 LTS | Python 3.12 (venv) | WEKA 3.8.7 (`D:\Software\weka\Weka-3-8-7\weka.jar`, 14,653,641 bytes)  
**Parent Cohort:** `data/processed/cohort_renal_bounded.csv` (SHA-256 verified)  
**Authoritative Artifact:** `reports/generated/weka_final_capability_audit.md`  

---

## 1. Executive Verdict: PARTIALLY SUPPORTED (Narrowed and Methodologically Framed)

| Proposed Claim Level | Verdict | Rationale |
| :--- | :---: | :--- |
| **Broad Claim:** *"WEKA is fundamentally broken / incapable of time-series processing"* | **REJECTED** | WEKA's Java filter architecture is extensible; a custom Java filter (`GroupedTimeSeriesDelta`) compiles cleanly and executes group-aware differences natively within WEKA. External per-patient partitioning also works. |
| **Technical Claim:** *"WEKA cannot perform date arithmetic"* | **REJECTED** | Standard built-in WEKA 3.8.7 filters (`DateToNumeric` + `TimeSeriesDelta` + `MathExpression`) CAN compute elapsed hours from ISO-8601 date attributes with **0.000000 hours error** on within-patient records. |
| **Narrowed Methodological Claim:** *"Standard out-of-the-box WEKA 3.8.7 built-in filters lack patient-group awareness, causing 100% cross-patient boundary contamination in longitudinal EHR batches"* | **FULLY SUPPORTED** | Across all tested standard WEKA filter combinations, **233 out of 233 (100.0%)** inter-patient transitions suffer cross-patient subtraction, producing phantom intervals of up to -453,046 hours (-51.7 years). |

**Executive Summary:**  
The candidate research contribution is **PARTIALLY SUPPORTED**. It must be strictly framed not as a theoretical discovery of software failure, but as a **methodological and architectural justification**: standard legacy data-mining suites (exemplified by WEKA 3.8.7) assume independent and identically distributed (I.I.D.) instances or continuous single-stream time series, lacking built-in panel/group-partitioning operators. When applied to multi-patient longitudinal EHR batches without heavy external scaffolding, standard WEKA silently corrupts all inter-patient boundary records. This empirical reality directly justifies the design and necessity of our specialized Python/Pandas irregular temporal preprocessing pipeline.

---

## 2. Environment and Exact Reproduction Instructions

### 2.1 Hardware and Runtime Environment
- **Operating System:** Windows 11 Enterprise (x86_64)
- **Java Runtime:** OpenJDK 64-Bit Server VM (build 21.0.8+9-LTS)
- **WEKA Installation:** WEKA 3.8.7 official release (`D:\Software\weka\Weka-3-8-7\weka.jar`)
- **Python Environment:** Python 3.12.x in `.venv` with `pandas 2.2.3`, `numpy 1.26.4`, `scipy 1.13.1`
- **JVM Execution Flags:** `java --add-opens java.base/java.lang=ALL-UNNAMED -cp "weka.jar;custom_classes"`

### 2.2 Step-by-Step Reproduction Commands

```bash
# 1. Compile the custom group-aware WEKA filter
javac -cp "D:\Software\weka\Weka-3-8-7\weka.jar" "scratch/custom/weka/GroupedTimeSeriesDelta.java"

# 2. Execute the comprehensive evaluation suite (synthetic edge cases + 5,000 real cohort records)
.venv\Scripts\python.exe "scratch/run_final_weka_audit.py"

# 3. Parse ARFF outputs, evaluate forward/backward references, and extract metrics
.venv\Scripts\python.exe "scratch/analyze_final_audit.py"
```

All source fixtures, input ARFF files, and generated output ARFF files reside in `reports/generated/weka_audit_fixtures/` and are tracked deterministically.

---

## 3. Corrected Experiment Design and Reference Definitions

### 3.1 Flaw in Previous Audit Logic
The preliminary audit report compared `TimeSeriesDelta -I 1` against backward previous-observation expectations. This was a directional mismatch:
- In WEKA documentation: `-I <num>` specifies instance offset. A negative offset (`-I -1`) translates backward ($X[i] - X[i-1]$), whereas a positive offset (`-I 1`) translates forward ($X[i+1] - X[i]$).
- Evaluating a forward difference against a backward reference conflates directional inversion with boundary contamination.
- Dataset boundary rows (row 0 for backward, row $N-1$ for forward) receive `?` (missing) simply because the filter hits the physical edge of the file, not because WEKA recognizes patient boundaries.

### 3.2 Formal Mathematical Reference Definitions

Let a dataset of $N$ rows be sorted deterministically by $(subject\_id, charttime, labevent\_id)$ ascending.

#### A. Backward Elapsed Time Reference ($\Delta t_{\text{bwd}}$)
For row $i \in \{0, \dots, N-1\}$:
$$\Delta t_{\text{bwd}}[i] = \begin{cases} 
\text{NaN}, & \text{if } i = 0 \quad (\text{first row of dataset}) \\
\text{NaN}, & \text{if } i > 0 \text{ and } subject\_id[i] \neq subject\_id[i-1] \quad (\text{inter-patient transition}) \\
\frac{charttime[i] - charttime[i-1]}{3600.0}, & \text{if } i > 0 \text{ and } subject\_id[i] = subject\_id[i-1] \quad (\text{within patient})
\end{cases}$$

#### B. Forward Elapsed Time Reference ($\Delta t_{\text{fwd}}$)
For row $i \in \{0, \dots, N-1\}$:
$$\Delta t_{\text{fwd}}[i] = \begin{cases} 
\text{NaN}, & \text{if } i = N - 1 \quad (\text{last row of dataset}) \\
\text{NaN}, & \text{if } i < N - 1 \text{ and } subject\_id[i] \neq subject\_id[i+1] \quad (\text{inter-patient transition}) \\
\frac{charttime[i+1] - charttime[i]}{3600.0}, & \text{if } i < N - 1 \text{ and } subject\_id[i] = subject\_id[i+1] \quad (\text{within patient})
\end{cases}$$

#### C. Strict Denominator Taxonomy
In a cohort of $N$ rows containing $P$ distinct patients:
1. **Total Rows:** $N$
2. **Dataset Physical Edge:** Exactly 1 row ($i=0$ for backward; $i=N-1$ for forward).
3. **Actual Inter-Patient Transitions:** Exactly $P - 1$ rows.
4. **Valid Within-Patient Trajectory Rows:** Exactly $N - P$ rows.

---

## 4. Results Across All Tested WEKA Approaches

Seven distinct preprocessing workflows were tested against the identical input data:

| Workflow Identifier | Description & Filter Pipeline | Date Arithmetic | Patient Boundary Isolation | Valid Row Accuracy | Feasibility / Overhead |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Workflow A (W1)** | `TimeSeriesDelta -R 5 -I -1` on creatinine | N/A (Computes $\Delta\text{Cr}$, not $\Delta t$) | **FAILED (100% contaminated)** | $\Delta\text{Cr}$ matches | High (Single command) |
| **Workflow B (W2)** | `TimeSeriesDelta -R 5 -I 1` on creatinine (Forward) | N/A (Computes forward $\Delta\text{Cr}$) | **FAILED (100% contaminated)** | $\Delta\text{Cr}$ matches | High (Single command) |
| **Workflow C (W3)** | `TimeSeriesTranslate -R 5 -I -1` on creatinine | N/A (Shifts values, no diff) | **FAILED (100% contaminated)** | Shifted values match | High (Single command) |
| **Workflow D (W4)** | **Native MultiFilter Backward:**<br>`DateToNumeric -R 4`<br>+ `TimeSeriesDelta -R 4 -I -1`<br>+ `MathExpression -E (A/3600000) -R 4 -V` | **EXACT (0.000000h error)** | **FAILED (100% contaminated)** | **100% exact match** on within-patient rows | High (Single standard WEKA CLI command) |
| **Workflow E (W5)** | **Native MultiFilter Forward:**<br>`DateToNumeric -R 4`<br>+ `TimeSeriesDelta -R 4 -I 1`<br>+ `MathExpression -E (A/3600000) -R 4 -V` | **EXACT (0.000000h error)** | **FAILED (100% contaminated)** | **100% exact match** on within-patient rows | High (Single standard WEKA CLI command) |
| **Workflow F (Per-Patient)** | External Python/Bash script splits into per-patient ARFFs, runs WEKA MultiFilter per patient, recombines | **EXACT (0.000000h error)** | **PERFECT (0 contaminations)** | **100% exact match** across all rows | **IMPRACTICAL:** 0.658s/patient = 2.5 min for 234 pts, **9.1 hours for 50k patients** |
| **Workflow G (Custom Java)** | Custom WEKA filter `GroupedTimeSeriesDelta -G 2 -R 4 -I -1` running inside WEKA `MultiFilter` | **EXACT (0.000000h error)** | **PERFECT (0 contaminations)** | **100% exact match** across all rows | High (Single command, <1s runtime, but requires custom Java code) |

---

## 5. Correct Boundary Counts with Transparent Denominators

### 5.1 Real 5,000-Record Serum Creatinine Breakdown

```
Total Cohort Input Rows (N):                    5,000
Distinct Subject IDs (P):                         234
Dataset Start Boundary (Row 0):                     1  (WEKA produces NaN by physical file edge)
Dataset End Boundary (Row 4999):                    1  (WEKA produces NaN by physical file edge)
Actual Inter-Patient Transitions (P - 1):         233
Valid Within-Patient Trajectory Rows (N - P):   4,766
```

### 5.2 Empirical Boundary Contamination Rate
- **Standard WEKA MultiFilter Backward (W4):**
  - Contaminated Inter-Patient Transitions: **233 / 233 (100.0%)**
  - Correctly Reset Inter-Patient Transitions: **0 / 233 (0.0%)**
  - Dataset Start Row: 1 / 1 (100.0% NaN, purely due to file start)
- **Standard WEKA MultiFilter Forward (W5):**
  - Contaminated Inter-Patient Forward Transitions: **233 / 233 (100.0%)**
  - Correctly Reset Inter-Patient Transitions: **0 / 233 (0.0%)**
  - Dataset End Row: 1 / 1 (100.0% NaN, purely due to file end)
- **Custom Grouped Java Filter (W6):**
  - Contaminated Inter-Patient Transitions: **0 / 233 (0.0%)**
  - Correctly Reset Inter-Patient Transitions: **233 / 233 (100.0%)**

---

## 6. Small Reproducible Examples: Edge Cases and Row-by-Row Proof

The synthetic test suite (`final_audit_synthetic_base.arff`) systematically evaluates six distinct clinical trajectory edge cases across 18 deterministic rows:

### 6.1 Row-by-Row Comparison Table

| Row | Subject | Timestamp | Cr | Clinical Scenario Description | Python Ref ($\Delta t_{\text{bwd}}$) | WEKA W4 (Standard) | WEKA W6 (Custom Grouped) | Evaluation & Verdict |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | 101 | 2180-01-01 08:00 | 1.0 | Standard multi-obs: First draw | `NaN` | `NaN` | `NaN` | Pass (File start) |
| **1** | 101 | 2180-01-01 20:00 | 1.2 | Standard multi-obs: +12h draw | **12.0h** | **12.0h** | **12.0h** | Pass (Exact match) |
| **2** | 101 | 2180-01-03 08:00 | 1.5 | Standard multi-obs: +36h draw | **36.0h** | **36.0h** | **36.0h** | Pass (Exact match) |
| **3** | **102** | 2180-02-01 10:00 | 0.9 | **Single-observation patient** | `NaN` | **698.0h** | `NaN` | **CONTAMINATED:** WEKA subtracted P101 row 2! |
| **4** | **103** | 2180-03-01 12:00 | 2.0 | Co-temporal duplicate: First draw | `NaN` | **698.0h** | `NaN` | **CONTAMINATED:** WEKA subtracted P102 row 3! |
| **5** | 103 | 2180-03-01 12:00 | 2.1 | Co-temporal duplicate: Second draw | **0.0h** | **0.0h** | **0.0h** | Pass (Zero delta detected) |
| **6** | 103 | 2180-03-02 12:00 | 1.8 | Co-temporal duplicate: Next day | **24.0h** | **24.0h** | **24.0h** | Pass (Exact match) |
| **7** | **104** | 2180-04-01 09:00 | 0.8 | Missing measurement: First draw | `NaN` | **717.0h** | `NaN` | **CONTAMINATED:** WEKA subtracted P103 row 6! |
| **8** | 104 | 2180-04-01 15:00 | `?` | Missing measurement: Null analyte | **6.0h** | **6.0h** | **6.0h** | Pass (Time delta independent of analyte) |
| **9** | 104 | 2180-04-02 09:00 | 1.1 | Missing measurement: Recovered | **18.0h** | **18.0h** | **18.0h** | Pass (Exact match) |
| **10** | **201** | 2185-06-01 10:00 | 1.0 | Calendar inversion: Patient in 2185 | `NaN` | **45,265.0h** | `NaN` | **CONTAMINATED:** +5.16 year phantom gap! |
| **11** | 201 | 2185-06-02 10:00 | 1.1 | Calendar inversion: Next day | **24.0h** | **24.0h** | **24.0h** | Pass (Exact match) |
| **12** | **202** | 2181-01-01 10:00 | 0.7 | Calendar inversion: Patient in 2181 | `NaN` | **-38,712.0h** | `NaN` | **CONTAMINATED:** **-4.42 year negative gap!** |
| **13** | 202 | 2181-01-02 10:00 | 0.8 | Calendar inversion: Next day | **24.0h** | **24.0h** | **24.0h** | Pass (Exact match) |
| **14** | **301** | 2180-01-01 10:00 | 1.0 | Perturbation pair: Patient 301 t0 | `NaN` | **-8,808.0h** | `NaN` | **CONTAMINATED:** Subtracted P202 row 13! |
| **15** | 301 | 2180-01-02 10:00 | 1.2 | Perturbation pair: Patient 301 t1 | **24.0h** | **24.0h** | **24.0h** | Pass (Exact match) |
| **16** | **302** | 2180-01-03 10:00 | 1.5 | Perturbation pair: Patient 302 t0 | `NaN` | **24.0h** | `NaN` | **CONTAMINATED:** Subtracted P301 row 15! |
| **17** | 302 | 2180-01-04 10:00 | 1.6 | Perturbation pair: Patient 302 t1 | **24.0h** | **24.0h** | **24.0h** | Pass (Exact match) |

### 6.2 The Perturbation Test: Proof of Cross-Patient Feature Leakage

To establish causal feature leakage:
1. In `final_audit_synthetic_base.arff`, Patient 301's second draw occurs on `2180-01-02 10:00:00`.
2. In `final_audit_synthetic_pert.arff`, Patient 301's second draw is perturbed by +8 days to `2180-01-10 10:00:00`. Patient 302 is **completely untouched**.
3. **Outcome in Independent Python Pipeline:**
   - Patient 302's initial $\Delta t_{\text{bwd}}$ remains `NaN` in both base and perturbed runs ($\Delta = 0.00\text{h}$). Patient 302 is strictly isolated.
4. **Outcome in Standard WEKA Pipeline:**
   - In Base run: Patient 302 Row 16 receives **+24.0 hours**.
   - In Perturbed run: Patient 302 Row 16 receives **-168.0 hours**.
   - **Cross-Patient Leakage:** Altering Patient 301's discharge timing shifted Patient 302's feature value by **-192.0 hours (-8.0 days)**.

---

## 7. Real-Cohort Validation Metrics (5,000 Real MIMIC-IV Records)

When evaluated on 5,000 real serum-creatinine lab draws (`itemid=50912`) across 234 patients from `cohort_renal_bounded.csv`:

### 7.1 Quantitative Findings
1. **Within-Patient Accuracy:**
   - For all 4,766 non-boundary observations, WEKA's native `MultiFilter` achieves **100.00% exact numerical agreement** with the Python reference.
   - **Mean Absolute Error:** `0.000000 hours`
   - **Maximum Absolute Error:** `0.000000 hours`
   - This disproves any claim that WEKA cannot accurately represent floating-point time intervals.
2. **Boundary Contamination Rate:**
   - Total inter-patient transitions: 233
   - Number of contaminated boundary rows: **233 (100.0%)**
   - Number of correctly reset boundary rows: **0 (0.0%)**
3. **Distribution of Phantom Inter-Patient Intervals:**
   - **Minimum Phantom Interval:** `-453,046.58 hours` (-51.7 years)
   - **Maximum Phantom Interval:** `+226,336.85 hours` (+25.8 years)
   - **Median Absolute Phantom Interval:** `24,192.50 hours` (~2.76 years)

```
Top 5 Contaminated Inter-Patient Transitions in Real Cohort:
  Row   18 | Patient 10000084 | WEKA Phantom Interval:  -172,861.50 hours  (-19.7 years)
  Row   24 | Patient 10000117 | WEKA Phantom Interval:  +117,733.67 hours  (+13.4 years)
  Row   40 | Patient 10000161 | WEKA Phantom Interval:  -187,026.12 hours  (-21.3 years)
  Row   43 | Patient 10000560 | WEKA Phantom Interval:  +226,336.85 hours  (+25.8 years)
  Row   47 | Patient 10000635 | WEKA Phantom Interval:  -453,046.58 hours  (-51.7 years)
```

### 7.2 Physiological Impact on Renal Feature Pipeline
In `src/temporal/temporal_features.py`, discrete creatinine velocity is computed as:
$$v_{\text{bio}}[i] = \frac{\text{Cr}[i] - \text{Cr}[i-1]}{\Delta t_{\text{hours}}[i]}$$
If $\Delta t_{\text{hours}}$ at an inter-patient transition is $-172,861.5\text{ hours}$:
- The first observation of Patient 10000084 is assigned a phantom velocity of $\frac{0.8 - 0.8}{-172861.5} \approx 0.0\text{ mg/dL/h}$, falsely signaling perfect renal stability.
- Where creatinine levels differ across patients (e.g. Row 40: 1.1 mg/dL vs preceding patient's 0.9 mg/dL), a non-zero velocity is calculated between two unrelated human beings.
- When second-order non-uniform acceleration ($a_{\text{bio}}$) is computed, the corrupted velocity propagates into the acceleration divided-difference formula, corrupting the subsequent within-patient observation as well.

---

## 8. Resolution via Per-Patient Partitioning vs. Custom WEKA Processing

Can WEKA be made to work correctly for this task? We investigated both potential remedies:

### 8.1 Approach 1: External Per-Patient Partitioning (Scaffolding)
- **Concept:** An external driver partitions the cohort by `subject_id`, writes $P$ individual ARFF files, executes the standard WEKA MultiFilter on each file independently, and concatenates the results.
- **Boundary Result:** 0 contaminated rows (100% boundary isolation achieved).
- **Performance & Scalability Benchmark:**
  - Evaluated on a 25-patient real cohort slice: **16.44 seconds** (average **0.658 seconds per patient**).
  - Projected for 234 patients: **153.9 seconds (2.56 minutes)**.
  - Projected for 50,000 patients (full MIMIC-IV): **32,875 seconds (9.13 hours)** with 50,000 separate disk I/O operations and JVM invocations.
  - **Comparison with Python:** In-memory vectorized `pandas.groupby('subject_id')` processes 303,460 rows in **0.15 seconds** (over **200,000x faster** than per-patient WEKA subprocesses).
- **Verdict:** Operationally impractical for clinical digital twin scaling.

### 8.2 Approach 2: Custom Group-Aware WEKA Java Filter
- **Concept:** Developed `GroupedTimeSeriesDelta` extending `SimpleBatchFilter` and implementing `UnsupervisedFilter` and `OptionHandler` with `-G <group_attr>` parameter.
- **Boundary Result:** **0 contaminated rows (100% boundary isolation achieved)** in a single pass.
- **Runtime:** **0.78 seconds** for the entire 5,000-row dataset in a single JVM instance.
- **Verdict:** Proves that WEKA's architecture is capable, but confirms that standard, out-of-the-box WEKA lacks this feature.

---

## 9. Limitations and Threats to Validity

1. **Sort Order Sensitivity:** WEKA instance filters assume the input physical order is meaningful. If the ARFF file is sorted globally by `charttime` rather than `(subject_id, charttime)`, inter-patient interleaving creates thousands of spurious cross-patient transitions.
2. **Missing Value Semantics:** Standard WEKA `TimeSeriesDelta` propagates missing values (`?`) correctly when `-M` is omitted, but does not distinguish between a missing observation within a patient and a reset at a patient boundary.
3. **Filter Chaining Complexity:** Achieving exact date arithmetic in native WEKA requires chaining three distinct filters (`DateToNumeric` $\rightarrow$ `TimeSeriesDelta` $\rightarrow$ `MathExpression`) inside a `MultiFilter`. While functional, this workflow is fragile and easily misconfigured (e.g. omitting `-V` in `MathExpression` corrupts all other attributes).

---

## 10. The Narrowest Defensible Research Claim

The only scientifically honest, peer-review-resilient research claim supported by this audit is:

> *"Standard, built-in preprocessing filters in legacy general-purpose data-mining suites (exemplified by WEKA 3.8.7) operate strictly on sequential instance offsets without support for panel-data group keys. When applied to multi-patient longitudinal EHR datasets in standard batch mode, these filters suffer 100% cross-patient boundary contamination, generating phantom inter-patient intervals (up to $\pm 10^5$ hours) that corrupt downstream physiological rate-of-change features. While this limitation can be resolved through external per-patient file partitioning or custom Java extensions, external partitioning incurs prohibitive computational overhead ($O(P)$ process invocations, projecting to >9 hours for 50,000 patients vs. <0.2 seconds in memory). This justifies the architecture of our specialized, patient-aware Python temporal representation engine as a necessary prerequisite for clinical digital twin deployment."*

---

## 11. Claims That the Evidence Does NOT Justify

Peer reviewers would rightfully reject the following unsupported claims:

1. ❌ **"WEKA is fundamentally flawed as a machine learning system."**  
   *Refutation:* WEKA was designed for cross-sectional tabular learning. Panel-data group-by operations were never part of its core architectural scope.
2. ❌ **"WEKA cannot perform date or timestamp arithmetic."**  
   *Refutation:* We demonstrated that WEKA's built-in `DateToNumeric` and `MathExpression` filters compute elapsed hours with 0.000000h error.
3. ❌ **"It is impossible to process panel EHR data in WEKA."**  
   *Refutation:* It is possible via external per-patient partitioning or by authoring a custom Java plugin (`GroupedTimeSeriesDelta`).
4. ❌ **"Our digital twin production pipeline suffers from this bug."**  
   *Refutation:* Our production code (`src/temporal/temporal_features.py`) already implements strict patient grouping via `groupby('subject_id')` and achieves 100% boundary isolation.
5. ❌ **"Discovering that WEKA lacks group-by is a novel computer science breakthrough."**  
   *Refutation:* It is a well-known characteristic of classic flat-file data-mining tools. The contribution is engineering and methodological justification, not theoretical novelty.

---

## 12. Final Recommendation: RETAIN (With Narrowed Framing)

### Recommendation: RETAIN as a Methodological / Engineering Justification Contribution
- **Action:** Retain the WEKA comparison in the thesis/paper, but strictly adopt the narrowed framing defined in Section 10.
- **Role in Research Narrative:** Use this audit in Section 3 ("System Architecture & Preprocessing Methodology") to explain why standard clinical data-mining workflows cannot simply rely on off-the-shelf tabular ML tools (WEKA/Orange/AutoML) for longitudinal EHR feature engineering without producing silent clinical data corruption.
- **Evidence Base:** Cite the 5,000-record empirical audit, the exact 233/233 boundary error rate, the perturbation leakage experiment, and the 200,000x efficiency advantage of vectorized Python panel processing.
