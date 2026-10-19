# WEKA Explorer Manual Reproduction Guide: Phantom Emergency Admission Imputation

**Target Audience:** Project Supervisor, Evaluators, and Viva Examiners  
**Estimated Time to Reproduce:** ~60 seconds (3 GUI clicks in WEKA Explorer)  
**Demonstrated Issue:** Inappropriate Missing-Value Mode Imputation on Integrated Relational Clinical Data (`ReplaceMissingValues`)  
**Artifact Location:** `reports/generated/weka_dataset_integration_audit/integrated_renal_sample.arff`  

---

## 1. Objective of Manual Verification

To demonstrate visually in the official **WEKA Explorer GUI** that applying standard tabular missing-value preprocessing (`weka.filters.unsupervised.attribute.ReplaceMissingValues`) to an integrated clinical dataset silently creates **747 phantom Emergency Department inpatient admissions** for genuine ambulatory/outpatient blood draws, while completely eliminating outpatient representation.

---

## 2. Prerequisites

1. WEKA 3.8.x installed (e.g. `D:\Software\weka\Weka-3-8-7` or standard installer).
2. The pre-generated, audited test file:  
   `D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_dataset_integration_audit\integrated_renal_sample.arff`

---

## 3. Step-by-Step Manual Reproduction Protocol

### Step 1: Launch WEKA and Open the Preprocess Tab
1. Start WEKA. In the main WEKA GUI Chooser window, click the **Explorer** button.
2. Ensure you are on the default **Preprocess** tab (top-left).

### Step 2: Open the Integrated Clinical Sample File
1. Click the **Open file...** button at the top-left of the Preprocess panel.
2. In the file chooser, navigate to:
   ```
   D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_dataset_integration_audit
   ```
3. Select `integrated_renal_sample.arff` and click **Open**.
4. Observe the **Current relation** panel:
   - **Relation:** `integrated_renal_sample`
   - **Instances:** `1600`
   - **Attributes:** `9`

---

### Step 3: Inspect the Baseline Outpatient Missingness
1. In the **Attributes** box on the left, scroll down and click on attribute **#8: `admission_type`**.
2. Look at the **Selected attribute** panel on the right:
   - **Name:** `admission_type`
   - **Type:** `Nominal`
   - **Missing:** **`747 (47%)`**  $\leftarrow$ *These are genuine outpatient and emergency draws prior to inpatient admission!*
   - **Distinct:** `7`
3. Look at the lower-right histogram panel:
   - Notice the frequency bars for existing inpatient admissions:
     - `EW EMER.`: `555`
     - `DIRECT EMER.`: `143`
     - `OBSERVATION ADMIT`: `70`
     - `DIRECT OBSERVATION`: `44`
     - `URGENT`: `22`
     - `EU OBSERVATION`: `12`
     - `SURGICAL SAME DAY ADMISSION`: `7`

---

### Step 4: Apply the Default WEKA Filter `ReplaceMissingValues`
1. At the top under the **Filter** section, click the **Choose** button.
2. Expand the folder tree:  
   `filters` $\rightarrow$ `unsupervised` $\rightarrow$ `attribute` $\rightarrow$ **`ReplaceMissingValues`**.
3. Click the **Apply** button on the right of the Choose box.

---

### Step 5: Verify the Silent Creation of Phantom Emergency Admissions
1. In the **Attributes** box, ensure attribute **#8: `admission_type`** is selected.
2. Look at the **Selected attribute** statistics panel on the right:
   - **Missing:** **`0 (0%)`**  $\leftarrow$ *All missingness has been eliminated.*
3. Look at the lower-right frequency table and histogram:
   - Look at the count for label `EW EMER.` (Emergency Ward Emergency Admission):
     - **Before Filter Count:** `555`
     - **After Filter Count:** **`1,302`**  ($555 + 747$)
   - **Result:** WEKA took all 747 outpatient and clinic laboratory observations and converted them into Emergency Inpatient Admissions!

---

### Step 6 (Optional): Inspect Patient 10000898 (Pure Ambulatory Patient)
1. In the Preprocess panel, click the **Edit...** button (top-right).
2. The WEKA Data Viewer table opens showing all 1,600 rows.
3. Scroll down or search for `subject_id = 10000898` (rows 694 to 717).
4. Notice that in the raw MIMIC-IV database, Patient 10000898 has **0 hospital admissions** in `admissions.csv`.
5. In WEKA's transformed table, all 24 routine blood draws for Patient 10000898 now state:  
   `admission_type = "EW EMER."`.

---

## 4. Verification Checklist

| Verification Check | Expected Outcome in WEKA Explorer | Confirmed? |
| :--- | :--- | :---: |
| File loads without error | 1,600 instances, 9 attributes | [ ] |
| Pre-filter `admission_type` missingness | Exactly **747 missing (46.69% / 47%)** | [ ] |
| Pre-filter `EW EMER.` count | Exactly **555** | [ ] |
| Post-filter `admission_type` missingness | Exactly **0 missing (0%)** | [ ] |
| Post-filter `EW EMER.` count | Exactly **1,302** (+747 jump) | [ ] |
| Outpatient representation preserved? | **NO (0% remaining; completely eliminated)** | [ ] |

---

## 5. Corresponding Python Reference & Causal Correction

In our Python pipeline (`src/temporal/canonical_schema.py` and `src/temporal/representation_pipeline.py`), relational clinical data is protected using explicit domain logic:

```python
# 1. Relational Preservation: Retain outpatient status as a first-class clinical state
df["encounter_type"] = np.where(df["hadm_id"].isna(), "AMBULATORY", "INPATIENT")

# 2. Never mode-impute inpatient admission metadata onto ambulatory draws
# Admission attributes (admittime, dischtime, admission_type) remain strictly null for outpatient labs.

# 3. Longitudinal missingness is handled via causal within-patient LOCF with staleness horizons:
# If creatinine is missing at time t, carry forward previous observation ONLY within the same patient,
# ONLY if elapsed time <= 24 hours, and tag with val_is_imputed = True.
```

To run the automated verification script:
```bash
.venv\Scripts\python.exe reports\generated\weka_dataset_integration_audit\run_integration_audit.py
```
