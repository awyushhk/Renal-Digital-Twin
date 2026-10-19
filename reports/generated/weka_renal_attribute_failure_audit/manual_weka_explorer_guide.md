# Manual WEKA Explorer Verification Guide
## Nephron-Aware Renal Digital Twin — WEKA Attribute Failure Audit

**Purpose:** Allow the project supervisor, examiners, or any third party to manually verify the two strongest audit findings using WEKA Explorer's GUI in under 5 minutes.

**WEKA Version:** 3.8.7  
**WEKA Installation:** `D:\Software\weka\Weka-3-8-7\`  
**Input Files:** `reports/generated/weka_renal_attribute_failure_audit/`

---

## Finding 1: STRING Attribute Silent Data Destruction (2 minutes)

This demonstrates that WEKA's batch filters silently drop all 1,578 data rows when the ARFF contains `STRING`-typed attributes, while returning exit code 0 (success) — no error or warning is produced.

### Steps

**Step 1 — Open the STRING ARFF (original, dangerous format):**
1. Launch WEKA Explorer (`D:\Software\weka\Weka-3-8-7\WekaGUI.jar` or via Start Menu)
2. Click **`Preprocess`** tab
3. Click **`Open file...`**
4. Navigate to:  
   `D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_renal_attribute_failure_audit\renal_audit_sample.arff`
5. Observe in the **Current relation** panel:  
   - **Instances: 1578**  
   - Attributes: 14  
   - Note attributes `hadm_id_str` (STRING) and `charttime` (STRING)

**Step 2 — Apply ReplaceMissingValues:**
1. Click the **`Filter`** panel → **`Choose`** button
2. Navigate: `weka.filters.unsupervised.attribute.ReplaceMissingValues`
3. Click **`Apply`**

**Step 3 — Observe the result:**
> **Expected:** Instances panel now shows **0 instances** (or an error is thrown)  
> **The original 1,578 rows have been completely destroyed with no warning.**

The output file (if saved) will be 641 bytes — containing only the ARFF header with zero data rows.

---

## Finding 2: Creatinine Feature Space Compression (3 minutes)

This demonstrates that WEKA's `Normalize` filter, when applied to the shared multi-analyte `valuenum` column, compresses the entire clinical range of serum creatinine to just **0.18% of the [0, 1] feature space**, making gradient-based models effectively blind to creatinine variation.

### Steps

**Step 1 — Open the corrected ARFF (no STRING attributes):**
1. Click **`Preprocess`** tab → **`Open file...`**
2. Navigate to:  
   `D:\CKD mini project\nephron-aware-renal-digital-twin\reports\generated\weka_renal_attribute_failure_audit\renal_audit_sample_v2.arff`
3. Observe:
   - **Instances: 1578**
   - Click attribute **`valuenum`** (attribute 6)
   - Note: **Min = 0.50**, **Max = 330.00**, **Mean = 50.37**
   - Histogram shows a wide range dominated by glucose (up to 330 mg/dL) and sodium (118–150 mEq/L)

**Step 2 — Note creatinine values before normalization:**
- Serum creatinine values in this sample range from **0.50 to 5.80 mg/dL**
- The clinical normal range is **0.6–1.2 mg/dL** (a range of 0.6 mg/dL)
- The entire creatinine dataset spans **5.30 mg/dL**

**Step 3 — Apply Normalize filter:**
1. Click **`Filter`** → **`Choose`** → `weka.filters.unsupervised.attribute.Normalize`
2. Ensure parameters: `-S 1.0 -T 0.0` (min-max normalization to [0, 1])
3. Click **`Apply`**

**Step 4 — Observe attribute `valuenum` after normalization:**
1. Click attribute **`valuenum`**
2. Observe:
   - **Min = 0.000000**, **Max = 1.000000**
   - The histogram is now dominated entirely by glucose/sodium values
   - Creatinine observations are clustered at values ≈ **0.001–0.003**

**Step 5 — Compute the creatinine compression ratio:**
```
Before normalization: Creatinine clinical range = 1.2 - 0.6 = 0.60 mg/dL
After normalization:  Creatinine clinical range = 0.60 / (330 - 0.50) = 0.00182

Creatinine occupies only 0.18% of the [0, 1] feature space.
```

> **Clinical consequence:** A gradient-based model (neural network, logistic regression, gradient boosting) trained on this normalized data will receive gradients from creatinine that are **555× smaller** than from glucose, effectively treating creatinine as a near-constant — eliminating the primary GFR surrogate from model learning.

---

## Finding 3: Global Mean Imputation — Cross-Analyte Incoherence (demonstration only)

This can be demonstrated with the synthetic censored ARFF.

**Step 1 — Open the censored ARFF:**
1. **`Open file...`** → `censored_creatinine_in_v2.arff` (7 rows with 4 NaN creatinine values)

**Step 2 — Apply ReplaceMissingValues:**
1. Filter → `ReplaceMissingValues` → Apply

**Step 3 — Observe the imputed values:**
- All 4 missing `valuenum` rows are filled with **1.31 mg/dL** (the global mean of the 3 reference rows)
- Row 1 (`<0.1 mg/dL` — undetectable creatinine): imputed as **1.31 mg/dL** → **26× upward distortion**
- Row 6 (`>15.0 mg/dL` — extreme renal failure): imputed as **1.31 mg/dL** → **11× downward distortion** — the emergency clinical signal is completely erased

---

## Python Solution Verification

To verify that the Python pipeline handles these cases correctly, run:

```powershell
cd "D:\CKD mini project\nephron-aware-renal-digital-twin"
python -m pytest tests/ -q --tb=short
```

Expected: **All tests pass** — confirming that the Python implementation enforces:
- Outpatient preservation via left-join null retention
- Analyte-specific normalization (z-score per `itemid`)
- LOCF with patient-grouped imputation (no cross-patient mean leakage)
- Censored value harmonization before ARFF export

---

## File Reference

| File | Size | Purpose |
|:---|:---:|:---|
| `renal_audit_sample.arff` | 183 KB | **Use for Finding 1** — contains STRING attributes |
| `renal_audit_sample_v2.arff` | 167 KB | **Use for Finding 2** — WEKA-compatible (no STRING) |
| `exp_a_replacemissing_out.arff` | 641 B | CLI output with STRING ARFF (0 data rows) |
| `exp_b_normalize_v2_out.arff` | 163 KB | Normalized output — creatinine compressed |
| `censored_creatinine_in_v2.arff` | <1 KB | Synthetic censored ARFF for Finding 3 |
| `audit_results.json` | 6 KB | Machine-readable complete results |
