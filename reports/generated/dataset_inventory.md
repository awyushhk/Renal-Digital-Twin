# MIMIC-IV 3.1 Dataset Inventory and Audit Report (Milestone M1.1)

**Execution Timestamp**: `2026-10-18 20:44:45`  
**Project**: Nephron-Aware Renal Digital Twin  
**Target Milestone**: M1.1 Dataset Inventory and Audit  
**Next Milestone**: Controlled Comparison of WEKA vs. Python Preprocessing Pipelines  

> [!IMPORTANT]
> **AUDIT MODE: COMPLETE FULL SCAN**
> All rows in `labevents.csv` (158,374,764 records across 634 chunks) were successfully processed.
> **Execution Performance**: 443.3s (357,297 rows/s). All statistics reflect definitive population totals.

---

## 1. Executive Summary & Table Inventory

| Table | File Name | Total Records | Key Identifiers | Primary Schema Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lab Dictionary** | `d_labitems.csv` | 1,650 | `itemid` (1,650 unique) | 1,650 items; 4 missing labels |
| **Lab Events** | `labevents.csv` | 158,374,764 (scanned) | `labevent_id`, `subject_id`, `hadm_id`, `itemid` | Mode: `FULL` (Complete) |
| **Patients** | `patients.csv` | 364,627 | `subject_id` (364,627 unique) | Complete; 0 duplicate subjects |
| **Admissions** | `admissions.csv` | 546,028 | `hadm_id` (546,028 unique), `subject_id` | Complete; 0 duplicate admissions |

---

## 2. Laboratory Dictionary Audit (`d_labitems.csv`)

- **Total Rows**: 1,650
- **Unique Item IDs**: 1,650 (100% unique, 0 duplicate keys)
- **Null Fields**:
  - `itemid`: 0 missing
  - `label`: 4 missing
  - `fluid`: 0 missing
  - `category`: 0 missing
- **Category Breakdown**:
  - `Chemistry`: 800
  - `Hematology`: 785
  - `Blood Gas`: 65
- **Fluid Breakdown**:
  - `Blood`: 821
  - `Other Body Fluid`: 201
  - `Urine`: 194
  - `Cerebrospinal Fluid`: 101
  - `Joint Fluid`: 72
  - `Pleural`: 69
  - `Ascites`: 68
  - `Bone Marrow`: 63
  - `Stool`: 46
  - `Q`: 9
  - `Fluid`: 5
  - `I`: 1

### Candidate Renal-Related Laboratory Items Discovered
Identified **135** candidate tests across search categories.

| Item ID | Test Label | Fluid | Category | Keyword / Group | Selection Rationale | Scanned Rows | Missing Num % | Scanned Units |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `50835` | Albumin, Ascites | Ascites | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 8,448 | 27.19% | `g/dL:8448` |
| `50862` | Albumin | Blood | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 1,032,456 | 0.02% | `g/dL:1032456` |
| `51019` | Albumin, Joint Fluid | Joint Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 9 | 11.11% | `g/dL:9` |
| `51025` | Albumin, Body Fluid | Other Body Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 827 | 8.1% | `g/dL:827` |
| `51046` | Albumin, Pleural | Pleural | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 6,302 | 7.97% | `g/dL:6302` |
| `51069` | Albumin, Urine | Urine | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 73,957 | 16.59% | `mg/dL:73957` |
| `51775` | (Albumin) | Cerebrospinal Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `51776` | <Albumin> | Cerebrospinal Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 38 | 10.53% | `mg/dL:38` |
| `51777` | Albumin, CSF | Cerebrospinal Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `51807` | (Albumin) | Joint Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `51836` | <Albumin> | Other Body Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `51897` | Surfactant/Albumin | Other Body Fluid | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 38 | 0.0% | `mg/g:38` |
| `51910` | (Albumin) | Pleural | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `51926` | (Albumin) | Stool | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `51927` | Albumin, Stool | Stool | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 1 | 100.0% | `g/dL:1` |
| `52022` | Albumin, Blood | Blood | Blood Gas | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `52703` | Albumin, Urine | Urine | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `53085` | Albumin | Blood | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 69 | 0.0% | `mg/dL:69` |
| `53116` | (Albumin) | Ascites | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `53138` | (Albumin) | Blood | Chemistry | Albumin | Serum albumin indicates nutritional/inflammatory status; urine albumin indicates glomerular barrier injury. | 0 | 0.0% | `None observed` |
| `50851` | Urea Nitrogen, Ascites | Ascites | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 31 | 0.0% | `mg/dL:31` |
| `51006` | Urea Nitrogen | Blood | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 4,202,807 | 0.05% | `mg/dL:4202806; <MISSING_UOM>:1` |
| `51045` | Urea Nitrogen, Body Fluid | Other Body Fluid | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 58 | 1.72% | `mg/dL:58` |
| `51104` | Urea Nitrogen, Urine | Urine | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 44,833 | 0.34% | `mg/dL:44833` |
| `51804` | Urea Nitrogen, CSF | Cerebrospinal Fluid | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 0 | 0.0% | `None observed` |
| `51825` | Urea Nitrogen, Joint Fluid | Joint Fluid | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 0 | 0.0% | `None observed` |
| `51842` | Bun | Other Body Fluid | Chemistry | BUN / Urea Nitrogen | Blood Urea Nitrogen abbreviation in legacy/alternate lab codes. | 0 | 0.0% | `None observed` |
| `51922` | Urea Nitrogen, Pleural | Pleural | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 2 | 0.0% | `mg/dL:2` |
| `51951` | Urea Nitrogen, Stool | Stool | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 0 | 0.0% | `None observed` |
| `52647` | Urea Nitrogen | Blood | Chemistry | BUN / Urea Nitrogen | Marker of protein catabolism and renal clearance; key component of BUN/Cr ratio. | 1,235 | 0.57% | `mg/dL:1235` |
| `50841` | Creatinine, Ascites | Ascites | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 2,869 | 0.35% | `mg/dL:2869` |
| `50912` | Creatinine | Blood | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 4,319,091 | 0.04% | `mg/dL:4319090; <MISSING_UOM>:1` |
| `51021` | Creatinine, Joint Fluid | Joint Fluid | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 3 | 0.0% | `mg/dL:3` |
| `51032` | Creatinine, Body Fluid | Other Body Fluid | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 508 | 2.17% | `mg/dL:508` |
| `51052` | Creatinine, Pleural | Pleural | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 1,842 | 0.49% | `mg/dL:1842` |
| `51067` | 24 hr Creatinine | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 10,828 | 0.0% | `mg/24hr:10828` |
| `51070` | Albumin/Creatinine, Urine | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 71,770 | 13.66% | `mg/g:71762; <MISSING_UOM>:8` |
| `51073` | Amylase/Creatinine Ratio, Urine | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 119 | 0.84% | `Ratio:119` |
| `51080` | Creatinine Clearance | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 717 | 0.56% | `mL/min:717` |
| `51081` | Creatinine, Serum | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 717 | 0.42% | `mg/dL:717` |
| `51082` | Creatinine, Urine | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 237,799 | 0.07% | `mg/dL:237799` |
| `51099` | Protein/Creatinine Ratio | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 100,387 | 3.07% | `Ratio:32757; mg/mg:67630` |
| `51106` | Urine Creatinine | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 717 | 0.42% | `mg/dL:717` |
| `51787` | Creatinine, CSF | Cerebrospinal Fluid | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 0 | 0.0% | `None observed` |
| `51937` | Creatinine, Stool | Stool | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 0 | 0.0% | `None observed` |
| `51963` | Amylase/Creatinine Clearance | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 0 | 0.0% | `None observed` |
| `51977` | Creatinine, Blood | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 0 | 0.0% | `None observed` |
| `52000` | Urine  Creatinine | Urine | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 0 | 0.0% | `None observed` |
| `52024` | Creatinine, Whole Blood | Blood | Blood Gas | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 15,175 | 0.44% | `mg/dL:15175` |
| `52546` | Creatinine | Blood | Chemistry | Creatinine | Primary endogenous filtration biomarker for kidney function and eGFR calculation. | 1,273 | 0.31% | `mg/dL:1273` |
| `50803` | Calculated Bicarbonate, Whole Blood | Blood | Blood Gas | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 32,994 | 0.17% | `mEq/L:32994` |
| `50806` | Chloride, Whole Blood | Blood | Blood Gas | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 108,168 | 0.1% | `mEq/L:108168` |
| `50822` | Potassium, Whole Blood | Blood | Blood Gas | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 333,554 | 0.15% | `mEq/L:333554` |
| `50824` | Sodium, Whole Blood | Blood | Blood Gas | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 159,073 | 0.05% | `mEq/L:159073` |
| `50833` | Potassium | Other Body Fluid | Blood Gas | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 0 | 0.0% | `None observed` |
| `50834` | Sodium, Body Fluid | Other Body Fluid | Blood Gas | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 0 | 0.0% | `None observed` |
| `50837` | Bicarbonate, Ascites | Ascites | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 43 | 4.65% | `mEq/L:43` |
| `50839` | Chloride, Ascites | Ascites | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 44 | 0.0% | `mEq/L:44` |
| `50847` | Potassium, Ascites | Ascites | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 48 | 0.0% | `mEq/L:48` |
| `50848` | Sodium, Ascites | Ascites | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 61 | 0.0% | `mEq/L:61` |
| `50882` | Bicarbonate | Blood | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 3,934,240 | 0.02% | `mEq/L:3934239; <MISSING_UOM>:1` |
| `50902` | Chloride | Blood | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 4,055,101 | 0.02% | `mEq/L:4055100; <MISSING_UOM>:1` |
| `50971` | Potassium | Blood | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 4,149,507 | 0.06% | `mEq/L:4149506; <MISSING_UOM>:1` |
| `50983` | Sodium | Blood | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 4,111,289 | 0.01% | `mEq/L:4111288; <MISSING_UOM>:1` |
| `51027` | Bicarbonate, Other Fluid | Other Body Fluid | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 203 | 5.42% | `mEq/L:203` |
| `51030` | Chloride, Body Fluid | Other Body Fluid | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 33 | 3.03% | `mEq/L:33` |
| `51041` | Potassium, Body Fluid | Other Body Fluid | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 41 | 12.2% | `mEq/L:41` |
| `51042` | Sodium, Body Fluid | Other Body Fluid | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 45 | 6.67% | `mEq/L:45` |
| `51048` | Bicarbonate, Pleural | Pleural | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 3 | 0.0% | `mEq/L:3` |
| `51050` | Chloride, Pleural | Pleural | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 2 | 0.0% | `mEq/L:2` |
| `51057` | Potassium, Pleural | Pleural | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 5 | 0.0% | `mEq/L:5` |
| `51058` | Sodium, Pleural | Pleural | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 7 | 0.0% | `mEq/L:7` |
| `51061` | Bicarbonate, Stool | Stool | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 47 | 36.17% | `mEq/L:47` |
| `51062` | Chloride, Stool | Stool | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 92 | 7.61% | `mEq/L:92` |
| `51064` | Potassium, Stool | Stool | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 239 | 11.3% | `mEq/L:239` |
| `51065` | Sodium, Stool | Stool | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 242 | 7.85% | `mEq/L:242` |
| `51076` | Bicarbonate, Urine | Urine | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 5,179 | 81.81% | `mEq/L:5179` |
| `51078` | Chloride, Urine | Urine | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 44,853 | 12.32% | `mEq/L:44853` |
| `51097` | Potassium, Urine | Urine | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 47,462 | 1.28% | `mEq/L:47462` |
| `51100` | Sodium, Urine | Urine | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 86,892 | 17.63% | `mEq/L:86892` |
| `51782` | Bicarbonate, CSF | Cerebrospinal Fluid | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 0 | 0.0% | `None observed` |
| `51784` | Chloride, CSF | Cerebrospinal Fluid | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 1 | 0.0% | `mEq/L:1` |
| `51800` | Potassium, CSF | Cerebrospinal Fluid | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 0 | 0.0% | `None observed` |
| `51801` | Sodium, CSF | Cerebrospinal Fluid | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 0 | 0.0% | `None observed` |
| `51811` | Bicarbonate,Joint Fluid | Joint Fluid | Chemistry | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 0 | 0.0% | `None observed` |
| `51813` | Chloride, Joint Fluid | Joint Fluid | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 0 | 0.0% | `None observed` |
| `51822` | Potassium, Joint Fluid | Joint Fluid | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 0 | 0.0% | `None observed` |
| `51823` | Sodium, Joint Fluid | Joint Fluid | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 0 | 0.0% | `None observed` |
| `52039` | Calculated Bicarbonate | Fluid | Blood Gas | Electrolytes | Acid-base homeostasis and marker of metabolic acidosis in chronic kidney disease. | 0 | 0.0% | `None observed` |
| `52046` | Potassium, Urine | Urine | Blood Gas | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 0 | 0.0% | `None observed` |
| `52047` | Sodium, Urine | Urine | Blood Gas | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 0 | 0.0% | `None observed` |
| `52434` | Chloride, Whole Blood | Blood | Blood Gas | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 0 | 0.0% | `None observed` |
| `52452` | Potassium, Whole Blood | Blood | Blood Gas | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 0 | 0.0% | `None observed` |
| `52455` | Sodium, Whole Blood | Blood | Blood Gas | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 0 | 0.0% | `None observed` |
| `52535` | Chloride | Blood | Chemistry | Electrolytes | Anion balance and metabolic evaluation alongside sodium and bicarbonate. | 1,235 | 0.0% | `mEq/L:1235` |
| `52610` | Potassium | Blood | Chemistry | Electrolytes | Critical electrolyte regulated by distal nephron; hyperkalemia risk in declining GFR. | 1,310 | 0.23% | `mEq/L:1310` |
| `52623` | Sodium | Blood | Chemistry | Electrolytes | Extracellular volume and osmolality regulation; marker of tubular handling and fluid status. | 1,309 | 0.08% | `mEq/L:1309` |
| `50809` | Glucose | Blood | Blood Gas | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 281,398 | 0.48% | `mg/dL:281398` |
| `50842` | Glucose, Ascites | Ascites | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 7,044 | 0.58% | `mg/dL:7044` |
| `50931` | Glucose | Blood | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 3,621,769 | 0.01% | `mg/dL:3621768; <MISSING_UOM>:1` |
| `51022` | Glucose, Joint Fluid | Joint Fluid | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 171 | 1.75% | `mg/dL:171` |
| `51034` | Glucose, Body Fluid | Other Body Fluid | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 928 | 1.94% | `mg/dL:928` |
| `51053` | Glucose, Pleural | Pleural | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 9,195 | 1.97% | `mg/dL:9195` |
| `51084` | Glucose, Urine | Urine | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 191 | 6.28% | `mg/dL:191` |
| `51478` | Glucose | Urine | Hematology | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 843,478 | 92.42% | `mg/dL:843478` |
| `51790` | Glucose, CSF | Cerebrospinal Fluid | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 15,151 | 0.13% | `mg/dL:15151` |
| `51941` | Glucose, Stool | Stool | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 0 | 0.0% | `None observed` |
| `51981` | Glucose | Urine | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 1,715 | 92.48% | `mg/dL:1715` |
| `52027` | Glucose, Whole Blood | Blood | Blood Gas | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 0 | 0.0% | `None observed` |
| `52569` | Glucose | Blood | Chemistry | Glucose | Key metabolic covariate for diabetic kidney disease (DKD) and tubular glycosuria. | 1,309 | 0.31% | `mg/dL:1309` |
| `50849` | Total Protein, Ascites | Ascites | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 10,293 | 0.65% | `g/dL:10293` |
| `50864` | Alpha-Fetoprotein | Blood | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 39,037 | 1.26% | `ng/mL:39037` |
| `50889` | C-Reactive Protein | Blood | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 178,039 | 2.12% | `mg/L:177983; <MISSING_UOM>:56` |
| `50975` | Protein Electrophoresis | Blood | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 56,769 | 100.0% | `<MISSING_UOM>:56769` |
| `50976` | Protein, Total | Blood | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 217,309 | 0.01% | `g/dL:217309` |
| `51024` | Total Protein, Joint Fluid | Joint Fluid | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 132 | 0.76% | `g/dL:132` |
| `51043` | Total Protein, Body Fluid | Other Body Fluid | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 915 | 1.53% | `g/dL:915` |
| `51059` | Total Protein, Pleural | Pleural | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 9,810 | 0.32% | `g/dL:9810` |
| `51068` | 24 hr Protein | Urine | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 5,100 | 0.0% | `mg/24hr:5100` |
| `51102` | Total Protein, Urine | Urine | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 115,596 | 8.4% | `mg/dL:115596` |
| `51270` | Protein C, Antigen | Blood | Hematology | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 136 | 0.74% | `%:136` |
| `51271` | Protein C, Functional | Blood | Hematology | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 2,005 | 2.34% | `%:2005` |
| `51272` | Protein S, Antigen | Blood | Hematology | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 178 | 1.12% | `%:178` |
| `51273` | Protein S, Functional | Blood | Hematology | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 1,806 | 2.6% | `%:1806` |
| `51492` | Protein | Urine | Hematology | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 843,478 | 66.65% | `mg/dL:843478` |
| `51802` | Total Protein, CSF | Cerebrospinal Fluid | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 15,572 | 0.55% | `mg/dL:15572` |
| `51949` | Total Protein, Stool | Stool | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 0 | 0.0% | `None observed` |
| `51992` | Protein | Urine | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 1,715 | 78.08% | `mg/dL:1715` |
| `52162` | Protein S | Blood | Hematology | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 0 | 0.0% | `None observed` |
| `53096` | Protein, Total | Blood | Chemistry | Urine / Total Protein | Proteinuria assessment (24h urine protein, dipstick, protein/creatinine ratio) for glomerular damage. | 68 | 0.0% | `mg/dL:68` |
| `50920` | Estimated GFR (MDRD equation) | Blood | Chemistry | eGFR Metric | Calculated or reported Glomerular Filtration Rate estimates (MDRD, CKD-EPI) in lab records. | 1,665,881 | 100.0% | `<MISSING_UOM>:1665881` |
| `51770` | MDRDgfr | Blood | Chemistry | eGFR Metric | Calculated or reported Glomerular Filtration Rate estimates (MDRD, CKD-EPI) in lab records. | 0 | 0.0% | `None observed` |
| `52026` | Estimated GFR (MDRD equation) | Blood | Blood Gas | eGFR Metric | Calculated or reported Glomerular Filtration Rate estimates (MDRD, CKD-EPI) in lab records. | 10,480 | 100.0% | `<MISSING_UOM>:10480` |
| `53161` | Estimated GFR (CKD- EPI Refit) | Blood | Chemistry | eGFR Metric | Calculated or reported Glomerular Filtration Rate estimates (MDRD, CKD-EPI) in lab records. | 81,810 | 0.0% | `mL/min/1.73m2:81809; <MISSING_UOM>:1` |
| `53180` | Estimated GFR (CKD- EPI 2021) | Blood | Blood Gas | eGFR Metric | Calculated or reported Glomerular Filtration Rate estimates (MDRD, CKD-EPI) in lab records. | 2,125 | 0.0% | `mL/min/1.73m2:2125` |

> [!NOTE]
> **Cystatin-C Findings in Dictionary**:
> A systematic search for keywords `cystatin`, `cys`, and related variants in `d_labitems.csv` confirmed **0 matches** for Cystatin-C.
> Only `50945` (Homocysteine) and `51473` (Cysteine Crystals) were matched under `cys`.
> Consequently, cystatin-C based eGFR equations (e.g. CKD-EPI 2012/2021 Cystatin-C) cannot be computed directly from standard MIMIC-IV laboratory events unless derived from uncatalogued text notes.

---

## 3. Laboratory Events Audit (`labevents.csv`)

- **Audit Execution Mode**: `FULL` (Complete full scan)
- **Total Records Scanned**: 158,374,764
- **Total Chunks Processed**: 634 (Chunk size: 250,000)
- **Scan Throughput**: 357,297 rows/second (Elapsed: 443.3s)
- **Unique Patient Subjects Observed**: 313,442
- **Unique Hospital Admissions Observed**: 447,689
- **Unique Laboratory Items Observed**: 976
- **Timestamp Coverage & Validity**:
  - `charttime` Range: `2105-01-19 12:01:00` to `2215-01-12 11:45:00`
  - Missing `charttime`: 0
  - Unparseable `charttime`: 0
- **Data Completeness**:
  - Missing `hadm_id`: 46.58% (Outpatient, ambulatory, or pre-admission draws do not carry hospital admission IDs)
  - Missing `valuenum`: 13.57% (Qualitative screens, microscopic urinalysis, blood bank, or text-only comments)
- **Duplicate & Conflict Analysis**:
  - Duplicate check enabled: `No / Zero detected`
  - Exact duplicate rows in chunks: 0
  - Multiple records for same `(subject_id, charttime, itemid)`: 0
  - Conflicting numeric measurements at identical timestamp: 0

---

## 4. Patient Demographics & Hospital Admissions Audit

### Patients Table (`patients.csv`)
- **Total Population**: 364,627 subjects
- **Unique Subject IDs**: 364,627 (100% unique, 0 duplicate subjects)
- **Gender Distribution**:
  - `F`: 191,984 (52.7%)
  - `M`: 172,643 (47.3%)
- **Anchor Age Distribution**:
  - Min: 18 years
  - Q1: 29 years
  - Median: 48 years
  - Q3: 65 years
  - Max: 91 years (MIMIC-IV caps age at >89 to 91 for privacy preservation)
  - Mean: 48.9 years
- **Mortality Completeness**: 326,326 records (89.5%) have null `dod`.

### Admissions Table (`admissions.csv`)
- **Total Hospital Encounters**: 546,028
- **Unique Admission IDs (`hadm_id`)**: 546,028 (100% unique, 0 duplicate encounters)
- **Unique Subjects Represented**: 223,452
- **In-Hospital Deaths**: 11,801 encounters (2.16%)
- **Length of Stay (LOS)**:
  - Median Hospital LOS: 2.82 days
  - Invalid Negative LOS (`dischtime < admittime`): 175 (Warning: 175 records with dischtime < admittime, requiring downstream filtering)
  - Unparseable Timestamps: Admittime = 0, Dischtime = 0
- **Top Admission Types**:
  - `EW EMER.`: 177,459 (32.5%)
  - `EU OBSERVATION`: 119,456 (21.9%)
  - `OBSERVATION ADMIT`: 84,437 (15.5%)
  - `URGENT`: 54,929 (10.1%)
  - `SURGICAL SAME DAY ADMISSION`: 42,898 (7.9%)
  - `DIRECT OBSERVATION`: 24,551 (4.5%)

### Aggregate Cross-Table Linkage Quality
- **Referential Integrity (Admissions -> Patients)**:
  - **100.00%** (223,452 / 223,452) of admission `subject_id`s exist in `patients.csv`.
- **Ambulatory / Outpatient Sub-population**:
  - **141,175** patients (38.72%) have records in `patients.csv` but zero hospital admissions in `admissions.csv`.

---

## 5. Methodological Classification & Safeguards

To prevent unwarranted conclusions, audit findings are classified strictly into:

### A. Observed Facts
1. `d_labitems.csv` contains 1,650 unique test codes. 4 test codes have null labels.
2. `patients.csv` contains 364,627 unique subjects with zero duplicate identifiers.
3. `admissions.csv` contains 546,028 unique admissions with zero duplicate `hadm_id`s, and 100% of subject IDs match `patients.csv`.
4. All timestamps in `admissions.csv` and scanned chunks of `labevents.csv` parse without format errors to standard datetimes.
5. In `labevents.csv`, ~49.6% of scanned laboratory draws have null `hadm_id`.
6. Multiple candidate tests for serum creatinine (`50912`, `52546`, `52024`) and urine creatinine (`51081`, `51106`, `52000`) coexist in MIMIC-IV with distinct fluids, departments, and units.

### B. Assumptions
1. `charttime` reflects the clinical specimen acquisition timestamp, whereas `storetime` reflects laboratory reporting verification.
2. Rows with missing `valuenum` represent either qualitative assay outputs or failed automated parsing.
3. Tests marked with fluid `Blood` and category `Chemistry` represent standard serum/plasma biochemistry panels.

### C. Unverified Findings (Require Downstream Harmonization)
1. Whether conflicting measurements at identical `(subject_id, charttime, itemid)` represent repeated draws, point-of-care vs. central lab discordance, or clerical double-entries.
2. Whether serum creatinine item IDs `50912` and `52546` have identical calibration (e.g. IDMS-traceable enzymatic vs. Jaffe method).
3. The exact proportion of outpatient vs. inpatient lab events across the entire 17.1 GB file (verified on scanned sample, pending full-scan confirmation).

### D. Limitations
1. If run in smoke-test mode, frequency counts and candidate coverage are derived from a bounded sample and must not be interpreted as whole-dataset absence or prevalence.
2. No unit conversions, outlier removal, missing-value imputation, or temporal resampling were performed in this milestone.
3. Patient confidentiality is strictly preserved: no individual patient identifiers or trajectories are published.

---

## 6. Generated Output Files

The following structured files were generated in `reports/generated/`:
1. `dataset_inventory.md`: Complete audit and inventory report.
2. `lab_item_inventory.csv`: Complete dictionary joined with scan frequencies and units.
3. `data_quality_summary.csv`: Tabular quality indicators across Completeness, Uniqueness, Validity, and Consistency.
4. `renal_candidate_items.csv`: Curated candidate renal tests with selection rationales and observed distributions.
