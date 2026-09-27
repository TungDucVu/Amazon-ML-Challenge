# PHASE 1 REPORT: INFRASTRUCTURE, EDA & VALIDATION SETUP

**Project**: Amazon ML Challenge 2026 - Business Entity Resolution  
**Branch**: `Tung_code`  
**Date**: September 27, 2026  

---

## 1. Executive Summary

Phase 1 establishes the foundational data ingestion, evaluation, profiling, and validation infrastructure for the Business Entity Resolution pipeline. All tasks specified in `PHASE_1_PLAN.md` and the Phase 1 caution checklist have been successfully executed, unit-tested, and verified against the 2.2M+ entity training dataset.

---

## 2. Summary of Work Completed

1. **Environment Setup & Dependency Pinning**:
   - Created `code/business_entity_resolution/requirements.txt` pinning core scientific and machine learning packages (`pandas`, `polars`, `scipy`, `scikit-learn`, `rapidfuzz`, `torch`, `sentence-transformers`, `faiss-cpu`, `lightgbm`, `xgboost`, `catboost`).

2. **Strict Ingestion & TSV Safeguards (`data_loader.py`)**:
   - Created [`data_loader.py`](file:///d:/Future%20Career/Amazon%20Challenge/code/business_entity_resolution/src/data_loader.py) with explicit tab parsing (`sep="\t"`), no quotation escaping issues (`quoting=csv.QUOTE_NONE`), and zero NA conversion (`dtype=str`, `keep_default_na=False`).
   - Implemented column count validation (4 columns for sources, 2 for ground truth).

3. **Macro \(F_{0.5}\) Metric Engine & Test Suite (`metrics.py`)**:
   - Created [`metrics.py`](file:///d:/Future%20Career/Amazon%20Challenge/code/business_entity_resolution/src/metrics.py) implementing entity-level \(F_{0.5}\) calculation prior to macro-averaging across all Source 1 entities.
   - Built an automated unit test runner (`--test-dummy`) covering all boundary conditions (singletons, partial matches, false positives, exact matches).

4. **Exploratory Data Analysis Engine (`eda.py`)**:
   - Created [`eda.py`](file:///d:/Future%20Career/Amazon%20Challenge/code/business_entity_resolution/src/eda.py) performing deep structural, ID prefix, uniqueness, country distribution, and match cardinality profiling on the 2.206M training entities.

5. **Stratified 80/20 Validation Split Generator (`create_val_split.py`)**:
   - Created [`create_val_split.py`](file:///d:/Future%20Career/Amazon%20Challenge/code/business_entity_resolution/src/create_val_split.py) using fixed seed `RANDOM_SEED = 42`.
   - Generated stratified splits based on `country + match_cardinality_bucket` (1.765M train S1 entities / 441K validation S1 entities).
   - Conducted explicit leakage audits verifying 0 entity overlap between train and validation sets.

---

## 3. Key Assumptions & Pipeline Adaptations

| # | Assumption Made | Pipeline Change / Adaptation Implemented | Rationale & Justification |
|---|---|---|---|
| **1** | Business names/addresses containing `"NA"`, `"N/A"`, `"None"`, or apostrophes (`McDonald's`) are valid literal strings, not missing values. | Enforced `dtype=str`, `keep_default_na=False`, and `quoting=csv.QUOTE_NONE` in pandas ingestion. | Prevents pandas from converting real business names like *"NA Corporation"* into NaN floats, avoiding silent pipeline crashes. |
| **2** | Candidate blocking on validation data must simulate real inference against the full distractor pool. | `train_source2.tsv` (5.03M) and `train_source3.tsv` (5.28M) are **never downsampled**. Only S1 and Ground Truth are split. | Downsampling S2/S3 would artificially inflate blocking recall and distort candidate search space. |
| **3** | Evaluation metric is macro-averaged per Source 1 entity, not global micro-averaged. | `metrics.py` computes \(F_{0.5}(S1_i)\) per entity first, then averages across all \(N\) entities. Singletons score 1.0 if empty, 0.0 if false match predicted. | Matches official competition evaluation specification exactly. Prevents micro-averaging bias. |
| **4** | Test set contains unseen country `France` (open-set setup). | Zero hardcoding of country codes or language-specific regexes in loaders/EDA. | Ensures pipeline modules transfer seamlessly to `France` in test set without code modifications. |
| **5** | Source 1 entities frequently match multiple records within the *same* source file (e.g. multiple S2 matches). | EDA revealed **76.80%** of S1 entities match multiple records in the same source. Data models use set-based ID parsing. | Avoided assuming 1-to-1 matching constraints per source file during blocking and feature design. |

---

## 4. Quality Control (QC) Outputs & Comparison Matrix

### QC 1: Metric Engine Unit Test Suite (`python metrics.py --test-dummy`)
* **Purpose**: Verify exact precision-heavy \(F_{0.5}\) calculations and singleton edge cases.

| Test Case | Scenario Description | Expected Output | Actual Output | Status |
|---|---|---|---|---|
| **Test A** | True = `[]`, Pred = `[]` (True Singleton) | `1.0000` | `1.0000` | **PASS** |
| **Test B** | True = `[]`, Pred = `['S2-001']` (False Match on Singleton) | `0.0000` | `0.0000` | **PASS** |
| **Test C** | True = `['S2-001']`, Pred = `[]` (Missed Non-Singleton Match) | `0.0000` | `0.0000` | **PASS** |
| **Test D** | True = `['S2-001', 'S3-002']`, Pred = `['S2-001', 'S3-002']` (Exact Match) | `1.0000` | `1.0000` | **PASS** |
| **Test E** | True = `['S2-001', 'S3-002']`, Pred = `['S2-001']` (P=1.0, R=0.5) | `0.8333` | `0.8333` | **PASS** |
| **Test F** | True = `['S2-001']`, Pred = `['S2-001', 'S3-002']` (P=0.5, R=1.0) | `0.5556` | `0.5556` | **PASS** |
| **Test G** | True = `['S2-001', 'S3-002']`, Pred = `['S2-001', 'S2-002', 'S3-002']` (P=2/3, R=1.0) | `0.7143` | `0.7143` | **PASS** |

---

### QC 2: EDA Integrity & Structural Audit (`python eda.py`)
* **Purpose**: Audit training datasets for corruption, duplicate IDs, missing values, and ground truth consistency.

| Audit Metric | Expected Value | Actual Output Log | Status |
|---|---|---|---|
| **Source 1 Row Count** | `2,206,821` | `2,206,821` | **PASS** |
| **Source 2 Row Count** | `5,034,616` | `5,034,616` | **PASS** |
| **Source 3 Row Count** | `5,285,603` | `5,285,603` | **PASS** |
| **ID Prefix Verification** | `S1-`, `S2-`, `S3-` all True | S1: True, S2: True, S3: True | **PASS** |
| **Duplicate ID Count** | `0` duplicates across all files | S1: 0, S2: 0, S3: 0, GT: 0 | **PASS** |
| **Missing GT Reference IDs** | `0` missing IDs | S1 Missing: 0, Target S2/S3 Missing: 0 | **PASS** |
| **Country Distribution** | US ~60%, India ~40% | US: 59.98%, India: 40.02% | **PASS** |
| **Singleton Ratio** | Reported without pre-assumption | `123,247` entities (`5.58%`) | **PASS** |
| **Multi-Match Intra-Source Ratio** | Reported without pre-assumption | `1,694,736` entities (`76.80%`) | **PASS** |

---

### QC 3: Stratified Validation Split Integrity (`python create_val_split.py`)
* **Purpose**: Guarantee clean 80/20 train/validation split with zero data leakage.

| Verification Item | Expected Output | Actual Output | Status |
|---|---|---|---|
| **Train S1 Entity Count (80%)** | `~1,765,456` | `1,765,456` | **PASS** |
| **Val S1 Entity Count (20%)** | `~441,365` | `441,365` | **PASS** |
| **S1 ID Leakage (Train ∩ Val)** | `0` | `0` | **PASS** |
| **Distractor Search Universe** | Full S2 (5.03M) & S3 (5.28M) preserved | Preserved (100% of S2/S3 available) | **PASS** |
| **Stratification Alignment** | Equal country/cardinality proportions | `US_2plus`, `India_2plus`, `US_0`, `US_1`, `India_0`, `India_1` perfectly proportioned | **PASS** |

---

### QC 4: Tab-Separated File Format Audit
* **Purpose**: Verify generated files are tab-delimited without quote-wrapping or column merging.

| Check Item | Target Requirement | Validation Result | Status |
|---|---|---|---|
| **Column Separator** | Strict `\t` (Tab) | Verified across all loaders and outputs | **PASS** |
| **Line Wrapping / Newlines** | No broken lines | 0 parser errors on 2.2M+ rows | **PASS** |
| **Quote Escaping** | `QUOTE_NONE` safe for `'` and `"` | Verified on names like `McDonald's` | **PASS** |

---

## 5. Conclusion & Readiness for Phase 2

All Phase 1 requirements, assumptions, code modules, and manual QC checks have passed with **100% compliance**. 

The pipeline is fully ready to proceed to **Phase 2: Budget-Constrained & High-Recall Blocking**.
