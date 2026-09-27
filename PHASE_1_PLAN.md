# PHASE 1 IMPLEMENTATION PLAN: INFRASTRUCTURE, EDA & VALIDATION SETUP

## 1. Objective & Overview
Establish a robust, reproducible development foundation, clean tab-separated TSV data loading routines, perform comprehensive Exploratory Data Analysis (EDA), create a stratified 80/20 local validation split, and build an exact Macro-averaged \(F_{0.5}\) evaluation harness.

---

## 2. Detailed Task Breakdown

### Task 1.1: Git & Workspace Verification
- Ensure working branch is `Tung_code`.
- Verify `.gitignore` ignores large zip/TSV files while tracking code and documentation.

### Task 1.2: Environment & Dependencies Definition
- Create `code/business_entity_resolution/requirements.txt` with pinned versions:
  - `pandas>=2.0.0`, `polars>=0.20.0`
  - `numpy>=1.24.0`, `scipy>=1.10.0`
  - `scikit-learn>=1.2.0`, `rapidfuzz>=3.0.0`
  - `torch>=2.0.0`, `sentence-transformers>=2.2.2`
  - `faiss-cpu>=1.7.4`
  - `lightgbm>=3.3.5`, `xgboost>=1.7.5`, `catboost>=1.2`
  - `tqdm>=4.65.0`, `pyarrow>=12.0.0`

### Task 1.3: Data Ingestion & TSV Safeguard Module (`data_loader.py`)
- Location: `code/business_entity_resolution/src/data_loader.py`
- Functions:
  - `load_tsv(filepath: str) -> pd.DataFrame`: Always specify `sep="\t"`, `quoting=3` (quote none) to prevent silent column merging.
  - `validate_schema(df: pd.DataFrame, expected_columns: list) -> bool`: Verify required columns (`entity_id`, `business_name`, `business_address`, `country`).

### Task 1.4: Exploratory Data Analysis Engine (`eda.py`)
- Location: `code/business_entity_resolution/src/eda.py`
- Analytics to compute & log:
  - Total records per file across Source 1, Source 2, Source 3.
  - Country breakdown for train (`US`, `India`) and test (`US`, `India`, `France`).
  - Distribution of ground truth matches per S1 entity (proportion of singletons with 0 matches vs 1 match vs multi-matches).
  - Null value percentages and string length statistics for `business_name` and `business_address`.

### Task 1.5: Stratified Validation Split Generator (`create_val_split.py`)
- Location: `code/business_entity_resolution/src/create_val_split.py`
- Strategy:
  - Sample 20% of `train_source1.tsv` entities stratified by country (`US`, `India`) and match cardinality (singleton vs non-singleton).
  - Extract corresponding ground-truth matches for the holdout set into `dataset/val_split/val_ground_truth.tsv`.
  - Save `val_source1.tsv`, `val_source2.tsv`, `val_source3.tsv` into `dataset/val_split/`.

### Task 1.6: Macro \(F_{0.5}\) Scoring Engine (`metrics.py`)
- Location: `code/business_entity_resolution/src/metrics.py`
- Implement exact competition evaluation formula:
  \[ F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}} \]
- Compute macro-average across ALL S1 entities in the validation set (including singletons where empty match list scores 1.0).

---

## 3. Deliverables & Outputs
- `code/business_entity_resolution/requirements.txt`
- `code/business_entity_resolution/src/data_loader.py`
- `code/business_entity_resolution/src/eda.py`
- `code/business_entity_resolution/src/create_val_split.py`
- `code/business_entity_resolution/src/metrics.py`
- Validation dataset directory `dataset/val_split/`

---

## 4. Manual Quality Control (QC) Gate & Stop Instruction

> [!IMPORTANT]
> **STOP & PERFORM MANUAL QUALITY CONTROL BEFORE PROCEEDING TO PHASE 2**

### Step-by-Step QC Execution Instructions:

1. **Verify Git Branch & Clean Workspace**:
   ```bash
   git branch --show-current
   ```
   *Expected Output*: `Tung_code`

2. **Execute Data Loader & EDA Inspection**:
   ```bash
   python code/business_entity_resolution/src/eda.py
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm no single-column TSV parsing errors occurred.
   - [ ] Check total row count of `train_source1.tsv` matches file size expectations.
   - [ ] Verify ground truth singleton ratio (approx 50-70% of S1 records have 0 matches).

3. **Generate & Audit Validation Split**:
   ```bash
   python code/business_entity_resolution/src/create_val_split.py
   ```
   *Manual Verification Checklist*:
   - [ ] Inspect `dataset/val_split/val_source1.tsv` to ensure exact tab alignment.
   - [ ] Verify that no S1 entity in validation set overlaps with training S1 entities.

4. **Run Metric Verification Test**:
   ```bash
   python code/business_entity_resolution/src/metrics.py --test-dummy
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm metric engine returns 1.0 for perfect predictions.
   - [ ] Confirm metric engine returns 1.0 for correctly identified empty singletons.
   - [ ] Confirm metric engine penalizes false positive matches according to \(F_{0.5}\) precision weighting.

**Decision Checkpoint**: Once all checklist items pass, inform the assistant to proceed with **Phase 2 Implementation**.
