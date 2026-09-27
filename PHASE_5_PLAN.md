# PHASE 5 IMPLEMENTATION PLAN: POST-PROCESSING, VERIFICATION & MEMORY OPTIMIZATION

## 1. Objective & Overview
Generate the final test output TSV files (`matching_results.tsv` and `candidate_pairs.tsv`), strictly enforce the **subset constraint** (\(\text{IDs in } \texttt{matching\_results.tsv} \subseteq \text{IDs in } \texttt{candidate\_pairs.tsv}\)), execute the official validation tool (`utils/validate_submission.py`), and optimize memory footprint.

---

## 2. Detailed Task Breakdown

### Task 5.1: `matching_results.tsv` Exporter & Formatter (`export_matching.py`)
- Location: `code/business_entity_resolution/src/export_matching.py`
- Format: Tab-separated (`\t`).
- Header: `source1_entity_id\tmatched_entity_ids`
- Rules:
  - Every S1 entity in `test_source1.tsv` must appear exactly once.
  - Comma-separated list of matched S2-/S3- entity IDs, or empty string for singletons.
  - No duplicate IDs within any ID list.

### Task 5.2: Subset Constraint Verification Engine (`verify_subset.py`)
- Location: `code/business_entity_resolution/src/verify_subset.py`
- Enforce mandatory rule:
  \[ \forall S1_i: \text{matched\_entity\_ids}(S1_i) \subseteq \text{candidate\_entity\_ids}(S1_i) \]
- If any matched ID was not in candidates, throw explicit pipeline error and flag discrepancy before submission.

### Task 5.3: Automated Execution of Official Validator (`validate_submission.py`)
- Command:
  ```bash
  python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
  ```
- Must return `PASS` (exit code 0).

### Task 5.4: RAM & GPU Memory Footprint Optimization
- Enforce batch processing for FAISS dense vector search and GBDT inference.
- Include explicit `gc.collect()` and PyTorch CUDA memory clearing calls between execution stages.

---

## 3. Deliverables & Outputs
- `code/business_entity_resolution/src/export_matching.py`
- `code/business_entity_resolution/src/verify_subset.py`
- Formatted output files:
  - `output/matching_results.tsv`
  - `output/candidate_pairs.tsv`

---

## 4. Manual Quality Control (QC) Gate & Stop Instruction

> [!IMPORTANT]
> **STOP & PERFORM MANUAL QUALITY CONTROL BEFORE PROCEEDING TO PHASE 6**

### Step-by-Step QC Execution Instructions:

1. **Generate Test Set Output Files**:
   ```bash
   python code/business_entity_resolution/src/export_matching.py --model models/gbdt_ensemble.pkl --test-candidates output/candidate_pairs.tsv --test-dir dataset/test --out output/matching_results.tsv
   ```

2. **Run Subset Verification Script**:
   ```bash
   python code/business_entity_resolution/src/verify_subset.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm output prints **"SUBSET VERIFICATION PASS: All matched entity IDs exist within candidate pairs."**

3. **Run Official Competition Submission Validator**:
   ```bash
   python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm output prints **`PASS`** with exit code 0.
   - [ ] Confirm zero errors or warnings reported regarding formatting, missing IDs, or self-matches.

4. **Inspect Test Output File Stats**:
   ```bash
   wc -l output/matching_results.tsv
   wc -l output/candidate_pairs.tsv
   ```
   *Manual Verification Checklist*:
   - [ ] Line counts of both files match exactly (Header + row count equal to total `test_source1.tsv` entities).

**Decision Checkpoint**: Once official validator returns `PASS` and subset check succeeds, inform the assistant to proceed with **Phase 6 Implementation**.
